//  WHAT: A model's layer caches plus snapshots of its recurrent layers at known token counts, so a
//        session can rewind to a turn boundary instead of prefilling the whole prompt again.
//  WHY:  Qwen3-Coder-Next keeps 36 of its 48 layers as Gated DeltaNet state (`MambaCache`), which
//        has no position to trim back to, so a session could never reuse its cache on the default
//        model: every round prefilled 5–20k tokens, and prefill was more than half of a turn (127 s
//        of 227 s, measured 2026-10-07). The next round's prompt shares the previous prompt exactly —
//        the assistant turn is re-rendered by the template, so the generated tokens are not the
//        shared part — and a snapshot taken when a prompt has just been prefilled is that point.
//        Rewinding to it costs nothing; the round prefills only what is new.
//  PIN:  Snapshots hold REFERENCES, never copies. The recurrent state arrays are replaced each step
//        (`Qwen3Next.swift`: `cache[1] = newState`), never written in place, so a held reference
//        stays what it was. KV layers are trimmed by offset — O(1), and their buffers are left
//        alone: holding a view of a KV buffer would make every later step copy the whole buffer,
//        because MLX only updates a buffer in place while nothing else references it.
//  PIN:  The cache's position is the offset of a KV layer. A recurrent layer's `offset` never
//        moves (`ArraysCache.advance` touches lengths and padding only), so reading
//        `layers.first` on a hybrid model always says 0.
//  PIN:  A turn boundary is PINNED, outside the ring. A chat template that drops an older turn's reasoning
//        (Qwen3.5: `<think>` only after the last user query) re-renders the previous turn differently
//        once the next one is asked, so the next turn shares the prompt only up to the end of the
//        previous ask. Every round of that turn snapshots, and a four-slot ring had dropped the ask's
//        point by round five: each follow-up prefilled 30–40k tokens from zero (Craft, 2026-11-05).
//        One pinned snapshot per cache, replaced by the next pin, invalidated like any other.
//  PIN:  A MARK is kept outside the ring and apart from the pin (Craft, 2026-11-18, then 2026-11-22). A prompt names
//        the positions every later prompt of its session begins with: where the tools end, where the system prompt
//        does. When a session's history is rewritten (Craft's compaction replaces older turns with a summary), the
//        next prompt shares the last one only up to the last mark (~17k tokens); the turn pin sits past the rewrite
//        and the ring is newer still, so without the mark every rewrite read the system prompt and the tools again.
//        A mark is also where a cache is cut into the stretches kept as files (`SlotStretch`): the state at a
//        stretch's end is the mark's. A mark goes only when the cache is taken back before it.
//  PIN:  A MARK IS A POSITION ANY CACHE KEEPS (2026-11-22). A cache with no recurrent layer rewinds by trimming and
//        needs no state, but its marks are still where its stretches end: read back from a file it lost them, and a
//        keep up to one then threw.
//  PIN:  `release` and `adopt` hand the layers to the parallel engine's batch and back. While they are
//        out, the snapshots stay here, and nothing here references the buffers the batch writes.

import Foundation
import MLX
import MLXLMCommon

public final class CheckpointedCache {
    public struct Snapshot {
        public let tokenCount: Int
        /// Layer index → that layer's state arrays at `tokenCount`.
        public let recurrent: [Int: [MLXArray]]
    }

    public private(set) var layers: [KVCache]
    /// Newest last; at most `ring`.
    public private(set) var snapshots: [Snapshot] = []
    /// The last turn boundary, kept outside the ring (`snapshot(at:pinned:)`).
    public private(set) var pinned: Snapshot?
    /// The positions every prompt of the session begins with, ascending, kept outside the ring and apart from the pin
    /// (`snapshot(at:mark:)`): on a cache with no recurrent layer each holds its position alone.
    public private(set) var marks: [Snapshot] = []
    /// Every snapshot held, the pinned one and the marks included.
    public var snapshotCount: Int { snapshots.count + (pinned == nil ? 0 : 1) + marks.count }
    /// Every snapshot held, newest last within each kind.
    private var held: [Snapshot] { marks + (pinned.map { [$0] } ?? []) + snapshots }
    /// What the sessions share (`sharedSnapshotBytes`): the ring, and every mark but the last.
    private var shared: [Snapshot] { marks.dropLast() + snapshots }
    public let ring: Int
    /// Recurrent layers, by index: snapshot-able, never trimmable.
    private var recurrent: [Int: ArraysCache] = [:]
    /// Layers trimmed by offset.
    private var trimmable: [Int] = []
    /// Layers that are neither: the cache can only ever be appended to.
    private var opaque = 0

    public init(layers: [KVCache], ring: Int = 4) {
        self.layers = layers
        self.ring = max(1, ring)
        classify()
    }

    /// A cache read back from files (`SlotFile`): its layers, with the snapshots it was kept with that they still
    /// cover — the newest `ring` of the ring, the pinned turn boundary and the marks outside it.
    convenience init(layers: [KVCache], ring: Int, snapshots: [Snapshot], pinned: Snapshot?, marks: [Snapshot] = []) {
        self.init(layers: layers, ring: ring)
        guard canRewind else { return }
        let now = position
        self.marks = marks.filter { $0.tokenCount <= now }.sorted { $0.tokenCount < $1.tokenCount }
        guard !recurrent.isEmpty else { return }
        self.snapshots = Array(snapshots.filter { $0.tokenCount <= now }.sorted { $0.tokenCount < $1.tokenCount }.suffix(self.ring))
        self.pinned = pinned.flatMap { $0.tokenCount <= now ? $0 : nil }
    }

    private func classify() {
        recurrent = [:]
        trimmable = []
        opaque = 0
        for (index, layer) in layers.enumerated() {
            if let arrays = layer as? ArraysCache {
                recurrent[index] = arrays
            } else if layer.isTrimmable {
                trimmable.append(index)
            } else {
                opaque += 1
            }
        }
    }

    /// Tokens the cache holds.
    public var position: Int { trimmable.first.map { layers[$0].offset } ?? 0 }

    // MARK: - Bytes (shapes only: nothing is evaluated)

    /// The attention layers' buffers as allocated, capacity included.
    public var kvBytes: Int {
        layers.enumerated().filter { recurrent[$0.offset] == nil }.flatMap { $0.element.innerState() }.reduce(0) { $0 + $1.nbytes }
    }

    /// The recurrent layers' live state.
    public var recurrentBytes: Int {
        recurrent.values.flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
    }

    /// What the snapshots hold.
    public var snapshotBytes: Int { Self.bytes(held) }

    /// What the snapshots the sessions share hold: the ring's, and every mark's but the last. The pin and the last
    /// mark are the session's own while its cache stays (`dropSharedSnapshot`).
    public var sharedSnapshotBytes: Int { Self.bytes(shared) }

    /// How many snapshots the sessions share.
    public var sharedSnapshotCount: Int { shared.count }

    private static func bytes(_ snapshots: some Sequence<Snapshot>) -> Int {
        snapshots.flatMap { $0.recurrent.values.flatMap { $0 } }.reduce(0) { $0 + $1.nbytes }
    }

    /// Whether `rewind` can ever succeed.
    public var canRewind: Bool { !layers.isEmpty && opaque == 0 && (!trimmable.isEmpty || recurrent.isEmpty) }

    /// Every layer trims by position: no snapshot is needed, and a rewind is exact.
    public var isFullyTrimmable: Bool { opaque == 0 && recurrent.isEmpty }

    /// Remembers the recurrent layers as they stand with `tokenCount` tokens in the cache.
    /// Taken right after a prompt has been prefilled, before the first generated token — or, `pinned`,
    /// where a turn's ask ends, kept outside the ring until the next pin replaces it — or, `mark`, at a position every
    /// prompt of the session begins with, kept until the cache is taken back before it.
    public func snapshot(at tokenCount: Int, pinned pin: Bool = false, mark: Bool = false) {
        guard canRewind else { return }
        // A cache that trims by position rewinds exactly without snapshots; a mark of its holds only where a stretch
        // ends (`SlotStretch`).
        if recurrent.isEmpty {
            if mark { place(Snapshot(tokenCount: tokenCount, recurrent: [:])) }
            return
        }
        var state: [Int: [MLXArray]] = [:]
        for (index, layer) in recurrent {
            let arrays = layer.state
            // A layer with no state yet cannot be restored; a snapshot of it would restore nothing.
            guard !arrays.isEmpty else { return }
            state[index] = arrays
        }
        snapshots.removeAll { $0.tokenCount >= tokenCount }
        if let kept = pinned, kept.tokenCount >= tokenCount { pinned = nil }
        marks.removeAll { $0.tokenCount > tokenCount }
        let taken = Snapshot(tokenCount: tokenCount, recurrent: state)
        if mark {
            place(taken)
        } else if pin {
            pinned = taken
        } else {
            snapshots.append(taken)
            if snapshots.count > ring { snapshots.removeFirst(snapshots.count - ring) }
        }
    }

    /// A mark at its position: it replaces one already there, and the marks past it go.
    private func place(_ mark: Snapshot) {
        marks.removeAll { $0.tokenCount >= mark.tokenCount }
        marks.append(mark)
    }

    /// The state a stretch ending at `tokenCount` is kept with (`SlotStretch`), nil being where the cache stands:
    /// nothing for a cache that trims by position, the live state at the cache's end, or a snapshot taken exactly
    /// there. Nil when the cache holds no state at that position.
    func state(at tokenCount: Int?) -> [Int: [MLXArray]]? {
        if recurrent.isEmpty { return [:] }
        guard let tokenCount else {
            let live = recurrent.mapValues(\.state)
            return live.values.contains(where: \.isEmpty) ? nil : live
        }
        return held.last { $0.tokenCount == tokenCount }?.recurrent
    }

    /// Rewinds to exactly `target` when every layer trims by position, otherwise to the newest
    /// snapshot at or before `target`. Returns the token count the cache now holds, or nil when
    /// nothing fits — the caller then starts a fresh cache.
    @discardableResult
    public func rewind(toAtMost target: Int) -> Int? { rewindPoint(toAtMost: target)?.position }

    /// `rewind(toAtMost:)`, saying which point it went back to: a mark, the pin or the ring when they stand at the
    /// same count, in that order, or a trim.
    public func rewindPoint(toAtMost target: Int) -> (position: Int, point: RewindPoint)? {
        guard canRewind else { return nil }
        let current = position
        if isFullyTrimmable {
            let count = max(0, min(target, current))
            trim(by: current - count)
            marks.removeAll { $0.tokenCount > count }
            return (count, .trim)
        }
        guard let snapshot = held.filter({ $0.tokenCount <= target }).max(by: { $0.tokenCount < $1.tokenCount }),
              snapshot.tokenCount <= current else { return nil }
        let point: RewindPoint = marks.contains { $0.tokenCount == snapshot.tokenCount } ? .mark
            : pinned?.tokenCount == snapshot.tokenCount ? .pin : .ring
        for (index, arrays) in snapshot.recurrent {
            recurrent[index]?.state = arrays
        }
        trim(by: current - snapshot.tokenCount)
        snapshots.removeAll { $0.tokenCount > snapshot.tokenCount }
        if let kept = pinned, kept.tokenCount > snapshot.tokenCount { pinned = nil }
        marks.removeAll { $0.tokenCount > snapshot.tokenCount }
        return (snapshot.tokenCount, point)
    }

    /// Drops one of the snapshots the sessions share: the ring's oldest while it holds more than one, then the earliest
    /// mark while a later one stands, then the ring's last. False when the session shares none.
    ///
    /// PIN: THE RING GOES FIRST (2026-11-20). The pin is where the next turn parts from this one and a mark where a
    /// rewritten history does; the ring's are this turn's rounds, the newest of which the next round needs and the
    /// oldest of which nothing does. Dropped by age instead, the pin went before the ring's newer snapshots, and the
    /// next turn read this one again from the mark before it.
    /// PIN: THE PIN AND THE LAST MARK ARE THE SESSION'S OWN (2026-11-22). One bound held every session's snapshots
    /// together, in megabytes chosen before a snapshot was 164 MiB: two sessions kept one ring entry each, and four
    /// started losing their pins, so a follow-up read its whole turn again. Neither is dropped here: they go with the
    /// cache, when its session is evicted.
    @discardableResult
    public func dropSharedSnapshot() -> Bool {
        if snapshots.count > 1 {
            snapshots.removeFirst()
            return true
        }
        if marks.count > 1 {
            marks.removeFirst()
            return true
        }
        guard !snapshots.isEmpty else { return false }
        snapshots.removeFirst()
        return true
    }

    /// Hands the layers to whoever decodes on them next and keeps only the snapshots.
    public func release() -> [KVCache] {
        let out = layers
        layers = []
        recurrent = [:]
        trimmable = []
        opaque = 0
        return out
    }

    /// Takes layers back — the same layout, more tokens — keeping the snapshots they still cover.
    public func adopt(_ newLayers: [KVCache]) {
        layers = newLayers
        classify()
        let now = position
        snapshots.removeAll { $0.tokenCount > now }
        if let kept = pinned, kept.tokenCount > now { pinned = nil }
        marks.removeAll { $0.tokenCount > now }
    }

    private func trim(by n: Int) {
        guard n > 0 else { return }
        for index in trimmable { layers[index].trim(n) }
    }
}
