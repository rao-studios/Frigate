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
//  PIN:  The end of the stable prefix is ANCHORED, outside the ring and apart from the pin (Craft, 2026-11-18). When a
//        session's history is rewritten (Craft's compaction replaces older turns with a summary), the next prompt
//        shares the last one only up to the system prompt and tools (~17k tokens); the turn pin sits past the rewrite
//        and the ring is newer still, so without the anchor every rewrite read the system prompt again. The anchor is
//        replaced only by the next anchor, and memory pressure drops it last.
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
    /// The end of the stable prefix, kept outside the ring and apart from the pin (`snapshot(at:anchored:)`).
    public private(set) var anchor: Snapshot?
    /// Every snapshot held, the pinned one and the anchor included: what memory pressure can drop.
    public var snapshotCount: Int { snapshots.count + (pinned == nil ? 0 : 1) + (anchor == nil ? 0 : 1) }
    /// Every snapshot held, newest last within each kind.
    private var held: [Snapshot] { (anchor.map { [$0] } ?? []) + (pinned.map { [$0] } ?? []) + snapshots }
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

    /// A cache read back from a file (`SlotFile`): its layers, with the snapshots it was kept with that they still
    /// cover — the newest `ring` of the ring, the pinned turn boundary and the anchor outside it.
    convenience init(layers: [KVCache], ring: Int, snapshots: [Snapshot], pinned: Snapshot?, anchor: Snapshot? = nil) {
        self.init(layers: layers, ring: ring)
        guard !recurrent.isEmpty, canRewind else { return }
        let now = position
        self.snapshots = Array(snapshots.filter { $0.tokenCount <= now }.sorted { $0.tokenCount < $1.tokenCount }.suffix(self.ring))
        self.pinned = pinned.flatMap { $0.tokenCount <= now ? $0 : nil }
        self.anchor = anchor.flatMap { $0.tokenCount <= now ? $0 : nil }
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
    public var snapshotBytes: Int {
        held.flatMap { $0.recurrent.values.flatMap { $0 } }.reduce(0) { $0 + $1.nbytes }
    }

    /// Whether `rewind` can ever succeed.
    public var canRewind: Bool { !layers.isEmpty && opaque == 0 && (!trimmable.isEmpty || recurrent.isEmpty) }

    /// Every layer trims by position: no snapshot is needed, and a rewind is exact.
    public var isFullyTrimmable: Bool { opaque == 0 && recurrent.isEmpty }

    /// Remembers the recurrent layers as they stand with `tokenCount` tokens in the cache.
    /// Taken right after a prompt has been prefilled, before the first generated token — or, `pinned`,
    /// where a turn's ask ends, kept outside the ring until the next pin replaces it — or, `anchored`, where the stable
    /// prefix ends, kept until the next anchor replaces it.
    public func snapshot(at tokenCount: Int, pinned pin: Bool = false, anchored: Bool = false) {
        guard canRewind else { return }
        // A cache that trims by position rewinds exactly without snapshots; its anchor holds only where the stable
        // prefix ends, for a keep up to it (`SlotExtent.anchor`).
        if recurrent.isEmpty {
            if anchored { anchor = Snapshot(tokenCount: tokenCount, recurrent: [:]) }
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
        if let kept = anchor, kept.tokenCount > tokenCount { anchor = nil }
        let taken = Snapshot(tokenCount: tokenCount, recurrent: state)
        if anchored {
            anchor = taken
        } else if pin {
            pinned = taken
        } else {
            snapshots.append(taken)
            if snapshots.count > ring { snapshots.removeFirst(snapshots.count - ring) }
        }
    }

    /// Rewinds to exactly `target` when every layer trims by position, otherwise to the newest
    /// snapshot at or before `target`. Returns the token count the cache now holds, or nil when
    /// nothing fits — the caller then starts a fresh cache.
    @discardableResult
    public func rewind(toAtMost target: Int) -> Int? { rewindPoint(toAtMost: target)?.position }

    /// `rewind(toAtMost:)`, saying which point it went back to: the anchor, the pin or the ring when they stand at the
    /// same count, in that order, or a trim.
    public func rewindPoint(toAtMost target: Int) -> (position: Int, point: RewindPoint)? {
        guard canRewind else { return nil }
        let current = position
        if isFullyTrimmable {
            let count = max(0, min(target, current))
            trim(by: current - count)
            if let kept = anchor, kept.tokenCount > count { anchor = nil }
            return (count, .trim)
        }
        guard let snapshot = held.filter({ $0.tokenCount <= target }).max(by: { $0.tokenCount < $1.tokenCount }),
              snapshot.tokenCount <= current else { return nil }
        let point: RewindPoint = anchor?.tokenCount == snapshot.tokenCount ? .anchor
            : pinned?.tokenCount == snapshot.tokenCount ? .pin : .ring
        for (index, arrays) in snapshot.recurrent {
            recurrent[index]?.state = arrays
        }
        trim(by: current - snapshot.tokenCount)
        snapshots.removeAll { $0.tokenCount > snapshot.tokenCount }
        if let kept = pinned, kept.tokenCount > snapshot.tokenCount { pinned = nil }
        if let kept = anchor, kept.tokenCount > snapshot.tokenCount { anchor = nil }
        return (snapshot.tokenCount, point)
    }

    /// Drops one snapshot: the ring's oldest while the ring holds any, then the pin, the anchor last. False when none is
    /// kept.
    ///
    /// PIN: THE RING GOES FIRST (2026-11-20). The pin is where the next turn parts from this one and the anchor where a
    /// rewritten history does; the ring's are this turn's rounds, the newest of which the next round needs and the
    /// oldest of which nothing does. Dropped by age instead, the pin went before the ring's newer snapshots, and the
    /// next turn read this one again from the anchor.
    @discardableResult
    public func dropOldestSnapshot() -> Bool {
        if !snapshots.isEmpty {
            snapshots.removeFirst()
            return true
        }
        if pinned != nil {
            pinned = nil
            return true
        }
        guard anchor != nil else { return false }
        anchor = nil
        return true
    }

    /// The cache as it stood at the anchor, for a keep up to it: its layers cut to the anchor's count (the recurrent ones
    /// the anchor's own state) and the anchor as its only snapshot. Nil without an anchor.
    func trimmedToAnchor() -> (layers: [[MLXArray]], anchor: Snapshot)? {
        guard let anchor, anchor.tokenCount <= position else { return nil }
        let count = anchor.tokenCount
        let cut = layers.enumerated().map { index, layer -> [MLXArray] in
            if let held = anchor.recurrent[index] { return held }
            return layer.state.map { $0[.ellipsis, ..<count, 0...] }
        }
        return (cut, anchor)
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
        if let kept = anchor, kept.tokenCount > now { anchor = nil }
    }

    private func trim(by n: Int) {
        guard n > 0 else { return }
        for index in trimmable { layers[index].trim(n) }
    }
}
