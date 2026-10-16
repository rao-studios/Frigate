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
        snapshots.flatMap { $0.recurrent.values.flatMap { $0 } }.reduce(0) { $0 + $1.nbytes }
    }

    /// Whether `rewind` can ever succeed.
    public var canRewind: Bool { !layers.isEmpty && opaque == 0 && (!trimmable.isEmpty || recurrent.isEmpty) }

    /// Every layer trims by position: no snapshot is needed, and a rewind is exact.
    public var isFullyTrimmable: Bool { opaque == 0 && recurrent.isEmpty }

    /// Remembers the recurrent layers as they stand with `tokenCount` tokens in the cache.
    /// Taken right after a prompt has been prefilled, before the first generated token.
    public func snapshot(at tokenCount: Int) {
        guard !recurrent.isEmpty, canRewind else { return }
        var state: [Int: [MLXArray]] = [:]
        for (index, layer) in recurrent {
            let arrays = layer.state
            // A layer with no state yet cannot be restored; a snapshot of it would restore nothing.
            guard !arrays.isEmpty else { return }
            state[index] = arrays
        }
        snapshots.removeAll { $0.tokenCount >= tokenCount }
        snapshots.append(Snapshot(tokenCount: tokenCount, recurrent: state))
        if snapshots.count > ring { snapshots.removeFirst(snapshots.count - ring) }
    }

    /// Rewinds to exactly `target` when every layer trims by position, otherwise to the newest
    /// snapshot at or before `target`. Returns the token count the cache now holds, or nil when
    /// nothing fits — the caller then starts a fresh cache.
    @discardableResult
    public func rewind(toAtMost target: Int) -> Int? {
        guard canRewind else { return nil }
        let current = position
        if isFullyTrimmable {
            let count = max(0, min(target, current))
            trim(by: current - count)
            return count
        }
        guard let snapshot = snapshots.last(where: { $0.tokenCount <= target }), snapshot.tokenCount <= current else { return nil }
        for (index, arrays) in snapshot.recurrent {
            recurrent[index]?.state = arrays
        }
        trim(by: current - snapshot.tokenCount)
        snapshots.removeAll { $0.tokenCount > snapshot.tokenCount }
        return snapshot.tokenCount
    }

    /// Drops the oldest snapshot: a rewind can then reach only the newer ones. False when none is kept.
    @discardableResult
    public func dropOldestSnapshot() -> Bool {
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
    }

    private func trim(by n: Int) {
        guard n > 0 else { return }
        for index in trimmable { layers[index].trim(n) }
    }
}
