//  WHAT: The rows decoding together: their layer caches side by side — a `BatchKVCache` per
//        attention layer, a batched `MambaCache` per recurrent layer — and what each row still owes.
//  IN:   ParallelEngine, on its thread only.
//  PIN:  The first row's caches become the batch's own, uncopied: `ArraysCache.batchSize` answers 1
//        for an empty cache, so extending an empty batch would zero-fill a phantom row.

import Foundation
import MLX
import MLXLMCommon

enum LayerKind {
    case attention, recurrent
}

/// What the model's `newCache` says each layer keeps, checked once at engine start.
struct CacheLayout {
    let kinds: [LayerKind]

    init(sample layers: [KVCache]) throws {
        kinds = try layers.map { layer in
            if layer is ArraysCache { return .recurrent }
            if layer is KVCacheSimple { return .attention }
            throw ParallelError.unsupportedLayer(String(describing: type(of: layer)))
        }
    }
}

final class GenerationBatch {
    let layout: CacheLayout
    private(set) var caches: [KVCache] = []
    private(set) var rows: [Row] = []

    init(layout: CacheLayout) {
        self.layout = layout
    }

    var isEmpty: Bool { rows.isEmpty }
    var count: Int { rows.count }

    /// Adds a row with its prefilled, single-sequence layers.
    func extend(_ row: Row, layers: [KVCache]) {
        precondition(layers.count == layout.kinds.count)
        if rows.isEmpty {
            caches = layers.enumerated().map { index, layer in
                switch layout.kinds[index] {
                case .attention:
                    let batch = BatchKVCache()
                    batch.extend(layer as! KVCacheSimple)
                    return batch
                case .recurrent:
                    return layer
                }
            }
        } else {
            for (index, layer) in layers.enumerated() {
                switch layout.kinds[index] {
                case .attention: (caches[index] as! BatchKVCache).extend(layer as! KVCacheSimple)
                case .recurrent: (caches[index] as! ArraysCache).extend(other: layer as! ArraysCache)
                }
            }
        }
        rows.append(row)
    }

    /// One row's layers as a sequence of its own: contiguous copies, nothing shared with the batch.
    func extract(row index: Int) -> [KVCache] {
        caches.enumerated().map { layerIndex, cache in
            switch layout.kinds[layerIndex] {
            case .attention:
                return (cache as! BatchKVCache).extract(row: index)
            case .recurrent:
                let arrays = cache as! ArraysCache
                let single = MambaCache()
                single.state = arrays.state.map { contiguous($0[index ..< (index + 1)]) }
                single.offset = arrays.offset
                return single
            }
        }
    }

    /// Drops the rows at `remove`, keeping the others' order.
    func remove(_ remove: Set<Int>) {
        guard !remove.isEmpty else { return }
        let keep = rows.indices.filter { !remove.contains($0) }
        rows = keep.map { rows[$0] }
        if keep.isEmpty {
            caches = []
            return
        }
        let indices = MLXArray(keep.map(Int32.init))
        for (index, cache) in caches.enumerated() {
            switch layout.kinds[index] {
            case .attention: (cache as! BatchKVCache).filter(keep: keep)
            case .recurrent: (cache as! ArraysCache).filter(batchIndices: indices)
            }
        }
    }

    /// Whether one sampler can serve every row at once.
    var sharesSampling: Bool {
        guard let first = rows.first else { return false }
        return rows.allSatisfy { $0.samplingKey == first.samplingKey }
    }
}

/// One stream decoding in the batch, or waiting to.
final class Row {
    let id: StreamID
    let request: StreamRequest
    let continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation
    let sampler: LogitSampler
    var processor: LogitProcessor?
    let samplingKey: String
    /// The last sampled token, not yet fed to the model.
    var pending: Int32 = 0
    /// Generated tokens the cache holds (fed), in order.
    var fed: [Int] = []
    var emitted = 0
    var removed = false
    var finish: FinishReason?
    let queuedAt: Date
    var admittedAt: Date?
    var prefillSeconds: TimeInterval = 0
    var reused = 0
    var batchSteps = 0
    var batchRowsSeen = 0
    /// Tokens emitted at the last stats tick, and when, for a per-stream rate.
    var rateMark: (count: Int, at: Date)?
    var tokensPerSecond: Double = 0

    init(id: StreamID, request: StreamRequest, continuation: AsyncThrowingStream<StreamEvent, Error>.Continuation) {
        self.id = id
        self.request = request
        self.continuation = continuation
        self.sampler = request.sampling.sampler()
        self.processor = request.sampling.processor()
        let s = request.sampling
        samplingKey = "\(s.temperature)/\(s.topP)/\(s.topK)/\(s.minP)/\(s.seed.map(String.init) ?? "-")"
        queuedAt = Date()
    }

    var slot: SlotID? { request.slot }
    var promptCount: Int { request.tokens.count }
}
