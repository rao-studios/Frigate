//  WHAT: Inkling-2's fused decode attention (Inkling2DecodeAttention) against the plain ops it replaces — the float32
//        reference `Inkling2Attention.decode` runs for a batch — for a global visit's keys (newest last, a padded row,
//        a slice of a longer buffer as the cache hands it) and a sliding visit's ring (a full ring and one a short
//        sequence has not filled), one row and two.

import Foundation
import MLX
import Testing

@testable import MLXLLM

private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

@Suite("Inkling-2 decode attention", .enabled(if: mlxFunctional))
struct Inkling2DecodeAttentionTests {
    let hq = 24, hkv = 8, d = 128, r = 1024, scale = Float(1 / sqrt(128.0))

    /// `Inkling2Attention.decode`'s batched path, verbatim: float32 logits, the bias by distance, the hidden keys at
    /// −∞, the sink in the softmax's denominator.
    func reference(_ q: MLXArray, _ keys: MLXArray, _ values: MLXArray, bias: MLXArray, sinks: MLXArray,
                   distance: MLXArray, visible: MLXArray) -> MLXArray {
        let b = q.dim(0), s = keys.dim(2)
        let dist = broadcast(distance.expandedDimensions(axis: 1), to: [b, hq, s])
        var biased = takeAlong(bias[0..., 0], clip(dist, min: 0, max: r - 1), axis: -1).asType(.float32)
        biased = which(dist .< r, biased, MLXArray(Float(0)))
        let mask = which(visible.expandedDimensions(axis: 1), biased, MLXArray(-Float.infinity)).reshaped(b, hkv, -1, 1, s)
        let q32 = q.asType(.float32).reshaped(b, hkv, -1, 1, d)
        let k32 = keys.asType(.float32).expandedDimensions(axis: 2)
        let logits = matmul(q32, k32.swappedAxes(-1, -2)) * scale + mask
        let sinkLogits = sinks.asType(.float32).transposed(0, 2, 1).reshaped(b, hkv, -1, 1, 1)
        let weights = softmax(concatenated([logits, sinkLogits], axis: -1), axis: -1)[.ellipsis, ..<s]
        let out = matmul(weights, values.asType(.float32).expandedDimensions(axis: 2))
        return out.reshaped(b, hq, 1, d).transposed(0, 2, 1, 3)
    }

    func inputs(_ b: Int, _ s: Int, capacity: Int) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        MLXRandom.seed(7)
        // As the attention hands them over: q a transposed [B,1,Hq,D], the keys a slice of the cache's longer buffer.
        let q = (MLXRandom.normal([b, 1, hq, d]) * 2).asType(.bfloat16).transposed(0, 2, 1, 3)
        let keys = MLXRandom.normal([b, hkv, capacity, d]).asType(.bfloat16)[.ellipsis, ..<s, 0...]
        let values = MLXRandom.normal([b, hkv, capacity, d]).asType(.bfloat16)[.ellipsis, ..<s, 0...]
        let bias = (MLXRandom.normal([b, 1, hq, r]) * 2).asType(.bfloat16)
        let sinks = MLXRandom.normal([b, 1, hq]).asType(.bfloat16)
        return (q, keys, values, bias, sinks)
    }

    func close(_ got: MLXArray, _ want: MLXArray) -> Float {
        abs(got.asType(.float32) - want).max().item(Float.self)
    }

    @Test("a global visit's keys, newest last, one row and a padded pair", arguments: [(1, 700), (1, 3000), (2, 3000)])
    func global(_ shape: (Int, Int)) {
        let (b, s) = shape
        let (q, keys, values, bias, sinks) = inputs(b, s, capacity: s + 256)
        // The second row's first 1,100 columns are the first row's padding.
        let keyStart = MLXArray(Array([0, 1100].prefix(b)).map(Int32.init))
        let columns = MLXArray(Int32(0) ..< Int32(s))
        let want = reference(q, keys, values, bias: bias, sinks: sinks,
                             distance: broadcast(MLXArray(Int32(s - 1)) - columns, to: [b, s]),
                             visible: columns.expandedDimensions(axis: 0) .>= keyStart.expandedDimensions(axis: 1))
        let got = Inkling2Kernels.decodeAttention(q, keys, values, rel: bias, sinks: sinks, scale: Inkling2Constant([scale]),
                                                  layout: .newestLast(keyStart: keyStart))
        #expect(got.shape == [b, 1, hq, d])
        #expect(close(got, want) < 0.02, "bf16 output, float32 reference")
    }

    @Test("a sliding visit's ring, full and not yet filled", arguments: [[5000], [300], [5000, 300]])
    func ring(_ rows: [Int]) {
        let w = 1024, b = rows.count
        let (q, keys, values, bias, sinks) = inputs(b, w, capacity: w)
        let pos = MLXArray(rows.map(Int32.init))
        let slots = MLXArray(Int32(0) ..< Int32(w)).expandedDimensions(axis: 0)
        let p = pos.expandedDimensions(axis: 1)
        let back = remainder(remainder(p - slots, Int32(w)) + Int32(w), Int32(w))
        let want = reference(q, keys, values, bias: bias, sinks: sinks, distance: back, visible: (p - back) .>= 0)
        let got = Inkling2Kernels.decodeAttention(q, keys, values, rel: bias, sinks: sinks, scale: Inkling2Constant([scale]),
                                                  layout: .ring(positions: pos))
        #expect(close(got, want) < 0.02, "bf16 output, float32 reference")
    }
}
