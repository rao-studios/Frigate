//  WHAT: Inkling-2's decode convolutions (Inkling2DecodeConvolution) against the plain ops they replace — the short
//        convolution and the global layers' dynamic one, for one token on a kept state — output and next state, one row
//        and two, a channel count a threadgroup does not divide.

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

@Suite("Inkling-2 decode convolutions", .enabled(if: mlxFunctional))
struct Inkling2DecodeConvolutionTests {
    let k = 4

    /// A kept state as the model leaves it: earlier tokens, bfloat16 values held in float32.
    func state(_ b: Int, _ c: Int) -> MLXArray {
        MLXRandom.normal([b, k - 1, c]).asType(.bfloat16).asType(.float32)
    }

    func largest(_ got: MLXArray, _ want: MLXArray) -> Float {
        abs(got.asType(.float32) - want.asType(.float32)).max().item(Float.self)
    }

    @Test("the short convolution, as conv1d in the weight's dtype", arguments: [(1, 2048), (2, 1000)])
    func short(_ shape: (Int, Int)) {
        let (b, c) = shape
        MLXRandom.seed(11)
        let x = MLXRandom.normal([b, 1, c]).asType(.bfloat16)
        let kept = state(b, c)
        let weight = (MLXRandom.normal([c, k, 1]) * 0.5).asType(.bfloat16)
        // Inkling2ShortConvolution's plain path, verbatim.
        let window = concatenated([kept, x.asType(.float32)], axis: 1)
        let out = conv1d(window.asType(weight.dtype), weight, groups: c)
        let want = ((out.asType(.float32) + x.asType(.float32)) / sqrt(Float(k + 1))).asType(x.dtype)
        let (got, next) = Inkling2Kernels.shortConvolution(x, state: kept, weight: weight, kernel: k)
        #expect(got.shape == [b, 1, c] && got.dtype == .bfloat16)
        #expect(largest(got, want) < 0.02)
        #expect(largest(next, window[0..., 1..., 0...]) == 0)
    }

    @Test("the dynamic convolution, a static weight per channel plus a projected one per head",
          arguments: [(1, 1024), (2, 1024)])
    func dynamic(_ shape: (Int, Int)) {
        let (b, c) = shape
        let h = 8
        MLXRandom.seed(13)
        let x = MLXRandom.normal([b, 1, c]).asType(.bfloat16)
        let kept = state(b, c)
        let fixed = (MLXRandom.normal([k, c]) * 0.5).asType(.bfloat16)
        let projected = (MLXRandom.normal([b, 1, h * k]) * 0.5).asType(.bfloat16)
        // Inkling2DynamicConvolution's plain path, verbatim, for one token.
        let dynamic = projected.asType(.float32).reshaped(b, 1, h, 1, k)
        let statics = fixed.asType(.float32).T.reshaped(h, c / h, k)
        let weights = statics + dynamic
        let window = concatenated([kept, x.asType(.float32)], axis: 1)
        let windows = stacked((0 ..< k).map { lag in window[0..., (k - 1 - lag) ..< (k - lag), 0...] }, axis: -1)
            .reshaped(b, 1, h, c / h, k)
        let out = (windows * weights).sum(axis: -1).reshaped(b, 1, c)
        let want = ((out + x.asType(.float32)) / sqrt(Float(2 * k + 1))).asType(x.dtype)
        let (got, next) = Inkling2Kernels.dynamicConvolution(
            x, state: kept, fixed: fixed, dynamic: projected, heads: h, kernel: k)
        #expect(got.shape == [b, 1, c] && got.dtype == .bfloat16)
        #expect(largest(got, want) < 0.02)
        #expect(largest(next, window[0..., 1..., 0...]) == 0)
    }
}
