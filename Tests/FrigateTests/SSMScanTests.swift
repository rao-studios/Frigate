import Foundation
import MLX
@testable import MLXLLM
import Testing

/// MLX graph ops abort on Darwin unless a metallib is beside the test bundle. Same gate as Gemma4SanitizeTests.
private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

/// The fused Mamba-2 prefill scan (`ssmScanKernel`) against the chunked SSD form it replaces (`ssmAttn`): the same y and
/// the same carried state, from a zero state and from one carried in, across a 256-token step boundary.
@Suite("Mamba-2 fused prefill scan", .enabled(if: mlxFunctional))
struct SSMScanTests {
    private func relative(_ a: MLXArray, _ b: MLXArray) -> Float {
        let a = a.asType(.float32), b = b.asType(.float32)
        return (MLX.sqrt(MLX.sum((a - b) * (a - b))) / MLX.maximum(MLX.sqrt(MLX.sum(b * b)), MLXArray(Float(1e-12)))).item(Float.self)
    }

    private func inputs(l: Int, h: Int, dh: Int, g: Int, ds: Int, dtype: DType, carried: Bool)
        -> (x: MLXArray, A: MLXArray, B: MLXArray, C: MLXArray, D: MLXArray, dt: MLXArray, bias: MLXArray, state: MLXArray?)
    {
        MLXRandom.seed(11)
        let x = MLXRandom.normal([1, l, h, dh]).asType(dtype)
        let B = (MLXRandom.normal([1, l, g, ds]) * 0.3).asType(dtype)
        let C = (MLXRandom.normal([1, l, g, ds]) * 0.3).asType(dtype)
        let dt = (MLXRandom.normal([1, l, h]) * 0.5 - 1).asType(dtype)
        let A = MLXRandom.uniform(low: -1, high: 1.5, [h])
        let D = MLXRandom.normal([h]).asType(dtype)
        let bias = MLXRandom.normal([h]) * 0.1
        let state = carried ? MLXRandom.normal([1, h, dh, ds]) * 0.5 : nil
        return (x, A, B, C, D, dt, bias, state)
    }

    @Test(arguments: [false, true])
    func theFusedScanMatchesTheChunkedForm(carried: Bool) throws {
        // fp32 across a step boundary (256 + 44); then Nemotron's widths in bf16 (Dh 64, Ds 128, 2 heads to a group).
        for (l, h, dh, g, ds, dtype, tolerance) in [
            (300, 4, 8, 2, 32, DType.float32, Float(1e-4)),
            (300, 8, 64, 4, 128, DType.bfloat16, Float(2e-2)),
        ] {
            let i = inputs(l: l, h: h, dh: dh, g: g, ds: ds, dtype: dtype, carried: carried)
            let limit: (Float, Float) = (0.001, 100.0)
            let (refY, refState) = ssmAttn(
                x: i.x, ALog: i.A, B: i.B, C: i.C, D: i.D, dt: i.dt, dtBias: i.bias, state: i.state, timeStepLimit: limit)
            let fused = try #require(ssmScanKernel(
                hiddenStates: i.x, ALog: i.A, B: i.B, C: i.C, D: i.D, dt: i.dt, dtBias: i.bias, state: i.state,
                timeStepLimit: limit))
            #expect(fused.0.shape == refY.shape && fused.0.dtype == refY.dtype)
            #expect(fused.1.shape == refState.shape)
            #expect(relative(fused.0, refY) < tolerance, "y, \(dtype), carried \(carried)")
            #expect(relative(fused.1, refState) < tolerance, "state, \(dtype), carried \(carried)")
        }
    }

    /// One prompt takes the fused scan through `ssmUpdate`; a state width it cannot spread over the SIMD lanes does not.
    @Test func ssmUpdateRoutesOnePromptToTheFusedScan() {
        let i = inputs(l: 64, h: 4, dh: 8, g: 2, ds: 32, dtype: .float32, carried: false)
        let (y, _) = ssmUpdate(hiddenStates: i.x, ALog: i.A, B: i.B, C: i.C, D: i.D, dt: i.dt, dtBias: i.bias)
        let (refY, _) = ssmAttn(x: i.x, ALog: i.A, B: i.B, C: i.C, D: i.D, dt: i.dt, dtBias: i.bias)
        #expect(relative(y, refY) < 1e-4)
        let odd = inputs(l: 8, h: 2, dh: 4, g: 1, ds: 24, dtype: .float32, carried: false)
        #expect(ssmScanKernel(hiddenStates: odd.x, ALog: odd.A, B: odd.B, C: odd.C, D: odd.D, dt: odd.dt, dtBias: odd.bias,
                              state: nil, timeStepLimit: (0.001, 100.0)) == nil)
    }
}
