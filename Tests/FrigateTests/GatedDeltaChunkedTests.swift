//  WHAT: The chunked Gated DeltaNet (MLXLMCommon/GatedDelta.swift, `gatedDeltaChunked`) against the Metal kernel and
//        the per-position ops: the same y and final state from a carried state, over a length that is not a whole
//        number of chunks; the same gradients as the ops, which autograd has always differentiated; the halving inverse
//        against `(I - n)·inverse = I`; and a tiny Qwen3.5 whose logits under `GatedDeltaPath.differentiable` are its
//        kernel logits, with a gradient that reaches its input embeddings.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

@Suite("Gated DeltaNet, chunked", .enabled(if: mlxFunctional), .serialized)
struct GatedDeltaChunkedTests {
    /// Inputs as Qwen3.5's layer hands them over: q and k RMS-normed and scaled, a decay in (0, 1), β in (0, 1).
    private func inputs(T: Int, seed: UInt64 = 3) -> (q: MLXArray, k: MLXArray, v: MLXArray, logG: MLXArray, beta: MLXArray, state: MLXArray) {
        MLXRandom.seed(seed)
        let (B, Hk, Hv, Dk, Dv) = (1, 2, 4, 32, 32)
        let scale = pow(Float(Dk), -0.5)
        let q = MLXFast.rmsNorm(MLXRandom.normal([B, T, Hk, Dk]), weight: MLXArray.mlxNone, eps: 1e-6) * (scale * scale)
        let k = MLXFast.rmsNorm(MLXRandom.normal([B, T, Hk, Dk]), weight: MLXArray.mlxNone, eps: 1e-6) * scale
        let v = MLXRandom.normal([B, T, Hv, Dv])
        let logG = -exp(MLXRandom.normal([B, T, Hv]) * 0.5 - 1.5)
        let beta = sigmoid(MLXRandom.normal([B, T, Hv]))
        let state = MLXRandom.normal([B, Hv, Dv, Dk]) * 0.1
        return (q, k, v, logG, beta, state)
    }

    private func gap(_ a: MLXArray, _ b: MLXArray) -> Float { abs(a - b).max().item(Float.self) }

    @Test(arguments: [1, 63, 64, 150])
    func theChunksAreTheKernel(T: Int) {
        let x = inputs(T: T)
        let (y, state) = gatedDeltaChunked(q: x.q, k: x.k, v: x.v, logG: x.logG, beta: x.beta, state: x.state)
        let (yk, stateK) = gatedDeltaKernel(q: x.q, k: x.k, v: x.v, g: exp(x.logG), beta: x.beta, state: x.state)
        let (yo, stateO) = gatedDeltaOps(q: x.q, k: x.k, v: x.v, g: exp(x.logG), beta: x.beta, state: x.state)
        #expect(gap(y, yk) < 1e-4, "y against the kernel at T=\(T)")
        #expect(gap(state, stateK) < 1e-4, "state against the kernel at T=\(T)")
        #expect(gap(y, yo) < 1e-4, "y against the ops at T=\(T)")
        #expect(gap(state, stateO) < 1e-4, "state against the ops at T=\(T)")
    }

    @Test func theGradientsAreTheOps() {
        let x = inputs(T: 70)
        let target = MLXRandom.normal([1, 70, 4, 32])
        func loss(_ chunked: Bool) -> (MLXArray, [MLXArray]) {
            let f: ([MLXArray]) -> [MLXArray] = { a in
                let (y, s) = chunked
                    ? gatedDeltaChunked(q: a[0], k: a[1], v: a[2], logG: x.logG, beta: a[3], state: x.state)
                    : gatedDeltaOps(q: a[0], k: a[1], v: a[2], g: exp(x.logG), beta: a[3], state: x.state)
                return [(y * target).sum() + s.sum() * 0.1]
            }
            let (value, grads) = valueAndGrad(f)([x.q, x.k, x.v, x.beta])
            return (value[0], grads)
        }
        let (chunkedLoss, chunkedGrads) = loss(true)
        let (opsLoss, opsGrads) = loss(false)
        #expect(abs(chunkedLoss.item(Float.self) - opsLoss.item(Float.self)) < 1e-3)
        for (name, (a, b)) in zip(["q", "k", "v", "beta"], zip(chunkedGrads, opsGrads)) {
            let relative = gap(a, b) / max(abs(b).max().item(Float.self), 1e-6)
            #expect(relative < 1e-3, "d loss / d \(name): relative gap \(relative)")
        }
    }

    @Test func theHalvingInverseInverts() {
        MLXRandom.seed(5)
        let c = 64
        let rows = MLXArray(0 ..< Int32(c)).reshaped(c, 1)
        let cols = MLXArray(0 ..< Int32(c)).reshaped(1, c)
        let n = MLXRandom.normal([2, 3, c, c]) * 0.2 * (rows .> cols).asType(.float32)
        let identity = MLXArray.eye(c)
        let product = matmul(identity - n, unitLowerInverse(n))
        #expect(gap(product, broadcast(identity, to: product.shape)) < 1e-4)
    }

    @Test func aQwen35ReadsTheSameAndItsInputTakesAGradient() throws {
        let model = try SoftPromptTests.qwen35()
        #expect(model.takesInputGradients)
        let ids = MLXArray((0 ..< 80).map { Int32(($0 * 37) % 61 + 1) }).reshaped(1, 80)
        let kernel = model(ids, cache: nil)
        let chunked = GatedDeltaPath.$differentiable.withValue(true) { model(ids, cache: nil) }
        #expect(gap(kernel, chunked) < 1e-3)

        let embedded = model.embed(ids)
        let grads = GatedDeltaPath.$differentiable.withValue(true) {
            grad({ (e: MLXArray) -> MLXArray in model(embeddings: e, cache: nil)[0..., 40..., 0...].logSumExp(axis: -1).sum() })(embedded)
        }
        eval(grads)
        let early = abs(grads[0..., ..<40, 0...]).sum().item(Float.self)
        #expect(early.isFinite && early > 0, "a loss on later positions reaches earlier embeddings through the recurrence")
    }

    /// A quantized mixture of experts, as Qwopus is: the router's choice takes no gradient (`stopGradient`) and the
    /// experts' `gather_qmm` passes one to its input, so a loss reaches the embeddings rather than failing in MLX.
    @Test func aQuantizedMixtureOfExpertsPassesTheGradientThrough() throws {
        MLXRandom.seed(4)
        let json = """
        {
          "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 64, "num_attention_heads": 2,
          "num_key_value_heads": 1, "head_dim": 32, "linear_num_value_heads": 2, "linear_num_key_heads": 2,
          "linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4, "vocab_size": 64,
          "tie_word_embeddings": false, "full_attention_interval": 4, "num_experts": 4, "num_experts_per_tok": 2,
          "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64
        }
        """
        let model = Qwen35TextModel(try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8)))
        quantize(model: model, groupSize: 32, bits: 4)
        eval(model)
        let ids = MLXArray((0 ..< 70).map { Int32(($0 * 37) % 61 + 1) }).reshaped(1, 70)
        let embedded = model.embed(ids)
        let grads = try GatedDeltaPath.$differentiable.withValue(true) {
            try withError {
                let g = grad({ (e: MLXArray) -> MLXArray in model(embeddings: e, cache: nil).logSumExp(axis: -1).sum() })(embedded)
                eval(g)
                return g
            }
        }
        let size = abs(grads).sum().item(Float.self)
        #expect(size.isFinite && size > 0)
    }

    /// A gradient traced on one thread beside compiled forwards (`silu`) on another: the two used to take `evalLock`
    /// and a compiled function's lock in opposite orders and wait on each other for good (Transforms+Compile.swift).
    @Test(.timeLimit(.minutes(1))) func aGradientBesideAForwardOnAnotherThreadFinishes() async throws {
        let model = try SoftPromptTests.qwen35()
        let other = try SoftPromptTests.qwen35(seed: 9)
        let ids = MLXArray((0 ..< 80).map { Int32(($0 * 37) % 61 + 1) }).reshaped(1, 80)
        nonisolated(unsafe) let (lent, beside, embedded) = (model, other, model.embed(ids))
        async let gradients: Void = Task.detached {
            for _ in 0 ..< 20 {
                let g = GatedDeltaPath.$differentiable.withValue(true) {
                    grad({ (e: MLXArray) -> MLXArray in lent(embeddings: e, cache: nil).logSumExp(axis: -1).sum() })(embedded)
                }
                eval(g)
            }
        }.value
        nonisolated(unsafe) let tokens = ids
        async let forwards: Void = Task.detached {
            for _ in 0 ..< 200 { eval(beside(tokens, cache: nil)) }
        }.value
        _ = await (gradients, forwards)
    }
}
