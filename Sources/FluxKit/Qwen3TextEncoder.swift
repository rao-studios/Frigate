//
//  Qwen3TextEncoder.swift
//  FluxKit
//
//  Qwen3-4B forward pass used purely as FLUX.2 Klein's text encoder: runs the first
//  N decoder layers and returns the hidden states after layers 9/18/27 (matching
//  diffusers' `hidden_states[k]` indexing, where index 0 is the embedding output),
//  interleaved per-token into the DiT's 7680-dim context. Layers past the deepest tap
//  are never built or run. Ported from mflux's Qwen3TextEncoder.
//

import Foundation
import MLX
import MLXFast
import MLXNN

public final class Qwen3TextEncoder {
    public struct Config {
        public var hiddenSize = 2560
        public var numLayers = 36
        public var numHeads = 32
        public var numKVHeads = 8
        public var headDim = 128
        public var intermediateSize = 9728
        public var ropeTheta: Float = 1_000_000
        public var rmsEps: Float = 1e-6
        public init() {}
    }

    final class Layer {
        let inputNorm: ExplicitRMSNorm
        let postAttnNorm: ExplicitRMSNorm
        let qProj: Linear
        let kProj: Linear
        let vProj: Linear
        let oProj: Linear
        let qNorm: ExplicitRMSNorm
        let kNorm: ExplicitRMSNorm
        let gateProj: Linear
        let upProj: Linear
        let downProj: Linear

        init(store: TensorStore, prefix: String, config: Config) throws {
            let attnDim = config.numHeads * config.headDim
            let kvDim = config.numKVHeads * config.headDim
            inputNorm = ExplicitRMSNorm(weight: try store.rmsNormWeight("\(prefix).input_layernorm"), eps: config.rmsEps)
            postAttnNorm = ExplicitRMSNorm(weight: try store.rmsNormWeight("\(prefix).post_attention_layernorm"), eps: config.rmsEps)
            qProj = try store.anyLinear("\(prefix).self_attn.q_proj", inFeatures: config.hiddenSize)
            kProj = try store.anyLinear("\(prefix).self_attn.k_proj", inFeatures: config.hiddenSize)
            vProj = try store.anyLinear("\(prefix).self_attn.v_proj", inFeatures: config.hiddenSize)
            oProj = try store.anyLinear("\(prefix).self_attn.o_proj", inFeatures: attnDim)
            qNorm = ExplicitRMSNorm(weight: try store.rmsNormWeight("\(prefix).self_attn.q_norm"), eps: config.rmsEps)
            kNorm = ExplicitRMSNorm(weight: try store.rmsNormWeight("\(prefix).self_attn.k_norm"), eps: config.rmsEps)
            gateProj = try store.anyLinear("\(prefix).mlp.gate_proj", inFeatures: config.hiddenSize)
            upProj = try store.anyLinear("\(prefix).mlp.up_proj", inFeatures: config.hiddenSize)
            downProj = try store.anyLinear("\(prefix).mlp.down_proj", inFeatures: config.intermediateSize)
            _ = kvDim
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray, cos: MLXArray, sin: MLXArray, config: Config) -> MLXArray {
            let (b, s, _) = (x.dim(0), x.dim(1), x.dim(2))

            var h = inputNorm(x)
            var q = qProj(h).reshaped(b, s, config.numHeads, config.headDim)
            var k = kProj(h).reshaped(b, s, config.numKVHeads, config.headDim)
            var v = vProj(h).reshaped(b, s, config.numKVHeads, config.headDim)

            q = qNorm(q).transposed(0, 2, 1, 3)
            k = kNorm(k).transposed(0, 2, 1, 3)
            v = v.transposed(0, 2, 1, 3)

            (q, k) = Self.applyRotary(q: q, k: k, cos: cos, sin: sin)

            let scale = 1.0 / Float(config.headDim).squareRoot()
            var attn = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: mask.asType(q.dtype))
            attn = attn.transposed(0, 2, 1, 3).reshaped(b, s, config.numHeads * config.headDim)
            h = x + oProj(attn)

            let normed = postAttnNorm(h)
            let mlp = downProj(silu(gateProj(normed)) * upProj(normed))
            return h + mlp
        }

        /// Standard HF rotate-half rotary (cos/sin already duplicated across halves).
        static func applyRotary(q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
            func rotateHalf(_ x: MLXArray) -> MLXArray {
                let half = x.dim(-1) / 2
                let x1 = x[.ellipsis, ..<half]
                let x2 = x[.ellipsis, half...]
                return concatenated([-x2, x1], axis: -1)
            }
            let qOut = q * cos + rotateHalf(q) * sin
            let kOut = k * cos + rotateHalf(k) * sin
            return (qOut.asType(q.dtype), kOut.asType(k.dtype))
        }
    }

    public let config: Config
    let embedTokens: TokenEmbedding
    let layers: [Layer]
    /// hidden_states indices to tap (1-based over layer outputs; 0 = embeddings).
    let tapIndices: [Int]

    /// Builds only the layers needed to reach the deepest tap.
    public init(componentDir: URL, taps: [Int] = [9, 18, 27], config: Config = Config()) throws {
        self.config = config
        self.tapIndices = taps
        let store = try TensorStore(componentDir: componentDir)
        if store.contains("embed_tokens.scales") {
            embedTokens = .quantized(try store.quantizedEmbedding("embed_tokens", dimensions: config.hiddenSize))
        } else {
            embedTokens = .full(try store.take("embed_tokens.weight"))
        }
        let deepest = taps.max() ?? config.numLayers
        var built: [Layer] = []
        for i in 0..<deepest {
            built.append(try Layer(store: store, prefix: "layers.\(i)", config: config))
        }
        layers = built
        store.drain()
    }

    /// Returns (B, S, taps.count * hiddenSize): per-token concatenation of the tapped
    /// layers' hidden states, in tap order.
    public func promptEmbeddings(inputIDs: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let b = inputIDs.dim(0)
        let s = inputIDs.dim(1)

        var hidden = embedTokens(inputIDs).asType(Flux2Pipeline.activationDType)

        // Additive causal + padding mask, [B, 1, S, S].
        let idx = MLXArray(0..<Int32(s))
        let causalBool = idx.expandedDimensions(axis: 0) .> idx.expandedDimensions(axis: 1)
        let causal = MLX.where(causalBool, MLXArray(Float(-1e9)), MLXArray(Float(0)))
            .expandedDimensions(axes: [0, 1])
        let padding = MLX.where(attentionMask .== 1, MLXArray(Float(0)), MLXArray(Float(-1e9)))
            .expandedDimensions(axes: [1, 2])
        let mask = (causal + padding).asType(.float32)

        // Rotary tables for positions 0..<s, duplicated across halves (HF layout).
        let positions = MLXArray(0..<Int32(s)).asType(.float32)
        let halfDim = config.headDim / 2
        let exponents = MLXArray(0..<Int32(halfDim)).asType(.float32) * (2.0 / Float(config.headDim))
        let invFreq = 1.0 / MLX.pow(MLXArray(config.ropeTheta), exponents)
        let freqs = positions.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        let emb = concatenated([freqs, freqs], axis: -1)
        let cos = MLX.cos(emb).asType(.float32).expandedDimensions(axes: [0, 1])
        let sin = MLX.sin(emb).asType(.float32).expandedDimensions(axes: [0, 1])

        var taps: [Int: MLXArray] = [:]
        if tapIndices.contains(0) { taps[0] = hidden }
        for (i, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cos: cos, sin: sin, config: config)
            let hiddenStateIndex = i + 1
            if tapIndices.contains(hiddenStateIndex) {
                taps[hiddenStateIndex] = hidden
            }
            // Bound intermediate memory: force materialization per few layers.
            if i % 6 == 5 { eval(hidden) }
        }

        let stacked = tapIndices.compactMap { taps[$0] }
        // (B, S, n*hidden): per-token concat in tap order — matches
        // torch.stack(dim=1).permute(0,2,1,3).reshape(B, S, n*H).
        let out = concatenated(stacked.map { $0.expandedDimensions(axis: 2) }, axis: 2)
        return out.reshaped(b, s, stacked.count * config.hiddenSize)
    }
}
