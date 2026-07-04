//
//  Flux2Transformer.swift
//  FluxKit
//
//  FLUX.2 Klein DiT: 5 joint (double-stream) blocks + 20 parallel single-stream
//  blocks at dim 3072 (24 heads × 128), shared per-stream adaLN modulation computed
//  once from the timestep embedding, 4-axis RoPE (t,h,w,l · 32 dims each, θ=2000,
//  interleaved-pair application), SwiGLU feed-forwards, adaLN-continuous output.
//  Ported from mflux's Flux2Transformer against the mflux checkpoint tensor names.
//

import Foundation
import MLX
import MLXFast
import MLXNN

public final class Flux2Transformer {
    public struct Config {
        public var inChannels = 128
        public var numLayers = 5
        public var numSingleLayers = 20
        public var headDim = 128
        public var numHeads = 24
        public var jointDim = 7680
        public var timestepChannels = 256
        public var mlpRatio: Float = 3.0
        public var ropeTheta: Float = 2000
        public var ropeAxes = [32, 32, 32, 32]
        public var eps: Float = 1e-6
        public var innerDim: Int { numHeads * headDim }
        public init() {}
    }

    // MARK: - Sub-modules

    final class DoubleBlock {
        let attnToQ: Linear, attnToK: Linear, attnToV: Linear, attnToOut: Linear
        let addQ: Linear, addK: Linear, addV: Linear, toAddOut: Linear
        let normQ: ExplicitRMSNorm, normK: ExplicitRMSNorm
        let normAddedQ: ExplicitRMSNorm, normAddedK: ExplicitRMSNorm
        let ffIn: Linear, ffOut: Linear
        let ffCtxIn: Linear, ffCtxOut: Linear
        let config: Config

        init(store: TensorStore, prefix: String, config: Config) throws {
            self.config = config
            let dim = config.innerDim
            let mlpInner = Int(Float(dim) * config.mlpRatio)
            attnToQ = try store.anyLinear("\(prefix).attn.to_q", inFeatures: dim)
            attnToK = try store.anyLinear("\(prefix).attn.to_k", inFeatures: dim)
            attnToV = try store.anyLinear("\(prefix).attn.to_v", inFeatures: dim)
            attnToOut = try store.anyLinear("\(prefix).attn.to_out", inFeatures: dim)
            addQ = try store.anyLinear("\(prefix).attn.add_q_proj", inFeatures: dim)
            addK = try store.anyLinear("\(prefix).attn.add_k_proj", inFeatures: dim)
            addV = try store.anyLinear("\(prefix).attn.add_v_proj", inFeatures: dim)
            toAddOut = try store.anyLinear("\(prefix).attn.to_add_out", inFeatures: dim)
            normQ = ExplicitRMSNorm(weight: try store.rmsNormWeight("\(prefix).attn.norm_q"), eps: 1e-5)
            normK = ExplicitRMSNorm(weight: try store.rmsNormWeight("\(prefix).attn.norm_k"), eps: 1e-5)
            normAddedQ = ExplicitRMSNorm(weight: try store.rmsNormWeight("\(prefix).attn.norm_added_q"), eps: 1e-5)
            normAddedK = ExplicitRMSNorm(weight: try store.rmsNormWeight("\(prefix).attn.norm_added_k"), eps: 1e-5)
            ffIn = try store.anyLinear("\(prefix).ff.linear_in", inFeatures: dim)
            ffOut = try store.anyLinear("\(prefix).ff.linear_out", inFeatures: mlpInner)
            ffCtxIn = try store.anyLinear("\(prefix).ff_context.linear_in", inFeatures: dim)
            ffCtxOut = try store.anyLinear("\(prefix).ff_context.linear_out", inFeatures: mlpInner)
        }

        /// Returns (encoderHidden, hidden).
        func callAsFunction(hidden: MLXArray, encoder: MLXArray,
                            modImg: [MLXArray], modTxt: [MLXArray],
                            ropeCos: MLXArray, ropeSin: MLXArray) -> (MLXArray, MLXArray) {
            // modImg/modTxt: 6 arrays each — [shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP]
            let normed = layerNorm(hidden, eps: config.eps)
            let modded = (1 + modImg[1]) * normed + modImg[0]
            let ctxNormed = layerNorm(encoder, eps: config.eps)
            let ctxModded = (1 + modTxt[1]) * ctxNormed + modTxt[0]

            let (attnOut, ctxAttnOut) = jointAttention(img: modded, txt: ctxModded,
                                                       ropeCos: ropeCos, ropeSin: ropeSin)

            var h = hidden + modImg[2] * attnOut
            var c = encoder + modTxt[2] * ctxAttnOut

            let hNorm = (1 + modImg[4]) * layerNorm(h, eps: config.eps) + modImg[3]
            h = h + modImg[5] * ffOut(swiGLU(ffIn(hNorm)))

            let cNorm = (1 + modTxt[4]) * layerNorm(c, eps: config.eps) + modTxt[3]
            c = c + modTxt[5] * ffCtxOut(swiGLU(ffCtxIn(cNorm)))

            return (c, h)
        }

        private func jointAttention(img: MLXArray, txt: MLXArray,
                                    ropeCos: MLXArray, ropeSin: MLXArray) -> (MLXArray, MLXArray) {
            let b = img.dim(0)
            let heads = config.numHeads
            let hd = config.headDim

            func heads4(_ x: MLXArray) -> MLXArray {
                x.reshaped(b, x.dim(1), heads, hd).transposed(0, 2, 1, 3)
            }

            var q = normQ(heads4(attnToQ(img)))
            var k = normK(heads4(attnToK(img)))
            let v = heads4(attnToV(img))

            let eq = normAddedQ(heads4(addQ(txt)))
            let ek = normAddedK(heads4(addK(txt)))
            let ev = heads4(addV(txt))

            // Text tokens lead the joint sequence.
            q = concatenated([eq, q], axis: 2)
            k = concatenated([ek, k], axis: 2)
            let vAll = concatenated([ev, v], axis: 2)

            (q, k) = applyRopeInterleaved(q: q, k: k, cos: ropeCos, sin: ropeSin)

            let scale = 1.0 / Float(hd).squareRoot()
            var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: vAll,
                                                        scale: scale, mask: nil)
            out = out.transposed(0, 2, 1, 3).reshaped(b, -1, heads * hd)

            let txtLen = txt.dim(1)
            let ctxOut = toAddOut(out[0..., ..<txtLen, 0...])
            let imgOut = attnToOut(out[0..., txtLen..., 0...])
            return (imgOut, ctxOut)
        }
    }

    final class SingleBlock {
        let toQKVMLP: Linear
        let toOut: Linear
        let normQ: ExplicitRMSNorm, normK: ExplicitRMSNorm
        let config: Config
        let mlpHidden: Int

        init(store: TensorStore, prefix: String, config: Config) throws {
            self.config = config
            let dim = config.innerDim
            mlpHidden = Int(Float(dim) * config.mlpRatio)
            toQKVMLP = try store.anyLinear("\(prefix).attn.to_qkv_mlp_proj", inFeatures: dim)
            toOut = try store.anyLinear("\(prefix).attn.to_out", inFeatures: dim + mlpHidden)
            normQ = ExplicitRMSNorm(weight: try store.rmsNormWeight("\(prefix).attn.norm_q"), eps: 1e-5)
            normK = ExplicitRMSNorm(weight: try store.rmsNormWeight("\(prefix).attn.norm_k"), eps: 1e-5)
        }

        func callAsFunction(_ hidden: MLXArray, mod: [MLXArray],
                            ropeCos: MLXArray, ropeSin: MLXArray) -> MLXArray {
            // mod: [shift, scale, gate]
            let b = hidden.dim(0)
            let s = hidden.dim(1)
            let heads = config.numHeads
            let hd = config.headDim
            let inner = heads * hd

            let normed = (1 + mod[1]) * layerNorm(hidden, eps: config.eps) + mod[0]

            let proj = toQKVMLP(normed)
            let qkv = proj[0..., 0..., ..<(3 * inner)]
            let mlp = proj[0..., 0..., (3 * inner)...]

            var q = qkv[0..., 0..., ..<inner].reshaped(b, s, heads, hd).transposed(0, 2, 1, 3)
            var k = qkv[0..., 0..., inner..<(2 * inner)].reshaped(b, s, heads, hd).transposed(0, 2, 1, 3)
            let v = qkv[0..., 0..., (2 * inner)...].reshaped(b, s, heads, hd).transposed(0, 2, 1, 3)

            q = normQ(q)
            k = normK(k)
            (q, k) = applyRopeInterleaved(q: q, k: k, cos: ropeCos, sin: ropeSin)

            let scale = 1.0 / Float(hd).squareRoot()
            var attn = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                         scale: scale, mask: nil)
            attn = attn.transposed(0, 2, 1, 3).reshaped(b, s, inner)

            let out = toOut(concatenated([attn, swiGLU(mlp)], axis: -1))
            return hidden + mod[2] * out
        }
    }

    // MARK: - Model

    public let config: Config
    let xEmbedder: Linear
    let contextEmbedder: Linear
    let timeLinear1: Linear
    let timeLinear2: Linear
    let doubleModImg: Linear
    let doubleModTxt: Linear
    let singleMod: Linear
    let doubleBlocks: [DoubleBlock]
    let singleBlocks: [SingleBlock]
    let normOutLinear: Linear
    let projOut: Linear

    public init(componentDir: URL, config: Config = Config()) throws {
        self.config = config
        let store = try TensorStore(componentDir: componentDir)
        let dim = config.innerDim
        xEmbedder = try store.anyLinear("x_embedder", inFeatures: config.inChannels)
        contextEmbedder = try store.anyLinear("context_embedder", inFeatures: config.jointDim)
        timeLinear1 = try store.anyLinear("time_guidance_embed.linear_1", inFeatures: config.timestepChannels)
        timeLinear2 = try store.anyLinear("time_guidance_embed.linear_2", inFeatures: dim)
        doubleModImg = try store.anyLinear("double_stream_modulation_img.linear", inFeatures: dim)
        doubleModTxt = try store.anyLinear("double_stream_modulation_txt.linear", inFeatures: dim)
        singleMod = try store.anyLinear("single_stream_modulation.linear", inFeatures: dim)
        doubleBlocks = try (0..<config.numLayers).map {
            try DoubleBlock(store: store, prefix: "transformer_blocks.\($0)", config: config)
        }
        singleBlocks = try (0..<config.numSingleLayers).map {
            try SingleBlock(store: store, prefix: "single_transformer_blocks.\($0)", config: config)
        }
        normOutLinear = try store.anyLinear("norm_out.linear", inFeatures: dim)
        projOut = try store.anyLinear("proj_out", inFeatures: dim)
        store.drain()
    }

    /// hidden: (B, imgSeq, 128) packed latents · encoder: (B, txtSeq, 7680) ·
    /// timestep in [0, 1000] · ids: (S, 4). Returns the velocity prediction (B, imgSeq, 128).
    public func callAsFunction(hidden: MLXArray, encoder: MLXArray, timestep: Float,
                               imgIDs: MLXArray, txtIDs: MLXArray) -> MLXArray {
        let dtype = Flux2Pipeline.activationDType

        // Timestep embedding (sinusoidal 256 → MLP 3072), flip_sin_to_cos.
        let temb = timestepEmbedding(MLXArray([timestep]), channels: config.timestepChannels)
        let t = timeLinear2(silu(timeLinear1(temb.asType(dtype))))

        // Shared modulations: double = 2 param sets (6 chunks), single = 1 set (3 chunks).
        func modChunks(_ linear: Linear, sets: Int) -> [MLXArray] {
            let m = linear(silu(t)).expandedDimensions(axis: 1)
            return split(m, parts: 3 * sets, axis: -1)
        }
        let modImg = modChunks(doubleModImg, sets: 2)
        let modTxt = modChunks(doubleModTxt, sets: 2)
        let modSingle = modChunks(singleMod, sets: 1)

        var h = xEmbedder(hidden.asType(dtype))
        var c = contextEmbedder(encoder.asType(dtype))

        // 4-axis RoPE; text tokens lead the joint sequence.
        let (txtCos, txtSin) = ropeTables(ids: txtIDs)
        let (imgCos, imgSin) = ropeTables(ids: imgIDs)
        let cos = concatenated([txtCos, imgCos], axis: 0)
        let sin = concatenated([txtSin, imgSin], axis: 0)

        for block in doubleBlocks {
            (c, h) = block(hidden: h, encoder: c, modImg: modImg, modTxt: modTxt,
                           ropeCos: cos, ropeSin: sin)
        }

        var joint = concatenated([c, h], axis: 1)
        for block in singleBlocks {
            joint = block(joint, mod: modSingle, ropeCos: cos, ropeSin: sin)
        }

        var out = joint[0..., c.dim(1)..., 0...]

        // adaLN-continuous: linear(silu(temb)) → [scale, shift].
        let adaln = normOutLinear(silu(t))
        let dim = config.innerDim
        let scale = adaln[0..., ..<dim].expandedDimensions(axis: 1)
        let shift = adaln[0..., dim...].expandedDimensions(axis: 1)
        out = layerNorm(out, eps: config.eps) * (1 + scale) + shift
        return projOut(out)
    }

    /// Per-axis rotary tables from 4-column integer ids, θ=2000, dims (32,32,32,32).
    func ropeTables(ids: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        var cosParts: [MLXArray] = []
        var sinParts: [MLXArray] = []
        let pos = ids.asType(.float32)
        for (axis, dim) in config.ropeAxes.enumerated() {
            let exponents = MLXArray(stride(from: 0, to: Int32(dim), by: 2).map { Float($0) }) / Float(dim)
            let omega = 1.0 / MLX.pow(MLXArray(config.ropeTheta), exponents)
            let out = pos[0..., axis].expandedDimensions(axis: 1) * omega.expandedDimensions(axis: 0)
            cosParts.append(MLX.cos(out))
            sinParts.append(MLX.sin(out))
        }
        return (concatenated(cosParts, axis: -1), concatenated(sinParts, axis: -1))
    }
}

// MARK: - Shared math

/// LayerNorm without affine, computed in float32.
func layerNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    let dtype = x.dtype
    let x32 = x.asType(.float32)
    let mean = x32.mean(axis: -1, keepDims: true)
    let variance = x32.variance(axis: -1, keepDims: true)
    return ((x32 - mean) * rsqrt(variance + eps)).asType(dtype)
}

/// SwiGLU with the gate in the leading half (FLUX.2 convention).
func swiGLU(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return silu(x1) * x2
}

/// Rotary application over interleaved (real, imag) pairs — FLUX convention, float32.
/// q/k: (B, H, S, D) · cos/sin: (S, D/2).
func applyRopeInterleaved(q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray) -> (MLXArray, MLXArray) {
    let outDtype = q.dtype
    let cosB = cos.reshaped(1, 1, cos.dim(0), cos.dim(1))
    let sinB = sin.reshaped(1, 1, sin.dim(0), sin.dim(1))

    func mix(_ x: MLXArray) -> MLXArray {
        let xF = x.asType(.float32)
        let shape = xF.shape
        let paired = xF.reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
        let real = paired[.ellipsis, 0]
        let imag = paired[.ellipsis, 1]
        let out0 = real * cosB - imag * sinB
        let out1 = imag * cosB + real * sinB
        return stacked([out0, out1], axis: -1).reshaped(shape).asType(outDtype)
    }
    return (mix(q), mix(k))
}

/// Sinusoidal timestep embedding, flip_sin_to_cos=true, downscale_freq_shift=0.
func timestepEmbedding(_ timesteps: MLXArray, channels: Int) -> MLXArray {
    let half = channels / 2
    let exponent = -logf(10000) * MLXArray(0..<Int32(half)).asType(.float32) / Float(half)
    let freqs = MLX.exp(exponent)
    let args = timesteps.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
    // flip_sin_to_cos → [cos, sin]
    return concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
}
