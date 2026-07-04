//
//  Flux2VAE.swift
//  FluxKit
//
//  FLUX.2 VAE decode path only (txt2img): BN-denormalize packed latents using the
//  checkpoint's running stats, un-patchify 128→32ch, post-quant conv, then the conv
//  decoder (GroupNorm resnets, single-head mid attention, nearest-×2 upsampling).
//  Convolutions run NHWC (MLX layout); weights convert from PyTorch OIHW at load.
//  Ported from mflux's Flux2VAE/Flux2Decoder. scaling=1, shift=0 for FLUX.2.
//

import Foundation
import MLX
import MLXFast
import MLXNN

public final class Flux2VAEDecoder {
    static let latentChannels = 32
    static let blockOut = [128, 256, 512, 512]
    static let groups = 32
    static let eps: Float = 1e-6
    static let bnEps: Float = 1e-4

    final class Resnet {
        let norm1W: MLXArray, norm1B: MLXArray
        let norm2W: MLXArray, norm2B: MLXArray
        let conv1W: MLXArray, conv1B: MLXArray
        let conv2W: MLXArray, conv2B: MLXArray
        let shortcutW: MLXArray?, shortcutB: MLXArray?

        init(store: TensorStore, prefix: String) throws {
            norm1W = try store.take("\(prefix).norm1.weight")
            norm1B = try store.take("\(prefix).norm1.bias")
            norm2W = try store.take("\(prefix).norm2.weight")
            norm2B = try store.take("\(prefix).norm2.bias")
            (conv1W, conv1B) = try store.conv2d("\(prefix).conv1")
            (conv2W, conv2B) = try store.conv2d("\(prefix).conv2")
            if store.contains("\(prefix).conv_shortcut.weight") {
                let sc = try store.conv2d("\(prefix).conv_shortcut")
                shortcutW = sc.weight
                shortcutB = sc.bias
            } else {
                shortcutW = nil
                shortcutB = nil
            }
        }

        /// NHWC in/out.
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = groupNorm(x, weight: norm1W, bias: norm1B)
            h = silu(h)
            h = conv2dSame(h, weight: conv1W, bias: conv1B)
            h = groupNorm(h, weight: norm2W, bias: norm2B)
            h = silu(h)
            h = conv2dSame(h, weight: conv2W, bias: conv2B)
            var residual = x
            if let shortcutW, let shortcutB {
                residual = conv2d1x1(x, weight: shortcutW, bias: shortcutB)
            }
            return h + residual
        }
    }

    final class MidAttention {
        let normW: MLXArray, normB: MLXArray
        let toQ: Linear, toK: Linear, toV: Linear, toOut: Linear

        init(store: TensorStore, prefix: String) throws {
            normW = try store.take("\(prefix).group_norm.weight")
            normB = try store.take("\(prefix).group_norm.bias")
            let channels = normW.dim(0)
            toQ = try store.anyLinear("\(prefix).to_q", inFeatures: channels, bias: true)
            toK = try store.anyLinear("\(prefix).to_k", inFeatures: channels, bias: true)
            toV = try store.anyLinear("\(prefix).to_v", inFeatures: channels, bias: true)
            toOut = try store.anyLinear("\(prefix).to_out", inFeatures: channels, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            let normed = groupNorm(x, weight: normW, bias: normB)
            let q = toQ(normed).reshaped(b, h * w, 1, c).transposed(0, 2, 1, 3)
            let k = toK(normed).reshaped(b, h * w, 1, c).transposed(0, 2, 1, 3)
            let v = toV(normed).reshaped(b, h * w, 1, c).transposed(0, 2, 1, 3)
            let scale = 1.0 / Float(c).squareRoot()
            var attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                             scale: scale, mask: nil)
            attended = attended.transposed(0, 2, 1, 3).reshaped(b, h, w, c)
            return x + toOut(attended)
        }
    }

    let bnMean: MLXArray
    let bnStd: MLXArray
    let postQuantW: MLXArray, postQuantB: MLXArray
    let convInW: MLXArray, convInB: MLXArray
    let midResnet1: Resnet
    let midAttention: MidAttention
    let midResnet2: Resnet
    let upBlocks: [[Resnet]]
    let upsampleConvs: [(MLXArray, MLXArray)?]
    let normOutW: MLXArray, normOutB: MLXArray
    let convOutW: MLXArray, convOutB: MLXArray

    public init(componentDir: URL) throws {
        let store = try TensorStore(componentDir: componentDir)
        bnMean = try store.take("bn.running_mean").asType(.float32)
        bnStd = MLX.sqrt(try store.take("bn.running_var").asType(.float32) + Self.bnEps)
        let pq = try store.conv2d("post_quant_conv")
        postQuantW = pq.weight
        postQuantB = pq.bias
        let ci = try store.conv2d("decoder.conv_in")
        convInW = ci.weight
        convInB = ci.bias
        midResnet1 = try Resnet(store: store, prefix: "decoder.mid_block.resnets.0")
        midAttention = try MidAttention(store: store, prefix: "decoder.mid_block.attentions.0")
        midResnet2 = try Resnet(store: store, prefix: "decoder.mid_block.resnets.1")

        var blocks: [[Resnet]] = []
        var ups: [(MLXArray, MLXArray)?] = []
        let numBlocks = Self.blockOut.count
        for i in 0..<numBlocks {
            var resnets: [Resnet] = []
            for j in 0..<3 {
                resnets.append(try Resnet(store: store, prefix: "decoder.up_blocks.\(i).resnets.\(j)"))
            }
            blocks.append(resnets)
            if i < numBlocks - 1 {
                let up = try store.conv2d("decoder.up_blocks.\(i).upsamplers.0.conv")
                ups.append(up)
            } else {
                ups.append(nil)
            }
        }
        upBlocks = blocks
        upsampleConvs = ups
        normOutW = try store.take("decoder.conv_norm_out.weight")
        normOutB = try store.take("decoder.conv_norm_out.bias")
        let co = try store.conv2d("decoder.conv_out")
        convOutW = co.weight
        convOutB = co.bias
        store.drain()
    }

    /// packed: (B, 128, Hp, Wp) BN-normalized DiT-space latents → image (B, H*16, W*16, 3) in [-1, 1].
    public func decodePacked(_ packed: MLXArray) -> MLXArray {
        // BN denormalize in the 128-channel packed space.
        var latents = packed.asType(.float32) * bnStd.reshaped(1, -1, 1, 1) + bnMean.reshaped(1, -1, 1, 1)

        // Unpatchify 128→32 channels, 2× spatial.
        let (b, c, hp, wp) = (latents.dim(0), latents.dim(1), latents.dim(2), latents.dim(3))
        latents = latents.reshaped(b, c / 4, 2, 2, hp, wp)
            .transposed(0, 1, 4, 2, 5, 3)
            .reshaped(b, c / 4, hp * 2, wp * 2)

        // NCHW → NHWC for the conv stack.
        var h = latents.transposed(0, 2, 3, 1).asType(Flux2Pipeline.activationDType)
        h = conv2d1x1(h, weight: postQuantW, bias: postQuantB)
        h = conv2dSame(h, weight: convInW, bias: convInB)

        h = midResnet1(h)
        h = midAttention(h)
        h = midResnet2(h)
        eval(h)

        for (i, resnets) in upBlocks.enumerated() {
            for resnet in resnets {
                h = resnet(h)
            }
            if let (w, bias) = upsampleConvs[i] {
                // Nearest ×2 upsample then 3×3 conv.
                h = MLX.repeated(h, count: 2, axis: 1)
                h = MLX.repeated(h, count: 2, axis: 2)
                h = conv2dSame(h, weight: w, bias: bias)
            }
            eval(h)
        }

        h = groupNorm(h, weight: normOutW, bias: normOutB)
        h = silu(h)
        h = conv2dSame(h, weight: convOutW, bias: convOutB)
        return h
    }
}

// MARK: - Conv/GroupNorm helpers (NHWC, PyTorch-compatible numerics)

func conv2dSame(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
    MLX.conv2d(x, weight, stride: [1, 1], padding: [1, 1]) + bias
}

func conv2d1x1(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
    MLX.conv2d(x, weight, stride: [1, 1], padding: [0, 0]) + bias
}

/// GroupNorm over NHWC with PyTorch grouping semantics (channels split into
/// contiguous groups, normalized over (H, W, C/G)), computed in float32.
func groupNorm(_ x: MLXArray, weight: MLXArray, bias: MLXArray, groups: Int = 32, eps: Float = 1e-6) -> MLXArray {
    let dtype = x.dtype
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    var v = x.asType(.float32).reshaped(b, h * w, groups, c / groups).transposed(0, 2, 1, 3)
        .reshaped(b, groups, -1)
    let mean = v.mean(axis: -1, keepDims: true)
    let variance = v.variance(axis: -1, keepDims: true)
    v = (v - mean) * rsqrt(variance + eps)
    v = v.reshaped(b, groups, h * w, c / groups).transposed(0, 2, 1, 3).reshaped(b, h, w, c)
    return (v * weight.asType(.float32) + bias.asType(.float32)).asType(dtype)
}
