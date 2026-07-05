//
//  ObscurDinov2.swift
//  ObscurKit
//
//  Self-contained DINOv2-S/14 for Signet ingest and projection pretraining — a copy of
//  CLR's bundled ViT so ObscurKit (and the training CLI) need no CLRCore dependency.
//  The checkpoint and preprocessing are pinned to CLR's `dinov2_small.safetensors` so
//  offline pretraining and on-device ingest stay embedding-identical.
//

import Foundation
import CoreGraphics
import MLX
import MLXNN

private func dinoAttention(_ x: MLXArray, q: Linear, k: Linear, v: Linear, proj: Linear,
                           heads: Int, scale: Float) -> MLXArray {
    let (b, l) = (x.dim(0), x.dim(1))
    let qa = q(x).reshaped(b, l, heads, -1).transposed(0, 2, 1, 3)
    let ka = k(x).reshaped(b, l, heads, -1).transposed(0, 2, 1, 3)
    let va = v(x).reshaped(b, l, heads, -1).transposed(0, 2, 1, 3)
    let scores = softmax(matmul(qa, ka.transposed(0, 1, 3, 2)) * scale, axis: -1)
    let o = matmul(scores, va).transposed(0, 2, 1, 3).reshaped(b, l, -1)
    return proj(o)
}

final class ObscurDinoAttention: Module {
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    init(dims: Int, heads: Int) {
        self.heads = heads
        self.scale = pow(Float(dims / heads), -0.5)
        self._q.wrappedValue = Linear(dims, dims, bias: true)
        self._k.wrappedValue = Linear(dims, dims, bias: true)
        self._v.wrappedValue = Linear(dims, dims, bias: true)
        self._proj.wrappedValue = Linear(dims, dims, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        dinoAttention(x, q: q, k: k, v: v, proj: proj, heads: heads, scale: scale)
    }
}

final class ObscurDinoMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(dims: Int, hidden: Int) {
        self._fc1.wrappedValue = Linear(dims, hidden, bias: true)
        self._fc2.wrappedValue = Linear(hidden, dims, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

final class ObscurDinoBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: ObscurDinoAttention
    @ParameterInfo(key: "ls1") var ls1: MLXArray
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: ObscurDinoMLP
    @ParameterInfo(key: "ls2") var ls2: MLXArray

    init(dims: Int, heads: Int, mlpHidden: Int, eps: Float) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dims, eps: eps)
        self._attn.wrappedValue = ObscurDinoAttention(dims: dims, heads: heads)
        self._ls1.wrappedValue = MLXArray.ones([dims])
        self._norm2.wrappedValue = LayerNorm(dimensions: dims, eps: eps)
        self._mlp.wrappedValue = ObscurDinoMLP(dims: dims, hidden: mlpHidden)
        self._ls2.wrappedValue = MLXArray.ones([dims])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + ls1 * attn(norm1(x))
        return h + ls2 * mlp(norm2(h))
    }
}

/// DINOv2-S/14. `patchTokens` returns the (B, 256, 384) per-patch features (CLS row
/// excluded) that the Signet projection consumes.
public final class ObscurDinov2: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: Conv2d
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    let blocks: [ObscurDinoBlock]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    public init(hidden: Int = 384, layers: Int = 12, heads: Int = 6,
                patch: Int = 14, eps: Float = 1e-6) {
        self._patchEmbed.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: hidden,
            kernelSize: .init(patch), stride: .init(patch))
        self._clsToken.wrappedValue = MLXArray.zeros([1, 1, hidden])
        self._posEmbed.wrappedValue = MLXArray.zeros([1, 16 * 16 + 1, hidden])
        self.blocks = (0 ..< layers).map { _ in
            ObscurDinoBlock(dims: hidden, heads: heads, mlpHidden: hidden * 4, eps: eps)
        }
        self._norm.wrappedValue = LayerNorm(dimensions: hidden, eps: eps)
        super.init()
    }

    /// pixels: (B, 224, 224, 3) NHWC, ImageNet-normalized. Returns (B, 256, 384) fp32.
    public func patchTokens(_ pixels: MLXArray) -> MLXArray {
        let dt = clsToken.dtype
        let b = pixels.dim(0)
        var p = patchEmbed(pixels.asType(dt))
        p = p.flattened(start: 1, end: 2)
        let cls = broadcast(clsToken, to: [b, 1, clsToken.dim(2)])
        var x = concatenated([cls, p], axis: 1)
        x = x + posEmbed
        for blk in blocks { x = blk(x) }
        x = norm(x)
        return x[0..., 1..., 0...].asType(.float32)
    }

    public func loadWeights(url: URL) throws {
        let w = try loadArrays(url: url)
        try update(parameters: ModuleParameters.unflattened(w), verify: [.all])
        eval(self)
    }
}

// MARK: - Preprocessing (pinned to CLR's Dinov2Preprocess)

public enum ObscurDinoPreprocess {
    static let mean: [Float] = [0.485, 0.456, 0.406]
    static let std: [Float] = [0.229, 0.224, 0.225]

    /// CGImage → (1, 224, 224, 3) NHWC, ImageNet-normalized. Short-edge resize to 256
    /// then center-crop 224 (matches CLR).
    public static func input(_ cg: CGImage, target: Int = 224, shortEdge: Int = 256) -> MLXArray {
        let w0 = cg.width, h0 = cg.height
        let scale = Float(shortEdge) / Float(min(w0, h0))
        let rw = Int((Float(w0) * scale).rounded()), rh = Int((Float(h0) * scale).rounded())

        let cs = CGColorSpaceCreateDeviceRGB()
        var buffer = [UInt8](repeating: 0, count: rw * rh * 4)
        guard let ctx = CGContext(data: &buffer, width: rw, height: rh, bitsPerComponent: 8,
                                  bytesPerRow: rw * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return MLXArray.zeros([1, target, target, 3])
        }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: rw, height: rh))

        let ox = (rw - target) / 2, oy = (rh - target) / 2
        var pixels = [Float](repeating: 0, count: target * target * 3)
        for y in 0..<target {
            for x in 0..<target {
                let src = ((y + oy) * rw + (x + ox)) * 4
                for c in 0..<3 {
                    let v = Float(buffer[src + c]) / 255.0
                    pixels[(y * target + x) * 3 + c] = (v - mean[c]) / std[c]
                }
            }
        }
        return MLXArray(pixels, [1, target, target, 3])
    }
}
