/// MLXAccelerate — batched image ops for tile-grid pipelines (CLR v1.8 inference).
///
/// The single-image ops in ImageOps.swift dispatch one tiny Metal kernel per tile;
/// a 512×512 image at 32×32 tiles issues ~100k dispatches and starves the GPU on
/// launch overhead. These variants operate on a whole *batch* of tiles in one lazy
/// graph:
///
///  - `batchedFilter2D` / `batchedSobelGradients` / `batchedGaussianBlur` take
///    `[N, H, W]` stacks and run a single NHWC conv2d. Border handling defaults to
///    `.reflect101` — cv2's BORDER_DEFAULT for filter2D/Sobel/GaussianBlur — which
///    the zero-padded single-image ops do not match.
///
///  - `WarpTapTable` precomputes a multi-homography bicubic warp as gather tables.
///    For fixed tile size and fixed homographies, the gather indices and bicubic
///    weights are content-independent, so T trials × H×W pixels reduce to 16
///    `take(axis: 1)` gathers over the whole batch. Matches cv2.warpPerspective
///    with INTER_CUBIC + BORDER_REFLECT_101 (cv2's exact fixed-point coordinate
///    quantization available behind `quantizeLikeCV2`).

import Foundation
import MLX

// MARK: - Border modes

public enum BorderMode2D: Sendable {
    /// Zero padding (matches the single-image ops in ImageOps.swift).
    case zero
    /// Mirror without repeating the edge pixel — cv2.BORDER_REFLECT_101, the
    /// default border for cv2.filter2D / cv2.Sobel / cv2.GaussianBlur.
    case reflect101
}

/// Reflect-101 index: maps any integer coordinate into [0, n) by mirroring about
/// the edges without repeating them (cv2 borderInterpolate, BORDER_REFLECT_101).
@inline(__always)
public func reflect101Index(_ p: Int, _ n: Int) -> Int {
    if n == 1 { return 0 }
    var r = p
    while r < 0 || r >= n {
        if r < 0 { r = -r }
        if r >= n { r = 2 * (n - 1) - r }
    }
    return r
}

/// Reflect-101 pad a `[N, H, W, C]` batch on the H (axis 1) and W (axis 2) axes
/// via two index gathers.
public func reflectPad101(
    _ x: MLXArray, top: Int, bottom: Int, left: Int, right: Int,
    stream: StreamOrDevice = .default
) -> MLXArray {
    if top == 0 && bottom == 0 && left == 0 && right == 0 { return x }
    let H = x.shape[1], W = x.shape[2]
    var out = x
    if top != 0 || bottom != 0 {
        let rows = MLXArray((0..<(H + top + bottom)).map { Int32(reflect101Index($0 - top, H)) })
        out = out.take(rows, axis: 1, stream: stream)
    }
    if left != 0 || right != 0 {
        let cols = MLXArray((0..<(W + left + right)).map { Int32(reflect101Index($0 - left, W)) })
        out = out.take(cols, axis: 2, stream: stream)
    }
    return out
}

// MARK: - Batched convolutions

/// Batched `cv2.filter2D`: correlate each `[H, W]` map in a `[N, H, W]` stack with
/// a `(kH, kW)` kernel. Uses cv2's default anchor `(kW/2, kH/2)` — pad top/left by
/// `k/2` and bottom/right by `k-1-k/2`, then a valid conv2d returns exactly `[H, W]`
/// (correct for even kernels too, e.g. the 2×2 local-variance box).
public func batchedFilter2D(
    _ input: MLXArray, kernel: MLXArray,
    border: BorderMode2D = .reflect101,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let N = input.shape[0], H = input.shape[1], W = input.shape[2]
    let kH = kernel.shape[0], kW = kernel.shape[1]
    let k4 = kernel.reshaped([1, kH, kW, 1])
    let pt = kH / 2, pl = kW / 2
    let pb = kH - 1 - pt, pr = kW - 1 - pl
    var x = input.reshaped([N, H, W, 1])
    switch border {
    case .reflect101:
        x = reflectPad101(x, top: pt, bottom: pb, left: pl, right: pr, stream: stream)
    case .zero:
        x = padded(
            x, widths: [IntOrPair(0), IntOrPair((pt, pb)), IntOrPair((pl, pr)), IntOrPair(0)],
            stream: stream)
    }
    return MLX.conv2d(x, k4, stream: stream).reshaped([N, H, W])
}

/// Batched `cv2.Sobel(ksize: 3)` horizontal and vertical gradients on `[N, H, W]`.
public func batchedSobelGradients(
    _ input: MLXArray,
    border: BorderMode2D = .reflect101,
    stream: StreamOrDevice = .default
) -> (dx: MLXArray, dy: MLXArray) {
    let kx = MLXArray([-1, 0, 1, -2, 0, 2, -1, 0, 1] as [Float], [3, 3])
    let ky = MLXArray([-1, -2, -1, 0, 0, 0, 1, 2, 1] as [Float], [3, 3])
    return (
        dx: batchedFilter2D(input, kernel: kx, border: border, stream: stream),
        dy: batchedFilter2D(input, kernel: ky, border: border, stream: stream)
    )
}

/// Batched `cv2.GaussianBlur` on `[N, H, W]`. The kernel is built in Double
/// (matching cv2.getGaussianKernel) then cast to float32; cv2 filters as two
/// separable float passes, which agrees with this 2-D product kernel up to
/// float rounding (~1e-7 relative).
public func batchedGaussianBlur(
    _ input: MLXArray, kernelSize: Int, sigma: Float,
    border: BorderMode2D = .reflect101,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let half = kernelSize / 2
    let s2 = 2.0 * Double(sigma) * Double(sigma)
    var data = [Double](repeating: 0, count: kernelSize * kernelSize)
    var sum = 0.0
    for i in 0..<kernelSize {
        for j in 0..<kernelSize {
            let x = Double(j - half), y = Double(i - half)
            let v = Foundation.exp(-(x * x + y * y) / s2)
            data[i * kernelSize + j] = v
            sum += v
        }
    }
    let kernel = MLXArray(data.map { Float($0 / sum) }, [kernelSize, kernelSize])
    return batchedFilter2D(input, kernel: kernel, border: border, stream: stream)
}

// MARK: - Precomputed bicubic warp (cv2.warpPerspective INTER_CUBIC + BORDER_REFLECT_101)

/// Host-side tap tables for a multi-homography bicubic warp: 16 (index, weight)
/// pairs per destination pixel, flattened over `[trial][y][x]`. Pure host data —
/// unit-testable without an MLX device.
public struct WarpTapTableHost: Sendable {
    /// 16 taps × `[T·H·W]` flat gather indices into the source pixel axis.
    public var indices: [[Int32]]
    /// 16 taps × `[T·H·W]` bicubic weights (wy·wx).
    public var weights: [[Float]]
}

/// GPU-resident tap tables: constant MLXArrays, built once and evaluated eagerly.
public struct WarpTapTable {
    /// 16 taps × int32 `[T·H·W]`.
    public let indices: [MLXArray]
    /// 16 taps × float32 `[T·H·W, 1]` (trailing 1 broadcasts over channels).
    public let weights: [MLXArray]
    public let trials: Int
    public let height: Int
    public let width: Int
}

/// cv2 `interpolateCubic` weights (A = −0.75) for taps at offsets {−1, 0, +1, +2}
/// around `floor(coord)`, given the fractional part `f`.
@inline(__always)
private func cv2CubicWeights(_ f: Double) -> [Double] {
    let A = -0.75
    let w0 = ((A * (f + 1) - 5 * A) * (f + 1) + 8 * A) * (f + 1) - 4 * A
    let w1 = ((A + 2) * f - (A + 3)) * f * f + 1
    let g = 1 - f
    let w2 = ((A + 2) * g - (A + 3)) * g * g + 1
    return [w0, w1, w2, 1 - w0 - w1 - w2]
}

/// Build bicubic warp tap tables for `T` destination→source homographies over an
/// `height × width` tile.
///
/// - Parameters:
///   - dstToSrc: `T` row-major 3×3 matrices in Double, each mapping destination
///     pixel coords to source coords (i.e. already inverted, as cv2.warpPerspective
///     does internally with the matrix you pass it).
///   - sourceIsTrialStacked: `false` → indices address a `[N, H·W, C]` source (the
///     forward warp samples the original tile, shared by all trials). `true` →
///     indices are offset by `t·H·W` so trial `t` samples only its own block of a
///     `[N, T·H·W, C]` trial-stacked source (the inverse warp samples the forward-
///     warped stack).
///   - quantizeLikeCV2: replicate cv2's 1/32-pixel fixed-point coordinate
///     quantization (`X = cvRound(sx·32)`, round-half-even) for bit-near parity
///     with cv2; `false` keeps exact double coordinates (sample points differ from
///     cv2 by ≤ 1/64 px).
public func buildBicubicWarpTapsHost(
    dstToSrc: [[Double]],
    height: Int, width: Int,
    sourceIsTrialStacked: Bool,
    quantizeLikeCV2: Bool = false
) -> WarpTapTableHost {
    let T = dstToSrc.count
    let count = T * height * width
    var idx = [[Int32]](repeating: [Int32](repeating: 0, count: count), count: 16)
    var wts = [[Float]](repeating: [Float](repeating: 0, count: count), count: 16)

    for t in 0..<T {
        let m = dstToSrc[t]
        let base = sourceIsTrialStacked ? Int32(t * height * width) : Int32(0)
        for y in 0..<height {
            for x in 0..<width {
                let xd = Double(x), yd = Double(y)
                let den = m[6] * xd + m[7] * yd + m[8]
                let sxNum = m[0] * xd + m[1] * yd + m[2]
                let syNum = m[3] * xd + m[4] * yd + m[5]
                var ix: Int, iy: Int
                var fx: Double, fy: Double
                if quantizeLikeCV2 {
                    // cv2 WarpPerspectiveInvoker: W = 32/w; X = cvRound(X0·W);
                    // ix = X >> 5; fx = (X & 31)/32.
                    let W = den != 0 ? 32.0 / den : 0
                    let X = Int((sxNum * W).rounded(.toNearestOrEven))
                    let Y = Int((syNum * W).rounded(.toNearestOrEven))
                    ix = X >> 5; fx = Double(X & 31) / 32.0
                    iy = Y >> 5; fy = Double(Y & 31) / 32.0
                } else {
                    let invW = den != 0 ? 1.0 / den : 0
                    let sx = sxNum * invW, sy = syNum * invW
                    ix = Int(sx.rounded(.down)); fx = sx - Double(ix)
                    iy = Int(sy.rounded(.down)); fy = sy - Double(iy)
                }
                let wx = cv2CubicWeights(fx)
                let wy = cv2CubicWeights(fy)
                let p = (t * height + y) * width + x
                for j in 0..<4 {
                    let srcY = reflect101Index(iy - 1 + j, height)
                    for i in 0..<4 {
                        let srcX = reflect101Index(ix - 1 + i, width)
                        let k = j * 4 + i
                        idx[k][p] = base + Int32(srcY * width + srcX)
                        wts[k][p] = Float(wy[j] * wx[i])
                    }
                }
            }
        }
    }
    return WarpTapTableHost(indices: idx, weights: wts)
}

/// Build the GPU-resident tap tables (see `buildBicubicWarpTapsHost`). The constant
/// arrays are evaluated once here so reuse never re-uploads.
public func buildBicubicWarpTable(
    dstToSrc: [[Double]],
    height: Int, width: Int,
    sourceIsTrialStacked: Bool,
    quantizeLikeCV2: Bool = false,
    stream: StreamOrDevice = .default
) -> WarpTapTable {
    let host = buildBicubicWarpTapsHost(
        dstToSrc: dstToSrc, height: height, width: width,
        sourceIsTrialStacked: sourceIsTrialStacked, quantizeLikeCV2: quantizeLikeCV2)
    let count = dstToSrc.count * height * width
    let indices = host.indices.map { MLXArray($0, [count]) }
    let weights = host.weights.map { MLXArray($0, [count, 1]) }
    eval(indices + weights)
    return WarpTapTable(
        indices: indices, weights: weights,
        trials: dstToSrc.count, height: height, width: width)
}

/// Apply a warp tap table to a batch of flattened sources.
///
/// `source` is `[N, S, C]` where `S` is `H·W` (forward tables) or `T·H·W`
/// (trial-stacked tables); returns `[N, T·H·W, C]`. Accumulates tap-by-tap so MLX
/// can free each gather as soon as its multiply-add completes.
public func applyWarpTable(
    _ source: MLXArray, _ table: WarpTapTable,
    stream: StreamOrDevice = .default
) -> MLXArray {
    var acc = source.take(table.indices[0], axis: 1, stream: stream) * table.weights[0]
    for k in 1..<table.indices.count {
        acc = acc + source.take(table.indices[k], axis: 1, stream: stream) * table.weights[k]
    }
    return acc
}

// MARK: - Double-precision homography helpers

/// `cv2.getPerspectiveTransform` solved and returned in Double (the Float-returning
/// variant in ImageOps.swift rounds the result, which is too coarse for building
/// warp tap tables that must match cv2's double-precision coordinate math).
public func getPerspectiveTransformD(src: [[Float]], dst: [[Float]]) -> [Double] {
    var A = [[Double]](repeating: [Double](repeating: 0, count: 8), count: 8)
    var b = [Double](repeating: 0, count: 8)
    for i in 0..<4 {
        let x = Double(src[i][0]), y = Double(src[i][1])
        let u = Double(dst[i][0]), v = Double(dst[i][1])
        A[2 * i] = [x, y, 1, 0, 0, 0, -u * x, -u * y]
        A[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y]
        b[2 * i] = u
        b[2 * i + 1] = v
    }
    guard let h = gaussianElimination(A, b) else {
        return [1, 0, 0, 0, 1, 0, 0, 0, 1]
    }
    return h + [1.0]
}

/// 3×3 matrix inverse via cofactors, all in Double (matches np.linalg.inv to
/// ~1e-15). Returns identity if singular.
public func invertMatrix3x3D(_ m: [Double]) -> [Double] {
    let a = m[0], b = m[1], c = m[2]
    let d = m[3], e = m[4], f = m[5]
    let g = m[6], h = m[7], k = m[8]
    let det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)
    guard Swift.abs(det) > 1e-12 else { return [1, 0, 0, 0, 1, 0, 0, 0, 1] }
    let inv = 1.0 / det
    return [
        (e * k - f * h) * inv, (c * h - b * k) * inv, (b * f - c * e) * inv,
        (f * g - d * k) * inv, (a * k - c * g) * inv, (c * d - a * f) * inv,
        (d * h - e * g) * inv, (b * g - a * h) * inv, (a * e - b * d) * inv,
    ]
}
