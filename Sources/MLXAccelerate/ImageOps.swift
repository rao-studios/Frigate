/// MLXAccelerate — Linux-compatible image processing ops backed by MLX.
///
/// Mirrors the subset of Accelerate/vImage used by the CLR v1.8 inference pipeline:
/// Gaussian blur, Sobel gradients, arbitrary filter2D, and perspective warp with
/// bilinear sampling.  All convolution ops dispatch through MLX conv2d (NHWC format)
/// so they run on CPU, CUDA, or Metal with no platform-specific code.

import Foundation
import MLX

// MARK: - Gaussian Blur

/// Gaussian blur on a single-channel float residual map.
///
/// Input shape:  `(H, W)`  — single channel.
/// Output shape: `(H, W)`  — same-padded (border clamped via conv2d reflect padding
/// is not available; we use zero-padding which matches cv2 for interior pixels).
public func gaussianBlur(
    _ input: MLXArray,
    kernelSize: Int,
    sigma: Float,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let H = input.shape[0], W = input.shape[1]
    let kernel = makeGaussianKernel(size: kernelSize, sigma: sigma)
        .reshaped([1, kernelSize, kernelSize, 1])
    let batched = input.reshaped([1, H, W, 1])
    let out = MLX.conv2d(batched, kernel, padding: IntOrPair(kernelSize / 2), stream: stream)
    return out.reshaped([H, W])
}

// MARK: - Sobel Gradients

/// Compute horizontal and vertical Sobel gradient maps on a single-channel array.
///
/// Input shape: `(H, W)`. Returns `(dx, dy)` each of shape `(H, W)`.
/// Matches `cv2.Sobel(R, CV_32F, dx=1, dy=0, ksize=3)` and the vertical counterpart.
public func sobelGradients(
    _ input: MLXArray,
    stream: StreamOrDevice = .default
) -> (dx: MLXArray, dy: MLXArray) {
    let H = input.shape[0], W = input.shape[1]
    let batched = input.reshaped([1, H, W, 1])
    // Row-major Sobel kernels shaped (1, 3, 3, 1) in NHWC weight format (C_out, kH, kW, C_in).
    let kx = MLXArray([-1, 0, 1, -2, 0, 2, -1, 0, 1] as [Float], [1, 3, 3, 1])
    let ky = MLXArray([-1, -2, -1, 0, 0, 0, 1, 2, 1] as [Float], [1, 3, 3, 1])
    let dx = MLX.conv2d(batched, kx, padding: 1, stream: stream).reshaped([H, W])
    let dy = MLX.conv2d(batched, ky, padding: 1, stream: stream).reshaped([H, W])
    return (dx, dy)
}

// MARK: - filter2D

/// Convolve a single-channel 2D float array with an arbitrary 2D kernel.
///
/// Input shape: `(H, W)`. Kernel shape: `(kH, kW)`.
/// Output shape: `(H, W)` with floor-divided same-padding.
/// Matches `cv2.filter2D(R, -1, kernel)` for odd-sized kernels.
public func filter2D(
    _ input: MLXArray,
    kernel: MLXArray,
    stream: StreamOrDevice = .default
) -> MLXArray {
    let H = input.shape[0], W = input.shape[1]
    let kH = kernel.shape[0], kW = kernel.shape[1]
    let k = kernel.reshaped([1, kH, kW, 1])
    let batched = input.reshaped([1, H, W, 1])
    let out = MLX.conv2d(batched, k, padding: IntOrPair((kH / 2, kW / 2)), stream: stream)
    // Even-sized kernels (e.g. 2×2) produce output size = input + 1 per padded dim.
    // Slice back to [H, W] so callers always get the expected shape.
    return out[0..<1, 0..<H, 0..<W, 0..<1].reshaped([H, W])
}

// MARK: - Perspective Warp

/// Apply a perspective warp to an image using inverse (destination→source) mapping
/// with bilinear interpolation.
///
/// - Parameters:
///   - input:  `(H, W, C)` float32 image (values in any range, e.g. 0–255).
///   - forwardMatrix:  9-element row-major 3×3 homography that maps **src→dst**
///     (same convention as `cv2.warpPerspective`).
///   - outputSize: `(outH, outW)` of the destination image.
///
/// Internally inverts `forwardMatrix` to get the dst→src mapping, then samples
/// using bilinear interpolation with border-clamp (≈ `cv2.BORDER_REPLICATE`).
/// All arithmetic runs through MLX so it dispatches to the active backend.
public func perspectiveWarp(
    _ input: MLXArray,
    forwardMatrix m: [Float],
    outputSize: (Int, Int),
    stream: StreamOrDevice = .default
) -> MLXArray {
    let H = input.shape[0], W = input.shape[1], C = input.shape[2]
    let (outH, outW) = outputSize

    // Invert the forward matrix to get the inverse (dst → src) mapping.
    let mInv = invertMatrix3x3(m)
    let mInvArr = MLXArray(mInv, [3, 3])  // (3, 3)

    // Destination coordinate grid (N, 3) in homogeneous form [x, y, 1].
    // Use a cache keyed by output size — warpNoise calls this 32 times with
    // the same tile dimensions so rebuilding from scratch every call is wasteful.
    let coords = coordGrid(outH: outH, outW: outW)  // (N, 3)

    // Apply M_inv: source_coords = coords @ mInv^T  →  (N, 3)
    let transformed = MLX.matmul(coords, mInvArr.transposed())

    // Extract and normalise: sx = tx/tw, sy = ty/tw
    let txArr = transformed[.ellipsis, 0..<1].squeezed(axis: -1)   // (N,)
    let tyArr = transformed[.ellipsis, 1..<2].squeezed(axis: -1)
    let twArr = transformed[.ellipsis, 2..<3].squeezed(axis: -1)
    let eps   = MLXArray(Float(1e-8))
    let safeW  = MLX.maximum(MLX.abs(twArr), eps) * MLX.sign(twArr + eps)
    let srcX   = txArr / safeW   // (N,) source x coords
    let srcY   = tyArr / safeW   // (N,) source y coords

    // Integer corner coordinates for bilinear sampling.
    let x0 = MLX.floor(srcX).asType(.int32)  // (N,)
    let y0 = MLX.floor(srcY).asType(.int32)
    let x1 = x0 + 1
    let y1 = y0 + 1

    // Clamp to [0, W-1] × [0, H-1].
    let x0c = clip(x0, min: 0, max: W - 1)
    let x1c = clip(x1, min: 0, max: W - 1)
    let y0c = clip(y0, min: 0, max: H - 1)
    let y1c = clip(y1, min: 0, max: H - 1)

    // Flat indices into the (H*W, C) flattened image.
    let Warr  = MLXArray(Int32(W))
    let idx00 = y0c * Warr + x0c   // (N,)
    let idx01 = y0c * Warr + x1c
    let idx10 = y1c * Warr + x0c
    let idx11 = y1c * Warr + x1c

    // Gather pixel values at the four corners.
    let flat  = input.reshaped([-1, C])  // (H*W, C)
    let p00   = flat.take(idx00, axis: 0)  // (N, C)
    let p01   = flat.take(idx01, axis: 0)
    let p10   = flat.take(idx10, axis: 0)
    let p11   = flat.take(idx11, axis: 0)

    // Fractional offsets for bilinear weighting — shape (N, 1) for broadcasting over C.
    let dx = (srcX - MLX.floor(srcX)).expandedDimensions(axis: 1).asType(.float32)  // (N, 1)
    let dy = (srcY - MLX.floor(srcY)).expandedDimensions(axis: 1).asType(.float32)

    let one = MLXArray(Float(1.0))
    let result = p00.asType(.float32) * (one - dx) * (one - dy)
               + p01.asType(.float32) * dx * (one - dy)
               + p10.asType(.float32) * (one - dx) * dy
               + p11.asType(.float32) * dx * dy

    return result.reshaped([outH, outW, C])
}

// MARK: - Coord grid cache

/// Returns a reusable `(outH*outW, 3)` homogeneous coordinate grid for bilinear warp.
/// Results are cached by (outH, outW) so repeated calls with the same tile size
/// (e.g. 32×32 across all N_TRIALS) skip the array construction entirely.
private nonisolated(unsafe) var _coordGridCache: [Int: MLXArray] = [:]
private func coordGrid(outH: Int, outW: Int) -> MLXArray {
    let key = outH &* 65536 &+ outW
    if let cached = _coordGridCache[key] { return cached }
    let ys = MLXArray(Array(0..<outH).map { Float($0) }, [outH])
    let xs = MLXArray(Array(0..<outW).map { Float($0) }, [outW])
    let gridY = broadcast(ys.expandedDimensions(axis: 1), to: [outH, outW])
    let gridX = broadcast(xs.expandedDimensions(axis: 0), to: [outH, outW])
    let onesG = MLXArray.ones([outH * outW], type: Float.self)
    let grid  = MLX.stacked([gridX.reshaped([-1]), gridY.reshaped([-1]), onesG], axis: 1)
    MLX.eval(grid)   // materialise once; safe here since it's constant data
    _coordGridCache[key] = grid
    return grid
}

// MARK: - Homography

/// Compute a 3×3 perspective transform matrix from four point correspondences.
///
/// Equivalent to `cv2.getPerspectiveTransform(src, dst)`.
/// - Parameters:
///   - src: 4 source points as `[[x, y], ...]`
///   - dst: 4 destination points as `[[x, y], ...]`
/// - Returns: 9-element row-major 3×3 matrix (forward: src → dst).
public func getPerspectiveTransform(src: [[Float]], dst: [[Float]]) -> [Float] {
    // Build 8×8 system A·h = b  where h = [h00,h01,h02,h10,h11,h12,h20,h21], h22=1.
    var A = [[Double]](repeating: [Double](repeating: 0, count: 8), count: 8)
    var b = [Double](repeating: 0, count: 8)
    for i in 0..<4 {
        let x = Double(src[i][0]), y = Double(src[i][1])
        let u = Double(dst[i][0]), v = Double(dst[i][1])
        A[2*i]   = [x, y, 1, 0, 0, 0, -u*x, -u*y]
        A[2*i+1] = [0, 0, 0, x, y, 1, -v*x, -v*y]
        b[2*i]   = u
        b[2*i+1] = v
    }
    guard let h = gaussianElimination(A, b) else {
        return [1,0,0, 0,1,0, 0,0,1]
    }
    return h.map(Float.init) + [Float(1.0)]
}

// MARK: - Internal helpers

/// 3×3 matrix inverse via cofactors. Returns identity if singular.
public func invertMatrix3x3(_ m: [Float]) -> [Float] {
    let a=Double(m[0]),b=Double(m[1]),c=Double(m[2])
    let d=Double(m[3]),e=Double(m[4]),f=Double(m[5])
    let g=Double(m[6]),h=Double(m[7]),k=Double(m[8])
    let det = a*(e*k - f*h) - b*(d*k - f*g) + c*(d*h - e*g)
    guard Swift.abs(det) > 1e-12 else { return [1,0,0, 0,1,0, 0,0,1] }
    let inv = 1.0 / det
    return [
        Float((e*k - f*h)*inv), Float((c*h - b*k)*inv), Float((b*f - c*e)*inv),
        Float((f*g - d*k)*inv), Float((a*k - c*g)*inv), Float((c*d - a*f)*inv),
        Float((d*h - e*g)*inv), Float((b*g - a*h)*inv), Float((a*e - b*d)*inv),
    ]
}

/// Build a normalised Gaussian kernel of size `size × size` with standard deviation `sigma`.
private func makeGaussianKernel(size: Int, sigma: Float) -> MLXArray {
    let half = size / 2
    var data = [Float](repeating: 0, count: size * size)
    var sum: Float = 0
    for i in 0..<size {
        for j in 0..<size {
            let x = Float(j - half), y = Float(i - half)
            let v = Foundation.exp(-(x*x + y*y) / (2 * sigma * sigma))
            data[i * size + j] = v
            sum += v
        }
    }
    let invSum = 1.0 / sum
    return MLXArray(data.map { $0 * invSum }, [size, size])
}

/// Gaussian elimination on an (n×n) system, returns nil if singular.
private func gaussianElimination(_ A: [[Double]], _ b: [Double]) -> [Double]? {
    let n = b.count
    var aug = A.enumerated().map { (i, row) in row + [b[i]] }
    for col in 0..<n {
        // Partial pivot
        var maxRow = col
        for row in (col+1)..<n where Swift.abs(aug[row][col]) > Swift.abs(aug[maxRow][col]) {
            maxRow = row
        }
        aug.swapAt(col, maxRow)
        guard Swift.abs(aug[col][col]) > 1e-12 else { return nil }
        let pivot = aug[col][col]
        for j in col...n { aug[col][j] /= pivot }
        for row in 0..<n where row != col {
            let factor = aug[row][col]
            for j in col...n { aug[row][j] -= factor * aug[col][j] }
        }
    }
    return (0..<n).map { aug[$0][n] }
}
