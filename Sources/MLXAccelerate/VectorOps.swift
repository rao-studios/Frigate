/// MLXAccelerate — Batched vector-quantization ops backed by MLX.
///
/// Serves product-quantization workloads (Totem's PartitionQuantizer): pairwise
/// distance matrices, batched k-means codebook training, nearest-centroid
/// encoding, and ADC distance tables. All ops are expressed as MLX graphs so
/// they run on CPU, CUDA, or Metal with no platform-specific code.
///
/// Each public function scopes the MLX default device via
/// `Device.withDefaultDevice` so that *every* op in its graph (including
/// operators like `*` and `+`, which cannot take a `stream:` argument) lands on
/// the selected backend.
///
/// Shape convention: a leading batch axis `s` (e.g. PQ subvector count) lets a
/// caller train/encode all subspaces in a single dispatch:
///   data       `(s, n, d)` — n vectors of dimension d per subspace
///   centroids  `(s, k, d)` — k centroids per subspace
///
/// Functions return lazy `MLXArray`s unless documented otherwise; callers
/// control `eval`/materialization.

import Foundation
import MLX

// MARK: - Device Selection

public enum VectorOpsDevice {
    /// Resolved compute target for MLXAccelerate vector ops.
    ///
    /// GPU on Linux (CUDA build); CPU on Darwin — the vendored package ships no
    /// compiled metallib, so MLX's Metal backend cannot initialize there and any
    /// GPU-stream op aborts the process. Override with
    /// `FRIGATE_VECTOROPS_DEVICE=cpu|gpu`.
    public static let recommended: Device = {
        switch ProcessInfo.processInfo.environment["FRIGATE_VECTOROPS_DEVICE"]?.lowercased() {
        case "cpu": return .cpu
        case "gpu": return .gpu
        default:
//            #if os(Linux)
//            return .gpu
//            #else
//            return .cpu
//            #endif
            return .gpu
        }
    }()
}

// MARK: - Pairwise Squared Distances

/// Squared Euclidean distance from every row of `x` to every row of `y`.
///
/// `x`: `(..., n, d)`, `y`: `(..., k, d)` → `(..., n, k)` where
/// `out[..., i, j] = ‖x[..., i, :] − y[..., j, :]‖²`.
///
/// Computed as `‖x‖² + ‖y‖² − 2·x·yᵀ` (one matmul instead of n×k vector
/// subtractions), clamped at 0 to absorb floating-point cancellation.
public func pairwiseSquaredDistances(
    _ x: MLXArray,
    _ y: MLXArray,
    device: Device = VectorOpsDevice.recommended
) -> MLXArray {
    Device.withDefaultDevice(device) {
        let xNorms = (x * x).sum(axis: -1, keepDims: true)                       // (..., n, 1)
        let yNorms = swappedAxes((y * y).sum(axis: -1, keepDims: true), -1, -2)  // (..., 1, k)
        let cross = matmul(x, swappedAxes(y, -1, -2))                            // (..., n, k)
        return maximum(xNorms + yNorms - 2 * cross, MLXArray(Float(0)))
    }
}

// MARK: - Nearest Centroids (encode)

/// Assign each vector to its nearest centroid.
///
/// `data`: `(s, n, d)`, `centroids`: `(s, k, d)` →
/// `codes (s, n)` int32 centroid indices and `squaredDistances (s, n)`,
/// the squared distance of each vector to its assigned centroid.
public func nearestCentroids(
    _ data: MLXArray,
    centroids: MLXArray,
    device: Device = VectorOpsDevice.recommended
) -> (codes: MLXArray, squaredDistances: MLXArray) {
    Device.withDefaultDevice(device) {
        let d2 = pairwiseSquaredDistances(data, centroids, device: device)  // (s, n, k)
        let codes = argMin(d2, axis: -1)                                    // (s, n)
        let minD2 = d2.min(axis: -1)                                        // (s, n)
        return (codes, minD2)
    }
}

// MARK: - ADC Distance Table

/// Per-subspace distance table for Asymmetric Distance Computation.
///
/// `query`: `(s, d)` — one query split into s subvectors;
/// `codebooks`: `(s, k, d)` → `(s, k)` where `out[i, j]` is the **Euclidean**
/// (sqrt) distance from query subvector i to centroid j of codebook i.
public func centroidDistanceTable(
    query: MLXArray,
    codebooks: MLXArray,
    device: Device = VectorOpsDevice.recommended
) -> MLXArray {
    Device.withDefaultDevice(device) {
        let q = query.expandedDimensions(axis: 1)                             // (s, 1, d)
        let d2 = pairwiseSquaredDistances(q, codebooks, device: device)       // (s, 1, k)
        return sqrt(d2.squeezed(axis: 1))                                     // (s, k)
    }
}

// MARK: - Batched K-Means

/// Batched Lloyd's k-means over a leading subspace axis.
///
/// `data`: `(s, n, d)` with `n >= k`. Trains `s` independent codebooks in one
/// graph per iteration. Initialization is k-means++ (D² sampling via the
/// Gumbel-max trick, one dispatch per seed) for `k <= 64`; larger `k` falls
/// back to sampling k distinct data rows per subspace, since k sequential
/// seeding rounds would dominate runtime.
///
/// Differences from a textbook scalar implementation (acceptable for codebook
/// training, where downstream tests assert assignment properties rather than
/// exact centroids):
/// - Empty clusters retain their previous centroid instead of reseeding to a
///   random vector.
/// - Convergence is checked on the max centroid shift across all subspaces.
///
/// Returns materialized `centroids (s, k, d)` float32, final `codes (s, n)`
/// int32 assignments, and `squaredDistances (s, n)` — each vector's squared
/// distance to its assigned centroid (callers use these for reconstruction-
/// error calibration without a second encode pass).
public func kmeans(
    _ data: MLXArray,
    k: Int,
    maxIterations: Int = 20,
    tolerance: Float = 1e-3,
    device: Device = VectorOpsDevice.recommended
) -> (centroids: MLXArray, codes: MLXArray, squaredDistances: MLXArray) {
    Device.withDefaultDevice(device) {
        let n = data.shape[1]
        precondition(data.ndim == 3, "kmeans expects (s, n, d) data")
        precondition(n >= k, "kmeans requires n >= k (caller should pass vectors through when n < k)")

        var centroids = kmeansSeeds(data, k: k)                                 // (s, k, d)

        let toleranceSquared = tolerance * tolerance
        for _ in 0..<maxIterations {
            let d2 = pairwiseSquaredDistances(data, centroids, device: device)  // (s, n, k)
            let codes = argMin(d2, axis: -1)                                    // (s, n)

            // One-hot assignment matrix → per-cluster sums and counts in two matmul-shaped ops.
            let clusterIds = MLXArray(Array(0..<Int32(k)))                      // (k,)
            let oneHot = (codes.expandedDimensions(axis: -1) .== clusterIds)
                .asType(.float32)                                               // (s, n, k)
            let counts = oneHot.sum(axis: 1)                                    // (s, k)
            let sums = matmul(swappedAxes(oneHot, -1, -2), data)                // (s, k, d)
            let safeCounts = maximum(counts, MLXArray(Float(1)))
                .expandedDimensions(axis: -1)                                   // (s, k, 1)
            let updated = sums / safeCounts

            // Empty clusters keep their previous centroid.
            let empty = (counts .== MLXArray(Float(0))).expandedDimensions(axis: -1)  // (s, k, 1)
            let next = which(empty, centroids, updated)

            let shift2 = ((next - centroids) * (next - centroids))
                .sum(axis: -1)
                .max()                                                          // scalar
            centroids = next
            eval(centroids, shift2)
            if shift2.item(Float.self) < toleranceSquared { break }
        }

        let (codes, squaredDistances) = nearestCentroids(data, centroids: centroids, device: device)
        eval(codes, squaredDistances)
        return (centroids, codes, squaredDistances)
    }
}

// MARK: - Seeding

/// K-means++ seeds for `k <= 64` (D² sampling, one dispatch per seed);
/// k distinct random rows per subspace otherwise.
/// Runs on the caller's scoped default device.
private func kmeansSeeds(_ data: MLXArray, k: Int) -> MLXArray {
    let s = data.shape[0], n = data.shape[1], d = data.shape[2]

    func gatherRows(_ rowIndices: [Int32], rowsPerSubspace: Int) -> MLXArray {
        // indices (s, rows, 1) broadcast to (s, rows, d) for takeAlong on axis 1.
        let idx = MLXArray(rowIndices, [s, rowsPerSubspace, 1]).asType(.int32)
        return takeAlong(data, broadcast(idx, to: [s, rowsPerSubspace, d]), axis: 1)
    }

    guard k <= 64 else {
        // Distinct random rows per subspace.
        var indices: [Int32] = []
        indices.reserveCapacity(s * k)
        for _ in 0..<s {
            indices.append(contentsOf: Array(0..<Int32(n)).shuffled().prefix(k))
        }
        return gatherRows(indices, rowsPerSubspace: k)
    }

    // K-means++ — first seed uniform per subspace.
    let first = (0..<s).map { _ in Int32.random(in: 0..<Int32(n)) }
    var centroids = gatherRows(first, rowsPerSubspace: 1)                // (s, 1, d)
    var minSqDist: MLXArray? = nil                                       // (s, n)

    for _ in 1..<k {
        let newest = centroids[0..., (centroids.shape[1] - 1)...]        // (s, 1, d)
        let diff = data - newest
        let d2 = (diff * diff).sum(axis: -1)                             // (s, n)
        let dist = minSqDist.map { minimum($0, d2) } ?? d2
        minSqDist = dist

        // Gumbel-max sampling: argmax(log(D²) + G) ~ Categorical(D² / ΣD²).
        // log(0) = -inf correctly excludes already-chosen points.
        let uniform = (0..<(s * n)).map { _ in Float.random(in: Float.ulpOfOne..<1) }
        let gumbel = -log(-log(MLXArray(uniform, [s, n])))
        let scores = log(dist) + gumbel
        let idx = argMax(scores, axis: -1)                               // (s,)
            .asType(.int32)
            .reshaped([s, 1, 1])
        let chosen = takeAlong(data, broadcast(idx, to: [s, 1, d]), axis: 1)
        centroids = concatenated([centroids, chosen], axis: 1)
    }

    return centroids
}
