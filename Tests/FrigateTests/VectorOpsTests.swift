import Foundation
import Testing
import MLX
@testable import MLXAccelerate

// MARK: - Scalar references

private func refSquaredDistance(_ a: [Float], _ b: [Float]) -> Float {
    var sum: Float = 0
    for i in 0..<a.count {
        let d = a[i] - b[i]
        sum += d * d
    }
    return sum
}

private func seededVectors(_ count: Int, dim: Int, seed: UInt64) -> [[Float]] {
    var state = seed &* 6364136223846793005 &+ 1442695040888963407
    return (0..<count).map { _ in
        (0..<dim).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int(bitPattern: UInt(state >> 33)) % 1000) / 500.0 - 1.0
        }
    }
}

/// MLX graph ops abort on Darwin: the vendored Cmlx compiles the Metal backend
/// but ships no metallib, so the Metal allocator fatals on first array creation.
/// These tests therefore run on Linux (CUDA/CPU backends) by default, or
/// anywhere with `FRIGATE_MLX_TESTS=1` (e.g. a future Darwin CPU-only build).
private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

// MARK: - CPU kernels (run everywhere)

@Suite("VectorOpsCPU")
struct VectorOpsCPUTests {

    @Test("squaredEuclidean matches scalar reference")
    func cpuSquaredEuclidean() {
        for dim in [1, 7, 8, 9, 63, 64, 65, 1024] {
            let a = seededVectors(1, dim: dim, seed: 60)[0]
            let b = seededVectors(1, dim: dim, seed: 61)[0]
            let expected = refSquaredDistance(a, b)
            let got = squaredEuclidean(a, b)
            #expect(abs(got - expected) <= max(1e-4, expected * 1e-5), "dim \(dim)")
        }
    }

    @Test("vDSP_dotpr matches scalar reference")
    func cpuDotProduct() {
        let a = seededVectors(1, dim: 1024, seed: 70)[0]
        let b = seededVectors(1, dim: 1024, seed: 71)[0]
        var expected: Float = 0
        for i in 0..<1024 { expected += a[i] * b[i] }
        var got: Float = 0
        vDSP_dotpr(a, 1, b, 1, &got, 1024)
        #expect(abs(got - expected) <= max(1e-2, abs(expected) * 1e-4))
    }
}

// MARK: - MLX graph ops (gated)

@Suite("VectorOps", .enabled(if: mlxFunctional))
struct VectorOpsTests {

    @Test("pairwiseSquaredDistances matches scalar reference (batched 3D)")
    func pairwiseMatchesReference() {
        let s = 4, n = 7, k = 5, d = 16
        let xRows = seededVectors(s * n, dim: d, seed: 1)
        let yRows = seededVectors(s * k, dim: d, seed: 2)
        let x = MLXArray(xRows.flatMap { $0 }, [s, n, d])
        let y = MLXArray(yRows.flatMap { $0 }, [s, k, d])

        let out = pairwiseSquaredDistances(x, y).asArray(Float.self)  // (s, n, k)

        for si in 0..<s {
            for ni in 0..<n {
                for ki in 0..<k {
                    let expected = refSquaredDistance(xRows[si * n + ni], yRows[si * k + ki])
                    let got = out[si * n * k + ni * k + ki]
                    #expect(abs(got - expected) <= max(1e-3, expected * 1e-4),
                            "mismatch at (\(si),\(ni),\(ki)): \(got) vs \(expected)")
                }
            }
        }
    }

    @Test("pairwiseSquaredDistances is non-negative even for identical rows")
    func pairwiseNonNegative() {
        let rows = seededVectors(6, dim: 32, seed: 3)
        let x = MLXArray(rows.flatMap { $0 }, [1, 6, 32])
        let out = pairwiseSquaredDistances(x, x).asArray(Float.self)
        for v in out { #expect(v >= 0) }
    }

    @Test("nearestCentroids returns scalar-argmin assignments and distances")
    func nearestCentroidsOptimal() {
        let s = 3, n = 10, k = 4, d = 8
        let dataRows = seededVectors(s * n, dim: d, seed: 10)
        let centRows = seededVectors(s * k, dim: d, seed: 11)
        let data = MLXArray(dataRows.flatMap { $0 }, [s, n, d])
        let cents = MLXArray(centRows.flatMap { $0 }, [s, k, d])

        let (codes, d2) = nearestCentroids(data, centroids: cents)
        let codesOut = codes.asArray(Int32.self)
        let d2Out = d2.asArray(Float.self)

        for si in 0..<s {
            for ni in 0..<n {
                var best = 0
                var bestDist = Float.infinity
                for ki in 0..<k {
                    let dist = refSquaredDistance(dataRows[si * n + ni], centRows[si * k + ki])
                    if dist < bestDist { bestDist = dist; best = ki }
                }
                #expect(Int(codesOut[si * n + ni]) == best)
                #expect(abs(d2Out[si * n + ni] - bestDist) <= max(1e-3, bestDist * 1e-4))
            }
        }
    }

    @Test("centroidDistanceTable matches scalar sqrt distances")
    func distanceTableMatchesReference() {
        let s = 4, k = 6, d = 16
        let queryRows = seededVectors(s, dim: d, seed: 20)
        let centRows = seededVectors(s * k, dim: d, seed: 21)
        let query = MLXArray(queryRows.flatMap { $0 }, [s, d])
        let cents = MLXArray(centRows.flatMap { $0 }, [s, k, d])

        let table = centroidDistanceTable(query: query, codebooks: cents).asArray(Float.self)  // (s, k)

        for si in 0..<s {
            for ki in 0..<k {
                let expected = refSquaredDistance(queryRows[si], centRows[si * k + ki]).squareRoot()
                let got = table[si * k + ki]
                #expect(abs(got - expected) <= max(1e-3, expected * 1e-3))
            }
        }
    }

    @Test("kmeans codes are optimal assignments to returned centroids")
    func kmeansCodesOptimal() {
        let s = 2, n = 60, k = 4, d = 8
        let dataRows = seededVectors(s * n, dim: d, seed: 30)
        let data = MLXArray(dataRows.flatMap { $0 }, [s, n, d])

        let (centroids, codes, d2) = kmeans(data, k: k)
        #expect(centroids.shape == [s, k, d])
        #expect(codes.shape == [s, n])
        #expect(d2.shape == [s, n])

        let centOut = centroids.asArray(Float.self)
        let codesOut = codes.asArray(Int32.self)
        let d2Out = d2.asArray(Float.self)

        for si in 0..<s {
            for ni in 0..<n {
                let assigned = Int(codesOut[si * n + ni])
                let row = dataRows[si * n + ni]
                let assignedDist = refSquaredDistance(
                    row, Array(centOut[(si * k + assigned) * d ..< (si * k + assigned + 1) * d]))
                #expect(abs(d2Out[si * n + ni] - assignedDist) <= max(1e-3, assignedDist * 1e-4))
                for ki in 0..<k {
                    let dist = refSquaredDistance(
                        row, Array(centOut[(si * k + ki) * d ..< (si * k + ki + 1) * d]))
                    #expect(assignedDist <= dist + 1e-3,
                            "code \(assigned) is not optimal at (\(si),\(ni)): \(assignedDist) > \(dist) for centroid \(ki)")
                }
            }
        }
    }

    @Test("kmeans beats single-centroid reconstruction error")
    func kmeansReducesError() {
        let n = 80, k = 8, d = 8
        let dataRows = seededVectors(n, dim: d, seed: 40)
        let data = MLXArray(dataRows.flatMap { $0 }, [1, n, d])

        let (centroids, codes, _) = kmeans(data, k: k)
        let centOut = centroids.asArray(Float.self)
        let codesOut = codes.asArray(Int32.self)

        // Mean of all rows = best single centroid.
        var mean = [Float](repeating: 0, count: d)
        for row in dataRows { for i in 0..<d { mean[i] += row[i] } }
        for i in 0..<d { mean[i] /= Float(n) }

        var kmeansError: Float = 0
        var singleError: Float = 0
        for ni in 0..<n {
            let c = Int(codesOut[ni])
            kmeansError += refSquaredDistance(dataRows[ni], Array(centOut[c * d ..< (c + 1) * d]))
            singleError += refSquaredDistance(dataRows[ni], mean)
        }
        #expect(kmeansError < singleError)
    }

    @Test("kmeans large-k path (random-row seeding) produces valid codes")
    func kmeansLargeK() {
        let s = 2, n = 300, k = 128, d = 4
        let dataRows = seededVectors(s * n, dim: d, seed: 50)
        let data = MLXArray(dataRows.flatMap { $0 }, [s, n, d])

        let (centroids, codes, _) = kmeans(data, k: k, maxIterations: 5)
        #expect(centroids.shape == [s, k, d])
        let codesOut = codes.asArray(Int32.self)
        for c in codesOut { #expect(c >= 0 && c < Int32(k)) }
    }
}
