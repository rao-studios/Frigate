import Foundation
import Testing
import MLX
@testable import MLXAccelerate

/// MLX graph ops abort on Darwin test runs (no metallib in the test bundle — see
/// VectorOpsTests.swift). Host-side tap-table math runs everywhere; the gated
/// suite covers the MLX graph paths on Linux or with FRIGATE_MLX_TESTS=1.
private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

// MARK: - Host-side references

/// Reference correlation with cv2 semantics (anchor k/2, chosen border) — direct loops.
private func refFilter2D(
    _ src: [[Float]], _ kernel: [[Float]], reflect101: Bool
) -> [[Float]] {
    let H = src.count, W = src[0].count
    let kH = kernel.count, kW = kernel[0].count
    let ay = kH / 2, ax = kW / 2
    var out = [[Float]](repeating: [Float](repeating: 0, count: W), count: H)
    for y in 0..<H {
        for x in 0..<W {
            var acc: Float = 0
            for j in 0..<kH {
                for i in 0..<kW {
                    let sy = y + j - ay, sx = x + i - ax
                    let v: Float
                    if reflect101 {
                        v = src[reflect101Index(sy, H)][reflect101Index(sx, W)]
                    } else {
                        v = (sy >= 0 && sy < H && sx >= 0 && sx < W) ? src[sy][sx] : 0
                    }
                    acc += kernel[j][i] * v
                }
            }
            out[y][x] = acc
        }
    }
    return out
}

private func seededGrid(_ h: Int, _ w: Int, seed: UInt64) -> [[Float]] {
    var state = seed &* 6364136223846793005 &+ 1442695040888963407
    return (0..<h).map { _ in
        (0..<w).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int(bitPattern: UInt(state >> 33)) % 1000) / 500.0 - 1.0
        }
    }
}

// MARK: - Host math (runs everywhere)

@Suite("BatchedImageOpsHost")
struct BatchedImageOpsHostTests {

    @Test("reflect101Index matches cv2 borderInterpolate")
    func reflect101() {
        // n = 5: ... 2 1 | 0 1 2 3 4 | 3 2 ...
        #expect(reflect101Index(-2, 5) == 2)
        #expect(reflect101Index(-1, 5) == 1)
        #expect(reflect101Index(0, 5) == 0)
        #expect(reflect101Index(4, 5) == 4)
        #expect(reflect101Index(5, 5) == 3)
        #expect(reflect101Index(6, 5) == 2)
        #expect(reflect101Index(-1, 1) == 0)
        // 32-wide tile, tap range −3..34 used by the warp tables
        #expect(reflect101Index(-3, 32) == 3)
        #expect(reflect101Index(32, 32) == 30)
        #expect(reflect101Index(34, 32) == 28)
    }

    @Test("getPerspectiveTransformD maps src corners onto dst corners")
    func homographyD() {
        let src: [[Float]] = [[0, 0], [32, 0], [32, 32], [0, 32]]
        let dst: [[Float]] = [[1.5, -0.7], [33.1, 0.4], [31.2, 33.0], [-0.9, 31.6]]
        let m = getPerspectiveTransformD(src: src, dst: dst)
        for c in 0..<4 {
            let x = Double(src[c][0]), y = Double(src[c][1])
            let w = m[6] * x + m[7] * y + m[8]
            let u = (m[0] * x + m[1] * y + m[2]) / w
            let v = (m[3] * x + m[4] * y + m[5]) / w
            #expect(abs(u - Double(dst[c][0])) < 1e-9, "corner \(c) x")
            #expect(abs(v - Double(dst[c][1])) < 1e-9, "corner \(c) y")
        }
    }

    @Test("invertMatrix3x3D inverts to identity")
    func inverseD() {
        let m: [Double] = [1.02, 0.03, -1.4, -0.01, 0.97, 2.2, 0.0001, -0.0002, 1.0]
        let inv = invertMatrix3x3D(m)
        // m · inv = I
        for r in 0..<3 {
            for c in 0..<3 {
                var acc = 0.0
                for k in 0..<3 { acc += m[r * 3 + k] * inv[k * 3 + c] }
                #expect(abs(acc - (r == c ? 1.0 : 0.0)) < 1e-12, "(\(r),\(c))")
            }
        }
    }

    @Test("identity warp taps: weight 1 on the center tap, index = pixel")
    func identityTaps() {
        let eye: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        let t = buildBicubicWarpTapsHost(
            dstToSrc: [eye], height: 8, width: 8, sourceIsTrialStacked: false)
        // f = 0 → cubic weights (0, 1, 0, 0); the unit tap is k = 1*4 + 1 = 5.
        for p in 0..<64 {
            #expect(abs(t.weights[5][p] - 1) < 1e-7)
            #expect(t.indices[5][p] == Int32(p))
            for k in 0..<16 where k != 5 {
                #expect(abs(t.weights[k][p]) < 1e-7, "tap \(k) px \(p)")
            }
        }
    }

    @Test("bicubic weights at f=0.5 match cv2 interpolateCubic (A=-0.75)")
    func halfPixelWeights() {
        // Pure x-translation by 0.5: dst (x,y) → src (x+0.5, y).
        let shift: [Double] = [1, 0, 0.5, 0, 1, 0, 0, 0, 1]
        let t = buildBicubicWarpTapsHost(
            dstToSrc: [shift], height: 8, width: 8, sourceIsTrialStacked: false)
        // wy = (0,1,0,0), wx = (−0.09375, 0.59375, 0.59375, −0.09375)
        let p = 3 * 8 + 4   // interior pixel
        let expected: [Float] = [-0.09375, 0.59375, 0.59375, -0.09375]
        for i in 0..<4 {
            #expect(abs(t.weights[4 + i][p] - expected[i]) < 1e-7, "tap x-offset \(i)")
        }
        // weights on the j≠1 rows are zero
        #expect(abs(t.weights[0][p]) < 1e-7)
        #expect(abs(t.weights[9][p]) < 1e-7)
    }

    @Test("trial-stacked tables offset indices by t·H·W")
    func trialStacking() {
        let eye: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        let t = buildBicubicWarpTapsHost(
            dstToSrc: [eye, eye, eye], height: 4, width: 4, sourceIsTrialStacked: true)
        for trial in 0..<3 {
            for p in 0..<16 {
                #expect(t.indices[5][trial * 16 + p] == Int32(trial * 16 + p))
            }
        }
    }

    @Test("quantizeLikeCV2 snaps fractions to 1/32 steps")
    func quantization() {
        // Translation by 0.013 px (< 1/64) quantizes to fraction 0 → unit center tap.
        let tiny: [Double] = [1, 0, 0.013, 0, 1, 0, 0, 0, 1]
        let t = buildBicubicWarpTapsHost(
            dstToSrc: [tiny], height: 8, width: 8, sourceIsTrialStacked: false,
            quantizeLikeCV2: true)
        #expect(abs(t.weights[5][0] - 1) < 1e-7)
        // Translation by 0.5 + 1/128 rounds to 16/32 = 0.5 exactly (round-half-even on 16.25 → 16).
        let half: [Double] = [1, 0, 0.5 + 1.0 / 128.0, 0, 1, 0, 0, 0, 1]
        let q = buildBicubicWarpTapsHost(
            dstToSrc: [half], height: 8, width: 8, sourceIsTrialStacked: false,
            quantizeLikeCV2: true)
        let p = 3 * 8 + 4
        #expect(abs(q.weights[5][p] - 0.59375) < 1e-7)
    }
}

// MARK: - MLX graph ops (gated)

@Suite("BatchedImageOps", .enabled(if: mlxFunctional))
struct BatchedImageOpsMLXTests {

    @Test("take(axis: 1) broadcasts indices over the batch dim")
    func takeSemantics() {
        // [2, 4, 3] source; indices [0, 2, 2] along axis 1 → [2, 3, 3]
        let src = MLXArray((0..<24).map(Float.init), [2, 4, 3])
        let out = src.take(MLXArray([0, 2, 2] as [Int32]), axis: 1)
        #expect(out.shape == [2, 3, 3])
        let v = out.asArray(Float.self)
        // batch 0 row 1 should be source row 2: [6, 7, 8]
        #expect(v[3] == 6 && v[4] == 7 && v[5] == 8)
        // batch 1 row 0 should be source row 0 of batch 1: [12, 13, 14]
        #expect(v[9] == 12 && v[10] == 13 && v[11] == 14)
    }

    @Test("batchedFilter2D matches host reference (odd 3×3, reflect101)")
    func filterOdd() {
        let grid = seededGrid(9, 7, seed: 11)
        let kernel: [[Float]] = [[0.1, -0.2, 0.3], [0.0, 0.5, -0.1], [0.2, 0.1, 0.1]]
        let expected = refFilter2D(grid, kernel, reflect101: true)
        let out = batchedFilter2D(
            MLXArray(grid.flatMap { $0 }, [1, 9, 7]),
            kernel: MLXArray(kernel.flatMap { $0 }, [3, 3])
        ).asArray(Float.self)
        for y in 0..<9 {
            for x in 0..<7 {
                #expect(abs(out[y * 7 + x] - expected[y][x]) < 1e-5, "(\(y),\(x))")
            }
        }
    }

    @Test("batchedFilter2D matches host reference (even 2×2 box, cv2 anchor)")
    func filterEven() {
        let grid = seededGrid(8, 8, seed: 12)
        let kernel: [[Float]] = [[0.25, 0.25], [0.25, 0.25]]
        let expected = refFilter2D(grid, kernel, reflect101: true)
        let out = batchedFilter2D(
            MLXArray(grid.flatMap { $0 }, [1, 8, 8]),
            kernel: MLXArray(kernel.flatMap { $0 }, [2, 2])
        ).asArray(Float.self)
        for y in 0..<8 {
            for x in 0..<8 {
                #expect(abs(out[y * 8 + x] - expected[y][x]) < 1e-5, "(\(y),\(x))")
            }
        }
    }

    @Test("identity warp table is a passthrough")
    func identityWarp() {
        let eye: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        let table = buildBicubicWarpTable(
            dstToSrc: [eye], height: 4, width: 4, sourceIsTrialStacked: false)
        let tiles = MLXArray((0..<(2 * 16 * 3)).map(Float.init), [2, 16, 3])
        let out = applyWarpTable(tiles, table).asArray(Float.self)
        let expected = tiles.asArray(Float.self)
        for i in 0..<expected.count {
            #expect(abs(out[i] - expected[i]) < 1e-5, "i \(i)")
        }
    }
}
