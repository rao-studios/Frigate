//  WHAT: Frigate's two changes to how MLX hands work to the GPU: `GPU.scheduling` (the command-buffer commit limits and
//        the in-flight cap an engine sets around a decode step, `DecodeSchedulingModel`) and the custom-kernel source
//        written once per kernel name. Results stay the same under any scheduling, and two kernels that share a name
//        never share a source.

import Foundation
import MLX
import Testing

private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

@Suite("GPU scheduling and kernel sources", .enabled(if: mlxFunctional), .serialized)
struct GPUSchedulingTests {
    @Test("the scheduling set is the scheduling read, and MLX's own is all zeros")
    func roundTrip() {
        let before = GPU.scheduling
        defer { GPU.scheduling = before }
        #expect(GPU.Scheduling.mlx == GPU.Scheduling(ops: 0, megabytes: 0, inFlight: 0))
        GPU.scheduling = GPU.Scheduling(ops: 200, megabytes: 100_000, inFlight: 64)
        #if !os(Linux)
        #expect(GPU.scheduling == GPU.Scheduling(ops: 200, megabytes: 100_000, inFlight: 64))
        #endif
        GPU.scheduling = .mlx
        #expect(GPU.scheduling == .mlx)
    }

    @Test("a chain of small kernels gives the same result committed one by one, one in flight, or all at once")
    func resultsDoNotDependOnScheduling() {
        let before = GPU.scheduling
        defer { GPU.scheduling = before }
        let x = MLXRandom.normal([256, 256], key: MLXRandom.key(3))
        func chain() -> MLXArray {
            var y = x
            for i in 0 ..< 200 { y = tanh(y * Float(1 + i % 3)) + x }
            return y.sum(axis: 0)
        }
        GPU.scheduling = .mlx
        let reference = chain()
        eval(reference)
        for scheduling in [GPU.Scheduling(ops: 1, megabytes: 1, inFlight: 1),
                           GPU.Scheduling(ops: 100_000, megabytes: 100_000, inFlight: 1_000)] {
            GPU.scheduling = scheduling
            let y = chain()
            eval(y)
            #expect(abs(y - reference).max().item(Float.self) == 0, "\(scheduling)")
        }
    }

    @Test("two kernels that share a name keep their own sources, each call after the first included")
    func aSharedNameKeepsItsOwnSource() {
        let double = MLXFast.metalKernel(
            name: "frigate_shared_name", inputNames: ["x"], outputNames: ["out"],
            source: "uint i = thread_position_in_grid.x; out[i] = x[i] * 2;")
        let negate = MLXFast.metalKernel(
            name: "frigate_shared_name", inputNames: ["x"], outputNames: ["out"],
            source: "uint i = thread_position_in_grid.x; out[i] = -x[i];")
        let x = MLXArray((0 ..< 64).map { Float($0) })
        for _ in 0 ..< 3 {
            let a = double([x], grid: (64, 1, 1), threadGroup: (64, 1, 1), outputShapes: [[64]], outputDTypes: [.float32])[0]
            let b = negate([x], grid: (64, 1, 1), threadGroup: (64, 1, 1), outputShapes: [[64]], outputDTypes: [.float32])[0]
            #expect(abs(a - x * 2).max().item(Float.self) == 0)
            #expect(abs(b + x).max().item(Float.self) == 0)
        }
    }
}
