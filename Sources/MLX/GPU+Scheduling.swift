//
//  GPU+Scheduling.swift
//  MLX
//
//  WHAT: How MLX hands work to the GPU, set at run time: how many kernels or megabytes of inputs go in one command
//        buffer before it is committed, and how many committed buffers may run before an evaluation waits for one.
//  WHY:  MLX reads the first two once, at start (`MLX_MAX_OPS_PER_BUFFER`, `MLX_MAX_MB_PER_BUFFER`), and fixes the third
//        at 10, so one process can't give a decode step's thousands of small kernels a deeper queue than a prefill's
//        large ones.
//  PIN:  APPLE ONLY. The binding lives in mlx-c's metal.cpp, which Linux builds leave out; there `scheduling` is always
//        MLX's own and setting it does nothing.
//  FRIGATE(gpu-scheduling): added, not in upstream mlx-swift. The vendored MLX reads the overrides (device.cpp,
//        transforms.cpp) and mlx-c's metal.cpp binds them.
//

import Cmlx
import Foundation

extension GPU {
    /// `ops` kernels or `megabytes` of inputs go in one command buffer before it is committed, and `inFlight`
    /// committed buffers may run before an evaluation waits for one. Zero is MLX's own: the device's limits by
    /// architecture (50 ops and 50 MB on an M-series Max) and 10 in flight.
    public struct Scheduling: Equatable, Sendable {
        public var ops: Int
        public var megabytes: Int
        public var inFlight: Int

        public init(ops: Int = 0, megabytes: Int = 0, inFlight: Int = 0) {
            self.ops = ops
            self.megabytes = megabytes
            self.inFlight = inFlight
        }

        /// MLX's own.
        public static let mlx = Scheduling()
    }

    /// Runs `body` with `scheduling` set and puts back what was set before, holding the eval lock throughout: an
    /// evaluation on another thread never encodes under it, and two engines never put back each other's.
    public static func withScheduling<R>(_ scheduling: Scheduling, _ body: () throws -> R) rethrows -> R {
        try withEvalLock {
            let before = Self.scheduling
            Self.scheduling = scheduling
            defer { Self.scheduling = before }
            return try body()
        }
    }

    /// What is set now; set it through `withScheduling` unless it should stay.
    public static var scheduling: Scheduling {
        get {
            #if os(Linux)
                return .mlx
            #else
                var ops: Int32 = 0, megabytes: Int32 = 0, inFlight: Int32 = 0
                mlx_gpu_scheduling(&ops, &megabytes, &inFlight)
                return Scheduling(ops: Int(ops), megabytes: Int(megabytes), inFlight: Int(inFlight))
            #endif
        }
        set {
            #if !os(Linux)
                mlx_set_gpu_scheduling(Int32(newValue.ops), Int32(newValue.megabytes), Int32(newValue.inFlight))
            #endif
        }
    }
}
