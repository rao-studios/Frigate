//  WHAT: A gradient traced on one thread beside compiled forwards on another (MLX/Transforms+Compile.swift). `vjp` holds
//        `evalLock` while it traces, and the trace calls compiled functions (`silu`); a compiled call that took its own
//        lock before `evalLock` waited on the trace, which waited on it, for good.

import Foundation
import MLX
import MLXNN
import Testing

private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

@Suite("Compiled functions and gradients on two threads", .enabled(if: mlxFunctional))
struct CompiledLockTests {
    @Test("a gradient through a compiled function beside its forwards on another thread finishes", .timeLimit(.minutes(1)))
    func aGradientBesideACompiledForwardFinishes() async throws {
        nonisolated(unsafe) let x = MLXRandom.normal([64, 64], key: MLXRandom.key(7))
        async let gradients: Void = Task.detached {
            for _ in 0 ..< 200 { eval(grad({ (a: MLXArray) -> MLXArray in silu(a).sum() })(x)) }
        }.value
        async let forwards: Void = Task.detached {
            for _ in 0 ..< 2_000 { eval(silu(x)) }
        }.value
        _ = await (gradients, forwards)
    }
}
