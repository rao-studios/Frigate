import XCTest

/// A test of the parallel engine that evaluates arrays: skipped on Darwin unless a metallib sits beside the test bundle
/// (`Scripts/test-mlx.sh`), as every MLX-touching suite here is.
class EngineTestCase: XCTestCase {
    static var mlxFunctional: Bool {
        #if os(Linux)
        return true
        #else
        return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
        #endif
    }

    override func setUpWithError() throws {
        try XCTSkipUnless(Self.mlxFunctional, "needs MLX: run Scripts/test-mlx.sh")
    }
}
