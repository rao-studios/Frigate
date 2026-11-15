import Foundation
import MLX
import Testing

/// MLX graph ops abort on Darwin unless a metallib is beside the test bundle. Same gate as Gemma4SanitizeTests.
private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

/// The multi-block merge sort past 1,023 blocks. A float32 row is sorted in blocks of 2,048, and each merge pass
/// partitions the row's blocks first. The partition kernel offset each row's partitions by its threadgroup width,
/// min(blocks + 1, 1024), and the merge kernel read them at blocks + 1, so past 2,095,104 columns the rows of one call
/// overlapped: argPartition over Braid's 3,038,932-entry provenance index returned repeats, zeros and ids out of range
/// for every row of a batched query, while a lone row was right (2026-11-15). Upstream MLX offsets by blocks + 1.
@Suite("Sort past 1,023 blocks", .enabled(if: mlxFunctional))
struct SortTests {
    /// Three rows at 2,097,152 columns (1,024 blocks, the first width past the bound) and at 3,038,932 (1,484 blocks).
    static let widths = [2_097_152, 3_038_932]

    @Test func argPartitionKeepsEveryRowsTopEntries() {
        MLXRandom.seed(7)
        let m = 24
        for n in Self.widths {
            let scores = MLXRandom.normal([3, n])
            let top = argPartition(scores, kth: n - m, axis: -1)[0..., (n - m) ..< n]
            let picked = takeAlong(scores, top, axis: -1)
            eval(scores, top, picked)
            for row in 0 ..< 3 {
                let expected = Array(scores[row].asArray(Float.self).sorted(by: >).prefix(m))
                #expect(picked[row].asArray(Float.self).sorted(by: >) == expected, "\(n) columns, row \(row)")
                #expect(Set(top[row].asArray(UInt32.self)).count == m, "\(n) columns, row \(row): a repeated index")
            }
        }
    }

    @Test func sortOrdersEveryRow() {
        MLXRandom.seed(9)
        for n in Self.widths {
            let values = MLXRandom.normal([3, n])
            let sortedRows = MLX.sorted(values, axis: -1)
            eval(values, sortedRows)
            for row in 0 ..< 3 {
                #expect(sortedRows[row].asArray(Float.self) == values[row].asArray(Float.self).sorted(), "\(n) columns, row \(row)")
            }
        }
    }
}
