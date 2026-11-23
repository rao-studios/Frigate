import FrigateParallel
import MLX
import MLXLMCommon
import XCTest

/// The batched cache against the single-sequence cache it is made of: a row decoded in a batch
/// must hold exactly what it would hold alone.
final class BatchKVCacheTests: EngineTestCase, @unchecked Sendable {
    private let heads = 2
    private let dim = 8

    /// `[1, heads, n, dim]` of distinct values.
    private func block(_ n: Int, seed: Int) -> MLXArray {
        MLXArray(0 ..< (heads * n * dim)).asType(.float32).reshaped(1, heads, n, dim) + Float(seed * 1000)
    }

    private func simple(fed lengths: [Int], seed: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        var at = 0
        for n in lengths {
            _ = cache.update(keys: block(n, seed: seed + at), values: block(n, seed: seed + at + 7))
            at += n
        }
        return cache
    }

    private func equal(_ a: MLXArray, _ b: MLXArray) -> Bool {
        a.shape == b.shape && allClose(a, b).item(Bool.self)
    }

    func testTheFirstRowIsAdoptedAndReadsBackWhole() {
        let row = simple(fed: [5], seed: 1)
        let batch = BatchKVCache()
        batch.extend(row)
        XCTAssertEqual(batch.rows, 1)
        XCTAssertEqual(batch.cursor, 5)
        XCTAssertEqual(batch.leftPadding, [0])
        let back = batch.extract(row: 0)
        XCTAssertEqual(back.offset, 5)
        XCTAssertTrue(equal(back.state[0], row.state[0]))
        XCTAssertTrue(equal(back.state[1], row.state[1]))
    }

    func testRowsOfDifferentLengthsLineUpOnTheirLastToken() {
        let short = simple(fed: [3], seed: 1)
        let long = simple(fed: [6], seed: 2)
        let batch = BatchKVCache()
        batch.extend(short)
        batch.extend(long)
        XCTAssertEqual(batch.cursor, 6)
        XCTAssertEqual(batch.leftPadding, [3, 0])
        XCTAssertEqual(batch.lengths, [3, 6])
        XCTAssertEqual(batch.batchOffset.asArray(Int32.self), [3, 6])
        XCTAssertTrue(equal(batch.extract(row: 0).state[0], short.state[0]))
        XCTAssertTrue(equal(batch.extract(row: 1).state[1], long.state[1]))
    }

    func testAStepOnTheBatchIsAStepOnEachRow() {
        let a = simple(fed: [4], seed: 1)
        let b = simple(fed: [2], seed: 2)
        let batch = BatchKVCache()
        batch.extend(a)
        batch.extend(b)
        // One token for each row, as a decode step feeds them.
        let stepKeys = concatenated([block(1, seed: 11), block(1, seed: 12)], axis: 0)
        let stepValues = concatenated([block(1, seed: 21), block(1, seed: 22)], axis: 0)
        let (keys, values) = batch.update(keys: stepKeys, values: stepValues)
        XCTAssertEqual(keys.shape, [2, heads, 5, dim])
        XCTAssertEqual(values.shape, [2, heads, 5, dim])
        _ = a.update(keys: block(1, seed: 11), values: block(1, seed: 21))
        _ = b.update(keys: block(1, seed: 12), values: block(1, seed: 22))
        XCTAssertTrue(equal(batch.extract(row: 0).state[0], a.state[0]))
        XCTAssertTrue(equal(batch.extract(row: 0).state[1], a.state[1]))
        XCTAssertTrue(equal(batch.extract(row: 1).state[0], b.state[0]))
        XCTAssertTrue(equal(batch.extract(row: 1).state[1], b.state[1]))
    }

    func testTheMaskHidesEachRowsPaddingAndStaysCausal() {
        let batch = BatchKVCache()
        batch.extend(simple(fed: [2], seed: 1))
        batch.extend(simple(fed: [4], seed: 2))
        guard case .array(let mask) = batch.makeMask(n: 1, windowSize: nil, returnArray: false) else {
            return XCTFail("a padded batch needs an array mask")
        }
        XCTAssertEqual(mask.shape, [2, 1, 1, 5])
        let bits = mask.asType(.int32).asArray(Int32.self)
        XCTAssertEqual(Array(bits[0 ..< 5]), [0, 0, 1, 1, 1], "row 0: two tokens of padding, then its two tokens and the new one")
        XCTAssertEqual(Array(bits[5 ..< 10]), [1, 1, 1, 1, 1], "row 1 has no padding")
        // Several query positions: causal within the new tokens.
        guard case .array(let wide) = batch.makeMask(n: 2, windowSize: nil, returnArray: false) else { return XCTFail() }
        XCTAssertEqual(wide.shape, [2, 1, 2, 6])
        let w = wide.asType(.int32).asArray(Int32.self)
        // Flattened [2, 1, 2, 6]: row 0's two queries at 0..<12, row 1's at 12..<24.
        XCTAssertEqual(Array(w[0 ..< 6]), [0, 0, 1, 1, 1, 0], "row 0, first query does not see the second new token")
        XCTAssertEqual(Array(w[6 ..< 12]), [0, 0, 1, 1, 1, 1], "row 0, second query sees both new tokens, never its padding")
        XCTAssertEqual(Array(w[18 ..< 24]), [1, 1, 1, 1, 1, 1], "row 1, second query sees everything")
    }

    func testAnUnpaddedBatchKeepsTheSymbolicMask() {
        let batch = BatchKVCache()
        batch.extend(simple(fed: [3], seed: 1))
        batch.extend(simple(fed: [3], seed: 2))
        if case .none = batch.makeMask(n: 1, windowSize: nil, returnArray: false) {} else { XCTFail("one token, no padding: no mask") }
        if case .causal = batch.makeMask(n: 3, windowSize: nil, returnArray: false) {} else { XCTFail("several tokens, no padding: causal") }
    }

    func testTheOffsetIsPerRowThroughTheExistential() {
        let batch = BatchKVCache()
        batch.extend(simple(fed: [1], seed: 1))
        batch.extend(simple(fed: [4], seed: 2))
        let any: any KVCache = batch
        guard case .batch(let offsets) = any.ropeOffset else { return XCTFail("a batch answers per-row offsets") }
        XCTAssertEqual(offsets.asArray(Int32.self), [1, 4])
        XCTAssertEqual(any.offset, 4, "the scalar offset is the longest row's")
    }

    func testFilteringDropsRowsAndTheirSharedPadding() {
        let batch = BatchKVCache()
        batch.extend(simple(fed: [2], seed: 1))
        batch.extend(simple(fed: [5], seed: 2))
        batch.extend(simple(fed: [3], seed: 3))
        XCTAssertEqual(batch.leftPadding, [3, 0, 2])
        let before = batch.extract(row: 2).state[0]
        batch.filter(keep: [0, 2])
        XCTAssertEqual(batch.rows, 2)
        XCTAssertEqual(batch.leftPadding, [1, 0], "the padding both rows shared is gone")
        XCTAssertEqual(batch.cursor, 3)
        XCTAssertTrue(equal(batch.extract(row: 1).state[0], before))
        batch.filter(keep: [])
        XCTAssertEqual(batch.rows, 0)
        XCTAssertEqual(batch.cursor, 0)
    }
}
