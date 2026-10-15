//  WHAT: The key/value cache of one attention layer for B sequences decoded as one batch: one
//        padded buffer `[B, kvHeads, capacity, headDim]`, one shared write cursor, and a left
//        padding per row so rows of different lengths line up on their last token.
//  IN:   GenerationBatch — adopts one sequence's `KVCacheSimple` as the first row, extends by a
//        row, extracts a row back out, filters out rows that finished.
//  OUT:  The model reads it as any `KVCache`: `update` appends every row's new tokens at the
//        cursor, `makeMask` hides each row's padding, `ropeOffset` gives each row its own position.
//  PIN:  The mask and the offsets are per row; the cursor is not. A row's position is
//        `cursor − leftPadding`, which the model's RoPE takes as a `[B]` array
//        (`BatchPositionedKVCache`); `offset` reports the cursor, the longest row's length, which is
//        what a scalar reader expects. `ropeOffset` is implemented here, not left to a protocol
//        extension: through `any KVCache` the extension on `KVCache` would answer `.scalar`.

import Foundation
import MLX
import MLXFast
import MLXLMCommon

public final class BatchKVCache: KVCache, BatchPositionedKVCache {
    public private(set) var keys: MLXArray?
    public private(set) var values: MLXArray?
    /// The shared write cursor: how many columns of the buffer hold tokens or padding.
    public private(set) var cursor = 0
    /// Per row: how many leading columns are padding, not tokens.
    public private(set) var leftPadding: [Int] = []
    /// The buffer grows in multiples of this.
    public var step = 256

    public init() {}

    public var rows: Int { leftPadding.count }
    public var offset: Int { cursor }
    public var maxSize: Int? { nil }
    public var isTrimmable: Bool { true }
    public var isPadded: Bool { leftPadding.contains { $0 > 0 } }
    /// Each row's token count.
    public var lengths: [Int] { leftPadding.map { cursor - $0 } }

    public var batchOffset: MLXArray { MLXArray(leftPadding.map { Int32(cursor - $0) }) }
    public var ropeOffset: RoPEOffset { .batch(batchOffset) }

    public func innerState() -> [MLXArray] { [keys, values].compactMap { $0 } }

    // MARK: - KVCache

    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let previous = cursor
        let n = newKeys.dim(2)
        let batch = newKeys.dim(0)
        precondition(batch == rows, "BatchKVCache: \(batch) rows of keys for \(rows) rows of cache")
        if keys == nil || previous + n > keys!.dim(2) {
            let kvHeads = newKeys.dim(1)
            let nSteps = (step + n - 1) / step
            let newK = MLXArray.zeros([batch, kvHeads, nSteps * step, newKeys.dim(3)], dtype: newKeys.dtype)
            let newV = MLXArray.zeros([batch, kvHeads, nSteps * step, newValues.dim(3)], dtype: newValues.dtype)
            if var k = keys, var v = values {
                if previous % step != 0 {
                    k = k[.ellipsis, ..<previous, 0...]
                    v = v[.ellipsis, ..<previous, 0...]
                }
                keys = concatenated([k, newK], axis: 2)
                values = concatenated([v, newV], axis: 2)
            } else {
                keys = newK
                values = newV
            }
        }
        cursor += n
        keys![.ellipsis, previous ..< cursor, 0...] = newKeys
        values![.ellipsis, previous ..< cursor, 0...] = newValues
        return (keys![.ellipsis, ..<cursor, 0...], values![.ellipsis, ..<cursor, 0...])
    }

    /// Causal over the columns written so far, and never a row's padding: `[B, 1, n, cursor + n]`
    /// when any row is padded, the symbolic answer otherwise.
    public func makeMask(n: Int, windowSize: Int?, returnArray: Bool) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard isPadded else {
            if n == 1 { return .none }
            if returnArray || (windowSize != nil && n > windowSize!) {
                return .array(createCausalMask(n: n, offset: cursor, windowSize: windowSize))
            }
            return .causal
        }
        let total = cursor + n
        let columns = MLXArray(Int32(0) ..< Int32(total))[.newAxis, 0...]            // [1, L]
        let queries = MLXArray(Int32(cursor) ..< Int32(total))[0..., .newAxis]       // [n, 1]
        var causal = columns .<= queries                                             // [n, L]
        if let windowSize {
            causal = causal & (columns .> (queries - MLXArray(Int32(windowSize))))
        }
        let padding = MLXArray(leftPadding.map(Int32.init))[0..., .newAxis]         // [B, 1]
        let valid = columns .>= padding                                              // [B, L]
        let mask = causal[.newAxis, .newAxis, 0..., 0...] & valid[0..., .newAxis, .newAxis, 0...]   // [B, 1, n, L]
        return .array(mask)
    }

    public var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            return [keys[.ellipsis, ..<cursor, 0...], values[.ellipsis, ..<cursor, 0...]]
        }
        set {
            precondition(newValue.count == 2, "BatchKVCache state is keys and values")
            keys = newValue[0]
            values = newValue[1]
            cursor = newValue[0].dim(2)
            if leftPadding.count != newValue[0].dim(0) {
                leftPadding = Array(repeating: 0, count: newValue[0].dim(0))
            }
        }
    }

    /// `[leftPadding as "a,b,c", cursor]`.
    public var metaState: [String] {
        get { [leftPadding.map(String.init).joined(separator: ","), "\(cursor)"] }
        set {
            if let first = newValue.first {
                leftPadding = first.isEmpty ? [] : first.split(separator: ",").compactMap { Int($0) }
            }
            if newValue.count > 1, let c = Int(newValue[1]) { cursor = c }
        }
    }

    /// Trims every row by the same count.
    @discardableResult
    public func trim(_ n: Int) -> Int {
        let trimmed = min(cursor, n)
        cursor -= trimmed
        return trimmed
    }

    public func copy() -> any KVCache {
        let new = BatchKVCache()
        new.step = step
        new.cursor = cursor
        new.leftPadding = leftPadding
        new.keys = keys?[.ellipsis]
        new.values = values?[.ellipsis]
        return new
    }

    public func prepare(lengths: [Int]?) {}
    public func prepare(lengths: MLXArray?) {}
    public func finalize() {}

    // MARK: - Rows

    /// Appends one sequence as a row. The first row's buffer becomes the batch's own, uncopied;
    /// a later row is placed with its last token on the cursor, padded on the left — and when it is
    /// longer than every row so far, the rows already here gain that much padding instead.
    public func extend(_ row: KVCacheSimple) {
        let rowState = row.state
        guard rowState.count == 2 else {
            leftPadding.append(cursor)
            return
        }
        let (rowKeys, rowValues) = (rowState[0], rowState[1])
        let length = rowKeys.dim(2)
        guard let batchKeys = keys, let batchValues = values, rows > 0 else {
            let capacity = Self.roundUp(max(length, 1), to: step)
            keys = Self.placed(rowKeys, length: length, padding: 0, capacity: capacity)
            values = Self.placed(rowValues, length: length, padding: 0, capacity: capacity)
            cursor = length
            leftPadding = [0]
            return
        }
        var k = batchKeys
        var v = batchValues
        if length > cursor {
            let extra = length - cursor
            let capacity = Self.roundUp(length, to: step)
            k = Self.placed(k[.ellipsis, ..<cursor, 0...], length: cursor, padding: extra, capacity: capacity)
            v = Self.placed(v[.ellipsis, ..<cursor, 0...], length: cursor, padding: extra, capacity: capacity)
            leftPadding = leftPadding.map { $0 + extra }
            cursor = length
        }
        let capacity = k.dim(2)
        let padding = cursor - length
        keys = concatenated([k, Self.placed(rowKeys, length: length, padding: padding, capacity: capacity)], axis: 0)
        values = concatenated([v, Self.placed(rowValues, length: length, padding: padding, capacity: capacity)], axis: 0)
        leftPadding.append(padding)
    }

    /// One row's tokens as a sequence of its own: a contiguous copy, so the batch keeps exclusive
    /// use of its buffer.
    public func extract(row index: Int) -> KVCacheSimple {
        let simple = KVCacheSimple()
        simple.step = step
        guard let keys, let values else { return simple }
        let padding = leftPadding[index]
        guard cursor > padding else { return simple }
        simple.state = [
            contiguous(keys[index ..< (index + 1), 0..., padding ..< cursor, 0...]),
            contiguous(values[index ..< (index + 1), 0..., padding ..< cursor, 0...]),
        ]
        return simple
    }

    /// Keeps the rows at `keep`, in that order, and drops padding every remaining row shares.
    public func filter(keep: [Int]) {
        guard !keep.isEmpty else {
            keys = nil
            values = nil
            leftPadding = []
            cursor = 0
            return
        }
        guard let keys, let values else {
            leftPadding = keep.map { leftPadding[$0] }
            return
        }
        let indices = MLXArray(keep.map(Int32.init))
        var k = keys[indices]
        var v = values[indices]
        var padding = keep.map { leftPadding[$0] }
        let common = padding.min() ?? 0
        if common > 0 {
            k = k[.ellipsis, common..., 0...]
            v = v[.ellipsis, common..., 0...]
            cursor -= common
            padding = padding.map { $0 - common }
        }
        self.keys = contiguous(k)
        self.values = contiguous(v)
        leftPadding = padding
    }

    // MARK: - Helpers

    /// `[1, kvHeads, capacity, D]`: `padding` zero columns, the tokens, zeros to `capacity`.
    static func placed(_ array: MLXArray, length: Int, padding: Int, capacity: Int) -> MLXArray {
        var parts: [MLXArray] = []
        let batch = array.dim(0), heads = array.dim(1), dim = array.dim(3)
        if padding > 0 { parts.append(MLXArray.zeros([batch, heads, padding, dim], dtype: array.dtype)) }
        if length > 0 { parts.append(array[.ellipsis, ..<length, 0...]) }
        let tail = capacity - padding - length
        if tail > 0 { parts.append(MLXArray.zeros([batch, heads, tail, dim], dtype: array.dtype)) }
        return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
    }

    static func roundUp(_ n: Int, to step: Int) -> Int {
        ((n + step - 1) / step) * step
    }
}
