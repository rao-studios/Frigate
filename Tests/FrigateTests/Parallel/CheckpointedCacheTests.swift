import MLX
@testable import FrigateParallel
import MLXLMCommon
import XCTest

/// The cache a hybrid model keeps between rounds: KV layers trimmed by offset, recurrent layers
/// restored from the snapshot taken when the previous prompt had just been prefilled.
/// No array is ever evaluated here, so these tests touch no GPU.
final class CheckpointedCacheTests: XCTestCase {

    /// Feeds `n` tokens into a KV layer (lazily: nothing is evaluated).
    private func feed(_ kv: KVCacheSimple, _ n: Int) {
        _ = kv.update(keys: MLXArray.zeros([1, 1, n, 4]), values: MLXArray.zeros([1, 1, n, 4]))
    }

    private func set(_ layer: MambaCache, _ tag: Float) -> (MLXArray, MLXArray) {
        let conv = MLXArray([tag, tag, tag])
        let state = MLXArray([tag * 10])
        layer[0] = conv
        layer[1] = state
        return (conv, state)
    }

    func testPositionReadsAKVLayerNotARecurrentOne() {
        let kv = KVCacheSimple()
        let recurrent = MambaCache()
        let cache = CheckpointedCache(layers: [recurrent, kv])
        feed(kv, 12)
        XCTAssertEqual(cache.position, 12, "a recurrent layer's offset never moves; the KV layer's is the position")
        XCTAssertFalse(cache.isFullyTrimmable)
        XCTAssertTrue(cache.canRewind)
    }

    func testRewindRestoresTheSnapshotsRecurrentStateAndTrimsTheKVLayer() {
        let kv = KVCacheSimple()
        let recurrent = MambaCache()
        let cache = CheckpointedCache(layers: [recurrent, kv])
        feed(kv, 10)
        let (conv, state) = set(recurrent, 1)
        cache.snapshot(at: 10)
        feed(kv, 4)
        _ = set(recurrent, 2)
        XCTAssertEqual(cache.position, 14)

        XCTAssertEqual(cache.rewind(toAtMost: 12), 10, "the newest snapshot at or before the target")
        XCTAssertEqual(kv.offset, 10)
        XCTAssertTrue(recurrent[0] === conv, "the conv state is the held reference, not a copy")
        XCTAssertTrue(recurrent[1] === state)
    }

    func testRewindPicksTheNewestSnapshotThatFits() {
        let kv = KVCacheSimple()
        let recurrent = MambaCache()
        let cache = CheckpointedCache(layers: [recurrent, kv])
        feed(kv, 10); _ = set(recurrent, 1); cache.snapshot(at: 10)
        feed(kv, 10); let (conv20, _) = set(recurrent, 2); cache.snapshot(at: 20)
        feed(kv, 10); _ = set(recurrent, 3)
        XCTAssertEqual(cache.rewind(toAtMost: 25), 20)
        XCTAssertTrue(recurrent[0] === conv20)
        XCTAssertEqual(cache.snapshots.map(\.tokenCount), [10, 20], "snapshots past the rewind point are gone")
        feed(kv, 3)
        XCTAssertEqual(cache.rewind(toAtMost: 15), 10)
        XCTAssertEqual(cache.snapshots.map(\.tokenCount), [10])
    }

    func testNothingFitsLeavesTheCacheAloneAndSaysSo() {
        let kv = KVCacheSimple()
        let recurrent = MambaCache()
        let cache = CheckpointedCache(layers: [recurrent, kv])
        feed(kv, 10); _ = set(recurrent, 1); cache.snapshot(at: 10)
        feed(kv, 5)
        XCTAssertNil(cache.rewind(toAtMost: 5))
        XCTAssertEqual(kv.offset, 15)
        XCTAssertEqual(cache.snapshots.count, 1)
    }

    func testAFullyTrimmableCacheRewindsExactlyWithoutSnapshots() {
        let kv = KVCacheSimple()
        let cache = CheckpointedCache(layers: [kv])
        feed(kv, 30)
        cache.snapshot(at: 30)
        XCTAssertTrue(cache.snapshots.isEmpty, "trimmable layers need no snapshot")
        XCTAssertTrue(cache.isFullyTrimmable)
        XCTAssertEqual(cache.rewind(toAtMost: 7), 7)
        XCTAssertEqual(kv.offset, 7)
    }

    func testTheRingKeepsTheNewestSnapshots() {
        let kv = KVCacheSimple()
        let recurrent = MambaCache()
        let cache = CheckpointedCache(layers: [recurrent, kv], ring: 2)
        for count in [10, 20, 30] {
            feed(kv, 10); _ = set(recurrent, Float(count)); cache.snapshot(at: count)
        }
        XCTAssertEqual(cache.snapshots.map(\.tokenCount), [20, 30])
    }

    func testASnapshotAtAnEarlierCountReplacesLaterOnes() {
        let kv = KVCacheSimple()
        let recurrent = MambaCache()
        let cache = CheckpointedCache(layers: [recurrent, kv])
        feed(kv, 10); _ = set(recurrent, 1); cache.snapshot(at: 10)
        feed(kv, 10); _ = set(recurrent, 2); cache.snapshot(at: 20)
        cache.snapshot(at: 15)
        XCTAssertEqual(cache.snapshots.map(\.tokenCount), [10, 15])
    }

    /// A rewind says where it went: the ring, the pin or a mark. A cache that trims by position keeps a mark of its
    /// count alone, gone once trimmed past.
    func testARewindSaysWhereItWent() {
        let (cache, _) = filled()
        let rewinds: [(target: Int, position: Int, point: RewindPoint)] = [(45, 40, .ring), (35, 30, .pin), (25, 20, .mark), (15, 10, .mark)]
        for (target, position, point) in rewinds {
            let went = cache.rewindPoint(toAtMost: target)
            XCTAssertEqual(went?.position, position, "to at most \(target)")
            XCTAssertEqual(went?.point, point, "to at most \(target)")
        }
        XCTAssertEqual(cache.marks.map(\.tokenCount), [10], "taken back before a mark, the mark goes")
        XCTAssertNil(cache.rewindPoint(toAtMost: 5), "nothing before the first mark")

        let kv = KVCacheSimple()
        let trimmable = CheckpointedCache(layers: [kv])
        feed(kv, 30)
        trimmable.snapshot(at: 12, mark: true)
        XCTAssertEqual(trimmable.marks.map(\.tokenCount), [12])
        XCTAssertEqual(trimmable.rewindPoint(toAtMost: 20)?.point, .trim)
        XCTAssertEqual(trimmable.marks.map(\.tokenCount), [12], "trimmed to after it, the mark stays")
        _ = trimmable.rewindPoint(toAtMost: 8)
        XCTAssertTrue(trimmable.marks.isEmpty, "trimmed past it, it goes")
    }

    /// Two marks at 10 and 20, the pin at 30, the ring at 40 and 50, and five tokens past it.
    private func filled() -> (CheckpointedCache, KVCacheSimple) {
        let kv = KVCacheSimple()
        let recurrent = MambaCache()
        let cache = CheckpointedCache(layers: [recurrent, kv])
        for count in [10, 20] { feed(kv, 10); _ = set(recurrent, Float(count)); cache.snapshot(at: count, mark: true) }
        feed(kv, 10); _ = set(recurrent, 30); cache.snapshot(at: 30, pinned: true)
        for count in [40, 50] { feed(kv, 10); _ = set(recurrent, Float(count)); cache.snapshot(at: count) }
        feed(kv, 5)
        return (cache, kv)
    }

    /// PIN: what the sessions share goes, and a session's own stays. Memory short drops the ring's older snapshots, then
    /// the earlier marks, then the ring's last; the pin (where the next turn parts) and the last mark (where a
    /// rewritten history does) are never dropped while the cache stays.
    func testOnlyWhatSessionsShareIsDropped() {
        let (cache, _) = filled()
        XCTAssertEqual(cache.snapshotCount, 5)
        XCTAssertEqual(cache.sharedSnapshotCount, 3, "the ring's two and the first mark")
        var left: [[Int]] = []
        while cache.dropSharedSnapshot() {
            left.append(cache.snapshots.map(\.tokenCount) + [-1] + cache.marks.map(\.tokenCount))
        }
        XCTAssertEqual(left, [[50, -1, 10, 20], [50, -1, 20], [-1, 20]])
        XCTAssertEqual(cache.pinned?.tokenCount, 30, "the pin is the session's own")
        XCTAssertEqual(cache.marks.map(\.tokenCount), [20], "and so is its last mark")
        XCTAssertEqual(cache.sharedSnapshotCount, 0)
        XCTAssertEqual(cache.sharedSnapshotBytes, 0)
    }

    /// The state a stretch is kept with: the live state where the cache stands, a snapshot's where one was taken, and
    /// none anywhere else. A cache that trims by position needs none, wherever the stretch ends.
    func testAStretchEndsWhereTheCacheHoldsState() {
        let (cache, _) = filled()
        XCTAssertNotNil(cache.state(at: nil), "where the cache stands")
        for count in [10, 20, 30, 50] { XCTAssertNotNil(cache.state(at: count), "a snapshot at \(count)") }
        XCTAssertNil(cache.state(at: 25), "no snapshot was taken there")
        let kv = KVCacheSimple()
        let trimmable = CheckpointedCache(layers: [kv])
        feed(kv, 30)
        XCTAssertEqual(trimmable.state(at: 17)?.isEmpty, true, "nothing to keep but the position")
    }

    /// PIN: a cache read back keeps its marks, recurrent layers or not. One with no recurrent layer lost them, and a
    /// keep up to its mark then threw.
    func testACacheReadBackKeepsItsMarks() {
        let kv = KVCacheSimple()
        feed(kv, 30)
        let mark = CheckpointedCache.Snapshot(tokenCount: 12, recurrent: [:])
        let past = CheckpointedCache.Snapshot(tokenCount: 40, recurrent: [:])
        let read = CheckpointedCache(layers: [kv], ring: 4, snapshots: [], pinned: nil, marks: [past, mark])
        XCTAssertEqual(read.marks.map(\.tokenCount), [12], "the mark it covers; one past its end is left out")
    }

    func testARecurrentLayerWithoutStateIsNotSnapshotted() {
        let kv = KVCacheSimple()
        let recurrent = MambaCache()
        let cache = CheckpointedCache(layers: [recurrent, kv])
        feed(kv, 10)
        cache.snapshot(at: 10)
        XCTAssertTrue(cache.snapshots.isEmpty)
        XCTAssertNil(cache.rewind(toAtMost: 10))
    }

    /// A rotating cache that has wrapped can no longer be trimmed by position, and it is not a
    /// recurrent layer either: nothing can be rewound, so the session starts a fresh cache.
    func testAnOpaqueLayerCannotRewind() {
        let rotating = RotatingKVCache(maxSize: 2)
        _ = rotating.update(keys: MLXArray.zeros([1, 1, 3, 4]), values: MLXArray.zeros([1, 1, 3, 4]))
        XCTAssertFalse(rotating.isTrimmable)
        let cache = CheckpointedCache(layers: [rotating, KVCacheSimple()])
        XCTAssertFalse(cache.canRewind)
        XCTAssertNil(cache.rewind(toAtMost: 0))
    }
}
