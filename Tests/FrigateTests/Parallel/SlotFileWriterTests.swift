@testable import FrigateParallel
import MLX
import XCTest

/// The engine's own safetensors writer against MLX's reader: what it writes from memory, off the engine's thread, is
/// the file MLX's `save` would have written.
final class SlotFileWriterTests: EngineTestCase, @unchecked Sendable {
    private func folder() -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("frigate-writer-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        return folder
    }

    /// Every type a cache holds, a slice that is not one piece of memory, a scalar and an empty array come back as they
    /// were, with the metadata beside them, and the bytes on disk are MLX's own for the same arrays.
    func testWhatItWritesReadsBackThroughMLX() throws {
        let url = folder().appendingPathComponent("arrays.safetensors")
        let wide = MLXArray(Array(0 ..< 48).map(Float.init), [1, 2, 6, 4])
        let arrays: [String: MLXArray] = [
            "floats": wide,
            // Columns 1..<4 of every head: two runs of memory, not one.
            "slice": wide[.ellipsis, 1 ..< 4, 0...],
            "half": MLXArray([1.5, -2.25, 3] as [Float]).asType(.float16),
            "brain": MLXArray([0.5, -8, 1024] as [Float]).asType(.bfloat16),
            "ints": MLXArray([7, -9, 11] as [Int32]),
            "scalar": MLXArray(Float(4.5)),
            "empty": MLXArray.zeros([0, 4]),
        ]
        eval(Array(arrays.values))
        let metadata = ["format": "test", "meta.model": "stub"]
        try SlotFileWriter.write(arrays: arrays, metadata: metadata, to: url)

        let (read, said) = try loadArraysAndMetadata(url: url)
        XCTAssertEqual(said, metadata)
        XCTAssertEqual(Set(read.keys), Set(arrays.keys))
        for (name, array) in arrays {
            let back = try XCTUnwrap(read[name], name)
            XCTAssertEqual(back.shape, array.shape, name)
            XCTAssertEqual(back.dtype, array.dtype, name)
            XCTAssertEqual(back.asType(.float32).asArray(Float.self), array.asType(.float32).asArray(Float.self), name)
        }
        XCTAssertEqual(read["slice"]?.asArray(Float.self).prefix(4).map { $0 }, [4, 5, 6, 7], "the slice's own columns, in order")

        let reference = url.deletingLastPathComponent().appendingPathComponent("reference.safetensors")
        try save(arrays: arrays, metadata: metadata, url: reference)
        let (theirs, _) = try loadArraysAndMetadata(url: reference)
        for (name, array) in theirs {
            XCTAssertEqual(read[name]?.asType(.float32).asArray(Float.self), array.asType(.float32).asArray(Float.self), "\(name) as MLX writes it")
        }
    }

    /// PIN: written beside and moved over, and what a write cut short left behind is swept. A hidden file an hour old
    /// goes at the next write into its folder; one a write may still be making stays.
    func testAWriteSweepsWhatACutShortWriteLeft() throws {
        let folder = folder()
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let stale = folder.appendingPathComponent(".gpu-\(UUID().uuidString).safetensors")
        let fresh = folder.appendingPathComponent(".gpu-\(UUID().uuidString).safetensors")
        let kept = folder.appendingPathComponent("kept.safetensors")
        for file in [stale, fresh, kept] { try Data("x".utf8).write(to: file) }
        let old = Date().addingTimeInterval(-SlotFileWriter.leftoverAge - 60)
        try FileManager.default.setAttributes([.modificationDate: old], ofItemAtPath: stale.path)
        try FileManager.default.setAttributes([.modificationDate: old], ofItemAtPath: kept.path)

        let array = MLXArray([1, 2, 3] as [Int32])
        eval(array)
        try SlotFileWriter.write(arrays: ["tokens": array], metadata: [:], to: folder.appendingPathComponent("new.safetensors"))
        let left = Set(try FileManager.default.contentsOfDirectory(atPath: folder.path))
        XCTAssertEqual(left, [fresh.lastPathComponent, "kept.safetensors", "new.safetensors"],
                       "the stale leftover went; a file in place, however old, and a write under way stay")
    }
}
