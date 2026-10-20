import Foundation
import Testing
import MLX

// A safetensors file's bytes depend only on its tensors and metadata. Swift seeds its hasher
// per process, so a dictionary's iteration order changes from run to run; the vendored writer
// used to lay the data out in that order (through the C++ map it was inserted into), and the
// same tensors saved twice gave two different files. It now lays them out in key order.

/// MLX graph ops abort on Darwin without a metallib (see VectorOpsTests): these run on Linux,
/// or anywhere with `FRIGATE_MLX_TESTS=1` after scripts/build-metallib.sh.
private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

@Suite("Safetensors layout", .enabled(if: mlxFunctional))
struct SafetensorsLayoutTests {
    @Test("the same tensors in two iteration orders save to the same bytes, laid out in key order")
    func layoutFollowsKeys() throws {
        let names = (0 ..< 40).map { String(format: "layer.%02d.weight", $0) }
        var forward: [String: MLXArray] = [:]
        for (i, name) in names.enumerated() {
            forward[name] = MLXArray((0 ..< (i + 1) * 3).map { Float($0) * 0.5 + Float(i) })
        }
        var backward: [String: MLXArray] = [:]
        backward.reserveCapacity(4096)
        for name in names.reversed() { backward[name] = forward[name]! }
        try #require(Array(forward.keys) != Array(backward.keys), "the two dictionaries must iterate in different orders")

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("frigate-st-\(UUID().uuidString.prefix(8))", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let a = directory.appendingPathComponent("a.safetensors")
        let b = directory.appendingPathComponent("b.safetensors")
        try save(arrays: forward, metadata: ["format": "test", "z": "1", "a": "2"], url: a)
        try save(arrays: backward, metadata: ["a": "2", "z": "1", "format": "test"], url: b)

        let bytes = try Data(contentsOf: a)
        #expect(bytes == (try Data(contentsOf: b)), "the two saves differ")

        // The data section runs in key order.
        let headerLength = Int(bytes.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        let header = try #require(try JSONSerialization.jsonObject(with: bytes.subdata(in: 8 ..< 8 + headerLength)) as? [String: Any])
        func start(_ name: String) -> Int { ((header[name] as? [String: Any])?["data_offsets"] as? [Int])?.first ?? -1 }
        #expect(names.sorted { start($0) < start($1) } == names.sorted(), "the data is not in key order")

        // And it reads back as written.
        let loaded = try loadArrays(url: a)
        for name in names {
            #expect(arrayEqual(try #require(loaded[name]), forward[name]!).item(Bool.self), "\(name) reads back different")
        }
    }
}
