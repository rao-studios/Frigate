//  WHAT: Writes evaluated arrays to a safetensors file straight from their memory, on whatever thread calls it: the
//        header, then each array's bytes in name order, beside the file and moved over it.
//  WHY:  A slot file was written by MLX's own `save` on the engine's thread, which holds the evaluation lock for the
//        whole write: nothing decoded, in any session, while 1.3–2.7 GB went to disk (Craft, 2026-11-22). The arrays
//        of a kept cache are already evaluated, so writing them needs neither the engine's thread nor that lock.
//  PIN:  EVALUATED BEFORE IT IS HANDED OVER. `SlotFileCodec.gather` evaluates every array on the engine's thread;
//        here `asData` finds nothing left to compute, takes the evaluation lock for an instant, and reads memory.
//  PIN:  THE ARRAYS ARE HELD, SO THE ENGINE NEVER WRITES UNDER THEM. A slice of a key-value buffer shares the buffer.
//        MLX updates a buffer in place only while nothing else references it, so a session that decodes on while its
//        file is written copies the buffer once and leaves the bytes being written as they were.
//  PIN:  Written beside and moved over. A write cut short (a quit, a full disk) never replaces a file that reads.

import Foundation
import MLX

enum SlotFileWriter {
    /// Hidden files a write left beside its target and never moved: removed from a folder once older than this.
    static let leftoverAge: TimeInterval = 3_600

    /// Writes `arrays` and `metadata` to `url` as safetensors. Every array must be evaluated already.
    static func write(arrays: [String: MLXArray], metadata: [String: String], to url: URL) throws {
        let names = arrays.keys.sorted()
        var header: [String: Any] = ["__metadata__": metadata]
        var offset = 0
        for name in names {
            guard let array = arrays[name] else { continue }
            guard let type = safetensorsName(array.dtype) else {
                throw ParallelError.stretchRefused("\(name) is of a type a slot file does not hold (\(array.dtype))")
            }
            header[name] = ["dtype": type, "shape": array.shape, "data_offsets": [offset, offset + array.nbytes]] as [String: Any]
            offset += array.nbytes
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        // The reference writer pads the header to eight bytes, so the data begins aligned.
        while json.count % 8 != 0 { json.append(0x20) }

        let manager = FileManager.default
        let folder = url.deletingLastPathComponent()
        try manager.createDirectory(at: folder, withIntermediateDirectories: true)
        sweep(folder)
        let beside = folder.appendingPathComponent(".\(url.deletingPathExtension().lastPathComponent)-\(UUID().uuidString).safetensors")
        guard manager.createFile(atPath: beside.path, contents: nil) else {
            throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: beside.path])
        }
        do {
            let handle = try FileHandle(forWritingTo: beside)
            defer { try? handle.close() }
            var length = UInt64(json.count).littleEndian
            try handle.write(contentsOf: Data(bytes: &length, count: 8))
            try handle.write(contentsOf: json)
            for name in names {
                guard let array = arrays[name], array.nbytes > 0 else { continue }
                // A slice of a wider buffer is copied into one piece here, an array at a time; anything else is read
                // where it lies.
                let bytes = array.asData(access: .noCopyIfContiguous).data
                guard bytes.count == array.nbytes else {
                    throw ParallelError.stretchRefused("\(name) reads \(bytes.count) bytes, not its \(array.nbytes)")
                }
                try handle.write(contentsOf: bytes)
            }
            try handle.synchronize()
            guard rename(beside.path, url.path) == 0 else { throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path]) }
        } catch {
            try? manager.removeItem(at: beside)
            throw error
        }
    }

    /// Removes what a write cut short left in `folder`: the hidden files a write makes beside its target, once they
    /// are older than any write still under way could be.
    static func sweep(_ folder: URL, now: Date = Date()) {
        let manager = FileManager.default
        let keys: Set<URLResourceKey> = [.contentModificationDateKey]
        let files = (try? manager.contentsOfDirectory(at: folder, includingPropertiesForKeys: Array(keys), options: [])) ?? []
        for file in files where file.lastPathComponent.hasPrefix(".") && file.pathExtension == "safetensors" {
            let written = (try? file.resourceValues(forKeys: keys).contentModificationDate) ?? .distantPast
            if now.timeIntervalSince(written) > leftoverAge { try? manager.removeItem(at: file) }
        }
    }

    /// A type as a safetensors header names it; nil for one the format has no name for.
    static func safetensorsName(_ type: DType) -> String? {
        switch type {
        case .bool: return "BOOL"
        case .uint8: return "U8"
        case .uint16: return "U16"
        case .uint32: return "U32"
        case .uint64: return "U64"
        case .int8: return "I8"
        case .int16: return "I16"
        case .int32: return "I32"
        case .int64: return "I64"
        case .float16: return "F16"
        case .float32: return "F32"
        case .bfloat16: return "BF16"
        case .float64: return "F64"
        default: return nil
        }
    }
}
