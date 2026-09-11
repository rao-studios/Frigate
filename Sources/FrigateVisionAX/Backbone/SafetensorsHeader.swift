//
//  SafetensorsHeader.swift
//  FrigateVisionAX
//
//  WHAT: A safetensors file's metadata, read without loading a single tensor.
//  IN:   BackboneSelection, deciding whether the MLX weights belong to this model
//  PIN:  NO MLX HERE. The backbone is chosen before any MLX op runs — with no metallib the
//        first GPU op aborts the process — so the check that ties the MLX weights to the ONNX
//        backbone they came from reads the header by hand: eight bytes of little-endian
//        length, then that much JSON.
//

import Foundation

enum SafetensorsHeader {

    enum Failure: Error, CustomStringConvertible {
        case unreadable(String)
        case lfsPointer(String)
        case malformed(String)

        var description: String {
            switch self {
            case .unreadable(let name): return "\(name) could not be opened"
            case .lfsPointer(let name): return "\(name) is a git-lfs pointer, not weights — run `git lfs pull`"
            case .malformed(let reason): return reason
            }
        }
    }

    /// The file's `__metadata__` block. Empty when the file has none.
    static func metadata(at url: URL) throws -> [String: String] {
        let name = url.lastPathComponent
        guard let handle = FileHandle(forReadingAtPath: url.path) else {
            throw Failure.unreadable(name)
        }
        defer { try? handle.close() }

        let prefix = (try? handle.read(upToCount: 8)) ?? Data()
        if prefix.starts(with: Data("version ".utf8)) {
            throw Failure.lfsPointer(name)
        }
        guard prefix.count == 8 else {
            throw Failure.malformed("\(name) is shorter than a safetensors header")
        }
        let length = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard length > 1, length < 64 << 20 else {
            throw Failure.malformed("\(name) declares a \(length)-byte header, which is not safetensors")
        }
        let json = (try? handle.read(upToCount: Int(length))) ?? Data()
        guard json.count == Int(length),
              let header = (try? JSONSerialization.jsonObject(with: json)) as? [String: Any]
        else {
            throw Failure.malformed("\(name) has no readable JSON header")
        }
        return header["__metadata__"] as? [String: String] ?? [:]
    }
}
