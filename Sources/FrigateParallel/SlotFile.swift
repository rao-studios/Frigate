//  WHAT: A session's cache as a file, `frigate-slot/1`: the slot's layers, the tokens they hold, and the snapshots
//        it can rewind to — the pinned turn boundary and the ring — in one safetensors file, with what the caller
//        wrote beside them. `ParallelEngine.keep(slot:to:metadata:)` writes one; `take(slot:from:expecting:)` reads
//        one back into a slot, which the next request then continues by its common prefix, as if the slot had never
//        left the engine.
//  WHY:  A session kept over a relaunch (Craft, 2026-11-17) re-read its whole conversation on its first ask —
//        30–60k tokens, minutes on a large model — though the cache it had was the same one. Kept beside the session,
//        the cache comes back in the time it takes to read the file.
//  PIN:  A file fits only the engine that could have written it. Its layout (attention or recurrent, layer by layer)
//        must be this engine's, and every key the caller expects (the model's id and weights, say) must read the same:
//        a cache another model wrote would decode garbage, or fail on its first forward pass.
//  PIN:  Read whole when taken. MLX loads a safetensors file lazily; a file cut short would fail at the first
//        evaluation, mid-generation. `take` evaluates every array at once, so a torn file is refused there and the
//        session reads its prompt as it would have without one.
//  PIN:  Written beside and moved over. A write cut short (a quit, a full disk) never replaces a file that reads.

import Foundation
import MLX
import MLXLMCommon

/// What a caller is told of a session's cache kept as a file.
public struct SlotFile: Sendable, Equatable {
    public static let format = "frigate-slot/1"
    /// The tokens the cache holds: what the next request on the slot need not read again.
    public let tokens: Int
    /// The file's size on disk.
    public let bytes: Int
    /// What the caller wrote beside the cache, and what `take` holds a file to.
    public let metadata: [String: String]

    public init(tokens: Int, bytes: Int, metadata: [String: String]) {
        self.tokens = tokens; self.bytes = bytes; self.metadata = metadata
    }
}

/// Writes and reads `frigate-slot/1`. On the engine's thread only: every array is evaluated there.
///
/// Arrays: `tokens` (int32), `layers.<layer>.<n>` (each layer's state), `pinned.<layer>.<n>` and `ring.<k>.<layer>.<n>`
/// (the snapshots' recurrent arrays). Metadata: `format`, `layout`, `tokens`, `pinned` and `ring` (the snapshots'
/// token counts), and the caller's own keys as `meta.<key>`.
enum SlotFileCodec {
    static func write(_ cache: CheckpointedCache, tokens: [Int], layout: CacheLayout, metadata: [String: String],
                      to url: URL) throws -> SlotFile {
        var arrays: [String: MLXArray] = ["tokens": MLXArray(tokens.map(Int32.init))]
        for (index, layer) in cache.layers.enumerated() {
            for (n, array) in layer.state.enumerated() { arrays["layers.\(index).\(n)"] = array }
        }
        func put(_ snapshot: CheckpointedCache.Snapshot, under prefix: String) {
            for (index, state) in snapshot.recurrent {
                for (n, array) in state.enumerated() { arrays["\(prefix).\(index).\(n)"] = array }
            }
        }
        if let pinned = cache.pinned { put(pinned, under: "pinned") }
        for (k, snapshot) in cache.snapshots.enumerated() { put(snapshot, under: "ring.\(k)") }
        var header = ["format": SlotFile.format, "layout": layout.signature, "tokens": String(tokens.count)]
        header["pinned"] = cache.pinned.map { String($0.tokenCount) }
        if !cache.snapshots.isEmpty { header["ring"] = cache.snapshots.map { String($0.tokenCount) }.joined(separator: ",") }
        for (key, value) in metadata { header["meta." + key] = value }

        let fm = FileManager.default
        let folder = url.deletingLastPathComponent()
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        let beside = folder.appendingPathComponent(".\(url.deletingPathExtension().lastPathComponent)-\(UUID().uuidString).safetensors")
        do {
            try save(arrays: arrays, metadata: header, url: beside)
            guard rename(beside.path, url.path) == 0 else { throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: url.path]) }
        } catch {
            try? fm.removeItem(at: beside)
            throw error
        }
        return SlotFile(tokens: tokens.count, bytes: size(of: url), metadata: metadata)
    }

    static func read(from url: URL, layout: CacheLayout, expecting: [String: String],
                     ring: Int) throws -> (cache: CheckpointedCache, tokens: [Int], file: SlotFile) {
        func refuse(_ why: String) -> ParallelError { .slotFileRefused("\(url.lastPathComponent): \(why)") }
        guard FileManager.default.fileExists(atPath: url.path) else { throw refuse("no such file") }
        guard !cutShort(url) else { throw refuse("cut short") }
        let (arrays, header) = try loadArraysAndMetadata(url: url)
        guard header["format"] == SlotFile.format else { throw refuse("not a \(SlotFile.format) file") }
        guard header["layout"] == layout.signature else { throw refuse("written for another cache layout") }
        var metadata: [String: String] = [:]
        for (key, value) in header where key.hasPrefix("meta.") { metadata[String(key.dropFirst("meta.".count))] = value }
        for (key, value) in expecting.sorted(by: { $0.key < $1.key }) where metadata[key] != value {
            throw refuse("its \(key) is \(metadata[key].map { "\"\($0)\"" } ?? "missing"), not \"\(value)\"")
        }
        guard let tokenArray = arrays["tokens"] else { throw refuse("no tokens") }
        eval(Array(arrays.values))
        let tokens = tokenArray.asArray(Int32.self).map(Int.init)
        guard !tokens.isEmpty, header["tokens"] == String(tokens.count) else { throw refuse("its tokens do not match its header") }

        let state = grouped(arrays, under: "layers")
        var layers: [KVCache] = []
        for (index, kind) in layout.kinds.enumerated() {
            let held = state[index] ?? []
            switch kind {
            case .attention:
                guard held.count == 2 else { throw refuse("layer \(index) holds no keys and values") }
                let layer = KVCacheSimple()
                layer.state = held
                layers.append(layer)
            case .recurrent:
                guard !held.isEmpty else { throw refuse("layer \(index) holds no state") }
                // As `GenerationBatch.extract` hands a row's recurrent layer back to its slot.
                let layer = MambaCache()
                layer.state = held
                layers.append(layer)
            }
        }
        // A snapshot that does not cover every recurrent layer would rewind some and not others: it is left out.
        let recurrent = Set(layout.kinds.indices.filter { layout.kinds[$0] == .recurrent })
        func snapshot(_ count: Int?, under prefix: String) -> CheckpointedCache.Snapshot? {
            guard let count else { return nil }
            let held = grouped(arrays, under: prefix)
            guard Set(held.keys) == recurrent, held.values.allSatisfy({ !$0.isEmpty }) else { return nil }
            return CheckpointedCache.Snapshot(tokenCount: count, recurrent: held)
        }
        let counts = (header["ring"] ?? "").split(separator: ",").map { Int($0) }
        let snapshots = counts.enumerated().compactMap { k, count in snapshot(count, under: "ring.\(k)") }
        let cache = CheckpointedCache(layers: layers, ring: ring, snapshots: snapshots,
                                      pinned: snapshot(header["pinned"].flatMap { Int($0) }, under: "pinned"))
        if layout.kinds.contains(.attention), cache.position != tokens.count {
            throw refuse("its layers hold \(cache.position) tokens, its header \(tokens.count)")
        }
        return (cache, tokens, SlotFile(tokens: tokens.count, bytes: size(of: url), metadata: metadata))
    }

    /// `<prefix>.<layer>.<n>` → layer → its arrays in order.
    private static func grouped(_ arrays: [String: MLXArray], under prefix: String) -> [Int: [MLXArray]] {
        var found: [Int: [Int: MLXArray]] = [:]
        for (key, array) in arrays where key.hasPrefix(prefix + ".") {
            let rest = key.dropFirst(prefix.count + 1).split(separator: ".")
            guard rest.count == 2, let layer = Int(rest[0]), let n = Int(rest[1]) else { continue }
            found[layer, default: [:]][n] = array
        }
        return found.mapValues { held in held.keys.sorted().compactMap { held[$0] } }
    }

    /// Whether the file ends before its own header says its data does. MLX reads a safetensors file's arrays only when
    /// they are evaluated, and a read past the end there is no error a caller can catch.
    private static func cutShort(_ url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return true }
        defer { try? handle.close() }
        let size = UInt64(Self.size(of: url))
        guard let head = try? handle.read(upToCount: 8), head.count == 8 else { return true }
        let length = head.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard length <= size - 8, let json = try? handle.read(upToCount: Int(length)), json.count == Int(length),
              let header = try? JSONSerialization.jsonObject(with: json) as? [String: Any] else { return true }
        let end = header.values.compactMap { ($0 as? [String: Any])?["data_offsets"] as? [NSNumber] }
            .compactMap { $0.last?.uint64Value }.max() ?? 0
        return 8 + length + end > size
    }

    private static func size(of url: URL) -> Int {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.intValue ?? 0
    }
}
