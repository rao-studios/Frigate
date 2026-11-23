//  WHAT: A session's cache as files, `frigate-slot/1`. A file holds a stretch of the cache (`SlotStretch`): the layers'
//        keys and values for the tokens from its start to its end, the recurrent state at its end, and the snapshots
//        inside it a session rewinds to — its marks, the pinned turn boundary, the ring's newest — with every token
//        before its end and what the caller wrote beside them. `ParallelEngine.keep(slot:to:stretch:metadata:)` writes
//        one; `take(slot:from:expecting:)` reads a chain of them into a slot, which the next request then continues
//        by its common prefix, as if the slot had never left the engine.
//  WHY:  A session kept over a relaunch (Craft, 2026-11-17) re-read its whole conversation on its first ask —
//        30–60k tokens, minutes on a large model — though the cache it had was the same one. Kept beside the session,
//        the cache comes back in the time it takes to read the file.
//  PIN:  A file fits only the engine that could have written it. Its layout (attention or recurrent, layer by layer)
//        must be this engine's, and every key the caller expects (the model's id and weights, say) must read the same:
//        a cache another model wrote would decode garbage, or fail on its first forward pass.
//  PIN:  A STRETCH FITS ONLY THE TOKENS IT WAS READ AFTER. A file carries every token before its end, not only its own,
//        so a stretch is taken only onto a slot that holds exactly the tokens before its start: token for token, as the
//        engine continues any cache. Thirty thousand tokens are 120 KB; the keys and values they stand for are 1 GB.
//  PIN:  Read whole when taken. MLX loads a safetensors file lazily; a file cut short would fail at the first
//        evaluation, mid-generation. `take` evaluates every array at once, so a torn file is refused there and the
//        session reads its prompt as it would have without one.
//  PIN:  ONLY THE SNAPSHOTS A RESUME NEEDS (2026-11-22). A file carried the whole ring, four snapshots of 164 MiB on
//        Inkling-2. The ring's older entries are a turn's earlier rounds, which nothing rewinds to once the turn has
//        ended; the newest is where the next prompt parts from the reply the template renders again. A snapshot that
//        stands at the file's end is the file's own state and is written once.

import Foundation
import MLX
import MLXLMCommon

/// What a caller is told of a session's cache kept as a file.
public struct SlotFile: Sendable, Equatable {
    public static let format = "frigate-slot/1"
    /// The tokens the cache holds through the file's end: what the next request on the slot need not read again.
    public let tokens: Int
    /// The first token the file's layers hold (`SlotStretch.start`): zero for a file that stands alone.
    public let start: Int
    /// The file's size on disk; a chain's files together when several were taken.
    public let bytes: Int
    /// What the caller wrote beside the cache, and what `take` holds a file to.
    public let metadata: [String: String]

    public init(tokens: Int, start: Int = 0, bytes: Int, metadata: [String: String]) {
        self.tokens = tokens; self.start = start; self.bytes = bytes; self.metadata = metadata
    }

    /// A file's stretch and every token before its end, read from its header and its token array alone: what a caller
    /// needs to find the files a stretch stands on, or to read their tokens again when those files are gone. Nil for a
    /// file that is not a slot file, or is cut short. Called from any thread: an error MLX raises reading the file is
    /// caught here, where outside the engine's thread it would end the process.
    public static func head(of url: URL) -> (file: SlotFile, tokens: [Int])? {
        try? withError { try SlotFileCodec.head(of: url) }
    }
}

/// A stretch gathered for writing: its arrays, every one evaluated, and its header. Made on the engine's thread
/// (`SlotFileCodec.gather`), written on another (`SlotFileCodec.write`).
struct SlotFileImage: @unchecked Sendable {
    let arrays: [String: MLXArray]
    let header: [String: String]
    let tokens: Int
    let start: Int
    let metadata: [String: String]
}

/// Writes and reads `frigate-slot/1`. Gathering and reading run on the engine's thread, where every array is
/// evaluated; writing runs anywhere.
///
/// Arrays: `tokens` (int32, every token before the file's end), `layers.<layer>.<n>` (an attention layer's keys and
/// values for the stretch's own tokens, a recurrent layer's state at its end), `mark.<k>.<layer>.<n>`,
/// `pinned.<layer>.<n>` and `ring.<layer>.<n>` (the recurrent arrays of the snapshots inside the stretch that do not
/// stand at its end). Metadata: `format`, `layout`, `start`, `tokens`, `marks` (token counts, ascending), `pinned`
/// and `ring` (a token count each), and the caller's own keys as `meta.<key>`.
enum SlotFileCodec {
    // MARK: - Writing

    /// A stretch of `cache` as it will be written: the layers cut to it, the state at its end, and the snapshots
    /// inside it. Throws when the stretch cannot be cut (`ParallelError.stretchRefused`).
    static func gather(_ cache: CheckpointedCache, tokens: [Int], layout: CacheLayout, stretch: SlotStretch,
                       metadata: [String: String]) throws -> SlotFileImage {
        let position = layout.kinds.contains(.attention) ? cache.position : tokens.count
        let end = stretch.end ?? position, start = stretch.start
        guard start >= 0, start < end, end <= position, end <= tokens.count else {
            throw ParallelError.stretchRefused("the stretch \(start)..<\(end) is not inside the \(min(position, tokens.count)) tokens it holds")
        }
        guard let state = cache.state(at: end == position ? nil : end) else {
            throw ParallelError.stretchRefused("it holds no state at token \(end)")
        }
        var arrays: [String: MLXArray] = [:]
        for (index, layer) in cache.layers.enumerated() {
            let held: [MLXArray]
            switch layout.kinds[index] {
            case .attention:
                let whole = layer.state
                guard whole.count == 2 else { throw ParallelError.stretchRefused("layer \(index) holds no keys and values") }
                held = start == 0 && end == position ? whole : whole.map { $0[.ellipsis, start ..< end, 0...] }
            case .recurrent:
                guard let at = state[index], !at.isEmpty else { throw ParallelError.stretchRefused("layer \(index) holds no state") }
                held = at
            }
            for (n, array) in held.enumerated() { arrays["layers.\(index).\(n)"] = array }
        }
        // A snapshot at the stretch's end is the layers' own state: only its place is written.
        func put(_ snapshot: CheckpointedCache.Snapshot, under prefix: String) {
            guard snapshot.tokenCount < end else { return }
            for (index, held) in snapshot.recurrent {
                for (n, array) in held.enumerated() { arrays["\(prefix).\(index).\(n)"] = array }
            }
        }
        let inside: (CheckpointedCache.Snapshot) -> Bool = { $0.tokenCount > start && $0.tokenCount <= end }
        var header = ["format": SlotFile.format, "layout": layout.signature, "start": String(start), "tokens": String(end)]
        let marks = cache.marks.filter(inside)
        for (k, mark) in marks.enumerated() { put(mark, under: "mark.\(k)") }
        if !marks.isEmpty { header["marks"] = marks.map { String($0.tokenCount) }.joined(separator: ",") }
        if let pinned = cache.pinned, inside(pinned) {
            put(pinned, under: "pinned")
            header["pinned"] = String(pinned.tokenCount)
        }
        if let newest = cache.snapshots.last(where: inside) {
            put(newest, under: "ring")
            header["ring"] = String(newest.tokenCount)
        }
        arrays["tokens"] = MLXArray(tokens.prefix(end).map(Int32.init))
        for (key, value) in metadata { header["meta." + key] = value }
        eval(Array(arrays.values))
        return SlotFileImage(arrays: arrays, header: header, tokens: end, start: start, metadata: metadata)
    }

    /// Writes a gathered stretch to `url`, beside it and moved over.
    static func write(_ image: SlotFileImage, to url: URL) throws -> SlotFile {
        try SlotFileWriter.write(arrays: image.arrays, metadata: image.header, to: url)
        return SlotFile(tokens: image.tokens, start: image.start, bytes: size(of: url), metadata: image.metadata)
    }

    // MARK: - Reading

    /// A file's stretch and every token before its end: its header and its token array, nothing else evaluated.
    static func head(of url: URL) throws -> (file: SlotFile, tokens: [Int]) {
        let opened = try open(url)
        return (opened.file, opened.tokens)
    }

    /// The files of a chain as one cache: each stretch standing on the tokens before it — the stretch before it in
    /// `urls`, or, for the first, what `base` holds (nothing, when the first begins at zero).
    static func read(from urls: [URL], onto base: (cache: CheckpointedCache, tokens: [Int])?, layout: CacheLayout,
                     expecting: [String: String], ring: Int) throws -> (cache: CheckpointedCache, tokens: [Int], file: SlotFile) {
        guard let first = urls.first else { throw ParallelError.slotFileRefused("no file to take") }
        let recurrent = Set(layout.kinds.indices.filter { layout.kinds[$0] == .recurrent })
        var held: [Int] = []
        var pieces: [Int: [[MLXArray]]] = [:]
        var state: [Int: [MLXArray]] = [:]
        var marks: [CheckpointedCache.Snapshot] = []
        var pinned: CheckpointedCache.Snapshot?
        var newest: CheckpointedCache.Snapshot?
        var bytes = 0
        var metadata: [String: String] = [:]
        var begins = 0

        for url in urls {
            func refuse(_ why: String) -> ParallelError { .slotFileRefused("\(url.lastPathComponent): \(why)") }
            let opened = try open(url)
            guard opened.header["layout"] == layout.signature else { throw refuse("written for another cache layout") }
            for (key, value) in expecting.sorted(by: { $0.key < $1.key }) where opened.file.metadata[key] != value {
                throw refuse("its \(key) is \(opened.file.metadata[key].map { "\"\($0)\"" } ?? "missing"), not \"\(value)\"")
            }
            let start = opened.file.start, end = opened.file.tokens
            if url == first {
                begins = start
                if start > 0 {
                    // It stands on what the slot holds: exactly the tokens before it, with the cache standing there.
                    guard let base, base.tokens.count == start,
                          !layout.kinds.contains(.attention) || base.cache.position == start else {
                        throw refuse("it begins at token \(start), and the slot holds \(base?.tokens.count ?? 0)")
                    }
                    held = base.tokens
                    for (index, kind) in layout.kinds.enumerated() where kind == .attention {
                        let stands = base.cache.layers[index].state
                        guard stands.count == 2 else { throw refuse("the slot's layer \(index) holds no keys and values") }
                        pieces[index] = [stands]
                    }
                    marks = base.cache.marks
                    pinned = base.cache.pinned
                }
            }
            guard start == held.count, opened.tokens.starts(with: held) else {
                throw refuse("it does not begin where the \(held.count) tokens before it end")
            }
            eval(Array(opened.arrays.values))
            let layers = grouped(opened.arrays, under: "layers")
            for (index, kind) in layout.kinds.enumerated() {
                let own = layers[index] ?? []
                switch kind {
                case .attention:
                    guard own.count == 2, own.allSatisfy({ $0.ndim >= 2 && $0.shape[$0.ndim - 2] == end - start }) else {
                        throw refuse("layer \(index) does not hold the keys and values of its \(end - start) tokens")
                    }
                    pieces[index, default: []].append(own)
                case .recurrent:
                    guard !own.isEmpty else { throw refuse("layer \(index) holds no state") }
                    state[index] = own
                }
            }
            // A snapshot that does not cover every recurrent layer would rewind some and not others: it is left out.
            func snapshot(_ count: Int?, under prefix: String) -> CheckpointedCache.Snapshot? {
                guard let count, count > start, count <= end else { return nil }
                let at = count == end ? state : grouped(opened.arrays, under: prefix)
                guard Set(at.keys) == recurrent, at.values.allSatisfy({ !$0.isEmpty }) else { return nil }
                return CheckpointedCache.Snapshot(tokenCount: count, recurrent: at)
            }
            let counts = (opened.header["marks"] ?? "").split(separator: ",").map { Int($0) }
            marks += counts.enumerated().compactMap { k, count in snapshot(count, under: "mark.\(k)") }
            pinned = snapshot(opened.header["pinned"].flatMap { Int($0) }, under: "pinned") ?? pinned
            newest = snapshot(opened.header["ring"].flatMap { Int($0) }, under: "ring")
            held = opened.tokens
            bytes += opened.file.bytes
            metadata = opened.file.metadata
        }

        var layers: [KVCache] = []
        for (index, kind) in layout.kinds.enumerated() {
            switch kind {
            case .attention:
                let parts = pieces[index] ?? []
                let layer = KVCacheSimple()
                layer.state = parts.count == 1 ? parts[0] : (0 ..< 2).map { n in concatenated(parts.map { $0[n] }, axis: -2) }
                layers.append(layer)
            case .recurrent:
                // As `GenerationBatch.extract` hands a row's recurrent layer back to its slot.
                let layer = MambaCache()
                layer.state = state[index] ?? []
                layers.append(layer)
            }
        }
        let cache = CheckpointedCache(layers: layers, ring: ring, snapshots: newest.map { [$0] } ?? [], pinned: pinned, marks: marks)
        if layout.kinds.contains(.attention), cache.position != held.count {
            throw ParallelError.slotFileRefused("\(first.lastPathComponent): its layers hold \(cache.position) tokens, its header \(held.count)")
        }
        return (cache, held, SlotFile(tokens: held.count, start: begins, bytes: bytes, metadata: metadata))
    }

    /// A file opened and held to its own header: a slot file, not cut short, its token array read and matching its
    /// count, its stretch inside it. Its other arrays are not evaluated.
    private static func open(_ url: URL) throws -> (file: SlotFile, tokens: [Int], arrays: [String: MLXArray], header: [String: String]) {
        func refuse(_ why: String) -> ParallelError { .slotFileRefused("\(url.lastPathComponent): \(why)") }
        guard FileManager.default.fileExists(atPath: url.path) else { throw refuse("no such file") }
        guard !cutShort(url) else { throw refuse("cut short") }
        let (arrays, header) = try loadArraysAndMetadata(url: url)
        guard header["format"] == SlotFile.format else { throw refuse("not a \(SlotFile.format) file") }
        guard let start = header["start"].flatMap({ Int($0) }) else { throw refuse("it does not say where its stretch begins") }
        var metadata: [String: String] = [:]
        for (key, value) in header where key.hasPrefix("meta.") { metadata[String(key.dropFirst("meta.".count))] = value }
        guard let tokenArray = arrays["tokens"] else { throw refuse("no tokens") }
        let tokens = tokenArray.asArray(Int32.self).map(Int.init)
        guard !tokens.isEmpty, header["tokens"] == String(tokens.count) else { throw refuse("its tokens do not match its header") }
        guard start >= 0, start < tokens.count else { throw refuse("its stretch begins at \(start), past its \(tokens.count) tokens") }
        return (SlotFile(tokens: tokens.count, start: start, bytes: size(of: url), metadata: metadata), tokens, arrays, header)
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
        guard size >= 8, let head = try? handle.read(upToCount: 8), head.count == 8 else { return true }
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
