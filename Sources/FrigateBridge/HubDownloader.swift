// Concrete `Downloader` for MLXLMCommon 3.x, backed by the vendored swift-transformers `Hub`.
//
// mlx-swift-lm 3.x deliberately dropped its hard dependency on the HuggingFace Hub and
// expects the host application to supply the concrete implementations. Upstream ships that
// wiring as the `MLXHuggingFace` macro target, which pulls in swift-syntax; this package is
// fully vendored and cross-compiles to Linux, so it wires the protocols by hand instead —
// the "integration package" shape upstream documents in MLXLMCommon's `upgrade.md`.

import Foundation
import Hub
import MLXLMCommon

/// Downloads model snapshots through `HubApi`.
public struct HubDownloader: Downloader {
    let hub: HubApi

    public init(hub: HubApi = HubDownloader.defaultHub) {
        self.hub = hub
    }

    /// `HubApi.shared` puts snapshots in ~/Documents/huggingface. When the
    /// launcher names a home for models (`HF_HOME` — Ambient sets it beside its
    /// data, out of Documents), snapshots go under `$HF_HOME/snapshots`; the
    /// Hub's cache already follows `HF_HOME` on its own.
    ///
    /// New downloads land there. `download` looks wider first: a complete copy
    /// in any of `snapshotRoots()` — `$HF_HOME/snapshots`, `$HF_HOME`,
    /// ~/Documents/huggingface — is used where it sits rather than fetched again.
    public static let defaultHub: HubApi = {
        guard let home = ProcessInfo.processInfo.environment["HF_HOME"], !home.isEmpty else {
            return .shared
        }
        return HubApi(downloadBase: URL(fileURLWithPath: home).appendingPathComponent("snapshots"))
    }()

    public func download(
        id: String,
        revision: String?,
        matching patterns: [String],
        useLatest: Bool,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> URL {
        // `snapshot` revalidates every file against the Hub, so unless the caller asks for the
        // latest, a complete copy already on disk is returned without touching the network.
        // The on-disk layout records no revision, so a pinned `revision` is not checked here.
        if !useLatest, let local = localSnapshot(id: id, matching: patterns) {
            let done = Progress(totalUnitCount: 1)
            done.completedUnitCount = 1
            progressHandler(done)
            return local
        }
        return try await hub.snapshot(
            from: id,
            revision: revision ?? "main",
            matching: patterns,
            progressHandler: progressHandler)
    }

    /// This hub's own snapshot directory — where `snapshot` would write, and for `defaultHub`
    /// the first of `snapshotRoots()` — then each of `snapshotRoots()` in order.
    private func localSnapshot(id: String, matching patterns: [String]) -> URL? {
        let own = hub.localRepoLocation(HubApi.Repo(id: id))
        if Self.isMaterialized(own, matching: patterns) { return own }
        return Self.materializedSnapshot(id: id, matching: patterns)
    }
}

// MARK: - Snapshots already on disk

extension HubDownloader {
    /// Where a complete `models/<org>/<repo>` may already sit, in lookup order:
    /// `$HF_HOME/snapshots` (where `defaultHub` downloads), `$HF_HOME`, then `documents`
    /// (`HubApi.shared`'s layout, ~/Documents/huggingface). Without `HF_HOME`, only
    /// `documents`. A root repeating an earlier one (by standardized path) is dropped.
    public static func snapshotRoots(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        documents: URL = HubDownloader.sharedDownloadBase
    ) -> [URL] {
        var roots: [URL] = []
        if let home = environment["HF_HOME"], !home.isEmpty {
            let base = URL(fileURLWithPath: home)
            roots += [base.appending(path: "snapshots"), base]
        }
        roots.append(documents)
        var seen: Set<String> = []
        return roots.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }

    /// The first `<root>/models/<id>` holding a complete snapshot, or `nil`.
    ///
    /// Complete means `config.json`; the weights — every shard `model.safetensors.index.json`
    /// names, else `model.safetensors`, else any `*.safetensors`, each non-empty; and, when
    /// `patterns` would fetch `tokenizer.json`, `tokenizer.json` or `tokenizer_config.json`.
    /// It is not checked pattern by pattern (older repos carry no `*.jinja`), and it is not
    /// HubApi's offline mode, which SHA-256s every LFS file — 13 GB for a 24B model — on
    /// each load.
    public static func materializedSnapshot(
        id: String,
        matching patterns: [String],
        roots: [URL] = HubDownloader.snapshotRoots()
    ) -> URL? {
        for root in roots {
            let candidate = root.appending(path: "models").appending(path: id)
            if isMaterialized(candidate, matching: patterns) { return candidate }
        }
        return nil
    }

    /// `HubApi.shared`'s download base, ~/Documents/huggingface. `HubApi` keeps
    /// `downloadBase` internal, so this repeats how it picks the default.
    @usableFromInline
    static var sharedDownloadBase: URL {
        let documents =
            FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appending(path: "Documents")
        return documents.appending(component: "huggingface")
    }

    private static func isMaterialized(_ directory: URL, matching patterns: [String]) -> Bool {
        guard fileSize(directory.appending(path: "config.json")) != nil, hasWeights(directory) else {
            return false
        }
        guard wantsTokenizer(patterns) else { return true }
        return fileSize(directory.appending(path: "tokenizer.json")) != nil
            || fileSize(directory.appending(path: "tokenizer_config.json")) != nil
    }

    /// Every shard the safetensors index names, else `model.safetensors`, else any
    /// `*.safetensors` — non-empty. An index that does not parse counts as no weights.
    private static func hasWeights(_ directory: URL) -> Bool {
        let index = directory.appending(path: "model.safetensors.index.json")
        if fileSize(index) != nil {
            guard let data = try? Data(contentsOf: index),
                let weightMap = try? JSONDecoder().decode(SafetensorsIndex.self, from: data).weightMap,
                !weightMap.isEmpty
            else { return false }
            return Set(weightMap.values).allSatisfy { isNonEmptyFile(directory.appending(path: $0)) }
        }
        if isNonEmptyFile(directory.appending(path: "model.safetensors")) { return true }
        let entries =
            (try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []
        return entries.contains { $0.pathExtension == "safetensors" && isNonEmptyFile($0) }
    }

    private struct SafetensorsIndex: Decodable {
        let weightMap: [String: String]

        enum CodingKeys: String, CodingKey {
            case weightMap = "weight_map"
        }
    }

    /// Whether `patterns` would fetch the tokenizer. No patterns fetches everything, as in HubApi.
    private static func wantsTokenizer(_ patterns: [String]) -> Bool {
        patterns.isEmpty
            || patterns.contains { $0.lowercased().contains("tokenizer") || glob($0, matches: "tokenizer.json") }
    }

    /// `fnmatch`-style `*` and `?` — the wildcards Hub patterns use — kept in Swift so this
    /// target needs nothing past Foundation on Linux.
    private static func glob(_ pattern: String, matches name: String) -> Bool {
        let pattern = Array(pattern)
        let name = Array(name)
        var p = 0
        var n = 0
        var star: Int?
        var retry = 0
        while n < name.count {
            if p < pattern.count, pattern[p] == "*" {
                star = p
                retry = n
                p += 1
            } else if p < pattern.count, pattern[p] == "?" || pattern[p] == name[n] {
                p += 1
                n += 1
            } else if let star {
                p = star + 1
                retry += 1
                n = retry
            } else {
                return false
            }
        }
        return pattern[p...].allSatisfy { $0 == "*" }
    }

    /// The size of the regular file at `url`, following symlinks, or `nil` when there is none.
    private static func fileSize(_ url: URL) -> Int? {
        guard
            let values = try? url.resolvingSymlinksInPath()
                .resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
            values.isRegularFile == true
        else { return nil }
        return values.fileSize ?? 0
    }

    private static func isNonEmptyFile(_ url: URL) -> Bool {
        (fileSize(url) ?? 0) > 0
    }
}
