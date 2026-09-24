//
//  HubDownloaderSnapshotTests.swift
//  FrigateTests
//
//  WHAT: A model already on disk loads without the network.
//  WHY:  The Rao launcher sets HF_HOME=~/Documents/huggingface, so `defaultHub` snapshots into
//        ~/Documents/huggingface/snapshots/models/<org>/<repo> — while a model fetched before
//        sits at ~/Documents/huggingface/models/<org>/<repo>, `HubApi.shared`'s layout.
//        Snapshotting again meant a 13 GB download. `HubDownloader` now takes a complete copy
//        from `$HF_HOME/snapshots`, `$HF_HOME` or ~/Documents/huggingface first.
//  HOW:  Temporary directory trees only — no MLX, no network. The `download` tests hand the
//        downloader an offline hub, so a lookup that misses throws instead of reaching the Hub.
//

import Foundation
import FrigateBridge
import Hub
import Testing

@Suite("HubDownloader snapshot lookup")
struct HubDownloaderSnapshotTests {

    static let repoID = "org/repo"
    /// MLXLMCommon's `modelDownloadPatterns`. No fixture carries a `*.jinja`.
    static let modelPatterns = ["*.safetensors", "*.json", "*.jinja"]
    static let shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]

    // MARK: - Fixtures

    /// A fresh directory under the system temporary directory; the caller removes it.
    static func scratch() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "HubDownloaderSnapshotTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    static func remove(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    enum Weights {
        /// Two shards, named by `model.safetensors.index.json`.
        case sharded
        /// A lone `model.safetensors`.
        case single
        /// A `*.safetensors` by another name, and no index.
        case other
        case none
    }

    /// Writes `<root>/models/<id>`: config.json, both tokenizer files and non-empty weights.
    @discardableResult
    static func makeRepo(
        under root: URL, id: String = HubDownloaderSnapshotTests.repoID, weights: Weights = .sharded
    ) throws -> URL {
        let repo = root.appending(path: "models").appending(path: id)
        try FileManager.default.createDirectory(at: repo, withIntermediateDirectories: true)
        try write("{}", to: repo.appending(path: "config.json"))
        try write("{}", to: repo.appending(path: "tokenizer.json"))
        try write("{}", to: repo.appending(path: "tokenizer_config.json"))
        switch weights {
        case .sharded:
            for shard in shards { try write("weights", to: repo.appending(path: shard)) }
            // Two tensors share the first shard: a shard is one file however many tensors name it.
            try write(
                """
                {"metadata": {"total_size": 14},
                 "weight_map": {"embed.weight": "\(shards[0])", "layers.0.weight": "\(shards[0])",
                                "lm_head.weight": "\(shards[1])"}}
                """,
                to: repo.appending(path: "model.safetensors.index.json"))
        case .single:
            try write("weights", to: repo.appending(path: "model.safetensors"))
        case .other:
            try write("weights", to: repo.appending(path: "weights.safetensors"))
        case .none:
            break
        }
        return repo
    }

    static func write(_ text: String, to url: URL) throws {
        try Data(text.utf8).write(to: url)
    }

    /// The lookup over `root` alone.
    static func found(
        in root: URL, matching patterns: [String] = HubDownloaderSnapshotTests.modelPatterns
    ) -> String? {
        HubDownloader.materializedSnapshot(id: repoID, matching: patterns, roots: [root])?.path
    }

    // MARK: - snapshotRoots

    @Test func rootsWithHFHome() {
        let roots = HubDownloader.snapshotRoots(
            environment: ["HF_HOME": "/hf"], documents: URL(fileURLWithPath: "/docs/huggingface"))
        #expect(roots.map(\.path) == ["/hf/snapshots", "/hf", "/docs/huggingface"])
    }

    @Test func rootsWithoutHFHome() {
        let documents = URL(fileURLWithPath: "/docs/huggingface")
        #expect(HubDownloader.snapshotRoots(environment: [:], documents: documents).map(\.path) == [documents.path])
        #expect(
            HubDownloader.snapshotRoots(environment: ["HF_HOME": ""], documents: documents).map(\.path)
                == [documents.path])
    }

    /// The Rao launcher's case: `HF_HOME` is ~/Documents/huggingface itself.
    @Test func rootsDropARepeatedRoot() {
        let roots = HubDownloader.snapshotRoots(
            environment: ["HF_HOME": "/docs/huggingface/"], documents: URL(fileURLWithPath: "/docs/huggingface"))
        #expect(roots.map(\.path) == ["/docs/huggingface/snapshots", "/docs/huggingface"])
    }

    /// Without `HF_HOME` the lookup is exactly `HubApi.shared`'s layout.
    @Test func documentsDefaultsToTheSharedHubBase() {
        let roots = HubDownloader.snapshotRoots(environment: [:])
        let shared = HubApi.shared.localRepoLocation(HubApi.Repo(id: Self.repoID))
        #expect(roots.map { $0.appending(path: "models").appending(path: Self.repoID).path } == [shared.path])
    }

    /// New downloads still go where `defaultHub` puts them, and that is looked in first.
    @Test func theFirstRootIsWhereTheDefaultHubDownloads() {
        let first = HubDownloader.snapshotRoots().first?.appending(path: "models").appending(path: Self.repoID)
        let destination = HubDownloader.defaultHub.localRepoLocation(HubApi.Repo(id: Self.repoID))
        #expect(first?.standardizedFileURL.path == destination.standardizedFileURL.path)
    }

    // MARK: - Precedence

    @Test func theSnapshotsRootWinsWhenBothAreComplete() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let home = scratch.appending(path: "hf")
        let documents = scratch.appending(path: "docs")
        let snapshots = try Self.makeRepo(under: home.appending(path: "snapshots"))
        try Self.makeRepo(under: home)
        try Self.makeRepo(under: documents)

        let roots = HubDownloader.snapshotRoots(environment: ["HF_HOME": home.path], documents: documents)
        let found = HubDownloader.materializedSnapshot(id: Self.repoID, matching: Self.modelPatterns, roots: roots)
        #expect(found?.path == snapshots.path)
    }

    @Test func hfHomeServesWhenOnlyItIsComplete() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let home = scratch.appending(path: "hf")
        let documents = scratch.appending(path: "docs")
        // A download under snapshots that never finished does not shadow the complete copy.
        let partial = try Self.makeRepo(under: home.appending(path: "snapshots"))
        try FileManager.default.removeItem(at: partial.appending(path: Self.shards[1]))
        let model = try Self.makeRepo(under: home)
        try Self.makeRepo(under: documents)

        let roots = HubDownloader.snapshotRoots(environment: ["HF_HOME": home.path], documents: documents)
        let found = HubDownloader.materializedSnapshot(id: Self.repoID, matching: Self.modelPatterns, roots: roots)
        #expect(found?.path == model.path)
    }

    @Test func documentsServesWhenOnlyItHasTheModel() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let home = scratch.appending(path: "hf")
        let documents = scratch.appending(path: "docs")
        let model = try Self.makeRepo(under: documents)

        let roots = HubDownloader.snapshotRoots(environment: ["HF_HOME": home.path], documents: documents)
        let found = HubDownloader.materializedSnapshot(id: Self.repoID, matching: Self.modelPatterns, roots: roots)
        #expect(found?.path == model.path)
    }

    @Test func nothingOnDiskFindsNothing() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let roots = HubDownloader.snapshotRoots(
            environment: ["HF_HOME": scratch.appending(path: "hf").path], documents: scratch.appending(path: "docs"))
        #expect(HubDownloader.materializedSnapshot(id: Self.repoID, matching: Self.modelPatterns, roots: roots) == nil)
    }

    // MARK: - Complete and incomplete copies

    @Test func aShardedCopyWithoutAChatTemplateIsComplete() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch)
        #expect(Self.found(in: scratch) == repo.path)
    }

    @Test func aSingleModelSafetensorsIsComplete() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch, weights: .single)
        #expect(Self.found(in: scratch) == repo.path)
    }

    @Test func anyNonEmptySafetensorsIsComplete() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch, weights: .other)
        #expect(Self.found(in: scratch) == repo.path)
    }

    @Test func noWeightsIsIncomplete() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        try Self.makeRepo(under: scratch, weights: .none)
        #expect(Self.found(in: scratch) == nil)
    }

    @Test func aZeroByteModelSafetensorsIsIncomplete() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch, weights: .single)
        try Data().write(to: repo.appending(path: "model.safetensors"))
        #expect(Self.found(in: scratch) == nil)
    }

    @Test func missingConfigIsIncomplete() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch)
        try FileManager.default.removeItem(at: repo.appending(path: "config.json"))
        #expect(Self.found(in: scratch) == nil)
    }

    @Test func aShardTheIndexNamesIsMissing() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch)
        try FileManager.default.removeItem(at: repo.appending(path: Self.shards[1]))
        #expect(Self.found(in: scratch) == nil)
    }

    @Test func aZeroByteShardIsIncomplete() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch)
        try Data().write(to: repo.appending(path: Self.shards[1]))
        #expect(Self.found(in: scratch) == nil)
    }

    /// Every shard is on disk; only the index is broken — and it is not bypassed for a scan.
    @Test(arguments: [
        #"{"weight_map": "#,  // truncated
        #"{"metadata": {"total_size": 14}}"#,  // no weight_map
        #"{"weight_map": {}}"#,  // names no shard
        #"{"weight_map": {"embed.weight": 1}}"#,  // not a file name
    ])
    func aMalformedIndexIsIncomplete(index: String) throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch)
        try Self.write(index, to: repo.appending(path: "model.safetensors.index.json"))
        #expect(Self.found(in: scratch) == nil)
    }

    /// Patterns that would fetch `tokenizer.json` — or name a tokenizer, or (as in HubApi) fetch
    /// everything — need a tokenizer on disk.
    @Test(arguments: [["*.json"], HubDownloaderSnapshotTests.modelPatterns, ["*.safetensors", "tokenizer.model"], ["*"], []])
    func missingTokenizerIsIncompleteWhenThePatternsFetchIt(patterns: [String]) throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch)
        try FileManager.default.removeItem(at: repo.appending(path: "tokenizer.json"))
        try FileManager.default.removeItem(at: repo.appending(path: "tokenizer_config.json"))
        #expect(Self.found(in: scratch, matching: patterns) == nil)
    }

    @Test func noTokenizerIsNeededWhenThePatternsSkipIt() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch)
        try FileManager.default.removeItem(at: repo.appending(path: "tokenizer.json"))
        try FileManager.default.removeItem(at: repo.appending(path: "tokenizer_config.json"))
        #expect(Self.found(in: scratch, matching: ["*.safetensors", "*.jinja"]) == repo.path)
    }

    @Test func tokenizerConfigAloneIsATokenizer() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch)
        try FileManager.default.removeItem(at: repo.appending(path: "tokenizer.json"))
        #expect(Self.found(in: scratch) == repo.path)
    }

    /// A shard linked in from elsewhere counts by its target — and a dangling link does not.
    @Test func aSymlinkedShardCountsByItsTarget() throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let repo = try Self.makeRepo(under: scratch)
        let shard = repo.appending(path: Self.shards[0])
        let blob = scratch.appending(path: "blob")
        try FileManager.default.moveItem(at: shard, to: blob)
        try FileManager.default.createSymbolicLink(at: shard, withDestinationURL: blob)
        #expect(Self.found(in: scratch) == repo.path)

        try FileManager.default.removeItem(at: blob)
        #expect(Self.found(in: scratch) == nil)
    }

    // MARK: - download

    /// A hub whose snapshots live under `root`, offline so nothing here can reach the Hub:
    /// a lookup that misses falls through to `snapshot`, which throws instead.
    static func offlineDownloader(root: URL) -> HubDownloader {
        HubDownloader(hub: HubApi(downloadBase: root, cache: nil, useOfflineMode: true))
    }

    /// A repo id no other directory on this machine holds, so only the scratch hub can supply it.
    static func uniqueID() -> String {
        "frigate-tests/\(UUID().uuidString)"
    }

    @Test func downloadReturnsTheCopyOnDisk() async throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let id = Self.uniqueID()
        let repo = try Self.makeRepo(under: scratch, id: id)
        let log = ProgressLog()

        let url = try await Self.offlineDownloader(root: scratch).download(
            id: id, revision: nil, matching: Self.modelPatterns, useLatest: false,
            progressHandler: { log.record($0) })
        #expect(url.path == repo.path)
        #expect(log.fractions == [1.0])
    }

    @Test func downloadFallsThroughAnIncompleteCopy() async throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let id = Self.uniqueID()
        let repo = try Self.makeRepo(under: scratch, id: id)
        try FileManager.default.removeItem(at: repo.appending(path: Self.shards[1]))

        await #expect(throws: HubApi.EnvironmentError.self) {
            try await Self.offlineDownloader(root: scratch).download(
                id: id, revision: nil, matching: Self.modelPatterns, useLatest: false,
                progressHandler: { _ in })
        }
    }

    @Test func useLatestSkipsTheLookup() async throws {
        let scratch = try Self.scratch()
        defer { Self.remove(scratch) }
        let id = Self.uniqueID()
        try Self.makeRepo(under: scratch, id: id)

        await #expect(throws: HubApi.EnvironmentError.self) {
            try await Self.offlineDownloader(root: scratch).download(
                id: id, revision: nil, matching: Self.modelPatterns, useLatest: true,
                progressHandler: { _ in })
        }
    }
}

/// Collects what a `@Sendable` progress handler reports.
private final class ProgressLog: @unchecked Sendable {
    private let lock = NSLock()
    private var reported: [Double] = []

    func record(_ progress: Progress) {
        lock.lock()
        reported.append(progress.fractionCompleted)
        lock.unlock()
    }

    var fractions: [Double] {
        lock.lock()
        defer { lock.unlock() }
        return reported
    }
}
