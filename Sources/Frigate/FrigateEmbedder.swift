import Foundation
import FrigateBridge
import Hub
import MLX
import Tokenizers
import mlx_embeddings

/// On-device text embedding via an MLX model downloaded from HuggingFace Hub.
///
/// GPU safety rule: never call MLX.Memory.*, Stream.*, or any CommandEncoder API
/// from inside `container.perform`. The CUDA allocator is active during that closure
/// and re-entry causes SIGSEGV. All memory management happens after `perform` returns.
public actor FrigateEmbedder {

    /// The model, its prompts, width and vector space. See `Profile`.
    public nonisolated let profile: Profile
    /// The caller's own models folder, or nil for Frigate's default home.
    private let modelsHome: URL?
    private let statusBox: StatusBox
    private var loadedContainer: mlx_embeddings.ModelContainer?
    private var loadingTask: Task<(mlx_embeddings.ModelContainer, URL), Error>?
    /// `Tokenizer` is `Sendable`; cached here so tokenization runs on this actor
    /// *before* `container.perform` — CPU tokenization no longer serializes under
    /// the model lock, overlapping with another request's GPU evaluation.
    private var cachedTokenizer: Tokenizers.Tokenizer?
    /// The special tokens this tokenizer wraps every text in, probed once.
    private var tokenFrame: TokenFrame?
    /// Prompts the snapshot's `config_sentence_transformers.json` names, used for a
    /// role the profile leaves unset.
    private var snapshotPrompts: [Role: String] = [:]
    private var requestsSinceCacheClear = 0

    // Batch limits tuned for RTX 3090 / sm_86. Override via environment without
    // changing the defaults: FRIGATE_MAX_BATCH, FRIGATE_CACHE_LIMIT (bytes),
    // FRIGATE_CACHE_CLEAR_INTERVAL (requests; 0 disables periodic clearing).
    static let maxInputsPerBatch: Int =
        ProcessInfo.processInfo.environment["FRIGATE_MAX_BATCH"].flatMap(Int.init) ?? 8
    /// Steady-state GPU allocator pool. Set once at model load — the previous
    /// drop-to-zero + clearCache after *every* request forced full buffer
    /// reallocation per request.
    static let cacheLimitBytes: Int =
        ProcessInfo.processInfo.environment["FRIGATE_CACHE_LIMIT"].flatMap(Int.init) ?? 20 * 1_024 * 1_024
    /// Clear the allocator pool every N requests (conservative VRAM hygiene for
    /// long-running CUDA deployments). 0 = never; use `trimMemory()` on demand.
    static let cacheClearInterval: Int =
        ProcessInfo.processInfo.environment["FRIGATE_CACHE_CLEAR_INTERVAL"].flatMap(Int.init) ?? 32

    /// The files a snapshot needs: weights, config, tokenizer, prompts, pooling config,
    /// and the licence/notice text a redistributed model carries.
    static let snapshotPatterns = ["*.safetensors", "*.json", "*.txt"]

    /// `modelsHome`: a folder of the caller's own (Rao's apps pass ~/.rao/models/huggingface),
    /// where the snapshot is fetched and looked for — `HubDownloader(home:)`. Nil is
    /// Frigate's default home (`HubDownloader.defaultHome()`).
    public init(profile: Profile = .voyage4Nano, modelsHome: URL? = nil) {
        // Raise SDPA LRU cache from 256 → 2048 so varying sequence lengths
        // across sub-batches don't trigger "Cache thrashing" fatal error.
        setenv("MLX_CUDA_SDPA_CACHE_SIZE", "2048", 0)
        self.profile = profile
        self.modelsHome = modelsHome
        self.statusBox = StatusBox(Status(
            model: profile.model, revision: profile.revision,
            vectorSpace: profile.vectorSpace, phase: .idle))
    }

    /// `org/repo`, `org/repo@<revision>` or a directory path — see `Profile.resolve`.
    public init(modelId: String, modelsHome: URL? = nil) {
        self.init(profile: .resolve(modelId), modelsHome: modelsHome)
    }

    /// Where the model is: idle, downloading (with a fraction), loading, ready, failed.
    public nonisolated var status: Status { statusBox.get() }

    // MARK: - Public API

    /// Return L2-normalised embeddings for each input string.
    public func embed(_ texts: [String], role: Role = .document) async throws -> [[Float]] {
        try await embedWithUsage(texts, role: role).embeddings
    }

    /// Return L2-normalised embeddings plus the total prompt-token count
    /// (post-truncation) — the usage datum host servers report per request.
    public func embedWithUsage(_ texts: [String], role: Role = .document) async throws
        -> (embeddings: [[Float]], promptTokens: Int) {
        let container = try await loadedContainer()
        let tokenizer = await self.tokenizer(from: container)
        let prompt = profile.prompt(for: role) ?? snapshotPrompts[role] ?? ""
        let frame = tokenFrame
        let maxTokens = profile.maxTokens

        // Tokenize outside the model lock.
        let tokenized: [[Int]] = texts.map { text in
            let input = prompt + text
            guard let frame else {
                return Array(tokenizer.encode(text: input, addSpecialTokens: true).prefix(maxTokens))
            }
            return frame.fit(tokenizer.encode(text: input, addSpecialTokens: false), maxTokens: maxTokens)
        }
        let promptTokens = tokenized.reduce(0) { $0 + $1.count }
        let padId = tokenizer.eosTokenId ?? 0

        // Similar lengths batch together (less padding); results go back in order.
        let order = tokenized.indices.sorted { tokenized[$0].count < tokenized[$1].count }
        var embeddings = [[Float]](repeating: [], count: tokenized.count)
        let dimension = profile.outputDimension
        for start in stride(from: 0, to: order.count, by: Self.maxInputsPerBatch) {
            let indices = Array(order[start..<min(start + Self.maxInputsPerBatch, order.count)])
            let batch = indices.map { tokenized[$0] }
            // The model lock is taken per sub-batch, so a query arriving mid-way
            // through a long indexing request runs between two of its batches.
            let rows = try await container.perform { model, _ in
                try Self.runBatch(batch, padId: padId, dimension: dimension, model: model)
            }
            for (index, row) in zip(indices, rows) { embeddings[index] = row }
        }
        // CUDA context is idle here — safe to touch the allocator.
        clearCacheIfDue()
        return (embeddings, promptTokens)
    }

    /// Download and load the model, then run one short pass so the first real request
    /// does not pay for kernel compilation.
    public func warmup() async throws {
        _ = try await embedWithUsage(["warm up"], role: .query)
    }

    /// Drop the GPU allocator pool now (e.g. on host memory pressure), then
    /// restore the steady-state cache limit.
    public func trimMemory() {
        MLX.Memory.cacheLimit = 0
        MLX.Memory.clearCache()
        MLX.Memory.cacheLimit = Self.cacheLimitBytes
        requestsSinceCacheClear = 0
    }

    // MARK: - Internal

    private func clearCacheIfDue() {
        guard Self.cacheClearInterval > 0 else { return }
        requestsSinceCacheClear += 1
        if requestsSinceCacheClear >= Self.cacheClearInterval {
            trimMemory()
        }
    }

    private func tokenizer(from container: mlx_embeddings.ModelContainer) async -> Tokenizers.Tokenizer {
        if let t = cachedTokenizer { return t }
        let t = await container.perform { _, tok in tok }
        cachedTokenizer = t
        let probe = "a"
        tokenFrame = TokenFrame.probe(
            with: t.encode(text: probe, addSpecialTokens: true),
            without: t.encode(text: probe, addSpecialTokens: false))
        return t
    }

    private func loadedContainer() async throws -> mlx_embeddings.ModelContainer {
        if let c = loadedContainer { return c }
        if let t = loadingTask { return try await t.value.0 }

        let profile = self.profile
        let statusBox = self.statusBox
        let modelsHome = self.modelsHome
        let task = Task<(mlx_embeddings.ModelContainer, URL), Error> {
            let directory = try await Self.snapshotDirectory(for: profile, modelsHome: modelsHome, status: statusBox)
            statusBox.set { $0.phase = .loading; $0.fraction = nil; $0.error = nil }
            let container = try await mlx_embeddings.loadModelContainer(
                hub: modelsHome.map { HubDownloader.hub(home: $0) } ?? HubDownloader.defaultHub,
                configuration: mlx_embeddings.ModelConfiguration(directory: directory))
            return (container, directory)
        }
        loadingTask = task

        do {
            let (c, directory) = try await task.value
            loadedContainer = c
            snapshotPrompts = Self.snapshotPrompts(in: directory)
            loadingTask = nil
            MLX.Memory.cacheLimit = Self.cacheLimitBytes
            statusBox.set { $0.phase = .ready; $0.fraction = nil; $0.error = nil }
            return c
        } catch {
            loadingTask = nil
            statusBox.set { $0.phase = .failed; $0.fraction = nil; $0.error = "\(error)" }
            throw error
        }
    }

    /// One local folder holding the whole snapshot: fetched through `HubDownloader`
    /// (under `modelsHome/snapshots`, else the default home's; pinned revision honoured,
    /// progress reported), or the profile's own directory. The model and tokenizer then
    /// load from it with no further network call.
    private static func snapshotDirectory(
        for profile: Profile, modelsHome: URL?, status: StatusBox
    ) async throws -> URL {
        let directory: URL
        switch profile.source {
        case .directory(let url):
            directory = url
        case .hub(let id, let revision):
            status.set { $0.phase = .downloading; $0.fraction = 0; $0.error = nil }
            // A home of the caller's own, or the default one: ~/Documents is not searched.
            let downloader = modelsHome.map(HubDownloader.init(home:))
                ?? HubDownloader(hub: HubDownloader.defaultHub, lookupRoots: HubDownloader.snapshotRoots(includeDocuments: false))
            directory = try await downloader.download(
                id: id, revision: revision, matching: snapshotPatterns, useLatest: false,
                progressHandler: { progress in
                    let fraction = progress.fractionCompleted
                    status.set { $0.fraction = fraction.isFinite ? fraction : nil }
                })
        }
        return directory
    }

    /// Prompts from `config_sentence_transformers.json`, when the snapshot has one.
    static func snapshotPrompts(in directory: URL) -> [Role: String] {
        struct File: Decodable { let prompts: [String: String]? }
        guard let data = try? Data(contentsOf: directory.appending(path: "config_sentence_transformers.json")),
            let prompts = (try? JSONDecoder().decode(File.self, from: data))?.prompts
        else { return [:] }
        var result: [Role: String] = [:]
        if let query = prompts["query"] { result[.query] = query }
        if let document = prompts["document"] ?? prompts["passage"] { result[.document] = document }
        return result
    }

    private static func runBatch(
        _ batch: [[Int]],
        padId: Int,
        dimension: Int?,
        model: any mlx_embeddings.EmbeddingModel
    ) throws -> [[Float]] {
        guard !batch.isEmpty else { return [] }
        let rawMax = batch.map { $0.count }.max() ?? 16
        // Power-of-2 padding keeps SDPA shapes within a small set, reducing
        // LRU cache misses even with MLX_CUDA_SDPA_CACHE_SIZE=2048.
        var maxLen = 1
        while maxLen < rawMax { maxLen <<= 1 }

        let padded = MLX.stacked(batch.map { tokens in
            MLXArray(tokens + Array(repeating: padId, count: maxLen - tokens.count))
        })
        // The mask comes from each text's length, not from comparing ids with the pad
        // id: a pad id that is also a real token (voyage's `<|endoftext|>`) would
        // otherwise be hidden wherever it appears in the text.
        let maskValues = batch.flatMap { tokens in
            Array(repeating: true, count: tokens.count)
                + Array(repeating: false, count: maxLen - tokens.count)
        }
        let attentionMask = MLXArray(maskValues, [batch.count, maxLen])
        let tokenTypeIds = MLXArray.zeros(like: padded)

        let output = model(padded, positionIds: nil, tokenTypeIds: tokenTypeIds, attentionMask: attentionMask)
        var embeddings = output.textEmbeds
        if let dimension, dimension < embeddings.dim(1) {
            embeddings = normalizeEmbeddings(embeddings[0..., ..<dimension])
        }
        embeddings = embeddings.asType(.float32)
        MLX.eval(embeddings)

        // One bridge copy for the whole (batch, dim) matrix instead of a
        // GPU sync + copy per row.
        let rows = embeddings.shape[0]
        let dim = embeddings.shape[1]
        let flat = embeddings.asArray(Float.self)
        guard flat.allSatisfy(\.isFinite) else {
            throw EmbedderFailure.nonFinite(rows: rows)
        }
        return (0..<rows).map { Array(flat[$0 * dim ..< ($0 + 1) * dim]) }
    }

    public enum EmbedderFailure: Error, CustomStringConvertible {
        /// The model produced NaN or infinity — a float16 overflow, typically.
        case nonFinite(rows: Int)

        public var description: String {
            switch self {
            case .nonFinite(let rows):
                return "the embedding model produced non-finite values in a batch of \(rows)"
            }
        }
    }
}
