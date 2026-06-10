import Foundation
import MLX
import Tokenizers
import mlx_embeddings

/// On-device text embedding via an MLX model downloaded from HuggingFace Hub.
///
/// GPU safety rule: never call MLX.Memory.*, Stream.*, or any CommandEncoder API
/// from inside `container.perform`. The CUDA allocator is active during that closure
/// and re-entry causes SIGSEGV. All memory management happens after `perform` returns.
public actor FrigateEmbedder {

    private let modelId: String
    private var loadedContainer: mlx_embeddings.ModelContainer?
    private var loadingTask: Task<mlx_embeddings.ModelContainer, Error>?
    /// `Tokenizer` is `Sendable`; cached here so tokenization runs on this actor
    /// *before* `container.perform` — CPU tokenization no longer serializes under
    /// the model lock, overlapping with another request's GPU evaluation.
    private var cachedTokenizer: Tokenizer?
    private var requestsSinceCacheClear = 0

    // Batch limits tuned for RTX 3090 / sm_86. Override via environment without
    // changing the defaults: FRIGATE_MAX_BATCH, FRIGATE_CACHE_LIMIT (bytes),
    // FRIGATE_CACHE_CLEAR_INTERVAL (requests; 0 disables periodic clearing).
    static let maxInputsPerBatch: Int =
        ProcessInfo.processInfo.environment["FRIGATE_MAX_BATCH"].flatMap(Int.init) ?? 8
    // Caps attention matrix: O(seq²). 512 → 4× less VRAM than 1024.
    static let maxTokensPerSequence = 512
    /// Steady-state GPU allocator pool. Set once at model load — the previous
    /// drop-to-zero + clearCache after *every* request forced full buffer
    /// reallocation per request.
    static let cacheLimitBytes: Int =
        ProcessInfo.processInfo.environment["FRIGATE_CACHE_LIMIT"].flatMap(Int.init) ?? 20 * 1_024 * 1_024
    /// Clear the allocator pool every N requests (conservative VRAM hygiene for
    /// long-running CUDA deployments). 0 = never; use `trimMemory()` on demand.
    static let cacheClearInterval: Int =
        ProcessInfo.processInfo.environment["FRIGATE_CACHE_CLEAR_INTERVAL"].flatMap(Int.init) ?? 32

    public init(modelId: String = "mlx-community/snowflake-arctic-embed-m-v1.5") {
        // Raise SDPA LRU cache from 256 → 2048 so varying sequence lengths
        // across sub-batches don't trigger "Cache thrashing" fatal error.
        setenv("MLX_CUDA_SDPA_CACHE_SIZE", "2048", 0)
        self.modelId = modelId
    }

    // MARK: - Public API

    /// Return L2-normalised embeddings for each input string.
    public func embed(_ texts: [String]) async throws -> [[Float]] {
        try await embedWithUsage(texts).embeddings
    }

    /// Return L2-normalised embeddings plus the total prompt-token count
    /// (post-truncation) — the usage datum host servers report per request.
    public func embedWithUsage(_ texts: [String]) async throws
        -> (embeddings: [[Float]], promptTokens: Int) {
        let container = try await loadedContainer()
        let tokenizer = await self.tokenizer(from: container)

        // Tokenize outside the model lock.
        let tokenized = texts.map {
            Array(tokenizer.encode(text: $0, addSpecialTokens: true).prefix(Self.maxTokensPerSequence))
        }
        let promptTokens = tokenized.reduce(0) { $0 + $1.count }
        let padId = tokenizer.eosTokenId ?? 0

        let embeddings = await container.perform { model, _ in
            Self.runBatches(tokenized: tokenized, padId: padId, model: model)
        }
        // CUDA context is idle here — safe to touch the allocator.
        clearCacheIfDue()
        return (embeddings, promptTokens)
    }

    /// Preload the model without running inference (warms up GPU memory).
    public func warmup() async throws {
        _ = try await loadedContainer()
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

    private func tokenizer(from container: mlx_embeddings.ModelContainer) async -> Tokenizer {
        if let t = cachedTokenizer { return t }
        let t = await container.perform { _, tok in tok }
        cachedTokenizer = t
        return t
    }

    private func loadedContainer() async throws -> mlx_embeddings.ModelContainer {
        if let c = loadedContainer { return c }
        if let t = loadingTask { return try await t.value }

        let modelId = self.modelId
        let task = Task<mlx_embeddings.ModelContainer, Error> {
            let config = mlx_embeddings.ModelConfiguration(id: modelId)
            return try await mlx_embeddings.loadModelContainer(configuration: config)
        }
        loadingTask = task

        do {
            let c = try await task.value
            loadedContainer = c
            loadingTask = nil
            MLX.Memory.cacheLimit = Self.cacheLimitBytes
            return c
        } catch {
            loadingTask = nil
            throw error
        }
    }

    private static func runBatches(
        tokenized: [[Int]],
        padId: Int,
        model: any mlx_embeddings.EmbeddingModel
    ) -> [[Float]] {
        var result: [[Float]] = []
        result.reserveCapacity(tokenized.count)

        for batchStart in stride(from: 0, to: tokenized.count, by: maxInputsPerBatch) {
            let batchEnd = min(batchStart + maxInputsPerBatch, tokenized.count)
            let batch = Array(tokenized[batchStart..<batchEnd])

            let rawMax = batch.map { $0.count }.max() ?? 16
            // Power-of-2 padding keeps SDPA shapes within a small set, reducing
            // LRU cache misses even with MLX_CUDA_SDPA_CACHE_SIZE=2048.
            var maxLen = 1
            while maxLen < rawMax { maxLen <<= 1 }

            let paddedArrays = batch.map { tokens in
                MLXArray(tokens + Array(repeating: padId, count: maxLen - tokens.count))
            }
            guard !paddedArrays.isEmpty else { continue }

            let padded = MLX.stacked(paddedArrays)
            let attentionMask = padded .!= MLXArray(padId)
            let tokenTypeIds = MLXArray.zeros(like: padded)

            let output = model(padded, positionIds: nil, tokenTypeIds: tokenTypeIds, attentionMask: attentionMask)
            let embeddings = output.textEmbeds
            MLX.eval(embeddings)

            // One bridge copy for the whole (batch, dim) matrix instead of a
            // GPU sync + copy per row.
            let rows = embeddings.shape[0]
            let dim = embeddings.shape[1]
            let flat = embeddings.asArray(Float.self)
            for i in 0..<rows {
                result.append(Array(flat[i * dim ..< (i + 1) * dim]))
            }
        }

        return result
    }
}
