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

    // Batch limits tuned for RTX 3090 / sm_86
    static let maxInputsPerBatch = 8
    // Caps attention matrix: O(seq²). 512 → 4× less VRAM than 1024.
    static let maxTokensPerSequence = 512

    public init(modelId: String = "mlx-community/snowflake-arctic-embed-m-v1.5") {
        // Raise SDPA LRU cache from 256 → 2048 so varying sequence lengths
        // across sub-batches don't trigger "Cache thrashing" fatal error.
        setenv("MLX_CUDA_SDPA_CACHE_SIZE", "2048", 0)
        self.modelId = modelId
    }

    // MARK: - Public API

    /// Return L2-normalised embeddings for each input string.
    public func embed(_ texts: [String]) async throws -> [[Float]] {
        let container = try await loadedContainer()
        let embeddings = await container.perform { model, tokenizer in
            Self.runBatches(texts: texts, model: model, tokenizer: tokenizer)
        }
        // CUDA context is idle here — safe to touch the allocator.
        MLX.Memory.cacheLimit = 0
        MLX.Memory.clearCache()
        MLX.Memory.cacheLimit = 20 * 1_024 * 1_024
        return embeddings
    }

    /// Preload the model without running inference (warms up GPU memory).
    public func warmup() async throws {
        _ = try await loadedContainer()
    }

    // MARK: - Internal

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
            MLX.Memory.cacheLimit = 20 * 1_024 * 1_024
            return c
        } catch {
            loadingTask = nil
            throw error
        }
    }

    private static func runBatches(
        texts: [String],
        model: any mlx_embeddings.EmbeddingModel,
        tokenizer: any Tokenizer
    ) -> [[Float]] {
        var result: [[Float]] = []

        for batchStart in stride(from: 0, to: texts.count, by: maxInputsPerBatch) {
            let batchEnd = min(batchStart + maxInputsPerBatch, texts.count)
            let batch = Array(texts[batchStart..<batchEnd])

            let tokenized = batch.map {
                Array(tokenizer.encode(text: $0, addSpecialTokens: true).prefix(maxTokensPerSequence))
            }

            let rawMax = tokenized.map { $0.count }.max() ?? 16
            // Power-of-2 padding keeps SDPA shapes within a small set, reducing
            // LRU cache misses even with MLX_CUDA_SDPA_CACHE_SIZE=2048.
            var maxLen = 1
            while maxLen < rawMax { maxLen <<= 1 }

            let padId = tokenizer.eosTokenId ?? 0
            let paddedArrays = tokenized.map { tokens in
                MLXArray(tokens + Array(repeating: padId, count: maxLen - tokens.count))
            }
            guard !paddedArrays.isEmpty else { continue }

            let padded = MLX.stacked(paddedArrays)
            let attentionMask = padded .!= MLXArray(padId)
            let tokenTypeIds = MLXArray.zeros(like: padded)

            let output = model(padded, positionIds: nil, tokenTypeIds: tokenTypeIds, attentionMask: attentionMask)
            let embeddings = output.textEmbeds
            MLX.eval(embeddings)

            for i in 0..<embeddings.shape[0] {
                result.append(embeddings[i].asArray(Float.self))
            }
        }

        return result
    }
}
