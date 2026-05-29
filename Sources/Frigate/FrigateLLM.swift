import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// On-device LLM inference via an MLX model downloaded from HuggingFace Hub.
///
/// Token stream terminates after `maxTokens` or when the model emits an EOS token.
public actor FrigateLLM {

    private let modelId: String
    private var loadedContainer: MLXLMCommon.ModelContainer?
    private var loadingTask: Task<MLXLMCommon.ModelContainer, Error>?

    public init(modelId: String = "mlx-community/Qwen3-0.6B-4bit") {
        self.modelId = modelId
    }

    // MARK: - Public API

    /// Stream generated tokens for a plain-text prompt.
    ///
    /// Collect the full response with:
    /// ```swift
    /// var response = ""
    /// for await token in try await llm.generate(prompt: "Hello") {
    ///     response += token
    /// }
    /// ```
    public func generate(
        prompt: String,
        maxTokens: Int = 512
    ) async throws -> AsyncStream<String> {
        let container = try await loadedContainer()
        let userInput = UserInput(prompt: .text(prompt))
        let lmInput = try await container.prepare(input: userInput)
        let genStream = try await container.generate(
            input: lmInput,
            parameters: GenerateParameters(maxTokens: maxTokens)
        )
        return AsyncStream { continuation in
            Task {
                for await generation in genStream {
                    if case .chunk(let text) = generation {
                        continuation.yield(text)
                    }
                }
                continuation.finish()
            }
        }
    }

    /// Preload the model without running inference (warms up GPU memory).
    public func warmup() async throws {
        _ = try await loadedContainer()
    }

    // MARK: - Internal

    private func loadedContainer() async throws -> MLXLMCommon.ModelContainer {
        if let c = loadedContainer { return c }
        if let t = loadingTask { return try await t.value }

        let modelId = self.modelId
        let task = Task<MLXLMCommon.ModelContainer, Error> {
            let config = MLXLMCommon.ModelConfiguration(id: modelId)
            return try await LLMModelFactory.shared.loadContainer(configuration: config)
        }
        loadingTask = task

        do {
            let c = try await task.value
            loadedContainer = c
            loadingTask = nil
            return c
        } catch {
            loadingTask = nil
            throw error
        }
    }
}
