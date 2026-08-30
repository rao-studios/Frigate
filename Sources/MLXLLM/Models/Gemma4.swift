//
//  Gemma4.swift
//  mlx-swift-lm (Frigate)
//
//  Text-path load for gemma4 / gemma4_unified. Vision and audio towers are
//  dropped in sanitize so a unified 12B checkpoint can load as an LLM.
//  Weights themselves are never vendored here — Hub download is the caller's
//  job (Mary Settings).
//
//  Port of https://github.com/ml-explore/mlx-swift-lm Libraries/MLXLLM/Models/Gemma4.swift
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Nested `text_config` wrapper used by `gemma4` and `gemma4_unified`.
public struct Gemma4Configuration: Decodable, Sendable {
    var modelType: String = "gemma4"
    var textConfig: Gemma4TextConfiguration
    var vocabSize: Int = 262144

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case vocabSize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4"
        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        if var textConfig = try container.decodeIfPresent(
            Gemma4TextConfiguration.self, forKey: .textConfig)
        {
            textConfig.vocabSize = self.vocabSize
            self.textConfig = textConfig
        } else {
            self.textConfig = try Gemma4TextConfiguration(from: decoder)
        }
    }
}

/// Outer module for unified checkpoints whose language weights live under
/// `language_model`.
public class Gemma4Model: Module, LLMModel, KVCacheDimensionProvider {
    public var vocabularySize: Int { languageModel.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    @ModuleInfo(key: "language_model") var languageModel: Gemma4TextModel

    public init(_ config: Gemma4Configuration) {
        self._languageModel.wrappedValue = Gemma4TextModel(config.textConfig)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// Drop vision/audio towers so a `gemma4_unified` Hub snapshot loads on
    /// the text path. Key rewrite is public so tests can pin it without
    /// constructing tensors (which needs a Metal library).
    public static func remappedCheckpointKey(_ key: String) -> String? {
        var k = key
        let startsWithModel = k.hasPrefix("model.")
        k = k.replacingOccurrences(of: "model.", with: "", options: .anchored)
        if k.hasPrefix("vision_tower") || k.hasPrefix("multi_modal_projector")
            || k.hasPrefix("audio_tower") || k.hasPrefix("embed_audio")
            || k.hasPrefix("embed_vision") || k.hasPrefix("vision_embedder")
            || k.contains("vision_embedder")
        {
            return nil
        }
        if !startsWithModel {
            return k
        }
        if k.hasPrefix("language_model") {
            k = k.replacingOccurrences(
                of: "language_model.", with: "language_model.model.", options: .anchored)
        }
        return k
    }

    public static func droppingMultimodalKeys(
        _ weights: [String: MLXArray]
    ) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if let remapped = remappedCheckpointKey(key) {
                sanitized[remapped] = value
            }
        }
        return sanitized
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        languageModel.sanitize(weights: Self.droppingMultimodalKeys(weights))
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.newCache(parameters: parameters)
    }
}

extension Gemma4Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.loraLayers
    }
}
