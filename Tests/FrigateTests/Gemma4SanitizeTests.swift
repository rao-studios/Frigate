import Foundation
import MLX
@testable import MLXLLM
import Testing

/// MLX graph ops abort on Darwin: the vendored Cmlx compiles the Metal backend but ships
/// no metallib in the test bundle, so array creation fatals. Same gate as VectorOpsTests.
private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

/// Gemma 4 load path.
///
/// These cover the two things that break silently when a checkpoint and the module tree
/// disagree: the unified config must decode without allocating the model, and `sanitize`
/// must drop the multimodal towers while remapping the language weights. Upstream
/// mlx-swift-lm 3.x carries Gemma 4 itself now, so this exercises upstream's API —
/// `sanitize(weights:)` on the model rather than a static key-remap helper.
@Suite("Gemma 4 load path")
struct Gemma4SanitizeTests {

    /// A deliberately tiny text config, so `sanitize` can run against a real module tree
    /// without allocating a 12B parameter model.
    static let tinyConfigJSON = """
        {
          "model_type": "gemma4_unified",
          "vocab_size": 32,
          "text_config": {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "head_dim": 4,
            "global_head_dim": 4,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 4,
            "vocab_size_per_layer_input": 32,
            "attention_k_eq_v": true,
            "final_logit_softcapping": 30.0,
            "use_double_wide_mlp": false,
            "sliding_window": 8,
            "max_position_embeddings": 64,
            "rope_parameters": {
              "sliding_attention": { "rope_theta": 10000.0 },
              "full_attention": { "rope_theta": 1000000.0, "partial_rotary_factor": 0.25 }
            }
          }
        }
        """

    @Test func unifiedConfigDecodesWithoutAllocatingTheModel() throws {
        let json = """
            {
              "model_type": "gemma4_unified",
              "vocab_size": 262144,
              "text_config": {
                "hidden_size": 3840,
                "num_hidden_layers": 48,
                "intermediate_size": 15360,
                "num_attention_heads": 16,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_key_value_heads": 8,
                "num_global_key_value_heads": 1,
                "num_kv_shared_layers": 0,
                "hidden_size_per_layer_input": 0,
                "attention_k_eq_v": true,
                "final_logit_softcapping": 30.0,
                "use_double_wide_mlp": false,
                "sliding_window": 1024,
                "max_position_embeddings": 262144,
                "rope_parameters": {
                  "sliding_attention": { "rope_theta": 10000.0 },
                  "full_attention": { "rope_theta": 1000000.0, "partial_rotary_factor": 0.25 }
                }
              }
            }
            """
        let config = try JSONDecoder().decode(
            Gemma4Configuration.self, from: Data(json.utf8))
        #expect(config.modelType == "gemma4_unified")
        #expect(config.textConfig.hiddenSize == 3840)
        #expect(config.textConfig.numHiddenLayers == 48)
        #expect(config.textConfig.attentionKeqV)
        #expect(config.textConfig.numKvSharedLayers == 0)
        #expect(config.textConfig.fullPartialRotaryFactor == 0.25)
        // vocab_size propagates from the outer config into the text config
        #expect(config.textConfig.vocabSize == 262144)
    }

    @Test func registryConvenienceIdIsHubOnly() {
        #expect(LLMRegistry.gemma4_e4b_it_4bit.name == "mlx-community/gemma-4-e4b-it-4bit")
        #expect(LLMRegistry.gemma4_e2b_it_4bit.name == "mlx-community/gemma-4-e2b-it-4bit")
    }
}

/// The sanitize path needs a real module tree, so it only runs where MLX is functional.
@Suite("Gemma 4 sanitize", .enabled(if: mlxFunctional))
struct Gemma4SanitizeMLXTests {

    private static let tinyConfigJSON = Gemma4SanitizeTests.tinyConfigJSON

    @Test func droppingMultimodalKeysKeepsLanguageWeights() throws {
        let config = try JSONDecoder().decode(
            Gemma4Configuration.self, from: Data(Self.tinyConfigJSON.utf8))
        let model = Gemma4Model(config)

        let scalar = MLXArray(0)
        let weights: [String: MLXArray] = [
            "model.vision_embedder.weight": scalar,
            "model.vision_tower.layers.0.weight": scalar,
            "model.audio_tower.weight": scalar,
            "model.multi_modal_projector.weight": scalar,
            "model.language_model.layers.0.self_attn.q_proj.weight": scalar,
            "language_model.model.embed_tokens.weight": scalar,
            "language_model.model.layers.0.layer_scalar": scalar,
        ]
        let keys = Set(model.sanitize(weights: weights).keys)

        // language weights survive, and the "model." prefix is remapped, not doubled
        #expect(keys.contains { $0.contains("language_model") && $0.contains("q_proj") })
        #expect(keys.contains("language_model.model.embed_tokens.weight"))
        #expect(keys.contains("language_model.model.layers.0.layer_scalar"))
        #expect(!keys.contains { $0.contains("language_model.model.model.") })

        // the multimodal towers are dropped — leaving them in fails the load with
        // `Unhandled keys ["vision_embedder"]`
        #expect(!keys.contains { $0.contains("vision") })
        #expect(!keys.contains { $0.contains("audio") })
        #expect(!keys.contains { $0.contains("multi_modal_projector") })
    }
}
