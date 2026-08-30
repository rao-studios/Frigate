import Foundation
@testable import MLXLLM
import Testing

@Suite("Gemma 4 load path")
struct Gemma4SanitizeTests {

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
                "enable_moe_block": false,
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
        #expect(config.textConfig.enableMoEBlock == false)
    }

    @Test func droppingMultimodalKeysKeepsLanguageWeights() {
        let remapped = [
            "model.vision_embedder.weight",
            "model.vision_tower.layers.0.weight",
            "model.audio_tower.weight",
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "language_model.model.embed_tokens.weight",
            "language_model.model.layers.0.layer_scalar",
        ].compactMap { Gemma4Model.remappedCheckpointKey($0) }
        #expect(remapped.contains {
            $0.contains("language_model") && $0.contains("q_proj")
        })
        #expect(remapped.contains("language_model.model.embed_tokens.weight"))
        #expect(remapped.contains("language_model.model.layers.0.layer_scalar"))
        #expect(!remapped.contains { $0.contains("language_model.model.model.") })
        #expect(!remapped.contains { $0.contains("vision") })
        #expect(!remapped.contains { $0.contains("audio") })
    }

    @Test func registryConvenienceIdIsHubOnly() {
        #expect(
            LLMRegistry.gemma4_12b_coder_4bit.name
                == "mlx-community/gemma-4-12b-coder-fable5-composer2.5-4bit")
    }
}
