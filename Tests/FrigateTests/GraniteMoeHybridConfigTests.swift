import Foundation
@testable import MLXLLM
import Testing

/// Granite 4.0's Mamba time step, decoded from a config: the clamp is transformers' (0, +inf) unless the config names one.
/// A higher floor raised every small step and turned a long prompt into "1.1.1.1" (H-Micro past ~2k tokens, 2026-11-28).
@Suite("Granite 4.0 hybrid config")
struct GraniteMoeHybridConfigTests {
    /// H-Micro's config.json less its layer list; it names no `time_step_limit`.
    private func config(adding extra: String = "") throws -> GraniteMoeHybridConfiguration {
        let json = """
            {
              "model_type": "granitemoehybrid", "vocab_size": 100352, "hidden_size": 2048, "intermediate_size": 8192,
              "num_hidden_layers": 2, "max_position_embeddings": 131072, "num_attention_heads": 32,
              "num_key_value_heads": 8, "attention_bias": false, "embedding_multiplier": 12,
              "attention_multiplier": 0.015625, "logits_scaling": 8, "residual_multiplier": 0.22,
              "layer_types": ["mamba", "attention"], "rms_norm_eps": 1e-05, "rope_theta": 10000,
              "position_embedding_type": "nope", "mamba_n_heads": 64, "mamba_d_head": 64, "mamba_d_state": 128,
              "mamba_d_conv": 4, "mamba_n_groups": 1, "mamba_conv_bias": true, "mamba_proj_bias": false\(extra)
            }
            """
        return try JSONDecoder().decode(GraniteMoeHybridConfiguration.self, from: Data(json.utf8))
    }

    @Test func unsetTheTimeStepIsUnclamped() throws {
        let args = try config()
        #expect(args.timeStepMinimum == 0)
        #expect(args.timeStepMaximum == .infinity)
    }

    @Test func aNamedLimitIsKept() throws {
        let args = try config(adding: #", "time_step_limit": [0.001, 100.0]"#)
        #expect(args.timeStepMinimum == 0.001)
        #expect(args.timeStepMaximum == 100)
    }
}
