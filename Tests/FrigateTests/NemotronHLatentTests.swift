import Foundation
import MLX
import MLXNN
@testable import MLXLLM
import Testing

/// MLX graph ops abort on Darwin unless a metallib is beside the test bundle. Same gate as Gemma4SanitizeTests.
private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

/// Nemotron 3 Super's LatentMoE: routed experts run in a `moe_latent_size` space, projected down before them
/// (`fc1_latent_proj`) and back up after (`fc2_latent_proj`); the shared expert stays on the hidden size. Without the
/// projections the module tree has no place for `backbone.layers.N.mixer.fc1_latent_proj.*`, and a strict load fails.
@Suite("Nemotron-H latent MoE")
struct NemotronHLatentTests {
    /// Mamba, MoE, attention: one layer of each, small enough to build.
    static func tinyConfig(latent: Int?) -> String {
        """
        {
          "model_type": "nemotron_h", "vocab_size": 32, "hidden_size": 16, "num_hidden_layers": 3,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
          "mamba_num_heads": 2, "mamba_head_dim": 8, "ssm_state_size": 4, "conv_kernel": 4, "n_groups": 1,
          "intermediate_size": 32, "moe_intermediate_size": 12, "moe_shared_expert_intermediate_size": 24,
          "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2,
          \(latent.map { "\"moe_latent_size\": \($0)," } ?? "")
          "hybrid_override_pattern": "ME*"
        }
        """
    }

    @Test func theLatentSizeIsReadAndOptional() throws {
        let latent = try JSONDecoder().decode(NemotronHConfiguration.self, from: Data(Self.tinyConfig(latent: 8).utf8))
        #expect(latent.moeLatentSize == 8)
        let plain = try JSONDecoder().decode(NemotronHConfiguration.self, from: Data(Self.tinyConfig(latent: nil).utf8))
        #expect(plain.moeLatentSize == nil, "Nemotron 3 Nano and 3.5 Lightning have no latent space")
    }
}

@Suite("Nemotron-H latent MoE module tree", .enabled(if: mlxFunctional))
struct NemotronHLatentMLXTests {
    private func shapes(latent: Int?) throws -> [String: [Int]] {
        let config = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: Data(NemotronHLatentTests.tinyConfig(latent: latent).utf8))
        let model = NemotronHModel(config)
        return Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
    }

    @Test func routedExpertsRunInTheLatentSpaceBetweenTwoProjections() throws {
        let latent = try shapes(latent: 8)
        #expect(latent["backbone.layers.1.mixer.fc1_latent_proj.weight"] == [8, 16])
        #expect(latent["backbone.layers.1.mixer.fc2_latent_proj.weight"] == [16, 8])
        // The experts take the latent width in; the shared expert keeps the hidden one.
        #expect(latent["backbone.layers.1.mixer.switch_mlp.fc1.weight"]?.last == 8)
        #expect(latent["backbone.layers.1.mixer.shared_experts.up_proj.weight"]?.last == 16)

        let plain = try shapes(latent: nil)
        #expect(plain["backbone.layers.1.mixer.fc1_latent_proj.weight"] == nil)
        #expect(plain["backbone.layers.1.mixer.switch_mlp.fc1.weight"]?.last == 16)
    }
}
