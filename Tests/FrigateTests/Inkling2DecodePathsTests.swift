//  WHAT: Which fused paths a decoded token takes is the model's own (`Inkling2Model.fuseDecode`): turned off on one
//        model, another built beside it keeps both. On a model of a few small layers, never loaded.

import Foundation
import MLX
import Testing

@testable import MLXLLM

private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

@Suite("Inkling-2 decode paths", .enabled(if: mlxFunctional))
struct Inkling2DecodePathsTests {
    /// One sliding layer and one global, each small.
    func small() throws -> Inkling2Model {
        let text: [String: Any] = [
            "hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 32, "unpadded_vocab_size": 30,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "swa_num_attention_heads": 2, "swa_num_key_value_heads": 1, "swa_head_dim": 32,
            "sliding_window_size": 8, "local_layer_ids": [0], "d_rel": 16, "rel_extent": 8,
            "log_scaling_n_floor": 1, "log_scaling_alpha": 0.5, "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 1.0, "rel_l2_clip_threshold": 1.0, "sconv_kernel_size": 4,
            "n_layers_per_attn_res_block": 1, "n_routed_experts": 4, "num_experts_per_tok": 2,
            "intermediate_size": 32, "mlp_gate_softcap": 1.0, "mlp_up_softcap": 1.0, "route_scale": 1.0,
            "final_logit_softcapping": 30.0,
        ]
        let data = try JSONSerialization.data(withJSONObject: ["model_type": "inkling2_loop", "text_config": text])
        return Inkling2Model(try JSONDecoder().decode(Inkling2Configuration.self, from: data))
    }

    /// The attention's switches, then the convolutions'.
    func paths(_ model: Inkling2Model) -> (attention: [Bool], convolutions: [Bool]) {
        let modules = model.modules()
        return (modules.compactMap { ($0 as? Inkling2Attention)?.fusesDecode },
                modules.compactMap {
                    ($0 as? Inkling2ShortConvolution)?.fusesDecode ?? ($0 as? Inkling2DynamicConvolution)?.fusesDecode
                })
    }

    @Test("turned off on one model, another keeps both")
    func theModelsOwn() throws {
        let one = try small(), other = try small()
        #expect(paths(one).attention == [true, true])
        #expect(paths(one).convolutions.count == 8, "four short convolutions a sliding layer, two and two dynamic a global")
        #expect(!paths(one).convolutions.contains(false))

        one.fuseDecode(attention: false, convolutions: false)
        #expect(paths(one).attention == [false, false])
        #expect(!paths(one).convolutions.contains(true))
        #expect(paths(other).attention == [true, true])
        #expect(!paths(other).convolutions.contains(false))

        one.fuseDecode(attention: true, convolutions: false)
        #expect(paths(one).attention == [true, true])
        #expect(!paths(one).convolutions.contains(true))
    }
}
