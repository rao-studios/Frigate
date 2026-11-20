//  WHAT: SoftPromptModel's promise (MLXLMCommon/SoftPrompt.swift): a model's own token embeddings, read through the
//        embeddings path, give the token path's logits bit for bit — whole, and spliced between token pieces into one
//        cache as a soft span is. On tiny random Qwen3.5 (Gated DeltaNet + attention) and Nemotron-H (Mamba + attention)
//        models, so both recurrent families and both caches are covered. And the fingerprint: a model is itself, a
//        different model is not, and no bridge crosses widths.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

@Suite("Soft prompts", .enabled(if: mlxFunctional), .serialized)
struct SoftPromptTests {
    /// Three Gated DeltaNet layers and one attention layer, as Qwen3.5's interval of 4 lays them out. The key heads are
    /// 32 wide: the Gated DeltaNet kernel spreads a key over 32 threads (`Dk / 32`), and a narrower key fails to build.
    static func qwen35(hidden: Int = 32, seed: UInt64 = 1) throws -> Qwen35TextModel {
        MLXRandom.seed(seed)
        let json = """
        {
          "hidden_size": \(hidden), "num_hidden_layers": 4, "intermediate_size": 64, "num_attention_heads": 2,
          "num_key_value_heads": 1, "head_dim": 16, "linear_num_value_heads": 2, "linear_num_key_heads": 2,
          "linear_key_head_dim": 32, "linear_value_head_dim": 32, "linear_conv_kernel_dim": 4, "vocab_size": 64,
          "tie_word_embeddings": true, "full_attention_interval": 4
        }
        """
        let model = Qwen35TextModel(try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8)))
        eval(model)
        return model
    }

    static func nemotron() throws -> NemotronHModel {
        MLXRandom.seed(2)
        let json = """
        {
          "model_type": "nemotron_h", "vocab_size": 64, "hidden_size": 16, "num_hidden_layers": 3,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
          "mamba_num_heads": 2, "mamba_head_dim": 8, "ssm_state_size": 4, "conv_kernel": 4, "n_groups": 1,
          "intermediate_size": 32, "moe_intermediate_size": 12, "moe_shared_expert_intermediate_size": 24,
          "n_routed_experts": 4, "n_shared_experts": 1, "num_experts_per_tok": 2, "tie_word_embeddings": true,
          "hybrid_override_pattern": "M*M"
        }
        """
        let model = NemotronHModel(try JSONDecoder().decode(NemotronHConfiguration.self, from: Data(json.utf8)))
        eval(model)
        return model
    }

    private let ids = MLXArray((0 ..< 24).map { Int32(($0 * 37) % 61 + 1) }).reshaped(1, 24)

    /// The whole prompt read through the embeddings path is the token path.
    private func wholeIsTheTokens(_ model: any SoftPromptModel) {
        let tokens = model(ids, cache: model.newCache(parameters: nil))
        let embedded = model(embeddings: model.embed(ids), cache: model.newCache(parameters: nil))
        #expect(abs(tokens - embedded).max().item(Float.self) == 0)
    }

    /// Tokens, then a span of the model's own embeddings, then tokens, into one cache: the token path read in the
    /// same three pieces. Decoding on after it continues from the same state.
    private func aSpanBetweenTokensIsTheTokens(_ model: any SoftPromptModel) {
        let cut = [0, 8, 20, 24]
        let plain = model.newCache(parameters: nil)
        let spliced = model.newCache(parameters: nil)
        for k in 0 ..< 3 {
            let piece = ids[0..., cut[k] ..< cut[k + 1]]
            let a = model(piece, cache: plain)
            let b = k == 1 ? model(embeddings: model.embed(piece), cache: spliced) : model(piece, cache: spliced)
            #expect(abs(a - b).max().item(Float.self) == 0, "piece \(k)")
        }
        let next = MLXArray([Int32(5)]).reshaped(1, 1)
        let a = model(next, cache: plain)
        let b = model(next, cache: spliced)
        #expect(abs(a - b).max().item(Float.self) == 0, "the token after the prompt")
    }

    @Test func qwen35EmbeddingsAreItsTokens() throws {
        let model = try Self.qwen35()
        #expect(model.hiddenSize == 32)
        #expect(model.embed(ids).shape == [1, 24, 32])
        wholeIsTheTokens(model)
        aSpanBetweenTokensIsTheTokens(model)
    }

    @Test func nemotronEmbeddingsAreItsTokens() throws {
        let model = try Self.nemotron()
        #expect(model.hiddenSize == 16)
        wholeIsTheTokens(model)
        aSpanBetweenTokensIsTheTokens(model)
    }

    @Test func aFingerprintKnowsItsModel() throws {
        let model = try Self.qwen35()
        // The tiny vocabulary is 64 wide; the fingerprint's rows sit past it, so read its own ids modulo 64.
        let print = model.embed(MLXArray(SoftPrompt.fingerprintIDs.map { $0 % 64 }).reshaped(1, 64)).squeezed(axis: 0)
        #expect(abs(SoftPrompt.likeness(print, print) - 1) < 1e-5)
        let other = try Self.qwen35(seed: 9)
        let otherPrint = other.embed(MLXArray(SoftPrompt.fingerprintIDs.map { $0 % 64 }).reshaped(1, 64)).squeezed(axis: 0)
        #expect(SoftPrompt.likeness(print, otherPrint) < 0.5)
        let wide = try Self.qwen35(hidden: 48)
        let widePrint = wide.embed(MLXArray(SoftPrompt.fingerprintIDs.map { $0 % 64 }).reshaped(1, 64)).squeezed(axis: 0)
        #expect(SoftPrompt.likeness(print, widePrint) == 0)
    }
}
