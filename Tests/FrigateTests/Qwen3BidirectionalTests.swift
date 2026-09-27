//
//  Qwen3BidirectionalTests.swift
//  FrigateTests
//
//  WHAT: voyage-4-nano's architecture in mlx_embeddings — a Qwen3 stack with no causal
//        mask, a projection head, mean pooling — chosen by its config, loaded from a
//        checkpoint laid out like the real one, and behaving like an encoder.
//  HOW:  A tiny config (hidden 64, 2 layers). Decoding runs anywhere; anything that
//        allocates arrays is gated like Gemma4SanitizeTests (FRIGATE_MLX_TESTS=1 on
//        Darwin, where the test bundle has no metallib).
//

import Foundation
import MLX
import MLXNN
@testable import mlx_embeddings
import Testing

private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

enum TinyQwen3 {
    static func config(bidirectional: Bool = true, labels: String = #""num_labels": 32,"#) -> String {
        """
        {
          "model_type": "qwen3",
          "hidden_size": 64,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 16,
          "rms_norm_eps": 1e-6,
          "vocab_size": 64,
          "rope_theta": 1000000,
          \(labels)
          "tie_word_embeddings": true,
          "use_bidirectional_attention": \(bidirectional)
        }
        """
    }

    static func writeConfig(_ json: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(path: "qwen3-bidirectional-\(UUID().uuidString).json")
        try Data(json.utf8).write(to: url)
        return url
    }
}

@Suite("Qwen3 bidirectional config")
struct Qwen3BidirectionalConfigTests {

    @Test func theFlagAndTheHeadWidthDecode() throws {
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: Data(TinyQwen3.config().utf8))
        #expect(config.useBidirectionalAttention)
        #expect(config.numLabels == 32)
    }

    @Test func id2labelStandsInForNumLabels() throws {
        let json = TinyQwen3.config(labels: #""id2label": {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"},"#)
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8))
        #expect(config.numLabels == 3)
    }

    @Test func aCausalQwen3ConfigIsUnchanged() throws {
        let json = TinyQwen3.config(bidirectional: false, labels: "")
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8))
        #expect(!config.useBidirectionalAttention)
        #expect(config.numLabels == nil)
    }

    /// The creator refuses before allocating anything, so this needs no MLX.
    @Test func bidirectionalWithoutAHeadWidthIsRefused() throws {
        let url = try TinyQwen3.writeConfig(TinyQwen3.config(labels: ""))
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: (any Error).self) {
            _ = try ModelType(rawValue: "qwen3").createModel(configuration: url)
        }
    }
}

@Suite("Qwen3 bidirectional model", .enabled(if: mlxFunctional))
struct Qwen3BidirectionalModelTests {

    /// A model with deterministic random weights, loaded through `sanitize` and
    /// `update(verify: .all)` from keys laid out like the real checkpoint.
    static func loadedModel() throws -> Qwen3BidirectionalModel {
        let url = try TinyQwen3.writeConfig(TinyQwen3.config())
        defer { try? FileManager.default.removeItem(at: url) }
        let created = try ModelType(rawValue: "qwen3").createModel(configuration: url)
        let model = try #require(created as? Qwen3BidirectionalModel)

        MLXRandom.seed(7)
        var checkpoint: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            // The checkpoint's own names: `model.*` and a bare `linear.weight`, in bf16.
            checkpoint[key] = (MLXRandom.normal(value.shape) * 0.05).asType(.bfloat16)
        }
        checkpoint["model.layers.0.self_attn.rotary_emb.inv_freq"] = MLXArray.zeros([8])
        let weights = model.sanitize(weights: checkpoint)
        #expect(weights["linear.weight"] != nil)
        #expect(weights["model.linear.weight"] == nil)
        #expect(weights.keys.allSatisfy { !$0.contains("inv_freq") })
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        return model
    }

    static func embed(
        _ model: Qwen3BidirectionalModel, _ rows: [[Int32]], width: Int
    ) -> (pooled: MLXArray, hidden: MLXArray) {
        let ids = MLX.stacked(rows.map { MLXArray($0 + Array(repeating: 0, count: width - $0.count)) })
        let mask = MLX.stacked(rows.map {
            MLXArray(Array(repeating: true, count: $0.count) + Array(repeating: false, count: width - $0.count))
        })
        let out = model(ids, positionIds: nil, tokenTypeIds: nil, attentionMask: mask)
        eval(out.textEmbeds)
        return (out.textEmbeds.asType(.float32), out.hiddenStates!.asType(.float32))
    }

    @Test func itIsChosenByTheConfigAndProjectsToTheHeadWidth() throws {
        let model = try Self.loadedModel()
        let (pooled, _) = Self.embed(model, [[1, 2, 3, 4, 5], [6, 7, 8]], width: 8)
        #expect(pooled.shape == [2, 32])
        let norms = sqrt((pooled * pooled).sum(axis: 1)).asArray(Float.self)
        #expect(norms.allSatisfy { abs($0 - 1) < 1e-3 })
    }

    /// Padding is invisible: the same text alone and in a longer, padded batch.
    @Test func paddingDoesNotChangeAnEmbedding() throws {
        let model = try Self.loadedModel()
        let text: [Int32] = [3, 1, 4, 1, 5]
        let alone = Self.embed(model, [text], width: 8).pooled
        let batched = Self.embed(model, [[9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2], text], width: 16).pooled
        let cosine = (alone[0] * batched[1]).sum().item(Float.self)
        #expect(cosine > 0.9999)
    }

    /// Not causal: the first token's state depends on the last one.
    @Test func theFirstTokenSeesTheLastOne() throws {
        let model = try Self.loadedModel()
        let a = Self.embed(model, [[1, 2, 3, 4, 5]], width: 8).hidden
        let b = Self.embed(model, [[1, 2, 3, 4, 60]], width: 8).hidden
        let moved = abs(a[0, 0] - b[0, 0]).max().item(Float.self)
        #expect(moved > 1e-4)
    }

    /// An 8-bit checkpoint quantizes the embedding and the head where it carries scales.
    @Test func aQuantizedHeadAndEmbeddingLoadAndRun() throws {
        let model = try Self.loadedModel()
        quantize(model: model, groupSize: 32, bits: 8) { path, _ in
            path == "linear" || path == "model.embed_tokens"
        }
        let (pooled, _) = Self.embed(model, [[1, 2, 3]], width: 4)
        #expect(pooled.shape == [1, 32])
        let finite = pooled.asArray(Float.self).allSatisfy { $0.isFinite }
        #expect(finite)
    }
}
