//
//  SoftPrompt.swift
//  MLXLMCommon
//
//  WHAT: A language model whose prompt can carry vectors where tokens would be: a soft span. A tower outside the model
//        (a vision encoder, or Braid's commons through a trained bridge) turns what it read into vectors of the model's
//        width; the model reads them in the prompt as it reads its own token embeddings, and decodes on token ids after.
//  PIN:  THE SAME PATH AS TOKENS. `callAsFunction(embeddings:cache:)` runs exactly what `callAsFunction(_:cache:)` runs
//        after its embedding lookup, so `embed(ids)` spliced in as a span gives the token prompt's logits bit for bit
//        (SoftPromptTests). A model adopts it by letting its inner model take embeddings in place of ids.
//  PIN:  POSITIONS ARE THE CACHE'S. Attention layers rotate by their cache's offset and recurrent layers carry no
//        positions, so a span takes its place in the sequence like the tokens it stands for. A span that needs other
//        positions (an image's three-axis rotary) is not this protocol's.
//

import Foundation
import MLX

public protocol SoftPromptModel: LanguageModel {
    /// The width a soft span's vectors must have: the model's hidden size.
    var hiddenSize: Int { get }
    /// The embeddings of `ids` ([B, L]), [B, L, hiddenSize], in the embedding's dtype.
    func embed(_ ids: MLXArray) -> MLXArray
    /// Logits for `embeddings` ([B, L, hiddenSize]) read into `cache`, as the same positions' tokens would be.
    func callAsFunction(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray
    /// Whether a loss on the logits can be read back to the embeddings: every layer differentiable (a Gated DeltaNet
    /// runs its chunked form under `GatedDeltaPath.differentiable`). A bridge trains through a model only when it does.
    var takesInputGradients: Bool { get }
}

extension SoftPromptModel {
    public var takesInputGradients: Bool { false }
}

public enum SoftPrompt {
    /// The rows a fingerprint samples: spread over the vocabulary's first 32,768 ids, which every vocabulary Frigate
    /// loads holds, and fixed, so two models' fingerprints compare row for row.
    public static let fingerprintIDs: [Int32] = (0 ..< 64).map { Int32($0 * 509 + 7) }

    /// The model's embedding rows at `fingerprintIDs`, [64, hiddenSize] in float32: what a bridge trained against one
    /// model compares with another's to say whether it fits (fine-tunes of one base keep their embeddings close).
    public static func fingerprint(_ model: any SoftPromptModel) -> MLXArray {
        model.embed(MLXArray(fingerprintIDs).reshaped(1, fingerprintIDs.count)).squeezed(axis: 0).asType(.float32)
    }

    /// How alike two fingerprints are: the mean cosine of their rows, 1 for the same embeddings; 0 when the widths
    /// differ, since no bridge crosses widths.
    public static func likeness(_ a: MLXArray, _ b: MLXArray) -> Float {
        guard a.shape == b.shape else { return 0 }
        let dot = (a * b).sum(axis: -1)
        let norms = sqrt((a * a).sum(axis: -1)) * sqrt((b * b).sum(axis: -1))
        return (dot / maximum(norms, 1e-12)).mean().item(Float.self)
    }
}
