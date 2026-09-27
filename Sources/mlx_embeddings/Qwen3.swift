import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXLinalg

private class Attention: Module {
  let args: Qwen3Configuration
  let scale: Float

  @ModuleInfo(key: "q_proj") var wq: Linear
  @ModuleInfo(key: "k_proj") var wk: Linear
  @ModuleInfo(key: "v_proj") var wv: Linear
  @ModuleInfo(key: "o_proj") var wo: Linear

  @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

  let rope: RoPE

  public init(_ args: Qwen3Configuration) {
    self.args = args

    let dim = args.hiddenSize
    let heads = args.attentionHeads
    let kvHeads = args.kvHeads

    let headDim = args.headDim
    self.scale = pow(Float(headDim), -0.5)

    _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
    _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
    _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
    _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

    _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
    _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

    let ropeScale: Float
    if let ropeScaling = args.ropeScaling, ropeScaling["type"] == .string("linear"),
      let factor = ropeScaling["factor"]
    {
      if let v = factor.asFloat() {
        ropeScale = 1 / v
      } else {
        fatalError("ropeScaling.factor must be a float")
      }
    } else {
      ropeScale = 1
    }

    self.rope = RoPE(
      dimensions: headDim, traditional: false, base: args.ropeTheta,
      scale: ropeScale)
  }

  public func callAsFunction(
    _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
  ) -> MLXArray {
    let (B, L) = (x.dim(0), x.dim(1))

    var queries = wq(x)
    var keys = wk(x)
    var values = wv(x)

    // prepare the queries, keys and values for the attention computation
    queries = qNorm(queries.reshaped(B, L, args.attentionHeads, -1)).transposed(0, 2, 1, 3)
    keys = kNorm(keys.reshaped(B, L, args.kvHeads, -1)).transposed(0, 2, 1, 3)
    values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

    if let cache {
      queries = rope(queries, offset: cache.offset)
      keys = rope(keys, offset: cache.offset)
      (keys, values) = cache.update(keys: keys, values: values)
    } else {
      queries = rope(queries)
      keys = rope(keys)
    }

    let output = MLXFast.scaledDotProductAttention(
      queries: queries, keys: keys, values: values, scale: scale, mask: mask
    )
    .transposed(0, 2, 1, 3)
    .reshaped(B, L, -1)

    return wo(output)
  }
}

private class MLP: Module, UnaryLayer {
  @ModuleInfo(key: "gate_proj") var gate: Linear
  @ModuleInfo(key: "down_proj") var down: Linear
  @ModuleInfo(key: "up_proj") var up: Linear

  public init(dimensions: Int, hiddenDimensions: Int) {
    _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    down(silu(gate(x)) * up(x))
  }
}

private class TransformerBlock: Module {
  @ModuleInfo(key: "self_attn") var attention: Attention
  let mlp: MLP

  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  public init(_ args: Qwen3Configuration) {
    _attention.wrappedValue = Attention(args)
    self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
    _inputLayerNorm.wrappedValue = RMSNorm(
      dimensions: args.hiddenSize, eps: args.rmsNormEps)
    _postAttentionLayerNorm.wrappedValue = RMSNorm(
      dimensions: args.hiddenSize, eps: args.rmsNormEps)
  }

  public func callAsFunction(
    _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache?
  ) -> MLXArray {
    var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
    let h = x + r
    r = mlp(postAttentionLayerNorm(h))
    let out = h + r
    return out
  }
}

private class Qwen3ModelInner: Module {
  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

  fileprivate let layers: [TransformerBlock]
  let norm: RMSNorm

  public init(_ args: Qwen3Configuration) {
    precondition(args.vocabularySize > 0)

    _embedTokens.wrappedValue = Embedding(
      embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

    self.layers = (0..<args.hiddenLayers)
      .map { _ in
        TransformerBlock(args)
      }
    self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
  }

  public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
    var h = embedTokens(inputs)

    let mask: MLXArray? = createAttentionMask(h: h, cache: cache)

    for (i, layer) in layers.enumerated() {
      h = layer(h, mask: mask, cache: cache?[i])
    }

    return norm(h)
  }

  /// Bidirectional pass: every token sees every real token; only padding is hidden.
  /// `keyMask` is bool `[B, 1, 1, L]` over keys. No cache, positions 0..<L — right
  /// padding leaves the real tokens' positions where the checkpoint trained them.
  public func callAsFunction(_ inputs: MLXArray, keyMask: MLXArray) -> MLXArray {
    var h = embedTokens(inputs)
    for layer in layers {
      h = layer(h, mask: keyMask, cache: nil)
    }
    return norm(h)
  }
}

public class Qwen3Model: Module, EmbeddingModel {
  public let vocabularySize: Int
  public let kvHeads: [Int]

  @ModuleInfo(key: "model") private var model: Qwen3ModelInner
  let configuration: Qwen3Configuration

  public init(_ args: Qwen3Configuration) {
    self.configuration = args
    self.vocabularySize = args.vocabularySize
    self.kvHeads = (0..<args.hiddenLayers).map { _ in args.kvHeads }
    self._model.wrappedValue = Qwen3ModelInner(args)
  }

  public func callAsFunction(
    _ inputIds: MLXArray, positionIds: MLXArray? = nil, tokenTypeIds: MLXArray? = nil,
    attentionMask: MLXArray? = nil
  )
    -> EmbeddingModelOutput
  {
    let out = model(inputIds, cache: nil)
    var text_embeds = lastTokenPooling(
      lastHiddenState: out,
      attentionMask: attentionMask!)
    text_embeds = normalizeEmbeddings(text_embeds)
    return EmbeddingModelOutput(
      hiddenStates: out,
      poolerOutput: nil,
      textEmbeds: text_embeds)
  }

  public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitizedWeights = [String: MLXArray]()
    
    for (key, value) in weights {
      // Skip unused keys
      if key.contains("self_attn.rotary_emb.inv_freq") || key.contains("lm_head") {
        continue
      }
      
      var newKey = key
      if !newKey.hasPrefix("model.") {
        newKey = "model." + newKey
      }
      
      sanitizedWeights[newKey] = value
    }
    
    return sanitizedWeights
  }
}

/// A Qwen3 encoder trained as a bidirectional embedder with a projection head —
/// `voyageai/voyage-4-nano` and its conversions (`use_bidirectional_attention: true`).
///
/// The upstream PyTorch module is `Qwen3BidirectionalModel`: the Qwen3 stack with every
/// layer non-causal, then `linear` (hidden → `num_labels`, no bias) on each token, then
/// sentence-transformers mean pooling over the attention mask — prompt tokens included —
/// and L2 normalisation. Matryoshka truncation, when a caller wants fewer dimensions,
/// happens after this, in `FrigateEmbedder`.
public class Qwen3BidirectionalModel: Module, EmbeddingModel {
  public let vocabularySize: Int

  @ModuleInfo(key: "model") private var model: Qwen3ModelInner
  @ModuleInfo(key: "linear") var linear: Linear
  let configuration: Qwen3Configuration

  public init(_ args: Qwen3Configuration) {
    precondition((args.numLabels ?? 0) > 0, "a bidirectional Qwen3 embedder needs num_labels")
    self.configuration = args
    self.vocabularySize = args.vocabularySize
    self._model.wrappedValue = Qwen3ModelInner(args)
    self._linear.wrappedValue = Linear(args.hiddenSize, args.numLabels ?? 0, bias: false)
  }

  public func callAsFunction(
    _ inputIds: MLXArray, positionIds: MLXArray? = nil, tokenTypeIds: MLXArray? = nil,
    attentionMask: MLXArray? = nil
  )
    -> EmbeddingModelOutput
  {
    let mask = (attentionMask ?? MLXArray.ones(like: inputIds)).asType(.bool)
    let hidden = model(inputIds, keyMask: mask.expandedDimensions(axes: [1, 2]))
    let projected = linear(hidden)
    let pooled = normalizeEmbeddings(
      meanPooling(lastHiddenState: projected, attentionMask: mask))
    return EmbeddingModelOutput(hiddenStates: projected, poolerOutput: nil, textEmbeds: pooled)
  }

  /// The checkpoint keeps the head at the top level (`linear.weight`) beside `model.*`.
  /// Anything half-precision is carried as bfloat16: this model overflows float16 on
  /// real text (NaN rows), and the compute dtype is whatever the weights are.
  public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized = [String: MLXArray]()
    for (key, value) in weights {
      if key.contains("self_attn.rotary_emb.inv_freq") || key.contains("lm_head") { continue }
      let newKey = key.hasPrefix("model.") || key.hasPrefix("linear.") ? key : "model." + key
      sanitized[newKey] = value.dtype == .float16 ? value.asType(.bfloat16) : value
    }
    return sanitized
  }
}

public struct Qwen3Configuration: Codable, Sendable {
  var hiddenSize: Int
  var hiddenLayers: Int
  var intermediateSize: Int
  var attentionHeads: Int
  var rmsNormEps: Float
  var vocabularySize: Int
  var kvHeads: Int
  var ropeTheta: Float = 1_000_000
  var headDim: Int
  var ropeScaling: [String: StringOrNumber]? = nil
  var tieWordEmbeddings = false
  var maxPositionEmbeddings: Int = 32768
  /// Set by bidirectional embedders (voyage-4-nano): no causal mask, a projection head.
  var useBidirectionalAttention = false
  /// The projection head's width: `num_labels`, else the size of `id2label`.
  var numLabels: Int? = nil

  enum CodingKeys: String, CodingKey {
    case hiddenSize = "hidden_size"
    case hiddenLayers = "num_hidden_layers"
    case intermediateSize = "intermediate_size"
    case attentionHeads = "num_attention_heads"
    case rmsNormEps = "rms_norm_eps"
    case vocabularySize = "vocab_size"
    case kvHeads = "num_key_value_heads"
    case ropeTheta = "rope_theta"
    case headDim = "head_dim"
    case ropeScaling = "rope_scaling"
    case tieWordEmbeddings = "tie_word_embeddings"
    case maxPositionEmbeddings = "max_position_embeddings"
    case useBidirectionalAttention = "use_bidirectional_attention"
    case numLabels = "num_labels"
    case id2label = "id2label"
  }

  public init(from decoder: Decoder) throws {
    // custom implementation to handle optional keys with required values
    let container: KeyedDecodingContainer<Qwen3Configuration.CodingKeys> =
      try decoder.container(
        keyedBy: Qwen3Configuration.CodingKeys.self)

    self.hiddenSize = try container.decode(
      Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenSize)
    self.hiddenLayers = try container.decode(
      Int.self, forKey: Qwen3Configuration.CodingKeys.hiddenLayers)
    self.intermediateSize = try container.decode(
      Int.self, forKey: Qwen3Configuration.CodingKeys.intermediateSize)
    self.attentionHeads = try container.decode(
      Int.self, forKey: Qwen3Configuration.CodingKeys.attentionHeads)
    self.rmsNormEps = try container.decode(
      Float.self, forKey: Qwen3Configuration.CodingKeys.rmsNormEps)
    self.vocabularySize = try container.decode(
      Int.self, forKey: Qwen3Configuration.CodingKeys.vocabularySize)
    self.kvHeads = try container.decode(Int.self, forKey: Qwen3Configuration.CodingKeys.kvHeads)
    self.ropeTheta =
      try container.decodeIfPresent(
        Float.self, forKey: Qwen3Configuration.CodingKeys.ropeTheta)
      ?? 1_000_000
    self.headDim = try container.decode(
      Int.self, forKey: Qwen3Configuration.CodingKeys.headDim)
    self.ropeScaling = try container.decodeIfPresent(
      [String: StringOrNumber].self, forKey: Qwen3Configuration.CodingKeys.ropeScaling)
    self.tieWordEmbeddings =
      try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    self.maxPositionEmbeddings =
      try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
    self.useBidirectionalAttention =
      try container.decodeIfPresent(Bool.self, forKey: .useBidirectionalAttention) ?? false
    if let labels = try container.decodeIfPresent(Int.self, forKey: .numLabels) {
      self.numLabels = labels
    } else if let map = try container.decodeIfPresent([String: String].self, forKey: .id2label) {
      self.numLabels = map.count
    }
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(hiddenSize, forKey: .hiddenSize)
    try container.encode(hiddenLayers, forKey: .hiddenLayers)
    try container.encode(intermediateSize, forKey: .intermediateSize)
    try container.encode(attentionHeads, forKey: .attentionHeads)
    try container.encode(rmsNormEps, forKey: .rmsNormEps)
    try container.encode(vocabularySize, forKey: .vocabularySize)
    try container.encode(kvHeads, forKey: .kvHeads)
    try container.encode(ropeTheta, forKey: .ropeTheta)
    try container.encode(headDim, forKey: .headDim)
    try container.encodeIfPresent(ropeScaling, forKey: .ropeScaling)
    try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
    try container.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
    try container.encode(useBidirectionalAttention, forKey: .useBidirectionalAttention)
    try container.encodeIfPresent(numLabels, forKey: .numLabels)
  }
}
