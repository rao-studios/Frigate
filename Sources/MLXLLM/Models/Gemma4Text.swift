//
//  Gemma4Text.swift
//  mlx-swift-lm (Frigate)
//
//  Decoder-only Gemma 4 text backbone. Adapted to Frigate's LanguageModel /
//  KVCache APIs. MoE and per-layer embeddings stay optional; the 12B unified
//  coder checkpoint uses neither.
//
//  Port of https://github.com/ml-explore/mlx-swift-lm Libraries/MLXLLM/Models/Gemma4Text.swift
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct Gemma4TextConfiguration: Decodable, Sendable {
    var modelType: String = "gemma4_text"
    var hiddenSize: Int = 1536
    var numHiddenLayers: Int = 35
    var intermediateSize: Int = 6144
    var numAttentionHeads: Int = 8
    var headDim: Int = 256
    var globalHeadDim: Int = 512
    var rmsNormEps: Float = 1e-6
    var vocabSize: Int = 262144
    var vocabSizePerLayerInput: Int = 262144
    var numKeyValueHeads: Int = 1
    var numGlobalKeyValueHeads: Int?
    var numKvSharedLayers: Int = 20
    var hiddenSizePerLayerInput: Int = 256
    var slidingWindow: Int = 512
    var slidingWindowPattern: Int = 5
    var maxPositionEmbeddings: Int = 131072
    var attentionKeqV: Bool = false
    var finalLogitSoftcapping: Float = 30.0
    var useDoubleWideMlp: Bool = true
    var enableMoEBlock: Bool = false
    var layerTypes: [String] = []
    var tieWordEmbeddings: Bool = true
    var slidingRopeTheta: Float = 10_000.0
    var fullRopeTheta: Float = 1_000_000.0
    var fullPartialRotaryFactor: Float = 1.0

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case vocabSizePerLayerInput = "vocab_size_per_layer_input"
        case numKeyValueHeads = "num_key_value_heads"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
        case numKvSharedLayers = "num_kv_shared_layers"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionKeqV = "attention_k_eq_v"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case enableMoEBlock = "enable_moe_block"
        case layerTypes = "layer_types"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
    }

    enum NestedKeys: String, CodingKey {
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let nested = try decoder.container(keyedBy: NestedKeys.self)
        let container =
            if nested.contains(.textConfig) {
                try nested.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)
            } else {
                try decoder.container(keyedBy: CodingKeys.self)
            }

        self.modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_text"
        let isUnified = modelType == "gemma4_unified_text" || modelType == "gemma4_unified"
        self.hiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? (isUnified ? 3840 : 1536)
        self.numHiddenLayers =
            try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? (isUnified ? 48 : 35)
        self.intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? (isUnified ? 15_360 : 6144)
        self.numAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads)
            ?? (isUnified ? 16 : 8)
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        self.globalHeadDim =
            try container.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabSize =
            try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 262144
        self.vocabSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .vocabSizePerLayerInput)
            ?? self.vocabSize
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? (isUnified ? 8 : 1)
        self.numGlobalKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)
            ?? (isUnified ? 1 : nil)
        self.numKvSharedLayers =
            try container.decodeIfPresent(Int.self, forKey: .numKvSharedLayers)
            ?? (isUnified ? 0 : 20)
        self.hiddenSizePerLayerInput =
            try container.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput)
            ?? (isUnified ? 0 : 256)
        self.slidingWindow =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
            ?? (isUnified ? 1024 : 512)
        self.slidingWindowPattern =
            try container.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? (isUnified ? 262144 : 131072)
        self.attentionKeqV =
            try container.decodeIfPresent(Bool.self, forKey: .attentionKeqV) ?? isUnified
        self.finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        self.useDoubleWideMlp =
            try container.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? !isUnified
        self.enableMoEBlock =
            try container.decodeIfPresent(Bool.self, forKey: .enableMoEBlock) ?? false
        if let decoded = try container.decodeIfPresent([String].self, forKey: .layerTypes) {
            self.layerTypes = decoded
        } else {
            var pattern = [String]()
            for i in 0 ..< slidingWindowPattern {
                pattern.append(i == slidingWindowPattern - 1 ? "full_attention" : "sliding_attention")
            }
            var types = [String]()
            while types.count < numHiddenLayers {
                types.append(contentsOf: pattern)
            }
            self.layerTypes = Array(types.prefix(numHiddenLayers))
        }
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        if let ropeParams = try? container.decode(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
        {
            if let sliding = ropeParams["sliding_attention"] {
                self.slidingRopeTheta = sliding["rope_theta"]?.asFloat() ?? 10_000.0
            }
            if let full = ropeParams["full_attention"] {
                self.fullRopeTheta = full["rope_theta"]?.asFloat() ?? 1_000_000.0
                self.fullPartialRotaryFactor =
                    full["partial_rotary_factor"]?.asFloat() ?? (isUnified ? 0.25 : 1.0)
            }
        } else if isUnified {
            self.fullPartialRotaryFactor = 0.25
        }
    }
}

// MARK: - Helpers

private class RMSNormNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

// MARK: - Attention

private class Gemma4Attention: Module {
    let layerIdx: Int
    let layerType: String
    let isSliding: Bool
    let effectiveHeadDim: Int
    let nHeads: Int
    let nKvHeads: Int
    let useKeqV: Bool
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear?
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
    @ModuleInfo(key: "v_norm") var vNorm: RMSNormNoScale
    let rope: OffsetLayer

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.layerIdx = layerIdx
        self.layerType = config.layerTypes[layerIdx]
        self.isSliding = layerType == "sliding_attention"
        self.effectiveHeadDim = isSliding ? config.headDim : config.globalHeadDim
        let dim = config.hiddenSize
        self.nHeads = config.numAttentionHeads
        self.useKeqV = config.attentionKeqV && !isSliding
        if useKeqV, let globalKvHeads = config.numGlobalKeyValueHeads {
            self.nKvHeads = globalKvHeads
        } else {
            self.nKvHeads = config.numKeyValueHeads
        }
        self.scale = 1.0

        self._qProj.wrappedValue = Linear(dim, nHeads * effectiveHeadDim, bias: false)
        let firstKvShared = config.numHiddenLayers - config.numKvSharedLayers
        let isKvShared = layerIdx >= firstKvShared && firstKvShared > 0
        if !isKvShared {
            self._kProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
            if !useKeqV {
                self._vProj.wrappedValue = Linear(dim, nKvHeads * effectiveHeadDim, bias: false)
            }
        }
        self._oProj.wrappedValue = Linear(nHeads * effectiveHeadDim, dim, bias: false)
        self._qNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        if !isKvShared {
            self._kNorm.wrappedValue = RMSNorm(dimensions: effectiveHeadDim, eps: config.rmsNormEps)
        }
        self._vNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)

        if isSliding {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.slidingRopeTheta, traditional: false,
                scalingConfig: nil, maxPositionEmbeddings: nil)
        } else {
            self.rope = initializeRope(
                dims: effectiveHeadDim, base: config.fullRopeTheta, traditional: false,
                scalingConfig: [
                    "type": .string("proportional"),
                    "partial_rotary_factor": .float(config.fullPartialRotaryFactor),
                ],
                maxPositionEmbeddings: nil)
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        var queries = qProj(x).reshaped(B, L, nHeads, effectiveHeadDim)
        queries = qNorm(queries)
        queries = queries.transposed(0, 2, 1, 3)

        guard let kProj, let kNorm else {
            fatalError("Gemma4Attention layer \(layerIdx) has no k_proj; KV-shared layers are not on this path.")
        }
        let kRaw = kProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim)
        var keys = kNorm(kRaw).transposed(0, 2, 1, 3)
        var values: MLXArray
        if let vProj {
            values = vNorm(vProj(x).reshaped(B, L, nKvHeads, effectiveHeadDim))
                .transposed(0, 2, 1, 3)
        } else {
            values = vNorm(kRaw).transposed(0, 2, 1, 3)
        }

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        var adjustedMask = mask
        if case .array(let maskArray) = mask {
            let keysSeqLen = keys.dim(-2)
            if maskArray.dim(-1) != keysSeqLen {
                adjustedMask = .array(maskArray[.ellipsis, 0 ..< keysSeqLen])
            }
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: adjustedMask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return oProj(output)
    }
}

// MARK: - MLP

private class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        let firstKvShared = config.numHiddenLayers - config.numKvSharedLayers
        let isKvShared = layerIdx >= firstKvShared && firstKvShared > 0
        let useDoubleWide = config.useDoubleWideMlp && isKvShared
        let intermediateSize = config.intermediateSize * (useDoubleWide ? 2 : 1)
        self._gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder layer

private class Gemma4DecoderLayer: Module {
    let layerType: String

    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4TextConfiguration, layerIdx: Int) {
        self.layerType = config.layerTypes[layerIdx]
        self._selfAttn.wrappedValue = Gemma4Attention(config, layerIdx: layerIdx)
        self._mlp.wrappedValue = Gemma4MLP(config, layerIdx: layerIdx)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._preFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postFeedforwardLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._layerScalar.wrappedValue = MLXArray.ones([1], dtype: .float16)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let residual = x
        let attnOut = selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        var out = residual + postAttentionLayernorm(attnOut)
        let residual2 = out
        out = mlp(preFeedforwardLayernorm(out))
        return (residual2 + postFeedforwardLayernorm(out)) * layerScalar
    }
}

// MARK: - Inner model

fileprivate class Gemma4TextModelInner: Module {
    let config: Gemma4TextConfiguration
    let embedScale: Float

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.embedScale = Float(config.hiddenSize).squareRoot()
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
            Gemma4DecoderLayer(config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs) * embedScale
        var fullCache: [KVCache?]
        if let cache {
            fullCache = cache.map { Optional($0) }
            while fullCache.count < config.numHiddenLayers { fullCache.append(nil) }
        } else {
            fullCache = Array(repeating: nil, count: config.numHiddenLayers)
        }

        var maskByType = [String: MLXFast.ScaledDotProductAttentionMaskMode]()
        for (i, layer) in layers.enumerated() {
            if maskByType[layer.layerType] == nil {
                if layer.layerType == "sliding_attention" {
                    maskByType[layer.layerType] = createAttentionMask(
                        h: h, cache: fullCache[i], windowSize: config.slidingWindow)
                } else {
                    maskByType[layer.layerType] = createAttentionMask(h: h, cache: fullCache[i])
                }
            }
        }

        for (idx, layer) in layers.enumerated() {
            h = layer(h, mask: maskByType[layer.layerType] ?? .none, cache: fullCache[idx])
        }
        return norm(h)
    }
}

// MARK: - Public text model

public class Gemma4TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    fileprivate let config: Gemma4TextConfiguration
    @ModuleInfo fileprivate var model: Gemma4TextModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Gemma4TextConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabSize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self._model.wrappedValue = Gemma4TextModelInner(config)
        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        if let lmHead {
            out = lmHead(out)
        } else {
            out = model.embedTokens.asLinear(out)
        }
        out = tanh(out / config.finalLogitSoftcapping) * config.finalLogitSoftcapping
        return out
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var processed = Gemma4Model.droppingMultimodalKeys(weights)
        if config.tieWordEmbeddings {
            processed["lm_head.weight"] = nil
        }
        let firstKvShared = config.numHiddenLayers - config.numKvSharedLayers
        var sanitized = [String: MLXArray]()
        for (key, value) in processed {
            if key.contains("self_attn.rotary_emb")
                || key.contains("input_max") || key.contains("input_min")
                || key.contains("output_max") || key.contains("output_min")
            {
                continue
            }
            if firstKvShared > 0,
                key.contains("self_attn.k_proj")
                    || key.contains("self_attn.v_proj")
                    || key.contains("self_attn.k_norm"),
                let layerIdx = Self.decoderLayerIndex(in: key),
                layerIdx >= firstKvShared
            {
                continue
            }
            sanitized[key] = value
        }
        return sanitized
    }

    private static func decoderLayerIndex(in key: String) -> Int? {
        guard let range = key.range(of: "layers.") else { return nil }
        let digits = key[range.upperBound...].prefix { $0.isNumber }
        return Int(digits)
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.numHiddenLayers).map { i in
            if config.layerTypes[i] == "full_attention" {
                let cache = KVCacheSimple()
                cache.step = 1024
                return cache
            }
            return RotatingKVCache(maxSize: config.slidingWindow, keep: 0)
        }
    }
}

extension Gemma4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.map { $0.selfAttn }
    }
}
