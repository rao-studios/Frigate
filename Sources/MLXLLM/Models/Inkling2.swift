//
//  Inkling2.swift
//  MLXLLM
//
//  WHAT: Thinking Machines' Inkling-2 looped text model (`inkling2_loop`; Inkling-2-Mini: 26.4B parameters, 3.4B active,
//        Apache-2.0), ported from mlx-lm's `inkling2` package (PR 2086). 24 layers, layers 6–17 run three passes,
//        so a token makes 48 layer visits (40 sliding, 8 global). Each layer reads the stream through depth attention
//        over block states (a block every 6 layers); a looped layer's later passes mix in messages from the pass
//        before. A 128-expert top-8 MoE per layer; final logits softcapped at 30 and cut to the unpadded vocabulary.
//        The vision tower and the MTP drafter are not loaded.
//  IN:   The checkpoint as Thinking Machines publish it (`model.llm.*`, experts NVFP4 / MXFP8 with FP8 scales): no
//        conversion. `sanitize` renames, as mlx-lm's checkpoint.py does, and quantizes the experts as they are.
//  PIN:  TWO CACHES A VISIT, ATTENTION THEN CONVOLUTION. `newCache` gives each visit [attention, ArraysCache(4)]: a
//        `KVCacheSimple` for a global visit, the sliding window's `ArraysCache(2)` for a sliding one
//        (Inkling2Attention's PIN), and the four short convolutions' last K−1 inputs. 32,768 bytes of KV a token
//        (8 global visits × 2 × 8 heads × 128 × bf16); about 165 MB of window and convolution state a sequence.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Inkling2TextConfiguration: Codable, Sendable {
    public struct Loop: Codable, Sendable {
        public var coreStart: Int
        public var coreEnd: Int
        public var passes: Int
        public var gateRank: Int

        enum CodingKeys: String, CodingKey {
            case coreStart = "core_start"
            case coreEnd = "core_end"
            case passes = "num_passes"
            case gateRank = "gate_rank"
        }
    }

    public var hiddenSize: Int
    public var hiddenLayers: Int
    public var vocabularySize: Int
    public var unpaddedVocabularySize: Int
    public var attentionHeads: Int
    public var keyValueHeads: Int
    public var headDim: Int
    public var swaAttentionHeads: Int
    public var swaKeyValueHeads: Int
    public var swaHeadDim: Int
    public var slidingWindow: Int
    public var localLayerIds: [Int]
    public var dRel: Int
    public var relExtent: Int
    public var logScalingFloor: Int
    public var logScalingAlpha: Float
    public var rmsNormEps: Float
    public var qkL2ClipThreshold: Float
    public var relL2ClipThreshold: Float
    public var sconvKernelSize: Int
    public var layersPerBlock: Int
    public var routedExperts: Int
    public var expertsPerToken: Int
    public var intermediateSize: Int
    public var mlpGateSoftcap: Float
    public var mlpUpSoftcap: Float
    public var routeScale: Float
    public var finalLogitSoftcapping: Float
    public var loop: Loop?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case unpaddedVocabularySize = "unpadded_vocab_size"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case swaAttentionHeads = "swa_num_attention_heads"
        case swaKeyValueHeads = "swa_num_key_value_heads"
        case swaHeadDim = "swa_head_dim"
        case slidingWindow = "sliding_window_size"
        case localLayerIds = "local_layer_ids"
        case dRel = "d_rel"
        case relExtent = "rel_extent"
        case logScalingFloor = "log_scaling_n_floor"
        case logScalingAlpha = "log_scaling_alpha"
        case rmsNormEps = "rms_norm_eps"
        case qkL2ClipThreshold = "qk_l2_clip_threshold"
        case relL2ClipThreshold = "rel_l2_clip_threshold"
        case sconvKernelSize = "sconv_kernel_size"
        case layersPerBlock = "n_layers_per_attn_res_block"
        case routedExperts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case intermediateSize = "intermediate_size"
        case mlpGateSoftcap = "mlp_gate_softcap"
        case mlpUpSoftcap = "mlp_up_softcap"
        case routeScale = "route_scale"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case loop = "loop_config"
    }

    /// The looped layers, empty without a loop.
    var core: Range<Int> { loop.map { $0.coreStart ..< $0.coreEnd } ?? 0 ..< 0 }

    /// The layer each visit runs, in order: the prelude, the core once per pass, the rest.
    var visits: [Int] {
        guard let loop else { return Array(0 ..< hiddenLayers) }
        return Array(0 ..< loop.coreStart)
            + Array(repeating: Array(core), count: loop.passes).flatMap { $0 }
            + Array(loop.coreEnd ..< hiddenLayers)
    }
}

public struct Inkling2Configuration: Codable, Sendable {
    public var modelType: String
    public var text: Inkling2TextConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
    }
}

final class Inkling2DecoderLayer: Module {
    let index: Int
    let sliding: Bool
    let isBlockBoundary: Bool
    let kernel: Int

    @ModuleInfo(key: "attention_input") var attentionInput: Inkling2BlockResidual?
    @ModuleInfo(key: "mlp_input") var mlpInput: Inkling2BlockResidual
    @ModuleInfo(key: "mlp") var mlp: Inkling2SparseMoe
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm
    @ModuleInfo(key: "post_attention_convolution") var postAttentionConv: Inkling2ShortConvolution
    @ModuleInfo(key: "post_mlp_convolution") var postMlpConv: Inkling2ShortConvolution
    @ModuleInfo(key: "q_message_gate") var qMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "k_message_gate") var kMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "v_message_gate") var vMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "mlp_message_gate") var mlpMessageGate: Inkling2MessageGate?
    @ModuleInfo(key: "self_attn") var selfAttention: Inkling2Attention
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm?
    @ModuleInfo(key: "q_input_layernorm") var qInputNorm: RMSNorm?
    @ModuleInfo(key: "k_input_layernorm") var kInputNorm: RMSNorm?
    @ModuleInfo(key: "v_input_layernorm") var vInputNorm: RMSNorm?

    init(_ config: Inkling2TextConfiguration, index: Int) {
        self.index = index
        sliding = config.localLayerIds.contains(index)
        isBlockBoundary = index % config.layersPerBlock == 0
        kernel = config.sconvKernelSize
        let d = config.hiddenSize, eps = config.rmsNormEps
        _attentionInput.wrappedValue = index > 0 ? Inkling2BlockResidual(dimensions: d, queries: sliding ? 1 : 3) : nil
        _mlpInput.wrappedValue = Inkling2BlockResidual(dimensions: d)
        _mlp.wrappedValue = Inkling2SparseMoe(config)
        _postAttentionNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        _postAttentionConv.wrappedValue = Inkling2ShortConvolution(channels: d, kernel: kernel)
        _postMlpConv.wrappedValue = Inkling2ShortConvolution(channels: d, kernel: kernel)
        if let loop = config.loop, config.core.contains(index) {
            _qMessageGate.wrappedValue = Inkling2MessageGate(dimensions: d, rank: loop.gateRank)
            _kMessageGate.wrappedValue = Inkling2MessageGate(dimensions: d, rank: loop.gateRank)
            _vMessageGate.wrappedValue = Inkling2MessageGate(dimensions: d, rank: loop.gateRank)
            _mlpMessageGate.wrappedValue = Inkling2MessageGate(dimensions: d, rank: loop.gateRank)
        }
        if sliding {
            _selfAttention.wrappedValue = Inkling2SlidingAttention(config)
            _inputNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        } else {
            _selfAttention.wrappedValue = Inkling2FullAttention(config)
            _qInputNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
            _kInputNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
            _vInputNorm.wrappedValue = RMSNorm(dimensions: d, eps: eps)
        }
    }

    private func mix(_ states: (MLXArray, MLXArray, MLXArray), _ messages: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        (qMessageGate!(states.0, messages[.ellipsis, 0, 0...]),
         kMessageGate!(states.1, messages[.ellipsis, 1, 0...]),
         vMessageGate!(states.2, messages[.ellipsis, 2, 0...]))
    }

    /// h [B,L,D] in the model's dtype; `blocks` the block states so far, [K,B,L,D] float32.
    func callAsFunction(
        _ h: MLXArray, blocks: MLXArray?, attentionCache: KVCache?, convCache: ArraysCache?,
        positions: Inkling2Positions, messages: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let dtype = h.dtype
        var conv: [MLXArray?] = (0 ..< 4).map { convCache?[$0] }
        var blocks = blocks
        let blockResidual = h
        let normedResidual = Inkling2Ops.rms(blockResidual.asType(.float32))
        let inputs = attentionInput.map { $0(blocks!, normedResidual) }                       // [B,L,Q,D]

        var states: (MLXArray, MLXArray, MLXArray)
        if sliding {
            let x = inputs?[.ellipsis, 0, 0...] ?? h
            if let messages {
                let mixed = mix((x, x, x), messages)
                states = (inputNorm!(mixed.0).asType(dtype), inputNorm!(mixed.1).asType(dtype),
                          inputNorm!(mixed.2).asType(dtype))
            } else {
                let s = inputNorm!(x).asType(dtype)
                states = (s, s, s)
            }
        } else {
            var qkv = inputs.map { ($0[.ellipsis, 0, 0...], $0[.ellipsis, 1, 0...], $0[.ellipsis, 2, 0...]) } ?? (h, h, h)
            if let messages { qkv = mix(qkv, messages) }
            states = (qInputNorm!(qkv.0).asType(dtype), kInputNorm!(qkv.1).asType(dtype), vInputNorm!(qkv.2).asType(dtype))
        }
        var x = selfAttention(states.0, states.1, states.2, cache: attentionCache, conv: &conv, positions: positions)
        (x, conv[2]) = postAttentionConv(x, state: conv[2])

        var y = x.asType(.float32)
        let residual: MLXArray
        if isBlockBoundary {
            residual = y
            let block = normedResidual.expandedDimensions(axis: 0)
            blocks = blocks.map { concatenated([$0, block]) } ?? block
        } else {
            y = blockResidual.asType(.float32) + y
            residual = y
        }
        y = mlpInput(blocks!, Inkling2Ops.rms(y))[.ellipsis, 0, 0...]
        if let messages { y = mlpMessageGate!(y, messages[.ellipsis, 3, 0...]) }
        let mlpOut = mlp(postAttentionNorm(y).asType(dtype))
        let z: MLXArray
        (z, conv[3]) = postMlpConv(mlpOut, state: conv[3])

        if let convCache {
            for i in 0 ..< 4 { convCache[i] = conv[i]! }
        }
        return (Inkling2Fused.residual([residual, z])[0], blocks)
    }
}

public final class Inkling2TextModel: Module {
    let config: Inkling2TextConfiguration
    let visits: [Int]

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_norm") var embedNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Inkling2DecoderLayer]
    @ModuleInfo(key: "final_input") var finalInput: Inkling2BlockResidual
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "message_input") var messageInput: Inkling2MessageInput?

    init(_ config: Inkling2TextConfiguration) {
        self.config = config
        visits = config.visits
        let d = config.hiddenSize
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabularySize, dimensions: d)
        _embedNorm.wrappedValue = RMSNorm(dimensions: d, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { Inkling2DecoderLayer(config, index: $0) }
        _finalInput.wrappedValue = Inkling2BlockResidual(dimensions: d)
        _norm.wrappedValue = RMSNorm(dimensions: d, eps: config.rmsNormEps)
        _messageInput.wrappedValue = config.loop == nil ? nil : Inkling2MessageInput(dimensions: d, loopedLayers: config.core.count)
    }

    /// The first global visit, whose cache says where the forward's tokens sit.
    var positionVisit: Int? { visits.firstIndex { !config.localLayerIds.contains($0) } }

    /// The first hidden state of `inputs`: the token embedding, normalized. A picture's features stand in for its
    /// body's tokens after this norm, with no norm of their own (NeMo's multimodal.py).
    func embed(_ inputs: MLXArray) -> MLXArray { embedNorm(embedTokens(inputs)) }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(embeddings: embed(inputs), cache: cache)
    }

    /// The forward from its first hidden state, as `embed` writes it.
    func callAsFunction(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embeddings
        let dtype = h.dtype
        let positions = Inkling2Positions(positionVisit.flatMap { cache?[2 * $0].ropeOffset })
        var visit = 0
        var blocks: MLXArray?

        func run(_ range: Range<Int>, bank: (MLXArray, MLXArray)?) {
            for (i, layer) in range.enumerated() {
                let messages = bank.map { messageInput!($0, layer: i) }
                (h, blocks) = layers[layer](
                    h, blocks: blocks, attentionCache: cache?[2 * visit],
                    convCache: cache.map { Inkling2Model.arrays($0[2 * visit + 1], holding: "a visit's convolution windows") },
                    positions: positions, messages: messages)
                visit += 1
            }
        }

        if let loop = config.loop {
            run(0 ..< loop.coreStart, bank: nil)
            let prelude = (h, blocks)
            let preludeBlocks = blocks?.dim(0) ?? 0
            for pass in 0 ..< loop.passes {
                var bank: (MLXArray, MLXArray)?
                if pass > 0 {
                    bank = (blocks![(preludeBlocks + 1)...], Inkling2Ops.rms(h.asType(.float32)))
                    (h, blocks) = prelude
                }
                run(config.core, bank: bank)
            }
            run(loop.coreEnd ..< config.hiddenLayers, bank: nil)
        } else {
            run(0 ..< config.hiddenLayers, bank: nil)
        }

        h = finalInput(blocks!, Inkling2Ops.rms(h.asType(.float32)))[.ellipsis, 0, 0...]
        return norm(h).asType(dtype)
    }
}

public final class Inkling2Inner: Module {
    @ModuleInfo(key: "language_model") var languageModel: Inkling2TextModel

    init(_ config: Inkling2TextConfiguration) {
        _languageModel.wrappedValue = Inkling2TextModel(config)
    }
}

public final class Inkling2Model: Module, LLMModel, DecodeSchedulingModel {
    public let modelType: String
    let configuration: Inkling2TextConfiguration
    @ModuleInfo(key: "model") var model: Inkling2Inner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: Inkling2Configuration) {
        modelType = config.modelType
        configuration = config.text
        _model.wrappedValue = Inkling2Inner(config.text)
        _lmHead.wrappedValue = Linear(config.text.hiddenSize, config.text.vocabularySize, bias: false)
    }

    public var vocabularySize: Int { configuration.unpaddedVocabularySize }

    /// A decoded token is ~4,100 dispatches over about 5 GB of weights: with MLX's 10 buffers in flight the encoder
    /// waits on the GPU for ~40% of its encoding, and the 50 MB limit commits a buffer every few projections. 64 in flight
    /// and no MB limit: 1k 42.1 → 48.6 tok/s, 32k 38.0 → 42.7 (release, M4 Max, 2026-11-21). More ops a buffer
    /// (100, 200) lost at 32k, so the ops limit stays the device's own.
    ///
    /// PIN: A DECODED TOKEN IS BOUND BY THE GPU, NOT BY THE THREAD THAT ENCODES IT. Of a 21.4 ms step at 1k the GPU is
    /// busy 21.3 (each command buffer's `GPUEndTime − GPUStartTime`, summed); the engine's thread works 15.3 and waits
    /// the rest inside `asyncEval`, on the buffers in flight (thread CPU time against the clock, release, 2026-11-23).
    /// The 0.02 ms once read as the GPU's wait is the read of the token sampled a step earlier. So work taken off the
    /// thread becomes waiting, and the rate moves only with what the GPU does a token. Each part left out in turn, at
    /// 1k: the attention less its kernel 7.2 ms (six projections a visit), the experts 5.1, the attention kernel and
    /// its ring writes 2.4, the unembedding 1.7, and about 5 of norms, casts, routing and depth attention, small
    /// dispatches each waiting on the one before.
    public var decodeScheduling: GPU.Scheduling { GPU.Scheduling(megabytes: 100_000, inFlight: 64) }

    /// Which of a decoded token's fused paths this model takes: the attention kernel (Inkling2DecodeAttention) and the
    /// convolution kernels (Inkling2DecodeConvolution). Both from the start; a test turns them off on the model it
    /// holds, to read the plain ops beside them.
    ///
    /// PIN: THE MODEL'S OWN, NEVER THE PROCESS'S. They were two statics, which one model's test turned off for every
    /// model loaded.
    func fuseDecode(attention: Bool, convolutions: Bool) {
        for module in modules() {
            switch module {
            case let module as Inkling2Attention: module.fusesDecode = attention
            case let module as Inkling2ShortConvolution: module.fusesDecode = convolutions
            case let module as Inkling2DynamicConvolution: module.fusesDecode = convolutions
            default: break
            }
        }
    }

    /// `layer` as the arrays `newCache` made for it: a sliding visit's window, a visit's convolution windows.
    ///
    /// PIN: NEVER `as?`. A visit handed another kind of cache took nil for it and decoded on without a word: zeros for
    /// its convolution windows, no window at all for its attention. A forward cannot throw, so a cache this model did
    /// not make stops here, saying what it is.
    static func arrays(_ layer: KVCache, holding what: StaticString) -> ArraysCache {
        guard let arrays = layer as? ArraysCache else {
            preconditionFailure(
                "Inkling2: \(what) is kept in an ArraysCache, as newCache(parameters:) makes it; this is a \(type(of: layer))")
        }
        return arrays
    }

    /// The vision tower is not loaded (this is the text model).
    public var excludedCheckpointNamespaces: [String] { ["model.visual"] }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(model.languageModel(inputs, cache: cache))
    }

    /// The logits of a forward whose first hidden state is `embeddings` [1, L, hidden] (`Inkling2OwnVision`).
    func callAsFunction(embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(model.languageModel(embeddings: embeddings, cache: cache))
    }

    private func logits(_ hidden: MLXArray) -> MLXArray {
        let cap = configuration.finalLogitSoftcapping
        let logits = tanh(lmHead(hidden).asType(.float32) / cap) * cap
        return logits[.ellipsis, ..<configuration.unpaddedVocabularySize]
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let local = Set(configuration.localLayerIds)
        return configuration.visits.flatMap { layer -> [KVCache] in
            [local.contains(layer) ? ArraysCache(size: 2) : KVCacheSimple(), ArraysCache(size: 4)]
        }
    }

    public var loraLayers: [Module] { model.languageModel.layers }

    // MARK: - The checkpoint

    /// mlx-lm checkpoint.py's renames, in order.
    static let renames: [(NSRegularExpression, String)] = [
        (#"^model\.llm\.unembed\."#, "lm_head."),
        (#"^model\.llm\.embed\."#, "model.language_model.embed_tokens."),
        (#"^model\.llm\."#, "model.language_model."),
        (#"\.attn_res\.mlp_queries\.(\d+)$"#, ".layers.$1.mlp_input.query"),
        (#"\.attn_res\.output_query$"#, ".final_input.query"),
        (#"\.attn\.wq_du\."#, ".self_attn.q_proj."),
        (#"\.attn\.wk_dv\."#, ".self_attn.k_proj."),
        (#"\.attn\.wv_dv\."#, ".self_attn.v_proj."),
        (#"\.attn\.wr_du\."#, ".self_attn.r_proj."),
        (#"\.attn\.wo_ud\."#, ".self_attn.o_proj."),
        (#"\.attn\.gate_attn\."#, ".self_attn.gate_proj."),
        (#"\.attn\.attn_sink_proj\."#, ".self_attn.sink_proj."),
        (#"\.controllers\.attn\."#, ".q_message_gate."),
        (#"\.controllers\.(k|v|mlp)\."#, ".$1_message_gate."),
        (#"_message_gate\.down_weight$"#, "_message_gate.down_proj.weight"),
        (#"_message_gate\.forget_weight$"#, "_message_gate.forget_gate_proj.weight"),
        (#"_message_gate\.gain_weight$"#, "_message_gate.message_gate_proj.weight"),
        (#"\.attn\."#, ".self_attn."),
        (#"\.attn_sconv\."#, ".post_attention_convolution."),
        (#"\.mlp_sconv\."#, ".post_mlp_convolution."),
        (#"sconv\.weight$"#, "sconv.conv1d.weight"),
        (#"convolution\.weight$"#, "convolution.conv1d.weight"),
        (#"sconv\.static_w$"#, "sconv.static_weight"),
        (#"sconv\.weight_proj$"#, "sconv.weight_proj.weight"),
        (#"\.mlp_norm\."#, ".post_attention_layernorm."),
        (#"\.attn_norm\."#, ".input_layernorm."),
        (#"\.(q|k|v)_input_norm\."#, ".$1_input_layernorm."),
        (#"\.mlp\.gate\.bias$"#, ".mlp.gate.e_score_correction_bias"),
    ].map { (try! NSRegularExpression(pattern: $0.0), $0.1) }

    static let attentionQuery = try! NSRegularExpression(pattern: #"^model\.llm\.attn_res\.attn_queries\.(\d+)$"#)
    static let expertTensor = try! NSRegularExpression(pattern: #"^(.+\.mlp\.experts)\.(w13|w2)_weight(\w*)$"#)

    static func rename(_ key: String) -> String {
        let whole = NSRange(key.startIndex..., in: key)
        if let m = attentionQuery.firstMatch(in: key, range: whole), let r = Range(m.range(at: 1), in: key),
           let i = Int(key[r]) {
            // Layer 0 has no attention input, so query i is for layer i + 1.
            return "model.language_model.layers.\(i + 1).attention_input.query"
        }
        var name = key
        for (pattern, template) in renames {
            name = pattern.stringByReplacingMatches(
                in: name, range: NSRange(name.startIndex..., in: name), withTemplate: template)
        }
        return name
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var amax: [String: MLXArray] = [:], denominator: [String: MLXArray] = [:]
        var messageQueries: [String: MLXArray] = [:]
        let castToFloat = [".mlp.gate.weight", ".forget_gate_proj.weight", ".message_gate_proj.weight"]
        for (key, value) in weights {
            // The MTP drafter's tensors carry no `model.llm.` prefix; the vision tower is excluded before this.
            guard key.hasPrefix("model.llm.") else { continue }
            let name = Self.rename(key)
            var v = value
            if name.hasSuffix("conv1d.weight") { v = v.swappedAxes(1, 2) }
            if name.hasSuffix("cross_pass_queries") || name.hasSuffix("cross_pass_kv_queries") {
                messageQueries[name.contains("_kv_") ? "kv" : "q"] = v
                continue
            }
            if castToFloat.contains(where: name.hasSuffix) {
                out[name] = v.asType(.float32)
                continue
            }
            let whole = NSRange(name.startIndex..., in: name)
            if let m = Self.expertTensor.firstMatch(in: name, range: whole),
               let prefix = Range(m.range(at: 1), in: name), let which = Range(m.range(at: 2), in: name),
               let suffix = Range(m.range(at: 3), in: name) {
                let module = "\(name[prefix]).\(name[which] == "w13" ? "gate_up_proj" : "down_proj")"
                switch name[suffix] {
                case "_amax": amax[module] = v
                case "_outer_scale_denominator": denominator[module] = v
                case "_scale": out["\(module).scales"] = v
                default: out["\(module).weight"] = v.view(dtype: .uint32)
                }
                continue
            }
            out[name] = v
        }

        if let q = messageQueries["q"], let kv = messageQueries["kv"] {
            // The checkpoint has [query, mlp] and [key, value] rows per looped layer.
            let d = q.dim(-1)
            let q2 = q.reshaped(-1, 2, d), kv2 = kv.reshaped(-1, 2, d)
            out["model.language_model.message_input.query"] = concatenated(
                [q2[0..., ..<1], kv2, q2[0..., 1...]], axis: 1)
        }

        // NVFP4's per-expert scale: amax · (6 · 448) / denominator.
        for (module, a) in amax {
            guard let den = denominator[module] else { continue }
            out["\(module).global_scale"] = a.asType(.float32) * (6.0 * 448.0 / den.asType(.float32))
        }

        // The experts as quantized as the tensors say: (bits, group) from the packed weight and the scales.
        let modes: [String: QuantizationMode] = ["4/16": .nvfp4, "4/32": .mxfp4, "8/32": .mxfp8]
        func format(_ module: String, inputs: Int) -> (Inkling2SwitchLinear.Format, Bool)? {
            guard let scales = out["\(module).scales"], scales.dtype == .uint8, let weight = out["\(module).weight"]
            else { return nil }
            let group = inputs / scales.dim(-1), bits = weight.dim(-1) * 32 / inputs
            guard let mode = modes["\(bits)/\(group)"] else { return nil }
            return (.init(bits: bits, groupSize: group, mode: mode), out["\(module).global_scale"] != nil)
        }
        for (i, layer) in model.languageModel.layers.enumerated() {
            let experts = layer.mlp.experts
            let path = "model.language_model.layers.\(i).mlp.experts"
            if let gu = format("\(path).gate_up_proj", inputs: experts.dimensions),
               let dn = format("\(path).down_proj", inputs: experts.hidden) {
                experts.adopt(gateUp: gu.0, gateUpGlobal: gu.1, down: dn.0, downGlobal: dn.1)
            }
        }
        return out
    }
}
