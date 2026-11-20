//
//  Mamba3.swift
//  MLXLLM
//
//  WHAT: Mamba-3 MIMO (state-spaces/mamba, 2026-07-28 release), ported from Braid's BraidKit (Mamba3Block.swift,
//        Mamba3Config.swift): the `mamba_ssm` config layout with Braid's `braid_*` keys, a layer of x + mixer(norm(x))
//        then + mlp(norm2(·)), logits tied to the embedding. Registered as `mamba3` (LLMModelFactory); Braid publishes
//        the commons as `rao-studios/braid-commons-mamba3-1.5b`.
//        `read(_:cache:tap:)` reads text into the recurrent state without the vocabulary projection, the one use
//        Craft makes of it: a conversation read into a fixed-size state, and a key per span.
//  PIN:  THE STATE PER LAYER IS ONE `ArraysCache(size: 4)`: [0] ssm [B,H,P,N], [1] angle [B,H,S] modulo 2π, [2] k
//        [B,R,H,N] (the last position's rotated, biased keys), [3] v [B,H,P] (its x). All four are written by every
//        forward and replaced, never written in place: a snapshot holds the arrays it was taken with. Each is a buffer
//        of its own (`owned`): a slice shares its source's buffer, and the state's slices of a read's k, x and chunk
//        states kept every layer's whole tensors alive, 5.8 GB after a 1,024-token read against the state's 54 MB
//        (Mamba3LiveTests.theStateHoldsOnlyItself, 2026-11-19).
//  PIN:  WEIGHTS MAY BE NARROWER THAN THE ARITHMETIC. The projections run in their weights' dtype (BF16 when loaded by
//        `Mamba3Loader` with `.bfloat16`); the residual stream, the norms and the scan run in float32
//        (`residual_in_fp32`, and SSD3's PIN). With float32 weights every cast is a no-op and the arithmetic is Braid's.
//  PIN:  A's ACTIVATION HAS NO DEFAULT. A config without `braid_a_activation` is refused, as Braid refuses it: REAPER.md
//        D30/D40 pin it per checkpoint (heavy-tail for the 1.5b).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Mamba3Configuration: Codable, Sendable, ModelConfigurationValidating {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var vocabSize: Int
    public var dIntermediate: Int
    public var stateSize: Int
    public var expand: Int
    public var headDim: Int
    public var nGroups: Int
    public var ropeFraction: Float
    public var aFloor: Float
    public var mimoRank: Int
    public var tieWordEmbeddings: Bool
    public var isMimo: Bool
    public var isOutprojNorm: Bool
    public var rmsNorm: Bool
    public var attentionLayers: [Int]
    /// Braid: the training kernel's chunk in positions (64 at rank 4: 256 rows of the score matrix), kept as the file
    /// says. Reading chunks by `readChunk`.
    public var chunkSize: Int
    /// The positions the chunked scan takes at once when reading: a split of the work only, the scan is the same at
    /// any length (Mamba3Tests hold it at 8, the live oracle at this). The score work grows with the chunk (L·c·R²·N)
    /// while the hand-off between chunks is one product, so 32 read fastest on an M4 Max (2026-11-19, 4 × 1,024 tokens
    /// at BF16: 2,530 tok/s against 2,350 at 64 and 2,449 at 16). Not coded: no checkpoint carries it.
    public var readChunk = 32
    /// Braid: the first block of the commons' tail; blocks 0..<cut are a Thread's own.
    public var cut: Int
    /// Braid: the residual tap (after how many blocks) a key reads.
    public var tapAfterBlocks: Int
    /// Braid: A's activation, `heavy_tail` or `softplus`.
    public var aActivation: String?

    public var layerNormEpsilon: Float { 1e-5 }
    /// The inner width, `expand · d_model`.
    public var dInner: Int { expand * hiddenSize }
    public var numHeads: Int { dInner / headDim }
    /// The gated MLP's hidden width: `d_intermediate` rounded up to a multiple of 128, as `GatedMLP` does.
    public var mlpHidden: Int { (dIntermediate + 127) / 128 * 128 }
    /// Rotary angles per head: half of `d_state · rope_fraction`, rounded down to an even split.
    public var rotaryAngles: Int {
        var split = Int(Float(stateSize) * ropeFraction)
        split -= split % 2
        return split / 2
    }
    /// The in_proj's output: z, x, B, C (rank-wide), dt, A, trap, angles.
    public var projection: Int { 2 * dInner + 2 * stateSize * nGroups * mimoRank + 3 * numHeads + rotaryAngles }
    /// Bytes of one stream's state over every layer, in float32.
    public var stateBytes: Int {
        4 * numHiddenLayers * (numHeads * headDim * stateSize + numHeads * rotaryAngles + mimoRank * numHeads * stateSize + numHeads * headDim)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type", dModel = "d_model", nLayer = "n_layer", vocabSize = "vocab_size"
        case dIntermediate = "d_intermediate", ssmCfg = "ssm_cfg", tieEmbeddings = "tie_embeddings", rmsNorm = "rms_norm"
        case padVocabSizeMultiple = "pad_vocab_size_multiple", attnLayerIdx = "attn_layer_idx"
        case chunkSize = "braid_chunk_size", cut = "braid_cut", tapAfterBlocks = "braid_tap_after_blocks"
        case aActivation = "braid_a_activation"
    }

    enum SSMKeys: String, CodingKey {
        case layer, dState = "d_state", expand, headdim, ngroups, ropeFraction = "rope_fraction", aFloor = "A_floor"
        case isMimo = "is_mimo", mimoRank = "mimo_rank", isOutprojNorm = "is_outproj_norm"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let s = try c.nestedContainer(keyedBy: SSMKeys.self, forKey: .ssmCfg)
        let layer = try s.decodeIfPresent(String.self, forKey: .layer)
        guard layer == "Mamba3" else {
            throw DecodingError.dataCorruptedError(forKey: .layer, in: s, debugDescription: "ssm_cfg.layer is \(layer ?? "absent"), not Mamba3")
        }
        hiddenSize = try c.decode(Int.self, forKey: .dModel)
        numHiddenLayers = try c.decode(Int.self, forKey: .nLayer)
        let pad = try c.decodeIfPresent(Int.self, forKey: .padVocabSizeMultiple) ?? 8
        let declared = try c.decode(Int.self, forKey: .vocabSize)
        vocabSize = declared % pad == 0 ? declared : declared + pad - declared % pad
        dIntermediate = try c.decode(Int.self, forKey: .dIntermediate)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieEmbeddings) ?? true
        rmsNorm = try c.decodeIfPresent(Bool.self, forKey: .rmsNorm) ?? true
        attentionLayers = try c.decodeIfPresent([Int].self, forKey: .attnLayerIdx) ?? []
        stateSize = try s.decodeIfPresent(Int.self, forKey: .dState) ?? 128
        expand = try s.decodeIfPresent(Int.self, forKey: .expand) ?? 2
        headDim = try s.decodeIfPresent(Int.self, forKey: .headdim) ?? 64
        nGroups = try s.decodeIfPresent(Int.self, forKey: .ngroups) ?? 1
        ropeFraction = try s.decodeIfPresent(Float.self, forKey: .ropeFraction) ?? 0.5
        aFloor = try s.decodeIfPresent(Float.self, forKey: .aFloor) ?? 1e-4
        isMimo = try s.decodeIfPresent(Bool.self, forKey: .isMimo) ?? false
        mimoRank = try s.decodeIfPresent(Int.self, forKey: .mimoRank) ?? 4
        isOutprojNorm = try s.decodeIfPresent(Bool.self, forKey: .isOutprojNorm) ?? false
        chunkSize = try c.decodeIfPresent(Int.self, forKey: .chunkSize) ?? 64
        cut = try c.decodeIfPresent(Int.self, forKey: .cut) ?? numHiddenLayers / 2
        tapAfterBlocks = try c.decodeIfPresent(Int.self, forKey: .tapAfterBlocks) ?? numHiddenLayers / 2
        aActivation = try c.decodeIfPresent(String.self, forKey: .aActivation)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode("mamba3", forKey: .modelType)
        try c.encode(hiddenSize, forKey: .dModel)
        try c.encode(numHiddenLayers, forKey: .nLayer)
        try c.encode(vocabSize, forKey: .vocabSize)
        try c.encode(dIntermediate, forKey: .dIntermediate)
        try c.encode(tieWordEmbeddings, forKey: .tieEmbeddings)
        try c.encode(rmsNorm, forKey: .rmsNorm)
        try c.encode(attentionLayers, forKey: .attnLayerIdx)
        var s = c.nestedContainer(keyedBy: SSMKeys.self, forKey: .ssmCfg)
        try s.encode("Mamba3", forKey: .layer)
        try s.encode(stateSize, forKey: .dState)
        try s.encode(expand, forKey: .expand)
        try s.encode(headDim, forKey: .headdim)
        try s.encode(nGroups, forKey: .ngroups)
        try s.encode(ropeFraction, forKey: .ropeFraction)
        try s.encode(aFloor, forKey: .aFloor)
        try s.encode(isMimo, forKey: .isMimo)
        try s.encode(mimoRank, forKey: .mimoRank)
        try s.encode(isOutprojNorm, forKey: .isOutprojNorm)
        try c.encode(chunkSize, forKey: .chunkSize)
        try c.encode(cut, forKey: .cut)
        try c.encode(tapAfterBlocks, forKey: .tapAfterBlocks)
        try c.encodeIfPresent(aActivation, forKey: .aActivation)
    }

    public enum Refusal: Error, CustomStringConvertible, Equatable {
        case unsupported(String)
        public var description: String {
            switch self { case .unsupported(let why): return "mamba3: \(why)" }
        }
    }

    /// What the port reads: MIMO without the out-proj norm, RMSNorm, tied embeddings, no attention layers, a pinned
    /// activation, a rotary fraction the kernels support.
    public func validateModelConfiguration() throws {
        guard isMimo else { throw Refusal.unsupported("only MIMO checkpoints are read; this one is SISO") }
        guard !isOutprojNorm else { throw Refusal.unsupported("checkpoints with the out-proj norm are not read") }
        guard rmsNorm else { throw Refusal.unsupported("only RMSNorm checkpoints are read") }
        guard attentionLayers.isEmpty else { throw Refusal.unsupported("the checkpoint has attention layers") }
        guard tieWordEmbeddings else { throw Refusal.unsupported("the embeddings are not tied") }
        guard dInner % headDim == 0, numHeads % nGroups == 0, mimoRank >= 1 else {
            throw Refusal.unsupported("d_inner \(dInner), head dim \(headDim), groups \(nGroups), rank \(mimoRank) do not fit")
        }
        guard ropeFraction == 0.5 || ropeFraction == 1.0, rotaryAngles > 0, rotaryAngles <= stateSize / 2 else {
            throw Refusal.unsupported("rope fraction \(ropeFraction) is not 0.5 or 1.0")
        }
        guard let aActivation, ["heavy_tail", "softplus"].contains(aActivation) else {
            throw Refusal.unsupported("braid_a_activation is \(aActivation ?? "absent"); it is pinned per checkpoint")
        }
        guard chunkSize >= 1, readChunk >= 1, (1 ... numHiddenLayers).contains(tapAfterBlocks) else {
            throw Refusal.unsupported("chunk \(chunkSize), read chunk \(readChunk) or tap \(tapAfterBlocks) is out of range")
        }
    }
}

final class Mamba3Mixer: Module {
    let config: Mamba3Configuration

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "B_bias") var bBias: MLXArray
    @ParameterInfo(key: "C_bias") var cBias: MLXArray
    @ModuleInfo(key: "B_norm") var bNorm: RMSNorm
    @ModuleInfo(key: "C_norm") var cNorm: RMSNorm
    @ParameterInfo(key: "mimo_x") var mimoX: MLXArray
    @ParameterInfo(key: "mimo_z") var mimoZ: MLXArray
    @ParameterInfo(key: "mimo_o") var mimoO: MLXArray
    @ParameterInfo(key: "D") var dSkip: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: Mamba3Configuration) {
        self.config = config
        let (h, r, n, p) = (config.numHeads, config.mimoRank, config.stateSize, config.headDim)
        _inProj.wrappedValue = Linear(config.hiddenSize, config.projection, bias: false)
        _dtBias.wrappedValue = MLXArray.zeros([h])
        _bBias.wrappedValue = MLXArray.ones([h, r, n])
        _cBias.wrappedValue = MLXArray.ones([h, r, n])
        _bNorm.wrappedValue = RMSNorm(dimensions: n, eps: 1e-5)
        _cNorm.wrappedValue = RMSNorm(dimensions: n, eps: 1e-5)
        _mimoX.wrappedValue = MLXArray.ones([h, r, p]) / Float(r)
        _mimoZ.wrappedValue = MLXArray.ones([h, r, p])
        _mimoO.wrappedValue = MLXArray.ones([h, r, p]) / Float(r)
        _dSkip.wrappedValue = MLXArray.ones([h])
        _outProj.wrappedValue = Linear(config.dInner, config.hiddenSize, bias: false)
        super.init()
    }

    /// The preprocessed inputs at every position of h [B,L,D], in float32.
    private struct Inputs {
        var z: MLXArray, x: MLXArray, bm: MLXArray, cm: MLXArray, adt: MLXArray, dt: MLXArray, lam: MLXArray, increments: MLXArray
    }

    private func inputs(_ h: MLXArray) -> Inputs {
        let c = config
        let (b, l) = (h.dim(0), h.dim(1))
        let (hh, r, g, n, p) = (c.numHeads, c.mimoRank, c.nGroups, c.stateSize, c.headDim)
        let bc = n * g * r
        let projected = inProj(h.asType(inProj.weight.dtype)).asType(.float32)
        let parts = MLX.split(projected, indices: [c.dInner, 2 * c.dInner, 2 * c.dInner + bc, 2 * c.dInner + 2 * bc,
                                                   2 * c.dInner + 2 * bc + hh, 2 * c.dInner + 2 * bc + 2 * hh,
                                                   2 * c.dInner + 2 * bc + 3 * hh], axis: -1)
        let z = parts[0].reshaped([b, l, hh, p]), x = parts[1].reshaped([b, l, hh, p])
        // B and C: [B,L,R,G,N], normed over N; each scan spreads the groups over the heads.
        let bm = bNorm(parts[2].reshaped([b, l, r, g, n]))
        let cm = cNorm(parts[3].reshaped([b, l, r, g, n]))
        let a = minimum(-activation(parts[5]), MLXArray(-c.aFloor))
        let dt = softplus(parts[4] + dtBias)
        let lam = sigmoid(parts[6])
        let increments = tanh(parts[7]).expandedDimensions(axis: 2) * dt.expandedDimensions(axis: 3) * Float(Double.pi)
        return Inputs(z: z, x: x, bm: bm, cm: cm, adt: a * dt, dt: dt, lam: lam, increments: increments)
    }

    private func activation(_ x: MLXArray) -> MLXArray {
        if config.aActivation == "softplus" { return softplus(x) }
        // f(x) = 1 + x for x ≥ 0, 1 / (1 − x) for x < 0 (mamba_ssm's heavy_tail_activation).
        return maximum(x, MLXArray(Float(0))) + 1 / (1 - minimum(x, MLXArray(Float(0))))
    }

    /// The biases [H,R,N] as [R,H,N], the layout q and k carry one token at a time.
    private var biases: (q: MLXArray, k: MLXArray) { (cBias.transposed(1, 0, 2), bBias.transposed(1, 0, 2)) }

    /// B or C at one position [B,R,G,N] spread over the heads: [B,R,H,N].
    private func heads(_ t: MLXArray) -> MLXArray { repeated(t, count: config.numHeads / config.nGroups, axis: 2) }

    /// B or C [B,L,R,G,N] plus its bias [H,R,N], rotated by `tables` (`SSD3.rotationTables` over [B,H,L,1,S]), head-major:
    /// [B,H,L,R,N]. The groups spread over the heads inside the one fused pass that writes it.
    private func headMajor(_ t: MLXArray, bias: MLXArray, tables: (c: MLXArray, s: MLXArray)) -> MLXArray {
        let (b, l, r, g, n) = (t.dim(0), t.dim(1), t.dim(2), t.dim(3), t.dim(4))
        let (heads, perGroup, half) = (config.numHeads, config.numHeads / g, n / 2)
        let grouped = t.transposed(0, 3, 1, 2, 4).reshaped([b, g, 1, l, r, 2, half])
        let biased = bias.reshaped([1, g, perGroup, 1, r, 2, half])
        return Self.rotation([grouped, biased, SSD3.swapHalves(grouped), SSD3.swapHalves(biased),
                              tables.c.reshaped([b, g, perGroup, l, 1, 1, half]),
                              tables.s.reshaped([b, g, perGroup, l, 1, 2, half])])[0]
            .reshaped([b, heads, l, r, n])
    }

    /// (t + bias) ⊙ C + (swap(t) + swap(bias)) ⊙ S, fused; primitive ops only (SSD3.gate's PIN).
    private static let rotation: @Sendable ([MLXArray]) -> [MLXArray] = compile { a in
        [(a[0] + a[1]) * a[4] + (a[2] + a[3]) * a[5]]
    }

    /// `x` in a buffer of its own, for the state (see PIN): an element-wise op writes a fresh one, since MLX donates an
    /// input's buffer only to an output about its size, and a slice's buffer is its whole source's.
    static func owned(_ x: MLXArray) -> MLXArray { x * 1 }

    private func project(_ y: MLXArray) -> MLXArray { outProj(y.asType(outProj.weight.dtype)).asType(.float32) }

    /// h [B,L,D] float32; the layer's state in `cache` (empty = zeros), replaced by the state after the last position.
    func callAsFunction(_ h: MLXArray, cache: ArraysCache?) -> MLXArray {
        let (b, l) = (h.dim(0), h.dim(1))
        let m = inputs(h)
        let ssm = cache?[0], angle0 = cache?[1], kPrev = cache?[2], xPrev = cache?[3]
        if l == 1, let ssm, let angle0, let kPrev, let xPrev {
            let angle = SSD3.wrap(angle0 + m.increments[0..., 0])                                   // [B,H,S]
            let cosA = cos(angle).expandedDimensions(axis: 1), sinA = sin(angle).expandedDimensions(axis: 1)
            let q = SSD3.rotate(heads(m.cm[0..., 0]) + biases.q, cos: cosA, sin: sinA)               // [B,R,H,N]
            let k = SSD3.rotate(heads(m.bm[0..., 0]) + biases.k, cos: cosA, sin: sinA)
            let x = m.x[0..., 0]
            let (y, next) = SSD3.step(q: q, k: k, x: x, adt: m.adt[0..., 0], dt: m.dt[0..., 0], lam: m.lam[0..., 0],
                                      mimoX: mimoX, mimoZ: mimoZ, mimoO: mimoO, d: dSkip, z: m.z[0..., 0],
                                      ssm: ssm, kPrev: kPrev, xPrev: xPrev)
            cache?[0] = next; cache?[1] = angle; cache?[2] = k; cache?[3] = Self.owned(x)
            return project(y.reshaped([b, 1, config.dInner]))
        }
        let angle = SSD3.angles(increments: m.increments, angle0: angle0)                         // [B,L,H,S]
        let angleH = angle.transposed(0, 2, 1, 3).expandedDimensions(axis: 3)                   // [B,H,L,1,S]
        let tables = SSD3.rotationTables(cos: cos(angleH), sin: sin(angleH), half: config.stateSize / 2)
        let q = headMajor(m.cm, bias: cBias, tables: tables)                                     // [B,H,L,R,N]
        let k = headMajor(m.bm, bias: bBias, tables: tables)
        let (y, next) = SSD3.chunked(q: q, k: k, x: m.x, adt: m.adt, dt: m.dt, lam: m.lam, mimoX: mimoX, mimoZ: mimoZ,
                                     mimoO: mimoO, d: dSkip, z: m.z, ssm: ssm, kPrev: kPrev, xPrev: xPrev,
                                     chunk: config.readChunk)
        cache?[0] = Self.owned(next)
        cache?[1] = SSD3.wrap(angle[0..., l - 1])
        cache?[2] = Self.owned(k[0..., 0..., l - 1].transposed(0, 2, 1, 3))                      // [B,R,H,N]
        cache?[3] = Self.owned(m.x[0..., l - 1])
        return project(y.reshaped([b, l, config.dInner]))
    }
}

/// `mamba_ssm`'s `GatedMLP`: fc1 to twice the hidden width, x · silu(gate), fc2 back.
final class Mamba3MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: Mamba3Configuration) {
        _fc1.wrappedValue = Linear(config.hiddenSize, 2 * config.mlpHidden, bias: false)
        _fc2.wrappedValue = Linear(config.mlpHidden, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(fc1(x.asType(fc1.weight.dtype)), parts: 2, axis: -1)
        return fc2(parts[0] * silu(parts[1])).asType(.float32)
    }
}

final class Mamba3Block: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mixer") var mixer: Mamba3Mixer
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Mamba3MLP

    init(_ config: Mamba3Configuration) {
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        _mixer.wrappedValue = Mamba3Mixer(config)
        _norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        _mlp.wrappedValue = Mamba3MLP(config)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let r = x + mixer(norm(x), cache: cache)
        return r + mlp(norm2(r))
    }
}

public final class Mamba3Backbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: Embedding
    let layers: [Mamba3Block]
    @ModuleInfo(key: "norm_f") var normF: RMSNorm

    init(_ config: Mamba3Configuration) {
        _embeddings.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        layers = (0 ..< config.numHiddenLayers).map { _ in Mamba3Block(config) }
        _normF.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        super.init()
    }

    /// The residual stream after the embedding, in float32 (`residual_in_fp32`).
    func embed(_ tokens: MLXArray) -> MLXArray { embeddings(tokens).asType(.float32) }
}

/// What `Mamba3Model.read` leaves behind besides the state it advanced.
public struct Mamba3Reading {
    /// Σ over the positions read of the residual after `tap` blocks, float32 [B, D]. A sum, so a span read in pieces
    /// pools by adding the pieces' sums and counts.
    public let tappedSum: MLXArray
    /// The final normed hidden state at the last position read, float32 [B, D].
    public let last: MLXArray
    /// Positions read.
    public let count: Int
    /// With a stride: at every `stride`-th position of the piece and at its last, the residual after `tap` blocks and
    /// the final normed hidden state side by side, float32 [B, ⌈count / stride⌉, 2·D]. What a bridge projects into a
    /// core model's embeddings: each row is the stream so far, as of that position.
    public var strided: MLXArray? = nil

    /// The positions a strided read keeps out of `count`: every `stride`-th, and the last.
    public static func stridePositions(count: Int, stride: Int) -> [Int] {
        precondition(count > 0 && stride > 0, "a strided read keeps positions of a piece")
        var positions = Array(Swift.stride(from: stride - 1, to: count, by: stride))
        if positions.last != count - 1 { positions.append(count - 1) }
        return positions
    }
}

public final class Mamba3Model: Module, LLMModel {
    public let config: Mamba3Configuration
    @ModuleInfo(key: "backbone") var backbone: Mamba3Backbone

    public init(_ config: Mamba3Configuration) {
        self.config = config
        _backbone.wrappedValue = Mamba3Backbone(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = backbone.embed(inputs)
        for (i, layer) in backbone.layers.enumerated() {
            h = layer(h, cache: cache?[i] as? ArraysCache)
        }
        let hidden = backbone.normF(h)
        return backbone.embeddings.asLinear(hidden.asType(backbone.embeddings.weight.dtype))
    }

    /// Reads `tokens` [B,L] into `cache` (one `newCache` per stream, carried across reads) without the vocabulary
    /// projection, which at the 1.5b is 263M of its parameters. Each layer is evaluated as it goes, so the graph never
    /// grows with the piece.
    /// With `stride`, the reading also keeps `strided` (`Mamba3Reading.stridePositions`).
    public func read(_ tokens: MLXArray, cache: [KVCache], tap: Int? = nil, stride: Int? = nil) -> Mamba3Reading {
        let tap = tap ?? config.tapAfterBlocks
        precondition((1 ... config.numHiddenLayers).contains(tap), "tap \(tap) is outside 1...\(config.numHiddenLayers)")
        precondition(cache.count == config.numHiddenLayers, "one cache per layer")
        let positions = stride.map { MLXArray(Mamba3Reading.stridePositions(count: tokens.dim(1), stride: $0).map(Int32.init)) }
        var h = backbone.embed(tokens)
        var tapped: MLXArray?
        var tappedRows: MLXArray?
        for (i, layer) in backbone.layers.enumerated() {
            let layerCache = cache[i] as? ArraysCache
            h = layer(h, cache: layerCache)
            if i + 1 == tap {
                tapped = h.sum(axis: 1)
                tappedRows = positions.map { h.take($0, axis: 1) }
            }
            eval([h] + (layerCache?.state ?? []) + [tapped, tappedRows].compactMap { $0 })
        }
        let last = backbone.normF(h[0..., (h.dim(1) - 1)...]).squeezed(axis: 1)
        var reading = Mamba3Reading(tappedSum: tapped!, last: last, count: tokens.dim(1))
        if let positions, let tappedRows {
            reading.strided = concatenated([tappedRows, backbone.normF(h.take(positions, axis: 1))], axis: -1)
            eval(last, reading.strided!)
        } else {
            eval(last)
        }
        return reading
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< config.numHiddenLayers).map { _ in ArraysCache(size: 4) }
    }

    /// The checkpoint names the embedding `backbone.embedding`; a tied `lm_head`, when a conversion wrote one, is the
    /// embedding again and is dropped.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Mamba3Loader.renamed(weights)
    }

    public var loraLayers: [Module] { backbone.layers }
}
