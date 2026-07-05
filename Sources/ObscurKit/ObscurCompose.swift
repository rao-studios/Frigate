//
//  ObscurCompose.swift
//  ObscurKit
//
//  Composition + the decoupled cross-attention branch. A generation composes one or
//  more banks into per-layer K/V (concatenation along the sequence dim; per-corpus
//  logit bias added pre-softmax so softmax renormalizes across banks of any size).
//  The branch reuses the base block's image-stream queries — no new Q weights — and is
//  scaled by per-layer gates that init to 0 (bit-exact base behavior at install).
//  Attention over the bank is computed manually (not fused SDPA) because attribution
//  requires the softmax weights; banks are small, so this is cheap.
//

import Foundation
import MLX

public enum ObscurEntrySelection {
    case all
    case explicit([ObscurEntryID])
    /// kNN in DINOv2-mean space against a query embedding. Mandatory past a few
    /// hundred entries; linear scan here — swap in CLR's HNSW when corpora demand it.
    case topK(k: Int, query: MLXArray)
}

/// Per-layer scalar gates, grouped for UX as structure (double-stream) and texture
/// (single-stream). Values are the *effective* gates for one generation.
public struct ObscurGates {
    public var perLayer: [ObscurLayerRef: Float]

    public init(perLayer: [ObscurLayerRef: Float]) {
        self.perLayer = perLayer
    }

    /// Zero everywhere — the adapter is a no-op (safe insertion).
    public static func zero(for layers: [ObscurLayerRef]) -> ObscurGates {
        .init(perLayer: Dictionary(uniqueKeysWithValues: layers.map { ($0, 0) }))
    }

    /// Grouped dials: `structure` drives double-stream sites, `texture` single-stream.
    public static func grouped(structure: Float, texture: Float,
                               layers: [ObscurLayerRef]) -> ObscurGates {
        var perLayer: [ObscurLayerRef: Float] = [:]
        for layer in layers {
            switch layer {
            case .doubleStream: perLayer[layer] = structure
            case .singleStream: perLayer[layer] = texture
            }
        }
        return .init(perLayer: perLayer)
    }
}

/// Union of one or more banks prepared for a single generation: per hooked layer,
/// K/V concatenated over all selected entries, plus a per-key-token logit bias and
/// the entry spans needed to slice attention mass back to entries.
public struct ObscurComposedKV {
    public struct Span {
        public let corpusID: ObscurCorpusID
        public let entryID: ObscurEntryID
        public let range: Range<Int>       // token rows in the concatenated bank
    }

    /// layer → (k: (N, D), v: (N, D)) fp16.
    public let kv: [ObscurLayerRef: (k: MLXArray, v: MLXArray)]
    /// (N,) fp32 — per-token logit bias (each token carries its corpus's bias).
    public let logitBias: MLXArray
    public let spans: [Span]
    public let tokenCount: Int

    public var isEmpty: Bool { tokenCount == 0 }

    public init(kv: [ObscurLayerRef: (k: MLXArray, v: MLXArray)],
                logitBias: MLXArray, spans: [Span], tokenCount: Int) {
        self.kv = kv
        self.logitBias = logitBias
        self.spans = spans
        self.tokenCount = tokenCount
    }

    /// A single-entry bank of `tokenCount` tokens (bias 0) — the training-time bank
    /// (projection output for one image), differentiable: `kv` is left un-eval'd.
    public static func singleEntry(kv: [ObscurLayerRef: (k: MLXArray, v: MLXArray)],
                                   tokenCount: Int,
                                   corpusID: ObscurCorpusID = "train",
                                   entryID: ObscurEntryID = "entry") -> ObscurComposedKV {
        ObscurComposedKV(kv: kv,
                         logitBias: MLXArray.zeros([tokenCount], type: Float.self),
                         spans: [Span(corpusID: corpusID, entryID: entryID, range: 0..<tokenCount)],
                         tokenCount: tokenCount)
    }
}

public enum ObscurComposer {
    /// Compose selected entries of the given banks through `projection`.
    /// K/V per entry are memoized on the bank until entries or projection change.
    public static func compose(banks: [ObscurBank],
                               selection: ObscurEntrySelection,
                               projection: ObscurProjection) throws -> ObscurComposedKV {
        var spans: [ObscurComposedKV.Span] = []
        var biasChunks: [Float] = []
        var perLayerK: [ObscurLayerRef: [MLXArray]] = [:]
        var perLayerV: [ObscurLayerRef: [MLXArray]] = [:]
        var cursor = 0

        for bank in banks {
            let ids = select(from: bank, selection: selection)
            for id in ids {
                guard let entry = bank.entries[id] else { continue }
                let kv: [ObscurLayerRef: (k: MLXArray, v: MLXArray)]
                if let memo = bank.kvMemo[id] {
                    kv = memo
                } else {
                    kv = projection.project(entry.projectionInput)
                    bank.kvMemo[id] = kv
                }
                let tokens = ObscurProjection.pooledTokens
                for (layer, pair) in kv {
                    perLayerK[layer, default: []].append(pair.k)
                    perLayerV[layer, default: []].append(pair.v)
                }
                spans.append(.init(corpusID: bank.corpusID, entryID: id,
                                   range: cursor..<(cursor + tokens)))
                biasChunks.append(contentsOf: Array(repeating: bank.logitBias, count: tokens))
                cursor += tokens
            }
        }

        guard cursor > 0 else { throw ObscurKitError.emptyBank }

        var kv: [ObscurLayerRef: (k: MLXArray, v: MLXArray)] = [:]
        for layer in projection.hookedLayers {
            guard let ks = perLayerK[layer], let vs = perLayerV[layer] else { continue }
            kv[layer] = (concatenated(ks, axis: 0), concatenated(vs, axis: 0))
        }
        return ObscurComposedKV(kv: kv,
                                logitBias: MLXArray(biasChunks),
                                spans: spans,
                                tokenCount: cursor)
    }

    static func select(from bank: ObscurBank, selection: ObscurEntrySelection) -> [ObscurEntryID] {
        switch selection {
        case .all:
            return bank.order
        case .explicit(let ids):
            return bank.order.filter { ids.contains($0) }
        case .topK(let k, let query):
            guard bank.count > k else { return bank.order }
            let q = query.asType(.float32)
            let qNorm = sqrt(q.square().sum()).item(Float.self)
            let scored: [(ObscurEntryID, Float)] = bank.order.compactMap { id in
                guard let entry = bank.entries[id] else { return nil }
                let m = entry.dinoMean.asType(.float32)
                let cosine = (q * m).sum().item(Float.self)
                    / max(qNorm * sqrt(m.square().sum()).item(Float.self), 1e-6)
                return (id, cosine)
            }
            return scored.sorted { $0.1 > $1.1 }.prefix(k).map { $0.0 }
        }
    }
}

// MARK: - Attribution

/// Accumulates gate-weighted attention mass per entry, per layer, per denoising step.
/// With grid dimensions set, also accumulates the SPATIAL map: attention mass per image
/// region (query token) per entry — the data behind the Influence Cloud. A faithful
/// record of what the model read — a causal-input record, not an influence certificate.
public final class ObscurAttributionRecorder {
    // step → layerKey → entry masses (aligned with spans order)
    private var masses: [Int: [String: [Float]]] = [:]
    private let spans: [ObscurComposedKV.Span]
    public var currentStep: Int = 0

    /// Image-token grid (packedW × packedH). Zero ⇒ spatial recording disabled.
    public let gridWidth: Int
    public let gridHeight: Int
    /// (S × E) row-major accumulator, gate-weighted, summed over layers and steps.
    private var spatialAccum: [Float]

    var wantsSpatial: Bool { gridWidth > 0 && gridHeight > 0 }

    public init(spans: [ObscurComposedKV.Span], gridWidth: Int = 0, gridHeight: Int = 0) {
        self.spans = spans
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.spatialAccum = [Float](repeating: 0,
                                    count: max(gridWidth * gridHeight * spans.count, 0))
    }

    func record(layer: ObscurLayerRef, entryMasses: [Float]) {
        masses[currentStep, default: [:]][layer.key] = entryMasses
    }

    /// masses: (S × E) row-major for this layer/step, already gate-weighted.
    func recordSpatial(_ values: [Float]) {
        guard values.count == spatialAccum.count else { return }
        for i in 0..<values.count {
            spatialAccum[i] += values[i]
        }
    }

    /// The per-region influence map accumulated across the generation, or nil when
    /// spatial recording was disabled.
    public func influenceMap() -> ObscurInfluenceMap? {
        guard wantsSpatial, !spans.isEmpty else { return nil }
        let s = gridWidth * gridHeight
        let e = spans.count
        var relief = [Float](repeating: 0, count: s)
        var shares = [Float](repeating: 0, count: s * e)
        var entryTotals = [Float](repeating: 0, count: e)
        for i in 0..<s {
            var total: Float = 0
            for j in 0..<e {
                let value = spatialAccum[i * e + j]
                total += value
                entryTotals[j] += value
            }
            relief[i] = total
            if total > 0 {
                for j in 0..<e {
                    shares[i * e + j] = spatialAccum[i * e + j] / total
                }
            }
        }
        let grandTotal = max(entryTotals.reduce(0, +), 1e-12)
        let maxRelief = max(relief.max() ?? 0, 1e-12)
        return ObscurInfluenceMap(gridWidth: gridWidth,
                                  gridHeight: gridHeight,
                                  entryIDs: spans.map(\.entryID),
                                  regionRelief: relief.map { $0 / maxRelief },
                                  entryShares: shares,
                                  entryTotals: entryTotals.map { $0 / grandTotal })
    }

    public func report(generationID: UUID, modelVersion: String,
                       projectionVersion: String = ObscurProjection.schema,
                       gates: ObscurGates,
                       corpora: [ObscurCorpusID: Float]) -> ObscurAttributionReport {
        // Total gate-weighted mass per entry across steps and layers.
        var perEntryTotal = [Float](repeating: 0, count: spans.count)
        var perEntryLayer: [String: [Float]] = [:]
        for (_, layers) in masses {
            for (layerKey, entryMasses) in layers {
                for (i, m) in entryMasses.enumerated() where i < spans.count {
                    perEntryTotal[i] += m
                    perEntryLayer[layerKey, default: [Float](repeating: 0, count: spans.count)][i] += m
                }
            }
        }
        let total = max(perEntryTotal.reduce(0, +), 1e-12)

        var entries: [String: ObscurAttributionReport.EntryShare] = [:]
        for (i, span) in spans.enumerated() {
            var perLayer: [String: Float] = [:]
            for (layerKey, values) in perEntryLayer {
                let layerTotal = max(values.reduce(0, +), 1e-12)
                perLayer[layerKey] = values[i] / layerTotal
            }
            entries[span.entryID] = .init(attentionShare: perEntryTotal[i] / total,
                                          perLayerShare: perLayer,
                                          corpusID: span.corpusID)
        }
        return ObscurAttributionReport(generationID: generationID,
                                       timestamp: Date(),
                                       modelVersion: modelVersion,
                                       projectionVersion: projectionVersion,
                                       corpora: corpora,
                                       entries: entries)
    }
}

/// Spatial influence map for one generation: for every image region (packed token),
/// how much the adapter branch read from each bank entry. Region order is row-major
/// over the packed grid; each region maps to a 16×16-pixel patch of the output.
public struct ObscurInfluenceMap: Codable {
    public let gridWidth: Int
    public let gridHeight: Int
    public let entryIDs: [String]
    /// (S,) — total adapter attention per region, max-normalized to 0…1 (Z relief).
    public let regionRelief: [Float]
    /// (S × E) row-major — per region, each entry's share (sums to 1 where relief > 0).
    public let entryShares: [Float]
    /// (E,) — whole-generation share per entry (matches the aggregate report).
    public let entryTotals: [Float]

    public func shares(atRegion index: Int) -> [Float] {
        let e = entryIDs.count
        return Array(entryShares[(index * e)..<((index + 1) * e)])
    }
}

public struct ObscurAttributionReport: Codable {
    public struct EntryShare: Codable {
        public let attentionShare: Float       // normalized over the adapter branch
        public let perLayerShare: [String: Float]
        public let corpusID: ObscurCorpusID
    }
    public let generationID: UUID
    public let timestamp: Date
    public let modelVersion: String
    public let projectionVersion: String
    public let corpora: [ObscurCorpusID: Float]   // corpusID → logitBias snapshot
    public let entries: [String: EntryShare]
}

// MARK: - Injection context (consumed by FluxKit at hooked attention sites)

/// Everything one generation's adapter pass needs: composed K/V, gates, recorder.
public final class ObscurInjectionContext {
    public let composed: ObscurComposedKV
    public let gates: ObscurGates
    public let recorder: ObscurAttributionRecorder?
    let headCount: Int
    let headDim: Int

    public init(composed: ObscurComposedKV, gates: ObscurGates,
                recorder: ObscurAttributionRecorder?,
                headCount: Int = 24, headDim: Int = 128) {
        self.composed = composed
        self.gates = gates
        self.recorder = recorder
        self.headCount = headCount
        self.headDim = headDim
    }

    public func isHooked(_ layer: ObscurLayerRef) -> Bool {
        composed.kv[layer] != nil && (gates.perLayer[layer] ?? 0) != 0
    }

    /// Whether this layer participates at all (even at gate 0 we skip compute, but the
    /// recorder still needs spans on layers that ran — gate 0 layers record nothing).
    public func gate(_ layer: ObscurLayerRef) -> Float {
        gates.perLayer[layer] ?? 0
    }

    /// The decoupled cross-attention branch at one hooked site.
    /// query: (B, H, S, D) pre-RoPE, RMS-normed image-stream queries from the base
    /// block. `baseOutput`: the block's own attention output for the same image tokens,
    /// (B, S, H*D) — the branch is RMS-matched to it per token, which makes the gate
    /// dimensionless: gate 1.0 injects at most a base-attention-sized signal, never an
    /// unbounded one (an uncalibrated projection at raw scale flattens the whole
    /// generation into a constant-color latent). Returns the gated branch output to add
    /// to `baseOutput`, or nil when this layer is unhooked/gated off.
    public func apply(query: MLXArray, baseOutput: MLXArray, layer: ObscurLayerRef) -> MLXArray? {
        guard let (k, v) = composed.kv[layer] else { return nil }
        let gate = gate(layer)
        guard gate != 0 else { return nil }

        let outDtype = query.dtype
        let b = query.dim(0)
        let s = query.dim(2)
        let n = composed.tokenCount

        // Bank K/V (N, H*D) → (1, H, N, D), fp32 for the softmax.
        let q = query.asType(.float32)
        let kHeads = k.asType(.float32).reshaped(1, n, headCount, headDim).transposed(0, 2, 1, 3)
        let vHeads = v.asType(.float32).reshaped(1, n, headCount, headDim).transposed(0, 2, 1, 3)

        let scale = 1.0 / Float(headDim).squareRoot()
        var logits = matmul(q, kHeads.transposed(0, 1, 3, 2)) * scale     // (B, H, S, N)
        logits = logits + composed.logitBias.reshaped(1, 1, 1, n)
        let attn = softmax(logits, axis: -1)

        if let recorder {
            // Attention mass per bank token, summed over batch/heads/queries, then per
            // entry span, gate-weighted.
            let perToken = attn.sum(axes: [0, 1, 2])                       // (N,)
            eval(perToken)
            let tokenMass: [Float] = perToken.asArray(Float.self)
            let entryMasses = composed.spans.map { span in
                tokenMass[span.range].reduce(0, +) * gate
            }
            recorder.record(layer: layer, entryMasses: entryMasses)

            // Spatial map: mass per image region per entry (S × E), gate-weighted.
            if recorder.wantsSpatial, recorder.gridWidth * recorder.gridHeight == s {
                let spatial = attn.sum(axes: [0, 1])                       // (S, N)
                let columns = composed.spans.map { span in
                    spatial[0..., span.range.lowerBound..<span.range.upperBound].sum(axis: -1)
                }
                let regionByEntry = stacked(columns, axis: -1)             // (S, E)
                eval(regionByEntry)
                recorder.recordSpatial(regionByEntry.asArray(Float.self).map { $0 * gate })
            }
        }

        var out = matmul(attn, vHeads)                                     // (B, H, S, D)
            .transposed(0, 2, 1, 3)
            .reshaped(b, s, headCount * headDim)

        // Per-token RMS match against the base attention output. Eps inside the sqrt:
        // a (near-)zero branch otherwise has an infinite sqrt gradient at 0, which NaNs
        // training when backprop runs through this scale.
        let base32 = baseOutput.asType(.float32)
        let baseRMS = sqrt(base32.square().mean(axis: -1, keepDims: true) + 1e-12)
        let branchRMS = sqrt(out.square().mean(axis: -1, keepDims: true) + 1e-12)
        out = out * (baseRMS / branchRMS)

        return (out * gate).asType(outDtype)
    }
}
