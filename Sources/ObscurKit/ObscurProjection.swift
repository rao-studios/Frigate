//
//  ObscurProjection.swift
//  ObscurKit
//
//  DINOv2 token space → per-layer adapter K/V space: learned pooling (T→32 query
//  cross-attention), a shared 2-layer MLP trunk, and per-hooked-layer K/V linear heads.
//  ~31M params fp16. Schema `obscur.projection.v1`, persisted as safetensors so the
//  offline-pretrained artifact (spec §6 P1) drops in over the seeded random init that
//  ships until then. Hooked-layer set is a build-time constant here.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Identifies an attention site inside Klein where the adapter injects.
public enum ObscurLayerRef: Hashable, Codable, Comparable, CustomStringConvertible {
    case doubleStream(index: Int)
    case singleStream(index: Int)

    public var description: String {
        switch self {
        case .doubleStream(let i): return "double.\(i)"
        case .singleStream(let i): return "single.\(i)"
        }
    }

    var key: String { description }

    public static func < (lhs: ObscurLayerRef, rhs: ObscurLayerRef) -> Bool {
        lhs.sortRank < rhs.sortRank
    }

    private var sortRank: Int {
        switch self {
        case .doubleStream(let i): return i
        case .singleStream(let i): return 1000 + i
        }
    }
}

public final class ObscurProjection {
    /// Default hooked set: every other double-stream + every other single-stream block
    /// of klein-4B (5 double / 20 single) → 13 sites. Ablate offline (P1); build-time.
    public static let defaultHookedLayers: [ObscurLayerRef] =
        [0, 2, 4].map { .doubleStream(index: $0) }
        + stride(from: 0, to: 20, by: 2).map { .singleStream(index: $0) }

    public static let schema = "obscur.projection.v1"
    public static let dinoDim = 384       // DINOv2-S — pinned to CLR's bundled checkpoint
    public static let modelDim = 3072     // klein-4B inner dim
    public static let pooledTokens = 32   // tokens/entry after learned pooling
    static let trunkHidden = 768

    public let hookedLayers: [ObscurLayerRef]

    // Learned pooling queries (32, 384).
    var poolQueries: MLXArray
    // Trunk: 384 → 768 → 384 with GELU.
    var trunkIn: MLXArray    // (768, 384)
    var trunkOut: MLXArray   // (384, 768)
    // Per-layer heads: layerKey → (Wk (3072, 384), Wv (3072, 384)).
    var heads: [ObscurLayerRef: (k: MLXArray, v: MLXArray)]

    /// Seeded random init — the placeholder until the offline-pretrained artifact lands.
    /// Persist after creation so results stay stable across sessions.
    public init(hookedLayers: [ObscurLayerRef] = ObscurProjection.defaultHookedLayers,
                seed: UInt64 = 0) {
        self.hookedLayers = hookedLayers
        var key = MLXRandom.key(seed)
        func draw(_ shape: [Int], scale: Float) -> MLXArray {
            let (k1, k2) = MLXRandom.split(key: key)
            key = k1
            return (MLXRandom.normal(shape, key: k2) * scale).asType(.float16)
        }
        let dinoScale = 1.0 / Float(Self.dinoDim).squareRoot()
        let trunkScale = 1.0 / Float(Self.trunkHidden).squareRoot()
        poolQueries = draw([Self.pooledTokens, Self.dinoDim], scale: dinoScale)
        trunkIn = draw([Self.trunkHidden, Self.dinoDim], scale: dinoScale)
        trunkOut = draw([Self.dinoDim, Self.trunkHidden], scale: trunkScale)
        var heads: [ObscurLayerRef: (MLXArray, MLXArray)] = [:]
        for layer in hookedLayers {
            heads[layer] = (draw([Self.modelDim, Self.dinoDim], scale: dinoScale),
                            draw([Self.modelDim, Self.dinoDim], scale: dinoScale))
        }
        self.heads = heads
    }

    // MARK: - Forward

    /// Pool raw DINOv2 tokens (T, 384) → (32, 384) via learned-query attention.
    public func pool(_ tokens: MLXArray) -> MLXArray {
        let t = tokens.asType(.float32)
        let q = poolQueries.asType(.float32)
        let scale = 1.0 / Float(Self.dinoDim).squareRoot()
        let logits = matmul(q, t.transposed(1, 0)) * scale       // (32, T)
        let attn = softmax(logits, axis: -1)
        return matmul(attn, t)                                    // (32, 384)
    }

    /// Full projection: raw tokens → per-hooked-layer (k, v), each (32, 3072) fp16.
    public func project(_ tokens: MLXArray) -> [ObscurLayerRef: (k: MLXArray, v: MLXArray)] {
        var x = pool(tokens)                                      // (32, 384) fp32
        let h = gelu(matmul(x, trunkIn.asType(.float32).transposed(1, 0)))
        x = x + matmul(h, trunkOut.asType(.float32).transposed(1, 0))
        var out: [ObscurLayerRef: (MLXArray, MLXArray)] = [:]
        for layer in hookedLayers {
            guard let head = heads[layer] else { continue }
            let k = matmul(x, head.k.asType(.float32).transposed(1, 0)).asType(.float16)
            let v = matmul(x, head.v.asType(.float32).transposed(1, 0)).asType(.float16)
            out[layer] = (k, v)
        }
        return out
    }

    // MARK: - Persistence (versioned safetensors)

    public func save(to url: URL) throws {
        var tensors: [String: MLXArray] = [
            "pool.queries": poolQueries,
            "trunk.in": trunkIn,
            "trunk.out": trunkOut,
        ]
        for (layer, head) in heads {
            tensors["head.\(layer.key).k"] = head.k
            tensors["head.\(layer.key).v"] = head.v
        }
        try MLX.save(arrays: tensors, metadata: ["schema": Self.schema], url: url)
    }

    public static func load(from url: URL) throws -> ObscurProjection {
        let tensors = try loadArrays(url: url)
        guard let pool = tensors["pool.queries"],
              let tIn = tensors["trunk.in"],
              let tOut = tensors["trunk.out"] else {
            throw ObscurKitError.corruptStore("projection missing trunk tensors")
        }
        var layers: [ObscurLayerRef] = []
        var heads: [ObscurLayerRef: (MLXArray, MLXArray)] = [:]
        for name in tensors.keys where name.hasPrefix("head.") && name.hasSuffix(".k") {
            let key = String(name.dropFirst("head.".count).dropLast(".k".count))
            guard let layer = ObscurLayerRef(key: key),
                  let k = tensors[name], let v = tensors["head.\(key).v"] else { continue }
            layers.append(layer)
            heads[layer] = (k, v)
        }
        let projection = ObscurProjection(hookedLayers: layers.sorted())
        projection.poolQueries = pool
        projection.trunkIn = tIn
        projection.trunkOut = tOut
        projection.heads = heads
        return projection
    }
}

extension ObscurLayerRef {
    init?(key: String) {
        let parts = key.split(separator: ".")
        guard parts.count == 2, let index = Int(parts[1]) else { return nil }
        switch parts[0] {
        case "double": self = .doubleStream(index: index)
        case "single": self = .singleStream(index: index)
        default: return nil
        }
    }
}
