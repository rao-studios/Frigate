//
//  WeightLoading.swift
//  FluxKit
//
//  Loads the mflux-exported FLUX.2 Klein checkpoint: sharded safetensors indexed by
//  model.safetensors.index.json, with MLX-native affine quantization stored as
//  `weight`/`scales`/`biases` triplets (bits/group size inferred from shapes). Builders
//  construct concrete MLXNN layers with explicit arrays so a missing or misshapen
//  tensor fails loudly at load, not silently at inference.
//

import Foundation
import MLX
import MLXNN

public enum FluxKitError: Error, LocalizedError {
    case missingTensor(String)
    case missingFile(String)
    case malformedCheckpoint(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .missingTensor(let name): return "Checkpoint is missing tensor '\(name)'"
        case .missingFile(let name): return "Checkpoint is missing file '\(name)'"
        case .malformedCheckpoint(let reason): return "Malformed checkpoint: \(reason)"
        case .cancelled: return "Generation was cancelled"
        }
    }
}

/// One component directory (text_encoder / transformer / vae) fully loaded into memory.
public final class TensorStore {
    private var tensors: [String: MLXArray]

    public init(componentDir: URL) throws {
        let indexURL = componentDir.appendingPathComponent("model.safetensors.index.json")
        guard let indexData = try? Data(contentsOf: indexURL),
              let index = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
              let weightMap = index["weight_map"] as? [String: String] else {
            throw FluxKitError.missingFile(indexURL.lastPathComponent + " in \(componentDir.lastPathComponent)")
        }
        var merged: [String: MLXArray] = [:]
        for shard in Set(weightMap.values) {
            let shardURL = componentDir.appendingPathComponent(shard)
            guard FileManager.default.fileExists(atPath: shardURL.path) else {
                throw FluxKitError.missingFile("\(componentDir.lastPathComponent)/\(shard)")
            }
            let arrays = try loadArrays(url: shardURL)
            merged.merge(arrays) { a, _ in a }
        }
        self.tensors = merged
    }

    public func take(_ name: String) throws -> MLXArray {
        guard let t = tensors[name] else { throw FluxKitError.missingTensor(name) }
        return t
    }

    public func contains(_ name: String) -> Bool { tensors[name] != nil }

    /// Free everything still held (post-build residue).
    public func drain() { tensors.removeAll() }

    // MARK: - Layer builders

    /// Linear that is quantized in the checkpoint (weight/scales/biases triplet).
    /// Bits and group size are inferred: packed uint32 columns = in*bits/32,
    /// scales columns = in/groupSize.
    public func quantizedLinear(_ prefix: String, inFeatures: Int, bias: Bool = false) throws -> QuantizedLinear {
        let weight = try take("\(prefix).weight")
        let scales = try take("\(prefix).scales")
        let biases = try take("\(prefix).biases")
        let bits = weight.dim(1) * 32 / inFeatures
        let groupSize = inFeatures / scales.dim(1)
        let b = bias ? try take("\(prefix).bias") : nil
        return QuantizedLinear(weight: weight, bias: b, scales: scales, biases: biases,
                               groupSize: groupSize, bits: bits)
    }

    /// Linear stored unquantized.
    public func linear(_ prefix: String, bias: Bool = false) throws -> Linear {
        Linear(weight: try take("\(prefix).weight"),
               bias: bias ? try take("\(prefix).bias") : nil)
    }

    /// Linear that may or may not be quantized in the checkpoint.
    public func anyLinear(_ prefix: String, inFeatures: Int, bias: Bool = false) throws -> Linear {
        if contains("\(prefix).scales") {
            return try quantizedLinear(prefix, inFeatures: inFeatures, bias: bias)
        }
        return try linear(prefix, bias: bias)
    }

    public func quantizedEmbedding(_ prefix: String, dimensions: Int) throws -> ExplicitQuantizedEmbedding {
        let weight = try take("\(prefix).weight")
        let scales = try take("\(prefix).scales")
        let biases = try take("\(prefix).biases")
        let bits = weight.dim(1) * 32 / dimensions
        let groupSize = dimensions / scales.dim(1)
        return ExplicitQuantizedEmbedding(weight: weight, scales: scales, biases: biases,
                                          groupSize: groupSize, bits: bits)
    }

    public func rmsNormWeight(_ prefix: String) throws -> MLXArray {
        try take("\(prefix).weight")
    }

    /// Conv2d weights in the mflux checkpoint are already MLX-native OHWI (O, kH, kW, I).
    public func conv2d(_ prefix: String) throws -> (weight: MLXArray, bias: MLXArray) {
        (try take("\(prefix).weight"), try take("\(prefix).bias"))
    }
}

/// Token embedding that may be quantized or full precision in the checkpoint.
public enum TokenEmbedding {
    case quantized(ExplicitQuantizedEmbedding)
    case full(MLXArray)

    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        switch self {
        case .quantized(let q): return q(ids)
        case .full(let weight):
            let shape = ids.shape
            return weight[ids.flattened()].reshaped(shape + [-1])
        }
    }
}

/// Token embedding stored as an MLX affine-quantized triplet. Gathers the packed rows
/// for the input ids and dequantizes just those (the checkpoint's `QuantizedEmbedding`
/// init doesn't accept pre-quantized arrays, so we do the lookup ourselves).
public final class ExplicitQuantizedEmbedding {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    let groupSize: Int
    let bits: Int

    init(weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
    }

    /// ids: (B, S) → (B, S, dimensions).
    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let shape = ids.shape
        let flat = ids.flattened()
        let rows = dequantized(weight[flat], scales: scales[flat], biases: biases[flat],
                               groupSize: groupSize, bits: bits, mode: .affine)
        return rows.reshaped(shape + [-1])
    }
}

/// RMSNorm with an explicit weight, computed in float32 like the reference.
final class ExplicitRMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(weight: MLXArray, eps: Float) {
        self.weight = weight
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let x32 = x.asType(.float32)
        let normed = x32 * rsqrt(x32.square().mean(axis: -1, keepDims: true) + eps)
        return (normed * weight.asType(.float32)).asType(dtype)
    }
}
