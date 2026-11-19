//
//  Mamba3Loader.swift
//  MLXLLM
//
//  WHAT: Loads a `mamba3` checkpoint folder into a `Mamba3Model`, optionally narrowing its weights as they load: the
//        published commons is F32 (6.0 GB), BF16 values widened, so BF16 holds it exactly in half the memory.
//  PIN:  ONE TENSOR AT A TIME. Each tensor is read and narrowed on the CPU, evaluated, and put back in place of its
//        float32 original, which is then released: the float32 copy never sits whole beside the narrowed one. Reading
//        on the CPU first is Braid's fix (MAMBA3.md): a GPU cast inside one command buffer waited on a 3 GB read from
//        a slow disk and hit Metal's GPU timeout.
//  PIN:  ONLY MATRICES NARROW. The projections (`in_proj`, `out_proj`, `fc1`, `fc2`) and the embedding take the weight
//        dtype; the norms, `D`, `dt_bias`, the B/C biases and the MIMO projections stay float32, as the scan reads
//        them (Mamba3.swift's PIN).
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum Mamba3Loader {
    /// The model in `directory` (config.json and its safetensors), with matrices in `weightDType` (nil keeps the
    /// checkpoint's dtype), and the config refused as `Mamba3Configuration.validateModelConfiguration` refuses it.
    public static func load(from directory: URL, weightDType: DType? = .bfloat16) throws -> Mamba3Model {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder.json5().decode(Mamba3Configuration.self, from: data)
        try config.validateModelConfiguration()
        return try load(config: config, directory: directory, weightDType: weightDType)
    }

    /// As `load(from:weightDType:)`, with the config given (a test pins the activation of a fixture whose config
    /// leaves it out).
    public static func load(config: Mamba3Configuration, directory: URL, weightDType: DType? = .bfloat16) throws -> Mamba3Model {
        let model = Mamba3Model(config)
        var weights: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            // The checkpoint's own files only (`model.safetensors`, `model-0000k-of-0000n.safetensors`): an ExFAT volume
            // keeps a `._model.safetensors` (AppleDouble) beside each, and a folder may hold other arrays.
            .filter { $0.pathExtension == "safetensors" && $0.lastPathComponent.hasPrefix("model") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        for file in files {
            for (key, value) in try loadArrays(url: file) { weights[key] = value }
        }
        weights = renamed(weights)
        if let weightDType {
            for key in weights.keys.sorted() {
                let original = weights[key]!
                eval(original)
                guard narrows(key), original.dtype != weightDType else { continue }
                let narrow = original.asType(weightDType, stream: .cpu)
                eval(narrow)
                weights[key] = narrow
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    /// The checkpoint's names as the model's: `backbone.embedding` → `backbone.embeddings`; a tied `lm_head` dropped.
    static func renamed(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights where key != "lm_head.weight" {
            out[key == "backbone.embedding.weight" ? "backbone.embeddings.weight" : key] = value
        }
        return out
    }

    /// Whether `key` is a matrix that takes the weight dtype (see PIN).
    static func narrows(_ key: String) -> Bool {
        key == "backbone.embeddings.weight" || ["in_proj.weight", "out_proj.weight", "fc1.weight", "fc2.weight"].contains { key.hasSuffix($0) }
    }
}
