//  WHAT: A Qwen3.5 text model (MLXLLM's, loaded without its tower) reading a picture with the tower its checkpoint
//        carries: this module's `Qwen35` built over the loaded model's own arrays, with only `vision_tower.*` read
//        from the checkpoint. The picture's processing, its placement and the M-RoPE positions are `Qwen35`'s own.
//  PIN:  SHARED, NEVER COPIED. The language model's parameters are the loaded model's arrays, handed over by name
//        (both modules name them `language_model.…`), quantized where those arrays carry scales. `prepare()` is not
//        called: it would fuse new copies of the delta-net projections, and the unfused path reads the shared ones.

import CoreGraphics
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public final class Qwen35OwnVision: OwnVision, @unchecked Sendable {
    let model: Qwen35
    let processor: Qwen3VLProcessor
    let tokenizer: any Tokenizer
    let stops: Set<Int>

    init(model: Qwen35, processor: Qwen3VLProcessor, tokenizer: any Tokenizer, stops: Set<Int>) {
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.stops = stops
    }

    static let towerPrefix = "vision_tower."

    /// The tower of the checkpoint in `directory`, over `text` (the Qwen3.5 the host loaded from there); nil when the
    /// checkpoint is not a Qwen3.5 or carries no `vision_tower` tensors.
    public static func over(_ text: any LanguageModel, tokenizer: any Tokenizer, directory: URL) async throws -> Qwen35OwnVision? {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: data)
        guard base.modelType == "qwen3_5" || base.modelType == "qwen3_5_moe" else { return nil }
        let files = try OwnVisionCheckpoint.files(in: directory, prefix: towerPrefix)
        var tower: [String: MLXArray] = [:]
        for url in files {
            for (key, value) in try loadArrays(url: url) where key.hasPrefix(towerPrefix) { tower[key] = value }
        }
        guard !tower.isEmpty,
              let model = try await VLMTypeRegistry.shared.createModel(configuration: data, modelType: base.modelType) as? Qwen35
        else { return nil }

        let shared = Dictionary(text.parameters().flattened(), uniquingKeysWith: { first, _ in first })
            .filter { $0.key.hasPrefix("language_model.") }
        if let perLayer = base.perLayerQuantization {
            quantize(model: model) { path, _ in
                guard shared["\(path).scales"] != nil else { return nil }
                return perLayer.quantization(layer: path)?.asTuple
            }
        }
        let vision = Qwen3VLVision.VisionModel.sanitize(weights: tower, config: model.config.visionConfiguration)
        try model.update(parameters: ModuleParameters.unflattened(shared.merging(vision) { first, _ in first }), verify: [.all])
        eval(model.parameters().flattened().filter { $0.0.hasPrefix(towerPrefix) }.map(\.1))

        let processor = Qwen3VLProcessor(
            Qwen3VLProcessorConfiguration(qwen35VisionConfiguration: model.config.visionConfiguration), tokenizer: tokenizer)
        var stops = base.effectiveEOSTokenIds
        for token in ["<|im_end|>", "<|endoftext|>"] { if let id = tokenizer.convertTokenToId(token) { stops.insert(id) } }
        if let eos = tokenizer.eosTokenId { stops.insert(eos) }
        return Qwen35OwnVision(model: model, processor: processor, tokenizer: tokenizer, stops: stops)
    }

    public func reader(image: CGImage, question: String, maxTokens: Int) async throws -> @Sendable () throws -> String {
        let input = UserInput(chat: [.user(question, images: [.ciImage(CIImage(cgImage: image))])],
                              additionalContext: ["enable_thinking": false])
        let prepared = Prepared(input: try await processor.prepare(input: input))
        return { [self] in
            try withError {
                let iterator = try TokenIterator(input: prepared.input, model: model,
                                                 parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0))
                var tokens: [Int] = []
                for token in iterator {
                    if stops.contains(token) { break }
                    tokens.append(token)
                }
                return tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
    }
}

/// The processed picture and prompt, read only by the forward that follows.
private final class Prepared: @unchecked Sendable {
    let input: LMInput
    init(input: LMInput) { self.input = input }
}
