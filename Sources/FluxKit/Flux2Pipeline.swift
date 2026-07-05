//
//  Flux2Pipeline.swift
//  FluxKit
//
//  End-to-end FLUX.2 Klein text-to-image over an mflux-format checkpoint directory:
//  Qwen3 prompt encoding (chat-templated, padded to 512, layers 9/18/27 interleaved),
//  4-step flow-match Euler denoising through the DiT, BN-denorm + VAE decode.
//
//  Components are loaded and released in stages so peak memory stays near the largest
//  single component (~2.5 GB for the 4-bit checkpoint) — required for iPhone.
//

import CoreGraphics
import Foundation
import Hub
import MLX
import MLXRandom
import ObscurKit
import Tokenizers

public final class Flux2Pipeline {
    /// bfloat16 matches mflux numerics (M-series and A17+ GPUs).
    public static let activationDType: DType = .bfloat16

    public enum Phase {
        case loadingTextEncoder
        case encodingPrompt
        case loadingTransformer
        case denoising(step: Int, total: Int)
        case decoding
    }

    let modelDir: URL

    public init(modelDirectory: URL) {
        self.modelDir = modelDirectory
    }

    /// Synchronous; run on a background executor. `onPhase` fires on the calling thread.
    /// `adapter` installs a Signet bank branch for this generation (see ObscurKit);
    /// its recorder — when set — accrues per-entry attribution across steps/layers.
    public func generate(prompt: String,
                         width: Int = 1024,
                         height: Int = 1024,
                         steps: Int = 4,
                         seed: UInt64 = UInt64.random(in: 0..<UInt64.max),
                         adapter: ObscurInjectionContext? = nil,
                         onPhase: ((Phase) -> Void)? = nil,
                         isCancelled: (() -> Bool)? = nil) throws -> CGImage {
        func checkCancelled() throws {
            if isCancelled?() == true { throw FluxKitError.cancelled }
        }

        // Keep MLX's recycled-buffer cache small for the duration of the run: freed
        // component weights must return to the OS, not sit in the cache (Jetsam counts
        // cached buffers against the app on iOS).
        let previousCacheLimit = Memory.cacheLimit
        Memory.cacheLimit = 128 * 1024 * 1024
        defer {
            Memory.cacheLimit = previousCacheLimit
            Memory.clearCache()
        }

        // ── 1. Prompt → context embeddings (Qwen3 taps), then free the encoder ──
        onPhase?(.loadingTextEncoder)
        let (inputIDs, attentionMask) = try tokenize(prompt: prompt)
        try checkCancelled()

        var promptEmbeds: MLXArray
        do {
            let encoder = try Qwen3TextEncoder(componentDir: modelDir.appendingPathComponent("text_encoder"))
            onPhase?(.encodingPrompt)
            promptEmbeds = encoder.promptEmbeddings(inputIDs: inputIDs, attentionMask: attentionMask)
            eval(promptEmbeds)
        }
        Memory.clearCache()
        try checkCancelled()

        // ── 2. Latents, ids, schedule ──
        let latentH = 2 * (height / 16)   // VAE /8, then /2 packing
        let latentW = 2 * (width / 16)
        let packedH = latentH / 2
        let packedW = latentW / 2
        let imageSeqLen = packedH * packedW

        // Match mflux: sample NCHW then pack to (B, H*W, C).
        let key = MLXRandom.key(seed)
        var latents = MLXRandom.normal([1, 128, packedH, packedW], key: key)
            .asType(Self.activationDType)
            .reshaped(1, 128, imageSeqLen)
            .transposed(0, 2, 1)

        let imgIDs = Self.latentIDs(packedH: packedH, packedW: packedW)
        let txtIDs = Self.textIDs(count: promptEmbeds.dim(1))
        let scheduler = Flux2Scheduler(imageSequenceLength: imageSeqLen, steps: steps)

        // ── 3. Denoise, then free the DiT ──
        onPhase?(.loadingTransformer)
        do {
            let transformer = try Flux2Transformer(componentDir: modelDir.appendingPathComponent("transformer"))
            try checkCancelled()
            for i in 0..<steps {
                adapter?.recorder?.currentStep = i
                let noise = transformer(hidden: latents, encoder: promptEmbeds,
                                        timestep: scheduler.timesteps[i],
                                        imgIDs: imgIDs, txtIDs: txtIDs,
                                        adapter: adapter)
                latents = scheduler.step(noise: noise, latents: latents, index: i)
                eval(latents)
                onPhase?(.denoising(step: i + 1, total: steps))
                try checkCancelled()
            }
        }
        promptEmbeds = MLXArray()
        Memory.clearCache()

        // ── 4. Decode ──
        onPhase?(.decoding)
        let packed = latents.reshaped(1, packedH, packedW, 128).transposed(0, 3, 1, 2)
        var imageArray: MLXArray
        do {
            let vae = try Flux2VAEDecoder(componentDir: modelDir.appendingPathComponent("vae"))
            imageArray = vae.decodePacked(packed)   // NHWC, [-1, 1]
            eval(imageArray)
        }
        Memory.clearCache()
        try checkCancelled()

        return try Self.toCGImage(imageArray)
    }

    /// Encode a prompt to Qwen3 context embeddings, loading and freeing the encoder.
    /// Exposed for the projection trainer (which caches one empty-prompt embedding).
    public func promptEmbeddings(for prompt: String) throws -> MLXArray {
        let (inputIDs, attentionMask) = try tokenize(prompt: prompt)
        let encoder = try Qwen3TextEncoder(componentDir: modelDir.appendingPathComponent("text_encoder"))
        let embeds = encoder.promptEmbeddings(inputIDs: inputIDs, attentionMask: attentionMask)
        eval(embeds)
        Memory.clearCache()
        return embeds
    }

    // MARK: - Prompt tokenization

    /// Qwen3 chat template for one user message, generation prompt, thinking disabled —
    /// rendered directly (the template is fixed for this call shape). Padded/truncated
    /// to 512 with <|endoftext|> (151643), right-padding, mask over real tokens.
    func tokenize(prompt: String, maxLength: Int = 512) throws -> (ids: MLXArray, mask: MLXArray) {
        let templated = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        let tokenizer = try loadTokenizer()
        var ids = tokenizer.encode(text: templated)
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
        }
        let realCount = ids.count
        let padID = 151_643
        ids.append(contentsOf: Array(repeating: padID, count: maxLength - realCount))
        var mask = Array(repeating: Int32(1), count: realCount)
        mask.append(contentsOf: Array(repeating: Int32(0), count: maxLength - realCount))
        let idArray = MLXArray(ids.map(Int32.init)).reshaped(1, maxLength)
        let maskArray = MLXArray(mask).reshaped(1, maxLength)
        return (idArray, maskArray)
    }

    private func loadTokenizer() throws -> Tokenizer {
        let dir = modelDir.appendingPathComponent("tokenizer")
        let configURL = dir.appendingPathComponent("tokenizer_config.json")
        let dataURL = dir.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            throw FluxKitError.missingFile("tokenizer/tokenizer.json")
        }
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [NSString: Any] ?? [:]
        let data = try JSONSerialization.jsonObject(with: Data(contentsOf: dataURL)) as? [NSString: Any] ?? [:]
        return try AutoTokenizer.from(tokenizerConfig: Config(config), tokenizerData: Config(data))
    }

    // MARK: - Position ids

    /// (S, 4) coords (t=0, h, w, l=0) row-major over the packed grid.
    static func latentIDs(packedH: Int, packedW: Int) -> MLXArray {
        var coords = [Int32]()
        coords.reserveCapacity(packedH * packedW * 4)
        for h in 0..<packedH {
            for w in 0..<packedW {
                coords.append(contentsOf: [0, Int32(h), Int32(w), 0])
            }
        }
        return MLXArray(coords).reshaped(packedH * packedW, 4)
    }

    /// (S, 4) coords (t=0, h=0, w=0, l=token index).
    static func textIDs(count: Int) -> MLXArray {
        var coords = [Int32]()
        coords.reserveCapacity(count * 4)
        for l in 0..<count {
            coords.append(contentsOf: [0, 0, 0, Int32(l)])
        }
        return MLXArray(coords).reshaped(count, 4)
    }

    // MARK: - Output

    /// NHWC [-1, 1] float → RGBA CGImage.
    static func toCGImage(_ image: MLXArray) throws -> CGImage {
        let h = image.dim(1)
        let w = image.dim(2)
        var rgb = (clip(image[0] * 0.5 + 0.5, min: 0, max: 1) * 255).round().asType(.uint8)
        let alpha = MLXArray.full([h, w, 1], values: MLXArray(UInt8(255)), type: UInt8.self)
        rgb = concatenated([rgb, alpha], axis: -1)
        eval(rgb)
        let bytes: [UInt8] = rgb.asArray(UInt8.self)

        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let cgImage = CGImage(width: w, height: h,
                                    bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                                    space: CGColorSpaceCreateDeviceRGB(),
                                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                                    provider: provider, decode: nil,
                                    shouldInterpolate: false, intent: .defaultIntent) else {
            throw FluxKitError.malformedCheckpoint("could not create CGImage from decoded latents")
        }
        return cgImage
    }
}
