//
//  ObscurCampaign.swift
//  FluxKit
//
//  Production P1 training campaign over a real-image corpus (the T9 atlas layout):
//  Phase A builds a resumable preprocessed cache (VAE-encoded packed x0 latents +
//  DINOv2 tokens, sharded safetensors), Phase B trains the projection with the
//  flow-matching objective at the distilled model's timesteps, gate-strength jitter
//  (calibrates the influence dial; classical conditioning dropout is a structural
//  no-op for a bias-free projection), LR warmup+cosine, a held-out validation split,
//  checkpoint/resume of optimizer state, and periodic style-transfer previews.
//
//  Honors the atlas dataset's own rules: `anchors` are never trained; staging,
//  rejected, blur, and shot-render pools are excluded.
//

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXNN
import MLXRandom
import ObscurKit
import UniformTypeIdentifiers

public final class ObscurCampaign {
    public struct Config {
        public var modelDir: URL
        public var dinoWeightsURL: URL
        public var datasetRoot: URL
        public var cacheDir: URL
        public var campaignDir: URL
        public var outputURL: URL
        public var maxImages: Int
        public var resolution: Int
        public var steps: Int
        public var learningRate: Float
        public var warmupSteps: Int
        public var valEvery: Int
        public var checkpointEvery: Int
        public var previewEvery: Int
        public var seed: UInt64

        public init(modelDir: URL, dinoWeightsURL: URL, datasetRoot: URL,
                    cacheDir: URL, campaignDir: URL, outputURL: URL,
                    maxImages: Int = 8000, resolution: Int = 256, steps: Int = 20000,
                    learningRate: Float = 1e-4, warmupSteps: Int = 200,
                    valEvery: Int = 500, checkpointEvery: Int = 500,
                    previewEvery: Int = 2000, seed: UInt64 = 0) {
            self.modelDir = modelDir
            self.dinoWeightsURL = dinoWeightsURL
            self.datasetRoot = datasetRoot
            self.cacheDir = cacheDir
            self.campaignDir = campaignDir
            self.outputURL = outputURL
            self.maxImages = maxImages
            self.resolution = resolution
            self.steps = steps
            self.learningRate = learningRate
            self.warmupSteps = warmupSteps
            self.valEvery = valEvery
            self.checkpointEvery = checkpointEvery
            self.previewEvery = previewEvery
            self.seed = seed
        }
    }

    static let shardSize = 256
    /// Atlas pools used for style pretraining (real + generated both count as style).
    static let includedPools = ["art", "photo", "cgi", "human"]
    static let excludedPathParts = ["anchors", "_incoming", "_rejected", "shots", "blur",
                                    ".obscur", "._"]

    let config: Config
    let logSink: (String) -> Void
    private var logHandle: FileHandle?

    public init(config: Config, log: @escaping (String) -> Void) {
        self.config = config
        self.logSink = log
    }

    func log(_ message: String) {
        logSink(message)
        let line = "\(ISO8601DateFormatter().string(from: Date())) \(message)\n"
        if logHandle == nil {
            let url = config.campaignDir.appendingPathComponent("train.log")
            try? FileManager.default.createDirectory(at: config.campaignDir,
                                                     withIntermediateDirectories: true)
            if !FileManager.default.fileExists(atPath: url.path) {
                FileManager.default.createFile(atPath: url.path, contents: nil)
            }
            logHandle = try? FileHandle(forWritingTo: url)
            _ = try? logHandle?.seekToEnd()
        }
        try? logHandle?.write(contentsOf: Data(line.utf8))
    }

    // MARK: - Phase A: dataset cache

    struct CachePlan: Codable {
        var schema = "obscur.p1cache.v1"
        var resolution: Int
        var shardSize: Int
        var sources: [String]      // dataset-relative paths, deterministic order
        var shardCount: Int
    }

    func planURL() -> URL { config.cacheDir.appendingPathComponent("plan.json") }
    func shardURL(_ index: Int) -> URL {
        config.cacheDir.appendingPathComponent(String(format: "shard-%04d.safetensors", index))
    }

    /// Scan the dataset, fix the sample plan, and build any missing shards. Resumable:
    /// existing shards are kept; a plan mismatch (resolution/limit change) rebuilds all.
    func buildCache() throws -> CachePlan {
        try FileManager.default.createDirectory(at: config.cacheDir, withIntermediateDirectories: true)

        var plan: CachePlan
        if let data = try? Data(contentsOf: planURL()),
           let existing = try? JSONDecoder().decode(CachePlan.self, from: data),
           existing.resolution == config.resolution,
           existing.sources.count <= config.maxImages {
            plan = existing
            log("cache plan: reusing (\(plan.sources.count) images)")
        } else {
            let sources = Self.scan(root: config.datasetRoot, limit: config.maxImages,
                                    seed: config.seed)
            guard !sources.isEmpty else {
                throw FluxKitError.malformedCheckpoint("no images found under \(config.datasetRoot.path)")
            }
            plan = CachePlan(resolution: config.resolution, shardSize: Self.shardSize,
                             sources: sources,
                             shardCount: (sources.count + Self.shardSize - 1) / Self.shardSize)
            try JSONEncoder().encode(plan).write(to: planURL())
            log("cache plan: \(sources.count) images → \(plan.shardCount) shards")
        }

        let missing = (0..<plan.shardCount).filter {
            !FileManager.default.fileExists(atPath: shardURL($0).path)
        }
        guard !missing.isEmpty else {
            log("cache: complete (\(plan.shardCount) shards)")
            return plan
        }

        log("cache: building \(missing.count) shard(s)…")
        let encoder = try Flux2VAEEncoder(componentDir: config.modelDir.appendingPathComponent("vae"))
        let dino = ObscurDinov2()
        try dino.loadWeights(url: config.dinoWeightsURL)

        for shardIndex in missing {
            let start = shardIndex * Self.shardSize
            let end = min(start + Self.shardSize, plan.sources.count)
            var tensors: [String: MLXArray] = [:]
            var kept = 0
            for i in start..<end {
                let url = config.datasetRoot.appendingPathComponent(plan.sources[i])
                guard let cg = Self.loadImage(url) else { continue }
                let image = Self.imageTensor(cg, resolution: config.resolution)
                let x0 = encoder.encodePackedNormalized(image)[0].asType(.float16)
                let tokens = dino.patchTokens(ObscurDinoPreprocess.input(cg))[0].asType(.float16)
                eval(x0, tokens)
                tensors["\(i - start).x0"] = x0
                tensors["\(i - start).dino"] = tokens
                kept += 1
            }
            try MLX.save(arrays: tensors, url: shardURL(shardIndex))
            Memory.clearCache()
            log("  shard \(shardIndex + 1)/\(plan.shardCount) (\(kept)/\(end - start) images)")
        }
        return plan
    }

    /// Deterministic scan: included pools only, exclusions applied, sorted then
    /// seed-shuffled so the plan is reproducible.
    static func scan(root: URL, limit: Int, seed: UInt64) -> [String] {
        let extensions: Set<String> = ["jpg", "jpeg", "png", "webp"]
        var paths: [String] = []
        for pool in includedPools {
            let poolURL = root.appendingPathComponent(pool)
            guard let enumerator = FileManager.default.enumerator(
                at: poolURL, includingPropertiesForKeys: nil,
                options: [.skipsHiddenFiles]) else { continue }
            for case let file as URL in enumerator {
                let name = file.lastPathComponent
                guard !name.hasPrefix("._"),
                      extensions.contains(file.pathExtension.lowercased()) else { continue }
                let relative = file.path.replacingOccurrences(of: root.path + "/", with: "")
                guard !excludedPathParts.contains(where: { relative.contains($0) }) else { continue }
                paths.append(relative)
            }
        }
        paths.sort()
        // Seeded Fisher–Yates so art (the largest pool) doesn't dominate the head.
        var rng = SplitMix64(seed: seed &+ 0x9E37_79B9)
        for i in stride(from: paths.count - 1, to: 0, by: -1) {
            let j = Int(rng.next() % UInt64(i + 1))
            paths.swapAt(i, j)
        }
        return Array(paths.prefix(limit))
    }

    static func loadImage(_ url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    /// Center-crop + resize to (1, res, res, 3) NHWC in [-1, 1].
    static func imageTensor(_ cg: CGImage, resolution: Int) -> MLXArray {
        let target = resolution
        let scale = Float(target) / Float(min(cg.width, cg.height))
        let rw = Int((Float(cg.width) * scale).rounded()), rh = Int((Float(cg.height) * scale).rounded())
        var buffer = [UInt8](repeating: 0, count: rw * rh * 4)
        let cs = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: &buffer, width: rw, height: rh, bitsPerComponent: 8,
                                  bytesPerRow: rw * 4, space: cs,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            return MLXArray.zeros([1, target, target, 3])
        }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: rw, height: rh))
        let ox = (rw - target) / 2, oy = (rh - target) / 2
        var pixels = [Float](repeating: 0, count: target * target * 3)
        for y in 0..<target {
            for x in 0..<target {
                let src = ((y + oy) * rw + (x + ox)) * 4
                for c in 0..<3 {
                    pixels[(y * target + x) * 3 + c] = Float(buffer[src + c]) / 127.5 - 1
                }
            }
        }
        return MLXArray(pixels, [1, target, target, 3])
    }

    // MARK: - Phase B: training

    public func train(resume: Bool) throws -> URL {
        let plan = try buildCache()
        try FileManager.default.createDirectory(at: config.campaignDir, withIntermediateDirectories: true)

        // Validation split: the final shard is held out entirely.
        let valShard = plan.shardCount - 1
        let trainShards = Array(0..<max(valShard, 1))
        log("split: \(trainShards.count) train shard(s), 1 val shard")

        log("loading frozen DiT + prompt embedding…")
        let pipeline = Flux2Pipeline(modelDirectory: config.modelDir)
        let (emptyIDs, emptyMask) = try pipeline.tokenize(prompt: " ")
        var promptEmbeds: MLXArray
        do {
            let textEncoder = try Qwen3TextEncoder(componentDir: config.modelDir.appendingPathComponent("text_encoder"))
            promptEmbeds = textEncoder.promptEmbeddings(inputIDs: emptyIDs, attentionMask: emptyMask)
            eval(promptEmbeds)
        }
        Memory.clearCache()
        let dit = try Flux2Transformer(componentDir: config.modelDir.appendingPathComponent("transformer"))
        let txtIDs = Flux2Pipeline.textIDs(count: promptEmbeds.dim(1))

        // Geometry (fixed by cache resolution).
        let packedSide = config.resolution / 16
        let imgIDs = Flux2Pipeline.latentIDs(packedH: packedSide, packedW: packedSide)
        let scheduler = Flux2Scheduler(imageSequenceLength: packedSide * packedSide, steps: 4)
        let sigmas = (0..<4).map { scheduler.sigmas[$0] }

        // Projection params (+ optimizer state), fresh or resumed.
        let projection = ObscurProjection(seed: 0)
        let hooked = projection.hookedLayers
        var params = projection.trainableParameters()
        var m = params.mapValues { MLXArray.zeros(like: $0) }
        var v = params.mapValues { MLXArray.zeros(like: $0) }
        var startStep = 0
        if resume, let restored = try? loadCheckpoint(params: &params, m: &m, v: &v) {
            startStep = restored
            log("resumed from checkpoint at step \(startStep)")
        }
        eval(Array(params.values))

        var rng = SplitMix64(seed: config.seed &+ UInt64(startStep) &* 0x1234_5678)
        var shardCache: (index: Int, tensors: [String: MLXArray])? = nil

        func sample(_ shardIndex: Int, _ item: Int) throws -> (MLXArray, MLXArray)? {
            if shardCache?.index != shardIndex {
                shardCache = (shardIndex, try loadArrays(url: shardURL(shardIndex)))
            }
            guard let x0 = shardCache?.tensors["\(item).x0"],
                  let dino = shardCache?.tensors["\(item).dino"] else { return nil }
            return (x0.expandedDimensions(axis: 0), dino)
        }

        let beta1: Float = 0.9, beta2: Float = 0.999, epsAdam: Float = 1e-8
        var stepInEpoch = 0
        var order: [(Int, Int)] = []

        func reshuffle(epoch: Int) {
            order.removeAll()
            var epochRNG = SplitMix64(seed: config.seed &+ UInt64(epoch) &* 7919)
            var shards = trainShards
            for i in stride(from: shards.count - 1, to: 0, by: -1) {
                shards.swapAt(i, Int(epochRNG.next() % UInt64(i + 1)))
            }
            for s in shards {
                let count = min(Self.shardSize, plan.sources.count - s * Self.shardSize)
                var items = Array(0..<count)
                for i in stride(from: items.count - 1, to: 0, by: -1) {
                    items.swapAt(i, Int(epochRNG.next() % UInt64(i + 1)))
                }
                order.append(contentsOf: items.map { (s, $0) })
            }
            stepInEpoch = 0
        }

        var epoch = 0
        reshuffle(epoch: 0)

        let started = Date()
        var emaLoss: Float = -1

        for step in startStep..<config.steps {
            if stepInEpoch >= order.count {
                epoch += 1
                reshuffle(epoch: epoch)
            }
            let (shardIndex, item) = order[stepInEpoch]
            stepInEpoch += 1
            guard let (x0, dino) = try sample(shardIndex, item) else { continue }

            let sigma = sigmas[Int(rng.next() % 4)]
            let gateValue = 0.5 + Float(rng.next() % 1000) / 2000.0    // U[0.5, 1.0]
            let noiseKey = MLXRandom.key(rng.next())
            let noise = MLXRandom.normal(x0.shape, key: noiseKey).asType(x0.dtype)

            let stepLossFn: (ModuleParameters) -> [MLXArray] = { p in
                let dict = Dictionary(uniqueKeysWithValues: p.flattened())
                let kv = ObscurProjection.projectFunctional(dino, params: dict, hookedLayers: hooked)
                let composed = ObscurComposedKV.singleEntry(kv: kv, tokenCount: ObscurProjection.pooledTokens)
                let gates = ObscurGates.grouped(structure: gateValue, texture: gateValue, layers: hooked)
                let ctx = ObscurInjectionContext(composed: composed, gates: gates, recorder: nil)
                let xt = (1 - sigma) * x0 + sigma * noise
                let vTarget = noise - x0
                let vPred = dit(hidden: xt, encoder: promptEmbeds, timestep: sigma * 1000,
                                imgIDs: imgIDs, txtIDs: txtIDs, adapter: ctx)
                return [(vPred - vTarget).square().mean()]
            }
            let (values, grads) = valueAndGrad(interceptingArguments(stepLossFn))(
                ModuleParameters.unflattened(params), 0)
            let gradDict = Dictionary(uniqueKeysWithValues: grads.flattened())

            // LR schedule: linear warmup → cosine decay to 10%.
            let lr: Float
            if step < config.warmupSteps {
                lr = config.learningRate * Float(step + 1) / Float(config.warmupSteps)
            } else {
                let progress = Float(step - config.warmupSteps)
                    / Float(max(config.steps - config.warmupSteps, 1))
                lr = config.learningRate * (0.1 + 0.9 * 0.5 * (1 + cos(Float.pi * progress)))
            }

            let t = Float(step + 1)
            let bc1 = 1 - pow(beta1, t)
            let bc2 = 1 - pow(beta2, t)
            for (key, g) in gradDict {
                m[key] = beta1 * m[key]! + (1 - beta1) * g
                v[key] = beta2 * v[key]! + (1 - beta2) * (g * g)
                params[key] = params[key]! - lr * (m[key]! / bc1) / (sqrt(v[key]! / bc2) + epsAdam)
            }
            eval(Array(params.values) + Array(m.values) + Array(v.values))

            let loss = values[0].item(Float.self)
            emaLoss = emaLoss < 0 ? loss : 0.98 * emaLoss + 0.02 * loss

            if step % 25 == 0 || step == config.steps - 1 {
                let rate = Double(step - startStep + 1) / max(Date().timeIntervalSince(started), 1)
                let remaining = Double(config.steps - step - 1) / max(rate, 0.01)
                log(String(format: "step %d/%d  loss %.4f  ema %.4f  lr %.2e  eta %.1fh",
                           step, config.steps, loss, emaLoss, lr, remaining / 3600))
            }
            if (step + 1) % config.checkpointEvery == 0 || step == config.steps - 1 {
                try saveCheckpoint(step: step + 1, params: params, m: m, v: v)
            }
            if (step + 1) % config.valEvery == 0 || step == config.steps - 1 {
                let valLoss = try validate(valShard: valShard, plan: plan, params: params,
                                           hooked: hooked, dit: dit, promptEmbeds: promptEmbeds,
                                           imgIDs: imgIDs, txtIDs: txtIDs, sigmas: sigmas)
                log(String(format: "val   %d  loss %.4f", step + 1, valLoss))
            }
            if config.previewEvery > 0, (step + 1) % config.previewEvery == 0 {
                try? preview(step: step + 1, plan: plan, valShard: valShard, params: params,
                             projection: projection, pipeline: pipeline)
            }
            Memory.clearCache()
        }

        projection.setTrainableParameters(params)
        try projection.save(to: config.outputURL)
        let manifest: [String: String] = [
            "schema": ObscurProjection.schema,
            "datasetRoot": config.datasetRoot.path,
            "images": "\(plan.sources.count)",
            "resolution": "\(config.resolution)",
            "steps": "\(config.steps)",
            "finalEMA": String(format: "%.4f", emaLoss),
            "wallclockHours": String(format: "%.2f", Date().timeIntervalSince(started) / 3600),
        ]
        try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted])
            .write(to: config.campaignDir.appendingPathComponent("campaign-manifest.json"))
        log("✓ saved projection → \(config.outputURL.path)")
        return config.outputURL
    }

    // MARK: - Validation & preview

    func validate(valShard: Int, plan: CachePlan, params: [String: MLXArray],
                  hooked: [ObscurLayerRef], dit: Flux2Transformer, promptEmbeds: MLXArray,
                  imgIDs: MLXArray, txtIDs: MLXArray, sigmas: [Float]) throws -> Float {
        let tensors = try loadArrays(url: shardURL(valShard))
        let count = min(24, Self.shardSize)
        var total: Float = 0
        var n = 0
        for i in 0..<count {
            guard let x0raw = tensors["\(i).x0"], let dino = tensors["\(i).dino"] else { continue }
            let x0 = x0raw.expandedDimensions(axis: 0)
            let sigma = sigmas[i % 4]
            let noise = MLXRandom.normal(x0.shape, key: MLXRandom.key(UInt64(5000 + i))).asType(x0.dtype)
            let kv = ObscurProjection.projectFunctional(dino, params: params, hookedLayers: hooked)
            let composed = ObscurComposedKV.singleEntry(kv: kv, tokenCount: ObscurProjection.pooledTokens)
            let gates = ObscurGates.grouped(structure: 0.75, texture: 0.75, layers: hooked)
            let ctx = ObscurInjectionContext(composed: composed, gates: gates, recorder: nil)
            let xt = (1 - sigma) * x0 + sigma * noise
            let vPred = dit(hidden: xt, encoder: promptEmbeds, timestep: sigma * 1000,
                            imgIDs: imgIDs, txtIDs: txtIDs, adapter: ctx)
            let loss = ((vPred - (noise - x0)).square().mean())
            eval(loss)
            total += loss.item(Float.self)
            n += 1
        }
        Memory.clearCache()
        return n > 0 ? total / Float(n) : .nan
    }

    /// Style-transfer preview: fixed prompt/seed conditioned on the first val image.
    func preview(step: Int, plan: CachePlan, valShard: Int, params: [String: MLXArray],
                 projection: ObscurProjection, pipeline: Flux2Pipeline) throws {
        let tensors = try loadArrays(url: shardURL(valShard))
        guard let dino = tensors["0.dino"] else { return }
        projection.setTrainableParameters(params)
        let kv = projection.project(dino)
        let composed = ObscurComposedKV.singleEntry(kv: kv, tokenCount: ObscurProjection.pooledTokens)
        let gates = ObscurGates.grouped(structure: 0.8, texture: 0.8,
                                        layers: projection.hookedLayers)
        let ctx = ObscurInjectionContext(composed: composed, gates: gates, recorder: nil)
        let image = try pipeline.generate(prompt: "a portrait", width: 256, height: 256,
                                          steps: 4, seed: 7, adapter: ctx)
        let url = config.campaignDir.appendingPathComponent(String(format: "preview-%06d.png", step))
        if let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) {
            CGImageDestinationAddImage(dest, image, nil)
            CGImageDestinationFinalize(dest)
            log("  preview → \(url.lastPathComponent)")
        }
        Memory.clearCache()
    }

    // MARK: - Checkpointing (resume, not restart)

    func checkpointURL() -> URL { config.campaignDir.appendingPathComponent("checkpoint.safetensors") }

    func saveCheckpoint(step: Int, params: [String: MLXArray],
                        m: [String: MLXArray], v: [String: MLXArray]) throws {
        var tensors: [String: MLXArray] = ["meta.step": MLXArray(Int32(step))]
        for (k, a) in params { tensors["p.\(k)"] = a }
        for (k, a) in m { tensors["m.\(k)"] = a }
        for (k, a) in v { tensors["v.\(k)"] = a }
        try MLX.save(arrays: tensors, url: checkpointURL())
    }

    func loadCheckpoint(params: inout [String: MLXArray],
                        m: inout [String: MLXArray],
                        v: inout [String: MLXArray]) throws -> Int {
        let tensors = try loadArrays(url: checkpointURL())
        guard let stepArray = tensors["meta.step"] else {
            throw FluxKitError.malformedCheckpoint("checkpoint missing step")
        }
        for key in params.keys {
            if let p = tensors["p.\(key)"] { params[key] = p }
            if let mm = tensors["m.\(key)"] { m[key] = mm }
            if let vv = tensors["v.\(key)"] { v[key] = vv }
        }
        return Int(stepArray.item(Int32.self))
    }
}

/// Deterministic small RNG for shuffles/sampling (keeps the campaign reproducible
/// without threading MLX keys through host-side control flow).
struct SplitMix64 {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Adapts a closure over parameters into the shape `valueAndGrad` expects.
private func interceptingArguments(
    _ f: @escaping (ModuleParameters) -> [MLXArray]
) -> (ModuleParameters, Int) -> [MLXArray] {
    { params, _ in f(params) }
}
