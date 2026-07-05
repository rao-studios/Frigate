//
//  main.swift
//  flux2-cli
//
//  macOS verification harness for FluxKit: downloads the FLUX.2 Klein 4-bit checkpoint
//  via Hub (resumable) and generates a PNG. Used to visually verify the port before it
//  ships inside the iOS app.
//
//    swift run -c release flux2-cli --prompt "a lighthouse at dawn" --out /tmp/flux2.png
//

import ArgumentParser
import CoreGraphics
import FluxKit
import MLX
import MLXRandom
import ObscurKit
import Foundation
import Hub
import ImageIO
import UniformTypeIdentifiers

@main
struct Flux2CLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "flux2-cli",
        abstract: "Generate an image with FLUX.2 Klein 4-bit (MLX)."
    )

    @Option(name: .long, help: "Prompt text.")
    var prompt: String = ""

    @Option(name: .long, help: "Output PNG path.")
    var out: String = "/tmp/flux2.png"

    @Option(name: .long, help: "Image width.")
    var width: Int = 512

    @Option(name: .long, help: "Image height.")
    var height: Int = 512

    @Option(name: .long, help: "Denoising steps.")
    var steps: Int = 4

    @Option(name: .long, help: "Random seed.")
    var seed: UInt64 = 42

    @Option(name: .long, help: "Model directory (defaults to Hub download location).")
    var modelDir: String?

    @Flag(name: .long, help: "Run the Signet adapter gate test: zero-gate bit-exactness, attribution sum, influence sweep.")
    var testAdapter = false

    @Flag(name: .long, help: "Bootstrap P1: pretrain the Signet projection and save obscur.projection.v1.safetensors.")
    var trainProjection = false

    @Option(name: .long, help: "P1 training steps.")
    var trainSteps: Int = 400

    @Option(name: .long, help: "DINOv2 weights (dinov2_small.safetensors).")
    var dinoWeights: String = "/Users/ritesh/Documents/projects/seer/Ra/applications/CLR/Sources/CLRCore/Resources/dinov2_small.safetensors"

    @Flag(name: .long, help: "Production P1: train the projection on a real-image dataset (atlas layout).")
    var trainCampaign = false

    @Option(name: .long, help: "Campaign dataset root (e.g. /Volumes/T9/datasets/atlas).")
    var dataset: String = "/Volumes/T9/datasets/atlas"

    @Option(name: .long, help: "Preprocessed cache dir (default <dataset>/.obscur-p1-cache).")
    var cacheDir: String?

    @Option(name: .long, help: "Campaign dir for checkpoints/logs/previews.")
    var campaignDir: String = NSString(string: "~/Documents/obscur-p1-campaign").expandingTildeInPath

    @Option(name: .long, help: "Campaign: max dataset images.")
    var maxImages: Int = 8000

    @Option(name: .long, help: "Campaign: optimizer steps.")
    var campaignSteps: Int = 20000

    @Flag(name: .long, help: "Campaign: resume from the checkpoint in --campaign-dir.")
    var resume = false

    @Option(name: .long, help: "P3: reference image to ingest as a 1-image Signet for the trained-vs-random comparison.")
    var testProjection: String?

    @Option(name: .long, help: "P3: trained projection artifact to compare against random.")
    var projection: String?

    /// Diverse subjects × styles for the bootstrap P1 self-supervised dataset.
    static let trainingPrompts: [String] = {
        let subjects = ["a mountain lake", "a city street at night", "a bowl of fruit",
                        "a portrait of a woman", "a red sports car", "a forest path",
                        "a cup of coffee", "a sailboat on the sea", "a field of sunflowers",
                        "an old stone bridge", "a snowy village", "a desert at dusk"]
        let styles = ["photorealistic", "oil painting", "watercolor", "pencil sketch",
                      "anime style", "cyberpunk neon", "impressionist", "low-poly 3D render"]
        var prompts: [String] = []
        for (i, s) in subjects.enumerated() {
            prompts.append("\(s), \(styles[i % styles.count])")
            prompts.append("\(s), \(styles[(i + 3) % styles.count])")
        }
        return prompts
    }()

    /// MLX resolves its Metal kernels from `<binary dir>/mlx.metallib` first. Frigate
    /// ships no metallib (CLR's resource bundle carries it), so `swift build` output
    /// lacks one and any clean wipes a hand-copied file. Self-heal at launch: copy from
    /// `FLUX2_METALLIB` (env) or CLR's known resource path when missing.
    static func ensureMetallib() {
        let binaryDir = URL(fileURLWithPath: CommandLine.arguments[0])
            .resolvingSymlinksInPath().deletingLastPathComponent()
        let target = binaryDir.appendingPathComponent("mlx.metallib")
        guard !FileManager.default.fileExists(atPath: target.path) else { return }

        let candidates = [
            ProcessInfo.processInfo.environment["FLUX2_METALLIB"],
            "/Users/ritesh/Documents/projects/seer/Ra/applications/CLR/Sources/CLRCore/Resources/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib",
        ].compactMap { $0 }

        for candidate in candidates where FileManager.default.fileExists(atPath: candidate) {
            if (try? FileManager.default.copyItem(at: URL(fileURLWithPath: candidate), to: target)) != nil {
                print("• restored mlx.metallib → \(target.path)")
                return
            }
        }
        print("⚠️ mlx.metallib not found next to the binary and no source available — set FLUX2_METALLIB to a metallib path if MLX fails to start.")
    }

    func run() async throws {
        Self.ensureMetallib()
        let repoID = "mlx-community/flux2-klein-4b-4bit"
        let dir: URL
        if let modelDir {
            dir = URL(fileURLWithPath: modelDir)
        } else {
            let hub = HubApi()
            print("Ensuring checkpoint \(repoID) is downloaded…")
            dir = try await hub.snapshot(from: repoID) { progress, speed in
                let pct = Int(progress.fractionCompleted * 100)
                let mbps = (speed ?? 0) / 1_048_576
                print("\r  download \(pct)% (\(String(format: "%.1f", mbps)) MB/s)   ", terminator: "")
                fflush(stdout)
            }
            print("\n  checkpoint at \(dir.path)")
        }

        let pipeline = Flux2Pipeline(modelDirectory: dir)

        if trainCampaign {
            let datasetURL = URL(fileURLWithPath: dataset)
            let campaignURL = URL(fileURLWithPath: campaignDir)
            let cacheURL = cacheDir.map { URL(fileURLWithPath: $0) }
                ?? datasetURL.appendingPathComponent(".obscur-p1-cache")
            let outputURL = out.hasSuffix(".safetensors")
                ? URL(fileURLWithPath: out)
                : campaignURL.appendingPathComponent("obscur.projection.v1.safetensors")
            let campaign = ObscurCampaign(
                config: .init(modelDir: dir,
                              dinoWeightsURL: URL(fileURLWithPath: dinoWeights),
                              datasetRoot: datasetURL,
                              cacheDir: cacheURL,
                              campaignDir: campaignURL,
                              outputURL: outputURL,
                              maxImages: maxImages,
                              steps: campaignSteps)) { print($0); fflush(stdout) }
            _ = try campaign.train(resume: resume)
            return
        }

        if trainProjection {
            let trainer = ObscurProjectionTrainer(modelDirectory: dir,
                                                  dinoWeightsURL: URL(fileURLWithPath: dinoWeights))
            let start = Date()
            _ = try trainer.train(prompts: Self.trainingPrompts, steps: trainSteps,
                                  outputURL: URL(fileURLWithPath: out)) { print($0) }
            print(String(format: "✓ P1 training done in %.0fs", Date().timeIntervalSince(start)))
            return
        }

        if let refPath = testProjection {
            try runProjectionTest(pipeline: pipeline, refPath: refPath)
            return
        }

        if testAdapter {
            try runAdapterGateTest(pipeline: pipeline)
            return
        }

        let start = Date()
        let image = try pipeline.generate(prompt: prompt, width: width, height: height,
                                          steps: steps, seed: seed) { phase in
            switch phase {
            case .loadingTextEncoder: print("• loading text encoder…")
            case .encodingPrompt: print("• encoding prompt…")
            case .loadingTransformer: print("• loading transformer…")
            case .denoising(let step, let total): print("• denoise step \(step)/\(total)")
            case .decoding: print("• decoding latents…")
            }
        }
        let elapsed = Date().timeIntervalSince(start)

        let url = URL(fileURLWithPath: out)
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw ValidationError("cannot create \(out)")
        }
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
        print("✓ wrote \(out) (\(image.width)×\(image.height)) in \(String(format: "%.1f", elapsed))s")
    }

    // MARK: - P3 projection validation (trained vs random, visual style transfer)

    /// Ingest a reference image as a 1-entry Signet, then generate the SAME neutral
    /// prompt/seed three ways — base (no adapter), random projection, trained projection —
    /// so the trained projection's style-transfer can be compared against noise.
    func runProjectionTest(pipeline: Flux2Pipeline, refPath: String) throws {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: refPath) as CFURL, nil),
              let refImage = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw ValidationError("cannot read reference image \(refPath)")
        }
        let dino = ObscurDinov2()
        try dino.loadWeights(url: URL(fileURLWithPath: dinoWeights))
        let refTokens = dino.patchTokens(ObscurDinoPreprocess.input(refImage))[0].asType(.float16)
        MLX.eval(refTokens)

        func bank(_ proj: ObscurProjection) -> ObscurInjectionContext {
            let kv = proj.project(refTokens)
            let composed = ObscurComposedKV.singleEntry(kv: kv, tokenCount: ObscurProjection.pooledTokens)
            let gates = ObscurGates.grouped(structure: 0.8, texture: 0.8, layers: proj.hookedLayers)
            return ObscurInjectionContext(composed: composed, gates: gates, recorder: nil)
        }

        func gen(_ adapter: ObscurInjectionContext?, _ suffix: String) throws {
            let image = try pipeline.generate(prompt: prompt.isEmpty ? "a photograph" : prompt,
                                              width: 256, height: 256, steps: steps, seed: seed,
                                              adapter: adapter)
            let url = URL(fileURLWithPath: out.replacingOccurrences(of: ".png", with: "-\(suffix).png"))
            if let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) {
                CGImageDestinationAddImage(dest, image, nil)
                CGImageDestinationFinalize(dest)
                print("  wrote \(url.lastPathComponent)")
            }
        }

        print("• baseline (no Signet)…");     try gen(nil, "base")
        print("• random projection…");        try gen(bank(ObscurProjection(seed: 999)), "random")
        if let projPath = projection {
            print("• trained projection…")
            let trained = try ObscurProjection.load(from: URL(fileURLWithPath: projPath))
            try gen(bank(trained), "trained")
        }
        print("✓ projection test complete")
    }

    // MARK: - Signet adapter gate test (spec build-order gate 4)

    /// Builds a synthetic 3-entry bank (random DINO-shaped tokens), then verifies:
    /// 1. zero gates → bit-identical pixels vs. no adapter installed;
    /// 2. influence > 0 → attribution shares sum to 1 over the adapter branch;
    /// 3. influence sweep visibly changes pixels (monotonic pixel delta).
    func runAdapterGateTest(pipeline: Flux2Pipeline) throws {
        let projection = ObscurProjection()
        let bank = ObscurBank(corpusID: "gate-test")
        for i in 0..<3 {
            let tokens = MLXRandom.normal([256, ObscurProjection.dinoDim],
                                          key: MLXRandom.key(UInt64(100 + i))).asType(.float16)
            bank.insert(ObscurEntry(id: "entry-\(i)", corpusID: bank.corpusID,
                                    dinoTokens: tokens,
                                    provenance: .init(sourceCaptureID: nil,
                                                      signatureCID: nil, capturedAt: nil)))
        }
        let composed = try ObscurComposer.compose(banks: [bank], selection: .all,
                                                  projection: projection)
        print("composed bank: \(composed.tokenCount) tokens over \(composed.spans.count) entries")

        func writePNG(_ image: CGImage, suffix: String) {
            let url = URL(fileURLWithPath: out.replacingOccurrences(of: ".png", with: "-\(suffix).png"))
            guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
            CGImageDestinationAddImage(dest, image, nil)
            CGImageDestinationFinalize(dest)
            print("  wrote \(url.path)")
        }

        func pixels(influence: Float, recorder: ObscurAttributionRecorder?) throws -> [UInt8] {
            var adapter: ObscurInjectionContext?
            let gates = ObscurGates.grouped(structure: influence, texture: influence,
                                            layers: projection.hookedLayers)
            adapter = ObscurInjectionContext(composed: composed, gates: gates, recorder: recorder)
            let image = try pipeline.generate(prompt: prompt, width: 256, height: 256,
                                              steps: steps, seed: seed, adapter: adapter)
            writePNG(image, suffix: String(format: "inf%03d", Int(influence * 100)))
            guard let data = image.dataProvider?.data as Data? else { return [] }
            return [UInt8](data)
        }

        print("• baseline (no adapter)…")
        let baseImage = try pipeline.generate(prompt: prompt, width: 256, height: 256,
                                              steps: steps, seed: seed)
        writePNG(baseImage, suffix: "base")
        let base = [UInt8]((baseImage.dataProvider?.data as Data?) ?? Data())

        print("• zero-gate install…")
        let zero = try pixels(influence: 0, recorder: nil)
        print(zero == base ? "  PASS: zero-gate bit-exact (\(base.count) bytes identical)"
                           : "  FAIL: zero-gate pixels differ!")

        print("• influence sweep…")
        var previousDelta = 0.0
        for influence: Float in [0.25, 0.75, 1.0] {
            let recorder = ObscurAttributionRecorder(spans: composed.spans,
                                                     gridWidth: 16, gridHeight: 16)
            let px = try pixels(influence: influence, recorder: recorder)
            let delta = zip(px, base).reduce(0.0) { $0 + abs(Double($1.0) - Double($1.1)) }
                / Double(max(base.count, 1))
            let report = recorder.report(generationID: UUID(), modelVersion: "klein-4b-4bit",
                                         gates: .grouped(structure: influence, texture: influence,
                                                         layers: projection.hookedLayers),
                                         corpora: [bank.corpusID: bank.logitBias])
            let shareSum = report.entries.values.reduce(0) { $0 + $1.attentionShare }
            print(String(format: "  influence %.2f: mean pixel delta %.3f, attribution sum %.4f %@",
                         influence, delta, shareSum,
                         abs(shareSum - 1) < 1e-3 ? "PASS" : "FAIL"))
            for (id, share) in report.entries.sorted(by: { $0.key < $1.key }) {
                print(String(format: "    %@ share %.4f", id, share.attentionShare))
            }

            // Spatial map invariants: per-region shares sum to 1 (where relief > 0), and
            // the per-entry totals derived from the spatial map match the aggregate report.
            if let map = recorder.influenceMap() {
                let e = map.entryIDs.count
                var worstRegionSum: Float = 1
                for i in 0..<(map.gridWidth * map.gridHeight) where map.regionRelief[i] > 0 {
                    let sum = map.shares(atRegion: i).reduce(0, +)
                    if abs(sum - 1) > abs(worstRegionSum - 1) { worstRegionSum = sum }
                }
                var worstEntryDelta: Float = 0
                for (j, id) in map.entryIDs.enumerated() {
                    let aggregate = report.entries[id]?.attentionShare ?? 0
                    worstEntryDelta = max(worstEntryDelta, abs(map.entryTotals[j] - aggregate))
                }
                let reliefOK = map.regionRelief.max().map { abs($0 - 1) < 1e-4 } ?? false
                print(String(format: "  spatial: worst region sum %.5f %@ · spatial↔aggregate Δ %.5f %@ · relief max %@",
                             worstRegionSum, abs(worstRegionSum - 1) < 1e-3 ? "PASS" : "FAIL",
                             worstEntryDelta, worstEntryDelta < 1e-3 ? "PASS" : "FAIL",
                             reliefOK ? "PASS" : "FAIL"))
            } else {
                print("  spatial: FAIL (no map produced)")
            }

            if delta <= previousDelta {
                print("  WARN: pixel delta not increasing with influence")
            }
            previousDelta = delta
        }
        print("✓ adapter gate test complete")
    }
}
