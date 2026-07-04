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
    var prompt: String

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

    func run() async throws {
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

        func pixels(influence: Float, recorder: ObscurAttributionRecorder?) throws -> [UInt8] {
            var adapter: ObscurInjectionContext?
            let gates = ObscurGates.grouped(structure: influence, texture: influence,
                                            layers: projection.hookedLayers)
            adapter = ObscurInjectionContext(composed: composed, gates: gates, recorder: recorder)
            let image = try pipeline.generate(prompt: prompt, width: 256, height: 256,
                                              steps: steps, seed: seed, adapter: adapter)
            guard let data = image.dataProvider?.data as Data? else { return [] }
            return [UInt8](data)
        }

        print("• baseline (no adapter)…")
        let baseImage = try pipeline.generate(prompt: prompt, width: 256, height: 256,
                                              steps: steps, seed: seed)
        let base = [UInt8]((baseImage.dataProvider?.data as Data?) ?? Data())

        print("• zero-gate install…")
        let zero = try pixels(influence: 0, recorder: nil)
        print(zero == base ? "  PASS: zero-gate bit-exact (\(base.count) bytes identical)"
                           : "  FAIL: zero-gate pixels differ!")

        print("• influence sweep…")
        var previousDelta = 0.0
        for influence: Float in [0.25, 0.75] {
            let recorder = ObscurAttributionRecorder(spans: composed.spans)
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
            if delta <= previousDelta {
                print("  WARN: pixel delta not increasing with influence")
            }
            previousDelta = delta
        }
        print("✓ adapter gate test complete")
    }
}
