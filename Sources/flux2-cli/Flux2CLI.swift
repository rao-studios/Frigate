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
}
