//
//  RegionBackboneTests.swift
//  FrigateVisionAXTests
//
//  WHAT: The classifier split at its backbone — prepare in C, any backbone, the head in C —
//        and the MLX backbone that runs in that slot on Metal.
//  PIN:  TWO GATES. The split path and the choice of backbone run everywhere and never touch
//        MLX: they put the engine's own ONNX backbone behind the seam, so the answer can be
//        held to EQUALITY with the single call. The MLX backbone needs a metallib in the test
//        bundle — without one the first GPU op aborts the whole test process — so those tests
//        run only with FRIGATE_MLX_TESTS=1 after scripts/build-metallib.sh, the gate
//        BatchedImageOpsTests uses.
//

import CVisionAX
import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import FrigateVisionAX

private let mlxTestsEnabled = ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"

@Suite struct RegionBackboneTests {

    // MARK: - Fixtures

    struct Expected: Codable {
        var image: String
        var boxes: [[Int]]
        var expectedClassIndex: [Int]
        var expectedRole: [String]
        var probabilities: [[Double]]
    }

    static func fixtureURL(_ name: String) throws -> URL {
        try #require(
            Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures"),
            "missing fixture \(name)")
    }

    static func tinySpecURL() throws -> URL { try fixtureURL("tiny-classifier.json") }

    static func tinySpec() throws -> ClassifierSpec {
        try AXTreeJSON.decode(ClassifierSpec.self, from: Data(contentsOf: tinySpecURL()))
    }

    static func tinyExpected() throws -> Expected {
        try AXTreeJSON.decode(
            Expected.self, from: Data(contentsOf: fixtureURL("tiny-classifier.expected.json")))
    }

    static func tinyImage() throws -> CGImage {
        let url = try fixtureURL("tiny-classifier.image.png")
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    static func boxes(_ expected: Expected) -> [CGRect] {
        expected.boxes.map { CGRect(x: $0[0], y: $0[1], width: $0[2], height: $0[3]) }
    }

    /// A backbone that answers with a shape the tiny head (8 channels, stride 8) was not built for.
    final class MisshapenBackbone: RegionBackbone, @unchecked Sendable {
        let name = "misshapen"
        let channels: Int
        let heightDelta: Int

        init(channels: Int = 8, heightDelta: Int = 0) {
            self.channels = channels
            self.heightDelta = heightDelta
        }

        func features(
            image: UnsafeBufferPointer<Float>, height: Int, width: Int
        ) throws -> RegionFeatures {
            let rows = height / 8 + heightDelta
            let columns = width / 8
            return RegionFeatures(
                values: [Float](repeating: 0, count: channels * rows * columns),
                channels: channels, height: rows, width: columns)
        }
    }

    // MARK: - The split path, on the ONNX backbone — everywhere

    @Test func theSplitPathAnswersExactlyLikeTheSingleCall() throws {
        let reference = try RegionClassifier(specURL: Self.tinySpecURL(), backbone: .onnx)
        let split = try RegionClassifier(
            specURL: Self.tinySpecURL(), injecting: ONNXRegionBackbone(reference))
        let image = try Self.tinyImage()
        let boxes = Self.boxes(try Self.tinyExpected()) + [CGRect(x: 16, y: 16, width: 96, height: 48)]
        let engine = try VisionEngine()

        let single = try engine.classifyRegions(in: image, boxes: boxes, using: reference)
        let viaSplit = try engine.classifyRegions(in: image, boxes: boxes, using: split)

        #expect(viaSplit == single, "the same features through the same head must answer bit for bit")
        #expect(split.backboneDescription == "onnx-reference")
        #expect(split.runCounts.backbone == 0, "the split classifier ran its own ONNX backbone")
        #expect(split.runCounts.head == 1)
    }

    @Test func theSplitPathStillMatchesPython() throws {
        let reference = try RegionClassifier(specURL: Self.tinySpecURL(), backbone: .onnx)
        let split = try RegionClassifier(
            specURL: Self.tinySpecURL(), injecting: ONNXRegionBackbone(reference))
        let expected = try Self.tinyExpected()
        let labels = try VisionEngine().classifyRegions(
            in: try Self.tinyImage(), boxes: Self.boxes(expected), using: split)

        #expect(labels.count == expected.expectedClassIndex.count)
        for (index, label) in labels.enumerated() {
            #expect(label.classIndex == expected.expectedClassIndex[index])
            let reference = expected.probabilities[index][label.classIndex]
            #expect(abs(label.confidence - reference) < 1e-4,
                    "box \(index): \(label.confidence) vs Python's \(reference)")
        }
    }

    @Test func aMapOfTheWrongSizeIsRefusedByName() throws {
        let split = try RegionClassifier(
            specURL: Self.tinySpecURL(), injecting: MisshapenBackbone(heightDelta: -1))
        do {
            _ = try VisionEngine().classifyRegions(
                in: try Self.tinyImage(), boxes: Self.boxes(try Self.tinyExpected()), using: split)
            Issue.record("a map one row short reached the head")
        } catch let error as RegionClassifierError {
            #expect("\(error)".contains("at stride 8"), "\(error)")
        }
    }

    @Test func aMapOfTheWrongWidthIsRefusedByTheHead() throws {
        let split = try RegionClassifier(
            specURL: Self.tinySpecURL(), injecting: MisshapenBackbone(channels: 7))
        do {
            _ = try VisionEngine().classifyRegions(
                in: try Self.tinyImage(), boxes: Self.boxes(try Self.tinyExpected()), using: split)
            Issue.record("a 7-channel map reached an 8-channel head")
        } catch let error as VisionEngineError {
            #expect("\(error)".contains("channels"), "\(error)")
        }
    }

    // MARK: - Choosing the backbone — everywhere, without touching MLX

    @Test func theEnvironmentOverridesOnlyAutomatic() throws {
        let spec = try Self.tinySpec()
        let directory = try Self.tinySpecURL().deletingLastPathComponent()

        let forced = BackboneSelection.resolve(
            .automatic, spec: spec, modelDirectory: directory,
            environment: ["FRIGATE_VISION_BACKBONE": "onnx"], metalLibrary: { nil })
        #expect(forced.backbone == nil)
        #expect(forced.description == "onnx-cpu (FRIGATE_VISION_BACKBONE=onnx)")

        let asked = BackboneSelection.resolve(
            .onnx, spec: spec, modelDirectory: directory,
            environment: ["FRIGATE_VISION_BACKBONE": "mlx"], metalLibrary: { nil })
        #expect(asked.description == "onnx-cpu", "an explicit .onnx is not the environment's to change")
    }

    @Test func noMetallibKeepsTheONNXBackboneAndSaysSo() throws {
        let choice = BackboneSelection.resolve(
            .mlx, spec: try Self.tinySpec(),
            modelDirectory: try Self.tinySpecURL().deletingLastPathComponent(),
            environment: [:], metalLibrary: { nil })
        #expect(choice.backbone == nil)
        #expect(choice.description.hasPrefix("onnx-cpu (mlx requested, but no mlx.metallib"),
                "\(choice.description)")
    }

    @Test func aModelWithoutMLXWeightsStaysOnItsONNXBackbone() throws {
        // A metallib that "exists" gets past the first check; the tiny model ships no MLX
        // weights, which is where the choice must stop — before anything loads.
        let choice = BackboneSelection.resolve(
            .automatic, spec: try Self.tinySpec(),
            modelDirectory: try Self.tinySpecURL().deletingLastPathComponent(),
            environment: [:], metalLibrary: { URL(fileURLWithPath: "/dev/null") })
        #expect(choice.backbone == nil)
        #expect(choice.description.contains("no MLX backbone"), "\(choice.description)")
    }

    @Test func theShippedMLXWeightsBelongToTheShippedModel() throws {
        guard let models = RegionClassifier.resourceBundleLocation() else { return }
        let spec = try AXTreeJSON.decode(
            ClassifierSpec.self,
            from: Data(contentsOf: models.appendingPathComponent("region-classifier.json")))
        let name = try #require(
            spec.files.backboneMLX,
            "the bundled model ships no MLX backbone — run `vxtrain export-mlx` on it")
        let metadata = try SafetensorsHeader.metadata(at: models.appendingPathComponent(name))
        #expect(metadata["arch"] == BackboneSelection.architecture)
        #expect(metadata["source_sha256"] == spec.sha256?["backbone"])
        #expect(
            BackboneSelection.availability(
                spec: spec, modelDirectory: models, metalLibrary: URL(fileURLWithPath: "/dev/null"))
                == .available(models.appendingPathComponent(name)))
    }

    @Test func weightsFromAnotherExportAreRefused() throws {
        guard let models = RegionClassifier.resourceBundleLocation() else { return }
        var spec = try AXTreeJSON.decode(
            ClassifierSpec.self,
            from: Data(contentsOf: models.appendingPathComponent("region-classifier.json")))
        guard spec.files.backboneMLX != nil else { return }
        spec.sha256?["backbone"] = String(repeating: "0", count: 64)
        let availability = BackboneSelection.availability(
            spec: spec, modelDirectory: models, metalLibrary: URL(fileURLWithPath: "/dev/null"))
        guard case .unavailable(let reason) = availability else {
            Issue.record("weights converted from another backbone were accepted")
            return
        }
        #expect(reason.contains("different ONNX backbone"), "\(reason)")
    }

    @Test func automaticSaysWhatItChose() throws {
        guard let classifier = try RegionClassifier.bundled() else { return }
        if MetalLibrary.locate() == nil {
            #expect(classifier.backbone == nil)
            #expect(classifier.backboneDescription.contains("metallib"),
                    "\(classifier.backboneDescription)")
        } else if ProcessInfo.processInfo.environment[BackboneSelection.environmentKey] == nil {
            #expect(classifier.backboneDescription == "mlx-metal",
                    "\(classifier.backboneDescription)")
        }
    }

    // MARK: - The MLX backbone — FRIGATE_MLX_TESTS=1, after scripts/build-metallib.sh

    static let paritySizes = [
        CGSize(width: 320, height: 200),    // no resize, heavy padding
        CGSize(width: 900, height: 752),    // the bench captures' size
        CGSize(width: 1280, height: 800),
        CGSize(width: 1920, height: 1080),  // INTER_AREA down to 1600, the fractional path
    ]

    /// SyntheticScreen's layout, scaled to `size`.
    static func page(_ size: CGSize) -> CGImage {
        let base = SyntheticScreen.Layout.imageSize
        let sx = size.width / base.width
        let sy = size.height / base.height
        let rects = [
            SyntheticScreen.Layout.window, SyntheticScreen.Layout.toolbar,
            SyntheticScreen.Layout.content, SyntheticScreen.Layout.button,
        ].map { CGRect(x: $0.minX * sx, y: $0.minY * sy, width: $0.width * sx, height: $0.height * sy) }
        return SyntheticScreen.image(rects: rects, size: size)
    }

    static func withPreparedTensor(
        _ image: CGImage, classifier: RegionClassifier,
        _ body: (UnsafeBufferPointer<Float>, Int, Int) throws -> Void
    ) throws {
        let buffer = try #require(VisionImageBuffer(image: image))
        var prepared: OpaquePointer?
        let status = buffer.withImageView { view -> vx_status in
            var view = view
            return vx_classifier_prepare_image(classifier.handle, &view, &prepared)
        }
        try #require(status == VX_OK, "\(classifier.lastError)")
        let handle = try #require(prepared)
        defer { vx_prepared_image_free(handle) }
        var height: Int32 = 0
        var width: Int32 = 0
        let tensor = try #require(vx_prepared_image_tensor(handle, &height, &width))
        try body(
            UnsafeBufferPointer(start: tensor, count: 3 * Int(height) * Int(width)),
            Int(height), Int(width))
    }

    static func difference(_ got: [Float], _ want: [Float]) -> (max: Double, mean: Double, scale: Double) {
        var largest = 0.0
        var total = 0.0
        var scale = 0.0
        for (a, b) in zip(got, want) {
            let delta = Double(abs(a - b))
            largest = max(largest, delta)
            total += delta
            scale = max(scale, Double(abs(b)))
        }
        return (largest, total / Double(max(1, min(got.count, want.count))), scale)
    }

    @Test(.enabled(if: mlxTestsEnabled)) func theBundledModelRunsOnMetal() throws {
        let classifier = try #require(try RegionClassifier.bundled(backbone: .mlx))
        #expect(classifier.backboneDescription == "mlx-metal", "\(classifier.backboneDescription)")
    }

    @Test(.enabled(if: mlxTestsEnabled)) func mlxFeaturesMatchTheONNXBackbone() throws {
        let cpu = try #require(try RegionClassifier.bundled(backbone: .onnx))
        let gpu = try #require(try RegionClassifier.bundled(backbone: .mlx))
        let metal = try #require(gpu.backbone, "\(gpu.backboneDescription)")
        let reference = ONNXRegionBackbone(cpu)

        for size in Self.paritySizes {
            try Self.withPreparedTensor(Self.page(size), classifier: cpu) { tensor, height, width in
                let want = try reference.features(image: tensor, height: height, width: width)
                let got = try metal.features(image: tensor, height: height, width: width)
                #expect(got.channels == want.channels && got.height == want.height
                        && got.width == want.width)
                let drift = Self.difference(got.values, want.values)
                print(String(
                    format: "[parity] %4.0fx%-4.0f -> %dx%dx%d  max|d| %.2e  mean|d| %.2e  max|ref| %.2f",
                    size.width, size.height, want.channels, want.height, want.width,
                    drift.max, drift.mean, drift.scale))
                #expect(drift.max <= 1e-3 * max(1, drift.scale), "features drift at \(size): \(drift.max)")
            }
        }
    }

    @Test(.enabled(if: mlxTestsEnabled)) func mlxProbabilitiesMatchTheCPUClassifier() throws {
        let cpu = try #require(try RegionClassifier.bundled(backbone: .onnx))
        let gpu = try #require(try RegionClassifier.bundled(backbone: .mlx))
        try #require(gpu.backbone != nil, "\(gpu.backboneDescription)")
        let engine = try VisionEngine()

        var compared = 0
        var disagreements = 0
        var worst = 0.0
        for size in Self.paritySizes {
            let image = Self.page(size)
            let detection = try engine.detectRegions(in: image, title: "parity")
            let boxes = RegionRelabeler.classifiableNodes(in: detection.window).map(\.frame)
            guard !boxes.isEmpty else { continue }
            let want = try engine.classifyRegions(in: image, boxes: boxes, using: cpu)
            let got = try engine.classifyRegions(in: image, boxes: boxes, using: gpu)
            for (a, b) in zip(want, got) {
                compared += 1
                if a.classIndex == b.classIndex {
                    worst = max(worst, abs(a.confidence - b.confidence))
                } else if abs(a.confidence - b.confidence) > 1e-3 {
                    // Two different winners at clearly different confidences: not a near-tie
                    // that fp32 accumulation order could flip, but a real disagreement.
                    disagreements += 1
                }
            }
        }
        print("[parity] \(compared) boxes, max |dprob| \(worst), \(disagreements) argmax disagreements")
        #expect(compared > 0)
        #expect(worst <= 1e-3)
        #expect(disagreements == 0)
        #expect(gpu.runCounts.backbone == 0, "the MLX classifier ran its ONNX backbone")
    }

    @Test(.enabled(if: mlxTestsEnabled)) func manyThreadsShareOneMLXClassifier() throws {
        let gpu = try #require(try RegionClassifier.bundled(backbone: .mlx))
        let engine = try VisionEngine()
        let image = Self.page(CGSize(width: 900, height: 752))
        let boxes = RegionRelabeler.classifiableNodes(
            in: try engine.detectRegions(in: image, title: "threads").window).map(\.frame)
        let first = try engine.classifyRegions(in: image, boxes: boxes, using: gpu)

        let lock = NSLock()
        var answers: [[RegionLabel]] = []
        DispatchQueue.concurrentPerform(iterations: 8) { _ in
            let labels = (try? engine.classifyRegions(in: image, boxes: boxes, using: gpu)) ?? []
            lock.lock()
            answers.append(labels)
            lock.unlock()
        }
        #expect(answers.count == 8)
        #expect(answers.allSatisfy { $0 == first })
    }

    /// `VISIONAX_BENCH=/dir/of/pngs FRIGATE_MLX_TESTS=1 swift test --filter benchmarkRegionBackbone`
    @Test(.enabled(if: mlxTestsEnabled)) func benchmarkRegionBackbone() throws {
        guard let root = ProcessInfo.processInfo.environment["VISIONAX_BENCH"] else { return }
        let cpu = try #require(try RegionClassifier.bundled(backbone: .onnx))
        let gpu = try #require(try RegionClassifier.bundled(backbone: .mlx))
        let engine = try VisionEngine()
        let files = try FileManager.default.contentsOfDirectory(atPath: root)
            .filter { $0.hasSuffix(".png") }.sorted()

        func milliseconds(_ body: () throws -> Void) rethrows -> Double {
            let started = ContinuousClock.now
            try body()
            return started.duration(to: ContinuousClock.now).milliseconds
        }
        func median(_ repeats: Int, _ body: () throws -> Void) rethrows -> Double {
            var samples: [Double] = []
            for _ in 0..<repeats { samples.append(try milliseconds(body)) }
            return samples.sorted()[samples.count / 2]
        }

        print("[bench] classify phase, median of 7 — \(gpu.backboneDescription) vs \(cpu.backboneDescription)")
        var largest: (image: CGImage, boxes: [CGRect])?
        for file in files {
            let url = URL(fileURLWithPath: root).appendingPathComponent(file)
            guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { continue }
            let boxes = RegionRelabeler.classifiableNodes(
                in: try engine.detectRegions(in: image, title: file).window).map(\.frame)
            let first = try milliseconds { _ = try engine.classifyRegions(in: image, boxes: boxes, using: gpu) }
            let onCPU = try median(7) { _ = try engine.classifyRegions(in: image, boxes: boxes, using: cpu) }
            let onGPU = try median(7) { _ = try engine.classifyRegions(in: image, boxes: boxes, using: gpu) }
            let label = String(file.prefix(28)).padding(toLength: 28, withPad: " ", startingAt: 0)
            print(String(
                format: "  %@ %4d boxes  %4dx%-4d  cpu %7.1fms  mlx %7.1fms  x%.1f  (first mlx %.1fms)",
                label, boxes.count, image.width, image.height, onCPU, onGPU, onCPU / onGPU, first))
            if image.width * image.height > (largest.map { $0.image.width * $0.image.height } ?? 0) {
                largest = (image, boxes)
            }
        }

        // MEMORY: fifty reads of the largest capture. MLX keeps freed buffers in a cache, and a
        // long-running process (Mary, a Sand session) must not grow with every page read.
        if let (image, boxes) = largest {
            let before = Self.residentMegabytes()
            for _ in 0..<50 { _ = try engine.classifyRegions(in: image, boxes: boxes, using: gpu) }
            let after = Self.residentMegabytes()
            for _ in 0..<50 { _ = try engine.classifyRegions(in: image, boxes: boxes, using: gpu) }
            let later = Self.residentMegabytes()
            print(String(format: "[bench] resident %.0f MB -> %.0f MB after 50 mlx reads -> %.0f MB after 100",
                         before, after, later))
        }
    }

    static func residentMegabytes() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Double(info.resident_size) / 1_048_576 : -1
    }
}
