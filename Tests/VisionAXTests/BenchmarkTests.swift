//
//  BenchmarkTests.swift
//  VisionAXTests
//
//  WHAT: What each lane costs, measured on real captures rather than guessed.
//  OUT:  a table, printed
//  PIN:  ENV-GATED AND NOT AN ASSERTION. A timing that fails a build is a timing that
//        fails on a busy machine, and the number worth having is the one you can read
//        beside what else was running. The instrument matters more than the threshold:
//        every claim about "the media lane is cheap" or "the classifier dominates" in
//        the docs should come from here.
//        MEDIAN OF REPEATS, AND THE FIRST RUN REPORTED SEPARATELY. The first perception
//        pays for the ONNX session, the Vision request and the glyph banks; a mean that
//        folds that in describes a cost nobody pays twice.
//

import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import VisionAX

@Suite struct BenchmarkTests {

    struct Sample {
        var name: String
        var image: CGImage
    }

    struct Timing {
        var name: String
        var first: Duration
        var median: Duration
        var detail: String
    }

    /// `VISIONAX_BENCH=/dir/with/pngs swift test --filter benchmarkTheLanes`
    @Test func benchmarkTheLanes() throws {
        guard let root = ProcessInfo.processInfo.environment["VISIONAX_BENCH"] else { return }
        let repeats = Int(ProcessInfo.processInfo.environment["VISIONAX_REPEATS"] ?? "") ?? 5
        let samples = try load(from: URL(fileURLWithPath: root))
        guard !samples.isEmpty else {
            Issue.record("no PNGs under \(root)")
            return
        }

        let engine = try VisionEngine()
        let classifier = try? RegionClassifier.bundled()
        print("""

        VisionAX benchmark
        ──────────────────
        \(samples.count) captures, \(repeats) repeats each, classifier \
        \(classifier == nil ? "ABSENT" : "loaded")
        """)

        for sample in samples {
            let size = "\(sample.image.width)×\(sample.image.height)"
            print("\n\(sample.name)  \(size)")
            var rows: [Timing] = []

            rows.append(measure("detect (Canny)", repeats: repeats) {
                let detection = try engine.detectRegions(
                    in: sample.image, title: "", options: .standard)
                return "\(detection.nodeCount) nodes, \(detection.contourCount) contours"
            })

            rows.append(measure("text (fast)", repeats: repeats) {
                let runs = try TextRecognizer.runs(in: sample.image)
                return "\(runs.count) runs"
            })

            rows.append(measure("text (accurate)", repeats: repeats) {
                let runs = try TextRecognizer.runs(in: sample.image, accurate: true)
                return "\(runs.count) runs"
            })

            if let classifier {
                let detection = try engine.detectRegions(
                    in: sample.image, title: "", options: .standard)
                let boxes = RegionRelabeler.classifiableNodes(in: detection.window)
                    .map(\.frame)
                rows.append(measure("classify", repeats: repeats) {
                    let labels = try engine.classifyRegions(
                        in: sample.image, boxes: boxes, using: classifier)
                    return "\(labels.count) boxes"
                })
            }

            rows.append(measure("media", repeats: repeats) {
                let reading = try engine.readMediaControls(in: sample.image)
                return reading.controlsVisible
                    ? "\(reading.controls.count) controls, \(reading.playback.rawValue)"
                    : "no transport"
            })

            // WHAT THE LANE ACTUALLY DOES LIVE: two frames, so the picture's motion is
            // a witness. The second frame is the same image here — the cost is the
            // comparison, not the difference it finds.
            rows.append(measure("media (2 frames)", repeats: repeats) {
                let reading = try engine.readMediaControls(
                    in: sample.image, previous: sample.image)
                return "motion \(reading.motion.map { String(format: "%.3f", $0) } ?? "—")"
            })

            // The icon bank, over every box it would really be asked about.
            let detected = try engine.detectRegions(
                in: sample.image, title: "", options: .standard)
            let iconBoxes = PageMapBuilder.iconCandidates(
                in: detected, lines: TextLines.lines(from: []))
                .map { $0.frame }
            rows.append(measure("icons (\(iconBoxes.count) boxes)", repeats: repeats) {
                let matched = try engine.matchIcons(in: sample.image, boxes: iconBoxes)
                return "\(matched.filter { $0.name != nil }.count) named"
            })

            rows.append(measure("perceive .elements", repeats: repeats) {
                let scene = try engine.perceive(
                    image: sample.image,
                    projection: ScreenProjection(origin: .zero, pixelsPerPoint: 1),
                    classifier: classifier,
                    lanes: [.regions, .text])
                let map = scene.pageMap()
                return "\(map.elements.count) rows, \(map.actionable.count) actionable, "
                    + "\(Int(map.labeledFraction * 100))% named"
            })

            rows.append(measure("perceive .media", repeats: repeats) {
                let scene = try engine.perceive(
                    image: sample.image,
                    projection: ScreenProjection(origin: .zero, pixelsPerPoint: 1),
                    lanes: [.media, .text])
                return scene.media?.controlsVisible == true ? "transport" : "no transport"
            })

            // The map alone, over a perception already in hand — what a second query costs.
            let scene = try engine.perceive(
                image: sample.image,
                projection: ScreenProjection(origin: .zero, pixelsPerPoint: 1),
                classifier: classifier, lanes: [.regions, .text])
            rows.append(measure("pageMap (built)", repeats: repeats) {
                "\(scene.pageMap().elements.count) rows"
            })

            for row in rows {
                print(String(
                    format: "  %-20@ first %7.1fms   median %7.1fms   %@",
                    row.name as NSString,
                    milliseconds(row.first), milliseconds(row.median),
                    row.detail as NSString))
            }
        }
    }

    /// ONE CALL IN, ONE CALL OUT — the package as a unit.
    ///
    /// `VISIONAX_BENCH=/dir/of/pngs swift test --filter benchmarkOnePipelineCall`
    ///
    /// PIN: THE PHASES COME FROM INSIDE THE CALL, not from calling the lanes separately.
    /// Timed from outside, the parts add up to more than the whole — each call rebuilds
    /// the image buffer, the union is invisible, and the classifier appears to pay for a
    /// conversion the detector already did. This is the one number to quote for the
    /// package, and the breakdown under it always sums to it.
    @Test func benchmarkOnePipelineCall() throws {
        guard let root = ProcessInfo.processInfo.environment["VISIONAX_BENCH"] else { return }
        let repeats = Int(ProcessInfo.processInfo.environment["VISIONAX_REPEATS"] ?? "") ?? 5
        let samples = try load(from: URL(fileURLWithPath: root))
        guard !samples.isEmpty else {
            Issue.record("no PNGs under \(root)")
            return
        }

        let engine = try VisionEngine()
        let classifier = try? RegionClassifier.bundled()
        let projection = ScreenProjection(origin: .zero, pixelsPerPoint: 1)

        func run(
            _ sample: Sample, lanes: VisionLanes, useClassifier: Bool
        ) throws -> (VisionScene, PageMap, Duration) {
            let scene = try engine.perceive(
                image: sample.image, projection: projection,
                classifier: useClassifier ? classifier : nil, lanes: lanes)
            let began = ContinuousClock.now
            let map = scene.pageMap()
            return (scene, map, began.duration(to: ContinuousClock.now))
        }

        for sample in samples {
            for (label, lanes, wantsClassifier) in [
                ("page read  (.regions + .text, classifier)", VisionLanes([.regions, .text]), true),
                ("media read (.media + .text, no classifier)", VisionLanes([.media, .text]), false),
            ] {
                // Warm, then take the median call and print ITS breakdown — an average
                // of breakdowns is a shape no single call ever had.
                _ = try run(sample, lanes: lanes, useClassifier: wantsClassifier)
                var runs: [(VisionScene, PageMap, Duration)] = []
                for _ in 0 ..< max(1, repeats) {
                    runs.append(try run(sample, lanes: lanes, useClassifier: wantsClassifier))
                }
                let ordered = runs.sorted { $0.0.timing.total < $1.0.timing.total }
                let (scene, map, mapCost) = ordered[ordered.count / 2]

                print("""

                \(sample.name)  \(sample.image.width)×\(sample.image.height)  —  \(label)
                \(scene.timing.table())
                  pageMap()      \(String(format: "%7.1fms", mapCost.milliseconds))
                  ────────────────────────
                  ENTRY→EXIT     \(String(format: "%7.1fms", (scene.timing.total + mapCost).milliseconds))
                  out: \(map.elements.count) rows, \(map.actionable.count) actionable, \
                \(scene.text?.count ?? 0) text runs\
                \(scene.media?.controlsVisible == true ? ", transport found" : "")
                """)
            }
        }
    }

    // MARK: - Machinery

    private func measure(
        _ name: String, repeats: Int, _ body: () throws -> String
    ) -> Timing {
        var timings: [Duration] = []
        var detail = ""
        for _ in 0 ..< max(1, repeats) {
            let started = ContinuousClock.now
            detail = (try? body()) ?? "failed"
            timings.append(started.duration(to: ContinuousClock.now))
        }
        let sorted = timings.sorted()
        return Timing(
            name: name, first: timings[0],
            median: sorted[sorted.count / 2], detail: detail)
    }

    private func milliseconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
    }

    private func load(from directory: URL) throws -> [Sample] {
        try FileManager.default
            .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "png" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { url in
                guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
                else { return nil }
                return Sample(name: url.deletingPathExtension().lastPathComponent, image: image)
            }
    }
}
