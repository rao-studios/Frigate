//
//  RecallDiagnosticTests.swift
//  VisionAXTests
//
//  WHAT: What proposal recall would be, on real samples already harvested.
//  OUT:  a table, printed
//  PIN:  MEASURED BEFORE RE-HARVESTING, NOT AFTER. A change to the proposal set costs
//        an hour of crawling to evaluate the honest way; this replays samples that are
//        already on disk, which answers the same question in seconds and is the reason
//        a bad idea gets thrown away instead of shipped.
//        ENV-GATED. It needs a dataset, which no other machine has.
//

import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import VisionAX

@Suite struct RecallDiagnosticTests {

    /// `VISIONAX_DATASET=/path swift test --filter measureProposalRecall`
    @Test func measureProposalRecall() throws {
        guard let root = ProcessInfo.processInfo.environment["VISIONAX_DATASET"] else { return }
        let limit = Int(ProcessInfo.processInfo.environment["VISIONAX_LIMIT"] ?? "") ?? 40
        let directory = URL(fileURLWithPath: root)
        let samples = try FileManager.default
            .contentsOfDirectory(at: directory.appendingPathComponent("samples"),
                                 includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }

        let decoder = JSONDecoder()
        let engine = try VisionEngine()
        var totals: [String: (found: Int, total: Int)] = [:]

        func add(_ key: String, _ report: RecallReport) {
            let existing = totals[key] ?? (0, 0)
            totals[key] = (existing.found + report.overall.found,
                           existing.total + report.overall.total)
        }

        var used = 0
        for url in samples {
            guard used < limit else { break }
            guard let data = try? Data(contentsOf: url),
                  let sample = try? decoder.decode(HarvestSample.self, from: data),
                  sample.origin.url != nil
            else { continue }
            let imageURL = directory.appendingPathComponent(sample.imagePath)
            guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
            else { continue }
            used += 1

            let truth = sample.elements
            let detection = try engine.detectRegions(
                in: image, title: sample.origin.title, options: sample.canny)
            add("canny only", ProposalMatcher.match(
                proposals: ProposalMatcher.proposals(from: detection.window),
                elements: truth).recall)

            let runs = (try? TextRecognizer.runs(in: image)) ?? []
            let lines = TextLines.lines(from: runs)
            var existing: [CGRect] = []
            detection.window.root?.forEachNode { if let frame = $0.frame { existing.append(frame) } }

            let lineBoxes = TextLines.proposals(fromLines: lines, notCovering: existing)
            add("+ text lines", ProposalMatcher.match(
                proposals: ProposalMatcher.proposals(
                    from: ProposalUnion.union(detection, boxes: lineBoxes).window),
                elements: truth).recall)

        }

        print("proposal recall over \(used) real samples")
        for key in ["canny only", "+ text lines"] {
            guard let value = totals[key], value.total > 0 else { continue }
            print(String(format: "  %-14@ %5d/%5d  %.3f", key as NSString,
                         value.found, value.total,
                         Double(value.found) / Double(value.total)))
        }
    }
}
