//
//  PageDumpTests.swift
//  VisionAXTests
//
//  WHAT: One real capture, taken apart — proposals, text, roles, groups, the map.
//  OUT:  a table, printed
//  PIN:  ENV-GATED, LIKE THE ROW DUMP. This is an instrument, not an assertion: it
//        exists so a page that reads badly can be understood without a browser in the
//        loop, which is the difference between fixing the rule and guessing at it.
//

import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import VisionAX

@Suite struct PageDumpTests {

    /// `VISIONAX_PAGE_FILE=/path.png swift test --filter dumpPage`
    @Test func dumpPage() throws {
        guard let path = ProcessInfo.processInfo.environment["VISIONAX_PAGE_FILE"] else { return }
        let url = URL(fileURLWithPath: path)
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            Issue.record("could not read \(path)")
            return
        }

        let engine = try VisionEngine()
        let classifier = try? RegionClassifier.bundled()
        let scene = try engine.perceive(
            image: image,
            projection: ScreenProjection(origin: .zero, pixelsPerPoint: 1),
            classifier: classifier,
            lanes: [.regions, .text])

        print("\(image.width)×\(image.height)  classifier \(classifier == nil ? "none" : "loaded")")
        print("nodes \(scene.nodes.count)  text runs \(scene.text?.count ?? 0)  "
            + "lines \(TextLines.lines(from: scene.text ?? []).count)  icons \(scene.icons.count)")

        let map = scene.pageMap()
        print("map: \(map.elements.count) rows, \(map.actionable.count) actionable, "
            + "\(map.groups.count) groups, \(Int(map.labeledFraction * 100))% named")

        var byKind: [String: Int] = [:]
        for group in map.groups { byKind[group.kind.rawValue, default: 0] += 1 }
        print("groups: " + byKind.sorted { $0.key < $1.key }
            .map { "\($0.key) \($0.value)" }.joined(separator: ", "))

        var bySource: [String: Int] = [:]
        for element in map.elements { bySource[element.labelSource.rawValue, default: 0] += 1 }
        print("labels: " + bySource.sorted { $0.key < $1.key }
            .map { "\($0.key) \($0.value)" }.joined(separator: ", "))

        var byAffordance: [String: Int] = [:]
        for element in map.elements { byAffordance[element.affordance.rawValue, default: 0] += 1 }
        print("affords: " + byAffordance.sorted { $0.key < $1.key }
            .map { "\($0.key) \($0.value)" }.joined(separator: ", "))

        print("\nfirst 30 actionable rows")
        for element in map.actionable.prefix(30) {
            let frame = element.frame
            print(String(
                format: "  %-30@ %-12@ %-13@ %4d,%4d %3d×%3d %@",
                String(element.label.prefix(30)) as NSString,
                (element.role ?? "—") as NSString,
                element.labelSource.rawValue as NSString,
                Int(frame.minX), Int(frame.minY), Int(frame.width), Int(frame.height),
                (element.hints.joined(separator: ",")) as NSString))
        }

        if let band = ProcessInfo.processInfo.environment["VISIONAX_BAND"] {
            let parts = band.split(separator: "-").compactMap { Int($0) }
            if parts.count == 2 {
                print("\nrows with minY in \(parts[0])...\(parts[1])")
                for element in map.elements
                where Int(element.frame.minY) >= parts[0] && Int(element.frame.minY) <= parts[1] {
                    print(String(
                        format: "  %-40@ %-10@ %-12@ %4d,%4d %3d×%3d",
                        String(element.label.prefix(40)) as NSString,
                        element.affordance.rawValue as NSString,
                        element.labelSource.rawValue as NSString,
                        Int(element.frame.minX), Int(element.frame.minY),
                        Int(element.frame.width), Int(element.frame.height)))
                }
            }
        }

        let blocks = TextLines.blocks(from: TextLines.lines(from: scene.text ?? []))
        print("\ntext blocks with 2+ lines")
        for block in blocks where block.lines.count >= 2 {
            let first = block.lines[0]
            let rest = block.lines.dropFirst().map(\.frame.height).sorted()
            let median = rest.isEmpty ? 0 : rest[rest.count / 2]
            print(String(
                format: "  %-40@ h%.0f vs %.0f  lines %d  at %4d,%4d",
                String(first.string.prefix(40)) as NSString,
                first.frame.height, median, block.lines.count,
                Int(first.frame.minX), Int(first.frame.minY)))
        }

        print("\nlongest 15 text lines")
        for line in TextLines.lines(from: scene.text ?? [])
            .sorted(by: { $0.string.count > $1.string.count })
            .prefix(15) {
            let inside = map.elements.first {
                $0.frame.insetBy(dx: -2, dy: -2).contains(line.frame)
            }
            print(String(
                format: "  %-46@ %4d,%4d %3d×%3d  in: %@",
                String(line.string.prefix(46)) as NSString,
                Int(line.frame.minX), Int(line.frame.minY),
                Int(line.frame.width), Int(line.frame.height),
                (inside.map { "\($0.affordance.rawValue) \($0.label.prefix(20))" } ?? "—") as NSString))
        }
    }
}
