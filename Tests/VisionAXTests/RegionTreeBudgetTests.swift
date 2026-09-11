//
//  RegionTreeBudgetTests.swift
//  VisionAXTests
//
//  WHAT: The filters and the walk budget — what the engine refuses to emit.
//  OUT:  CannyRegionDetector (size floor, dedup), RegionTreeBuilder (depth, nodes)
//  PIN:  Budget semantics are Mary's AXTreeWalker: a node AT max_depth is KEPT and
//        its children withheld; exactly max_nodes nodes are emitted.
//

import CoreGraphics
import Foundation
import Testing
@testable import VisionAX

@Suite struct RegionTreeBudgetTests {

    private func detect(_ options: CannyOptions) throws -> VisionDetection {
        try VisionEngine().detectRegions(
            in: SyntheticScreen.image(), title: "synthetic", options: options)
    }

    @Test func sizeFloorDropsSmallBoxes() throws {
        var large = CannyOptions.standard
        large.minWidth = 400
        large.minHeight = 400
        let detection = try detect(large)

        for node in SyntheticScreen.nodes(in: detection.window).dropFirst() {
            let frame = try #require(node.frame)
            #expect(frame.width >= 400 && frame.height >= 400)
        }
        // The button (160×44) cannot survive that floor.
        let button = SyntheticScreen.closest(to: SyntheticScreen.Layout.button, in: detection.window)
        #expect(button == nil || button!.error > 4)
    }

    @Test func maxNodesTruncates() throws {
        var tight = CannyOptions.standard
        tight.maxNodes = 2
        let detection = try detect(tight)

        #expect(detection.nodeCount == 2)
        #expect(detection.window.isTruncated)
    }

    @Test func maxDepthKeepsTheNodeAndWithholdsItsChildren() throws {
        var shallow = CannyOptions.standard
        shallow.maxDepth = 1
        let detection = try detect(shallow)

        let root = try #require(detection.window.root)
        #expect(!root.children.isEmpty, "depth-1 nodes are kept")
        for child in root.children {
            #expect(child.children.isEmpty, "their children are withheld")
        }
        #expect(detection.window.isTruncated)
    }

    @Test func theBudgetKeepsTheLargestBoxes() throws {
        // The keep cap drops the SMALLEST candidates, so a budget of root + 2 must
        // spend it on the window and the content pane, never on the button.
        var tiny = CannyOptions.standard
        tiny.maxNodes = 3
        let detection = try detect(tiny)

        #expect(detection.nodeCount == 3)
        #expect(detection.window.isTruncated)

        let kept = SyntheticScreen.nodes(in: detection.window).dropFirst().compactMap(\.frame)
        let areas = kept.map { $0.width * $0.height }
        #expect(areas.count == 2)
        let buttonArea = SyntheticScreen.Layout.button.width * SyntheticScreen.Layout.button.height
        #expect(areas.allSatisfy { $0 > buttonArea * 4 }, "the budget went to small boxes")
    }

    @Test func aFullBudgetDoesNotReportTruncation() throws {
        let detection = try detect(.standard)
        #expect(!detection.window.isTruncated)
        #expect(detection.nodeCount < CannyOptions.standard.maxNodes)
    }

    @Test func oneStrokeBecomesOneBox() throws {
        // A single stroked rectangle yields an outer and an inner contour. Dedup must
        // collapse them, or every box in every screenshot would arrive doubled.
        let rect = CGRect(x: 100, y: 100, width: 300, height: 200)
        let image = SyntheticScreen.image(rects: [rect], lineWidth: 3)
        let detection = try VisionEngine().detectRegions(in: image, title: "one-stroke")

        let matches = SyntheticScreen.nodes(in: detection.window).dropFirst().filter { node in
            guard let frame = node.frame else { return false }
            return abs(frame.midX - rect.midX) < 8 && abs(frame.midY - rect.midY) < 8
        }
        #expect(matches.count == 1, "found \(matches.count) boxes for one stroke")
    }

    @Test func anEmptyImageIsJustTheRoot() throws {
        let image = SyntheticScreen.image(rects: [])
        let detection = try VisionEngine().detectRegions(in: image, title: "blank")

        #expect(detection.nodeCount == 1)
        let root = try #require(detection.window.root)
        #expect(root.children.isEmpty)
        #expect(root.category == .window)
    }

    @Test func detectedRegionsCarryNoLabelYet() throws {
        let detection = try detect(.standard)
        for node in SyntheticScreen.nodes(in: detection.window).dropFirst() {
            #expect(node.role == VisionAX.regionRole)
            #expect(node.category == .other)
            #expect(node.label == nil)
            #expect(node.subrole == nil)
        }
    }
}
