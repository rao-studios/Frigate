//
//  SyntheticScreenTests.swift
//  VisionAXTests
//
//  WHAT: Drawn rectangles in, the same rectangles out — nested, ordered, placed.
//  OUT:  VisionEngine, RegionTreeBridge
//

import CoreGraphics
import Foundation
import Testing
@testable import VisionAX

@Suite struct SyntheticScreenTests {

    /// Canny reports a stroke's OUTER edge, so a detected box sits one line width
    /// outside the rect that was drawn — 2pt here, on every side.
    private let tolerance: CGFloat = 4

    @Test func rootCoversTheWholeImage() throws {
        let image = SyntheticScreen.image()
        let detection = try VisionEngine().detectRegions(
            in: image, title: "synthetic", wantsEdgeMap: true)
        if let edges = detection.edges { SyntheticScreen.dump(edges, named: "edges.png") }

        let root = try #require(detection.window.root)
        #expect(root.role == VisionAX.windowRole)
        #expect(root.category == .window)
        #expect(root.frame == CGRect(x: 0, y: 0, width: 800, height: 600))
        #expect(detection.window.frame == root.frame)
        #expect(detection.window.isMain)
        #expect(!detection.window.isTruncated)
    }

    @Test func everyDrawnRectangleIsFound() throws {
        let image = SyntheticScreen.image()
        let detection = try VisionEngine().detectRegions(in: image, title: "synthetic")

        for rect in [
            SyntheticScreen.Layout.window,
            SyntheticScreen.Layout.toolbar,
            SyntheticScreen.Layout.content,
            SyntheticScreen.Layout.button,
        ] {
            let match = try #require(SyntheticScreen.closest(to: rect, in: detection.window))
            #expect(match.error <= tolerance, "\(rect) missed by \(match.error)")
        }
    }

    @Test func nestingFollowsContainment() throws {
        let image = SyntheticScreen.image()
        let detection = try VisionEngine().detectRegions(in: image, title: "synthetic")
        let window = detection.window

        let windowNode = try #require(SyntheticScreen.closest(to: SyntheticScreen.Layout.window, in: window)).node
        let toolbarNode = try #require(SyntheticScreen.closest(to: SyntheticScreen.Layout.toolbar, in: window)).node
        let contentNode = try #require(SyntheticScreen.closest(to: SyntheticScreen.Layout.content, in: window)).node
        let buttonNode = try #require(SyntheticScreen.closest(to: SyntheticScreen.Layout.button, in: window)).node

        // The four are distinct nodes, not one box matched four times.
        #expect(Set([windowNode.id, toolbarNode.id, contentNode.id, buttonNode.id]).count == 4)

        #expect(SyntheticScreen.depth(of: windowNode.id, in: window) == 1)
        #expect(SyntheticScreen.depth(of: toolbarNode.id, in: window) == 2)
        #expect(SyntheticScreen.depth(of: contentNode.id, in: window) == 2)
        #expect(SyntheticScreen.depth(of: buttonNode.id, in: window) == 3)

        #expect(windowNode.children.contains { $0.id == toolbarNode.id })
        #expect(windowNode.children.contains { $0.id == contentNode.id })
        #expect(contentNode.children.contains { $0.id == buttonNode.id })
    }

    @Test func siblingsAreInReadingOrder() throws {
        let image = SyntheticScreen.image()
        let detection = try VisionEngine().detectRegions(in: image, title: "synthetic")
        let window = detection.window
        let windowNode = try #require(SyntheticScreen.closest(to: SyntheticScreen.Layout.window, in: window)).node

        // The toolbar sits above the content pane, so it comes first.
        let orderedTops = windowNode.children.compactMap { $0.frame?.minY }
        #expect(orderedTops == orderedTops.sorted())
    }

    @Test func specksBelowTheSizeFloorAreDropped() throws {
        let image = SyntheticScreen.image()
        let detection = try VisionEngine().detectRegions(in: image, title: "synthetic")
        let options = CannyOptions.standard

        // Nothing lands near the speck — the floor dropped it before the tree.
        let match = SyntheticScreen.closest(to: SyntheticScreen.Layout.speck, in: detection.window)
        #expect(match == nil || match!.error > tolerance,
                "a 6pt mark survived an \(options.minWidth)pt floor")

        // And the floor holds for every node, not just this one.
        for node in SyntheticScreen.nodes(in: detection.window).dropFirst() {
            let frame = try #require(node.frame)
            #expect(frame.width >= CGFloat(options.minWidth))
            #expect(frame.height >= CGFloat(options.minHeight))
        }
    }

    @Test func idsArePreOrderAndUnique() throws {
        let image = SyntheticScreen.image()
        let detection = try VisionEngine().detectRegions(in: image, title: "synthetic")
        let nodes = SyntheticScreen.nodes(in: detection.window)

        let ids = nodes.map(\.id.raw)
        #expect(Set(ids).count == ids.count)
        #expect(ids == ids.sorted(), "pre-order ids must ascend along a pre-order walk")
        #expect(ids.first == 1)
        #expect(detection.nodeCount == nodes.count)
    }

    @Test func subtreeCountsAreConsistent() throws {
        let image = SyntheticScreen.image()
        let detection = try VisionEngine().detectRegions(in: image, title: "synthetic")
        for node in SyntheticScreen.nodes(in: detection.window) {
            var counted = 0
            node.forEachNode { _ in counted += 1 }
            #expect(node.subtreeCount == counted)
        }
    }

    @Test func edgeMapMatchesTheImage() throws {
        let image = SyntheticScreen.image()
        let detection = try VisionEngine().detectRegions(
            in: image, title: "synthetic", wantsEdgeMap: true)
        let edges = try #require(detection.edges)
        #expect(edges.width == image.width)
        #expect(edges.height == image.height)
        #expect(edges.bitsPerPixel == 8)
        // AND IT IS OFF BY DEFAULT — the map is a full-frame buffer, and a caller that
        // did not ask for one must not pay for it.
        let quiet = try VisionEngine().detectRegions(in: image, title: "synthetic")
        #expect(quiet.edges == nil)
        #expect(quiet.nodeCount == detection.nodeCount)
        #expect(detection.contourCount >= detection.nodeCount - 1)
    }

    @Test func onnxRuntimeIsLinked() throws {
        // Answering requires building the shared OrtEnv, so this proves the static
        // archive is really linked — not just that its header parsed.
        let version = try #require(VisionAX.onnxRuntimeVersion)
        #expect(version.hasPrefix("1."), "unexpected ONNX Runtime version \(version)")
    }

    @Test func theEngineReportsItsOwnVersion() {
        #expect(!VisionAX.version.isEmpty)
        #expect(VisionAX.version != "0.0.0", "VISIONAX_VERSION was not defined at compile time")
    }

    @Test func openCVIsLinked() {
        #expect(VisionAX.openCVVersion.hasPrefix("4."))
    }
}
