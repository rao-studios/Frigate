//
//  SyntheticScreenSupport.swift
//  VisionAXTests
//
//  WHAT: A drawn "screenshot" with known rectangles, so assertions can name what
//        the engine should have found.
//  OUT:  SyntheticScreenTests, RegionTreeBudgetTests
//  PIN:  Strokes, not fills — a filled rectangle on a white ground still produces
//        one Canny loop, but a stroked one is what real UI chrome looks like and it
//        is the case the dedup rules exist for.
//
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import VisionAX

enum SyntheticScreen {

    /// A window frame, a toolbar strip and a content pane inside it, and one button
    /// inside the pane — the nesting the tree should reproduce.
    struct Layout {
        static let imageSize = CGSize(width: 800, height: 600)
        static let window = CGRect(x: 20, y: 20, width: 760, height: 560)
        static let toolbar = CGRect(x: 40, y: 40, width: 720, height: 60)
        static let content = CGRect(x: 40, y: 120, width: 720, height: 440)
        static let button = CGRect(x: 80, y: 160, width: 160, height: 44)
        /// Below the 8pt default floor once stroked — must not survive into the
        /// tree. Canny reports a stroke's OUTER edge, so the detected box is the
        /// drawn rect grown by one line width on each side: 2 + 2 + 2 = 6.
        static let speck = CGRect(x: 600, y: 500, width: 2, height: 2)
    }

    /// `rects` are stroked dark on a light ground, in the order given.
    static func image(
        rects: [CGRect] = [
            Layout.window, Layout.toolbar, Layout.content, Layout.button, Layout.speck,
        ],
        size: CGSize = Layout.imageSize,
        lineWidth: CGFloat = 2
    ) -> CGImage {
        let width = Int(size.width)
        let height = Int(size.height)
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue)!

        context.setFillColor(CGColor(red: 0.96, green: 0.96, blue: 0.96, alpha: 1))
        context.fill(CGRect(origin: .zero, size: size))
        context.setStrokeColor(CGColor(red: 0.1, green: 0.1, blue: 0.1, alpha: 1))
        context.setLineWidth(lineWidth)
        for rect in rects {
            // The context is y-up; flip each rect so the caller's coordinates are the
            // TOP-LEFT ones the engine will report back.
            let flipped = CGRect(
                x: rect.minX, y: size.height - rect.maxY,
                width: rect.width, height: rect.height)
            context.stroke(flipped)
        }
        return context.makeImage()!
    }

    /// Every node in the tree, root included, pre-order.
    static func nodes(in window: AXWindowSnapshot) -> [AXNodeSnapshot] {
        guard let root = window.root else { return [] }
        var all: [AXNodeSnapshot] = []
        root.forEachNode { all.append($0) }
        return all
    }

    /// The node whose frame is closest to `rect`, and how far off it is.
    static func closest(
        to rect: CGRect, in window: AXWindowSnapshot
    ) -> (node: AXNodeSnapshot, error: CGFloat)? {
        let scored = nodes(in: window).compactMap { node -> (AXNodeSnapshot, CGFloat)? in
            guard let frame = node.frame else { return nil }
            let error = max(
                abs(frame.minX - rect.minX), abs(frame.minY - rect.minY),
                abs(frame.maxX - rect.maxX), abs(frame.maxY - rect.maxY))
            return (node, error)
        }
        return scored.min { $0.1 < $1.1 }
    }

    /// Depth of the node with this id, root = 0, or nil when it is not in the tree.
    static func depth(of id: AXNodeID, in window: AXWindowSnapshot) -> Int? {
        guard let root = window.root else { return nil }
        var depth: Int?
        root.forEachNode(withAncestors: { node, ancestors in
            if node.id == id { depth = ancestors.count }
        })
        return depth
    }

    /// Set VISIONAX_DUMP_DIR to eyeball what a failing test actually saw.
    static func dump(_ image: CGImage, named name: String) {
        guard let dir = ProcessInfo.processInfo.environment["VISIONAX_DUMP_DIR"] else { return }
        let url = URL(fileURLWithPath: dir).appendingPathComponent(name)
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let destination = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
    }
}
