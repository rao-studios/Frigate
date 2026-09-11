//
//  VisionDetection.swift
//  FrigateVisionAX
//
//  WHAT: One detection's whole result — the tree, the edges, and what it cost.
//  IN:   VisionEngine.detectRegions
//  OUT:  VisionAXBench, Mary
//  PIN:  `window.frame` is the IMAGE's pixel bounds, not a screen rect. A caller
//        placing these on screen divides by the screenshot's backing scale.
//

import CoreGraphics
import Foundation
import VisionAXCore

public struct VisionDetection: Sendable {
    /// The tree, in the shape Mary walks. Root role "AXWindow"; every detected region
    /// role "VXRegion", category .other, label nil until a classifier runs.
    public let window: AXWindowSnapshot
    /// The raw Canny map, 8-bit gray, image-sized — present only when the caller asked
    /// for it. NIL IS THE DEFAULT, and deliberately: the map is a full-frame buffer
    /// nobody but the bench looks at, and building one per perception tick is a
    /// megabyte of copying to throw away.
    public let edges: CGImage?
    public let options: CannyOptions
    /// Nodes in the tree, root included.
    public let nodeCount: Int
    /// Contours found before size filtering and dedup — a tuning signal: a large gap
    /// between this and `nodeCount` means the filters are doing most of the work.
    public let contourCount: Int
    public let duration: Duration
    /// Every node's raw answer, threshold not applied — kept so a caller can move the
    /// threshold without paying for inference again.
    public let labels: [AXNodeID: RegionLabel]?
    /// What the classifier cost, separate from detection.
    public let classificationDuration: Duration?
    /// The threshold the tree in `window` was built with.
    public let minimumConfidence: Double?

    public init(
        window: AXWindowSnapshot,
        edges: CGImage? = nil,
        options: CannyOptions,
        nodeCount: Int,
        contourCount: Int,
        duration: Duration,
        labels: [AXNodeID: RegionLabel]? = nil,
        classificationDuration: Duration? = nil,
        minimumConfidence: Double? = nil
    ) {
        self.window = window
        self.edges = edges
        self.options = options
        self.nodeCount = nodeCount
        self.contourCount = contourCount
        self.duration = duration
        self.labels = labels
        self.classificationDuration = classificationDuration
        self.minimumConfidence = minimumConfidence
    }

    /// True once a classifier has run over this detection.
    public var isClassified: Bool { labels != nil }

    /// The same answers at a different threshold. PURE — no model runs, so a caller can
    /// drag a confidence slider without re-inferring.
    public func relabeled(minimumConfidence: Double) -> VisionDetection {
        guard let labels else { return self }
        return VisionDetection(
            window: RegionRelabeler.relabeled(
                window, labels: labels, minimumConfidence: minimumConfidence),
            edges: edges,
            options: options,
            nodeCount: nodeCount,
            contourCount: contourCount,
            duration: duration,
            labels: labels,
            classificationDuration: classificationDuration,
            minimumConfidence: minimumConfidence)
    }

    /// How many nodes carry a role a classifier named, at the tree's own threshold.
    public var namedCount: Int {
        var count = 0
        window.root?.forEachNode { node in
            if node.role != VisionAX.regionRole && node.role != VisionAX.windowRole { count += 1 }
        }
        return count
    }

    /// The image's pixel bounds — the plane every frame in the tree lives on.
    public var imageBounds: CGRect {
        if let frame = window.frame { return frame }
        guard let edges else { return .zero }
        return CGRect(x: 0, y: 0, width: edges.width, height: edges.height)
    }

    /// The tree as JSON, in the house format (sorted keys, ISO-8601, trailing newline).
    public func json(prettyPrinted: Bool = true) throws -> String {
        try AXTreeJSON.encode(window, prettyPrinted: prettyPrinted)
    }
}
