//
//  VisionEngine.swift
//  FrigateVisionAX
//
//  WHAT: The native face of the C engine. Owns one vx_engine for its lifetime.
//  IN:   CGImage + CannyOptions (+ a RegionClassifier, to name what was found)
//  OUT:  VisionDetection (AXWindowSnapshot + edge map)
//  PIN:  The C engine holds no per-call state, so one instance serves any number
//        of concurrent calls — hence @unchecked Sendable rather than an actor.
//

import CVisionAX
import CoreGraphics
import Foundation
import VisionAXCore

public enum VisionEngineError: Error, Equatable, CustomStringConvertible {
    /// vx_engine_create returned NULL.
    case engineUnavailable
    /// The CGImage could not be redrawn into an 8-bit BGRA buffer.
    case unreadableImage
    /// The engine rejected the call or failed inside OpenCV.
    case engine(status: UInt32, message: String)
    /// The engine produced an edge map that would not become a CGImage.
    case edgeMapUnavailable

    public var description: String {
        switch self {
        case .engineUnavailable: return "the vision engine could not be created"
        case .unreadableImage: return "the image could not be read as 8-bit BGRA"
        case .engine(let status, let message): return "engine failed (\(status)): \(message)"
        case .edgeMapUnavailable: return "the engine's edge map could not be rendered"
        }
    }
}

public final class VisionEngine: @unchecked Sendable {
    /// `vx_engine` is opaque in the header, so Swift imports the handle as an
    /// OpaquePointer rather than a typed pointer. Visible to the module so a sibling
    /// file can add a capability without reopening this one.
    let handle: OpaquePointer

    public init() throws {
        guard let handle = vx_engine_create() else {
            throw VisionEngineError.engineUnavailable
        }
        self.handle = handle
    }

    deinit {
        vx_engine_destroy(handle)
    }

    /// Canny edges → bounding boxes → an AXTree-shaped containment tree.
    ///
    /// `wantsEdgeMap` is off by default: the C engine already accepts a NULL edge map,
    /// and only the bench draws one.
    public func detectRegions(
        in image: CGImage,
        title: String,
        options: CannyOptions = .standard,
        wantsEdgeMap: Bool = false
    ) throws -> VisionDetection {
        guard let buffer = VisionImageBuffer(image: image) else {
            throw VisionEngineError.unreadableImage
        }
        return try detectRegions(
            buffer: buffer, title: title, options: options, wantsEdgeMap: wantsEdgeMap)
    }

    /// The same detection over a buffer the caller already built — the seam that lets
    /// `perceive` convert one CGImage once and hand it to every lane.
    func detectRegions(
        buffer: VisionImageBuffer,
        title: String,
        options: CannyOptions,
        wantsEdgeMap: Bool
    ) throws -> VisionDetection {

        var tree = vx_region_tree()
        var edgeMap = vx_edge_map()
        var cOptions = options.toC()
        let started = ContinuousClock.now

        let status: vx_status = buffer.withImageView { view in
            var view = view
            // The C entry takes NULL for "no edge map"; a ternary cannot produce an
            // inout pointer, so the two calls are written out.
            if wantsEdgeMap {
                return vx_engine_detect_regions(handle, &view, &cOptions, &tree, &edgeMap)
            }
            return vx_engine_detect_regions(handle, &view, &cOptions, &tree, nil)
        }
        defer {
            vx_region_tree_free(&tree)
            vx_edge_map_free(&edgeMap)
        }
        guard status == VX_OK else {
            throw VisionEngineError.engine(
                status: status.rawValue,
                message: String(cString: vx_status_message(status)))
        }

        let regions: [vx_region] = {
            guard let base = tree.regions, tree.count > 0 else { return [] }
            return Array(UnsafeBufferPointer(start: base, count: Int(tree.count)))
        }()
        var edges: CGImage?
        if wantsEdgeMap {
            guard let map = CGImage.fromEdgeMap(edgeMap) else {
                throw VisionEngineError.edgeMapUnavailable
            }
            edges = map
        }

        let bounds = CGRect(x: 0, y: 0, width: buffer.width, height: buffer.height)
        let window = RegionTreeBridge.window(
            from: regions, title: title, imageBounds: bounds,
            isTruncated: tree.truncated != 0)

        return VisionDetection(
            window: window,
            edges: edges,
            options: options,
            nodeCount: regions.count,
            contourCount: Int(tree.contour_count),
            duration: started.duration(to: ContinuousClock.now))
    }

    /// Names each box, in order. The low-level entry point: it takes rectangles rather
    /// than a tree, so a caller can classify any set of boxes — a detector's output, a
    /// hand-drawn selection — not only what Canny proposed.
    ///
    /// `boxes` are in the image's own pixel space, top-left origin, exactly like every
    /// frame in a detection.
    public func classifyRegions(
        in image: CGImage,
        boxes: [CGRect],
        using classifier: RegionClassifier
    ) throws -> [RegionLabel] {
        guard !boxes.isEmpty else { return [] }
        guard let buffer = VisionImageBuffer(image: image) else {
            throw VisionEngineError.unreadableImage
        }
        return try classifyRegions(buffer: buffer, boxes: boxes, using: classifier)
    }

    func classifyRegions(
        buffer: VisionImageBuffer,
        boxes: [CGRect],
        using classifier: RegionClassifier
    ) throws -> [RegionLabel] {
        guard !boxes.isEmpty else { return [] }
        let regions = boxes.map { box in
            var region = vx_region()
            region.x = Int32(box.origin.x.rounded())
            region.y = Int32(box.origin.y.rounded())
            region.width = Int32(box.size.width.rounded())
            region.height = Int32(box.size.height.rounded())
            return region
        }

        // THE SPLIT PATH, when the classifier carries a backbone of its own (MLX, on Metal):
        // the same preprocessing and the same RoiAlign head as the single call below, with
        // only the feature map computed elsewhere.
        if let backbone = classifier.backbone {
            return try classifyRegions(
                buffer: buffer, regions: regions, using: classifier, backbone: backbone)
        }

        var labels = vx_region_labels()
        let status: vx_status = buffer.withImageView { view in
            var view = view
            return regions.withUnsafeBufferPointer { regionBuffer in
                vx_engine_classify_regions(
                    handle, classifier.handle, &view,
                    regionBuffer.baseAddress, Int32(regionBuffer.count), &labels)
            }
        }
        defer { vx_region_labels_free(&labels) }
        try throwIfFailed(status, classifier: classifier)
        return try Self.regionLabels(labels, expected: boxes.count, classifier: classifier)
    }

    /// Preprocess in C, run `backbone`, then the head in C.
    private func classifyRegions(
        buffer: VisionImageBuffer,
        regions: [vx_region],
        using classifier: RegionClassifier,
        backbone: any RegionBackbone
    ) throws -> [RegionLabel] {
        var prepared: OpaquePointer?
        let prepareStatus: vx_status = buffer.withImageView { view in
            var view = view
            return vx_classifier_prepare_image(classifier.handle, &view, &prepared)
        }
        try throwIfFailed(prepareStatus, classifier: classifier)
        guard let prepared else {
            throw VisionEngineError.engine(
                status: VX_ERROR_INTERNAL.rawValue, message: "preprocessing returned no image")
        }
        defer { vx_prepared_image_free(prepared) }

        var height: Int32 = 0
        var width: Int32 = 0
        guard let tensor = vx_prepared_image_tensor(prepared, &height, &width) else {
            throw VisionEngineError.engine(
                status: VX_ERROR_INTERNAL.rawValue, message: "the prepared image has no tensor")
        }
        let features: RegionFeatures
        do {
            features = try backbone.features(
                image: UnsafeBufferPointer(start: tensor, count: 3 * Int(height) * Int(width)),
                height: Int(height), width: Int(width))
        } catch {
            throw RegionClassifierError.backboneFailed("\(backbone.name): \(error)")
        }

        // The head reads the image extent from the map's own shape, so a map of the wrong
        // size would crop every box silently. Refused here, by name, before C sees it.
        let stride = classifier.spec.head.stride
        guard features.height == Int(height) / stride,
              features.width == Int(width) / stride,
              features.values.count == features.channels * features.height * features.width
        else {
            throw RegionClassifierError.backboneFailed(
                "\(backbone.name) returned \(features.channels)×\(features.height)×\(features.width) "
                    + "for a \(height)×\(width) image; the head needs "
                    + "\(Int(height) / stride)×\(Int(width) / stride) at stride \(stride)")
        }

        var labels = vx_region_labels()
        let status: vx_status = features.values.withUnsafeBufferPointer { values in
            regions.withUnsafeBufferPointer { regionBuffer in
                vx_classifier_classify_features(
                    classifier.handle, prepared, values.baseAddress,
                    Int32(features.channels), Int32(features.height), Int32(features.width),
                    regionBuffer.baseAddress, Int32(regionBuffer.count), &labels)
            }
        }
        defer { vx_region_labels_free(&labels) }
        try throwIfFailed(status, classifier: classifier)
        return try Self.regionLabels(labels, expected: regions.count, classifier: classifier)
    }

    private func throwIfFailed(_ status: vx_status, classifier: RegionClassifier) throws {
        guard status == VX_OK else {
            // The model's own words beat "internal failure" every time.
            let message = status == VX_ERROR_MODEL
                ? classifier.lastError
                : String(cString: vx_status_message(status))
            throw VisionEngineError.engine(status: status.rawValue, message: message)
        }
    }

    private static func regionLabels(
        _ labels: vx_region_labels, expected: Int, classifier: RegionClassifier
    ) throws -> [RegionLabel] {
        guard let base = labels.labels, Int(labels.count) == expected else {
            throw RegionClassifierError.labelCountMismatch(
                expected: expected, got: Int(labels.count))
        }
        return UnsafeBufferPointer(start: base, count: Int(labels.count)).map {
            classifier.label(classIndex: $0.class_index, confidence: $0.confidence)
        }
    }

    /// Names every region in a detection and returns the detection with roles filled.
    ///
    /// `minimumConfidence` defaults to the one calibrated at training time and carried
    /// in the model's spec — a caller that has no opinion should not have to invent one.
    public func classifyRegions(
        in image: CGImage,
        detection: VisionDetection,
        using classifier: RegionClassifier,
        minimumConfidence: Double? = nil
    ) throws -> VisionDetection {
        guard let buffer = VisionImageBuffer(image: image) else {
            throw VisionEngineError.unreadableImage
        }
        return try classifyRegions(
            buffer: buffer, detection: detection, using: classifier,
            minimumConfidence: minimumConfidence)
    }

    func classifyRegions(
        buffer: VisionImageBuffer,
        detection: VisionDetection,
        using classifier: RegionClassifier,
        minimumConfidence: Double? = nil
    ) throws -> VisionDetection {
        let threshold = minimumConfidence ?? classifier.minimumConfidence
        let nodes = RegionRelabeler.classifiableNodes(in: detection.window)
        let started = ContinuousClock.now
        let labels = try classifyRegions(
            buffer: buffer, boxes: nodes.map(\.frame), using: classifier)
        let elapsed = started.duration(to: ContinuousClock.now)

        var byID: [AXNodeID: RegionLabel] = [:]
        byID.reserveCapacity(labels.count)
        for (node, label) in zip(nodes, labels) {
            byID[node.id] = label
        }

        return VisionDetection(
            window: RegionRelabeler.relabeled(
                detection.window, labels: byID, minimumConfidence: threshold),
            edges: detection.edges,
            options: detection.options,
            nodeCount: detection.nodeCount,
            contourCount: detection.contourCount,
            duration: detection.duration,
            labels: byID,
            classificationDuration: elapsed,
            minimumConfidence: threshold)
    }

    /// Detect then classify, in one call.
    public func detectAndClassify(
        in image: CGImage,
        title: String,
        options: CannyOptions = .standard,
        using classifier: RegionClassifier,
        minimumConfidence: Double? = nil,
        wantsEdgeMap: Bool = false
    ) throws -> VisionDetection {
        // ONE CONVERSION, NOT TWO. Building the BGRA buffer here and handing it to both
        // stages is the difference between one full-frame redraw per call and two.
        guard let buffer = VisionImageBuffer(image: image) else {
            throw VisionEngineError.unreadableImage
        }
        let detection = try detectRegions(
            buffer: buffer, title: title, options: options, wantsEdgeMap: wantsEdgeMap)
        return try classifyRegions(
            buffer: buffer, detection: detection, using: classifier,
            minimumConfidence: minimumConfidence)
    }
}
