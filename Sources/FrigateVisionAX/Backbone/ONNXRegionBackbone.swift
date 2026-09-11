//
//  ONNXRegionBackbone.swift
//  FrigateVisionAX
//
//  WHAT: A classifier's own ONNX backbone, behind the RegionBackbone seam.
//  IN:   any RegionClassifier — its C engine runs the backbone graph on CPU
//  OUT:  RegionFeatures, exactly what the single-call path hands its head
//  PIN:  THE REFERENCE, NOT A PRODUCTION PATH. The single call already runs this backbone
//        without copying its output; this exists so the split path can be held to EQUALITY
//        with it, and so the MLX backbone can be measured against the features it replaces.
//

import CVisionAX
import Foundation

final class ONNXRegionBackbone: RegionBackbone, @unchecked Sendable {
    let name = "onnx-reference"

    private let classifier: RegionClassifier

    init(_ classifier: RegionClassifier) {
        self.classifier = classifier
    }

    func features(
        image: UnsafeBufferPointer<Float>, height: Int, width: Int
    ) throws -> RegionFeatures {
        var map = vx_feature_map()
        let status = vx_classifier_backbone_features(
            classifier.handle, image.baseAddress, Int32(height), Int32(width), &map)
        defer { vx_feature_map_free(&map) }
        guard status == VX_OK, let data = map.data else {
            throw RegionClassifierError.backboneFailed("the ONNX backbone: \(classifier.lastError)")
        }
        let count = Int(map.channels) * Int(map.height) * Int(map.width)
        return RegionFeatures(
            values: Array(UnsafeBufferPointer(start: data, count: count)),
            channels: Int(map.channels), height: Int(map.height), width: Int(map.width))
    }
}
