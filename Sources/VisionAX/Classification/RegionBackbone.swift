//
//  RegionBackbone.swift
//  VisionAX
//
//  WHAT: A feature extractor that can stand in for the classifier's ONNX backbone.
//  IN:   the prepared tensor from vx_classifier_prepare_image — CHW floats
//  OUT:  RegionFeatures → vx_classifier_classify_features
//  PIN:  ONLY THE BACKBONE IS SWAPPABLE. Preprocessing stays in C++ (it is the contract with
//        Python's preprocess.py) and so does the RoiAlign head. A backbone sees the exact
//        tensor the ONNX backbone would have seen and must return the exact shape it would
//        have returned — the engine checks, by name, before the head runs.
//

import Foundation
import VisionAXCore

/// What a backbone hands the head: channels × height × width floats, CHW.
struct RegionFeatures {
    var values: [Float]
    var channels: Int
    var height: Int
    var width: Int
}

/// A feature extractor standing in for the ONNX backbone. One instance serves many threads.
protocol RegionBackbone: AnyObject, Sendable {
    /// What `RegionClassifier.backboneDescription` reports while this backbone is in use.
    var name: String { get }

    /// `image` is 3 × height × width floats, CHW, normalized and padded.
    func features(image: UnsafeBufferPointer<Float>, height: Int, width: Int) throws -> RegionFeatures
}
