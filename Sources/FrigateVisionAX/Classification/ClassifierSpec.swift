//
//  ClassifierSpec.swift
//  FrigateVisionAX
//
//  WHAT: The model's JSON sidecar — vocabulary, preprocessing, runtime knobs.
//  IN:   VisionAX's Training/visionax_train/export.py writes it; RegionClassifier reads it.
//  OUT:  vx_classifier_spec (numbers only)
//  PIN:  SWIFT PARSES THIS, NEVER C. The C engine takes a struct of numbers and has no
//        idea what a role is called, so a model can add a class without touching a line
//        of C++. The other half of that bargain is that this file must validate what C
//        cannot: that the vocabulary is one Mary can act on, and that the tensor names
//        are the ones Classifier.cpp binds by.
//

import CVisionAX
import Foundation
import VisionAXCore

public struct ClassifierSpec: Codable, Sendable, Equatable {
    public struct Files: Codable, Sendable, Equatable {
        public var backbone: String
        public var head: String
        /// The backbone again, as MLX weights (safetensors) for the Metal path. Optional: a
        /// model exported before that path existed still loads, on its ONNX backbone.
        public var backboneMLX: String?

        enum CodingKeys: String, CodingKey {
            case backbone, head
            case backboneMLX = "backbone_mlx"
        }
    }

    public struct IO: Codable, Sendable, Equatable {
        public var image: String
        public var features: String
        public var boxes: String
        public var probs: String

        /// The names Classifier.cpp binds by. A model that renames a tensor must be a
        /// deliberate, matching change on both sides.
        static let expected = IO(image: "image", features: "features", boxes: "boxes", probs: "probs")
    }

    public struct Preprocess: Codable, Sendable, Equatable {
        public var longSide: Int
        public var padMultiple: Int
        public var interpolation: String
        public var padValue: String
        public var channelOrder: String
        public var mean: [Float]
        public var std: [Float]

        enum CodingKeys: String, CodingKey {
            case longSide = "long_side"
            case padMultiple = "pad_multiple"
            case interpolation, padValue = "pad_value", channelOrder = "channel_order"
            case mean, std
        }
    }

    public struct Head: Codable, Sendable, Equatable {
        public var stride: Int
        public var contextScale: Double
        public var roiSize: Int

        enum CodingKeys: String, CodingKey {
            case stride
            case contextScale = "context_scale"
            case roiSize = "roi_size"
        }
    }

    public struct Runtime: Codable, Sendable, Equatable {
        public var maxBoxesPerRun: Int
        public var intraOpThreads: Int

        enum CodingKeys: String, CodingKey {
            case maxBoxesPerRun = "max_boxes_per_run"
            case intraOpThreads = "intra_op_threads"
        }
    }

    public var format: Int
    public var name: String
    public var version: String
    public var roles: [String]
    public var files: Files
    public var io: IO
    public var preprocess: Preprocess
    public var head: Head
    public var runtime: Runtime
    /// The threshold calibrated at training time — below it a region stays VXRegion.
    public var minConfidence: Double
    /// What each role affords, as the training run grouped it.
    ///
    /// PIN: SHIPPED WITH THE MODEL, NOT KEPT HERE. Which roles are pressable is a
    /// decision about THIS vocabulary, and a consumer holding its own copy is a copy
    /// that drifts the first time the vocabulary changes. Optional, so a spec exported
    /// before the table existed still loads and falls back to the built-in grouping.
    public var affordances: [String: [String]]?
    /// SHA-256 of each shipped file, keyed like `files`, as the exporter wrote them.
    ///
    /// PIN: THE MLX WEIGHTS ARE TIED TO THE ONNX BACKBONE THROUGH THIS. The converter stamps
    /// `sha256["backbone"]` into the safetensors it writes; weights whose stamp differs came
    /// from another export and are refused, so a retrain that skipped the conversion falls
    /// back to CPU instead of serving a stale network on the GPU.
    public var sha256: [String: String]?

    enum CodingKeys: String, CodingKey {
        case format, name, version, roles, files, io, preprocess, head, runtime
        case affordances, sha256
        case minConfidence = "min_confidence"
    }

    /// The only format this build understands.
    public static let supportedFormat = 1

    public var vocabulary: RoleVocabulary { RoleVocabulary(roles: roles) }

    /// Everything C needs, and nothing it does not.
    func toC() -> vx_classifier_spec {
        var spec = vx_classifier_spec_default()
        spec.long_side = Int32(preprocess.longSide)
        spec.pad_multiple = Int32(preprocess.padMultiple)
        for channel in 0..<3 {
            withUnsafeMutablePointer(to: &spec.mean) {
                $0.withMemoryRebound(to: Float.self, capacity: 3) { $0[channel] = preprocess.mean[channel] }
            }
            withUnsafeMutablePointer(to: &spec.std) {
                $0.withMemoryRebound(to: Float.self, capacity: 3) { $0[channel] = preprocess.std[channel] }
            }
        }
        spec.class_count = Int32(roles.count)
        spec.max_boxes_per_run = Int32(runtime.maxBoxesPerRun)
        spec.intra_op_threads = Int32(runtime.intraOpThreads)
        spec.execution_provider = 0
        spec.feature_stride = Int32(head.stride)
        return spec
    }

    /// Everything C cannot check for itself.
    func validated() throws -> ClassifierSpec {
        guard format == Self.supportedFormat else {
            throw RegionClassifierError.unsupportedSpecFormat(format)
        }
        guard io == IO.expected else {
            throw RegionClassifierError.unexpectedTensorNames(io)
        }
        guard preprocess.mean.count == 3, preprocess.std.count == 3 else {
            throw RegionClassifierError.malformedSpec("mean and std must have three channels")
        }
        guard head.stride > 0 else {
            throw RegionClassifierError.malformedSpec("head stride \(head.stride) — it must be positive")
        }
        if let affordances {
            let known = Set(roles)
            for (group, members) in affordances {
                guard PageAffordance(rawValue: group) != nil else {
                    throw RegionClassifierError.malformedSpec(
                        "affordance group \(group) is not one this build knows")
                }
                for role in members where !known.contains(role) {
                    throw RegionClassifierError.malformedSpec(
                        "affordance group \(group) names \(role), which is not in the vocabulary")
                }
            }
        }
        guard preprocess.channelOrder == "rgb" else {
            throw RegionClassifierError.malformedSpec(
                "channel order \(preprocess.channelOrder) — the engine feeds RGB")
        }
        guard preprocess.interpolation == "area" else {
            throw RegionClassifierError.malformedSpec(
                "interpolation \(preprocess.interpolation) — the engine resizes with INTER_AREA")
        }
        guard preprocess.padValue == "mean" else {
            throw RegionClassifierError.malformedSpec(
                "pad value \(preprocess.padValue) — the engine pads with the mean colour")
        }
        do {
            try vocabulary.validated()
        } catch let error as RoleVocabularyError {
            throw RegionClassifierError.vocabulary(error)
        }
        return self
    }
}
