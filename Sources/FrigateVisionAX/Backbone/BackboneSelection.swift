//
//  BackboneSelection.swift
//  FrigateVisionAX
//
//  WHAT: Which backbone a RegionClassifier runs — MLX on Metal, or the ONNX graph on CPU —
//        and a sentence saying why.
//  IN:   RegionClassifier.Backbone, the model's spec, FRIGATE_VISION_BACKBONE
//  OUT:  Choice → RegionClassifier.backbone and .backboneDescription
//  PIN:  EVERY CHECK RUNS BEFORE ANY MLX OP. With no metallib beside the binary, the first
//        GPU op aborts the process (plain `swift test`; a build nobody ran build-metallib.sh
//        for), so presence, the spec and the weights' provenance are settled from the
//        filesystem alone before MLX is touched.
//        THE ACCELERATOR IS OPTIONAL; THE MODEL IS NOT. A missing, stale or unloadable MLX
//        backbone leaves the classifier on its ONNX backbone and says why — it never takes
//        the classifier away, because the ONNX path is the whole model.
//        THE ENVIRONMENT OVERRIDES ONLY `.automatic`. Code that asks for `.onnx` or `.mlx` by
//        name gets what it asked for; FRIGATE_VISION_BACKBONE is the A/B switch for a process
//        that took the default — Sand, the app.
//

import Foundation
import VisionAXCore

enum BackboneSelection {
    static let environmentKey = "FRIGATE_VISION_BACKBONE"

    /// The only MLX architecture this build implements. The converter stamps it into the
    /// weights, and refuses any graph that is not this one.
    static let architecture = "resnet18-fpn8"

    struct Choice {
        var backbone: (any RegionBackbone)?
        var description: String
    }

    enum Availability: Equatable {
        case available(URL)
        case unavailable(String)
    }

    static func resolve(
        _ preference: RegionClassifier.Backbone,
        spec: ClassifierSpec,
        modelDirectory: URL,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        metalLibrary: () -> URL? = { MetalLibrary.locate() }
    ) -> Choice {
        var effective = preference
        var note = ""
        if preference == .automatic,
           let raw = environment[environmentKey]?.trimmingCharacters(in: .whitespaces).lowercased(),
           !raw.isEmpty {
            switch raw {
            case "onnx", "cpu":
                effective = .onnx
                note = "\(environmentKey)=\(raw)"
            case "mlx", "metal":
                effective = .mlx
                note = "\(environmentKey)=\(raw)"
            case "auto", "automatic":
                break
            default:
                note = "ignored \(environmentKey)=\(raw)"
            }
        }

        if effective == .onnx {
            return Choice(backbone: nil, description: note.isEmpty ? "onnx-cpu" : "onnx-cpu (\(note))")
        }

        switch availability(spec: spec, modelDirectory: modelDirectory, metalLibrary: metalLibrary()) {
        case .unavailable(let reason):
            let asked = effective == .mlx ? "mlx requested, but " : ""
            let suffix = note.isEmpty || effective == .mlx ? "" : "; \(note)"
            return Choice(backbone: nil, description: "onnx-cpu (\(asked)\(reason)\(suffix))")
        case .available(let weights):
            do {
                let backbone = try MLXRegionBackbone(weights: weights)
                return Choice(backbone: backbone, description: backbone.name)
            } catch {
                return Choice(
                    backbone: nil,
                    description: "onnx-cpu (the MLX backbone failed to load: \(error))")
            }
        }
    }

    /// Everything that decides whether the MLX backbone can run, answered without MLX.
    static func availability(
        spec: ClassifierSpec, modelDirectory: URL, metalLibrary: URL?
    ) -> Availability {
        guard metalLibrary != nil else {
            return .unavailable(
                "no mlx.metallib beside the binary — run Frigate's scripts/build-metallib.sh")
        }
        guard let name = spec.files.backboneMLX else {
            return .unavailable("the model ships no MLX backbone (its spec has no files.backbone_mlx)")
        }
        let url = modelDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .unavailable("\(name) is missing from \(modelDirectory.path)")
        }
        let metadata: [String: String]
        do {
            metadata = try SafetensorsHeader.metadata(at: url)
        } catch {
            return .unavailable("\(error)")
        }
        guard metadata["arch"] == architecture else {
            return .unavailable(
                "\(name) is a \(metadata["arch"] ?? "unlabelled") backbone; this build runs \(architecture)")
        }
        guard let source = metadata["source_sha256"] else {
            return .unavailable("\(name) does not say which ONNX backbone it was converted from")
        }
        guard let onnx = spec.sha256?["backbone"] else {
            return .unavailable(
                "the spec carries no sha256 for its ONNX backbone, so \(name) cannot be tied to it")
        }
        guard source == onnx else {
            return .unavailable(
                "\(name) was converted from a different ONNX backbone — re-run `vxtrain export-mlx`")
        }
        return .available(url)
    }
}
