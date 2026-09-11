//
//  RegionClassifier.swift
//  VisionAX
//
//  WHAT: A loaded model — two ONNX graphs plus the vocabulary that names their output — and
//        the backbone this process runs it on.
//  IN:   a spec JSON on disk, or the one bundled in Resources/Models
//  OUT:  VisionEngine.classifyRegions
//  PIN:  The two .onnx files are git-lfs objects. A clone without `git lfs pull` leaves
//        130-byte TEXT pointers in their place, and handing one to ONNX Runtime yields
//        "Protobuf parsing failed" — a message that sends you looking at the model
//        instead of at git. So the pointer is detected here, by name, and said out loud.
//        THE BACKBONE IS CHOSEN HERE, ONCE. `.automatic` runs it on MLX (Metal) when this
//        process can — a metallib beside the binary, MLX weights converted from this model's
//        own ONNX backbone — and on ONNX Runtime's CPU path otherwise; `backboneDescription`
//        says which, and why. FRIGATE_VISION_BACKBONE=onnx|mlx overrides `.automatic` for an
//        A/B without touching a caller. See Backbone/BackboneSelection.swift.
//        The C engine holds no per-call state, so one classifier serves many threads.
//

import CVisionAX
import Foundation
import VisionAXCore
import os

public enum RegionClassifierError: Error, CustomStringConvertible {
    case specUnreadable(URL, String)
    case unsupportedSpecFormat(Int)
    case unexpectedTensorNames(ClassifierSpec.IO)
    case malformedSpec(String)
    case vocabulary(RoleVocabularyError)
    case modelMissing(URL)
    case modelIsLFSPointer(URL)
    case modelFailed(String)
    case labelCountMismatch(expected: Int, got: Int)
    case backboneFailed(String)

    public var description: String {
        switch self {
        case .specUnreadable(let url, let reason):
            return "could not read the model spec at \(url.path): \(reason)"
        case .unsupportedSpecFormat(let format):
            return "model spec format \(format) — this build understands "
                + "\(ClassifierSpec.supportedFormat)"
        case .unexpectedTensorNames(let io):
            return "the spec names its tensors \(io.image)/\(io.features)/\(io.boxes)/\(io.probs); "
                + "the engine binds image/features/boxes/probs"
        case .malformedSpec(let reason):
            return "the model spec is not one this engine can serve: \(reason)"
        case .vocabulary(let error):
            return "the model's vocabulary is unusable: \(error)"
        case .modelMissing(let url):
            return "the model file \(url.lastPathComponent) is missing from \(url.deletingLastPathComponent().path)"
        case .modelIsLFSPointer(let url):
            return "\(url.lastPathComponent) is a git-lfs pointer, not a model — run `git lfs pull`"
        case .modelFailed(let message):
            return "ONNX Runtime rejected the model: \(message)"
        case .labelCountMismatch(let expected, let got):
            return "asked for \(expected) labels, the engine returned \(got)"
        case .backboneFailed(let message):
            return "the classifier's backbone failed: \(message)"
        }
    }
}

public final class RegionClassifier: @unchecked Sendable {
    /// Where the backbone runs. `.automatic` is the default everywhere, and the only
    /// preference FRIGATE_VISION_BACKBONE overrides.
    public enum Backbone: String, Sendable, CaseIterable {
        /// MLX on Metal when this process can run it; the ONNX backbone on CPU otherwise.
        case automatic
        /// The ONNX backbone on CPU — the reference path.
        case onnx
        /// MLX on Metal. Falls back to the ONNX backbone, saying why, when it cannot run.
        case mlx
    }

    /// `vx_classifier` is opaque in the header, so it arrives as an OpaquePointer.
    let handle: OpaquePointer
    public let spec: ClassifierSpec
    public let vocabulary: RoleVocabulary
    /// The directory the spec and the files it names were loaded from.
    public let modelDirectory: URL
    /// Which backbone runs, and — when it is not the one that could have — why:
    /// "mlx-metal", "onnx-cpu", or "onnx-cpu (<reason>)".
    public let backboneDescription: String
    /// Nil when the ONNX backbone inside the C engine runs.
    let backbone: (any RegionBackbone)?

    /// The confidence below which a region keeps `VXRegion` — calibrated at training
    /// time and carried in the spec, so serving does not need to re-guess it.
    public var minimumConfidence: Double { spec.minConfidence }

    /// `specURL` names a JSON sidecar; the files inside it are resolved relative to that
    /// file, so a model directory can be copied anywhere as a unit.
    public convenience init(specURL: URL, backbone preference: Backbone = .automatic) throws {
        try self.init(specURL: specURL) { spec, directory in
            BackboneSelection.resolve(preference, spec: spec, modelDirectory: directory)
        }
    }

    /// A classifier on a backbone the caller built — the seam a test uses to run the split
    /// path on a backbone whose answer it already knows.
    convenience init(specURL: URL, injecting backbone: any RegionBackbone) throws {
        try self.init(specURL: specURL) { _, _ in
            BackboneSelection.Choice(backbone: backbone, description: backbone.name)
        }
    }

    private init(
        specURL: URL,
        choosing: (ClassifierSpec, URL) -> BackboneSelection.Choice
    ) throws {
        let data: Data
        do {
            data = try Data(contentsOf: specURL)
        } catch {
            throw RegionClassifierError.specUnreadable(specURL, error.localizedDescription)
        }
        let decoded: ClassifierSpec
        do {
            decoded = try AXTreeJSON.decode(ClassifierSpec.self, from: data)
        } catch {
            throw RegionClassifierError.specUnreadable(specURL, "\(error)")
        }
        let spec = try decoded.validated()

        let directory = specURL.deletingLastPathComponent()
        let backbonePath = directory.appendingPathComponent(spec.files.backbone)
        let headPath = directory.appendingPathComponent(spec.files.head)
        try Self.requireRealModel(at: backbonePath)
        try Self.requireRealModel(at: headPath)

        var cSpec = spec.toC()
        var created: OpaquePointer?
        let status = backbonePath.path.withCString { backboneCString in
            headPath.path.withCString { headCString in
                vx_classifier_create(backboneCString, headCString, &cSpec, &created)
            }
        }
        guard status == VX_OK, let created else {
            throw RegionClassifierError.modelFailed(String(cString: vx_classifier_last_error(nil)))
        }

        let choice = choosing(spec, directory)
        self.handle = created
        self.spec = spec
        self.vocabulary = spec.vocabulary
        self.modelDirectory = directory
        self.backbone = choice.backbone
        self.backboneDescription = choice.description
        Self.log.notice(
            "region classifier \(spec.name, privacy: .public) \(spec.version, privacy: .public): backbone \(choice.description, privacy: .public)"
        )
    }

    private static let log = Logger(subsystem: "nyc.rao.frigate", category: "vision")

    deinit {
        vx_classifier_destroy(handle)
    }

    /// The model shipped in the package's resources, or nil when none was bundled.
    /// Throws only when a model IS there but cannot be loaded — a missing model is a
    /// configuration, a broken one is a fault.
    ///
    /// PIN: `Bundle.module` IS TOUCHED LAST, AND THAT ORDER IS LOAD-BEARING. SwiftPM
    /// generates an accessor that calls `fatalError` when the resource bundle is not
    /// beside the executable — so inside a copied .app, reaching for it first would
    /// TRAP rather than return nil. Looking in the host app's own Resources first means
    /// a properly assembled bundle never reaches the trapping path, and a machine with
    /// neither gets an honest nil.
    public static func bundled(
        searching extra: [URL] = [], backbone: Backbone = .automatic
    ) throws -> RegionClassifier? {
        for directory in extra + [Self.hostBundleModels].compactMap({ $0 }) {
            let spec = directory.appendingPathComponent("region-classifier.json")
            if FileManager.default.fileExists(atPath: spec.path) {
                return try RegionClassifier(specURL: spec, backbone: backbone)
            }
        }
        guard let url = Bundle.module.url(
            forResource: "region-classifier", withExtension: "json", subdirectory: "Models")
        else { return nil }
        return try RegionClassifier(specURL: url, backbone: backbone)
    }

    /// Where a host application copies this module's resource bundle — the layout
    /// `make-app.sh`-style assembly produces. SwiftPM names it `<package>_<target>`.
    static var hostBundleModels: URL? {
        Bundle.main.resourceURL?
            .appendingPathComponent("Frigate_VisionAX.bundle")
            .appendingPathComponent("Models")
    }

    /// The directory the model was actually loaded from, for a probe to print. Nil when
    /// no model is reachable at all — which is a state the media lane survives.
    public static func resourceBundleLocation() -> URL? {
        if let host = hostBundleModels,
           FileManager.default.fileExists(
            atPath: host.appendingPathComponent("region-classifier.json").path) {
            return host
        }
        return Bundle.module.url(
            forResource: "region-classifier", withExtension: "json", subdirectory: "Models")?
            .deletingLastPathComponent()
    }

    /// Run counters, for proving one image costs one backbone pass. `backbone` counts the
    /// ONNX backbone only — a classifier on the MLX backbone reports 0 there.
    public var runCounts: (backbone: Int, head: Int) {
        let stats = vx_classifier_get_stats(handle)
        return (Int(stats.backbone_runs), Int(stats.head_runs))
    }

    /// The engine's own words about the last failure on this classifier.
    var lastError: String {
        String(cString: vx_classifier_last_error(handle))
    }

    /// Turns a raw class index into the label callers see.
    func label(classIndex: Int32, confidence: Float) -> RegionLabel {
        RegionLabel(
            classIndex: Int(classIndex),
            role: vocabulary.role(at: Int(classIndex)) ?? RoleVocabulary.noneRole,
            confidence: Double(confidence))
    }

    private static func requireRealModel(at url: URL) throws {
        guard let handle = FileHandle(forReadingAtPath: url.path) else {
            throw RegionClassifierError.modelMissing(url)
        }
        defer { try? handle.close() }
        let head = (try? handle.read(upToCount: 64)) ?? Data()
        // git-lfs pointer files begin with exactly this line.
        if head.starts(with: Data("version https://git-lfs.github.com/spec".utf8)) {
            throw RegionClassifierError.modelIsLFSPointer(url)
        }
    }
}
