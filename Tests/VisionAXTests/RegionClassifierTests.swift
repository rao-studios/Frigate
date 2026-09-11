//
//  RegionClassifierTests.swift
//  VisionAXTests
//
//  WHAT: The ONNX Runtime path, end to end, against a model whose answer is arithmetic.
//  PIN:  `matchesPythonExactly` IS THE PREPROCESSING CONTRACT. The fixture's expected
//        probabilities were produced by Training/visionax_train/preprocess.py and ONNX
//        Runtime in Python; reproducing them here proves the C++ twin resizes, pads,
//        normalizes and scales boxes identically. When that test drifts, the model did
//        not get worse — one of the two preprocessors changed.
//

import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import VisionAX

@Suite struct RegionClassifierTests {

    // MARK: - Fixture

    struct Expected: Codable {
        var image: String
        var boxes: [[Int]]
        var expectedClassIndex: [Int]
        var expectedRole: [String]
        var probabilities: [[Double]]
    }

    private static func fixtureURL(_ name: String) throws -> URL {
        try #require(
            Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures"),
            "missing fixture \(name) — regenerate with Training/tools/make_test_model.py")
    }

    private func loadClassifier() throws -> RegionClassifier {
        try RegionClassifier(specURL: Self.fixtureURL("tiny-classifier.json"))
    }

    private func loadExpected() throws -> Expected {
        let data = try Data(contentsOf: Self.fixtureURL("tiny-classifier.expected.json"))
        return try AXTreeJSON.decode(Expected.self, from: data)
    }

    private func loadImage() throws -> CGImage {
        let url = try Self.fixtureURL("tiny-classifier.image.png")
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    private func boxes(_ expected: Expected) -> [CGRect] {
        expected.boxes.map {
            CGRect(x: $0[0], y: $0[1], width: $0[2], height: $0[3])
        }
    }

    // MARK: - The contract

    @Test func matchesPythonExactly() throws {
        let classifier = try loadClassifier()
        let expected = try loadExpected()
        let labels = try VisionEngine().classifyRegions(
            in: try loadImage(), boxes: boxes(expected), using: classifier)

        #expect(labels.count == expected.expectedClassIndex.count)
        for (index, label) in labels.enumerated() {
            #expect(label.classIndex == expected.expectedClassIndex[index])
            #expect(label.role == expected.expectedRole[index])
            let reference = expected.probabilities[index][label.classIndex]
            #expect(abs(label.confidence - reference) < 1e-4,
                    "box \(index): \(label.confidence) vs Python's \(reference)")
        }
    }

    @Test func theFixtureReadsColoursTheWayItWasBuiltTo() throws {
        let classifier = try loadClassifier()
        let expected = try loadExpected()
        let labels = try VisionEngine().classifyRegions(
            in: try loadImage(), boxes: boxes(expected), using: classifier)

        #expect(labels[0].role == "AXButton")      // the red block
        #expect(labels[1].role == "AXTextField")   // the blue block
        #expect(labels[2].role == "none")          // plain background
        #expect(labels[0].confidence > 0.9)
        #expect(labels[1].confidence > 0.9)
    }

    @Test func theFixtureVocabularyIsTheStandardOne() throws {
        // The generator carries its own copy of the role list; this is what keeps the
        // two honest.
        let classifier = try loadClassifier()
        #expect(classifier.vocabulary.roles == RoleVocabulary.standard.roles)
    }

    // MARK: - Shape and budget

    @Test func noBoxesIsSuccessNotFailure() throws {
        let labels = try VisionEngine().classifyRegions(
            in: try loadImage(), boxes: [], using: try loadClassifier())
        #expect(labels.isEmpty)
    }

    @Test func oneImageCostsOneBackbonePassHoweverManyBoxes() throws {
        let classifier = try loadClassifier()
        let image = try loadImage()
        let box = CGRect(x: 16, y: 16, width: 96, height: 48)

        // 1,500 boxes against a 512 chunk: one backbone run, three head runs.
        let labels = try VisionEngine().classifyRegions(
            in: image, boxes: Array(repeating: box, count: 1500), using: classifier)

        #expect(labels.count == 1500)
        #expect(labels.allSatisfy { $0.role == "AXButton" })
        let counts = classifier.runCounts
        #expect(counts.backbone == 1, "the backbone re-ran per chunk")
        #expect(counts.head == 3, "expected ceil(1500/512) head runs, got \(counts.head)")
    }

    @Test func chunkingDoesNotReorderAnswers() throws {
        let classifier = try loadClassifier()
        let expected = try loadExpected()
        let image = try loadImage()
        // Repeat the three fixture boxes past the chunk boundary; every row must still
        // answer for its own box.
        let repeated = Array(repeating: boxes(expected), count: 400).flatMap { $0 }
        let labels = try VisionEngine().classifyRegions(
            in: image, boxes: repeated, using: classifier)

        #expect(labels.count == 1200)
        for (index, label) in labels.enumerated() {
            #expect(label.role == expected.expectedRole[index % 3], "row \(index) drifted")
        }
    }

    @Test func aDifferentImageSizeStillWorks() throws {
        // The graphs were exported at 320×200; the tree they will really serve is any
        // screenshot at all.
        let classifier = try loadClassifier()
        let labels = try VisionEngine().classifyRegions(
            in: SyntheticScreen.image(),
            boxes: [CGRect(x: 80, y: 160, width: 160, height: 44)],
            using: classifier)
        #expect(labels.count == 1)
        #expect(labels[0].confidence > 0)
    }

    // MARK: - Failure modes

    @Test func aSpecWithARoleMaryDiscardsIsRefused() throws {
        let url = try Self.fixtureURL("tiny-classifier.json")
        var spec = try AXTreeJSON.decode(ClassifierSpec.self, from: Data(contentsOf: url))
        spec.roles[1] = "AXUnknown"
        #expect(throws: RegionClassifierError.self) { try spec.validated() }
    }

    @Test func aSpecThatRenamesATensorIsRefused() throws {
        let url = try Self.fixtureURL("tiny-classifier.json")
        var spec = try AXTreeJSON.decode(ClassifierSpec.self, from: Data(contentsOf: url))
        spec.io.boxes = "rois"
        #expect(throws: RegionClassifierError.self) { try spec.validated() }
    }

    @Test func anLFSPointerIsNamedAsSuch() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("visionax-lfs-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let specURL = try Self.fixtureURL("tiny-classifier.json")
        let spec = try AXTreeJSON.decode(ClassifierSpec.self, from: Data(contentsOf: specURL))
        try FileManager.default.copyItem(
            at: specURL, to: directory.appendingPathComponent("tiny-classifier.json"))
        // What a clone without `git lfs pull` actually leaves on disk.
        let pointer = """
            version https://git-lfs.github.com/spec/v1
            oid sha256:0000000000000000000000000000000000000000000000000000000000000000
            size 12345

            """
        for file in [spec.files.backbone, spec.files.head] {
            try Data(pointer.utf8).write(to: directory.appendingPathComponent(file))
        }

        #expect(throws: RegionClassifierError.self) {
            try RegionClassifier(specURL: directory.appendingPathComponent("tiny-classifier.json"))
        }
    }

    @Test func aMissingModelFileIsNamed() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("visionax-missing-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.copyItem(
            at: try Self.fixtureURL("tiny-classifier.json"),
            to: directory.appendingPathComponent("tiny-classifier.json"))

        #expect(throws: RegionClassifierError.self) {
            try RegionClassifier(specURL: directory.appendingPathComponent("tiny-classifier.json"))
        }
    }

    @Test func manyThreadsShareOneClassifier() throws {
        let classifier = try loadClassifier()
        let engine = try VisionEngine()
        let image = try loadImage()
        let expected = try loadExpected()
        let boxes = boxes(expected)

        // The C engine holds no per-call state; this is the assertion that says so.
        DispatchQueue.concurrentPerform(iterations: 8) { _ in
            let labels = try? engine.classifyRegions(in: image, boxes: boxes, using: classifier)
            #expect(labels?.first?.role == "AXButton")
        }
        #expect(classifier.runCounts.backbone == 8)
    }
}
