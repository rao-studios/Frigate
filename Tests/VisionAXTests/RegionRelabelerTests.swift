//
//  RegionRelabelerTests.swift
//  VisionAXTests
//
//  WHAT: A classified detection is still a tree Mary can walk.
//  PIN:  The threshold tests are the Mary-safety contract in practice: raise it and
//        every node falls back to VXRegion, which Mary ignores. Nothing may ever come
//        back with a label or a subrole — this step has no evidence for either.
//

import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import VisionAX

@Suite struct RegionRelabelerTests {

    private func fixtureClassifier() throws -> RegionClassifier {
        let url = try #require(Bundle.module.url(
            forResource: "tiny-classifier.json", withExtension: nil, subdirectory: "Fixtures"))
        return try RegionClassifier(specURL: url)
    }

    private func fixtureImage() throws -> CGImage {
        let url = try #require(Bundle.module.url(
            forResource: "tiny-classifier.image.png", withExtension: nil, subdirectory: "Fixtures"))
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    private func classifiedFixture() throws -> VisionDetection {
        let engine = try VisionEngine()
        let image = try fixtureImage()
        return try engine.detectAndClassify(
            in: image, title: "fixture", using: try fixtureClassifier())
    }

    @Test func classifiableNodesSkipTheRoot() throws {
        let detection = try VisionEngine().detectRegions(
            in: SyntheticScreen.image(), title: "synthetic")
        let nodes = RegionRelabeler.classifiableNodes(in: detection.window)
        #expect(nodes.count == detection.nodeCount - 1)
        #expect(!nodes.contains { $0.id == detection.window.root?.id })
    }

    @Test func aClassifiedTreeCarriesRolesAndCategories() throws {
        let detection = try classifiedFixture()
        #expect(detection.isClassified)
        #expect(detection.namedCount > 0, "the fixture's blocks should have been named")

        detection.window.root?.forEachNode { node in
            // Every role in the tree is either a vocabulary role or the safe fallback.
            let known = RoleVocabulary.standard.roles.contains(node.role)
            #expect(known || node.role == VisionAX.regionRole || node.role == VisionAX.windowRole,
                    "invented role \(node.role)")
            // And the category always agrees with the role it was derived from.
            #expect(node.category == AXNodeCategory.category(role: node.role))
        }
    }

    @Test func nothingGainsALabelOrSubrole() throws {
        let detection = try classifiedFixture()
        detection.window.root?.forEachNode { node in
            #expect(node.label == nil, "the classifier invented a label")
            #expect(node.subrole == nil, "the classifier invented a subrole")
        }
    }

    @Test func theRootStaysAWindow() throws {
        let detection = try classifiedFixture()
        #expect(detection.window.root?.role == VisionAX.windowRole)
        #expect(detection.window.root?.category == .window)
    }

    @Test func animpossibleThresholdLeavesEverythingUnnamed() throws {
        let detection = try classifiedFixture()
        let strict = detection.relabeled(minimumConfidence: 1.01)
        #expect(strict.namedCount == 0)
        strict.window.root?.forEachNode { node in
            #expect(node.role == VisionAX.regionRole || node.role == VisionAX.windowRole)
            #expect(node.category == .other || node.category == .window)
        }
    }

    @Test func loweringTheThresholdNamesAtLeastAsMuch() throws {
        let detection = try classifiedFixture()
        let loose = detection.relabeled(minimumConfidence: 0.0)
        let strict = detection.relabeled(minimumConfidence: 0.99)
        #expect(loose.namedCount >= strict.namedCount)
    }

    @Test func rethresholdingRunsNoModel() throws {
        let classifier = try fixtureClassifier()
        let detection = try VisionEngine().detectAndClassify(
            in: try fixtureImage(), title: "fixture", using: classifier)
        let before = classifier.runCounts

        _ = detection.relabeled(minimumConfidence: 0.1)
        _ = detection.relabeled(minimumConfidence: 0.9)

        #expect(classifier.runCounts == before, "re-thresholding re-ran inference")
    }

    @Test func theShapeOfTheTreeIsUntouched() throws {
        let engine = try VisionEngine()
        let image = try fixtureImage()
        let plain = try engine.detectRegions(in: image, title: "fixture")
        let classified = try engine.classifyRegions(
            in: image, detection: plain, using: try fixtureClassifier())

        #expect(classified.nodeCount == plain.nodeCount)
        #expect(classified.window.root?.subtreeCount == plain.window.root?.subtreeCount)

        var plainFrames: [CGRect?] = []
        var classifiedFrames: [CGRect?] = []
        plain.window.root?.forEachNode { plainFrames.append($0.frame) }
        classified.window.root?.forEachNode { classifiedFrames.append($0.frame) }
        #expect(plainFrames == classifiedFrames, "classification moved a box")
    }

    @Test func aClassifiedTreeStillRoundTripsThroughJSON() throws {
        let detection = try classifiedFixture()
        let json = try detection.json()
        let decoded = try AXTreeJSON.decode(AXWindowSnapshot.self, from: json)
        #expect(decoded == detection.window)
    }

    @Test func anUnclassifiedDetectionRethresholdsToItself() throws {
        let detection = try VisionEngine().detectRegions(
            in: SyntheticScreen.image(), title: "synthetic")
        #expect(!detection.isClassified)
        #expect(detection.relabeled(minimumConfidence: 0.5).window == detection.window)
    }
}
