//
//  BundledModelTests.swift
//  VisionAXTests
//
//  WHAT: The shipped model loads, names things, and never names them unsafely.
//  PIN:  Skips cleanly when no model is committed — a clone without `git lfs pull`
//        must fail on the POINTER, loudly, and a repo with no model at all must not
//        fail at all. Those are different situations and the difference matters.
//

import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import VisionAX

@Suite struct BundledModelTests {

    @Test func theBundledModelLoadsIfPresent() throws {
        guard let classifier = try RegionClassifier.bundled() else { return }
        #expect(classifier.vocabulary.roles == RoleVocabulary.standard.roles,
                "the shipped vocabulary drifted from RoleVocabulary.standard")
        #expect(classifier.minimumConfidence > 0 && classifier.minimumConfidence < 1)
        try classifier.spec.validated()
    }

    @Test func itNamesTheSyntheticScreenSafely() throws {
        guard let classifier = try RegionClassifier.bundled() else { return }
        let engine = try VisionEngine()
        let image = SyntheticScreen.image()
        let detection = try engine.detectAndClassify(
            in: image, title: "synthetic", using: classifier)

        #expect(detection.isClassified)
        // Whatever it decides, it may only ever use words Mary knows.
        detection.window.root?.forEachNode { node in
            let known = RoleVocabulary.standard.roles.contains(node.role)
                || node.role == VisionAX.regionRole || node.role == VisionAX.windowRole
            #expect(known, "invented role \(node.role)")
            #expect(node.category == AXNodeCategory.category(role: node.role))
            #expect(node.label == nil)
        }
        // And the tree it produces is still a tree Mary can decode.
        let json = try detection.json()
        #expect(try AXTreeJSON.decode(AXWindowSnapshot.self, from: json) == detection.window)
    }

    @Test func aRaisedThresholdOnlyEverRemovesNames() throws {
        guard let classifier = try RegionClassifier.bundled() else { return }
        let engine = try VisionEngine()
        let detection = try engine.detectAndClassify(
            in: SyntheticScreen.image(), title: "synthetic", using: classifier)
        let strict = detection.relabeled(minimumConfidence: 1.01)
        #expect(strict.namedCount == 0)
        #expect(detection.relabeled(minimumConfidence: 0).namedCount >= detection.namedCount)
    }
}
