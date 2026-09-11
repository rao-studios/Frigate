//
//  FrigateVisionTests.swift
//  FrigateTests
//
//  WHAT: The FrigateVision product opens onto Frigate's VisionAX module, and the AX types
//        of VisionAXCore arrive with it.
//  PIN:  IMPORTS `VisionAX` AND NOTHING ELSE, deliberately. The product keeps the name
//        FrigateVision; the module it vends is VisionAX, and that module re-exports
//        VisionAXCore — so one import must reach the engine, the AX types and the model's
//        resources. A test that also imported VisionAXCore would prove nothing.
//        macOS ONLY, like the target it exercises.
//

#if !os(Linux)

import CoreGraphics
import CoreText
import Foundation
import Testing
import VisionAX

@Suite("FrigateVision")
struct FrigateVisionTests {

    /// The engine builds and perceives — the C++ core, OpenCV, Apple's Vision framework
    /// and the Swift face, all reached through the one import.
    ///
    /// PIN: BOTH LANES, BECAUSE THE MAP NEEDS BOTH. A page of bare rectangles with no
    /// text lane yields an empty map by design — the map keeps classified nodes, text
    /// lines and icon-shaped boxes, and a borderless rectangle is none of those. Asking
    /// for `.text` too is what makes this a test of the whole door rather than of the
    /// detector alone.
    @Test func theEngineIsReachableThroughTheModule() throws {
        let engine = try VisionEngine()
        let scene = try engine.perceive(
            image: Self.drawnPage(),
            projection: ScreenProjection(origin: .zero, pixelsPerPoint: 1),
            lanes: [.regions, .text])

        #expect(scene.detection != nil)
        #expect(scene.nodes.count > 1, "the detector found nothing in a drawn page")
        #expect(scene.text?.isEmpty == false, "recognition read nothing from drawn words")
        // And the map, which is what a consumer actually acts on.
        let map = scene.pageMap()
        #expect(map.elements.isEmpty == false)
        #expect(map.elements.contains { $0.label.lowercased().contains("accept") })
    }

    /// The AX vocabulary lives in VisionAXCore and must still arrive through `VisionAX`.
    @Test func theAXTypesArriveThroughTheReExport() {
        #expect(AXNodeCategory.category(role: "AXButton") != .other)
        #expect(RoleVocabulary.standard.roles.first == RoleVocabulary.noneRole)
    }

    /// The resource bundle resolves.
    ///
    /// PIN: SwiftPM names a bundle `<defining package>_<target>`. The runtime target is
    /// defined in Frigate now, so the bundle is `Frigate_VisionAX.bundle` — the name a
    /// consumer's copy step (Mary's `make-app.sh`) must use, and the one
    /// `RegionClassifier.hostBundleModels` looks for.
    @Test func theModelBundleResolves() {
        // A model may legitimately be absent (git-lfs not pulled); the LOCATION must not.
        #expect(RegionClassifier.resourceBundleLocation() != nil)
        if let classifier = try? RegionClassifier.bundled() {
            #expect(classifier.spec.roles.isEmpty == false)
        }
    }

    /// A page with edges to find and words to read.
    private static func drawnPage() -> CGImage {
        let context = CGContext(
            data: nil, width: 600, height: 300, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue)!
        context.setFillColor(gray: 0.98, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 600, height: 300))
        context.setStrokeColor(gray: 0.1, alpha: 1)
        context.setLineWidth(2)
        context.stroke(CGRect(x: 40, y: 40, width: 520, height: 100))
        context.stroke(CGRect(x: 40, y: 190, width: 180, height: 56))

        // NO FLIP. Flipping the CTM to reach image coordinates mirrors the glyphs too,
        // and mirrored text does not recognize — measured, this test read nothing at all.
        // Everything here is drawn in the context's own bottom-left space; where the
        // words sit in the image does not matter to what is being proved.
        func write(_ text: String, at point: CGPoint, size: CGFloat) {
            // The Core Text keys, not AppKit's: this target links neither AppKit nor
            // UIKit, and `NSAttributedString.Key.font` comes from those.
            let attributed = NSAttributedString(
                string: text,
                attributes: [
                    NSAttributedString.Key(kCTFontAttributeName as String):
                        CTFontCreateWithName("Helvetica" as CFString, size, nil),
                    NSAttributedString.Key(kCTForegroundColorAttributeName as String):
                        CGColor(gray: 0.05, alpha: 1),
                ])
            let line = CTLineCreateWithAttributedString(attributed)
            context.textPosition = point
            CTLineDraw(line, context)
        }
        write("Accept all cookies", at: CGPoint(x: 70, y: 214), size: 26)
        write("This page would like to store a few things.", at: CGPoint(x: 70, y: 90), size: 22)
        return context.makeImage()!
    }
}

#endif
