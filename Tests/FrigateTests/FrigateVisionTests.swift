//
//  FrigateVisionTests.swift
//  FrigateTests
//
//  WHAT: The vision door opens, and it opens onto VisionAX's own types.
//  PIN:  IMPORTS `FrigateVision` AND NOTHING ELSE, deliberately. That is the whole claim
//        a re-export makes: a consumer names one module and gets the package behind it,
//        resources included. A test that also imported VisionAX would prove nothing.
//        macOS ONLY, like the target it exercises.
//

#if !os(Linux)

import CoreGraphics
import CoreText
import Foundation
import Testing
import FrigateVision

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
    @Test func theEngineIsReachableThroughTheReExport() throws {
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

    /// The resource bundle resolves through the indirection.
    ///
    /// PIN: SwiftPM names a bundle `<defining package>_<target>`, and VisionAX's target
    /// is still defined in the VisionAX package — so it stays `VisionAX_VisionAX.bundle`
    /// no matter who depends on it. A consumer's copy step (Mary's `make-app.sh`) keeps
    /// working, and this is the test that says so.
    @Test func theModelBundleIsFoundThroughTheReExport() {
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
