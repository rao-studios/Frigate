//
//  ScreenProjectionTests.swift
//  VisionAXTests
//
//  WHAT: The pixel↔point map, including the Retina case that makes every click wrong.
//  OUT:  ScreenProjection
//

import CoreGraphics
import Foundation
import Testing
@testable import VisionAX

@Suite struct ScreenProjectionTests {

    /// AT 1x THE TWO SPACES DIFFER ONLY BY THE WINDOW ORIGIN.
    @Test func oneTimesIsAPlainTranslation() {
        let projection = ScreenProjection(origin: CGPoint(x: 100, y: 50), pixelsPerPoint: 1)
        let rect = projection.screenRect(CGRect(x: 10, y: 20, width: 30, height: 40))
        #expect(rect == CGRect(x: 110, y: 70, width: 30, height: 40))
    }

    /// THE RETINA CASE. A 2x capture reports twice the pixels for the same points, and
    /// a projection that assumed 1x would aim every click at half the distance from the
    /// window's corner — the failure this type exists to prevent.
    @Test func twoTimesHalvesEveryDistance() {
        let projection = ScreenProjection(origin: CGPoint(x: 200, y: 100), pixelsPerPoint: 2)
        let rect = projection.screenRect(CGRect(x: 400, y: 200, width: 80, height: 40))
        #expect(rect == CGRect(x: 400, y: 200, width: 40, height: 20))
    }

    @Test(arguments: [1.0, 2.0, 3.0]) func pixelAndScreenRoundTrip(_ scale: Double) {
        let projection = ScreenProjection(origin: CGPoint(x: 37, y: 91), pixelsPerPoint: scale)
        let pixels = CGRect(x: 12, y: 34, width: 56, height: 78)
        let back = projection.pixelRect(projection.screenRect(pixels))
        #expect(abs(back.minX - pixels.minX) < 0.001)
        #expect(abs(back.minY - pixels.minY) < 0.001)
        #expect(abs(back.width - pixels.width) < 0.001)
        #expect(abs(back.height - pixels.height) < 0.001)
    }

    /// A CROP MOVES THE PROJECTION, NOT THE FRAMES. This is what lets a region of
    /// interest cost nothing: the detector reports in the crop's own coordinates and
    /// the offset lives here.
    @Test func aRegionOfInterestShiftsTheOrigin() {
        let projection = ScreenProjection(origin: CGPoint(x: 0, y: 0), pixelsPerPoint: 2)
        let cropped = projection.offset(byPixelROI: CGRect(x: 100, y: 60, width: 400, height: 300))
        // Pixel (0,0) of the crop is pixel (100,60) of the parent — 50pt, 30pt in.
        #expect(cropped.origin == CGPoint(x: 50, y: 30))
        #expect(cropped.pixelsPerPoint == 2)
        #expect(cropped.screenRect(CGRect(x: 0, y: 0, width: 20, height: 20))
            == CGRect(x: 50, y: 30, width: 10, height: 10))
    }

    @Test func theCentreIsWhereAClickGoes() {
        let projection = ScreenProjection(origin: CGPoint(x: 10, y: 10), pixelsPerPoint: 2)
        #expect(projection.screenPoint(centerOf: CGRect(x: 0, y: 0, width: 40, height: 20))
            == CGPoint(x: 20, y: 15))
    }

    /// A NONSENSE DENSITY MISPLACES A CLICK; IT DOES NOT TAKE THE PROCESS DOWN.
    @Test func aZeroScaleDegradesRatherThanDividesByZero() {
        let projection = ScreenProjection(origin: .zero, pixelsPerPoint: 0)
        let rect = projection.screenRect(CGRect(x: 4, y: 4, width: 8, height: 8))
        #expect(rect.width.isFinite)
        #expect(rect == CGRect(x: 4, y: 4, width: 8, height: 8))
    }
}
