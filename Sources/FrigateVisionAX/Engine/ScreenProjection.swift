//
//  ScreenProjection.swift
//  FrigateVisionAX
//
//  WHAT: The map between a detection's image pixels and the screen it was cut from.
//  IN:   the capturer — it alone knows the window origin and the measured density
//  OUT:  VisionScene.screenRect / screenPoint; every consumer that clicks
//  PIN:  THE SCALE IS MEASURED, NEVER ASSUMED. A detection carries no scale of its
//        own (the image does not know it), so a caller that guesses 2x is wrong on a
//        1x display and wrong again on any future density. This type is the one place
//        the two coordinate systems meet, and it is pure arithmetic so a test can pin it.
//

import CoreGraphics
import Foundation
import VisionAXCore

public struct ScreenProjection: Sendable, Equatable, Codable {

    /// Global, top-left-origin screen POINTS of the image's pixel (0, 0).
    public var origin: CGPoint

    /// Measured pixels per point — `image.width / window.frame.width`, never a constant.
    public var pixelsPerPoint: Double

    public init(origin: CGPoint, pixelsPerPoint: Double) {
        self.origin = origin
        self.pixelsPerPoint = pixelsPerPoint
    }

    /// The identity projection: pixels ARE points, anchored at the screen origin.
    /// For a detection over a file, where nothing on screen is involved.
    public static let imageSpace = ScreenProjection(origin: .zero, pixelsPerPoint: 1)

    /// A pixel rect from a detection, placed on the screen.
    public func screenRect(_ pixels: CGRect) -> CGRect {
        let scale = CGFloat(safeScale)
        return CGRect(
            x: origin.x + pixels.origin.x / scale,
            y: origin.y + pixels.origin.y / scale,
            width: pixels.width / scale,
            height: pixels.height / scale)
    }

    /// A screen rect, back into the image's pixel space.
    public func pixelRect(_ points: CGRect) -> CGRect {
        let scale = CGFloat(safeScale)
        return CGRect(
            x: (points.origin.x - origin.x) * scale,
            y: (points.origin.y - origin.y) * scale,
            width: points.width * scale,
            height: points.height * scale)
    }

    /// Where to aim at the middle of a detected box.
    public func screenPoint(centerOf pixels: CGRect) -> CGPoint {
        let rect = screenRect(pixels)
        return CGPoint(x: rect.midX.rounded(), y: rect.midY.rounded())
    }

    public func screenPoint(_ pixel: CGPoint) -> CGPoint {
        let scale = CGFloat(safeScale)
        return CGPoint(x: origin.x + pixel.x / scale, y: origin.y + pixel.y / scale)
    }

    /// The projection for a detection run over a CROPPED region of this image.
    ///
    /// Cropping is how a region of interest is expressed — the crop's pixel (0, 0) is
    /// the parent's `roi.origin`, so every frame the sub-detection reports lands on
    /// screen correctly without the detector knowing a crop happened.
    public func offset(byPixelROI roi: CGRect) -> ScreenProjection {
        let scale = CGFloat(safeScale)
        return ScreenProjection(
            origin: CGPoint(
                x: origin.x + roi.origin.x / scale,
                y: origin.y + roi.origin.y / scale),
            pixelsPerPoint: pixelsPerPoint)
    }

    /// A zero or negative density would divide the plane away; it is treated as 1
    /// rather than trapping, because a bad projection should misplace a click by a
    /// factor, not take the process down.
    private var safeScale: Double {
        pixelsPerPoint > 0 ? pixelsPerPoint : 1
    }
}
