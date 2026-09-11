//
//  TextRecognizer.swift
//  FrigateVisionAX
//
//  WHAT: The words in a picture, where they are.
//  IN:   VisionEngine.perceive, lane .text
//  OUT:  [TextRun] in image pixel space
//  PIN:  APPLE'S VISION, NOT A MODEL WE SHIP. Text recognition is the one perception
//        task the system already does well, on the ANE, with no download — spending a
//        second ONNX graph on it would be slower and worse.
//        `.fast` BY DEFAULT: the strings this reads are clocks and button labels a few
//        characters long, and the accurate path costs several times as much to resolve
//        ambiguities that a timestamp does not have.
//        Vision reports NORMALIZED, BOTTOM-LEFT rects; every frame here is flipped into
//        the top-left pixel space the rest of the package speaks, once, at this seam.
//

import CoreGraphics
import Foundation
import Vision
import VisionAXCore

/// One recognized string and the box it sits in.
public struct TextRun: Sendable, Equatable, Codable {
    public var string: String
    /// Image pixel space, top-left origin.
    public var frame: CGRect
    public var confidence: Double

    public init(string: String, frame: CGRect, confidence: Double) {
        self.string = string
        self.frame = frame
        self.confidence = confidence
    }
}

public enum TextRecognizerError: Error, CustomStringConvertible {
    case failed(String)

    public var description: String {
        switch self {
        case .failed(let reason): return "text recognition failed: \(reason)"
        }
    }
}

public enum TextRecognizer {

    /// The text inside one region, enlarged first.
    ///
    /// PIN: SMALL TEXT IS READ BY MAKING IT BIGGER. A player's clock is about fourteen
    /// points tall in a page-sized capture, and recognition at that size returns nothing
    /// at all — measured, on a timecode plainly legible to a person. Cropping to the
    /// transport and scaling up costs a fraction of what recognising the whole page
    /// costs, and it is the difference between a clock and no clock.
    public static func runs(
        in image: CGImage,
        within region: CGRect,
        magnify: Int = 3,
        accurate: Bool = false,
        minimumConfidence: Double = 0.3
    ) throws -> [TextRun] {
        let bounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let crop = region.integral.intersection(bounds)
        guard crop.width >= 8, crop.height >= 8, let cropped = image.cropping(to: crop) else {
            return []
        }
        let scale = max(1, magnify)
        let enlarged = Self.scaled(cropped, by: scale) ?? cropped
        let found = try runs(
            in: enlarged, accurate: accurate, minimumConfidence: minimumConfidence)
        // Back into the parent image's pixels: undo the magnification, then the crop.
        return found.map { run in
            TextRun(
                string: run.string,
                frame: CGRect(
                    x: crop.minX + run.frame.minX / CGFloat(scale),
                    y: crop.minY + run.frame.minY / CGFloat(scale),
                    width: run.frame.width / CGFloat(scale),
                    height: run.frame.height / CGFloat(scale)),
                confidence: run.confidence)
        }
    }

    static func scaled(_ image: CGImage, by factor: Int) -> CGImage? {
        let width = image.width * factor
        let height = image.height * factor
        guard let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()
    }

    /// Every run Vision found, newest API, in reading order as Vision returns them.
    public static func runs(
        in image: CGImage,
        accurate: Bool = false,
        minimumConfidence: Double = 0.3
    ) throws -> [TextRun] {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = accurate ? .accurate : .fast
        // A clock is not a word. Language correction turns "0:41" into something that
        // reads better and measures nothing.
        request.usesLanguageCorrection = false

        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        do {
            try handler.perform([request])
        } catch {
            throw TextRecognizerError.failed(error.localizedDescription)
        }

        let width = CGFloat(image.width)
        let height = CGFloat(image.height)
        return (request.results ?? []).compactMap { observation in
            guard let candidate = observation.topCandidates(1).first,
                  Double(candidate.confidence) >= minimumConfidence
            else { return nil }
            let box = observation.boundingBox
            return TextRun(
                string: candidate.string,
                frame: CGRect(
                    x: box.origin.x * width,
                    // Vision's origin is bottom-left; ours is top-left.
                    y: (1 - box.origin.y - box.height) * height,
                    width: box.width * width,
                    height: box.height * height),
                confidence: Double(candidate.confidence))
        }
    }
}
