//
//  VisionEngine+Perceive.swift
//  VisionAX
//
//  WHAT: The API door — one call, one image conversion, every lane the caller asked for.
//  IN:   a CGImage plus the projection that says where on screen it came from
//  OUT:  VisionScene
//  PIN:  A REGION IS A CROP, AND THE CROP MOVES THE PROJECTION — not the tree. Frames
//        stay in the coordinates of the pixels that were actually examined, and
//        `scene.projection` carries the offset, so a caller that asks for screen
//        coordinates gets the right answer without anything having rewritten a rect.
//        This is also what makes a region cheap: the detector never sees the pixels
//        outside it, so a page's worth of chrome costs nothing to skip.
//        ONE BGRA CONVERSION PER CALL. Every lane reads the same buffer.
//

import CVisionAX
import CoreGraphics
import Foundation
import VisionAXCore

/// Which readings a caller wants. Each costs real time, so none is implied.
public struct VisionLanes: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    /// Canny regions and, when a classifier is supplied, their roles.
    public static let regions = VisionLanes(rawValue: 1 << 0)
    /// Recognized text runs.
    public static let text = VisionLanes(rawValue: 1 << 1)
    /// The media transport.
    public static let media = VisionLanes(rawValue: 1 << 2)

    public static let all: VisionLanes = [.regions, .text, .media]
}

extension VisionEngine {

    /// One look, in whichever lanes were asked for.
    ///
    /// `previous` is a frame captured shortly BEFORE `image`, of the same size and
    /// region. It is what gives the media lane its motion witness; without it the lane
    /// still reads the transport, it just cannot say whether the picture moved.
    public func perceive(
        image: CGImage,
        projection: ScreenProjection,
        region: CGRect? = nil,
        options: CannyOptions = .standard,
        classifier: RegionClassifier? = nil,
        lanes: VisionLanes = .regions,
        previous: CGImage? = nil,
        title: String = "",
        minimumConfidence: Double? = nil,
        wantsEdgeMap: Bool = false,
        namesIcons: Bool = true,
        previousFraction: Double? = nil,
        previousElapsed: TimeInterval? = nil
    ) throws -> VisionScene {
        var timing = VisionTiming.Recorder()
        let bounds = CGRect(x: 0, y: 0, width: image.width, height: image.height)
        let roi = region.map { $0.integral.intersection(bounds) } ?? bounds
        guard roi.width >= 1, roi.height >= 1 else {
            throw VisionEngineError.unreadableImage
        }

        let prepared = try timing.measure(VisionTiming.Name.crop) {
            () -> (CGImage, VisionImageBuffer) in
            let cropped = roi == bounds ? image : image.cropping(to: roi)
            guard let cropped, let buffer = VisionImageBuffer(image: cropped) else {
                throw VisionEngineError.unreadableImage
            }
            return (cropped, buffer)
        }
        let cropped = prepared.0
        let buffer = prepared.1

        var media: MediaControlDetection?
        var text: [TextRun]?

        // TEXT BEFORE REGIONS, when the page's elements are what is wanted. The text
        // lane finds the boxes the edge detector cannot see — a link with no border is
        // invisible to Canny and perfectly legible to recognition — and those boxes have
        // to exist BEFORE the classifier runs, or the rows that matter most are never
        // offered to it. The media lane keeps the opposite order for its own reason,
        // below: its clock is legible only once the bar has been found.
        if lanes.contains(.text), !lanes.contains(.media) {
            // ACCURATE, NOT FAST, WHEN THE PAGE'S WORDS ARE THE POINT. Measured on a
            // real capture: fast recognition returned nine runs for a whole page and
            // spelled them "Sub8cribe" and "23M vlew8" — labels nobody could resolve a
            // phrase against. The clock keeps `.fast` below, where the strip is
            // magnified three times first and the answer is four digits.
            text = try timing.measure(VisionTiming.Name.text) {
                try TextRecognizer.runs(in: cropped, accurate: true)
            }
        }

        var detection: VisionDetection?
        if lanes.contains(.regions) {
            var found = try timing.measure(VisionTiming.Name.detect) {
                try detectRegions(
                    buffer: buffer, title: title, options: options,
                    wantsEdgeMap: wantsEdgeMap)
            }
            if let text, !text.isEmpty {
                found = timing.measure(VisionTiming.Name.union) {
                    ProposalUnion.union(found, textRuns: text)
                }
            }
            if let classifier {
                found = try timing.measure(VisionTiming.Name.classify) {
                    try classifyRegions(
                        buffer: buffer, detection: found, using: classifier,
                        minimumConfidence: minimumConfidence)
                }
            }
            detection = found
        }

        // MEDIA BEFORE TEXT, when both were asked for, and the order is the point: the
        // transport says WHERE the clock is, and reading a small timecode out of a whole
        // page at speed does not work — it is legible only once the bar has been found
        // and that strip alone is enlarged. Text on its own still reads the whole image.
        if lanes.contains(.media) {
            let before: VisionImageBuffer? = previous.flatMap { frame in
                let croppedPrevious = roi == bounds ? frame : frame.cropping(to: roi)
                return croppedPrevious.flatMap(VisionImageBuffer.init(image:))
            }
            let found = try timing.measure(VisionTiming.Name.media) {
                try readMediaControls(
                    buffer: buffer, previous: before, text: nil,
                    previousFraction: previousFraction, previousElapsed: previousElapsed)
            }
            if lanes.contains(.text), let bar = found.bar {
                text = try timing.measure(VisionTiming.Name.clock) {
                    try TextRecognizer.runs(in: cropped, within: bar, magnify: 3)
                }
                media = found.withClock(
                    MediaTimestamp.parse(text ?? [], within: bar),
                    previousElapsed: previousElapsed)
            } else {
                media = found
            }
        }
        if lanes.contains(.text), text == nil {
            text = try timing.measure(VisionTiming.Name.text) {
                try TextRecognizer.runs(in: cropped)
            }
        }

        // WORDLESS CONTROLS, NAMED. Asked only of the boxes that could be icons — small,
        // near-square, nothing written inside them — because the bank is twenty-two
        // overlap tests per box and a page's worth would be spent answering a question
        // about a few dozen of them.
        var icons: [AXNodeID: String] = [:]
        if lanes.contains(.regions), let detection, namesIcons {
            icons = timing.measure(VisionTiming.Name.icons) {
                var named: [AXNodeID: String] = [:]
                let candidates = PageMapBuilder.iconCandidates(
                    in: detection, lines: TextLines.lines(from: text ?? []))
                guard !candidates.isEmpty else { return named }
                let matched = (try? matchIcons(
                    buffer: buffer, boxes: candidates.map { $0.frame })) ?? []
                for (candidate, match) in zip(candidates, matched) {
                    if let name = match.name { named[candidate.id] = name }
                }
                return named
            }
        }

        return VisionScene(
            detection: detection,
            projection: projection.offset(byPixelROI: roi),
            regionOfInterest: roi,
            imageBounds: CGRect(x: 0, y: 0, width: buffer.width, height: buffer.height),
            capturedAt: Date(),
            text: text,
            media: media,
            icons: icons,
            affordances: classifier?.spec.affordances,
            timing: timing.finished(),
            duration: timing.elapsed)
    }

    /// The media transport alone, for a caller that has already cropped to the player.
    public func readMediaControls(
        in image: CGImage,
        previous: CGImage? = nil,
        text: [TextRun]? = nil,
        previousFraction: Double? = nil,
        previousElapsed: TimeInterval? = nil
    ) throws -> MediaControlDetection {
        guard let buffer = VisionImageBuffer(image: image) else {
            throw VisionEngineError.unreadableImage
        }
        let before = previous.flatMap(VisionImageBuffer.init(image:))
        return try readMediaControls(
            buffer: buffer, previous: before, text: text,
            previousFraction: previousFraction, previousElapsed: previousElapsed)
    }

    func readMediaControls(
        buffer: VisionImageBuffer,
        previous: VisionImageBuffer?,
        text: [TextRun]?,
        previousFraction: Double?,
        previousElapsed: TimeInterval?
    ) throws -> MediaControlDetection {
        var reading = vx_media_reading()
        let status: vx_status = buffer.withImageView { view in
            var view = view
            guard let previous else {
                return vx_engine_read_media_controls(handle, &view, nil, &reading)
            }
            return previous.withImageView { previousView in
                var previousView = previousView
                return vx_engine_read_media_controls(
                    handle, &view, &previousView, &reading)
            }
        }
        defer { vx_media_reading_free(&reading) }
        guard status == VX_OK else {
            throw VisionEngineError.engine(
                status: status.rawValue,
                message: String(cString: vx_status_message(status)))
        }

        func rect(_ raw: vx_rect) -> CGRect? {
            guard raw.width > 0, raw.height > 0 else { return nil }
            return CGRect(
                x: CGFloat(raw.x), y: CGFloat(raw.y),
                width: CGFloat(raw.width), height: CGFloat(raw.height))
        }
        func control(_ raw: vx_media_control) -> MediaControlDetection.Control? {
            guard let frame = rect(raw.frame) else { return nil }
            return MediaControlDetection.Control(
                frame: frame, glyph: MediaGlyph(raw.glyph), confidence: raw.confidence)
        }

        var controls: [MediaControlDetection.Control] = []
        if let base = reading.controls, reading.control_count > 0 {
            controls = UnsafeBufferPointer(start: base, count: Int(reading.control_count))
                .compactMap(control)
        }
        let bar = rect(reading.bar)
        let progress = rect(reading.progress).map {
            MediaControlDetection.Progress(
                frame: $0,
                fraction: reading.progress_fraction >= 0 ? reading.progress_fraction : 0)
        }
        let volumeTrack = rect(reading.volume_track).map {
            MediaControlDetection.Progress(
                frame: $0,
                fraction: reading.volume_fraction >= 0 ? reading.volume_fraction : 0)
        }
        // A clock is only read inside the bar: text elsewhere on the page is a caption
        // or a comment, and parsing it would report a video's length as a stranger's.
        let clock = MediaTimestamp.parse(text ?? [], within: bar)

        return MediaControlDetection.derive(
            controlsVisible: reading.controls_visible != 0,
            bar: bar,
            progress: reading.progress_fraction >= 0 ? progress : nil,
            motion: reading.motion >= 0 ? reading.motion : nil,
            controls: controls,
            centerGlyph: control(reading.center_glyph),
            elapsed: clock.elapsed,
            duration: clock.duration,
            volumeTrack: volumeTrack,
            previousFraction: previousFraction,
            previousElapsed: previousElapsed)
    }
}
