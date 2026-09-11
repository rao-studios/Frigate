//
//  SyntheticPlayerSupport.swift
//  VisionAXTests
//
//  WHAT: A drawn video player with a known transport, so the media lane can be pinned
//        without a network, a browser, or a screenshot in the repository.
//  OUT:  MediaControlTests
//  PIN:  THIS PROVES THE PIPELINE, NOT THE ARTWORK. A drawn player exercises the band
//        scan, the blob threshold, slot assignment and the witness derivation against
//        known answers; whether the glyph bank recognises YOUTUBE's particular play
//        triangle is a question only a real capture answers, and that lives in the
//        live probe. Keeping the two apart is what stops a green suite from being read
//        as "it works on the web".
//        The picture is NOISE on purpose: a flat background would let the band scan
//        find a "track" anywhere, and the whole point of the scan is that it does not.
//

import CoreGraphics
import Foundation
import VisionAX

enum SyntheticPlayer {

    struct Layout {
        static let size = CGSize(width: 1280, height: 720)
        /// The transport scrim, across the bottom.
        static let bar = CGRect(x: 0, y: 640, width: 1280, height: 80)
        /// The progress track: inset, thin, exactly what a real player draws.
        static let track = CGRect(x: 24, y: 652, width: 1232, height: 5)
        /// Where the control glyphs sit, left to right.
        static let controlY: CGFloat = 674
        static let controlSide: CGFloat = 28
        static let firstControlX: CGFloat = 28
        static let controlGap: CGFloat = 46
        /// How many trailing slots are pinned to the RIGHT end of the bar.
        ///
        /// PIN: A REAL PLAYER SPANS ITS BAR. Play sits at the far left and full screen
        /// at the far right, with the middle empty — that span is a structural signal
        /// the detector relies on to tell a transport from a row of icons sitting under
        /// an unrelated line. A fixture that bunched every control at the left was not
        /// a player, and it hid exactly the false positive the rule exists to catch.
        static let trailingSlots = 3
    }

    /// What to draw in each control slot, left to right.
    enum Slot {
        case play
        case pause
        case next
        case volume
        case muted
        case captions
        case settings
        case fullscreen
    }

    /// A player frame.
    ///
    /// `fraction` is how much of the track is filled. `seed` shifts the picture noise,
    /// which is how a caller makes two frames that differ (a playing video) or two that
    /// do not (a paused one).
    static func image(
        fraction: Double = 0.35,
        slots: [Slot] = [.play, .next, .volume, .captions, .settings, .fullscreen],
        controlsVisible: Bool = true,
        centerGlyph: Slot? = nil,
        seed: Int = 0,
        size: CGSize = Layout.size
    ) -> CGImage {
        let width = Int(size.width)
        let height = Int(size.height)
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue)!

        // The picture: coarse blocks of varying grey, shifted by the seed.
        var generator = SplitMix64(seed: UInt64(bitPattern: Int64(seed &+ 1)))
        let block = 40
        for y in stride(from: 0, to: height, by: block) {
            for x in stride(from: 0, to: width, by: block) {
                let level = 0.15 + Double(generator.next() % 60) / 255.0
                context.setFillColor(CGColor(red: level, green: level, blue: level, alpha: 1))
                context.fill(CGRect(x: x, y: y, width: block, height: block))
            }
        }

        guard controlsVisible else {
            if let centerGlyph { draw(centerGlyph, in: centerRect(size), into: context, size: size) }
            return context.makeImage()!
        }

        // The scrim, drawn dark so the glyphs read against it the way a player's does.
        context.setFillColor(CGColor(red: 0.04, green: 0.04, blue: 0.04, alpha: 1))
        context.fill(flipped(Layout.bar, in: size))

        // The track: played (red) then unplayed (grey), which is the two-tone step the
        // band scan looks for.
        let track = Layout.track
        // A PLAYER ALWAYS SHOWS SOME ACCENT, even at zero. YouTube draws a small red nub
        // and a scrubber knob on an unstarted video; a fixture with none of that is not
        // a player, and a bar with no colour in it anywhere is indistinguishable from a
        // page divider by anything looking at pixels.
        let split = max(
            track.minX + 4,
            track.minX + track.width * CGFloat(min(max(fraction, 0), 1)))
        context.setFillColor(CGColor(red: 0.35, green: 0.35, blue: 0.35, alpha: 1))
        context.fill(flipped(track, in: size))
        context.setFillColor(CGColor(red: 0.9, green: 0.05, blue: 0.1, alpha: 1))
        context.fill(flipped(
            CGRect(x: track.minX, y: track.minY, width: split - track.minX, height: track.height),
            in: size))

        for (rect, slot) in zip(layout(count: slots.count, width: size.width), slots) {
            draw(slot, in: rect, into: context, size: size)
        }
        if let centerGlyph { draw(centerGlyph, in: centerRect(size), into: context, size: size) }
        return context.makeImage()!
    }

    /// Where each control is drawn: a leading group from the left, a trailing group
    /// pinned to the right — the arrangement every player uses.
    static func layout(count: Int, width: CGFloat = Layout.size.width) -> [CGRect] {
        guard count > 0 else { return [] }
        let trailing = min(Layout.trailingSlots, max(0, count - 1))
        let leading = count - trailing
        var rects: [CGRect] = []
        for index in 0..<leading {
            rects.append(CGRect(
                x: Layout.firstControlX + CGFloat(index) * Layout.controlGap,
                y: Layout.controlY, width: Layout.controlSide, height: Layout.controlSide))
        }
        let rightEdge = width - Layout.firstControlX
        for index in 0..<trailing {
            let fromRight = CGFloat(trailing - 1 - index)
            rects.append(CGRect(
                x: rightEdge - Layout.controlSide - fromRight * Layout.controlGap,
                y: Layout.controlY, width: Layout.controlSide, height: Layout.controlSide))
        }
        return rects
    }

    /// The screen rect of the control at `index`, in top-left coordinates.
    static func controlRect(at index: Int, of count: Int = 6) -> CGRect {
        layout(count: count)[index]
    }

    private static func centerRect(_ size: CGSize) -> CGRect {
        CGRect(x: size.width / 2 - 34, y: size.height / 2 - 34, width: 68, height: 68)
    }

    /// Top-left rect → the y-up context's coordinates.
    private static func flipped(_ rect: CGRect, in size: CGSize) -> CGRect {
        CGRect(x: rect.minX, y: size.height - rect.maxY, width: rect.width, height: rect.height)
    }

    private static func draw(_ slot: Slot, in rect: CGRect, into context: CGContext, size: CGSize) {
        let box = flipped(rect, in: size)
        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.setStrokeColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        let w = box.width
        let h = box.height
        switch slot {
        case .play:
            context.beginPath()
            context.move(to: CGPoint(x: box.minX + w * 0.25, y: box.minY + h * 0.1))
            context.addLine(to: CGPoint(x: box.minX + w * 0.25, y: box.minY + h * 0.9))
            context.addLine(to: CGPoint(x: box.minX + w * 0.82, y: box.midY))
            context.closePath()
            context.fillPath()
        case .pause:
            context.fill(CGRect(x: box.minX + w * 0.26, y: box.minY + h * 0.13, width: w * 0.17, height: h * 0.74))
            context.fill(CGRect(x: box.minX + w * 0.56, y: box.minY + h * 0.13, width: w * 0.17, height: h * 0.74))
        case .next:
            context.beginPath()
            context.move(to: CGPoint(x: box.minX + w * 0.16, y: box.minY + h * 0.13))
            context.addLine(to: CGPoint(x: box.minX + w * 0.16, y: box.minY + h * 0.87))
            context.addLine(to: CGPoint(x: box.minX + w * 0.64, y: box.midY))
            context.closePath()
            context.fillPath()
            context.fill(CGRect(x: box.minX + w * 0.70, y: box.minY + h * 0.13, width: w * 0.13, height: h * 0.74))
        case .volume, .muted:
            context.fill(CGRect(x: box.minX + w * 0.10, y: box.minY + h * 0.38, width: w * 0.17, height: h * 0.24))
            context.beginPath()
            context.move(to: CGPoint(x: box.minX + w * 0.27, y: box.minY + h * 0.5))
            context.addLine(to: CGPoint(x: box.minX + w * 0.52, y: box.minY + h * 0.86))
            context.addLine(to: CGPoint(x: box.minX + w * 0.52, y: box.minY + h * 0.14))
            context.addLine(to: CGPoint(x: box.minX + w * 0.27, y: box.minY + h * 0.38))
            context.closePath()
            context.fillPath()
            context.setLineWidth(max(2, w * 0.09))
            if slot == .volume {
                context.addArc(center: CGPoint(x: box.minX + w * 0.52, y: box.midY),
                               radius: w * 0.20, startAngle: -0.9, endAngle: 0.9, clockwise: false)
                context.strokePath()
                context.addArc(center: CGPoint(x: box.minX + w * 0.52, y: box.midY),
                               radius: w * 0.34, startAngle: -0.9, endAngle: 0.9, clockwise: false)
                context.strokePath()
            } else {
                context.move(to: CGPoint(x: box.minX + w * 0.62, y: box.minY + h * 0.32))
                context.addLine(to: CGPoint(x: box.minX + w * 0.92, y: box.minY + h * 0.68))
                context.move(to: CGPoint(x: box.minX + w * 0.92, y: box.minY + h * 0.32))
                context.addLine(to: CGPoint(x: box.minX + w * 0.62, y: box.minY + h * 0.68))
                context.strokePath()
            }
        case .captions:
            context.setLineWidth(max(2, w * 0.09))
            context.stroke(CGRect(x: box.minX + w * 0.08, y: box.minY + h * 0.24, width: w * 0.84, height: h * 0.52))
            context.fill(CGRect(x: box.minX + w * 0.24, y: box.midY - h * 0.05, width: w * 0.22, height: h * 0.10))
            context.fill(CGRect(x: box.minX + w * 0.56, y: box.midY - h * 0.05, width: w * 0.22, height: h * 0.10))
        case .settings:
            context.setLineWidth(max(3, w * 0.18))
            context.addArc(center: CGPoint(x: box.midX, y: box.midY), radius: w * 0.34,
                           startAngle: 0, endAngle: .pi * 2, clockwise: false)
            context.strokePath()
        case .fullscreen:
            let arm = w * 0.28
            let t = max(2, w * 0.10)
            context.fill(CGRect(x: box.minX + w * 0.10, y: box.maxY - h * 0.10 - t, width: arm, height: t))
            context.fill(CGRect(x: box.minX + w * 0.10, y: box.maxY - h * 0.10 - arm, width: t, height: arm))
            context.fill(CGRect(x: box.maxX - w * 0.10 - arm, y: box.maxY - h * 0.10 - t, width: arm, height: t))
            context.fill(CGRect(x: box.maxX - w * 0.10 - t, y: box.maxY - h * 0.10 - arm, width: t, height: arm))
            context.fill(CGRect(x: box.minX + w * 0.10, y: box.minY + h * 0.10, width: arm, height: t))
            context.fill(CGRect(x: box.minX + w * 0.10, y: box.minY + h * 0.10, width: t, height: arm))
            context.fill(CGRect(x: box.maxX - w * 0.10 - arm, y: box.minY + h * 0.10, width: arm, height: t))
            context.fill(CGRect(x: box.maxX - w * 0.10 - t, y: box.minY + h * 0.10, width: t, height: arm))
        }
    }

    /// A WATCH PAGE: the player at the top, a page of content underneath.
    ///
    /// This is the layout that matters, and it is not the one a naive detector expects.
    /// A real watch page is mostly comments and thumbnails; the transport sits around
    /// 60% of the way up, not at the bottom of what the caller captured.
    static func watchPage(
        fraction: Double = 0.35,
        slots: [Slot] = [.play, .next, .volume, .captions, .settings, .fullscreen],
        seed: Int = 0,
        playerSize: CGSize = Layout.size,
        pageHeight: CGFloat = 1800
    ) -> CGImage {
        let player = image(fraction: fraction, slots: slots, seed: seed, size: playerSize)
        let width = Int(playerSize.width)
        let height = Int(pageHeight)
        let context = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue)!

        // A light page with rules and text-shaped blocks — the decorations that look
        // like a progress track to anything that only checks for a flat two-tone row.
        context.setFillColor(CGColor(red: 0.99, green: 0.99, blue: 0.99, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: CGFloat(width), height: pageHeight))
        var generator = SplitMix64(seed: 99)
        var y = pageHeight - playerSize.height - 60
        while y > 40 {
            context.setFillColor(CGColor(red: 0.85, green: 0.85, blue: 0.85, alpha: 1))
            // A full-width horizontal rule: flat, one colour, exactly the shape a track
            // scan must refuse when nothing sits under it.
            context.fill(CGRect(x: 0, y: y, width: CGFloat(width), height: 2))
            context.setFillColor(CGColor(red: 0.2, green: 0.2, blue: 0.2, alpha: 1))
            let runs = 2 + Int(generator.next() % 4)
            for line in 0..<runs {
                let lineWidth = CGFloat(200 + generator.next() % 600)
                context.fill(CGRect(x: 40, y: y - 24 - CGFloat(line) * 22, width: lineWidth, height: 12))
            }
            y -= 140
        }

        // The player, at the TOP of the page (y-up context, so the top is high y).
        context.draw(player, in: CGRect(
            x: 0, y: pageHeight - playerSize.height,
            width: playerSize.width, height: playerSize.height))
        return context.makeImage()!
    }
}

/// A tiny deterministic generator — the picture must be the same every run, or a
/// motion assertion becomes a coin flip.
struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
