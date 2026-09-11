//
//  MediaTimestamp.swift
//  VisionAX
//
//  WHAT: A player's clock, read out of recognized text.
//  IN:   [TextRun] from the bar
//  OUT:  elapsed / duration seconds
//  PIN:  PURE, AND SCOPED TO THE BAR. Timestamps look like ordinary text, and a page
//        full of comments has plenty of them; parsing anything outside the transport
//        would report a stranger's quoted timecode as this video's length.
//        A LONE TIME IS ELAPSED, NEVER DURATION. Players that show one number show the
//        position; guessing the other way round would make every seek look wrong.
//

import CoreGraphics
import Foundation
import VisionAXCore

public enum MediaTimestamp {

    public struct Clock: Sendable, Equatable {
        public var elapsed: TimeInterval?
        public var duration: TimeInterval?
        /// What a player counting down reports, when that is the only figure it shows.
        public var remaining: TimeInterval?

        public init(elapsed: TimeInterval? = nil, duration: TimeInterval? = nil, remaining: TimeInterval? = nil) {
            self.elapsed = elapsed
            self.duration = duration
            self.remaining = remaining
        }
    }

    /// `h:mm:ss`, `m:ss`, and the leading `-` a countdown wears.
    static let pattern = try? NSRegularExpression(
        pattern: #"(-?)(?:(\d{1,2}):)?(\d{1,2}):(\d{2})"#)

    /// Seconds for one timecode, or nil. Negative means the text counted down.
    public static func seconds(in text: String) -> (value: TimeInterval, isNegative: Bool)? {
        guard let pattern else { return nil }
        let range = NSRange(text.startIndex..., in: text)
        guard let match = pattern.firstMatch(in: text, range: range) else { return nil }

        func group(_ index: Int) -> String? {
            guard let range = Range(match.range(at: index), in: text) else { return nil }
            return String(text[range])
        }
        let negative = (group(1) ?? "") == "-"
        let hours = group(2).flatMap(Double.init) ?? 0
        guard let minutes = group(3).flatMap(Double.init),
              let secondsValue = group(4).flatMap(Double.init),
              secondsValue < 60
        else { return nil }
        return (hours * 3600 + minutes * 60 + secondsValue, negative)
    }

    /// The clock a transport's text runs describe.
    ///
    /// `bar` scopes the search. Nil means "these runs are already the bar's" — a caller
    /// that cropped to the transport itself.
    public static func parse(_ runs: [TextRun], within bar: CGRect?) -> Clock {
        let scoped = bar.map { frame in
            runs.filter { $0.frame.intersects(frame) }
        } ?? runs

        // A player writes "1:23 / 4:56" as one run or as two beside each other; both
        // are handled by reading every timecode in bar order and taking the first two.
        var found: [(value: TimeInterval, isNegative: Bool)] = []
        for run in scoped.sorted(by: { $0.frame.minX < $1.frame.minX }) {
            var remainder = Substring(run.string)
            while let parsed = seconds(in: String(remainder)) {
                found.append(parsed)
                guard let pattern,
                      let match = pattern.firstMatch(
                        in: String(remainder),
                        range: NSRange(remainder.startIndex..., in: remainder)),
                      let range = Range(match.range, in: remainder)
                else { break }
                remainder = remainder[range.upperBound...]
                if found.count >= 4 { break }
            }
        }

        guard let first = found.first else { return Clock() }
        if first.isNegative {
            // A countdown names what is LEFT; with a second figure the total is known.
            let total = found.dropFirst().first(where: { !$0.isNegative })?.value
            return Clock(
                elapsed: total.map { $0 - first.value },
                duration: total,
                remaining: first.value)
        }
        let second = found.dropFirst().first(where: { !$0.isNegative })?.value
        // A duration below the position is a misread, not a video that ends before it
        // starts — drop it rather than publish an impossible pair.
        let duration = second.flatMap { $0 >= first.value ? $0 : nil }
        return Clock(elapsed: first.value, duration: duration)
    }
}
