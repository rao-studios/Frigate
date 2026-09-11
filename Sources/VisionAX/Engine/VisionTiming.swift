//
//  VisionTiming.swift
//  VisionAX
//
//  WHAT: Where one perception's time went, phase by phase.
//  IN:   VisionEngine.perceive  OUT: VisionScene.timing
//  PIN:  MEASURED FROM INSIDE, BECAUSE FROM OUTSIDE IT IS NOT THE SAME CALL. Timing the
//        lanes by invoking them separately double-counts the image conversion, misses
//        the union entirely, and cannot see that the classifier is handed a buffer the
//        detector already built — the parts add up to more than the whole, which is a
//        way of being wrong about which part to improve.
//        THE PHASES SUM TO THE TOTAL, and `other` carries the remainder rather than
//        letting it vanish. A breakdown that does not add up is a breakdown nobody can
//        act on.
//        ALWAYS ON. Two clock reads per phase against tens of milliseconds of work is
//        not a cost worth a flag, and a measurement you have to remember to switch on is
//        one you do not have when it matters.
//

import Foundation
import VisionAXCore

public struct VisionTiming: Sendable, Equatable {

    public struct Phase: Sendable, Equatable {
        /// What was being done. Stable strings — a caller may key on them.
        public var name: String
        public var duration: Duration

        public init(name: String, duration: Duration) {
            self.name = name
            self.duration = duration
        }
    }

    /// In the order they ran.
    public var phases: [Phase]
    /// Entry to exit, including whatever the phases did not name.
    public var total: Duration

    public init(phases: [Phase] = [], total: Duration = .zero) {
        self.phases = phases
        self.total = total
    }

    /// The phase names this engine reports, so a consumer can lay out a table before it
    /// has a reading in hand.
    public enum Name {
        public static let crop = "crop + buffer"
        public static let text = "text"
        public static let detect = "detect"
        public static let union = "union text"
        public static let classify = "classify"
        public static let icons = "icons"
        public static let media = "media"
        public static let clock = "clock"
        /// Everything the named phases did not account for.
        public static let other = "other"
    }

    /// What the phases left unexplained — assembly, bookkeeping, the guard clauses.
    public var unaccounted: Duration {
        let named = phases.reduce(Duration.zero) { $0 + $1.duration }
        return total > named ? total - named : .zero
    }

    /// The phases plus the remainder, which is what a table should show.
    public var accounted: [Phase] {
        let remainder = unaccounted
        guard remainder > .milliseconds(0) else { return phases }
        return phases + [Phase(name: Name.other, duration: remainder)]
    }

    /// One line per phase and a total, for a probe or a log.
    ///
    /// PIN: PADDED IN SWIFT, NOT BY `%-14@`. A width on a `%@` does not pad an NSString
    /// on this platform, and a table whose columns do not line up is a table nobody
    /// reads down.
    public func table(indent: String = "  ") -> String {
        func row(_ name: String, _ duration: Duration, _ share: Double) -> String {
            let label = name.padding(toLength: 14, withPad: " ", startingAt: 0)
            let milliseconds = String(format: "%7.1fms", duration.milliseconds)
            let percent = String(format: "%4.0f%%", share)
            return "\(indent)\(label) \(milliseconds)  \(percent)"
        }
        let rows = accounted.map { phase in
            row(phase.name, phase.duration,
                total > .zero
                    ? phase.duration.milliseconds / total.milliseconds * 100
                    : 0)
        }
        return (rows + [row("TOTAL", total, 100)]).joined(separator: "\n")
    }

    // MARK: - Recording

    /// Collects phases while a perception runs. Not thread-safe by design: one
    /// perception is one call on one thread.
    struct Recorder {
        private var phases: [Phase] = []
        private let started = ContinuousClock.now

        mutating func measure<Value>(_ name: String, _ body: () throws -> Value) rethrows -> Value {
            let began = ContinuousClock.now
            let value = try body()
            phases.append(Phase(name: name, duration: began.duration(to: ContinuousClock.now)))
            return value
        }

        /// Fold a second stretch into a phase that already ran — the text lane reads the
        /// page once and, on the media path, the clock strip afterwards.
        mutating func add(_ name: String, _ duration: Duration) {
            if let index = phases.firstIndex(where: { $0.name == name }) {
                phases[index].duration += duration
            } else {
                phases.append(Phase(name: name, duration: duration))
            }
        }

        func finished() -> VisionTiming {
            VisionTiming(phases: phases, total: started.duration(to: ContinuousClock.now))
        }

        var elapsed: Duration { started.duration(to: ContinuousClock.now) }
    }
}

public extension Duration {
    /// Milliseconds as a Double. Every timing this package prints goes through here, so
    /// the arithmetic is written once.
    var milliseconds: Double {
        let parts = components
        return Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
    }
}
