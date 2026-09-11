//
//  MediaControlDetection.swift
//  VisionAX
//
//  WHAT: One look at a player's transport — where the controls are and what the
//        evidence says about whether it is playing.
//  IN:   VisionEngine.readMediaControls
//  OUT:  Mary's browsing lane; the bench's media overlay
//  PIN:  WITNESSES, NOT A VERDICT DRESSED AS ONE. `playback` is derived from three
//        independent observations — the picture moved, the progress fraction advanced,
//        the clock advanced — and `witnesses` names the ones that actually spoke. When
//        they disagree, a caller can see that they did instead of being handed the
//        majority answer with the disagreement thrown away.
//        SLOTS ARE POSITIONAL. The leftmost control in the bar is play/pause and the
//        rightmost is fullscreen in every player anyone has shipped; that layout, not
//        the artwork, is what makes this work on a player it has never seen.
//

import CoreGraphics
import Foundation
import VisionAXCore

public struct MediaControlDetection: Sendable, Equatable {

    public enum Playback: String, Sendable, Equatable, Codable {
        case playing
        case paused
        /// Nothing spoke. NOT "stopped" — a read that found no evidence and a player
        /// that is genuinely idle must stay distinguishable.
        case unknown
    }

    public struct Control: Sendable, Equatable {
        /// Image pixel space, like every other frame in a detection.
        public var frame: CGRect
        public var glyph: MediaGlyph
        /// Silhouette overlap, 0...1.
        public var confidence: Double

        public init(frame: CGRect, glyph: MediaGlyph, confidence: Double) {
            self.frame = frame
            self.glyph = glyph
            self.confidence = confidence
        }
    }

    public struct Progress: Sendable, Equatable {
        public var frame: CGRect
        /// How far along, 0...1.
        public var fraction: Double

        public init(frame: CGRect, fraction: Double) {
            self.frame = frame
            self.fraction = fraction
        }
    }

    /// True when a transport was found at all. False means the controls are hidden,
    /// which is a state, not a failure — hover and read again.
    public var controlsVisible: Bool
    /// The whole control bar band, when one was found.
    public var bar: CGRect?
    public var progress: Progress?
    /// How much the picture changed against the previous frame, 0...1; nil when only
    /// one frame was given.
    public var motion: Double?
    /// Every control found in the bar, left to right.
    public var controls: [Control]
    /// The big glyph drawn over the middle of a paused picture, when there is one.
    public var centerGlyph: Control?
    public var playback: Playback
    /// The observations that decided `playback`, in the order they were weighed.
    public var witnesses: [String]
    /// Read off the clock in the bar, when the text lane ran.
    public var elapsed: TimeInterval?
    public var duration: TimeInterval?
    /// The short track beside the volume control. Present only while the pointer is
    /// over that control, so nil usually means "not revealed" rather than "not there".
    public var volumeTrack: Progress?

    public init(
        controlsVisible: Bool = false,
        bar: CGRect? = nil,
        progress: Progress? = nil,
        volumeTrack: Progress? = nil,
        motion: Double? = nil,
        controls: [Control] = [],
        centerGlyph: Control? = nil,
        playback: Playback = .unknown,
        witnesses: [String] = [],
        elapsed: TimeInterval? = nil,
        duration: TimeInterval? = nil
    ) {
        self.controlsVisible = controlsVisible
        self.bar = bar
        self.progress = progress
        self.volumeTrack = volumeTrack
        self.motion = motion
        self.controls = controls
        self.centerGlyph = centerGlyph
        self.playback = playback
        self.witnesses = witnesses
        self.elapsed = elapsed
        self.duration = duration
    }

    // MARK: - The slots

    /// The transport button: the LEFTMOST control in the bar.
    public var playPause: Control? {
        // A glyph that names itself outranks position — a player that puts previous
        // before play would otherwise hand back the wrong button, and the cost of that
        // mistake is a skipped track instead of a pause.
        if let named = controls.first(where: { $0.glyph == .play || $0.glyph == .pause || $0.glyph == .replay }),
           named.confidence >= Self.namedGlyphFloor {
            return named
        }
        return controls.first
    }

    /// Fullscreen: the RIGHTMOST control.
    public var fullscreen: Control? {
        if let named = controls.last(where: { $0.glyph == .fullscreen || $0.glyph == .exitFullscreen }),
           named.confidence >= Self.namedGlyphFloor {
            return named
        }
        return controls.count >= 2 ? controls.last : nil
    }

    /// Volume: a speaker-shaped control if one was named nearby, else the control
    /// immediately after the transport.
    ///
    /// PIN: THE POSITION IS GOOD ENOUGH TO PRESS, AND NOT GOOD ENOUGH TO READ. Every
    /// player puts volume second, right after play, so a click aimed there lands on the
    /// right button — and a real YouTube speaker reads as `next` against these drawn
    /// silhouettes about half the time, so requiring the name would refuse to mute a
    /// video whose mute button is plainly there.
    /// BOTH RULES ARE SCOPED TO THE SAME NARROW REGION, because a speaker shape found in
    /// the MIDDLE of the bar is not a volume control: measured, it was a scrap of video
    /// showing between the buttons, named `volume` at 0.51, and preferring it over the
    /// real speaker sitting second from the left aimed a mute at nothing at all.
    public var volume: Control? {
        let leading = controls.filter { control in
            guard let bar, bar.width > 0 else { return true }
            return control.frame.midX < bar.minX + bar.width * Self.volumeReach
        }
        if let named = leading
            .filter({ $0.glyph == .volume || $0.glyph == .muted })
            .max(by: { $0.confidence < $1.confidence }) {
            return named
        }
        guard let transport = playPause,
              let next = leading.first(where: { $0.frame.minX > transport.frame.maxX })
        else { return nil }
        return next
    }

    /// Whether the volume control says the player is muted. Nil when no volume control
    /// was NAMED — a read that failed, not a player with sound on. A positional guess
    /// never answers this.
    public var isMuted: Bool? {
        guard let volume else { return nil }
        switch volume.glyph {
        case .muted: return true
        case .volume: return false
        default: return nil
        }
    }

    /// Below this, a glyph name is not worth outranking position with.
    static let namedGlyphFloor = 0.28
    /// How far along the bar a volume control may sit, as a share of its width.
    static let volumeReach = 0.3

    // MARK: - Deriving the verdict

    /// Motion above this counts as the picture moving. Low, because a talking head on a
    /// static background moves a small share of a coarse grid; noise sits below it.
    public static let motionFloor = 0.02
    /// The progress fraction must move by more than the track's own quantization —
    /// one pixel on a 1000px track is 0.001 — before it counts as advancing.
    public static let fractionFloor = 0.002

    /// The reading, with playback decided from every witness available.
    ///
    /// `previousFraction` and `previousElapsed` come from an EARLIER reading, which is
    /// what lets a caller prove a seek landed rather than merely that a click happened.
    public static func derive(
        controlsVisible: Bool,
        bar: CGRect?,
        progress: Progress?,
        motion: Double?,
        controls: [Control],
        centerGlyph: Control?,
        elapsed: TimeInterval? = nil,
        duration: TimeInterval? = nil,
        volumeTrack: Progress? = nil,
        previousFraction: Double? = nil,
        previousElapsed: TimeInterval? = nil
    ) -> MediaControlDetection {
        var witnesses: [String] = []
        // STRONG AND WEAK EVIDENCE ARE NOT THE SAME THING, and treating them alike was a
        // measured bug. A picture that MOVED is proof it is playing. A picture that did
        // not move is only consistent with paused — a held shot, a title card, a slow
        // pan below the grid's resolution all look identical to a stopped video, and one
        // of those was measured reporting a running video as paused while its own button
        // plainly showed a pause glyph.
        var playing = false
        var weaklyPaused = false

        if let motion, motion >= 0 {
            if motion > motionFloor {
                witnesses.append(String(format: "picture moved (%.3f)", motion))
                playing = true
            } else {
                witnesses.append(String(format: "picture still (%.3f)", motion))
                weaklyPaused = true
            }
        }
        if let fraction = progress?.fraction, let previousFraction {
            let delta = fraction - previousFraction
            if delta > fractionFloor {
                witnesses.append(String(format: "progress advanced (%.3f)", delta))
                playing = true
            } else if abs(delta) <= fractionFloor {
                witnesses.append("progress held")
                weaklyPaused = true
            }
        }
        if let elapsed, let previousElapsed {
            if elapsed > previousElapsed {
                witnesses.append("clock advanced")
                playing = true
            } else {
                witnesses.append("clock held")
                weaklyPaused = true
            }
        }

        // THE BUTTON SAYS WHAT PRESSING IT WOULD DO, so a pause glyph means the video is
        // running. It outranks stillness — which proves nothing on its own — and yields
        // to movement, which proves everything.
        let interim = MediaControlDetection(
            controlsVisible: controlsVisible, bar: bar, progress: progress,
            motion: motion, controls: controls, centerGlyph: centerGlyph)
        var glyphSays: Playback = .unknown
        if let transport = interim.playPause, transport.confidence >= namedGlyphFloor {
            if transport.glyph.impliesPlaying {
                glyphSays = .playing
            } else if transport.glyph.impliesPaused {
                glyphSays = .paused
            }
        }
        if glyphSays == .unknown, let centerGlyph, centerGlyph.glyph.impliesPaused,
           centerGlyph.confidence >= namedGlyphFloor {
            glyphSays = .paused
            witnesses.append("centered play glyph")
        } else if glyphSays != .unknown {
            witnesses.append(glyphSays == .playing ? "pause glyph" : "play glyph")
        }

        let verdict: Playback
        if playing {
            verdict = .playing
        } else if glyphSays != .unknown {
            verdict = glyphSays
        } else if weaklyPaused {
            verdict = .paused
        } else {
            verdict = .unknown
        }

        return MediaControlDetection(
            controlsVisible: controlsVisible,
            bar: bar,
            progress: progress,
            volumeTrack: volumeTrack,
            motion: motion,
            controls: controls,
            centerGlyph: centerGlyph,
            playback: verdict,
            witnesses: witnesses,
            elapsed: elapsed,
            duration: duration)
    }

    /// The same reading with a clock attached, and playback re-derived now that the
    /// clock can speak as a witness.
    public func withClock(
        _ clock: MediaTimestamp.Clock, previousElapsed: TimeInterval?
    ) -> MediaControlDetection {
        MediaControlDetection.derive(
            controlsVisible: controlsVisible, bar: bar, progress: progress, motion: motion,
            controls: controls, centerGlyph: centerGlyph,
            elapsed: clock.elapsed, duration: clock.duration,
            volumeTrack: volumeTrack,
            previousFraction: nil, previousElapsed: previousElapsed)
    }

    /// The same reading with its frames moved into another space — how a detection
    /// taken over a cropped region is reported in the parent image's coordinates.
    public func offset(by delta: CGPoint) -> MediaControlDetection {
        func moved(_ rect: CGRect) -> CGRect { rect.offsetBy(dx: delta.x, dy: delta.y) }
        var copy = self
        copy.bar = bar.map(moved)
        copy.volumeTrack = volumeTrack.map { Progress(frame: moved($0.frame), fraction: $0.fraction) }
        copy.progress = progress.map { Progress(frame: moved($0.frame), fraction: $0.fraction) }
        copy.controls = controls.map { Control(frame: moved($0.frame), glyph: $0.glyph, confidence: $0.confidence) }
        copy.centerGlyph = centerGlyph.map { Control(frame: moved($0.frame), glyph: $0.glyph, confidence: $0.confidence) }
        return copy
    }
}
