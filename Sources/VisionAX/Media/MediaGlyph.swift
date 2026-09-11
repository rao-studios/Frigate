//
//  MediaGlyph.swift
//  VisionAX
//
//  WHAT: What a transport control depicts — the closed vocabulary, Swift side.
//  IN:   vx_media_glyph
//  OUT:  MediaControlDetection
//  PIN:  A CLOSED SET, and `none` is a real answer. A control whose shape nothing
//        matched still has a frame, and a caller may still click it; inventing a
//        name for it is the one thing that would make that click unsafe.
//

import CVisionAX
import Foundation
import VisionAXCore

public enum MediaGlyph: String, Sendable, Equatable, CaseIterable, Codable {
    case none
    case play
    case pause
    case replay
    case volume
    case muted
    case fullscreen
    case exitFullscreen
    case settings
    case captions
    case next
    case previous
    case theater
    case miniplayer

    init(_ raw: vx_media_glyph) {
        switch raw {
        case VX_GLYPH_PLAY: self = .play
        case VX_GLYPH_PAUSE: self = .pause
        case VX_GLYPH_REPLAY: self = .replay
        case VX_GLYPH_VOLUME: self = .volume
        case VX_GLYPH_MUTED: self = .muted
        case VX_GLYPH_FULLSCREEN: self = .fullscreen
        case VX_GLYPH_EXIT_FULLSCREEN: self = .exitFullscreen
        case VX_GLYPH_SETTINGS: self = .settings
        case VX_GLYPH_CAPTIONS: self = .captions
        case VX_GLYPH_NEXT: self = .next
        case VX_GLYPH_PREVIOUS: self = .previous
        case VX_GLYPH_THEATER: self = .theater
        case VX_GLYPH_MINIPLAYER: self = .miniplayer
        default: self = .none
        }
    }

    /// Whether this glyph is the one a player wears while it is PLAYING — the button
    /// says what pressing it would do, so a pause glyph means the video is running.
    public var impliesPlaying: Bool { self == .pause }

    /// …and its mirror: a play or replay glyph means nothing is running right now.
    public var impliesPaused: Bool { self == .play || self == .replay }
}
