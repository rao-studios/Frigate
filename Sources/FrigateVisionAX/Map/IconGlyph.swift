//
//  IconGlyph.swift
//  FrigateVisionAX
//
//  WHAT: A wordless control, named by what it is drawn as.
//  IN:   VisionEngine.matchIcons
//  OUT:  the page map's label ladder
//  PIN:  A RUNG, NOT AN ORACLE. Every icon scores something against every other, so the
//        floor below is the whole difference between a name and a guess — and an icon
//        that scores under it keeps its position and is still pressable. Nothing here
//        may decide whether a control exists.
//        THE WORD IS WHAT A PERSON WOULD SAY. "search", not "magnifier"; "more", not
//        "kebab". The name is for resolving a phrase against, so it is spoken English
//        rather than an icon designer's vocabulary.
//

import CoreGraphics
import Foundation
import VisionAXCore

public enum IconGlyph: String, Sendable, Equatable, CaseIterable, Codable {
    case search
    case close
    case menu
    case back
    case forward
    case up
    case down
    case add
    case more
    case share
    case microphone
    case notifications
    case account
    case star
    case heart
    case delete
    case done
    case settings
    case home
    case filter
    case cart
    case download

    /// What to call it out loud. Several icons want a second word to be sayable.
    public var spokenName: String {
        switch self {
        case .search: return "search"
        case .close: return "close"
        case .menu: return "menu"
        case .back: return "back"
        case .forward: return "forward"
        case .up: return "up"
        case .down: return "down"
        case .add: return "add"
        case .more: return "more"
        case .share: return "share"
        case .microphone: return "microphone"
        case .notifications: return "notifications"
        case .account: return "account"
        case .star: return "star"
        case .heart: return "like"
        case .delete: return "delete"
        case .done: return "done"
        case .settings: return "settings"
        case .home: return "home"
        case .filter: return "filter"
        case .cart: return "cart"
        case .download: return "download"
        }
    }
}

public struct IconMatch: Sendable, Equatable {
    public var glyph: IconGlyph?
    /// Silhouette overlap, 0…1.
    public var confidence: Double

    public init(glyph: IconGlyph?, confidence: Double) {
        self.glyph = glyph
        self.confidence = confidence
    }

    /// Below this an answer is a coincidence rather than a reading.
    ///
    /// PIN: MEASURED TWICE, AND THE SECOND MEASUREMENT MOVED IT. Independently drawn
    /// icons score their own template between 0.79 and 0.996, which suggested a floor
    /// around 0.55 with room to spare. Then a real watch page put it to work and the
    /// bank named two dozen patches of VIDEO PICTURE — "notifications", "like", "down" —
    /// because a floor low enough to admit every true icon is also low enough to admit
    /// arbitrary content that happens to correlate. 0.72 sits above the picture noise
    /// and below every icon the fixture draws, and the fixture asserts that gap so it
    /// cannot be closed by accident.
    public static let floor: Double = 0.72

    public var name: String? {
        guard let glyph, confidence >= Self.floor else { return nil }
        return glyph.spokenName
    }
}
