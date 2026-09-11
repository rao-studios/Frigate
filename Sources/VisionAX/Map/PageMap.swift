//
//  PageMap.swift
//  VisionAX
//
//  WHAT: A page as things one can act on — named, grouped, and in reading order.
//  IN:   VisionScene.pageMap()
//  OUT:  Mary's page lane; the bench overlay
//  PIN:  READING ORDER IS THE INVARIANT, AND IT IS THE ONLY ORDINAL. Groups run top to
//        bottom, members run band by band and then leading to trailing. A consumer that
//        says "the first result" counts this array; there is no second ordinal stored
//        anywhere that could disagree with it.
//        NOTHING IS DROPPED FOR WANT OF A NAME. The roster's old rule — no label, no
//        row — is exactly what made a page of results look empty. Every box that can be
//        acted on survives, carrying WHERE its name came from, so a caller can tell a
//        classifier's word from a guess.
//        PIXELS, LIKE EVERY OTHER DETECTION. The scene projects to screen points; this
//        does not, so nothing here has to know a scale existed.
//

import CoreGraphics
import Foundation
import VisionAXCore

/// What a box can have done to it. A coarsening of role, not a second opinion about it.
public enum PageAffordance: String, Sendable, Equatable, CaseIterable, Codable {
    case press
    case fill
    case adjust
    case scroll
    case none

    /// What a shipped model says its roles afford, when it says anything.
    ///
    /// A spec's table wins over the built-in one below, because the table travels with
    /// the vocabulary it describes.
    public static func derived(role: String?, table: [String: [String]]?) -> PageAffordance {
        guard let role, let table else { return derived(role: role) }
        for (group, members) in table where members.contains(role) {
            if let affordance = PageAffordance(rawValue: group) { return affordance }
        }
        return .none
    }

    /// The standard table: which roles afford what. Data, not a judgement — the same
    /// grouping the training pipeline uses to score the classifier's marginals.
    public static func derived(role: String?) -> PageAffordance {
        switch role {
        case "AXLink", "AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton",
             "AXMenuButton", "AXMenuItem", "AXMenuBarItem", "AXDisclosureTriangle", "AXTab":
            return .press
        case "AXTextField", "AXTextArea", "AXComboBox", "AXSearchField":
            return .fill
        case "AXSlider":
            return .adjust
        case "AXScrollArea":
            return .scroll
        default:
            return .none
        }
    }
}

/// Where an element's name came from. A caller ranks its own confidence with this.
public enum PageLabelSource: String, Sendable, Equatable, CaseIterable, Codable {
    /// The classifier named the role and a label came with it.
    case classifier
    /// Words found inside the box.
    case textInside
    /// Words beside the box — a form's label for its field.
    case textAdjacent
    /// A drawn glyph the icon bank recognized.
    case icon
    /// "button 3". Nothing named it, so it is named by what it is and where it is.
    case synthesized
}

/// Why an element carries the affordance it does.
public enum PageAffordanceSource: String, Sendable, Equatable, CaseIterable, Codable {
    case classifier
    /// Its place in a group said so — the title line of a result row is pressable.
    case grouping
    /// Its shape said so — a long thin two-tone run is a track.
    case shape
    case unknown
}

public struct PageMapElement: Sendable, Equatable, Identifiable {
    public var id: AXNodeID
    /// Image pixel space, top-left origin.
    public var frame: CGRect
    /// Nil when nothing named the role — the element is still real and still pressable.
    public var role: String?
    public var subrole: String?
    public var category: AXNodeCategory
    public var affordance: PageAffordance
    public var affordanceSource: PageAffordanceSource
    public var label: String
    public var labelSource: PageLabelSource
    /// What else the page said about this row: a duration badge, a "sponsored" marker,
    /// grey placeholder text. Never folded into the label, so a caller can rank on them.
    public var hints: [String]
    public var groupID: Int?
    public var confidence: Double
    public var isEnabled: Bool

    public init(
        id: AXNodeID,
        frame: CGRect,
        role: String? = nil,
        subrole: String? = nil,
        category: AXNodeCategory = .other,
        affordance: PageAffordance = .none,
        affordanceSource: PageAffordanceSource = .unknown,
        label: String,
        labelSource: PageLabelSource = .synthesized,
        hints: [String] = [],
        groupID: Int? = nil,
        confidence: Double = 0,
        isEnabled: Bool = true
    ) {
        self.id = id
        self.frame = frame
        self.role = role
        self.subrole = subrole
        self.category = category
        self.affordance = affordance
        self.affordanceSource = affordanceSource
        self.label = label
        self.labelSource = labelSource
        self.hints = hints
        self.groupID = groupID
        self.confidence = confidence
        self.isEnabled = isEnabled
    }

    /// Can a person do anything with this, or is it only there to be read?
    public var isActionable: Bool { affordance != .none }
}

/// What a run of elements amounts to. Geometry, never a site's markup.
public enum PageGroupKind: String, Sendable, Equatable, CaseIterable, Codable {
    /// One horizontal band that reads as a unit — a search result, a row of a table.
    case row
    /// A picture with its words below or beside it.
    case card
    /// Rows of a shape, repeated. The thing "the third one" counts.
    case list
    /// Fields with their labels, and something to press.
    case form
    /// A run of small controls side by side.
    case toolbar
    /// A box over the page, with the page dimmed behind it.
    case overlay
    /// A horizontal strip that is not any of the above — a header, a footer.
    case band
}

public struct PageMapGroup: Sendable, Equatable, Identifiable {
    public var id: Int
    public var kind: PageGroupKind
    public var frame: CGRect
    public var memberIDs: [AXNodeID]
    /// The heading above it, when there is one. Becomes an element's container trail.
    public var title: String?

    public init(
        id: Int, kind: PageGroupKind, frame: CGRect,
        memberIDs: [AXNodeID], title: String? = nil
    ) {
        self.id = id
        self.kind = kind
        self.frame = frame
        self.memberIDs = memberIDs
        self.title = title
    }
}

public struct PageMap: Sendable, Equatable {
    /// In reading order. THE ordinal.
    public var elements: [PageMapElement]
    public var groups: [PageMapGroup]
    /// Whether a role classifier ran. False means every role is a shape's best guess.
    public var classified: Bool

    public init(
        elements: [PageMapElement] = [], groups: [PageMapGroup] = [], classified: Bool = false
    ) {
        self.elements = elements
        self.groups = groups
        self.classified = classified
    }

    public var actionable: [PageMapElement] { elements.filter(\.isActionable) }

    public func group(_ id: Int?) -> PageMapGroup? {
        guard let id else { return nil }
        return groups.first { $0.id == id }
    }

    /// How many rows carry a name that came from somewhere real. The number that says
    /// whether the map is working on this page.
    public var labeledFraction: Double {
        guard !elements.isEmpty else { return 0 }
        let named = elements.filter { $0.labelSource != .synthesized }.count
        return Double(named) / Double(elements.count)
    }
}
