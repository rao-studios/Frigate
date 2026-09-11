//
//  LabelLadder.swift
//  FrigateVisionAX
//
//  WHAT: What to call a box, and how sure that name is.
//  IN:   a frame, its role, the page's text lines, an icon name when one was matched
//  OUT:  a label plus where it came from
//  PIN:  EVERY RUNG ENDS IN A NAME. The bottom rung is "button 3" — a position said out
//        loud — because a control nobody can refer to is a control nobody can use, and
//        dropping it (which is what the old roster did) is how a page of icons read as
//        an empty page.
//        ADJACENT TEXT IS FOR FIELDS ONLY. A form labels its box from the outside; a
//        button labels itself from the inside. Reaching outside a button picks up the
//        sentence next to it and calls the button by it.
//

import CoreGraphics
import Foundation
import VisionAXCore

public enum LabelLadder {

    /// A run has to be this far inside a box to be that box's own words.
    static let insideShare: CGFloat = 0.7
    /// A field's label sits within this many box-heights to its leading side …
    static let adjacentLeadingReach: CGFloat = 1.5
    /// … or this many above it.
    static let adjacentAboveReach: CGFloat = 1.0
    /// A label longer than this is a paragraph that happens to be near a box.
    static let adjacentLengthLimit = 48

    public struct Named: Sendable, Equatable {
        public var label: String
        public var source: PageLabelSource

        public init(label: String, source: PageLabelSource) {
            self.label = label
            self.source = source
        }
    }

    /// The ladder, in order. `ordinal` is 1-based within the role, for the last rung.
    public static func name(
        frame: CGRect,
        role: String?,
        affordance: PageAffordance,
        classifierLabel: String?,
        lines: [TextLine],
        iconName: String?,
        ordinal: Int
    ) -> Named {
        if let classifierLabel, !classifierLabel.isEmpty {
            return Named(label: classifierLabel, source: .classifier)
        }
        if let inside = textInside(frame, lines: lines), !inside.isEmpty {
            return Named(label: inside, source: .textInside)
        }
        if affordance == .fill || affordance == .adjust,
           let adjacent = textAdjacent(frame, lines: lines) {
            return Named(label: adjacent, source: .textAdjacent)
        }
        if let iconName, !iconName.isEmpty {
            return Named(label: iconName, source: .icon)
        }
        return Named(label: "\(word(for: role, affordance: affordance)) \(ordinal)", source: .synthesized)
    }

    /// The words inside a box, joined in reading order.
    public static func textInside(_ frame: CGRect, lines: [TextLine]) -> String? {
        let inside = lines.filter { line in
            let overlap = line.frame.intersection(frame)
            guard !overlap.isNull else { return false }
            let area = line.frame.width * line.frame.height
            guard area > 0 else { return false }
            return (overlap.width * overlap.height) / area >= insideShare
        }
        guard !inside.isEmpty else { return nil }
        let joined = inside
            .sorted {
                $0.frame.minY == $1.frame.minY
                    ? $0.frame.minX < $1.frame.minX
                    : $0.frame.minY < $1.frame.minY
            }
            .map(\.string)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return joined.isEmpty ? nil : joined
    }

    /// The line that labels a field from outside it: to its leading side on the same
    /// band, or directly above and left-aligned with it.
    public static func textAdjacent(_ frame: CGRect, lines: [TextLine]) -> String? {
        let height = max(frame.height, 1)
        let leading = lines.filter { line in
            line.frame.maxX <= frame.minX
                && frame.minX - line.frame.maxX <= height * adjacentLeadingReach
                && abs(line.frame.midY - frame.midY) <= height * 0.6
                && line.string.count <= adjacentLengthLimit
        }
        .max { $0.frame.maxX < $1.frame.maxX }
        if let leading { return leading.string }

        return lines.filter { line in
            line.frame.maxY <= frame.minY
                && frame.minY - line.frame.maxY <= height * adjacentAboveReach
                && abs(line.frame.minX - frame.minX) <= height * 0.5
                && line.string.count <= adjacentLengthLimit
        }
        .max { $0.frame.maxY < $1.frame.maxY }?
        .string
    }

    /// What to call a role out loud. The bottom rung's noun.
    public static func word(for role: String?, affordance: PageAffordance) -> String {
        switch role {
        case "AXLink": return "link"
        case "AXButton": return "button"
        case "AXTextField", "AXSearchField": return "field"
        case "AXTextArea": return "text area"
        case "AXComboBox": return "combo box"
        case "AXCheckBox": return "checkbox"
        case "AXRadioButton": return "radio button"
        case "AXPopUpButton", "AXMenuButton": return "menu"
        case "AXMenuItem", "AXMenuBarItem": return "menu item"
        case "AXSlider": return "slider"
        case "AXTab": return "tab"
        case "AXImage": return "image"
        case "AXHeading": return "heading"
        case "AXRow": return "row"
        case "AXCell": return "cell"
        case "AXDisclosureTriangle": return "disclosure"
        default: break
        }
        switch affordance {
        case .press: return "button"
        case .fill: return "field"
        case .adjust: return "slider"
        case .scroll: return "area"
        case .none: return "item"
        }
    }

    // MARK: - Hints

    /// A duration badge — "12:34" — anywhere in these lines.
    public static func duration(in lines: [TextLine]) -> String? {
        for line in lines {
            if let match = line.string.range(
                of: #"\b\d{1,2}:\d{2}(:\d{2})?\b"#, options: .regularExpression) {
                return String(line.string[match])
            }
        }
        return nil
    }

    /// Words a page uses to mark paid placement. A HINT, never a filter: what it is for
    /// is ranking an organic result above a promoted one, not hiding anything.
    static let promotionWords: Set<String> = ["sponsored", "ad", "ads", "promoted", "advertisement"]

    public static func promotion(in lines: [TextLine]) -> String? {
        for line in lines {
            let folded = line.string.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
            guard folded.count <= 16 else { continue }
            if promotionWords.contains(folded) { return folded }
        }
        return nil
    }
}
