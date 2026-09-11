//
//  PageMapBuilder.swift
//  VisionAX
//
//  WHAT: A scene, assembled into the page map a consumer acts on.
//  IN:   VisionScene (regions + text)
//  OUT:  PageMap
//  PIN:  THIS DOES NOT GO THROUGH `roster`. The roster drops any node the classifier
//        did not name and any node without a label, which on a real results page is
//        almost all of them — measured: 33% proposal recall, no rows at all, and a page
//        of results that read as four chrome buttons. The map keeps what the roster
//        threw away and says how sure it is instead.
//        A TEXT LINE IS A THING TO PRESS WHEN IT LEADS A REPEATED BAND. A result title
//        is an anchor around a heading with no border to find an edge on; the band it
//        sits in, repeated down the page, is the evidence that it is a result. Marked
//        `.grouping`, so nothing pretends the classifier said it.
//        NO SITE, NO ENGINE, NO PRODUCT — geometry and words only.
//

import CoreGraphics
import Foundation
import VisionAXCore

public extension VisionScene {

    /// The page as things one can act on.
    ///
    /// `icons` names icon-only controls when a bank is available; without one they are
    /// still kept, and still pressable by position.
    func pageMap(icons: [AXNodeID: String]? = nil) -> PageMap {
        PageMapBuilder.build(
            nodes: detection?.window.root.map { root -> [AXNodeSnapshot] in
                var found: [AXNodeSnapshot] = []
                root.forEachNode { found.append($0) }
                return found
            } ?? [],
            text: text ?? [],
            imageBounds: imageBounds,
            classified: detection?.labels != nil,
            icons: icons ?? self.icons,
            affordances: affordances)
    }
}

public enum PageMapBuilder {

    /// A box smaller than this on either side is a stray mark, not a control.
    static let minimumSide: CGFloat = 8
    /// And a box smaller than THIS is not something a person can aim at.
    ///
    /// PIN: MEASURED ON A REAL PAGE. Without a floor, the map offered 813 "actionable"
    /// rows on one watch page — hundreds of them 8×8 and 12×8 fragments of letterforms
    /// that Canny found edges around. A listing of 813 things is not a listing, and a
    /// resolver asked to pick among them is picking among noise.
    static let minimumActionableSide: CGFloat = 16
    /// A promoted title line needs at least this many characters to be a name rather
    /// than a fragment of chrome.
    static let minimumPromotedLabel = 12
    /// How much taller than the lines below it a leading line must be to read as their
    /// title. Measured on a real results page: titles at 21px over snippets at 13–16.
    static let titleHeightRatio: CGFloat = 1.2
    /// And how much narrower. A heading wraps early; a paragraph's first line does not.
    static let titleWidthRatio: CGFloat = 0.85
    /// An unnamed box only joins the map on shape when it sits in a toolbar.
    static let iconMaximumSide: CGFloat = 72

    /// Boxes that could be a wordless control: small, near-square, and with nothing
    /// written inside them.
    ///
    /// PIN: THE FILTER IS THE POINT. Naming is twenty-two overlap tests per box, and
    /// almost every box on a page already has its own words in it. This is the handful
    /// that has none.
    public static func iconCandidates(
        in detection: VisionDetection, lines: [TextLine]
    ) -> [(id: AXNodeID, frame: CGRect)] {
        var found: [(id: AXNodeID, frame: CGRect)] = []
        detection.window.root?.forEachNode { node in
            guard node.role != VisionAX.windowRole, let frame = node.frame,
                  frame.width >= minimumActionableSide,
                  frame.height >= minimumActionableSide,
                  frame.width <= iconMaximumSide, frame.height <= iconMaximumSide
            else { return }
            let aspect = frame.width / frame.height
            guard PageGrouping.toolbarAspectRange.contains(aspect) else { return }
            guard LabelLadder.textInside(frame, lines: lines) == nil else { return }
            found.append((id: node.id, frame: frame))
        }
        // AN ICON KEEPS COMPANY. A control drawn without words sits in a row of other
        // controls drawn without words — that is what a toolbar is — and a lone blob in
        // the middle of a picture does not. Measured on a watch page, where the bank
        // cheerfully named two dozen patches of video "notifications" and "like" before
        // this rule; the shapes were genuinely similar, and the context was not.
        return found.filter { candidate in
            found.filter { other in
                other.id != candidate.id
                    && PageGrouping.shareABand(candidate.frame, other.frame)
            }.count >= PageGrouping.minimumToolbarMembers - 1
        }
    }

    public static func build(
        nodes: [AXNodeSnapshot],
        text: [TextRun],
        imageBounds: CGRect,
        classified: Bool,
        icons: [AXNodeID: String] = [:],
        affordances: [String: [String]]? = nil
    ) -> PageMap {
        let lines = TextLines.lines(from: text)

        // 1 — everything the classifier named, plus the boxes it left unnamed.
        var named: [PageMapElement] = []
        var unnamed: [AXNodeSnapshot] = []
        for node in nodes {
            guard node.role != VisionAX.windowRole, let frame = node.frame,
                  frame.width >= minimumSide, frame.height >= minimumSide
            else { continue }
            if node.role == VisionAX.regionRole {
                unnamed.append(node)
                continue
            }
            // A CLASSIFIED ROW TOO SMALL TO AIM AT IS NOT A CONTROL EITHER, whatever
            // the model called it. The same measurement: a model confident about an
            // 8-pixel box is confident about a letter.
            let affordance = PageAffordance.derived(role: node.role, table: affordances)
            let reachable = frame.width >= minimumActionableSide
                && frame.height >= minimumActionableSide
            named.append(PageMapElement(
                id: node.id,
                frame: frame,
                role: node.role,
                subrole: node.subrole,
                category: node.category,
                affordance: reachable ? affordance : .none,
                affordanceSource: reachable ? .classifier : .unknown,
                label: "",
                isEnabled: node.isEnabled))
        }

        // 2 — text lines nothing already accounts for.
        //
        // PIN: ANY NAMED BOX, NOT JUST AN ACTIONABLE ONE. A classified box takes its
        // label from the words inside it, so a line that also became a row of its own
        // put the same sentence in the map twice — measured on a live results page,
        // where one advert's title appeared three times and a resolver asked for it by
        // name had to call that an ambiguity.
        let claimed = named.map(\.frame)
        var nextID = (nodes.map(\.id.raw).max() ?? 0) + 1
        var fromText: [PageMapElement] = []
        for line in lines {
            let frame = line.frame.insetBy(
                dx: -TextLines.proposalPadding, dy: -TextLines.proposalPadding).integral
            guard frame.width >= minimumSide, frame.height >= minimumSide else { continue }
            guard !claimed.contains(where: { TextLines.covers($0, frame, atLeast: 0.6) })
            else { continue }
            fromText.append(PageMapElement(
                id: AXNodeID(raw: nextID),
                frame: frame,
                role: nil,
                category: .text,
                affordance: .none,
                affordanceSource: .unknown,
                label: line.string,
                labelSource: .textInside))
            nextID += 1
        }

        // 3 — unnamed boxes that are icon-shaped. Kept only where several sit together,
        //     which is what a toolbar looks like from outside.
        let iconShaped = unnamed.filter { node in
            guard let frame = node.frame, frame.height > 0,
                  frame.width >= minimumActionableSide, frame.height >= minimumActionableSide,
                  frame.width <= iconMaximumSide, frame.height <= iconMaximumSide
            else { return false }
            let aspect = frame.width / frame.height
            guard PageGrouping.toolbarAspectRange.contains(aspect) else { return false }
            return LabelLadder.textInside(frame, lines: lines) == nil
        }
        // A GUESS IS NOT AN OFFER. An unnamed box in a row of unnamed boxes is only
        // pressable when the icon bank actually recognized the shape drawn in it;
        // otherwise it stays in the map — visible, diagnosable — and is not offered as
        // something to act on. Without this every letterform Canny outlined became a
        // button nobody could name and nobody meant.
        var fromShape: [PageMapElement] = []
        for node in iconShaped {
            guard let frame = node.frame,
                  frame.width >= minimumActionableSide, frame.height >= minimumActionableSide
            else { continue }
            let neighbours = iconShaped.filter {
                guard $0.id != node.id, let other = $0.frame else { return false }
                return PageGrouping.shareABand(frame, other)
            }
            guard neighbours.count >= PageGrouping.minimumToolbarMembers - 1 else { continue }
            let named = icons[node.id]
            fromShape.append(PageMapElement(
                id: node.id,
                frame: frame,
                role: nil,
                category: named == nil ? .other : .interactive,
                affordance: named == nil ? .none : .press,
                affordanceSource: named == nil ? .unknown : .shape,
                label: named ?? "",
                labelSource: named == nil ? .synthesized : .icon,
                isEnabled: node.isEnabled))
        }

        var elements = named + fromText + fromShape
        guard !elements.isEmpty else { return PageMap(classified: classified) }

        // 4 — group them, then let the groups promote what they explain.
        let candidates = elements.map { element in
            PageGrouping.Candidate(
                id: element.id, frame: element.frame, affordance: element.affordance,
                category: element.category, isText: element.category == .text,
                textLength: element.label.count)
        }
        let groups = PageGrouping.groups(
            for: candidates, imageBounds: imageBounds, lines: lines)
        var groupByElement: [AXNodeID: Int] = [:]
        for group in groups where group.kind != .list {
            for id in group.memberIDs { groupByElement[id] = group.id }
        }
        let repeatedBands = Set(
            groups.filter { $0.kind == .row || $0.kind == .card }.map(\.id))

        elements = promote(
            elements, groupByElement: groupByElement, repeatedBands: repeatedBands,
            groups: groups, lines: lines)

        // 5 — name everything, in reading order, with the ordinal it will be counted by.
        elements = deduplicated(ordered(elements, groups: groups))
        var counts: [String: Int] = [:]
        elements = elements.map { element in
            var element = element
            element.groupID = groupByElement[element.id]
            let word = LabelLadder.word(for: element.role, affordance: element.affordance)
            counts[word, default: 0] += 1
            if element.label.isEmpty || element.labelSource == .synthesized {
                let named = LabelLadder.name(
                    frame: element.frame, role: element.role, affordance: element.affordance,
                    classifierLabel: nil, lines: lines, iconName: icons[element.id],
                    ordinal: counts[word] ?? 1)
                element.label = named.label
                element.labelSource = named.source
            }
            return element
        }
        return PageMap(elements: elements, groups: groups, classified: classified)
    }

    // MARK: - Promotion

    /// What a band explains about the things in it.
    ///
    /// The leading long line of a REPEATED band is that band's name and the thing a
    /// person means by "the first one" — so it becomes pressable, and any duration badge
    /// beside it joins its label, which is what makes it read as a video rather than a
    /// link.
    static func promote(
        _ elements: [PageMapElement],
        groupByElement: [AXNodeID: Int],
        repeatedBands: Set<Int>,
        groups: [PageMapGroup],
        lines: [TextLine]
    ) -> [PageMapElement] {
        var byGroup: [Int: [PageMapElement]] = [:]
        for element in elements {
            guard let group = groupByElement[element.id] else { continue }
            byGroup[group, default: []].append(element)
        }
        var promoted: Set<AXNodeID> = []
        var badges: [AXNodeID: String] = [:]
        var promotions: [AXNodeID: String] = [:]

        // A TITLE IS SET LARGER THAN WHAT IT INTRODUCES, and that is the whole rule.
        //
        // Repetition catches a list of results; it does not catch the one result at the
        // top of a page or a block that stands alone — measured on a live results page,
        // where every organic title read as ordinary text because the bands around them
        // were adverts of another shape. This works over TEXT BLOCKS rather than bands
        // for the same measured reason: a title and its snippet are four pixels apart
        // and a band merge compares gaps with their neighbours, which cannot separate
        // "the next line of this paragraph" from "the next row of this list".
        for block in TextLines.blocks(from: lines) where block.lines.count >= 2 {
            guard let first = block.lines.first,
                  first.string.count >= minimumPromotedLabel
            else { continue }
            // TALLER THAN WHAT FOLLOWS, AND NARROWER THAN IT.
            //
            // PIN: HEIGHT ALONE PROMOTED PROSE. Measured on a live article: the first
            // line of a paragraph is often a pixel or two taller than the rest — a
            // capital letter, an ascender, recognition noise — and the map duly offered
            // six sentences of body text as things to press. A heading is also SET
            // NARROW: it wraps early and its block runs to the measure below it, while a
            // paragraph's first line is exactly as wide as the paragraph. The two
            // together separate the one from the other; either alone does not.
            let heights = block.lines.dropFirst().map(\.frame.height).sorted()
            let widths = block.lines.dropFirst().map(\.frame.width).sorted()
            guard !heights.isEmpty else { continue }
            let median = heights[heights.count / 2]
            let medianWidth = widths[widths.count / 2]
            guard median > 0, medianWidth > 0,
                  first.frame.height >= median * titleHeightRatio,
                  first.frame.width <= medianWidth * titleWidthRatio
            else { continue }
            // The element that IS that line — the text row built from it.
            guard let element = elements.first(where: { candidate in
                candidate.category == .text
                    && candidate.affordance == .none
                    && candidate.frame.insetBy(dx: -1, dy: -1).contains(first.frame)
            }) else { continue }
            promoted.insert(element.id)
        }

        for (groupID, members) in byGroup where repeatedBands.contains(groupID) {
            // Already something to press in this band? Then the band is explained.
            let bandLines = lines.filter { line in
                guard let frame = groups.first(where: { $0.id == groupID })?.frame
                else { return false }
                return frame.intersects(line.frame)
            }
            let badge = LabelLadder.duration(in: bandLines)
            let promotion = LabelLadder.promotion(in: bandLines)
            // THE TOPMOST LINE, NOT THE LONGEST. A result's name is its first line and
            // its snippet is its longest one — picking by length names every row after
            // the sentence describing it, which is both unusable to say out loud and
            // wrong about what the row is.
            let titles = members
                .filter { $0.affordance == .none && $0.category == .text }
                .filter { $0.label.count >= minimumPromotedLabel }
                .sorted {
                    $0.frame.minY == $1.frame.minY
                        ? $0.frame.minX < $1.frame.minX
                        : $0.frame.minY < $1.frame.minY
                }
            guard members.contains(where: { $0.affordance == .press }) == false,
                  let title = titles.first
            else {
                if let badge, let existing = members.first(where: { $0.affordance == .press }) {
                    badges[existing.id] = badge
                }
                if let promotion, let existing = members.first(where: { $0.affordance == .press }) {
                    promotions[existing.id] = promotion
                }
                continue
            }
            promoted.insert(title.id)
            if let badge { badges[title.id] = badge }
            if let promotion { promotions[title.id] = promotion }
        }

        return elements.map { element in
            var element = element
            if promoted.contains(element.id) {
                element.affordance = .press
                element.affordanceSource = .grouping
                // A TEXT ROLE IS NOT KEPT ONCE THE ROW IS PRESSABLE. The classifier
                // often names a result's title AXStaticText — it is text, and it is
                // also the link — and a row that affords a press while claiming to be
                // static text reads as a contradiction to everything downstream.
                if element.role == nil || element.category == .text {
                    element.role = "AXLink"
                }
                element.category = .interactive
            }
            if let badge = badges[element.id] {
                element.hints.append(badge)
                // THE BADGE JOINS THE NAME, not just the hints. "12:34" beside a title is
                // how the web says "this is timed media", and a consumer deriving a kind
                // from the label alone cannot see a hint it was never given.
                if !element.label.contains(badge) {
                    element.label += " · \(badge)"
                }
            }
            if let promotion = promotions[element.id] {
                element.hints.append(promotion)
            }
            return element
        }
    }

    // MARK: - Duplicates

    /// The same thing, found twice, kept once.
    ///
    /// PIN: THE DETECTOR NESTS. A line of text often yields two boxes — the run and the
    /// padded block around it — and both classify the same way and take the same words,
    /// so the map offered one sentence twice. A resolver asked for it by name then had
    /// to call that an ambiguity and refuse, which is the worst possible outcome for a
    /// row that is not ambiguous at all. The larger box is kept: it is the one a click
    /// lands inside from the widest range of aims.
    static func deduplicated(_ elements: [PageMapElement]) -> [PageMapElement] {
        var kept: [PageMapElement] = []
        for element in elements {
            let label = folded(element.label)
            guard !label.isEmpty else {
                kept.append(element)
                continue
            }
            if let index = kept.firstIndex(where: {
                folded($0.label) == label && overlap($0.frame, element.frame) >= 0.5
            }) {
                // Keep whichever can be pressed, then whichever is bigger.
                let existing = kept[index]
                let better = element.isActionable && !existing.isActionable
                    || (element.isActionable == existing.isActionable
                        && element.frame.width * element.frame.height
                            > existing.frame.width * existing.frame.height)
                if better { kept[index] = element }
                continue
            }
            kept.append(element)
        }
        return kept
    }

    static func folded(_ value: String) -> String {
        value.lowercased().filter { $0.isLetter || $0.isNumber }
    }

    static func overlap(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let intersection = a.intersection(b)
        guard !intersection.isNull else { return 0 }
        let shared = intersection.width * intersection.height
        let union = a.width * a.height + b.width * b.height - shared
        guard union > 0 else { return 0 }
        return shared / union
    }

    // MARK: - Order

    /// Reading order: by group, top to bottom, then within a group by band and leading
    /// edge. Elements in no group sort by their own position among the groups.
    static func ordered(_ elements: [PageMapElement], groups: [PageMapGroup]) -> [PageMapElement] {
        elements.sorted { lhs, rhs in
            let leftBand = bandKey(lhs.frame)
            let rightBand = bandKey(rhs.frame)
            if leftBand != rightBand { return leftBand < rightBand }
            return lhs.frame.minX < rhs.frame.minX
        }
    }

    /// Vertical position, quantized so two things on one line do not swap by a pixel.
    static func bandKey(_ frame: CGRect) -> Int {
        Int((frame.minY / 8).rounded(.down))
    }
}
