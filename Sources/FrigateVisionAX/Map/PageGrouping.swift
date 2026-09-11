//
//  PageGrouping.swift
//  FrigateVisionAX
//
//  WHAT: Boxes gathered into the units a person means — rows, cards, lists, forms,
//        toolbars, overlays.
//  IN:   candidate frames + text lines
//  OUT:  PageMapGroup, and the reading order everything else counts
//  PIN:  GEOMETRY, NEVER MARKUP AND NEVER A SITE. A results page is a run of bands of
//        similar height at a regular pitch; so is a mail list and a table of orders.
//        That is the whole rule, and it is why nothing here has to be taught a site.
//        A LIST IS WHAT MAKES "THE FIRST ONE" MEAN ANYTHING. Without a repeated run
//        there is no ordinal to count, so this pass is not decoration — it is the
//        difference between a page of boxes and a page of results.
//        PURE. Rectangles in, groups out.
//

import CoreGraphics
import Foundation
import VisionAXCore

public enum PageGrouping {

    /// Two boxes are in one band when they share this much of the shorter one's height.
    static let bandOverlap: CGFloat = 0.5
    /// Bands repeat when their heights agree within this share and their leading edges
    /// within this many pixels.
    static let repeatHeightTolerance: CGFloat = 0.35
    static let repeatEdgeTolerance: CGFloat = 24
    /// How many similar bands in a row make a list. Two could be a coincidence.
    static let minimumRepeats = 3
    /// A toolbar's members are small, near-square and side by side.
    static let toolbarMaximumSide: CGFloat = 72
    static let toolbarAspectRange: ClosedRange<CGFloat> = 0.55 ... 1.8
    static let minimumToolbarMembers = 3
    /// A dialog sits inset from every edge and covers a good share of the page.
    static let overlayInset: CGFloat = 16
    static let overlayAreaRange: ClosedRange<CGFloat> = 0.08 ... 0.72

    /// One candidate box, as grouping sees it.
    public struct Candidate: Sendable, Equatable {
        public var id: AXNodeID
        public var frame: CGRect
        public var affordance: PageAffordance
        public var category: AXNodeCategory
        public var isText: Bool
        public var textLength: Int

        public init(
            id: AXNodeID, frame: CGRect, affordance: PageAffordance,
            category: AXNodeCategory, isText: Bool, textLength: Int
        ) {
            self.id = id
            self.frame = frame
            self.affordance = affordance
            self.category = category
            self.isText = isText
            self.textLength = textLength
        }
    }

    /// Groups in reading order, ids assigned in that order.
    public static func groups(
        for candidates: [Candidate], imageBounds: CGRect, lines: [TextLine]
    ) -> [PageMapGroup] {
        guard !candidates.isEmpty else { return [] }
        let bands = bands(of: candidates)
        guard !bands.isEmpty else { return [] }

        var groups: [PageMapGroup] = []
        var nextID = 0
        func take() -> Int {
            defer { nextID += 1 }
            return nextID
        }

        // THE OVERLAY FIRST, because everything inside it belongs to it and nothing
        // outside it can be reached while it is up.
        let overlayFrame = overlay(in: candidates, imageBounds: imageBounds)

        // Which bands repeat — the runs that make lists.
        let runs = repeatedRuns(in: bands)
        var listByBand: [Int: Int] = [:]
        for run in runs {
            let members = run.flatMap { bands[$0].map(\.id) }
            let frame = union(of: run.flatMap { bands[$0].map(\.frame) })
            let id = take()
            groups.append(PageMapGroup(
                id: id, kind: .list, frame: frame, memberIDs: members,
                title: heading(above: frame, in: lines)))
            for index in run { listByBand[index] = id }
        }

        for (index, band) in bands.enumerated() {
            let frame = union(of: band.map(\.frame))
            let kind: PageGroupKind
            if let overlayFrame, overlayFrame.contains(frame) {
                kind = .overlay
            } else if listByBand[index] != nil {
                kind = band.contains(where: { $0.category == .image }) ? .card : .row
            } else if isToolbar(band) {
                kind = .toolbar
            } else if band.contains(where: { $0.affordance == .fill }) {
                kind = .form
            } else if band.contains(where: { $0.category == .image }),
                      band.contains(where: \.isText) {
                kind = .card
            } else {
                kind = .band
            }
            groups.append(PageMapGroup(
                id: take(), kind: kind, frame: frame, memberIDs: band.map(\.id),
                title: listByBand[index].flatMap { id in groups.first { $0.id == id }?.title }
                    ?? heading(above: frame, in: lines)))
        }
        return groups
    }

    // MARK: - Bands

    /// Candidates gathered into horizontal bands, top to bottom, each in leading order.
    public static func bands(of candidates: [Candidate]) -> [[Candidate]] {
        let ordered = candidates
            .filter { $0.frame.width > 0 && $0.frame.height > 0 }
            .sorted {
                $0.frame.minY == $1.frame.minY
                    ? $0.frame.minX < $1.frame.minX
                    : $0.frame.minY < $1.frame.minY
            }
        var bands: [[Candidate]] = []
        for candidate in ordered {
            // A CANDIDATE THAT SPANS THE WHOLE BAND IS ITS CONTAINER, not a sibling of
            // the things inside it; it still bands with them, and the group's frame is
            // the union either way.
            let joined = bands.indices.last { index in
                bands[index].contains { shareABand($0.frame, candidate.frame) }
            }
            if let joined {
                bands[joined].append(candidate)
            } else {
                bands.append([candidate])
            }
        }
        // ROWS THAT BELONG TOGETHER ARE ONE ROW. A result is a title with a snippet
        // under it, and a card is a picture with words below: separated by a few pixels,
        // they are two bands geometrically and one thing to a person. Merging on a gap
        // measured against the bands' own height is what makes the repeat test see four
        // results instead of eight half-results.
        return merged(bands).map { $0.sorted { $0.frame.minX < $1.frame.minX } }
    }

    /// A band never merges across a gap wider than this share of its own height.
    static let stackedGapLimit: CGFloat = 0.6

    /// Bands that belong to one another, joined.
    ///
    /// PIN: THE COMPARISON IS LOCAL, NOT A THRESHOLD. What separates a title from its
    /// own snippet is that the gap above it is SMALLER than the gap to whatever comes
    /// next; a fixed number cannot know that, because the same six pixels mean "same
    /// row" on one page and "next row" on another. Comparing each gap with its
    /// neighbour is scale-free, and it is what makes four results read as four rows
    /// rather than eight halves.
    static func merged(_ bands: [[Candidate]]) -> [[Candidate]] {
        guard bands.count > 1 else { return bands }
        let frames = bands.map { union(of: $0.map(\.frame)) }
        // gaps[i] is the space above band i.
        var gaps: [CGFloat] = [.infinity]
        for index in 1 ..< frames.count {
            gaps.append(frames[index].minY - frames[index - 1].maxY)
        }

        var merged: [[Candidate]] = []
        var frame: CGRect = .null
        for index in bands.indices {
            let gap = gaps[index]
            // What this gap is being judged against: the space below, or — for the last
            // band, which has none — the space above the one before it.
            let neighbour = index + 1 < gaps.count
                ? gaps[index + 1]
                : (index >= 1 ? gaps[index - 1] : CGFloat.infinity)
            let heights = min(frame.isNull ? frames[index].height : frame.height,
                              frames[index].height)
            let overlaps = !frame.isNull && frame.maxX > frames[index].minX
                && frames[index].maxX > frame.minX
            let close = gap < neighbour
                && gap <= heights * stackedGapLimit
                && gap >= -heights * stackedGapLimit
            if overlaps, close, !merged.isEmpty {
                merged[merged.count - 1] += bands[index]
                frame = frame.union(frames[index])
            } else {
                merged.append(bands[index])
                frame = frames[index]
            }
        }
        return merged
    }

    static func shareABand(_ a: CGRect, _ b: CGRect) -> Bool {
        let overlap = min(a.maxY, b.maxY) - max(a.minY, b.minY)
        guard overlap > 0 else { return false }
        return overlap >= min(a.height, b.height) * bandOverlap
    }

    // MARK: - Repetition

    /// Runs of consecutive bands that look like each other. Index ranges into `bands`.
    static func repeatedRuns(in bands: [[Candidate]]) -> [[Int]] {
        let frames = bands.map { union(of: $0.map(\.frame)) }
        var runs: [[Int]] = []
        var current: [Int] = []
        for index in frames.indices {
            guard let last = current.last else {
                current = [index]
                continue
            }
            if alike(frames[last], frames[index]) {
                current.append(index)
            } else {
                if current.count >= minimumRepeats { runs.append(current) }
                current = [index]
            }
        }
        if current.count >= minimumRepeats { runs.append(current) }
        return runs
    }

    static func alike(_ a: CGRect, _ b: CGRect) -> Bool {
        guard a.height > 0, b.height > 0 else { return false }
        let ratio = abs(a.height - b.height) / max(a.height, b.height)
        guard ratio <= repeatHeightTolerance else { return false }
        guard abs(a.minX - b.minX) <= repeatEdgeTolerance else { return false }
        let widthRatio = abs(a.width - b.width) / max(a.width, b.width)
        return widthRatio <= repeatHeightTolerance
    }

    // MARK: - Shapes

    static func isToolbar(_ band: [Candidate]) -> Bool {
        let small = band.filter { candidate in
            let frame = candidate.frame
            guard frame.height <= toolbarMaximumSide, frame.width <= toolbarMaximumSide,
                  frame.height > 0
            else { return false }
            return toolbarAspectRange.contains(frame.width / frame.height)
        }
        guard small.count >= minimumToolbarMembers else { return false }
        // A band that is mostly words is a sentence with something small in it.
        return small.count * 2 >= band.count
    }

    /// A dialog's frame, when the page is showing one.
    ///
    /// PIN: SHAPE, NOT PIXELS. Reading the dimmed backdrop would be the stronger test
    /// and needs the image; a box inset from every edge that holds several things to
    /// press is the same thing said in geometry, and it is what makes a consent wall
    /// surface FIRST instead of being buried under the page it is covering.
    static func overlay(in candidates: [Candidate], imageBounds: CGRect) -> CGRect? {
        let area = imageBounds.width * imageBounds.height
        guard area > 0 else { return nil }
        let boxes = candidates.filter { candidate in
            let frame = candidate.frame
            guard frame.minX >= imageBounds.minX + overlayInset,
                  frame.minY >= imageBounds.minY + overlayInset,
                  frame.maxX <= imageBounds.maxX - overlayInset,
                  frame.maxY <= imageBounds.maxY - overlayInset
            else { return false }
            let share = (frame.width * frame.height) / area
            guard overlayAreaRange.contains(share) else { return false }
            let inside = candidates.filter {
                $0.id != candidate.id && frame.contains($0.frame) && $0.affordance == .press
            }
            return inside.count >= 2
        }
        return boxes.max { ($0.frame.width * $0.frame.height) < ($1.frame.width * $1.frame.height) }?
            .frame
    }

    // MARK: - Titles

    /// The nearest heading-shaped line above a group — its name, when it has one.
    static func heading(above frame: CGRect, in lines: [TextLine]) -> String? {
        let candidates = lines.filter { line in
            line.frame.maxY <= frame.minY
                && line.frame.minY >= frame.minY - max(frame.height, 120)
                && line.frame.maxX > frame.minX
                && line.frame.minX < frame.maxX
                && line.string.count <= 60
        }
        return candidates.max { $0.frame.maxY < $1.frame.maxY }?.string
    }

    static func union(of frames: [CGRect]) -> CGRect {
        guard let first = frames.first else { return .zero }
        return frames.dropFirst().reduce(first) { $0.union($1) }
    }
}
