//
//  TextLines.swift
//  FrigateVisionAX
//
//  WHAT: Recognized words gathered into lines and blocks, and offered as proposals.
//  IN:   [TextRun] from the text lane
//  OUT:  the page map; the harvester's ground-truth matching
//  PIN:  THE SAME FUNCTION AT HARVEST AND AT INFERENCE. A text line that becomes a
//        proposal only when the model runs is a box the model was never trained on, and
//        it answers `none` for exactly the rows that matter — a result title is an
//        anchor around a heading, and the detector's edge pass does not see it because
//        there is no edge to see. Both paths call `proposals(from:)`, so what the head
//        learns is what the head is asked.
//        PURE. No engine, no image, no state — geometry over rectangles, testable on
//        made-up numbers.
//

import CoreGraphics
import Foundation
import VisionAXCore

/// Words that share a baseline, joined.
public struct TextLine: Sendable, Equatable {
    public var string: String
    /// Image pixel space, top-left origin. The union of its runs.
    public var frame: CGRect
    public var runs: [TextRun]
    public var confidence: Double

    public init(string: String, frame: CGRect, runs: [TextRun], confidence: Double) {
        self.string = string
        self.frame = frame
        self.runs = runs
        self.confidence = confidence
    }
}

/// Lines stacked into a paragraph — a snippet, an address, a description.
public struct TextBlock: Sendable, Equatable {
    public var lines: [TextLine]
    public var frame: CGRect

    public init(lines: [TextLine], frame: CGRect) {
        self.lines = lines
        self.frame = frame
    }

    public var string: String {
        lines.map(\.string).joined(separator: " ")
    }
}

public enum TextLines {

    /// Two runs are on one line when their centres sit within this share of the
    /// median run height.
    static let baselineTolerance: CGFloat = 0.5
    /// And when the gap between them is no wider than this share of that height. A
    /// wider gap is two columns, not one sentence.
    static let horizontalGapLimit: CGFloat = 1.2
    /// Lines stack into a block when their left edges agree within this share of the
    /// height, and the vertical gap is under `blockGapLimit`.
    static let blockAlignmentTolerance: CGFloat = 0.5
    /// PIN: A FULL LINE HEIGHT, NOT A FRACTION OF ONE. Web line spacing runs around
    /// 1.4× the type size, so consecutive lines of one paragraph sit up to a whole line
    /// height apart — measured on a live results page, where a title and its own snippet
    /// were 12 pixels apart with a limit of 9.6 and were read as unrelated.
    static let blockGapLimit: CGFloat = 1.0
    /// Proposals are padded outwards by this many pixels, so the box a classifier sees
    /// includes the few pixels of margin a link's hit area actually has.
    static let proposalPadding: CGFloat = 4
    /// A proposal narrower or shorter than this is a stray mark.
    static let minimumProposalSide: CGFloat = 8

    /// Runs gathered into lines, in reading order.
    ///
    /// PIN: BASELINES FIRST, THEN LEADING EDGE. Joining runs in whatever order they
    /// arrive splits a sentence around anything that happens to sort between its words —
    /// a badge in the next column lands mid-phrase and takes half the line with it.
    /// Clustering by baseline and only then walking each band left to right cannot do
    /// that, because a word is only ever offered the words actually beside it.
    public static func lines(from runs: [TextRun]) -> [TextLine] {
        let usable = runs.filter { $0.frame.width > 0 && $0.frame.height > 0 }
        guard !usable.isEmpty else { return [] }
        let median = medianHeight(of: usable)

        // 1 — baselines.
        var bands: [[TextRun]] = []
        for run in usable.sorted(by: { $0.frame.midY < $1.frame.midY }) {
            let tolerance = max(median, run.frame.height) * baselineTolerance
            if let index = bands.indices.last(where: { index in
                guard let last = bands[index].last else { return false }
                return abs(last.frame.midY - run.frame.midY) <= tolerance
            }) {
                bands[index].append(run)
            } else {
                bands.append([run])
            }
        }

        // 2 — each band, leading to trailing, split where the gap is a column.
        var lines: [TextLine] = []
        for band in bands {
            var current: [TextRun] = []
            for run in band.sorted(by: { $0.frame.minX < $1.frame.minX }) {
                if let last = current.last {
                    let gap = run.frame.minX - last.frame.maxX
                    let limit = max(median, run.frame.height) * horizontalGapLimit
                    if gap > limit {
                        if let line = line(from: current) { lines.append(line) }
                        current = []
                    }
                }
                current.append(run)
            }
            if let line = line(from: current) { lines.append(line) }
        }

        return lines.sorted {
            $0.frame.minY == $1.frame.minY
                ? $0.frame.minX < $1.frame.minX
                : $0.frame.minY < $1.frame.minY
        }
    }

    static func line(from runs: [TextRun]) -> TextLine? {
        guard let first = runs.first else { return nil }
        let frame = runs.dropFirst().reduce(first.frame) { $0.union($1.frame) }
        let string = runs.map(\.string).joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !string.isEmpty else { return nil }
        return TextLine(
            string: string, frame: frame, runs: runs,
            confidence: runs.map(\.confidence).min() ?? 0)
    }

    /// Lines stacked into paragraphs.
    public static func blocks(from lines: [TextLine]) -> [TextBlock] {
        guard !lines.isEmpty else { return [] }
        var built: [[TextLine]] = []
        for line in lines {
            let height = line.frame.height
            let appended = built.indices.last { index in
                guard let last = built[index].last else { return false }
                guard abs(last.frame.minX - line.frame.minX)
                    <= max(height, last.frame.height) * blockAlignmentTolerance
                else { return false }
                let gap = line.frame.minY - last.frame.maxY
                return gap >= -height && gap <= max(height, last.frame.height) * blockGapLimit
            }
            if let appended {
                built[appended].append(line)
            } else {
                built.append([line])
            }
        }
        return built.compactMap { group in
            guard let first = group.first else { return nil }
            let frame = group.dropFirst().reduce(first.frame) { $0.union($1.frame) }
            return TextBlock(lines: group, frame: frame)
        }
    }

    /// The boxes a text lane contributes to the detector's proposals.
    ///
    /// One per LINE, not per run and not per block: a link is a line, a heading is a
    /// line, and a paragraph is several lines nobody clicks as a unit.
    public static func proposals(from runs: [TextRun]) -> [CGRect] {
        proposals(fromLines: lines(from: runs))
    }

    public static func proposals(fromLines lines: [TextLine]) -> [CGRect] {
        lines.compactMap { line in
            let padded = line.frame.insetBy(dx: -proposalPadding, dy: -proposalPadding).integral
            guard padded.width >= minimumProposalSide, padded.height >= minimumProposalSide
            else { return nil }
            return padded
        }
    }

    /// Proposals that no existing box already covers.
    ///
    /// `overlap` is deliberately high: a text line INSIDE a button is the button's name,
    /// not a second thing to press, and the detector already found the button.
    /// PIN: ONE BOX PER LINE, NOT PER RUN, AND THAT WAS MEASURED. A navigation bar is
    /// several links on one baseline, so offering each run separately looked like the
    /// obvious improvement — it moved real-page proposal recall from 0.727 to 0.728 over
    /// 150 harvested samples, for roughly twice the proposals and twice the inference.
    /// The line is where the boxes are.
    public static func proposals(
        fromLines lines: [TextLine], notCovering existing: [CGRect], overlap: CGFloat = 0.9
    ) -> [CGRect] {
        let boxes = proposals(fromLines: lines)
        var kept: [CGRect] = []
        for candidate in boxes {
            if existing.contains(where: { covers($0, candidate, atLeast: overlap) }) { continue }
            if kept.contains(where: { covers($0, candidate, atLeast: overlap) }) { continue }
            kept.append(candidate)
        }
        return kept
    }

    /// Does `box` account for `candidate` — same thing, near enough.
    static func covers(_ box: CGRect, _ candidate: CGRect, atLeast overlap: CGFloat) -> Bool {
        let intersection = box.intersection(candidate)
        guard !intersection.isNull else { return false }
        let area = candidate.width * candidate.height
        guard area > 0 else { return false }
        let shared = (intersection.width * intersection.height) / area
        guard shared >= overlap else { return false }
        // AND SIMILAR IN SIZE. A line inside a full-page container shares all of its own
        // area with it, and that container is not the line.
        let boxArea = box.width * box.height
        return boxArea <= area / overlap
    }

    static func medianHeight(of runs: [TextRun]) -> CGFloat {
        let heights = runs.map(\.frame.height).sorted()
        guard !heights.isEmpty else { return 0 }
        return heights[heights.count / 2]
    }
}
