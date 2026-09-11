//
//  PageMapTests.swift
//  VisionAXTests
//
//  WHAT: The map, over geometry whose answer is known before the test runs.
//  PIN:  MADE-UP RECTANGLES, ON PURPOSE. Every rule here — a line, a band, a repeat, a
//        label — is geometry, so it is provable without an image, and a failure names
//        the rule rather than blaming a screenshot.
//

import CoreGraphics
import Foundation
import Testing
@testable import VisionAX

@Suite struct TextLinesTests {

    private func run(_ string: String, _ frame: CGRect) -> TextRun {
        TextRun(string: string, frame: frame, confidence: 0.9)
    }

    /// WORDS ON ONE BASELINE ARE ONE LINE, and words a column away are not.
    @Test func runsOnABaselineJoinAndColumnsDoNot() {
        let lines = TextLines.lines(from: [
            run("Alpine", CGRect(x: 10, y: 100, width: 60, height: 16)),
            run("touring", CGRect(x: 74, y: 101, width: 60, height: 16)),
            run("boots", CGRect(x: 138, y: 100, width: 50, height: 16)),
            // Far to the trailing side: a second column, not a fourth word.
            run("Sponsored", CGRect(x: 600, y: 100, width: 80, height: 16)),
        ])
        #expect(lines.count == 2)
        #expect(lines[0].string == "Alpine touring boots")
        #expect(lines[1].string == "Sponsored")
    }

    /// A stack of aligned lines is a paragraph.
    @Test func alignedLinesStackIntoOneBlock() {
        let lines = TextLines.lines(from: [
            run("The first line of a snippet", CGRect(x: 10, y: 100, width: 200, height: 14)),
            run("and the second line of it", CGRect(x: 10, y: 118, width: 190, height: 14)),
        ])
        let blocks = TextLines.blocks(from: lines)
        #expect(blocks.count == 1)
        #expect(blocks[0].lines.count == 2)
    }

    /// A LINE INSIDE A BUTTON IS THE BUTTON'S NAME, not a second thing to press.
    @Test func aLineInsideAnExistingBoxIsNotProposedAgain() {
        let lines = TextLines.lines(from: [
            run("Accept", CGRect(x: 104, y: 204, width: 52, height: 14)),
        ])
        let button = CGRect(x: 100, y: 200, width: 60, height: 22)
        #expect(TextLines.proposals(fromLines: lines, notCovering: [button]).isEmpty)
        // And a whole-page container does not account for it either.
        let page = CGRect(x: 0, y: 0, width: 1_200, height: 900)
        #expect(TextLines.proposals(fromLines: lines, notCovering: [page]).count == 1)
    }
}

@Suite struct PageGroupingTests {

    private func candidate(
        _ id: UInt, _ frame: CGRect, _ affordance: PageAffordance = .none,
        category: AXNodeCategory = .text, text: Int = 0
    ) -> PageGrouping.Candidate {
        PageGrouping.Candidate(
            id: AXNodeID(raw: id), frame: frame, affordance: affordance,
            category: category, isText: category == .text, textLength: text)
    }

    /// A RUN OF SIMILAR BANDS IS A LIST, which is what makes "the first one" mean
    /// anything at all.
    @Test func repeatedBandsBecomeAList() {
        let rows = (0 ..< 5).map { index in
            candidate(
                UInt(index + 1),
                CGRect(x: 40, y: 100 + index * 90, width: 600, height: 70), text: 30)
        }
        let groups = PageGrouping.groups(
            for: rows, imageBounds: CGRect(x: 0, y: 0, width: 1_200, height: 900), lines: [])
        #expect(groups.contains { $0.kind == .list && $0.memberIDs.count == 5 })
        #expect(groups.filter { $0.kind == .row }.count == 5)
    }

    /// Two of anything is a coincidence.
    @Test func twoSimilarBandsAreNotAList() {
        let rows = (0 ..< 2).map { index in
            candidate(UInt(index + 1), CGRect(x: 40, y: 100 + index * 90, width: 600, height: 70))
        }
        let groups = PageGrouping.groups(
            for: rows, imageBounds: CGRect(x: 0, y: 0, width: 1_200, height: 900), lines: [])
        #expect(groups.allSatisfy { $0.kind != .list })
    }

    /// SMALL SQUARES SIDE BY SIDE ARE A TOOLBAR.
    @Test func aRunOfSmallSquaresIsAToolbar() {
        let icons = (0 ..< 5).map { index in
            candidate(
                UInt(index + 1), CGRect(x: 40 + index * 44, y: 20, width: 32, height: 32),
                .press, category: .interactive)
        }
        let groups = PageGrouping.groups(
            for: icons, imageBounds: CGRect(x: 0, y: 0, width: 1_200, height: 900), lines: [])
        #expect(groups.contains { $0.kind == .toolbar })
    }

    /// A DIALOG IS INSET FROM EVERY EDGE AND HOLDS THINGS TO PRESS.
    @Test func anInsetBoxHoldingButtonsIsAnOverlay() {
        let bounds = CGRect(x: 0, y: 0, width: 1_200, height: 900)
        let dialog = candidate(
            1, CGRect(x: 300, y: 250, width: 600, height: 300), .none, category: .container)
        let accept = candidate(
            2, CGRect(x: 700, y: 480, width: 120, height: 40), .press, category: .interactive)
        let reject = candidate(
            3, CGRect(x: 560, y: 480, width: 120, height: 40), .press, category: .interactive)
        let groups = PageGrouping.groups(
            for: [dialog, accept, reject], imageBounds: bounds, lines: [])
        #expect(groups.contains { $0.kind == .overlay })
    }
}

@Suite struct LabelLadderTests {

    private func line(_ string: String, _ frame: CGRect) -> TextLine {
        TextLine(
            string: string, frame: frame,
            runs: [TextRun(string: string, frame: frame, confidence: 0.9)], confidence: 0.9)
    }

    /// WORDS INSIDE A BOX ARE THE BOX'S NAME.
    @Test func wordsInsideNameTheBox() {
        let named = LabelLadder.name(
            frame: CGRect(x: 100, y: 200, width: 90, height: 30),
            role: "AXButton", affordance: .press, classifierLabel: nil,
            lines: [line("Accept all", CGRect(x: 106, y: 206, width: 70, height: 16))],
            iconName: nil, ordinal: 1)
        #expect(named.label == "Accept all")
        #expect(named.source == .textInside)
    }

    /// A FIELD IS NAMED FROM OUTSIDE, and a button never is.
    @Test func aFieldTakesTheLabelBesideItAndAButtonDoesNot() {
        let label = line("Email", CGRect(x: 20, y: 204, width: 44, height: 14))
        let box = CGRect(x: 80, y: 198, width: 200, height: 28)
        let field = LabelLadder.name(
            frame: box, role: "AXTextField", affordance: .fill, classifierLabel: nil,
            lines: [label], iconName: nil, ordinal: 2)
        #expect(field.label == "Email")
        #expect(field.source == .textAdjacent)

        let button = LabelLadder.name(
            frame: box, role: "AXButton", affordance: .press, classifierLabel: nil,
            lines: [label], iconName: nil, ordinal: 2)
        #expect(button.source == .synthesized)
        #expect(button.label == "button 2")
    }

    /// THE BOTTOM RUNG STILL NAMES IT. A control nobody can refer to is unusable, and
    /// dropping it is what made a page of icons read as an empty page.
    @Test func anUnnamedControlIsStillNamed() {
        let named = LabelLadder.name(
            frame: CGRect(x: 10, y: 10, width: 24, height: 24), role: nil,
            affordance: .press, classifierLabel: nil, lines: [], iconName: nil, ordinal: 4)
        #expect(named.label == "button 4")
        #expect(named.source == .synthesized)
    }

    @Test func aDurationBadgeIsFound() {
        #expect(LabelLadder.duration(in: [line("12:34", CGRect(x: 0, y: 0, width: 40, height: 12))]) == "12:34")
        #expect(LabelLadder.promotion(in: [line("Sponsored", CGRect(x: 0, y: 0, width: 60, height: 12))]) == "sponsored")
    }
}

@Suite struct PageMapBuilderTests {

    private func node(_ id: UInt, _ role: String, _ frame: CGRect) -> AXNodeSnapshot {
        AXNodeSnapshot(
            id: AXNodeID(raw: id), role: role, frame: frame,
            category: AXNodeCategory.category(role: role))
    }

    private func run(_ string: String, _ frame: CGRect) -> TextRun {
        TextRun(string: string, frame: frame, confidence: 0.9)
    }

    /// THE THING THIS WHOLE PASS EXISTS FOR: a page of result titles with no borders
    /// becomes a page of pressable rows in the order they are read.
    @Test func resultTitlesBecomePressableRowsInOrder() {
        let titles = ["Alpine touring boots reviewed", "The best touring boots this year",
                      "How to choose touring boots", "Touring boots buying guide"]
        var text: [TextRun] = []
        for (index, title) in titles.enumerated() {
            let top = CGFloat(140 + index * 96)
            text.append(run(title, CGRect(x: 60, y: top, width: 420, height: 20)))
            text.append(run(
                "A sentence of snippet under the title of the result.",
                CGRect(x: 60, y: top + 26, width: 520, height: 16)))
        }
        let map = PageMapBuilder.build(
            nodes: [node(1, VisionAX.windowRole, CGRect(x: 0, y: 0, width: 1_200, height: 900))],
            text: text,
            imageBounds: CGRect(x: 0, y: 0, width: 1_200, height: 900),
            classified: false)

        let pressable = map.elements.filter { $0.affordance == .press }
        #expect(pressable.count == titles.count)
        #expect(pressable.first?.label == titles[0])
        #expect(pressable.map(\.label) == titles)
        #expect(pressable.allSatisfy { $0.affordanceSource == .grouping })
        // And the snippet under each title is NOT a second thing to press.
        #expect(map.elements.contains { $0.affordance == .none && $0.label.contains("snippet") })
    }

    /// A DURATION BADGE JOINS THE NAME, which is how a row reads as timed media rather
    /// than as one more link.
    @Test func aDurationBadgeJoinsTheRowsName() {
        var text: [TextRun] = []
        for index in 0 ..< 4 {
            let top = CGFloat(140 + index * 96)
            text.append(run(
                "Something worth watching number \(index)",
                CGRect(x: 60, y: top, width: 420, height: 20)))
            text.append(run("1\(index):0\(index)", CGRect(x: 500, y: top, width: 44, height: 16)))
        }
        let map = PageMapBuilder.build(
            nodes: [node(1, VisionAX.windowRole, CGRect(x: 0, y: 0, width: 1_200, height: 900))],
            text: text,
            imageBounds: CGRect(x: 0, y: 0, width: 1_200, height: 900),
            classified: false)
        let first = map.elements.first { $0.affordance == .press }
        #expect(first?.label.contains("10:00") == true)
        #expect(first?.hints.contains("10:00") == true)
    }

    /// A CLASSIFIED CONTROL KEEPS ITS OWN ROLE and takes its words from inside itself.
    @Test func aClassifiedButtonIsNotSecondGuessed() {
        let map = PageMapBuilder.build(
            nodes: [
                node(1, VisionAX.windowRole, CGRect(x: 0, y: 0, width: 800, height: 600)),
                node(2, "AXButton", CGRect(x: 100, y: 200, width: 120, height: 36)),
                node(3, "AXTextField", CGRect(x: 100, y: 100, width: 300, height: 32)),
            ],
            text: [run("Accept all", CGRect(x: 112, y: 208, width: 80, height: 16))],
            imageBounds: CGRect(x: 0, y: 0, width: 800, height: 600),
            classified: true)
        let button = map.elements.first { $0.role == "AXButton" }
        #expect(button?.label == "Accept all")
        #expect(button?.affordance == .press)
        #expect(map.elements.first { $0.role == "AXTextField" }?.affordance == .fill)
    }

    /// NOTHING IS DROPPED FOR WANT OF A NAME.
    @Test func anUnnamedIconRowSurvives() {
        var nodes = [node(1, VisionAX.windowRole, CGRect(x: 0, y: 0, width: 800, height: 600))]
        for index in 0 ..< 5 {
            nodes.append(node(
                UInt(index + 2), VisionAX.regionRole,
                CGRect(x: 40 + index * 44, y: 20, width: 30, height: 30)))
        }
        var map = PageMapBuilder.build(
            nodes: nodes, text: [],
            imageBounds: CGRect(x: 0, y: 0, width: 800, height: 600), classified: true)
        #expect(map.elements.count == 5)
        #expect(map.elements.allSatisfy { !$0.label.isEmpty })
        #expect(map.actionable.isEmpty, "an unnamed box was offered as pressable")
        #expect(map.labeledFraction == 0)

        // Name them, and they become things to act on.
        let names = Dictionary(
            uniqueKeysWithValues: nodes.dropFirst().map { ($0.id, "search") })
        map = PageMapBuilder.build(
            nodes: nodes, text: [],
            imageBounds: CGRect(x: 0, y: 0, width: 800, height: 600),
            classified: true, icons: names)
        #expect(map.actionable.count == 5)
        #expect(map.actionable.allSatisfy { $0.label == "search" })
    }
}

@Suite struct ProposalUnionTests {

    /// A TEXT BOX THE DETECTOR MISSED JOINS THE TREE, under the node that contains it.
    @Test func aMissingTextBoxIsAddedUnderItsContainer() {
        let container = AXNodeSnapshot(
            id: AXNodeID(raw: 2), role: VisionAX.regionRole,
            frame: CGRect(x: 0, y: 100, width: 800, height: 200), category: .other)
        let root = AXNodeSnapshot(
            id: AXNodeID(raw: 1), role: VisionAX.windowRole,
            frame: CGRect(x: 0, y: 0, width: 800, height: 600), category: .window,
            children: [container])
        let detection = VisionDetection(
            window: AXWindowSnapshot(
                id: AXNodeID(raw: 1), title: "", frame: root.frame ?? .zero,
                isMain: true, isTruncated: false, root: root),
            options: .standard, nodeCount: 2, contourCount: 2, duration: .zero)

        let grown = ProposalUnion.union(detection, textRuns: [
            TextRun(
                string: "A title with no border at all",
                frame: CGRect(x: 40, y: 140, width: 300, height: 20), confidence: 0.9),
        ])
        var frames: [CGRect] = []
        grown.window.root?.forEachNode { if let frame = $0.frame { frames.append(frame) } }
        #expect(grown.nodeCount == 3)
        // Under the container, not the root.
        let added = grown.window.root?.children.first?.children.first
        #expect(added != nil)
        #expect(added?.frame?.contains(CGPoint(x: 100, y: 150)) == true)
        _ = frames
    }
}

@Suite struct IconGlyphTests {

    /// A SHAPE DRAWN SOMEWHERE ELSE IS STILL THAT SHAPE. Different size, different
    /// stroke weight, different proportions — which is the whole claim the bank makes.
    @Test(arguments: SyntheticIcon.drawable) func aDrawnIconIsRecognized(_ glyph: IconGlyph) throws {
        let engine = try VisionEngine()
        let image = try #require(SyntheticIcon.image(glyph))
        let box = CGRect(x: 0, y: 0, width: SyntheticIcon.side, height: SyntheticIcon.side)
        let matched = try engine.matchIcons(in: image, boxes: [box])
        #expect(matched.count == 1)
        #expect(
            matched[0].glyph == glyph,
            "\(glyph.rawValue) read as \(matched[0].glyph?.rawValue ?? "nothing") at \(matched[0].confidence)")
    }

    /// LIGHT ON DARK READS THE SAME AS DARK ON LIGHT. A player's chrome and a toolbar's
    /// are opposite polarities of the same shapes.
    @Test func polarityDoesNotChangeTheAnswer() throws {
        let engine = try VisionEngine()
        for glyph in [IconGlyph.search, .close, .menu, .add] {
            let image = try #require(SyntheticIcon.image(glyph, dark: true))
            let box = CGRect(x: 0, y: 0, width: SyntheticIcon.side, height: SyntheticIcon.side)
            let matched = try engine.matchIcons(in: image, boxes: [box])
            #expect(matched[0].glyph == glyph, "\(glyph.rawValue) inverted")
        }
    }

    /// A BOX WITH NOTHING IN IT IS NOT AN ICON, and neither is one too small to hold a
    /// shape — the normalization would scale a handful of pixels up and call whatever
    /// fell out a match.
    @Test func blankAndTinyBoxesAreNotNamed() throws {
        let engine = try VisionEngine()
        let blank = SyntheticIcon.blank()
        let matched = try engine.matchIcons(in: blank, boxes: [
            CGRect(x: 0, y: 0, width: SyntheticIcon.side, height: SyntheticIcon.side),
            CGRect(x: 0, y: 0, width: 6, height: 6),
        ])
        #expect(matched[0].name == nil)
        #expect(matched[1].name == nil)
    }

    /// THE FLOOR SEPARATES A READING FROM THE CLOSEST OF TWENTY-TWO SHAPES. Printed
    /// rather than merely asserted, because the number in `IconMatch.floor` has to be
    /// justified by the gap between a right answer and a wrong one — and that gap is
    /// what this prints.
    @Test func theFloorSitsBetweenRightAndWrongAnswers() throws {
        let engine = try VisionEngine()
        let box = CGRect(x: 0, y: 0, width: SyntheticIcon.side, height: SyntheticIcon.side)
        var correct: [Double] = []
        for glyph in SyntheticIcon.drawable {
            let image = try #require(SyntheticIcon.image(glyph))
            let matched = try engine.matchIcons(in: image, boxes: [box])
            guard matched[0].glyph == glyph else { continue }
            correct.append(matched[0].confidence)
        }
        let worstCorrect = correct.min() ?? 0
        for (glyph, score) in zip(SyntheticIcon.drawable, correct) {
            print("  \(glyph.rawValue): \(String(format: "%.3f", score))")
        }
        // Noise scores what an unrelated shape scores.
        let noise = try engine.matchIcons(in: SyntheticIcon.blank(), boxes: [box])[0].confidence
        print("icon scores — worst correct \(String(format: "%.3f", worstCorrect)), "
            + "blank \(String(format: "%.3f", noise)), floor \(IconMatch.floor)")
        #expect(correct.count == SyntheticIcon.drawable.count, "an icon was misread")
        #expect(worstCorrect >= IconMatch.floor, "a right answer falls below the floor")
        #expect(noise < IconMatch.floor)
    }

    /// THE NAME IS A WORD A PERSON WOULD SAY.
    @Test func namesAreSpokenEnglish() {
        #expect(IconGlyph.more.spokenName == "more")
        #expect(IconGlyph.heart.spokenName == "like")
        #expect(IconMatch(glyph: .search, confidence: 0.9).name == "search")
        // Below the floor it is the closest of twenty-two shapes, which is not a name.
        #expect(IconMatch(glyph: .search, confidence: 0.2).name == nil)
    }
}

@Suite struct VisionTimingTests {

    /// THE BREAKDOWN ADDS UP. A table whose parts do not sum to its total is a table
    /// nobody can act on — the missing time is always the part worth improving.
    @Test func thePhasesSumToTheTotal() throws {
        let engine = try VisionEngine()
        let scene = try engine.perceive(
            image: SyntheticScreen.image(),
            projection: ScreenProjection(origin: .zero, pixelsPerPoint: 1),
            lanes: [.regions, .text])

        let named = scene.timing.phases.reduce(Duration.zero) { $0 + $1.duration }
        #expect(named <= scene.timing.total, "a phase outlasted the call that ran it")
        #expect(!scene.timing.phases.isEmpty)

        let accounted = scene.timing.accounted.reduce(Duration.zero) { $0 + $1.duration }
        #expect(accounted == scene.timing.total || scene.timing.unaccounted == .zero)
        // And the total is the same span the scene reports.
        #expect(abs(scene.timing.total.milliseconds - scene.duration.milliseconds) < 1)
    }

    /// EVERY LANE THAT RAN IS NAMED, and no lane that did not run is.
    @Test func onlyTheLanesThatRanAreNamed() throws {
        let engine = try VisionEngine()
        let names = { (scene: VisionScene) in Set(scene.timing.phases.map(\.name)) }

        let regions = try engine.perceive(
            image: SyntheticScreen.image(),
            projection: ScreenProjection(origin: .zero, pixelsPerPoint: 1),
            lanes: [.regions])
        #expect(names(regions).contains(VisionTiming.Name.detect))
        #expect(!names(regions).contains(VisionTiming.Name.text))
        #expect(!names(regions).contains(VisionTiming.Name.media))

        let media = try engine.perceive(
            image: SyntheticPlayer.image(fraction: 0.4),
            projection: ScreenProjection(origin: .zero, pixelsPerPoint: 1),
            lanes: [.media])
        #expect(names(media).contains(VisionTiming.Name.media))
        #expect(!names(media).contains(VisionTiming.Name.detect))
    }

    /// THE TABLE IS READABLE, which is the only reason it exists.
    @Test func theTableLinesUpAndEndsInATotal() {
        let timing = VisionTiming(
            phases: [
                .init(name: VisionTiming.Name.text, duration: .milliseconds(150)),
                .init(name: VisionTiming.Name.classify, duration: .milliseconds(90)),
            ],
            total: .milliseconds(250))
        let lines = timing.table().split(separator: "\n").map(String.init)
        #expect(lines.count == 4, "two phases, the remainder, and a total")
        #expect(lines.last?.contains("TOTAL") == true)
        #expect(lines.contains { $0.contains("other") }, "the missing 10ms is shown")
        // Every row is the same width, or the columns do not line up.
        #expect(Set(lines.map(\.count)).count == 1)
    }
}
