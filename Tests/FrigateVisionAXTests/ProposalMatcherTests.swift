//
//  ProposalMatcherTests.swift
//  FrigateVisionAXTests
//
//  WHAT: The label rules — bands, ancestor duplicates, disagreement, recall.
//  PIN:  These cases are the ones that silently poison a dataset. Every number here is
//        an exact IoU worked out by hand, not a tolerance, so a change in the rule
//        fails loudly instead of shifting a metric three steps downstream.
//

import Foundation
import Testing
@testable import FrigateVisionAX
@testable import VisionAXCore

@Suite struct ProposalMatcherTests {

    private func element(
        _ index: Int, _ role: String, _ rect: PixelRect,
        parent: Int = -1, matchable: Bool = true
    ) -> GroundTruthElement {
        GroundTruthElement(
            index: index, role: role, rect: rect,
            parent: parent, matchable: matchable)
    }

    private func proposal(_ index: Int, _ rect: PixelRect) -> ProposalMatcher.Proposal {
        ProposalMatcher.Proposal(index: index, parent: -1, depth: 1, rect: rect)
    }

    // MARK: - Geometry

    @Test func iouOfIdenticalRectsIsOne() {
        let rect = PixelRect(x: 10, y: 10, width: 100, height: 50)
        #expect(rect.intersectionOverUnion(rect) == 1.0)
    }

    @Test func iouOfDisjointRectsIsZero() {
        let a = PixelRect(x: 0, y: 0, width: 10, height: 10)
        let b = PixelRect(x: 100, y: 100, width: 10, height: 10)
        #expect(a.intersectionOverUnion(b) == 0)
    }

    @Test func iouIsExactForAKnownOverlap() {
        // 100×100 and 100×100 offset by 50 in x: overlap 50×100 = 5000,
        // union 10000 + 10000 − 5000 = 15000 → 1/3.
        let a = PixelRect(x: 0, y: 0, width: 100, height: 100)
        let b = PixelRect(x: 50, y: 0, width: 100, height: 100)
        #expect(abs(a.intersectionOverUnion(b) - 1.0 / 3.0) < 1e-12)
    }

    @Test func roundingUsesEdgesNotSize() {
        // x 10.6 → 11, maxX 10.6+9.6=20.2 → 20, so width is 9, not 10.
        let rect = PixelRect(rounding: CGRect(x: 10.6, y: 0, width: 9.6, height: 4))
        #expect(rect.x == 11)
        #expect(rect.width == 9)
    }

    // MARK: - Bands

    @Test func aSquareMatchTakesTheRole() {
        let elements = [element(0, "AXButton", PixelRect(x: 0, y: 0, width: 100, height: 40))]
        let outcome = ProposalMatcher.match(
            proposals: [proposal(0, PixelRect(x: 0, y: 0, width: 100, height: 40))],
            elements: elements)
        #expect(outcome.regions[0].label == .role("AXButton"))
        #expect(outcome.regions[0].matchIoU == 1.0)
        #expect(outcome.regions[0].matchedElement == 0)
    }

    @Test func aDistantProposalIsNone() {
        let elements = [element(0, "AXButton", PixelRect(x: 0, y: 0, width: 100, height: 40))]
        let outcome = ProposalMatcher.match(
            proposals: [proposal(0, PixelRect(x: 500, y: 500, width: 100, height: 40))],
            elements: elements)
        #expect(outcome.regions[0].label == .none)
    }

    @Test func aHalfOverlapIsIgnoredNotNone() {
        // IoU 1/3: above the 0.3 floor, below the 0.5 bar — no defensible answer.
        let elements = [element(0, "AXButton", PixelRect(x: 0, y: 0, width: 100, height: 100))]
        let outcome = ProposalMatcher.match(
            proposals: [proposal(0, PixelRect(x: 50, y: 0, width: 100, height: 100))],
            elements: elements)
        #expect(outcome.regions[0].label == .ignore)
    }

    // MARK: - The two hard rules

    @Test func aDescendantDuplicatingItsAncestorStopsBeingMatchable() {
        // A button and the static text inside it, one pixel apart in each direction.
        let button = element(0, "AXButton", PixelRect(x: 0, y: 0, width: 100, height: 40))
        let text = element(1, "AXStaticText", PixelRect(x: 1, y: 1, width: 98, height: 38), parent: 0)
        let outcome = ProposalMatcher.match(
            proposals: [proposal(0, PixelRect(x: 0, y: 0, width: 100, height: 40))],
            elements: [button, text])

        #expect(outcome.elements[1].matchable == false, "the inner text should defer to the button")
        #expect(outcome.elements[0].matchable == true)
        // And so the proposal takes the role Mary can actually click.
        #expect(outcome.regions[0].label == .role("AXButton"))
    }

    @Test func aControlInsideAWrapperBeatsTheWrapper() {
        // The div only has a box at all because the field is inside it. Handing the
        // label to the group would cost Mary every form field on the page.
        let wrapper = element(0, "AXGroup", PixelRect(x: 0, y: 0, width: 200, height: 44))
        let field = element(1, "AXTextField", PixelRect(x: 1, y: 1, width: 198, height: 42), parent: 0)
        let outcome = ProposalMatcher.match(
            proposals: [proposal(0, PixelRect(x: 0, y: 0, width: 200, height: 44))],
            elements: [wrapper, field])

        #expect(outcome.elements[0].matchable == false, "the wrapper should defer")
        #expect(outcome.elements[1].matchable == true)
        #expect(outcome.regions[0].label == .role("AXTextField"))
    }

    @Test func rankPutsActionAboveReadingAboveGrouping() {
        #expect(ProposalMatcher.rank(of: "AXButton") > ProposalMatcher.rank(of: "AXStaticText"))
        #expect(ProposalMatcher.rank(of: "AXImage") > ProposalMatcher.rank(of: "AXStaticText"))
        #expect(ProposalMatcher.rank(of: "AXStaticText") > ProposalMatcher.rank(of: "AXGroup"))
        #expect(ProposalMatcher.rank(of: "AXGroup") > ProposalMatcher.rank(of: "VXRegion"))
    }

    @Test func aNestedElementWithItsOwnBoxStaysMatchable() {
        // A group containing a small button is NOT a duplicate — different boxes.
        let group = element(0, "AXGroup", PixelRect(x: 0, y: 0, width: 400, height: 200))
        let button = element(1, "AXButton", PixelRect(x: 10, y: 10, width: 80, height: 30), parent: 0)
        let outcome = ProposalMatcher.match(proposals: [], elements: [group, button])
        #expect(outcome.elements[0].matchable == true)
        #expect(outcome.elements[1].matchable == true)
    }

    @Test func twoStrongMatchesThatDisagreeBecomeIgnore() {
        // Siblings, not ancestor/descendant, so the dedup rule does not apply: a link
        // and a field stacked on nearly the same box. A coin flip here would teach the
        // model that the two are interchangeable.
        let link = element(0, "AXLink", PixelRect(x: 0, y: 0, width: 100, height: 40))
        let field = element(1, "AXTextField", PixelRect(x: 2, y: 0, width: 100, height: 40))
        let outcome = ProposalMatcher.match(
            proposals: [proposal(0, PixelRect(x: 1, y: 0, width: 100, height: 40))],
            elements: [link, field])
        #expect(outcome.regions[0].label == .ignore)
        #expect(outcome.regions[0].secondIoU >= 0.5)
    }

    @Test func twoStrongMatchesThatAgreeKeepTheRole() {
        let a = element(0, "AXCell", PixelRect(x: 0, y: 0, width: 100, height: 40))
        let b = element(1, "AXCell", PixelRect(x: 2, y: 0, width: 100, height: 40))
        let outcome = ProposalMatcher.match(
            proposals: [proposal(0, PixelRect(x: 1, y: 0, width: 100, height: 40))],
            elements: [a, b])
        #expect(outcome.regions[0].label == .role("AXCell"))
    }

    @Test func anUnmatchableElementNeverClaimsAProposal() {
        let hidden = element(0, "AXButton", PixelRect(x: 0, y: 0, width: 100, height: 40),
                             matchable: false)
        let outcome = ProposalMatcher.match(
            proposals: [proposal(0, PixelRect(x: 0, y: 0, width: 100, height: 40))],
            elements: [hidden])
        #expect(outcome.regions[0].label == .none)
        #expect(outcome.recall.overall.total == 0)
    }

    // MARK: - Recall

    @Test func recallCountsOnlyMatchableElements() {
        let found = element(0, "AXButton", PixelRect(x: 0, y: 0, width: 100, height: 40))
        let missed = element(1, "AXLink", PixelRect(x: 300, y: 300, width: 60, height: 20))
        let outcome = ProposalMatcher.match(
            proposals: [proposal(0, PixelRect(x: 0, y: 0, width: 100, height: 40))],
            elements: [found, missed])

        #expect(outcome.recall.overall.total == 2)
        #expect(outcome.recall.overall.found == 1)
        #expect(outcome.recall.byRole["AXButton"]?.found == 1)
        #expect(outcome.recall.byRole["AXLink"]?.found == 0)
    }

    @Test func recallBucketsBySmallestSide() {
        #expect(RecallReport.sizeBucket(forShortSide: 8) == "<16")
        #expect(RecallReport.sizeBucket(forShortSide: 16) == "16-32")
        #expect(RecallReport.sizeBucket(forShortSide: 31) == "16-32")
        #expect(RecallReport.sizeBucket(forShortSide: 64) == ">64")

        let wide = element(0, "AXRow", PixelRect(x: 0, y: 0, width: 900, height: 20))
        let outcome = ProposalMatcher.match(proposals: [], elements: [wide])
        #expect(outcome.recall.bySize["16-32"]?.total == 1, "a wide thin row buckets by its height")
    }

    @Test func reportsMerge() {
        let a = RecallReport(overall: RecallTally(total: 2, found: 1),
                             byRole: ["AXButton": RecallTally(total: 2, found: 1)])
        let b = RecallReport(overall: RecallTally(total: 3, found: 3),
                             byRole: ["AXButton": RecallTally(total: 1, found: 1),
                                      "AXLink": RecallTally(total: 2, found: 2)])
        let merged = a.merged(with: b)
        #expect(merged.overall == RecallTally(total: 5, found: 4))
        #expect(merged.byRole["AXButton"] == RecallTally(total: 3, found: 2))
        #expect(merged.byRole["AXLink"] == RecallTally(total: 2, found: 2))
    }

    // MARK: - Proposals from a real tree

    @Test func proposalsSkipTheRootAndKeepPreOrderParents() throws {
        let detection = try VisionEngine().detectRegions(
            in: SyntheticScreen.image(), title: "synthetic")
        let proposals = ProposalMatcher.proposals(from: detection.window)

        #expect(proposals.count == detection.nodeCount - 1, "the root is not a proposal")
        #expect(proposals.first?.parent == -1, "a top-level box's parent is the excluded root")
        for (slot, proposal) in proposals.enumerated() {
            #expect(proposal.index == slot)
            #expect(proposal.parent < slot, "a parent always precedes its child in pre-order")
        }
    }
}
