//
//  ProposalUnion.swift
//  VisionAX
//
//  WHAT: Text lines added to the detector's proposals, before anything is classified.
//  IN:   a detection + the text lane's runs
//  OUT:  the same tree with the missing boxes in it
//  PIN:  THE DETECTOR CANNOT SEE A LINK THAT HAS NO EDGE. A result title is an anchor
//        around a heading — no border, no fill, no shadow — so Canny finds nothing and
//        the row never becomes a proposal. MEASURED: 33% proposal recall on real pages
//        and 0 of 1,736 rows. The words are right there; the text lane already read
//        them; this is the two halves meeting.
//        THE SAME UNION RUNS AT HARVEST. A box the model meets only at inference is a
//        box it was never trained on, and it answers `none` for exactly the rows that
//        matter. `HarvestSession` calls this too, so training sees what serving sees.
//        APPENDED, NEVER INSERTED IN THE MIDDLE. Downstream code pairs labels with
//        nodes by walk order; a new node in the middle of the tree would renumber
//        everything after it.
//

import CoreGraphics
import Foundation
import VisionAXCore

public enum ProposalUnion {

    /// Text boxes a detection did not already find, as a detection.
    public static func union(
        _ detection: VisionDetection, textRuns: [TextRun]
    ) -> VisionDetection {
        let lines = TextLines.lines(from: textRuns)
        guard !lines.isEmpty else { return detection }
        var existing: [CGRect] = []
        detection.window.root?.forEachNode { node in
            if let frame = node.frame { existing.append(frame) }
        }
        let boxes = TextLines.proposals(fromLines: lines, notCovering: existing)
        guard !boxes.isEmpty else { return detection }
        return union(detection, boxes: boxes)
    }

    /// The same, for a caller that already has the boxes.
    public static func union(_ detection: VisionDetection, boxes: [CGRect]) -> VisionDetection {
        guard let root = detection.window.root, !boxes.isEmpty else { return detection }
        var nextID = highestID(in: root) + 1
        let grown = insert(into: root, boxes: boxes, nextID: &nextID)
        var count = 0
        grown.forEachNode { _ in count += 1 }
        return VisionDetection(
            window: AXWindowSnapshot(
                id: detection.window.id,
                title: detection.window.title,
                frame: detection.window.frame,
                isMain: detection.window.isMain,
                isTruncated: detection.window.isTruncated,
                root: grown),
            edges: detection.edges,
            options: detection.options,
            nodeCount: count,
            contourCount: detection.contourCount,
            duration: detection.duration,
            labels: detection.labels,
            classificationDuration: detection.classificationDuration,
            minimumConfidence: detection.minimumConfidence)
    }

    /// Each box under the smallest node that contains it.
    static func insert(
        into node: AXNodeSnapshot, boxes: [CGRect], nextID: inout UInt
    ) -> AXNodeSnapshot {
        guard !boxes.isEmpty else { return node }
        var mine: [CGRect] = []
        var byChild = [[CGRect]](repeating: [], count: node.children.count)
        for box in boxes {
            let owner = node.children.indices
                .filter { index in
                    guard let frame = node.children[index].frame else { return false }
                    return frame.contains(box)
                }
                .min { left, right in
                    let leftFrame = node.children[left].frame ?? .infinite
                    let rightFrame = node.children[right].frame ?? .infinite
                    return leftFrame.width * leftFrame.height
                        < rightFrame.width * rightFrame.height
                }
            if let owner {
                byChild[owner].append(box)
            } else {
                mine.append(box)
            }
        }

        var children = node.children.enumerated().map { index, child in
            insert(into: child, boxes: byChild[index], nextID: &nextID)
        }
        for box in mine.sorted(by: {
            $0.minY == $1.minY ? $0.minX < $1.minX : $0.minY < $1.minY
        }) {
            children.append(AXNodeSnapshot(
                id: AXNodeID(raw: nextID),
                role: VisionAX.regionRole,
                frame: box,
                category: .other))
            nextID += 1
        }
        return AXNodeSnapshot(
            id: node.id,
            role: node.role,
            subrole: node.subrole,
            label: node.label,
            frame: node.frame,
            isEnabled: node.isEnabled,
            isFocused: node.isFocused,
            category: node.category,
            children: children)
    }

    static func highestID(in node: AXNodeSnapshot) -> UInt {
        var highest: UInt = 0
        node.forEachNode { highest = max(highest, $0.id.raw) }
        return highest
    }
}
