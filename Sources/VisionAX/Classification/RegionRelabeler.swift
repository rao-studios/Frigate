//
//  RegionRelabeler.swift
//  VisionAX
//
//  WHAT: Turns per-region answers back into a tree Mary can walk.
//  IN:   VisionEngine.classifyRegions
//  OUT:  AXWindowSnapshot with role + category filled
//  PIN:  BELOW THE THRESHOLD A NODE KEEPS "VXRegion", AND THAT IS THE SAFE ANSWER, not
//        a cop-out. VXRegion categorizes as .other, and Mary drops .other nodes from
//        every roster and draws none of them — so an unsure guess costs her nothing,
//        while a confident wrong role gets clicked. `label` and `subrole` stay nil
//        because this step knows neither; inventing either would put words in Mary's
//        mouth that no pixel supports.
//

import CoreGraphics
import Foundation
import VisionAXCore

public enum RegionRelabeler {
    /// Every node a classifier should be asked about: pre-order, root excluded, only
    /// nodes that have a frame to crop.
    ///
    /// The root is the whole image, which is not an element; asking about it would
    /// spend a crop to be told `none`.
    public static func classifiableNodes(in window: AXWindowSnapshot) -> [(id: AXNodeID, frame: CGRect)] {
        guard let root = window.root else { return [] }
        var nodes: [(id: AXNodeID, frame: CGRect)] = []
        root.forEachNode(withAncestors: { node, ancestors in
            guard !ancestors.isEmpty, let frame = node.frame else { return }
            nodes.append((node.id, frame))
        })
        return nodes
    }

    /// Rebuilds the window with each node named by its label, where the label is sure
    /// enough. Nodes with no label are left exactly as they were.
    public static func relabeled(
        _ window: AXWindowSnapshot,
        labels: [AXNodeID: RegionLabel],
        minimumConfidence: Double
    ) -> AXWindowSnapshot {
        guard let root = window.root else { return window }
        var relabeled = window
        relabeled.root = rename(root, labels: labels, minimumConfidence: minimumConfidence, isRoot: true)
        return relabeled
    }

    private static func rename(
        _ node: AXNodeSnapshot,
        labels: [AXNodeID: RegionLabel],
        minimumConfidence: Double,
        isRoot: Bool
    ) -> AXNodeSnapshot {
        let children = node.children.map {
            rename($0, labels: labels, minimumConfidence: minimumConfidence, isRoot: false)
        }
        // The root is the image, not a region: it keeps AXWindow whatever a model says.
        let role: String
        if isRoot {
            role = node.role
        } else if let label = labels[node.id], label.names(atLeast: minimumConfidence) {
            role = label.role
        } else {
            role = VisionAX.regionRole
        }

        return AXNodeSnapshot(
            id: node.id,
            role: role,
            subrole: nil,
            label: nil,
            frame: node.frame,
            isEnabled: node.isEnabled,
            isFocused: node.isFocused,
            category: AXNodeCategory.category(role: role),
            children: children)
    }
}
