//
//  RegionTreeBridge.swift
//  VisionAX
//
//  WHAT: The engine's flat pre-order regions → a nested AXNodeSnapshot tree.
//  IN:   VisionEngine (vx_region array)
//  OUT:  AXWindowSnapshot — the shape Mary already walks
//  PIN:  NO LABELS YET. Every detected region is role "VXRegion", category .other,
//        label nil — the classifier step fills those in later, and a fabricated
//        label now would be indistinguishable from a real one then.
//

import CVisionAX
import CoreGraphics
import Foundation
import VisionAXCore

enum RegionTreeBridge {

    /// `regions` must be the engine's pre-order array: index 0 is the root, and every
    /// other node's `parent` points at a LOWER index.
    static func window(
        from regions: [vx_region],
        title: String,
        imageBounds: CGRect,
        isTruncated: Bool
    ) -> AXWindowSnapshot {
        guard !regions.isEmpty else {
            return AXWindowSnapshot(
                id: AXNodeID(raw: 1), title: title, frame: imageBounds,
                isMain: true, isTruncated: isTruncated,
                root: AXNodeSnapshot(
                    id: AXNodeID(raw: 1), role: VisionAX.windowRole,
                    frame: imageBounds, category: .window))
        }

        // Child index lists, in the order the engine emitted them (reading order).
        var childIndexes = [[Int]](repeating: [], count: regions.count)
        for (index, region) in regions.enumerated() where region.parent >= 0 {
            let parent = Int(region.parent)
            guard parent < childIndexes.count else { continue }
            childIndexes[parent].append(index)
        }

        // Bottom-up: a node's children are always at higher indexes in a pre-order
        // array, so one reverse pass builds every subtree without recursion.
        var built = [AXNodeSnapshot?](repeating: nil, count: regions.count)
        for index in stride(from: regions.count - 1, through: 0, by: -1) {
            let region = regions[index]
            let children = childIndexes[index].compactMap { built[$0] }
            let isRoot = region.parent < 0
            built[index] = AXNodeSnapshot(
                id: AXNodeID(raw: UInt(region.id)),
                role: isRoot ? VisionAX.windowRole : VisionAX.regionRole,
                subrole: nil,
                label: nil,
                frame: CGRect(
                    x: CGFloat(region.x), y: CGFloat(region.y),
                    width: CGFloat(region.width), height: CGFloat(region.height)),
                isEnabled: true,
                isFocused: false,
                category: isRoot ? .window : .other,
                children: children)
        }

        let root = built[0]
        return AXWindowSnapshot(
            id: root?.id ?? AXNodeID(raw: 1),
            title: title,
            frame: root?.frame ?? imageBounds,
            isMain: true,
            isMinimized: false,
            isTruncated: isTruncated,
            root: root)
    }
}
