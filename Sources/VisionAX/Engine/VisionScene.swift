//
//  VisionScene.swift
//  VisionAX
//
//  WHAT: One perception, answerable in screen coordinates — the type a consumer holds.
//  IN:   VisionEngine.perceive
//  OUT:  Mary's page lane; the bench
//  PIN:  THE SCENE ANSWERS IN POINTS, THE DETECTION SPEAKS IN PIXELS. Every query here
//        takes and returns SCREEN coordinates, because a caller that wants to click
//        something should never have to know a scale existed. The pixel frames stay
//        reachable underneath for anyone drawing over the image itself.
//        A HIT TEST PICKS THE SMALLEST CONTAINING BOX. Regions nest, and the largest
//        box containing a point is almost always the page; the smallest is the thing.
//

import CoreGraphics
import Foundation
import VisionAXCore

public struct VisionScene: Sendable {

    /// Nil when the regions lane was not asked for.
    public let detection: VisionDetection?
    /// Where these pixels were on screen. Already offset for the region of interest.
    public let projection: ScreenProjection
    /// The crop, in the ORIGINAL image's pixels — what was actually examined.
    public let regionOfInterest: CGRect
    /// The examined pixels' own bounds, origin zero.
    public let imageBounds: CGRect
    public let capturedAt: Date
    public let text: [TextRun]?
    public let media: MediaControlDetection?
    /// Names for wordless controls, by node. Empty when nothing was named — which is
    /// not the same as nothing being there.
    public let icons: [AXNodeID: String]
    /// What the model that named these roles says they afford. Nil without a model, or
    /// with one exported before the table existed.
    public let affordances: [String: [String]]?
    /// Where this perception's time went, phase by phase. The phases sum to
    /// `timing.total`, which is the same span as `duration`.
    public let timing: VisionTiming
    public let duration: Duration

    public init(
        detection: VisionDetection?,
        projection: ScreenProjection,
        regionOfInterest: CGRect,
        imageBounds: CGRect,
        capturedAt: Date = Date(),
        text: [TextRun]? = nil,
        media: MediaControlDetection? = nil,
        icons: [AXNodeID: String] = [:],
        affordances: [String: [String]]? = nil,
        timing: VisionTiming = VisionTiming(),
        duration: Duration = .zero
    ) {
        self.detection = detection
        self.projection = projection
        self.regionOfInterest = regionOfInterest
        self.imageBounds = imageBounds
        self.capturedAt = capturedAt
        self.text = text
        self.media = media
        self.icons = icons
        self.affordances = affordances
        self.timing = timing
        self.duration = duration
    }

    /// The examined region, on screen.
    public var screenBounds: CGRect { projection.screenRect(imageBounds) }

    // MARK: - Walking

    /// Every node in the tree, root first, pre-order. Empty without the regions lane.
    public var nodes: [AXNodeSnapshot] {
        guard let root = detection?.window.root else { return [] }
        var found: [AXNodeSnapshot] = []
        root.forEachNode { found.append($0) }
        return found
    }

    /// Nodes a classifier named — everything the caller can reason about by role.
    /// The root and unnamed `VXRegion`s are left out: an unnamed box is a proposal.
    public var namedNodes: [AXNodeSnapshot] {
        nodes.filter { $0.role != VisionAX.regionRole && $0.role != VisionAX.windowRole }
    }

    public func elements(role: String) -> [AXNodeSnapshot] {
        nodes.filter { $0.role == role }
    }

    public func elements(category: AXNodeCategory) -> [AXNodeSnapshot] {
        nodes.filter { $0.category == category }
    }

    // MARK: - Asking in screen coordinates

    /// The node's frame on screen.
    public func screenRect(of node: AXNodeSnapshot) -> CGRect? {
        node.frame.map(projection.screenRect)
    }

    public func screenRect(of id: AXNodeID) -> CGRect? {
        node(withID: id)?.frame.map(projection.screenRect)
    }

    /// One node by id, or nil once the tree it came from has been replaced.
    public func node(withID id: AXNodeID) -> AXNodeSnapshot? {
        guard let window = detection?.window else { return nil }
        if window.id == id { return window.root }
        return window.root?.subtree(withID: id)
    }

    /// Where to click the middle of a node.
    public func screenPoint(centerOf node: AXNodeSnapshot) -> CGPoint? {
        node.frame.map(projection.screenPoint(centerOf:))
    }

    /// The smallest node containing a screen point — the innermost thing under it.
    public func hitTest(screenPoint point: CGPoint) -> AXNodeSnapshot? {
        var best: AXNodeSnapshot?
        var bestArea = CGFloat.greatestFiniteMagnitude
        for node in nodes {
            guard let frame = node.frame else { continue }
            let onScreen = projection.screenRect(frame)
            guard onScreen.contains(point) else { continue }
            let area = onScreen.width * onScreen.height
            if area < bestArea {
                bestArea = area
                best = node
            }
        }
        return best
    }

    /// Nodes whose screen frame falls inside a screen rect.
    public func within(screenRect rect: CGRect) -> [AXNodeSnapshot] {
        nodes.filter { node in
            guard let frame = node.frame else { return false }
            return rect.contains(projection.screenRect(frame))
        }
    }

    /// The closest node to a screen point, by centre distance.
    public func nearest(to point: CGPoint, role: String? = nil) -> AXNodeSnapshot? {
        var best: AXNodeSnapshot?
        var bestDistance = CGFloat.greatestFiniteMagnitude
        for node in nodes {
            if let role, node.role != role { continue }
            guard let frame = node.frame else { continue }
            let onScreen = projection.screenRect(frame)
            let dx = onScreen.midX - point.x
            let dy = onScreen.midY - point.y
            let distance = dx * dx + dy * dy
            if distance < bestDistance {
                bestDistance = distance
                best = node
            }
        }
        return best
    }

    // MARK: - The roster

    /// The scene as flat roster rows, in reading order, in SCREEN points.
    ///
    /// `.other` nodes are dropped, which is the same rule Mary's own roster follows: an
    /// unnamed proposal is not something to offer anybody. Labels come from the text
    /// lane when it ran — the box a word sits inside is that box's name.
    public func roster(
        pid: pid_t,
        appName: String,
        windowTitle: String,
        windowID: AXNodeID = AXNodeID(raw: 1),
        limit: Int = 120
    ) -> [AXScreenElement] {
        var rows: [AXScreenElement] = []
        var ordinal = 0
        for node in nodes {
            guard node.category != .other, node.role != VisionAX.windowRole,
                  let frame = node.frame
            else { continue }
            let onScreen = projection.screenRect(frame)
            let label = node.label ?? labelFromText(in: frame)
            guard let label, !label.isEmpty else { continue }
            ordinal += 1
            rows.append(AXScreenElement(
                ordinal: ordinal,
                id: node.id,
                pid: pid,
                appName: appName,
                windowID: windowID,
                windowTitle: windowTitle,
                role: node.role,
                subrole: node.subrole,
                category: node.category,
                label: label,
                frame: onScreen,
                isEnabled: node.isEnabled,
                isFocused: node.isFocused))
            if rows.count >= limit { break }
        }
        return rows
    }

    /// The words inside a pixel frame, joined — a box's name when the tree has none.
    /// Runs must be MOSTLY inside: a caption clipped by a card's edge belongs to the
    /// card, and a run that merely touches it belongs to its neighbour.
    public func labelFromText(in frame: CGRect) -> String? {
        guard let text, !text.isEmpty else { return nil }
        let inside = text.filter { run in
            let overlap = run.frame.intersection(frame)
            guard !overlap.isNull else { return false }
            let runArea = run.frame.width * run.frame.height
            guard runArea > 0 else { return false }
            return (overlap.width * overlap.height) / runArea >= 0.7
        }
        guard !inside.isEmpty else { return nil }
        return inside
            .sorted { $0.frame.minY == $1.frame.minY ? $0.frame.minX < $1.frame.minX : $0.frame.minY < $1.frame.minY }
            .map(\.string)
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Media, in screen coordinates

    /// Where to click a transport control.
    public func screenPoint(of control: MediaControlDetection.Control) -> CGPoint {
        projection.screenPoint(centerOf: control.frame)
    }

    /// The progress track on screen, for a seek.
    public var progressScreenRect: CGRect? {
        media?.progress.map { projection.screenRect($0.frame) }
    }
}
