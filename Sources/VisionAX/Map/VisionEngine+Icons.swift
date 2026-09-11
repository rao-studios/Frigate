//
//  VisionEngine+Icons.swift
//  VisionAX
//
//  WHAT: Name the wordless controls in a frame.
//  IN:   the page map, for boxes with nothing written in them
//  OUT:  [AXNodeID: String] for the label ladder
//  PIN:  ASKED FOR ONLY WHERE IT COULD HELP. Every box on a page put through the bank
//        would be twenty-two overlap tests each for a page's worth of rows, to answer a
//        question about the few dozen boxes that have no words in them. The caller
//        filters first — small, near-square, no text inside — and this names those.
//

import CVisionAX
import CoreGraphics
import Foundation
import VisionAXCore

public extension VisionEngine {

    /// One answer per box, in the order they were given.
    func matchIcons(in image: CGImage, boxes: [CGRect]) throws -> [IconMatch] {
        guard !boxes.isEmpty else { return [] }
        guard let buffer = VisionImageBuffer(image: image) else {
            throw VisionEngineError.unreadableImage
        }
        return try matchIcons(buffer: buffer, boxes: boxes)
    }

    internal func matchIcons(
        buffer: VisionImageBuffer, boxes: [CGRect]
    ) throws -> [IconMatch] {
        guard !boxes.isEmpty else { return [] }
        var rects = boxes.map { box in
            vx_rect(
                x: Int32(box.origin.x.rounded()),
                y: Int32(box.origin.y.rounded()),
                width: Int32(box.size.width.rounded()),
                height: Int32(box.size.height.rounded()))
        }
        var matches = [vx_icon_match](
            repeating: vx_icon_match(glyph: VX_ICON_NONE, confidence: 0), count: boxes.count)

        let status: vx_status = buffer.withImageView { view in
            var view = view
            return rects.withUnsafeMutableBufferPointer { rectBuffer in
                matches.withUnsafeMutableBufferPointer { matchBuffer in
                    vx_engine_match_icons(
                        handle, &view, rectBuffer.baseAddress,
                        Int32(boxes.count), matchBuffer.baseAddress)
                }
            }
        }
        guard status == VX_OK else {
            throw VisionEngineError.engine(
                status: status.rawValue,
                message: String(cString: vx_status_message(status)))
        }
        return matches.map { IconMatch(glyph: IconGlyph($0.glyph), confidence: $0.confidence) }
    }
}

extension IconGlyph {
    /// Spelled out rather than bridged by raw value: the C enum and this one are
    /// separate vocabularies that happen to agree today.
    init?(_ raw: vx_icon_glyph) {
        switch raw {
        case VX_ICON_SEARCH: self = .search
        case VX_ICON_CLOSE: self = .close
        case VX_ICON_MENU: self = .menu
        case VX_ICON_BACK: self = .back
        case VX_ICON_FORWARD: self = .forward
        case VX_ICON_UP: self = .up
        case VX_ICON_DOWN: self = .down
        case VX_ICON_ADD: self = .add
        case VX_ICON_MORE: self = .more
        case VX_ICON_SHARE: self = .share
        case VX_ICON_MICROPHONE: self = .microphone
        case VX_ICON_NOTIFICATIONS: self = .notifications
        case VX_ICON_ACCOUNT: self = .account
        case VX_ICON_STAR: self = .star
        case VX_ICON_HEART: self = .heart
        case VX_ICON_DELETE: self = .delete
        case VX_ICON_DONE: self = .done
        case VX_ICON_SETTINGS: self = .settings
        case VX_ICON_HOME: self = .home
        case VX_ICON_FILTER: self = .filter
        case VX_ICON_CART: self = .cart
        case VX_ICON_DOWNLOAD: self = .download
        default: return nil
        }
    }
}
