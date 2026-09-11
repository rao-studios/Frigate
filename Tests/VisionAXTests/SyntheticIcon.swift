//
//  SyntheticIcon.swift
//  VisionAXTests
//
//  WHAT: Icons drawn INDEPENDENTLY of the bank, for the bank to read back.
//  PIN:  NOT THE BANK'S OWN BITMAP. Rendering each mask and feeding it straight back
//        would prove only that a picture equals itself. These are drawn here, at a
//        different size, with different stroke weights and different proportions —
//        which is the actual claim: one drawing stands for every version of a magnifier
//        anybody has shipped.
//

import CoreGraphics
import Foundation
@testable import VisionAX

enum SyntheticIcon {

    static let side = 44

    /// A drawn icon on a button, in either polarity.
    static func image(_ glyph: IconGlyph, dark: Bool = false) -> CGImage? {
        guard let draw = drawing(for: glyph) else { return nil }
        return render { context in
            context.setFillColor(gray: dark ? 0.10 : 0.97, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: side, height: side))
            let ink = CGColor(gray: dark ? 0.94 : 0.08, alpha: 1)
            context.setStrokeColor(ink)
            context.setFillColor(ink)
            context.setLineWidth(3)
            context.setLineCap(.butt)
            draw(context)
            context.strokePath()
        }
    }

    static func blank() -> CGImage {
        render { context in
            context.setFillColor(gray: 0.95, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        }
    }

    /// The glyphs this fixture can draw. Deliberately a subset: these are the ones with
    /// a shape simple enough to draw twice without copying the bank.
    static let drawable: [IconGlyph] = [
        .search, .close, .menu, .add, .done, .more, .back, .forward, .up, .down,
        .star, .heart, .home, .filter, .download,
    ]
    // A gear is not drawable here without copying the bank's own teeth, and a fixture
    // that copies the thing it is testing proves nothing. It is covered by the live
    // probe over real chrome instead.

    private static func drawing(for glyph: IconGlyph) -> ((CGContext) -> Void)? {
        let s = Double(side)
        let c = s / 2
        switch glyph {
        case .search:
            return { context in
                context.addEllipse(in: CGRect(x: s * 0.16, y: s * 0.16, width: s * 0.44, height: s * 0.44))
                context.strokePath()
                context.move(to: CGPoint(x: s * 0.56, y: s * 0.56))
                context.addLine(to: CGPoint(x: s * 0.84, y: s * 0.84))
            }
        case .close:
            return { context in
                context.move(to: CGPoint(x: s * 0.2, y: s * 0.2))
                context.addLine(to: CGPoint(x: s * 0.8, y: s * 0.8))
                context.move(to: CGPoint(x: s * 0.8, y: s * 0.2))
                context.addLine(to: CGPoint(x: s * 0.2, y: s * 0.8))
            }
        case .menu:
            return { context in
                for fraction in [0.28, 0.5, 0.72] {
                    context.move(to: CGPoint(x: s * 0.18, y: s * fraction))
                    context.addLine(to: CGPoint(x: s * 0.82, y: s * fraction))
                }
            }
        case .add:
            return { context in
                context.move(to: CGPoint(x: c, y: s * 0.16))
                context.addLine(to: CGPoint(x: c, y: s * 0.84))
                context.move(to: CGPoint(x: s * 0.16, y: c))
                context.addLine(to: CGPoint(x: s * 0.84, y: c))
            }
        case .done:
            return { context in
                context.move(to: CGPoint(x: s * 0.2, y: s * 0.52))
                context.addLine(to: CGPoint(x: s * 0.42, y: s * 0.74))
                context.addLine(to: CGPoint(x: s * 0.8, y: s * 0.24))
            }
        case .more:
            return { context in
                for fraction in [0.2, 0.5, 0.8] {
                    context.fillEllipse(in: CGRect(
                        x: c - s * 0.07, y: s * fraction - s * 0.07,
                        width: s * 0.14, height: s * 0.14))
                }
            }
        case .back, .forward, .up, .down:
            return { context in
                let arm = s * 0.24
                switch glyph {
                case .back:
                    context.move(to: CGPoint(x: c + arm * 0.6, y: c - arm))
                    context.addLine(to: CGPoint(x: c - arm * 0.5, y: c))
                    context.addLine(to: CGPoint(x: c + arm * 0.6, y: c + arm))
                case .forward:
                    context.move(to: CGPoint(x: c - arm * 0.6, y: c - arm))
                    context.addLine(to: CGPoint(x: c + arm * 0.5, y: c))
                    context.addLine(to: CGPoint(x: c - arm * 0.6, y: c + arm))
                case .up:
                    context.move(to: CGPoint(x: c - arm, y: c + arm * 0.6))
                    context.addLine(to: CGPoint(x: c, y: c - arm * 0.5))
                    context.addLine(to: CGPoint(x: c + arm, y: c + arm * 0.6))
                default:
                    context.move(to: CGPoint(x: c - arm, y: c - arm * 0.6))
                    context.addLine(to: CGPoint(x: c, y: c + arm * 0.5))
                    context.addLine(to: CGPoint(x: c + arm, y: c - arm * 0.6))
                }
            }
        case .star:
            return { context in
                var points: [CGPoint] = []
                for index in 0 ..< 10 {
                    let angle = -Double.pi / 2 + Double(index) * Double.pi / 5
                    let radius = index % 2 == 0 ? s * 0.42 : s * 0.18
                    points.append(CGPoint(x: c + radius * cos(angle), y: c + radius * sin(angle)))
                }
                context.addLines(between: points)
                context.closePath()
                context.fillPath()
            }
        case .heart:
            return { context in
                context.fillEllipse(in: CGRect(
                    x: s * 0.14, y: s * 0.22, width: s * 0.38, height: s * 0.38))
                context.fillEllipse(in: CGRect(
                    x: s * 0.48, y: s * 0.22, width: s * 0.38, height: s * 0.38))
                context.addLines(between: [
                    CGPoint(x: s * 0.12, y: s * 0.44),
                    CGPoint(x: s * 0.88, y: s * 0.44),
                    CGPoint(x: c, y: s * 0.9),
                ])
                context.closePath()
                context.fillPath()
            }
        case .home:
            return { context in
                context.addLines(between: [
                    CGPoint(x: c, y: s * 0.12),
                    CGPoint(x: s * 0.9, y: s * 0.48),
                    CGPoint(x: s * 0.1, y: s * 0.48),
                ])
                context.closePath()
                context.fillPath()
                context.stroke(CGRect(x: s * 0.24, y: s * 0.48, width: s * 0.52, height: s * 0.4))
            }
        case .settings:
            return { context in
                context.strokeEllipse(in: CGRect(
                    x: s * 0.32, y: s * 0.32, width: s * 0.36, height: s * 0.36))
                context.strokeEllipse(in: CGRect(
                    x: s * 0.1, y: s * 0.1, width: s * 0.8, height: s * 0.8))
            }
        case .filter:
            return { context in
                context.move(to: CGPoint(x: s * 0.12, y: s * 0.24))
                context.addLine(to: CGPoint(x: s * 0.88, y: s * 0.24))
                context.move(to: CGPoint(x: s * 0.26, y: s * 0.5))
                context.addLine(to: CGPoint(x: s * 0.74, y: s * 0.5))
                context.move(to: CGPoint(x: s * 0.38, y: s * 0.76))
                context.addLine(to: CGPoint(x: s * 0.62, y: s * 0.76))
            }
        case .download:
            return { context in
                context.move(to: CGPoint(x: c, y: s * 0.12))
                context.addLine(to: CGPoint(x: c, y: s * 0.62))
                context.move(to: CGPoint(x: s * 0.26, y: s * 0.42))
                context.addLine(to: CGPoint(x: c, y: s * 0.66))
                context.addLine(to: CGPoint(x: s * 0.74, y: s * 0.42))
                context.move(to: CGPoint(x: s * 0.16, y: s * 0.86))
                context.addLine(to: CGPoint(x: s * 0.84, y: s * 0.86))
            }
        default:
            return nil
        }
    }

    private static func render(_ body: (CGContext) -> Void) -> CGImage {
        let context = CGContext(
            data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue)!
        // IMAGE COORDINATES, NOT CORE GRAPHICS'. The bank draws with the origin at the
        // top left, the way a screenshot is indexed; a context left in its own bottom-up
        // space renders every chevron upside down, and the first run of this fixture
        // duly reported "up read as down".
        context.translateBy(x: 0, y: CGFloat(side))
        context.scaleBy(x: 1, y: -1)
        body(context)
        return context.makeImage()!
    }
}
