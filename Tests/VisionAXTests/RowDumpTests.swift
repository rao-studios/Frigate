//
//  RowDumpTests.swift
//  VisionAXTests
//
//  WHAT: The pixels along a row, printed. A tuning instrument, not an assertion.
//  OUT:  stdout, only when VISIONAX_ROW_DUMP is set
//  PIN:  KEPT BECAUSE THE NEXT PLAYER WILL NEED IT. Every real fix in the media lane so
//        far came from reading actual pixel values across the track — that a translucent
//        unplayed bar carries the video's own variation, and by how much — and guessing
//        at thresholds without them cost more time than writing this did.
//
//        VISIONAX_ROW_DUMP=1 VISIONAX_ROW_IMAGE=youtube-safari-playing \
//            VISIONAX_ROWS=508,511,514 swift test --filter dumpRows
//

import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import VisionAX

@Suite struct RowDumpTests {

    /// Read ANY image off disk and say what the media lane makes of it.
    ///
    /// VISIONAX_MEDIA_FILE=/path/to/capture.png swift test --filter readFile
    @Test func readFile() throws {
        guard let path = ProcessInfo.processInfo.environment["VISIONAX_MEDIA_FILE"] else { return }
        let source = try #require(
            CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        let engine = try VisionEngine()
        let scene = try engine.perceive(
            image: image, projection: .imageSpace, lanes: [.media, .text])
        let media = try #require(scene.media)
        print("FILE \(path) \(image.width)x\(image.height)")
        print("  controlsVisible \(media.controlsVisible)")
        print("  bar \(String(describing: media.bar))")
        print("  progress \(String(describing: media.progress))")
        print("  playback \(media.playback) witnesses \(media.witnesses)")
        print("  clock \(String(describing: media.elapsed)) / \(String(describing: media.duration))")
        print("  transport \(String(describing: media.playPause))")
        print("  volume \(String(describing: media.volume)) muted=\(String(describing: media.isMuted))")
        print("  fullscreen \(String(describing: media.fullscreen))")
        for control in media.controls {
            print("  control \(control.frame) \(control.glyph) \(String(format: "%.3f", control.confidence))")
        }
    }
    @Test func dumpRows() throws {
        guard ProcessInfo.processInfo.environment["VISIONAX_ROW_DUMP"] != nil else { return }
        let name = ProcessInfo.processInfo.environment["VISIONAX_ROW_IMAGE"] ?? "youtube-safari-playing"
        let url: URL
        if let path = ProcessInfo.processInfo.environment["VISIONAX_MEDIA_FILE"] {
            url = URL(fileURLWithPath: path)
        } else {
            url = try #require(Bundle.module.url(
                forResource: name, withExtension: "png", subdirectory: "Fixtures/media"))
        }
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        let width = image.width, height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        pixels.withUnsafeMutableBytes { raw in
            let ctx = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                    | CGBitmapInfo.byteOrder32Little.rawValue)!
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        func px(_ x: Int, _ y: Int) -> (Int, Int, Int) {
            let i = (y * width + x) * 4
            return (Int(pixels[i]), Int(pixels[i + 1]), Int(pixels[i + 2]))
        }
        let rows = (ProcessInfo.processInfo.environment["VISIONAX_ROWS"] ?? "508,510,512,514")
            .split(separator: ",").compactMap { Int($0) }
        for y in rows {
            var line = "y=\(y): "
            for x in stride(from: 0, to: width, by: 40) {
                let (b, g, r) = px(x, y)
                line += "\(x):(\(r),\(g),\(b)) "
            }
            print(line)
        }
    }
}
