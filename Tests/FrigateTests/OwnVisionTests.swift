//  WHAT: A loaded model reading a picture with its own tower (`OwnVision`). The tower's host-side arithmetic always;
//        on the checkpoints, with FRIGATE_MLX_TESTS=1 and FRIGATE_INKLING2 (or FRIGATE_QWEN35_VISION) naming a folder, a
//        picture of a number read back by the model.

import CoreGraphics
import CoreText
import Foundation
import FrigateBridge
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM
@testable import MLXVLM

private func folder(_ name: String) -> URL? {
    guard ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1",
          let path = ProcessInfo.processInfo.environment[name] else { return nil }
    return URL(fileURLWithPath: path)
}

/// A white picture with `text` written large in black.
func picture(_ text: String, width: Int = 640, height: Int = 200) -> CGImage {
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    context.setFillColor(gray: 1, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    let font = CTFontCreateWithName("Helvetica-Bold" as CFString, 96, nil)
    let attributes = [kCTFontAttributeName: font, kCTForegroundColorFromContextAttributeName: true] as CFDictionary
    let line = CTLineCreateWithAttributedString(CFAttributedStringCreate(nil, text as CFString, attributes))
    context.setFillColor(gray: 0, alpha: 1)
    context.textPosition = CGPoint(x: 40, y: 60)
    CTLineDraw(line, context)
    return context.makeImage()!
}

@Suite("Own vision: the tower's arithmetic")
struct OwnVisionArithmeticTests {
    @Test func theTentResamplesTheGridAndEachRowSumsToOne() {
        // The learned grid's own size is the grid itself.
        let same = Inkling2VisionEmbeddings.tent(4, grid: 4)
        #expect(same == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        for size in [1, 3, 17, 64] {
            let w = Inkling2VisionEmbeddings.tent(size, grid: 32)
            for row in 0 ..< size {
                let sum = w[row * 32 ..< (row + 1) * 32].reduce(0, +)
                #expect(abs(sum - 1) < 1e-5, "row \(row) of \(size)")
            }
        }
        // Half the grid: each position halfway between two of the grid's.
        let half = Inkling2VisionEmbeddings.tent(2, grid: 4)
        #expect(half == [0.5, 0.5, 0, 0, 0, 0, 0.5, 0.5])
    }

    @Test func theAnglesPutTheClassTokenAtZeroAndTheColumnFirst() {
        let a = Inkling2VisionBackbone.angles(height: 2, width: 3, headDim: 8)    // two frequencies: 1, 10000^-0.5
        #expect(a.count == 7 * 4)
        #expect(a[0 ..< 4].allSatisfy { $0 == 0 })
        // Token 1 is (row 1, column 1); token 6 is (row 2, column 3).
        func near(_ got: ArraySlice<Float>, _ want: [Float]) -> Bool { zip(got, want).allSatisfy { abs($0 - $1) < 1e-6 } }
        #expect(near(a[4 ..< 8], [1, 0.01, 1, 0.01]))
        #expect(near(a[24 ..< 28], [3, 0.03, 2, 0.02]))
    }

    @Test func aPictureIsCutIntoFortyPixelTilesPaddedWithMinusOne() throws {
        let (pixels, rows, columns) = try Inkling2VisionTower.pixels(picture("7", width: 81, height: 40), tile: 40)
        #expect(rows == 1 && columns == 3)
        #expect(pixels.count == 40 * 120 * 3)
        #expect(pixels[0] == 255, "the picture's white corner")
        #expect(pixels[(0 * 120 + 81) * 3] == -1, "past the picture")
        #expect(pixels[(39 * 120 + 80) * 3] == 255, "the picture's last pixel")
    }
}

@Suite("Own vision: Inkling-2 reads a picture", .enabled(if: folder("FRIGATE_INKLING2") != nil), .serialized)
struct Inkling2OwnVisionLiveTests {
    @Test func itReadsTheNumberInThePicture() async throws {
        let directory = folder("FRIGATE_INKLING2")!
        let container = try await loadModelContainer(from: directory, using: HubTokenizerLoader())
        let (model, tokenizer) = await container.perform { context in (SendableBox(context.model), SendableBox(context.tokenizer)) }
        let vision = try #require(try Inkling2OwnVision.over(model.consume(), tokenizer: tokenizer.consume(), directory: directory))
        let started = Date()
        let read = try await vision.reader(image: picture("4217"), question: "What number is written in the picture? Answer with the number only.",
                                           maxTokens: 32)
        let answer = try read()
        print("Inkling-2 read \"\(answer)\" in \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
        #expect(answer.contains("4217"))
    }
}

@Suite("Own vision: Qwen3.5 reads a picture", .enabled(if: folder("FRIGATE_QWEN35_VISION") != nil), .serialized)
struct Qwen35OwnVisionLiveTests {
    @Test func itReadsTheNumberInThePicture() async throws {
        let directory = folder("FRIGATE_QWEN35_VISION")!
        let container = try await loadModelContainer(from: directory, using: HubTokenizerLoader())
        let (model, tokenizer) = await container.perform { context in (SendableBox(context.model), SendableBox(context.tokenizer)) }
        let vision = try #require(try await Qwen35OwnVision.over(model.consume(), tokenizer: tokenizer.consume(), directory: directory))
        let started = Date()
        let read = try await vision.reader(image: picture("4217"), question: "What number is written in the picture? Answer with the number only.",
                                           maxTokens: 32)
        let answer = try read()
        print("Qwen3.5 read \"\(answer)\" in \(String(format: "%.1f", Date().timeIntervalSince(started))) s")
        #expect(answer.contains("4217"))
    }
}
