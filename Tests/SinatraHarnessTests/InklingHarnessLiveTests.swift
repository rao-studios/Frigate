//
//  InklingHarnessLiveTests.swift
//  SinatraHarnessTests
//
//  Copyright © 2026 Rao Studios LLC.
//
//  WHAT: Inkling-2 through the harness as Sewn runs it: an answer with no frame markers in it,
//        a call that parses, and a picture read by the model's own tower under the harness's gate.
//        With FRIGATE_MLX_TESTS=1 and FRIGATE_INKLING2 naming the checkpoint's folder (19.5 GB).
//

import CoreGraphics
import Foundation
import MLXLMCommon
import Testing
@testable import SinatraHarness

private var inklingFolder: URL? {
    guard mlxAvailable, let path = ProcessInfo.processInfo.environment["FRIGATE_INKLING2"] else { return nil }
    return URL(fileURLWithPath: path)
}

@Suite("Inkling-2 through the harness", .enabled(if: inklingFolder != nil), .serialized)
struct InklingHarnessLiveTests {
    nonisolated(unsafe) static var harness: Harness?

    private func loaded() async throws -> Harness {
        if let harness = Self.harness { return harness }
        let store = FileManager.default.temporaryDirectory.appending(component: "inkling-harness-\(UUID().uuidString)")
        let harness = Harness(storeDirectory: store)
        try await harness.load(directory: inklingFolder!, modelKey: "rao-studios/Inkling-2-Mini")
        Self.harness = harness
        return harness
    }

    private func run(_ input: UserInput, tools: [[String: any Sendable]]? = nil) async throws -> (text: String, calls: [ToolCall]) {
        let harness = try await loaded()
        let request = GenerateRequest(
            input: input, parameters: GenerateParameters(maxTokens: 200, temperature: 0), trace: .off, record: false,
            tools: tools)
        let handle = try await harness.generate(request)
        var text = ""
        var calls: [ToolCall] = []
        for await item in handle.stream {
            switch item {
            case .chunk(let chunk): text += chunk
            case .toolCall(let call): calls.append(call)
            default: break
            }
        }
        _ = await handle.completion.value
        return (text, calls)
    }

    @Test func anAnswerAtEffortNoneHasNoFrames() async throws {
        let input = UserInput(
            chat: [.system("You are terse."), .user("Name the capital of France in one word.")],
            additionalContext: ["reasoning_effort": "none"])
        let reply = try await run(input)
        #expect(!reply.text.contains("<|"), "\(reply.text)")
        #expect(reply.text.lowercased().contains("paris"), "\(reply.text)")
    }

    @Test func aCallParses() async throws {
        let tool: [String: any Sendable] = [
            "type": "function",
            "function": [
                "name": "get_weather",
                "description": "The weather in a city.",
                "parameters": [
                    "type": "object",
                    "properties": ["city": ["type": "string"] as [String: any Sendable]] as [String: any Sendable],
                    "required": ["city"],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
        let input = UserInput(
            chat: [.user("What's the weather in Tokyo? Use the tool.")], tools: [tool],
            additionalContext: ["reasoning_effort": "none"])
        let reply = try await run(input, tools: [tool])
        #expect(reply.calls.first?.function.name == "get_weather", "\(reply.text)")
        #expect(reply.calls.first?.function.arguments["city"] == .string("Tokyo"))
        #expect(!reply.text.contains("<|"), "\(reply.text)")
    }

    @Test func thePictureIsReadByItsOwnTower() async throws {
        let harness = try await loaded()
        #expect(await harness.hasOwnVision)
        let answer = try await harness.look(image: Self.solid(red: 1, green: 0, blue: 0), question: "What colour fills this picture? One word.", maxTokens: 16)
        #expect(answer?.lowercased().contains("red") == true, "\(answer ?? "nil")")
    }

    static func solid(red: CGFloat, green: CGFloat, blue: CGFloat) -> CGImage {
        let side = 256
        let context = CGContext(
            data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(red: red, green: green, blue: blue, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        return context.makeImage()!
    }
}
