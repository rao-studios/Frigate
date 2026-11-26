import Foundation
import Testing

@testable import MLXLMCommon

/// Inkling-2's reply frames come out as reasoning, answer and calls: what Sewn's harness and every
/// MLXLMCommon generation read, with no Inkling code of their own (Craft's ReplyFrames, ported).
@Suite("Inkling-2 reply frames")
struct InklingStreamAdapterTests {
    @Test func reasoningAnswerAndCallComeApart() {
        let events = decode(
            "<|message_model|><|content_thinking|>Look it up.<|end_message|>"
                + "<|message_model|><|content_text|>Checking.<|end_message|>"
                + #"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Tokyo","days":2}}<|end_message|>"#
                + "<|content_model_end_sampling|>")
        #expect(reasoning(events) == "Look it up.")
        #expect(response(events) == "Checking.")
        let calls = calls(events)
        #expect(calls.count == 1)
        #expect(calls.first?.function.name == "get_weather")
        #expect(calls.first?.function.arguments["city"] == .string("Tokyo"))
        #expect(calls.first?.function.arguments["days"] == .int(2))
        #expect(!events.contains { if case .protocolError = $0 { true } else { false } })
    }

    @Test func noMarkerReachesTheAnswer() {
        let events = decode("<|message_model|><|content_text|>Hello there.<|end_message|><|content_model_end_sampling|>")
        #expect(response(events) == "Hello there.")
        #expect(!response(events).contains("<|"))
    }

    @Test func aMarkerInsideACallsStringIsWhatTheCallWrites() {
        let events = decode(
            #"<|message_model|>write<|content_invoke_tool_json|>{"name":"write","args":{"text":"a <|end_message|> b"}}<|end_message|>"#)
        let calls = calls(events)
        #expect(calls.count == 1)
        #expect(calls.first?.function.arguments["text"] == .string("a <|end_message|> b"))
    }

    @Test func jsonThatDoesNotParseIsARejectedCallNotText() {
        let events = decode(#"<|message_model|>write<|content_invoke_tool_json|>{"name":"write","args":{<|end_message|>"#)
        #expect(calls(events).isEmpty)
        #expect(response(events).isEmpty)
        #expect(events.contains { if case .rejectedToolCall = $0 { true } else { false } })
    }

    @Test func textOutsideAFrameAndUnknownKindsPassThrough() {
        let events = decode("plain <|message_model|><|content_xml|><a/><|end_message|>")
        #expect(response(events) == "plain <a/>")
    }

    @Test func anOpenAnswerFlushesAtTheEnd() {
        let events = decode("<|message_model|><|content_text|>cut short")
        #expect(response(events) == "cut short")
    }

    @Test func theFormatIsNamedAndOwnsItsProtocol() {
        #expect(ToolCallFormat(toolParserType: "inkling") == .inkling)
        let parser = ToolCallFormat.inkling.createParser()
        let call = parser.parse(
            content: #"<|content_invoke_tool_json|>{"name":"ls","args":{"path":"."}}<|end_message|>"#, tools: nil)
        #expect(call?.function.name == "ls")
        #expect(call?.function.arguments["path"] == .string("."))
    }

    // MARK: - Helpers

    private func decode(_ text: String) -> [TokenStreamEvent] {
        let tokenizer = FrameTokenizer()
        var decoder = ToolCallFormat.inkling.makeTokenStreamDecoder(
            tokenizer: tokenizer, tools: nil, stopStrings: [])
        #expect(decoder is InklingStreamAdapter)
        var events: [TokenStreamEvent] = []
        for token in tokenizer.encode(text: text, addSpecialTokens: false) {
            if !decoder.push(token, emit: { events.append($0); return true }) { break }
        }
        _ = decoder.finish { events.append($0); return true }
        return events
    }

    private func reasoning(_ events: [TokenStreamEvent]) -> String {
        events.map { if case .reasoning(let text) = $0 { text } else { "" } }.joined()
    }

    private func response(_ events: [TokenStreamEvent]) -> String {
        events.map { if case .response(let text) = $0 { text } else { "" } }.joined()
    }

    private func calls(_ events: [TokenStreamEvent]) -> [ToolCall] {
        events.compactMap { if case .toolCall(let call) = $0 { call } else { nil } }
    }
}

/// Inkling's frame markers as single special tokens; every other character its own token.
private struct FrameTokenizer: Tokenizer {
    static let markers: [String: Int] = [
        "<|message_model|>": 200001, "<|content_text|>": 200004, "<|content_model_end_sampling|>": 200006,
        "<|content_thinking|>": 200008, "<|end_message|>": 200010, "<|content_xml|>": 200024,
        "<|content_invoke_tool_json|>": 200049,
    ]

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var ids: [Int] = []
        var rest = Substring(text)
        outer: while !rest.isEmpty {
            for (marker, id) in Self.markers where rest.hasPrefix(marker) {
                // Every marker becomes its special token, inside a call's string too: the open-string rule is under test.
                ids.append(id)
                rest = rest.dropFirst(marker.count)
                continue outer
            }
            ids.append(Int(rest.unicodeScalars.first!.value))
            rest = rest.dropFirst()
        }
        return ids
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { id in
            if let marker = Self.markers.first(where: { $0.value == id })?.key {
                return skipSpecialTokens ? "" : marker
            }
            return String(UnicodeScalar(UInt32(id)).map(Character.init) ?? "?")
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { Self.markers[token] }
    func convertIdToToken(_ id: Int) -> String? { decode(tokenIds: [id], skipSpecialTokens: false) }
    var bosToken: String? { nil }
    var eosToken: String? { "<|content_model_end_sampling|>" }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}
