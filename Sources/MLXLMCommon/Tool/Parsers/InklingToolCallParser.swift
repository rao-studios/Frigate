// Copyright © 2026 Rao Studios LLC.

import Foundation

/// Parser for an Inkling-2 call frame's payload:
/// `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{…}}<|end_message|>`.
///
/// The payload is the hosted Thinking Machines shape, `args` where the JSON format has
/// `arguments`; it is renamed and handed to the JSON parser, so typed conversion is shared.
/// The streaming path (`InklingStreamAdapter`) finds the frame by token id and passes only
/// the payload; `parse(content:)` also accepts the text form with its markers.
public struct InklingToolCallParser: ToolCallParser, Sendable {
    public static let callMarker = "<|content_invoke_tool_json|>"
    public static let endMarker = "<|end_message|>"

    public var startTag: String? { Self.callMarker }
    public var endTag: String? { Self.endMarker }

    private let json = JSONToolCallParser(startTag: "<tool_call>", endTag: "</tool_call>")

    public init() {}

    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        var text = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if !text.hasPrefix("{"), let start = text.range(of: Self.callMarker) {
            text = String(text[start.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        if text.hasSuffix(Self.endMarker) { text.removeLast(Self.endMarker.count) }
        return parsePayload(text)
    }

    /// The JSON between the call marker and the frame's end.
    func parsePayload(_ payload: String) -> ToolCall? {
        guard let data = payload.trimmingCharacters(in: .whitespacesAndNewlines).data(using: .utf8),
            var object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        if object["arguments"] == nil, let args = object.removeValue(forKey: "args") {
            object["arguments"] = args
        }
        if object["arguments"] == nil { object["arguments"] = [String: Any]() }
        guard let normalized = try? JSONSerialization.data(withJSONObject: object),
            let text = String(data: normalized, encoding: .utf8)
        else { return nil }
        return json.parsePayload(text)
    }
}
