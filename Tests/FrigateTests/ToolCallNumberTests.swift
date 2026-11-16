import Foundation
import MLXLMCommon
import Testing

/// A tool call's numbers keep their kind. `JSONSerialization` reads a JSON number as an NSNumber, whose 0 and 1 cast
/// to `Bool`, so `JSONValue.from` made every 0 and 1 inside an object or array parameter a boolean: Craft's `$mocks`
/// got `"tone": false` for the model's `"tone": 0`, refused it, and the model wrote the same call again (Seer R13).
@Suite("A tool call's numbers")
struct ToolCallNumberTests {
    @Test func aNumberThatJSONReadKeepsItsKind() throws {
        let text = #"{"tone": 0, "rank": 1, "royalty": 62, "on": true, "off": false, "cost": 0.14, "amount": 8.0, "none": null, "tones": [0, 1, true]}"#
        let read = try JSONSerialization.jsonObject(with: Data(text.utf8))
        #expect(JSONValue.from(read) == .object([
            "tone": .int(0), "rank": .int(1), "royalty": .int(62), "on": .bool(true), "off": .bool(false),
            "cost": .double(0.14), "amount": .int(8), "none": .null, "tones": .array([.int(0), .int(1), .bool(true)]),
        ]))
    }

    @Test func aSwiftValueKeepsItsKind() {
        let values: [(Any, JSONValue)] = [(true, .bool(true)), (false, .bool(false)), (0, .int(0)), (1, .int(1)), (0.5, .double(0.5)), ("0", .string("0"))]
        for (value, expected) in values {
            #expect(JSONValue.from(value) == expected, "\(value)")
        }
    }

    @Test func anXMLCallsObjectParameterKeepsItsZerosAndOnes() {
        let tools: [[String: any Sendable]] = [[
            "type": "function",
            "function": [
                "name": "mocks",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "records": ["type": "object"] as [String: any Sendable],
                        "seed": ["type": "integer"] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]]
        let processor = ToolCallProcessor(format: .xmlFunction, tools: tools)
        let call = """
            <tool_call>
            <function=mocks>
            <parameter=records>
            {"Owner": [{"name": "Ada", "tone": 0, "royalty": 1, "isAccount": true}]}
            </parameter>
            <parameter=seed>
            1
            </parameter>
            </function>
            </tool_call>
            """
        _ = processor.processChunk(call)
        processor.processEOS()
        let arguments = processor.toolCalls.first?.function.arguments
        #expect(arguments?["records"] == .object(["Owner": .array([.object(["name": .string("Ada"), "tone": .int(0), "royalty": .int(1), "isAccount": .bool(true)])])]))
        #expect(arguments?["seed"] == .int(1))
    }
}
