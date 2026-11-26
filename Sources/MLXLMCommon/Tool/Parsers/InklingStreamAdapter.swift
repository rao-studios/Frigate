// Copyright © 2026 Rao Studios LLC.
//
//  WHAT: The token-stream decoder for Inkling-2's reply frames. Inkling writes every reply
//        as frames, opened by the model itself (its template has no generation prompt):
//
//            <|message_model|><|content_thinking|>REASONING<|end_message|>
//            <|message_model|><|content_text|>ANSWER<|end_message|>
//            <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{…}}<|end_message|>
//            <|content_model_end_sampling|>
//
//        which come out as `.reasoning`, `.response` and `.toolCall` events.
//  PIN:  MARKERS ARE TOKEN IDS. Every marker above is one special token, so frames are found
//        by id: marker-shaped text the model writes inside a payload is never a boundary, and
//        a marker can't be cut across two chunks.
//  PIN:  A MARKER INSIDE A CALL'S STRING IS WHAT THE CALL WRITES. While the call's JSON has a
//        string open, `<|end_message|>` and `<|message_model|>` are content; the call closes at
//        the first end marker after its strings have (Craft's ReplyFrames, the same rule).
//  PIN:  NOTHING THE MODEL WROTE IS LOST. A content kind this table doesn't route, text outside
//        any frame, or a header too long to be a name goes out as answer text. JSON that doesn't
//        parse is a rejected call, never answer text.
//

private struct InklingControls {
    var message: Int
    var thinking: Int
    var text: Int
    var call: Int
    var end: Int
    var endSampling: Int
    /// Content kinds Inkling's vocabulary has that a reply doesn't route (`<|content_xml|>`,
    /// `<|content_invoke_tool_text|>`, …): their payload passes through as text.
    var otherContent: Set<Int>

    init?(tokenizer: any Tokenizer) {
        guard let message = tokenizer.convertTokenToId("<|message_model|>"),
            let thinking = tokenizer.convertTokenToId("<|content_thinking|>"),
            let text = tokenizer.convertTokenToId("<|content_text|>"),
            let call = tokenizer.convertTokenToId(InklingToolCallParser.callMarker),
            let end = tokenizer.convertTokenToId(InklingToolCallParser.endMarker),
            let endSampling = tokenizer.convertTokenToId("<|content_model_end_sampling|>"),
            Set([message, thinking, text, call, end, endSampling]).count == 6
        else { return nil }
        self.message = message
        self.thinking = thinking
        self.text = text
        self.call = call
        self.end = end
        self.endSampling = endSampling
        self.otherContent = Set(
            ["<|content_invoke_tool_text|>", "<|content_xml|>", "<|content_image|>", "<|content_tool_error|>"]
                .compactMap(tokenizer.convertTokenToId))
    }
}

private struct InklingProtocolDecoder {
    private enum State {
        case between
        case header(tokens: [Int])
        case reasoning
        case text
        case call(name: String, tokens: [Int])
    }

    private let tokenizer: any Tokenizer
    private let controls: InklingControls
    private let tools: [[String: any Sendable]]?
    private let validationPolicy: ToolCallValidationPolicy
    private let parser = InklingToolCallParser()
    private let maximumHeaderTokens = 64
    private var state: State = .between
    private var reasoningDetokenizer: any StreamingDetokenizer
    private var responseDetokenizer: any StreamingDetokenizer
    private(set) var endsInVisibleText = false

    var isInsideReasoning: Bool {
        if case .reasoning = state { return true }
        return false
    }

    var endSamplingToken: Int { controls.endSampling }

    init?(tokenizer: any Tokenizer, tools: [[String: any Sendable]]?, toolCallPolicy: ToolCallPolicy) {
        guard let controls = InklingControls(tokenizer: tokenizer) else { return nil }
        self.tokenizer = tokenizer
        self.controls = controls
        self.tools = tools
        self.validationPolicy = toolCallPolicy.validation
        self.reasoningDetokenizer = tokenizer.makeStreamingDetokenizer()
        self.responseDetokenizer = tokenizer.makeStreamingDetokenizer()
    }

    mutating func push(_ token: Int) -> [TokenStreamEvent] {
        endsInVisibleText = false
        if token == controls.endSampling { return finish() }

        switch state {
        case .between:
            if token == controls.message {
                state = .header(tokens: [])
                return []
            }
            if token == controls.end { return [] }
            // Text outside a frame: the answer, as written.
            return response(token)

        case .header(var tokens):
            if token == controls.thinking {
                state = .reasoning
                return []
            }
            if token == controls.text || controls.otherContent.contains(token) {
                state = .text
                return []
            }
            if token == controls.call {
                let name = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                state = .call(name: name, tokens: [])
                return []
            }
            if token == controls.end {
                state = .between
                return []
            }
            if token == controls.message {
                state = .header(tokens: [])
                return []
            }
            tokens.append(token)
            guard tokens.count <= maximumHeaderTokens else {
                // Not a name: what the model wrote is answer text.
                state = .text
                var events: [TokenStreamEvent] = []
                for held in tokens { events += response(held) }
                return events
            }
            state = .header(tokens: tokens)
            return []

        case .reasoning:
            if token == controls.end || token == controls.message {
                let events = flushReasoning()
                state = token == controls.message ? .header(tokens: []) : .between
                return events
            }
            reasoningDetokenizer.append(token: token)
            if let text = reasoningDetokenizer.next(), !text.isEmpty {
                endsInVisibleText = text.last?.isWhitespace == false
                return [.reasoning(text)]
            }
            return []

        case .text:
            if token == controls.end || token == controls.message {
                let events = flushResponse()
                state = token == controls.message ? .header(tokens: []) : .between
                return events
            }
            return response(token)

        case .call(let name, var tokens):
            if token == controls.end || token == controls.message {
                let payload = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
                if Self.hasOpenString(payload) {
                    // A marker inside one of the call's strings is what the call writes.
                    tokens.append(token)
                    state = .call(name: name, tokens: tokens)
                    return []
                }
                state = token == controls.message ? .header(tokens: []) : .between
                return [callEvent(name: name, payload: payload)]
            }
            tokens.append(token)
            state = .call(name: name, tokens: tokens)
            return []
        }
    }

    /// At the end of sampling (or of generation): what an open frame still holds.
    mutating func finish() -> [TokenStreamEvent] {
        defer { state = .between }
        switch state {
        case .reasoning:
            return flushReasoning()
        case .text:
            return flushResponse()
        case .call(let name, let tokens):
            let payload = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
            return [callEvent(name: name, payload: payload)]
        case .header(let tokens) where !tokens.isEmpty:
            var events: [TokenStreamEvent] = []
            for held in tokens { events += response(held) }
            return events + flushResponse()
        case .header, .between:
            return flushResponse()
        }
    }

    private mutating func response(_ token: Int) -> [TokenStreamEvent] {
        responseDetokenizer.append(token: token)
        if let text = responseDetokenizer.next(), !text.isEmpty {
            endsInVisibleText = text.last?.isWhitespace == false
            return [.response(text)]
        }
        return []
    }

    private mutating func flushReasoning() -> [TokenStreamEvent] {
        let text = reasoningDetokenizer.finish()
        reasoningDetokenizer = tokenizer.makeStreamingDetokenizer()
        return text.map { [.reasoning($0)] } ?? []
    }

    private mutating func flushResponse() -> [TokenStreamEvent] {
        let text = responseDetokenizer.finish()
        responseDetokenizer = tokenizer.makeStreamingDetokenizer()
        return text.map { [.response($0)] } ?? []
    }

    private func callEvent(name: String, payload: String) -> TokenStreamEvent {
        guard var call = parser.parsePayload(payload) else {
            return .rejectedToolCall(
                RejectedToolCall(
                    reason: .malformedSyntax, format: .inkling, toolName: name.isEmpty ? nil : name,
                    rawText: payload, detail: "the call's JSON does not parse"))
        }
        call = ToolArgumentNormalization.normalize(call, tools: tools)
        if validationPolicy == .strict,
            case .invalid(let violations) = ToolSchemaValidator.validate(
                arguments: call.function.arguments, forToolNamed: call.function.name, in: tools)
        {
            return .rejectedToolCall(
                RejectedToolCall(
                    reason: .invalidArguments, format: .inkling, toolName: call.function.name,
                    rawText: payload, detail: ToolSchemaValidator.describe(violations)))
        }
        return .toolCall(ToolCall(function: call.function, id: ToolCallFormat.inkling.generateToolCallID()))
    }

    /// Whether `json` ends inside a string literal.
    static func hasOpenString(_ json: String) -> Bool {
        var inString = false
        var escaped = false
        for character in json {
            if escaped {
                escaped = false
            } else if inString && character == "\\" {
                escaped = true
            } else if character == "\"" {
                inString.toggle()
            }
        }
        return inString
    }
}

/// Token-stream decoder for Inkling-2's reply frames and its JSON tool calls.
struct InklingStreamAdapter: TokenStreamDecoder {
    private var protocolDecoder: InklingProtocolDecoder
    private var stopStringFilter: StopStringFilter
    private(set) var rejectedToolCallCount = 0
    let additionalStopTokenIDs: Set<Int>
    let receivesStopTokens = true

    var isInsideReasoning: Bool { protocolDecoder.isInsideReasoning }

    var canEndForSteering: Bool {
        protocolDecoder.endsInVisibleText && stopStringFilter.buffer.isEmpty
    }

    init?(
        tokenizer: any Tokenizer,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>,
        toolCallPolicy: ToolCallPolicy = .init()
    ) {
        guard
            let decoder = InklingProtocolDecoder(
                tokenizer: tokenizer, tools: tools, toolCallPolicy: toolCallPolicy)
        else { return nil }
        self.protocolDecoder = decoder
        self.stopStringFilter = StopStringFilter(stopStrings: stopStrings)
        self.additionalStopTokenIDs = [decoder.endSamplingToken]
    }

    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        for event in protocolDecoder.push(token) where !emitEvent(event, emit: emit) { return false }
        return true
    }

    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool {
        for event in protocolDecoder.finish() where !emitEvent(event, emit: emit) { return false }
        if let text = stopStringFilter.finish() { return emit(.response(text)) }
        return true
    }

    private mutating func emitEvent(_ event: TokenStreamEvent, emit: (TokenStreamEvent) -> Bool) -> Bool {
        switch event {
        case .reasoning(let text):
            return emit(.reasoning(text))
        case .response(let text):
            let result = stopStringFilter.process(text)
            if let response = result.text, !emit(.response(response)) { return false }
            if result.stopped {
                _ = emit(.stop)
                return false
            }
            return true
        case .toolCall(let call):
            if let response = stopStringFilter.finish(), !emit(.response(response)) { return false }
            return emit(.toolCall(call))
        case .rejectedToolCall(let rejection):
            rejectedToolCallCount += 1
            if let response = stopStringFilter.finish(), !emit(.response(response)) { return false }
            return emit(.rejectedToolCall(rejection))
        case .protocolError(let message):
            return emit(.protocolError(message))
        case .stop:
            return emit(.stop)
        }
    }
}
