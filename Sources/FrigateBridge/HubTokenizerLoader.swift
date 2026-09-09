// Concrete `TokenizerLoader` / `Tokenizer` for MLXLMCommon 3.x, backed by the vendored
// swift-transformers `Tokenizers`. See HubDownloader.swift for why this is hand-wired.

import Foundation
import MLXLMCommon
import Tokenizers

/// Loads a swift-transformers tokenizer out of an already-downloaded model directory.
public struct HubTokenizerLoader: TokenizerLoader {
    public init() {}

    public func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        TokenizerAdapter(try await AutoTokenizer.from(modelFolder: directory))
    }
}

/// Adapts a swift-transformers `Tokenizers.Tokenizer` to MLXLMCommon's `Tokenizer`.
///
/// The two protocols line up almost exactly — `Message` and `ToolSpec` are both
/// `[String: any Sendable]` — so this only renames `tokens:` to `tokenIds:`.
struct TokenizerAdapter: MLXLMCommon.Tokenizer {
    let wrapped: any Tokenizers.Tokenizer

    init(_ wrapped: any Tokenizers.Tokenizer) {
        self.wrapped = wrapped
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        wrapped.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        wrapped.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { wrapped.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { wrapped.convertIdToToken(id) }

    var bosToken: String? { wrapped.bosToken }
    var eosToken: String? { wrapped.eosToken }
    var unknownToken: String? { wrapped.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        try wrapped.applyChatTemplate(
            messages: messages, tools: tools, additionalContext: additionalContext)
    }
}
