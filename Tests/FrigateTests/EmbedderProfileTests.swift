//
//  EmbedderProfileTests.swift
//  FrigateTests
//
//  WHAT: How FrigateEmbedder names a model and calls it — model specs, the known
//        voyage-4-nano profile, prompts per role, and the special-token frame that
//        keeps a truncated text's trailing token (the Qwen3-Embedding fix).
//  HOW:  Pure values; no MLX, no network.
//

import Foundation
import Frigate
import Testing

@Suite("FrigateEmbedder profiles")
struct EmbedderProfileTests {

    typealias Profile = FrigateEmbedder.Profile
    typealias TokenFrame = FrigateEmbedder.TokenFrame

    @Test func theKnownRepoResolvesToVoyage() {
        let profile = Profile.resolve(Profile.voyage4NanoRepo)
        #expect(profile == .voyage4Nano)
        #expect(profile.outputDimension == 1024)
        #expect(profile.vectorSpace == "voyage-4@1024")
        #expect(profile.prompt(for: .query) == "Represent the query for retrieving supporting documents: ")
        #expect(profile.prompt(for: .document) == "Represent the document for retrieval: ")
    }

    @Test func anAtRevisionOverridesThePinAndKeepsTheRest() {
        let sha = String(repeating: "ab", count: 20)
        let profile = Profile.resolve("\(Profile.voyage4NanoRepo)@\(sha)")
        #expect(profile.revision == sha)
        #expect(profile.model == Profile.voyage4NanoRepo)
        #expect(profile.queryPrompt == Profile.voyage4Nano.queryPrompt)
        #expect(profile.vectorSpace == "voyage-4@1024")
    }

    @Test func anUnknownRepoIsBare() {
        let profile = Profile.resolve("mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
        #expect(profile.source == .hub(id: "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ", revision: nil))
        #expect(profile.queryPrompt == nil && profile.documentPrompt == nil)
        #expect(profile.outputDimension == nil)
        #expect(profile.maxTokens == 512)
    }

    @Test func aPathIsADirectory() {
        let profile = Profile.resolve("/tmp/voyage")
        #expect(profile.source == .directory(URL(fileURLWithPath: "/tmp/voyage")))
        let home = Profile.resolve("~/models/voyage")
        #expect(home.model == (("~/models/voyage") as NSString).expandingTildeInPath)
    }

    @Test func modelIdInitResolvesTheSameWay() {
        let embedder = FrigateEmbedder(modelId: Profile.voyage4NanoRepo)
        #expect(embedder.profile == .voyage4Nano)
        #expect(embedder.status.phase == .idle)
        #expect(embedder.status.vectorSpace == "voyage-4@1024")
    }

    // MARK: - Token frame

    @Test func theFrameIsWhatSpecialTokensAdd() {
        #expect(TokenFrame.probe(with: [1, 7, 9], without: [7]) == TokenFrame(prefix: [1], suffix: [9]))
        #expect(TokenFrame.probe(with: [7], without: [7]) == TokenFrame())
        #expect(TokenFrame.probe(with: [7, 2], without: [7]) == TokenFrame(prefix: [], suffix: [2]))
        #expect(TokenFrame.probe(with: [1, 8], without: [7]) == nil, "a tokenizer that rewrites the body")
    }

    /// Qwen3-Embedding appends `<|endoftext|>` and pools on it: cutting the wrapped
    /// sequence at 512 dropped it. Cutting the body keeps it.
    @Test func truncationKeepsTheTrailingToken() {
        let eos = 151_643
        let frame = TokenFrame(prefix: [], suffix: [eos])
        let fitted = frame.fit(Array(0..<600), maxTokens: 512)
        #expect(fitted.count == 512)
        #expect(fitted.last == eos)
        #expect(Array(fitted.prefix(3)) == [0, 1, 2])
    }

    @Test func aShortTextIsUntouched() {
        let frame = TokenFrame(prefix: [101], suffix: [102])
        #expect(frame.fit([5, 6, 7], maxTokens: 512) == [101, 5, 6, 7, 102])
        #expect(TokenFrame().fit([5, 6, 7], maxTokens: 2) == [5, 6])
    }

    @Test func aBudgetSmallerThanTheFrameKeepsOnlyTheFrame() {
        let frame = TokenFrame(prefix: [1], suffix: [2])
        #expect(frame.fit([5, 6, 7], maxTokens: 1) == [1, 2])
    }
}
