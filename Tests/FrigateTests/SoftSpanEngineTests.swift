//  WHAT: Soft spans through `ParallelEngine` (FrigateParallel/ParallelEngine.swift, `forward`): a prompt whose middle is
//        `embed(ids)` carried as a span decodes what the token prompt decodes; a slot that read one span reuses it and
//        does not reuse another span at the same place (the placeholders carry the digest); and admission refuses a
//        span of the wrong width, one at the prompt's end, and placeholders with no span.

import Foundation
import MLX
import Testing

@testable import FrigateParallel
@testable import MLXLLM
@testable import MLXLMCommon

private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

@Suite("Soft spans through the engine", .enabled(if: mlxFunctional), .serialized)
struct SoftSpanEngineTests {
    private let ids: [Int] = (0 ..< 24).map { ($0 * 37) % 61 + 1 }

    private struct Run { var tokens: [Int] = []; var reused = -1; var finish: FinishInfo? }

    private func run(_ engine: ParallelEngine, _ tokens: [Int], soft: [SoftSpan] = [], slot: SlotID? = nil) async throws -> Run {
        var run = Run()
        let request = StreamRequest(tokens: tokens, sampling: GenerateParameters(temperature: 0), maxTokens: 8,
                                    stopTokens: [], slot: slot, soft: soft)
        for try await event in engine.insert(request).events {
            switch event {
            case .admitted(let reused, _, _, _): run.reused = reused
            case .token(let token): run.tokens.append(token)
            case .finished(let info): run.finish = info
            }
        }
        return run
    }

    /// `ids` with positions `range` carried as their own embeddings.
    private func spliced(_ model: Qwen35TextModel, _ range: Range<Int>, nudge: Float = 0) -> ([Int], SoftSpan) {
        let rows = model.embed(MLXArray(ids[range].map(Int32.init)).reshaped(1, range.count)).asType(.float32) + nudge
        let span = SoftSpan(start: range.lowerBound, vectors: rows.asArray(Float.self), width: model.hiddenSize)
        var tokens = ids
        tokens.replaceSubrange(range, with: span.placeholders)
        return (tokens, span)
    }

    @Test func aSpanOfItsOwnEmbeddingsDecodesAsTheTokensDo() async throws {
        let model = try SoftPromptTests.qwen35()
        let (tokens, span) = spliced(model, 5 ..< 12)
        var configuration = ParallelEngine.Configuration()
        configuration.prefillChunk = 8  // chunk edges fall inside the span and around it
        let engine = try ParallelEngine(model: model, configuration: configuration)
        defer { engine.shutdown() }
        let plain = try await run(engine, ids)
        let soft = try await run(engine, tokens, soft: [span])
        #expect(plain.tokens.count == 8)
        #expect(soft.finish?.reason == .maxTokens, "\(soft.finish?.message ?? "")")
        #expect(soft.tokens == plain.tokens)
    }

    @Test func aSlotReusesTheSpanItReadAndNotAnother() async throws {
        let model = try SoftPromptTests.qwen35()
        let engine = try ParallelEngine(model: model)
        defer { engine.shutdown() }
        let slot = SlotID()
        let (tokens, span) = spliced(model, 5 ..< 12)
        _ = try await run(engine, tokens, soft: [span], slot: slot)
        let again = try await run(engine, tokens + [3], soft: [span], slot: slot)
        #expect(again.reused >= tokens.count, "the slot holds the span it read")

        let (other, otherSpan) = spliced(model, 5 ..< 12, nudge: 0.01)
        let changed = try await run(engine, other + [3], soft: [otherSpan], slot: slot)
        #expect(changed.reused <= 5, "another span at the same place is read again")
        #expect(changed.finish?.reason == .maxTokens)
    }

    @Test func admissionRefusesWhatItCannotRead() async throws {
        let model = try SoftPromptTests.qwen35()
        let engine = try ParallelEngine(model: model)
        defer { engine.shutdown() }
        let (tokens, span) = spliced(model, 5 ..< 12)

        let narrow = SoftSpan(start: 5, vectors: [Float](repeating: 0, count: 7 * 16), width: 16)
        var narrowTokens = ids
        narrowTokens.replaceSubrange(5 ..< 12, with: narrow.placeholders)
        let wrongWidth = try await run(engine, narrowTokens, soft: [narrow])
        #expect(wrongWidth.finish?.reason == .failed)
        #expect(wrongWidth.finish?.message?.contains("wide") == true)

        let atTheEnd = SoftSpan(start: ids.count - 7, vectors: span.vectors, width: span.width)
        let endTokens = Array(ids.prefix(ids.count - 7)) + atTheEnd.placeholders
        let ending = try await run(engine, endTokens, soft: [atTheEnd])
        #expect(ending.finish?.reason == .failed)

        let orphan = try await run(engine, tokens)
        #expect(orphan.finish?.reason == .failed)
        #expect(orphan.finish?.message?.contains("no soft span") == true)
    }

    /// `borrow`: a gradient step through the engine's own model, on its thread, between streams.
    @Test func theEngineLendsItsModelBetweenStreams() async throws {
        let model = try SoftPromptTests.qwen35()
        let engine = try ParallelEngine(model: model)
        defer { engine.shutdown() }
        let before = try await run(engine, ids)
        let ids = self.ids
        let gradient: Float = try await engine.borrow { lent in
            let soft = lent as! any SoftPromptModel
            // A task-local does not cross into the engine's thread: the differentiable path is set inside the call.
            return GatedDeltaPath.$differentiable.withValue(true) {
                let embedded = soft.embed(MLXArray(ids.map(Int32.init)).reshaped(1, ids.count))
                let g = grad({ (e: MLXArray) -> MLXArray in soft(embeddings: e, cache: nil).logSumExp(axis: -1).sum() })(embedded)
                return abs(g).sum().item(Float.self)
            }
        }
        #expect(gradient.isFinite && gradient > 0)
        let after = try await run(engine, ids)
        #expect(after.tokens == before.tokens, "a borrowed step leaves the model as it was")

        engine.shutdown()
        await #expect(throws: ParallelError.self) { _ = try await engine.borrow { _ in 1 } }
    }
}
