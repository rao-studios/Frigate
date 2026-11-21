//  WHAT: The Inkling-2 port held to mlx-lm's (PR 2086) on the published checkpoint: a 96-token forward, a forward
//        long enough to wrap the 1,024-token window, and the same prompt read in 512-token chunks into a cache and
//        16 tokens decoded greedily. With FRIGATE_INKLING2 naming the checkpoint's folder and FRIGATE_INKLING2_GOLDEN
//        the reference's safetensors (written by the scratch golden.py beside the mirror work).
//  PIN:  EQUAL TO THE LAST BIT, EXCEPT THE RING'S ORDER. With mlx-lm's three Metal kernels ported (Inkling2Kernels) the
//        forwards match the reference exactly (measured 2026-11-20: 96/96 and 2,049/2,049, max diff 0.0). Decoding
//        matches exactly too when the prompt's read ends at a multiple of the window (2,048): mlx-lm's RotatingKVCache
//        then holds each position in the slot this port gives it, `p mod W`. Elsewhere the same keys sit in another
//        order, the fused attention sums them in that order, and BF16 rounds apart (1,833 tokens: every token equal,
//        max diff 1.55, cosine 0.99992). The bounds below admit that and nothing coarser; a port without the kernels
//        misses them (mlx-lm's own plain path agrees with its kernels on 89% of a long prompt's greedy choices).

import Foundation
import FrigateParallel
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

private var inklingFolder: URL? {
    guard ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1",
          let path = ProcessInfo.processInfo.environment["FRIGATE_INKLING2"] else { return nil }
    return URL(fileURLWithPath: path)
}

private var goldenFile: URL? {
    ProcessInfo.processInfo.environment["FRIGATE_INKLING2_GOLDEN"].map { URL(fileURLWithPath: $0) }
}

@Suite("Inkling-2, against mlx-lm", .enabled(if: inklingFolder != nil && goldenFile != nil), .serialized)
struct Inkling2LiveTests {
    /// Loaded once for the suite: 19.5 GB.
    nonisolated(unsafe) static let shared: Result<Inkling2Model, Error> = Result {
        let data = try Data(contentsOf: inklingFolder!.appendingPathComponent("config.json"))
        let model = Inkling2Model(try JSONDecoder().decode(Inkling2Configuration.self, from: data))
        try loadWeights(modelDirectory: inklingFolder!, model: model)
        return model
    }

    static func load() throws -> Inkling2Model { try shared.get() }

    /// Reads `tokens` into `cache` in chunks; the last token's logits.
    static func read(_ model: Inkling2Model, _ tokens: ArraySlice<Int>, into cache: [KVCache], chunk: Int = 512) -> MLXArray {
        var start = tokens.startIndex
        var last = MLXArray(0)
        while start < tokens.endIndex {
            let end = min(start + chunk, tokens.endIndex)
            last = model(MLXArray(tokens[start ..< end].map(Int32.init)).reshaped(1, -1), cache: cache)[0, -1]
            eval(last)
            start = end
        }
        return last
    }

    /// One stream through the engine to its end: the tokens it sampled and how much of its prompt it found kept.
    static func run(_ engine: ParallelEngine, _ request: StreamRequest) async throws -> (tokens: [Int], reused: Int) {
        var tokens: [Int] = [], reused = 0
        for try await event in engine.insert(request).events {
            switch event {
            case .admitted(let prefix, _, _, _): reused = prefix
            case .token(let token): tokens.append(token)
            case .finished: break
            }
        }
        return (tokens, reused)
    }

    static func ints(_ a: MLXArray) -> [Int] { a.asArray(Int32.self).map(Int.init) }

    /// Max |a − b| and the cosine of the two logit vectors.
    static func compare(_ a: MLXArray, _ b: MLXArray) -> (maxDiff: Float, cosine: Float) {
        let x = a.asType(.float32), y = b.asType(.float32)
        let diff = abs(x - y).max().item(Float.self)
        let cosine = ((x * y).sum() / (sqrt((x * x).sum()) * sqrt((y * y).sum()))).item(Float.self)
        return (diff, cosine)
    }

    @Test func matchesTheReference() throws {
        let golden = try loadArrays(url: goldenFile!)
        let model = try Self.load()

        // A short forward, no cache.
        let short = golden["short_tokens"]!
        let shortLogits = model(short.reshaped(1, -1), cache: nil)[0]
        let shortArgmax = Self.ints(argMax(shortLogits, axis: -1))
        let shortAgree = zip(shortArgmax, Self.ints(golden["short_argmax"]!)).filter { $0 == $1 }.count
        let shortLast = Self.compare(shortLogits[-1], golden["short_last"]!)
        print("[inkling2] short: argmax \(shortAgree)/\(shortArgmax.count), last max diff \(shortLast.maxDiff), cosine \(shortLast.cosine)")

        // A forward across the window, no cache.
        let long = golden["long_tokens"]!
        let longLogits = model(long.reshaped(1, -1), cache: nil)[0]
        let longArgmax = Self.ints(argMax(longLogits, axis: -1))
        let longAgree = zip(longArgmax, Self.ints(golden["long_argmax"]!)).filter { $0 == $1 }.count
        let longLast = Self.compare(longLogits[-1], golden["long_last"]!)
        print("[inkling2] long (\(longArgmax.count)): argmax \(longAgree)/\(longArgmax.count), last max diff \(longLast.maxDiff), cosine \(longLast.cosine)")

        // The same prompt read in chunks into a cache, then greedy decoding.
        let tokens = Self.ints(long)
        let cache = model.newCache(parameters: nil)
        var start = 0
        while start < tokens.count - 1 {
            let end = min(start + 512, tokens.count - 1)
            _ = model(MLXArray(tokens[start ..< end].map(Int32.init)).reshaped(1, -1), cache: cache)
            eval(cache.flatMap { $0.innerState() })
            start = end
        }
        var next = tokens.last!
        var decoded: [Int] = []
        let reference = Self.ints(golden["decode_tokens"]!)
        let referenceLogits = golden["decode_logits"]!
        var worst: (maxDiff: Float, cosine: Float) = (0, 1)
        for step in 0 ..< reference.count {
            let logits = model(MLXArray([Int32(next)]).reshaped(1, 1), cache: cache)[0, -1]
            let c = Self.compare(logits, referenceLogits[step])
            worst = (max(worst.maxDiff, c.maxDiff), min(worst.cosine, c.cosine))
            next = argMax(logits).item(Int.self)
            decoded.append(next)
            // Teacher-forced: the reference's token goes in, so one flip does not derail the rest.
            next = reference[step]
        }
        print("[inkling2] decode: \(decoded) vs \(reference); worst max diff \(worst.maxDiff), cosine \(worst.cosine)")

        #expect(shortAgree == shortArgmax.count && longAgree == longArgmax.count)
        #expect(shortLast.maxDiff < 0.01 && longLast.maxDiff < 0.01)
        #expect(decoded == reference)
        #expect(worst.cosine > 0.9995)
    }

    /// A rewind restores the sliding window's ring as it stood: the same tokens read again after it give the same
    /// logits to the last bit. PIN: the ring once took a prompt chunk by subscript writes, which swap the contents of the
    /// object a snapshot holds by reference, so the chunk after a snapshot rewrote it.
    @Test func aRewindRestoresTheRing() throws {
        let model = try Self.load()
        let tokens = Self.ints(try loadArrays(url: goldenFile!)["long_tokens"]!)
        let cache = CheckpointedCache(layers: model.newCache(parameters: nil))
        _ = Self.read(model, tokens[0 ..< 600], into: cache.layers)
        cache.snapshot(at: 600)
        let first = Self.read(model, tokens[600 ..< 1100], into: cache.layers)
        #expect(cache.position == 1100)
        #expect(cache.rewind(toAtMost: 700) == 600)
        let again = Self.read(model, tokens[600 ..< 1100], into: cache.layers)
        let c = Self.compare(first, again)
        print("[inkling2] rewind: max diff \(c.maxDiff), cosine \(c.cosine); per stream: recurrent \(cache.recurrentBytes) B, KV \(cache.kvBytes) B at \(cache.position) tokens")
        #expect(c.maxDiff == 0)
    }

    /// The engine Craft runs it in: a session's follow-up appends onto its slot, a prompt that diverges past the
    /// anchor rewinds to it, and two rows decoding together say what each says alone.
    /// PIN: WHOLE CHAT PROMPTS, ANSWERED. Each prompt ends where the model's answer starts (golden.py's `chat_*`, the
    /// template applied). Cut inside the person's message, greedy decoding sits on near-ties: mlx-lm itself continued
    /// the same 1,100 tokens four different ways as the chunks it read them in changed (measured 2026-11-20), and the
    /// engine stops a chunk at the anchor, so the anchored round reads in other chunks than the one alone.
    @Test func servesThroughTheEngine() async throws {
        let model = try Self.load()
        let golden = try loadArrays(url: goldenFile!)
        let a = Self.ints(golden["chat_long"]!), b = Self.ints(golden["chat_short"]!), c = Self.ints(golden["chat_branch"]!)
        var configuration = ParallelEngine.Configuration()
        configuration.prefillChunk = 512
        let engine = try ParallelEngine(model: model, configuration: configuration)
        defer { engine.shutdown() }
        func request(_ prompt: [Int], slot: SlotID? = nil, anchorAt: Int? = nil) -> StreamRequest {
            StreamRequest(tokens: prompt, sampling: GenerateParameters(temperature: 0), maxTokens: 12, stopTokens: [],
                          slot: slot, anchorAt: anchorAt)
        }

        // Alone, each on a cache of its own.
        let aloneA = try await Self.run(engine, request(a)).tokens
        let aloneB = try await Self.run(engine, request(b)).tokens
        let aloneC = try await Self.run(engine, request(c)).tokens

        // A session: its first round anchors at 640, its follow-up appends, then a prompt diverging at 739 rewinds to 640.
        let slot = SlotID()
        let round1 = try await Self.run(engine, request(a, slot: slot, anchorAt: 640))
        let followUp = a + round1.tokens + Array(b[b.count - 48 ..< b.count])
        let round2 = try await Self.run(engine, request(followUp, slot: slot))
        let rewound = try await Self.run(engine, request(c, slot: slot))
        print("[inkling2] engine: round 1 \(round1.tokens == aloneA), follow-up reused \(round2.reused)/\(followUp.count), rewound reused \(rewound.reused) \(rewound.tokens == aloneC)")
        #expect(round1.tokens == aloneA)
        #expect(round2.reused >= a.count && round2.tokens.count == 12)
        #expect(rewound.reused == 640 && rewound.tokens == aloneC)

        // Two rows at once.
        async let rowA = Self.run(engine, request(a))
        async let rowB = Self.run(engine, request(b))
        let (batchA, batchB) = try await (rowA.tokens, rowB.tokens)
        print("[inkling2] alone \(aloneA) \(aloneB) \(aloneC)")
        print("[inkling2] batch: \(batchA) vs \(aloneA); \(batchB) vs \(aloneB)")
        #expect(batchA == aloneA && batchB == aloneB)
    }
}
