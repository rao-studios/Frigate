//  WHAT: The fused decode on the published checkpoint: a 2,500-token prompt read into a cache (past the 1,024-slot ring
//        and the 1,024-distance bias), 32 tokens decoded by the plain ops (compile off), then the same 32 tokens fed
//        through the fused paths — the compiled chains and decode convolutions (Inkling2Fused,
//        Inkling2DecodeConvolution), then the decode attention as well (Inkling2DecodeAttention) — their logits step
//        by step. With FRIGATE_INKLING2 naming the checkpoint's folder.
//  PIN:  THE ATTENTION IS HELD TO THE MODEL'S OWN NOISE, NOT TO EQUALITY. Any change of rounding in a decoded token's
//        attention grows through the caches it writes and 48 visits: the plain path's one-row SDPA and its float32
//        branch, both exact in intent, part to a logit diff of 6.2 and a cosine of 0.957 by step 28 of this run,
//        32/32 top-1; the kernel against either is the same (2026-11-21). So the first step, on a cache both read
//        alike, is held to a rounding, and the run to its top-1 choices. The compiled chains and convolutions are held
//        to equality, bit for bit.

import Foundation
import MLX
import MLXLMCommon
import Testing

@testable import MLXLLM

private var checkpoint: URL? {
    guard ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1",
          let path = ProcessInfo.processInfo.environment["FRIGATE_INKLING2"] else { return nil }
    return URL(fileURLWithPath: path)
}

@Suite("Inkling-2 fused decode, on the checkpoint", .enabled(if: checkpoint != nil), .serialized)
struct Inkling2FusedDecodeLiveTests {
    /// The prompt read in chunks, then `steps` tokens: greedy, or the `forced` ones. `fused` turns on the compiled
    /// chains and decode convolutions, `attention` the decode attention. Returns the tokens and each step's logits.
    func decode(_ model: Inkling2Model, prompt: [Int], steps: Int, forced: [Int]? = nil, fused: Bool, attention: Bool)
        -> (tokens: [Int], logits: [MLXArray])
    {
        model.fuseDecode(attention: attention, convolutions: fused)
        compile(enable: fused)
        defer {
            model.fuseDecode(attention: true, convolutions: true)
            compile(enable: true)
        }
        let cache = model.newCache(parameters: nil)
        var start = 0
        while start < prompt.count - 1 {
            let end = min(start + 512, prompt.count - 1)
            _ = model(MLXArray(prompt[start ..< end].map(Int32.init)).reshaped(1, -1), cache: cache)
            eval(cache.flatMap { $0.innerState() })
            start = end
        }
        var next = prompt.last!, tokens: [Int] = [], logits: [MLXArray] = []
        for i in 0 ..< steps {
            let step = model(MLXArray([Int32(next)]).reshaped(1, 1), cache: cache)[0, -1].asType(.float32)
            eval(step)
            logits.append(step)
            tokens.append(argMax(step).item(Int.self))
            next = forced?[i] ?? tokens.last!
        }
        return (tokens, logits)
    }

    /// Top-1 choices alike, the largest logit diff, the lowest cosine, and the first step's diff.
    func compare(_ got: [MLXArray], _ want: [MLXArray]) -> (top: Int, diff: Float, cosine: Float, first: Float) {
        var top = 0, diff: Float = 0, cosine: Float = 1
        for (a, b) in zip(got, want) {
            if argMax(a).item(Int.self) == argMax(b).item(Int.self) { top += 1 }
            diff = max(diff, abs(a - b).max().item(Float.self))
            cosine = min(cosine, ((a * b).sum() / (sqrt((a * a).sum()) * sqrt((b * b).sum()))).item(Float.self))
        }
        return (top, diff, cosine, abs(got[0] - want[0]).max().item(Float.self))
    }

    @Test func decodesAsThePlainOpsDid() throws {
        let data = try Data(contentsOf: checkpoint!.appendingPathComponent("config.json"))
        let model = Inkling2Model(try JSONDecoder().decode(Inkling2Configuration.self, from: data))
        try loadWeights(modelDirectory: checkpoint!, model: model)
        let prompt = (0 ..< 2_500).map { 1_000 + ($0 * 7_919) % 60_000 }
        let plain = decode(model, prompt: prompt, steps: 32, fused: false, attention: false)

        let chains = decode(model, prompt: prompt, steps: 32, forced: plain.tokens, fused: true, attention: false)
        let c = compare(chains.logits, plain.logits)
        print("[inkling2] compiled chains + decode convolutions: \(c.top)/32 top-1, max diff \(c.diff)")
        #expect(c.diff == 0, "the same operations in the same order and dtypes")

        let all = decode(model, prompt: prompt, steps: 32, forced: plain.tokens, fused: true, attention: true)
        let a = compare(all.logits, plain.logits)
        print("[inkling2] all fused: \(a.top)/32 top-1, first step diff \(a.first), max diff \(a.diff), cosine \(a.cosine)")
        #expect(a.first < 0.25, "one rounding on a cache both read alike")
        #expect(a.top >= 30)
    }
}
