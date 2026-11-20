//  WHAT: One bridge training step through a real core model, measured: a small projector's vectors spliced in front of
//        a passage, cross-entropy on the passage, the gradient read back through the frozen model to the projector
//        (`SoftPromptModel`, `GatedDeltaPath.differentiable`). Prints seconds and peak memory per passage length, and
//        checks the gradient is finite and non-zero. What a trainer's "Check this model" asks before training.
//  IN:   FRIGATE_CORE_MODEL, a model folder (e.g. Qwopus3.5-122B-A10B 4-bit); FRIGATE_CORE_LENGTHS, comma-separated
//        passage lengths (default 256,512,1024).
//  PIN:  MEASURED 2026-11-20, M4 Max 128 GB, Qwopus3.5-122B-A10B 4-bit wired (65.5 GB): a step of 128 / 256 / 512 / 1,024
//        tokens took 2.8 / 3.2 / 4.9 / 9.1 s and peaked at 72.1 / 77.0 / 86.9 / 106.1 GB: about 40 MB a token for the
//        backward pass. Unwired, the 1,024 step pushed 36 GB of the process to swap and crawled: wire before loading.

import Foundation
import MLX
import MLXNN
import Testing

@testable import FrigateBridge
@testable import MLXLLM
@testable import MLXLMCommon

private let coreModel = ProcessInfo.processInfo.environment["FRIGATE_CORE_MODEL"]

@Suite("A bridge step through a real core model", .enabled(if: coreModel != nil), .serialized)
struct CoreModelBridgeStepTests {
    @Test func oneStepFitsAndReachesTheProjector() async throws {
        // Wired as Craft wires it (ResidentWeights), so macOS cannot page the weights out under the step's memory.
        if let bytes = GPU.maxRecommendedWorkingSetBytes() {
            _ = await WiredMemoryTicket(size: bytes, policy: Held(bytes: bytes)).start()
            print("wired limit \(bytes >> 20) MB")
        }
        Memory.cacheLimit = 2 << 30
        let context = try await loadModel(from: URL(fileURLWithPath: coreModel!), using: HubTokenizerLoader())
        let core = try #require(context.model as? any SoftPromptModel, "\(type(of: context.model)) takes no soft spans")
        #expect(core.takesInputGradients)
        let lengths = (ProcessInfo.processInfo.environment["FRIGATE_CORE_LENGTHS"] ?? "256,512,1024")
            .split(separator: ",").compactMap { Int($0) }
        let passage = String(repeating: try String(contentsOfFile: #filePath, encoding: .utf8), count: 8)
        let ids = context.tokenizer.encode(text: passage)
        let span = 32
        let dtype = core.embed(MLXArray([Int32(1)]).reshaped(1, 1)).dtype
        print("core \(type(of: context.model)) hidden \(core.hiddenSize) dtype \(dtype); weights resident \(Memory.activeMemory >> 20) MB")

        MLXRandom.seed(0)
        let projector = Linear(64, core.hiddenSize)
        let reading = MLXRandom.normal([1, span, 64])
        for length in lengths {
            let tokens = MLXArray(ids.prefix(length).map(Int32.init)).reshaped(1, -1)
            // The array form, not MLXNN's `valueAndGrad(model:)`: an MLX error inside comes back as empty results,
            // which that form indexes (a crash) where this one lets `withError` throw it.
            let keys = projector.trainableParameters().flattened().map(\.0)
            let start = projector.trainableParameters().flattened().map(\.1)
            let step = MLX.valueAndGrad({ (arrays: [MLXArray]) -> [MLXArray] in
                projector.update(parameters: ModuleParameters.unflattened(Array(zip(keys, arrays.prefix(keys.count)))))
                let soft = projector(reading).asType(dtype)
                let logits = core(embeddings: concatenated([soft, core.embed(tokens)], axis: 1), cache: nil)
                let n = tokens.dim(1)
                return [crossEntropy(logits: logits[0..., span ..< span + n - 1, 0...].asType(.float32),
                                     targets: tokens[0..., 1...], reduction: .mean)]
            }, argumentNumbers: Array(0 ..< keys.count))
            Memory.clearCache()
            Memory.peakMemory = 0
            let forwardBegan = Date()
            let plain = core(embeddings: concatenated([projector(reading).asType(dtype), core.embed(tokens)], axis: 1), cache: nil)
            eval(plain)
            print(String(format: "length %5d: forward %.1f s, peak %d MB", length, Date().timeIntervalSince(forwardBegan),
                         Memory.peakMemory >> 20))
            Memory.clearCache()
            Memory.peakMemory = 0
            let began = Date()
            let (loss, grads) = try GatedDeltaPath.$differentiable.withValue(true) {
                try withError { () throws -> (MLXArray, [MLXArray]) in
                    let (values, grads) = step(start)
                    guard let loss = values.first, grads.count == keys.count else { return (MLXArray(Float.nan), []) }
                    eval([loss] + grads)
                    return (loss, grads)
                }
            }
            projector.update(parameters: ModuleParameters.unflattened(Array(zip(keys, start))))
            let seconds = Date().timeIntervalSince(began)
            let weight = try #require(zip(keys, grads).first(where: { $0.0 == "weight" })?.1)
            let size = abs(weight).sum().item(Float.self)
            print(String(format: "length %5d: loss %.3f, %.1f s, peak %d MB, |dW| %.3e", length, loss.item(Float.self), seconds,
                         Memory.peakMemory >> 20, size))
            #expect(size.isFinite && size > 0)
        }
    }
}

private struct Held: WiredMemoryPolicy, Hashable {
    let bytes: Int
    func limit(baseline: Int, activeSizes: [Int]) -> Int { bytes }
}
