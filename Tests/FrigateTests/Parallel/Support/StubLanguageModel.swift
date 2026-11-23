import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The smallest model the parallel engine serves: one attention layer and one recurrent layer of tiny
/// dimensions, with logits that always favour one token. It drives the engine's bookkeeping — slots,
/// rows, caches into the batch and back out — and none of a real model's kernels.
///
/// What its layers hold says what was read: the attention layer's keys and values are the tokens themselves, one
/// column a token, and the recurrent layer counts the passes. So a cache cut into stretches and put back together
/// (`SlotStretch`) can be held to the cache it was cut from, value for value.
final class StubLanguageModel: Module, LanguageModel {
    /// Set, the next forward pass makes MLX raise an error, as a failed allocation would.
    final class Fault: @unchecked Sendable {
        private let lock = NSLock()
        private var armed = false
        func arm() { lock.withLock { armed = true } }
        func take() -> Bool { lock.withLock { defer { armed = false }; return armed } }
    }

    let fault = Fault()
    /// Whether the model keeps a recurrent layer beside its attention layer. Without one its cache trims by position.
    let recurrent: Bool
    /// How many of a row's last tokens its recurrent layer keeps as a third array, as Inkling-2's attention keeps a ring
    /// of the last 1,024: zero keeps none. What a row holds there is its own tokens, so rows mixed up in a batch show.
    let window: Int

    init(recurrent: Bool = true, window: Int = 0) {
        self.recurrent = recurrent
        self.window = window
        super.init()
    }

    static let heads = 2
    static let headDim = 4
    static let vocabulary = 16
    /// What every step samples: the argmax of the logits.
    static let token = 7

    func prepare(_ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: PrefillParameters) throws -> PrepareResult { .tokens(input.text) }

    /// `[B, L]` in, `[B, L, vocabulary]` out. The attention layer gains `L` columns a row, each its token's id;
    /// the recurrent layer's state is replaced with one a step further on, as a real layer's is.
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let batch = inputs.dim(0), length = inputs.dim(1)
        // A shape MLX refuses: it reports an error through its handler, as an allocation that failed does.
        if fault.take() { _ = inputs.reshaped(inputs.size + 1) }
        if let cache {
            let block = MLXArray.zeros([batch, Self.heads, length, Self.headDim]) + inputs.asType(.float32).reshaped(batch, 1, length, 1)
            _ = cache[0].update(keys: block, values: block)
            if cache.count > 1, let recurrent = cache[1] as? ArraysCache {
                recurrent[0] = MLXArray.zeros([batch, 3, 8])
                recurrent[1] = (recurrent[1] ?? MLXArray.zeros([batch, Self.heads, Self.headDim, Self.headDim])) + 1
                if window > 0 {
                    let before = recurrent[2] ?? MLXArray.zeros([batch, window])
                    recurrent[2] = concatenated([before, inputs.asType(.float32)], axis: 1)[0..., length...]
                }
            }
        }
        let peak = MLXArray((0 ..< Self.vocabulary).map { Float($0 == Self.token ? 1 : 0) }).reshaped(1, 1, Self.vocabulary)
        return MLXArray.zeros([batch, length, Self.vocabulary]) + peak
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        guard recurrent else { return [KVCacheSimple()] }
        return [KVCacheSimple(), window > 0 ? ArraysCache(size: 3) : MambaCache()]
    }
}
