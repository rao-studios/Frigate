//
//  Inkling2Layers.swift
//  MLXLLM
//
//  WHAT: The parts of Inkling-2's decoder layer around its attention (Inkling2Attention.swift): the short convolutions,
//        the softcapped SwiGLU experts behind a sigmoid router, the depth attention over block states (attention
//        residuals) and the cross-pass message gates of the looped layers. Ported from mlx-lm's `inkling2` package
//        (layers.py, PR 2086), which the golden test holds this port to.
//  PIN:  A STATE IS A BUFFER OF ITS OWN. A convolution's kept state is the last K−1 rows of its window, copied (`owned`):
//        a slice shares its source's buffer, and a prefill's window is the whole chunk (Mamba3.swift's PIN).
//  PIN:  THE EXPERTS' FORMAT COMES FROM THE CHECKPOINT. Layers 0–19 of Inkling-2-Mini are NVFP4 (4 bits, groups of 16,
//        a float32 scale per expert) and 20–23 MXFP8 (8 bits, groups of 32); `Inkling2Model.sanitize` reads each
//        layer's from its tensors' shapes and swaps in the module that holds it, as mlx-lm's sanitize does.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

enum Inkling2Ops {
    /// RMS norm without a weight, in the input's dtype (float32 everywhere the model calls it).
    static func rms(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        MLXFast.rmsNorm(x, weight: ones(x.dim(-1), x.dtype), eps: eps)
    }

    private struct OnesKey: Hashable { let width: Int; let dtype: DType }
    private static let onesLock = NSLock()
    nonisolated(unsafe) private static var onesMade: [OnesKey: MLXArray] = [:]

    /// `rms`'s weight, made once per width from host memory. `MLXArray.ones` is a `Full` primitive: a dispatch on every
    /// call, 339 of them in a decoded token (2026-11-21).
    static func ones(_ width: Int, _ dtype: DType) -> MLXArray {
        onesLock.lock()
        defer { onesLock.unlock() }
        let key = OnesKey(width: width, dtype: dtype)
        if let made = onesMade[key] { return made }
        let host = MLXArray([Float](repeating: 1, count: width))
        let made = dtype == .float32 ? host : host.asType(dtype)
        onesMade[key] = made
        return made
    }

    private static let rowsLock = NSLock()
    nonisolated(unsafe) private static var rowsMade: [[Int]: MLXArray] = [:]

    /// Each row's own index, shaped as `shape`: the left indices of a gathered matmul whose rows are not gathered.
    /// Made from host memory once per shape; left out, `gather_qmm` builds the same with an `arange` on every call
    /// (96 a decoded token, 2026-11-21). Nil past 256 rows, where MLX's own is as cheap.
    /// PIN: NEVER WITH SORTED INDICES. MLX takes its sorted-experts kernel only when no left indices are given
    /// (`right_sorted = sorted_indices && !lhs_indices_`); passing them halved prefill (718 → 299 tok/s at 1k).
    static func rows(_ shape: [Int]) -> MLXArray? {
        let count = shape.reduce(1, *)
        guard count <= 256 else { return nil }
        rowsLock.lock()
        defer { rowsLock.unlock() }
        if let made = rowsMade[shape] { return made }
        let made = MLXArray((0 ..< UInt32(count)).map { $0 }, shape)
        rowsMade[shape] = made
        return made
    }

    /// `x` in a buffer of its own (an element-wise op writes a fresh one).
    static func owned(_ x: MLXArray) -> MLXArray { x * 1 }

    /// Scales the last axis to an L2 norm of at most `threshold`, after an optional per-head scale.
    static func l2Clip(_ x: MLXArray, scale: MLXArray?, threshold: Float) -> MLXArray {
        let limit = MLXArray(threshold)
        if let scale { return Inkling2Fused.scaledL2Clip([x, scale, limit])[0] }
        return Inkling2Fused.l2Clip([x, limit])[0]
    }

    /// The SwiGLU of capped gate and up; the sigmoid is of the uncapped gate.
    static func softcappedSwiGLU(_ gate: MLXArray, _ up: MLXArray, gateCap: Float, upCap: Float) -> MLXArray {
        Inkling2Fused.softcappedSwiGLU([gate, up, MLXArray(gateCap), MLXArray(upCap)])[0]
    }

    /// Attention of learned queries over the depth sources of each token: `blocks` [K,B,L,D] and `partial` [B,L,D],
    /// float32, are keys and values both; `queries` [Q,D]. Returns [B,L,Q,D] float32.
    static func depthAttention(_ blocks: MLXArray, _ partial: MLXArray, _ queries: MLXArray) -> MLXArray {
        let q = queries.asType(.float32)
        let scale = pow(Float(partial.dim(-1)), -0.5)
        if Inkling2Kernels.available { return Inkling2Kernels.depthAttention(blocks, partial, q, scale: scale) }
        let stacked = movedAxis(concatenated([blocks, partial.expandedDimensions(axis: 0)], axis: 0),
                                source: 0, destination: -2)                                     // [B,L,N,D]
        let weights = softmax(matmul(stacked, q.T) * scale, axis: -2, precise: true)            // [B,L,N,Q]
        return matmul(weights.swappedAxes(-1, -2), stacked)
    }

    /// The window a convolution reads: its kept state (zeros before the first token), then `x` in float32.
    static func window(_ x: MLXArray, state: MLXArray?, kernel: Int) -> MLXArray {
        let x32 = x.asType(.float32)
        let state = state ?? MLXArray.zeros([x.dim(0), kernel - 1, x.dim(-1)], dtype: .float32)
        return concatenated([state, x32], axis: 1)
    }

    /// The state a window leaves: its last `n` rows, copied.
    static func keep(_ window: MLXArray, _ n: Int) -> MLXArray {
        owned(window[0..., (window.dim(1) - n)..., 0...])
    }
}

/// `scale` per head, then the L2 clip.
final class Inkling2ScaledL2Clip: Module {
    let threshold: Float
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(heads: Int, dimensions: Int, threshold: Float) {
        self.threshold = threshold
        _scale.wrappedValue = MLXArray.ones([heads, dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { Inkling2Ops.l2Clip(x, scale: scale, threshold: threshold) }
}

/// The relative position bias of each head: its clipped relative state times a learned projection to `extent` distances.
final class Inkling2RelativeLogits: Module {
    @ParameterInfo(key: "proj") var proj: MLXArray

    init(dRel: Int, extent: Int) {
        _proj.wrappedValue = MLXArray.zeros([dRel, extent])
    }

    func callAsFunction(_ relative: MLXArray) -> MLXArray { matmul(relative, proj.asType(relative.dtype)) }
}

/// A depthwise convolution's weight, `[C, K, 1]` (the checkpoint's `[C, 1, K]` swapped by sanitize).
final class Inkling2ConvWeight: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    init(channels: Int, kernel: Int) {
        _weight.wrappedValue = MLXArray.zeros([channels, kernel, 1])
    }
}

/// A causal depthwise convolution of `kernel` taps added to its input, scaled by 1/√(K+1).
final class Inkling2ShortConvolution: Module {
    let kernel: Int
    @ModuleInfo(key: "conv1d") var conv: Inkling2ConvWeight

    init(channels: Int, kernel: Int) {
        self.kernel = kernel
        _conv.wrappedValue = Inkling2ConvWeight(channels: channels, kernel: kernel)
    }

    /// Returns the output, in `x`'s dtype, and the state it leaves (the window's last K−1 rows, float32).
    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        if x.dim(1) == 1, let state, Inkling2Kernels.available, Inkling2Kernels.fusedConvolution {
            return Inkling2Kernels.shortConvolution(x, state: state, weight: conv.weight, kernel: kernel)
        }
        let residual = x.asType(.float32)
        let window = Inkling2Ops.window(x, state: state, kernel: kernel)
        let w = conv.weight
        // As the reference: the convolution runs in the weight's dtype.
        let out = conv1d(window.asType(w.dtype), w, groups: x.dim(-1))
        let y = (out.asType(.float32) + residual) / sqrt(Float(kernel + 1))
        return (y.asType(x.dtype), Inkling2Ops.keep(window, kernel - 1))
    }
}

/// The global layers' key and value convolution: a static weight per channel plus a dynamic one per head, projected
/// from the token's input. Scaled by 1/√(2K+1).
final class Inkling2DynamicConvolution: Module {
    let kernel: Int
    let heads: Int
    @ParameterInfo(key: "static_weight") var staticWeight: MLXArray
    @ModuleInfo(key: "weight_proj") var weightProj: Linear

    init(channels: Int, inputDimensions: Int, heads: Int, kernel: Int) {
        self.kernel = kernel
        self.heads = heads
        _staticWeight.wrappedValue = MLXArray.zeros([kernel, channels])
        _weightProj.wrappedValue = Linear(inputDimensions, heads * kernel, bias: false)
    }

    /// Returns the output, in `x`'s dtype, and the state it leaves (the window's last K−1 rows, float32).
    func callAsFunction(_ x: MLXArray, dynamicInput: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let (b, l, c) = (x.dim(0), x.dim(1), x.dim(2))
        let k = kernel, h = heads
        if l == 1, let state, Inkling2Kernels.available, Inkling2Kernels.fusedConvolution {
            return Inkling2Kernels.dynamicConvolution(
                x, state: state, fixed: staticWeight, dynamic: weightProj(dynamicInput), heads: h, kernel: k)
        }
        let dynamic = weightProj(dynamicInput).asType(.float32).reshaped(b, l, h, 1, k)
        let statics = staticWeight.asType(.float32).T.reshaped(h, c / h, k)
        let weights = statics + dynamic                                                   // [B,L,H,C/H,K]
        let window = Inkling2Ops.window(x, state: state, kernel: k)
        // Lag 0 is the token itself.
        let windows = stacked((0 ..< k).map { lag in window[0..., (k - 1 - lag) ..< (k - 1 - lag + l), 0...] }, axis: -1)
            .reshaped(b, l, h, c / h, k)
        let out = (windows * weights).sum(axis: -1).reshaped(b, l, c)
        let y = (out + x.asType(.float32)) / sqrt(Float(2 * k + 1))
        return (y.asType(x.dtype), Inkling2Ops.keep(window, k - 1))
    }
}

/// One expert projection for every expert, plain or quantized in the format sanitize found.
final class Inkling2SwitchLinear: Module {
    struct Format: Equatable {
        let bits: Int
        let groupSize: Int
        let mode: QuantizationMode
    }

    let format: Format?
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray?
    @ParameterInfo(key: "global_scale") var globalScale: MLXArray?

    init(inputDimensions: Int, outputDimensions: Int, experts: Int) {
        format = nil
        _weight.wrappedValue = MLXArray.zeros([experts, outputDimensions, inputDimensions], dtype: .bfloat16)
        _scales.wrappedValue = nil
        _globalScale.wrappedValue = nil
    }

    init(inputDimensions: Int, outputDimensions: Int, experts: Int, format: Format, globalScale: Bool) {
        self.format = format
        _weight.wrappedValue = MLXArray.zeros(
            [experts, outputDimensions, inputDimensions * format.bits / 32], dtype: .uint32)
        _scales.wrappedValue = MLXArray.zeros([experts, outputDimensions, inputDimensions / format.groupSize], dtype: .uint8)
        _globalScale.wrappedValue = globalScale ? MLXArray.zeros([experts], dtype: .float32) : nil
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sorted: Bool) -> MLXArray {
        if let format, let scales {
            return gatherQuantizedMM(
                x, weight, scales: scales, lhsIndices: sorted ? nil : Inkling2Ops.rows(Array(x.shape.dropLast(2))),
                rhsIndices: indices, transpose: true,
                groupSize: format.groupSize, bits: format.bits, mode: format.mode,
                globalScale: globalScale, sortedIndices: sorted)
        }
        return gatherMM(x, weight.swappedAxes(-1, -2), rhsIndices: indices, sortedIndices: sorted)
    }
}

/// The routed experts: a fused gate/up projection whose rows alternate gate, up (as the checkpoint), the softcapped
/// SwiGLU, and the down projection.
final class Inkling2Experts: Module {
    let dimensions: Int
    let hidden: Int
    let experts: Int
    let gateCap: Float
    let upCap: Float
    @ModuleInfo(key: "gate_up_proj") var gateUp: Inkling2SwitchLinear
    @ModuleInfo(key: "down_proj") var down: Inkling2SwitchLinear

    init(_ config: Inkling2TextConfiguration) {
        dimensions = config.hiddenSize
        hidden = config.intermediateSize
        experts = config.routedExperts
        gateCap = config.mlpGateSoftcap
        upCap = config.mlpUpSoftcap
        _gateUp.wrappedValue = Inkling2SwitchLinear(
            inputDimensions: dimensions, outputDimensions: 2 * hidden, experts: experts)
        _down.wrappedValue = Inkling2SwitchLinear(inputDimensions: hidden, outputDimensions: dimensions, experts: experts)
    }

    /// Swaps in the projections quantized as the checkpoint's tensors say.
    func adopt(gateUp gateUpFormat: Inkling2SwitchLinear.Format, gateUpGlobal: Bool,
               down downFormat: Inkling2SwitchLinear.Format, downGlobal: Bool) {
        let newGateUp = Inkling2SwitchLinear(
            inputDimensions: dimensions, outputDimensions: 2 * hidden, experts: experts, format: gateUpFormat,
            globalScale: gateUpGlobal)
        let newDown = Inkling2SwitchLinear(
            inputDimensions: hidden, outputDimensions: dimensions, experts: experts, format: downFormat,
            globalScale: downGlobal)
        update(modules: ModuleChildren.unflattened(["gate_up_proj": newGateUp, "down_proj": newDown]))
    }

    /// x [B,L,D], indices [B,L,k] → [B,L,k,D].
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size >= 64
        var idx = indices
        var inverseOrder: MLXArray?
        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }
        let gu = gateUp(x, idx, sorted: doSort)
        x = Inkling2Ops.softcappedSwiGLU(
            gu[.ellipsis, .stride(by: 2)], gu[.ellipsis, .stride(from: 1, by: 2)], gateCap: gateCap, upCap: upCap)
        x = down(x, idx, sorted: doSort)
        if let inverseOrder {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }
        return x.squeezed(axis: -2)
    }
}

/// Sigmoid scores over the experts; the top k by score plus a correction bias, weighted by their scores normalised and
/// scaled by `route_scale`.
final class Inkling2Router: Module {
    let topK: Int
    let routeScale: Float
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var bias: MLXArray

    init(_ config: Inkling2TextConfiguration) {
        topK = config.expertsPerToken
        routeScale = config.routeScale
        _weight.wrappedValue = MLXArray.zeros([config.routedExperts, config.hiddenSize], dtype: .float32)
        _bias.wrappedValue = MLXArray.zeros([config.routedExperts], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let logits = matmul(x.asType(.float32), weight.T)
        if Inkling2Kernels.available {
            return Inkling2Kernels.routing(logits, bias: bias, topK: topK, eps: 1e-20, routeScale: routeScale)
        }
        let scores = sigmoid(logits)
        let biased = scores + bias
        let indices = argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        return (indices, weights * routeScale)
    }
}

final class Inkling2SparseMoe: Module {
    @ModuleInfo(key: "gate") var gate: Inkling2Router
    @ModuleInfo(key: "experts") var experts: Inkling2Experts

    init(_ config: Inkling2TextConfiguration) {
        _gate.wrappedValue = Inkling2Router(config)
        _experts.wrappedValue = Inkling2Experts(config)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x)
        let states = experts(x, indices)
        return Inkling2Fused.combine([states, weights.expandedDimensions(axis: -1)])[0]
    }
}

/// The learned query (one, or one per q/k/v input) a layer reads the block states with.
final class Inkling2BlockResidual: Module {
    @ParameterInfo(key: "query") var query: MLXArray

    init(dimensions: Int, queries: Int = 1) {
        _query.wrappedValue = queries == 1
            ? MLXArray.zeros([dimensions], dtype: .float32)
            : MLXArray.zeros([queries, dimensions], dtype: .float32)
    }

    /// [B,L,Q,D] float32.
    func callAsFunction(_ blocks: MLXArray, _ partial: MLXArray) -> MLXArray {
        Inkling2Ops.depthAttention(blocks, partial, query.reshaped(-1, partial.dim(-1)))
    }
}

/// Mixes a looped layer's input with the message the previous pass left for it.
final class Inkling2MessageGate: Module {
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "forget_gate_proj") var forgetProj: Linear
    @ModuleInfo(key: "message_gate_proj") var gainProj: Linear

    init(dimensions: Int, rank: Int) {
        _downProj.wrappedValue = Linear(2 * dimensions, rank, bias: false)
        _forgetProj.wrappedValue = Linear(rank, dimensions, bias: false)
        _gainProj.wrappedValue = Linear(rank, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray, _ message: MLXArray) -> MLXArray {
        let states = concatenated([Inkling2Ops.rms(x), Inkling2Ops.rms(message)], axis: -1)
        let down = downProj(states.asType(downProj.weight.dtype)).asType(.float32)
        return Inkling2Fused.messageGate([forgetProj(down), gainProj(down), x, message])[0]
    }
}

/// The looped layers' message queries: query, key, value and mlp, per looped layer.
final class Inkling2MessageInput: Module {
    @ParameterInfo(key: "query") var query: MLXArray

    init(dimensions: Int, loopedLayers: Int) {
        _query.wrappedValue = MLXArray.zeros([loopedLayers, 4, dimensions], dtype: .float32)
    }

    /// The bank is the previous pass's later block states and its normed output. Returns [B,L,4,D].
    func callAsFunction(_ bank: (MLXArray, MLXArray), layer: Int) -> MLXArray {
        Inkling2Ops.depthAttention(bank.0, bank.1, query[layer])
    }
}
