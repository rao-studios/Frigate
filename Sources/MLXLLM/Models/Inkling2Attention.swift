//
//  Inkling2Attention.swift
//  MLXLLM
//
//  WHAT: Inkling-2's attention: q, k and r clipped to an L2 bound, v at unit RMS, a relative position bias per head over
//        the first `extent` distances, a sink logit per query and head, the output gated by sigmoid(gate). Sliding
//        layers see the last 1,024 positions with statically convolved keys and values; global layers see everything,
//        with dynamically convolved keys and values and queries scaled by tau = 1 + α·log(max((pos+1)/floor, 1)).
//        A prompt is read by mlx-lm's `relative_attention` Metal kernel (Inkling2Kernels); one token is decoded by the
//        fused SDPA with an additive mask when one row decodes, by the reference otherwise.
//  PIN:  THE SLIDING WINDOW IS STATE, NOT A GROWING CACHE. A sliding visit keeps `ArraysCache(size: 2)`: keys and values
//        `[B, Hkv, W, D]`, position p at slot p mod W. A prompt chunk reads the window unrolled in position order with
//        its own keys after it, then writes its last W keys back; a decoded token is put at its slot. Nothing grows
//        past W, a snapshot holds the arrays it was taken with (writes make new arrays), and the engine batches,
//        extracts and rewinds it as it does any recurrent layer's state.
//  PIN:  POSITIONS COME FROM A GLOBAL LAYER'S CACHE, READ ONCE BEFORE THE FORWARD. Every visit reads the same tokens,
//        so `Inkling2Positions` is taken from the first global visit's `ropeOffset`: a host offset for one sequence,
//        per-row offsets for a batch. A slot's position is derived from it (p − ((p − s) mod W)), never stored.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Where the forward's first token sits: one offset for a single sequence, one per row for a batch.
enum Inkling2Positions {
    case host(Int)
    case device(MLXArray)

    init(_ offset: RoPEOffset?) {
        switch offset {
        case .scalar(let o)?: self = .host(o)
        case .batch(let rows)?: self = .device(rows.asType(.int32))
        case nil: self = .host(0)
        }
    }

    /// [B] int32.
    func rows(_ batch: Int) -> MLXArray {
        switch self {
        case .host(let o): return MLXArray(Array(repeating: Int32(o), count: batch))
        case .device(let rows): return rows
        }
    }

    /// The positions of `length` tokens from here, float32: [1,L] or [B,L].
    func span(_ length: Int) -> MLXArray {
        let steps = MLXArray(Int32(0) ..< Int32(length))
        switch self {
        case .host(let o): return (steps + Int32(o)).asType(.float32).expandedDimensions(axis: 0)
        case .device(let rows): return (rows.expandedDimensions(axis: 1) + steps).asType(.float32)
        }
    }

    /// The host offset; a batch's rows are read back (a sync), which only a multi-token batched read needs.
    func host(row: Int) -> Int {
        switch self {
        case .host(let o): return o
        case .device(let rows): return Int(rows.asArray(Int32.self)[row])
        }
    }
}

/// The parts every attention shares; `Inkling2SlidingAttention` and `Inkling2FullAttention` add their convolutions.
class Inkling2Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let window: Int
    let scale: Float
    let qkClip: Float
    let eps: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "r_proj") var rProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "sink_proj") var sinkProj: Linear
    @ModuleInfo(key: "rel_logits_proj") var relLogits: Inkling2RelativeLogits
    @ModuleInfo(key: "q_l2_clip") var qClip: Inkling2ScaledL2Clip
    @ModuleInfo(key: "rel_l2_clip") var relClip: Inkling2ScaledL2Clip

    init(_ config: Inkling2TextConfiguration, sliding: Bool) {
        if sliding {
            heads = config.swaAttentionHeads
            kvHeads = config.swaKeyValueHeads
            headDim = config.swaHeadDim
            window = config.slidingWindow
        } else {
            heads = config.attentionHeads
            kvHeads = config.keyValueHeads
            headDim = config.headDim
            window = 0
        }
        let extent = sliding ? config.slidingWindow : config.relExtent
        scale = pow(Float(headDim), -0.5)
        qkClip = config.qkL2ClipThreshold
        eps = config.rmsNormEps
        let d = config.hiddenSize
        let qDim = heads * headDim, kvDim = kvHeads * headDim
        _qProj.wrappedValue = Linear(d, qDim, bias: false)
        _kProj.wrappedValue = Linear(d, kvDim, bias: false)
        _vProj.wrappedValue = Linear(d, kvDim, bias: false)
        _rProj.wrappedValue = Linear(d, heads * config.dRel, bias: false)
        _oProj.wrappedValue = Linear(qDim, d, bias: false)
        _gateProj.wrappedValue = Linear(d, qDim, bias: false)
        _sinkProj.wrappedValue = Linear(d, heads, bias: false)
        _relLogits.wrappedValue = Inkling2RelativeLogits(dRel: config.dRel, extent: extent)
        _qClip.wrappedValue = Inkling2ScaledL2Clip(heads: heads, dimensions: headDim, threshold: config.qkL2ClipThreshold)
        _relClip.wrappedValue = Inkling2ScaledL2Clip(heads: heads, dimensions: config.dRel, threshold: config.relL2ClipThreshold)
    }

    /// The layer's attention: q from `hq`, k and v from the convolved projections of `hk` and `hv`.
    func callAsFunction(
        _ hq: MLXArray, _ hk: MLXArray, _ hv: MLXArray, cache: KVCache?, conv: inout [MLXArray?],
        positions: Inkling2Positions
    ) -> MLXArray {
        fatalError("Inkling2Attention is abstract")
    }

    /// The projections clipped and normed, the bias computed, then attention by `attend`, gated and projected out.
    func attend(
        _ h: MLXArray, query: MLXArray, key: MLXArray, value: MLXArray, relative: MLXArray, tau: MLXArray?,
        attend: (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray
    ) -> MLXArray {
        let (b, l) = (h.dim(0), h.dim(1))
        let gate = gateProj(h)
        let sinks = sinkProj(h)
        var q = qClip(query.reshaped(b, l, heads, -1))
        let k = Inkling2Ops.l2Clip(key.reshaped(b, l, kvHeads, -1), scale: nil, threshold: qkClip)
        let v0 = value.reshaped(b, l, kvHeads, -1)
        let v = Inkling2Ops.rms(v0.asType(.float32), eps: eps).asType(v0.dtype)
        var r = relClip(relative.reshaped(b, l, heads, -1))
        if let tau {
            q = (q.asType(.float32) * tau).asType(q.dtype)
            r = (r.asType(.float32) * tau).asType(r.dtype)
        }
        let bias = relLogits(r)                                                           // [B,L,Hq,R]
        let out = attend(q.transposed(0, 2, 1, 3), k.transposed(0, 2, 1, 3), v.transposed(0, 2, 1, 3), bias, sinks)
        let gated = out.reshaped(b, l, -1).asType(.float32) * sigmoid(gate.asType(.float32))
        return oProj(gated.asType(gate.dtype))
    }

    /// One token's attention over `keys` [B,Hkv,S,D]: `distance` [B,S] or [S] from the query to each key, `visible` the
    /// same shape. q [B,Hq,1,D], bias [B,1,Hq,R], sinks [B,1,Hq]. Returns [B,1,Hq,D].
    func decode(
        _ q: MLXArray, _ keys: MLXArray, _ values: MLXArray, bias: MLXArray, sinks: MLXArray, distance: MLXArray,
        visible: MLXArray
    ) -> MLXArray {
        let b = q.dim(0), s = keys.dim(2), r = bias.dim(-1)
        let distance = distance.ndim == 1 ? distance.expandedDimensions(axis: 0) : distance          // [B?,S]
        let visible = visible.ndim == 1 ? visible.expandedDimensions(axis: 0) : visible
        if b == 1 {
            // The fused attention takes one row's sinks.
            let d = distance[0], seen = visible[0]
            var biased = take(bias[0, 0], clip(d, min: 0, max: r - 1), axis: -1)                     // [Hq,S]
            biased = which(d .< r, biased, zeros(like: biased))
            let mask = which(seen, biased, MLXArray(-Float.infinity).asType(biased.dtype)).asType(q.dtype)
            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values, scale: scale,
                mask: mask.expandedDimensions(axes: [0, 2]), sinks: sinks[0, 0].asType(q.dtype))
            return out.transposed(0, 2, 1, 3)
        }
        let hq = q.dim(1), hkv = keys.dim(1), dim = q.dim(3)
        let dist = broadcast(distance.expandedDimensions(axis: 1), to: [b, hq, s])
        var biased = takeAlong(bias[0..., 0], clip(dist, min: 0, max: r - 1), axis: -1).asType(.float32)   // [B,Hq,S]
        biased = which(dist .< r, biased, MLXArray(Float(0)))
        let mask = which(visible.expandedDimensions(axis: 1), biased, MLXArray(-Float.infinity))
            .reshaped(b, hkv, -1, 1, s)
        let q32 = q.asType(.float32).reshaped(b, hkv, -1, 1, dim)
        let k32 = keys.asType(.float32).expandedDimensions(axis: 2)
        let logits = matmul(q32, k32.swappedAxes(-1, -2)) * scale + mask                           // [B,Hkv,G,1,S]
        let sinkLogits = sinks.asType(.float32).transposed(0, 2, 1).reshaped(b, hkv, -1, 1, 1)
        let weights = softmax(concatenated([logits, sinkLogits], axis: -1), axis: -1)[.ellipsis, ..<s]
        let out = matmul(weights.asType(values.dtype), values.expandedDimensions(axis: 2))           // [B,Hkv,G,1,D]
        return out.reshaped(b, hq, 1, dim).transposed(0, 2, 1, 3)
    }
}

final class Inkling2SlidingAttention: Inkling2Attention {
    @ModuleInfo(key: "k_sconv") var kConv: Inkling2ShortConvolution
    @ModuleInfo(key: "v_sconv") var vConv: Inkling2ShortConvolution

    init(_ config: Inkling2TextConfiguration) {
        let dims = config.swaKeyValueHeads * config.swaHeadDim
        _kConv.wrappedValue = Inkling2ShortConvolution(channels: dims, kernel: config.sconvKernelSize)
        _vConv.wrappedValue = Inkling2ShortConvolution(channels: dims, kernel: config.sconvKernelSize)
        super.init(config, sliding: true)
    }

    override func callAsFunction(
        _ hq: MLXArray, _ hk: MLXArray, _ hv: MLXArray, cache: KVCache?, conv: inout [MLXArray?],
        positions: Inkling2Positions
    ) -> MLXArray {
        let key: MLXArray, value: MLXArray
        (key, conv[0]) = kConv(kProj(hk), state: conv[0])
        (value, conv[1]) = vConv(vProj(hv), state: conv[1])
        let ring = cache as? ArraysCache
        return attend(hq, query: qProj(hq), key: key, value: value, relative: rProj(hq), tau: nil) {
            q, k, v, bias, sinks in
            self.slide(q, k, v, bias: bias, sinks: sinks, ring: ring, positions: positions)
        }
    }

    /// q [B,Hq,L,D], k and v [B,Hkv,L,D]. Returns [B,L,Hq,D].
    private func slide(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, bias: MLXArray, sinks: MLXArray, ring: ArraysCache?,
        positions: Inkling2Positions
    ) -> MLXArray {
        let (b, l) = (q.dim(0), q.dim(2))
        let w = window
        guard let ring else {
            let keyStart = MLXArray.zeros([b], dtype: .int32)
            if l > 1 {
                return Inkling2Kernels.relativeAttention(
                    q, k, v, rel: bias, sinks: sinks, scale: scale, window: w, keyStart: keyStart)
            }
            return decode(q, k, v, bias: bias, sinks: sinks, distance: MLXArray([Int32(0)]),
                          visible: MLXArray([true]))
        }
        let shape = [b, k.dim(1), w, k.dim(3)]
        var ringK = ring[0] ?? MLXArray.zeros(shape, dtype: k.dtype)
        var ringV = ring[1] ?? MLXArray.zeros(shape, dtype: v.dtype)

        if l > 1 {
            // One sequence reads a prompt chunk: the window before it in position order, then the chunk's keys.
            precondition(b == 1, "Inkling2: a multi-token read is one sequence")
            let o = positions.host(row: 0)
            let n = min(o, w - 1)
            var keys = k, values = v
            if n > 0 {
                let slots = MLXArray((o - n ..< o).map { Int32($0 % w) })
                keys = concatenated([take(ringK, slots, axis: 2), k], axis: 2)
                values = concatenated([take(ringV, slots, axis: 2), v], axis: 2)
            }
            let out = Inkling2Kernels.relativeAttention(
                q, keys, values, rel: bias, sinks: sinks, scale: scale, window: w,
                keyStart: MLXArray.zeros([b], dtype: .int32))
            // The new ring, gathered: slot s holds the newest position p ≡ s (mod W), which `keys` (positions o − n
            // through o + l − 1) always holds once p ≥ 0; a slot no position has reached yet keeps its zeros.
            // PIN: NEVER A SUBSCRIPT WRITE. `ringK[…] = x` swaps the contents of the very object the cache holds, and a
            // CheckpointedCache snapshot holds that object by reference: the next chunk would rewrite the snapshot.
            let last = o + l - 1
            let held = (0 ..< w).map { last - ((last - $0) % w + w) % w }
            let gather = MLXArray(held.map { Int32(max($0 - (o - n), 0)) })
            ringK = take(keys, gather, axis: 2)
            ringV = take(values, gather, axis: 2)
            if last < w - 1 {
                let reached = MLXArray(held.map { $0 >= 0 }).reshaped(1, 1, w, 1)
                ringK = which(reached, ringK, ring[0] ?? MLXArray.zeros(shape, dtype: k.dtype))
                ringV = which(reached, ringV, ring[1] ?? MLXArray.zeros(shape, dtype: v.dtype))
            }
            ring[0] = ringK
            ring[1] = ringV
            return out
        }

        // One token per row, at its slot; every slot's position follows from the row's.
        let pos = positions.rows(b)                                                         // [B]
        let slot = remainder(pos, Int32(w))
        let index = broadcast(slot.reshaped(b, 1, 1, 1), to: [b, k.dim(1), 1, k.dim(3)])
        ringK = putAlong(ringK, index, values: k, axis: 2)
        ringV = putAlong(ringV, index, values: v, axis: 2)
        ring[0] = ringK
        ring[1] = ringV
        let slots = MLXArray(Int32(0) ..< Int32(w)).expandedDimensions(axis: 0)                // [1,W]
        let p = pos.expandedDimensions(axis: 1)                                             // [B,1]
        let back = remainder(remainder(p - slots, Int32(w)) + Int32(w), Int32(w))           // [B,W], 0 ..< W
        return decode(q, ringK, ringV, bias: bias, sinks: sinks, distance: back, visible: (p - back) .>= 0)
    }
}

final class Inkling2FullAttention: Inkling2Attention {
    let logScalingFloor: Float
    let logScalingAlpha: Float
    @ModuleInfo(key: "k_sconv") var kConv: Inkling2DynamicConvolution
    @ModuleInfo(key: "v_sconv") var vConv: Inkling2DynamicConvolution

    init(_ config: Inkling2TextConfiguration) {
        let dims = config.keyValueHeads * config.headDim
        _kConv.wrappedValue = Inkling2DynamicConvolution(
            channels: dims, inputDimensions: config.hiddenSize, heads: config.keyValueHeads,
            kernel: config.sconvKernelSize)
        _vConv.wrappedValue = Inkling2DynamicConvolution(
            channels: dims, inputDimensions: config.hiddenSize, heads: config.keyValueHeads,
            kernel: config.sconvKernelSize)
        logScalingFloor = Float(config.logScalingFloor)
        logScalingAlpha = config.logScalingAlpha
        super.init(config, sliding: false)
    }

    override func callAsFunction(
        _ hq: MLXArray, _ hk: MLXArray, _ hv: MLXArray, cache: KVCache?, conv: inout [MLXArray?],
        positions: Inkling2Positions
    ) -> MLXArray {
        let key: MLXArray, value: MLXArray
        (key, conv[0]) = kConv(kProj(hk), dynamicInput: hk, state: conv[0])
        (value, conv[1]) = vConv(vProj(hv), dynamicInput: hv, state: conv[1])
        let l = hq.dim(1)
        let p = positions.span(l)                                                           // [1|B, L]
        let tau = 1 + logScalingAlpha * log(maximum((p + 1) / logScalingFloor, 1.0))
        return attend(hq, query: qProj(hq), key: key, value: value, relative: rProj(hq),
                      tau: tau.expandedDimensions(axes: [2, 3])) { q, k, v, bias, sinks in
            self.global(q, k, v, bias: bias, sinks: sinks, cache: cache, positions: positions)
        }
    }

    private func global(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, bias: MLXArray, sinks: MLXArray, cache: KVCache?,
        positions: Inkling2Positions
    ) -> MLXArray {
        let (b, l) = (q.dim(0), q.dim(2))
        var keys = k, values = v
        if let cache { (keys, values) = cache.update(keys: k, values: v) }
        let s = keys.dim(2)
        // Each row's tokens end at its position + L; the columns before its first token are another row's padding.
        let ends = positions.rows(b) + Int32(l)                                              // [B]
        if l > 1 {
            return Inkling2Kernels.relativeAttention(
                q, keys, values, rel: bias, sinks: sinks, scale: scale, window: 0, keyStart: MLXArray(Int32(s)) - ends)
        }
        let columns = MLXArray(Int32(0) ..< Int32(s))
        let distance = MLXArray(Int32(s - 1)) - columns                                                // [S], the same for every row
        let visible = columns.expandedDimensions(axis: 0) .>= (MLXArray(Int32(s)) - ends).expandedDimensions(axis: 1)
        return decode(q, keys, values, bias: bias, sinks: sinks, distance: distance, visible: visible)
    }
}
