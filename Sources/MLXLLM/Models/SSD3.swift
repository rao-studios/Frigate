//
//  SSD3.swift
//  MLXLLM
//
//  WHAT: The Mamba-3 MIMO recurrence, chunked and one token at a time, ported from Braid's `SSD3`
//        (github.com/rao-studios/Braid, Sources/BraidKit/Model/SSD3.swift) without its resets. Per head, at
//        position t with α = exp(AΔ), λ = σ(trap), γ = λΔ and β = (1 − λ)Δα, over rank-R keys k and values v
//        (x times `mimo_x`) rotated by the cumulative angle:
//            h_t = α_t h_{t−1} + β_t Σ_r k_{t−1,r} ⊗ v_{t−1,r} + γ_t Σ_r k_{t,r} ⊗ v_{t,r}
//            y_{t,r} = h_t · q_{t,r} + D v_{t,r},  gated by silu(z_t ⊙ mimo_z_r),  summed through mimo_o.
//        Chunked (the reference's `mamba3_MIMO_chunk_ref`, rearranged for speed): within a chunk key s weighs γ_s at
//        its own position and γ_s + (1 − λ_{s+1})Δ_{s+1} under the decay after it, held in the chunk's mask; the rank
//        is flattened into the chunk's score matrix; every chunk is one batched product and the hand-off of the state
//        between chunks is one more. Plain MLX ops, two of them fused by `compile`; no kernels.
//  PIN:  HEAD-MAJOR. The chunked pass takes q and k as [B,H,L,R,N] (the mixer writes them so, the groups spread over the
//        heads in the same pass), so the chunks are reshapes, never transposed copies. Measured 2026-11-19 on the 1.5b,
//        4 × 1,024 tokens at BF16: 1,229 tok/s position-major with a chunk loop and the diagonal taken back out, 2,530
//        with all of this (Mamba3SpeedTests); the float32 read stays 3e-5 of Braid's oracle in logits.
//  PIN:  FLOAT32 THROUGHOUT, whatever the weights are: the projections hand this file f32 and the state stays f32.
//  PIN:  EVERY RUNNING SUM OVER A ROW-CONTIGUOUS COPY (`runningSum`). MLX's Metal scan treated a transposed (dense but
//        permuted) input as contiguous and returned wrong values, at times over its input (Braid MAMBA3.md, AUDIT.md
//        2026-11-10). A running sum over a length-1 axis is its input, never scanned: the same scan takes the stride
//        of a length-1 axis literally and wrote 64 × 9,440 floats past a 64-float output (AUDIT.md 2026-11-12).
//        `.contiguous()` copies nothing at length 1, so it does not help there.
//  PIN:  The rotation is the training kernel's: dimension n with N/2 + n for n < S (mamba3_mimo_fwd.py). The angle a
//        state carries is reduced modulo 2π, as the kernel's returned state is.
//

import Foundation
import MLX
import MLXNN

public enum SSD3 {
    /// The running sum along `axis` of a row-contiguous copy of `x`; `x` itself when the axis has length 1 (see PIN).
    public static func runningSum(_ x: MLXArray, axis: Int) -> MLXArray {
        guard x.dim(axis) > 1 else { return x }
        return cumsum(x.contiguous(), axis: axis)
    }

    /// For x […, q]: out[…, t, s] = Σ_{r=s+1}^{t} x_r for t ≥ s (0 on the diagonal), −inf above it.
    ///
    /// PIN: NOT `segsum(_:mask:)` in SSM.swift, which leaves the upper triangle at 0 (exp 1) when no mask is given.
    static func segsum(_ x: MLXArray) -> MLXArray {
        let q = x.dim(-1)
        let y = tril(repeated(x[.ellipsis, .newAxis], count: q, axis: -1), k: -1)
        let out = cumsum(y, axis: -2)
        return MLX.where(tril(MLXArray.ones([q, q], dtype: .bool), k: 0), out, MLXArray(-Float.infinity))
    }

    /// Rotates the last axis of `t` [..., N] by `cos`/`sin` [..., S]: dimension n with N/2 + n for n < S.
    public static func rotate(_ t: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
        let n = t.dim(-1), half = n / 2, angles = c.dim(-1)
        let a = t[.ellipsis, 0 ..< angles], b = t[.ellipsis, half ..< half + angles]
        var parts = [a * c - b * s]
        if angles < half { parts.append(t[.ellipsis, angles ..< half]) }
        parts.append(a * s + b * c)
        if angles < half { parts.append(t[.ellipsis, (half + angles) ..< n]) }
        return concatenated(parts, axis: -1)
    }

    /// `rotate` as two tables over the halves of the last axis, for t viewed as [..., 2, N/2]: C [..., 1, N/2] holds cos
    /// for the first S dimensions of a half and 1 after, S [..., 2, N/2] holds −sin over the first half's and +sin over
    /// the second's (0 after). Then rotate(t) = t ⊙ C + swap(t) ⊙ S, swap trading the halves: the same pairing and the
    /// same products, element-wise only, so `compile` makes it one pass over the output.
    public static func rotationTables(cos c: MLXArray, sin s: MLXArray, half: Int) -> (c: MLXArray, s: MLXArray) {
        var cHalf = c, sHalf = s
        if c.dim(-1) < half {
            let rest = Array(c.shape.dropLast()) + [half - c.dim(-1)]
            cHalf = concatenated([c, MLXArray.ones(rest, dtype: c.dtype)], axis: -1)
            sHalf = concatenated([s, MLXArray.zeros(rest, dtype: s.dtype)], axis: -1)
        }
        return (cHalf.expandedDimensions(axis: -2), stacked([-sHalf, sHalf], axis: -2))
    }

    /// t [..., 2, N/2] with its halves traded.
    public static func swapHalves(_ t: MLXArray) -> MLXArray {
        concatenated([t[.ellipsis, 1 ..< 2, 0...], t[.ellipsis, 0 ..< 1, 0...]], axis: -2)
    }

    /// The angle at every position: the carried angle plus the running sum of tanh(θ)·Δ·π.
    /// inc [B,L,H,S]; angle0 [B,H,S] or nil. Returns [B,L,H,S] (not reduced).
    public static func angles(increments inc: MLXArray, angle0: MLXArray?) -> MLXArray {
        let cs = runningSum(inc, axis: 1)
        return angle0.map { cs + $0.expandedDimensions(axis: 1) } ?? cs
    }

    /// x reduced to [0, 2π), as the kernel's returned angle state is.
    public static func wrap(_ x: MLXArray) -> MLXArray {
        let twoPi = MLXArray(Float(2 * Double.pi))
        return x - twoPi * floor(x / twoPi)
    }

    /// The chunked pass, head-major. q, k [B,H,L,R,N] rotated and biased; x, z [B,L,H,P]; adt, dt, lam [B,L,H] (AΔ ≤ 0,
    /// Δ, σ(trap)); mimoX, mimoZ, mimoO [H,R,P]; d [H]; ssm [B,H,P,N], kPrev [B,R,H,N] and xPrev [B,H,P] the state
    /// before the sequence or nil. Returns y [B,L,H,P] and the SSM state after the last position.
    ///
    /// Every chunk at once, the sequence padded to whole chunks with positions that neither decay nor write; only the
    /// hand-off of the state from one chunk to the next runs in order. Within a chunk, key s weighs γ_s at its own
    /// position and γ_s + (1 − λ_{s+1})Δ_{s+1} under the decay after it: the trapezoid lives in the mask, so the
    /// keys are never reweighted and no diagonal is taken back out.
    public static func chunked(q: MLXArray, k: MLXArray, x: MLXArray, adt: MLXArray, dt: MLXArray, lam: MLXArray,
                               mimoX: MLXArray, mimoZ: MLXArray, mimoO: MLXArray, d: MLXArray, z: MLXArray,
                               ssm: MLXArray?, kPrev: MLXArray?, xPrev: MLXArray?, chunk: Int) -> (y: MLXArray, ssm: MLXArray) {
        let (b, h, l, r, n) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3), q.dim(4))
        let p = x.dim(3)
        let v = x.transposed(0, 2, 1, 3).expandedDimensions(axis: 3) * mimoX.reshaped([1, h, 1, r, p])   // [B,H,L,R,P]
        let dtH = dt.swappedAxes(1, 2), lamH = lam.swappedAxes(1, 2)                    // [B,H,L]
        let gamma = lamH * dtH
        let nextTerm = concatenated([(dtH * (1 - lamH))[0..., 0..., 1...], MLXArray.zeros([b, h, 1], dtype: .float32)], axis: 2)
        let factor = gamma + nextTerm

        // The state the sequence enters with: the carried SSM state, plus the first position's β term on the
        // carried key and value (β_0 / α_0 = (1 − λ_0)Δ_0, since the carry-in decays it by α_0).
        var s = ssm?.asType(.float32) ?? MLXArray.zeros([b, h, p, n], dtype: .float32)
        if let kPrev, let xPrev {
            let vPrev = xPrev.expandedDimensions(axis: 1) * mimoX.transposed(1, 0, 2)       // [B,R,H,P]
            let outer = matmul(vPrev.transposed(0, 2, 3, 1), kPrev.transposed(0, 2, 1, 3))   // [B,H,P,N]
            let w = (dt[0..., 0] * (1 - lam[0..., 0])).reshaped([b, h, 1, 1])
            s = s + w * outer
        }

        let c = min(chunk, l), nc = (l + c - 1) / c, pad = nc * c - l
        // Along the positions (axis 2), padded with zeros to whole chunks.
        func padded(_ t: MLXArray) -> MLXArray {
            guard pad > 0 else { return t }
            var shape = t.shape
            shape[2] = pad
            return concatenated([t, MLXArray.zeros(shape, dtype: t.dtype)], axis: 2)
        }
        // [B,H,L,R,W] → [B,H,nc,c·R,W]: rows run over (position, rank), position-major, as the reference flattens them.
        func blocks(_ t: MLXArray) -> MLXArray { padded(t).reshaped([b, h, nc, c * r, t.dim(4)]) }
        func perChunk(_ t: MLXArray) -> MLXArray { padded(t).reshaped([b, h, nc, c]) }   // [B,H,L] → [B,H,nc,c]
        let dtA = perChunk(adt.swappedAxes(1, 2))
        let cs = runningSum(dtA, axis: -1)
        let fc = perChunk(factor).reshaped([b, h, nc, 1, c]), gc = perChunk(gamma).reshaped([b, h, nc, 1, c])
        let weight = MLX.where(MLXArray.eye(c, dtype: .bool), gc, exp(segsum(dtA)) * fc)   // [B,H,nc,c,c] (t, s)
        let qc = blocks(q), kc = blocks(k), vc = blocks(v)
        let scores = matmul(qc, kc.swappedAxes(-1, -2)).reshaped([b, h, nc, c, r, c, r])
            * weight.reshaped([b, h, nc, c, 1, c, 1])
        var y = matmul(scores.reshaped([b, h, nc, c * r, c * r]), vc)                    // [B,H,nc,cR,P]
        // Each chunk's own writes, weighted and decayed to its end, then the state each chunk enters with.
        let toEnd = exp(cs[.ellipsis, (c - 1)...] - cs) * fc.reshaped([b, h, nc, c])      // [B,H,nc,c]
        let decayed = (vc.reshaped([b, h, nc, c, r, p]) * toEnd.reshaped([b, h, nc, c, 1, 1])).reshaped([b, h, nc, c * r, p])
        let written = matmul(decayed.swappedAxes(-1, -2), kc)                           // [B,H,nc,P,N]
        // The state each chunk enters with and the state after the last, in one product: the chunks' total decays
        // (exp of running sums, as within a chunk) over the carried state and each chunk's writes.
        let totals = concatenated([MLXArray.zeros([b, h, 1], dtype: .float32), cs[.ellipsis, c - 1]], axis: -1)   // [B,H,nc+1]
        let states = concatenated([s.expandedDimensions(axis: 2), written], axis: 2).reshaped([b, h, nc + 1, p * n])
        let after = matmul(exp(segsum(totals)), states).reshaped([b, h, nc + 1, p, n])  // [B,H,nc+1,P,N]
        let carried = matmul(qc, after[0..., 0..., ..<nc].swappedAxes(-1, -2))          // [B,H,nc,cR,P]
        y = y + (exp(cs).reshaped([b, h, nc, c, 1, 1]) * carried.reshaped([b, h, nc, c, r, p])).reshaped([b, h, nc, c * r, p])
        var yh = y.reshaped([b, h, nc * c, r, p])                                        // [B,H,L',R,P]
        if pad > 0 { yh = yh[0..., 0..., ..<l] }
        let gated = gate([yh, d.reshaped([1, h, 1, 1, 1]), v, z.transposed(0, 2, 1, 3).expandedDimensions(axis: 3),
                          mimoZ.reshaped([1, h, 1, r, p]), mimoO.reshaped([1, h, 1, r, p])])[0]
        return (gated.sum(axis: 3).transposed(0, 2, 1, 3), after[0..., 0..., nc])        // [B,L,H,P]
    }

    /// D, the gate and the output mix, one pass: (y + D v) ⊙ silu(z ⊙ mimo_z) ⊙ mimo_o, summed over the rank after.
    ///
    /// PIN: PRIMITIVE OPS ONLY INSIDE A COMPILED CLOSURE, here and in the mixer's rotation. MLXNN's `silu` is itself
    /// compiled: tracing this closure holds its lock and MLX's eval lock and would wait for `silu`'s, while a model on
    /// another thread holds `silu`'s and waits for the eval lock. Frigate's whole suite deadlocked so (2026-11-19).
    private static let gate: @Sendable ([MLXArray]) -> [MLXArray] = compile { a in
        let g = a[3] * a[4]
        return [(a[0] + a[1] * a[2]) * (g * sigmoid(g)) * a[5]]
    }

    /// One token. q, k [B,R,H,N] rotated and biased; x, z [B,H,P]; adt, dt, lam [B,H]; the state's ssm [B,H,P,N],
    /// kPrev [B,R,H,N], xPrev [B,H,P]. Returns y [B,H,P] and the new SSM state.
    public static func step(q: MLXArray, k: MLXArray, x: MLXArray, adt: MLXArray, dt: MLXArray, lam: MLXArray,
                            mimoX: MLXArray, mimoZ: MLXArray, mimoO: MLXArray, d: MLXArray, z: MLXArray,
                            ssm: MLXArray, kPrev: MLXArray, xPrev: MLXArray) -> (y: MLXArray, ssm: MLXArray) {
        let h = x.dim(1)
        let mx = mimoX.transposed(1, 0, 2)                                               // [R,H,P]
        let v = x.expandedDimensions(axis: 1) * mx                                       // [B,R,H,P]
        let vPrev = xPrev.expandedDimensions(axis: 1) * mx
        let alpha = exp(adt)
        let beta = (1 - lam) * dt * alpha
        let gamma = lam * dt
        let prev = matmul(vPrev.transposed(0, 2, 3, 1), kPrev.transposed(0, 2, 1, 3))     // [B,H,P,N]
        let curr = matmul(v.transposed(0, 2, 3, 1), k.transposed(0, 2, 1, 3))
        let shape = [x.dim(0), h, 1, 1]
        let next = alpha.reshaped(shape) * ssm + beta.reshaped(shape) * prev + gamma.reshaped(shape) * curr
        var y = matmul(q.transposed(0, 2, 1, 3), next.swappedAxes(2, 3)).transposed(0, 2, 1, 3)   // [B,R,H,P]
        y = y + d.reshaped([1, 1, h, 1]) * v
        y = y * silu(z.expandedDimensions(axis: 1) * mimoZ.transposed(1, 0, 2))
        return ((y * mimoO.transposed(1, 0, 2)).sum(axis: 1), next)
    }
}
