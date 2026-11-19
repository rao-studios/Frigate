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
//        Chunked (the reference's `mamba3_MIMO_chunk_ref`): a key carries γ_s plus the next step's
//        (1 − λ_{s+1})Δ_{s+1} under the decay from s, the diagonal subtracts that second term where it has not
//        happened yet, and the rank is flattened into the chunk's score matrix. Plain MLX ops, no kernels.
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

    /// The chunked pass. q, k [B,L,R,H,N] rotated and biased; x [B,L,H,P]; adt, dt, lam [B,L,H] (AΔ ≤ 0, Δ, σ(trap));
    /// mimoX, mimoZ, mimoO [H,R,P]; d [H]; z [B,L,H,P]; ssm [B,H,P,N], kPrev [B,R,H,N] and xPrev [B,H,P] the state
    /// before the sequence or nil. Returns y [B,L,H,P] and the SSM state after the last position.
    public static func chunked(q: MLXArray, k: MLXArray, x: MLXArray, adt: MLXArray, dt: MLXArray, lam: MLXArray,
                               mimoX: MLXArray, mimoZ: MLXArray, mimoO: MLXArray, d: MLXArray, z: MLXArray,
                               ssm: MLXArray?, kPrev: MLXArray?, xPrev: MLXArray?, chunk: Int) -> (y: MLXArray, ssm: MLXArray) {
        let (b, l, r, h, n) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3), q.dim(4))
        let p = x.dim(3)
        let mx = mimoX.transposed(1, 0, 2)                                               // [R,H,P]
        let v = x.expandedDimensions(axis: 2) * mx                                       // [B,L,R,H,P]
        // The trapezoid: γ_s at s, and the next step's (1 − λ)Δ on the same key after it (zero at the end).
        let gamma = lam * dt                                                             // [B,L,H]
        let nextTerm = concatenated([(dt * (1 - lam))[0..., 1...], MLXArray.zeros([b, 1, h], dtype: .float32)], axis: 1)
        let factor = gamma + nextTerm                                                    // [B,L,H]
        let kw = k * factor.reshaped([b, l, 1, h, 1])                                   // [B,L,R,H,N]

        // The state the sequence enters with: the carried SSM state, plus the first position's β term on the
        // carried key and value (β_0 / α_0 = (1 − λ_0)Δ_0, since the carry-in decays it by α_0).
        var s = ssm?.asType(.float32) ?? MLXArray.zeros([b, h, p, n], dtype: .float32)
        if let kPrev, let xPrev {
            let vPrev = xPrev.expandedDimensions(axis: 1) * mx                             // [B,R,H,P]
            let outer = matmul(vPrev.transposed(0, 2, 3, 1), kPrev.transposed(0, 2, 1, 3))   // [B,H,P,N]
            let w = (dt[0..., 0] * (1 - lam[0..., 0])).reshaped([b, h, 1, 1])
            s = s + w * outer
        }

        var ys: [MLXArray] = []
        var start = 0
        while start < l {
            let c = min(chunk, l - start)
            let range = start ..< start + c
            let dtAc = adt[0..., range].swappedAxes(1, 2).contiguous()                  // [B,H,c], row-contiguous (see PIN)
            let cs = runningSum(dtAc, axis: -1)
            let decay = exp(segsum(dtAc))                                                // [B,H,c,c]
            // Rows and columns run over (position, rank), position-major, as the reference flattens them.
            let qc = q[0..., range].transposed(0, 3, 1, 2, 4).reshaped([b, h, c * r, n])
            let kc = kw[0..., range].transposed(0, 3, 1, 2, 4).reshaped([b, h, c * r, n])
            let vc = v[0..., range].transposed(0, 3, 1, 2, 4).reshaped([b, h, c * r, p])
            let decayR = repeated(repeated(decay, count: r, axis: 2), count: r, axis: 3)  // [B,H,cR,cR]
            var y = matmul(matmul(qc, kc.swappedAxes(2, 3)) * decayR, vc)              // [B,H,cR,P]
            let decayInR = repeated(exp(cs), count: r, axis: 2).expandedDimensions(axis: 3)   // [B,H,cR,1]
            y = y + decayInR * matmul(qc, s.swappedAxes(2, 3))                          // carry-in through each position
            let toEnd = exp(cs[0..., 0..., (c - 1)...] - cs)                              // [B,H,c]
            let toEndR = repeated(toEnd, count: r, axis: 2).expandedDimensions(axis: 3)  // [B,H,cR,1]
            let through = exp(cs[0..., 0..., c - 1]).reshaped([b, h, 1, 1])
            s = through * s + matmul((vc * toEndR).swappedAxes(2, 3), kc)               // [B,H,P,N]
            ys.append(y.reshaped([b, h, c, r, p]).transposed(0, 2, 3, 1, 4))            // [B,c,R,H,P]
            start += c
        }
        var y = concatenated(ys, axis: 1)                                                // [B,L,R,H,P]
        // The diagonal: the next step's term on a key has not happened at its own position.
        let qk = matmul(q.transposed(0, 1, 3, 2, 4), k.transposed(0, 1, 3, 4, 2))        // [B,L,H,R',R]
        let own = matmul(qk, v.transposed(0, 1, 3, 2, 4)).transposed(0, 1, 3, 2, 4)      // [B,L,R',H,P]
        y = y - nextTerm.reshaped([b, l, 1, h, 1]) * own
        y = y + d.reshaped([1, 1, 1, h, 1]) * v
        y = y * silu(z.expandedDimensions(axis: 2) * mimoZ.transposed(1, 0, 2))
        return ((y * mimoO.transposed(1, 0, 2)).sum(axis: 2), s)                         // [B,L,H,P]
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
