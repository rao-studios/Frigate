//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Compute G

func computeGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    return decay
}

// MARK: - Metal Kernel

private func makeGatedDeltaKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * g_[hv_idx];
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              } else {
                y[dv_idx] = static_cast<InT>(0);
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "g", "beta", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = hasMask ? "_mask" : ""

    return MLXFast.metalKernel(
        name: "gated_delta_step\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let kernel: MLXFast.MLXFastKernel?
    let kernelMasked: MLXFast.MLXFastKernel?

    private init() {
        kernel = makeGatedDeltaKernel(hasMask: false)
        kernelMasked = makeGatedDeltaKernel(hasMask: true)
    }
}

// MARK: - Kernel Dispatch

func gatedDeltaKernel(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype
    let stateType = state.dtype

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(T)]
    if let mask {
        selectedKernel = GatedDeltaKernelManager.shared.kernelMasked
        inputs.append(mask)
    } else {
        selectedKernel = GatedDeltaKernelManager.shared.kernel
    }

    guard let kernel = selectedKernel else {
        fatalError("Gated delta kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("StT", stateType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, stateType]
    )

    return (outputs[0], outputs[1])
}

// MARK: - Ops Fallback

private func gatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state = state + expandedDimensions(k, axis: -2) * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: -2)).sum(axis: -1)

    if let mask {
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = expandedDimensions(mask, axes: [1, 2, 3])
        } else if mask.ndim == 2 {
            expandedMask = expandedDimensions(mask, axes: [2, 3])
        } else if mask.ndim == 3 {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        state = MLX.where(expandedMask, state, oldState)
    }

    return (y.asType(q.dtype), state)
}

func gatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var q = q
    var k = k

    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: -2)
        k = repeated(k, count: repeatFactor, axis: -2)
    }

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)

    var ys = [MLXArray]()
    ys.reserveCapacity(T)

    for t in 0 ..< T {
        let qT = q[0..., t]
        let kT = k[0..., t]
        let vT = v[0..., t]
        let gT = g[0..., t]
        let betaT = beta[0..., t]
        let maskT = mask == nil ? nil : mask![0..., t]

        let (y, newState) = gatedDeltaStepOps(
            q: qT,
            k: kT,
            v: vT,
            g: gT,
            beta: betaT,
            state: state,
            mask: maskT
        )
        ys.append(y)
        state = newState
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

// MARK: - Chunked (differentiable)

/// Where the Gated DeltaNet layers run differentiably: inside `$differentiable.withValue(true)`, an unmasked update runs
/// `gatedDeltaChunked` instead of the Metal kernel, which has no gradient. What a bridge trained through a frozen
/// Qwen3.5 or Qwen3-Next reads its loss back through (`SoftPromptModel.takesInputGradients`).
public enum GatedDeltaPath {
    @TaskLocal public static var differentiable = false
}

/// `(I - n)⁻¹` for `n` strictly lower triangular in its last two axes (a power of two wide), by halves: the inverse of
/// `[[A, 0], [B, D]]` is `[[A⁻¹, 0], [-D⁻¹BA⁻¹, D⁻¹]]`. Matmuls only, so autograd keeps a few small products rather than
/// every row of a forward substitution.
func unitLowerInverse(_ n: MLXArray) -> MLXArray {
    let c = n.dim(-1)
    if c == 1 { return MLXArray.ones(n.shape, dtype: n.dtype) }
    let h = c / 2
    let a = unitLowerInverse(n[.ellipsis, ..<h, ..<h])
    let d = unitLowerInverse(n[.ellipsis, h..., h...])
    let b = matmul(matmul(d, n[.ellipsis, h..., ..<h]), a)
    var zeroShape = a.shape
    zeroShape[zeroShape.count - 1] = c - h
    let top = concatenated([a, MLXArray.zeros(zeroShape, dtype: a.dtype)], axis: -1)
    return concatenated([top, concatenated([b, d], axis: -1)], axis: -2)
}

/// The Gated DeltaNet recurrence of `gatedDeltaOps`, 64 positions a chunk (the WY form): within a chunk the delta rule
/// is a unit lower-triangular solve and matmuls, and only the state at each chunk's end is carried. The same y and
/// state as the kernel to float32 rounding (GatedDeltaChunkedTests), at a cost autograd can keep: one state a chunk,
/// not one a position. `logG` is the decay's log (`computeGatedDeltaG` before its `exp`); q and k arrive normed and
/// scaled, as the layer passes them to the kernel. No mask: a training batch is one unpadded row.
func gatedDeltaChunked(
    q: MLXArray, k: MLXArray, v: MLXArray, logG: MLXArray, beta: MLXArray, state: MLXArray, chunk: Int = 64
) -> (MLXArray, MLXArray) {
    let B = q.dim(0), T = q.dim(1), Hk = q.dim(2), Dk = q.dim(3), Hv = v.dim(2), Dv = v.dim(3)
    let outType = q.dtype
    var q = q.asType(.float32), k = k.asType(.float32)
    if Hv / Hk > 1 {
        q = repeated(q, count: Hv / Hk, axis: -2)
        k = repeated(k, count: Hv / Hk, axis: -2)
    }
    // [B, H, T, D], padded to whole chunks: a padded position has k = 0, β = 0 and log g = 0, so it moves nothing.
    let pad = (chunk - T % chunk) % chunk
    func heads(_ x: MLXArray) -> MLXArray {
        let x = x.transposed(0, 2, 1, 3)
        return pad == 0 ? x : padded(x, widths: [0, 0, [0, pad], 0] as [IntOrPair])
    }
    func gates(_ x: MLXArray) -> MLXArray {
        let x = x.asType(.float32).transposed(0, 2, 1)
        return pad == 0 ? x : padded(x, widths: [0, 0, [0, pad]] as [IntOrPair])
    }
    let n = (T + pad) / chunk
    let qh = heads(q).reshaped(B, Hv, n, chunk, Dk)
    let kh = heads(k).reshaped(B, Hv, n, chunk, Dk)
    let vh = heads(v.asType(.float32)).reshaped(B, Hv, n, chunk, Dv)
    let bh = gates(beta).reshaped(B, Hv, n, chunk)
    let g = cumsum(gates(logG).reshaped(B, Hv, n, chunk), axis: -1)

    let rows = MLXArray(0 ..< Int32(chunk)).reshaped(chunk, 1)
    let cols = MLXArray(0 ..< Int32(chunk)).reshaped(1, chunk)
    let lowerWithDiagonal = rows .>= cols
    let strictlyLower = rows .> cols
    // decay[i, j] = exp(g_i - g_j) on and below the diagonal, 0 above.
    let decay = exp(MLX.where(lowerWithDiagonal, expandedDimensions(g, axis: -1) - expandedDimensions(g, axis: -2), MLXArray(Float(0))))
        * lowerWithDiagonal.asType(.float32)
    let kBeta = kh * expandedDimensions(bh, axis: -1)
    let vBeta = vh * expandedDimensions(bh, axis: -1)
    let lower = -(matmul(kBeta, kh.swappedAxes(-1, -2)) * decay) * strictlyLower.asType(.float32)
    let solve = unitLowerInverse(lower)
    let values = matmul(solve, vBeta)
    let kCumDecay = matmul(solve, kBeta * expandedDimensions(exp(g), axis: -1))

    // The state as [B, H, Dk, Dv] (kᵀv), the layer's [B, H, Dv, Dk] transposed.
    var m = state.asType(.float32).swappedAxes(-1, -2)
    var outs: [MLXArray] = []
    outs.reserveCapacity(n)
    for i in 0 ..< n {
        let qi = qh[0..., 0..., i], ki = kh[0..., 0..., i], gi = g[0..., 0..., i]
        let attn = matmul(qi, ki.swappedAxes(-1, -2)) * decay[0..., 0..., i]
        let vNew = values[0..., 0..., i] - matmul(kCumDecay[0..., 0..., i], m)
        outs.append(matmul(qi * expandedDimensions(exp(gi), axis: -1), m) + matmul(attn, vNew))
        let last = gi[0..., 0..., (chunk - 1)...]
        m = m * expandedDimensions(exp(last), axis: -1)
            + matmul((ki * expandedDimensions(exp(last - gi), axis: -1)).swappedAxes(-1, -2), vNew)
    }
    let y = stacked(outs, axis: 2).reshaped(B, Hv, n * chunk, Dv)[0..., 0..., ..<T].transposed(0, 2, 1, 3)
    return (y.asType(outType), m.swappedAxes(-1, -2))
}

// MARK: - Public API

public func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let beta = sigmoid(b).asType(.float32)
    let g = computeGatedDeltaG(aLog, a, dtBias)

    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    // State kept in fp32 to match Python mlx-lm. Using q.dtype (bf16) loses
    // precision across T-step recurrence, compounding rounding error.
    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    if state.dtype != .float32 {
        state = state.asType(.float32)
    }

    if GatedDeltaPath.differentiable, mask == nil {
        let logG = -exp(aLog.asType(.float32)) * softplus(a + dtBias)
        return gatedDeltaChunked(q: q, k: k, v: v, logG: logG, beta: beta, state: state)
    }

    if GatedDeltaKernelManager.shared.kernel != nil {
        return gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}
