//
//  SSM.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2025/10/01.
//

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/ssm.py

import Foundation
import MLX
import MLXNN

public func computeDt(_ dt: MLXArray, _ dtBias: MLXArray, _ timeStepLimit: (Float, Float))
    -> MLXArray
{
    let dt = softplus(dt.asType(.float32) + dtBias)
    return MLX.clip(dt, min: timeStepLimit.0, max: timeStepLimit.1)
}

private func makeSSMKernel() -> MLXFast.MLXFastKernel? {
    let source = """
            auto n = thread_position_in_grid.z;
            auto h_idx = n % H;
            auto g_idx = n / G;
            constexpr int n_per_t = Ds / 32;

            auto x = X + n * Dh;
            out += n * Dh;
            auto i_state = state_in + n * Dh * Ds;
            auto o_state = state_out + n * Dh * Ds;

            // C and B have shape [batch, group, state_dim]
            // C and B need to be offset by group size
            auto C_ = C + g_idx * Ds;
            auto B_ = B + g_idx * Ds;

            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;

            auto dt_ = static_cast<float>(dt[n]);
            auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
            auto dA = fast::exp(A * dt_);

            float acc = 0.0;
            auto x_ = static_cast<float>(x[d_idx]);

            for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * ds_idx + i;
                auto idx = d_idx * Ds + s_idx;
                auto dB_by_x = x_ * dt_ * static_cast<float>(B_[s_idx]);
                auto state = dA * i_state[idx] + dB_by_x;
                o_state[idx] = static_cast<U>(state);
                acc += state * C_[s_idx];
            }
            acc = simd_sum(acc);
            if (thread_index_in_simdgroup == 0) {
                out[d_idx] = static_cast<T>(acc + x_ * D[h_idx]);
            }
        """

    return MLXFast.metalKernel(
        name: "ssm_kernel",
        inputNames: ["X", "A_log", "B", "C", "D", "dt", "state_in"],
        outputNames: ["out", "state_out"],
        source: source
    )
}

/// The prefill scan as one kernel: the one-token kernel above with the prompt's steps looped inside it, so each lane
/// keeps its Ds/32 state values in registers from the first token to the last. Same recurrence as `ssmAttn`:
///
///     h_t = exp(dt_t * A) * h_{t-1} + dt_t * x_t * B_t        y_t = C_t . h_t + D * x_t
///
/// PIN: `ssmAttn` builds the chunked SSD form instead: per 256-token step and head, a 256 x 256 decay matrix and a
/// "surrogate attention" matrix, a dozen dispatches. On Nemotron 3 Super's layer 0 at 2,048 tokens that was 17.2 ms
/// against this kernel's 4.7 ms, with y within 0.3% and the carried state within 0.2% (bf16), and the model's greedy
/// tokens unchanged (prototyped over mlx-lm, 2026-11-04). The scan is about a quarter of a Mamba-2 layer's prefill;
/// the projections around it are the rest.
private func makeSSMScanKernel() -> MLXFast.MLXFastKernel? {
    let source = """
            auto n = thread_position_in_grid.z;          // batch * H + head
            auto b_idx = n / H;
            auto h_idx = n % H;
            auto g_idx = h_idx / G;                      // G heads share one group of B and C
            constexpr int n_per_t = Ds / 32;
            constexpr int NG = H / G;
            auto ds_idx = thread_position_in_threadgroup.x;
            auto d_idx = thread_position_in_grid.y;

            auto A = -fast::exp(static_cast<float>(A_log[h_idx]));
            auto Dv = static_cast<float>(D[h_idx]);
            float s[n_per_t];
            auto i_state = state_in + (n * Dh + d_idx) * Ds;
            for (int i = 0; i < n_per_t; ++i) { s[i] = static_cast<float>(i_state[n_per_t * ds_idx + i]); }

            for (int t = 0; t < L; ++t) {
                auto row = b_idx * L + t;
                auto dt_ = static_cast<float>(dt[row * H + h_idx]);
                auto dA = fast::exp(A * dt_);
                auto x_ = static_cast<float>(X[(row * H + h_idx) * Dh + d_idx]);
                auto B_ = B + (row * NG + g_idx) * Ds;
                auto C_ = C + (row * NG + g_idx) * Ds;
                float acc = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                    auto s_idx = n_per_t * ds_idx + i;
                    s[i] = dA * s[i] + x_ * dt_ * static_cast<float>(B_[s_idx]);
                    acc += s[i] * static_cast<float>(C_[s_idx]);
                }
                acc = simd_sum(acc);
                if (thread_index_in_simdgroup == 0) {
                    out[(row * H + h_idx) * Dh + d_idx] = static_cast<T>(acc + x_ * Dv);
                }
            }
            auto o_state = state_out + (n * Dh + d_idx) * Ds;
            for (int i = 0; i < n_per_t; ++i) { o_state[n_per_t * ds_idx + i] = static_cast<U>(s[i]); }
        """

    return MLXFast.metalKernel(
        name: "ssm_scan_kernel",
        inputNames: ["X", "A_log", "B", "C", "D", "dt", "state_in"],
        outputNames: ["out", "state_out"],
        source: source
    )
}

private final class SSMKernelManager: Sendable {
    static let shared = SSMKernelManager()

    let ssmKernel: MLXFast.MLXFastKernel?
    /// The fused prefill scan; nil when `FRIGATE_SSM_SCAN=0` asks for `ssmAttn` everywhere (for A/B checks).
    let scanKernel: MLXFast.MLXFastKernel?

    private init() {
        ssmKernel = makeSSMKernel()
        scanKernel = ProcessInfo.processInfo.environment["FRIGATE_SSM_SCAN"] == "0" ? nil : makeSSMScanKernel()
    }
}

/// `ssmAttn`'s contract for one prompt (no padding mask, no per-row lengths) in one dispatch: y `[b, l, h, dh]` in the
/// input's dtype, and the state after the last token. Nil when the shapes do not suit it (the state width must be a
/// multiple of 32, the SIMD width it is spread over) or the kernel is off; the caller then takes `ssmAttn`.
func ssmScanKernel(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray?,
    timeStepLimit: (Float, Float)
) -> (MLXArray, MLXArray)? {
    guard let kernel = SSMKernelManager.shared.scanKernel else { return nil }
    let (b, l, h, dh) = hiddenStates.shape4
    let (g, ds) = (B.dim(-2), B.dim(-1))
    guard ds % 32 == 0, g > 0, h % g == 0 else { return nil }
    let dt = computeDt(dt, dtBias, timeStepLimit)
    let state = state ?? MLXArray.zeros([b, h, dh, ds], dtype: .float32)
    let outputs = kernel(
        [hiddenStates, ALog, B, C, D, dt, state],
        template: [
            ("T", hiddenStates.dtype),
            ("U", state.dtype),
            ("Dh", dh),
            ("Ds", ds),
            ("H", h),
            ("G", h / g),
            ("L", l),
        ],
        grid: (32, dh, h * b),
        threadGroup: (32, 8, 1),
        outputShapes: [hiddenStates.shape, state.shape],
        outputDTypes: [hiddenStates.dtype, state.dtype]
    )
    return (outputs[0], outputs[1])
}

func ssmUpdateKernel(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray,
    timeStepLimit: (Float, Float)
) -> (MLXArray, MLXArray) {
    let (n, _, h, d) = hiddenStates.shape4
    let inputType = hiddenStates.dtype
    let stateType = state.dtype
    let (hb, ds) = (B.dim(-2), B.dim(-1))

    let dt = computeDt(dt, dtBias, timeStepLimit)

    guard let kernel = SSMKernelManager.shared.ssmKernel else {
        fatalError("SSM kernel not available")
    }

    let outputs = kernel(
        [hiddenStates, ALog, B, C, D, dt, state],
        template: [
            ("T", inputType),
            ("U", stateType),
            ("Dh", d),
            ("Ds", ds),
            ("H", h),
            ("G", h / hb),
        ],
        grid: (32, d, h * n),
        threadGroup: (32, 8, 1),
        outputShapes: [[n, 1, h, d], state.shape],
        outputDTypes: [inputType, stateType]
    )

    return (outputs[0], outputs[1])
}

public func segsum(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let l = x.dim(-1)
    var x = x

    if let mask = mask {
        let mask = MLX.expandedDimensions(mask, axis: 1)
        x = x * mask
    }

    x = MLX.repeated(x[.ellipsis, .newAxis], count: l, axis: -1)
    x = MLX.tril(x, k: -1)
    var xSegsum = MLX.cumsum(x, axis: -2)

    if let mask = mask {
        xSegsum = which(
            mask[.ellipsis, .newAxis, 0...] * mask[.ellipsis, .newAxis],
            xSegsum,
            MLXArray(Float(-Float.infinity), dtype: xSegsum.dtype)
        )
    }

    return xSegsum
}

public func ssmAttn(
    x: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    timeStepLimit: (Float, Float) = (0.001, 100.0),
    mask: MLXArray? = nil,
    lengths: MLXArray? = nil,
    step: Int = 256
) -> (MLXArray, MLXArray) {
    let (b, l, h, dh) = x.shape4
    let (_, _, g, d) = B.shape4

    let dt = computeDt(dt, dtBias, timeStepLimit)
    let repeats = h / g
    let A = -MLX.exp(ALog).asType(dt.dtype)
    let dtA = dt * A.reshaped(1, 1, -1)
    let dtx = dt.reshaped(b, l, h, 1) * x
    let stateType = state?.dtype ?? x.dtype

    func stepForward(
        dtx: MLXArray,
        dtA: MLXArray,
        B: MLXArray,
        C: MLXArray,
        state: MLXArray?,
        mask: MLXArray?,
        lengths: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let s = dtx.dim(1)
        var B = MLX.transposed(B, axes: [0, 2, 3, 1])

        var CB = MLX.swappedAxes(C, 1, 2).matmul(B)
        CB = MLX.repeated(CB, count: repeats, axis: 1)

        var decay = MLX.exp(segsum(dtA.swappedAxes(1, 2), mask: mask))

        let surrogateAttentionMatrix = MLX.tril(CB * decay, k: 0)

        var y = surrogateAttentionMatrix.matmul(dtx.swappedAxes(1, 2))
        y = MLX.swappedAxes(y, 1, 2)

        if let lengths {
            var pos = MLX.minimum(lengths, MLXArray(step)) - 1
            pos = MLX.maximum(pos, MLXArray(0))
            pos = expandedDimensions(pos, axes: [1, 2, 3])
            decay = MLX.takeAlong(decay, pos, axis: 2)
        } else {
            decay = decay[0..., 0..., (-1)..., 0...]
        }
        decay = decay.transposed(0, 3, 1, 2)
        B = MLX.repeated(B, count: h / g, axis: 1).swappedAxes(2, 3)
        var dtxdecay = dtx * decay
        dtxdecay = dtxdecay.swappedAxes(1, 2).swappedAxes(2, 3)

        var nextState = dtxdecay.matmul(B)

        if var state {
            let expDtACumsum = MLX.exp(MLX.cumsum(dtA, axis: -2))
            nextState = nextState + expDtACumsum[0..., -1, 0..., .newAxis, .newAxis] * state
            state = state.reshaped(b, 1, g, repeats, dh, d)
            let C = C.reshaped(b, s, g, 1, d, 1)
            let yPrev = (state.matmul(C)).squeezed(axis: -1).flattened(start: 2, end: 3)
            y = y + expDtACumsum[.ellipsis, .newAxis] * yPrev
        }
        if let state, let lengths {
            nextState = MLX.where(
                expandedDimensions(lengths .< 0, axes: [1, 2, 3]), state, nextState)
        }

        return (y.asType(x.dtype), nextState.asType(stateType))
    }

    var ys = [MLXArray]()
    ys.reserveCapacity((l + step - 1) / step)

    var state = state
    var lengths = lengths
    for i in stride(from: 0, to: l, by: step) {
        let end = min(i + step, l)
        let maskChunk = mask == nil ? nil : mask![0..., i ..< end]
        let (y, nextState) = stepForward(
            dtx: dtx[0..., i ..< end],
            dtA: dtA[0..., i ..< end],
            B: B[0..., i ..< end],
            C: C[0..., i ..< end],
            state: state,
            mask: maskChunk,
            lengths: lengths
        )
        ys.append(y)
        state = nextState
        asyncEval(y, nextState)
        if let currentLengths = lengths {
            lengths = currentLengths - step
        }
    }

    let y = MLX.concatenated(ys, axis: 1) + x * D.reshaped(1, 1, h, 1)
    guard let state else {
        fatalError("SSM update did not produce a state")
    }
    eval(y, state)
    return (y, state)
}

public func ssmUpdate(
    hiddenStates: MLXArray,
    ALog: MLXArray,
    B: MLXArray,
    C: MLXArray,
    D: MLXArray,
    dt: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    timeStepLimit: (Float, Float) = (0.001, 100.0),
    mask: MLXArray? = nil,
    lengths: MLXArray? = nil,
    step: Int = 256
) -> (MLXArray, MLXArray) {
    let seqLen = hiddenStates.dim(1)

    if seqLen == 1,
        let state = state,
        SSMKernelManager.shared.ssmKernel != nil
    {
        return ssmUpdateKernel(
            hiddenStates: hiddenStates,
            ALog: ALog,
            B: B,
            C: C,
            D: D,
            dt: dt,
            dtBias: dtBias,
            state: state,
            timeStepLimit: timeStepLimit
        )
    } else if mask == nil, lengths == nil,
        let fused = ssmScanKernel(
            hiddenStates: hiddenStates, ALog: ALog, B: B, C: C, D: D, dt: dt, dtBias: dtBias,
            state: state, timeStepLimit: timeStepLimit)
    {
        // One prompt: the fused scan. A padded batch keeps the chunked form, which masks.
        return fused
    } else {
        return ssmAttn(
            x: hiddenStates,
            ALog: ALog,
            B: B,
            C: C,
            D: D,
            dt: dt,
            dtBias: dtBias,
            state: state,
            timeStepLimit: timeStepLimit,
            mask: mask,
            lengths: lengths,
            step: step
        )
    }
}
