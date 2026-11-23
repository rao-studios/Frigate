//
//  Inkling2DecodeConvolution.swift
//  MLXLLM
//
//  WHAT: A decoded token's convolutions for Inkling-2, one dispatch each: the short convolution (four a visit) and the
//        global layers' dynamic key and value convolution. Each reads the kept state and the token and writes both the
//        output and the next state (the state's last K−2 rows, then the token in float32).
//  WHY:  The plain ops spend ~13 primitives a short convolution and ~25 a dynamic one, the copy that keeps the state
//        included: ~2,700 of a token's 14,564 (2026-11-21), most of them dispatches the GPU ran one after another
//        (a decoded token is bound by the GPU: `Inkling2Model.decodeScheduling`).
//  PIN:  THE SAME ROUNDINGS AS THE PLAIN OPS. The short convolution rounds the window to the weight's dtype and the sum
//        to it again, as `conv1d` in the weight's dtype does; the dynamic one stays in float32 throughout, as its ops do.
//

import Foundation
import MLX

extension Inkling2Kernels {
    /// out = (conv(window) + x) / √(K+1); state' = window[1...]. Grid (C, B).
    static let shortConvolutionSource = """
            const int c = thread_position_in_grid.x;
            const int b = thread_position_in_grid.y;
            const int C = sizes[0];
            if (c >= C) return;
            constexpr int N = K - 1;
            const float xv = float(x[b * C + c]);
            float acc = 0.0f;
            for (int k = 0; k < N; k++) {
                const float s = state[(b * N + k) * C + c];
                acc += float(W(s)) * float(w[c * K + k]);
                if (k > 0) next[(b * N + k - 1) * C + c] = s;
            }
            acc += float(W(xv)) * float(w[c * K + N]);
            next[(b * N + N - 1) * C + c] = xv;
            out[b * C + c] = T((float(W(acc)) + xv) / divisor[0]);
        """

    static let shortConvolutionKernel = MLXFast.metalKernel(
        name: "inkling2_decode_short_conv",
        inputNames: ["x", "state", "w", "sizes", "divisor"],
        outputNames: ["out", "next"],
        source: shortConvolutionSource)

    /// x [B,1,C], state [B,K−1,C] float32, weight [C,K,1]; `sizes` is [C] and `divisor` [√(K+1)], the convolution's
    /// own. Returns the output [B,1,C] in x's dtype and the next state.
    static func shortConvolution(
        _ x: MLXArray, state: MLXArray, weight: MLXArray, kernel: Int, sizes: Inkling2Constant, divisor: Inkling2Constant
    ) -> (MLXArray, MLXArray) {
        let (b, c) = (x.dim(0), x.dim(2))
        let parts = shortConvolutionKernel(
            [x, state, weight, sizes.array, divisor.array],
            template: [("T", x.dtype), ("W", weight.dtype), ("K", kernel)],
            grid: (c, b, 1),
            threadGroup: (min(256, c), 1, 1),
            outputShapes: [[b, 1, c], [b, kernel - 1, c]],
            outputDTypes: [x.dtype, .float32])
        return (parts[0], parts[1])
    }

    /// out = (Σ_lag window[K−1−lag] · (static[lag] + dynamic[head, lag]) + x) / √(2K+1); state' = window[1...].
    static let dynamicConvolutionSource = """
            const int c = thread_position_in_grid.x;
            const int b = thread_position_in_grid.y;
            const int C = sizes[0];
            const int H = sizes[1];
            if (c >= C) return;
            constexpr int N = K - 1;
            const int h = c / (C / H);
            const float xv = float(x[b * C + c]);
            float acc = xv * (float(fixed[c]) + float(dyn[b * H * K + h * K]));
            for (int lag = 1; lag < K; lag++) {
                const float s = state[(b * N + N - lag) * C + c];
                acc += s * (float(fixed[lag * C + c]) + float(dyn[b * H * K + h * K + lag]));
            }
            for (int k = 1; k < N; k++) next[(b * N + k - 1) * C + c] = state[(b * N + k) * C + c];
            next[(b * N + N - 1) * C + c] = xv;
            out[b * C + c] = T((acc + xv) / divisor[0]);
        """

    static let dynamicConvolutionKernel = MLXFast.metalKernel(
        name: "inkling2_decode_dynamic_conv",
        inputNames: ["x", "state", "fixed", "dyn", "sizes", "divisor"],
        outputNames: ["out", "next"],
        source: dynamicConvolutionSource)

    /// x [B,1,C], state [B,K−1,C] float32, the static weight [K,C], the dynamic projection [B,1,H·K]; `sizes` is
    /// [C, H] and `divisor` [√(2K+1)], the convolution's own. Returns the output [B,1,C] in x's dtype and the next state.
    static func dynamicConvolution(
        _ x: MLXArray, state: MLXArray, fixed: MLXArray, dynamic: MLXArray, kernel: Int, sizes: Inkling2Constant,
        divisor: Inkling2Constant
    ) -> (MLXArray, MLXArray) {
        let (b, c) = (x.dim(0), x.dim(2))
        let parts = dynamicConvolutionKernel(
            [x, state, fixed, dynamic, sizes.array, divisor.array],
            template: [("T", x.dtype), ("K", kernel)],
            grid: (c, b, 1),
            threadGroup: (min(256, c), 1, 1),
            outputShapes: [[b, 1, c], [b, kernel - 1, c]],
            outputDTypes: [x.dtype, .float32])
        return (parts[0], parts[1])
    }
}
