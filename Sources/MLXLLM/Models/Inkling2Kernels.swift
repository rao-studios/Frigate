//
//  Inkling2Kernels.swift
//  MLXLLM
//
//  WHAT: mlx-lm's three Metal kernels for Inkling-2 (mlx_lm/models/inkling2/kernels.py, mlx_lm/kernels/expert_routing.py,
//        PR 2086), ported verbatim: causal attention with a relative position bias and a sink per query, attention of
//        learned queries over the depth sources of each token, and the expert routing after the router matmul.
//  PIN:  THE KERNELS ARE THE REFERENCE. With all three the port matches mlx-lm on the GPU; the plain ops in their
//        place change roundings, and on a long, repetitive prompt mlx-lm's own plain path disagrees with its kernels on
//        one greedy choice in nine (measured 2026-11-20, Inkling2LiveTests' PIN). The plain ops stay for the CPU.
//

import Foundation
import MLX

enum Inkling2Kernels {
    /// The kernels run on the GPU; the plain ops elsewhere.
    static var available: Bool { Device.defaultDevice().deviceType == .gpu }

    /// Queries per threadgroup, 8 for each of its 4 simdgroups; threads per threadgroup.
    static let bq = 32
    static let tg = 128

    /// Causal flash attention with a relative position bias and a sink logit per query. mlx-lm kernels.py, verbatim:
    /// the online softmax of each row starts from its sink logit, so the zero-value sink needs no key.
    static let relativeAttentionSource = """
            constexpr float LOG2E = 1.4426950408889634f;
            constexpr int BK = 16;
            constexpr int LDK = BK + 8;
            constexpr int LDV = D + 8;
            const int L = sizes[0];
            const int S = sizes[1];
            const uint tid = thread_position_in_threadgroup.x;
            const uint sg = simdgroup_index_in_threadgroup;
            const uint lane = thread_index_in_simdgroup;
            const int batch = threadgroup_position_in_grid.z;
            const int head = threadgroup_position_in_grid.y;
            const int kv_head = head / (HQ / HKV);
            const int q0 = threadgroup_position_in_grid.x * BQ;
            const int start = key_start[batch];
            const float qscale = scale * LOG2E;

            const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
            const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
            const int row = q0 + int(sg) * 8 + fm;
            const bool row_ok = row < L;
            const int diag = S - L + row;

            threadgroup half Ks[D * LDK];
            threadgroup half Vs[BK * LDV];

            simdgroup_half8x8 q_frag[D / 8];
            for (int db = 0; db < D / 8; ++db) {
                half2 qv = half2(0.0h);
                if (row_ok) {
                    const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                        + row * q_strides[2] + db * 8 + fn;
                    qv = half2(float(qp[0]), float(qp[1]));
                }
                q_frag[db].thread_elements()[0] = qv.x;
                q_frag[db].thread_elements()[1] = qv.y;
            }
            simdgroup_float8x8 acc[D / 8];
            for (int db = 0; db < D / 8; ++db) {
                acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            }
            float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
            float row_sum = 1.0f;
            const int bias_row = batch * L + (row_ok ? row : 0);
            const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

            const int first_diag = S - L + q0;
            const int last_diag = S - L + min(q0 + BQ, L) - 1;
            int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
            first_key = max(first_key, start);
            const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
            const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

            for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (int c = tid; c < BK * D / 8; c += TG) {
                    const int r = c / (D / 8);
                    const int col = (c % (D / 8)) * 8;
                    const int key = k0 + r;
                    vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
                    vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
                    if (key < S) {
                        auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                        auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                        kv[0] = kp[0];
                        kv[1] = kp[1];
                        vv[0] = vp[0];
                        vv[1] = vp[1];
                    }
                    for (int j = 0; j < 8; ++j) {
                        Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                        Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (q0 + int(sg) * 8 >= L) {
                    continue;
                }

                simdgroup_float8x8 scores[BK / 8];
                for (int kb = 0; kb < BK / 8; ++kb) {
                    scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    for (int db = 0; db < D / 8; ++db) {
                        simdgroup_half8x8 k_frag;
                        simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                        simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
                    }
                }

                float tile_max = -INFINITY;
                for (int kb = 0; kb < BK / 8; ++kb) {
                    for (int e = 0; e < 2; ++e) {
                        const int key = k0 + kb * 8 + fn + e;
                        const int d = diag - key;
                        const bool visible = row_ok && d >= 0 && key >= start
                            && (WINDOW == 0 || d < WINDOW);
                        float s = -INFINITY;
                        if (visible) {
                            s = scores[kb].thread_elements()[e] * qscale;
                            if (d < R) {
                                s += float(bias[d]) * LOG2E;
                            }
                        }
                        scores[kb].thread_elements()[e] = s;
                        tile_max = max(tile_max, s);
                    }
                }
                tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
                tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
                const float new_max = max(row_max, tile_max);
                const float factor = fast::exp2(row_max - new_max);
                float tile_sum = 0.0f;
                simdgroup_half8x8 probs[BK / 8];
                for (int kb = 0; kb < BK / 8; ++kb) {
                    for (int e = 0; e < 2; ++e) {
                        const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                        probs[kb].thread_elements()[e] = half(p);
                        tile_sum += p;
                    }
                }
                tile_sum += simd_shuffle_xor(tile_sum, 1);
                tile_sum += simd_shuffle_xor(tile_sum, 8);
                row_sum = row_sum * factor + tile_sum;
                row_max = new_max;
                for (int db = 0; db < D / 8; ++db) {
                    acc[db].thread_elements()[0] *= factor;
                    acc[db].thread_elements()[1] *= factor;
                    for (int kb = 0; kb < BK / 8; ++kb) {
                        simdgroup_half8x8 v_frag;
                        simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                        simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
                    }
                }
            }

            if (row_ok) {
                device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
                for (int db = 0; db < D / 8; ++db) {
                    op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
                    op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
                }
            }
        """

    static let relativeAttentionKernel = MLXFast.metalKernel(
        name: "inkling2_relative_attention",
        inputNames: ["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
        outputNames: ["out"],
        source: relativeAttentionSource,
        ensureRowContiguous: false)

    /// q [B,Hq,L,D], k and v [B,Hkv,S,D] with the queries at the last L keys; rel [B,L,Hq,R]; sinks [B,L,Hq];
    /// keyStart [B], each row's first key. Returns [B,L,Hq,D].
    static func relativeAttention(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, rel: MLXArray, sinks: MLXArray, scale: Float, window: Int,
        keyStart: MLXArray
    ) -> MLXArray {
        let (b, hq, l, d) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let (hkv, s) = (k.dim(1), k.dim(2))
        return relativeAttentionKernel(
            [q, k, v, contiguous(rel), contiguous(sinks.asType(.float32)), contiguous(keyStart.asType(.int32)),
             MLXArray([Int32(l), Int32(s)]), MLXArray(scale)],
            template: [
                ("T", q.dtype), ("D", d), ("HQ", hq), ("HKV", hkv), ("R", rel.dim(-1)), ("WINDOW", window),
                ("BQ", bq), ("TG", tg),
            ],
            grid: ((l + bq - 1) / bq * tg, hq, b),
            threadGroup: (tg, 1, 1),
            outputShapes: [[b, l, hq, d]],
            outputDTypes: [q.dtype])[0]
    }

    // MARK: - Depth attention

    static let depthAttentionSource = """
            constexpr int N = K + 1;
            uint row = threadgroup_position_in_grid.x;
            uint rows = threadgroups_per_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;

            float acc[N * Q];
            for (int i = 0; i < N * Q; i++) {
                acc[i] = 0.0f;
            }
            for (int d = tid; d < D; d += TG) {
                float x[N];
                for (int i = 0; i < K; i++) {
                    x[i] = blocks[(size_t(i) * rows + row) * D + d];
                }
                x[K] = partial[size_t(row) * D + d];
                for (int j = 0; j < Q; j++) {
                    float qv = queries[j * D + d];
                    for (int i = 0; i < N; i++) {
                        acc[i * Q + j] += x[i] * qv;
                    }
                }
            }

            threadgroup float buf[N * Q][TG / 32];
            for (int i = 0; i < N * Q; i++) {
                float v = simd_sum(acc[i]);
                if (lane == 0) {
                    buf[i][sg] = v;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float w[N * Q];
            for (int i = 0; i < N * Q; i++) {
                float t = 0.0f;
                for (int s = 0; s < TG / 32; s++) {
                    t += buf[i][s];
                }
                w[i] = t * scale;
            }
            for (int j = 0; j < Q; j++) {
                float m = -INFINITY;
                for (int i = 0; i < N; i++) {
                    m = max(m, w[i * Q + j]);
                }
                float z = 0.0f;
                for (int i = 0; i < N; i++) {
                    w[i * Q + j] = metal::exp(w[i * Q + j] - m);
                    z += w[i * Q + j];
                }
                for (int i = 0; i < N; i++) {
                    w[i * Q + j] /= z;
                }
            }

            for (int d = tid; d < D; d += TG) {
                float x[N];
                for (int i = 0; i < K; i++) {
                    x[i] = blocks[(size_t(i) * rows + row) * D + d];
                }
                x[K] = partial[size_t(row) * D + d];
                for (int j = 0; j < Q; j++) {
                    float o = 0.0f;
                    for (int i = 0; i < N; i++) {
                        o += w[i * Q + j] * x[i];
                    }
                    out[(size_t(row) * Q + j) * D + d] = o;
                }
            }
        """

    static let depthAttentionKernel = MLXFast.metalKernel(
        name: "inkling2_depth_attention",
        inputNames: ["blocks", "partial", "queries", "scale"],
        outputNames: ["out"],
        source: depthAttentionSource)

    /// Threads for a threadgroup that handles one row of `dim` values.
    static func rowThreads(_ dim: Int) -> Int { min(1024, max(32, dim / 8 / 32 * 32)) }

    /// blocks [K,…,D] and partial […,D], float32, also the values; queries [Q,D] float32. Returns […,Q,D] float32.
    static func depthAttention(
        _ blocks: MLXArray, _ partial: MLXArray, _ queries: MLXArray, scale: Inkling2Constant
    ) -> MLXArray {
        let d = partial.dim(-1), q = queries.dim(0), threads = rowThreads(d)
        return depthAttentionKernel(
            [blocks, partial, queries, scale.array],
            template: [("K", blocks.dim(0)), ("D", d), ("Q", q), ("TG", threads)],
            grid: (partial.size / d * threads, 1, 1),
            threadGroup: (threads, 1, 1),
            outputShapes: [Array(partial.shape.dropLast()) + [q, d]],
            outputDTypes: [.float32])[0]
    }

    // MARK: - Expert routing

    static let routingSource = """
            constexpr int R = (E + 31) / 32;
            uint token = thread_position_in_grid.x / 32;
            uint lane = thread_index_in_simdgroup;
            auto row = logits + size_t(token) * E;

            float score[R];
            float biased[R];
            for (int i = 0; i < R; i++) {
                int e = lane + 32 * i;
                score[i] = 0.0f;
                biased[i] = -INFINITY;
                if (e < E) {
                    float x = row[e] * multiplier;
                    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
                    score[i] = x < 0 ? y : 1.0f - y;
                    biased[i] = score[i] + bias[e];
                }
            }

            float w[K];
            for (int k = 0; k < K; k++) {
                float best = -INFINITY;
                int best_e = E;
                for (int i = 0; i < R; i++) {
                    int e = lane + 32 * i;
                    if (biased[i] > best || (biased[i] == best && e < best_e)) {
                        best = biased[i];
                        best_e = e;
                    }
                }
                float top = simd_max(best);
                int winner = simd_min(best == top ? best_e : E);
                float s = 0.0f;
                for (int i = 0; i < R; i++) {
                    if (lane + 32 * i == winner) {
                        s = score[i];
                        biased[i] = -INFINITY;
                    }
                }
                w[k] = simd_sum(s);
                if (lane == 0) {
                    inds[size_t(token) * K + k] = winner;
                }
            }

            float total = 0.0f;
            for (int k = 0; k < K; k++) {
                total = w[k] + total;
            }
            if (lane == 0) {
                for (int k = 0; k < K; k++) {
                    weights[size_t(token) * K + k] = w[k] / (total + eps) * scale;
                }
            }
        """

    static let routingKernel = MLXFast.metalKernel(
        name: "inkling2_expert_routing",
        inputNames: ["logits", "bias", "multiplier", "eps", "scale"],
        outputNames: ["inds", "weights"],
        source: routingSource)

    /// What the routing kernel reads beside the logits and the bias: the sigmoid's multiplier (1), the epsilon under
    /// the weights' sum, and the scale of the weights.
    struct Routing: Sendable {
        let multiplier = Inkling2Constant(Float(1))
        let eps: Inkling2Constant
        let routeScale: Inkling2Constant

        init(eps: Float, routeScale: Float) {
            self.eps = Inkling2Constant(eps)
            self.routeScale = Inkling2Constant(routeScale)
        }
    }

    /// The top-k experts (uint32) and their weights from the router's float32 logits […,E].
    static func routing(_ logits: MLXArray, bias: MLXArray, topK: Int, constants: Routing) -> (MLXArray, MLXArray) {
        let e = logits.dim(-1)
        let shape = Array(logits.shape.dropLast()) + [topK]
        let out = routingKernel(
            [logits, bias, constants.multiplier.array, constants.eps.array, constants.routeScale.array],
            template: [("E", e), ("K", topK)],
            grid: (logits.size / e * 32, 1, 1),
            threadGroup: (256, 1, 1),
            outputShapes: [shape, shape],
            outputDTypes: [.uint32, .float32])
        return (out[0], out[1])
    }
}
