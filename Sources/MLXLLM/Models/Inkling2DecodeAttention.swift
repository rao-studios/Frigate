//
//  Inkling2DecodeAttention.swift
//  MLXLLM
//
//  WHAT: One token's attention for Inkling-2, fused: every query head of a key-value group against each key row read
//        once, the relative bias looked up by distance and the hidden keys skipped in the kernel, an online softmax per
//        block of keys, then the blocks and the sink folded per query head. Two layouts: a global visit's keys in
//        position order with the newest last and each row's first key, and a sliding visit's ring, slot s holding the
//        newest position ≡ s (mod W).
//  WHY:  The masked SDPA built a mask per visit — distances, visibility, the bias gathered by distance, −∞ for the
//        hidden keys: ten primitives and a [Hq,S] array before the attention reads a key. 32k decode 36.6 / 36.7
//        tok/s against 35.8 / 34.4 by the SDPA, 1k level (release, 2026-11-21). A decoded token is bound by the GPU
//        (`Inkling2Model.decodeScheduling`): the kernel and the ring's two writes are 2.4 ms of a 1k token's 21.
//  PIN:  THE KEYS ARE READ WHERE THEY ARE. A global visit's keys are a slice of the cache's buffer, its heads a capacity
//        apart; the kernel reads them by their strides (`ensureRowContiguous: false`), never through a copy.
//

import Foundation
import MLX

extension Inkling2Kernels {
    /// How a decoded token's keys are laid out.
    enum DecodeLayout {
        /// A global visit: keys in position order, the newest last; each row's first key [B] (the columns before it are
        /// another row's padding). Distance = S − 1 − column.
        case newestLast(keyStart: MLXArray)
        /// A sliding visit's ring of W slots; each row's position [B]. Distance = (position − slot) mod W; a slot no
        /// position has reached is hidden.
        case ring(positions: MLXArray)
    }

    /// Simdgroups a threadgroup, keys a block (measured 2026-11-21: 4 × 256 was best at both a 1,024 ring and 32k).
    static let decodeSimdgroups = 4
    static let decodeChunk = 256

    static let decodeBlocksSource = """
            constexpr int E = D / 32;
            constexpr int G = HQ / HKV;
            const uint lane = thread_index_in_simdgroup;
            const uint sg = simdgroup_index_in_threadgroup;
            const int kvh = threadgroup_position_in_grid.y;
            const int blocks = sizes[1];
            const int b = threadgroup_position_in_grid.z / blocks;
            const int blk = threadgroup_position_in_grid.z % blocks;
            const int S = sizes[0];
            const int row_value = rows[b];
            const float qs = scale[0];

            float qv[G][E];
            for (int g = 0; g < G; g++) {
                const device T* qp = q + b * q_strides[0] + (kvh * G + g) * q_strides[1] + lane * E;
                for (int i = 0; i < E; i++) qv[g][i] = float(qp[i]) * qs;
            }
            float m[G], l[G], acc[G][E];
            for (int g = 0; g < G; g++) {
                m[g] = -INFINITY; l[g] = 0.0f;
                for (int i = 0; i < E; i++) acc[g][i] = 0.0f;
            }

            const int j0 = blk * CHUNK;
            const int j1 = min(j0 + CHUNK, S);
            const int first = RING ? j0 : max(j0, row_value);
            const device T* kb = k + b * k_strides[0] + kvh * k_strides[1] + lane * E;
            const device T* vb = v + b * v_strides[0] + kvh * v_strides[1] + lane * E;
            const device T* relb = rel + size_t(b) * HQ * R;
            for (int j = first + int(sg); j < j1; j += NS) {
                int dist;
                if (RING) {
                    dist = ((row_value - j) % S + S) % S;
                    if (row_value - dist < 0) continue;
                } else {
                    dist = S - 1 - j;
                }
                const device T* kr = kb + j * k_strides[2];
                const device T* vr = vb + j * v_strides[2];
                float kk[E], vv[E];
                for (int i = 0; i < E; i++) { kk[i] = float(kr[i]); vv[i] = float(vr[i]); }
                for (int g = 0; g < G; g++) {
                    float part = 0.0f;
                    for (int i = 0; i < E; i++) part += qv[g][i] * kk[i];
                    float s = simd_sum(part);
                    if (dist < R) s += float(relb[(kvh * G + g) * R + dist]);
                    const float mn = max(m[g], s);
                    const float c = fast::exp(m[g] - mn);
                    const float p = fast::exp(s - mn);
                    l[g] = l[g] * c + p;
                    for (int i = 0; i < E; i++) acc[g][i] = acc[g][i] * c + p * vv[i];
                    m[g] = mn;
                }
            }

            threadgroup float tm[NS][G];
            threadgroup float tl[NS][G];
            threadgroup float ta[NS][G][D];
            for (int g = 0; g < G; g++) {
                if (lane == 0) { tm[sg][g] = m[g]; tl[sg][g] = l[g]; }
                for (int i = 0; i < E; i++) ta[sg][g][lane * E + i] = acc[g][i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int g = int(sg); g < G; g += NS) {
                float M = -INFINITY;
                for (int s = 0; s < NS; s++) M = max(M, tm[s][g]);
                float L = 0.0f, A[E];
                for (int i = 0; i < E; i++) A[i] = 0.0f;
                if (M > -INFINITY) {
                    for (int s = 0; s < NS; s++) {
                        const float w = fast::exp(tm[s][g] - M);
                        L += tl[s][g] * w;
                        for (int i = 0; i < E; i++) A[i] += ta[s][g][lane * E + i] * w;
                    }
                }
                const int row = (b * HQ + kvh * G + g) * blocks + blk;
                if (lane == 0) { pm[row] = M; pl[row] = L; }
                for (int i = 0; i < E; i++) pa[size_t(row) * D + lane * E + i] = A[i];
            }
        """

    static let decodeFoldSource = """
            constexpr int E = D / 32;
            const uint lane = thread_index_in_simdgroup;
            const int h = threadgroup_position_in_grid.y;
            const int b = threadgroup_position_in_grid.z;
            const int blocks = sizes[1];
            const int base = (b * HQ + h) * blocks;
            const float sink = float(sinks[b * HQ + h]);
            float M = sink;
            for (int i = 0; i < blocks; i++) M = max(M, pm[base + i]);
            float L = fast::exp(sink - M);
            float A[E];
            for (int i = 0; i < E; i++) A[i] = 0.0f;
            for (int i = 0; i < blocks; i++) {
                const float w = fast::exp(pm[base + i] - M);
                L += pl[base + i] * w;
                for (int e = 0; e < E; e++) A[e] += pa[size_t(base + i) * D + lane * E + e] * w;
            }
            device T* op = out + size_t(b * HQ + h) * D + lane * E;
            for (int e = 0; e < E; e++) op[e] = T(A[e] / L);
        """

    static let decodeBlocksKernel = MLXFast.metalKernel(
        name: "inkling2_decode_blocks",
        inputNames: ["q", "k", "v", "rel", "rows", "sizes", "scale"],
        outputNames: ["pm", "pl", "pa"],
        source: decodeBlocksSource,
        ensureRowContiguous: false)

    static let decodeFoldKernel = MLXFast.metalKernel(
        name: "inkling2_decode_fold",
        inputNames: ["pm", "pl", "pa", "sinks", "sizes"],
        outputNames: ["out"],
        source: decodeFoldSource)

    /// What the kernels read of a ring of `keys` slots: their count, and the blocks they are read in. For a count that
    /// never changes; a global visit's grows with the prompt, and `decodeAttention` makes its own.
    static func decodeSizes(keys: Int) -> Inkling2Constant {
        Inkling2Constant([Int32(keys), Int32((keys + decodeChunk - 1) / decodeChunk)])
    }

    /// q [B,Hq,1,D], keys and values [B,Hkv,S,D] (any strides, the last dimension contiguous); rel [B,1,Hq,R]; sinks
    /// [B,1,Hq]; `scale` the attention's, as a list of one; `sizes` is `decodeSizes(keys: S)` where S never changes,
    /// made here otherwise. Returns [B,1,Hq,D].
    static func decodeAttention(
        _ q: MLXArray, _ keys: MLXArray, _ values: MLXArray, rel: MLXArray, sinks: MLXArray, scale: Inkling2Constant,
        sizes: Inkling2Constant? = nil, layout: DecodeLayout
    ) -> MLXArray {
        let (b, hq, d) = (q.dim(0), q.dim(1), q.dim(3))
        let (hkv, s) = (keys.dim(1), keys.dim(2))
        let blocks = (s + decodeChunk - 1) / decodeChunk
        let rows: MLXArray, ring: Bool
        switch layout {
        case .newestLast(let keyStart): (rows, ring) = (keyStart, false)
        case .ring(let positions): (rows, ring) = (positions, true)
        }
        let sizes = sizes?.array ?? MLXArray([Int32(s), Int32(blocks)])
        let template: [(String, any KernelTemplateArg)] = [
            ("T", q.dtype), ("D", d), ("HQ", hq), ("HKV", hkv), ("R", rel.dim(-1)), ("RING", ring),
            ("NS", decodeSimdgroups), ("CHUNK", decodeChunk),
        ]
        let parts = decodeBlocksKernel(
            [q, keys, values, contiguous(rel), contiguous(rows.asType(.int32)), sizes, scale.array],
            template: template,
            grid: (32 * decodeSimdgroups, hkv, b * blocks),
            threadGroup: (32 * decodeSimdgroups, 1, 1),
            outputShapes: [[b * hq * blocks], [b * hq * blocks], [b * hq * blocks * d]],
            outputDTypes: [.float32, .float32, .float32])
        return decodeFoldKernel(
            [parts[0], parts[1], parts[2], sinks, sizes],
            template: [("T", q.dtype), ("D", d), ("HQ", hq)],
            grid: (32, hq, b),
            threadGroup: (32, 1, 1),
            outputShapes: [[b, 1, hq, d]],
            outputDTypes: [q.dtype])[0]
    }
}
