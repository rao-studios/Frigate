//
//  Inkling2Fused.swift
//  MLXLLM
//
//  WHAT: Inkling-2's chains of small operations compiled into fused kernels: the L2 clips, the attention's output
//        gate, the softcapped SwiGLU, the experts' weighted sum, the message gate's mix and the layer's residual add.
//        Each runs the same operations in the same order and dtypes as the plain code it replaced, so outputs are the
//        same bit for bit (as MuseGlimmer.swift's compiled norms; Inkling2FusedDecodeLiveTests holds it on the
//        checkpoint: 32 decoded tokens, logit diff 0).
//  WHY:  A token was 14,564 primitives (a DOT of one decode step's graph, 2026-11-21); 2,695 were casts and 1,986
//        broadcasts of scalars, most of them inside these chains, and each one the GPU ran was a dispatch waiting on
//        the one before. With the decode convolutions: 7,966 primitives, 1k decode 34.4 → 41.9 tok/s, 32k 30.9 → 37.3
//        (`craft bench context`, release). A decoded token is bound by the GPU (`Inkling2Model.decodeScheduling`).
//  PIN:  NO SHAPES INSIDE. Every function is compiled shapeless, so a reshape, slice or expand stays outside it, done
//        by the caller; constants that differ between callers (the clip thresholds, the caps) arrive as scalar arrays.
//

import Foundation
import MLX

enum Inkling2Fused {
    /// [x, threshold]: the last axis scaled to an L2 norm of at most `threshold`, in x's dtype.
    static let l2Clip: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { a in
        let (x, threshold) = (a[0], a[1])
        let states = x.asType(.float32)
        let norm = sqrt((states * states).sum(axis: -1, keepDims: true))
        return [(states * minimum(threshold / (norm + 1e-6), 1.0)).asType(x.dtype)]
    }

    /// [x, scale, threshold]: `l2Clip` after a per-head scale.
    static let scaledL2Clip: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { a in
        let (x, scale, threshold) = (a[0], a[1], a[2])
        let states = x.asType(.float32) * scale
        let norm = sqrt((states * states).sum(axis: -1, keepDims: true))
        return [(states * minimum(threshold / (norm + 1e-6), 1.0)).asType(x.dtype)]
    }

    /// [x, tau]: x scaled by tau in float32, back in x's dtype.
    static let scaled: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { a in
        [(a[0].asType(.float32) * a[1]).asType(a[0].dtype)]
    }

    /// [out, gate]: the attention's output times the sigmoid of its gate, in the gate's dtype.
    static let outputGate: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { a in
        let (out, gate) = (a[0], a[1])
        return [(out.asType(.float32) * sigmoid(gate.asType(.float32))).asType(gate.dtype)]
    }

    /// [gate, up, gateCap, upCap]: the SwiGLU of capped gate and up; the sigmoid is of the uncapped gate.
    static let softcappedSwiGLU: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { a in
        let (gate, up, gateCap, upCap) = (a[0], a[1], a[2], a[3])
        let g = gate.asType(.float32), u = up.asType(.float32)
        let gc = tanh(g / gateCap) * gateCap
        let uc = tanh(u / upCap) * upCap
        return [(gc * sigmoid(g) * uc).asType(gate.dtype)]
    }

    /// [states [B,L,k,D], weights [B,L,k,1]]: the experts' outputs summed by their weights, in float32, then back.
    static let combine: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { a in
        let (states, weights) = (a[0], a[1])
        return [(states.asType(.float32) * weights).sum(axis: -2).asType(states.dtype)]
    }

    /// [forget, gain, x, message], the projections before their tanh: (1 − tanh forget)·x + tanh gain·message.
    static let messageGate: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { a in
        let (forget, gain, x, message) = (a[0], a[1], a[2], a[3])
        return [(1 - tanh(forget)) * x + tanh(gain) * message]
    }

    /// [residual float32, z]: their sum, in z's dtype.
    static let residual: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true) { a in
        [(a[0] + a[1].asType(.float32)).asType(a[1].dtype)]
    }
}
