//
//  DecodeSchedulingModel.swift
//  MLXLMCommon
//
//  WHAT: A model that decodes faster under its own GPU scheduling (`GPU.Scheduling`) says which; an engine sets it
//        around each decode step and nowhere else, so prefill and every other model keep MLX's own.
//  WHY:  How many command buffers may run at once, and how much goes in each, helps one model's decode and hurts
//        another's. 64 in flight with no MB limit, measured on an M4 Max in release (`craft bench context`,
//        2026-11-21): Inkling-2-Mini 42.1 → 48.6 tok/s at 1k and 38.0 → 42.7 at 32k, Qwopus 122B +3–5%, Ministral 8B
//        −3% at 1k and −13% at 8k. So a model opts in, with the scheduling measured on it.
//

import MLX

public protocol DecodeSchedulingModel {
    /// The scheduling a decode step on this model runs under.
    var decodeScheduling: GPU.Scheduling { get }
}
