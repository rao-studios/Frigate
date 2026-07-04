//
//  Flux2Scheduler.swift
//  FluxKit
//
//  FlowMatchEulerDiscrete for FLUX.2 Klein: sigmas linspace(1, 1/N) shifted by the
//  empirical mu (dynamic, sequence-length dependent, exponential form), terminal 0
//  appended. Step: x += (σ_next − σ) · v. Ported from mflux/diffusers Klein pipeline.
//

import Foundation
import MLX

public struct Flux2Scheduler {
    /// σ_0 … σ_{N-1}, plus terminal 0 at index N.
    public let sigmas: [Float]
    /// Timesteps fed to the DiT (σ·1000).
    public let timesteps: [Float]

    public init(imageSequenceLength: Int, steps: Int) {
        let mu = Self.empiricalMu(imageSeqLen: imageSequenceLength, numSteps: steps)
        var sig: [Float] = []
        for i in 0..<steps {
            // linspace(1.0, 1/steps, steps)
            let t = steps == 1 ? 1.0 : 1.0 - Double(i) * (1.0 - 1.0 / Double(steps)) / Double(steps - 1)
            sig.append(Float(Self.timeShiftExponential(mu: mu, t: t)))
        }
        self.timesteps = sig.map { $0 * 1000 }
        sig.append(0)
        self.sigmas = sig
    }

    /// x_{i+1} = x_i + (σ_{i+1} − σ_i) · v
    public func step(noise: MLXArray, latents: MLXArray, index: Int) -> MLXArray {
        let dt = sigmas[index + 1] - sigmas[index]
        return latents + dt * noise.asType(latents.dtype)
    }

    static func empiricalMu(imageSeqLen: Int, numSteps: Int) -> Double {
        let a1 = 8.73809524e-05, b1 = 1.89833333
        let a2 = 0.00016927, b2 = 0.45666666
        let seq = Double(imageSeqLen)
        if imageSeqLen > 4300 {
            return a2 * seq + b2
        }
        let m200 = a2 * seq + b2
        let m10 = a1 * seq + b1
        let a = (m200 - m10) / 190.0
        let b = m200 - 200.0 * a
        return a * Double(numSteps) + b
    }

    static func timeShiftExponential(mu: Double, t: Double) -> Double {
        exp(mu) / (exp(mu) + (1.0 / t - 1.0))
    }
}
