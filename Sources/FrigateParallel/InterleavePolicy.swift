//  WHAT: How the engine's one thread shares itself between the prompt it is reading and the rows it is decoding:
//        a decode step next, or a chunk of the prompt, and how many of its tokens.
//  IN:   ParallelEngine, once a pass: whose prompt is being read, whose rows decode, and what the last steps and
//        chunks took (`Clock`).
//  OUT:  `.step` or `.chunk(tokens)`.
//  PIN:  A SHARE OF TIME, NOT A COUNT OF TURNS. The engine alternated one decode step and one chunk, so a row decoding
//        beside a prompt got one token per chunk: on a 122B mixture, a step of about 50 ms against a 256-token chunk of
//        1 to 2.4 s, under a token a second for as long as the prompt took. A prompt now takes a share of the engine's
//        time: decode steps run until the rows have had (1 − s) / s of the last chunk's time, then one chunk runs, sized
//        to the time that keeps the share. Nothing here is tuned to a model; the times are the engine's own, as it runs.
//  PIN:  WHOSE WORK SETS THE SHARE. A prompt someone waits on more than on the rows (the turn's, beside its agents)
//        keeps the engine as it always had it: one step, then a whole chunk. Beside rows of its own priority a prompt
//        takes `even`; a prompt no one waits on beside rows someone does (an agent's prompt, a warm, beside the turn's
//        reply) takes `yielding`, so the reply the person reads keeps most of its rate.

import Foundation

public struct InterleavePolicy: Sendable, Equatable {
    public enum Next: Equatable, Sendable {
        case step
        case chunk(Int)
    }

    /// The prompt's share of the engine's time beside rows of its own priority.
    public var even: Double
    /// The prompt's share beside rows someone waits on more than on it.
    public var yielding: Double
    /// The smallest chunk a prompt beside decoding rows is read in: under it a chunk stops paying for the weights it reads.
    public var minChunk: Int
    /// The largest chunk, and the one a prompt is read in when no row decodes beside it.
    public var maxChunk: Int
    /// Decode steps the rows take between two chunks; the chunk is sized so the share holds.
    public var burst: Double = 4

    public init(even: Double = 0.5, yielding: Double = 0.2, minChunk: Int = 32, maxChunk: Int = 256) {
        self.even = Self.bounded(even)
        self.yielding = Self.bounded(yielding)
        self.maxChunk = max(1, maxChunk)
        self.minChunk = min(max(1, minChunk), self.maxChunk)
    }

    /// A share the rule can hold: past 1 the prompt has the engine as it always had it, and under a twentieth it would
    /// never be read.
    static func bounded(_ share: Double) -> Double { min(1, max(0.05, share)) }

    /// What the engine's time has gone to, as the engine measured it.
    public struct Clock: Sendable, Equatable {
        /// Decode steps since the last chunk, and the seconds they took.
        public var stepsSinceChunk: Int
        public var decodeSecondsSinceChunk: TimeInterval
        /// What the last chunk took, and a decode step, averaged.
        public var chunkSeconds: TimeInterval
        public var stepSeconds: TimeInterval
        /// What a chunk reads a second, averaged; nil until one has been read.
        public var prefillTokensPerSecond: Double?

        public init(stepsSinceChunk: Int = 0, decodeSecondsSinceChunk: TimeInterval = 0, chunkSeconds: TimeInterval = 0,
                    stepSeconds: TimeInterval = 0, prefillTokensPerSecond: Double? = nil) {
            self.stepsSinceChunk = stepsSinceChunk
            self.decodeSecondsSinceChunk = decodeSecondsSinceChunk
            self.chunkSeconds = chunkSeconds
            self.stepSeconds = stepSeconds
            self.prefillTokensPerSecond = prefillTokensPerSecond
        }

        /// How much a new reading moves an average.
        static let weight = 0.3

        /// A decode step took `seconds`. The first after a chunk stays out of the step's average: what it waited on
        /// finished while the chunk ran, so it reads short.
        public mutating func stepped(seconds: TimeInterval) {
            if stepsSinceChunk > 0 {
                stepSeconds = stepSeconds > 0 ? stepSeconds + Self.weight * (seconds - stepSeconds) : seconds
            }
            stepsSinceChunk += 1
            decodeSecondsSinceChunk += seconds
        }

        /// A chunk read `tokens` in `seconds`: the rows' time starts again from nothing.
        public mutating func read(tokens: Int, seconds: TimeInterval) {
            chunkSeconds = seconds
            if tokens > 0, seconds > 0 {
                let rate = Double(tokens) / seconds
                prefillTokensPerSecond = prefillTokensPerSecond.map { $0 + Self.weight * (rate - $0) } ?? rate
            }
            stepsSinceChunk = 0
            decodeSecondsSinceChunk = 0
        }
    }

    /// The share of the engine's time a prompt of `prompt`'s priority takes beside rows whose highest priority is
    /// `rows`: all of it when it outranks them, `even` beside its own, `yielding` beside rows someone waits on more.
    /// Nil with no rows decoding.
    public func share(prompt: StreamPriority, rows: StreamPriority?) -> Double? {
        guard let rows else { return nil }
        if prompt > rows { return 1 }
        return prompt == rows ? even : yielding
    }

    /// What the engine does next while `prompt`'s prompt is being read beside rows whose highest priority is `rows`.
    public func next(prompt: StreamPriority, rows: StreamPriority?, clock: Clock) -> Next {
        guard let share = share(prompt: prompt, rows: rows) else { return .chunk(maxChunk) }
        // Every chunk is followed by one step at least: no row decoding ever waits two chunks in a row.
        if clock.stepsSinceChunk == 0 { return .step }
        if share >= 1 { return .chunk(maxChunk) }
        if clock.decodeSecondsSinceChunk < clock.chunkSeconds * (1 - share) / share { return .step }
        guard let rate = clock.prefillTokensPerSecond, rate > 0, clock.stepSeconds > 0 else { return .chunk(minChunk) }
        // The time a chunk may take so that `burst` steps between two of them hold the share.
        let seconds = burst * clock.stepSeconds * share / (1 - share)
        return .chunk(min(maxChunk, max(minChunk, Int(rate * seconds))))
    }
}
