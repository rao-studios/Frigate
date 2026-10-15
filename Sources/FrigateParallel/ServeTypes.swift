//  WHAT: The engine's wire: what a caller asks for (a prompt, how to sample it, when to stop, which
//        session's cache to continue), what it gets back (admission, tokens, the end), and the
//        stats a front end shows. Values only — no array crosses the engine's thread.

import Foundation
import MLXLMCommon

public struct StreamID: Hashable, Sendable, CustomStringConvertible {
    public let raw: Int
    public init(raw: Int) { self.raw = raw }
    public var description: String { "s\(raw)" }
}

/// A session's resident cache in the engine: a request that names one continues it.
public struct SlotID: Hashable, Sendable {
    public let raw: UUID
    public init() { raw = UUID() }
}

public enum StreamPriority: Int, Sendable, Comparable {
    /// A worker, or anything else no one is waiting on.
    case background = 0
    /// The turn, and the asides it waits for.
    case interactive = 1
    public static func < (a: Self, b: Self) -> Bool { a.rawValue < b.rawValue }
}

public struct StreamRequest: Sendable {
    /// The whole prompt. What the slot already holds is not prefilled again.
    public var tokens: [Int]
    public var sampling: GenerateParameters
    public var maxTokens: Int
    public var stopTokens: Set<Int>
    public var priority: StreamPriority
    public var label: String
    /// The session's cache to continue and leave for the next round; nil runs on a cache of its own.
    public var slot: SlotID?

    public init(tokens: [Int], sampling: GenerateParameters, maxTokens: Int, stopTokens: Set<Int>,
                priority: StreamPriority = .interactive, label: String = "", slot: SlotID? = nil) {
        self.tokens = tokens; self.sampling = sampling; self.maxTokens = maxTokens; self.stopTokens = stopTokens
        self.priority = priority; self.label = label; self.slot = slot
    }
}

public enum FinishReason: String, Sendable {
    case stop, maxTokens, removed, failed
}

public struct FinishInfo: Sendable {
    public var reason: FinishReason
    public var promptTokens: Int
    /// Tokens the stream produced, the stop token included.
    public var generated: Int
    public var prefillSeconds: TimeInterval
    public var decodeSeconds: TimeInterval
    /// Rows in the batch, averaged over this stream's steps.
    public var meanBatchSize: Double
    public var message: String?
}

public enum StreamEvent: Sendable {
    /// The prompt is in the cache and the first token is on its way.
    case admitted(prefixReused: Int, prefilled: Int, queueWait: TimeInterval, prefillSeconds: TimeInterval)
    case token(Int)
    case finished(FinishInfo)
}

public struct StreamHandle: Sendable {
    public let id: StreamID
    public let events: AsyncThrowingStream<StreamEvent, Error>
}

public enum ServeError: Error, CustomStringConvertible {
    case slotBusy(String)
    case unsupportedLayer(String)
    case shutDown

    public var description: String {
        switch self {
        case .slotBusy(let label): return "the session's cache is still in use by \(label)"
        case .unsupportedLayer(let kind): return "the serve engine cannot batch a \(kind) layer cache"
        case .shutDown: return "the serve engine is shut down"
        }
    }
}

// MARK: - Stats

public enum StreamPhase: Sendable, Equatable {
    case queued
    case prefill(done: Int, total: Int)
    case decode
}

public struct StreamStat: Sendable {
    public var id: StreamID
    public var label: String
    public var phase: StreamPhase
    public var promptTokens: Int
    public var generated: Int
    public var tokensPerSecond: Double
}

public struct ServeStats: Sendable {
    public var streams: [StreamStat] = []
    /// Rows decoding together right now.
    public var batchSize = 0
    public var queued = 0
    /// The last decode step, in milliseconds.
    public var stepMs: Double = 0
    /// Tokens every row produced over the last second or so.
    public var aggregateTokensPerSecond: Double = 0
    public var at = Date()
    public init() {}
}
