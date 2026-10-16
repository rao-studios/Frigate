//  WHAT: The engine's wire: what a caller asks for (a prompt, how to sample it, when to stop, which
//        session's cache to continue), what it gets back (admission, tokens, the end), and the
//        stats a front end shows. Values only — no array crosses the engine's thread.

import Foundation
import MLX
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

public enum ParallelError: Error, CustomStringConvertible {
    case slotBusy(String)
    case unsupportedLayer(String)
    case shutDown

    public var description: String {
        switch self {
        case .slotBusy(let label): return "the session's cache is still in use by \(label)"
        case .unsupportedLayer(let kind): return "the parallel engine cannot batch a \(kind) layer cache"
        case .shutDown: return "the parallel engine is shut down"
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

public struct ParallelStats: Sendable {
    public var streams: [StreamStat] = []
    /// Rows decoding together right now.
    public var batchSize = 0
    public var queued = 0
    /// The last decode step, in milliseconds.
    public var stepMs: Double = 0
    /// Tokens every row produced over the last second or so.
    public var aggregateTokensPerSecond: Double = 0
    /// Sessions whose cache the engine holds between rounds, in flight or idle.
    public var residentSlots = 0
    /// What the engine holds on the GPU, and what MLX's allocator says of the process.
    public var memory = MemoryStats()
    /// The request at the head of the queue is held back until rows finish and free memory.
    public var heldBack = false
    /// Idle sessions whose caches the memory policy dropped, since the engine started.
    public var evictions = 0
    /// The last failure the engine contained — an MLX error on its thread — and when.
    public var incident: String?
    public var incidentAt: Date?
    public var at = Date()
    public init() {}
}

/// How much the engine may hold on the GPU, and what it does when that runs out.
///
/// PIN: refuse or wait, never crash. MLX's memory limit is soft — past it an allocation waits and
/// then happens anyway — and an allocation that fails on the engine's thread ended the process. The
/// engine counts what it holds and decides at admission, where a request can still be told no.
public struct MemoryPolicy: Sendable {
    /// What the process may hold on the GPU, all told. `Int.max` watches and holds nothing back.
    public var budgetBytes = Int.max
    /// The fraction of the budget past which a new request waits for rows to finish, after the engine
    /// has given back what idle sessions keep.
    public var highWater = 0.85
    /// Sessions whose caches stay between rounds; past it the least recently used idle one goes.
    public var maxResidentSlots = Int.max
    /// What idle sessions' caches may hold together.
    public var slotBudgetBytes = Int.max
    /// What recurrent snapshots may hold together; the oldest go first.
    public var snapshotBudgetBytes = Int.max
    /// Requests that may wait; past it a request ends at once, failed, rather than queue without bound.
    public var maxQueued = Int.max
    /// Where the engine reads the process's memory: MLX's allocator. A test reads its own.
    public var readMemory: @Sendable () -> (active: Int, cache: Int) = { (Memory.activeMemory, Memory.cacheMemory) }
    /// How the engine gives MLX's buffer cache back, on its own thread between steps. A test counts the calls.
    public var clearCache: @Sendable () -> Void = { Memory.clearCache() }
    public init() {}
}

/// What admission decides for the request at the head of the queue.
public enum Admission: Equatable, Sendable {
    case admit
    /// Wait: rows that are decoding will finish and give memory back.
    case `defer`
    /// Never: the reason ends the stream, failed.
    case reject(String)
}

/// Memory as the engine counts it: bytes read off array shapes (no evaluation), and MLX's allocator.
///
/// PIN: KV and recurrent state are counted apart. Attention KV grows with every token a row holds;
/// recurrent state is a fixed size per row whatever its context. A budget that read them as one number
/// would be wrong for a hybrid model and useless for a pure recurrent one.
public struct MemoryStats: Sendable, Equatable {
    /// MLX's allocator, process-wide: arrays alive now, freed buffers kept for reuse, and the most that
    /// was alive at once over the last few seconds.
    public var activeBytes = 0
    public var cacheBytes = 0
    public var peakBytes = 0
    /// Attention KV the engine holds: the batch's buffers, a prompt being prefilled, idle sessions.
    public var kvBytes = 0
    /// Recurrent state the engine holds, live: the batch's rows, a prefill, idle sessions.
    public var recurrentBytes = 0
    /// Recurrent snapshots the idle and running sessions keep to rewind to. An upper bound: a snapshot
    /// taken at a prompt's end shares its arrays with the live state until the next step replaces them.
    public var snapshotBytes = 0
    /// The batch's own KV buffers, the longest row's token count, and the padding columns the shorter
    /// rows carry so their last tokens line up with it.
    public var batchKVBytes = 0
    public var batchMaxTokens = 0
    public var paddingTokens = 0
    /// Every session the engine keeps a cache for.
    public var slots: [SlotStat] = []
    public init() {}

    /// Everything the engine itself holds.
    public var engineBytes: Int { kvBytes + recurrentBytes + snapshotBytes }
}

/// One session's cache in the engine.
public struct SlotStat: Sendable, Equatable {
    public var slot: SlotID
    /// Tokens its cache holds.
    public var tokens: Int
    public var kvBytes: Int
    public var recurrentBytes: Int
    public var snapshots: Int
    public var snapshotBytes: Int
    /// A stream is running on it now: its layers are in the batch, and it cannot be evicted.
    public var inFlight: Bool
    public var lastUsed: Date
}
