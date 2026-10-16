//  WHAT: One thread serving N streams on one loaded model: a queue of prompts prefilled one at a
//        time in chunks, a batch of rows decoded together one token a step, and a resident cache per
//        session so the next round prefills only what is new.
//  IN:   `insert` a StreamRequest (any thread), `remove` a stream, `forget` a slot (or all of them),
//        `stats`.
//  OUT:  StreamEvent per stream: admitted, token…, finished. ParallelStats for the agent map.
//  PIN:  Every MLX evaluation in this engine happens on its one thread. Arrays never cross it:
//        tokens leave as integers, stats as values, and a stream's cache lives here between rounds.
//        Decode and prefill interleave on the same thread — one decode step, then one prefill
//        chunk — so a long prompt joining never stalls the rows already decoding for more than a
//        chunk, and the rows never starve the prompt.
//  PIN:  Rows are decoded as one forward pass: B tokens in, B logits out. That is the whole point.
//        Independent generations only split the GPU's bandwidth (measured 2026-10-15: two streams
//        34 + 34 tok/s against 67 alone); one batched step reads the weights once for all rows.
//  PIN:  A forgotten slot stays forgotten. A slot's cache is the most memory a session holds here —
//        its KV and its recurrent snapshots — and nothing else frees it. Forgetting one mid-stream
//        used to nil the cache only for `settle` to rebuild it from the row's layers when the row
//        ended, so a session reset during a generation kept everything it meant to drop.

import Foundation
import MLX
import MLXLMCommon

public final class ParallelEngine: @unchecked Sendable {
    public struct Configuration: Sendable {
        /// Rows decoded together at most; what the batch admits before a request waits.
        public var maxRows = 4
        /// Prompt tokens prefilled per chunk between decode steps.
        public var prefillChunk = 256
        /// Decode steps between two prefill chunks (1: strictly alternate).
        public var decodeStepsPerChunk = 1
        /// How many recurrent-state snapshots a slot keeps.
        public var checkpointRing = 4
        /// What the engine may hold, and what it does past it.
        public var memory = MemoryPolicy()
        public init() {}
    }

    private enum Command {
        case insert(Row)
        case remove(StreamID)
        case forget(SlotID)
        case forgetAll
        case setLimits(maxRows: Int?, prefillChunk: Int?)
        case setPolicy(MemoryPolicy)
        case trimSlots(toBytes: Int)
        case clearCache
        case shutdown
    }

    private final class Slot {
        var cache: CheckpointedCache?
        var tokens: [Int] = []
        var inFlight: StreamID?
        /// Forgotten while a stream was using it: the slot goes when that stream settles.
        var forgotten = false
        /// When a stream last began or ended on it: what an eviction reads, oldest first.
        var lastUsed = Date()
    }

    /// A prompt being prefilled, one chunk at a time, on single-sequence layers.
    private final class PrefillJob {
        let row: Row
        let layers: [KVCache]
        let slotCache: CheckpointedCache?
        let suffix: [Int]
        var done = 0
        let startedAt = Date()

        init(row: Row, layers: [KVCache], slotCache: CheckpointedCache?, suffix: [Int]) {
            self.row = row; self.layers = layers; self.slotCache = slotCache; self.suffix = suffix
        }
        var isDone: Bool { done >= suffix.count }
    }

    public let configuration: Configuration
    private let model: any LanguageModel
    private let layout: CacheLayout
    private let condition = NSCondition()
    private var commands: [Command] = []
    private var running = true
    /// `shutdown` was called: a stream inserted after it ends at once rather than wait for a thread that is gone.
    private var closed = false
    private var nextID = 0

    // Engine-thread state.
    private var queue: [Row] = []
    private var prefill: PrefillJob?
    private var batch: GenerationBatch
    private var slots: [SlotID: Slot] = [:]
    /// The last step's samples, one per row and aligned with `batch.rows`: fed at the next step and
    /// read then, while that step runs — the pipelining `TokenIterator` does for one sequence.
    private var pendingTokens: MLXArray?
    private var stepsSinceChunk = 0
    private var lastStepAt: Date?
    private var lastStepSeconds: TimeInterval = 0
    private var rateWindow: [(at: Date, tokens: Int)] = []
    /// MLX's peak, read and reset at every stats: the highest of the last few seconds is what is published.
    private var peakWindow: [(at: Date, bytes: Int)] = []
    /// The limits as they stand now: the configuration's until `setLimits` moves them.
    private var maxRows: Int
    private var prefillChunkSize: Int
    private var policy: MemoryPolicy
    /// Learned from sessions as they settle: what a token of KV costs, and a row of recurrent state.
    private var kvBytesPerToken = 0
    private var recurrentBytesPerRow = 0
    private var heldBack = false
    private var evictions = 0
    private var incident: (at: Date, text: String)?

    private let statsLock = NSLock()
    private var latest = ParallelStats()

    /// Takes the model: from here on, only this engine evaluates it.
    public init(model: any LanguageModel, configuration: Configuration = Configuration()) throws {
        self.model = model
        self.configuration = configuration
        maxRows = max(1, configuration.maxRows)
        prefillChunkSize = max(1, configuration.prefillChunk)
        policy = configuration.memory
        layout = try CacheLayout(sample: model.newCache(parameters: nil))
        batch = GenerationBatch(layout: layout)
        let runner = Thread { [self] in self.loop() }
        runner.name = "frigate.parallel"
        runner.qualityOfService = .userInteractive
        runner.stackSize = 16 << 20
        runner.start()
    }

    // MARK: - API (any thread)

    public func insert(_ request: StreamRequest) -> StreamHandle {
        let (stream, continuation) = AsyncThrowingStream<StreamEvent, Error>.makeStream(bufferingPolicy: .unbounded)
        let (id, open): (StreamID, Bool) = condition.withLock {
            nextID += 1
            return (StreamID(raw: nextID), !closed)
        }
        guard open else {
            continuation.finish(throwing: ParallelError.shutDown)
            return StreamHandle(id: id, events: stream)
        }
        let row = Row(id: id, request: request, continuation: continuation)
        continuation.onTermination = { [weak self] reason in
            if case .cancelled = reason { self?.remove(id) }
        }
        post(.insert(row))
        return StreamHandle(id: id, events: stream)
    }

    /// Ends a stream wherever it is: queued, prefilling or decoding. Its cache, if it was a slot's,
    /// keeps what was prefilled or decoded so far.
    public func remove(_ id: StreamID) { post(.remove(id)) }

    /// Drops a session's resident cache (the session reset, or the session gone). A stream still
    /// running on it finishes, and its cache is dropped when it does.
    public func forget(slot: SlotID) { post(.forget(slot)) }

    /// Drops every session's resident cache: the model is about to be released.
    public func forgetAll() { post(.forgetAll) }

    /// Moves the batch's width and the prefill chunk while the engine runs: the levers memory has. A
    /// narrower batch admits nothing new until rows finish; it never stops a row already decoding.
    public func setLimits(maxRows: Int? = nil, prefillChunk: Int? = nil) { post(.setLimits(maxRows: maxRows, prefillChunk: prefillChunk)) }

    /// Replaces the memory policy.
    public func setPolicy(_ policy: MemoryPolicy) { post(.setPolicy(policy)) }

    /// Drops idle sessions' caches, least recently used first, until they hold at most `bytes`.
    public func trimSlots(toBytes bytes: Int) { post(.trimSlots(toBytes: bytes)) }

    /// Gives MLX's buffer cache back to the system, between two steps.
    public func clearCache() { post(.clearCache) }

    public var stats: ParallelStats { statsLock.withLock { latest } }

    public func shutdown() {
        condition.withLock { closed = true }
        post(.shutdown)
    }

    private func post(_ command: Command) {
        condition.lock()
        commands.append(command)
        condition.signal()
        condition.unlock()
    }

    // MARK: - The loop (engine thread)

    private func loop() {
        while true {
            // Commands first; sleep only when there is nothing to do.
            condition.lock()
            while commands.isEmpty && queue.isEmpty && prefill == nil && batch.isEmpty && running {
                condition.wait()
            }
            let pending = commands
            commands.removeAll()
            let alive = running
            condition.unlock()
            for command in pending { apply(command) }
            if !alive { break }

            // PIN: contained. An MLX error on this thread — an allocation that failed — used to end the
            // process. Under `withError` it ends the rows it hit, failed, and the engine goes on.
            if !batch.isEmpty, prefill == nil || stepsSinceChunk < configuration.decodeStepsPerChunk {
                do { try withError { decodeStep() } } catch { failBatch("decoding failed: \(error)") }
                stepsSinceChunk += 1
            }
            admit()
            if let job = prefill {
                do { try withError { prefillChunk(job) } } catch { failPrefill(job, "prefill failed: \(error)") }
                stepsSinceChunk = 0
            }
            publishStats()
        }
        // Shut down: every stream still here ends.
        for row in queue { end(row, reason: .removed, layers: nil) }
        if let job = prefill { end(job.row, reason: .removed, layers: nil) }
        for (index, row) in batch.rows.enumerated().reversed() { end(row, reason: .removed, layers: batch.extract(row: index)) }
        batch.remove(Set(batch.rows.indices))
        pendingTokens = nil
        // Nothing serves these sessions any more: their caches go with the engine.
        slots.removeAll()
        publishStats()
    }

    private func apply(_ command: Command) {
        switch command {
        case .insert(let row):
            guard queue.count < policy.maxQueued else {
                end(row, reason: .failed, layers: nil, message: "the engine already has \(queue.count) requests waiting")
                return
            }
            queue.append(row)
            queue.sort { a, b in a.priority > b.priority || (a.priority == b.priority && a.id.raw < b.id.raw) }
        case .remove(let id):
            if let index = queue.firstIndex(where: { $0.id == id }) {
                let row = queue.remove(at: index)
                end(row, reason: .removed, layers: nil)
            } else if let job = prefill, job.row.id == id {
                prefill = nil
                // What was prefilled stays with the slot: the next round starts from there.
                if let cache = job.slotCache {
                    let kept = cache.position
                    cache.snapshot(at: kept)
                    settle(job.row, layers: nil, keptTokens: Array(job.row.request.tokens.prefix(kept)))
                }
                end(job.row, reason: .removed, layers: nil)
            } else if let row = batch.rows.first(where: { $0.id == id }) {
                row.removed = true
            }
        case .forget(let slot):
            forget(slot)
        case .forgetAll:
            for slot in Array(slots.keys) { forget(slot) }
            policy.clearCache()
        case .setLimits(let rows, let chunk):
            if let rows { maxRows = max(1, rows) }
            if let chunk { prefillChunkSize = max(1, chunk) }
        case .setPolicy(let next):
            policy = next
            enforceSlotBudget()
        case .trimSlots(let bytes):
            var evicted = false
            while idleSlotBytes > bytes, evictLeastRecentlyUsed() { evicted = true }
            if evicted { policy.clearCache() }
        case .clearCache:
            policy.clearCache()
        case .shutdown:
            running = false
        }
    }

    /// An idle slot goes now; one a stream is using goes when that stream settles (`settle`).
    private func forget(_ id: SlotID) {
        guard let slot = slots[id] else { return }
        guard slot.inFlight != nil else {
            slots[id] = nil
            return
        }
        slot.forgotten = true
        slot.cache = nil
        slot.tokens = []
    }

    // MARK: - Admission and prefill

    private func admit() {
        guard prefill == nil, batch.count < maxRows, let next = queue.first else { return }
        switch admission(for: next) {
        case .defer:
            heldBack = true
            return
        case .reject(let reason):
            heldBack = false
            queue.removeFirst()
            end(next, reason: .failed, layers: nil, message: reason)
            return
        case .admit:
            heldBack = false
        }
        let row = queue.removeFirst()
        row.admittedAt = Date()
        var layers: [KVCache]
        var prefix = 0
        var slotCache: CheckpointedCache?
        if let slotID = row.slot {
            let slot = slots[slotID] ?? Slot()
            slots[slotID] = slot
            if let busy = slot.inFlight {
                row.continuation.finish(throwing: ParallelError.slotBusy("\(busy)"))
                return
            }
            slot.lastUsed = Date()
            if let cache = slot.cache, !slot.tokens.isEmpty, !cache.layers.isEmpty {
                let common = Self.commonPrefix(slot.tokens, row.request.tokens)
                // The model needs at least one new token.
                let wanted = common == row.request.tokens.count ? max(0, common - 1) : common
                if wanted == slot.tokens.count, cache.position == wanted {
                    prefix = wanted
                } else if let position = cache.rewind(toAtMost: wanted) {
                    prefix = position
                } else {
                    slot.cache = CheckpointedCache(layers: model.newCache(parameters: nil), ring: configuration.checkpointRing)
                }
            } else {
                slot.cache = CheckpointedCache(layers: model.newCache(parameters: nil), ring: configuration.checkpointRing)
            }
            slot.inFlight = row.id
            slotCache = slot.cache
            layers = slot.cache!.layers
        } else {
            layers = model.newCache(parameters: nil)
        }
        row.reused = prefix
        row.processor?.prompt(MLXArray(row.request.tokens.map(Int32.init)))
        prefill = PrefillJob(row: row, layers: layers, slotCache: slotCache, suffix: Array(row.request.tokens[prefix...]))
    }

    private func prefillChunk(_ job: PrefillJob) {
        let row = job.row
        if row.removed {
            prefill = nil
            end(row, reason: .removed, layers: nil)
            return
        }
        let upTo = min(job.done + prefillChunkSize, job.suffix.count)
        let piece = Array(job.suffix[job.done ..< upTo])
        let last = upTo == job.suffix.count
        if !piece.isEmpty {
            let input = MLXArray(piece.map(Int32.init)).reshaped(1, piece.count)
            let logits = model(input, cache: job.layers)
            if last {
                row.pending = sample(logits: logits[0..., -1, 0...], row: row)
            } else {
                eval(job.layers.flatMap { $0.innerState() })
            }
        } else if last {
            // Nothing to prefill (an exact repeat): the last prompt token is fed as the first step.
            row.pending = Int32(row.request.tokens.last ?? 0)
        }
        job.done = upTo
        guard last else { return }
        prefill = nil
        row.prefillSeconds = Date().timeIntervalSince(job.startedAt)
        job.slotCache?.snapshot(at: row.promptCount)
        let layers = job.slotCache?.release() ?? job.layers
        row.continuation.yield(.admitted(prefixReused: row.reused, prefilled: job.suffix.count,
                                         queueWait: row.admittedAt.map { $0.timeIntervalSince(row.queuedAt) } ?? 0,
                                         prefillSeconds: row.prefillSeconds))
        // The first token, sampled from the prompt's last logits, is fed at the next step and
        // delivered then, like every token after it.
        batch.extend(row, layers: layers)
        let first = MLXArray([row.pending])
        pendingTokens = pendingTokens.map { concatenated([$0, first], axis: 0) } ?? first
    }

    // MARK: - Decode

    private func decodeStep() {
        guard let fed = pendingTokens else { return }
        let rows = batch.rows
        let count = rows.count
        let now = Date()
        if let last = lastStepAt { lastStepSeconds = now.timeIntervalSince(last) }
        lastStepAt = now
        let logits = model(fed.reshaped(count, 1), cache: batch.caches)
        var last = logits[0..., -1, 0...]                                   // [B, V]
        if rows.contains(where: { $0.processor != nil }) {
            last = concatenated(rows.indices.map { b -> MLXArray in
                var slice = last[b ..< (b + 1)]
                if var processor = rows[b].processor {
                    slice = processor.process(logits: slice)
                    rows[b].processor = processor
                }
                return slice
            }, axis: 0)
        }
        let sampled: MLXArray
        if batch.sharesSampling {
            sampled = rows[0].sampler.sample(logits: last).asType(.int32)
        } else {
            sampled = concatenated(rows.indices.map { rows[$0].sampler.sample(logits: last[$0 ..< ($0 + 1)]) }, axis: 0).asType(.int32)
        }
        for (b, row) in rows.enumerated() {
            if var processor = row.processor {
                processor.didSample(token: sampled[b ..< (b + 1)])
                row.processor = processor
            }
        }
        asyncEval(sampled)
        // The tokens just fed were sampled last step: read them now, while this step runs.
        let values = fed.asArray(Int32.self)
        var finished: Set<Int> = []
        for (b, row) in rows.enumerated() {
            let token = Int(values[b])
            row.fed.append(token)
            row.batchSteps += 1
            row.batchRowsSeen += count
            if row.removed {
                row.finish = .removed
                finished.insert(b)
                continue
            }
            if emit(row, token: token) { finished.insert(b) }
        }
        rateWindow.append((Date(), count))
        guard !finished.isEmpty else {
            pendingTokens = sampled
            return
        }
        // A row that ended has its stop (or last) token in the cache already: it was fed this step.
        for b in finished.sorted() {
            end(rows[b], reason: rows[b].finish ?? .stop, layers: batch.extract(row: b))
        }
        let keep = rows.indices.filter { !finished.contains($0) }
        batch.remove(finished)
        pendingTokens = keep.isEmpty ? nil : sampled[MLXArray(keep.map(Int32.init))]
    }

    /// Delivers one token; true when it ended the stream.
    private func emit(_ row: Row, token: Int) -> Bool {
        row.emitted += 1
        row.continuation.yield(.token(token))
        if row.request.stopTokens.contains(token) {
            row.finish = .stop
            return true
        }
        if row.emitted >= row.request.maxTokens {
            row.finish = .maxTokens
            return true
        }
        return false
    }

    private func sample(logits: MLXArray, row: Row) -> Int32 {
        var slice = logits                                                   // [1, V]
        if var processor = row.processor {
            slice = processor.process(logits: slice)
            row.processor = processor
        }
        let token = row.sampler.sample(logits: slice)
        if var processor = row.processor {
            processor.didSample(token: token)
            row.processor = processor
        }
        let sampled = token.asType(.int32)
        eval(sampled)
        return sampled.asArray(Int32.self)[0]
    }

    // MARK: - Ending

    /// The slot takes its layers back with what was fed; the stream hears `finished`.
    private func end(_ row: Row, reason: FinishReason, layers: [KVCache]?, message: String? = nil) {
        settle(row, layers: layers, keptTokens: row.request.tokens + row.fed)
        let info = FinishInfo(reason: reason, promptTokens: row.promptCount, generated: row.emitted,
                              prefillSeconds: row.prefillSeconds,
                              decodeSeconds: row.admittedAt.map { Date().timeIntervalSince($0) - row.prefillSeconds } ?? 0,
                              meanBatchSize: row.batchSteps > 0 ? Double(row.batchRowsSeen) / Double(row.batchSteps) : 0,
                              message: message)
        row.continuation.yield(.finished(info))
        row.continuation.finish()
    }

    // MARK: - Failure, contained

    /// Every row of the batch ends failed: what they held is dropped, not settled back.
    private func failBatch(_ message: String) {
        for row in batch.rows.reversed() { end(row, reason: .failed, layers: nil, message: message) }
        batch.remove(Set(batch.rows.indices))
        pendingTokens = nil
        record(message)
    }

    /// The prompt being prefilled ends failed, and its session's cache with it.
    private func failPrefill(_ job: PrefillJob, _ message: String) {
        prefill = nil
        if let slotID = job.row.slot { slots[slotID]?.cache = nil }
        end(job.row, reason: .failed, layers: nil, message: message)
        record(message)
    }

    private func record(_ message: String) {
        incident = (Date(), message)
        policy.clearCache()
    }

    // MARK: - Memory policy

    /// Whether `row` fits now: what it would add against what the process holds — idle sessions' caches
    /// and MLX's buffer cache given back first when that is what stands in the way.
    private func admission(for row: Row) -> Admission {
        guard policy.budgetBytes < Int.max else { return .admit }
        let estimate = estimatedBytes(for: row)
        if estimate > policy.budgetBytes {
            return .reject("this prompt would need about \(Self.gb(estimate)) on the GPU, more than the \(Self.gb(policy.budgetBytes)) budget")
        }
        let high = Int(Double(policy.budgetBytes) * policy.highWater)
        var memory = policy.readMemory()
        if memory.active + memory.cache + estimate <= high { return .admit }
        // Idle sessions go first, least recently used: what they held moves to the buffer cache.
        var evicted = false
        while memory.active + estimate > high, evictLeastRecentlyUsed(except: row.slot) {
            evicted = true
            memory = policy.readMemory()
        }
        // PIN: the buffer cache goes back only after an eviction or when that alone admits the row — never
        // on every step a row waits. Cleared each step, the batch would lose its buffers and allocate afresh
        // for every token, exactly while memory is tight.
        if evicted || memory.active + estimate <= high {
            policy.clearCache()
            memory = policy.readMemory()
            if memory.active + memory.cache + estimate <= high { return .admit }
        }
        // Rows decoding now will finish and give memory back: wait for them.
        if !batch.isEmpty { return .defer }
        // Nothing running will free anything; what is held is the model and what cannot go.
        if memory.active + estimate > policy.budgetBytes {
            return .reject("the model and what it holds leave \(Self.gb(max(0, policy.budgetBytes - memory.active))) of the "
                           + "\(Self.gb(policy.budgetBytes)) budget, and this prompt needs about \(Self.gb(estimate))")
        }
        return .admit
    }

    /// What admitting `row` adds: KV for the tokens its session does not hold yet, a row of recurrent state
    /// for a session that holds none, and the padding every row takes on when it is longer than the longest.
    private func estimatedBytes(for row: Row) -> Int {
        let slot = row.slot.flatMap { slots[$0] }
        let held = slot.map { Self.commonPrefix($0.tokens, row.request.tokens) } ?? 0
        var bytes = max(0, row.request.tokens.count - held) * kvBytesPerToken
        if slot?.cache == nil { bytes += recurrentBytesPerRow }
        let longest = batch.lengths.longest
        if !batch.isEmpty, row.request.tokens.count > longest {
            bytes += (row.request.tokens.count - longest) * kvBytesPerToken * batch.count
        }
        return bytes
    }

    /// What idle sessions' caches hold together.
    private var idleSlotBytes: Int {
        slots.values.filter { $0.inFlight == nil }.compactMap(\.cache).reduce(0) { $0 + $1.kvBytes + $1.recurrentBytes + $1.snapshotBytes }
    }

    /// Drops the least recently used idle session's cache. False when every session is in use.
    @discardableResult
    private func evictLeastRecentlyUsed(except keep: SlotID? = nil) -> Bool {
        let idle = slots.filter { $0.value.inFlight == nil && $0.value.cache != nil && $0.key != keep }
        guard let victim = idle.min(by: { $0.value.lastUsed < $1.value.lastUsed })?.key else { return false }
        slots[victim] = nil
        evictions += 1
        return true
    }

    /// Holds sessions to the policy: how many keep a cache, what the idle ones hold, what the snapshots hold.
    private func enforceSlotBudget() {
        var freed = false
        while slots.values.filter({ $0.cache != nil }).count > policy.maxResidentSlots, evictLeastRecentlyUsed() { freed = true }
        while idleSlotBytes > policy.slotBudgetBytes, evictLeastRecentlyUsed() { freed = true }
        while slots.values.compactMap(\.cache).reduce(0, { $0 + $1.snapshotBytes }) > policy.snapshotBudgetBytes {
            // The oldest snapshot of the session with the most, then of the least recently used.
            let holders = slots.values.filter { ($0.cache?.snapshots.count ?? 0) > 0 }
            guard let holder = holders.max(by: { a, b in
                (a.cache?.snapshots.count ?? 0, b.lastUsed) < (b.cache?.snapshots.count ?? 0, a.lastUsed)
            }), holder.cache?.dropOldestSnapshot() == true else { break }
            freed = true
        }
        if freed { policy.clearCache() }
    }

    static func gb(_ bytes: Int) -> String { String(format: "%.1f GB", Double(bytes) / 1_073_741_824) }

    private func settle(_ row: Row, layers: [KVCache]?, keptTokens: [Int]) {
        guard let slotID = row.slot, let slot = slots[slotID], slot.inFlight == row.id else { return }
        slot.inFlight = nil
        slot.lastUsed = Date()
        // Forgotten while this row ran: what it hands back is dropped, not adopted.
        if slot.forgotten {
            slots[slotID] = nil
            return
        }
        guard let layers else {
            // Nothing decoded on the batch: the slot keeps the layers it still holds, if any.
            if slot.cache?.layers.isEmpty == false { slot.tokens = keptTokens } else { slot.cache = nil; slot.tokens = [] }
            return
        }
        if let cache = slot.cache {
            cache.adopt(layers)
        } else {
            slot.cache = CheckpointedCache(layers: layers, ring: configuration.checkpointRing)
        }
        slot.tokens = keptTokens
        // What a token and a row cost, learned from a session that just settled: what admission estimates with.
        if let cache = slot.cache, cache.position > 0 {
            kvBytesPerToken = max(kvBytesPerToken, cache.kvBytes / cache.position)
            recurrentBytesPerRow = max(recurrentBytesPerRow, cache.recurrentBytes)
        }
        enforceSlotBudget()
    }

    // MARK: - Stats

    private func publishStats() {
        let now = Date()
        rateWindow.removeAll { now.timeIntervalSince($0.at) > 1.5 }
        var stats = ParallelStats()
        stats.at = now
        stats.batchSize = batch.count
        stats.queued = queue.count
        stats.residentSlots = slots.values.filter { $0.cache != nil }.count
        stats.memory = memory(at: now)
        stats.heldBack = heldBack
        stats.evictions = evictions
        stats.incident = incident?.text
        stats.incidentAt = incident?.at
        stats.stepMs = lastStepSeconds * 1000
        if let first = rateWindow.first, rateWindow.count > 1 {
            let span = now.timeIntervalSince(first.at)
            stats.aggregateTokensPerSecond = span > 0 ? Double(rateWindow.map(\.tokens).reduce(0, +)) / span : 0
        }
        func stat(_ row: Row, phase: StreamPhase) -> StreamStat {
            if let mark = row.rateMark {
                let span = now.timeIntervalSince(mark.at)
                if span >= 1 {
                    row.tokensPerSecond = Double(row.emitted - mark.count) / span
                    row.rateMark = (row.emitted, now)
                }
            } else {
                row.rateMark = (row.emitted, now)
            }
            return StreamStat(id: row.id, label: row.request.label, phase: phase, promptTokens: row.promptCount,
                              generated: row.emitted, tokensPerSecond: row.tokensPerSecond)
        }
        stats.streams = batch.rows.map { stat($0, phase: .decode) }
        if let job = prefill { stats.streams.append(stat(job.row, phase: .prefill(done: job.done, total: job.suffix.count))) }
        stats.streams += queue.map { stat($0, phase: .queued) }
        statsLock.withLock { latest = stats }
    }

    /// What the engine holds, by kind and by session, and what MLX's allocator says. Shapes only.
    private func memory(at now: Date) -> MemoryStats {
        var memory = MemoryStats()
        let snapshot = Memory.snapshot()
        memory.activeBytes = snapshot.activeMemory
        memory.cacheBytes = snapshot.cacheMemory
        // MLX keeps the peak since it was last reset, process-wide: read it, reset it, and publish the
        // highest of the last five seconds, so a reader polling every few seconds still sees a spike.
        Memory.peakMemory = 0
        peakWindow.append((now, snapshot.peakMemory))
        peakWindow.removeAll { now.timeIntervalSince($0.at) > 5 }
        memory.peakBytes = peakWindow.map(\.bytes).max() ?? snapshot.peakMemory
        memory.batchKVBytes = batch.kvBytes
        let lengths = batch.lengths
        memory.batchMaxTokens = lengths.longest
        memory.paddingTokens = lengths.padding
        memory.kvBytes = batch.kvBytes
        memory.recurrentBytes = batch.recurrentBytes
        if let job = prefill, job.slotCache == nil {
            for layer in job.layers {
                let bytes = layer.innerState().reduce(0) { $0 + $1.nbytes }
                if layer is ArraysCache { memory.recurrentBytes += bytes } else { memory.kvBytes += bytes }
            }
        }
        for (id, slot) in slots {
            guard let cache = slot.cache else { continue }
            let stat = SlotStat(slot: id, tokens: slot.tokens.count, kvBytes: cache.kvBytes, recurrentBytes: cache.recurrentBytes,
                                snapshots: cache.snapshots.count, snapshotBytes: cache.snapshotBytes,
                                inFlight: slot.inFlight != nil, lastUsed: slot.lastUsed)
            memory.kvBytes += stat.kvBytes
            memory.recurrentBytes += stat.recurrentBytes
            memory.snapshotBytes += stat.snapshotBytes
            memory.slots.append(stat)
        }
        memory.slots.sort { $0.lastUsed > $1.lastUsed }
        return memory
    }

    static func commonPrefix(_ a: [Int], _ b: [Int]) -> Int {
        var index = 0
        let limit = min(a.count, b.count)
        while index < limit, a[index] == b[index] { index += 1 }
        return index
    }
}

extension Row {
    var priority: StreamPriority { request.priority }
}

extension NSCondition {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
