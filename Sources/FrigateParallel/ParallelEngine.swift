//  WHAT: One thread serving N streams on one loaded model: a queue of prompts prefilled one at a
//        time in chunks, a batch of rows decoded together one token a step, and a resident cache per
//        session so the next round prefills only what is new.
//  IN:   `insert` a StreamRequest (any thread), `remove` a stream, `forget` a slot (or all of them),
//        `keep` a slot's cache to a file and `take` one back (`SlotFile`), `stats`.
//  OUT:  StreamEvent per stream: admitted, token…, finished. ParallelStats for the agent map.
//  PIN:  Every MLX evaluation in this engine happens on its one thread. Arrays never cross it:
//        tokens leave as integers, stats as values, and a stream's cache lives here between rounds.
//        Decode and prefill share that thread by time, not by turns: each pass `InterleavePolicy`
//        says whether the rows take a step or the prompt a chunk, and how big, from what the last
//        ones took. A prompt joining stalls the rows for one chunk sized to a few of their steps at
//        most, and the rows never starve the prompt.
//  PIN:  A prompt someone waits on is never held behind one no one does. A request of higher
//        priority sets the prompt being read aside, as it stands, layers and progress kept, and that
//        prompt reads on once nothing above it waits (`admit`). Before, a worker's prompt or a warm
//        held every request behind it until its last chunk.
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
        /// Prompt tokens read in one chunk at most, and the chunk a prompt is read in when no row decodes.
        public var prefillChunk = 256
        /// The share of the engine's time a prompt being read takes from rows of its own priority decoding beside it
        /// (`InterleavePolicy.even`), and from rows someone waits on more (`yielding`).
        public var prefillShare = 0.5
        public var yieldingPrefillShare = 0.2
        /// The smallest chunk a prompt beside decoding rows is read in.
        public var minPrefillChunk = 32
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
        case hold
        case resume(ceiling: Int?)
        case shutdown
        case keep(SlotID, URL, SlotStretch, [String: String], @Sendable (Result<SlotFile, Error>) -> Void)
        case holdSlot(SlotID, Bool)
        case take(SlotID, [URL], [String: String], @Sendable (Result<SlotFile, Error>) -> Void)
        case exclusive(@Sendable () -> Void)
    }

    private final class Slot {
        var cache: CheckpointedCache?
        var tokens: [Int] = []
        var inFlight: StreamID?
        /// Forgotten while a stream was using it: the slot goes when that stream settles.
        var forgotten = false
        /// When a stream last began or ended on it: what an eviction reads, oldest first.
        var lastUsed = Date()
        /// How many of its session's scopes hold it between two of a turn's requests (`holding(slot:)`): not idle, and
        /// evicted only when a request would otherwise be refused.
        var holds = 0
        var isHeld: Bool { holds > 0 }
        /// Its cache came from a file (`take`) since its last request.
        var taken = false
    }

    /// A prompt being prefilled, one chunk at a time, on single-sequence layers.
    private final class PrefillJob {
        let row: Row
        let layers: [KVCache]
        let slotCache: CheckpointedCache?
        let suffix: [Int]
        /// Where in `suffix` the turn boundary ends, when it is inside it and a slot keeps the cache.
        let pin: Int?
        /// Where in `suffix` the prompt's marks stand, those inside it, when a slot keeps the cache.
        let marks: [Int]
        var done = 0
        let startedAt = Date()

        init(row: Row, layers: [KVCache], slotCache: CheckpointedCache?, suffix: [Int], pin: Int? = nil, marks: [Int] = []) {
            self.row = row; self.layers = layers; self.slotCache = slotCache; self.suffix = suffix; self.pin = pin
            self.marks = marks
        }
        var isDone: Bool { done >= suffix.count }
        /// The next point past `done` a chunk must end at: a mark or the pin, whichever comes first.
        var nextStop: Int? { (marks + [pin].compactMap { $0 }).filter { $0 > done }.min() }
    }

    public let configuration: Configuration
    private let model: any LanguageModel
    /// The GPU scheduling a decode step runs under, when the model gives one (`DecodeSchedulingModel`).
    private let decodeScheduling: GPU.Scheduling?
    private let layout: CacheLayout
    private let condition = NSCondition()
    /// Where kept caches are written, one at a time, off the engine's thread (`SlotFileWriter`).
    private let files = DispatchQueue(label: "frigate.parallel.files", qos: .utility)
    private var commands: [Command] = []
    private var running = true
    /// `shutdown` was called: a stream inserted after it ends at once rather than wait for a thread that is gone.
    private var closed = false
    private var nextID = 0

    // Engine-thread state.
    private var queue: [Row] = []
    private var prefill: PrefillJob?
    /// Prompts set aside mid-read for a request someone waits on more, in the order they read on: priority, then age.
    private var suspended: [PrefillJob] = []
    /// How a pass is shared between the rows and the prompt being read, and what the engine's time went to.
    private var interleave: InterleavePolicy
    private var clock = InterleavePolicy.Clock()
    private var batch: GenerationBatch
    private var slots: [SlotID: Slot] = [:]
    /// The last step's samples, one per row and aligned with `batch.rows`: fed at the next step and
    /// read then, while that step runs — the pipelining `TokenIterator` does for one sequence.
    private var pendingTokens: MLXArray?
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
    /// Paused at the memory ceiling: no step runs (`holding()`).
    private var held: MemoryHold?
    /// The ceiling as the host last raised it; the policy's until then.
    private var raisedCeiling: Int?
    private var evictions = 0
    private var incident: (at: Date, text: String)?

    private let statsLock = NSLock()
    private var latest = ParallelStats()

    /// Takes the model: from here on, only this engine evaluates it.
    public init(model: any LanguageModel, configuration: Configuration = Configuration()) throws {
        self.model = model
        decodeScheduling = (model as? any DecodeSchedulingModel)?.decodeScheduling
        self.configuration = configuration
        maxRows = max(1, configuration.maxRows)
        prefillChunkSize = max(1, configuration.prefillChunk)
        policy = configuration.memory
        interleave = InterleavePolicy(even: configuration.prefillShare, yielding: configuration.yieldingPrefillShare,
                                      minChunk: configuration.minPrefillChunk, maxChunk: prefillChunkSize)
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

    /// Pauses every step, as if the ceiling were passed, until `resume`: the host saw memory run short
    /// where the engine cannot (macOS's own pressure).
    public func hold() { post(.hold) }

    /// Lets a held engine go on. A ceiling raises the line it holds at from here on; nil keeps it, so an
    /// engine still over it holds again at once.
    public func resume(ceiling: Int? = nil) { post(.resume(ceiling: ceiling)) }

    /// Writes a stretch of a session's cache to `url` (`SlotFile`, `frigate-slot/1`): its layers for the stretch's
    /// tokens, the state at its end and the snapshots inside it, with every token before its end and `metadata` beside
    /// them. The slot keeps its cache. Throws when the slot holds nothing, a stream is using it, the stretch cannot be
    /// cut (`ParallelError.stretchRefused`), or the file cannot be written.
    ///
    /// PIN: GATHERED ON THE ENGINE'S THREAD, WRITTEN OFF IT (2026-11-22). The engine cuts the stretch between two
    /// passes, which costs it the time to evaluate a few slices; the bytes go to disk on a queue of their own
    /// (`SlotFileWriter`), and the engine reads and decodes on meanwhile. The call returns once the file is in place.
    public func keep(slot: SlotID, to url: URL, stretch: SlotStretch = .whole, metadata: [String: String] = [:]) async throws -> SlotFile {
        try await answered { .keep(slot, url, stretch, metadata, $0) }
    }

    /// Runs `body` with a session's cache held against eviction: the tools and agents a turn runs between two of its
    /// requests find the cache where the last request left it. The hold ends when `body` does, however it ends.
    ///
    /// PIN: A TURN'S SLOT IS NOT IDLE BETWEEN ITS ROUNDS (2026-11-20). An agent the turn called could evict the turn's
    /// own cache as least recently used, and the round after the agent read the whole conversation again.
    /// PIN: A HOLD IS A SCOPE (2026-11-22). It was two calls, `hold` and `unhold`, that every path out of a turn had to
    /// pair by hand; a path that missed the second left a slot nothing could ever evict. A scope has one way out.
    public func holding<T>(slot: SlotID, isolation: isolated (any Actor)? = #isolation,
                           _ body: () async throws -> T) async rethrows -> T {
        post(.holdSlot(slot, true))
        defer { post(.holdSlot(slot, false)) }
        return try await body()
    }

    /// Reads a session's cache from `urls` into `slot`: a chain of stretches, each beginning where the one before it
    /// ends. A chain that begins at token zero replaces what the slot held; one that begins later is taken onto the
    /// slot's own cache, which must hold exactly the tokens before it. The next request on the slot continues the
    /// cache by their common prefix. Throws, leaving the slot as it was, when a stream is using the slot or a file does
    /// not fit: not a slot file, cut short, another cache layout, a value in `expecting` its metadata does not hold,
    /// or a stretch that does not begin where the tokens before it end.
    public func take(slot: SlotID, from urls: [URL], expecting: [String: String] = [:]) async throws -> SlotFile {
        try await answered { .take(slot, urls, expecting, $0) }
    }

    /// `take(slot:from:expecting:)` for a file that stands alone.
    public func take(slot: SlotID, from url: URL, expecting: [String: String] = [:]) async throws -> SlotFile {
        try await take(slot: slot, from: [url], expecting: expecting)
    }

    /// Runs `body` on the engine's thread between two passes, nothing else running: a forward of the engine's model
    /// that is not a stream of it (a picture read by the model's own tower, `OwnVision`) never runs beside a step.
    /// The rows decoding wait while it runs.
    public func exclusive<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<T, Error>) in
            let posted: Bool = condition.withLock {
                guard !closed else { return false }
                commands.append(.exclusive { continuation.resume(with: Result { try body() }) })
                condition.signal()
                return true
            }
            if !posted { continuation.resume(throwing: ParallelError.shutDown) }
        }
    }

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

    /// Posts a command the engine's thread answers, and waits for the answer.
    ///
    /// PIN: checked and posted under one lock. `shutdown` closes the engine under that lock before it posts its own
    /// command, so a command posted while open is ahead of it and answered before the loop ends; one posted after is
    /// answered at once. Posted without the check, it would wait for a thread that is gone.
    private func answered(_ command: (@escaping @Sendable (Result<SlotFile, Error>) -> Void) -> Command) async throws -> SlotFile {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<SlotFile, Error>) in
            let posted: Bool = condition.withLock {
                guard !closed else { return false }
                commands.append(command { continuation.resume(with: $0) })
                condition.signal()
                return true
            }
            if !posted { continuation.resume(throwing: ParallelError.shutDown) }
        }
    }

    // MARK: - The loop (engine thread)

    private func loop() {
        while true {
            // Commands first; sleep only when there is nothing to do.
            condition.lock()
            while commands.isEmpty && queue.isEmpty && prefill == nil && suspended.isEmpty && batch.isEmpty && running {
                condition.wait()
            }
            let pending = commands
            commands.removeAll()
            let alive = running
            condition.unlock()
            for command in pending { apply(command) }
            if !alive { break }
            if holding() {
                publishStats()
                // Paused: wake for a command, or to look at memory again.
                condition.lock()
                if commands.isEmpty, running { _ = condition.wait(until: Date().addingTimeInterval(0.25)) }
                condition.unlock()
                continue
            }

            // PIN: contained. An MLX error on this thread — an allocation that failed — used to end the
            // process. Under `withError` it ends the rows it hit, failed, and the engine goes on.
            admit()
            switch nextWork() {
            case .step?:
                let began = Date()
                do { try withError { decodeStep() } } catch { failBatch("decoding failed: \(error)") }
                clock.stepped(seconds: Date().timeIntervalSince(began))
            case .chunk(let tokens)?:
                guard let job = prefill else { break }
                let began = Date(), before = job.done
                do { try withError { prefillChunk(job, tokens: tokens) } } catch { failPrefill(job, "prefill failed: \(error)") }
                if job.done > before { clock.read(tokens: job.done - before, seconds: Date().timeIntervalSince(began)) }
            case nil:
                break
            }
            publishStats()
        }
        // Shut down: every stream still here ends.
        for row in queue { end(row, reason: .removed, layers: nil) }
        if let job = prefill { end(job.row, reason: .removed, layers: nil) }
        for job in suspended { end(job.row, reason: .removed, layers: nil) }
        suspended.removeAll()
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
            guard !row.request.prefillOnly || row.slot != nil else {
                end(row, reason: .failed, layers: nil, message: "a prompt read only to be kept needs a slot to keep it in")
                return
            }
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
                stopReading(job)
            } else if let index = suspended.firstIndex(where: { $0.row.id == id }) {
                stopReading(suspended.remove(at: index))
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
            if let chunk {
                prefillChunkSize = max(1, chunk)
                interleave = InterleavePolicy(even: interleave.even, yielding: interleave.yielding,
                                              minChunk: configuration.minPrefillChunk, maxChunk: prefillChunkSize)
            }
        case .setPolicy(let next):
            policy = next
            enforceSlotBudget()
        case .trimSlots(let bytes):
            var evicted = false
            while idleSlotBytes > bytes, evictLeastRecentlyUsed() { evicted = true }
            if evicted { policy.clearCache() }
        case .clearCache:
            policy.clearCache()
        case .hold:
            held = MemoryHold(bytes: ceilingReading(), ceiling: ceiling, asked: true)
        case .resume(let raised):
            if let raised { raisedCeiling = raised }
            held = nil
        case .shutdown:
            running = false
        case .keep(let id, let url, let stretch, let metadata, let answer):
            do {
                let image = try withError { try gatherSlot(id, stretch: stretch, metadata: metadata) }
                files.async { answer(Result { try withError { try SlotFileCodec.write(image, to: url) } }) }
            } catch {
                answer(.failure(error))
            }
        case .holdSlot(let id, let held):
            let slot = slots[id] ?? Slot()
            slot.holds = max(0, slot.holds + (held ? 1 : -1))
            // A slot made only to be held, and let go before it ever read a prompt, leaves nothing behind.
            if slot.isHeld || slot.cache != nil || slot.inFlight != nil { slots[id] = slot } else { slots[id] = nil }
            if !held { enforceSlotBudget() }
        case .take(let id, let urls, let expecting, let answer):
            answer(Result { try withError { try takeSlot(id, from: urls, expecting: expecting) } })
        case .exclusive(let body):
            body()
        }
    }

    /// Cuts a stretch of an idle slot's cache for writing, leaving the cache in place (`keep(slot:to:stretch:metadata:)`).
    private func gatherSlot(_ id: SlotID, stretch: SlotStretch, metadata: [String: String]) throws -> SlotFileImage {
        guard let slot = slots[id] else { throw ParallelError.slotEmpty }
        if let busy = slot.inFlight { throw ParallelError.slotBusy("\(busy)") }
        guard let cache = slot.cache, !cache.layers.isEmpty, !slot.tokens.isEmpty else { throw ParallelError.slotEmpty }
        return try SlotFileCodec.gather(cache, tokens: slot.tokens, layout: layout, stretch: stretch, metadata: metadata)
    }

    /// Reads a chain of kept stretches into a slot that no stream is using (`take(slot:from:expecting:)`), then holds
    /// every slot to the memory policy again, as a slot that settles is.
    private func takeSlot(_ id: SlotID, from urls: [URL], expecting: [String: String]) throws -> SlotFile {
        if let busy = slots[id]?.inFlight { throw ParallelError.slotBusy("\(busy)") }
        let standing = slots[id].flatMap { slot in
            slot.cache.flatMap { $0.layers.isEmpty || slot.tokens.isEmpty ? nil : (cache: $0, tokens: slot.tokens) }
        }
        let read = try SlotFileCodec.read(from: urls, onto: standing, layout: layout, expecting: expecting,
                                          ring: configuration.checkpointRing)
        let slot = slots[id] ?? Slot()
        slot.cache = read.cache
        slot.tokens = read.tokens
        slot.forgotten = false
        slot.taken = true
        slot.lastUsed = Date()
        slots[id] = slot
        learnCosts(from: read.cache)
        enforceSlotBudget()
        return read.file
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

    // MARK: - The ceiling

    private var ceiling: Int { raisedCeiling ?? policy.ceilingBytes }

    private func ceilingReading() -> Int {
        if let read = policy.readCeilingBytes { return read() }
        let memory = policy.readMemory()
        return memory.active + memory.cache
    }

    /// Whether the engine is paused at its ceiling. Past it the buffer cache goes back first; if that is
    /// not enough, no step runs until the host lets it go (`resume`), or memory falls back under the
    /// high-water mark of the line — what stopping the turns gives back.
    ///
    /// PIN: back under the high-water mark, not back under the line. A process that sits at its line —
    /// giving the cache back takes it a hair under, the next step a hair over — was paused and let go
    /// every few seconds (watched in the app, 2026-10-17), so the person was asked and un-asked and no
    /// answer ever stuck.
    ///
    /// PIN: a stream stopped while the engine is paused still ends. A removed row only ends at the next
    /// decode step, and a paused engine takes none — so Stop would wait on the very hold it was meant to
    /// relieve. Its cache is dropped, not settled: copying it out is memory the engine does not have.
    private func holding() -> Bool {
        if var current = held {
            endRemovedRows()
            let reading = ceilingReading()
            if !current.asked, Double(reading) <= Double(current.ceiling) * policy.highWater {
                held = nil
                return false
            }
            // The most it reached while paused: what the person is told was past the line, even after the
            // buffer cache has gone back.
            current.bytes = max(current.bytes, reading)
            held = current
            return true
        }
        let line = ceiling
        guard line < Int.max, ceilingReading() > line else { return false }
        policy.clearCache()
        let reading = ceilingReading()
        guard reading > line else { return false }
        held = MemoryHold(bytes: reading, ceiling: line, asked: false)
        incident = (Date(), "paused at \(Self.gb(reading)), past the \(Self.gb(line)) ceiling")
        endRemovedRows()
        return true
    }

    /// Ends the batch's removed rows now, without a step; their sessions' caches go with them.
    private func endRemovedRows() {
        let rows = batch.rows
        let gone = Set(rows.indices.filter { rows[$0].removed })
        guard !gone.isEmpty else { return }
        for index in gone.sorted() { end(rows[index], reason: .removed, layers: nil) }
        batch.remove(gone)
        let keep = rows.indices.filter { !gone.contains($0) }
        pendingTokens = keep.isEmpty ? nil : pendingTokens?[MLXArray(keep.map(Int32.init))]
    }

    // MARK: - Admission and prefill

    /// Starts reading the next prompt: the first request that may start, unless a prompt set aside comes before it.
    ///
    /// PIN: a request someone waits on more than on the prompt being read sets that prompt aside (`suspend`) once the
    /// memory policy has let it in; one of equal priority waits its turn, as every request always did.
    /// A fresh cache for one row.
    /// PIN: CHECKED AT START. `init` built one from the same model with the same (no) parameters and threw if it could
    /// not, so this one cannot fail.
    private func freshLayers() -> [KVCache] {
        try! model.newCache(parameters: nil)
    }

    private func admit() {
        let next = queue.firstIndex(where: mayStart)
        if let job = prefill {
            guard let index = next, queue[index].priority > job.row.priority else { return }
        } else if let waiting = suspended.first, mayStart(waiting.row),
                  next.map({ queue[$0].priority <= waiting.row.priority }) ?? true {
            prefill = suspended.removeFirst()
            return
        }
        guard let index = next else { return }
        switch admission(for: queue[index]) {
        case .defer:
            heldBack = true
            return
        case .reject(let reason):
            heldBack = false
            let refused = queue.remove(at: index)
            end(refused, reason: .failed, layers: nil, message: reason)
            return
        case .admit:
            heldBack = false
        }
        if let job = prefill { suspend(job) }
        let row = queue.remove(at: index)
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
            // PIN: FROM A FILE ONLY WHEN THE FILE'S CACHE WAS GONE ON FROM. It was said of any request after a take,
            // so a prompt that shared nothing with the kept cache, and read from the start, was recorded "fresh from
            // file".
            let taken = slot.taken
            slot.taken = false
            row.admission = .fresh
            if let cache = slot.cache, !slot.tokens.isEmpty, !cache.layers.isEmpty {
                let common = Self.commonPrefix(slot.tokens, row.request.tokens)
                // PIN: a prompt read only to be kept that the slot already holds, whole, reads nothing and leaves the
                // cache as it is: the slot may hold more the next request shares (a reply a template renders again
                // token for token), and rewinding to the prompt's end would throw that away to read it again.
                if row.request.prefillOnly, common == row.request.tokens.count {
                    row.reused = common
                    row.admission = .append
                    row.fromFile = taken
                    row.continuation.yield(.admitted(prefixReused: common, prefilled: 0,
                                                     queueWait: Date().timeIntervalSince(row.queuedAt), prefillSeconds: 0))
                    end(row, reason: .prefilled, layers: nil)
                    return
                }
                // The model needs at least one new token.
                let wanted = common == row.request.tokens.count ? max(0, common - 1) : common
                if wanted == slot.tokens.count, cache.position == wanted {
                    prefix = wanted
                    row.admission = .append
                    row.fromFile = taken
                } else if let rewound = cache.rewindPoint(toAtMost: wanted) {
                    prefix = rewound.position
                    row.admission = .rewind(rewound.point)
                    row.fromFile = taken && prefix > 0
                } else {
                    slot.cache = CheckpointedCache(layers: freshLayers(), ring: configuration.checkpointRing)
                }
            } else {
                slot.cache = CheckpointedCache(layers: freshLayers(), ring: configuration.checkpointRing)
            }
            slot.inFlight = row.id
            slotCache = slot.cache
            layers = slot.cache!.layers
        } else {
            layers = freshLayers()
            row.admission = .fresh
        }
        row.reused = prefix
        if !row.request.prefillOnly { row.processor?.prompt(MLXArray(row.request.tokens.map(Int32.init))) }
        // A boundary the cache already holds, or the prompt's very end, needs no pin of its own. A prompt read only to
        // be kept may end at a mark: nothing is sampled after it, and its end is where a stretch is cut.
        let count = row.request.tokens.count
        let inside: (Int?) -> Int? = { at in at.flatMap { slotCache != nil && $0 > prefix && $0 < count ? $0 - prefix : nil } }
        let marks = row.request.marksAt.compactMap { at in
            inside(at) ?? (row.request.prefillOnly && slotCache != nil && at == count && at > prefix ? at - prefix : nil)
        }
        prefill = PrefillJob(row: row, layers: layers, slotCache: slotCache, suffix: Array(row.request.tokens[prefix...]),
                             pin: inside(row.request.pinAt), marks: marks)
    }

    /// A request may start reading now: one that joins the batch needs a row free there; one read only to be kept
    /// never joins it.
    private func mayStart(_ row: Row) -> Bool { row.request.prefillOnly || batch.count < maxRows }

    /// The prompt being read steps aside for a request someone waits on more, its layers and its progress kept, and
    /// reads on from there once nothing above it waits.
    private func suspend(_ job: PrefillJob) {
        prefill = nil
        suspended.append(job)
        suspended.sort { a, b in a.row.priority > b.row.priority || (a.row.priority == b.row.priority && a.row.id.raw < b.row.id.raw) }
    }

    /// A prompt removed before it was read to the end: what was read stays with its slot, so the next round starts from
    /// there.
    private func stopReading(_ job: PrefillJob) {
        if let cache = job.slotCache {
            let kept = cache.position
            cache.snapshot(at: kept)
            settle(job.row, layers: nil, keptTokens: Array(job.row.request.tokens.prefix(kept)))
        }
        end(job.row, reason: .removed, layers: nil)
    }

    /// What this pass is for: a decode step, a chunk of the prompt being read and how many of its tokens, or nothing.
    private func nextWork() -> InterleavePolicy.Next? {
        guard let job = prefill else { return batch.isEmpty ? nil : .step }
        let rows = batch.isEmpty ? nil : batch.rows.map(\.priority).max()
        return interleave.next(prompt: job.row.priority, rows: rows, clock: clock)
    }

    private func prefillChunk(_ job: PrefillJob, tokens: Int) {
        let row = job.row
        if row.removed {
            prefill = nil
            end(row, reason: .removed, layers: nil)
            return
        }
        var upTo = min(job.done + max(1, tokens), job.suffix.count)
        // A chunk ends at each mark and at the turn boundary, so the cache stands exactly there once.
        if let stop = job.nextStop, upTo > stop { upTo = stop }
        let piece = Array(job.suffix[job.done ..< upTo])
        let last = upTo == job.suffix.count
        // A prompt read only to be kept samples nothing: its last chunk is evaluated as any other is.
        let keeps = row.request.prefillOnly
        if !piece.isEmpty {
            let input = MLXArray(piece.map(Int32.init)).reshaped(1, piece.count)
            let logits = model(input, cache: job.layers)
            if last, !keeps {
                row.pending = sample(logits: logits[0..., -1, 0...], row: row)
            } else {
                eval(job.layers.flatMap { $0.innerState() })
                if job.marks.contains(upTo) { job.slotCache?.snapshot(at: row.reused + upTo, mark: true) }
                if let pin = job.pin, upTo == pin { job.slotCache?.snapshot(at: row.reused + pin, pinned: true) }
            }
        } else if last, !keeps {
            // Nothing to prefill (an exact repeat): the last prompt token is fed as the first step.
            row.pending = Int32(row.request.tokens.last ?? 0)
        }
        job.done = upTo
        row.prefilled = upTo
        guard last else { return }
        prefill = nil
        row.prefillSeconds = Date().timeIntervalSince(job.startedAt)
        job.slotCache?.snapshot(at: row.promptCount)
        let admitted = StreamEvent.admitted(prefixReused: row.reused, prefilled: job.suffix.count,
                                            queueWait: row.admittedAt.map { $0.timeIntervalSince(row.queuedAt) } ?? 0,
                                            prefillSeconds: row.prefillSeconds)
        if keeps {
            // Read and kept: the layers stay with the slot, nothing joins the batch, and the slot holds the prompt.
            row.continuation.yield(admitted)
            end(row, reason: .prefilled, layers: nil)
            if let cache = job.slotCache { learnCosts(from: cache) }
            enforceSlotBudget()
            return
        }
        let layers = job.slotCache?.release() ?? job.layers
        row.continuation.yield(admitted)
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
        if let decodeScheduling {
            GPU.withScheduling(decodeScheduling) { asyncEval(sampled) }
        } else {
            asyncEval(sampled)
        }
        // The tokens just fed were sampled last step: read them now, while this step runs.
        let values = fed.asArray(Int32.self)
        var finished: Set<Int> = []
        for (b, row) in rows.enumerated() {
            let token = Int(values[b])
            row.fed.append(token)
            row.batchSteps += 1
            row.batchRowsSeen += count
            if row.removed {
                // PIN: A REMOVED STREAM HEARS ITS LAST TOKEN. It was fed this step, so its slot keeps it; unheard, the
                // request that continued the slot (a thought closed and fed on, Craft 2026-11-25) parted from the
                // cache one token short, and a cache with recurrent layers went back to its last snapshot to read the
                // whole generation again: 16,388 tokens in 31.7 s after a thought that filled its room.
                row.emitted += 1
                row.continuation.yield(.token(token))
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
                              decodeSeconds: row.admittedAt.map { max(0, Date().timeIntervalSince($0) - row.prefillSeconds) } ?? 0,
                              meanBatchSize: row.batchSteps > 0 ? Double(row.batchRowsSeen) / Double(row.batchSteps) : 0,
                              message: message, prefilled: row.prefilled, admission: row.admission, fromFile: row.fromFile)
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
        // PIN: A HELD SESSION GIVES WAY BEFORE A REQUEST IS REFUSED (2026-11-22). A turn waiting on its person (a
        // permission prompt, for as long as they take) holds its cache, and a held cache was never evicted: another
        // session short of room was refused where, before holds, it had evicted and gone on. The held turn's next round
        // reads its conversation again, which is what the hold spares it while there is room, and no more than that.
        var gaveWay = false
        while memory.active + estimate > policy.budgetBytes, evictLeastRecentlyUsed(except: row.slot, held: true) {
            gaveWay = true
            memory = policy.readMemory()
        }
        if gaveWay {
            policy.clearCache()
            memory = policy.readMemory()
        }
        // Nothing running will free anything; what is left is the model and what cannot go.
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
        // A prompt read only to be kept never joins the batch, so it pads no row.
        if !row.request.prefillOnly, !batch.isEmpty, row.request.tokens.count > longest {
            bytes += (row.request.tokens.count - longest) * kvBytesPerToken * batch.count
        }
        return bytes
    }

    /// What idle sessions' caches hold together: the sessions no stream is using and no turn holds.
    private var idleSlotBytes: Int {
        slots.values.filter { $0.inFlight == nil && !$0.isHeld }.compactMap(\.cache).reduce(0) { $0 + Self.bytes($1) }
    }

    private static func bytes(_ cache: CheckpointedCache) -> Int { cache.kvBytes + cache.recurrentBytes + cache.snapshotBytes }

    /// Drops the least recently used idle session's cache. False when every session is in use or held. With `held`,
    /// the sessions a turn holds are taken too, after every other: a held slot stays, emptied, so its turn still holds
    /// what it reads next.
    @discardableResult
    private func evictLeastRecentlyUsed(except keep: SlotID? = nil, held: Bool = false) -> Bool {
        let idle = slots.filter { $0.value.inFlight == nil && (held || !$0.value.isHeld) && $0.value.cache != nil && $0.key != keep }
        guard let victim = idle.min(by: { a, b in (a.value.isHeld ? 1 : 0, a.value.lastUsed) < (b.value.isHeld ? 1 : 0, b.value.lastUsed) }) else {
            return false
        }
        if victim.value.isHeld {
            victim.value.cache = nil
            victim.value.tokens = []
        } else {
            slots[victim.key] = nil
        }
        evictions += 1
        return true
    }

    /// Holds sessions to the policy: how many keep a cache, what the idle ones hold, what the snapshots they share hold.
    ///
    /// PIN: THE SNAPSHOT BOUND IS OVER WHAT THE SESSIONS SHARE, AND NEVER UNDER ONE RING (2026-11-22). Each session
    /// keeps its pin and its last mark outside it (`CheckpointedCache.dropSharedSnapshot`), and the bound is at least
    /// one ring of the model's own snapshots, whatever the policy's megabytes say: a number chosen for a model whose
    /// snapshot was a few megabytes left a model whose snapshot is 164 MiB room for six among every session.
    private func enforceSlotBudget() {
        var freed = false
        while slots.values.filter({ $0.cache != nil }).count > policy.maxResidentSlots, evictLeastRecentlyUsed() { freed = true }
        while idleSlotBytes > policy.slotBudgetBytes, evictLeastRecentlyUsed() { freed = true }
        let bound = max(policy.snapshotBudgetBytes, configuration.checkpointRing * recurrentBytesPerRow)
        while slots.values.compactMap(\.cache).reduce(0, { $0 + $1.sharedSnapshotBytes }) > bound {
            // The oldest shared snapshot of the session with the most, then of the least recently used.
            let holders = slots.values.filter { ($0.cache?.sharedSnapshotCount ?? 0) > 0 }
            guard let holder = holders.max(by: { a, b in
                (a.cache?.sharedSnapshotCount ?? 0, b.lastUsed) < (b.cache?.sharedSnapshotCount ?? 0, a.lastUsed)
            }), holder.cache?.dropSharedSnapshot() == true else { break }
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
        if let cache = slot.cache { learnCosts(from: cache) }
        enforceSlotBudget()
    }

    /// What a token and a row cost, learned from a session that just settled: what admission estimates with.
    private func learnCosts(from cache: CheckpointedCache) {
        guard cache.position > 0 else { return }
        kvBytesPerToken = max(kvBytesPerToken, cache.kvBytes / cache.position)
        recurrentBytesPerRow = max(recurrentBytesPerRow, cache.recurrentBytes)
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
        stats.suspended = suspended.count
        stats.prefillTokensPerSecond = clock.prefillTokensPerSecond ?? 0
        stats.prefillShare = prefill.flatMap { job in
            batch.isEmpty ? nil : interleave.share(prompt: job.row.priority, rows: batch.rows.map(\.priority).max())
        }
        stats.memory = memory(at: now)
        stats.heldBack = heldBack
        stats.held = held
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
                              generated: row.emitted, tokensPerSecond: row.tokensPerSecond,
                              admission: row.admission, fromFile: row.fromFile)
        }
        stats.streams = batch.rows.map { stat($0, phase: .decode) }
        if let job = prefill { stats.streams.append(stat(job.row, phase: .prefill(done: job.done, total: job.suffix.count))) }
        stats.streams += suspended.map { stat($0.row, phase: .prefill(done: $0.done, total: $0.suffix.count)) }
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
        for job in [prefill].compactMap({ $0 }) + suspended where job.slotCache == nil {
            for layer in job.layers {
                let bytes = layer.innerState().reduce(0) { $0 + $1.nbytes }
                if layer is ArraysCache { memory.recurrentBytes += bytes } else { memory.kvBytes += bytes }
            }
        }
        for (id, slot) in slots {
            guard let cache = slot.cache else { continue }
            let stat = SlotStat(slot: id, tokens: slot.tokens.count, kvBytes: cache.kvBytes, recurrentBytes: cache.recurrentBytes,
                                snapshots: cache.snapshotCount, snapshotBytes: cache.snapshotBytes,
                                inFlight: slot.inFlight != nil, held: slot.isHeld, lastUsed: slot.lastUsed)
            if slot.isHeld { memory.heldBytes += Self.bytes(cache) }
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
