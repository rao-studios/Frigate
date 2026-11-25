@testable import FrigateParallel
import MLX
import MLXLMCommon
import XCTest

/// The parallel engine's bookkeeping on a stub model: what a session holds between rounds, and when
/// it lets go. The numbers a real model's kernels produce are the live test's and the bench's.
final class ParallelEngineTests: EngineTestCase, @unchecked Sendable {
    private func request(slot: SlotID?, tokens: Int = 40, maxTokens: Int = 4) -> StreamRequest {
        StreamRequest(tokens: Array(1 ... tokens), sampling: GenerateParameters(temperature: 0), maxTokens: maxTokens,
                      stopTokens: [], slot: slot)
    }

    /// Runs a stream to its end and returns how it finished.
    @discardableResult
    private func finish(_ handle: StreamHandle) async throws -> FinishInfo? {
        var info: FinishInfo?
        for try await event in handle.events { if case .finished(let done) = event { info = done } }
        return info
    }

    /// The engine publishes its stats once a loop: wait for them to say what the test expects.
    private func stats(_ engine: ParallelEngine, until settled: (ParallelStats) -> Bool) async -> ParallelStats {
        for _ in 0 ..< 300 {
            let stats = engine.stats
            if settled(stats) { return stats }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return engine.stats
    }

    func testASessionKeepsItsCacheBetweenRoundsUntilForgotten() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let slot = SlotID()
        let info = try await finish(engine.insert(request(slot: slot)))
        XCTAssertEqual(info?.reason, .maxTokens)
        XCTAssertEqual(info?.generated, 4)
        let kept = await stats(engine) { $0.residentSlots == 1 }
        XCTAssertEqual(kept.residentSlots, 1, "the session's cache stays for its next round")

        // The next round continues the slot: only what is new is prefilled.
        var reused = 0
        for try await event in engine.insert(request(slot: slot, tokens: 50)).events {
            if case .admitted(let prefix, _, _, _) = event { reused = prefix }
        }
        XCTAssertGreaterThan(reused, 0)

        engine.forget(slot: slot)
        let gone = await stats(engine) { $0.residentSlots == 0 }
        XCTAssertEqual(gone.residentSlots, 0)
    }

    /// PIN: a slot forgotten while its stream runs stays forgotten. The engine used to drop the cache
    /// only to adopt the row's layers back into the slot when the row ended, so a session reset in the
    /// middle of a generation kept everything it meant to free.
    func testASlotForgottenWhileItsStreamRunsStaysForgotten() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let slot = SlotID()
        let handle = engine.insert(request(slot: slot, maxTokens: 100_000))
        var events = handle.events.makeAsyncIterator()
        while let event = try await events.next() { if case .admitted = event { break } }
        engine.forget(slot: slot)
        engine.remove(handle.id)
        var reason: FinishReason?
        while let event = try await events.next() { if case .finished(let info) = event { reason = info.reason } }
        XCTAssertEqual(reason, .removed)
        let after = await stats(engine) { $0.residentSlots == 0 && $0.batchSize == 0 }
        XCTAssertEqual(after.residentSlots, 0, "what the row handed back was dropped, not adopted")
    }

    func testForgetAllDropsEverySession() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        for _ in 0 ..< 3 { try await finish(engine.insert(request(slot: SlotID()))) }
        let held = await stats(engine) { $0.residentSlots == 3 }
        XCTAssertEqual(held.residentSlots, 3)
        engine.forgetAll()
        let gone = await stats(engine) { $0.residentSlots == 0 }
        XCTAssertEqual(gone.residentSlots, 0)
    }

    /// A shut-down engine holds no session's cache.
    func testShutdownDropsWhatTheEngineHeld() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        try await finish(engine.insert(request(slot: SlotID())))
        _ = await stats(engine) { $0.residentSlots == 1 }
        engine.shutdown()
        let after = await stats(engine) { $0.residentSlots == 0 }
        XCTAssertEqual(after.residentSlots, 0)
    }

    // MARK: - A prompt read ahead, and who goes first

    private func readAhead(slot: SlotID?, tokens: Int, priority: StreamPriority = .background) -> StreamRequest {
        StreamRequest(tokens: Array(1 ... tokens), sampling: GenerateParameters(temperature: 0), maxTokens: 0, stopTokens: [],
                      priority: priority, slot: slot, prefillOnly: true)
    }

    /// What one stream said: how it was admitted, how many tokens it gave, how it finished.
    private struct Heard {
        var admitted: (reused: Int, prefilled: Int)?
        var tokens = 0
        var ids: [Int] = []
        var finished: FinishInfo?
    }

    private func hear(_ handle: StreamHandle) async throws -> Heard {
        var heard = Heard()
        for try await event in handle.events {
            switch event {
            case .admitted(let reused, let prefilled, _, _): heard.admitted = (reused, prefilled)
            case .token(let id): heard.tokens += 1; heard.ids.append(id)
            case .finished(let info): heard.finished = info
            }
        }
        return heard
    }

    /// Which streams finished first, in order.
    private final class Order: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: [String] = []
        var names: [String] { lock.withLock { stored } }
        func note(_ name: String) { lock.withLock { stored.append(name) } }
    }

    /// Hears a stream to its end, then notes that it finished.
    private func hear(_ handle: StreamHandle, noting name: String, in order: Order) async throws -> Heard {
        let heard = try await hear(handle)
        order.note(name)
        return heard
    }

    /// Waits until some stream has read part of its prompt.
    private func reading(_ engine: ParallelEngine) async {
        _ = await stats(engine) { stats in
            stats.streams.contains { if case .prefill(let done, _) = $0.phase { return done > 0 }; return false }
        }
    }

    /// An engine that reads in chunks of 32, so a long prompt takes many passes.
    private func smallChunks() throws -> ParallelEngine {
        var configuration = ParallelEngine.Configuration()
        configuration.prefillChunk = 32
        return try ParallelEngine(model: StubLanguageModel(), configuration: configuration)
    }

    /// PIN: A REMOVED STREAM HEARS ITS LAST TOKEN. A stream removed while it decodes hears every token its slot keeps, so
    /// a request that goes on from it (a thought closed and fed on, in Craft) reads only what is new. Its last token was
    /// fed and kept but never heard, and a cache with a recurrent layer went back to its last snapshot to read the whole
    /// reply again: 16,388 tokens in 31.7 s after a thought that filled its room.
    func testARemovedStreamHearsEveryTokenItsSlotKeeps() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let slot = SlotID()
        let handle = engine.insert(request(slot: slot, tokens: 20, maxTokens: 100_000))
        var ids: [Int] = []
        var finished: FinishInfo?
        for try await event in handle.events {
            switch event {
            case .token(let id):
                ids.append(id)
                if ids.count == 5 { engine.remove(handle.id) }
            case .finished(let info): finished = info
            case .admitted: break
            }
        }
        XCTAssertEqual(finished?.reason, .removed)
        let kept = await stats(engine) { $0.memory.slots.first?.inFlight == false }
        XCTAssertEqual(kept.memory.slots.first?.tokens, 20 + ids.count, "the slot keeps the prompt and every token heard")
        let next = try await hear(engine.insert(StreamRequest(tokens: Array(1 ... 20) + ids + [3], sampling: GenerateParameters(temperature: 0),
                                                              maxTokens: 1, stopTokens: [], slot: slot)))
        XCTAssertEqual(next.admitted?.reused, 20 + ids.count, "it goes on from where the stream stood")
        XCTAssertEqual(next.admitted?.prefilled, 1, "and reads only what is new")
    }

    /// A request's components shape its sampling as a `TokenIterator`'s do: their processor sees the row's logits after
    /// the penalties, here keeping the stub's one token from the first sample and no other.
    func testARequestsComponentsShapeItsSampling() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let barred = GenerationComponents().appendingLogitProcessor { FirstBarred(token: StubLanguageModel.token) }
        let heard = try await hear(engine.insert(StreamRequest(tokens: Array(1 ... 20), sampling: GenerateParameters(temperature: 0),
                                                               maxTokens: 3, stopTokens: [], components: barred)))
        XCTAssertEqual(heard.ids.count, 3)
        XCTAssertNotEqual(heard.ids.first, StubLanguageModel.token, "the first sample is kept from it")
        XCTAssertEqual(Array(heard.ids.dropFirst()), [StubLanguageModel.token, StubLanguageModel.token], "the rest are the model's")
    }

    /// Keeps one token from the first sample.
    private struct FirstBarred: LogitProcessor {
        let token: Int
        var sampled = false
        mutating func prompt(_ prompt: MLXArray) {}
        func process(logits: MLXArray) -> MLXArray {
            guard !sampled, let barred = SuppressTokensProcessor(tokenIds: [token]) else { return logits }
            return barred.process(logits: logits)
        }
        mutating func didSample(token: MLXArray) { sampled = true }
    }

    /// PIN: a prompt read ahead is kept in its slot and nothing else: no token is sampled, no row joins the batch, and the
    /// request that follows reads only what comes after it. Read again, it reads nothing.
    func testAPromptReadAheadIsKeptInItsSlotAndTheNextRequestContinuesIt() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let slot = SlotID()
        let read = try await hear(engine.insert(readAhead(slot: slot, tokens: 40)))
        XCTAssertEqual(read.tokens, 0, "nothing is sampled")
        XCTAssertEqual(read.admitted?.reused, 0)
        XCTAssertEqual(read.admitted?.prefilled, 40)
        XCTAssertEqual(read.finished?.reason, .prefilled)
        XCTAssertEqual(read.finished?.prefilled, 40)
        let held = await stats(engine) { $0.memory.slots.first?.inFlight == false }
        XCTAssertEqual(held.memory.slots.first?.tokens, 40, "the slot holds the prompt and nothing after it")
        XCTAssertEqual(held.memory.slots.first?.snapshots, 1, "a snapshot at the prompt's end")
        XCTAssertEqual(held.batchSize, 0, "nothing joined the batch")

        let again = try await hear(engine.insert(readAhead(slot: slot, tokens: 40)))
        XCTAssertEqual(again.admitted?.reused, 40)
        XCTAssertEqual(again.admitted?.prefilled, 0, "a prompt the slot holds is not read twice")
        XCTAssertEqual(again.finished?.reason, .prefilled)

        let next = try await hear(engine.insert(request(slot: slot, tokens: 50)))
        XCTAssertEqual(next.admitted?.reused, 40, "the next request continues the prompt read ahead")
        XCTAssertEqual(next.admitted?.prefilled, 10)
        XCTAssertEqual(next.finished?.reason, .maxTokens)
    }

    /// PIN: a slot that holds the whole prompt read ahead and more is left as it is. Rewound to the prompt's end, it would
    /// throw away a reply the next request shares, only to read it again.
    func testAPromptReadAheadThatTheSlotHoldsLeavesItsCacheAlone() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let slot = SlotID()
        try await finish(engine.insert(request(slot: slot, tokens: 40, maxTokens: 4)))
        let read = try await hear(engine.insert(readAhead(slot: slot, tokens: 40)))
        XCTAssertEqual(read.admitted?.reused, 40)
        XCTAssertEqual(read.admitted?.prefilled, 0)
        let held = await stats(engine) { $0.memory.slots.first?.inFlight == false }
        XCTAssertEqual(held.memory.slots.first?.tokens, 44, "the prompt and the four tokens fed, still there")
        // The stub samples one token, always; the next prompt carries what was fed, then something new.
        let fed = Array(repeating: StubLanguageModel.token, count: 4)
        let next = try await hear(engine.insert(StreamRequest(tokens: Array(1 ... 40) + fed + [3], sampling: GenerateParameters(temperature: 0),
                                                              maxTokens: 1, stopTokens: [], slot: slot)))
        XCTAssertEqual(next.admitted?.reused, 44)
        XCTAssertEqual(next.admitted?.prefilled, 1)
    }

    // MARK: - A session's cache as a file

    /// What a call that should be refused said, or nil when it was not.
    private func refusal(_ body: () async throws -> SlotFile) async -> String? {
        do {
            _ = try await body()
            return nil
        } catch {
            return "\(error)"
        }
    }

    /// PIN: a slot kept to a file and taken by another engine is the slot it was (`frigate-slot/1`): the next request
    /// reuses what it held, and a rewind still reaches the ring's snapshot, the pinned turn boundary and the mark, each
    /// said in the request's `admission`. Kept up to the mark, it is the stable prefix alone, which a fresh slot takes
    /// and continues. A file another model wrote, one cut short, a slot in use and a slot that holds nothing are refused,
    /// and a shut-down engine answers at once.
    func testASlotKeptToAFileIsTakenBackAndContinued() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("frigate-slot-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        let url = folder.appendingPathComponent("gpu.safetensors")
        let model = ["model": "stub"]

        // The stable prefix ends at 10, marked; the ask at 30, pinned; the ring's snapshot is at the prompt's end, 40;
        // four tokens are fed after it.
        let first = try ParallelEngine(model: StubLanguageModel())
        let slot = SlotID()
        let opened = try await finish(first.insert(StreamRequest(tokens: Array(1 ... 40), sampling: GenerateParameters(temperature: 0),
                                                                 maxTokens: 4, stopTokens: [], slot: slot, pinAt: 30, marksAt: [10])))
        XCTAssertEqual(opened?.admission, .fresh)
        XCTAssertEqual(opened?.fromFile, false)
        let held = await stats(first) { $0.memory.slots.first?.inFlight == false }
        let before = try XCTUnwrap(held.memory.slots.first)
        XCTAssertEqual(before.snapshots, 3)
        let kept = try await first.keep(slot: slot, to: url, metadata: model)
        XCTAssertEqual(kept.tokens, 44)
        XCTAssertEqual(kept.start, 0)
        XCTAssertGreaterThan(kept.bytes, 0)
        XCTAssertEqual(kept.metadata, model)
        first.shutdown()
        let closed = await refusal { try await first.keep(slot: slot, to: url) }
        XCTAssertEqual(closed, ParallelError.shutDown.description)

        let second = try ParallelEngine(model: StubLanguageModel())
        defer { second.shutdown() }
        let taken = SlotID()
        let torn = folder.appendingPathComponent("torn.safetensors")
        let whole = try Data(contentsOf: url)
        try whole.prefix(whole.count / 2).write(to: torn)
        let refused: [(String?, String, String)] = [
            (await refusal { try await second.take(slot: taken, from: url, expecting: ["model": "other"]) },
             "its model is \"stub\", not \"other\"", "another model's"),
            (await refusal { try await second.take(slot: taken, from: torn) }, "cut short", "a file cut short"),
            (await refusal { try await second.take(slot: taken, from: folder.appendingPathComponent("none.safetensors")) },
             "no such file", "no file"),
            (await refusal { try await second.keep(slot: taken, to: torn) }, "holds no cache", "a slot that holds nothing"),
        ]
        for (said, expected, label) in refused {
            XCTAssertTrue(said?.contains(expected) == true, "\(label): \(said ?? "taken")")
        }
        XCTAssertEqual(second.stats.residentSlots, 0, "nothing refused was taken")

        let file = try await second.take(slot: taken, from: url, expecting: model)
        XCTAssertEqual(file.tokens, 44)
        let back = await stats(second) { $0.residentSlots == 1 }
        let after = try XCTUnwrap(back.memory.slots.first)
        XCTAssertEqual(after.tokens, before.tokens)
        XCTAssertEqual(after.snapshots, before.snapshots, "the ring's snapshot, the pinned one and the mark")
        XCTAssertEqual(after.recurrentBytes, before.recurrentBytes)
        XCTAssertEqual(after.snapshotBytes, before.snapshotBytes)

        // The next request continues past the prompt (the ring's snapshot); a later one that parts at 35 rewinds to the
        // pinned turn boundary, and one that parts at 20 (a history rewritten after the stable prefix) to the mark.
        let next = try await hear(second.insert(request(slot: taken, tokens: 50)))
        XCTAssertEqual(next.admitted?.reused, 40)
        XCTAssertEqual(next.admitted?.prefilled, 10)
        XCTAssertEqual(next.finished?.admission, .rewind(.ring))
        XCTAssertEqual(next.finished?.fromFile, true, "the first request after a take says so")
        let parted = try await hear(second.insert(StreamRequest(tokens: Array(1 ... 35) + [3], sampling: GenerateParameters(temperature: 0),
                                                                maxTokens: 1, stopTokens: [], slot: taken)))
        XCTAssertEqual(parted.admitted?.reused, 30)
        XCTAssertEqual(parted.finished?.admission, .rewind(.pin))
        XCTAssertEqual(parted.finished?.fromFile, false)
        let rewritten = try await hear(second.insert(StreamRequest(tokens: Array(1 ... 20) + [3], sampling: GenerateParameters(temperature: 0),
                                                                   maxTokens: 1, stopTokens: [], slot: taken)))
        XCTAssertEqual(rewritten.admitted?.reused, 10)
        XCTAssertEqual(rewritten.finished?.admission, .rewind(.mark))
        let appended = try await hear(second.insert(StreamRequest(tokens: Array(1 ... 20) + [3] + rewritten.ids + [7],
                                                                  sampling: GenerateParameters(temperature: 0),
                                                                  maxTokens: 1, stopTokens: [], slot: taken)))
        XCTAssertEqual(appended.finished?.admission, .append, "a prompt that goes on from where the cache stands")

        // Kept up to the mark: the stable prefix and nothing after it, taken by a fresh slot, which continues it. A
        // stretch ends only where the cache holds state, and begins before it ends.
        let prefixURL = folder.appendingPathComponent("prefix.safetensors")
        let prefix = try await second.keep(slot: taken, to: prefixURL, stretch: SlotStretch(upTo: 10), metadata: model)
        XCTAssertEqual(prefix.tokens, 10)
        let empty = await refusal { try await second.keep(slot: SlotID(), to: prefixURL, stretch: SlotStretch(upTo: 10)) }
        XCTAssertNotNil(empty, "a slot that holds nothing keeps no stretch")
        let stateless = await refusal { try await second.keep(slot: taken, to: torn, stretch: SlotStretch(upTo: 15)) }
        XCTAssertTrue(stateless?.contains("no state at token 15") == true, stateless ?? "kept")
        let backwards = await refusal { try await second.keep(slot: taken, to: torn, stretch: SlotStretch(from: 10, upTo: 10)) }
        XCTAssertTrue(backwards?.contains("not inside") == true, backwards ?? "kept")
        let opener = SlotID()
        _ = try await second.take(slot: opener, from: prefixURL, expecting: model)
        let fromPrefix = try await hear(second.insert(request(slot: opener, tokens: 40)))
        XCTAssertEqual(fromPrefix.admitted?.reused, 10)
        XCTAssertEqual(fromPrefix.finished?.admission, .append)
        XCTAssertEqual(fromPrefix.finished?.fromFile, true)
        second.forget(slot: opener)

        // A slot in use is neither written nor replaced.
        let long = second.insert(request(slot: taken, maxTokens: 100_000))
        var events = long.events.makeAsyncIterator()
        while let event = try await events.next() { if case .admitted = event { break } }
        let busyKeep = await refusal { try await second.keep(slot: taken, to: url) }
        let busyTake = await refusal { try await second.take(slot: taken, from: url, expecting: model) }
        XCTAssertTrue(busyKeep?.contains("still in use") == true, busyKeep ?? "kept")
        XCTAssertTrue(busyTake?.contains("still in use") == true, busyTake ?? "taken")
        second.remove(long.id)
        while let _ = try await events.next() {}
    }

    /// A session read with marks at 10 and 20, its ask pinned at 30, the ring's snapshot at the prompt's end, 40, and four
    /// tokens fed after it.
    private func marked(_ engine: ParallelEngine, slot: SlotID) async throws {
        try await finish(engine.insert(StreamRequest(tokens: Array(1 ... 40), sampling: GenerateParameters(temperature: 0),
                                                     maxTokens: 4, stopTokens: [], slot: slot, pinAt: 30, marksAt: [10, 20])))
        _ = await stats(engine) { $0.memory.slots.contains { $0.slot == slot && !$0.inFlight } }
    }

    /// PIN: a cache cut at its marks and kept as stretches is the cache it was once the stretches are taken back in
    /// order: kept whole again, it is byte for byte the file the first engine wrote whole. Each file holds only its own
    /// tokens' keys and values, with the state at its end.
    func testACacheKeptAsStretchesIsTakenBackAsAChain() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("frigate-stretch-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        func file(_ name: String) -> URL { folder.appendingPathComponent(name + ".safetensors") }
        let model = ["model": "stub"]

        let first = try ParallelEngine(model: StubLanguageModel())
        let slot = SlotID()
        try await marked(first, slot: slot)
        let before = try XCTUnwrap(first.stats.memory.slots.first)
        XCTAssertEqual(before.snapshots, 4, "two marks, the pin and the ring's snapshot")
        let whole = try await first.keep(slot: slot, to: file("whole"), metadata: model)
        let shared = try await first.keep(slot: slot, to: file("shared"), stretch: SlotStretch(upTo: 10), metadata: model)
        let project = try await first.keep(slot: slot, to: file("project"), stretch: SlotStretch(from: 10, upTo: 20), metadata: model)
        let session = try await first.keep(slot: slot, to: file("session"), stretch: SlotStretch(from: 20), metadata: model)
        first.shutdown()
        XCTAssertEqual([shared.start, shared.tokens], [0, 10])
        XCTAssertEqual([project.start, project.tokens], [10, 20])
        XCTAssertEqual([session.start, session.tokens], [20, 44])
        XCTAssertLessThan(session.bytes, whole.bytes, "a stretch holds less than the whole")
        let head = try XCTUnwrap(SlotFile.head(of: file("session")))
        XCTAssertEqual(head.file, session)
        XCTAssertEqual(head.tokens.count, 44, "every token before its end, so what it stands on can be read again")

        let second = try ParallelEngine(model: StubLanguageModel())
        defer { second.shutdown() }
        let taken = SlotID()
        let lone = await refusal { try await second.take(slot: taken, from: [file("project")], expecting: model) }
        XCTAssertTrue(lone?.contains("begins at token 10, and the slot holds 0") == true, lone ?? "taken")
        let gap = await refusal { try await second.take(slot: taken, from: [file("shared"), file("session")], expecting: model) }
        XCTAssertTrue(gap?.contains("does not begin where the 10 tokens before it end") == true, gap ?? "taken")
        XCTAssertEqual(second.stats.residentSlots, 0, "nothing refused was taken")

        let chain = try await second.take(slot: taken, from: [file("shared"), file("project"), file("session")], expecting: model)
        XCTAssertEqual([chain.start, chain.tokens], [0, 44])
        XCTAssertEqual(chain.bytes, shared.bytes + project.bytes + session.bytes)
        let back = await stats(second) { $0.residentSlots == 1 }
        let after = try XCTUnwrap(back.memory.slots.first)
        XCTAssertEqual(after.tokens, 44)
        XCTAssertEqual(after.snapshots, before.snapshots, "the marks at the two files' ends, the pin and the ring's snapshot")
        XCTAssertEqual(after.snapshotBytes, before.snapshotBytes)
        XCTAssertEqual(after.recurrentBytes, before.recurrentBytes)
        _ = try await second.keep(slot: taken, to: file("again"), metadata: model)
        XCTAssertEqual(try Data(contentsOf: file("again")), try Data(contentsOf: file("whole")),
                       "kept whole again, it is the file the first engine wrote")

        // It rewinds as the cache it was cut from: past the prompt to the ring, and a system prompt changed after the
        // first mark to that mark.
        let next = try await hear(second.insert(request(slot: taken, tokens: 50)))
        XCTAssertEqual(next.admitted?.reused, 40)
        XCTAssertEqual(next.finished?.admission, .rewind(.ring))
        XCTAssertEqual(next.finished?.fromFile, true)
        let changed = try await hear(second.insert(StreamRequest(tokens: Array(1 ... 15) + [3], sampling: GenerateParameters(temperature: 0),
                                                                 maxTokens: 1, stopTokens: [], slot: taken)))
        XCTAssertEqual(changed.admitted?.reused, 10)
        XCTAssertEqual(changed.finished?.admission, .rewind(.mark))
    }

    /// PIN: a stretch stands on the tokens before it, however the slot came to hold them. A slot that read the first ten
    /// tokens itself takes the stretches after them; one that read other tokens is refused, token for token.
    func testAStretchIsTakenOntoWhatTheSlotRead() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("frigate-stretch-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        func file(_ name: String) -> URL { folder.appendingPathComponent(name + ".safetensors") }
        let first = try ParallelEngine(model: StubLanguageModel())
        let slot = SlotID()
        try await marked(first, slot: slot)
        _ = try await first.keep(slot: slot, to: file("whole"))
        _ = try await first.keep(slot: slot, to: file("project"), stretch: SlotStretch(from: 10, upTo: 20))
        _ = try await first.keep(slot: slot, to: file("session"), stretch: SlotStretch(from: 20))
        first.shutdown()

        let second = try ParallelEngine(model: StubLanguageModel())
        defer { second.shutdown() }
        let reader = SlotID()
        var prefix = readAhead(slot: reader, tokens: 10)
        prefix.marksAt = [10]
        let read = try await hear(second.insert(prefix))
        XCTAssertEqual(read.finished?.reason, .prefilled)
        let held = await stats(second) { $0.memory.slots.first?.inFlight == false }
        XCTAssertEqual(held.memory.slots.first?.snapshots, 2, "a prompt read only to be kept marks its own end, beside the ring's snapshot there")
        let onto = try await second.take(slot: reader, from: [file("project"), file("session")])
        XCTAssertEqual([onto.start, onto.tokens], [10, 44])
        _ = try await second.keep(slot: reader, to: file("again"))
        XCTAssertEqual(try Data(contentsOf: file("again")), try Data(contentsOf: file("whole")),
                       "read then taken, it is the cache that read every token")
        let next = try await hear(second.insert(request(slot: reader, tokens: 50)))
        XCTAssertEqual(next.admitted?.reused, 40)
        XCTAssertEqual(next.finished?.fromFile, true)

        let other = SlotID()
        try await finish(second.insert(StreamRequest(tokens: Array(2 ... 11), sampling: GenerateParameters(temperature: 0), maxTokens: 0,
                                                     stopTokens: [], priority: .background, slot: other, prefillOnly: true)))
        let refused = await refusal { try await second.take(slot: other, from: [file("project")]) }
        XCTAssertTrue(refused?.contains("does not begin where the 10 tokens before it end") == true, refused ?? "taken")
        let stats = await stats(second) { $0.memory.slots.contains { $0.slot == other } }
        XCTAssertEqual(stats.memory.slots.first { $0.slot == other }?.tokens, 10, "a slot that refuses keeps what it held")
    }

    /// PIN: from a file only when the file's cache was gone on from. A prompt that shares nothing with the cache a slot
    /// took reads from the start, and its record must not say a file spared it anything.
    func testAPromptThatSharesNothingWithATakenCacheIsNotFromAFile() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("frigate-stretch-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        let url = folder.appendingPathComponent("gpu.safetensors")
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let slot = SlotID()
        try await marked(engine, slot: slot)
        _ = try await engine.keep(slot: slot, to: url)
        let taken = SlotID()
        _ = try await engine.take(slot: taken, from: url)
        let other = try await hear(engine.insert(StreamRequest(tokens: Array(100 ... 140), sampling: GenerateParameters(temperature: 0),
                                                               maxTokens: 1, stopTokens: [], slot: taken)))
        XCTAssertEqual(other.admitted?.reused, 0)
        XCTAssertEqual(other.finished?.admission, .fresh)
        XCTAssertEqual(other.finished?.fromFile, false)
    }

    /// PIN: a cache with no recurrent layer keeps its marks through a file, and is cut anywhere: it needs no state where
    /// a stretch ends. Read back, it had lost its mark, and a keep up to it threw.
    func testACacheThatTrimsByPositionKeepsItsMarksThroughAFile() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("frigate-stretch-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        func file(_ name: String) -> URL { folder.appendingPathComponent(name + ".safetensors") }
        let first = try ParallelEngine(model: StubLanguageModel(recurrent: false))
        let slot = SlotID()
        try await marked(first, slot: slot)
        XCTAssertEqual(first.stats.memory.slots.first?.snapshots, 2, "its two marks: a position each, no state")
        _ = try await first.keep(slot: slot, to: file("whole"))
        first.shutdown()

        let second = try ParallelEngine(model: StubLanguageModel(recurrent: false))
        defer { second.shutdown() }
        let taken = SlotID()
        _ = try await second.take(slot: taken, from: file("whole"))
        let back = await stats(second) { $0.residentSlots == 1 }
        XCTAssertEqual(back.memory.slots.first?.snapshots, 2, "the marks came back with it")
        let prefix = try await second.keep(slot: taken, to: file("prefix"), stretch: SlotStretch(upTo: 20))
        XCTAssertEqual(prefix.tokens, 20)
        let anywhere = try await second.keep(slot: taken, to: file("anywhere"), stretch: SlotStretch(from: 20, upTo: 33))
        XCTAssertEqual([anywhere.start, anywhere.tokens], [20, 33])
        let refused = await refusal { try await second.take(slot: SlotID(), from: file("whole"), expecting: ["model": "stub"]) }
        XCTAssertTrue(refused?.contains("its model is missing") == true, refused ?? "taken")
        let hybrid = try ParallelEngine(model: StubLanguageModel())
        defer { hybrid.shutdown() }
        let layout = await refusal { try await hybrid.take(slot: SlotID(), from: file("whole")) }
        XCTAssertTrue(layout?.contains("another cache layout") == true, layout ?? "taken")
    }

    /// PIN: a hold ends with its scope, however the scope ends. Inside it the session is not idle: a trim of idle
    /// sessions leaves it alone. After it, thrown out of or not, the session is idle again.
    func testAHoldEndsWithItsScope() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let slot = SlotID()
        try await finish(engine.insert(request(slot: slot)))
        struct Stopped: Error {}
        for throwing in [false, true] {
            do {
                try await engine.holding(slot: slot) {
                    // Two scopes at once, as a turn inside a turn: the inner one's end leaves the outer's hold standing.
                    await engine.holding(slot: slot) {}
                    engine.trimSlots(toBytes: 0)
                    let held = await stats(engine) { $0.memory.slots.first?.held == true }
                    XCTAssertEqual(held.residentSlots, 1, "held, it is not trimmed as an idle session is")
                    XCTAssertEqual(held.memory.heldBytes, held.memory.engineBytes)
                    if throwing { throw Stopped() }
                }
            } catch is Stopped {}
            let free = await stats(engine) { $0.memory.slots.first?.held == false }
            XCTAssertEqual(free.memory.slots.first?.held, false, throwing ? "thrown out of, the hold ended" : "the hold ended")
            XCTAssertEqual(free.memory.heldBytes, 0)
        }
        engine.trimSlots(toBytes: 0)
        let gone = await stats(engine) { $0.residentSlots == 0 }
        XCTAssertEqual(gone.residentSlots, 0, "let go, it is idle and trimmed")
    }

    /// PIN: a held session gives way before a request is refused. A turn waiting on its person holds its cache for as
    /// long as they take; another session short of room was refused where, before holds, it had evicted and gone on.
    func testAHeldSessionGivesWayBeforeARequestIsRefused() async throws {
        let gauge = Gauge()
        // Over the budget until a cache goes: what an evicted cache held comes back when the engine gives the buffer cache
        // back, which it does only after an eviction or when that alone admits the request. A budget in megabytes: once
        // a session has ended the engine knows what a prompt costs, and a prompt's own cost must fit.
        let budget = 64 * 1_048_576
        let engine = try governed(gauge, budget: budget, cleared: { if gauge.value.active > budget { gauge.set(budget / 10) } })
        defer { engine.shutdown() }
        let waiting = SlotID()
        try await finish(engine.insert(request(slot: waiting)))
        try await engine.holding(slot: waiting) {
            _ = await stats(engine) { $0.memory.slots.first?.held == true }
            gauge.set(budget + budget / 2)
            let info = try await finish(engine.insert(request(slot: SlotID())))
            XCTAssertEqual(info?.reason, .maxTokens, info?.message ?? "admitted once the held session gave way")
            let after = await stats(engine) { $0.evictions == 1 && $0.residentSlots == 1 }
            XCTAssertEqual(after.evictions, 1)
            XCTAssertFalse(after.memory.slots.contains { $0.slot == waiting }, "the held session's cache went")
            // Its turn goes on, reading its conversation again, and what it reads is held as before.
            let again = try await hear(engine.insert(request(slot: waiting)))
            XCTAssertEqual(again.finished?.admission, .fresh)
            let held = await stats(engine) { $0.memory.slots.contains { $0.slot == waiting && !$0.inFlight } }
            XCTAssertEqual(held.memory.slots.first { $0.slot == waiting }?.held, true, "still inside its scope")
        }
    }

    /// PIN: the snapshot bound is over what the sessions share, and never under one ring of the model's own. Three
    /// sessions on a ring of one, with a bound of nothing: one ring snapshot stays among them, and every session keeps
    /// its pin and its last mark.
    func testTheSnapshotBoundSparesEachSessionsPinAndLastMark() async throws {
        var configuration = ParallelEngine.Configuration()
        configuration.checkpointRing = 1
        configuration.memory.snapshotBudgetBytes = 0
        let engine = try ParallelEngine(model: StubLanguageModel(), configuration: configuration)
        defer { engine.shutdown() }
        let slots = [SlotID(), SlotID(), SlotID()]
        for slot in slots { try await marked(engine, slot: slot) }
        let settled = await stats(engine) { $0.memory.slots.count == 3 && $0.memory.slots.map(\.snapshots).reduce(0, +) == 7 }
        XCTAssertEqual(settled.memory.slots.map(\.snapshots).sorted(), [2, 2, 3],
                       "the pin and the last mark each, and the one ring snapshot the bound leaves among them")
        XCTAssertEqual(settled.residentSlots, 3)
        // What stays is what a follow-up and a rewritten history rewind to.
        for slot in slots {
            let rewritten = try await hear(engine.insert(StreamRequest(tokens: Array(1 ... 25) + [3], sampling: GenerateParameters(temperature: 0),
                                                                       maxTokens: 1, stopTokens: [], slot: slot)))
            XCTAssertEqual(rewritten.admitted?.reused, 20)
            XCTAssertEqual(rewritten.finished?.admission, .rewind(.mark))
        }
    }

    func testAPromptReadAheadWithoutASlotIsRefused() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let heard = try await hear(engine.insert(readAhead(slot: nil, tokens: 40)))
        XCTAssertEqual(heard.finished?.reason, .failed)
        XCTAssertNil(heard.admitted)
        XCTAssertTrue(heard.finished?.message?.contains("slot") ?? false, heard.finished?.message ?? "no message")
    }

    /// A prompt read ahead and stopped before its end keeps what it read: the next request on the slot starts there.
    func testAPromptReadAheadThatIsStoppedKeepsWhatItRead() async throws {
        let engine = try smallChunks()
        defer { engine.shutdown() }
        let slot = SlotID()
        let long = 100_000
        let handle = engine.insert(readAhead(slot: slot, tokens: long))
        await reading(engine)
        engine.remove(handle.id)
        let heard = try await hear(handle)
        XCTAssertEqual(heard.finished?.reason, .removed)
        let read = try XCTUnwrap(heard.finished?.prefilled)
        XCTAssertGreaterThan(read, 0)
        XCTAssertLessThan(read, long)
        let next = try await hear(engine.insert(request(slot: slot, tokens: read + 8, maxTokens: 1)))
        XCTAssertEqual(next.admitted?.reused, read, "the next request starts where the stopped one got to")
        XCTAssertEqual(next.admitted?.prefilled, 8)
    }

    /// PIN: a stream says how it found its slot's cache while its prompt is still being read, not only when it ends
    /// (`StreamStat.admission`): whoever shows a long read can say why it is that long. A read from nothing says fresh;
    /// what it read stays in the slot, and the prompt that goes on from there says append.
    func testAStreamBeingReadSaysHowItFoundItsCache() async throws {
        let engine = try smallChunks()
        defer { engine.shutdown() }
        let slot = SlotID()
        func whileReading(_ handle: StreamHandle) async -> StreamStat? {
            await stats(engine) { stats in
                stats.streams.contains { stat in
                    guard stat.id == handle.id, case .prefill(let done, _) = stat.phase else { return false }
                    return done > 0
                }
            }.streams.first { $0.id == handle.id }
        }
        let first = engine.insert(readAhead(slot: slot, tokens: 100_000))
        let opened = await whileReading(first)
        XCTAssertEqual(opened?.admission, .fresh)
        XCTAssertEqual(opened?.fromFile, false)
        engine.remove(first.id)
        let stopped = try await hear(first)
        let read = try XCTUnwrap(stopped.finished?.prefilled)
        let second = engine.insert(readAhead(slot: slot, tokens: read + 100_000))
        let continued = await whileReading(second)
        XCTAssertEqual(continued?.admission, .append, "it goes on from what the stopped read left in the slot")
        engine.remove(second.id)
        _ = try await hear(second)
    }

    /// PIN: a request someone waits on is never held behind a prompt no one does: it sets the prompt being read aside,
    /// runs to its end, and the prompt reads on from where it stood, every token read once.
    func testARequestSomeoneWaitsOnGoesAheadOfAPromptReadAhead() async throws {
        let engine = try smallChunks()
        defer { engine.shutdown() }
        let long = 20_000
        let order = Order()
        let background = engine.insert(readAhead(slot: SlotID(), tokens: long))
        async let read = hear(background, noting: "read ahead", in: order)
        await reading(engine)
        let ask = try await hear(engine.insert(request(slot: SlotID(), tokens: 40, maxTokens: 4)))
        order.note("ask")
        XCTAssertEqual(ask.finished?.reason, .maxTokens)
        let heard = try await read
        XCTAssertEqual(order.names, ["ask", "read ahead"])
        XCTAssertEqual(heard.finished?.reason, .prefilled)
        XCTAssertEqual(heard.finished?.prefilled, long, "set aside and read on: every token read once")
    }

    /// One of the prompt's own priority waits its turn, as every request always did.
    func testARequestOfTheSamePriorityWaitsForThePromptBeingRead() async throws {
        let engine = try smallChunks()
        defer { engine.shutdown() }
        let order = Order()
        let background = engine.insert(readAhead(slot: SlotID(), tokens: 20_000))
        async let read = hear(background, noting: "read ahead", in: order)
        await reading(engine)
        let same = StreamRequest(tokens: Array(1 ... 40), sampling: GenerateParameters(temperature: 0), maxTokens: 4, stopTokens: [],
                                 priority: .background, slot: SlotID())
        let other = try await hear(engine.insert(same))
        order.note("other")
        _ = try await read
        XCTAssertEqual(other.finished?.reason, .maxTokens)
        XCTAssertEqual(order.names, ["read ahead", "other"])
    }

    // MARK: - What the engine counts

    /// A session's bytes are its arrays' shapes. The stub's KV is 2 heads × 4 dims of float32 a token,
    /// keys and values (64 bytes), and its recurrent state a `[1, 3, 8]` and a `[1, 2, 4, 4]` float32 array.
    func testASessionsBytesAreItsArraysShapes() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let slot = SlotID()
        try await finish(engine.insert(request(slot: slot, tokens: 40, maxTokens: 4)))
        let stats = await stats(engine) { $0.memory.slots.first?.inFlight == false }
        let held = try XCTUnwrap(stats.memory.slots.first)
        XCTAssertEqual(held.slot, slot)
        XCTAssertEqual(held.tokens, 44, "the prompt and the four tokens fed")
        // Settled out of the batch as a contiguous copy of exactly its tokens.
        XCTAssertEqual(held.kvBytes, 44 * 64)
        XCTAssertEqual(held.recurrentBytes, 3 * 8 * 4 + 2 * 4 * 4 * 4)
        XCTAssertEqual(held.snapshots, 1, "one snapshot, at the prompt's end")
        XCTAssertEqual(held.snapshotBytes, held.recurrentBytes)
        XCTAssertFalse(held.inFlight)
        XCTAssertEqual(stats.memory.kvBytes, held.kvBytes)
        XCTAssertEqual(stats.memory.recurrentBytes, held.recurrentBytes)
        XCTAssertGreaterThan(stats.memory.activeBytes, 0)
        XCTAssertGreaterThan(stats.memory.peakBytes, 0)
    }

    /// Rows of different lengths share one buffer: the batch says how long its longest row is and how
    /// much padding the shorter one carries to line up with it.
    func testTheBatchReportsItsBuffersAndPadding() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let short = engine.insert(request(slot: SlotID(), tokens: 30, maxTokens: 100_000))
        let long = engine.insert(request(slot: SlotID(), tokens: 70, maxTokens: 100_000))
        let both = await stats(engine) { $0.batchSize == 2 }
        XCTAssertEqual(both.batchSize, 2)
        XCTAssertGreaterThanOrEqual(both.memory.batchMaxTokens, 70)
        XCTAssertGreaterThan(both.memory.paddingTokens, 0)
        XCTAssertEqual(both.memory.batchKVBytes % (2 * 256 * 64), 0, "two rows of 256-token steps")
        XCTAssertTrue(both.memory.slots.allSatisfy(\.inFlight))
        engine.remove(short.id)
        engine.remove(long.id)
        try await finish(short)
        try await finish(long)
    }

    /// An insert after shutdown ends at once rather than waiting for a thread that is gone.
    func testAStreamInsertedAfterShutdownEndsAtOnce() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        engine.shutdown()
        do {
            _ = try await finish(engine.insert(request(slot: nil)))
            XCTFail("a shut-down engine runs nothing")
        } catch let error as ParallelError {
            XCTAssertEqual(error.description, ParallelError.shutDown.description)
        }
    }

    // MARK: - The memory policy

    /// What the policy reads as the process's memory, set by the test.
    private final class Gauge: @unchecked Sendable {
        private let lock = NSLock()
        private var stored = (active: 0, cache: 0)
        var value: (active: Int, cache: Int) { lock.withLock { stored } }
        func set(_ active: Int) { lock.withLock { stored = (active, 0) } }
    }

    /// How often the engine gave the buffer cache back.
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var stored = 0
        var value: Int { lock.withLock { stored } }
        func add() { lock.withLock { stored += 1 } }
    }

    private func governed(_ gauge: Gauge, budget: Int = 1_000, highWater: Double = 0.85, maxResident: Int = .max,
                          clears: Counter = Counter(), cleared: (@Sendable () -> Void)? = nil,
                          model: StubLanguageModel = StubLanguageModel()) throws -> ParallelEngine {
        var configuration = ParallelEngine.Configuration()
        configuration.memory.budgetBytes = budget
        configuration.memory.highWater = highWater
        configuration.memory.maxResidentSlots = maxResident
        configuration.memory.readMemory = { gauge.value }
        configuration.memory.clearCache = { clears.add(); cleared?() }
        return try ParallelEngine(model: model, configuration: configuration)
    }

    /// Past the high-water mark a request waits while rows are decoding, and goes in once they give memory back.
    /// PIN: a request held back costs the batch nothing. The buffer cache goes back after an eviction or
    /// when that alone admits the request — never once a step while it waits, which would make every row
    /// allocate afresh for every token exactly while memory is tight.
    func testAHeldBackRequestLeavesTheBufferCacheAlone() async throws {
        let gauge = Gauge()
        let clears = Counter()
        let engine = try governed(gauge, clears: clears)
        defer { engine.shutdown() }
        let first = engine.insert(request(slot: SlotID(), maxTokens: 100_000))
        _ = await stats(engine) { $0.batchSize == 1 }
        gauge.set(900)
        let second = engine.insert(request(slot: SlotID(), maxTokens: 4))
        let held = await stats(engine) { $0.heldBack && $0.queued == 1 }
        XCTAssertTrue(held.heldBack, "held back while the first row decodes")
        XCTAssertEqual(held.batchSize, 1)
        let before = (clears: clears.value, generated: held.streams.map(\.generated).max() ?? 0)
        let later = await stats(engine) { ($0.streams.map(\.generated).max() ?? 0) >= before.generated + 50 }
        XCTAssertGreaterThanOrEqual(later.streams.map(\.generated).max() ?? 0, before.generated + 50, "the batch kept decoding")
        XCTAssertTrue(later.heldBack)
        XCTAssertEqual(clears.value, before.clears, "fifty steps held back, not one clear")
        gauge.set(100)
        let info = try await finish(second)
        XCTAssertEqual(info?.reason, .maxTokens, "admitted once memory came back")
        engine.remove(first.id)
        try await finish(first)
    }

    /// With nothing decoding that could give memory back, a request that cannot fit is refused, and says why.
    func testARequestThatCannotFitIsRefusedWithTheReason() async throws {
        let gauge = Gauge()
        gauge.set(1_500)
        let engine = try governed(gauge)
        defer { engine.shutdown() }
        let info = try await finish(engine.insert(request(slot: SlotID(), maxTokens: 4)))
        XCTAssertEqual(info?.reason, .failed)
        XCTAssertTrue(info?.message?.contains("budget") ?? false, info?.message ?? "no message")
    }

    /// PIN: the least recently used idle session goes first, and a session in use never does — nor one held while its
    /// turn's tools run (`holding(slot:)`), until its scope ends.
    func testEvictionTakesTheIdleNeverTheRunning() async throws {
        let gauge = Gauge()
        let engine = try governed(gauge, budget: .max, maxResident: 1)
        defer { engine.shutdown() }
        let running = SlotID()
        let busy = engine.insert(request(slot: running, maxTokens: 100_000))
        _ = await stats(engine) { $0.batchSize == 1 }
        try await finish(engine.insert(request(slot: SlotID(), maxTokens: 4)))
        let after = await stats(engine) { $0.evictions == 1 }
        XCTAssertEqual(after.evictions, 1)
        XCTAssertEqual(after.memory.slots.map(\.slot), [running], "the running session kept its cache")
        engine.remove(busy.id)
        try await finish(busy)

        // Held between its rounds, the conversation outlives an agent's session made beside it: the agent's goes instead.
        // Let go, it is the least recently used again.
        try await engine.holding(slot: running) {
            try await finish(engine.insert(request(slot: SlotID(), maxTokens: 4)))
            let beside = await stats(engine) { $0.evictions == 2 }
            XCTAssertEqual(beside.memory.slots.map(\.slot), [running], "a held session is not evicted")
            XCTAssertEqual(beside.memory.slots.first?.held, true)
            XCTAssertGreaterThan(beside.memory.heldBytes, 0, "what it holds is on a line of its own")
        }
        let next = SlotID()
        try await finish(engine.insert(request(slot: next, maxTokens: 4)))
        let letGo = await stats(engine) { $0.evictions == 3 }
        XCTAssertEqual(letGo.memory.slots.map(\.slot), [next], "let go, it is evicted as any idle session is")
    }

    /// A narrower batch admits nothing new until rows finish, and stops none that is decoding.
    func testNarrowingTheBatchNeverStopsARow() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let a = engine.insert(request(slot: SlotID(), maxTokens: 100_000))
        let b = engine.insert(request(slot: SlotID(), maxTokens: 100_000))
        _ = await stats(engine) { $0.batchSize == 2 }
        engine.setLimits(maxRows: 1)
        let c = engine.insert(request(slot: SlotID(), maxTokens: 4))
        let narrowed = await stats(engine) { $0.queued == 1 }
        XCTAssertEqual(narrowed.batchSize, 2, "both rows still decode")
        XCTAssertEqual(narrowed.queued, 1, "the new one waits")
        engine.remove(a.id)
        engine.remove(b.id)
        try await finish(a)
        try await finish(b)
        let info = try await finish(c)
        XCTAssertEqual(info?.reason, .maxTokens)
    }

    /// PIN: an MLX error on the engine's thread ends the rows it hit, failed, and the engine goes on. It
    /// used to end the process.
    func testAnErrorOnTheEngineThreadFailsTheRowsAndTheEngineGoesOn() async throws {
        let model = StubLanguageModel()
        let engine = try ParallelEngine(model: model)
        defer { engine.shutdown() }
        let handle = engine.insert(request(slot: SlotID(), maxTokens: 100_000))
        var events = handle.events.makeAsyncIterator()
        while let event = try await events.next() { if case .admitted = event { break } }
        model.fault.arm()
        var ended: FinishInfo?
        while let event = try await events.next() { if case .finished(let info) = event { ended = info } }
        XCTAssertEqual(ended?.reason, .failed)
        XCTAssertNotNil(ended?.message)
        let reported = await stats(engine) { $0.incident != nil }
        XCTAssertNotNil(reported.incident)
        let next = try await finish(engine.insert(request(slot: SlotID(), maxTokens: 4)))
        XCTAssertEqual(next?.reason, .maxTokens, "the engine still serves")
    }

    /// Two sessions decode as rows of one batch and each settles back into its own slot.
    func testRowsOfOneBatchSettleIntoTheirOwnSlots() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        async let one = finish(engine.insert(request(slot: SlotID(), tokens: 30, maxTokens: 12)))
        async let two = finish(engine.insert(request(slot: SlotID(), tokens: 70, maxTokens: 12)))
        let (a, b) = try await (one, two)
        XCTAssertEqual(a?.generated, 12)
        XCTAssertEqual(b?.generated, 12)
        let after = await stats(engine) { $0.residentSlots == 2 }
        XCTAssertEqual(after.residentSlots, 2)
    }

    /// PIN: a row's recurrent state stays its own through a batch. Three sessions of different lengths decode as rows of
    /// one batch on a stub whose recurrent layer keeps a window of each row's last four tokens, as Inkling-2's attention
    /// keeps its ring; settled and kept, each session's file holds its own tokens as keys and its own window. The real
    /// model's tests hold batching by greedy tokens only, which a near-tie hides.
    func testThreeRowsDecodingTogetherKeepTheirOwnWindows() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("frigate-rows-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        let engine = try ParallelEngine(model: StubLanguageModel(window: 4))
        defer { engine.shutdown() }
        let prompts = [Array(1 ... 30), Array((40 ... 110).reversed()), Array(200 ... 250)]
        let slots = prompts.map { _ in SlotID() }
        let handles = zip(prompts, slots).map { prompt, slot in
            engine.insert(StreamRequest(tokens: prompt, sampling: GenerateParameters(temperature: 0), maxTokens: 100_000,
                                        stopTokens: [], slot: slot))
        }
        let together = await stats(engine) { $0.batchSize == 3 && $0.streams.allSatisfy { $0.generated >= 8 } }
        XCTAssertEqual(together.batchSize, 3, "the three decode as rows of one batch")
        // One leaves while the others decode on, so the batch narrows under them.
        engine.remove(handles[1].id)
        try await finish(handles[1])
        let narrowed = together.streams.map(\.generated).max() ?? 0
        _ = await stats(engine) { $0.batchSize == 2 && $0.streams.allSatisfy { $0.generated >= narrowed + 8 } }
        for handle in [handles[0], handles[2]] {
            engine.remove(handle.id)
            try await finish(handle)
        }
        for (index, slot) in slots.enumerated() {
            let url = folder.appendingPathComponent("row-\(index).safetensors")
            let kept = try await engine.keep(slot: slot, to: url)
            let arrays = try loadArrays(url: url)
            let tokens = try XCTUnwrap(arrays["tokens"]).asArray(Int32.self).map { Float($0) }
            XCTAssertEqual(kept.tokens, tokens.count)
            XCTAssertGreaterThan(tokens.count, prompts[index].count + 8, "it decoded past its prompt")
            XCTAssertEqual(Array(tokens.prefix(prompts[index].count)), prompts[index].map { Float($0) }, "row \(index) holds its own prompt")
            let keys = try XCTUnwrap(arrays["layers.0.0"])
            XCTAssertEqual(keys[0, 0, 0..., 0].asArray(Float.self), tokens, "row \(index): its keys are its own tokens, a column each")
            let window = try XCTUnwrap(arrays["layers.1.2"])
            XCTAssertEqual(window.asArray(Float.self), Array(tokens.suffix(4)), "row \(index): its window is its own last four tokens")
        }
    }

    // MARK: - The ceiling

    /// An engine paused at `ceiling` by `gauge`'s reading; clearing the cache does what `onClear` says.
    private func ceilinged(_ gauge: Gauge, ceiling: Int = 1_000, clears: Counter = Counter(),
                           onClear: @escaping @Sendable () -> Void = {}) throws -> ParallelEngine {
        var configuration = ParallelEngine.Configuration()
        configuration.memory.ceilingBytes = ceiling
        configuration.memory.readCeilingBytes = { gauge.value.active }
        configuration.memory.readMemory = { gauge.value }
        configuration.memory.clearCache = { clears.add(); onClear() }
        return try ParallelEngine(model: StubLanguageModel(), configuration: configuration)
    }

    private func generated(_ stats: ParallelStats) -> Int { stats.streams.map(\.generated).max() ?? 0 }

    /// PIN: past the ceiling nothing steps — the one limit that holds, since MLX's own never refuses an
    /// allocation — and the engine goes on by itself only once memory is well back under the line.
    func testPastTheCeilingNoStepRunsUntilMemoryComesBack() async throws {
        let gauge = Gauge()
        let clears = Counter()
        let engine = try ceilinged(gauge, clears: clears)
        defer { engine.shutdown() }
        let row = engine.insert(request(slot: SlotID(), maxTokens: 100_000))
        _ = await stats(engine) { $0.batchSize == 1 }
        gauge.set(1_200)
        let held = await stats(engine) { $0.held != nil }
        XCTAssertEqual(held.held?.ceiling, 1_000)
        XCTAssertEqual(held.held?.asked, false)
        XCTAssertGreaterThan(clears.value, 0, "the buffer cache went back before the engine paused")
        let paused = generated(held)
        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertLessThanOrEqual(generated(engine.stats), paused + 1, "no step while paused")
        // A hair under the line is not back: the person is still being asked.
        gauge.set(990)
        try await Task.sleep(nanoseconds: 600_000_000)
        XCTAssertNotNil(engine.stats.held, "a process sitting at its line stays paused until it is answered")
        XCTAssertLessThanOrEqual(generated(engine.stats), paused + 1)
        gauge.set(100)
        let going = await stats(engine) { $0.held == nil && self.generated($0) > paused + 20 }
        XCTAssertNil(going.held)
        engine.remove(row.id)
        try await finish(row)
    }

    /// When giving the buffer cache back is enough, the engine never pauses.
    func testTheBufferCacheGoesBackBeforeTheEnginePauses() async throws {
        let gauge = Gauge()
        let engine = try ceilinged(gauge) { gauge.set(100) }
        defer { engine.shutdown() }
        let row = engine.insert(request(slot: SlotID(), maxTokens: 100_000))
        _ = await stats(engine) { $0.batchSize == 1 }
        gauge.set(1_200)
        let before = generated(engine.stats)
        let later = await stats(engine) { self.generated($0) > before + 50 }
        XCTAssertNil(later.held)
        engine.remove(row.id)
        try await finish(row)
    }

    /// Continue raises the line: the engine goes on, and pauses again only past the new one.
    func testResumeRaisesTheCeiling() async throws {
        let gauge = Gauge()
        let engine = try ceilinged(gauge)
        defer { engine.shutdown() }
        let row = engine.insert(request(slot: SlotID(), maxTokens: 100_000))
        _ = await stats(engine) { $0.batchSize == 1 }
        gauge.set(1_200)
        let held = await stats(engine) { $0.held != nil }
        engine.resume(ceiling: 2_000)
        let going = await stats(engine) { $0.held == nil && self.generated($0) > self.generated(held) + 20 }
        XCTAssertNil(going.held, "1 200 is under the raised line")
        gauge.set(2_500)
        let again = await stats(engine) { $0.held != nil }
        XCTAssertEqual(again.held?.ceiling, 2_000)
        engine.remove(row.id)
        try await finish(row)
    }

    /// PIN: Stop reaches a paused engine. A removed row used to end only at the next decode step, and a
    /// paused engine takes none.
    func testAStreamStoppedWhileThePauseHoldsEnds() async throws {
        let gauge = Gauge()
        let engine = try ceilinged(gauge)
        defer { engine.shutdown() }
        let row = engine.insert(request(slot: SlotID(), maxTokens: 100_000))
        _ = await stats(engine) { $0.batchSize == 1 }
        gauge.set(1_200)
        _ = await stats(engine) { $0.held != nil }
        engine.remove(row.id)
        let info = try await finish(row)
        XCTAssertEqual(info?.reason, .removed)
        let after = await stats(engine) { $0.batchSize == 0 }
        XCTAssertEqual(after.batchSize, 0)
    }

    /// A pause the host asked for (macOS's critical pressure) holds until it is let go, whatever memory says.
    func testAnAskedForPauseHoldsUntilResumed() async throws {
        let gauge = Gauge()
        let engine = try ceilinged(gauge)
        defer { engine.shutdown() }
        let row = engine.insert(request(slot: SlotID(), maxTokens: 100_000))
        _ = await stats(engine) { $0.batchSize == 1 }
        engine.hold()
        let held = await stats(engine) { $0.held != nil }
        XCTAssertEqual(held.held?.asked, true)
        try await Task.sleep(nanoseconds: 300_000_000)
        XCTAssertNotNil(engine.stats.held, "memory is fine, and it still waits for the host")
        XCTAssertLessThanOrEqual(generated(engine.stats), generated(held) + 1)
        engine.resume()
        let going = await stats(engine) { $0.held == nil && self.generated($0) > self.generated(held) + 20 }
        XCTAssertNil(going.held)
        engine.remove(row.id)
        try await finish(row)
    }

    /// A forward that is not a stream (a picture read by the model's own tower) runs on the engine's thread between two
    /// passes: no row decodes while it runs, what it returns or throws comes back, and the rows go on after it.
    func testAnExclusiveBodyRunsWithNoStepBesideIt() async throws {
        let engine = try ParallelEngine(model: StubLanguageModel())
        defer { engine.shutdown() }
        let row = engine.insert(request(slot: SlotID(), maxTokens: 100_000))
        let counted = Counter()
        let consumer = Task {
            for try await event in row.events { if case .token = event { counted.add() } }
        }
        while counted.value < 3 { try await Task.sleep(nanoseconds: 1_000_000) }

        let (before, after) = try await engine.exclusive { () -> (Int, Int) in
            let before = counted.value
            Thread.sleep(forTimeInterval: 0.2)
            return (before, counted.value)
        }
        XCTAssertEqual(before, after, "no token while the body ran")
        let resumed = counted.value
        while counted.value <= resumed + 2 { try await Task.sleep(nanoseconds: 1_000_000) }

        struct Refused: Error {}
        do {
            _ = try await engine.exclusive { () -> Int in throw Refused() }
            XCTFail("the body's error comes back")
        } catch is Refused {}

        engine.remove(row.id)
        _ = try await consumer.value
        engine.shutdown()
        do {
            _ = try await engine.exclusive { 1 }
            XCTFail("a shut-down engine runs nothing")
        } catch ParallelError.shutDown {}
    }
}

private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var value: Int { lock.withLock { count } }
    func add() { lock.withLock { count += 1 } }
}
