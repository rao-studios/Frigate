import FrigateParallel
import XCTest

/// How the parallel engine shares its one thread between the prompt it reads and the rows it decodes, as rows: the
/// policy is a function of whose work each is and of what the engine's last steps and chunks took.
final class InterleavePolicyTests: XCTestCase {
    typealias Clock = InterleavePolicy.Clock

    /// A step of 50 ms, a chunk of 200 ms, a prompt read at 400 tokens a second.
    private func clock(steps: Int, decoded: TimeInterval, rate: Double? = 400) -> Clock {
        Clock(stepsSinceChunk: steps, decodeSecondsSinceChunk: decoded, chunkSeconds: 0.2, stepSeconds: 0.05,
              prefillTokensPerSecond: rate)
    }

    func testThePromptTakesItsShareOfTheEnginesTime() {
        let policy = InterleavePolicy(even: 0.5, yielding: 0.2, minChunk: 32, maxChunk: 256)
        let rows: [(label: String, prompt: StreamPriority, rows: StreamPriority?, clock: Clock, next: InterleavePolicy.Next)] = [
            ("no row decodes: whole chunks", .background, nil, clock(steps: 0, decoded: 0), .chunk(256)),
            ("every chunk is followed by a step", .background, .background, clock(steps: 0, decoded: 0), .step),
            ("the prompt outranks the rows: one step, then a whole chunk", .interactive, .background,
             clock(steps: 1, decoded: 0.05), .chunk(256)),
            ("even: steps until the rows have had the chunk's time", .background, .background,
             clock(steps: 3, decoded: 0.15), .step),
            ("even: then a chunk of four steps' time", .background, .background, clock(steps: 4, decoded: 0.21), .chunk(80)),
            ("yielding: steps until the rows have had four times the chunk's time", .background, .interactive,
             clock(steps: 15, decoded: 0.75), .step),
            ("yielding: then a chunk of one step's time, no smaller than the smallest", .background, .interactive,
             clock(steps: 17, decoded: 0.85), .chunk(32)),
            ("a prompt read fast is bounded by the largest chunk", .background, .background,
             clock(steps: 4, decoded: 0.21, rate: 10_000), .chunk(256)),
            ("nothing read yet: the smallest chunk", .background, .background,
             Clock(stepsSinceChunk: 1, decodeSecondsSinceChunk: 0.05), .chunk(32)),
        ]
        for row in rows {
            XCTAssertEqual(policy.next(prompt: row.prompt, rows: row.rows, clock: row.clock), row.next, row.label)
        }
    }

    /// PIN: whose work it is sets the share. A prompt someone waits on more than on the rows keeps the engine as it
    /// always had it; one beside rows of its own priority takes `even`; one no one waits on beside rows someone does takes
    /// `yielding`.
    func testWhoseWorkSetsTheShare() {
        let policy = InterleavePolicy(even: 0.5, yielding: 0.2)
        XCTAssertNil(policy.share(prompt: .background, rows: nil), "no row decodes: nothing to share")
        XCTAssertEqual(policy.share(prompt: .interactive, rows: .background), 1)
        XCTAssertEqual(policy.share(prompt: .background, rows: .background), 0.5)
        XCTAssertEqual(policy.share(prompt: .interactive, rows: .interactive), 0.5)
        XCTAssertEqual(policy.share(prompt: .background, rows: .interactive), 0.2)
        // Past 1 a share is the old alternation; under a twentieth a prompt would never be read.
        let bounded = InterleavePolicy(even: 3, yielding: 0)
        XCTAssertEqual(bounded.even, 1)
        XCTAssertEqual(bounded.yielding, 0.05)
        XCTAssertEqual(InterleavePolicy(minChunk: 64, maxChunk: 16).minChunk, 16, "the smallest chunk is never past the largest")
    }

    /// What the engine's time went to, as the clock keeps it: steps and their seconds since the last chunk, a step's
    /// average without the first after a chunk, and a reading rate each chunk moves.
    func testTheClockKeepsWhatTheEnginesTimeWentTo() {
        var clock = Clock()
        clock.read(tokens: 256, seconds: 1)
        XCTAssertEqual(clock.prefillTokensPerSecond, 256)
        XCTAssertEqual(clock.chunkSeconds, 1)
        clock.stepped(seconds: 0.002)
        XCTAssertEqual(clock.stepSeconds, 0, "the first step after a chunk waited on nothing: it stays out of the average")
        clock.stepped(seconds: 0.05)
        clock.stepped(seconds: 0.05)
        XCTAssertEqual(clock.stepSeconds, 0.05, accuracy: 1e-9)
        XCTAssertEqual(clock.stepsSinceChunk, 3)
        XCTAssertEqual(clock.decodeSecondsSinceChunk, 0.102, accuracy: 1e-9)
        clock.read(tokens: 32, seconds: 0.25)
        XCTAssertEqual(clock.stepsSinceChunk, 0)
        XCTAssertEqual(clock.decodeSecondsSinceChunk, 0)
        XCTAssertEqual(clock.prefillTokensPerSecond ?? 0, 256 + 0.3 * (128 - 256), accuracy: 1e-9, "an average each chunk moves")
    }
}
