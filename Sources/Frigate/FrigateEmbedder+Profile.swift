import Foundation

extension FrigateEmbedder {

    /// What a text is for. Asymmetric embedders (voyage-4-nano) prompt a search query
    /// differently from a passage being indexed; symmetric ones ignore it.
    public enum Role: String, Sendable, Codable {
        case query
        case document
    }

    /// One embedding model and how to call it: where it comes from (pinned), the
    /// prompt per role, the output width, and the name of the vector space it writes —
    /// what hosts stamp beside an index so vectors from two models never mix.
    public struct Profile: Sendable, Equatable {
        public enum Source: Sendable, Equatable {
            /// A Hugging Face repo, pinned to a commit when `revision` is a 40-hex sha.
            case hub(id: String, revision: String?)
            /// A snapshot already on disk (development, tests, air-gapped hosts).
            case directory(URL)
        }

        public var source: Source
        /// Prepended to a query. `nil`: take the snapshot's
        /// `config_sentence_transformers.json` prompt, if it names one.
        public var queryPrompt: String?
        /// Prepended to a document. `nil`: as above.
        public var documentPrompt: String?
        /// Matryoshka width: the first `outputDimension` dimensions, renormalised.
        /// `nil` keeps the model's own width.
        public var outputDimension: Int?
        /// Token budget per text, prompt and special tokens included.
        public var maxTokens: Int
        /// The vector space this profile writes, e.g. `voyage-4@1024`.
        public var vectorSpace: String

        public init(
            source: Source,
            queryPrompt: String? = nil,
            documentPrompt: String? = nil,
            outputDimension: Int? = nil,
            maxTokens: Int = 512,
            vectorSpace: String? = nil
        ) {
            self.source = source
            self.queryPrompt = queryPrompt
            self.documentPrompt = documentPrompt
            self.outputDimension = outputDimension
            self.maxTokens = maxTokens
            self.vectorSpace = vectorSpace ?? Self.defaultVectorSpace(source: source, dimension: outputDimension)
        }

        /// The repo id, or the directory's path.
        public var model: String {
            switch source {
            case .hub(let id, _): return id
            case .directory(let url): return url.path
            }
        }

        public var revision: String? {
            if case .hub(_, let revision) = source { return revision }
            return nil
        }

        /// The profile's own prompt for `role`, before any snapshot fallback.
        public func prompt(for role: Role) -> String? {
            role == .query ? queryPrompt : documentPrompt
        }

        private static func defaultVectorSpace(source: Source, dimension: Int?) -> String {
            let name: String
            switch source {
            case .hub(let id, _): name = id
            case .directory(let url): name = url.lastPathComponent
            }
            return dimension.map { "\(name)@\($0)" } ?? name
        }
    }
}

// MARK: - Known models

extension FrigateEmbedder.Profile {

    /// The Hugging Face repo Rao publishes voyage-4-nano from: an MLX conversion of
    /// `voyageai/voyage-4-nano` @ 67fabc9b (Apache-2.0; NOTICE carried in the repo),
    /// 8-bit — 370 MB, and within 0.001 nDCG@10 of the bfloat16 original on every
    /// benchmark in scripts/embedders (mean cosine 0.9996).
    public static let voyage4NanoRepo = "rao-studios/voyage-4-nano-mlx-8bit"
    /// The published commit. Bumping it is how a new conversion ships; the vector
    /// space stays `voyage-4@1024` as long as the weights are the same model.
    public static let voyage4NanoRevision: String? = "5b5f4aff4bd03236bb957dc3b9400fa0eedc8f4c"

    /// voyage-4-nano, cut to 1024 dimensions (Matryoshka). Its space is shared with
    /// Voyage's hosted voyage-4 models.
    public static let voyage4Nano = FrigateEmbedder.Profile(
        source: .hub(id: voyage4NanoRepo, revision: voyage4NanoRevision),
        queryPrompt: "Represent the query for retrieving supporting documents: ",
        documentPrompt: "Represent the document for retrieval: ",
        outputDimension: 1024,
        maxTokens: 512,
        vectorSpace: "voyage-4@1024")

    /// Profiles looked up by repo id. Anything else resolves to a bare profile that
    /// reads its prompts from the snapshot and keeps the model's own width.
    public static let known: [String: FrigateEmbedder.Profile] = [
        voyage4NanoRepo: voyage4Nano
    ]

    /// `org/repo`, `org/repo@<revision>`, or a path (`/…`, `~/…`, `./…`).
    /// A known repo keeps its prompts, width and space; `@<revision>` overrides the pin.
    public static func resolve(_ spec: String) -> FrigateEmbedder.Profile {
        let spec = spec.trimmingCharacters(in: .whitespacesAndNewlines)
        if spec.hasPrefix("/") || spec.hasPrefix("~") || spec.hasPrefix(".") {
            let path = (spec as NSString).expandingTildeInPath
            return FrigateEmbedder.Profile(source: .directory(URL(fileURLWithPath: path)))
        }
        let parts = spec.split(separator: "@", maxSplits: 1).map(String.init)
        let id = parts[0]
        let pinned = parts.count > 1 && !parts[1].isEmpty ? parts[1] : nil
        guard var profile = known[id] else {
            return FrigateEmbedder.Profile(source: .hub(id: id, revision: pinned))
        }
        if let pinned { profile.source = .hub(id: id, revision: pinned) }
        return profile
    }
}

// MARK: - Status

extension FrigateEmbedder {

    /// Where the model is in its life: hosts surface this (Thread's `/health`).
    public struct Status: Sendable, Codable, Equatable {
        public enum Phase: String, Sendable, Codable {
            case idle, downloading, loading, ready, failed
        }

        public var model: String
        public var revision: String?
        public var vectorSpace: String
        public var phase: Phase
        /// 0…1 while downloading.
        public var fraction: Double?
        public var error: String?

        public init(
            model: String, revision: String?, vectorSpace: String, phase: Phase,
            fraction: Double? = nil, error: String? = nil
        ) {
            self.model = model
            self.revision = revision
            self.vectorSpace = vectorSpace
            self.phase = phase
            self.fraction = fraction
            self.error = error
        }
    }

    /// A lock-guarded status any thread can read without hopping onto the actor.
    final class StatusBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Status

        init(_ value: Status) { self.value = value }

        func get() -> Status { lock.withLock { value } }
        func set(_ update: (inout Status) -> Void) { lock.withLock { update(&value) } }
    }
}

// MARK: - Token budget

extension FrigateEmbedder {

    /// The special tokens a tokenizer wraps around every text, learned once by
    /// encoding a probe with and without them. Truncating the wrapped sequence would
    /// cut the trailing token last-token-pooling models read (Qwen3-Embedding's
    /// `<|endoftext|>`), so the body is cut and the frame put back around it.
    public struct TokenFrame: Sendable, Equatable {
        public var prefix: [Int]
        public var suffix: [Int]

        public init(prefix: [Int] = [], suffix: [Int] = []) {
            self.prefix = prefix
            self.suffix = suffix
        }

        /// The frame `with` adds around `without`, or `nil` when `without` is not a
        /// contiguous run inside it (a tokenizer that rewrites the body).
        public static func probe(with: [Int], without: [Int]) -> TokenFrame? {
            guard !without.isEmpty, without.count <= with.count else {
                return without.isEmpty ? TokenFrame(prefix: with) : nil
            }
            for start in 0...(with.count - without.count)
            where Array(with[start..<(start + without.count)]) == without {
                return TokenFrame(
                    prefix: Array(with[..<start]),
                    suffix: Array(with[(start + without.count)...]))
            }
            return nil
        }

        /// `body` cut to fit `maxTokens` once the frame is around it.
        public func fit(_ body: [Int], maxTokens: Int) -> [Int] {
            let room = max(0, maxTokens - prefix.count - suffix.count)
            return prefix + body.prefix(room) + suffix
        }
    }
}
