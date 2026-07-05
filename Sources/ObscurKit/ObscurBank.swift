//
//  ObscurBank.swift
//  ObscurKit
//
//  Data layer for Signets — Obscur's per-corpus, attention-bank adapters over FLUX.2
//  Klein. Each registered image contributes discrete KV rows derived from its DINOv2
//  patch tokens; the base model is never modified. Invariants (see the ObscurAdapter
//  spec): append = insert rows, delete = drop rows, attribution is measured, base
//  frozen. Entries live in DINOv2 space so they stay portable across projection
//  versions; K/V are materialized on demand.
//

import Foundation
import MLX

public typealias ObscurCorpusID = String
public typealias ObscurEntryID = String

public enum ObscurKitError: Error, LocalizedError {
    case missingProjection
    case emptyBank
    case corruptStore(String)

    public var errorDescription: String? {
        switch self {
        case .missingProjection: return "No Obscur projection is available"
        case .emptyBank: return "The Signet has no ingested entries"
        case .corruptStore(let what): return "Signet store is corrupt: \(what)"
        }
    }
}

/// Provenance for one entry — aligned with CLR's signing chain. The entry id itself is
/// content-addressed (SHA-256 of the source image bytes).
public struct ObscurProvenance: Codable, Equatable {
    public let sourceCaptureID: UUID?     // SignatureStore capture, when ingested from one
    public let signatureCID: String?      // CLR residual signature content id
    public let capturedAt: Date?

    public init(sourceCaptureID: UUID?, signatureCID: String?, capturedAt: Date?) {
        self.sourceCaptureID = sourceCaptureID
        self.signatureCID = signatureCID
        self.capturedAt = capturedAt
    }
}

public enum ObscurRefinementState: Codable, Equatable {
    case raw
    case refined(steps: Int, loss: Float)
}

/// One registered image. Immutable after refinement; identified by content hash.
/// Tensors are kept out of the Codable manifest and stored in the bank safetensors.
public struct ObscurEntry: Identifiable {
    public let id: ObscurEntryID
    public let corpusID: ObscurCorpusID
    /// DINOv2 patch tokens, (T_dino, 384) — frozen at ingest, projection input.
    public var dinoTokens: MLXArray
    /// Mean-pooled DINOv2 vector (384,) — the kNN key for `topK` selection.
    public var dinoMean: MLXArray
    /// Optional per-image learned tokens (T1 refinement output), DINOv2 space.
    public var learnedTokens: MLXArray?
    public let provenance: ObscurProvenance
    public var refinementState: ObscurRefinementState
    public let ingestedAt: Date

    public init(id: ObscurEntryID, corpusID: ObscurCorpusID,
                dinoTokens: MLXArray, provenance: ObscurProvenance,
                ingestedAt: Date = Date()) {
        self.id = id
        self.corpusID = corpusID
        self.dinoTokens = dinoTokens
        self.dinoMean = dinoTokens.mean(axis: 0)
        self.learnedTokens = nil
        self.provenance = provenance
        self.refinementState = .raw
        self.ingestedAt = ingestedAt
    }

    /// Projection input: learned tokens once refined, raw DINO tokens otherwise.
    public var projectionInput: MLXArray { learnedTokens ?? dinoTokens }
}

/// A corpus = one Signet: one profile's body of signed work. Self-contained artifact.
public final class ObscurBank {
    public let corpusID: ObscurCorpusID
    public private(set) var entries: [ObscurEntryID: ObscurEntry]
    /// Insertion-ordered ids so composition and attribution are deterministic.
    public private(set) var order: [ObscurEntryID]
    /// Per-corpus attention logit bias (composition mixing weight, log-space).
    public var logitBias: Float
    /// Semantic motifs (k-means over entry dinoMeans) — set at forge time.
    public var motifs: [ObscurMotif] = []
    /// In-memory K/V memo per entry (invalidated on entry change / projection change).
    var kvMemo: [ObscurEntryID: [ObscurLayerRef: (k: MLXArray, v: MLXArray)]] = [:]

    public init(corpusID: ObscurCorpusID, logitBias: Float = 0) {
        self.corpusID = corpusID
        self.entries = [:]
        self.order = []
        self.logitBias = logitBias
    }

    /// Append = insert rows. Never requires touching shared weights.
    public func insert(_ entry: ObscurEntry) {
        if entries[entry.id] == nil { order.append(entry.id) }
        entries[entry.id] = entry
        kvMemo[entry.id] = nil
    }

    /// Delete = drop rows. Removes the entry's influence entirely.
    public func remove(_ id: ObscurEntryID) {
        entries[id] = nil
        order.removeAll { $0 == id }
        kvMemo[id] = nil
    }

    public var isEmpty: Bool { entries.isEmpty }
    public var count: Int { entries.count }

    public func invalidateKV() { kvMemo.removeAll() }
}

// MARK: - Persistence

/// One directory per corpus:
///   manifest.json      — entries (ids, provenance, refinement state), logitBias, versions
///   bank.safetensors   — dinoTokens / dinoMean / learnedTokens per entry
///   attribution/       — per-generation reports
public enum ObscurStore {
    struct Manifest: Codable {
        struct EntryMeta: Codable {
            let id: ObscurEntryID
            let provenance: ObscurProvenance
            let refinementState: ObscurRefinementState
            let ingestedAt: Date
        }
        var schema = "obscur.bank.v1"
        var corpusID: ObscurCorpusID
        var logitBias: Float
        var entries: [EntryMeta]
        // Absent in pre-motif banks — they recluster on next forge.
        var motifs: [ObscurMotif]? = nil
    }

    public static func save(_ bank: ObscurBank, to dir: URL) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var manifest = Manifest(corpusID: bank.corpusID, logitBias: bank.logitBias,
                                entries: [], motifs: bank.motifs.isEmpty ? nil : bank.motifs)
        var tensors: [String: MLXArray] = [:]
        for id in bank.order {
            guard let entry = bank.entries[id] else { continue }
            manifest.entries.append(.init(id: id, provenance: entry.provenance,
                                          refinementState: entry.refinementState,
                                          ingestedAt: entry.ingestedAt))
            tensors["\(id).dino"] = entry.dinoTokens
            tensors["\(id).mean"] = entry.dinoMean
            if let learned = entry.learnedTokens {
                tensors["\(id).learned"] = learned
            }
        }
        try MLX.save(arrays: tensors, url: dir.appendingPathComponent("bank.safetensors"))
        let data = try JSONEncoder().encode(manifest)
        try data.write(to: dir.appendingPathComponent("manifest.json"))
    }

    public static func load(from dir: URL) throws -> ObscurBank {
        let manifestURL = dir.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL) else {
            throw ObscurKitError.corruptStore("missing manifest at \(dir.lastPathComponent)")
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        let tensors = try loadArrays(url: dir.appendingPathComponent("bank.safetensors"))
        let bank = ObscurBank(corpusID: manifest.corpusID, logitBias: manifest.logitBias)
        bank.motifs = manifest.motifs ?? []
        for meta in manifest.entries {
            guard let dino = tensors["\(meta.id).dino"] else {
                throw ObscurKitError.corruptStore("missing tokens for entry \(meta.id)")
            }
            var entry = ObscurEntry(id: meta.id, corpusID: manifest.corpusID,
                                    dinoTokens: dino, provenance: meta.provenance,
                                    ingestedAt: meta.ingestedAt)
            entry.learnedTokens = tensors["\(meta.id).learned"]
            entry.refinementState = meta.refinementState
            bank.insert(entry)
        }
        return bank
    }
}
