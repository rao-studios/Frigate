//
//  ObscurMotif.swift
//  ObscurKit
//
//  Conceptual anchoring v1: a Signet's corpus clustered into semantic motifs — k-means
//  (MLXAccelerate, k-means++ seeding) over the DINOv2 mean vectors each entry already
//  carries. Motifs structure attribution (region → motif in the Influence Cloud) and
//  let generation draw from a chosen concept rather than the whole corpus. Labeled by
//  numeral + medoid image; VLM naming can layer on later without schema change.
//

import Foundation
import MLX
import MLXAccelerate

public struct ObscurMotif: Codable, Equatable, Identifiable {
    public let index: Int
    public let entryIDs: [ObscurEntryID]
    /// The entry nearest the cluster centroid — the motif's visual identity.
    public let medoidEntryID: ObscurEntryID

    public var id: Int { index }

    /// "I", "II", "III"… — the v1 label.
    public var numeral: String {
        let numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
        return index < numerals.count ? numerals[index] : "\(index + 1)"
    }

    public init(index: Int, entryIDs: [ObscurEntryID], medoidEntryID: ObscurEntryID) {
        self.index = index
        self.entryIDs = entryIDs
        self.medoidEntryID = medoidEntryID
    }
}

public enum ObscurMotifClusterer {
    /// Cluster the bank's entries into motifs. Corpora with fewer than 6 entries get a
    /// single motif (clustering tiny sets is noise); otherwise k = min(5, √n). Entries
    /// keep the bank's insertion order within each motif for deterministic covers.
    public static func cluster(bank: ObscurBank) -> [ObscurMotif] {
        let ids = bank.order
        let n = ids.count
        guard n > 0 else { return [] }

        let vectors = ids.compactMap { bank.entries[$0]?.dinoMean.asType(.float32) }
        guard vectors.count == n else { return [] }

        let k = n < 6 ? 1 : min(5, Int(Double(n).squareRoot()))
        guard k > 1 else {
            return [ObscurMotif(index: 0, entryIDs: ids,
                                medoidEntryID: medoid(ids: ids, vectors: vectors))]
        }

        // (1, n, 384) batch for MLXAccelerate's Lloyd's iteration.
        let data = stacked(vectors, axis: 0).expandedDimensions(axis: 0)
        let result = kmeans(data, k: k)
        eval(result.codes, result.squaredDistances)
        let codes: [Int32] = result.codes[0].asArray(Int32.self)
        let distances: [Float] = result.squaredDistances[0].asArray(Float.self)

        var motifs: [ObscurMotif] = []
        for cluster in 0..<k {
            var members: [ObscurEntryID] = []
            var best: (id: ObscurEntryID, distance: Float)? = nil
            for (i, code) in codes.enumerated() where Int(code) == cluster {
                members.append(ids[i])
                if best == nil || distances[i] < best!.distance {
                    best = (ids[i], distances[i])
                }
            }
            guard !members.isEmpty, let medoid = best?.id else { continue }
            motifs.append(ObscurMotif(index: motifs.count, entryIDs: members,
                                      medoidEntryID: medoid))
        }
        // Degenerate clustering (empty clusters collapsing to one) → single motif.
        if motifs.count <= 1 {
            return [ObscurMotif(index: 0, entryIDs: ids,
                                medoidEntryID: medoid(ids: ids, vectors: vectors))]
        }
        return motifs
    }

    /// Entry nearest the mean of the given vectors.
    static func medoid(ids: [ObscurEntryID], vectors: [MLXArray]) -> ObscurEntryID {
        guard ids.count > 1 else { return ids[0] }
        let stack = stacked(vectors, axis: 0)                 // (n, d)
        let mean = stack.mean(axis: 0, keepDims: true)        // (1, d)
        let distances = (stack - mean).square().sum(axis: -1) // (n,)
        eval(distances)
        let values: [Float] = distances.asArray(Float.self)
        var bestIndex = 0
        for (i, v) in values.enumerated() where v < values[bestIndex] {
            bestIndex = i
        }
        return ids[bestIndex]
    }
}
