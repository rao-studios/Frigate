/// FrigateBoost — pure-Swift XGBoost tree-ensemble inference.
///
/// Loads a model exported with `booster.save_model("model.json")` (XGBoost ≥ 2.0
/// JSON format) and evaluates it with zero runtime dependencies — no libxgboost needed.
///
/// Usage:
/// ```swift
/// let boost = try FrigateBoost(modelURL: URL(fileURLWithPath: "clr_v1.8.xgb.json"))
/// let probs = try await boost.predict(features: [[0.1, 0.001, 0.5, ...]])
/// ```

import Foundation

// MARK: - Internal model

/// Raw storage for one tree loaded from XGBoost JSON.
private struct XGTree: Sendable {
    let leftChildren:  [Int32]
    let rightChildren: [Int32]
    let splitIndices:  [Int32]
    let splitConds:    [Float]
    let baseWeights:   [Float]

    /// Walk the tree for a single feature vector and return the leaf value.
    func leafValue(for features: [Float]) -> Float {
        var node: Int = 0
        while leftChildren[node] != -1 {
            let fi = Int(splitIndices[node])
            let fv = fi < features.count ? features[fi] : 0
            node = fv < splitConds[node]
                ? Int(leftChildren[node])
                : Int(rightChildren[node])
        }
        return baseWeights[node]
    }
}

// MARK: - JSON Codable

private struct XGBModel: Decodable {
    let learner: Learner

    struct Learner: Decodable {
        let learnerModelParam: LearnerModelParam
        let gradientBooster: GradientBooster
        let objective: Objective

        enum CodingKeys: String, CodingKey {
            case learnerModelParam = "learner_model_param"
            case gradientBooster  = "gradient_booster"
            case objective
        }
    }

    struct LearnerModelParam: Decodable {
        let baseScore: String   // stored as e.g. "[5E-1]" or "0.5" in XGBoost 2.x

        enum CodingKeys: String, CodingKey { case baseScore = "base_score" }

        var baseScoreFloat: Float {
            // XGBoost 2.x wraps the value in brackets: "[5E-1]"
            var s = baseScore.trimmingCharacters(in: .whitespaces)
            if s.hasPrefix("[") && s.hasSuffix("]") {
                s = String(s.dropFirst().dropLast())
            }
            return Float(s) ?? 0.5
        }
    }

    struct GradientBooster: Decodable {
        let model: BoosterModel

        struct BoosterModel: Decodable {
            let trees: [TreeData]
        }
    }

    struct Objective: Decodable {
        let name: String
    }

    struct TreeData: Decodable {
        let leftChildren:  [Int32]
        let rightChildren: [Int32]
        let splitIndices:  [Int32]
        let splitConditions: [Float]
        let baseWeights:   [Float]

        enum CodingKeys: String, CodingKey {
            case leftChildren   = "left_children"
            case rightChildren  = "right_children"
            case splitIndices   = "split_indices"
            case splitConditions = "split_conditions"
            case baseWeights    = "base_weights"
        }
    }
}

// MARK: - XGBoostTreeModel

/// In-memory XGBoost model that can evaluate arbitrary feature vectors.
public struct XGBoostTreeModel: Sendable {
    private let trees: [XGTree]
    private let baseScoreLogit: Float  // logit(base_score); 0 for base_score=0.5

    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        let decoded = try JSONDecoder().decode(XGBModel.self, from: data)

        let bs = decoded.learner.learnerModelParam.baseScoreFloat
        // base_score is a probability; convert to log-odds (logit) to add to margin.
        // For bs=0.5, logit=0 — no adjustment needed.
        let clamped = max(1e-7, min(1 - 1e-7, Double(bs)))
        baseScoreLogit = Float(log(clamped / (1.0 - clamped)))

        trees = decoded.learner.gradientBooster.model.trees.map { t in
            XGTree(
                leftChildren:  t.leftChildren,
                rightChildren: t.rightChildren,
                splitIndices:  t.splitIndices,
                splitConds:    t.splitConditions,
                baseWeights:   t.baseWeights
            )
        }
    }

    /// Predict P(class=1) for a batch of feature vectors.
    ///
    /// - Parameter features: `[[Float]]` — each inner array is one sample with
    ///   `n_features` elements in the same order as training.
    /// - Returns: `[Float]` of probabilities, one per input row.
    public func predictBatch(_ features: [[Float]]) -> [Float] {
        features.map { fv in
            var margin = baseScoreLogit
            for tree in trees { margin += tree.leafValue(for: fv) }
            return 1.0 / (1.0 + Foundation.exp(-margin))
        }
    }

    /// Predict P(class=1) for a single feature vector.
    public func predict(_ features: [Float]) -> Float {
        var margin = baseScoreLogit
        for tree in trees { margin += tree.leafValue(for: features) }
        return 1.0 / (1.0 + Foundation.exp(-margin))
    }
}

// MARK: - FrigateBoost actor

/// Thread-safe async wrapper for XGBoost inference.
///
/// Follows the actor pattern of `FrigateLLM` / `FrigateEmbedder`.
public actor FrigateBoost {
    private let model: XGBoostTreeModel

    /// Load an XGBoost model from a JSON file.
    ///
    /// - Parameter modelURL: Path to a `.json` file exported with
    ///   `booster.save_model("model.json")`.
    public init(modelURL: URL) throws {
        model = try XGBoostTreeModel(url: modelURL)
    }

    /// Predict P(generated) for a batch of 10-element feature vectors.
    ///
    /// - Parameter features: Array of feature vectors.  Each inner `[Float]` must
    ///   have exactly 10 elements in the order:
    ///   `[fragility_rate, local_var, circ_var, composite, cross_corr, cross_ratio,
    ///     seam_h, seam_v, seam_h_spec, seam_v_spec]`
    /// - Returns: `[Float]` of P(generated) values, one per row.
    public func predict(features: [[Float]]) -> [Float] {
        model.predictBatch(features)
    }
}
