//  WHAT: The Mamba-3 port (Models/Mamba3.swift, SSD3.swift, Mamba3Loader.swift) against Braid's oracle: the tiny
//        fixture's normed output, logits and final states (Fixtures/mamba3-tiny, from Braid's
//        `Scripts/mamba3-oracle.py fixture`) under both activations; one token at a time against the sequence; a read
//        in pieces against one read; the BF16 loader. With FRIGATE_MAMBA3_COMMONS naming the 1.5b's folder (the HF
//        layout) and FRIGATE_MAMBA3_ORACLE Braid's `oracle/reference.safetensors`, the live conformance: logits within
//        1e-3 of the oracle at float32, the prompts tokenised to the oracle's ids.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import FrigateBridge
@testable import MLXLLM

private var mlxFunctional: Bool {
    #if os(Linux)
    return true
    #else
    return ProcessInfo.processInfo.environment["FRIGATE_MLX_TESTS"] == "1"
    #endif
}

private let fixture = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/mamba3-tiny")

private func tinyConfig(activation: String? = "heavy_tail", chunk: Int = 8) throws -> Mamba3Configuration {
    var config = try JSONDecoder().decode(Mamba3Configuration.self, from: Data(contentsOf: fixture.appendingPathComponent("config.json")))
    config.aActivation = activation
    config.readChunk = chunk
    return config
}

/// ‖a − b‖ / ‖b‖, in float32.
private func relative(_ a: MLXArray, _ b: MLXArray) -> Float {
    let a = a.asType(.float32), b = b.asType(.float32)
    return (sqrt(sum((a - b) * (a - b))) / maximum(sqrt(sum(b * b)), MLXArray(Float(1e-12)))).item(Float.self)
}

/// Angles compared on the circle: the smaller way round, at most π.
private func angleGap(_ a: MLXArray, _ b: MLXArray) -> Float {
    let d = abs(SSD3.wrap(a - b))
    return minimum(d, Float(2 * Double.pi) - d).max().item(Float.self)
}

@Suite("Mamba-3 configuration")
struct Mamba3ConfigurationTests {
    @Test func theCommonsShapeIsBraids() throws {
        let json = """
            {"model_type":"mamba3","d_model":2048,"d_intermediate":3824,"n_layer":24,"vocab_size":128256,
             "ssm_cfg":{"layer":"Mamba3","d_state":128,"expand":2,"headdim":64,"ngroups":1,"rope_fraction":0.5,
                        "A_floor":0.0001,"is_mimo":true,"mimo_rank":4,"is_outproj_norm":false},
             "attn_layer_idx":[],"rms_norm":true,"tie_embeddings":true,"pad_vocab_size_multiple":16,
             "braid_a_activation":"heavy_tail","braid_chunk_size":64,"braid_cut":12,"braid_tap_after_blocks":12}
            """
        let config = try JSONDecoder().decode(Mamba3Configuration.self, from: Data(json.utf8))
        try config.validateModelConfiguration()
        #expect(config.projection == 9_440)
        #expect(config.mlpHidden == 3_840)
        #expect(config.numHeads == 64)
        #expect(config.rotaryAngles == 32)
        #expect(config.stateBytes == 54_067_200)   // Braid's pack.json `stateBytesPerStream`
        #expect(config.tapAfterBlocks == 12)
    }

    @Test func aConfigWithoutTheActivationIsRefused() throws {
        #expect(throws: Mamba3Configuration.Refusal.self) { try tinyConfig(activation: nil).validateModelConfiguration() }
        try tinyConfig(activation: "softplus").validateModelConfiguration()
    }

    @Test(arguments: [
        ("backbone.embeddings.weight", true), ("backbone.layers.3.mixer.in_proj.weight", true),
        ("backbone.layers.3.mixer.out_proj.weight", true), ("backbone.layers.3.mlp.fc1.weight", true),
        ("backbone.layers.3.mlp.fc2.weight", true), ("backbone.layers.3.mixer.B_norm.weight", false),
        ("backbone.layers.3.norm.weight", false), ("backbone.layers.3.mixer.D", false),
        ("backbone.layers.3.mixer.mimo_x", false), ("backbone.norm_f.weight", false),
    ])
    func onlyMatricesNarrow(key: String, narrows: Bool) {
        #expect(Mamba3Loader.narrows(key) == narrows)
    }
}

@Suite("Mamba-3 against Braid's oracle", .enabled(if: mlxFunctional), .serialized)
struct Mamba3OracleTests {
    private let reference: [String: MLXArray]
    private let tokens: MLXArray

    init() throws {
        reference = try loadArrays(url: fixture.appendingPathComponent("reference.safetensors"))
        tokens = reference["tokens"]!.asType(.int32).reshaped([1, -1])
    }

    private func model(_ activation: String = "heavy_tail", chunk: Int = 8, dtype: DType? = nil) throws -> Mamba3Model {
        try Mamba3Loader.load(config: tinyConfig(activation: activation, chunk: chunk), directory: fixture, weightDType: dtype)
    }

    @Test(arguments: ["heavy_tail", "softplus"])
    func theSequenceIsTheOracles(activation: String) throws {
        let model = try model(activation)
        let cache = model.newCache(parameters: nil)
        let logits = model(tokens, cache: cache)
        #expect(relative(logits[0], reference["\(activation).logits"]!) < 1e-4)
        for (i, layer) in cache.enumerated() {
            let state = layer as! ArraysCache
            #expect(relative(state[0]![0], reference["\(activation).layer.\(i).ssm"]!) < 1e-4)
            #expect(angleGap(state[1]![0], reference["\(activation).layer.\(i).angle"]!) < 1e-4)
            #expect(relative(state[2]![0], reference["\(activation).layer.\(i).k"]!) < 1e-4)
            #expect(relative(state[3]![0], reference["\(activation).layer.\(i).v"]!) < 1e-4)
        }
    }

    @Test func oneTokenAtATimeIsTheSequence() throws {
        let model = try model()
        let ids = tokens.asArray(Int32.self)
        let whole = model(tokens, cache: model.newCache(parameters: nil))
        let cache = model.newCache(parameters: nil)
        var worst: Float = 0
        for (t, id) in ids.enumerated() {
            let logits = model(MLXArray([id], [1, 1]), cache: cache)
            worst = max(worst, relative(logits[0, 0], whole[0, t]))
        }
        #expect(worst < 1e-4)
    }

    @Test func aReadInPiecesIsOneRead() throws {
        let model = try model()
        let ids = tokens.asArray(Int32.self)
        let one = model.newCache(parameters: nil)
        let whole = model.read(tokens, cache: one, tap: 1)
        let pieces = model.newCache(parameters: nil)
        let a = model.read(MLXArray(Array(ids[..<9]), [1, 9]), cache: pieces, tap: 1)
        let single = model.read(MLXArray(Array(ids[9 ..< 10]), [1, 1]), cache: pieces, tap: 1)   // a length-1 piece (SSD3's PIN)
        let b = model.read(MLXArray(Array(ids[10...]), [1, ids.count - 10]), cache: pieces, tap: 1)
        #expect(a.count + single.count + b.count == whole.count)
        #expect(relative(a.tappedSum + single.tappedSum + b.tappedSum, whole.tappedSum) < 1e-4)
        #expect(relative(b.last, whole.last) < 1e-4)
        #expect(relative(b.last, reference["heavy_tail.final"]![(ids.count - 1)...]) < 1e-4)
        for (x, y) in zip(one, pieces) {
            #expect(relative((x as! ArraysCache)[0]!, (y as! ArraysCache)[0]!) < 1e-4)
        }
    }

    @Test func theTapIsTheResidualAfterItsBlocks() throws {
        let model = try model()
        let reading = model.read(tokens, cache: model.newCache(parameters: nil), tap: 1)
        let h = model.backbone.layers[0](model.backbone.embed(tokens), cache: ArraysCache(size: 4))
        #expect(relative(reading.tappedSum, h.sum(axis: 1)) < 1e-5)
    }

    @Test func theBF16LoaderNarrowsOnlyTheMatrices() throws {
        let narrow = try model(dtype: .bfloat16)
        let wide = try model()
        for (key, value) in narrow.parameters().flattened() {
            #expect(value.dtype == (Mamba3Loader.narrows(key) ? .bfloat16 : .float32), "\(key)")
        }
        let gap = relative(narrow(tokens, cache: narrow.newCache(parameters: nil)), wide(tokens, cache: wide.newCache(parameters: nil)))
        #expect(gap < 3e-2)
    }

    @Test func theRegistryBuildsIt() async throws {
        var data = try Data(contentsOf: fixture.appendingPathComponent("config.json"))
        var json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        json["braid_a_activation"] = "heavy_tail"
        data = try JSONSerialization.data(withJSONObject: json)
        let model = try await LLMTypeRegistry.shared.createModel(configuration: data, modelType: "mamba3")
        #expect(model is Mamba3Model)
    }
}

/// The 1.5b itself, when this Mac holds it: FRIGATE_MAMBA3_COMMONS (the HF folder) and FRIGATE_MAMBA3_ORACLE (Braid's
/// `oracle/reference.safetensors`, with its `reference.json` beside it).
private var livePaths: (commons: URL, oracle: URL)? {
    let env = ProcessInfo.processInfo.environment
    guard let c = env["FRIGATE_MAMBA3_COMMONS"], let o = env["FRIGATE_MAMBA3_ORACLE"] else { return nil }
    return (URL(fileURLWithPath: c), URL(fileURLWithPath: o))
}

@Suite("Mamba-3, the live commons", .enabled(if: mlxFunctional && livePaths != nil), .serialized)
struct Mamba3LiveTests {

    /// Fixtures/llama3-tokens.json, recorded with Hugging Face's `tokenizers` (scripts/llama3-token-fixture.py).
    @Test func theTokenizerIsHuggingFaces() async throws {
        let (commons, _) = livePaths!
        let fixtureURL = fixture.deletingLastPathComponent().appendingPathComponent("llama3-tokens.json")
        let recorded = try JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL)) as! [String: Any]
        let tokenizer = try await HubTokenizerLoader().load(from: commons)
        for case let row as [String: Any] in recorded["cases"] as! [Any] {
            let text = row["text"] as! String
            #expect(tokenizer.encode(text: text, addSpecialTokens: true) == row["ids"] as! [Int], "\(text)")
        }
    }

    @Test func theCommonsIsTheOraclesAtFloat32AndCloseAtBF16() async throws {
        let (commons, oracle) = livePaths!
        let reference = try loadArrays(url: oracle)
        let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: oracle.deletingPathExtension().appendingPathExtension("json"))) as! [String: Any]
        let prompts = meta["prompts"] as! [[String: Any]]
        let tokenizer = try await HubTokenizerLoader().load(from: commons)
        for dtype in [nil, DType.bfloat16] as [DType?] {
            let model = try Mamba3Loader.load(from: commons, weightDType: dtype)
            for (i, prompt) in prompts.enumerated() {
                let ids = reference["prompt.\(i).tokens"]!.asArray(Int32.self).map(Int.init)
                if dtype == nil {
                    // The oracle opens every stream with BOS 128000 (Braid D31); the tokenizer adds it.
                    let encoded = tokenizer.encode(text: prompt["text"] as! String, addSpecialTokens: true)
                    #expect(encoded == ids, "prompt \(i): \(encoded) against the oracle's \(ids)")
                }
                let logits = model(MLXArray(ids.map(Int32.init), [1, ids.count]), cache: model.newCache(parameters: nil))[0]
                let want = reference["prompt.\(i).logits"]!
                let gap = abs(logits.asType(.float32) - want).max().item(Float.self)
                print(String(format: "MAMBA3 LIVE %@ prompt %d: max |Δ logit| %.3e, relative %.3e", dtype == nil ? "f32" : "bf16", i, gap, relative(logits, want)))
                if dtype == nil { #expect(gap < 1e-3) } else { #expect(relative(logits, want) < 5e-2) }
            }
        }
    }

    /// The state after a read holds itself and nothing of the read: its slices of k, x and the chunk states once kept
    /// every layer's whole tensors (5.8 GB after 1,024 tokens against the state's 54 MB).
    @Test func theStateHoldsOnlyItself() throws {
        let model = try Mamba3Loader.load(from: livePaths!.commons, weightDType: .bfloat16)
        eval(model)
        Memory.clearCache()
        let base = Memory.activeMemory
        let cache = model.newCache(parameters: nil)
        do {
            let tokens = (0 ..< 1_024).map { Int32(($0 * 7_919) % 120_000 + 100) }
            let reading = model.read(MLXArray(tokens).reshaped(1, 1_024), cache: cache, tap: 12)
            eval(reading.tappedSum, reading.last)
        }
        Memory.clearCache()
        let held = Memory.activeMemory - base
        let state = cache.reduce(0) { sum, layer in sum + (0 ..< 4).reduce(0) { $0 + (layer as! ArraysCache)[$1]!.nbytes } }
        print(String(format: "MAMBA3 LIVE held after a 1,024-token read: %.1f MB, the state %.1f MB", Double(held) / 1e6, Double(state) / 1e6))
        #expect(held < 2 * state)
    }
}
