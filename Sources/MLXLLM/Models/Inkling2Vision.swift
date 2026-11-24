//  WHAT: Inkling-2's own vision tower (`model.visual.*`, bf16 in the published checkpoint) and the read of a picture
//        through the text model already loaded: the picture is cut into 40-pixel tiles with no resize, each tile is
//        four native 20-pixel patches through a 23-layer ViT, the four are merged into one feature of the text's width,
//        and the features stand in for the image body's tokens after the text's embedding norm.
//  IN:   NeMo AutoModel's released reference, nemo_automodel/components/models/inkling2/{processing,vision,
//        multimodal}.py: every constant and reshape here follows it.
//  OUT:  `Inkling2OwnVision`, an `OwnVision` over a loaded `Inkling2Model`.

import CoreGraphics
import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct Inkling2VisionConfiguration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var mlpRatio: Double
    var nativePatchSize: Int
    var mergeSize: Int
    var projectionHiddenDim: Int
    var projectorBias: Bool
    var textHiddenSize: Int
    var gridSize: Int
    var layerNormEps: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size", hiddenLayers = "num_hidden_layers", attentionHeads = "num_attention_heads"
        case mlpRatio = "mlp_ratio", nativePatchSize = "native_patch_size", mergeSize = "adapter_merge_size"
        case projectionHiddenDim = "projection_hidden_dim", projectorBias = "projector_bias"
        case textHiddenSize = "text_hidden_size", gridSize = "posemb_grid_size", layerNormEps = "layer_norm_eps"
    }

    /// One tile's side in pixels, and one image token's: four native patches.
    var tile: Int { nativePatchSize * mergeSize }
}

// MARK: - The tower

final class Inkling2VisionEmbeddings: Module {
    let gridSize: Int
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ParameterInfo(key: "position_embedding") var positionEmbedding: MLXArray
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ParameterInfo(key: "cls_position_embedding") var clsPositionEmbedding: MLXArray

    init(_ config: Inkling2VisionConfiguration) {
        gridSize = config.gridSize
        let d = config.hiddenSize
        _patchEmbedding.wrappedValue = Linear(3 * config.nativePatchSize * config.nativePatchSize, d, bias: false)
        _positionEmbedding.wrappedValue = MLXArray.zeros([config.gridSize * config.gridSize, d])
        _classEmbedding.wrappedValue = MLXArray.zeros([d])
        _clsPositionEmbedding.wrappedValue = MLXArray.zeros([d])
    }

    /// The tent weights that resample the learned grid's `grid` positions to `size`: row i is centred at
    /// `(i + 0.5) · grid / size`, each sample at `j + 0.5` weighs `max(0, 1 − |centre − sample|)`, and a row sums to one.
    static func tent(_ size: Int, grid: Int) -> [Float] {
        var weights = [Float](repeating: 0, count: size * grid)
        for i in 0 ..< size {
            let centre = (Float(i) + 0.5) * Float(grid) / Float(size)
            var sum: Float = 0
            for j in 0 ..< grid {
                let w = max(0, 1 - abs(centre - (Float(j) + 0.5)))
                weights[i * grid + j] = w
                sum += w
            }
            for j in 0 ..< grid { weights[i * grid + j] /= sum }
        }
        return weights
    }

    /// `pixels` [height·width, 1200] in native row-major order → [1 + height·width, hidden], the class token first.
    func callAsFunction(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        let g = gridSize, d = positionEmbedding.dim(-1)
        let grid = positionEmbedding.asType(.float32).reshaped(g, g * d)
        let rows = MLXArray(Self.tent(height, grid: g), [height, g])
        let columns = MLXArray(Self.tent(width, grid: g), [width, g])
        // einsum("hi,wj,ijd->hwd"): over the grid's rows, then its columns.
        let byRow = matmul(rows, grid).reshaped(height, g, d)
        let positions = matmul(columns, byRow).reshaped(height * width, d).asType(positionEmbedding.dtype)
        let hidden = patchEmbedding(pixels) + positions
        return concatenated([(classEmbedding + clsPositionEmbedding).reshaped(1, d), hidden], axis: 0)
    }
}

final class Inkling2VisionAttention: Module {
    let heads: Int
    let headDim: Int
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: Inkling2VisionConfiguration) {
        heads = config.attentionHeads
        headDim = config.hiddenSize / config.attentionHeads
        let d = config.hiddenSize
        _qProj.wrappedValue = Linear(d, d)
        _kProj.wrappedValue = Linear(d, d)
        _vProj.wrappedValue = Linear(d, d)
        _outProj.wrappedValue = Linear(d, d)
    }

    /// The interleaved pairs of `x` [tokens, heads, headDim] turned by `cos`/`sin` [tokens, 1, headDim/2], in float32.
    static func rotate(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let pairs = x.asType(.float32).reshaped(x.dim(0), x.dim(1), -1, 2)
        let even = pairs[.ellipsis, 0], odd = pairs[.ellipsis, 1]
        return stacked([even * cos - odd * sin, even * sin + odd * cos], axis: -1).reshaped(x.shape).asType(x.dtype)
    }

    /// Full attention over one image, its class token included.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let n = x.dim(0)
        let q = Self.rotate(qProj(x).reshaped(n, heads, headDim), cos: cos, sin: sin)
        let k = Self.rotate(kProj(x).reshaped(n, heads, headDim), cos: cos, sin: sin)
        let v = vProj(x).reshaped(n, heads, headDim)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q.transposed(1, 0, 2).expandedDimensions(axis: 0),
            keys: k.transposed(1, 0, 2).expandedDimensions(axis: 0),
            values: v.transposed(1, 0, 2).expandedDimensions(axis: 0),
            scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return outProj(attended[0].transposed(1, 0, 2).reshaped(n, -1))
    }
}

final class Inkling2VisionMLP: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc") var fc: Linear
    @ModuleInfo(key: "c_proj") var proj: Linear

    init(_ config: Inkling2VisionConfiguration) {
        let inner = Int(Double(config.hiddenSize) * config.mlpRatio)
        _fc.wrappedValue = Linear(config.hiddenSize, inner)
        _proj.wrappedValue = Linear(inner, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(gelu(fc(x))) }
}

final class Inkling2VisionLayerScale: Module, UnaryLayer {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(_ dimensions: Int) { _gamma.wrappedValue = MLXArray.ones([dimensions]) }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

final class Inkling2VisionLayer: Module {
    @ModuleInfo(key: "ln_1") var norm1: LayerNorm
    @ModuleInfo(key: "ln_2") var norm2: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: Inkling2VisionAttention
    @ModuleInfo(key: "mlp") var mlp: Inkling2VisionMLP
    @ModuleInfo(key: "ls_1") var scale1: Inkling2VisionLayerScale
    @ModuleInfo(key: "ls_2") var scale2: Inkling2VisionLayerScale

    init(_ config: Inkling2VisionConfiguration) {
        _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _attention.wrappedValue = Inkling2VisionAttention(config)
        _mlp.wrappedValue = Inkling2VisionMLP(config)
        _scale1.wrappedValue = Inkling2VisionLayerScale(config.hiddenSize)
        _scale2.wrappedValue = Inkling2VisionLayerScale(config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + scale1(attention(norm1(x), cos: cos, sin: sin))
        return h + scale2(mlp(norm2(h)))
    }
}

final class Inkling2VisionBackbone: Module {
    let headDim: Int
    @ModuleInfo(key: "embeddings") var embeddings: Inkling2VisionEmbeddings
    @ModuleInfo(key: "ln_pre") var preNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [Inkling2VisionLayer]

    init(_ config: Inkling2VisionConfiguration) {
        headDim = config.hiddenSize / config.attentionHeads
        _embeddings.wrappedValue = Inkling2VisionEmbeddings(config)
        _preNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in Inkling2VisionLayer(config) }
    }

    /// The rotary angles of each token: (row + 1, column + 1) per patch, (0, 0) for the class token; the column's
    /// angles first, then the row's, each over `headDim / 4` frequencies `10000^(−4k/headDim)`.
    static func angles(height: Int, width: Int, headDim: Int) -> [Float] {
        let frequencies = stride(from: 0, to: headDim, by: 4).map { pow(10_000, -Float($0) / Float(headDim)) }
        var out = [Float](repeating: 0, count: frequencies.count * 2)       // the class token's
        out.reserveCapacity((1 + height * width) * frequencies.count * 2)
        for index in 0 ..< height * width {
            let y = Float(index / width + 1), x = Float(index % width + 1)
            out += frequencies.map { x * $0 }
            out += frequencies.map { y * $0 }
        }
        return out
    }

    /// `pixels` [height·width, 1200] → [height·width, hidden], the class token dropped.
    func callAsFunction(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        let tokens = 1 + height * width
        let angles = MLXArray(Self.angles(height: height, width: width, headDim: headDim), [tokens, 1, headDim / 2])
        let cos = MLX.cos(angles), sin = MLX.sin(angles)
        var h = preNorm(embeddings(pixels, height: height, width: width))
        for layer in layers { h = layer(h, cos: cos, sin: sin) }
        return h[1...]
    }
}

/// `model.visual`: the backbone and the projector into the text's width.
final class Inkling2VisionTower: Module {
    let config: Inkling2VisionConfiguration
    @ModuleInfo(key: "backbone") var backbone: Inkling2VisionBackbone
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear

    static let mean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    static let std: [Float] = [0.26862954, 0.2613026, 0.2757771]

    init(_ config: Inkling2VisionConfiguration) {
        self.config = config
        let merged = config.mergeSize * config.mergeSize * config.hiddenSize
        _backbone.wrappedValue = Inkling2VisionBackbone(config)
        _projIn.wrappedValue = Linear(merged, config.projectionHiddenDim, bias: config.projectorBias)
        _projOut.wrappedValue = Linear(config.projectionHiddenDim, config.textHiddenSize, bias: config.projectorBias)
    }

    /// One feature a tile: `tiles` [rows·columns, 40, 40, 3] as the processor normalized them (bf16), in row-major
    /// tile order → [rows·columns, textHidden].
    func callAsFunction(_ tiles: MLXArray, rows h: Int, columns w: Int) -> MLXArray {
        let m = config.mergeSize, p = config.nativePatchSize, d = config.hiddenSize
        let dtype = projOut.weight.dtype
        // Back to [−1, 1], in the processor's dtype as the reference does it.
        let std = MLXArray(Self.std).asType(.bfloat16), mean = MLXArray(Self.mean).asType(.bfloat16)
        let pixels = (2 * (tiles * std + mean) - 1).transposed(0, 3, 1, 2).asType(dtype)      // [N, 3, 40, 40]
        let native = pixels.reshaped(h, w, 3, m, p, m, p).transposed(0, 3, 1, 5, 4, 6, 2).reshaped(h * w * m * m, p * p * 3)
        let hidden = backbone(native, height: h * m, width: w * m)
        let merged = hidden.reshaped(h, m, w, m, d).transposed(0, 2, 1, 3, 4).reshaped(h * w, m * m * d)
        return projOut(gelu(projIn(merged)))
    }

    /// The picture as the released processor cuts it: no resize, `(side + 39) / 40` tiles a side, the pixels past the
    /// picture −1 (in 0–255), normalized with CLIP's constants and cast to bf16. A picture with alpha is laid on an
    /// 8-pixel 204/153 checkerboard first. The pixels, padded, as [rows·tile, columns·tile, 3] on the host.
    static func pixels(_ image: CGImage, tile: Int) throws -> (pixels: [Float], rows: Int, columns: Int) {
        let width = image.width, height = image.height
        guard width > 0, height > 0,
              let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw OwnVisionError.noPicture }
        switch image.alphaInfo {
        case .none, .noneSkipFirst, .noneSkipLast: break
        default:
            for y in stride(from: 0, to: height, by: 8) {
                for x in stride(from: 0, to: width, by: 8) {
                    // CoreGraphics counts rows from the bottom; the checkerboard is the picture's own, from its top.
                    let light = (y / 8 + x / 8) % 2 == 0
                    context.setFillColor(gray: light ? 204 / 255 : 153 / 255, alpha: 1)
                    context.fill(CGRect(x: x, y: height - min(y + 8, height), width: min(8, width - x), height: min(8, height - y)))
                }
            }
        }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        guard let data = context.data else { throw OwnVisionError.noPicture }
        let bytes = data.bindMemory(to: UInt8.self, capacity: width * height * 4)

        let rows = (height + tile - 1) / tile, columns = (width + tile - 1) / tile
        let paddedWidth = columns * tile
        var padded = [Float](repeating: -1, count: rows * tile * paddedWidth * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let from = (y * width + x) * 4, to = (y * paddedWidth + x) * 3
                padded[to] = Float(bytes[from])
                padded[to + 1] = Float(bytes[from + 1])
                padded[to + 2] = Float(bytes[from + 2])
            }
        }
        return (padded, rows, columns)
    }

    /// The padded pixels as the processor hands them over: [rows·columns, tile, tile, 3] in row-major tile order.
    static func tiles(_ pixels: [Float], rows: Int, columns: Int, tile: Int) -> MLXArray {
        let normalized = (MLXArray(pixels, [rows * tile, columns * tile, 3]) / 255 - MLXArray(mean)) / MLXArray(std)
        return normalized.reshaped(rows, tile, columns, tile, 3).transposed(0, 2, 1, 3, 4)
            .reshaped(rows * columns, tile, tile, 3).asType(.bfloat16)
    }
}

// MARK: - The read

/// Inkling-2 reading a picture with its own tower, over the text model the host already loaded.
public final class Inkling2OwnVision: OwnVision, @unchecked Sendable {
    let model: Inkling2Model
    let tower: Inkling2VisionTower
    let tokenizer: any Tokenizer

    /// The image body's placeholder in the template, and the id its tokens carry into the model.
    static let body = "<|unused_200054|>"
    static let contentImage = "<|content_image|>"
    /// A direct answer: the model's message opened as text, as Craft opens it when thinking is off.
    static let answerOpening = ["<|message_model|>", "<|content_text|>"]
    static let stops = ["<|end_message|>", "<|content_model_end_sampling|>", "<|message_user|>", "<|message_system|>",
                        "<|message_tool|>"]

    init(model: Inkling2Model, tower: Inkling2VisionTower, tokenizer: any Tokenizer) {
        self.model = model
        self.tower = tower
        self.tokenizer = tokenizer
    }

    /// The tower of the checkpoint in `directory`, over `model` when it is an Inkling-2 the host loaded from there; nil
    /// when it is not, or the checkpoint carries no vision config or no `model.visual` tensors.
    public static func over(_ model: any LanguageModel, tokenizer: any Tokenizer, directory: URL) throws -> Inkling2OwnVision? {
        guard let model = model as? Inkling2Model else { return nil }
        struct Config: Decodable {
            let visionConfig: Inkling2VisionConfiguration?
            enum CodingKeys: String, CodingKey { case visionConfig = "vision_config" }
        }
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        guard let config = try JSONDecoder().decode(Config.self, from: data).visionConfig else { return nil }
        let prefix = "model.visual."
        var weights: [String: MLXArray] = [:]
        for url in try OwnVisionCheckpoint.files(in: directory, prefix: prefix) {
            for (key, value) in try loadArrays(url: url) where key.hasPrefix(prefix) {
                weights[String(key.dropFirst(prefix.count))] = value
            }
        }
        guard !weights.isEmpty else { return nil }
        let tower = Inkling2VisionTower(config)
        try tower.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(tower)
        return Inkling2OwnVision(model: model, tower: tower, tokenizer: tokenizer)
    }

    /// The prompt's tokens with the image body expanded to `count` tokens of the content-image id, and where it begins.
    func prompt(question: String, count: Int) throws -> (tokens: [Int], start: Int) {
        guard let body = tokenizer.convertTokenToId(Self.body), let image = tokenizer.convertTokenToId(Self.contentImage)
        else { throw OwnVisionError.placeholder("the tokenizer has no image tokens") }
        let messages: [[String: any Sendable]] = [[
            "role": "user",
            "content": [["type": "image"], ["type": "text", "text": question]] as [[String: any Sendable]],
        ]]
        var tokens = try tokenizer.applyChatTemplate(messages: messages, tools: nil,
                                                     additionalContext: ["reasoning_effort": "none"])
        let opening = Self.answerOpening.compactMap(tokenizer.convertTokenToId)
        guard opening.count == Self.answerOpening.count else { throw OwnVisionError.placeholder("no answer opening") }
        tokens += opening
        guard let start = tokens.firstIndex(of: body) else { throw OwnVisionError.placeholder(Self.body) }
        tokens.replaceSubrange(start ... start, with: repeatElement(image, count: count))
        return (tokens, start)
    }

    public func reader(image: CGImage, question: String, maxTokens: Int) async throws -> @Sendable () throws -> String {
        let tile = tower.config.tile
        let (pixels, rows, columns) = try Inkling2VisionTower.pixels(image, tile: tile)
        let (tokens, start) = try prompt(question: question, count: rows * columns)
        let stops = Set(Self.stops.compactMap(tokenizer.convertTokenToId))
        return { [self] in
            try withError {
                let tiles = Inkling2VisionTower.tiles(pixels, rows: rows, columns: columns, tile: tile)
                let features = tower(tiles, rows: rows, columns: columns)
                let text = model.model.languageModel.embed(MLXArray(tokens.map(Int32.init)).reshaped(1, -1))
                let count = rows * columns
                let embeddings = concatenated([
                    text[0..., ..<start], features.asType(text.dtype).expandedDimensions(axis: 0),
                    text[0..., (start + count)...],
                ], axis: 1)
                return decode(embeddings: embeddings, stops: stops, maxTokens: maxTokens)
            }
        }
    }

    /// The prompt read in chunks of 512, then greedy tokens until a stop or `maxTokens`.
    func decode(embeddings: MLXArray, stops: Set<Int>, maxTokens: Int) -> String {
        let cache = model.newCache(parameters: nil)
        let length = embeddings.dim(1)
        var start = 0
        var logits = MLXArray(0)
        while start < length {
            let end = min(start + 512, length)
            logits = model(embeddings: embeddings[0..., start ..< end], cache: cache)
            eval(logits)
            start = end
        }
        var next = argMax(logits[0, -1]).item(Int.self)
        var tokens: [Int] = []
        while !stops.contains(next), tokens.count < maxTokens {
            tokens.append(next)
            logits = model(MLXArray([Int32(next)]).reshaped(1, 1), cache: cache)
            next = argMax(logits[0, -1]).item(Int.self)
        }
        return tokenizer.decode(tokenIds: tokens, skipSpecialTokens: true).trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
