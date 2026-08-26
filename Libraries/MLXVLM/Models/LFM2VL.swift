// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/lfm2_vl

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Vision

private enum Vision {

    fileprivate class Attention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(dims: Int, numHeads: Int, bias: Bool = true) {
            precondition(
                dims % numHeads == 0,
                "The input feature dimensions should be divisible by the number of heads")

            self.numHeads = numHeads
            let headDim = dims / numHeads
            self.scale = pow(Float(headDim), -0.5)

            self._qProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._kProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._vProj.wrappedValue = Linear(dims, dims, bias: bias)
            self._outProj.wrappedValue = Linear(dims, dims, bias: bias)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            let (B, L, _) = (queries.dim(0), queries.dim(1), queries.dim(2))
            let S = keys.dim(1)

            queries = queries.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)
            values = values.reshaped(B, S, numHeads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values,
                scale: scale,
                mask: mask.map { .array($0) } ?? .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return outProj(output)
        }
    }

    fileprivate class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear

        init(config: LFM2VLConfiguration.VisionConfiguration) {
            self._fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            self._fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(geluApproximate(fc1(x)))
        }
    }

    fileprivate class EncoderLayer: Module {
        let embedDim: Int
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(config: LFM2VLConfiguration.VisionConfiguration) {
            self.embedDim = config.hiddenSize

            self._selfAttn.wrappedValue = Attention(
                dims: config.hiddenSize, numHeads: config.numAttentionHeads, bias: true)
            self._layerNorm1.wrappedValue = LayerNorm(
                dimensions: embedDim, eps: config.layerNormEps)
            self.mlp = MLP(config: config)
            self._layerNorm2.wrappedValue = LayerNorm(
                dimensions: embedDim, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            var r = selfAttn(layerNorm1(x), mask: mask)
            var h = x + r
            r = mlp(layerNorm2(h))
            h = h + r
            return h
        }
    }

    fileprivate class Encoder: Module {
        var layers: [EncoderLayer]

        init(config: LFM2VLConfiguration.VisionConfiguration, visionFeatureLayer: Int = -1) {
            // Determine how many layers to create
            let numLayers: Int

            // visionFeatureLayer == -1 means use all layers
            // Other negative values are Python-style indices from the end (e.g., -2 = second to last)
            if visionFeatureLayer == -1 {
                numLayers = config.numHiddenLayers
            } else {
                // Convert negative indices to positive (e.g., -2 with 27 layers -> 25)
                let actualLayer =
                    visionFeatureLayer < 0
                    ? config.numHiddenLayers + visionFeatureLayer
                    : visionFeatureLayer

                if actualLayer >= 0 && actualLayer < config.numHiddenLayers {
                    numLayers = actualLayer + 1
                } else {
                    numLayers = config.numHiddenLayers
                }
            }
            self.layers = (0 ..< numLayers).map { _ in EncoderLayer(config: config) }
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false, mask: MLXArray? = nil)
            -> [MLXArray]?
        {
            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil

            var h = x
            for layer in layers {
                h = layer(h, mask: mask)
                if outputHiddenStates {
                    encoderStates?.append(h)
                }
            }

            return encoderStates
        }
    }

    fileprivate class VisionEmbeddings: Module {
        let config: LFM2VLConfiguration.VisionConfiguration
        let embedDim: Int
        let imageSize: Int
        let patchSize: Int
        let numPatches: Int
        let positionEmbeddingSize: Int

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(config: LFM2VLConfiguration.VisionConfiguration) {
            self.config = config
            self.embedDim = config.hiddenSize
            self.imageSize = config.imageSize
            self.patchSize = config.patchSize
            self.numPatches = config.numPatches
            self.positionEmbeddingSize = Int(sqrt(Double(numPatches)))

            self._patchEmbedding.wrappedValue = Linear(
                config.numChannels * patchSize * patchSize,
                embedDim
            )
            self._positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPatches, dimensions: embedDim)
        }

        /// Resize positional embeddings using bicubic interpolation
        static func resizePositionalEmbeddings(
            positionalEmbeddings: MLXArray,
            spatialShapes: MLXArray,
            maxLength: Int
        ) -> MLXArray {
            let batchSize = spatialShapes.dim(0)
            let srcH = positionalEmbeddings.dim(0)
            let srcW = positionalEmbeddings.dim(1)
            let embedDim = positionalEmbeddings.dim(-1)
            let sourceDtype = positionalEmbeddings.dtype

            let resultedPositionalEmbeddings = MLXArray.zeros(
                [batchSize, maxLength, embedDim], dtype: sourceDtype)

            // Reshape from [H, W, embedDim] to [1, embedDim, H, W] once before loop
            let reshapedEmbeddings =
                positionalEmbeddings
                .transposed(2, 0, 1)
                .reshaped(1, embedDim, srcH, srcW)

            for i in 0 ..< batchSize {
                let shape = spatialShapes[i]
                let targetH = shape[0].item(Int.self)
                let targetW = shape[1].item(Int.self)

                // Bicubic interpolation
                let interpolated = bicubicInterpolate(
                    reshapedEmbeddings,
                    size: (targetH, targetW)
                )

                // Reshape to [targetH * targetW, embedDim]
                let resizedEmbeddings =
                    interpolated
                    .reshaped(embedDim, targetH * targetW)
                    .transposed(1, 0)

                let numPositions = targetH * targetW
                resultedPositionalEmbeddings[i, 0 ..< numPositions] = resizedEmbeddings
                // Fill remaining positions with the first embedding
                if numPositions < maxLength {
                    for j in numPositions ..< maxLength {
                        resultedPositionalEmbeddings[i, j] = resizedEmbeddings[0]
                    }
                }
            }

            return resultedPositionalEmbeddings
        }

        func callAsFunction(_ pixelValues: MLXArray, spatialShapes: MLXArray) -> MLXArray {
            let targetDtype = patchEmbedding.weight.dtype
            let patchEmbeds = patchEmbedding(pixelValues.asType(targetDtype))

            let positionalEmbeddings = positionEmbedding.weight.reshaped(
                positionEmbeddingSize, positionEmbeddingSize, -1
            )

            let resizedPositionalEmbeddings = VisionEmbeddings.resizePositionalEmbeddings(
                positionalEmbeddings: positionalEmbeddings,
                spatialShapes: spatialShapes,
                maxLength: pixelValues.dim(1)
            )

            let embeddings = patchEmbeds + resizedPositionalEmbeddings
            return embeddings
        }
    }

    fileprivate class VisionModel: Module {
        let modelType: String

        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayernorm: LayerNorm

        init(config: LFM2VLConfiguration.VisionConfiguration, visionFeatureLayer: Int = -1) {
            self.modelType = config.modelType

            self.embeddings = VisionEmbeddings(config: config)
            self.encoder = Encoder(config: config, visionFeatureLayer: visionFeatureLayer)
            self._postLayernorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(
            _ x: MLXArray,
            outputHiddenStates: Bool = false,
            spatialShapes: MLXArray,
            attentionMask: MLXArray? = nil
        ) -> (encoderOutputs: [MLXArray]?, embeddings: MLXArray, lastHiddenState: MLXArray) {
            var embeds = embeddings(x, spatialShapes: spatialShapes)
            embeds = embeds.asType(embeddings.patchEmbedding.weight.dtype)

            let encoderOutputs = encoder(
                embeds, outputHiddenStates: outputHiddenStates, mask: attentionMask)
            let lastHiddenState = postLayernorm(encoderOutputs?.last ?? embeds)

            return (encoderOutputs, embeds, lastHiddenState)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (k, v) in weights {
                if k.contains("position_ids") {
                    continue
                } else {
                    sanitizedWeights[k] = v
                }
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Language Model Components (LFM2)

private enum Language {

    fileprivate class LFM2Attention: Module {
        let scale: Float
        let headDim: Int
        let heads: Int
        let kvHeads: Int

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

        let rope: RoPE

        init(_ config: LFM2VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.attentionHeads
            self.kvHeads = config.kvHeads
            self.headDim = dim / heads
            self.scale = pow(Float(headDim), -0.5)

            self._qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
            self._kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            self._outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

            self._qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            self._kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)

            self.rope = RoPE(
                dimensions: headDim,
                traditional: false,
                base: config.ropeTheta
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))

            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = qLayerNorm(queries.reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
            keys = kLayerNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

            if let cache {
                queries = rope(queries, offset: cache.offset)
                keys = rope(keys, offset: cache.offset)
            } else {
                queries = rope(queries)
                keys = rope(keys)
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return outProj(output)
        }
    }

    fileprivate class LFM2ShortConv: Module {
        let lCache: Int
        let hiddenSize: Int

        @ModuleInfo(key: "conv") var conv: Conv1d
        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LFM2VLConfiguration.TextConfiguration, layerIdx: Int) {
            self.lCache = config.convLCache
            self.hiddenSize = config.hiddenSize
            let bias = config.convBias

            self._conv.wrappedValue = Conv1d(
                inputChannels: config.hiddenSize,
                outputChannels: config.hiddenSize,
                kernelSize: lCache,
                groups: config.hiddenSize,
                bias: bias
            )

            self._inProj.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: bias)
            self._outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: bias)
        }

        func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
            let BCx = inProj(x)
            let BCxSplit = BCx.split(parts: 3, axis: -1)
            let B = BCxSplit[0]
            let C = BCxSplit[1]
            let xPart = BCxSplit[2]
            var Bx = B * xPart

            var state: MLXArray? = nil
            if let cache {
                state = cache[0]
            }
            if state == nil {
                state = MLXArray.zeros([Bx.dim(0), lCache - 1, hiddenSize], dtype: Bx.dtype)
            }

            Bx = concatenated([state!, Bx], axis: -2)
            if let cache {
                cache[0] = contiguous(Bx[0..., (Bx.dim(1) - (lCache - 1))..., 0...])
                cache.advance(x.dim(1))
            }

            let convOut = conv(Bx)
            let y = C * convOut
            return outProj(y)
        }
    }

    fileprivate class LFM2MLP: Module, UnaryLayer {
        @ModuleInfo(key: "w1") var w1: Linear
        @ModuleInfo(key: "w2") var w2: Linear
        @ModuleInfo(key: "w3") var w3: Linear

        init(_ config: LFM2VLConfiguration.TextConfiguration) {
            var adjustedFFDim = config.blockFFDim

            if config.blockAutoAdjustFFDim {
                adjustedFFDim = Int(Float(2 * adjustedFFDim) / 3.0)
                adjustedFFDim = Int(config.blockFFNDimMultiplier * Float(adjustedFFDim))
                adjustedFFDim =
                    config.blockMultipleOf
                    * ((adjustedFFDim + config.blockMultipleOf - 1) / config.blockMultipleOf)
            }

            self._w1.wrappedValue = Linear(config.blockDim, adjustedFFDim, bias: false)
            self._w2.wrappedValue = Linear(adjustedFFDim, config.blockDim, bias: false)
            self._w3.wrappedValue = Linear(config.blockDim, adjustedFFDim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            w2(silu(w1(x)) * w3(x))
        }
    }

    fileprivate class LFM2DecoderLayer: Module {
        let isAttentionLayer: Bool

        @ModuleInfo(key: "self_attn") var attention: LFM2Attention?
        @ModuleInfo(key: "conv") var conv: LFM2ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: LFM2MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: LFM2VLConfiguration.TextConfiguration, layerIdx: Int) {
            self.isAttentionLayer = config.fullAttnIdxs.contains(layerIdx)

            if isAttentionLayer {
                self._attention.wrappedValue = LFM2Attention(config)
            } else {
                self._conv.wrappedValue = LFM2ShortConv(config, layerIdx: layerIdx)
            }

            self._feedForward.wrappedValue = LFM2MLP(config)
            self._operatorNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            self._ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let r: MLXArray
            if isAttentionLayer {
                r = attention!(operatorNorm(x), mask: mask, cache: cache)
            } else {
                r = conv!(operatorNorm(x), cache: cache as? MambaCache)
            }
            let h = x + r
            let out = h + feedForward(ffnNorm(h))
            return out
        }
    }

    fileprivate class LFM2ModelInner: Module {
        let config: LFM2VLConfiguration.TextConfiguration
        let vocabularySize: Int
        let numHiddenLayers: Int

        let layers: [LFM2DecoderLayer]

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

        init(_ config: LFM2VLConfiguration.TextConfiguration) {
            self.config = config
            self.vocabularySize = config.vocabularySize
            self.numHiddenLayers = config.hiddenLayers

            precondition(vocabularySize > 0)

            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: vocabularySize, dimensions: config.hiddenSize)

            self.layers = (0 ..< numHiddenLayers).map { i in
                LFM2DecoderLayer(config, layerIdx: i)
            }

            self._embeddingNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ inputs: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
            cache: [KVCache]? = nil, inputEmbeddings: MLXArray? = nil
        ) -> MLXArray {
            var h = inputEmbeddings ?? embedTokens(inputs)

            let mask =
                mask
                ?? {
                    let firstAttnIdx = config.fullAttnIdxs.first ?? 0
                    let c =
                        cache != nil && firstAttnIdx < cache!.count ? cache![firstAttnIdx] : nil
                    return createAttentionMask(h: h, cache: c)
                }()

            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            return embeddingNorm(h)
        }
    }

    fileprivate class LanguageModel: Module, KVCacheDimensionProvider {
        let config: LFM2VLConfiguration.TextConfiguration
        let modelType: String
        let model: LFM2ModelInner

        var kvHeads: [Int]

        init(_ config: LFM2VLConfiguration.TextConfiguration) {
            self.config = config
            self.modelType = config.modelType

            self.model = LFM2ModelInner(config)

            self.kvHeads = (0 ..< config.hiddenLayers).map { layerIdx in
                config.fullAttnIdxs.contains(layerIdx) ? config.kvHeads : 0
            }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            mask: MLXArray? = nil,
            cache: [KVCache]? = nil,
            inputsEmbeds: MLXArray? = nil
        ) -> LMOutput {
            var out = model(
                inputs ?? MLXArray([0]), cache: cache, inputEmbeddings: inputsEmbeds)
            out = model.embedTokens.asLinear(out)
            return LMOutput(logits: out)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()

            for (name, param) in weights {
                var sanitizedParam = param

                if name.contains("conv.weight") {
                    if param.shape[param.shape.count - 1] > param.dim(1) {
                        sanitizedParam = param.transposed(0, 2, 1)
                    }
                }

                sanitizedWeights[name] = sanitizedParam
            }

            return sanitizedWeights
        }
    }
}

// MARK: - Multi-modal Projector

private class Lfm2VlMultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm?
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(config: LFM2VLConfiguration) {
        let inChannels =
            config.visionConfiguration.hiddenSize
            * (config.downsampleFactor * config.downsampleFactor)

        if config.projectorUseLayernorm {
            self._layerNorm.wrappedValue = LayerNorm(dimensions: inChannels)
        }

        self._linear1.wrappedValue = Linear(
            inChannels, config.projectorHiddenSize, bias: config.projectorBias)
        self._linear2.wrappedValue = Linear(
            config.projectorHiddenSize, config.textConfiguration.hiddenSize,
            bias: config.projectorBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if let layerNorm {
            x = layerNorm(x)
        }
        x = linear1(x)
        x = gelu(x)
        x = linear2(x)
        return x
    }
}

// MARK: - PixelUnshuffleBlock

private class PixelUnshuffleBlock: Module, UnaryLayer {
    let factor: Int

    init(factor: Int) {
        self.factor = factor
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        var (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

        // Pad width if necessary
        if w % factor != 0 {
            let padW = factor - (w % factor)
            let padding = MLXArray.zeros([n, padW, h, c], dtype: x.dtype)
            x = concatenated([x, padding], axis: 1)
            (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        }

        // Pad height if necessary
        if h % factor != 0 {
            let padH = factor - (h % factor)
            let padding = MLXArray.zeros([n, w, padH, c], dtype: x.dtype)
            x = concatenated([x, padding], axis: 2)
            (n, w, h, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        }

        x = x.reshaped(n, w, h / factor, c * factor)
        x = x.transposed(0, 2, 1, 3)
        x = x.reshaped(n, h / factor, w / factor, c * factor * factor)
        x = x.transposed(0, 2, 1, 3)

        return x
    }
}

// MARK: - Processor

/// Resolves a token id, treating the tokenizer's unknown id as "absent from this vocabulary".
///
/// `convertTokenToId` maps a token that is not in the vocabulary to the unknown id rather than
/// to nil, so a bare `??` fallback behind it can never fire.
private func resolveTokenId(_ token: String, _ tokenizer: any Tokenizer) -> Int? {
    let resolved = tokenizer.convertTokenToId(token)
    return resolved == tokenizer.unknownTokenId ? nil : resolved
}

/// LFM2 VL VLM `UserInputProcessor`.
///
/// Port of `Lfm2VlImageProcessor` / `Lfm2VlProcessor` (transformers, `models/lfm2_vl`). The
/// geometry is driven entirely by the checkpoint's own processor config: LFM2.5-VL ships
/// `do_image_splitting: false` (one `smart_resize`d unit per image) while LFM2-VL ships it
/// enabled (a grid of tiles plus a thumbnail).
///
/// This is meant to be used with ``LFM2VL`` and is typically created by ``VLMModelFactory``.
public struct LFM2VLProcessor: UserInputProcessor {

    /// How one image expands after the `resize_and_split` decision.
    struct ImageLayout: Equatable {
        let rows: Int
        let columns: Int
        /// Size after `smart_resize`; doubles as the thumbnail size in multi-tile mode.
        let resized: CGSize
        /// Pixel size of every unit handed to the vision encoder, in emission order.
        let units: [CGSize]

        var isMultiTile: Bool { rows > 1 || columns > 1 }
    }

    /// A single image's expanded placeholder run.
    struct ExpandedImage {
        let tokens: [Int]
        /// How many of `tokens` are the image token itself — must equal the encoder's rows.
        let imageTokenCount: Int
    }

    private let config: LFM2VLProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenId: Int?

    // Not present in any config file: `Lfm2VlProcessor.__init__` reads them off the tokenizer,
    // which our `Tokenizer` protocol does not expose. These are its defaults.
    private let imageStartToken = "<|image_start|>"
    private let imageEndToken = "<|image_end|>"
    private let imageThumbnailToken = "<|img_thumbnail|>"

    public init(_ config: LFM2VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        // The placeholder id is model specific (396 in LFM2-VL, 124907 in LFM2.5-VL): resolve it
        // from the vocabulary so the expansion below agrees with
        // `LFM2VLConfiguration.imageTokenIndex`, which `mergeInputIdsWithImageFeatures` uses to
        // count the positions to fill.
        self.imageTokenId = resolveTokenId(config.imageToken, tokenizer)
    }

    // MARK: Geometry

    /// `round_by_factor`. Python's `round` is half-to-even; `Double.rounded()` is not, and the
    /// difference decides both the resize target and the tiling threshold (720 → 704, not 736).
    private func roundByFactor(_ number: Int, _ factor: Int) -> Int {
        Int((Double(number) / Double(factor)).rounded(.toNearestOrEven)) * factor
    }

    private func ceilDiv(_ a: Int, _ b: Int) -> Int {
        (a + b - 1) / b
    }

    private var pixelsPerToken: Int {
        config.encoderPatchSize * config.encoderPatchSize * config.downsampleFactor
            * config.downsampleFactor
    }

    /// Port of `smart_resize`. Returns `(width, height)` — width first, like Python.
    func smartResize(height: Int, width: Int) -> (width: Int, height: Int) {
        let totalFactor = config.encoderPatchSize * config.downsampleFactor
        let minPixels = config.minImageTokens * pixelsPerToken
        let maxPixels = config.maxImageTokens * pixelsPerToken

        var hBar = max(totalFactor, roundByFactor(height, totalFactor))
        var wBar = max(totalFactor, roundByFactor(width, totalFactor))

        if hBar * wBar > maxPixels {
            let beta = (Double(height) * Double(width) / Double(maxPixels)).squareRoot()
            hBar = max(
                totalFactor,
                Int((Double(height) / beta / Double(totalFactor)).rounded(.down)) * totalFactor)
            wBar = max(
                totalFactor,
                Int((Double(width) / beta / Double(totalFactor)).rounded(.down)) * totalFactor)
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / (Double(height) * Double(width))).squareRoot()
            hBar = Int((Double(height) * beta / Double(totalFactor)).rounded(.up)) * totalFactor
            wBar = Int((Double(width) * beta / Double(totalFactor)).rounded(.up)) * totalFactor
        }

        return (wBar, hBar)
    }

    /// Port of `_is_image_too_large`.
    func isImageTooLarge(height: Int, width: Int) -> Bool {
        let totalFactor = config.encoderPatchSize * config.downsampleFactor
        let hBar = max(config.encoderPatchSize, roundByFactor(height, totalFactor))
        let wBar = max(config.encoderPatchSize, roundByFactor(width, totalFactor))
        let limit = Double(config.maxImageTokens * pixelsPerToken) * config.maxPixelsTolerance
        return Double(hBar * wBar) > limit
    }

    /// Port of `_target_ratios`, as `(width, height)` pairs ordered by tile count.
    ///
    /// Python sorts a `set` by tile count, leaving the order within a count unspecified; ordering
    /// by `(count, width, height)` matches it for every tie that `find_closest_aspect_ratio`
    /// resolves by area, since those candidates always differ in tile count.
    func targetRatios(minTiles: Int, maxTiles: Int) -> [(Int, Int)] {
        guard minTiles <= maxTiles else { return [] }
        var seen = Set<[Int]>()
        var ratios = [(Int, Int)]()
        for n in minTiles ... maxTiles {
            for w in 1 ... n {
                for h in 1 ... n where w * h >= minTiles && w * h <= maxTiles {
                    if seen.insert([w, h]).inserted {
                        ratios.append((w, h))
                    }
                }
            }
        }
        return ratios.sorted {
            ($0.0 * $0.1, $0.0, $0.1) < ($1.0 * $1.1, $1.0, $1.1)
        }
    }

    /// Port of `find_closest_aspect_ratio`.
    func closestAspectRatio(
        _ aspectRatio: Double, _ ratios: [(Int, Int)], width: Int, height: Int, imageSize: Int
    ) -> (Int, Int) {
        var bestDiff = Double.infinity
        var best = (1, 1)
        let area = Double(width * height)

        for ratio in ratios {
            let diff = abs(aspectRatio - Double(ratio.0) / Double(ratio.1))
            if diff < bestDiff {
                bestDiff = diff
                best = ratio
            } else if diff == bestDiff {
                let targetArea = Double(imageSize * imageSize * ratio.0 * ratio.1)
                if area > 0.5 * targetArea {
                    best = ratio
                }
            }
        }

        return best
    }

    /// Port of `resize_and_split`'s decision, without touching pixels.
    func layout(height: Int, width: Int) -> ImageLayout {
        // `_preprocess` clamps the bounds when splitting is off and `resize_and_split` then
        // re-derives the flag from the clamped bounds.
        let bounds = config.doImageSplitting ? (config.minTiles, config.maxTiles) : (1, 1)
        let splitting = !(bounds.0 == 1 && bounds.1 == 1)

        let (newWidth, newHeight) = smartResize(height: height, width: width)
        let resized = CGSize(width: newWidth, height: newHeight)

        guard splitting, isImageTooLarge(height: height, width: width) else {
            return ImageLayout(rows: 1, columns: 1, resized: resized, units: [resized])
        }

        let ratios = targetRatios(minTiles: bounds.0, maxTiles: bounds.1)
        // `crop_image_to_patches` returns (grid_width, grid_height) and the caller binds them
        // as (num_cols, num_rows) — columns first.
        let (columns, rows) = closestAspectRatio(
            Double(width) / Double(height), ratios,
            width: width, height: height, imageSize: config.tileSize)

        let tile = CGSize(width: config.tileSize, height: config.tileSize)
        var units = Array(repeating: tile, count: rows * columns)
        if config.useThumbnail && rows * columns != 1 {
            units.append(resized)
        }

        return ImageLayout(rows: rows, columns: columns, resized: resized, units: units)
    }

    // MARK: Token sequence

    /// Port of `_compute_tokens_per_tile`.
    var tokensPerTile: Int {
        let patches = config.tileSize / config.encoderPatchSize
        let downsampled = ceilDiv(patches, config.downsampleFactor)
        return downsampled * downsampled
    }

    /// Port of `_compute_tokens_for_image`.
    func tokensForImage(height: Int, width: Int) -> Int {
        ceilDiv(height / config.encoderPatchSize, config.downsampleFactor)
            * ceilDiv(width / config.encoderPatchSize, config.downsampleFactor)
    }

    /// Port of `_build_image_tokens`.
    ///
    /// Marker ids missing from the vocabulary are skipped rather than faked: the image token
    /// count — the only thing the feature merge counts — never depends on them.
    func expand(layout: ImageLayout, imageTokenId: Int) -> ExpandedImage {
        var tokens = [Int]()
        var imageTokens = 0
        let special = config.useImageSpecialTokens

        func appendMarker(_ token: String) {
            if special, let id = resolveTokenId(token, tokenizer) {
                tokens.append(id)
            }
        }
        func appendImageTokens(_ count: Int) {
            tokens.append(contentsOf: repeatElement(imageTokenId, count: count))
            imageTokens += count
        }

        appendMarker(imageStartToken)

        let forImage = tokensForImage(
            height: Int(layout.resized.height), width: Int(layout.resized.width))

        if layout.isMultiTile {
            let perTile = tokensPerTile
            for row in 0 ..< layout.rows {
                for column in 0 ..< layout.columns {
                    appendMarker("<|img_row_\(row + 1)_col_\(column + 1)|>")
                    appendImageTokens(perTile)
                }
            }
            if config.useThumbnail {
                appendMarker(imageThumbnailToken)
                appendImageTokens(forImage)
            }
        } else {
            appendImageTokens(forImage)
        }

        appendMarker(imageEndToken)

        return ExpandedImage(tokens: tokens, imageTokenCount: imageTokens)
    }

    // MARK: Pixels

    /// Resize → sRGB → normalize → `[height, width, channels]`.
    ///
    /// Bicubic is a deliberate divergence: LFM2.5-VL's MLX conversion omits `resample`, so the
    /// Python class default (bilinear) would apply, but the source checkpoint asks for bicubic.
    private func unitArray(from image: CIImage, size: CGSize) -> MLXArray {
        let resized = image.toSRGB().resampled(to: size, method: .bicubic)
        let normalized = resized.normalized(
            mean: config.imageMeanTuple, std: config.imageStdTuple)
        // [1, C, H, W] -> [H, W, C]
        return MediaProcessing.asMLXArray(normalized).transposed(0, 2, 3, 1)[0]
    }

    /// Port of `convert_image_to_patches`: `[H, W, C]` → `[patches, patchSize² · C]`, row-major,
    /// each patch flattened as (row, column, channel).
    func patches(from unit: MLXArray) -> MLXArray {
        let patchSize = config.encoderPatchSize
        let channels = unit.dim(2)
        let rows = unit.dim(0) / patchSize
        let columns = unit.dim(1) / patchSize

        return unit[0 ..< rows * patchSize, 0 ..< columns * patchSize, 0...]
            .reshaped(rows, patchSize, columns, patchSize, channels)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(rows * columns, patchSize * patchSize * channels)
    }

    /// The units one image expands into, already normalized, in emission order.
    ///
    /// Tiles are cut from the resized array rather than from the `CIImage` so that row 0 is
    /// unambiguously the top row (CoreImage's origin is bottom-left).
    func units(from image: CIImage, layout: ImageLayout) -> [MLXArray] {
        guard layout.isMultiTile else {
            return [unitArray(from: image, size: layout.resized)]
        }

        let tileSize = config.tileSize
        let grid = unitArray(
            from: image,
            size: CGSize(
                width: tileSize * layout.columns, height: tileSize * layout.rows))

        var units = [MLXArray]()
        for row in 0 ..< layout.rows {
            for column in 0 ..< layout.columns {
                let top = row * tileSize
                let left = column * tileSize
                units.append(grid[top ..< top + tileSize, left ..< left + tileSize, 0...])
            }
        }
        if config.useThumbnail {
            // Python resizes the original image, not the tiled one.
            units.append(unitArray(from: image, size: layout.resized))
        }

        return units
    }

    // MARK: Entry point

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)

        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext
        )

        // Text-only input
        if input.images.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        guard let imageTokenId else {
            throw VLMError.processing(
                "The tokenizer has no image placeholder token \(config.imageToken)")
        }

        var allPatches = [MLXArray]()
        var frames = [THW]()
        var expansions = [ExpandedImage]()

        for imageInput in input.images {
            let image = MediaProcessing.apply(
                try imageInput.asCIImage(), processing: input.processing)
            let layout = layout(
                height: Int(image.extent.height), width: Int(image.extent.width))

            for unit in units(from: image, layout: layout) {
                let patches = patches(from: unit)
                allPatches.append(patches)
                frames.append(
                    THW(
                        1, unit.dim(0) / config.encoderPatchSize,
                        unit.dim(1) / config.encoderPatchSize))
            }
            expansions.append(expand(layout: layout, imageTokenId: imageTokenId))
        }

        // Replace each image placeholder with that image's expansion. The LFM2 templates emit
        // exactly one placeholder per image, so adjacent images produce adjacent placeholders:
        // consuming a whole run for a single image would drop every image but the first and leave
        // `mergeInputIdsWithImageFeatures` with fewer positions than features.
        //
        // `imageTokenId` above is already resolved per-model from the vocabulary (not
        // hardcoded), which is what upstream's equivalent fix (#576) does for its own,
        // run-consuming placeholder algorithm -- not applicable here.
        var newPromptTokens = [Int]()
        var imageIdx = 0
        for token in promptTokens {
            guard token == imageTokenId else {
                newPromptTokens.append(token)
                continue
            }
            guard imageIdx < expansions.count else {
                throw VLMError.processing(
                    "More image placeholders than images: at least \(imageIdx + 1) placeholders "
                        + "for \(expansions.count) images")
            }
            newPromptTokens.append(contentsOf: expansions[imageIdx].tokens)
            imageIdx += 1
        }
        guard imageIdx == expansions.count else {
            throw VLMError.processing(
                "Fewer image placeholders than images: \(imageIdx) placeholders "
                    + "for \(expansions.count) images")
        }
        promptTokens = newPromptTokens

        // The encoder emits `ceil(h / downsample) · ceil(w / downsample)` rows per unit — the
        // same rounding `_compute_tokens_*` uses. Checking it here turns any divergence into a
        // recoverable error instead of the `fatalError` in the feature merge.
        let featureRows = frames.reduce(0) {
            $0 + ceilDiv($1.h, config.downsampleFactor) * ceilDiv($1.w, config.downsampleFactor)
        }
        let emitted = expansions.reduce(0) { $0 + $1.imageTokenCount }
        guard emitted == featureRows else {
            throw VLMError.processing(
                "Image token count \(emitted) does not match the \(featureRows) rows the "
                    + "encoder will produce")
        }

        // `pad_along_first_dim`: units have different patch counts (tiles vs thumbnail, or two
        // images of different aspect ratios), so they are padded to a common length before being
        // stacked. `LFM2VL.prepare` rebuilds the matching attention mask from `frames`, and the
        // vision encoder masks the padding out. Python pads to a fixed `max_num_patches`; padding
        // to the batch maximum is numerically identical and up to 4x cheaper on device.
        let target = allPatches.map { $0.dim(0) }.max() ?? 0
        let padded = allPatches.map { patches -> MLXArray in
            let missing = target - patches.dim(0)
            guard missing > 0 else { return patches }
            let padding = MLXArray.zeros([missing, patches.dim(1)], dtype: patches.dtype)
            return concatenated([patches, padding], axis: 0)
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: LMInput.ProcessedImage(
                pixels: stacked(padded, axis: 0),
                frames: frames
            )
        )
    }
}

// MARK: - Model

/// LFM2 VL VLM
///
/// This is typically created by ``VLMModelFactory``.
public class LFM2VL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionModel: Vision.VisionModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        Lfm2VlMultiModalProjector
    @ModuleInfo(key: "language_model") private var languageModel: Language.LanguageModel
    @ModuleInfo(key: "pixel_unshuffle") private var pixelUnshuffle: PixelUnshuffleBlock?

    public let config: LFM2VLConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.model.layers.map { $0 as Module }
    }

    public init(_ config: LFM2VLConfiguration) {
        self.config = config

        self._visionModel.wrappedValue = Vision.VisionModel(
            config: config.visionConfiguration,
            visionFeatureLayer: config.visionFeatureLayer
        )

        if config.downsampleFactor > 1 {
            self._pixelUnshuffle.wrappedValue = PixelUnshuffleBlock(factor: config.downsampleFactor)
        }

        self._multiModalProjector.wrappedValue = Lfm2VlMultiModalProjector(config: config)
        self._languageModel.wrappedValue = Language.LanguageModel(config.textConfiguration)
    }

    private func getInputEmbeddings(
        inputIds: MLXArray,
        pixelValues: MLXArray?,
        spatialShapes: MLXArray?,
        pixelAttentionMask: MLXArray?
    ) -> MLXArray {
        // Ensure inputIds has batch dimension
        var batchedInputIds = inputIds
        if inputIds.ndim == 1 {
            batchedInputIds = inputIds.expandedDimensions(axis: 0)
        }

        var inputsEmbeds = languageModel.model.embedTokens(batchedInputIds)

        // Ensure embeddings have batch dimension
        if inputsEmbeds.ndim == 2 {
            inputsEmbeds = inputsEmbeds.expandedDimensions(axis: 0)
        }

        guard let pixelValues, let spatialShapes, let pixelAttentionMask else {
            return inputsEmbeds
        }

        // Units of different patch counts (tiles vs thumbnail, or images of different aspect
        // ratios) are padded to a common length by the processor. Those padded patches must not
        // take part in the vision self-attention: Python feeds `pixel_attention_mask` to the
        // tower (`Lfm2VlModel.get_image_features`). Masking keys only, so every query keeps at
        // least one finite score.
        var visionMask: MLXArray?
        if pixelValues.dim(0) > 1 {
            let dtype = visionModel.embeddings.patchEmbedding.weight.dtype
            visionMask =
                MLX
                .where(
                    pixelAttentionMask .> 0, MLXArray(Float(0)), MLXArray(-Float.infinity)
                )
                .asType(dtype)
                .reshaped(pixelValues.dim(0), 1, 1, pixelValues.dim(1))
        }

        // Get the output hidden states from the vision model
        let visionOutput = visionModel(
            pixelValues, outputHiddenStates: true, spatialShapes: spatialShapes,
            attentionMask: visionMask)
        let hiddenStates = visionOutput.lastHiddenState

        // Get feature lengths from attention mask
        let imgFeatureLengths = sum(pixelAttentionMask, axis: 1)

        var imageFeatures = [MLXArray]()

        for imgIdx in 0 ..< hiddenStates.dim(0) {
            var feature = hiddenStates[imgIdx]
            let featureLength = imgFeatureLengths[imgIdx].item(Int.self)

            // Slice to valid features
            feature = feature[0 ..< featureLength].expandedDimensions(axis: 0)

            // Get spatial dimensions
            let featureOrgH = spatialShapes[imgIdx, 0].item(Int.self)
            let featureOrgW = spatialShapes[imgIdx, 1].item(Int.self)

            // Reshape to spatial dimensions
            feature = feature.reshaped(1, featureOrgH, featureOrgW, -1)

            // Apply pixel unshuffle if configured
            if let pixelUnshuffle {
                feature = pixelUnshuffle(feature)
            }

            // Project to language model dimension
            var imgEmbedding = multiModalProjector(feature)

            // Flatten back
            imgEmbedding = imgEmbedding.reshaped(-1, imgEmbedding.dim(-1))
            imageFeatures.append(imgEmbedding)
        }

        let concatenatedImageFeatures = concatenated(imageFeatures, axis: 0)

        // Merge image features with text embeddings
        return mergeInputIdsWithImageFeatures(
            imageFeatures: concatenatedImageFeatures,
            inputsEmbeds: inputsEmbeds,
            inputIds: inputIds,
            imageTokenIndex: config.imageTokenIndex
        )
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int
    ) -> MLXArray {
        // Find image token positions
        var imageIndices = [Int]()
        for (i, v) in inputIds.flattened().asArray(Int.self).enumerated() {
            if v == imageTokenIndex {
                imageIndices.append(i)
            }
        }

        let nImageFeatures = imageFeatures.dim(0)
        if imageIndices.count != nImageFeatures {
            fatalError(
                "Image features and image tokens do not match: tokens: \(imageIndices.count), features \(nImageFeatures)"
            )
        }

        // Make sure shapes match before assignment
        var result = inputsEmbeds
        if result.ndim == 2 {
            result = result.expandedDimensions(axis: 0)
        }

        // Assign image features to the image token positions
        if imageFeatures.ndim == 2 {
            let reshapedFeatures = imageFeatures.expandedDimensions(axis: 0)
            result[0..., MLXArray(imageIndices), 0...] = reshapedFeatures
        } else {
            result[0..., MLXArray(imageIndices), 0...] = imageFeatures
        }

        return result
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?, prefill: PrefillParameters
    ) throws
        -> PrepareResult
    {
        let dtype = visionModel.embeddings.patchEmbedding.weight.dtype

        // Get image data if available
        let pixelValues = input.image?.pixels.asType(dtype)

        var spatialShapes: MLXArray? = nil
        var pixelAttentionMask: MLXArray? = nil

        if let pixels = pixelValues, let frames = input.image?.frames, !frames.isEmpty {
            // One frame per unit fed to the encoder: a whole image, or a tile, or a thumbnail.

            // Convert frames to spatial shapes array [numUnits, 2]
            let shapeArrays = frames.map { MLXArray([$0.h, $0.w]) }
            spatialShapes = stacked(shapeArrays, axis: 0)

            // Mark the valid patches of each unit. The target is the padded width the processor
            // produced, not the widest frame, so the mask always lines up with `pixels`.
            let paddedLength = pixels.dim(1)
            let maskArrays = frames.map { frame -> MLXArray in
                let numPatches = min(frame.h * frame.w, paddedLength)
                let mask = MLXArray.ones([numPatches]).asType(.int32)
                guard numPatches < paddedLength else { return mask }
                let padding = MLXArray.zeros([paddedLength - numPatches]).asType(.int32)
                return concatenated([mask, padding], axis: 0)
            }
            pixelAttentionMask = stacked(maskArrays, axis: 0)
        } else if let pixels = pixelValues {
            // Fallback: infer spatial shapes from pixel dimensions (assumes square)
            let numPatches = pixels.dim(1)
            let side = Int(sqrt(Double(numPatches)))
            spatialShapes = MLXArray([side, side]).expandedDimensions(axis: 0)
            pixelAttentionMask = MLXArray.ones([1, numPatches]).asType(.int32)
        }

        let inputEmbeddings = getInputEmbeddings(
            inputIds: input.text.tokens,
            pixelValues: pixelValues,
            spatialShapes: spatialShapes,
            pixelAttentionMask: pixelAttentionMask
        )

        let result = try withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            let totalPositions = inputEmbeddings.dim(1)
            let processed = try prefill.forEachChunk(total: totalPositions) { range in
                _ = languageModel(
                    nil, cache: cache, inputsEmbeds: inputEmbeddings[0..., range, 0...])
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }

            let result = languageModel(
                nil, cache: cache, inputsEmbeds: inputEmbeddings[0..., processed..., 0...])
            prefill.progress?(totalPositions, totalPositions)
            return result
        }

        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        func transformKey(_ key: String) -> String {
            var key = key

            if key.contains("vision_tower") {
                key =
                    key
                    .replacingOccurrences(of: "model.", with: "")
                    .replacingOccurrences(of: "vision_encoder", with: "encoder")
                    .replacingOccurrences(of: "vision_embeddings", with: "embeddings")
                    .replacingOccurrences(of: "vision_post_layernorm", with: "post_layernorm")
            }

            if key.contains("language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            }

            if key.contains("multi_modal_projector") {
                key = key.replacingOccurrences(
                    of: "model.multi_modal_projector", with: "multi_modal_projector")
            }

            return key
        }

        var sanitizedWeights = [String: MLXArray]()
        for (k, v) in weights {
            let newKey = transformKey(k)

            // Handle conv weight transposition
            var value = v
            if newKey.contains("conv.weight") {
                if v.shape[v.shape.count - 1] > v.dim(1) {
                    value = v.transposed(0, 2, 1)
                }
            }

            sanitizedWeights[newKey] = value
        }

        return sanitizedWeights
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        let textConfig = config.textConfiguration
        return try (0 ..< textConfig.hiddenLayers).map { layerIdx in
            if textConfig.fullAttnIdxs.contains(layerIdx) {
                try makeAttentionKVCache(parameters: parameters)
            } else {
                MambaCache()
            }
        }
    }
}

// MARK: - Configuration

/// Configuration for ``LFM2VL``
public struct LFM2VLConfiguration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let vocabularySize: Int
        private let _normEps: Float?
        public var normEps: Float { _normEps ?? 1e-5 }
        private let _convBias: Bool?
        public var convBias: Bool { _convBias ?? false }
        private let _convLCache: Int?
        public var convLCache: Int { _convLCache ?? 3 }
        private let _blockDim: Int?
        public var blockDim: Int { _blockDim ?? hiddenSize }
        private let _blockFFDim: Int?
        public var blockFFDim: Int { _blockFFDim ?? hiddenSize }
        private let _blockMultipleOf: Int?
        public var blockMultipleOf: Int { _blockMultipleOf ?? 256 }
        private let _blockFFNDimMultiplier: Float?
        public var blockFFNDimMultiplier: Float { _blockFFNDimMultiplier ?? 1.0 }
        private let _blockAutoAdjustFFDim: Bool?
        public var blockAutoAdjustFFDim: Bool { _blockAutoAdjustFFDim ?? true }
        private let _fullAttnIdxs: [Int]?
        private let layerTypes: [String]?
        public var fullAttnIdxs: [Int] {
            if let fullAttnIdxs = _fullAttnIdxs {
                return fullAttnIdxs
            }

            if let layerTypes {
                return layerTypes.enumerated().compactMap { index, layerType in
                    layerType == "full_attention" ? index : nil
                }
            }

            return Array(0 ..< hiddenLayers)
        }
        private let _ropeTheta: Float?
        public var ropeTheta: Float { _ropeTheta ?? 1000000.0 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case _normEps = "norm_eps"
            case _convBias = "conv_bias"
            case _convLCache = "conv_L_cache"
            case _blockDim = "block_dim"
            case _blockFFDim = "block_ff_dim"
            case _blockMultipleOf = "block_multiple_of"
            case _blockFFNDimMultiplier = "block_ffn_dim_multiplier"
            case _blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
            case _fullAttnIdxs = "full_attn_idxs"
            case layerTypes = "layer_types"
            case _ropeTheta = "rope_theta"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        private let _numChannels: Int?
        public var numChannels: Int { _numChannels ?? 3 }
        private let _imageSize: Int?
        public var imageSize: Int { _imageSize ?? 224 }
        private let _patchSize: Int?
        public var patchSize: Int { _patchSize ?? 16 }
        private let _numPatches: Int?
        public var numPatches: Int { _numPatches ?? 256 }
        private let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case _numChannels = "num_channels"
            case _imageSize = "image_size"
            case _patchSize = "patch_size"
            case _numPatches = "num_patches"
            case _layerNormEps = "layer_norm_eps"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _downsampleFactor: Int?
    public var downsampleFactor: Int { _downsampleFactor ?? 2 }
    private let _imageTokenId: Int?
    public var imageTokenIndex: Int { _imageTokenId ?? 396 }
    private let _projectorBias: Bool?
    public var projectorBias: Bool { _projectorBias ?? true }
    private let _projectorHiddenSize: Int?
    public var projectorHiddenSize: Int { _projectorHiddenSize ?? 2560 }
    private let _projectorUseLayernorm: Bool?
    public var projectorUseLayernorm: Bool { _projectorUseLayernorm ?? true }
    private let _visionFeatureLayer: Int?
    /// Which vision encoder layer to use for features. -1 means use all layers (default).
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    private let _doImageSplitting: Bool?
    public var doImageSplitting: Bool { _doImageSplitting ?? true }
    private let _maxImageTokens: Int?
    public var maxImageTokens: Int { _maxImageTokens ?? 256 }
    private let _maxNumPatches: Int?
    public var maxNumPatches: Int { _maxNumPatches ?? 1024 }
    private let _minImageTokens: Int?
    public var minImageTokens: Int { _minImageTokens ?? 64 }
    private let _minTiles: Int?
    public var minTiles: Int { _minTiles ?? 2 }
    private let _useThumbnail: Bool?
    public var useThumbnail: Bool { _useThumbnail ?? false }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _downsampleFactor = "downsample_factor"
        case _imageTokenId = "image_token_id"
        case _projectorBias = "projector_bias"
        case _projectorHiddenSize = "projector_hidden_size"
        case _projectorUseLayernorm = "projector_use_layernorm"
        case _visionFeatureLayer = "vision_feature_layer"
        case _doImageSplitting = "do_image_splitting"
        case _maxImageTokens = "max_image_tokens"
        case _maxNumPatches = "max_num_patches"
        case _minImageTokens = "min_image_tokens"
        case _minTiles = "min_tiles"
        case _useThumbnail = "use_thumbnail"
    }
}

/// Configuration for ``LFM2VLProcessor``
///
/// Defaults mirror the class attributes of `Lfm2VlImageProcessor` in transformers, so a
/// checkpoint that omits a key behaves like the Python reference.
public struct LFM2VLProcessorConfiguration: Codable, Sendable {
    private let _imageMean: [CGFloat]?
    private let _imageStd: [CGFloat]?
    private let _tileSize: Int?
    private let _encoderPatchSize: Int?
    private let _minTiles: Int?
    private let _maxTiles: Int?
    private let _downsampleFactor: Int?
    private let _doImageSplitting: Bool?
    private let _useThumbnail: Bool?
    private let _minImageTokens: Int?
    private let _maxImageTokens: Int?
    private let _maxPixelsTolerance: Double?
    private let _imageToken: String?
    private let _useImageSpecialTokens: Bool?

    // Default values matching LFM2 VL models
    public var imageMean: [CGFloat] {
        _imageMean ?? [0.5, 0.5, 0.5]
    }
    public var imageStd: [CGFloat] {
        _imageStd ?? [0.5, 0.5, 0.5]
    }
    public var tileSize: Int { _tileSize ?? 512 }
    public var encoderPatchSize: Int { _encoderPatchSize ?? 16 }
    public var minTiles: Int { _minTiles ?? 2 }
    public var maxTiles: Int { _maxTiles ?? 10 }
    public var downsampleFactor: Int { _downsampleFactor ?? 2 }
    public var doImageSplitting: Bool { _doImageSplitting ?? true }
    public var useThumbnail: Bool { _useThumbnail ?? true }
    public var minImageTokens: Int { _minImageTokens ?? 64 }
    public var maxImageTokens: Int { _maxImageTokens ?? 256 }
    public var maxPixelsTolerance: Double { _maxPixelsTolerance ?? 2.0 }
    public var imageToken: String { _imageToken ?? "<image>" }
    public var useImageSpecialTokens: Bool { _useImageSpecialTokens ?? true }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }
    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case _imageMean = "image_mean"
        case _imageStd = "image_std"
        case _tileSize = "tile_size"
        case _encoderPatchSize = "encoder_patch_size"
        case _minTiles = "min_tiles"
        case _maxTiles = "max_tiles"
        case _downsampleFactor = "downsample_factor"
        case _doImageSplitting = "do_image_splitting"
        case _useThumbnail = "use_thumbnail"
        case _minImageTokens = "min_image_tokens"
        case _maxImageTokens = "max_image_tokens"
        case _maxPixelsTolerance = "max_pixels_tolerance"
        case _imageToken = "image_token"
        case _useImageSpecialTokens = "use_image_special_tokens"
    }

    private enum ContainerKeys: String, CodingKey {
        case imageProcessor = "image_processor"
    }

    /// The image processor settings sit at the top level of `preprocessor_config.json`
    /// (LFM2-VL) but nested under `image_processor` in `processor_config.json` (LFM2.5-VL).
    /// Decoding only the flat layout would silently fall back to defaults — including
    /// `do_image_splitting`, whose default is the opposite of what LFM2.5-VL declares.
    public init(from decoder: any Decoder) throws {
        let top = try decoder.container(keyedBy: CodingKeys.self)
        let outer = try decoder.container(keyedBy: ContainerKeys.self)
        let image = try? outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .imageProcessor)
        let c = image ?? top

        _imageMean = try c.decodeIfPresent([CGFloat].self, forKey: ._imageMean)
        _imageStd = try c.decodeIfPresent([CGFloat].self, forKey: ._imageStd)
        _tileSize = try c.decodeIfPresent(Int.self, forKey: ._tileSize)
        _encoderPatchSize = try c.decodeIfPresent(Int.self, forKey: ._encoderPatchSize)
        _minTiles = try c.decodeIfPresent(Int.self, forKey: ._minTiles)
        _maxTiles = try c.decodeIfPresent(Int.self, forKey: ._maxTiles)
        _downsampleFactor = try c.decodeIfPresent(Int.self, forKey: ._downsampleFactor)
        _doImageSplitting = try c.decodeIfPresent(Bool.self, forKey: ._doImageSplitting)
        _useThumbnail = try c.decodeIfPresent(Bool.self, forKey: ._useThumbnail)
        _minImageTokens = try c.decodeIfPresent(Int.self, forKey: ._minImageTokens)
        _maxImageTokens = try c.decodeIfPresent(Int.self, forKey: ._maxImageTokens)
        _maxPixelsTolerance = try c.decodeIfPresent(Double.self, forKey: ._maxPixelsTolerance)
        _imageToken = try c.decodeIfPresent(String.self, forKey: ._imageToken)
        // Processor-level flag: never nested inside `image_processor`.
        _useImageSpecialTokens = try top.decodeIfPresent(
            Bool.self, forKey: ._useImageSpecialTokens)
    }
}

// MARK: - Chat conventions

extension LFM2VL {
    public var toolCallFormat: ToolCallFormat? { .lfm2 }
}
