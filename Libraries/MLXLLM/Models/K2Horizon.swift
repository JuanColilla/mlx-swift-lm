//
//  K2Horizon.swift
//  mlx-swift-lm
//
//  Port of the IFM K2-Horizon family (`model_type: k2_horizon`): dense
//  checkpoints (3.7B, 7B) and the MoE + MoVA checkpoint (36B-A4B).
//
//  Reference: `modeling_k2_horizon.py` shipped with the Hugging Face
//  checkpoints. Llama-like decoder with three additions: grouped RMSNorm,
//  an optional per-head attention gate, and, on sparse layers, a sigmoid
//  router whose bias is used for selection only. MoVA replaces `v_proj`
//  with routed value experts.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

public struct K2HorizonConfiguration: Codable, Sendable {
    public enum AttentionGate: String, Codable, Sendable {
        case softplus
        case silu
    }

    public enum RouterScore: String, Codable, Sendable {
        case sigmoid
        case softmax
    }

    public var modelType: String = "k2_horizon"
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var ropeHeadDim: Int
    public var ropeTheta: Float
    public var rmsNormEps: Float
    public var maxPositionEmbeddings: Int
    public var tieWordEmbeddings: Bool
    public var attentionBias: Bool
    public var queryKeyNorm: Bool
    public var layerNormGroups: Int
    public var attentionGate: AttentionGate?
    public var decoderSparseStep: Int
    public var mlpOnlyLayers: [Int]
    public var moeIntermediateSize: Int
    public var numExperts: Int
    public var numExpertsPerToken: Int
    public var numSharedExperts: Int
    public var normTopkProb: Bool
    public var moeGateBias: Bool
    public var routerScore: RouterScore
    public var routerScalingFactor: Float
    public var movaNumExperts: Int
    public var movaNumExpertsPerToken: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case ropeHeadDim = "rope_head_dim"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case queryKeyNorm = "query_key_norm"
        case layerNormGroups = "layernorm_num_groups"
        case attentionGate = "attention_gate_func"
        case decoderSparseStep = "decoder_sparse_step"
        case mlpOnlyLayers = "mlp_only_layers"
        case moeIntermediateSize = "moe_intermediate_size"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case normTopkProb = "norm_topk_prob"
        case moeGateBias = "moe_gate_bias"
        case routerScore = "router_score_func"
        case routerScalingFactor = "router_scaling_factor"
        case movaNumExperts = "mova_num_experts"
        case movaNumExpertsPerToken = "mova_num_experts_per_tok"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "k2_horizon"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        ropeHeadDim = try c.decodeIfPresent(Int.self, forKey: .ropeHeadDim) ?? headDim
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        queryKeyNorm = try c.decodeIfPresent(Bool.self, forKey: .queryKeyNorm) ?? true
        layerNormGroups = try c.decodeIfPresent(Int.self, forKey: .layerNormGroups) ?? 1
        attentionGate = try c.decodeIfPresent(AttentionGate.self, forKey: .attentionGate)
        decoderSparseStep = try c.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 768
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 128
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
        numSharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? false
        moeGateBias = try c.decodeIfPresent(Bool.self, forKey: .moeGateBias) ?? false
        routerScore = try c.decodeIfPresent(RouterScore.self, forKey: .routerScore) ?? .softmax
        routerScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routerScalingFactor) ?? 1.0
        movaNumExperts = try c.decodeIfPresent(Int.self, forKey: .movaNumExperts) ?? 0
        movaNumExpertsPerToken =
            try c.decodeIfPresent(Int.self, forKey: .movaNumExpertsPerToken) ?? 0

        // Transformers 5 nests the base under `rope_parameters`; older
        // configs keep a top-level `rope_theta`.
        let ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        let flatTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        ropeTheta = ropeParameters?["rope_theta"]?.asFloat() ?? flatTheta
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(ropeHeadDim, forKey: .ropeHeadDim)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(queryKeyNorm, forKey: .queryKeyNorm)
        try c.encode(layerNormGroups, forKey: .layerNormGroups)
        try c.encodeIfPresent(attentionGate, forKey: .attentionGate)
        try c.encode(decoderSparseStep, forKey: .decoderSparseStep)
        try c.encode(mlpOnlyLayers, forKey: .mlpOnlyLayers)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(numExperts, forKey: .numExperts)
        try c.encode(numExpertsPerToken, forKey: .numExpertsPerToken)
        try c.encode(numSharedExperts, forKey: .numSharedExperts)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(moeGateBias, forKey: .moeGateBias)
        try c.encode(routerScore, forKey: .routerScore)
        try c.encode(routerScalingFactor, forKey: .routerScalingFactor)
        try c.encode(movaNumExperts, forKey: .movaNumExperts)
        try c.encode(movaNumExpertsPerToken, forKey: .movaNumExpertsPerToken)
    }

    /// Mirrors `K2HorizonDecoderLayer.__init__`: sparse unless listed in
    /// `mlp_only_layers`, there are no experts, or the sparse step skips it.
    public func isSparseLayer(_ index: Int) -> Bool {
        !mlpOnlyLayers.contains(index) && numExperts > 0
            && (index + 1).isMultiple(of: decoderSparseStep)
    }

    public func usesMoVA(_ index: Int) -> Bool {
        isSparseLayer(index) && movaNumExperts > 0
    }
}

public struct K2HorizonConfigurationError: LocalizedError, Sendable {
    public let message: String
    public var errorDescription: String? { "K2HorizonConfiguration: \(message)" }
}

extension K2HorizonConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard hiddenSize.isMultiple(of: layerNormGroups) else {
            throw K2HorizonConfigurationError(
                message:
                    "hidden_size (\(hiddenSize)) must divide by layernorm_num_groups (\(layerNormGroups))"
            )
        }
        guard ropeHeadDim.isMultiple(of: 2), ropeHeadDim <= headDim else {
            throw K2HorizonConfigurationError(
                message:
                    "rope_head_dim (\(ropeHeadDim)) must be even and at most head_dim (\(headDim))"
            )
        }
        guard decoderSparseStep > 0 else {
            throw K2HorizonConfigurationError(message: "decoder_sparse_step must be positive")
        }
        if numExperts > 0, numExpertsPerToken < 1 || numExpertsPerToken > numExperts {
            throw K2HorizonConfigurationError(message: "num_experts_per_tok out of range")
        }
        if movaNumExperts > 0,
            movaNumExpertsPerToken < 1 || movaNumExpertsPerToken > movaNumExperts
        {
            throw K2HorizonConfigurationError(message: "mova_num_experts_per_tok out of range")
        }
    }
}

// MARK: - Grouped RMSNorm

/// RMSNorm whose variance is taken per group of `hidden / groups` features
/// with one weight vector over the full width (`K2HorizonRMSNorm`).
public class K2HorizonGroupedRMSNorm: Module, UnaryLayer {
    @ParameterInfo(key: "weight") public var weight: MLXArray

    public let groups: Int
    public let eps: Float

    public init(dimensions: Int, groups: Int, eps: Float) {
        precondition(dimensions.isMultiple(of: groups), "dimensions must divide into groups")
        self.groups = groups
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if groups == 1 {
            return MLXFast.rmsNorm(x, weight: weight, eps: eps)
        }
        let shape = x.shape
        let width = shape[shape.count - 1]
        var h = x.asType(.float32)
            .reshaped(Array(shape.dropLast()) + [groups, width / groups])
        let variance = MLX.square(h).mean(axis: -1, keepDims: true)
        h = (h * MLX.rsqrt(variance + eps)).reshaped(shape)
        return (weight.asType(.float32) * h).asType(x.dtype)
    }
}

// MARK: - Routing

/// Bias-free router logits: the checkpoint stores the router as a `Linear`
/// with bias, but the bias only steers selection (see `k2HorizonRoute`).
func k2HorizonRouterLogits(_ router: Linear, _ x: MLXArray) -> MLXArray {
    if let quantized = router as? QuantizedLinear {
        guard quantized.globalScale == nil else {
            // NVFP4 needs the global-scale post-processing; recover the
            // bias-free product from the full forward instead.
            var out = quantized(x)
            if let bias = quantized.bias { out = out - bias }
            return out
        }
        return quantizedMM(
            x, quantized.weight, scales: quantized.scales, biases: quantized.biases,
            transpose: true, groupSize: quantized.groupSize, bits: quantized.bits,
            mode: quantized.mode)
    }
    return MLX.matmul(x, router.weight.T)
}

/// `calc_router_weights` / `K2HorizonSparseMoeBlock.forward`: scores in
/// float32, bias added only to the selection copy, weights taken from the
/// unbiased scores, optionally normalized, then scaled.
func k2HorizonRoute(
    logits: MLXArray, selectionBias: MLXArray?, score: K2HorizonConfiguration.RouterScore,
    topK: Int, normalize: Bool, scalingFactor: Float
) -> (indices: MLXArray, weights: MLXArray) {
    let scores: MLXArray
    switch score {
    case .sigmoid: scores = MLX.sigmoid(logits.asType(.float32))
    case .softmax: scores = MLX.softmax(logits.asType(.float32), axis: -1, precise: true)
    }
    let selection = selectionBias.map { scores + $0.asType(.float32) } ?? scores

    // Ascending winner order, the same `argpartition(kth=-k)[..., -k:]`
    // the Python port uses, so expert outputs are reduced in the same order.
    var indices: MLXArray
    var weights: MLXArray
    if supportsFusedRouterTopK(selection, k: topK) {
        (indices, weights) = fusedRouterTopK(
            selection: selection, values: scores, k: topK, normalize: normalize,
            order: .ascending)
    } else {
        let kth = selection.dim(-1) - topK
        indices = MLX.argPartition(selection, kth: kth, axis: -1)[.ellipsis, kth...]
        weights = MLX.takeAlong(scores, indices, axis: -1)
        if normalize {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
    }
    if scalingFactor != 1 {
        weights = weights * scalingFactor
    }
    return (indices, weights)
}

// MARK: - Attention

/// Shared by the resident attention and the streamed one of
/// `K2HorizonStreamed.swift`, which differ only in where the routed values
/// come from.
///
/// Full RoPE when `rope_head_dim == head_dim`; otherwise rotate the pairs
/// `(i, i + head_dim / 2)` for `i < rope_head_dim / 2` and pass the rest
/// through, which is what the split/interleave dance in the reference amounts
/// to.
func k2HorizonApplyRope(
    _ rope: RoPE, to x: MLXArray, headDim: Int, ropeHeadDim: Int, offset: RoPEOffset?
) -> MLXArray {
    if ropeHeadDim == headDim {
        return applyRotaryPosition(rope, to: x, offset: offset)
    }
    let (B, H, L) = (x.dim(0), x.dim(1), x.dim(2))
    let half = headDim / 2
    let rotatingHalf = ropeHeadDim / 2
    let pairs = x.reshaped(B, H, L, 2, half)
    let rotating = pairs[.ellipsis, ..<rotatingHalf].reshaped(B, H, L, ropeHeadDim)
    let passthrough = pairs[.ellipsis, rotatingHalf...]
    let rotated = applyRotaryPosition(rope, to: rotating, offset: offset)
        .reshaped(B, H, L, 2, rotatingHalf)
    return concatenated([rotated, passthrough], axis: -1).reshaped(B, H, L, headDim)
}

/// The optional per-head output gate.
func k2HorizonApplyGate(
    _ gateProj: Linear?, function: K2HorizonConfiguration.AttentionGate?,
    output: MLXArray, input x: MLXArray, heads: Int, headDim: Int
) -> MLXArray {
    guard let gateProj, let function else { return output }
    var gate = gateProj(x).reshaped(x.dim(0), x.dim(1), heads, headDim)
    switch function {
    case .silu:
        gate = MLXNN.silu(gate)
    case .softplus:
        // `F.softplus(gate, beta=ln 2)`
        let beta = Float(M_LN2)
        gate = MLX.logAddExp(gate * beta, 0) / beta
    }
    return output * gate
}

/// Queries, keys and values (already projected, `[B, L, heads * headDim]`)
/// through RoPE, the cache, attention, the gate and the output projection.
func k2HorizonAttend(
    queries: MLXArray, keys: MLXArray, values: MLXArray, input x: MLXArray,
    heads: Int, kvHeads: Int, headDim: Int, ropeHeadDim: Int, scale: Float, rope: RoPE,
    gateProj: Linear?, gateFunction: K2HorizonConfiguration.AttentionGate?, oProj: Linear,
    mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
) -> MLXArray {
    let (B, L) = (x.dim(0), x.dim(1))
    var queries = queries.reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
    var keys = keys.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
    let values = values.reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

    let offset = cache?.ropeOffset
    queries = k2HorizonApplyRope(
        rope, to: queries, headDim: headDim, ropeHeadDim: ropeHeadDim, offset: offset)
    keys = k2HorizonApplyRope(
        rope, to: keys, headDim: headDim, ropeHeadDim: ropeHeadDim, offset: offset)

    var output = attentionWithCacheUpdate(
        queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
    )
    .transposed(0, 2, 1, 3)

    output = k2HorizonApplyGate(
        gateProj, function: gateFunction, output: output, input: x, heads: heads,
        headDim: headDim)
    return oProj(output.reshaped(B, L, -1))
}

/// What `K2HorizonDecoderLayer` holds as `self_attn`: the resident attention
/// or, on a streamed MoVA layer, the one that reads its value experts from
/// disk.
public protocol K2HorizonAttentionLayer {
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray
}

public class K2HorizonAttention: Module, K2HorizonAttentionLayer {
    let args: K2HorizonConfiguration
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let ropeHeadDim: Int
    let scale: Float
    let usesMoVA: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: K2HorizonGroupedRMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: K2HorizonGroupedRMSNorm?
    @ModuleInfo(key: "v_router") var vRouter: Linear?
    @ModuleInfo(key: "switch_v") var switchV: SwitchLinear?

    let rope: RoPE

    public init(_ args: K2HorizonConfiguration, layerIdx: Int) {
        self.args = args
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.ropeHeadDim = args.ropeHeadDim
        self.scale = pow(Float(headDim), -0.5)
        self.usesMoVA = args.usesMoVA(layerIdx)

        let dim = args.hiddenSize
        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        if usesMoVA {
            _vRouter.wrappedValue = Linear(dim, args.movaNumExperts, bias: args.moeGateBias)
            _switchV.wrappedValue = SwitchLinear(
                inputDims: dim, outputDims: kvHeads * headDim, numExperts: args.movaNumExperts,
                bias: false)
        } else {
            _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        }

        if args.attentionGate != nil {
            _gateProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        }

        if args.queryKeyNorm {
            _qNorm.wrappedValue = K2HorizonGroupedRMSNorm(
                dimensions: heads * headDim, groups: heads, eps: args.rmsNormEps)
            _kNorm.wrappedValue = K2HorizonGroupedRMSNorm(
                dimensions: kvHeads * headDim, groups: kvHeads, eps: args.rmsNormEps)
        }

        self.rope = RoPE(dimensions: ropeHeadDim, traditional: false, base: args.ropeTheta)
    }

    /// MoVA values: top-k value experts per token, `silu` on each expert
    /// output, weighted by the router.
    func routedValues(_ x: MLXArray) -> MLXArray {
        guard let vRouter, let switchV else { fatalError("routedValues on a non-MoVA layer") }
        let (indices, weights) = k2HorizonRoute(
            logits: k2HorizonRouterLogits(vRouter, x), selectionBias: vRouter.bias,
            score: args.routerScore, topK: args.movaNumExpertsPerToken,
            normalize: args.movaNumExpertsPerToken > 1,
            scalingFactor: args.routerScalingFactor)

        // Same gather layout as `SwitchGLU`: one expert row per assignment.
        var h = MLX.expandedDimensions(x, axes: [-2, -3])
        let doSort = indices.size >= 64
        var idx = indices
        var inverseOrder = MLXArray()
        if doSort {
            (h, idx, inverseOrder) = gatherSort(x: h, indices: indices)
        }
        var y = switchV(h, idx, sortedIndices: doSort)
        if doSort {
            y = scatterUnsort(x: y, invOrder: inverseOrder, shape: indices.shape)
        }
        y = MLXNN.silu(MLX.squeezed(y, axis: -2))
        return weightedExpertSum(y, weights.asType(x.dtype))
    }

    func applyRope(_ x: MLXArray, offset: RoPEOffset?) -> MLXArray {
        k2HorizonApplyRope(
            rope, to: x, headDim: headDim, ropeHeadDim: ropeHeadDim, offset: offset)
    }

    func applyGate(_ output: MLXArray, input x: MLXArray) -> MLXArray {
        k2HorizonApplyGate(
            gateProj, function: args.attentionGate, output: output, input: x, heads: heads,
            headDim: headDim)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var queries = qProj(x)
        var keys = kProj(x)
        if let qNorm { queries = qNorm(queries) }
        if let kNorm { keys = kNorm(keys) }
        let values = usesMoVA ? routedValues(x) : vProj!(x)

        return k2HorizonAttend(
            queries: queries, keys: keys, values: values, input: x,
            heads: heads, kvHeads: kvHeads, headDim: headDim, ropeHeadDim: ropeHeadDim,
            scale: scale, rope: rope, gateProj: gateProj, gateFunction: args.attentionGate,
            oProj: oProj, mask: mask, cache: cache)
    }
}

// MARK: - Feed-forward

public class K2HorizonMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    public init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(compiledSiluProduct(gateProj(x), upProj(x)))
    }
}

public class K2HorizonSparseMoEBlock: Module, UnaryLayer {
    let args: K2HorizonConfiguration

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: K2HorizonMLP?

    public init(_ args: K2HorizonConfiguration) {
        self.args = args
        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: args.moeGateBias)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts, bias: false)
        if args.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = K2HorizonMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.moeIntermediateSize * args.numSharedExperts)
        }
    }

    func route(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        k2HorizonRoute(
            logits: k2HorizonRouterLogits(gate, x), selectionBias: gate.bias,
            score: args.routerScore, topK: args.numExpertsPerToken,
            normalize: args.normTopkProb, scalingFactor: args.routerScalingFactor)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = route(x)
        var y = weightedExpertSum(switchMLP(x, indices), weights.asType(x.dtype))
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

// MARK: - Decoder

public class K2HorizonDecoderLayer: Module, TransformerLayer {
    @ModuleInfo(key: "self_attn") var selfAttn: Module & K2HorizonAttentionLayer
    @ModuleInfo(key: "mlp") var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: K2HorizonGroupedRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: K2HorizonGroupedRMSNorm

    /// FORK(JuanColilla): R-56 — whether this layer reads its routed experts
    /// (MLP, and value experts on a MoVA layer) from disk.
    public let usesStreamedExperts: Bool

    public init(_ args: K2HorizonConfiguration, layerIdx: Int) {
        // FORK(JuanColilla): R-56 — streaming is chosen at construction, never
        // by a flag on a live module. Only sparse layers stream; the dense
        // `mlp_only_layers` have no experts and are built as always. The
        // value experts stream only if the index actually has them, which
        // a `switch_v` checkpoint guarantees.
        if args.isSparseLayer(layerIdx), let session = ExpertStreaming.activeSession {
            if args.usesMoVA(layerIdx), session.bank(for: .value) != nil {
                _selfAttn.wrappedValue = K2HorizonStreamedAttention(
                    args, layerIdx: layerIdx, session: session)
            } else {
                _selfAttn.wrappedValue = K2HorizonAttention(args, layerIdx: layerIdx)
            }
            _mlp.wrappedValue = K2HorizonStreamedSparseMoEBlock(
                args, layerIndex: layerIdx, session: session)
            self.usesStreamedExperts = true
        } else if args.isSparseLayer(layerIdx) {
            _selfAttn.wrappedValue = K2HorizonAttention(args, layerIdx: layerIdx)
            _mlp.wrappedValue = K2HorizonSparseMoEBlock(args)
            self.usesStreamedExperts = false
        } else {
            _selfAttn.wrappedValue = K2HorizonAttention(args, layerIdx: layerIdx)
            self.usesStreamedExperts = false
            _mlp.wrappedValue = K2HorizonMLP(
                dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        }
        _inputLayerNorm.wrappedValue = K2HorizonGroupedRMSNorm(
            dimensions: args.hiddenSize, groups: args.layerNormGroups, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = K2HorizonGroupedRMSNorm(
            dimensions: args.hiddenSize, groups: args.layerNormGroups, eps: args.rmsNormEps)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class K2HorizonModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "norm") var norm: K2HorizonGroupedRMSNorm

    public var layers: [TransformerLayer]

    public init(_ args: K2HorizonConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { K2HorizonDecoderLayer(args, layerIdx: $0) }
        _norm.wrappedValue = K2HorizonGroupedRMSNorm(
            dimensions: args.hiddenSize, groups: args.layerNormGroups, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

public class K2HorizonModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: K2HorizonConfiguration

    public let model: K2HorizonModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: K2HorizonConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = K2HorizonModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Accepts the stacked MLX layout as-is and folds the per-expert
    /// Hugging Face layout (`mlp.experts.E.*`, `self_attn.v_experts.E.*`)
    /// into `switch_mlp` / `switch_v`.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        sanitized = filterLMHeadWeights(
            from: sanitized, tiedWordEmbeddings: configuration.tieWordEmbeddings)

        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer)"
            for name in ["gate_proj", "up_proj", "down_proj"] {
                stackExperts(
                    &sanitized, count: configuration.numExperts,
                    source: { "\(prefix).mlp.experts.\($0).\(name)" },
                    destination: "\(prefix).mlp.switch_mlp.\(name)")
            }
            stackExperts(
                &sanitized, count: configuration.movaNumExperts,
                source: { "\(prefix).self_attn.v_experts.\($0)" },
                destination: "\(prefix).self_attn.switch_v")
        }
        return sanitized
    }

    /// Stacks `source(e).<suffix>` for every expert into
    /// `destination.<suffix>`; a no-op when the layout is already stacked.
    private func stackExperts(
        _ weights: inout [String: MLXArray], count: Int,
        source: (Int) -> String, destination: String
    ) {
        guard count > 0 else { return }
        for suffix in ["weight", "scales", "biases"] {
            let keys = (0 ..< count).map { "\(source($0)).\(suffix)" }
            let stack = keys.compactMap { weights[$0] }
            guard stack.count == count else { continue }
            for key in keys {
                weights.removeValue(forKey: key)
            }
            weights["\(destination).\(suffix)"] = MLX.stacked(stack)
        }
    }
}

extension K2HorizonModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
