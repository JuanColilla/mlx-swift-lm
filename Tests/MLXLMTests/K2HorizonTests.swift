import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

/// Offline checks for the K2-Horizon port: configuration decoding against
/// the published `config.json` shapes, forward passes on tiny synthetic
/// dense and MoE+MoVA models, the routing rule, the grouped norm, partial
/// RoPE, `sanitize`, and prefill/decode cache consistency. Numeric parity
/// against Python mlx-lm lives in `K2HorizonMacExperiment` (opt-in).
final class K2HorizonTests: XCTestCase {

    // MARK: - Configurations

    /// Verbatim excerpt of `IFM/K2-Horizon-MoVA-36B-A4B/config.json`.
    private static let publishedMoEConfig = """
        {
          "architectures": ["K2HorizonForCausalLM"],
          "attention_bias": false,
          "attention_gate_func": "softplus",
          "bos_token_id": 0,
          "decoder_sparse_step": 1,
          "dtype": "bfloat16",
          "eos_token_id": 1,
          "head_dim": 128,
          "hidden_act": "silu",
          "hidden_size": 2560,
          "intermediate_size": 6144,
          "layernorm_num_groups": 2,
          "max_position_embeddings": 524288,
          "mlp_only_layers": [0, 1, 2],
          "model_type": "k2_horizon",
          "moe_gate_bias": true,
          "moe_intermediate_size": 768,
          "mova_num_experts": 64,
          "mova_num_experts_per_tok": 4,
          "norm_topk_prob": true,
          "num_attention_heads": 32,
          "num_experts": 100,
          "num_experts_per_tok": 8,
          "num_hidden_layers": 48,
          "num_key_value_heads": 8,
          "num_shared_experts": 1,
          "output_router_logits": false,
          "pad_token_id": null,
          "query_key_norm": false,
          "rms_norm_eps": 1e-06,
          "rope_head_dim": 128,
          "rope_parameters": { "rope_theta": 10000000.0, "rope_type": "default" },
          "router_aux_loss_coef": 0.001,
          "router_scaling_factor": 2.5,
          "router_score_func": "sigmoid",
          "sliding_window": null,
          "tie_word_embeddings": false,
          "use_cache": true,
          "use_sliding_window": false,
          "vocab_size": 250624
        }
        """

    /// Verbatim excerpt of `IFM/K2-Horizon-3.7B/config.json`.
    private static let publishedDenseConfig = """
        {
          "architectures": ["K2HorizonForCausalLM"],
          "attention_bias": false,
          "attention_gate_func": null,
          "decoder_sparse_step": 1,
          "head_dim": 128,
          "hidden_size": 2560,
          "intermediate_size": 10240,
          "layernorm_num_groups": 2,
          "max_position_embeddings": 524288,
          "mlp_only_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
          "model_type": "k2_horizon",
          "moe_gate_bias": false,
          "moe_intermediate_size": 0,
          "mova_num_experts": 0,
          "mova_num_experts_per_tok": 0,
          "norm_topk_prob": true,
          "num_attention_heads": 32,
          "num_experts": 0,
          "num_experts_per_tok": 0,
          "num_hidden_layers": 36,
          "num_key_value_heads": 8,
          "num_shared_experts": 0,
          "query_key_norm": false,
          "rms_norm_eps": 1e-06,
          "rope_head_dim": 128,
          "rope_parameters": { "rope_theta": 10000000.0, "rope_type": "default" },
          "router_scaling_factor": 1.0,
          "router_score_func": "sigmoid",
          "tie_word_embeddings": false,
          "vocab_size": 250624
        }
        """

    private func decode(_ json: String) throws -> K2HorizonConfiguration {
        try JSONDecoder().decode(K2HorizonConfiguration.self, from: Data(json.utf8))
    }

    private func tinyDense(
        headDim: Int = 8, ropeHeadDim: Int? = nil, layerNormGroups: Int = 2,
        queryKeyNorm: Bool = false, gate: String? = nil
    ) throws -> K2HorizonConfiguration {
        try decode(
            """
            {
              "model_type": "k2_horizon",
              "vocab_size": 64,
              "hidden_size": 32,
              "intermediate_size": 48,
              "num_hidden_layers": 2,
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": \(headDim),
              "rope_head_dim": \(ropeHeadDim ?? headDim),
              "rope_parameters": { "rope_theta": 10000000.0, "rope_type": "default" },
              "layernorm_num_groups": \(layerNormGroups),
              "query_key_norm": \(queryKeyNorm),
              "attention_gate_func": \(gate.map { "\"\($0)\"" } ?? "null"),
              "num_experts": 0,
              "mova_num_experts": 0,
              "rms_norm_eps": 1e-6,
              "tie_word_embeddings": false
            }
            """)
    }

    private func tinyMoE() throws -> K2HorizonConfiguration {
        try decode(
            """
            {
              "model_type": "k2_horizon",
              "vocab_size": 64,
              "hidden_size": 32,
              "intermediate_size": 48,
              "moe_intermediate_size": 16,
              "num_hidden_layers": 3,
              "mlp_only_layers": [0],
              "num_attention_heads": 4,
              "num_key_value_heads": 2,
              "head_dim": 8,
              "rope_parameters": { "rope_theta": 10000000.0 },
              "layernorm_num_groups": 2,
              "query_key_norm": false,
              "attention_gate_func": "softplus",
              "num_experts": 6,
              "num_experts_per_tok": 2,
              "num_shared_experts": 1,
              "norm_topk_prob": true,
              "moe_gate_bias": true,
              "router_score_func": "sigmoid",
              "router_scaling_factor": 2.5,
              "mova_num_experts": 5,
              "mova_num_experts_per_tok": 2,
              "rms_norm_eps": 1e-6,
              "tie_word_embeddings": false
            }
            """)
    }

    private func randomize(_ model: Module) throws {
        var parameters = [String: MLXArray]()
        for (key, value) in model.parameters().flattened() {
            parameters[key] = MLXRandom.normal(value.shape, scale: 0.2).asType(value.dtype)
        }
        try model.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        eval(model)
    }

    // MARK: - Configuration decoding

    func testPublishedMoEConfigDecodes() throws {
        let config = try decode(Self.publishedMoEConfig)
        try config.validateModelConfiguration()
        XCTAssertEqual(config.ropeTheta, 1e7)
        XCTAssertEqual(config.ropeHeadDim, 128)
        XCTAssertEqual(config.mlpOnlyLayers, [0, 1, 2])
        XCTAssertEqual(config.attentionGate, .softplus)
        XCTAssertTrue(config.moeGateBias)
        XCTAssertEqual(config.routerScore, .sigmoid)
        XCTAssertEqual(config.routerScalingFactor, 2.5)
        XCTAssertEqual(config.layerNormGroups, 2)
        XCTAssertFalse(config.queryKeyNorm)
        XCTAssertFalse(config.isSparseLayer(2))
        XCTAssertTrue(config.isSparseLayer(3))
        XCTAssertTrue(config.usesMoVA(3))
        XCTAssertFalse(config.usesMoVA(0))
        XCTAssertEqual((0 ..< 48).filter(config.isSparseLayer).count, 45)
    }

    func testPublishedDenseConfigDecodes() throws {
        let config = try decode(Self.publishedDenseConfig)
        try config.validateModelConfiguration()
        XCTAssertEqual(config.ropeTheta, 1e7)
        XCTAssertEqual(config.mlpOnlyLayers.count, 36)
        XCTAssertNil(config.attentionGate)
        XCTAssertFalse(config.moeGateBias)
        XCTAssertEqual(config.intermediateSize, 10240)
        XCTAssertFalse((0 ..< 36).contains(where: config.isSparseLayer))
    }

    func testDataclassDefaultsApplyWhenKeysAreMissing() throws {
        let config = try decode(
            """
            {"model_type": "k2_horizon", "vocab_size": 16, "hidden_size": 8,
             "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
             "head_dim": 8, "rope_theta": 5000, "num_experts": 0}
            """)
        XCTAssertTrue(config.queryKeyNorm)
        XCTAssertFalse(config.normTopkProb)
        XCTAssertEqual(config.routerScore, .softmax)
        XCTAssertEqual(config.routerScalingFactor, 1.0)
        XCTAssertEqual(config.layerNormGroups, 1)
        XCTAssertEqual(config.ropeHeadDim, 8)
        XCTAssertEqual(config.ropeTheta, 5000)
        XCTAssertNil(config.attentionGate)
        XCTAssertEqual(config.mlpOnlyLayers, [])
    }

    func testRegisteredInTypeRegistry() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(Self.publishedDenseConfig.utf8), modelType: "k2_horizon")
        XCTAssertTrue(model is K2HorizonModel)
    }

    // MARK: - Forward passes

    func testDenseForwardWithVerifiedRandomWeights() throws {
        let model = K2HorizonModel(try tinyDense())
        try randomize(model)
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.v_proj.weight"))
        XCTAssertFalse(
            keys.contains(where: { $0.contains("gate_proj") && $0.contains("self_attn") }))
        XCTAssertFalse(keys.contains(where: { $0.contains("switch_v") || $0.contains("v_router") }))
        XCTAssertTrue(keys.contains("lm_head.weight"))

        let logits = model(MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5), cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, 64])
        XCTAssertFalse(MLX.isNaN(logits).any().item(Bool.self))
    }

    func testMoEForwardWithVerifiedRandomWeights() throws {
        let model = K2HorizonModel(try tinyMoE())
        try randomize(model)
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("model.layers.0.mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.v_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate.bias"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.shared_experts.up_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.self_attn.switch_v.weight"))
        XCTAssertFalse(keys.contains("model.layers.1.self_attn.switch_v.bias"))
        XCTAssertTrue(keys.contains("model.layers.1.self_attn.v_router.bias"))
        XCTAssertFalse(keys.contains("model.layers.1.self_attn.v_proj.weight"))
        XCTAssertEqual(
            model.parameters().flattened().first {
                $0.0 == "model.layers.1.self_attn.switch_v.weight"
            }?
            .1.shape, [5, 16, 32])
        XCTAssertEqual(
            model.parameters().flattened().first {
                $0.0 == "model.layers.1.mlp.switch_mlp.down_proj.weight"
            }?.1.shape, [6, 32, 16])

        let logits = model(MLXArray([1, 2, 3, 4, 5] as [Int32]).reshaped(1, 5), cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, 64])
        XCTAssertFalse(MLX.isNaN(logits).any().item(Bool.self))
    }

    func testPrefillMatchesIncrementalDecode() throws {
        for config in [try tinyDense(queryKeyNorm: true, gate: "silu"), try tinyMoE()] {
            let model = K2HorizonModel(config)
            try randomize(model)
            let tokens: [Int32] = [3, 9, 27, 4, 12, 40]

            let full = model(MLXArray(tokens).reshaped(1, tokens.count), cache: nil)
            let cache = try model.newCache(parameters: nil)
            var incremental = [MLXArray]()
            incremental.append(model(MLXArray(Array(tokens[..<3])).reshaped(1, 3), cache: cache))
            for token in tokens[3...] {
                incremental.append(model(MLXArray([token]).reshaped(1, 1), cache: cache))
            }
            let stitched = concatenated(incremental, axis: 1)
            eval(full, stitched)
            XCTAssertEqual(stitched.shape, full.shape)
            let difference = MLX.max(MLX.abs(full - stitched)).item(Float.self)
            XCTAssertLessThan(difference, 1e-3, "cache route diverged from full prefill")
        }
    }

    // MARK: - Components

    func testGroupedRMSNormMatchesManualComputation() {
        let norm = K2HorizonGroupedRMSNorm(dimensions: 8, groups: 2, eps: 1e-6)
        let weight: [Float] = [1, 2, 3, 4, 0.5, 0.25, 2, 1]
        norm.update(parameters: ModuleParameters.unflattened(["weight": MLXArray(weight)]))
        let x: [Float] = [1, 2, 3, 4, 10, 20, 30, 40]
        let out = norm(MLXArray(x).reshaped(1, 1, 8)).reshaped(8).asArray(Float.self)

        func rms(_ v: ArraySlice<Float>) -> Float {
            (v.map { $0 * $0 }.reduce(0, +) / Float(v.count) + 1e-6).squareRoot()
        }
        let r0 = rms(x[0 ..< 4])
        let r1 = rms(x[4 ..< 8])
        for i in 0 ..< 8 {
            let expected = x[i] / (i < 4 ? r0 : r1) * weight[i]
            XCTAssertEqual(out[i], expected, accuracy: 1e-4, "feature \(i)")
        }
    }

    func testGroupedRMSNormWithOneGroupIsPlainRMSNorm() {
        let grouped = K2HorizonGroupedRMSNorm(dimensions: 8, groups: 1, eps: 1e-5)
        let plain = RMSNorm(dimensions: 8, eps: 1e-5)
        let x = MLXRandom.normal([2, 3, 8])
        let difference = MLX.max(MLX.abs(grouped(x) - plain(x))).item(Float.self)
        XCTAssertLessThan(difference, 1e-6)
    }

    private func sigmoid(_ v: Float) -> Float { 1 / (1 + expf(-v)) }

    func testRouterBiasSteersSelectionOnly() {
        let logits = MLXArray([2, 1, 0, -1] as [Float]).reshaped(1, 1, 4)
        let bias = MLXArray([0, 0, 1, 0] as [Float])
        let (indices, weights) = k2HorizonRoute(
            logits: logits, selectionBias: bias, score: .sigmoid, topK: 2, normalize: false,
            scalingFactor: 1)
        let idx = indices.reshaped(-1).asArray(Int32.self).map(Int.init)
        let w = weights.reshaped(-1).asArray(Float.self)
        let m = Dictionary(uniqueKeysWithValues: zip(idx, w))
        XCTAssertEqual(Set(m.keys), [0, 2], "the bias must move expert 2 into the top-k")
        XCTAssertEqual(m[0] ?? .nan, sigmoid(2), accuracy: 1e-5)
        XCTAssertEqual(
            m[2] ?? .nan, sigmoid(0), accuracy: 1e-5, "weights come from unbiased scores")
    }

    func testRouterNormalizesAndScales() {
        let logits = MLXArray([2, 1, 0, -1] as [Float]).reshaped(1, 1, 4)
        let (indices, weights) = k2HorizonRoute(
            logits: logits, selectionBias: nil, score: .sigmoid, topK: 2, normalize: true,
            scalingFactor: 2.5)
        let idx = indices.reshaped(-1).asArray(Int32.self).map(Int.init)
        let w = weights.reshaped(-1).asArray(Float.self)
        let m = Dictionary(uniqueKeysWithValues: zip(idx, w))
        XCTAssertEqual(Set(m.keys), [0, 1])
        let denominator = sigmoid(2) + sigmoid(1)
        XCTAssertEqual(m[0] ?? .nan, 2.5 * sigmoid(2) / denominator, accuracy: 1e-5)
        XCTAssertEqual(m[1] ?? .nan, 2.5 * sigmoid(1) / denominator, accuracy: 1e-5)
    }

    func testRouterBatchedRowsMatchSingleRowPath() {
        // Single rows take the fused Metal kernel; batched rows take the
        // argPartition chain. Both must agree.
        let logits = MLXRandom.normal([3, 4, 12])
        let bias = MLXRandom.normal([12])
        let batched = k2HorizonRoute(
            logits: logits, selectionBias: bias, score: .sigmoid, topK: 3, normalize: true,
            scalingFactor: 2.5)
        for b in 0 ..< 3 {
            for l in 0 ..< 4 {
                let single = k2HorizonRoute(
                    logits: logits[b, l].reshaped(1, 1, 12), selectionBias: bias, score: .sigmoid,
                    topK: 3, normalize: true, scalingFactor: 2.5)
                XCTAssertEqual(
                    Set(single.indices.reshaped(-1).asArray(Int32.self)),
                    Set(batched.indices[b, l].asArray(Int32.self)))
                let s = Dictionary(
                    uniqueKeysWithValues: zip(
                        single.indices.reshaped(-1).asArray(Int32.self),
                        single.weights.reshaped(-1).asArray(Float.self)))
                let m = Dictionary(
                    uniqueKeysWithValues: zip(
                        batched.indices[b, l].asArray(Int32.self),
                        batched.weights[b, l].asArray(Float.self)))
                for (k, v) in s {
                    XCTAssertEqual(v, m[k] ?? .nan, accuracy: 1e-5)
                }
            }
        }
    }

    func testBiasFreeRouterLogitsIgnoreTheBias() {
        let router = Linear(64, 3, bias: true)
        router.update(
            parameters: ModuleParameters.unflattened([
                "weight": MLXRandom.normal([3, 64]), "bias": MLXArray([10, 20, 30] as [Float]),
            ]))
        let x = MLXRandom.normal([2, 64])
        let expected = MLX.matmul(x, router.weight.T)
        let plain = k2HorizonRouterLogits(router, x)
        XCTAssertLessThan(MLX.max(MLX.abs(plain - expected)).item(Float.self), 1e-6)

        let quantized = QuantizedLinear(router, groupSize: 32, bits: 8)
        let viaQuantized = k2HorizonRouterLogits(quantized, x)
        let withBias = quantized(x)
        XCTAssertLessThan(
            MLX.max(MLX.abs(viaQuantized - (withBias - quantized.bias!))).item(Float.self), 1e-3)
    }

    func testSoftplusGateMatchesReference() throws {
        let model = K2HorizonModel(try tinyDense(gate: "softplus"))
        let attention = (model.model.layers[0] as! K2HorizonDecoderLayer).selfAttn as! K2HorizonAttention
        try randomize(attention)
        let x = MLXRandom.normal([1, 3, 32])
        let output = MLXArray.ones([1, 3, 4, 8])
        let gated = attention.applyGate(output, input: x)
        let ln2 = Float(M_LN2)
        let raw = attention.gateProj!(x).reshaped(1, 3, 4, 8).asArray(Float.self)
        let got = gated.asArray(Float.self)
        for i in 0 ..< raw.count {
            let expected = log1pf(expf(ln2 * raw[i])) / ln2
            XCTAssertEqual(got[i], expected, accuracy: 1e-4)
        }
    }

    func testPartialRoPEGeneralPathReducesToFullRoPE() throws {
        let full = K2HorizonAttention(try tinyDense(headDim: 8), layerIdx: 0)
        let x = MLXRandom.normal([1, 4, 5, 8])
        let reference = full.applyRope(x, offset: .scalar(3))
        // Force the general path with rope_head_dim == head_dim by building
        // the pairs manually.
        let pairs = x.reshaped(1, 4, 5, 2, 4)
        let rotating = pairs[.ellipsis, ..<4].reshaped(1, 4, 5, 8)
        let rotated = applyRotaryPosition(full.rope, to: rotating, offset: .scalar(3))
            .reshaped(1, 4, 5, 2, 4)
        let general = rotated.reshaped(1, 4, 5, 8)
        XCTAssertLessThan(MLX.max(MLX.abs(general - reference)).item(Float.self), 1e-6)
    }

    func testPartialRoPEKeepsTrailingPairsUntouched() throws {
        let attention = K2HorizonAttention(try tinyDense(headDim: 8, ropeHeadDim: 4), layerIdx: 0)
        let x = MLXRandom.normal([1, 2, 3, 8])
        let y = attention.applyRope(x, offset: .scalar(7))
        XCTAssertEqual(y.shape, x.shape)
        // Pairs (i, i + 4) for i >= 2 pass through unchanged.
        let xp = x.reshaped(1, 2, 3, 2, 4)[.ellipsis, 2...]
        let yp = y.reshaped(1, 2, 3, 2, 4)[.ellipsis, 2...]
        XCTAssertLessThan(MLX.max(MLX.abs(xp - yp)).item(Float.self), 1e-7)
        // The rotated pairs at position 7 differ from the input.
        let xr = x.reshaped(1, 2, 3, 2, 4)[.ellipsis, ..<2]
        let yr = y.reshaped(1, 2, 3, 2, 4)[.ellipsis, ..<2]
        XCTAssertGreaterThan(MLX.max(MLX.abs(xr - yr)).item(Float.self), 1e-3)
        // Rotation preserves the norm of each pair.
        let xn = MLX.sqrt(MLX.square(xr).sum(axis: -2))
        let yn = MLX.sqrt(MLX.square(yr).sum(axis: -2))
        XCTAssertLessThan(MLX.max(MLX.abs(xn - yn)).item(Float.self), 1e-4)
    }

    func testMoVASortedGatherMatchesPerTokenPath() throws {
        // 20 tokens x top-2 = 40 assignments stays on the unsorted path;
        // 40 tokens x top-2 = 80 crosses the gatherSort threshold (64).
        let model = K2HorizonModel(try tinyMoE())
        try randomize(model)
        let attention = (model.model.layers[1] as! K2HorizonDecoderLayer).selfAttn as! K2HorizonAttention
        let x = MLXRandom.normal([1, 40, 32])
        let batched = attention.routedValues(x)
        let perToken = concatenated(
            (0 ..< 40).map { attention.routedValues(x[0..., $0 ..< ($0 + 1), 0...]) }, axis: 1)
        eval(batched, perToken)
        XCTAssertEqual(batched.shape, [1, 40, 16])
        XCTAssertLessThan(MLX.max(MLX.abs(batched - perToken)).item(Float.self), 1e-4)

        let block =
            (model.model.layers[1] as! K2HorizonDecoderLayer).mlp as! K2HorizonSparseMoEBlock
        let moeBatched = block(x)
        let moePerToken = concatenated(
            (0 ..< 40).map { block(x[0..., $0 ..< ($0 + 1), 0...]) }, axis: 1)
        XCTAssertLessThan(MLX.max(MLX.abs(moeBatched - moePerToken)).item(Float.self), 1e-4)
    }

    func testFullPrefillMatchesTokenByTokenDecodeOnMoE() throws {
        let model = K2HorizonModel(try tinyMoE())
        try randomize(model)
        let tokens = (0 ..< 40).map { Int32(($0 * 7) % 64) }
        let full = model(MLXArray(tokens).reshaped(1, 40), cache: nil)
        let cache = try model.newCache(parameters: nil)
        let stepwise = concatenated(
            tokens.map { model(MLXArray([$0]).reshaped(1, 1), cache: cache) }, axis: 1)
        eval(full, stepwise)
        XCTAssertLessThan(MLX.max(MLX.abs(full - stepwise)).item(Float.self), 1e-3)
    }

    // MARK: - sanitize

    func testSanitizeStacksHuggingFaceExpertsAndDropsRotaryBuffers() throws {
        let config = try tinyMoE()
        let model = K2HorizonModel(config)
        var weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "model.layers.1.mlp.switch_mlp.up_proj.weight": MLXArray.zeros([6, 16, 32]),
        ]
        for e in 0 ..< 6 {
            weights["model.layers.2.mlp.experts.\(e).gate_proj.weight"] = MLXArray.zeros([16, 32])
            weights["model.layers.2.mlp.experts.\(e).up_proj.weight"] = MLXArray.zeros([16, 32])
            weights["model.layers.2.mlp.experts.\(e).down_proj.weight"] = MLXArray.zeros([32, 16])
        }
        for e in 0 ..< 5 {
            weights["model.layers.2.self_attn.v_experts.\(e).weight"] = MLXArray.zeros([16, 32])
        }
        let sanitized = model.sanitize(weights: weights)

        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape, [6, 16, 32])
        XCTAssertEqual(
            sanitized["model.layers.2.mlp.switch_mlp.gate_proj.weight"]?.shape, [6, 16, 32])
        XCTAssertEqual(
            sanitized["model.layers.2.mlp.switch_mlp.up_proj.weight"]?.shape, [6, 16, 32])
        XCTAssertEqual(
            sanitized["model.layers.2.mlp.switch_mlp.down_proj.weight"]?.shape, [6, 32, 16])
        XCTAssertEqual(sanitized["model.layers.2.self_attn.switch_v.weight"]?.shape, [5, 16, 32])
        XCTAssertFalse(
            sanitized.keys.contains { $0.contains(".experts.") || $0.contains("v_experts") })
    }

    func testSanitizeLeavesStackedLayoutAlone() throws {
        let model = K2HorizonModel(try tinyMoE())
        let stacked: [String: MLXArray] = [
            "model.layers.1.mlp.switch_mlp.gate_proj.weight": MLXArray.zeros([6, 16, 32]),
            "model.layers.1.mlp.switch_mlp.gate_proj.scales": MLXArray.zeros([6, 16, 1]),
            "model.layers.1.self_attn.switch_v.weight": MLXArray.zeros([5, 16, 32]),
            "model.layers.1.mlp.gate.bias": MLXArray.zeros([6]),
            "lm_head.weight": MLXArray.zeros([64, 32]),
        ]
        let sanitized = model.sanitize(weights: stacked)
        XCTAssertEqual(Set(sanitized.keys), Set(stacked.keys))
    }
}
