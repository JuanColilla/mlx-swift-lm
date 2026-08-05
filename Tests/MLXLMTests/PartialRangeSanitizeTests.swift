import Foundation
import MLX
import Testing

@testable import MLXLLM

@Suite struct PartialRangeSanitizeTests {
    /// Simula un shard que solo tiene las capas 4..<8 de un modelo de 12
    /// capas con 2 expertos, sin incluir la capa 0.
    @Test func deepseekV3SanitizeHandlesRangeWithoutLayerZero() throws {
        let model = DeepseekV3Model(try Self.deepseekV3Configuration())
        let sanitized = model.sanitize(weights: Self.partialExpertWeights())

        #expect(sanitized["model.layers.5.mlp.switch_mlp.gate_proj.weight"] != nil)
    }

    @Test func qwen3MoESanitizeHandlesRangeWithoutLayerZero() throws {
        let model = Qwen3MoEModel(try Self.qwen3MoEConfiguration())
        let sanitized = model.sanitize(weights: Self.partialExpertWeights())

        #expect(sanitized["model.layers.5.mlp.switch_mlp.gate_proj.weight"] != nil)
    }

    @Test func qwen3NextSanitizeHandlesRangeWithoutLayerZero() throws {
        let model = Qwen3NextModel(try Self.qwen3NextConfiguration())
        let sanitized = model.sanitize(weights: Self.partialExpertWeights())

        #expect(sanitized["model.layers.5.mlp.switch_mlp.gate_proj.weight"] != nil)
    }

    @Test func deepseekV3SanitizeSkipsIncompleteExpertLayer() throws {
        let model = DeepseekV3Model(try Self.deepseekV3Configuration())
        let sanitized = model.sanitize(weights: Self.partialExpertWeightsWithIncompleteLayer())

        #expect(sanitized["model.layers.5.mlp.switch_mlp.gate_proj.weight"] == nil)
        #expect(sanitized["model.layers.6.mlp.switch_mlp.gate_proj.weight"] != nil)
    }

    @Test func qwen3MoESanitizeSkipsIncompleteExpertLayer() throws {
        let model = Qwen3MoEModel(try Self.qwen3MoEConfiguration())
        let sanitized = model.sanitize(weights: Self.partialExpertWeightsWithIncompleteLayer())

        #expect(sanitized["model.layers.5.mlp.switch_mlp.gate_proj.weight"] == nil)
        #expect(sanitized["model.layers.6.mlp.switch_mlp.gate_proj.weight"] != nil)
    }

    @Test func qwen3NextSanitizeSkipsIncompleteExpertLayer() throws {
        let model = Qwen3NextModel(try Self.qwen3NextConfiguration())
        let sanitized = model.sanitize(weights: Self.partialExpertWeightsWithIncompleteLayer())

        #expect(sanitized["model.layers.5.mlp.switch_mlp.gate_proj.weight"] == nil)
        #expect(sanitized["model.layers.6.mlp.switch_mlp.gate_proj.weight"] != nil)
    }

    @Test func llamaSanitizePreservesPartialLayerRange() {
        let model = LlamaModel(Self.llamaConfiguration())
        let weights = Self.partialLayerWeights(suffix: "self_attn.q_proj.weight")
        let sanitized = model.sanitize(weights: weights)

        #expect(Set(sanitized.keys) == Set(weights.keys))
        for layer in 4 ..< 8 {
            #expect(sanitized["model.layers.\(layer).self_attn.q_proj.weight"] != nil)
        }
    }

    @Test func qwen3SanitizePreservesPartialLayerRange() throws {
        let model = Qwen3Model(try Self.qwen3Configuration())
        let weights = Self.partialLayerWeights(suffix: "self_attn.q_proj.weight")
        let sanitized = model.sanitize(weights: weights)

        #expect(Set(sanitized.keys) == Set(weights.keys))
        for layer in 4 ..< 8 {
            #expect(sanitized["model.layers.\(layer).self_attn.q_proj.weight"] != nil)
        }
    }

    @Test func lfm2SanitizeTransformsPartialLayerRange() throws {
        let model = LFM2Model(try Self.lfm2Configuration())
        let weights = Self.partialLFM2Weights()
        let sanitized = model.sanitize(weights: weights)

        for layer in 4 ..< 8 {
            let key = "model.layers.\(layer).conv.conv.weight"
            #expect(sanitized[key]?.shape == [4, 2, 1])
        }
    }

    @Test func gptOSSSanitizeTransformsPartialLayerRange() throws {
        let model = GPTOSSModel(try Self.gptOSSConfiguration())
        let sanitized = model.sanitize(weights: Self.partialGPTOSSWeights())

        for layer in 4 ..< 8 {
            #expect(sanitized["model.layers.\(layer).mlp.experts.gate_proj.weight"] != nil)
            #expect(sanitized["model.layers.\(layer).mlp.experts.up_proj.weight"] != nil)
            #expect(sanitized["model.layers.\(layer).mlp.experts.down_proj.weight"] != nil)
        }
    }

    @Test func glm4MoELiteSanitizeFusesPartialLayerRange() throws {
        let model = GLM4MoELiteModel(try Self.glm4MoELiteConfiguration())
        let sanitized = model.sanitize(weights: Self.partialExpertWeights())

        for layer in 4 ..< 8 {
            #expect(sanitized["model.layers.\(layer).mlp.switch_mlp.gate_proj.weight"] != nil)
        }
    }

    private static func partialExpertWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for layer in 4 ..< 8 {
            for expert in 0 ..< 2 {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    weights[
                        "model.layers.\(layer).mlp.experts.\(expert).\(projection).weight"
                    ] = MLXArray.zeros([4, 4])
                }
            }
        }
        return weights
    }

    private static func partialExpertWeightsWithIncompleteLayer() -> [String: MLXArray] {
        var weights = partialExpertWeights()
        weights["model.layers.5.mlp.experts.1.gate_proj.weight"] = nil
        return weights
    }

    private static func partialLayerWeights(suffix: String) -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for layer in 4 ..< 8 {
            weights["model.layers.\(layer).\(suffix)"] = MLXArray.zeros([4, 4])
        }
        return weights
    }

    private static func partialLFM2Weights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for layer in 4 ..< 8 {
            weights["model.layers.\(layer).conv.conv.weight"] = MLXArray.zeros([4, 1, 2])
        }
        return weights
    }

    private static func partialGPTOSSWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        for layer in 4 ..< 8 {
            let prefix = "model.layers.\(layer).mlp.experts"
            weights["\(prefix).gate_up_proj"] = MLXArray.zeros([2, 8, 4])
            weights["\(prefix).down_proj"] = MLXArray.zeros([2, 4, 4])
        }
        return weights
    }

    private static func llamaConfiguration() -> LlamaConfiguration {
        LlamaConfiguration(
            hiddenSize: 4,
            hiddenLayers: 12,
            intermediateSize: 4,
            attentionHeads: 1,
            rmsNormEps: 0.000001,
            vocabularySize: 8,
            kvHeads: 1
        )
    }

    private static func qwen3Configuration() throws -> Qwen3Configuration {
        let json = """
            {
              "hidden_size": 4,
              "num_hidden_layers": 12,
              "intermediate_size": 4,
              "num_attention_heads": 1,
              "rms_norm_eps": 0.000001,
              "vocab_size": 8,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "tie_word_embeddings": true
            }
            """
        return try JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8))
    }

    private static func lfm2Configuration() throws -> LFM2Configuration {
        let json = """
            {
              "model_type": "lfm2",
              "vocab_size": 8,
              "hidden_size": 4,
              "num_hidden_layers": 12,
              "num_attention_heads": 1,
              "num_key_value_heads": 1,
              "norm_eps": 0.000001,
              "block_dim": 4,
              "block_ff_dim": 4,
              "block_multiple_of": 4,
              "block_auto_adjust_ff_dim": false
            }
            """
        return try JSONDecoder().decode(LFM2Configuration.self, from: Data(json.utf8))
    }

    private static func gptOSSConfiguration() throws -> GPTOSSConfiguration {
        let json = """
            {
              "model_type": "gpt_oss",
              "num_hidden_layers": 12,
              "num_local_experts": 2,
              "num_experts_per_tok": 1,
              "vocab_size": 8,
              "rms_norm_eps": 0.00001,
              "hidden_size": 4,
              "intermediate_size": 4,
              "head_dim": 4,
              "num_attention_heads": 1,
              "num_key_value_heads": 1,
              "sliding_window": 8
            }
            """
        return try JSONDecoder().decode(GPTOSSConfiguration.self, from: Data(json.utf8))
    }

    private static func glm4MoELiteConfiguration() throws -> GLM4MoELiteConfiguration {
        let json = """
            {
              "model_type": "glm4_moe_lite",
              "vocab_size": 8,
              "hidden_size": 4,
              "intermediate_size": 4,
              "moe_intermediate_size": 4,
              "num_hidden_layers": 12,
              "num_attention_heads": 1,
              "num_key_value_heads": 1,
              "n_routed_experts": 2,
              "routed_scaling_factor": 1,
              "kv_lora_rank": 4,
              "qk_rope_head_dim": 2,
              "qk_nope_head_dim": 2,
              "v_head_dim": 2,
              "norm_topk_prob": true,
              "n_group": 1,
              "topk_group": 1,
              "num_experts_per_tok": 1,
              "first_k_dense_replace": 0,
              "max_position_embeddings": 16,
              "rms_norm_eps": 0.000001,
              "rope_theta": 10000,
              "attention_bias": false,
              "partial_rotary_factor": 1,
              "num_nextn_predict_layers": 0
            }
            """
        return try JSONDecoder().decode(GLM4MoELiteConfiguration.self, from: Data(json.utf8))
    }

    private static func deepseekV3Configuration() throws -> DeepseekV3Configuration {
        let json = """
            {
              "vocab_size": 8,
              "hidden_size": 4,
              "intermediate_size": 4,
              "moe_intermediate_size": 4,
              "num_hidden_layers": 12,
              "num_attention_heads": 1,
              "num_key_value_heads": 1,
              "n_routed_experts": 2,
              "routed_scaling_factor": 1,
              "kv_lora_rank": 4,
              "q_lora_rank": 4,
              "qk_rope_head_dim": 2,
              "v_head_dim": 2,
              "qk_nope_head_dim": 2,
              "norm_topk_prob": true,
              "n_group": 1,
              "topk_group": 1,
              "num_experts_per_tok": 1,
              "moe_layer_freq": 1,
              "first_k_dense_replace": 0,
              "max_position_embeddings": 16,
              "rms_norm_eps": 0.000001,
              "rope_theta": 10000,
              "attention_bias": false
            }
            """
        return try JSONDecoder().decode(DeepseekV3Configuration.self, from: Data(json.utf8))
    }

    private static func qwen3MoEConfiguration() throws -> Qwen3MoEConfiguration {
        let json = """
            {
              "model_type": "qwen3_moe",
              "hidden_size": 4,
              "num_hidden_layers": 12,
              "intermediate_size": 4,
              "num_attention_heads": 1,
              "num_experts": 2,
              "num_experts_per_tok": 1,
              "decoder_sparse_step": 1,
              "mlp_only_layers": [],
              "moe_intermediate_size": 4,
              "rms_norm_eps": 0.000001,
              "vocab_size": 8,
              "num_key_value_heads": 1,
              "head_dim": 4
            }
            """
        return try JSONDecoder().decode(Qwen3MoEConfiguration.self, from: Data(json.utf8))
    }

    private static func qwen3NextConfiguration() throws -> Qwen3NextConfiguration {
        let json = """
            {
              "model_type": "qwen3_next",
              "hidden_size": 4,
              "num_hidden_layers": 12,
              "intermediate_size": 4,
              "num_attention_heads": 1,
              "linear_num_value_heads": 1,
              "linear_num_key_heads": 1,
              "linear_key_head_dim": 4,
              "linear_value_head_dim": 4,
              "linear_conv_kernel_dim": 2,
              "num_experts": 2,
              "num_experts_per_tok": 1,
              "decoder_sparse_step": 1,
              "shared_expert_intermediate_size": 4,
              "mlp_only_layers": [],
              "moe_intermediate_size": 4,
              "rms_norm_eps": 0.000001,
              "vocab_size": 8,
              "num_key_value_heads": 1,
              "head_dim": 4,
              "full_attention_interval": 4
            }
            """
        return try JSONDecoder().decode(Qwen3NextConfiguration.self, from: Data(json.utf8))
    }
}
