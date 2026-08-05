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
