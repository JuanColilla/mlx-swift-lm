import Testing

@testable import MLXLLM
@testable import MLXLMCommon

@Suite struct AutoParallelSelectionTests {
    @Test func selectionKeepsOnlyLocalRangeAndGlobalsAlways() {
        let selection = WeightLoadingSelection.pipelineLayers(
            range: 4 ..< 8, sourcePrefixes: ["model.layers."], destinationPrefix: "model.layers."
        )
        #expect(selection.includes("model.layers.5.self_attn.q_proj.weight"))
        #expect(!selection.includes("model.layers.2.self_attn.q_proj.weight"))
        #expect(selection.includes("model.embed_tokens.weight"))
        #expect(selection.includes("model.norm.weight"))
        #expect(selection.includes("lm_head.weight"))
    }

    @Test func rewriteRenumbersToLocalZeroBasedIndex() {
        let selection = WeightLoadingSelection.pipelineLayers(
            range: 4 ..< 8, sourcePrefixes: ["model.layers."], destinationPrefix: "model.layers."
        )
        #expect(
            selection.rewrite("model.layers.5.self_attn.q_proj.weight")
                == "model.layers.1.self_attn.q_proj.weight")
    }

    @Test func sourcePrefixTableCoversAllTenFamilies() {
        for type in [
            "llama", "deepseek_v3", "qwen3_moe", "qwen3", "lfm2", "gpt_oss", "glm4_moe_lite",
            "qwen3_next",
        ] {
            #expect(PipelineWeightLayout.sourcePrefixes(forModelType: type) == ["model.layers."])
        }
        for type in ["qwen3_5", "qwen3_5_moe"] {
            #expect(
                PipelineWeightLayout.sourcePrefixes(forModelType: type)?.contains("model.layers.")
                    == true)
        }
        #expect(PipelineWeightLayout.sourcePrefixes(forModelType: "unknown_arch") == nil)
    }
}
