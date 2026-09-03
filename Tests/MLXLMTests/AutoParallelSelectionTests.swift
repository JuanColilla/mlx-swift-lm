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

    /// R-55 2.3 (variante asimétrica): un rank que no muestrea no necesita
    /// `lm_head` — en Ornith 1.5 son 826 MB residentes en cada worker para
    /// calcular unos logits que nadie lee.
    @Test func excludingTheOutputHeadLeavesEverythingElseAlone() {
        let selection = WeightLoadingSelection.pipelineLayers(
            range: 4 ..< 8, sourcePrefixes: ["model.layers."], destinationPrefix: "model.layers.",
            includesOutputHead: false
        )
        #expect(!selection.includes("lm_head.weight"))
        #expect(!selection.includes("lm_head.scales"))
        #expect(!selection.includes("lm_head.biases"))
        // El resto de globales sigue haciendo falta en todos los ranks: la
        // tabla de embeddings da la plantilla de shape del `recvLike` y la
        // norm final es un vector.
        #expect(selection.includes("model.embed_tokens.weight"))
        #expect(selection.includes("model.norm.weight"))
        #expect(selection.includes("model.layers.5.self_attn.q_proj.weight"))
        #expect(!selection.includes("model.layers.2.self_attn.q_proj.weight"))
    }

    /// El valor por defecto es el comportamiento anterior: todo rank carga su
    /// head salvo que se le pida explícitamente lo contrario.
    @Test func theOutputHeadIsIncludedByDefault() {
        let selection = WeightLoadingSelection.pipelineLayers(
            range: 0 ..< 4, sourcePrefixes: ["model.layers."], destinationPrefix: "model.layers."
        )
        #expect(selection.includes("lm_head.weight"))
    }

    /// Un tensor de capa cuyo nombre EMPIEZA por algo parecido no puede caer
    /// por el filtro del head: el prefijo lleva el punto a propósito.
    @Test func onlyTheOutputHeadItselfIsExcluded() {
        let selection = WeightLoadingSelection.pipelineLayers(
            range: 0 ..< 4, sourcePrefixes: ["model.layers."], destinationPrefix: "model.layers.",
            includesOutputHead: false
        )
        #expect(selection.includes("lm_head_extra.weight"))
        #expect(selection.includes("model.lm_head.weight"))
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
            #expect(PipelineWeightLayout.destinationPrefix(forModelType: type) == "model.layers.")
        }
        let qwen35Prefixes = [
            "model.language_model.layers.", "language_model.model.layers.",
            "model.layers.", "language_model.layers.", "layers.",
        ]
        for type in ["qwen3_5", "qwen3_5_moe"] {
            #expect(
                PipelineWeightLayout.sourcePrefixes(forModelType: type) == qwen35Prefixes)
            #expect(
                PipelineWeightLayout.destinationPrefix(forModelType: type)
                    == "language_model.model.layers.")
        }
        #expect(PipelineWeightLayout.sourcePrefixes(forModelType: "unknown_arch") == nil)
    }

    @Test func wrappingUsesShardStartAsSetLayersOffset() {
        let metadata = ShardMetadata(
            modelMeta: ModelMetadata(
                modelId: "test/lfm2", modelType: "lfm2", prettyName: "LFM2 Test",
                storageSize: MemorySize(), nLayers: 6, hiddenSize: 8),
            deviceRank: 1,
            worldSize: 2,
            startLayer: 2,
            endLayer: 5,
            nLayers: 6)

        #expect(
            pipelineAutoParallelWrapBoundaryShardOffset(for: metadata) == metadata.startLayer)
    }
}
