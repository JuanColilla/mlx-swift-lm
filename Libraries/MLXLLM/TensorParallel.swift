import Foundation
import MLX
import MLXNN
import MLXLMCommon

// Tensor parallel is the secondary, optional sharding mechanism (ADR INV-MODEL-01, spec-plan
// §2): it requires the full model to be resident in memory at some point before the
// non-local portion can be dropped (peak memory = full model), and it synchronizes over the
// network on every layer — it does not solve "models bigger than one device", which is the
// primary goal pipeline parallel (`AutoParallel.swift`) already covers.
//
// None of the 7 homogeneous architectures ported so far (Llama, DeepseekV3, Qwen3, Qwen3MoE,
// LFM2, GPTOSS, GLM4MoELite) implement a `.shard(group:)` method on their attention/MLP
// submodules — adding it is out of scope for this port and would need its own design pass
// (splitting `Linear`/`SwitchGLU` into `AllToShardedLinear`/`ShardedToAllLinear`, mutable
// head counts, etc., per spec-plan §4.2 point 3-4, all marked optional there). This file
// keeps the dispatch shape Infer Ring uses so a future pass can fill in real sharding without
// touching call sites, but fails loudly instead of silently no-op'ing or guessing at an API
// that doesn't exist yet.

/// Dispatch point for tensor-parallel sharding. Not implemented for any architecture yet —
/// pipeline-parallel (`pipelineAutoParallel`) is the supported path. See file header.
public func tensorAutoParallel(
    model: any LanguageModel,
    group: DistributedGroup
) -> any LanguageModel {
    if model is Qwen3MoEModel {
        fatalError("tensor-parallel no implementado para Qwen3MoEModel — pipeline-parallel es la vía soportada")
    }
    if model is Qwen3Model {
        fatalError("tensor-parallel no implementado para Qwen3Model — pipeline-parallel es la vía soportada")
    }
    if model is GPTOSSModel {
        fatalError("tensor-parallel no implementado para GPTOSSModel — pipeline-parallel es la vía soportada")
    }

    fatalError("tensor-parallel no implementado para \(type(of: model)) — pipeline-parallel es la vía soportada")
}
