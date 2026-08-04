import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Pipeline sharding for hybrid (Gated DeltaNet + full attention) architectures — Qwen3.5 /
// Qwen3.5MoE / Qwen3Next. See knowledge/superpowers/specs/2026-08-03-distributed-inference-
// pipeline-parallel-design.md §3/§4.3 in MLXHub for why these can't reuse the generic
// `PipelineFirstLayer`/`PipelineLastLayer` wrappers from AutoParallel.swift: their decoder
// layers take `(x, attentionMask:, ssmMask:, cache:)`, not `TransformerLayer`'s
// `(x, mask:, cache:)`, and the two masks are computed once per forward from indices into the
// FULL layer array — indices that go out of range once a shard doesn't contain layer 0 or the
// first full-attention layer. The sharding logic therefore lives inside
// `Qwen35TextModelInner.callAsFunction`/`Qwen3NextModelInner.callAsFunction` themselves
// (Qwen35.swift / Qwen3Next.swift), not in wrapper types here — this file only slices `layers`
// and recomputes the local mask indices.
//
// The public helpers take the shard fields explicitly so callers can derive them from
// `ShardMetadata` or from another topology planner without coupling model mutation to a
// particular transport DTO.
//
// Two concrete functions, not one generic over a shared protocol: `Qwen35TextModelInner` and
// `Qwen3NextModelInner` are structurally identical but intentionally don't conform to
// `TransformerLayer` (that's the whole point), and introducing a new protocol just to share
// ~20 lines of identical-but-unenforced-as-such logic between exactly two types would be
// speculative abstraction for a family that isn't growing a third member any time soon.

/// Pipeline-shards a `Qwen35TextModelInner` (used by both `Qwen35Model` and its MoE subclass
/// `Qwen35MoEModel`) in place: keeps only `layers[startLayer..<endLayer)`, recomputes the local
/// full-attention/linear-attention mask indices, and wires up the pipeline recv/send/allGather
/// performed by `Qwen35TextModelInner.callAsFunction`.
public func pipelineShardHybrid(
    model: Qwen35TextModel,
    group: DistributedGroup,
    startLayer: Int,
    endLayer: Int,
    deviceRank: Int,
    worldSize: Int
) {
    let safeStart = max(0, min(startLayer, model.model.layers.count))
    let safeEnd = max(safeStart, min(endLayer, model.model.layers.count))
    pipelineShardHybrid(
        model: model.model,
        group: group,
        startLayer: safeStart,
        endLayer: safeEnd,
        deviceRank: deviceRank,
        worldSize: worldSize
    )
    model.kvHeads = Array(model.kvHeads[safeStart ..< safeEnd])
}

/// Low-level variant for runtimes that own the outer model metadata separately.
public func pipelineShardHybrid(
    model: Qwen35TextModelInner,
    group: DistributedGroup,
    startLayer: Int,
    endLayer: Int,
    deviceRank: Int,
    worldSize: Int
) {
    let fullLayers = model.layers
    let safeStart = max(0, min(startLayer, fullLayers.count))
    let safeEnd = max(safeStart, min(endLayer, fullLayers.count))
    let localLayers = Array(fullLayers[safeStart ..< safeEnd])

    model.layers = localLayers
    model.faIdx = localLayers.firstIndex { !$0.isLinear }
    model.ssmIdx = localLayers.firstIndex { $0.isLinear }
    model.pipelineRank = deviceRank
    model.pipelineWorldSize = worldSize
    model.pipelineGroup = group

}

/// Same as `pipelineShardHybrid(model:group:startLayer:endLayer:deviceRank:worldSize:)` but for
/// `Qwen3NextModelInner`, which has the identical hybrid-layer/mask-index structure.
public func pipelineShardHybridNext(
    model: Qwen3NextModel,
    group: DistributedGroup,
    startLayer: Int,
    endLayer: Int,
    deviceRank: Int,
    worldSize: Int
) {
    let safeStart = max(0, min(startLayer, model.model.layers.count))
    let safeEnd = max(safeStart, min(endLayer, model.model.layers.count))
    pipelineShardHybridNext(
        model: model.model,
        group: group,
        startLayer: safeStart,
        endLayer: safeEnd,
        deviceRank: deviceRank,
        worldSize: worldSize
    )
    model.kvHeads = Array(model.kvHeads[safeStart ..< safeEnd])
}

/// Low-level variant for runtimes that own the outer model metadata separately.
public func pipelineShardHybridNext(
    model: Qwen3NextModelInner,
    group: DistributedGroup,
    startLayer: Int,
    endLayer: Int,
    deviceRank: Int,
    worldSize: Int
) {
    let fullLayers = model.layers
    let safeStart = max(0, min(startLayer, fullLayers.count))
    let safeEnd = max(safeStart, min(endLayer, fullLayers.count))
    let localLayers = Array(fullLayers[safeStart ..< safeEnd])

    model.layers = localLayers
    model.faIdx = localLayers.firstIndex { !$0.isLinear }
    model.ssmIdx = localLayers.firstIndex { $0.isLinear }
    model.pipelineRank = deviceRank
    model.pipelineWorldSize = worldSize
    model.pipelineGroup = group

}
