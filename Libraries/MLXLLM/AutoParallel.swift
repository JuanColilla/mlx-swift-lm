import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Adapted from Infer Ring's Ring/AutoParallel.swift (MIT), retargeted against this
// package's real model types and property names.

// MARK: - Auto Parallel Entry Point

/// Shards a homogeneous-architecture model for pipeline parallelism: keeps only
/// `modelShardMeta.startLayer..<modelShardMeta.endLayer` locally, and wraps the first/last
/// layer of that range with the send/recv/allGather glue needed to stitch ranks together.
///
/// Hybrid architectures with multiple mask kinds per forward pass (Qwen3.5/Qwen3Next) do not
/// go through here — they don't have layers conforming to `TransformerLayer` (see
/// `AutoParallelHybrid.swift`, ADR INV-MODEL-01).
public func pipelineAutoParallel(
    model: any LanguageModel,
    group: DistributedGroup,
    modelShardMeta: ShardMetadata
) -> any LanguageModel {
    let layers = getLayers(from: model)

    guard !layers.isEmpty else {
        print("Warning: pipelineAutoParallel could not find any layers in model")
        return model
    }

    let startLayer = modelShardMeta.startLayer
    let endLayer = modelShardMeta.endLayer
    let deviceRank = modelShardMeta.deviceRank
    let worldSize = modelShardMeta.worldSize

    let safeStart = max(0, min(startLayer, layers.count))
    let safeEnd = max(safeStart, min(endLayer, layers.count))

    let subsetLayers = Array(layers[safeStart ..< safeEnd])
    guard !subsetLayers.isEmpty else {
        print("Warning: pipelineAutoParallel layer range [\(safeStart), \(safeEnd)) is empty")
        return model
    }

    let first = PipelineFirstLayer(originalLayer: subsetLayers[0], r: deviceRank, group: group)
    let last = PipelineLastLayer(
        originalLayer: subsetLayers.last!, r: deviceRank, s: worldSize, group: group)

    var newLayers = subsetLayers
    newLayers[0] = first
    newLayers[newLayers.count - 1] = last
    if model is GPTOSSModel {
        // MLX GPTOSS bugged? layer output becomes f32 from bf16 — replicated as-is from
        // Infer Ring, not re-investigated (ADR INV-MODEL-01).
        last.forceDType = .bfloat16
    }

    setLayers(on: model, newLayers: newLayers, shardOffset: safeStart)

    return model
}

// MARK: - Wrapper Layers

public class PipelineWrappedLayer: Module {
    public let originalLayer: any TransformerLayer

    public init(originalLayer: any TransformerLayer) {
        self.originalLayer = originalLayer
        super.init()
    }

    fileprivate func callOriginal(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        originalLayer(x, mask: mask, cache: cache)
    }
}

/// Wraps the first layer of a rank's shard: receives the hidden state produced by the
/// previous rank instead of trusting the locally-computed embedding (rank 0 is the only
/// rank whose local embedding is actually used).
public final class PipelineFirstLayer: PipelineWrappedLayer, TransformerLayer {
    public let r: Int
    public let group: DistributedGroup

    public init(originalLayer: any TransformerLayer, r: Int, group: DistributedGroup) {
        self.r = r
        self.group = group
        super.init(originalLayer: originalLayer)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var x = x
        if r != 0 {
            x = group.recvLike(x, source: Int32(r - 1))
        }
        return callOriginal(x, mask: mask, cache: cache)
    }
}

/// Wraps the last layer of a rank's shard: forwards the output to the next rank (unless this
/// is the last rank in the pipeline), then broadcasts the final hidden state — computed only
/// by the last rank — to every rank via `allGather` + trim to the last segment, since
/// `DistributedGroup` has no native broadcast primitive. Every rank ends up with the same
/// final hidden state and can independently apply its own final norm / lm_head.
public final class PipelineLastLayer: PipelineWrappedLayer, TransformerLayer {
    public let r: Int
    public let s: Int
    public let group: DistributedGroup
    var forceDType: DType? = nil

    public init(originalLayer: any TransformerLayer, r: Int, s: Int, group: DistributedGroup) {
        self.r = r
        self.s = s
        self.group = group
        super.init(originalLayer: originalLayer)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var output = callOriginal(x, mask: mask, cache: cache)

        if let forceDType {
            output = output.asType(forceDType)
        }

        if r != s - 1 {
            output = group.send(output, dest: Int32((r + 1) % s))
        }

        let gathered = group.allGather(output)
        // MLX evaluates lazily: `send`/`allGather` only run their real
        // network I/O when something forces materialization of this node.
        // A non-sampling rank never calls `.item()`/`.eval()` on anything
        // downstream of its own forward pass (its logits are discarded),
        // so without this, its half of the pipeline's send/allGather never
        // executes — the sampling rank blocks forever waiting for a
        // contribution the other rank's process never actually sent, until
        // iOS's Metal GPU-watchdog aborts it. Every rank must materialize
        // this collective, every forward pass, regardless of whether its
        // own result is used.
        eval(gathered)
        let batchSize = output.dim(0)
        let totalSize = gathered.dim(0)
        let startIndex = totalSize - batchSize

        if output.dim(1) > 1, var cache {
            // Dependency edge so batched prefill isn't deadlocked waiting on the collective.
            cache.state = depends(
                inputs: cache.state, dependencies: Array(gathered[startIndex ..< totalSize]))
        }

        return gathered[startIndex ..< totalSize]
    }
}

// MARK: - Layer Access Per Architecture

/// Gets the transformer layers from a homogeneous-architecture `LanguageModel`.
func getLayers(from model: any LanguageModel) -> [TransformerLayer] {
    if let llama = model as? LlamaModel {
        return llama.model.layers
    }
    if let deepseek = model as? DeepseekV3Model {
        return deepseek.model.layers
    }
    if let qwen = model as? Qwen3MoEModel {
        return qwen.model.layers
    }
    if let qwen = model as? Qwen3Model {
        return qwen.model.layers
    }
    if let lfm = model as? LFM2Model {
        return lfm.model.layers
    }
    if let gpt = model as? GPTOSSModel {
        return gpt.model.layers
    }
    if let glm = model as? GLM4MoELiteModel {
        return glm.model.layers
    }

    return []
}

/// Replaces the transformer layers on a homogeneous-architecture `LanguageModel` with the
/// sharded subset, fixing up whatever per-architecture bookkeeping depends on layer count/
/// offset (KV cache metadata, sliding/full attention indices, etc.).
func setLayers(on model: any LanguageModel, newLayers: [TransformerLayer], shardOffset: Int) {
    func localKVHeads(_ heads: [Int]) -> [Int] {
        let end = min(shardOffset + newLayers.count, heads.count)
        guard shardOffset >= 0, shardOffset < end else { return [] }
        return Array(heads[shardOffset ..< end])
    }

    if let llama = model as? LlamaModel {
        llama.model.layers = newLayers
        llama.kvHeads = localKVHeads(llama.kvHeads)
    } else if let deepseek = model as? DeepseekV3Model {
        deepseek.model.layers = newLayers
        deepseek.model.endIdx = newLayers.count
        deepseek.model.numLayers = newLayers.count
        deepseek.kvHeads = localKVHeads(deepseek.kvHeads)
    } else if let qwen = model as? Qwen3MoEModel {
        qwen.model.layers = newLayers
        qwen.kvHeads = localKVHeads(qwen.kvHeads)
    } else if let qwen = model as? Qwen3Model {
        qwen.model.layers = newLayers
        qwen.kvHeads = localKVHeads(qwen.kvHeads)
    } else if let lfm = model as? LFM2Model {
        lfm.model.layers = newLayers
        lfm.shardOffset = shardOffset
        lfm.kvHeads = localKVHeads(lfm.kvHeads)
    } else if let gpt = model as? GPTOSSModel {
        gpt.model.layers = newLayers
        gpt.model.layerTypes = Array(
            gpt.model.layerTypes[shardOffset ..< shardOffset + newLayers.count])
        gpt.model.slidingAttentionIndex =
            gpt.model.layerTypes.firstIndex(of: "sliding_attention") ?? 0
        gpt.model.fullAttentionIndex = gpt.model.layerTypes.firstIndex(of: "full_attention") ?? 0
        gpt.kvHeads = localKVHeads(gpt.kvHeads)
    } else if let glm = model as? GLM4MoELiteModel {
        glm.model.layers = newLayers
        glm.kvHeads = localKVHeads(glm.kvHeads)
    } else {
        print("Couldn't update hidden layers for model \(String(describing: type(of: model)))")
    }
}
