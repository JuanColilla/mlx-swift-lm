import Foundation
import MLX
import MLXNN
import MLXLMCommon

// Adapted from Infer Ring's Ring/AutoParallel.swift (MIT), retargeted against this
// package's real model types and property names.

// MARK: - Auto Parallel Entry Point

public enum PipelineWeightLayout {
    public static func sourcePrefixes(forModelType modelType: String) -> [String]? {
        switch modelType {
        case "llama", "deepseek_v3", "qwen3_moe", "qwen3", "lfm2", "gpt_oss", "glm4_moe_lite",
            "qwen3_next":
            return ["model.layers."]
        case "qwen3_5", "qwen3_5_moe":
            return [
                "model.language_model.layers.", "language_model.model.layers.",
                "model.layers.", "language_model.layers.", "layers.",
            ]
        default:
            return nil
        }
    }

    public static func destinationPrefix(forModelType modelType: String) -> String {
        switch modelType {
        case "qwen3_5", "qwen3_5_moe": return "language_model.model.layers."
        default: return "model.layers."
        }
    }
}

public func pipelineAutoParallelSelection(
    model: any LanguageModel, modelShardMeta: ShardMetadata
) -> WeightLoadingSelection? {
    let layers = getLayers(from: model)
    guard !layers.isEmpty else { return nil }
    let safeStart = max(0, min(modelShardMeta.startLayer, layers.count))
    let safeEnd = max(safeStart, min(modelShardMeta.endLayer, layers.count))
    guard safeStart < safeEnd else { return nil }

    setLayers(on: model, newLayers: Array(layers[safeStart ..< safeEnd]), shardOffset: safeStart)

    return .pipelineLayers(
        range: safeStart ..< safeEnd,
        sourcePrefixes: PipelineWeightLayout.sourcePrefixes(
            forModelType: modelShardMeta.modelMeta.modelType) ?? ["model.layers."],
        destinationPrefix: PipelineWeightLayout.destinationPrefix(
            forModelType: modelShardMeta.modelMeta.modelType)
    )
}

public func pipelineAutoParallelWrapBoundaries(
    model: any LanguageModel, group: DistributedGroup, modelShardMeta: ShardMetadata
) {
    let layers = getLayers(from: model)
    guard !layers.isEmpty else { return }
    let deviceRank = modelShardMeta.deviceRank
    let worldSize = modelShardMeta.worldSize

    let first = PipelineFirstLayer(originalLayer: layers[0], r: deviceRank, group: group)
    let last = PipelineLastLayer(
        originalLayer: layers.last!, r: deviceRank, s: worldSize, group: group)
    var wrapped = layers
    wrapped[0] = first
    wrapped[wrapped.count - 1] = last
    if model is GPTOSSModel {
        last.forceDType = .bfloat16
    }
    setLayers(
        on: model,
        newLayers: wrapped,
        shardOffset: pipelineAutoParallelWrapBoundaryShardOffset(for: modelShardMeta))
}

func pipelineAutoParallelWrapBoundaryShardOffset(for modelShardMeta: ShardMetadata) -> Int {
    modelShardMeta.startLayer
}

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
    guard pipelineAutoParallelSelection(model: model, modelShardMeta: modelShardMeta) != nil else {
        return model
    }
    pipelineAutoParallelWrapBoundaries(model: model, group: group, modelShardMeta: modelShardMeta)
    return model
}

// MARK: - Wrapper Layers

public class PipelineWrappedLayer: Module {
    public let originalLayer: Module

    public init(originalLayer: Module) {
        self.originalLayer = originalLayer
        super.init()
    }

    fileprivate func callOriginal(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        if let layer = originalLayer as? TransformerLayer {
            return layer(x, mask: mask, cache: cache)
        } else if let layer = originalLayer as? UnaryLayer {
            return layer(x)
        }
        fatalError("\(type(of: self)): originalLayer signature not supported")
    }
}

/// Wraps the first layer of a rank's shard: receives the hidden state produced by the
/// previous rank instead of trusting the locally-computed embedding (rank 0 is the only
/// rank whose local embedding is actually used).
public final class PipelineFirstLayer: PipelineWrappedLayer, TransformerLayer {
    public let r: Int
    public let group: DistributedGroup

    public init(originalLayer: Module, r: Int, group: DistributedGroup) {
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

    public init(originalLayer: Module, r: Int, s: Int, group: DistributedGroup) {
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
            cache.state = depends(inputs: cache.state, dependencies: Array(gathered[startIndex..<totalSize]))
        }

        return gathered[startIndex..<totalSize]
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
/// offset (KV cache rebuild, sliding/full attention indices, etc.).
func setLayers(on model: any LanguageModel, newLayers: [TransformerLayer], shardOffset: Int) {
    if let llama = model as? LlamaModel {
        llama.model.layers = newLayers
        llama.model.rebuildCaches()
    } else if let deepseek = model as? DeepseekV3Model {
        deepseek.model.layers = newLayers
        deepseek.model.endIdx = newLayers.count
        deepseek.model.numLayers = newLayers.count
        deepseek.model.rebuildCaches()
    } else if let qwen = model as? Qwen3MoEModel {
        qwen.model.layers = newLayers
        qwen.model.rebuildCaches()
    } else if let qwen = model as? Qwen3Model {
        qwen.model.layers = newLayers
        qwen.model.rebuildCaches()
    } else if let lfm = model as? LFM2Model {
        lfm.model.layers = newLayers
        lfm.shardOffset = shardOffset
        lfm.model.rebuildCaches()
    } else if let gpt = model as? GPTOSSModel {
        gpt.model.layers = newLayers
        gpt.model.layerTypes = Array(gpt.model.layerTypes[shardOffset..<shardOffset + newLayers.count])
        gpt.model.slidingAttentionIndex = gpt.model.layerTypes.firstIndex(of: "sliding_attention") ?? 0
        gpt.model.fullAttentionIndex = gpt.model.layerTypes.firstIndex(of: "full_attention") ?? 0
        gpt.model.rebuildCaches()
    } else if let glm = model as? GLM4MoELiteModel {
        glm.model.layers = newLayers
        glm.model.rebuildCaches()
    } else {
        print("Couldn't update hidden layers for model \(String(describing: type(of: model)))")
    }
}
