import Foundation
import MLX
import MLXLMCommon
import MLXNN

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

/// - Parameter includesOutputHead: pass `false` on a rank whose `lm_head`
///   has already been removed with `dropPipelineOutputHead(from:)`. The two
///   have to agree; see that function for why.
public func pipelineAutoParallelSelection(
    model: any LanguageModel, modelShardMeta: ShardMetadata, includesOutputHead: Bool = true
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
            forModelType: modelShardMeta.modelMeta.modelType),
        includesOutputHead: includesOutputHead
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

// MARK: - Rank-aware output head (R-55 2.3, asymmetric variant)

/// Removes `lm_head` from a model that is **not** the rank that samples.
///
/// In the current pipeline every rank ends the forward pass holding the same
/// final hidden state (the `allGather` broadcast above) and every rank
/// applies its own `norm` + `lm_head`, but only the sampling rank ever reads
/// the result. On an untied checkpoint that projection is a full
/// `vocab × hidden` matrix — 826 MB on a 248k-vocabulary 6-bit model — held
/// resident on every device to compute logits nobody looks at.
///
/// Returns whether the head was actually removed, and that return value is
/// the point: the caller must pass `includesOutputHead: false` to
/// `WeightLoadingSelection.pipelineLayers` **exactly when** this returns
/// `true`. Deriving both from one call is what stops the module tree and the
/// weight selection from disagreeing — `update(parameters:verify: [.all])`
/// rejects a tree with an unfed `lm_head`, and equally rejects weights for a
/// module that is no longer there.
///
/// Returns `false`, leaving the model untouched, when:
/// - the checkpoint ties its embedding and output matrices, so `lmHead` is
///   already `nil` and the forward pass uses `embedTokens.asLinear` — there
///   is nothing to save;
/// - the architecture declares `lmHead` non-optional (`DeepseekV3Model`,
///   `GPTOSSModel`, `GLM4MoELiteModel`). Making those optional is a change to
///   their forward passes, not to this function, and until then they keep the
///   old behaviour rather than failing to load.
///
/// **The logits are still built, just never evaluated.** With `lmHead` gone
/// the forward falls through to `embedTokens.asLinear(out)`, which on an
/// untied model is arithmetically wrong — and harmless, because MLX is lazy
/// and a non-sampling rank never materializes its logits (MLXHub's
/// `runLoop` reads `nextTokenLogits` only when `isSampler`). A caller that
/// *does* evaluate them on a rank where this returned `true` gets garbage,
/// which is why this is named for the pipeline and not offered as a general
/// memory optimization.
/// The half of `dropPipelineOutputHead(from:)` each architecture implements
/// itself, because `@ModuleInfo`'s backing storage is private to the class
/// that declares it. Conformance is the architecture's statement that its
/// forward pass survives a missing head — today by falling through to
/// `embedTokens.asLinear`.
public protocol PipelineOutputHeadRemovable {
    /// Removes `lm_head` if there is one to remove, and says whether it did.
    @discardableResult
    func removePipelineOutputHead() -> Bool
}

extension LlamaModel: PipelineOutputHeadRemovable {}
extension Qwen3Model: PipelineOutputHeadRemovable {}
extension Qwen3NextModel: PipelineOutputHeadRemovable {}
/// `Qwen35MoEModel` inherits this: it subclasses `Qwen35Model`.
extension Qwen35TextModel: PipelineOutputHeadRemovable {}

@discardableResult
public func dropPipelineOutputHead(from model: any LanguageModel) -> Bool {
    if let removable = model as? PipelineOutputHeadRemovable {
        return removable.removePipelineOutputHead()
    }
    // `Qwen35Model` is the multimodal wrapper: the text tower it delegates to
    // is the one holding the head.
    if let wrapper = model as? Qwen35Model {
        return wrapper.languageModel.removePipelineOutputHead()
    }
    return false
}
