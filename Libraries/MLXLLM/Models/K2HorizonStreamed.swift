// FORK(JuanColilla): R-56 expert streaming — the K2-Horizon blocks that read
// their routed experts from disk instead of holding them resident.
//
// Separate types on purpose, not branches inside the resident ones; the
// reasoning is the one written at the top of `Qwen35StreamedMoE.swift`. What
// is different here is that K2-Horizon MoVA routes *two* families of experts
// per sparse layer: the feed-forward experts (`mlp.switch_mlp`, 100 of them,
// top-8) and the value projection of attention (`self_attn.switch_v`, 64 of
// them, top-4). Each has its own router, its own count, its own row size and
// therefore its own slot bank; the session keeps them apart by
// `ExpertFamily`. Everything else — `gate`, `v_router`, `shared_experts`,
// q/k/o and the norms — stays resident.
//
// The routing arithmetic is not duplicated: both the resident and the
// streamed blocks call `k2HorizonRouterLogits` and `k2HorizonRoute`, the same
// discipline as `lfm2MoeRoute`.
//
// Dense layers (`mlp_only_layers`, or the ones the sparse step skips) never
// come through here; `K2HorizonDecoderLayer.init` only picks these types for
// a sparse layer, and the offset index only has records for those.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The host-side half of a streamed routing step: the choice comes back to
/// the CPU, the session resolves it, and the block multiplies.
///
/// The one mandatory synchronization of the design: the router's choice has
/// to be on the host before a single `pread` can be issued. It also
/// guarantees the previous step's gather has been evaluated, which is what
/// lets the bank's scatter donate its buffer instead of copying the whole
/// bank.
private func k2HorizonStreamedRouting(
    _ indices: MLXArray, tokenCount: Int, topK: Int, session: ExpertStreamingSession
) -> [Int] {
    let flatIndices = indices.reshaped(tokenCount, topK).asType(.uint32)
    eval(flatIndices)
    session.noteRouterEval()
    return flatIndices.asArray(UInt32.self).map { Int($0) }
}

// MARK: - Attention

/// MoVA attention whose value experts live in the value slot bank.
public final class K2HorizonStreamedAttention: Module, K2HorizonAttentionLayer {
    let args: K2HorizonConfiguration
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let ropeHeadDim: Int
    let scale: Float
    let topK: Int
    /// The decoder layer index, which is what the offset index is keyed by.
    let layerIndex: Int
    let session: ExpertStreamingSession

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: K2HorizonGroupedRMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: K2HorizonGroupedRMSNorm?
    @ModuleInfo(key: "v_router") var vRouter: Linear

    /// Not a `@ModuleInfo`: it owns no parameters. The bank it multiplies with
    /// is streaming state, and the checkpoint no longer carries `switch_v.*`
    /// keys for `loadWeights` to verify against.
    let switchV: StreamedSwitchLinear

    let rope: RoPE

    public init(_ args: K2HorizonConfiguration, layerIdx: Int, session: ExpertStreamingSession) {
        precondition(args.usesMoVA(layerIdx), "streamed MoVA attention on a layer without it")
        guard let bank = session.bank(for: .value) else {
            fatalError("the streaming session has no value-expert bank for layer \(layerIdx)")
        }
        self.args = args
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.ropeHeadDim = args.ropeHeadDim
        self.scale = pow(Float(headDim), -0.5)
        self.topK = args.movaNumExpertsPerToken
        self.layerIndex = layerIdx
        self.session = session
        self.switchV = StreamedSwitchLinear(
            bank: bank,
            groupSize: session.configuration.groupSize,
            bits: session.configuration.bits)

        let dim = args.hiddenSize
        _qProj.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)
        _vRouter.wrappedValue = Linear(dim, args.movaNumExperts, bias: args.moeGateBias)

        if args.attentionGate != nil {
            _gateProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
        }
        if args.queryKeyNorm {
            _qNorm.wrappedValue = K2HorizonGroupedRMSNorm(
                dimensions: heads * headDim, groups: heads, eps: args.rmsNormEps)
            _kNorm.wrappedValue = K2HorizonGroupedRMSNorm(
                dimensions: kvHeads * headDim, groups: kvHeads, eps: args.rmsNormEps)
        }

        self.rope = RoPE(dimensions: ropeHeadDim, traditional: false, base: args.ropeTheta)
    }

    /// The same routing as the resident attention, by construction: both call
    /// `k2HorizonRoute` with the router's bias steering the selection only.
    func route(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        k2HorizonRoute(
            logits: k2HorizonRouterLogits(vRouter, x), selectionBias: vRouter.bias,
            score: args.routerScore, topK: topK, normalize: topK > 1,
            scalingFactor: args.routerScalingFactor)
    }

    /// MoVA values from the bank: top-k value experts per token, `silu` on
    /// each expert output, weighted by the router — the resident
    /// `routedValues` with the matmul redirected.
    func routedValues(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = route(x)

        let tokenCount = x.size / x.dim(-1)
        let flatX = x.reshaped(tokenCount, x.dim(-1))
        let flatWeights = weights.reshaped(tokenCount, topK).asType(x.dtype)
        let experts = k2HorizonStreamedRouting(
            indices, tokenCount: tokenCount, topK: topK, session: session)

        let resolution: StreamedExpertResolution
        do {
            resolution = try session.resolve(
                layer: layerIndex, family: .value, tokenCount: tokenCount, experts: experts)
        } catch {
            // TD-054(a). `callAsFunction` cannot throw, so the failure leaves
            // sideways: it is latched on the session, whose handler cancels
            // the generation, and the values of this layer degrade to zero.
            //
            // This is a degradation, not a recovery. The token's activations
            // are wrong and are already being written into the KV cache, so
            // the host must discard the turn's cache as well as stop the
            // stream. Returning something the process survives is what buys
            // the host the chance to do either.
            session.recordFailure(error)
            return MLXArray.zeros(
                [x.dim(0), x.dim(1), kvHeads * headDim], dtype: x.dtype)
        }

        let localIndices = resolution.indices.reshaped(tokenCount, topK)
        var y: MLXArray
        if let pools = resolution.pools {
            y = switchV(flatX, localIndices: localIndices, pools: pools)
        } else {
            y = switchV(flatX, slots: localIndices)
        }
        y = MLXNN.silu(y)
        return weightedExpertSum(y, flatWeights).reshaped(x.dim(0), x.dim(1), -1)
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var queries = qProj(x)
        var keys = kProj(x)
        if let qNorm { queries = qNorm(queries) }
        if let kNorm { keys = kNorm(keys) }
        let values = routedValues(x)

        return k2HorizonAttend(
            queries: queries, keys: keys, values: values, input: x,
            heads: heads, kvHeads: kvHeads, headDim: headDim, ropeHeadDim: ropeHeadDim,
            scale: scale, rope: rope, gateProj: gateProj, gateFunction: args.attentionGate,
            oProj: oProj, mask: mask, cache: cache)
    }
}

// MARK: - Feed-forward

/// The sparse MoE block whose routed experts live in the MLP slot bank.
public final class K2HorizonStreamedSparseMoEBlock: Module, UnaryLayer {
    let args: K2HorizonConfiguration
    let topK: Int
    /// The decoder layer index, not the position among sparse layers.
    let layerIndex: Int
    let session: ExpertStreamingSession

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "shared_experts") var sharedExperts: K2HorizonMLP?

    /// Not a `@ModuleInfo`: it owns no parameters. The bank it multiplies with
    /// is streaming state, and the checkpoint no longer carries
    /// `switch_mlp.*` keys for `loadWeights` to verify against.
    let switchMLP: StreamedSwitchGLU

    public init(_ args: K2HorizonConfiguration, layerIndex: Int, session: ExpertStreamingSession) {
        self.args = args
        self.topK = args.numExpertsPerToken
        self.layerIndex = layerIndex
        self.session = session
        self.switchMLP = StreamedSwitchGLU(
            bank: session.bank,
            groupSize: session.configuration.groupSize,
            bits: session.configuration.bits)

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: args.moeGateBias)
        if args.numSharedExperts > 0 {
            _sharedExperts.wrappedValue = K2HorizonMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.moeIntermediateSize * args.numSharedExperts)
        }
    }

    /// The same routing as the resident block, by construction.
    func route(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        k2HorizonRoute(
            logits: k2HorizonRouterLogits(gate, x), selectionBias: gate.bias,
            score: args.routerScore, topK: topK,
            normalize: args.normTopkProb, scalingFactor: args.routerScalingFactor)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = route(x)

        let tokenCount = x.size / x.dim(-1)
        let flatX = x.reshaped(tokenCount, x.dim(-1))
        let flatWeights = weights.reshaped(tokenCount, topK).asType(x.dtype)
        let experts = k2HorizonStreamedRouting(
            indices, tokenCount: tokenCount, topK: topK, session: session)

        let shared = sharedExperts.map { $0(x) }

        let resolution: StreamedExpertResolution
        do {
            resolution = try session.resolve(
                layer: layerIndex, family: .mlp, tokenCount: tokenCount, experts: experts)
        } catch {
            // TD-054(a): latch, let the host cancel and drop the turn's KV
            // cache, and degrade to the shared expert alone — the routed
            // contribution is zero. See the note in the attention above.
            session.recordFailure(error)
            return shared ?? MLXArray.zeros(x.shape, dtype: x.dtype)
        }

        let localIndices = resolution.indices.reshaped(tokenCount, topK)
        let combined: MLXArray
        if let pools = resolution.pools {
            combined = switchMLP.callAndWeightedReduce(
                flatX, localIndices: localIndices, pools: pools, weights: flatWeights)
        } else {
            combined = switchMLP.callAndWeightedReduce(
                flatX, slots: localIndices, weights: flatWeights)
        }

        var y = combined.reshaped(x.shape)
        if let shared { y = y + shared }
        return y
    }
}
