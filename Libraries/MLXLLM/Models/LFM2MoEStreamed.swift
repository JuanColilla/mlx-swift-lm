// FORK(JuanColilla): R-56 expert streaming — the LFM2.5 MoE block that reads
// its routed experts from disk instead of holding them resident.
//
// A separate type on purpose, not a branch inside `Lfm2MoeSparseMoeBlock`;
// the reasoning is the one written at the top of `Qwen35StreamedMoE.swift`:
// a resident block can end up inside a `compile` trace, and a trace cannot
// contain host control flow — reading the router on the CPU, choosing
// victims, issuing `pread`. `LFM2MoEModelInner` has no compiled decode
// schedule today, so nothing else in this architecture has to be gated; that
// is the only structural difference from the Qwen 3.5 port.
//
// The one thing this family has and Qwen 3.5 does not: `num_dense_layers`.
// The first two decoder layers carry a plain MLP and no `switch_mlp` tensors,
// so the streamed layers are 2…23 while the streamed *state* is keyed by the
// decoder index. Everything downstream already agrees with that convention —
// `ExpertOffsetIndex.records(forLayer:)` searches by the layer number it
// parsed out of the checkpoint key, `ExpertKey` carries it, and the
// prefetcher's prediction state is a dictionary — so the block passes
// `layerIdx` straight through and nothing renumbers it.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Lfm2MoeStreamedSparseMoeBlock: Module, UnaryLayer {
    let args: LFM2MoEConfiguration
    let numExperts: Int
    let topK: Int
    let normTopKProb: Bool
    let useExpertBias: Bool
    /// The decoder layer index, not the position among MoE layers.
    let layerIndex: Int
    let session: ExpertStreamingSession

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "expert_bias") var expertBias: MLXArray?

    /// Not a `@ModuleInfo`: it owns no parameters. The bank it multiplies with
    /// is streaming state, and the checkpoint no longer carries
    /// `switch_mlp.*` keys for `loadWeights` to verify against.
    let switchMLP: StreamedSwitchGLU

    init(_ args: LFM2MoEConfiguration, layerIndex: Int, session: ExpertStreamingSession) {
        self.args = args
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerToken
        self.normTopKProb = args.normTopkProb
        self.useExpertBias = args.useExpertBias
        self.layerIndex = layerIndex
        self.session = session
        self.switchMLP = StreamedSwitchGLU(
            bank: session.bank,
            groupSize: session.configuration.groupSize,
            bits: session.configuration.bits)

        _gate.wrappedValue = Linear(args.hiddenSize, numExperts, bias: false)
        if useExpertBias {
            _expertBias.wrappedValue = MLXArray.zeros([numExperts])
        }
    }

    /// The same routing as the resident block, by construction: both call
    /// `lfm2MoeRoute`.
    func route(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        lfm2MoeRoute(
            gateLogits: gate(x),
            expertBias: useExpertBias ? expertBias : nil,
            topK: topK,
            normTopKProb: normTopKProb,
            routedScalingFactor: args.routedScalingFactor)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = route(x)

        let tokenCount = x.size / x.dim(-1)
        let flatX = x.reshaped(tokenCount, x.dim(-1))
        let flatIndices = indices.reshaped(tokenCount, topK)
        let flatScores = weights.reshaped(tokenCount, topK).asType(x.dtype)

        // The one mandatory synchronization of the design: the router's choice
        // has to be on the host before a single `pread` can be issued. It also
        // guarantees the previous layer's gather has been evaluated, which is
        // what lets the bank's scatter donate its buffer instead of copying
        // the whole bank.
        eval(flatIndices)
        session.noteRouterEval()
        let experts = flatIndices.asArray(UInt32.self).map { Int($0) }

        let resolution: StreamedExpertResolution
        do {
            resolution = try session.resolve(
                layer: layerIndex, tokenCount: tokenCount, experts: experts)
        } catch {
            // TD-054(a). `UnaryLayer` cannot throw, so the failure leaves
            // sideways: it is latched on the session, whose handler cancels
            // the generation, and this layer degrades to zero. LFM2.5 has no
            // shared expert, so unlike Qwen 3.5 there is nothing left to
            // return — the whole feed-forward contribution of this layer
            // vanishes.
            //
            // This is a degradation, not a recovery. The token's activations
            // are wrong and are already being written into the KV cache, so
            // the host must discard the turn's cache as well as stop the
            // stream. Returning something the process survives is what buys
            // the host the chance to do either; a `fatalError` here used to
            // take the whole app with it.
            session.recordFailure(error)
            return MLXArray.zeros(x.shape, dtype: x.dtype)
        }

        let localIndices = resolution.indices.reshaped(tokenCount, topK)
        let combined: MLXArray
        if let pools = resolution.pools {
            combined = switchMLP.callAndWeightedReduce(
                flatX, localIndices: localIndices, pools: pools, weights: flatScores)
        } else {
            combined = switchMLP.callAndWeightedReduce(
                flatX, slots: localIndices, weights: flatScores)
        }

        return combined.reshaped(x.shape)
    }
}
