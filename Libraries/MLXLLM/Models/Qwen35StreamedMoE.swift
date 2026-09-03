// FORK(JuanColilla): R-56 expert streaming — the Qwen 3.5 MoE block that
// reads its routed experts from disk instead of holding them resident.
//
// A separate type on purpose, not a branch inside `Qwen35SparseMoeBlock`:
//
//  * The resident decode path is traced with `compile`, and a trace cannot
//    contain host control flow — reading the router on the CPU, choosing
//    victims, issuing `pread`. It also bakes the weights it captured, which a
//    mutable bank contradicts by definition.
//  * SwiftLM#84 is the precedent: a synchronization gate that only streaming
//    needed leaked into the resident path and cost it 3.4x. The resident block
//    is not modified by this file.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class Qwen35StreamedSparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int
    let layerIndex: Int
    let session: ExpertStreamingSession

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen3NextMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    /// Not a `@ModuleInfo`: it owns no parameters. The bank it multiplies with
    /// is streaming state, and the checkpoint no longer carries `switch_mlp.*`
    /// keys for `loadWeights` to verify against.
    let switchMLP: StreamedSwitchGLU

    init(_ args: Qwen35TextConfiguration, layerIndex: Int, session: ExpertStreamingSession) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok
        self.layerIndex = layerIndex
        self.session = session
        self.switchMLP = StreamedSwitchGLU(
            bank: session.bank,
            groupSize: session.configuration.groupSize,
            bits: session.configuration.bits)

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _sharedExpert.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let (inds, scores) = moeRouterTopK(gates, k: topK, normalize: normTopkProb)

        let tokenCount = x.size / x.dim(-1)
        let flatX = x.reshaped(tokenCount, x.dim(-1))
        let flatIndices = inds.reshaped(tokenCount, topK)
        let flatScores = scores.reshaped(tokenCount, topK)

        // The one mandatory synchronization of the design: the router's choice
        // has to be on the host before a single `pread` can be issued. It also
        // guarantees the previous layer's gather has been evaluated, which is
        // what lets the bank's scatter donate its buffer instead of copying
        // the whole bank.
        eval(flatIndices)
        let experts = flatIndices.asArray(UInt32.self).map { Int($0) }

        let resolution: StreamedExpertResolution
        do {
            resolution = try session.resolve(layer: layerIndex, experts: experts)
        } catch {
            // TD: `UnaryLayer` cannot throw, so a read failure mid-generation
            // has no path back to the caller. Phase 2 needs a cancellation
            // route through the generation loop; until then failing loudly
            // beats returning a silently wrong activation.
            fatalError("expert streaming failed on layer \(layerIndex): \(error)")
        }

        let indices = resolution.indices.reshaped(tokenCount, topK)
        let combined: MLXArray
        if let pools = resolution.pools {
            combined = switchMLP.callAndWeightedReduce(
                flatX, localIndices: indices, pools: pools, weights: flatScores)
        } else {
            combined = switchMLP.callAndWeightedReduce(
                flatX, slots: indices, weights: flatScores)
        }

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return combined.reshaped(x.shape) + sharedY
    }
}
