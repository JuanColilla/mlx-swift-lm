// FORK(JuanColilla): R-56 expert streaming — the compute side.
//
// The quantized gather matmul does not care where its weights came from: it
// indexes axis 0 of `w`, and `scales`/`biases` only have to share `w`'s batch
// dimensions. A bank of `[slots, …]` satisfies that by construction, which is
// why streaming needs no new kernel — only a different set of arrays and a
// translation from expert ids to slot ids.
//
// Deliberately NOT a subclass of `SwitchGLU`/`QuantizedSwitchLinear`:
//
//  * `SwitchGLU.init` allocates three dense `[numExperts, out, in]` weights,
//    which for a bank-sized module is exactly the memory streaming exists to
//    avoid.
//  * A registered parameter would be checked by `loadWeights`'s
//    `verify: [.all]` against a checkpoint that no longer carries those keys.
//  * `supportsDirectWeightedReduction` compares the *exact* type of each
//    projection with `ObjectIdentifier(QuantizedSwitchLinear.self)`, so even a
//    subclass would not qualify for the fused prefill kernel.

import Foundation
import MLX
import MLXNN

/// Bank-backed replacement for `SwitchGLU` in a streamed MoE layer.
///
/// Holds no parameters: the arrays it multiplies with belong to the slot bank
/// and change under it as experts are installed.
public final class StreamedSwitchGLU: Module, @unchecked Sendable {
    public let bank: ExpertSlotBank
    public let groupSize: Int
    public let bits: Int

    private let activationProduct: @Sendable (MLXArray, MLXArray) -> MLXArray

    public init(bank: ExpertSlotBank, groupSize: Int, bits: Int) {
        self.bank = bank
        self.groupSize = groupSize
        self.bits = bits
        self.activationProduct = compiledSiluProduct
        super.init()
    }

    /// Project through the slot bank and reduce, mirroring
    /// `SwitchGLU.callAndWeightedReduce`'s established path.
    ///
    /// The fused reduction kernel is not used here, and loses nothing in
    /// decode: it requires `indices.size >= 64`, which a single token with
    /// top-8 never reaches — the resident path already falls back to
    /// `weightedExpertSum` there.
    public func callAndWeightedReduce(
        _ x: MLXArray, slots: MLXArray, weights: MLXArray
    ) -> MLXArray {
        weightedExpertSum(project(x, slots, pools: bankPools), weights)
    }

    /// The prefill variant: the experts live in transient staged arrays rather
    /// than in the bank, so the sweep never evicts the decode working set.
    public func callAndWeightedReduce(
        _ x: MLXArray, localIndices: MLXArray, pools: [MLXArray], weights: MLXArray
    ) -> MLXArray {
        weightedExpertSum(project(x, localIndices, pools: pools), weights)
    }

    private var bankPools: [MLXArray] {
        ExpertPiece.all.map { bank.pool($0) }
    }

    private func project(_ x: MLXArray, _ indices: MLXArray, pools: [MLXArray]) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64
        var idx = indices
        var inverseOrder = MLXArray()
        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = matmul(.up, x, idx, pools: pools, sorted: doSort)
        let xGate = matmul(.gate, x, idx, pools: pools, sorted: doSort)
        var out = matmul(
            .down, activationProduct(xGate, xUp), idx, pools: pools, sorted: doSort)

        if doSort {
            out = scatterUnsort(x: out, invOrder: inverseOrder, shape: indices.shape)
        }
        return MLX.squeezed(out, axis: -2)
    }

    private func matmul(
        _ projection: ExpertProjection, _ x: MLXArray, _ indices: MLXArray,
        pools: [MLXArray], sorted: Bool
    ) -> MLXArray {
        MLX.gatherQuantizedMM(
            x,
            pools[ExpertPiece(projection, .weight).slot],
            scales: pools[ExpertPiece(projection, .scales).slot],
            biases: pools[ExpertPiece(projection, .biases).slot],
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            sortedIndices: sorted)
    }
}

/// The routing decision, resolved on the host, plus the indices the matmul
/// needs. Shared by every streamed MoE block so the policy lives in one place.
public struct StreamedExpertResolution {
    /// Indices into whichever pools `pools` holds.
    public let indices: MLXArray
    /// The arrays to multiply with: the bank, or transient staged arrays.
    public let pools: [MLXArray]?

    public init(indices: MLXArray, pools: [MLXArray]?) {
        self.indices = indices
        self.pools = pools
    }
}

extension ExpertStreamingSession {

    /// Resolve one layer's routing choice to slot or staged indices.
    ///
    /// Two regimes, and the split is the point:
    ///
    ///  * **Decode.** A handful of unique experts; they go through the bank,
    ///    which is what makes the next token cheap.
    ///  * **Prefill.** A prompt of a few hundred tokens selects nearly every
    ///    expert of every layer. Admitting that sweep would leave the bank
    ///    holding the last experts seen instead of the working set, so by
    ///    default it is read into transient staging and dropped.
    ///
    /// `experts` must already be the flattened, host-side routing choice.
    public func resolve(layer: Int, experts: [Int]) throws -> StreamedExpertResolution {
        let unique = Array(Set(experts)).sorted()
        let useBank = unique.count <= bank.slotCount && (experts.count <= 8 || configuration.admitOnSweep)

        if useBank {
            let slots = try bank.ensure(
                keys: unique.map { ExpertKey(layer: layer, expert: $0) })
            let mapping = Dictionary(uniqueKeysWithValues: zip(unique, slots))
            return StreamedExpertResolution(
                indices: MLXArray(experts.map { UInt32(mapping[$0]!) }),
                pools: nil)
        }

        let (arrays, sortedExperts) = try store.readRuns(layer: layer, experts: unique)
        var local = [Int: UInt32]()
        for (row, expert) in sortedExperts.enumerated() { local[expert] = UInt32(row) }
        return StreamedExpertResolution(
            indices: MLXArray(experts.map { local[$0]! }),
            pools: arrays)
    }
}
