// FORK(JuanColilla): R-56 expert streaming — the `lfm2_moe` port.
//
// Two questions this file answers that the Qwen 3.5 suite could not ask,
// because Qwen 3.5 has experts in every layer:
//
//  1. Does the offset index survive a checkpoint whose MoE layers start at 2?
//     `num_dense_layers` puts a hole at the front, and every piece of
//     streaming state is keyed by a layer number. An index that renumbered
//     them by "n-th MoE layer" would send decoder layer 2 to the record of
//     layer 0 and multiply the wrong experts, silently.
//  2. Does a streamed LFM2.5 model produce the resident model's logits?
//     Unit tests cannot see a wrong layer mapping — every read would still be
//     a valid expert of a valid layer. Only the logits can.
//
// XCTest rather than Swift Testing: the `swift-testing` runner in this fork
// cannot find MLX's metallib (see AGENTS.md).

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class LFM2MoEStreamedTests: XCTestCase {

    // MARK: - Fixtures

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-lfm2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    /// A checkpoint shaped like `lfm2_moe`: `feed_forward.` instead of `mlp.`
    /// as the block prefix, dense layers 0 and 1 with no `switch_mlp`, and a
    /// per-layer `feed_forward.gate` that must not be confused with
    /// `switch_mlp.gate_proj`.
    private func gappedTensors(
        experts: Int, denseLayers: Int, totalLayers: Int
    ) -> [SyntheticExpertCheckpoint.Tensor] {
        var tensors = [SyntheticExpertCheckpoint.Tensor]()
        for layer in 0 ..< totalLayers {
            let block = "model.layers.\(layer).feed_forward"
            guard layer >= denseLayers else {
                // A dense layer: plain projections, no expert axis. The parser
                // must ignore these rather than index them as layer records.
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    tensors.append(
                        .init(
                            name: "\(block).\(projection).weight", dtype: "U32",
                            shape: [128, 64]))
                }
                continue
            }
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let base = "\(block).switch_mlp.\(projection)"
                tensors.append(
                    .init(name: "\(base).weight", dtype: "U32", shape: [experts, 64, 64]))
                tensors.append(
                    .init(name: "\(base).scales", dtype: "BF16", shape: [experts, 128, 64]))
                tensors.append(
                    .init(name: "\(base).biases", dtype: "BF16", shape: [experts, 128, 64]))
            }
            tensors.append(
                .init(name: "\(block).gate.weight", dtype: "BF16", shape: [experts, 16]))
            tensors.append(.init(name: "\(block).expert_bias", dtype: "F32", shape: [experts]))
        }
        return tensors
    }

    // MARK: - The hole at the front

    func testIndexKeepsTheCheckpointLayerNumbersAcrossDenseLayers() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.write(
            gappedTensors(experts: 8, denseLayers: 2, totalLayers: 5),
            to: directory.appending(path: "model.safetensors"))

        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        // Three MoE layers out of five, numbered 2…4 — not 0…2.
        XCTAssertEqual(index.layerCount, 3)
        XCTAssertEqual(index.expertCount, 8)
        XCTAssertEqual(index.layers.map(\.layer), [2, 3, 4])
        XCTAssertNil(index.records(forLayer: 0))
        XCTAssertNil(index.records(forLayer: 1))
        XCTAssertNotNil(index.records(forLayer: 2))
        XCTAssertNotNil(index.records(forLayer: 4))

        // The `feed_forward.` prefix is parsed, and the per-layer router
        // `feed_forward.gate.weight` is not mistaken for a projection.
        let record = try XCTUnwrap(index.records(forLayer: 3))
        XCTAssertEqual(record[ExpertPiece(.gate, .weight)].rowShape, [64, 64])
        XCTAssertEqual(record[ExpertPiece(.up, .scales)].rowShape, [128, 64])
    }

    func testStoreRefusesADenseLayerInsteadOfReadingTheWrongRows() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.write(
            gappedTensors(experts: 8, denseLayers: 2, totalLayers: 5),
            to: directory.appending(path: "model.safetensors"))
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let store = try ExpertResidencyStore(index: index, modelDirectory: directory)

        XCTAssertThrowsError(try store.readBatch(keys: [ExpertKey(layer: 0, expert: 0)])) { error in
            guard case ExpertResidencyError.unknownLayer(let layer) = error else {
                return XCTFail("expected unknownLayer, got \(error)")
            }
            XCTAssertEqual(layer, 0)
        }

        // And the layers that do exist read without complaint.
        let arrays = try store.readBatch(keys: [ExpertKey(layer: 2, expert: 0)])
        XCTAssertEqual(arrays.count, ExpertPiece.all.count)
    }

    // MARK: - Resident vs streamed, end to end

    /// `full_attn_idxs` covers every layer on purpose: the conv layers carry a
    /// `conv.weight` that `sanitize` transposes, so a checkpoint saved from a
    /// live model would not round-trip. `num_dense_layers: 2` with
    /// `num_hidden_layers: 4` is the point of the fixture — layers 0 and 1 are
    /// dense, 2 and 3 are streamed.
    static let configJSON = """
        {
          "model_type": "lfm2_moe",
          "vocab_size": 128,
          "hidden_size": 1024,
          "intermediate_size": 2048,
          "moe_intermediate_size": 512,
          "num_hidden_layers": 4,
          "num_experts": 16,
          "num_experts_per_tok": 4,
          "norm_topk_prob": true,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "max_position_embeddings": 4096,
          "use_expert_bias": true,
          "num_dense_layers": 2,
          "norm_eps": 0.00001,
          "conv_bias": false,
          "conv_L_cache": 3,
          "rope_theta": 1000000.0,
          "routed_scaling_factor": 1.0,
          "full_attn_idxs": [0, 1, 2, 3]
        }
        """

    private func configuration() throws -> LFM2MoEConfiguration {
        try JSONDecoder().decode(LFM2MoEConfiguration.self, from: Data(Self.configJSON.utf8))
    }

    private func writeCheckpoint() throws -> URL {
        let directory = try temporaryDirectory()
        MLXRandom.seed(56)
        let model = LFM2MoEModel(try configuration())
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)

        var weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) })
        // A real `expert_bias` rather than the zeros the initializer plants:
        // with zeros the bias branch of `lfm2MoeRoute` is indistinguishable
        // from its absence, and the selection it is supposed to change never
        // changes. Written into the checkpoint rather than into the live
        // module, which MLXNN refuses.
        for layer in 2 ... 3 {
            weights["model.layers.\(layer).feed_forward.expert_bias"] =
                MLXRandom.normal([16]) * 0.5
        }
        eval(Array(weights.values))
        XCTAssertNotNil(
            weights["model.layers.2.feed_forward.expert_bias"],
            "expert_bias must be a real parameter, otherwise this test never exercises it")
        XCTAssertNotNil(weights["model.layers.2.feed_forward.switch_mlp.gate_proj.weight"])
        XCTAssertNil(
            weights["model.layers.0.feed_forward.switch_mlp.gate_proj.weight"],
            "layer 0 is dense")
        try save(arrays: weights, url: directory.appending(path: "model.safetensors"))
        return directory
    }

    private func loadResident(from directory: URL) throws -> LFM2MoEModel {
        let model = LFM2MoEModel(try configuration())
        quantize(model: model, groupSize: 64, bits: 4)
        try loadWeights(modelDirectory: directory, model: model)
        return model
    }

    private func loadStreamed(
        from directory: URL, bankSlots: Int
    ) throws -> (model: LFM2MoEModel, session: ExpertStreamingSession) {
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let session = try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(bankCapacityBytes: bankSlots * index.bytesPerExpert))

        let model = try ExpertStreaming.withSession(session) {
            let model = LFM2MoEModel(try configuration())
            quantize(model: model, groupSize: 64, bits: 4)
            try loadWeights(modelDirectory: directory, model: model)
            return model
        }
        return (model, session)
    }

    func testCheckpointGeometryMatchesTheBankTemplate() throws {
        let directory = try writeCheckpoint()
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        XCTAssertEqual(index.expertCount, 16)
        XCTAssertEqual(index.layerCount, 2)
        XCTAssertEqual(index.layers.map(\.layer), [2, 3])
        // Scales and biases are float32 here, not bfloat16 as in a published
        // checkpoint: `quantize` keeps the dtype of the weights it is given.
        XCTAssertEqual(index.bytesPerExpert, 3 * (262_144 + 32_768 + 32_768))
    }

    func testStreamedLoadSkipsTheRoutedExpertsAndPicksTheStreamedBlock() throws {
        let directory = try writeCheckpoint()
        let (model, _) = try loadStreamed(from: directory, bankSlots: 16)

        XCTAssertTrue(model.model.layers[0].feedForward is LFM2MoEMLP)
        XCTAssertTrue(model.model.layers[1].feedForward is LFM2MoEMLP)

        for layer in 2 ... 3 {
            let block = try XCTUnwrap(
                model.model.layers[layer].feedForward as? Lfm2MoeStreamedSparseMoeBlock)
            // The decoder index, not the position among MoE layers. A 0 here
            // would be the silent failure this whole file exists to catch.
            XCTAssertEqual(block.layerIndex, layer)
        }

        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertFalse(keys.contains { $0.contains("switch_mlp") })
        XCTAssertTrue(keys.contains("model.layers.2.feed_forward.gate.weight"))
        XCTAssertTrue(keys.contains("model.layers.2.feed_forward.expert_bias"))
    }

    /// Teacher-forced logit comparison, the acceptance criterion of the port.
    ///
    /// Two streamed arms: a bank that holds every expert of both layers, and a
    /// bank of four slots, which cannot — so the second one evicts and
    /// re-reads between layers and between tokens. A layer/slot mix-up shows
    /// up in the small bank first.
    func testStreamedLogitsMatchTheResidentModel() throws {
        let directory = try writeCheckpoint()
        let prompt = MLXArray([3, 17, 42, 5, 91, 60, 7, 23] as [Int32]).reshaped(1, 8)
        let forcing: [Int32] = [11, 64, 2, 88, 30, 19, 45, 7]

        let resident = try loadResident(from: directory)
        let reference = try run(model: resident, prompt: prompt, forcing: forcing)

        for slots in [32, 4] {
            let (model, session) = try loadStreamed(from: directory, bankSlots: slots)
            let streamed = try run(model: model, prompt: prompt, forcing: forcing)

            XCTAssertNil(session.lastFailure, "streaming failed with a \(slots)-slot bank")
            XCTAssertGreaterThan(session.bank.statistics.hits + session.bank.statistics.misses, 0)

            var worstRelative: Float = 0
            for (a, b) in zip(reference, streamed) {
                let difference = MLX.max(MLX.abs(a - b))
                let scale = MLX.max(MLX.abs(a))
                eval(difference, scale)
                worstRelative = max(
                    worstRelative, difference.item(Float.self) / scale.item(Float.self))
            }
            print(
                "R56 | LFM2 streamed vs resident, \(slots)-slot bank: "
                    + "worst relative logit difference \(String(format: "%.7f", worstRelative))")
            XCTAssertLessThan(
                worstRelative, 1e-4,
                "streamed logits diverged from the resident model with a \(slots)-slot bank")
        }
    }

    /// Prefill plus teacher-forced decode steps; one logits row per step.
    private func run(
        model: LFM2MoEModel, prompt: MLXArray, forcing: [Int32]
    ) throws -> [MLXArray] {
        let cache = try model.newCache(parameters: nil)
        var rows = [MLXArray]()

        var logits = model(prompt, cache: cache)
        rows.append(logits[0..., -1, 0...].asType(.float32))
        eval(rows.last!)

        for token in forcing {
            logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)
            rows.append(logits[0..., -1, 0...].asType(.float32))
            eval(rows.last!)
        }
        return rows
    }
}
