// FORK(JuanColilla): R-56 expert streaming — the K2-Horizon MoVA port.
//
// What this file proves that the Qwen 3.5 and LFM2 suites cannot:
//
//  1. Two families of routed experts per layer — MLP experts and value
//     experts — each in its own bank, both skipped by the resident load.
//     The failure this guards against is silent: a `switch_v` the load
//     filter did not recognize is materialized resident, and the model
//     still produces the right logits.
//  2. Expert rows that are not page multiples (the MLP `scales`/`biases`
//     here, as in the real 36B) go through the padded staging path.
//  3. A streamed K2 model produces the resident model's logits, with a bank
//     that holds everything and one that has to evict.
//
// XCTest rather than Swift Testing: the `swift-testing` runner in this fork
// cannot find MLX's metallib (see AGENTS.md).

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class K2HorizonStreamedTests: XCTestCase {

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-k2-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    /// Sized so the MLP and value expert rows mix page-aligned and unaligned
    /// pieces once quantized at group 64 / 4 bits (scales stay float32 here,
    /// `quantize` keeps the dtype it is given):
    ///
    ///  * `gate_proj`/`up_proj` weight `[256, 512/8]` u32 → 16 KiB, aligned;
    ///    scales `[256, 8]` f32 → 8 KiB, not aligned.
    ///  * `switch_v` weight `[128, 64]` u32 → 32 KiB, aligned; scales
    ///    `[128, 8]` f32 → 4 KiB, not aligned.
    ///
    /// `vocab_size` 128 because the prompt below uses ids up to 91: an id past
    /// the vocabulary gathers out of bounds and returns garbage that differs
    /// between two otherwise identical models.
    ///
    /// Layer 0 is dense (`mlp_only_layers`), layers 1 and 2 are sparse with
    /// MoVA, so the two families are keyed by the decoder index 1…2.
    static let configJSON = """
        {
          "model_type": "k2_horizon",
          "vocab_size": 128,
          "hidden_size": 512,
          "intermediate_size": 384,
          "moe_intermediate_size": 256,
          "num_hidden_layers": 3,
          "mlp_only_layers": [0],
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 64,
          "rope_parameters": { "rope_theta": 10000000.0 },
          "layernorm_num_groups": 2,
          "query_key_norm": false,
          "attention_gate_func": "softplus",
          "num_experts": 6,
          "num_experts_per_tok": 2,
          "num_shared_experts": 1,
          "norm_topk_prob": true,
          "moe_gate_bias": true,
          "router_score_func": "sigmoid",
          "router_scaling_factor": 2.5,
          "mova_num_experts": 5,
          "mova_num_experts_per_tok": 2,
          "rms_norm_eps": 1e-6,
          "tie_word_embeddings": false
        }
        """

    private func configuration() throws -> K2HorizonConfiguration {
        try JSONDecoder().decode(K2HorizonConfiguration.self, from: Data(Self.configJSON.utf8))
    }

    private func writeCheckpoint() throws -> URL {
        let directory = try temporaryDirectory()
        MLXRandom.seed(56)
        let model = K2HorizonModel(try configuration())
        var parameters = [String: MLXArray]()
        for (key, value) in model.parameters().flattened() {
            parameters[key] = MLXRandom.normal(value.shape, scale: 0.2).asType(value.dtype)
        }
        try model.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)

        let weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) })
        XCTAssertNotNil(weights["model.layers.1.mlp.switch_mlp.gate_proj.scales"])
        XCTAssertNotNil(weights["model.layers.1.self_attn.switch_v.weight"])
        XCTAssertNotNil(weights["model.layers.1.self_attn.v_router.bias"])
        XCTAssertNil(weights["model.layers.0.mlp.switch_mlp.gate_proj.weight"], "layer 0 is dense")
        XCTAssertNil(weights["model.layers.0.self_attn.switch_v.weight"], "layer 0 is dense")
        try save(arrays: weights, url: directory.appending(path: "model.safetensors"))
        return directory
    }

    private func loadResident(from directory: URL) throws -> K2HorizonModel {
        let model = K2HorizonModel(try configuration())
        quantize(model: model, groupSize: 64, bits: 4)
        try loadWeights(modelDirectory: directory, model: model)
        return model
    }

    private func loadStreamed(
        from directory: URL, mlpSlots: Int, valueSlots: Int
    ) throws -> (model: K2HorizonModel, session: ExpertStreamingSession) {
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let session = try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(
                bankCapacityBytes: 0,
                bankCapacityBytesPerFamily: [
                    .mlp: mlpSlots * index.mlp.bytesPerExpert,
                    .value: valueSlots * index.family(.value)!.bytesPerExpert,
                ]))

        let model = try ExpertStreaming.withSession(session) {
            let model = K2HorizonModel(try configuration())
            quantize(model: model, groupSize: 64, bits: 4)
            try loadWeights(modelDirectory: directory, model: model)
            return model
        }
        return (model, session)
    }

    // MARK: - Geometry

    func testCheckpointHasBothFamiliesWithMixedAlignment() throws {
        let directory = try writeCheckpoint()
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        XCTAssertEqual(index.families.map(\.family), [.mlp, .value])
        let mlp = index.mlp
        XCTAssertEqual(mlp.expertCount, 6)
        XCTAssertEqual(mlp.layers.map(\.layer), [1, 2])
        XCTAssertTrue(mlp.template[ExpertPiece(.gate, .weight)].isPageAligned)
        XCTAssertFalse(mlp.template[ExpertPiece(.gate, .scales)].isPageAligned)
        XCTAssertEqual(mlp.template[ExpertPiece(.gate, .scales)].rowBytes, 8192)

        let value = try index.requireFamily(.value)
        XCTAssertEqual(value.expertCount, 5)
        XCTAssertEqual(value.layers.map(\.layer), [1, 2])
        XCTAssertEqual(value.template[ExpertPiece(.value, .weight)].rowShape, [128, 64])
        XCTAssertTrue(value.template[ExpertPiece(.value, .weight)].isPageAligned)
        XCTAssertFalse(value.template[ExpertPiece(.value, .scales)].isPageAligned)
        XCTAssertEqual(value.template[ExpertPiece(.value, .scales)].rowBytes, 4096)

        XCTAssertNoThrow(try index.validateQuantization(groupSize: 64, bits: 4))
    }

    // MARK: - Construction and load

    func testStreamedLoadSkipsBothFamiliesAndPicksTheStreamedTypes() throws {
        let directory = try writeCheckpoint()
        let (model, session) = try loadStreamed(from: directory, mlpSlots: 12, valueSlots: 10)

        let dense = try XCTUnwrap(model.model.layers[0] as? K2HorizonDecoderLayer)
        XCTAssertFalse(dense.usesStreamedExperts)
        XCTAssertTrue(dense.mlp is K2HorizonMLP)
        XCTAssertTrue(dense.selfAttn is K2HorizonAttention)

        for layer in 1 ... 2 {
            let decoder = try XCTUnwrap(model.model.layers[layer] as? K2HorizonDecoderLayer)
            XCTAssertTrue(decoder.usesStreamedExperts)
            let moe = try XCTUnwrap(decoder.mlp as? K2HorizonStreamedSparseMoEBlock)
            XCTAssertEqual(moe.layerIndex, layer)
            XCTAssertNotNil(moe.sharedExperts)
            let attention = try XCTUnwrap(decoder.selfAttn as? K2HorizonStreamedAttention)
            XCTAssertEqual(attention.layerIndex, layer)
        }

        // Neither family's stacked tensors are parameters of the streamed
        // model. A `switch_v` here would mean 3,8 GB resident on the real
        // checkpoint with nothing reporting it.
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertFalse(keys.contains { $0.contains("switch_mlp") }, "switch_mlp loaded resident")
        XCTAssertFalse(keys.contains { $0.contains("switch_v") }, "switch_v loaded resident")
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.gate.bias"))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.shared_experts.up_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.self_attn.v_router.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.self_attn.v_router.bias"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.v_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.mlp.gate_proj.weight"))

        XCTAssertEqual(session.bank.slotCount, 12)
        XCTAssertEqual(session.bank(for: .value)?.slotCount, 10)
    }

    /// Without a session the same checkpoint loads resident, `switch_v`
    /// included — the control for the test above.
    func testResidentLoadKeepsBothFamilies() throws {
        let directory = try writeCheckpoint()
        let model = try loadResident(from: directory)
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("model.layers.1.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(keys.contains("model.layers.1.self_attn.switch_v.weight"))
        let decoder = try XCTUnwrap(model.model.layers[1] as? K2HorizonDecoderLayer)
        XCTAssertFalse(decoder.usesStreamedExperts)
        XCTAssertTrue(decoder.selfAttn is K2HorizonAttention)
    }

    // MARK: - Resident vs streamed, end to end

    /// Teacher-forced logit comparison, the acceptance criterion of the port.
    ///
    /// Two streamed arms: banks that hold every expert of both families, and
    /// banks of two slots per family — the top-K of both routers — which
    /// cannot, so they evict and re-read between layers and between tokens.
    /// A layer/slot/family mix-up shows up in the small banks first.
    func testStreamedLogitsMatchTheResidentModel() throws {
        let directory = try writeCheckpoint()
        let prompt = MLXArray([3, 17, 42, 5, 91, 60, 7, 23] as [Int32]).reshaped(1, 8)
        let forcing: [Int32] = [11, 64, 2, 88, 30, 19, 45, 7]

        let resident = try loadResident(from: directory)
        let reference = try run(model: resident, prompt: prompt, forcing: forcing)

        for (mlpSlots, valueSlots) in [(12, 10), (2, 2)] {
            let (model, session) = try loadStreamed(
                from: directory, mlpSlots: mlpSlots, valueSlots: valueSlots)
            let streamed = try run(model: model, prompt: prompt, forcing: forcing)

            XCTAssertNil(session.lastFailure, "streaming failed with \(mlpSlots)/\(valueSlots) slots")
            let mlpStatistics = session.bank.statistics
            let valueStatistics = try XCTUnwrap(session.bank(for: .value)).statistics
            XCTAssertGreaterThan(mlpStatistics.hits + mlpStatistics.misses, 0)
            XCTAssertGreaterThan(valueStatistics.hits + valueStatistics.misses, 0)
            if valueSlots == 2 {
                XCTAssertGreaterThan(valueStatistics.evictions, 0, "the small value bank never evicted")
                XCTAssertGreaterThan(mlpStatistics.evictions, 0, "the small MLP bank never evicted")
            }
            // Every streamed step reads its router back: two per sparse
            // layer (value, then MLP) per forward pass.
            XCTAssertEqual(session.syncCounters.routerEvals, 2 * 2 * (1 + forcing.count))

            var worstRelative: Float = 0
            for (a, b) in zip(reference, streamed) {
                let difference = MLX.max(MLX.abs(a - b))
                let scale = MLX.max(MLX.abs(a))
                eval(difference, scale)
                worstRelative = max(
                    worstRelative, difference.item(Float.self) / scale.item(Float.self))
            }
            print(
                "R56 | K2 streamed vs resident, \(mlpSlots) MLP / \(valueSlots) value slots: "
                    + "worst relative logit difference \(String(format: "%.7f", worstRelative)) "
                    + "| MLP hit \(String(format: "%.0f%%", mlpStatistics.hitRate * 100)) "
                    + "| value hit \(String(format: "%.0f%%", valueStatistics.hitRate * 100))")
            XCTAssertLessThan(
                worstRelative, 1e-4,
                "streamed logits diverged from the resident model with \(mlpSlots)/\(valueSlots) slots")
        }
    }

    /// The negative control: an index whose value-expert records of the two
    /// sparse layers are exchanged makes the streamed model read layer 2's
    /// value experts for layer 1 and vice versa. The logits must move, or
    /// the comparison above proves nothing about the value family.
    func testSwappedValueLayersDivergeFromTheResidentModel() throws {
        let directory = try writeCheckpoint()
        let prompt = MLXArray([3, 17, 42, 5, 91, 60, 7, 23] as [Int32]).reshaped(1, 8)
        let forcing: [Int32] = [11, 64, 2, 88]

        let resident = try loadResident(from: directory)
        let reference = try run(model: resident, prompt: prompt, forcing: forcing)

        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let value = try index.requireFamily(.value)
        let swapped = ExpertOffsetIndex(
            fingerprint: index.fingerprint, shardFiles: index.shardFiles,
            families: [
                index.mlp,
                ExpertFamilyIndex(
                    family: .value, expertCount: value.expertCount,
                    layers: [
                        ExpertLayerRecords(layer: 1, pieces: value.layers[1].pieces),
                        ExpertLayerRecords(layer: 2, pieces: value.layers[0].pieces),
                    ]),
            ])
        let session = try ExpertStreamingSession(
            index: swapped, modelDirectory: directory,
            configuration: .init(
                bankCapacityBytes: 0,
                bankCapacityBytesPerFamily: [
                    .mlp: 12 * index.mlp.bytesPerExpert, .value: 10 * value.bytesPerExpert,
                ]))
        let model = try ExpertStreaming.withSession(session) {
            let model = K2HorizonModel(try configuration())
            quantize(model: model, groupSize: 64, bits: 4)
            try loadWeights(modelDirectory: directory, model: model)
            return model
        }
        let streamed = try run(model: model, prompt: prompt, forcing: forcing)

        var worstRelative: Float = 0
        for (a, b) in zip(reference, streamed) {
            let difference = MLX.max(MLX.abs(a - b))
            let scale = MLX.max(MLX.abs(a))
            eval(difference, scale)
            worstRelative = max(worstRelative, difference.item(Float.self) / scale.item(Float.self))
        }
        print(
            "R56 | K2 control (value layers swapped): worst relative logit difference "
                + String(format: "%.5f", worstRelative))
        XCTAssertNil(session.lastFailure)
        XCTAssertGreaterThan(
            worstRelative, 1e-3, "swapping the value experts of two layers changed nothing")
    }

    /// Prefill plus teacher-forced decode steps; one logits row per step.
    private func run(
        model: K2HorizonModel, prompt: MLXArray, forcing: [Int32]
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
