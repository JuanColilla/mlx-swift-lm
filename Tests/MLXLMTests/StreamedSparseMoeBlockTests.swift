// FORK(JuanColilla): R-56 tasks 1.5 and 1.6 — the streamed MoE block, the
// load hook, and the construction-time factory.
//
// The load-bearing assertion of the whole feature is here: a model whose
// routed experts are read from disk one batch at a time must produce the same
// logits as the same model held resident. Everything else is plumbing.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class StreamedSparseMoeBlockTests: XCTestCase {

    /// Sized so every expert row is a whole number of 16 KiB pages, which the
    /// offset index requires: with `group_size` 64 and 4 bits, a `scales` row
    /// is `outputDims * inputDims / 64 * 2` bytes, so 1024/512 hidden and
    /// intermediate dimensions are the smallest convenient pair that works.
    ///
    /// `full_attention_interval: 1` makes every layer full attention: the
    /// gated DeltaNet layers carry a `conv1d.weight` that `sanitize` rewrites,
    /// so a checkpoint saved from a live model would not round-trip.
    static let configJSON = """
        {
          "model_type": "qwen3_5_text",
          "hidden_size": 1024,
          "num_hidden_layers": 2,
          "intermediate_size": 2048,
          "num_attention_heads": 8,
          "num_key_value_heads": 2,
          "head_dim": 64,
          "vocab_size": 128,
          "full_attention_interval": 1,
          "num_experts": 16,
          "num_experts_per_tok": 8,
          "moe_intermediate_size": 512,
          "shared_expert_intermediate_size": 512,
          "norm_topk_prob": true,
          "tie_word_embeddings": false
        }
        """

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-model-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func configuration() throws -> Qwen35TextConfiguration {
        try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(Self.configJSON.utf8))
    }

    /// Builds a quantized model, writes it as a safetensors checkpoint, and
    /// returns the directory. Because the text model's `sanitize` does not
    /// rename anything, its parameter paths are the checkpoint keys.
    private func writeCheckpoint() throws -> URL {
        let directory = try temporaryDirectory()
        MLXRandom.seed(56)
        let model = Qwen35TextModel(try configuration())
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)

        let weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) })
        try save(arrays: weights, url: directory.appending(path: "model.safetensors"))
        return directory
    }

    private func loadResident(from directory: URL) throws -> Qwen35TextModel {
        let model = Qwen35TextModel(try configuration())
        quantize(model: model, groupSize: 64, bits: 4)
        try loadWeights(modelDirectory: directory, model: model)
        return model
    }

    private func loadStreamed(
        from directory: URL, bankSlots: Int, admitOnSweep: Bool = false
    ) throws -> (model: Qwen35TextModel, session: ExpertStreamingSession) {
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let session = try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(
                bankCapacityBytes: bankSlots * index.bytesPerExpert,
                admitOnSweep: admitOnSweep))

        let model = try ExpertStreaming.withSession(session) {
            let model = Qwen35TextModel(try configuration())
            quantize(model: model, groupSize: 64, bits: 4)
            try loadWeights(modelDirectory: directory, model: model)
            return model
        }
        return (model, session)
    }

    // MARK: - Index over a saved checkpoint

    func testCheckpointRowsArePageAligned() throws {
        let directory = try writeCheckpoint()
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        XCTAssertEqual(index.expertCount, 16)
        XCTAssertEqual(index.layerCount, 2)
        // Scales and biases are float32 here, not bfloat16 as in a published
        // checkpoint: `quantize` keeps the dtype of the weights it is given,
        // and a model built from `MLXRandom` is float32. Both are page
        // multiples, which is what the index requires.
        XCTAssertEqual(index.bytesPerExpert, 3 * (262_144 + 32_768 + 32_768))
    }

    // MARK: - The load hook

    func testStreamedLoadSkipsTheRoutedExperts() throws {
        let directory = try writeCheckpoint()
        let (model, session) = try loadStreamed(from: directory, bankSlots: 16)

        let keys = model.parameters().flattened().map(\.0)
        XCTAssertFalse(
            keys.contains { ExpertOffsetIndex.isRoutedExpertKey($0) },
            "a streamed model must not hold routed expert parameters")
        XCTAssertTrue(keys.contains { $0.contains("shared_expert.gate_proj.weight") })
        XCTAssertTrue(keys.contains { $0.contains("mlp.gate.weight") })
        XCTAssertEqual(session.bank.slotCount, 16)
    }

    func testActivationIsScopedToConstruction() throws {
        let directory = try writeCheckpoint()
        _ = try loadStreamed(from: directory, bankSlots: 16)
        XCTAssertNil(
            ExpertStreaming.activeSession,
            "the session must not outlive the load that needed it")

        let resident = try loadResident(from: directory)
        XCTAssertTrue(
            resident.parameters().flattened().map(\.0)
                .contains { ExpertOffsetIndex.isRoutedExpertKey($0) },
            "with no session active the resident path must be unchanged")
    }

    // MARK: - Equivalence

    /// Prefill: the sweep goes through transient staging, so its result must
    /// still equal the resident model's.
    func testStreamedPrefillMatchesResident() throws {
        let directory = try writeCheckpoint()
        let resident = try loadResident(from: directory)
        let (streamed, _) = try loadStreamed(from: directory, bankSlots: 8)

        let tokens = MLXArray((0 ..< 24).map { Int32(($0 * 7) % 128) }).reshaped(1, 24)
        let expected = resident(tokens, cache: nil)
        let actual = streamed(tokens, cache: nil)
        eval(expected, actual)

        XCTAssertEqual(actual.shape, expected.shape)
        assertClose(actual, expected)
    }

    /// Decode: every token goes through the bank, with a bank far smaller than
    /// the expert count so eviction and reinstallation actually happen.
    func testStreamedDecodeMatchesResidentAcrossEvictions() throws {
        let directory = try writeCheckpoint()
        let resident = try loadResident(from: directory)
        let (streamed, session) = try loadStreamed(from: directory, bankSlots: 10)

        let residentCache = try resident.newCache(parameters: nil)
        let streamedCache = try streamed.newCache(parameters: nil)

        let prompt = MLXArray((0 ..< 8).map { Int32($0 * 3) }).reshaped(1, 8)
        _ = resident(prompt, cache: residentCache)
        _ = streamed(prompt, cache: streamedCache)

        for step in 0 ..< 12 {
            let token = MLXArray([Int32((step * 11 + 5) % 128)]).reshaped(1, 1)
            let expected = resident(token, cache: residentCache)
            let actual = streamed(token, cache: streamedCache)
            eval(expected, actual)
            assertClose(actual, expected, message: "decode step \(step)")
        }

        let statistics = session.bank.statistics
        XCTAssertGreaterThan(statistics.misses, 0)
        XCTAssertGreaterThan(
            statistics.evictions, 0,
            "a 10-slot bank over 2 layers x 16 experts must have evicted")
    }

    /// With the bank big enough to hold every expert of every layer, the
    /// second pass over the same tokens must be pure hits.
    func testFullyResidentBankReachesTotalHitRate() throws {
        let directory = try writeCheckpoint()
        let (streamed, session) = try loadStreamed(
            from: directory, bankSlots: 32, admitOnSweep: true)

        let cache = try streamed.newCache(parameters: nil)
        let prompt = MLXArray((0 ..< 32).map { Int32($0 * 5) }).reshaped(1, 32)
        _ = streamed(prompt, cache: cache)
        eval(streamed(MLXArray([Int32(3)]).reshaped(1, 1), cache: cache))

        session.bank.resetStatistics()
        for step in 0 ..< 8 {
            eval(
                streamed(
                    MLXArray([Int32((step * 11 + 5) % 128)]).reshaped(1, 1), cache: cache))
        }

        XCTAssertEqual(
            session.bank.statistics.misses, 0,
            "every expert of every layer fits; nothing should miss")
        XCTAssertEqual(session.bank.statistics.hitRate, 1)
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: MLXArray, message: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let difference = MLX.max(MLX.abs(actual.asType(.float32) - expected.asType(.float32)))
        eval(difference)
        XCTAssertLessThan(
            difference.item(Float.self), 1e-3,
            message.isEmpty ? "streamed output diverged" : message,
            file: file, line: line)
    }
}
