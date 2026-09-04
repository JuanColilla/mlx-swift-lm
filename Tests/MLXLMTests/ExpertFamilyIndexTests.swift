// FORK(JuanColilla): R-56 — the second expert family (K2-Horizon MoVA's
// `switch_v`) in the offset index, and the capacity split between families.
//
// No MLXArray is created anywhere in this file on purpose: index construction
// and validation must be verifiable where no Metal device exists.

import Foundation
import XCTest

@testable import MLXLMCommon

final class ExpertFamilyIndexTests: XCTestCase {

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-family-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func writeMoVA(
        mlpExperts: Int = 6, valueExperts: Int = 4, denseLayers: Int = 1, totalLayers: Int = 3
    ) throws -> URL {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.write(
            SyntheticExpertCheckpoint.movaTensors(
                mlpExperts: mlpExperts, valueExperts: valueExperts, denseLayers: denseLayers,
                totalLayers: totalLayers),
            to: directory.appending(path: "model.safetensors"))
        return directory
    }

    // MARK: - Key classification

    func testParsesValueExpertKeys() throws {
        let parsed = try ExpertOffsetIndex.parse(key: "model.layers.3.self_attn.switch_v.scales")
        XCTAssertEqual(parsed?.layer, 3)
        XCTAssertEqual(parsed?.piece, ExpertPiece(.value, .scales))
        XCTAssertEqual(parsed?.piece.family, .value)
        XCTAssertEqual(parsed?.piece.slot, 1)

        XCTAssertNil(try ExpertOffsetIndex.parse(key: "model.layers.3.self_attn.v_router.weight"))
        XCTAssertNil(try ExpertOffsetIndex.parse(key: "model.layers.0.self_attn.v_proj.weight"))
    }

    /// The failure this guards against is silent: a `switch_v` key that
    /// `isRoutedExpertKey` does not recognize is not an error, it is 3,8 GB
    /// of value experts materialized resident by `loadWeights` while the
    /// session believes it is streaming them.
    func testValueExpertKeysAreRoutedExpertKeys() {
        for component in ["weight", "scales", "biases"] {
            XCTAssertTrue(
                ExpertOffsetIndex.isRoutedExpertKey("model.layers.3.self_attn.switch_v.\(component)"),
                "switch_v.\(component) must be filtered out of the resident load")
        }
        XCTAssertTrue(
            ExpertOffsetIndex.isRoutedExpertKey(
                "model.language_model.layers.3.self_attn.switch_v.weight"))
        XCTAssertFalse(ExpertOffsetIndex.isRoutedExpertKey("model.layers.3.self_attn.v_router.weight"))
        XCTAssertFalse(ExpertOffsetIndex.isRoutedExpertKey("model.layers.3.self_attn.v_router.bias"))
        XCTAssertFalse(ExpertOffsetIndex.isRoutedExpertKey("model.layers.0.self_attn.v_proj.weight"))
    }

    func testPieceSlotsAreFamilyLocal() {
        XCTAssertEqual(ExpertFamily.mlp.pieces.count, 9)
        XCTAssertEqual(ExpertFamily.value.pieces.count, 3)
        XCTAssertEqual(ExpertFamily.mlp.pieces.map(\.slot), Array(0 ..< 9))
        XCTAssertEqual(ExpertFamily.value.pieces.map(\.slot), [0, 1, 2])
        XCTAssertEqual(ExpertPiece.all, ExpertFamily.mlp.pieces)
        // Attention runs before the feed-forward block within a layer.
        XCTAssertLessThan(ExpertFamily.value, ExpertFamily.mlp)
    }

    // MARK: - Construction

    func testBuildsBothFamiliesOverTheMoVAFixture() throws {
        let directory = try writeMoVA()
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        XCTAssertEqual(index.families.map(\.family), [.mlp, .value])

        let mlp = try index.requireFamily(.mlp)
        XCTAssertEqual(mlp.expertCount, 6)
        XCTAssertEqual(mlp.layers.map(\.layer), [1, 2])
        XCTAssertEqual(mlp.bytesPerExpert, 3 * (16384 + 8192 + 8192))
        XCTAssertNil(mlp.records(forLayer: 0))

        let value = try index.requireFamily(.value)
        XCTAssertEqual(value.expertCount, 4)
        XCTAssertEqual(value.layers.map(\.layer), [1, 2])
        XCTAssertEqual(value.template.pieces.count, 3)
        XCTAssertEqual(value.template[ExpertPiece(.value, .weight)].rowShape, [48, 64])
        XCTAssertEqual(value.template[ExpertPiece(.value, .weight)].rowBytes, 12288)
        XCTAssertEqual(value.bytesPerExpert, 12288 + 4096 + 4096)

        // The single-family conveniences describe the MLP family; the routed
        // total covers both.
        XCTAssertEqual(index.expertCount, 6)
        XCTAssertEqual(index.layerCount, 2)
        XCTAssertEqual(index.bytesPerExpert, mlp.bytesPerExpert)
        XCTAssertEqual(index.routedBytes, mlp.routedBytes + value.routedBytes)
        XCTAssertEqual(index.routedBytes, 2 * (6 * mlp.bytesPerExpert + 4 * value.bytesPerExpert))

        // Two families, two expert counts: this is what the single-count
        // index of Phase 1 would have rejected as inconsistent.
        XCTAssertNotEqual(mlp.expertCount, value.expertCount)

        // The rows that are not page multiples are recorded as such.
        XCTAssertTrue(mlp.template[ExpertPiece(.gate, .weight)].isPageAligned)
        XCTAssertFalse(mlp.template[ExpertPiece(.gate, .scales)].isPageAligned)
        XCTAssertFalse(value.template[ExpertPiece(.value, .weight)].isPageAligned)
    }

    func testSingleFamilyCheckpointHasNoValueFamily() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.writeWellFormed(experts: 4, layers: 2, to: directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        XCTAssertEqual(index.families.map(\.family), [.mlp])
        XCTAssertNil(index.family(.value))
        XCTAssertThrowsError(try index.requireFamily(.value)) { error in
            guard case ExpertOffsetIndexError.missingFamily(.value) = error else {
                return XCTFail("expected missingFamily, got \(error)")
            }
        }
    }

    func testValueFamilyAloneIsNotARoutedCheckpoint() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.write(
            [
                .init(name: "model.layers.0.self_attn.switch_v.weight", dtype: "U32",
                    shape: [4, 48, 64]),
                .init(name: "model.layers.0.self_attn.switch_v.scales", dtype: "BF16",
                    shape: [4, 32, 64]),
                .init(name: "model.layers.0.self_attn.switch_v.biases", dtype: "BF16",
                    shape: [4, 32, 64]),
            ], to: directory.appending(path: "model.safetensors"))

        XCTAssertThrowsError(try ExpertOffsetIndex.build(modelDirectory: directory)) { error in
            guard case ExpertOffsetIndexError.noRoutedExperts = error else {
                return XCTFail("expected noRoutedExperts, got \(error)")
            }
        }
    }

    func testIncompleteValueFamilyIsRejected() throws {
        let directory = try temporaryDirectory()
        var tensors = SyntheticExpertCheckpoint.wellFormedTensors(experts: 4, layers: 1)
        tensors.append(
            .init(name: "model.layers.0.self_attn.switch_v.weight", dtype: "U32", shape: [4, 48, 64]))
        try SyntheticExpertCheckpoint.write(
            tensors, to: directory.appending(path: "model.safetensors"))

        XCTAssertThrowsError(try ExpertOffsetIndex.build(modelDirectory: directory)) { error in
            guard case ExpertOffsetIndexError.incompletePiece(_, let missing) = error else {
                return XCTFail("expected incompletePiece, got \(error)")
            }
            XCTAssertTrue(missing.hasPrefix("switch_v."), missing)
        }
    }

    func testQuantizationIsValidatedForBothFamilies() throws {
        let directory = try writeMoVA()
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        // MLP weight row [64, 64] u32 at 4 bits → 512 inputs; scales [64, 64]
        // → group 8. Value weight row [48, 64] → 512 inputs; scales [32, 64]
        // → group 8 as well. Both agree at group 8 only.
        XCTAssertNoThrow(try index.validateQuantization(groupSize: 8, bits: 4))
        XCTAssertThrowsError(try index.validateQuantization(groupSize: 64, bits: 4))

        // A value family whose scales disagree with its weight fails even
        // though the MLP family is consistent.
        var tensors = SyntheticExpertCheckpoint.wellFormedTensors(experts: 4, layers: 1)
        tensors.append(
            .init(name: "model.layers.0.self_attn.switch_v.weight", dtype: "U32", shape: [4, 48, 64]))
        tensors.append(
            .init(name: "model.layers.0.self_attn.switch_v.scales", dtype: "BF16", shape: [4, 48, 8]))
        tensors.append(
            .init(name: "model.layers.0.self_attn.switch_v.biases", dtype: "BF16", shape: [4, 48, 8]))
        let inconsistent = try temporaryDirectory()
        try SyntheticExpertCheckpoint.write(
            tensors, to: inconsistent.appending(path: "model.safetensors"))
        let mixed = try ExpertOffsetIndex.build(modelDirectory: inconsistent)
        XCTAssertThrowsError(try mixed.validateQuantization(groupSize: 8, bits: 4)) { error in
            guard case ExpertOffsetIndexError.quantizationMismatch(let projection, _, _, _, _) = error
            else {
                return XCTFail("expected quantizationMismatch, got \(error)")
            }
            XCTAssertEqual(projection, "switch_v")
        }
    }

    // MARK: - Shard discovery

    /// The K2-Horizon checkpoint directory carries two diagnostic dumps,
    /// `k2-reference-logits.safetensors` and `k2-reference-layers.safetensors`,
    /// next to the weights. Neither may enter the index: through the
    /// weight map when there is one, and by name when there is not.
    func testDiagnosticDumpsNextToTheWeightsAreNotShards() throws {
        let directory = try writeMoVA()
        for dump in ["k2-reference-logits.safetensors", "k2-reference-layers.safetensors"] {
            try SyntheticExpertCheckpoint.write(
                [.init(name: "logits", dtype: "F32", shape: [4, 16])],
                to: directory.appending(path: dump))
        }

        // No `model.safetensors.index.json`: the directory listing decides.
        XCTAssertEqual(
            try ExpertOffsetIndex.routedWeightShards(in: directory).map(\.lastPathComponent),
            ["model.safetensors"])

        // With the index file, its weight map decides.
        let weightMap = ["weight_map": ["model.layers.1.mlp.switch_mlp.gate_proj.weight": "model.safetensors"]]
        try JSONSerialization.data(withJSONObject: weightMap)
            .write(to: directory.appending(path: "model.safetensors.index.json"))
        XCTAssertEqual(
            try ExpertOffsetIndex.routedWeightShards(in: directory).map(\.lastPathComponent),
            ["model.safetensors"])

        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        XCTAssertEqual(index.shardFiles, ["model.safetensors"])
    }

    // MARK: - Persistence

    func testRoundTripsBothFamiliesThroughJSON() throws {
        let directory = try writeMoVA()
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let url = directory.appending(path: "expert-offsets.json")
        try index.write(to: url)
        let reloaded = try ExpertOffsetIndex.load(from: url, expectingFingerprint: index.fingerprint)
        XCTAssertEqual(index, reloaded)
        XCTAssertEqual(reloaded.formatVersion, 2)
    }

    /// A version-1 index — persisted by the single-family design — has no
    /// family records. It has to say so, not fail to decode.
    func testVersionOneIndexIsReportedAsOutdated() throws {
        let legacy = """
            {"expertCount": 4, "fingerprint": "abc", "formatVersion": 1, "layers": [], \
            "shardFiles": ["model.safetensors"]}
            """
        XCTAssertThrowsError(try ExpertOffsetIndex.decoded(from: Data(legacy.utf8))) { error in
            guard case ExpertOffsetIndexError.unsupportedFormatVersion(1) = error else {
                return XCTFail("expected unsupportedFormatVersion(1), got \(error)")
            }
        }
    }

    // MARK: - Bank capacity split

    func testSingleFamilyGetsTheWholeBank() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.writeWellFormed(experts: 4, layers: 2, to: directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        XCTAssertEqual(
            ExpertStreamingSession.bankCapacitySplit(capacityBytes: 1 << 20, index: index),
            [.mlp: 1 << 20])
    }

    func testTwoFamiliesSplitInProportionToRoutedBytes() throws {
        let directory = try writeMoVA()
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let mlp = try index.requireFamily(.mlp)
        let value = try index.requireFamily(.value)

        // Large enough that the floor does not bite.
        let capacity = 1 << 30
        let split = ExpertStreamingSession.bankCapacitySplit(capacityBytes: capacity, index: index)
        let mlpShare = Double(mlp.routedBytes) / Double(index.routedBytes)
        let valueShare = Double(value.routedBytes) / Double(index.routedBytes)
        XCTAssertEqual(Double(split[.mlp]!), Double(capacity) * mlpShare, accuracy: 1)
        XCTAssertEqual(Double(split[.value]!), Double(capacity) * valueShare, accuracy: 1)
        XCTAssertEqual(split[.mlp]! + split[.value]!, capacity, accuracy: 2)

        // A tiny bank floors both families at the minimum slot count instead
        // of starving the minority family below one decode step.
        let tiny = ExpertStreamingSession.bankCapacitySplit(capacityBytes: 1, index: index)
        XCTAssertEqual(
            tiny[.value], ExpertStreamingSession.minimumBankSlotsPerFamily * value.bytesPerExpert)
        XCTAssertEqual(
            tiny[.mlp], ExpertStreamingSession.minimumBankSlotsPerFamily * mlp.bytesPerExpert)
    }

    // MARK: - The real checkpoint

    func testRealK2HorizonCheckpointGeometry() throws {
        let directory = try XCTUnwrap(
            K2StreamingTestCheckpoint.directory(),
            "no local K2-Horizon MoVA checkpoint; set MLX_R56_K2_MODEL_DIR to run this")
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        XCTAssertEqual(index.shardFiles.count, 4)
        XCTAssertFalse(index.shardFiles.contains { $0.hasPrefix("k2-reference") })

        let mlp = try index.requireFamily(.mlp)
        XCTAssertEqual(mlp.expertCount, 100)
        XCTAssertEqual(mlp.layers.map(\.layer), Array(3 ..< 48))
        XCTAssertEqual(mlp.template[ExpertPiece(.gate, .weight)].rowBytes, 983_040)
        XCTAssertEqual(mlp.template[ExpertPiece(.gate, .scales)].rowBytes, 61_440)
        XCTAssertFalse(mlp.template[ExpertPiece(.gate, .scales)].isPageAligned)
        XCTAssertEqual(mlp.template[ExpertPiece(.down, .weight)].rowShape, [2560, 96])
        XCTAssertEqual(mlp.bytesPerExpert, 3 * (983_040 + 2 * 61_440))

        let value = try index.requireFamily(.value)
        XCTAssertEqual(value.expertCount, 64)
        XCTAssertEqual(value.layers.map(\.layer), Array(3 ..< 48))
        XCTAssertEqual(value.template[ExpertPiece(.value, .weight)].rowShape, [1024, 320])
        XCTAssertEqual(value.template[ExpertPiece(.value, .weight)].rowBytes, 1_310_720)
        XCTAssertEqual(value.template[ExpertPiece(.value, .scales)].rowBytes, 81_920)
        XCTAssertTrue(value.template[ExpertPiece(.value, .scales)].isPageAligned)
        XCTAssertEqual(value.bytesPerExpert, 1_310_720 + 2 * 81_920)

        // config.json: affine, 4-bit, group 64 — for both families.
        XCTAssertNoThrow(try index.validateQuantization(groupSize: 64, bits: 4))
        XCTAssertThrowsError(try index.validateQuantization(groupSize: 64, bits: 8))

        print(
            """
            R56 | K2 index: MLP \(mlp.layerCount) layers × \(mlp.expertCount) experts × \
            \(String(format: "%.2f", Double(mlp.bytesPerExpert) / 1_048_576)) MiB = \
            \(String(format: "%.2f", Double(mlp.routedBytes) / 1e9)) GB; value \
            \(value.layerCount) × \(value.expertCount) × \
            \(String(format: "%.2f", Double(value.bytesPerExpert) / 1_048_576)) MiB = \
            \(String(format: "%.2f", Double(value.routedBytes) / 1e9)) GB; \
            routed \(String(format: "%.2f", Double(index.routedBytes) / 1e9)) GB
            """)
    }
}

/// Locates the local K2-Horizon MoVA 36B-A4B 4-bit checkpoint used by the
/// on-Mac experiments. `MLX_R56_K2_MODEL_DIR` wins; otherwise the conversion
/// directory the port was validated against.
enum K2StreamingTestCheckpoint {
    static func directory() -> URL? {
        let env = ProcessInfo.processInfo.environment
        let candidates = [
            env["MLX_R56_K2_MODEL_DIR"], env["TEST_RUNNER_MLX_R56_K2_MODEL_DIR"],
            NSHomeDirectory() + "/Documents/mlx-lm-converted/K2-Horizon-MoVA-36B-A4B-4bit",
        ]
        for case let path? in candidates
        where FileManager.default.fileExists(atPath: path + "/config.json") {
            return URL(fileURLWithPath: path)
        }
        return nil
    }
}
