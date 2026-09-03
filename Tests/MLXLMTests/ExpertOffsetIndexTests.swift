// FORK(JuanColilla): R-56 task 1.1 — pure tests for the expert offset index.
//
// No MLXArray is created anywhere in this file on purpose: index construction
// and validation must be verifiable where no Metal device exists.

import Foundation
import XCTest

@testable import MLXLMCommon

final class ExpertOffsetIndexTests: XCTestCase {

    func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-index-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    // MARK: - Key parsing

    func testParsesBothCheckpointPrefixes() throws {
        let text = try ExpertOffsetIndex.parse(
            key: "model.layers.7.mlp.switch_mlp.gate_proj.weight")
        XCTAssertEqual(text?.layer, 7)
        XCTAssertEqual(text?.piece, ExpertPiece(.gate, .weight))

        let multimodal = try ExpertOffsetIndex.parse(
            key: "model.language_model.layers.39.mlp.switch_mlp.down_proj.scales")
        XCTAssertEqual(multimodal?.layer, 39)
        XCTAssertEqual(multimodal?.piece, ExpertPiece(.down, .scales))
    }

    func testIgnoresNonExpertKeys() throws {
        XCTAssertNil(try ExpertOffsetIndex.parse(key: "model.layers.0.mlp.gate.weight"))
        XCTAssertNil(try ExpertOffsetIndex.parse(key: "model.embed_tokens.weight"))
        XCTAssertNil(
            try ExpertOffsetIndex.parse(key: "model.layers.0.mlp.shared_expert.up_proj.weight"))
    }

    /// TD-050: both unsupported layouts must be rejected loudly, never read as
    /// if they were the stacked one.
    func testRejectsFusedGateUpLayout() {
        XCTAssertThrowsError(
            try ExpertOffsetIndex.parse(key: "model.layers.0.mlp.switch_mlp.gate_up_proj.weight")
        ) { error in
            guard case ExpertOffsetIndexError.fusedGateUpLayout = error else {
                return XCTFail("expected fusedGateUpLayout, got \(error)")
            }
        }
    }

    func testRejectsPerExpertLayout() {
        XCTAssertThrowsError(
            try ExpertOffsetIndex.parse(key: "model.layers.0.mlp.experts.3.gate_proj.weight")
        ) { error in
            guard case ExpertOffsetIndexError.perExpertLayout = error else {
                return XCTFail("expected perExpertLayout, got \(error)")
            }
        }
    }

    func testRoutedExpertKeyClassification() {
        XCTAssertTrue(
            ExpertOffsetIndex.isRoutedExpertKey(
                "model.layers.0.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertFalse(
            ExpertOffsetIndex.isRoutedExpertKey(
                "model.layers.0.mlp.shared_expert.gate_proj.weight"))
        XCTAssertFalse(ExpertOffsetIndex.isRoutedExpertKey("model.layers.0.mlp.gate.weight"))
    }

    // MARK: - Construction over a synthetic checkpoint

    func testBuildsIndexOverSyntheticCheckpoint() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.writeWellFormed(experts: 4, layers: 2, to: directory)

        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        XCTAssertEqual(index.expertCount, 4)
        XCTAssertEqual(index.layerCount, 2)
        XCTAssertEqual(index.layers.map(\.layer), [0, 1])
        XCTAssertEqual(index.bytesPerExpert, 9 * 16384)
        XCTAssertEqual(index.routedBytes, 2 * 4 * 9 * 16384)

        for layer in index.layers {
            XCTAssertEqual(layer.pieces.count, 9)
            for piece in ExpertPiece.all {
                let record = layer[piece]
                XCTAssertEqual(record.rowBytes, 16384)
                XCTAssertEqual(record.rowBytes % ExpertOffsetIndex.pageSize, 0)
                XCTAssertEqual(
                    record.offset(ofExpert: 3), record.baseOffset + 3 * 16384)
            }
            XCTAssertEqual(layer[ExpertPiece(.gate, .weight)].dtype, .uint32)
            XCTAssertEqual(layer[ExpertPiece(.gate, .scales)].dtype, .bfloat16)
        }
    }

    /// The offsets have to be absolute: forgetting the 8-byte length field and
    /// the JSON header is the classic way to build an index that reads garbage.
    func testOffsetsIncludeTheHeader() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.writeWellFormed(experts: 2, layers: 1, to: directory)
        let url = directory.appending(path: "model.safetensors")

        let header = try SafetensorsHeader.read(url: url)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let first = index.layers[0][ExpertPiece(.gate, .weight)]

        XCTAssertGreaterThan(header.dataStart, 8)
        let entry = try XCTUnwrap(
            header.entries.first { $0.name.hasSuffix("switch_mlp.gate_proj.weight") })
        XCTAssertEqual(first.baseOffset, header.dataStart + entry.begin)

        let fileSize = try XCTUnwrap(
            FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int)
        XCTAssertLessThanOrEqual(
            Int(first.offset(ofExpert: index.expertCount - 1)) + first.rowBytes, fileSize)
    }

    func testRejectsRowsThatAreNotPageMultiples() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.write(
            [
                .init(name: "model.layers.0.mlp.switch_mlp.gate_proj.weight", dtype: "U32",
                    shape: [4, 8, 8]),
                .init(name: "model.layers.0.mlp.switch_mlp.gate_proj.scales", dtype: "BF16",
                    shape: [4, 8, 8]),
                .init(name: "model.layers.0.mlp.switch_mlp.gate_proj.biases", dtype: "BF16",
                    shape: [4, 8, 8]),
            ], to: directory.appending(path: "model.safetensors"))

        XCTAssertThrowsError(try ExpertOffsetIndex.build(modelDirectory: directory)) { error in
            guard case ExpertOffsetIndexError.unalignedRow = error else {
                return XCTFail("expected unalignedRow, got \(error)")
            }
        }
    }

    func testRejectsIncompleteLayer() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.write(
            [
                .init(name: "model.layers.0.mlp.switch_mlp.gate_proj.weight", dtype: "U32",
                    shape: [4, 64, 64]),
                .init(name: "model.layers.0.mlp.switch_mlp.gate_proj.scales", dtype: "BF16",
                    shape: [4, 128, 64]),
            ], to: directory.appending(path: "model.safetensors"))

        XCTAssertThrowsError(try ExpertOffsetIndex.build(modelDirectory: directory)) { error in
            guard case ExpertOffsetIndexError.incompletePiece = error else {
                return XCTFail("expected incompletePiece, got \(error)")
            }
        }
    }

    /// The multi-token-prediction head is a decoder layer with the same tensor
    /// names, so `mtp.layers.0.mlp.switch_mlp.*` collides with the real layer
    /// 0. `Qwen35TextModel.sanitize` drops every `mtp.` key, so the module tree
    /// never sees them and the index must not either.
    func testIgnoresTheMultiTokenPredictionHead() throws {
        XCTAssertNil(
            try ExpertOffsetIndex.parse(key: "mtp.layers.0.mlp.switch_mlp.gate_proj.weight"))
        XCTAssertNil(
            try ExpertOffsetIndex.parse(
                key: "model.mtp.layers.0.mlp.switch_mlp.down_proj.scales"))
    }

    func testBuildIgnoresMultiTokenPredictionTensors() throws {
        let directory = try temporaryDirectory()
        var tensors = SyntheticExpertCheckpoint.wellFormedTensors(experts: 4, layers: 1)
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let base = "mtp.layers.0.mlp.switch_mlp.\(projection)"
            tensors.append(
                .init(name: "\(base).weight", dtype: "U32", shape: [4, 64, 64]))
            tensors.append(
                .init(name: "\(base).scales", dtype: "BF16", shape: [4, 128, 64]))
            tensors.append(
                .init(name: "\(base).biases", dtype: "BF16", shape: [4, 128, 64]))
        }
        try SyntheticExpertCheckpoint.write(
            tensors, to: directory.appending(path: "model.safetensors"))

        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        XCTAssertEqual(index.layerCount, 1)

        // The real layer 0 comes first in the file, so a collision would have
        // been resolvable by order alone; the point is that it is not indexed
        // at all rather than indexed and then overwritten.
        let realWeightOffset = index.layers[0][ExpertPiece(.gate, .weight)].baseOffset
        let header = try SafetensorsHeader.read(
            url: directory.appending(path: "model.safetensors"))
        let real = try XCTUnwrap(
            header.entries.first {
                $0.name == "model.layers.0.mlp.switch_mlp.gate_proj.weight"
            })
        XCTAssertEqual(realWeightOffset, header.dataStart + real.begin)
    }

    /// Any other namespace that reuses the pattern must fail loudly rather than
    /// silently binding a layer to the wrong stack (TD-050).
    func testRejectsTwoTensorsClaimingTheSameLayerAndPiece() throws {
        let directory = try temporaryDirectory()
        var tensors = SyntheticExpertCheckpoint.wellFormedTensors(experts: 4, layers: 1)
        tensors.append(
            .init(
                name: "draft.layers.0.mlp.switch_mlp.gate_proj.weight", dtype: "U32",
                shape: [4, 64, 64]))
        try SyntheticExpertCheckpoint.write(
            tensors, to: directory.appending(path: "model.safetensors"))

        XCTAssertThrowsError(try ExpertOffsetIndex.build(modelDirectory: directory)) { error in
            guard case ExpertOffsetIndexError.ambiguousKey = error else {
                return XCTFail("expected ambiguousKey, got \(error)")
            }
        }
    }

    // MARK: - Persistence

    func testRoundTripsThroughJSON() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.writeWellFormed(experts: 4, layers: 3, to: directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        let url = directory.appending(path: "expert-offsets.json")
        try index.write(to: url)
        let reloaded = try ExpertOffsetIndex.load(
            from: url, expectingFingerprint: index.fingerprint)

        XCTAssertEqual(index, reloaded)
    }

    func testRejectsIndexBuiltForAnotherCheckpoint() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.writeWellFormed(experts: 4, layers: 1, to: directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let url = directory.appending(path: "expert-offsets.json")
        try index.write(to: url)

        XCTAssertThrowsError(
            try ExpertOffsetIndex.load(from: url, expectingFingerprint: "not-the-same")
        ) { error in
            guard case ExpertOffsetIndexError.fingerprintMismatch = error else {
                return XCTFail("expected fingerprintMismatch, got \(error)")
            }
        }
    }

    // MARK: - The real checkpoint

    func testRealQwen35MoECheckpointGeometry() throws {
        let directory = try XCTUnwrap(
            ExpertStreamingTestCheckpoint.directory(),
            "no local Qwen 3.5 MoE checkpoint; set MLX_R56_MODEL_DIR to run this")

        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        XCTAssertEqual(index.expertCount, 256)
        XCTAssertEqual(index.layerCount, 40)
        XCTAssertEqual(index.layers.map(\.layer), Array(0 ..< 40))

        // §2.3 of the design: 1.6875 MiB per expert in nine contiguous ranges.
        XCTAssertEqual(index.bytesPerExpert, 1_769_472)
        XCTAssertEqual(Double(index.bytesPerExpert) / 1_048_576, 1.6875)

        let layer = index.layers[0]
        XCTAssertEqual(layer[ExpertPiece(.gate, .weight)].rowBytes, 524_288)
        XCTAssertEqual(layer[ExpertPiece(.gate, .scales)].rowBytes, 32_768)
        XCTAssertEqual(layer[ExpertPiece(.down, .weight)].rowShape, [2048, 64])
        XCTAssertEqual(layer[ExpertPiece(.gate, .weight)].rowShape, [512, 256])
        XCTAssertEqual(layer[ExpertPiece(.down, .biases)].dtype, .bfloat16)

        // 18.1194 GB of routed experts, 88.9% of the checkpoint.
        XCTAssertEqual(Double(index.routedBytes) / 1e9, 18.1194, accuracy: 0.001)

        for layerRecords in index.layers {
            for piece in ExpertPiece.all {
                let record = layerRecords[piece]
                XCTAssertEqual(record.rowBytes % ExpertOffsetIndex.pageSize, 0)
                XCTAssertLessThan(record.shard, index.shardFiles.count)
            }
        }
    }

    /// Every routed expert range must sit inside its shard: an off-by-one in
    /// `dataStart` shows up here and nowhere else until a read returns noise.
    func testRealCheckpointRangesStayInsideTheirShards() throws {
        let directory = try XCTUnwrap(
            ExpertStreamingTestCheckpoint.directory(),
            "no local Qwen 3.5 MoE checkpoint; set MLX_R56_MODEL_DIR to run this")
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        var sizes = [Int64]()
        for name in index.shardFiles {
            let path = directory.appending(path: name).path
            sizes.append(
                Int64(try XCTUnwrap(FileManager.default.attributesOfItem(atPath: path)[.size]
                    as? Int)))
        }

        for layerRecords in index.layers {
            for piece in ExpertPiece.all {
                let record = layerRecords[piece]
                let last = record.offset(ofExpert: index.expertCount - 1) + Int64(record.rowBytes)
                XCTAssertGreaterThan(record.baseOffset, 8)
                XCTAssertLessThanOrEqual(last, sizes[record.shard])
            }
        }
    }
}

// MARK: - Checkpoint discovery

/// Locates the local Qwen 3.5 MoE checkpoint used by the on-Mac experiments.
///
/// `MLX_R56_MODEL_DIR` wins; otherwise the Hugging Face cache is probed. Tests
/// that need it skip when it is absent — a 20 GB checkpoint is not a fixture.
enum ExpertStreamingTestCheckpoint {
    static func directory() -> URL? {
        if let path = ProcessInfo.processInfo.environment["MLX_R56_MODEL_DIR"],
            FileManager.default.fileExists(atPath: path)
        {
            return URL(fileURLWithPath: path)
        }

        let cache = URL(fileURLWithPath: NSHomeDirectory())
            .appending(path: ".cache/huggingface/hub/models--qwen3.5-35b-a3b/snapshots")
        let snapshots =
            (try? FileManager.default.contentsOfDirectory(
                at: cache, includingPropertiesForKeys: nil)) ?? []
        return snapshots.first {
            FileManager.default.fileExists(
                atPath: $0.appending(path: "config.json").path)
        }
    }
}
