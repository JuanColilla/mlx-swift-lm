// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLXLMCommon

final class LoadWeightsTests: XCTestCase {

    func testLoadWeightsUsesSafetensorsIndexWeightMapWhenPresent() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("mtp.safetensors", in: directory)
        try writeEmptyFile("optiq_vision.safetensors", in: directory)
        try """
        {
          "metadata": { "total_size": 1 },
          "weight_map": {
            "model.norm.weight": "model.safetensors"
          }
        }
        """.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model.safetensors"])
    }

    func testPipelineSelectionKeepsOnlyAssignedLayersAndRenumbersThem() {
        let selection = WeightLoadingSelection.pipelineLayers(
            range: 12 ..< 18,
            sourcePrefixes: ["model.layers.", "language_model.model.layers."],
            destinationPrefix: "language_model.model.layers."
        )

        XCTAssertTrue(selection.includes("model.embed_tokens.weight"))
        XCTAssertTrue(selection.includes("model.layers.12.self_attn.q_proj.weight"))
        XCTAssertTrue(selection.includes("model.layers.17.self_attn.q_proj.weight"))
        XCTAssertFalse(selection.includes("model.layers.11.self_attn.q_proj.weight"))
        XCTAssertFalse(selection.includes("model.layers.18.self_attn.q_proj.weight"))
        XCTAssertEqual(
            selection.rewrite("language_model.model.layers.12.self_attn.q_proj.weight"),
            "language_model.model.layers.0.self_attn.q_proj.weight"
        )
        XCTAssertEqual(
            selection.rewrite("language_model.model.layers.17.self_attn.q_proj.weight"),
            "language_model.model.layers.5.self_attn.q_proj.weight"
        )
    }

    func testSafetensorsIndexSkipsFilesOwnedOnlyByOtherPipelineShards() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("shared.safetensors", in: directory)
        try writeEmptyFile("local.safetensors", in: directory)
        try writeEmptyFile("remote.safetensors", in: directory)
        try """
        {
          "metadata": { "total_size": 1 },
          "weight_map": {
            "model.embed_tokens.weight": "shared.safetensors",
            "model.layers.12.self_attn.q_proj.weight": "local.safetensors",
            "model.layers.18.self_attn.q_proj.weight": "remote.safetensors"
          }
        }
        """.data(using: .utf8)!.write(
            to: directory.appendingPathComponent("model.safetensors.index.json"))

        let selection = WeightLoadingSelection.pipelineLayers(
            range: 12 ..< 18,
            sourcePrefixes: ["model.layers."],
            destinationPrefix: "language_model.model.layers."
        )
        let names = try safetensorWeightURLs(in: directory, selection: selection).map(\.lastPathComponent)

        XCTAssertEqual(names, ["local.safetensors", "shared.safetensors"])
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("LoadWeightsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeEmptyFile(_ name: String, in directory: URL) throws {
        try Data().write(to: directory.appendingPathComponent(name))
    }
}
