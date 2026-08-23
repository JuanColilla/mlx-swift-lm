import Foundation
import Testing
import MLX
@testable import MLXLMCommon

@Suite("pipelineWeightURLs tolerates a same-path sliced file (TD-032)")
struct SafetensorsSlicedFileLoadTests {
    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(path: "td032-fork-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    /// Writes a minimal valid safetensors file containing exactly `keys`,
    /// each a single float32 element, with an optional `__metadata__`
    /// block — mirrors the fixture MLXHub's own slicer tests use.
    private func writeSafetensors(
        keys: [String], metadata: [String: String] = [:], to url: URL
    ) throws {
        var header: [String: Any] = [:]
        if !metadata.isEmpty {
            header["__metadata__"] = metadata
        }
        var cursor: Int64 = 0
        var payload = Data()
        for key in keys.sorted() {
            let bytes = withUnsafeBytes(of: Float32(1)) { Data($0) }
            header[key] = ["dtype": "F32", "shape": [1], "data_offsets": [cursor, cursor + 4]]
            payload.append(bytes)
            cursor += 4
        }
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var sizeLE = UInt64(headerData.count).littleEndian
        var full = Data(bytes: &sizeLE, count: 8)
        full.append(headerData)
        full.append(payload)
        try full.write(to: url)
    }

    @Test("pipelineWeightURLs finds the file by its index-declared name regardless of its real content, and metadata survives to loadArraysAndMetadata")
    func findsFileByDeclaredNameEvenWhenSliced() throws {
        let directory = try temporaryDirectory()

        // The published index claims TWO tensors live in this one file —
        // exactly the TD-032 shape (embed_tokens + a layer tensor sharing
        // a file). On disk, only ONE of them is actually present, WITH a
        // __metadata__ block: this simulates MLXHub having served a slice
        // (Task 1's SafetensorsTensorSlicer output) at the ORIGINAL
        // filename.
        let index: [String: Any] = [
            "weight_map": [
                "model.embed_tokens.weight": "model.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model.safetensors",
            ]
        ]
        let indexData = try JSONSerialization.data(withJSONObject: index)
        try indexData.write(to: directory.appending(path: "model.safetensors.index.json"))

        try writeSafetensors(
            keys: ["model.embed_tokens.weight"],
            metadata: ["format": "mlx"],
            to: directory.appending(path: "model.safetensors")
        )

        let selection = WeightLoadingSelection(includes: { $0 == "model.embed_tokens.weight" })
        let urls = try pipelineWeightURLs(in: directory, selection: selection)

        #expect(urls.map(\.lastPathComponent) == ["model.safetensors"])

        let (weights, metadata) = try loadArraysAndMetadata(url: urls[0])
        #expect(Set(weights.keys) == ["model.embed_tokens.weight"])
        // Confirms Task 1's fix matters at the exact boundary
        // loadWeights reads: Qwen35/FalconH1's sanitize(weights:metadata:)
        // would see this metadata, not an empty dict.
        #expect(metadata["format"] == "mlx")
    }
}
