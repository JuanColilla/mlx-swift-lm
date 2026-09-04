// FORK(JuanColilla): K2-Horizon acceptance run on the converted checkpoints.
//
// Teacher-forced logits of the Swift port against a dump produced by the
// Python mlx-lm port on the same token ids. Opt-in and slow: it loads a
// multi-GB checkpoint.
//
// ```sh
// MLX_K2_EXPERIMENT=1 MLX_K2_MODEL_DIR=/path/to/K2-Horizon-3.7B-bf16 \
//   xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
//   -skipPackagePluginValidation -only-testing:MLXLMTests/K2HorizonMacExperiment
// ```
//
// With `xcodebuild test`, prefix the variables with `TEST_RUNNER_`; both
// spellings are accepted here. The reference is `MLX_K2_REFERENCE` or
// `<model dir>/k2-reference-logits.safetensors`, holding `tokens` (int32,
// prompt followed by the forced continuation), `prompt_length` (int32) and
// `logits` (float32 `[steps, vocab]`: row 0 after the full prompt, row s
// after feeding forced token s-1). Without a reference the run only prints
// the Swift greedy continuation and its speed. A run that reports 0.00 s
// did not execute: look for the `K2 |` lines.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class K2HorizonMacExperiment: XCTestCase {

    private static func environment(_ name: String) -> String? {
        let env = ProcessInfo.processInfo.environment
        return env[name] ?? env["TEST_RUNNER_\(name)"]
    }

    private func requireCheckpoint() throws -> URL {
        try XCTSkipUnless(
            Self.environment("MLX_K2_EXPERIMENT") == "1",
            "opt-in experiment; set MLX_K2_EXPERIMENT=1")
        let path = try XCTUnwrap(
            Self.environment("MLX_K2_MODEL_DIR"), "set MLX_K2_MODEL_DIR to a converted checkpoint")
        XCTAssertTrue(FileManager.default.fileExists(atPath: path), "missing \(path)")
        return URL(fileURLWithPath: path)
    }

    private struct Reference {
        let tokens: [Int32]
        let promptLength: Int
        let logits: MLXArray
    }

    private func loadReference(_ directory: URL) throws -> Reference? {
        let path =
            Self.environment("MLX_K2_REFERENCE")
            ?? directory.appending(path: "k2-reference-logits.safetensors").path
        guard FileManager.default.fileExists(atPath: path) else { return nil }
        let arrays = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(arrays["tokens"]).asType(.int32).asArray(Int32.self)
        let promptLength = Int(
            try XCTUnwrap(arrays["prompt_length"]).asType(.int32).reshaped(-1)[0].item(Int32.self))
        let logits = try XCTUnwrap(arrays["logits"]).asType(.float32)
        return Reference(tokens: tokens, promptLength: promptLength, logits: logits)
    }

    private func loadModel(_ directory: URL) async throws -> (K2HorizonModel, Double) {
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let start = Date.timeIntervalSinceReferenceDate
        let created = try await LLMTypeRegistry.shared.createModel(
            configuration: data, modelType: "k2_horizon")
        let model = try XCTUnwrap(created as? K2HorizonModel)
        try loadWeights(
            modelDirectory: directory, model: model, quantization: base.quantization,
            perLayerQuantization: base.perLayerQuantization)
        eval(model)
        return (model, Date.timeIntervalSinceReferenceDate - start)
    }

    private struct Run {
        var tokens: [Int32]
        var logits: [MLXArray]
        var decodeTokensPerSecond: Double
        var prefillSeconds: Double
    }

    /// Prefill the prompt, then `steps - 1` decode steps, teacher-forced
    /// when a continuation is given.
    private func generate(
        model: K2HorizonModel, prompt: [Int32], steps: Int, forcing: [Int32]?
    ) throws -> Run {
        let cache = try model.newCache(parameters: nil)
        var produced = [Int32]()
        var captured = [MLXArray]()

        let prefillStart = Date.timeIntervalSinceReferenceDate
        var logits = model(MLXArray(prompt).reshaped(1, prompt.count), cache: cache)
        var row = logits[0..., -1, 0...].asType(.float32)
        var next = MLX.argMax(row, axis: -1)
        eval(row, next)
        let prefillSeconds = Date.timeIntervalSinceReferenceDate - prefillStart
        captured.append(row)
        produced.append(next.item(Int32.self))

        let decodeStart = Date.timeIntervalSinceReferenceDate
        for step in 1 ..< steps {
            let input = forcing.map { MLXArray([$0[step - 1]]) } ?? next
            logits = model(input.reshaped(1, 1), cache: cache)
            row = logits[0..., -1, 0...].asType(.float32)
            next = MLX.argMax(row, axis: -1)
            eval(row, next)
            captured.append(row)
            produced.append(next.item(Int32.self))
        }
        let decodeSeconds = Date.timeIntervalSinceReferenceDate - decodeStart
        return Run(
            tokens: produced, logits: captured,
            decodeTokensPerSecond: Double(steps - 1) / decodeSeconds,
            prefillSeconds: prefillSeconds)
    }

    /// Layer-by-layer prefill against `k2-reference-layers.safetensors`
    /// (`embeddings`, `hidden_<i>`, `final_norm`; float32 `[1, prompt, hidden]`)
    /// to locate where the two ports part ways.
    func testHiddenStatesLayerByLayer() async throws {
        let directory = try requireCheckpoint()
        let path = directory.appending(path: "k2-reference-layers.safetensors")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path.path), "no layer dump")
        let reference = try loadReference(directory)
        let tokens = try XCTUnwrap(reference).tokens
        let promptLength = try XCTUnwrap(reference).promptLength
        let layers = try MLX.loadArrays(url: path)
        let (model, _) = try await loadModel(directory)

        let prompt = MLXArray(Array(tokens[..<promptLength])).reshaped(1, promptLength)
        var h = model.model.embedTokens(prompt)
        let mask = createAttentionMask(h: h, cache: nil)

        func report(_ name: String, _ value: MLXArray) {
            guard let expected = layers[name] else { return }
            let mine = value.asType(.float32)
            let difference = MLX.abs(expected - mine)
            let worst = MLX.max(difference).item(Float.self)
            let mean = MLX.mean(difference).item(Float.self)
            let scale = MLX.max(MLX.abs(expected)).item(Float.self)
            let worstToken = MLX.argMax(MLX.max(difference, axis: -1).reshaped(-1)).item(Int32.self)
            print(
                String(
                    format: "K2 | %@: max |Δ| %.4f (token %d) mean %.5f scale %.2f", name, worst,
                    worstToken, mean, scale))
        }

        report("embeddings", h)
        for (i, layer) in model.model.layers.enumerated() {
            h = layer(h, mask: mask, cache: nil)
            eval(h)
            report("hidden_\(i)", h)
        }
        report("final_norm", model.model.norm(h))
    }

    func testLogitsMatchThePythonReference() async throws {
        let directory = try requireCheckpoint()
        let reference = try loadReference(directory)
        let (model, loadSeconds) = try await loadModel(directory)
        print(
            """
            K2 | \(directory.lastPathComponent) | load \(String(format: "%.1f", loadSeconds)) s \
            | \(model.numParameters() / 1_000_000) M parameters
            """)

        guard let reference else {
            // No dump yet: report the Swift greedy continuation so the
            // Python side can be compared by hand.
            let prompt: [Int32] = (Self.environment("MLX_K2_PROMPT_IDS") ?? "")
                .split(separator: ",").compactMap { Int32($0.trimmingCharacters(in: .whitespaces)) }
            let ids = prompt.isEmpty ? [0, 3737, 9024, 1010, 12244, 1029] : prompt
            let run = try generate(model: model, prompt: ids, steps: 16, forcing: nil)
            print("K2 | no reference dump; prompt ids \(ids)")
            print("K2 | Swift greedy continuation: \(run.tokens)")
            print(
                """
                K2 | prefill \(String(format: "%.2f", run.prefillSeconds)) s \
                | decode \(String(format: "%.2f", run.decodeTokensPerSecond)) tok/s \
                | peak GPU \(String(format: "%.2f", Double(GPU.peakMemory) / 1_073_741_824)) GiB
                """)
            throw XCTSkip("no reference dump at \(directory.path)/k2-reference-logits.safetensors")
        }

        let steps = reference.logits.dim(0)
        let prompt = Array(reference.tokens[..<reference.promptLength])
        let forced = Array(reference.tokens[reference.promptLength...])
        XCTAssertGreaterThanOrEqual(forced.count + 1, steps, "dump is missing forced tokens")

        let run = try generate(model: model, prompt: prompt, steps: steps, forcing: forced)

        // `MLX_K2_SWIFT_DUMP=<file>` writes this run in the reference format,
        // so two Swift builds can be compared with `MLX_K2_REFERENCE`.
        if let dump = Self.environment("MLX_K2_SWIFT_DUMP") {
            try MLX.save(
                arrays: [
                    "tokens": MLXArray(reference.tokens),
                    "prompt_length": MLXArray([Int32(reference.promptLength)]),
                    "logits": concatenated(run.logits.map { $0.reshaped(1, -1) }, axis: 0),
                ], url: URL(fileURLWithPath: dump))
        }

        var agreements = 0
        var worstAbsolute: Float = 0
        var worstRelative: Float = 0
        var meanAbsolute: Float = 0
        var disagreements = [String]()
        for step in 0 ..< steps {
            let python = reference.logits[step].reshaped(-1)
            let swift = run.logits[step].reshaped(-1)
            let pythonTop = MLX.argMax(python, axis: -1).item(Int32.self)
            let difference = MLX.max(MLX.abs(python - swift)).item(Float.self)
            let scale = MLX.max(MLX.abs(python)).item(Float.self)
            worstAbsolute = max(worstAbsolute, difference)
            meanAbsolute += MLX.mean(MLX.abs(python - swift)).item(Float.self) / Float(steps)
            worstRelative = max(worstRelative, difference / scale)
            let pythonAtSwift = python[Int(run.tokens[step])].item(Float.self)
            let pythonAtTop = python[Int(pythonTop)].item(Float.self)
            let swiftAtTop = swift[Int(pythonTop)].item(Float.self)
            let swiftAtSwift = swift[Int(run.tokens[step])].item(Float.self)
            disagreements.append(
                String(
                    format:
                        "step %d: max |Δ| %.4f | python top %d (%.3f, swift %.3f) | swift top %d (%.3f, python %.3f)",
                    step, difference, pythonTop, pythonAtTop, swiftAtTop, run.tokens[step],
                    swiftAtSwift, pythonAtSwift))
            if pythonTop == run.tokens[step] {
                agreements += 1
            } else {
                // Two ports built on different MLX kernels drift by bf16 /
                // 4-bit noise; a flip only counts as a failure when the
                // reference preferred its token by more than that noise.
                XCTAssertLessThanOrEqual(
                    pythonAtTop - pythonAtSwift, difference,
                    "step \(step): Python preferred \(pythonTop) by a clear margin")
            }
        }

        print(
            """
            K2 | \(directory.lastPathComponent) teacher-forced \(steps) steps: \
            top-1 agreement \(agreements)/\(steps), \
            worst absolute logit difference \(String(format: "%.5f", worstAbsolute)), \
            mean absolute \(String(format: "%.5f", meanAbsolute)), \
            worst relative \(String(format: "%.6f", worstRelative))
            """)
        print(
            """
            K2 | prefill \(prompt.count) tokens in \(String(format: "%.2f", run.prefillSeconds)) s \
            | decode \(String(format: "%.2f", run.decodeTokensPerSecond)) tok/s \
            | peak GPU \(String(format: "%.2f", Double(GPU.peakMemory) / 1_073_741_824)) GiB
            """)
        for line in disagreements {
            print("K2 | \(line)")
        }

        XCTAssertGreaterThanOrEqual(agreements, steps - 1, "top-1 must match at nearly every step")
        XCTAssertLessThan(worstRelative, 0.15)
    }
}
