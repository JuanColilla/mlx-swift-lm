// FORK(JuanColilla): R-56 expert streaming — the K2-Horizon MoVA acceptance
// run on the real 36B-A4B 4-bit checkpoint.
//
// The synthetic round-trip in `K2HorizonStreamedTests` proves the wiring;
// this proves it against the converted checkpoint, whose sparse layers are
// 3…47, whose MLP scales rows are not page multiples, and whose two expert
// families have different counts (100 MLP, 64 value). Same instrument as the
// Qwen 3.5 P1c and the LFM2 run: teacher-forced logits, resident against
// streamed, on the tokens of the Python reference dump so the streamed model
// is also compared with the Python port for free.
//
// Opt-in and slow — it loads a 20 GB checkpoint three or four times:
//
// ```sh
// MLX_R56_K2_EXPERIMENT=1 swift test --scratch-path /tmp/r56 \
//     --filter K2HorizonStreamedMacExperiment
// ```
//
// A run that reports 0.00 s did not execute: check for the `R56 |` lines.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class K2HorizonStreamedMacExperiment: XCTestCase {

    private static func environment(_ name: String) -> String? {
        let env = ProcessInfo.processInfo.environment
        return env[name] ?? env["TEST_RUNNER_\(name)"]
    }

    private func requireCheckpoint() throws -> URL {
        try XCTSkipUnless(
            Self.environment("MLX_R56_K2_EXPERIMENT") == "1",
            "opt-in experiment; set MLX_R56_K2_EXPERIMENT=1")
        return try XCTUnwrap(
            K2StreamingTestCheckpoint.directory(),
            "no local K2-Horizon MoVA checkpoint; set MLX_R56_K2_MODEL_DIR")
    }

    private struct Reference {
        let prompt: [Int32]
        let forced: [Int32]
        let logits: MLXArray
    }

    /// The Python dump the resident port was accepted against: `tokens`
    /// (prompt followed by the forced continuation), `prompt_length`, and one
    /// float32 logits row per step.
    private func loadReference(_ directory: URL) throws -> Reference? {
        let path = directory.appending(path: "k2-reference-logits.safetensors")
        guard FileManager.default.fileExists(atPath: path.path) else { return nil }
        let arrays = try MLX.loadArrays(url: path)
        let tokens = try XCTUnwrap(arrays["tokens"]).asType(.int32).asArray(Int32.self)
        let promptLength = Int(
            try XCTUnwrap(arrays["prompt_length"]).asType(.int32).reshaped(-1)[0].item(Int32.self))
        return Reference(
            prompt: Array(tokens[..<promptLength]), forced: Array(tokens[promptLength...]),
            logits: try XCTUnwrap(arrays["logits"]).asType(.float32))
    }

    private func baseConfiguration(_ directory: URL) throws -> (Data, BaseConfiguration) {
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        return (data, try JSONDecoder().decode(BaseConfiguration.self, from: data))
    }

    private func loadResident(_ directory: URL) async throws -> (K2HorizonModel, Double) {
        let (data, base) = try baseConfiguration(directory)
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

    /// The same load as the resident one, inside an active session. The expert
    /// quantization comes from `config.quantization`, never from a constant.
    private func loadStreamed(
        _ directory: URL, index: ExpertOffsetIndex, bankBytes: Int,
        perFamily: [ExpertFamily: Int]? = nil
    ) async throws -> (K2HorizonModel, ExpertStreamingSession, Double) {
        let (data, base) = try baseConfiguration(directory)
        let quantization = try XCTUnwrap(base.quantization, "checkpoint is not quantized")
        let session = try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(
                bankCapacityBytes: bankBytes,
                groupSize: quantization.groupSize, bits: quantization.bits,
                bankCapacityBytesPerFamily: perFamily))

        let start = Date.timeIntervalSinceReferenceDate
        let model = try await ExpertStreaming.withSession(session) {
            let created = try await LLMTypeRegistry.shared.createModel(
                configuration: data, modelType: "k2_horizon")
            let model = try XCTUnwrap(created as? K2HorizonModel)
            try loadWeights(
                modelDirectory: directory, model: model, quantization: base.quantization,
                perLayerQuantization: base.perLayerQuantization)
            eval(model)
            return model
        }
        return (model, session, Date.timeIntervalSinceReferenceDate - start)
    }

    private struct Run {
        var tokens: [Int32]
        var logits: [MLXArray]
        var prefillSeconds: Double
        var decodeTokensPerSecond: Double
    }

    /// Prefill plus `steps - 1` teacher-forced decode steps.
    private func generate(
        model: K2HorizonModel, prompt: [Int32], forced: [Int32], steps: Int
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
            logits = model(MLXArray([forced[step - 1]]).reshaped(1, 1), cache: cache)
            row = logits[0..., -1, 0...].asType(.float32)
            next = MLX.argMax(row, axis: -1)
            eval(row, next)
            captured.append(row)
            produced.append(next.item(Int32.self))
        }
        let decodeSeconds = Date.timeIntervalSinceReferenceDate - decodeStart
        return Run(
            tokens: produced, logits: captured, prefillSeconds: prefillSeconds,
            decodeTokensPerSecond: Double(steps - 1) / decodeSeconds)
    }

    private struct Comparison {
        var agreements = 0
        var worstAbsolute: Float = 0
        var worstRelative: Float = 0
    }

    private func compare(_ reference: Run, _ candidate: Run) -> Comparison {
        var result = Comparison()
        for (step, (a, b)) in zip(reference.logits, candidate.logits).enumerated() {
            let difference = MLX.max(MLX.abs(a - b))
            let scale = MLX.max(MLX.abs(a))
            eval(difference, scale)
            result.worstAbsolute = max(result.worstAbsolute, difference.item(Float.self))
            result.worstRelative = max(
                result.worstRelative, difference.item(Float.self) / scale.item(Float.self))
            if reference.tokens[step] == candidate.tokens[step] { result.agreements += 1 }
        }
        return result
    }

    private func describe(_ session: ExpertStreamingSession, bankBytes: Int, index: ExpertOffsetIndex)
        -> String
    {
        let mlp = session.bank
        let value = session.bank(for: .value)!
        let store = session.store.statistics
        return """
            banks MLP \(mlp.slotCount) slots \
            (\(String(format: "%.2f", Double(mlp.bytesResident) / 1_073_741_824)) GiB, \
            hit \(String(format: "%.1f%%", mlp.statistics.hitRate * 100)), \
            \(mlp.statistics.evictions) evictions) / value \(value.slotCount) slots \
            (\(String(format: "%.2f", Double(value.bytesResident) / 1_073_741_824)) GiB, \
            hit \(String(format: "%.1f%%", value.statistics.hitRate * 100)), \
            \(value.statistics.evictions) evictions) \
            | asked \(String(format: "%.2f", Double(bankBytes) / 1_073_741_824)) GiB of \
            \(String(format: "%.2f", Double(index.routedBytes) / 1_073_741_824)) GiB routed \
            | read \(String(format: "%.0f", store.megabytesPerSecond)) MB/s \
            | sync \(session.syncCounters.total) evals \
            (\(session.syncCounters.routerEvals) router)
            """
    }

    /// Resident vs streamed on the reference tokens, two bank sizes, then two
    /// negative controls with the records of two layers exchanged in one
    /// family at a time.
    func testStreamedLogitsMatchTheResidentModel() async throws {
        let directory = try requireCheckpoint()
        let reference = try loadReference(directory)
        let prompt = reference?.prompt ?? [0, 3737, 9024, 1010, 12244, 1029]
        let steps = reference.map { $0.logits.dim(0) } ?? 16

        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let (_, base) = try baseConfiguration(directory)
        let quantization = try XCTUnwrap(base.quantization)
        try index.validateQuantization(groupSize: quantization.groupSize, bits: quantization.bits)
        print(
            """
            R56 | K2 checkpoint \(directory.lastPathComponent) | \(quantization.bits)-bit \
            group \(quantization.groupSize) | MLP \(index.mlp.layerCount)×\(index.mlp.expertCount) \
            | value \(index.family(.value)!.layerCount)×\(index.family(.value)!.expertCount) \
            | routed \(String(format: "%.2f", Double(index.routedBytes) / 1e9)) GB
            """)

        // 1. Resident.
        let residentRun: Run
        let forced: [Int32]
        do {
            let (resident, loadSeconds) = try await loadResident(directory)
            let greedy = try generate(model: resident, prompt: prompt, forced: [], steps: 1)
            forced = reference?.forced ?? greedy.tokens
            residentRun = try generate(model: resident, prompt: prompt, forced: forced, steps: steps)
            print(
                """
                R56 | K2 resident | load \(String(format: "%.1f", loadSeconds)) s \
                | prefill \(prompt.count) tokens \(String(format: "%.2f", residentRun.prefillSeconds)) s \
                | decode \(String(format: "%.2f", residentRun.decodeTokensPerSecond)) tok/s \
                | peak GPU \(String(format: "%.2f", Double(GPU.peakMemory) / 1_073_741_824)) GiB
                """)
            if let reference {
                var agreements = 0
                for step in 0 ..< steps {
                    let pythonTop = MLX.argMax(reference.logits[step].reshaped(-1), axis: -1)
                        .item(Int32.self)
                    if pythonTop == residentRun.tokens[step] { agreements += 1 }
                }
                print("R56 | K2 resident vs Python reference: top-1 agreement \(agreements)/\(steps)")
            }
        }
        GPU.clearCache()
        GPU.resetPeakMemory()

        // 2. Streamed, a bank large enough to hold the working set.
        for bankBytes in [3 << 30, 256 << 20] {
            let (streamed, session, loadSeconds) = try await loadStreamed(
                directory, index: index, bankBytes: bankBytes)
            let split = ExpertStreamingSession.bankCapacitySplit(capacityBytes: bankBytes, index: index)
            let run = try generate(model: streamed, prompt: prompt, forced: forced, steps: steps)
            let comparison = compare(residentRun, run)

            print(
                """
                R56 | K2 streamed \(String(format: "%.2f", Double(bankBytes) / 1_073_741_824)) GiB \
                | load \(String(format: "%.1f", loadSeconds)) s \
                | prefill \(String(format: "%.2f", run.prefillSeconds)) s \
                | decode \(String(format: "%.2f", run.decodeTokensPerSecond)) tok/s \
                | peak GPU \(String(format: "%.2f", Double(GPU.peakMemory) / 1_073_741_824)) GiB \
                | split MLP \(String(format: "%.2f", Double(split[.mlp]!) / 1_073_741_824)) GiB, \
                value \(String(format: "%.2f", Double(split[.value]!) / 1_073_741_824)) GiB
                """)
            print("R56 | K2 streamed | \(describe(session, bankBytes: bankBytes, index: index))")
            print(
                """
                R56 | K2 streamed vs resident, teacher-forced \(steps) steps: \
                top-1 agreement \(comparison.agreements)/\(steps), \
                worst absolute logit difference \(String(format: "%.6f", comparison.worstAbsolute)), \
                worst relative \(String(format: "%.7f", comparison.worstRelative))
                """)
            if let reference {
                var agreements = 0
                for step in 0 ..< steps {
                    let pythonTop = MLX.argMax(reference.logits[step].reshaped(-1), axis: -1)
                        .item(Int32.self)
                    if pythonTop == run.tokens[step] { agreements += 1 }
                }
                print("R56 | K2 streamed vs Python reference: top-1 agreement \(agreements)/\(steps)")
            }

            XCTAssertNil(session.lastFailure)
            XCTAssertEqual(comparison.agreements, steps, "top-1 must match at every step")
            XCTAssertEqual(
                comparison.worstAbsolute, 0,
                "streamed logits are expected to be bit-identical to the resident model")
            if bankBytes == 256 << 20 {
                XCTAssertGreaterThan(session.bank.statistics.evictions, 0, "the MLP bank never evicted")
                XCTAssertGreaterThan(
                    session.bank(for: .value)!.statistics.evictions, 0, "the value bank never evicted")
            }
            GPU.clearCache()
            GPU.resetPeakMemory()
        }

        // 3. Controls: exchange the records of the first two sparse layers in
        //    one family and leave the other intact.
        let controlSteps = min(4, steps)
        let residentControl = Run(
            tokens: Array(residentRun.tokens.prefix(controlSteps)),
            logits: Array(residentRun.logits.prefix(controlSteps)),
            prefillSeconds: 0, decodeTokensPerSecond: 0)
        for family in [ExpertFamily.value, .mlp] {
            let swapped = ExpertOffsetIndex(
                fingerprint: index.fingerprint, shardFiles: index.shardFiles,
                families: index.families.map { familyIndex in
                    guard familyIndex.family == family else { return familyIndex }
                    var layers = familyIndex.layers
                    layers[0] = ExpertLayerRecords(
                        layer: familyIndex.layers[0].layer, pieces: familyIndex.layers[1].pieces)
                    layers[1] = ExpertLayerRecords(
                        layer: familyIndex.layers[1].layer, pieces: familyIndex.layers[0].pieces)
                    return ExpertFamilyIndex(
                        family: family, expertCount: familyIndex.expertCount, layers: layers)
                })
            let (streamed, session, _) = try await loadStreamed(
                directory, index: swapped, bankBytes: 1 << 30)
            let run = try generate(model: streamed, prompt: prompt, forced: forced, steps: controlSteps)
            let comparison = compare(residentControl, run)
            print(
                """
                R56 | K2 control (\(family.rawValue) layers \(index.mlp.layers[0].layer)↔\
                \(index.mlp.layers[1].layer) swapped): top-1 agreement \
                \(comparison.agreements)/\(controlSteps), worst absolute logit difference \
                \(String(format: "%.4f", comparison.worstAbsolute)), worst relative \
                \(String(format: "%.5f", comparison.worstRelative))
                """)
            XCTAssertNil(session.lastFailure)
            XCTAssertGreaterThan(
                comparison.worstRelative, 1e-3,
                "swapping two \(family.rawValue) layers changed nothing: the comparison is blind to it")
            GPU.clearCache()
        }
    }
}
