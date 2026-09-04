// FORK(JuanColilla): R-56 expert streaming — the `lfm2_moe` acceptance run on
// the real checkpoint.
//
// The synthetic round-trip in `LFM2MoEStreamedTests` proves the wiring; this
// proves it against the published checkpoint, whose MoE layers start at 2
// (`num_dense_layers`) and whose weights are a single `model.safetensors`.
// Same instrument as the Qwen 3.5 P1c: teacher-forced logits, resident
// against streamed.
//
// `LiquidAI/LFM2.5-8B-A1B-MLX-4bit` is the canonical repository — it is what
// the app installs — and `mlx-community` publishes a conversion of the same
// model. They differ in how the router is stored: LiquidAI leaves
// `feed_forward.gate.weight` unquantized, mlx-community quantizes it to 8
// bits with a per-layer override in `config.quantization`. Neither touches
// the routed experts, which are 4-bit affine with group 64 in both, so the
// streaming path is identical; the quantization this test hands the session
// is read from the checkpoint rather than written here as a constant.
//
// Opt-in and slow — it loads a 4.5 GB checkpoint twice:
//
// ```sh
// MLX_R56_LFM2_EXPERIMENT=1 swift test \
//     --scratch-path /tmp/r56 --filter LFM2MoEStreamedMacExperiment
// ```
//
// A run that reports 0.00x s did not execute: check for the `R56 |` lines.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class LFM2MoEStreamedMacExperiment: XCTestCase {

    /// `MLX_R56_LFM2_MODEL_DIR` wins; otherwise the Hugging Face cache, with
    /// the LiquidAI repository preferred over the mlx-community conversion.
    static func checkpointDirectory() -> URL? {
        if let path = ProcessInfo.processInfo.environment["MLX_R56_LFM2_MODEL_DIR"],
            FileManager.default.fileExists(atPath: path)
        {
            return URL(fileURLWithPath: path)
        }
        let repositories = [
            "models--LiquidAI--LFM2.5-8B-A1B-MLX-4bit",
            "models--mlx-community--LFM2.5-8B-A1B-MLX-4bit",
        ]
        for repository in repositories {
            let cache = URL(fileURLWithPath: NSHomeDirectory())
                .appending(path: ".cache/huggingface/hub/\(repository)/snapshots")
            let snapshots =
                (try? FileManager.default.contentsOfDirectory(
                    at: cache, includingPropertiesForKeys: nil)) ?? []
            if let snapshot = snapshots.first(where: {
                FileManager.default.fileExists(atPath: $0.appending(path: "config.json").path)
            }) {
                return snapshot
            }
        }
        return nil
    }

    private func requireCheckpoint() throws -> URL {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLX_R56_LFM2_EXPERIMENT"] == "1",
            "opt-in experiment; set MLX_R56_LFM2_EXPERIMENT=1")
        return try XCTUnwrap(
            Self.checkpointDirectory(),
            "no local LFM2.5-8B-A1B checkpoint; set MLX_R56_LFM2_MODEL_DIR")
    }

    private func configurations(_ directory: URL) throws -> (
        model: LFM2MoEConfiguration, base: BaseConfiguration
    ) {
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        return (
            try JSONDecoder().decode(LFM2MoEConfiguration.self, from: data),
            try JSONDecoder().decode(BaseConfiguration.self, from: data)
        )
    }

    /// Byte-level BPE vocabulary read straight out of `tokenizer.json`.
    ///
    /// Not a tokenizer: greedy longest-match over whole words, enough to feed
    /// the model a real English sentence. This checkpoint ships no
    /// `vocab.json`, which is why it is read from `model.vocab` here.
    /// Correctness is judged on logits, not on the text.
    struct MinimalVocabulary {
        let idOfToken: [String: Int32]
        let tokenOfID: [Int32: String]

        init(directory: URL) throws {
            let data = try Data(contentsOf: directory.appending(path: "tokenizer.json"))
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            let model = json?["model"] as? [String: Any]
            let raw = (model?["vocab"] as? [String: Int]) ?? [:]
            idOfToken = raw.mapValues { Int32($0) }
            tokenOfID = Dictionary(
                raw.map { (Int32($0.value), $0.key) }, uniquingKeysWith: { first, _ in first })
        }

        func encode(_ text: String) -> [Int32] {
            var ids = [Int32]()
            for (position, word) in text.split(separator: " ").enumerated() {
                var remaining = Substring(position == 0 ? String(word) : "\u{0120}" + word)
                while !remaining.isEmpty {
                    var length = remaining.count
                    while length > 0 {
                        if let id = idOfToken[String(remaining.prefix(length))] {
                            ids.append(id)
                            remaining = remaining.dropFirst(length)
                            break
                        }
                        length -= 1
                    }
                    if length == 0 { remaining = remaining.dropFirst() }
                }
            }
            return ids
        }

        func decode(_ ids: [Int32]) -> String {
            ids.map { tokenOfID[$0] ?? "<\($0)>" }
                .joined()
                .replacingOccurrences(of: "\u{0120}", with: " ")
                .replacingOccurrences(of: "\u{010A}", with: "\n")
        }
    }

    private static let promptText = "The capital of France is the city of"

    private struct Run {
        var tokens: [Int32]
        var logits: [MLXArray]
        var loadSeconds: Double
        var decodeTokensPerSecond: Double
    }

    /// Prefill plus `steps - 1` decode steps, optionally teacher-forced.
    private func generate(
        model: LFM2MoEModel, promptTokens: [Int32], steps: Int, loadSeconds: Double,
        forcing: [Int32]? = nil
    ) throws -> Run {
        let cache = try model.newCache(parameters: nil)
        var produced = [Int32]()
        var captured = [MLXArray]()

        let prompt = MLXArray(promptTokens).reshaped(1, promptTokens.count)
        var logits = model(prompt, cache: cache)
        var row = logits[0..., -1, 0...]
        var next = MLX.argMax(row, axis: -1)
        eval(next)
        captured.append(row.asType(.float32))
        eval(captured.last!)
        produced.append(next.item(Int32.self))

        let decodeStart = Date.timeIntervalSinceReferenceDate
        for step in 1 ..< steps {
            let input = forcing.map { MLXArray([$0[step - 1]]) } ?? next
            logits = model(input.reshaped(1, 1), cache: cache)
            row = logits[0..., -1, 0...]
            next = MLX.argMax(row, axis: -1)
            eval(next)
            captured.append(row.asType(.float32))
            eval(captured.last!)
            produced.append(next.item(Int32.self))
        }
        let decodeSeconds = Date.timeIntervalSinceReferenceDate - decodeStart

        return Run(
            tokens: produced, logits: captured, loadSeconds: loadSeconds,
            decodeTokensPerSecond: Double(steps - 1) / decodeSeconds)
    }

    private func loadResident(_ directory: URL) throws -> (LFM2MoEModel, Double) {
        let (configuration, base) = try configurations(directory)
        let start = Date.timeIntervalSinceReferenceDate
        let model = LFM2MoEModel(configuration)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        try model.prepare()
        return (model, Date.timeIntervalSinceReferenceDate - start)
    }

    private func loadStreamed(_ directory: URL, bankBytes: Int) throws -> (
        LFM2MoEModel, ExpertStreamingSession, Double
    ) {
        let (configuration, base) = try configurations(directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        // The *expert* quantization: the root of `config.quantization`, read
        // from the checkpoint rather than written here. Any per-layer override
        // applies to `feed_forward.gate`, which the bank never holds, and a
        // wrong pair here would not fail — it would decode the expert weights
        // as another bit depth and generate fluent nonsense.
        let quantization = try XCTUnwrap(base.quantization, "checkpoint is not quantized")
        let session = try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(
                bankCapacityBytes: bankBytes,
                groupSize: quantization.groupSize, bits: quantization.bits))

        let start = Date.timeIntervalSinceReferenceDate
        let model = try ExpertStreaming.withSession(session) {
            let model = LFM2MoEModel(configuration)
            try loadWeights(
                modelDirectory: directory, model: model,
                perLayerQuantization: base.perLayerQuantization)
            try model.prepare()
            return model
        }
        return (model, session, Date.timeIntervalSinceReferenceDate - start)
    }

    // MARK: - The index over the published checkpoint

    func testIndexOverTheRealCheckpoint() throws {
        let directory = try requireCheckpoint()
        let (configuration, base) = try configurations(directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        let expected = Array(configuration.numDenseLayers ..< configuration.hiddenLayers)
        XCTAssertEqual(index.layers.map(\.layer), expected)
        XCTAssertEqual(index.expertCount, configuration.numExperts)
        for dense in 0 ..< configuration.numDenseLayers {
            XCTAssertNil(index.records(forLayer: dense))
        }
        let quantization = try XCTUnwrap(base.quantization)
        try index.validateQuantization(
            groupSize: quantization.groupSize, bits: quantization.bits)

        print(
            """
            R56 | LFM2 checkpoint: \(directory.path) \
            | quantization \(quantization.bits)-bit group \(quantization.groupSize)
            """)
        print(
            """
            R56 | LFM2 index: \(index.layerCount) MoE layers of \
            \(configuration.hiddenLayers) (first \(index.layers.first?.layer ?? -1)), \
            \(index.expertCount) experts, \
            \(String(format: "%.2f", Double(index.bytesPerExpert) / 1_048_576)) MiB/expert, \
            \(String(format: "%.2f", Double(index.routedBytes) / 1_073_741_824)) GiB routed, \
            shards \(index.shardFiles)
            """)
    }

    // MARK: - Resident vs streamed logits

    func testStreamedLogitsMatchTheResidentModel() throws {
        let directory = try requireCheckpoint()
        let vocabulary = try MinimalVocabulary(directory: directory)
        let promptTokens = vocabulary.encode(Self.promptText)
        XCTAssertGreaterThan(promptTokens.count, 3, "the prompt did not tokenize")
        let steps = 16

        let reference: Run
        let forcing: [Int32]
        do {
            let (resident, loadSeconds) = try loadResident(directory)
            reference = try generate(
                model: resident, promptTokens: promptTokens, steps: steps,
                loadSeconds: loadSeconds)
            forcing = reference.tokens
            print(
                """
                R56 | LFM2 resident | load \(String(format: "%.1f", loadSeconds)) s \
                | decode \(String(format: "%.2f", reference.decodeTokensPerSecond)) tok/s \
                | peak GPU \
                \(String(format: "%.2f", Double(GPU.peakMemory) / 1_073_741_824)) GiB
                """)
            print("R56 | LFM2 resident completion: \(vocabulary.decode(reference.tokens))")
        }
        GPU.clearCache()
        GPU.resetPeakMemory()

        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let (streamed, session, streamedLoad) = try loadStreamed(
            directory, bankBytes: 1 << 30)
        let streamedRun = try generate(
            model: streamed, promptTokens: promptTokens, steps: steps,
            loadSeconds: streamedLoad, forcing: forcing)

        var worstAbsolute: Float = 0
        var worstRelative: Float = 0
        var agreements = 0
        for step in 0 ..< steps {
            let a = reference.logits[step]
            let b = streamedRun.logits[step]
            let difference = MLX.max(MLX.abs(a - b))
            let scale = MLX.max(MLX.abs(a))
            eval(difference, scale)
            worstAbsolute = max(worstAbsolute, difference.item(Float.self))
            worstRelative = max(
                worstRelative, difference.item(Float.self) / scale.item(Float.self))
            if reference.tokens[step] == streamedRun.tokens[step] { agreements += 1 }
        }

        print(
            """
            R56 | LFM2 streamed | load \(String(format: "%.1f", streamedLoad)) s \
            | decode \(String(format: "%.2f", streamedRun.decodeTokensPerSecond)) tok/s \
            | peak GPU \(String(format: "%.2f", Double(GPU.peakMemory) / 1_073_741_824)) GiB \
            | bank \(String(format: "%.2f", Double(1 << 30) / 1_073_741_824)) GiB of \
            \(String(format: "%.2f", Double(index.routedBytes) / 1_073_741_824)) GiB routed \
            | hit \(String(format: "%.1f%%", session.bank.statistics.hitRate * 100)) \
            | read \(String(format: "%.0f", session.store.statistics.megabytesPerSecond)) MB/s
            """)
        print(
            """
            R56 | LFM2 teacher-forced: top-1 agreement \(agreements)/\(steps), \
            worst absolute logit difference \(String(format: "%.5f", worstAbsolute)), \
            worst relative \(String(format: "%.6f", worstRelative))
            """)

        XCTAssertNil(session.lastFailure)
        XCTAssertGreaterThan(session.bank.statistics.hits + session.bank.statistics.misses, 0)
        XCTAssertEqual(
            agreements, steps,
            "streamed logits should pick the same token at every teacher-forced position")
        XCTAssertLessThan(
            worstRelative, 0.01,
            "streamed logits diverged from the resident model by more than rounding")
    }
}
