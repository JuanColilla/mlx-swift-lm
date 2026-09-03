// FORK(JuanColilla): R-56 task 1.8 — P1, P1b and P3 on the real
// Qwen3.5-35B-A3B 4-bit checkpoint, on a Mac.
//
// Opt-in and slow: they load a 20 GB checkpoint, several times.
//
// ```sh
// MLX_R56_MAC_EXPERIMENTS=1 swift test --filter ExpertStreamingMacExperiments
// ```
//
// Results are written up in MLXHub's
// `knowledge/articles/expertStreamingMacExperiments.md`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class ExpertStreamingMacExperiments: XCTestCase {

    struct Measurement {
        var label: String
        var loadSeconds: Double
        var timeToFirstToken: Double
        var decodeTokensPerSecond: Double
        var peakGPUBytes: Int
        var hitRate: Double?
        var readMegabytesPerSecond: Double?
        var tokens: [Int32]
        /// One logits row per step, kept only when the caller asks: the
        /// resident/streamed comparison has to be made on logits, not on a
        /// free-running greedy sequence that amplifies any tie.
        var logits: [MLXArray] = []
    }

    private func requireCheckpoint() throws -> URL {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLX_R56_MAC_EXPERIMENTS"] == "1",
            "opt-in experiment")
        return try XCTUnwrap(
            ExpertStreamingTestCheckpoint.directory(),
            "no local Qwen 3.5 MoE checkpoint; set MLX_R56_MODEL_DIR")
    }

    private func configurations(_ directory: URL) throws -> (
        model: Qwen35Configuration, base: BaseConfiguration
    ) {
        let data = try Data(contentsOf: directory.appending(path: "config.json"))
        return (
            try JSONDecoder().decode(Qwen35Configuration.self, from: data),
            try JSONDecoder().decode(BaseConfiguration.self, from: data)
        )
    }

    /// A minimal byte-level BPE vocabulary reader.
    ///
    /// Not a tokenizer: it maps whole words to ids by lookup and joins the
    /// pieces back for printing, which is enough to feed the model a real
    /// English prompt and read what it answered. Correctness is still judged
    /// by comparing ids against the resident model, not by reading the text.
    struct MinimalVocabulary {
        let idOfToken: [String: Int32]
        let tokenOfID: [Int32: String]

        init(directory: URL) throws {
            let data = try Data(contentsOf: directory.appending(path: "vocab.json"))
            let raw = try JSONDecoder().decode([String: Int32].self, from: data)
            idOfToken = raw
            tokenOfID = Dictionary(uniqueKeysWithValues: raw.map { ($0.value, $0.key) })
        }

        /// Greedy longest-match over the vocabulary, with the byte-level
        /// space marker. Good enough for a plain sentence of common words.
        func encode(_ text: String) -> [Int32] {
            var ids = [Int32]()
            for (position, word) in text.split(separator: " ").enumerated() {
                var remaining = Substring(position == 0 ? String(word) : "\u{0120}" + word)
                while !remaining.isEmpty {
                    var length = remaining.count
                    while length > 0 {
                        let candidate = String(remaining.prefix(length))
                        if let id = idOfToken[candidate] {
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

    private static let promptText =
        "The capital of France is the city of"

    /// Runs prefill plus `tokens - 1` decode steps.
    ///
    /// `forcing` replaces the sampled token at each step, so two models can be
    /// compared step by step on the same input instead of drifting apart after
    /// the first near-tie.
    private func generate(
        model: Qwen35MoEModel, tokens: Int, label: String,
        loadSeconds: Double, session: ExpertStreamingSession?,
        promptTokens: [Int32], forcing: [Int32]? = nil, captureLogits: Bool = false
    ) throws -> Measurement {
        let cache = try model.newCache(parameters: nil)
        GPU.resetPeakMemory()

        var captured = [MLXArray]()
        let prompt = MLXArray(promptTokens).reshaped(1, promptTokens.count)
        let prefillStart = Date.timeIntervalSinceReferenceDate
        var logits = model(prompt, cache: cache)
        var row = logits[0..., -1, 0...]
        var next = MLX.argMax(row, axis: -1)
        eval(next)
        let timeToFirstToken = Date.timeIntervalSinceReferenceDate - prefillStart
        if captureLogits { captured.append(row.asType(.float32)); eval(captured.last!) }

        var produced = [next.item(Int32.self)]
        let decodeStart = Date.timeIntervalSinceReferenceDate
        for step in 1 ..< tokens {
            let input = forcing.map { MLXArray([$0[step - 1]]) } ?? next
            logits = model(input.reshaped(1, 1), cache: cache)
            row = logits[0..., -1, 0...]
            next = MLX.argMax(row, axis: -1)
            eval(next)
            if captureLogits { captured.append(row.asType(.float32)); eval(captured.last!) }
            produced.append(next.item(Int32.self))
        }
        let decodeSeconds = Date.timeIntervalSinceReferenceDate - decodeStart

        return Measurement(
            label: label,
            loadSeconds: loadSeconds,
            timeToFirstToken: timeToFirstToken,
            decodeTokensPerSecond: Double(tokens - 1) / decodeSeconds,
            peakGPUBytes: GPU.peakMemory,
            hitRate: session?.bank.statistics.hitRate,
            readMegabytesPerSecond: session?.store.statistics.megabytesPerSecond,
            tokens: produced,
            logits: captured)
    }

    private func loadResident(_ directory: URL) throws -> (Qwen35MoEModel, Double) {
        let (configuration, base) = try configurations(directory)
        let start = Date.timeIntervalSinceReferenceDate
        let model = Qwen35MoEModel(configuration)
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        try model.prepare()
        return (model, Date.timeIntervalSinceReferenceDate - start)
    }

    private func loadStreamed(_ directory: URL, bankBytes: Int) throws -> (
        Qwen35MoEModel, ExpertStreamingSession, Double
    ) {
        let (configuration, base) = try configurations(directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let session = try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(bankCapacityBytes: bankBytes))

        let start = Date.timeIntervalSinceReferenceDate
        let model = try ExpertStreaming.withSession(session) {
            let model = Qwen35MoEModel(configuration)
            try loadWeights(
                modelDirectory: directory, model: model,
                perLayerQuantization: base.perLayerQuantization)
            try model.prepare()
            return model
        }
        return (model, session, Date.timeIntervalSinceReferenceDate - start)
    }

    private func report(_ measurement: Measurement) {
        let hit = measurement.hitRate.map { String(format: "%.1f%%", $0 * 100) } ?? "n/a"
        let read = measurement.readMegabytesPerSecond.map { String(format: "%.0f", $0) } ?? "n/a"
        print(
            """
            R56 | \(measurement.label) | load \(String(format: "%.1f", measurement.loadSeconds)) s \
            | TTFT \(String(format: "%.2f", measurement.timeToFirstToken)) s \
            | decode \(String(format: "%.2f", measurement.decodeTokensPerSecond)) tok/s \
            | peak GPU \(String(format: "%.2f", Double(measurement.peakGPUBytes) / 1_073_741_824)) GiB \
            | hit \(hit) | read \(read) MB/s
            """)
    }

    // MARK: - P1 · does it work?

    func testP1StreamedGenerationWithOneGibibyteBank() throws {
        let directory = try requireCheckpoint()
        let vocabulary = try MinimalVocabulary(directory: directory)
        let promptTokens = vocabulary.encode(Self.promptText)
        let (model, session, loadSeconds) = try loadStreamed(
            directory, bankBytes: 1 << 30)

        let measurement = try generate(
            model: model, tokens: 24, label: "P1 streamed, 1 GiB bank",
            loadSeconds: loadSeconds, session: session, promptTokens: promptTokens)
        report(measurement)
        print("R56 | P1 prompt: \(Self.promptText)")
        print("R56 | P1 completion: \(vocabulary.decode(measurement.tokens))")

        XCTAssertEqual(measurement.tokens.count, 24)
        XCTAssertGreaterThan(session.bank.statistics.hits + session.bank.statistics.misses, 0)
    }

    // MARK: - P1b · the price of synchronization

    /// Three arms in one process so the numbers are comparable: the resident
    /// model, streaming with a bank that cannot miss after warm-up, and
    /// streaming with a 1 GiB bank. The gap between arms one and two is the
    /// cost of the ~80 host round trips per token; the gap between two and
    /// three is the I/O.
    func testP1bResidentAgainstFullBankAndSmallBank() throws {
        let directory = try requireCheckpoint()
        let vocabulary = try MinimalVocabulary(directory: directory)
        let promptTokens = vocabulary.encode(Self.promptText)
        var measurements = [Measurement]()

        do {
            let (model, loadSeconds) = try loadResident(directory)
            measurements.append(
                try generate(
                    model: model, tokens: 24, label: "P1b resident",
                    loadSeconds: loadSeconds, session: nil, promptTokens: promptTokens))
            report(measurements[0])
        }
        GPU.clearCache()

        do {
            let index = try ExpertOffsetIndex.build(modelDirectory: directory)
            let full = index.routedBytes
            let (model, session, loadSeconds) = try loadStreamed(directory, bankBytes: full)
            // Warm-up pass so the bank holds the working set before timing.
            _ = try generate(
                model: model, tokens: 4, label: "warm-up", loadSeconds: 0, session: session, promptTokens: promptTokens)
            session.bank.resetStatistics()
            session.store.resetStatistics()
            measurements.append(
                try generate(
                    model: model, tokens: 24, label: "P1b streamed, full bank",
                    loadSeconds: loadSeconds, session: session,
                    promptTokens: promptTokens))
            report(measurements[1])
        }
        GPU.clearCache()

        do {
            let (model, session, loadSeconds) = try loadStreamed(directory, bankBytes: 1 << 30)
            _ = try generate(
                model: model, tokens: 4, label: "warm-up", loadSeconds: 0, session: session, promptTokens: promptTokens)
            session.bank.resetStatistics()
            session.store.resetStatistics()
            measurements.append(
                try generate(
                    model: model, tokens: 24, label: "P1b streamed, 1 GiB bank",
                    loadSeconds: loadSeconds, session: session,
                    promptTokens: promptTokens))
            report(measurements[2])
        }

        // Correctness is the point of running all three: streaming must not
        // change a single token.
        print("R56 | P1b resident completion: \(vocabulary.decode(measurements[0].tokens))")
        print("R56 | P1b streamed completion: \(vocabulary.decode(measurements[1].tokens))")
        XCTAssertEqual(measurements[1].tokens, measurements[0].tokens)
        XCTAssertEqual(measurements[2].tokens, measurements[0].tokens)
    }

    // MARK: - P1c · how close are the logits, really?

    /// Teacher-forced comparison of resident and streamed logits.
    ///
    /// Free-running greedy decoding is the wrong instrument for this question:
    /// one near-tie flips a token and the two sequences never meet again, which
    /// says nothing about how close the computation is. Feeding both models the
    /// same tokens and comparing the logits row by row does.
    func testP1cLogitAgreementUnderTeacherForcing() throws {
        let directory = try requireCheckpoint()
        let vocabulary = try MinimalVocabulary(directory: directory)
        let promptTokens = vocabulary.encode(Self.promptText)
        let steps = 16

        let (resident, residentLoad) = try loadResident(directory)
        let residentRun = try generate(
            model: resident, tokens: steps, label: "P1c resident",
            loadSeconds: residentLoad, session: nil, promptTokens: promptTokens,
            captureLogits: true)
        let forcing = residentRun.tokens

        // Control: the same model, run again in the same process. Without this
        // a difference between two models says nothing, because it could be
        // the machine rather than the code.
        let residentRepeat = try generate(
            model: resident, tokens: steps, label: "P1c resident repeat",
            loadSeconds: 0, session: nil, promptTokens: promptTokens,
            forcing: forcing, captureLogits: true)
        var controlWorst: Float = 0
        for step in 0 ..< steps {
            let difference = MLX.max(
                MLX.abs(residentRun.logits[step] - residentRepeat.logits[step]))
            let scale = MLX.max(MLX.abs(residentRun.logits[step]))
            eval(difference, scale)
            controlWorst = max(
                controlWorst, difference.item(Float.self) / scale.item(Float.self))
        }
        print("R56 | P1c control resident-vs-itself worst relative \(String(format: "%.6f", controlWorst))")

        GPU.clearCache()

        let (streamed, session, streamedLoad) = try loadStreamed(directory, bankBytes: 1 << 30)
        let streamedRun = try generate(
            model: streamed, tokens: steps, label: "P1c streamed",
            loadSeconds: streamedLoad, session: session, promptTokens: promptTokens,
            forcing: forcing, captureLogits: true)

        var worstDifference: Float = 0
        var worstRelative: Float = 0
        var agreements = 0
        for step in 0 ..< steps {
            let a = residentRun.logits[step]
            let b = streamedRun.logits[step]
            let difference = MLX.max(MLX.abs(a - b))
            let scale = MLX.max(MLX.abs(a))
            eval(difference, scale)
            worstDifference = max(worstDifference, difference.item(Float.self))
            worstRelative = max(worstRelative, difference.item(Float.self) / scale.item(Float.self))
            if residentRun.tokens[step] == streamedRun.tokens[step] { agreements += 1 }
        }

        print(
            """
            R56 | P1c teacher-forced: top-1 agreement \(agreements)/\(steps), \
            worst absolute logit difference \(String(format: "%.4f", worstDifference)), \
            worst relative \(String(format: "%.5f", worstRelative))
            """)

        XCTAssertGreaterThanOrEqual(
            agreements, steps - 2,
            "streamed logits should pick the same token at almost every position")
        XCTAssertLessThan(
            worstRelative, 0.01,
            "streamed logits diverged from the resident model by more than rounding")
    }

    // MARK: - P1d · where does the difference come from?

    /// Isolates the streamed expert path from everything else: the full GLU of
    /// layer 0 computed from the checkpoint tensors with global expert ids,
    /// against the same computation over rows read by the store with local
    /// ids. Also compares every one of the nine pieces byte for byte.
    func testP1dSubsetMatmulEquivalence() throws {
        let directory = try requireCheckpoint()
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let store = try ExpertResidencyStore(index: index, modelDirectory: directory)

        let record = index.records(forLayer: 0)![ExpertPiece(.gate, .weight)]
        let shard = index.shardURL(record.shard, relativeTo: directory)
        let loaded = try loadArrays(url: shard)
        func tensor(_ piece: ExpertPiece) throws -> MLXArray {
            let suffix = "\(piece.projection.rawValue).\(piece.component.rawValue)"
            let key = try XCTUnwrap(
                loaded.keys.first {
                    $0.contains("layers.0.mlp.switch_mlp.") && $0.hasSuffix(suffix)
                })
            return loaded[key]!
        }
        let checkpoint = try ExpertPiece.all.map { try tensor($0) }

        let tokens = 6
        let topK = 8
        MLXRandom.seed(56)
        let inputDims = checkpoint[ExpertPiece(.gate, .weight).slot].dim(2) * 8
        let x = MLXRandom.normal([tokens, 1, 1, inputDims]).asType(.bfloat16)
        let chosen: [UInt32] = (0 ..< tokens * topK).map { UInt32(($0 * 29 + 7) % 256) }
        let globalIndices = MLXArray(chosen).reshaped(tokens, topK)

        let unique = Array(Set(chosen)).sorted()
        var local = [UInt32: UInt32]()
        for (row, expert) in unique.enumerated() { local[expert] = UInt32(row) }
        let localIndices = MLXArray(chosen.map { local[$0]! }).reshaped(tokens, topK)
        let uniqueArray = MLXArray(unique)

        let staged = try store.readBatch(
            keys: unique.map { ExpertKey(layer: 0, expert: Int($0)) })

        // Every piece, byte for byte, against MLX's own gather of the same rows.
        for piece in ExpertPiece.all {
            let expected = checkpoint[piece.slot][uniqueArray]
            eval(expected)
            XCTAssertEqual(
                expected.asData(access: .copy).data,
                staged[piece.slot].asData(access: .copy).data,
                "piece \(piece.projection.rawValue).\(piece.component.rawValue)")
        }

        func project(
            _ pools: [MLXArray], _ projection: ExpertProjection, _ input: MLXArray,
            _ idx: MLXArray
        ) -> MLXArray {
            MLX.gatherQuantizedMM(
                input,
                pools[ExpertPiece(projection, .weight).slot],
                scales: pools[ExpertPiece(projection, .scales).slot],
                biases: pools[ExpertPiece(projection, .biases).slot],
                rhsIndices: idx, transpose: true, groupSize: 64, bits: 4, mode: .affine,
                sortedIndices: false)
        }

        func glu(_ pools: [MLXArray], _ idx: MLXArray) -> MLXArray {
            let up = project(pools, .up, x, idx)
            let gate = project(pools, .gate, x, idx)
            return MLX.squeezed(
                project(pools, .down, MLXNN.silu(gate) * up, idx), axis: -2)
        }

        func worstRelative(_ a: MLXArray, _ b: MLXArray) -> Float {
            let difference = MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32)))
            let scale = MLX.max(MLX.abs(a.asType(.float32)))
            eval(difference, scale)
            return difference.item(Float.self) / max(scale.item(Float.self), 1e-8)
        }

        let reference = glu(checkpoint, globalIndices)
        let gathered = glu(ExpertPiece.all.map { checkpoint[$0.slot][uniqueArray] }, localIndices)
        let fromDisk = glu(staged, localIndices)

        print(
            """
            R56 | P1d full-vs-gathered \(String(format: "%.6f", worstRelative(reference, gathered))) \
            | full-vs-disk \(String(format: "%.6f", worstRelative(reference, fromDisk)))
            """)

        XCTAssertLessThan(worstRelative(reference, gathered), 1e-3)
        XCTAssertLessThan(worstRelative(reference, fromDisk), 1e-3)
    }

    // MARK: - P1e · block against block

    /// Runs layer 0's MoE block from both loaded models on the same input and
    /// compares the pieces: the router's choice, the routed combination and
    /// the shared expert. Narrows the divergence to one of them.
    func testP1eBlockAgainstBlock() throws {
        let directory = try requireCheckpoint()
        let (resident, _) = try loadResident(directory)
        let (streamed, _, _) = try loadStreamed(directory, bankBytes: 1 << 30)

        let residentBlock = try XCTUnwrap(
            resident.languageModel.model.layers[0].mlp as? Qwen35SparseMoeBlock)
        let streamedBlock = try XCTUnwrap(
            streamed.languageModel.model.layers[0].mlp as? Qwen35StreamedSparseMoeBlock)

        MLXRandom.seed(56)
        let x = MLXRandom.normal([1, 6, 2048]).asType(.bfloat16)
        eval(x)

        func worstRelative(_ a: MLXArray, _ b: MLXArray) -> Float {
            let difference = MLX.max(MLX.abs(a.asType(.float32) - b.asType(.float32)))
            let scale = MLX.max(MLX.abs(a.asType(.float32)))
            eval(difference, scale)
            return difference.item(Float.self) / max(scale.item(Float.self), 1e-8)
        }

        let residentGate = residentBlock.gate(x)
        let streamedGate = streamedBlock.gate(x)
        eval(residentGate, streamedGate)
        print("R56 | P1e router logits \(String(format: "%.6f", worstRelative(residentGate, streamedGate)))")

        let residentShared =
            sigmoid(residentBlock.sharedExpertGate(x)) * residentBlock.sharedExpert(x)
        let streamedShared =
            sigmoid(streamedBlock.sharedExpertGate(x)) * streamedBlock.sharedExpert(x)
        eval(residentShared, streamedShared)
        print("R56 | P1e shared expert \(String(format: "%.6f", worstRelative(residentShared, streamedShared)))")

        // Same router choice, same input, two ways of supplying the experts.
        let gates = MLX.softmax(residentBlock.gate(x), axis: -1, precise: true)
        let (inds, scores) = moeRouterTopK(gates, k: 8, normalize: true)
        let flatX = x.reshaped(6, 2048)
        let flatIndices = inds.reshaped(6, 8)
        let flatScores = scores.reshaped(6, 8)
        eval(flatIndices, flatScores)
        let experts = flatIndices.asArray(UInt32.self).map { Int($0) }
        let resolution = try streamedBlock.session.resolve(layer: 0, experts: experts)
        eval(resolution.indices)

        // The rows the store serves must be the rows the resident model holds.
        // This is the assertion that caught the MTP key collision: the index
        // had bound layer 0 to the multi-token-prediction head's experts, and
        // every other check still passed.
        let residentParameters = Dictionary(
            uniqueKeysWithValues: residentBlock.parameters().flattened().map { ($0.0, $0.1) })
        let uniqueExperts = MLXArray(Array(Set(experts)).sorted().map { UInt32($0) })
        let residentRows = residentParameters["switch_mlp.gate_proj.weight"]![uniqueExperts]
        let streamedRows = resolution.pools![ExpertPiece(.gate, .weight).slot]
        eval(residentRows, streamedRows)
        XCTAssertEqual(
            residentRows.asData(access: .copy).data,
            streamedRows.asData(access: .copy).data,
            "the store served different bytes than the resident model holds")

        let residentCombined = residentBlock.switchMLP.callAndWeightedReduce(
            flatX, flatIndices, weights: flatScores, fuseSortedReduction: true)
        let streamedCombined = streamedBlock.switchMLP.callAndWeightedReduce(
            flatX, localIndices: resolution.indices.reshaped(6, 8),
            pools: resolution.pools!, weights: flatScores)
        eval(residentCombined, streamedCombined)
        print("R56 | P1e routed combination \(String(format: "%.6f", worstRelative(residentCombined, streamedCombined)))")
        XCTAssertLessThan(worstRelative(residentCombined, streamedCombined), 1e-4)

        let residentOut = residentBlock.forward(x)
        let streamedOut = streamedBlock(x)
        eval(residentOut, streamedOut)
        print("R56 | P1e block output \(String(format: "%.6f", worstRelative(residentOut, streamedOut)))")

        XCTAssertLessThan(worstRelative(residentGate, streamedGate), 1e-4)
        XCTAssertLessThan(worstRelative(residentShared, streamedShared), 1e-4)
        XCTAssertLessThan(worstRelative(residentOut, streamedOut), 1e-3)
    }

    // MARK: - P1f · what does prefill cost in memory?

    /// Peak allocator memory against prompt length, with the bank fixed.
    ///
    /// The prefill sweep reads every expert the prompt touched into transient
    /// staging. If those arrays stay alive across all 40 layers instead of
    /// being released layer by layer, the peak grows with the prompt — which
    /// is survivable on a 128 GB Mac and fatal under a 6 GiB process ceiling.
    func testP1fPrefillStagingPeak() throws {
        let directory = try requireCheckpoint()
        let vocabulary = try MinimalVocabulary(directory: directory)

        let prompts = [
            "Paris",
            "The capital of France is the city of",
            String(repeating: "The capital of France is the city of Paris and ", count: 4),
        ]

        for prompt in prompts {
            let tokens = vocabulary.encode(prompt)
            let (model, session, _) = try loadStreamed(directory, bankBytes: 1 << 30)
            GPU.clearCache()
            let measurement = try generate(
                model: model, tokens: 2, label: "P1f prompt \(tokens.count) tokens",
                loadSeconds: 0, session: session, promptTokens: tokens)
            let staged = session.store.statistics.bytes
            print(
                """
                R56 | P1f prompt \(tokens.count) tokens | peak GPU \
                \(String(format: "%.2f", Double(measurement.peakGPUBytes) / 1_073_741_824)) GiB \
                | bytes read \(String(format: "%.2f", Double(staged) / 1_073_741_824)) GiB
                """)
            GPU.clearCache()
        }
    }

    // MARK: - P3 · a stable bank against per-token materialization

    /// The "intentionally bad" diagnostic: a bank of eight slots holds exactly
    /// the experts of one layer of one token, so every layer of every token
    /// evicts everything and reinstalls. That is per-token materialization by
    /// another name, and the comparison against a bank that fits the working
    /// set is what makes the case for a stable bank in this stack.
    func testP3StableBankAgainstPerTokenMaterialization() throws {
        let directory = try requireCheckpoint()
        let vocabulary = try MinimalVocabulary(directory: directory)
        let promptTokens = vocabulary.encode(Self.promptText)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)

        let sizes: [(String, Int)] = [
            ("P3 bank 8 slots (per-token materialization)", 8 * index.bytesPerExpert),
            ("P3 bank 1 GiB", 1 << 30),
            ("P3 bank 3 GiB", 3 << 30),
        ]

        for (label, bytes) in sizes {
            let (model, session, loadSeconds) = try loadStreamed(directory, bankBytes: bytes)
            _ = try generate(
                model: model, tokens: 4, label: "warm-up", loadSeconds: 0, session: session, promptTokens: promptTokens)
            session.bank.resetStatistics()
            session.store.resetStatistics()
            let measurement = try generate(
                model: model, tokens: 16, label: label, loadSeconds: loadSeconds,
                session: session, promptTokens: promptTokens)
            report(measurement)
            GPU.clearCache()
        }
    }
}
