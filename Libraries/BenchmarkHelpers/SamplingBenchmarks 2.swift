// Microbenchmarks for the logit sampling and penalty-processing primitives in
// MLXLMCommon (`Evaluate.swift`). These operate on synthetic logits so they can
// run without downloading or loading any model, isolating the cost of sampling
// itself from prefill/decode.
//
// See DOCS/tech-debt-and-research-backlog.md, "Perfilador de sampling".

import Foundation
import MLX
import MLXLMCommon

public enum SamplingBenchmarkDefaults {
    /// Representative vocabulary sizes: a small/legacy tokenizer, and the
    /// larger vocabularies used by recent Qwen/Llama/Gemma families.
    public static let vocabularySizes = [32_000, 128_000, 152_000]
    public static let runs = 25
    public static let warmupRuns = 3
    /// Context window used to seed repetition/presence/frequency penalty rings.
    public static let penaltyContextSize = 64
}

/// Timing for a single sampler or logit processor, measured on synthetic logits
/// of a given vocabulary size.
public struct SamplingBenchmarkResult: Sendable {
    public let name: String
    public let vocabularySize: Int
    public let stats: BenchmarkStats

    public init(name: String, vocabularySize: Int, stats: BenchmarkStats) {
        self.name = name
        self.vocabularySize = vocabularySize
        self.stats = stats
    }

    public func printSummary() {
        print(
            "\(name) (vocab=\(vocabularySize)): "
                + "median \(String(format: "%.3f", stats.median))ms, "
                + "mean \(String(format: "%.3f", stats.mean))ms, "
                + "stddev \(String(format: "%.3f", stats.stdDev))ms"
        )
    }
}

/// Generate deterministic synthetic logits, shaped `[1, vocabularySize]`, to
/// drive sampling benchmarks without a real model.
private func syntheticLogits(vocabularySize: Int, seed: UInt64) -> MLXArray {
    let logits = MLXRandom.normal([1, vocabularySize], key: MLXRandom.key(seed))
    eval(logits)
    return logits
}

/// Generate a plausible "recent token" context for penalty processors:
/// random token ids in `[0, vocabularySize)`.
private func syntheticContext(vocabularySize: Int, size: Int, seed: UInt64) -> MLXArray {
    let tokens = MLXRandom.randInt(
        low: 0, high: vocabularySize, [size], key: MLXRandom.key(seed))
    eval(tokens)
    return tokens
}

/// Time `runs` invocations of `body(logits)` after `warmupRuns` untimed
/// warm-up calls, returning per-call latency statistics in milliseconds.
private func measureSampling(
    name: String,
    vocabularySize: Int,
    runs: Int,
    warmupRuns: Int,
    logits: MLXArray,
    _ body: (MLXArray) -> MLXArray
) -> SamplingBenchmarkResult {
    for _ in 0 ..< warmupRuns {
        eval(body(logits))
    }

    var times: [Double] = []
    times.reserveCapacity(runs)
    for _ in 0 ..< runs {
        let start = CFAbsoluteTimeGetCurrent()
        eval(body(logits))
        times.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
    }

    return SamplingBenchmarkResult(
        name: name, vocabularySize: vocabularySize, stats: BenchmarkStats(times: times))
}

/// Benchmark the built-in ``LogitSampler`` implementations (argmax, categorical,
/// top-p, top-k, min-p, and all three filters combined) on synthetic logits of
/// the given vocabulary size.
///
/// This isolates sampling cost from model inference cost -- useful for judging
/// whether `topP`/`topK`/`minP` filters are worth their overhead at a given
/// vocabulary size, per DOCS/tech-debt-and-research-backlog.md item 8.
public func benchmarkSamplers(
    vocabularySize: Int = 152_000,
    runs: Int = SamplingBenchmarkDefaults.runs,
    warmupRuns: Int = SamplingBenchmarkDefaults.warmupRuns
) -> [SamplingBenchmarkResult] {
    let logits = syntheticLogits(vocabularySize: vocabularySize, seed: 0)

    let argMax = ArgMaxSampler()
    let categorical = CategoricalSampler(temperature: 0.8, seed: 1)
    let topP = TopPSampler(temperature: 0.8, topP: 0.95, seed: 2)
    let topK = TopPSampler(temperature: 0.8, topK: 50, seed: 3)
    let minP = TopPSampler(temperature: 0.8, minP: 0.05, seed: 4)
    let combined = TopPSampler(temperature: 0.8, topP: 0.95, topK: 50, minP: 0.05, seed: 5)

    return [
        measureSampling(
            name: "argMax", vocabularySize: vocabularySize, runs: runs, warmupRuns: warmupRuns,
            logits: logits
        ) { argMax.sample(logits: $0) },
        measureSampling(
            name: "categorical", vocabularySize: vocabularySize, runs: runs,
            warmupRuns: warmupRuns, logits: logits
        ) { categorical.sample(logits: $0) },
        measureSampling(
            name: "topP", vocabularySize: vocabularySize, runs: runs, warmupRuns: warmupRuns,
            logits: logits
        ) { topP.sample(logits: $0) },
        measureSampling(
            name: "topK", vocabularySize: vocabularySize, runs: runs, warmupRuns: warmupRuns,
            logits: logits
        ) { topK.sample(logits: $0) },
        measureSampling(
            name: "minP", vocabularySize: vocabularySize, runs: runs, warmupRuns: warmupRuns,
            logits: logits
        ) { minP.sample(logits: $0) },
        measureSampling(
            name: "topP+topK+minP", vocabularySize: vocabularySize, runs: runs,
            warmupRuns: warmupRuns, logits: logits
        ) { combined.sample(logits: $0) },
    ]
}

/// Benchmark the ``LogitProcessor`` penalty implementations (repetition,
/// presence, frequency, and all three combined via ``PenaltyProcessor``) on
/// synthetic logits and a synthetic recent-token context.
public func benchmarkPenaltyProcessors(
    vocabularySize: Int = 152_000,
    contextSize: Int = SamplingBenchmarkDefaults.penaltyContextSize,
    runs: Int = SamplingBenchmarkDefaults.runs,
    warmupRuns: Int = SamplingBenchmarkDefaults.warmupRuns
) -> [SamplingBenchmarkResult] {
    let logits = syntheticLogits(vocabularySize: vocabularySize, seed: 100)
    let context = syntheticContext(vocabularySize: vocabularySize, size: contextSize, seed: 101)

    var repetition = RepetitionContext(repetitionPenalty: 1.1, repetitionContextSize: contextSize)
    repetition.prompt(context)

    var presence = PresencePenaltyContext(
        presencePenalty: 1.0, presenceContextSize: contextSize)
    presence.prompt(context)

    var frequency = FrequencyPenaltyContext(
        frequencyPenalty: 0.5, frequencyContextSize: contextSize)
    frequency.prompt(context)

    let combined = PenaltyProcessor(
        repetitionContext: repetition, presenceContext: presence, frequencyContext: frequency)

    return [
        measureSampling(
            name: "repetitionPenalty", vocabularySize: vocabularySize, runs: runs,
            warmupRuns: warmupRuns, logits: logits
        ) { repetition.process(logits: $0) },
        measureSampling(
            name: "presencePenalty", vocabularySize: vocabularySize, runs: runs,
            warmupRuns: warmupRuns, logits: logits
        ) { presence.process(logits: $0) },
        measureSampling(
            name: "frequencyPenalty", vocabularySize: vocabularySize, runs: runs,
            warmupRuns: warmupRuns, logits: logits
        ) { frequency.process(logits: $0) },
        measureSampling(
            name: "allPenaltiesCombined", vocabularySize: vocabularySize, runs: runs,
            warmupRuns: warmupRuns, logits: logits
        ) { combined.process(logits: $0) },
    ]
}

/// Run both sampler and penalty-processor microbenchmarks across
/// ``SamplingBenchmarkDefaults/vocabularySizes``, printing a summary for each.
public func benchmarkSamplingSuite(
    vocabularySizes: [Int] = SamplingBenchmarkDefaults.vocabularySizes,
    runs: Int = SamplingBenchmarkDefaults.runs,
    warmupRuns: Int = SamplingBenchmarkDefaults.warmupRuns
) -> [SamplingBenchmarkResult] {
    var results: [SamplingBenchmarkResult] = []
    for vocabularySize in vocabularySizes {
        let samplerResults = benchmarkSamplers(
            vocabularySize: vocabularySize, runs: runs, warmupRuns: warmupRuns)
        let penaltyResults = benchmarkPenaltyProcessors(
            vocabularySize: vocabularySize, runs: runs, warmupRuns: warmupRuns)
        for result in samplerResults + penaltyResults {
            result.printSummary()
        }
        results.append(contentsOf: samplerResults)
        results.append(contentsOf: penaltyResults)
    }
    return results
}
