// JSON reporting for BenchmarkHelpers results, so runs can be diffed across
// commits/presets instead of only read as console output.
//
// See DOCS/tech-debt-and-research-backlog.md, "Benchmark suite reproducible".

import Foundation

/// A single named timing measurement, ready for JSON serialization.
///
/// Constructed from ``BenchmarkStats`` (see `BenchmarkStats.entry(name:context:)`)
/// so every existing benchmark function in this module can be reported without
/// changing its return type.
public struct BenchmarkReportEntry: Codable, Sendable, Equatable {
    /// Name of the thing measured, e.g. "argMax", "topP", "LLM decode".
    public let name: String
    /// Free-form context for grouping/filtering, e.g. a model id, vocabulary
    /// size, or preset name. Not parsed -- purely descriptive.
    public let context: String?
    public let unitMs: Bool
    public let mean: Double
    public let median: Double
    public let stdDev: Double
    public let min: Double
    public let max: Double
    /// Additional scalar metrics that don't fit the mean/median/stdDev shape,
    /// e.g. "tokensPerSecond", "promptTokenCount".
    public let metrics: [String: Double]

    public init(
        name: String,
        context: String? = nil,
        stats: BenchmarkStats,
        metrics: [String: Double] = [:]
    ) {
        self.name = name
        self.context = context
        self.unitMs = true
        self.mean = stats.mean
        self.median = stats.median
        self.stdDev = stats.stdDev
        self.min = stats.min
        self.max = stats.max
        self.metrics = metrics
    }
}

extension BenchmarkStats {
    /// Wrap this result as a ``BenchmarkReportEntry`` for JSON reporting.
    public func entry(name: String, context: String? = nil, metrics: [String: Double] = [:])
        -> BenchmarkReportEntry
    {
        BenchmarkReportEntry(name: name, context: context, stats: self, metrics: metrics)
    }
}

extension LLMGenerationStats {
    /// Wrap this result as a ``BenchmarkReportEntry`` pair (prefill + decode),
    /// including throughput as extra metrics since tok/s is the number most
    /// benchmark comparisons actually care about.
    public func entries(name: String, context: String? = nil) -> [BenchmarkReportEntry] {
        [
            promptTimeStats.entry(
                name: "\(name).prefill", context: context,
                metrics: [
                    "tokensPerSecond": prefillTokensPerSecond,
                    "tokenCount": Double(promptTokenCount),
                ]),
            generateTimeStats.entry(
                name: "\(name).decode", context: context,
                metrics: [
                    "tokensPerSecond": decodeTokensPerSecond,
                    "tokenCount": Double(generationTokenCount),
                ]),
        ]
    }
}

extension SamplingBenchmarkResult {
    /// Wrap this result as a ``BenchmarkReportEntry``.
    public var entry: BenchmarkReportEntry {
        stats.entry(name: name, context: "vocab=\(vocabularySize)")
    }
}

/// A full benchmark run: a set of named entries plus enough metadata to
/// compare runs across commits, devices, and presets.
///
/// JSON output is stable (sorted keys, ISO-8601 dates) so it can be diffed
/// directly or fed into a script that compares two `BenchmarkReport` files.
public struct BenchmarkReport: Codable, Sendable {
    /// Bump when the shape of ``BenchmarkReportEntry`` changes incompatibly.
    public static let currentSchemaVersion = 1

    public let schemaVersion: Int
    public let generatedAt: Date
    /// Free-form label for what produced this report, e.g. a commit SHA,
    /// branch name, or "mlx-community/Qwen3-0.6B-4bit @ affine4". Callers own
    /// how they populate this -- BenchmarkHelpers has no git dependency.
    public let label: String?
    public let platform: String
    public let entries: [BenchmarkReportEntry]

    public init(
        label: String? = nil,
        platform: String = BenchmarkReport.currentPlatformDescription(),
        generatedAt: Date = Date(),
        entries: [BenchmarkReportEntry]
    ) {
        self.schemaVersion = Self.currentSchemaVersion
        self.generatedAt = generatedAt
        self.label = label
        self.platform = platform
        self.entries = entries
    }

    /// A short platform description, e.g. "macOS 15.2 arm64", useful for
    /// telling apart benchmark runs from different devices at a glance.
    public static func currentPlatformDescription() -> String {
        let info = ProcessInfo.processInfo
        #if arch(arm64)
            let arch = "arm64"
        #elseif arch(x86_64)
            let arch = "x86_64"
        #else
            let arch = "unknown-arch"
        #endif
        #if os(macOS)
            return "macOS \(info.operatingSystemVersionString) \(arch)"
        #elseif os(iOS)
            return "iOS \(info.operatingSystemVersionString) \(arch)"
        #elseif os(tvOS)
            return "tvOS \(info.operatingSystemVersionString) \(arch)"
        #elseif os(visionOS)
            return "visionOS \(info.operatingSystemVersionString) \(arch)"
        #else
            return "unknown-platform \(arch)"
        #endif
    }

    private static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    /// Serialize this report as pretty-printed, key-sorted JSON.
    public func jsonData() throws -> Data {
        try Self.makeEncoder().encode(self)
    }

    /// Serialize and write this report to `url`.
    public func write(to url: URL) throws {
        try jsonData().write(to: url, options: .atomic)
    }

    private static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    /// Load a previously written report for comparison.
    public static func load(from url: URL) throws -> BenchmarkReport {
        try makeDecoder().decode(BenchmarkReport.self, from: Data(contentsOf: url))
    }
}

/// A single named comparison between two benchmark entries with the same
/// `name`/`context`, expressed as a percentage change in the median.
///
/// Positive `medianDeltaPercent` means the candidate is slower than the
/// baseline; negative means faster.
public struct BenchmarkComparison: Codable, Sendable, Equatable {
    public let name: String
    public let context: String?
    public let baselineMedian: Double
    public let candidateMedian: Double
    public let medianDeltaPercent: Double

    public init(name: String, context: String?, baselineMedian: Double, candidateMedian: Double) {
        self.name = name
        self.context = context
        self.baselineMedian = baselineMedian
        self.candidateMedian = candidateMedian
        self.medianDeltaPercent =
            baselineMedian == 0 ? 0 : ((candidateMedian - baselineMedian) / baselineMedian) * 100
    }
}

extension BenchmarkReport {
    /// Compare this report (the candidate) against a `baseline` report,
    /// matching entries by `(name, context)`. Entries present in only one of
    /// the two reports are skipped -- this compares what both runs measured
    /// in common, e.g. after adding/removing a benchmark between commits.
    public func compared(against baseline: BenchmarkReport) -> [BenchmarkComparison] {
        var baselineByKey: [String: BenchmarkReportEntry] = [:]
        for entry in baseline.entries {
            baselineByKey["\(entry.name)|\(entry.context ?? "")"] = entry
        }

        return entries.compactMap { candidate in
            guard let base = baselineByKey["\(candidate.name)|\(candidate.context ?? "")"] else {
                return nil
            }
            return BenchmarkComparison(
                name: candidate.name,
                context: candidate.context,
                baselineMedian: base.median,
                candidateMedian: candidate.median
            )
        }
    }
}
