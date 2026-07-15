// Copyright © 2026 Apple Inc.

import BenchmarkHelpers
import Foundation
import Testing

struct BenchmarkReportTests {
    @Test func reportRoundTripsThroughDisk() throws {
        let generatedAt = Date(timeIntervalSince1970: 1_750_000_000)
        let entry = BenchmarkStats(times: [1, 2, 3]).entry(
            name: "decode",
            context: "test-model",
            metrics: ["tokensPerSecond": 42]
        )
        let report = BenchmarkReport(
            label: "candidate",
            platform: "macOS test arm64",
            generatedAt: generatedAt,
            entries: [entry]
        )
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        defer { try? FileManager.default.removeItem(at: url) }

        try report.write(to: url)
        let decoded = try BenchmarkReport.load(from: url)

        #expect(decoded.schemaVersion == BenchmarkReport.currentSchemaVersion)
        #expect(decoded.generatedAt == generatedAt)
        #expect(decoded.label == "candidate")
        #expect(decoded.platform == "macOS test arm64")
        #expect(decoded.entries == [entry])
    }

    @Test func comparisonMatchesEntriesAndReportsMedianDelta() {
        let baseline = BenchmarkReport(
            entries: [
                BenchmarkStats(times: [10, 20, 30]).entry(
                    name: "decode", context: "test-model"),
                BenchmarkStats(times: [1]).entry(name: "baseline-only"),
            ]
        )
        let candidate = BenchmarkReport(
            entries: [
                BenchmarkStats(times: [15, 30, 45]).entry(
                    name: "decode", context: "test-model"),
                BenchmarkStats(times: [1]).entry(name: "candidate-only"),
            ]
        )

        let comparisons = candidate.compared(against: baseline)

        #expect(
            comparisons == [
                BenchmarkComparison(
                    name: "decode",
                    context: "test-model",
                    baselineMedian: 20,
                    candidateMedian: 30
                )
            ])
        #expect(comparisons.first?.medianDeltaPercent == 50)
    }
}
