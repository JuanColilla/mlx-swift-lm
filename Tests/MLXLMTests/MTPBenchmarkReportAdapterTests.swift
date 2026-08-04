// Copyright © 2026 Apple Inc.

import BenchmarkHelpers
import Foundation
import MLXLMCommon
import Testing

@Suite
struct MTPBenchmarkReportAdapterTests {
    @Test
    func completionInfoMapsToComparableBenchmarkJSON() throws {
        let info = GenerateCompletionInfo(
            promptTokenCount: 8,
            generationTokenCount: 12,
            promptTime: 0.25,
            generationTime: 0.5,
            stopReason: .length,
            proposedDraftTokens: 9,
            acceptedDraftTokens: 6
        )
        let entry = info.mtpBenchmarkEntry(
            context: "target+drafter@block4",
            generationTimesMilliseconds: [500, 510, 490]
        )
        let report = BenchmarkReport(
            label: "candidate-commit",
            platform: "test-platform",
            generatedAt: Date(timeIntervalSince1970: 0),
            entries: [entry]
        )

        let data = try report.jsonData()
        let decoded = try JSONDecoder.benchmarkReportDecoder.decode(
            BenchmarkReport.self,
            from: data
        )
        let metrics = try #require(decoded.entries.first?.metrics)

        #expect(metrics["tokensPerSecond"] == 24)
        #expect(metrics["generationTokenCount"] == 12)
        #expect(metrics["proposedDraftTokens"] == 9)
        #expect(metrics["acceptedDraftTokens"] == 6)
        #expect(metrics["acceptanceRate"] == 2.0 / 3.0)
        #expect(metrics["didPassthrough"] == 0)
    }
}

extension JSONDecoder {
    fileprivate static var benchmarkReportDecoder: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
