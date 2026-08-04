// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

extension GenerateCompletionInfo {
    /// Converts an MTP completion into a reusable ``BenchmarkReportEntry``.
    ///
    /// Integration tests own the timing samples and model/drafter context;
    /// this adapter standardizes the scalar MTP metrics stored in report JSON
    /// so runs can be compared across commits and devices.
    ///
    /// - Parameters:
    ///   - name: Stable benchmark name, such as `"mtp.decode"`.
    ///   - context: Target, drafter, block size, and preset description.
    ///   - generationTimesMilliseconds: Repeated end-to-end decode timings.
    public func mtpBenchmarkEntry(
        name: String = "mtp.decode",
        context: String? = nil,
        generationTimesMilliseconds: [Double]
    ) -> BenchmarkReportEntry {
        let proposed = proposedDraftTokens ?? 0
        let accepted = acceptedDraftTokens ?? 0
        let acceptanceRate = proposed > 0 ? Double(accepted) / Double(proposed) : 0

        return BenchmarkStats(times: generationTimesMilliseconds).entry(
            name: name,
            context: context,
            metrics: [
                "tokensPerSecond": tokensPerSecond,
                "generationTokenCount": Double(generationTokenCount),
                "proposedDraftTokens": Double(proposed),
                "acceptedDraftTokens": Double(accepted),
                "acceptanceRate": acceptanceRate,
                "didPassthrough": passthroughReason == nil ? 0 : 1,
            ]
        )
    }
}
