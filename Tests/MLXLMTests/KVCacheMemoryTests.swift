// Copyright © 2026 Apple Inc.

import Testing

@testable import MLXLMCommon

@Suite
struct KVCacheMemoryTests {
    @Test func testEstimateKVCacheBytesForBF16LlamaShape() throws {
        let bytes = try estimateKVCacheBytes(
            numLayers: 28,
            kvHeads: 8,
            headDim: 128,
            maxTokens: 32_768
        )

        #expect(bytes == 3_758_096_384)
    }

    @Test func testEstimateQuantizedKVCacheBytesIncludesAffineMetadata() throws {
        let bytes = try estimateKVCacheBytes(
            numLayers: 28,
            kvHeads: 8,
            headDim: 128,
            maxTokens: 32_768,
            kvBits: 4,
            kvGroupSize: 64
        )

        #expect(bytes == 1_056_964_608)
    }

    @Test func testEstimateQuantizedKVCacheBytesMirrorsEffectiveGroupSizeResolution() throws {
        let bytes = try estimateKVCacheBytes(
            numLayers: 1,
            kvHeads: 1,
            headDim: 32,
            maxTokens: 1,
            kvBits: 4,
            kvGroupSize: 64
        )

        // 32 bytes of packed K/V payload + 8 bytes of scale/bias metadata.
        #expect(bytes == 40)
    }

    @Test func testEstimateKVCacheBytesAllowsAnEmptyCache() throws {
        let bytes = try estimateKVCacheBytes(
            numLayers: Int.max,
            kvHeads: 8,
            headDim: 128,
            maxTokens: 0
        )

        #expect(bytes == 0)
    }

    @Test func testEstimateKVCacheBytesRejectsInvalidConfiguration() {
        #expect(throws: KVCacheMemoryEstimationError.self) {
            _ = try estimateKVCacheBytes(
                numLayers: -1,
                kvHeads: 8,
                headDim: 128,
                maxTokens: 1
            )
        }
        #expect(throws: KVCacheMemoryEstimationError.self) {
            _ = try estimateKVCacheBytes(
                numLayers: 1,
                kvHeads: 8,
                headDim: 128,
                maxTokens: 1,
                bytesPerElement: 0
            )
        }
        #expect(throws: KVCacheMemoryEstimationError.self) {
            _ = try estimateKVCacheBytes(
                numLayers: 1,
                kvHeads: 8,
                headDim: 128,
                maxTokens: 1,
                kvBits: 7
            )
        }
        #expect(throws: KVCacheMemoryEstimationError.self) {
            _ = try estimateKVCacheBytes(
                numLayers: 1,
                kvHeads: 8,
                headDim: 24,
                maxTokens: 1,
                kvBits: 4
            )
        }
    }

    @Test func testEstimateKVCacheBytesReportsArithmeticOverflow() {
        do {
            _ = try estimateKVCacheBytes(
                numLayers: Int.max,
                kvHeads: 2,
                headDim: 128,
                maxTokens: 1
            )
            Issue.record("Expected arithmetic overflow to throw")
        } catch let error as KVCacheMemoryEstimationError {
            #expect(error == .arithmeticOverflow)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}
