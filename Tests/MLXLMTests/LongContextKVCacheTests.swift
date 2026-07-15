// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private func kvTensor(_ values: [Float]) -> MLXArray {
    MLXArray(values, [1, 1, values.count, 1])
}

private func expectArraysClose(_ lhs: [MLXArray], _ rhs: [MLXArray]) {
    #expect(lhs.count == rhs.count)
    for (left, right) in zip(lhs, rhs) {
        #expect(left.shape == right.shape)
        #expect(allClose(left, right).item(Bool.self))
    }
}

private final class KVCacheSelectionModel: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [1, 1] }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }
}

@Suite(.serialized)
struct LongContextKVCacheTests {
    @Test
    func testKVCacheSimpleTrimRewindsLogicalStateAndAllowsOverwrite() {
        let cache = KVCacheSimple()
        _ = cache.update(
            keys: kvTensor([0, 1, 2, 3, 4, 5]),
            values: kvTensor([10, 11, 12, 13, 14, 15])
        )

        #expect(cache.trim(-1) == 0)
        #expect(cache.offset == 6)
        #expect(cache.trim(2) == 2)
        #expect(cache.offset == 4)
        #expect(cache.state[0].shape == [1, 1, 4, 1])

        let (keys, values) = cache.update(
            keys: kvTensor([40, 50]),
            values: kvTensor([140, 150])
        )

        #expect(keys.asArray(Float.self) == [0, 1, 2, 3, 40, 50])
        #expect(values.asArray(Float.self) == [10, 11, 12, 13, 140, 150])
    }

    @Test
    func testTrimPromptCacheKeepsLayerOffsetsAligned() {
        let caches = [KVCacheSimple(), KVCacheSimple()]
        for cache in caches {
            _ = cache.update(keys: kvTensor([0, 1, 2, 3]), values: kvTensor([4, 5, 6, 7]))
        }

        #expect(canTrimPromptCache(caches))
        #expect(trimPromptCache(caches, numTokens: 3) == 3)
        #expect(caches.map(\.offset) == [1, 1])
        #expect(trimPromptCache(caches, numTokens: -1) == 0)
        #expect(caches.map(\.offset) == [1, 1])
    }

    @Test
    func testRotatingKVCacheRefusesTrimAfterReachingWindow() {
        let cache = RotatingKVCache(maxSize: 8)
        _ = cache.update(keys: kvTensor([0, 1, 2, 3]), values: kvTensor([4, 5, 6, 7]))

        #expect(cache.isTrimmable)
        #expect(cache.trim(2) == 2)
        #expect(cache.offset == 2)

        _ = cache.update(
            keys: kvTensor([10, 11, 12, 13, 14, 15]),
            values: kvTensor([20, 21, 22, 23, 24, 25])
        )

        #expect(cache.offset == 8)
        #expect(!cache.isTrimmable)
        #expect(cache.trim(1) == 0)
        #expect(cache.offset == 8)
    }

    @Test
    func testRotatingKVCacheMaskTracksWrappedSingleTokenWindow() throws {
        let cache = RotatingKVCache(maxSize: 8)
        for token in 0 ... 8 {
            _ = cache.update(
                keys: kvTensor([Float(token)]),
                values: kvTensor([Float(token)])
            )
        }

        let mode = cache.makeMask(n: 1, windowSize: 4, returnArray: false)
        guard case .array(let mask) = mode else {
            Issue.record("Expected an array mask after the rotating cache wraps")
            return
        }

        #expect(mask.shape == [8])
        #expect(mask.asArray(Bool.self) == [true, true, false, false, false, false, true, true])
    }

    @Test
    func testRotatingKVCacheRoundTripPreservesWrappedIndex() throws {
        let original = RotatingKVCache(maxSize: 8)
        for token in 0 ... 10 {
            _ = original.update(
                keys: kvTensor([Float(token)]),
                values: kvTensor([Float(token + 100)])
            )
        }
        #expect(original.offset == 11)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        try savePromptCache(url: url, cache: [original])
        let (loaded, _) = try loadPromptCache(url: url)
        let restored = try #require(loaded.first as? RotatingKVCache)

        #expect(restored.metaState == original.metaState)
        expectArraysClose(restored.state, original.state)

        let originalResult = original.update(keys: kvTensor([99]), values: kvTensor([199]))
        let restoredResult = restored.update(keys: kvTensor([99]), values: kvTensor([199]))
        expectArraysClose(
            [originalResult.0, originalResult.1],
            [restoredResult.0, restoredResult.1]
        )
        #expect(restored.metaState == original.metaState)
    }

    @Test
    func testMaxKVSizeAndKVBitsKeepRotatingCacheUnquantized() {
        let model = KVCacheSelectionModel()
        let parameters = GenerateParameters(
            maxKVSize: 8,
            kvBits: 4,
            kvGroupSize: 64,
            quantizedKVStart: 0
        )
        var caches = model.newCache(parameters: parameters)
        for cache in caches {
            _ = cache.update(
                keys: MLXArray.ones([1, 1, 1, 64], dtype: .bfloat16),
                values: MLXArray.ones([1, 1, 1, 64], dtype: .bfloat16)
            )
        }

        maybeQuantizeKVCache(
            cache: &caches,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart
        )

        #expect(caches.count == 2)
        #expect(caches.allSatisfy { $0 is RotatingKVCache })
        #expect(caches.allSatisfy { !($0 is QuantizedKVCache) })
    }
}
