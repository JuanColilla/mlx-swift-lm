// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import Testing

@testable import MLXLMCommon

@Suite(.serialized)
struct WalshHadamardKVCacheTests {
    @Test func nativeTransformIsItsOwnInverseAtSupportedHeadDimensions() {
        for dimension in [64, 128] {
            MLXRandom.seed(UInt64(dimension))
            let input = MLXRandom.normal([1, 2, 3, dimension]).asType(.float32)
            let reconstructed = hadamardTransform(hadamardTransform(input))

            #expect(
                allClose(input, reconstructed, rtol: 1e-5, atol: 1e-5).item(Bool.self),
                "orthonormal WHT round trip failed for head dimension \(dimension)"
            )
        }
    }

    @Test func dimensionValidationUsesTheReversiblePowerOfTwoSubset() {
        #expect(isWalshHadamardDimensionSupported(64, dtype: .float32))
        #expect(isWalshHadamardDimensionSupported(128, dtype: .float32))
        #expect(isWalshHadamardDimensionSupported(8192, dtype: .float32))

        #expect(!isWalshHadamardDimensionSupported(80, dtype: .float16))
        #expect(!isWalshHadamardDimensionSupported(96, dtype: .bfloat16))
        #expect(!isWalshHadamardDimensionSupported(288, dtype: .float32))
        #expect(!isWalshHadamardDimensionSupported(16_384, dtype: .float32))
        #expect(!isWalshHadamardDimensionSupported(64, dtype: .int32))
    }

    @Test func inverseTransformRestoresAttentionBasisWithinAffine8Tolerance() {
        MLXRandom.seed(42)
        let queries = (MLXRandom.normal([1, 2, 3, 64]) / 4).asType(.float32)
        let keys = (MLXRandom.normal([1, 2, 3, 64]) / 4).asType(.float32)
        let values = (MLXRandom.normal([1, 2, 3, 64]) / 4).asType(.float32)
        let scale = Float(1 / sqrt(64.0))

        let reference = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: nil,
            scale: scale
        )
        let cache = WalshHadamardQuantizedKVCache(groupSize: 64, bits: 8)
        let actual = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale
        )

        #expect(!(cache is any QuantizedKVCacheProtocol))
        #expect(
            allClose(reference, actual, rtol: 0.03, atol: 0.01).item(Bool.self),
            "WHT cache must return to the original attention basis before SDPA"
        )
    }

    @Test func repeatedUpdatesReconstructTheCompleteSequence() {
        MLXRandom.seed(99)
        let firstKeys = (MLXRandom.normal([1, 2, 2, 64]) / 4).asType(.float32)
        let firstValues = (MLXRandom.normal([1, 2, 2, 64]) / 4).asType(.float32)
        let nextKeys = (MLXRandom.normal([1, 2, 1, 64]) / 4).asType(.float32)
        let nextValues = (MLXRandom.normal([1, 2, 1, 64]) / 4).asType(.float32)
        let cache = WalshHadamardQuantizedKVCache(groupSize: 64, bits: 8)

        _ = cache.update(keys: firstKeys, values: firstValues)
        let reconstructed = cache.update(keys: nextKeys, values: nextValues)
        let expectedKeys = concatenated([firstKeys, nextKeys], axis: 2)
        let expectedValues = concatenated([firstValues, nextValues], axis: 2)

        #expect(cache.offset == 3)
        #expect(
            allClose(expectedKeys, reconstructed.0, rtol: 0.03, atol: 0.01)
                .item(Bool.self)
        )
        #expect(
            allClose(expectedValues, reconstructed.1, rtol: 0.03, atol: 0.01)
                .item(Bool.self)
        )
    }

    @Test func whtSchemeIsExplicitOptInAndAffineBehaviorIsUnchanged() throws {
        let keys = MLXArray.ones([1, 1, 2, 64], dtype: .float32)
        let values = MLXArray.ones([1, 1, 2, 64], dtype: .float32) * 2

        var defaultCache: [KVCache] = [KVCacheSimple()]
        _ = defaultCache[0].update(keys: keys, values: values)
        maybeQuantizeKVCache(
            cache: &defaultCache,
            kvBits: nil,
            quantizedKVStart: 0
        )
        #expect(defaultCache[0] is KVCacheSimple)

        var affineCache: [KVCache] = [KVCacheSimple()]
        _ = affineCache[0].update(keys: keys, values: values)
        maybeQuantizeKVCache(
            cache: &affineCache,
            kvBits: nil,
            quantizedKVStart: 0,
            kvScheme: "affine4"
        )
        #expect(affineCache[0] is QuantizedKVCache)

        var whtCache: [KVCache] = [KVCacheSimple()]
        _ = whtCache[0].update(keys: keys, values: values)
        maybeQuantizeKVCache(
            cache: &whtCache,
            kvBits: nil,
            quantizedKVStart: 0,
            kvScheme: "wht4"
        )
        let transformed = try #require(whtCache[0] as? WalshHadamardQuantizedKVCache)
        #expect(transformed.bits == 4)
        #expect(transformed.groupSize == 64)

        var deferredCache: [KVCache] = [KVCacheSimple()]
        _ = deferredCache[0].update(keys: keys, values: values)
        maybeQuantizeKVCache(
            cache: &deferredCache,
            kvBits: nil,
            quantizedKVStart: 2,
            kvScheme: "wht8"
        )
        #expect(deferredCache[0] is KVCacheSimple)
    }

    @Test func unsupportedDimensionsFailRecoverablyAndDynamicPolicyFallsBack() throws {
        let keys = MLXArray.ones([1, 1, 1, 288], dtype: .float32)
        let values = MLXArray.ones([1, 1, 1, 288], dtype: .float32)
        let simple = KVCacheSimple()
        _ = simple.update(keys: keys, values: values)

        #expect(throws: KVCacheError.self) {
            try simple.walshHadamardQuantized(groupSize: 32, bits: 4)
        }

        var cache: [KVCache] = [simple]
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: nil,
            quantizedKVStart: 0,
            kvScheme: "wht4"
        )
        #expect(cache[0] is KVCacheSimple)
        #expect(cache[0].offset == simple.offset)
    }

    @Test func promptCacheRoundTripPreservesWHTIdentityAndContinuation() throws {
        MLXRandom.seed(7)
        let firstKeys = MLXRandom.normal([1, 1, 2, 64]).asType(.float32)
        let firstValues = MLXRandom.normal([1, 1, 2, 64]).asType(.float32)
        let nextKeys = MLXRandom.normal([1, 1, 1, 64]).asType(.float32)
        let nextValues = MLXRandom.normal([1, 1, 1, 64]).asType(.float32)

        let cache = WalshHadamardQuantizedKVCache(groupSize: 64, bits: 8)
        _ = cache.update(keys: firstKeys, values: firstValues)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        try savePromptCache(url: url, cache: [cache])
        let (loaded, _) = try loadPromptCache(url: url)
        let restored = try #require(loaded.first as? WalshHadamardQuantizedKVCache)

        #expect(restored.metaState == cache.metaState)
        #expect(restored.offset == cache.offset)

        let originalContinuation = cache.update(keys: nextKeys, values: nextValues)
        let restoredContinuation = restored.update(keys: nextKeys, values: nextValues)
        #expect(
            allClose(
                originalContinuation.0,
                restoredContinuation.0,
                rtol: 0,
                atol: 0
            ).item(Bool.self)
        )
        #expect(
            allClose(
                originalContinuation.1,
                restoredContinuation.1,
                rtol: 0,
                atol: 0
            ).item(Bool.self)
        )
    }
}
