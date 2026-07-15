// Copyright © 2026 Apple Inc.

import Foundation

/// Errors produced while estimating KV-cache memory.
public enum KVCacheMemoryEstimationError: Error, Equatable, Sendable {
    /// A dimension or count was negative.
    case negativeValue(parameter: String, value: Int)
    /// The byte width of the unquantized element type must be positive.
    case invalidBytesPerElement(Int)
    /// Affine quantization only supports the bit widths implemented by MLX.
    case unsupportedQuantizationBits(Int)
    /// No supported MLX group size divides the supplied head dimension.
    case incompatibleQuantizationGroupSize(requested: Int, headDim: Int)
    /// The result cannot be represented by `Int` on the current platform.
    case arithmeticOverflow
}

extension KVCacheMemoryEstimationError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .negativeValue(let parameter, let value):
            "KV cache estimation requires a nonnegative \(parameter); received \(value)."
        case .invalidBytesPerElement(let value):
            "KV cache estimation requires bytesPerElement greater than zero; received \(value)."
        case .unsupportedQuantizationBits(let bits):
            "KV cache estimation supports affine quantization with 1, 2, 3, 4, 5, 6, or 8 bits; received \(bits)."
        case .incompatibleQuantizationGroupSize(let requested, let headDim):
            "KV cache head dimension \(headDim) is not divisible by a supported affine group size (32, 64, or 128) near the requested size \(requested)."
        case .arithmeticOverflow:
            "The estimated KV cache size exceeds the largest byte count representable by Int."
        }
    }
}

/// Estimate the logical storage required by a dense key-value cache.
///
/// The estimate includes every layer, K and V, and affine quantization metadata
/// (one scale and one bias per group). It intentionally excludes allocator
/// alignment, the cache's growth-step spare capacity, and temporary attention
/// workspace. Use ``WiredMemoryUtils/tune(context:tokenCount:parameters:seedText:resetPeakMemory:)``
/// when a runtime measurement is available.
///
/// When `kvBits` is non-nil, the effective group size mirrors
/// `QuantizedKVCache`: the nearest of 32, 64, or 128 that divides `headDim`.
/// Ties prefer the smaller group. `bytesPerElement` describes both the
/// unquantized K/V values and the affine scale/bias dtype; its default models
/// BF16 and FP16 caches.
///
/// - Parameters:
///   - numLayers: Number of attention layers that own a KV cache.
///   - kvHeads: Number of KV heads per layer.
///   - headDim: Elements in each key/value head.
///   - maxTokens: Maximum number of cached tokens.
///   - kvBits: Optional affine quantization bit width.
///   - kvGroupSize: Preferred affine quantization group size.
///   - bytesPerElement: Byte width of unquantized values and quantization metadata.
/// - Returns: The logical cache size in bytes.
/// - Throws: ``KVCacheMemoryEstimationError`` for invalid input or arithmetic overflow.
public func estimateKVCacheBytes(
    numLayers: Int,
    kvHeads: Int,
    headDim: Int,
    maxTokens: Int,
    kvBits: Int? = nil,
    kvGroupSize: Int = 64,
    bytesPerElement: Int = 2
) throws -> Int {
    for (parameter, value) in [
        ("numLayers", numLayers),
        ("kvHeads", kvHeads),
        ("headDim", headDim),
        ("maxTokens", maxTokens),
    ] where value < 0 {
        throw KVCacheMemoryEstimationError.negativeValue(parameter: parameter, value: value)
    }
    guard bytesPerElement > 0 else {
        throw KVCacheMemoryEstimationError.invalidBytesPerElement(bytesPerElement)
    }

    let scalarCount = try checkedProduct([2, numLayers, kvHeads, headDim, maxTokens])
    guard let kvBits else {
        return try checkedProduct([scalarCount, bytesPerElement])
    }

    let supportedBits = [1, 2, 3, 4, 5, 6, 8]
    guard supportedBits.contains(kvBits) else {
        throw KVCacheMemoryEstimationError.unsupportedQuantizationBits(kvBits)
    }
    guard kvGroupSize > 0,
        let effectiveGroupSize = resolvedEstimationGroupSize(
            requested: kvGroupSize,
            headDim: headDim
        )
    else {
        throw KVCacheMemoryEstimationError.incompatibleQuantizationGroupSize(
            requested: kvGroupSize,
            headDim: headDim
        )
    }

    let payloadBits = try checkedProduct([scalarCount, kvBits])
    let payloadBytes = payloadBits / 8 + (payloadBits.isMultiple(of: 8) ? 0 : 1)

    let groupCount = scalarCount / effectiveGroupSize
    let metadataBytes = try checkedProduct([groupCount, 2, bytesPerElement])
    return try checkedSum(payloadBytes, metadataBytes)
}

private func resolvedEstimationGroupSize(requested: Int, headDim: Int) -> Int? {
    let compatible = [32, 64, 128].filter { headDim.isMultiple(of: $0) }
    guard !compatible.isEmpty else { return nil }
    return compatible.min { lhs, rhs in
        let lhsDistance = requested >= lhs ? requested - lhs : lhs - requested
        let rhsDistance = requested >= rhs ? requested - rhs : rhs - requested
        if lhsDistance == rhsDistance {
            return lhs < rhs
        }
        return lhsDistance < rhsDistance
    }
}

private func checkedProduct(_ values: [Int]) throws -> Int {
    guard !values.contains(0) else { return 0 }
    return try values.reduce(1) { partialResult, value in
        let (result, overflow) = partialResult.multipliedReportingOverflow(by: value)
        guard !overflow else {
            throw KVCacheMemoryEstimationError.arithmeticOverflow
        }
        return result
    }
}

private func checkedSum(_ lhs: Int, _ rhs: Int) throws -> Int {
    let (result, overflow) = lhs.addingReportingOverflow(rhs)
    guard !overflow else {
        throw KVCacheMemoryEstimationError.arithmeticOverflow
    }
    return result
}
