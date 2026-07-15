// Copyright © 2026 Apple Inc.

import MLX

/// Experimental affine KV cache with an orthonormal Walsh-Hadamard pre-transform.
///
/// This cache intentionally does not conform to ``QuantizedKVCacheProtocol``.
/// The current quantized attention path consumes cached keys and values directly,
/// so doing so would rotate the attention basis without compensating the queries
/// and output. Instead, this prototype dequantizes and applies the inverse WHT
/// before returning arrays to the regular attention path:
///
/// ```
/// stored K = quantize(H(K))      returned K = H(dequantize(stored K))
/// stored V = quantize(H(V))      returned V = H(dequantize(stored V))
/// ```
///
/// MLX's default Hadamard scaling is orthonormal, therefore `H⁻¹ = H`.
/// This preserves attention semantics modulo affine quantization error, at the
/// cost of dequantizing and transforming the complete cache on each update.
final class WalshHadamardQuantizedKVCache: BaseKVCache {
    static let formatVersion = "wht-affine-v1"

    private var storage: QuantizedKVCache

    var groupSize: Int { storage.groupSize }
    var bits: Int { storage.bits }

    init(groupSize: Int = 64, bits: Int = 4) {
        self.storage = QuantizedKVCache(groupSize: groupSize, bits: bits)
        super.init()
    }

    override func innerState() -> [MLXArray] {
        storage.innerState()
    }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        precondition(
            isWalshHadamardKVShapeSupported(keys: keys, values: values),
            "Walsh-Hadamard KV cache received unsupported head dimensions or dtypes"
        )
        precondition(
            keys.dim(3).isMultiple(of: groupSize)
                && values.dim(3).isMultiple(of: groupSize),
            "Walsh-Hadamard KV cache head dimensions must be divisible by groupSize"
        )

        let transformedKeys = hadamardTransform(keys)
        let transformedValues = hadamardTransform(values)
        _ = storage.updateQuantized(keys: transformedKeys, values: transformedValues)
        offset = storage.offset

        let transformedState = storage.toUnquantized().state
        precondition(
            transformedState.count == 2,
            "Walsh-Hadamard KV cache failed to reconstruct its quantized state"
        )

        return (
            hadamardTransform(transformedState[0]),
            hadamardTransform(transformedState[1])
        )
    }

    override var state: [MLXArray] {
        get { storage.state }
        set { storage.state = newValue }
    }

    override var metaState: [String] {
        get { [Self.formatVersion] + storage.metaState }
        set {
            precondition(
                newValue.count == 5 && newValue[0] == Self.formatVersion,
                "WalshHadamardQuantizedKVCache metaState must contain its format and four quantization values"
            )
            storage.metaState = Array(newValue.dropFirst())
            offset = storage.offset
        }
    }

    override var isTrimmable: Bool { true }

    @discardableResult
    override func trim(_ n: Int) -> Int {
        let trimmed = storage.trim(n)
        offset = storage.offset
        return trimmed
    }

    override func copy() -> any KVCache {
        let new = WalshHadamardQuantizedKVCache(groupSize: groupSize, bits: bits)
        let copiedState = state
        if !copiedState.isEmpty {
            new.state = copiedState.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }
}

extension KVCacheSimple {
    /// Convert a populated cache to the experimental WHT + affine representation.
    ///
    /// The conversion is recoverably rejected when either head dimension cannot
    /// use MLX's native WHT or affine quantization group sizes.
    func walshHadamardQuantized(
        groupSize: Int = 64,
        bits: Int = 4
    ) throws -> WalshHadamardQuantizedKVCache {
        let currentState = state
        guard currentState.count == 2 else {
            throw KVCacheError(
                message: "Walsh-Hadamard KV cache conversion requires a populated KVCacheSimple"
            )
        }

        let keys = currentState[0]
        let values = currentState[1]
        guard isWalshHadamardKVShapeSupported(keys: keys, values: values) else {
            throw KVCacheError(
                message:
                    "Walsh-Hadamard KV cache does not support key/value head dimensions \(keys.dim(3))/\(values.dim(3)) with dtypes \(keys.dtype)/\(values.dtype)"
            )
        }
        guard
            let effectiveGroupSize = resolvedWalshHadamardGroupSize(
                requested: groupSize,
                keyHeadDim: keys.dim(3),
                valueHeadDim: values.dim(3)
            )
        else {
            throw KVCacheError(
                message:
                    "Walsh-Hadamard KV cache requires key/value head dimensions divisible by 32, 64, or 128"
            )
        }

        let cache = WalshHadamardQuantizedKVCache(
            groupSize: effectiveGroupSize,
            bits: bits
        )
        _ = cache.update(keys: keys, values: values)
        return cache
    }
}

func resolveWalshHadamardScheme(_ scheme: String?) -> (bits: Int, groupSize: Int)? {
    switch scheme {
    case "wht4": return (4, 64)
    case "wht8": return (8, 64)
    default: return nil
    }
}

func isWalshHadamardDimensionSupported(_ dimension: Int, dtype: DType) -> Bool {
    let maximumPowerOfTwo: Int
    switch dtype {
    case .float32:
        maximumPowerOfTwo = 8192
    case .float16, .bfloat16:
        maximumPowerOfTwo = 16_384
    default:
        return false
    }

    guard dimension > 0 else { return false }
    // MLX accepts additional composite dimensions, but its public operation
    // does not expose the transpose/inverse for those base matrices. Applying
    // the operation twice is only guaranteed by this prototype for the
    // conventional power-of-two Walsh-Hadamard matrix.
    return (dimension & (dimension - 1)) == 0
        && dimension <= maximumPowerOfTwo
}

private func isWalshHadamardKVShapeSupported(
    keys: MLXArray,
    values: MLXArray
) -> Bool {
    keys.ndim == 4
        && values.ndim == 4
        && isWalshHadamardDimensionSupported(keys.dim(3), dtype: keys.dtype)
        && isWalshHadamardDimensionSupported(values.dim(3), dtype: values.dtype)
}

private func resolvedWalshHadamardGroupSize(
    requested: Int,
    keyHeadDim: Int,
    valueHeadDim: Int
) -> Int? {
    let requested = max(1, requested)
    let compatible = [32, 64, 128].filter {
        keyHeadDim.isMultiple(of: $0) && valueHeadDim.isMultiple(of: $0)
    }
    guard !compatible.isEmpty else { return nil }
    return compatible.min { lhs, rhs in
        let lhsDistance = abs(lhs - requested)
        let rhsDistance = abs(rhs - requested)
        return lhsDistance == rhsDistance ? lhs < rhs : lhsDistance < rhsDistance
    }
}
