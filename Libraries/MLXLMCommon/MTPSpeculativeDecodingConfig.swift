// Copyright © 2026 Apple Inc.

import Foundation

/// Errors produced while configuring high-level MTP speculative decoding.
public enum MTPSpeculativeDecodingConfigError: Error, Sendable, Equatable {
    /// An MTP round must contain one verifier bonus token and at least one drafted token.
    case invalidBlockSize(Int)
}

/// High-level MTP speculative-decoding configuration for ``ChatSession``.
///
/// MTP uses an auxiliary drafter that consumes hidden state and shared K/V
/// emitted by the target model. Unlike classic speculative decoding, the MTP
/// drafter owns no per-session KV cache.
///
/// Prefer the deferred initializer when the drafter may not fit beside the
/// target model. When a memory policy is present, `ChatSession` evaluates the
/// supplied byte estimate before invoking the loader.
public struct MTPSpeculativeDecodingConfig: Sendable {
    package enum DrafterSource: Sendable {
        case loaded(MTPDrafterContainer)
        case deferred(bytes: Int, @Sendable () async throws -> MTPDrafterContainer)
    }

    package let drafterSource: DrafterSource

    /// The eagerly supplied drafter container, or `nil` for deferred configurations.
    public var drafter: MTPDrafterContainer? {
        guard case .loaded(let drafter) = drafterSource else { return nil }
        return drafter
    }

    /// Total tokens in one MTP verification round.
    ///
    /// The drafter proposes `blockSize - 1` tokens; the remaining position is
    /// the verifier bonus/correction token.
    public let blockSize: Int

    /// Optional resident-memory policy applied to target + drafter weights.
    ///
    /// `additionalBytes` on the policy can account for KV cache, workspace,
    /// and application headroom without the framework inventing a threshold.
    public let memoryPolicy: SpeculativeDecodingMemoryPolicy?

    /// Creates a configuration with an already loaded MTP drafter.
    public init(
        drafter: MTPDrafterContainer,
        blockSize: Int = 4,
        memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil
    ) throws {
        guard blockSize >= 2 else {
            throw MTPSpeculativeDecodingConfigError.invalidBlockSize(blockSize)
        }
        self.drafterSource = .loaded(drafter)
        self.blockSize = blockSize
        self.memoryPolicy = memoryPolicy
    }

    /// Creates a configuration whose drafter is loaded only after memory admission.
    ///
    /// - Parameters:
    ///   - drafterBytes: Estimated resident bytes for the drafter weights.
    ///     Negative values are normalized to zero, matching ``SpeculativeDecodingConfig``.
    ///   - blockSize: Total tokens per MTP verification round; must be at least two.
    ///   - memoryPolicy: Optional policy evaluated before and after loading.
    ///   - loadDrafter: Deferred drafter loader. It is cached after the first admitted load.
    public init(
        drafterBytes: Int,
        blockSize: Int = 4,
        memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil,
        loadDrafter: @escaping @Sendable () async throws -> MTPDrafterContainer
    ) throws {
        guard blockSize >= 2 else {
            throw MTPSpeculativeDecodingConfigError.invalidBlockSize(blockSize)
        }
        self.drafterSource = .deferred(bytes: max(0, drafterBytes), loadDrafter)
        self.blockSize = blockSize
        self.memoryPolicy = memoryPolicy
    }

    package var estimatedDrafterBytes: Int? {
        guard case .deferred(let bytes, _) = drafterSource else { return nil }
        return bytes
    }

    package func loadDrafter() async throws -> MTPDrafterContainer {
        switch drafterSource {
        case .loaded(let drafter):
            drafter
        case .deferred(_, let load):
            try await load()
        }
    }
}
