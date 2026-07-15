// Copyright © 2026 Apple Inc.

import Foundation
import MLX

private func defaultSpeculativeDecodingMemoryLimit() -> Int? {
    guard let bytes = GPU.maxRecommendedWorkingSetBytes(), bytes > 0 else {
        return nil
    }
    return bytes
}

/// Configuration errors for ``AdaptiveSpeculativeDecodingPolicy``.
public enum AdaptiveSpeculativeDecodingPolicyError: Error, Sendable, Equatable {
    /// `warmUpRounds` must be at least one.
    case invalidWarmUpRounds

    /// `observationWindowRounds` must be at least one.
    case invalidObservationWindowRounds

    /// `minimumObservedDraftTokens` must be at least one.
    case invalidMinimumObservedDraftTokens

    /// `minimumAcceptanceRate` must be finite and between zero and one, inclusive.
    case invalidMinimumAcceptanceRate
}

/// Opt-in policy for abandoning speculative decoding when recent measured
/// acceptance is persistently low.
///
/// The policy does not prescribe a universal target/draft threshold. Callers
/// must provide one based on their own benchmark data. A decision is made only
/// after all three evidence gates are satisfied: the warm-up has completed,
/// the rolling window is full, and the window contains enough drafted tokens.
public struct AdaptiveSpeculativeDecodingPolicy: Sendable, Hashable {
    /// Minimum total speculative rounds before the first decision.
    public let warmUpRounds: Int

    /// Number of most-recent speculative rounds used for each decision.
    public let observationWindowRounds: Int

    /// Minimum number of drafted tokens required inside the rolling window.
    public let minimumObservedDraftTokens: Int

    /// Acceptance-rate floor. Falling strictly below it triggers autoregressive mode.
    public let minimumAcceptanceRate: Double

    /// Creates an explicit adaptive policy.
    ///
    /// There is intentionally no framework default: useful thresholds depend
    /// on the target/draft pair, prompt distribution, and device measurements.
    public init(
        warmUpRounds: Int,
        observationWindowRounds: Int,
        minimumObservedDraftTokens: Int,
        minimumAcceptanceRate: Double
    ) throws {
        guard warmUpRounds > 0 else {
            throw AdaptiveSpeculativeDecodingPolicyError.invalidWarmUpRounds
        }
        guard observationWindowRounds > 0 else {
            throw AdaptiveSpeculativeDecodingPolicyError.invalidObservationWindowRounds
        }
        guard minimumObservedDraftTokens > 0 else {
            throw AdaptiveSpeculativeDecodingPolicyError.invalidMinimumObservedDraftTokens
        }
        guard minimumAcceptanceRate.isFinite,
            (0 ... 1).contains(minimumAcceptanceRate)
        else {
            throw AdaptiveSpeculativeDecodingPolicyError.invalidMinimumAcceptanceRate
        }

        self.warmUpRounds = warmUpRounds
        self.observationWindowRounds = observationWindowRounds
        self.minimumObservedDraftTokens = minimumObservedDraftTokens
        self.minimumAcceptanceRate = minimumAcceptanceRate
    }
}

/// Current state of an opt-in adaptive speculative-decoding policy.
public enum AdaptiveSpeculativeDecodingState: Sendable, Hashable {
    /// Waiting for the configured number of total speculative rounds.
    case warmingUp

    /// Warm-up completed, but the rolling observation window is not full yet.
    case collectingWindow

    /// The window is full but contains too few drafted-token observations.
    case insufficientDraftTokens

    /// The latest complete window met the acceptance-rate floor.
    case monitoring

    /// Recent acceptance fell below the configured floor; generation is now autoregressive.
    case autoregressive
}

/// Why adaptive speculative decoding changed modes.
public enum AdaptiveSpeculativeDecodingFallbackReason: Sendable, Hashable {
    /// Recent measured acceptance fell strictly below the configured minimum.
    case acceptanceRateBelowMinimum
}

/// Explainable final state for an opt-in adaptive speculative-decoding pass.
public struct AdaptiveSpeculativeDecodingTelemetry: Sendable, Equatable {
    /// Policy used for this pass.
    public let policy: AdaptiveSpeculativeDecodingPolicy

    /// Current policy state.
    public private(set) var state: AdaptiveSpeculativeDecodingState

    /// Number of full, sufficiently sampled windows evaluated against the threshold.
    public private(set) var evaluatedWindowCount: Int

    /// Target-model calls made after switching to autoregressive mode.
    public private(set) var autoregressiveTargetModelCallCount: Int

    /// Number of rounds represented by the latest rolling observation window.
    public private(set) var observedRoundCount: Int

    /// Drafted tokens represented by the latest rolling observation window.
    public private(set) var observedDraftTokenCount: Int

    /// Accepted draft tokens represented by the latest rolling observation window.
    public private(set) var observedAcceptedDraftTokenCount: Int

    /// Total speculative round after which autoregressive mode was selected.
    public private(set) var fallbackAfterRoundCount: Int?

    /// Structured reason for switching modes.
    public private(set) var fallbackReason: AdaptiveSpeculativeDecodingFallbackReason?

    public init(policy: AdaptiveSpeculativeDecodingPolicy) {
        self.policy = policy
        self.state = .warmingUp
        self.evaluatedWindowCount = 0
        self.autoregressiveTargetModelCallCount = 0
        self.observedRoundCount = 0
        self.observedDraftTokenCount = 0
        self.observedAcceptedDraftTokenCount = 0
        self.fallbackAfterRoundCount = nil
        self.fallbackReason = nil
    }

    /// Acceptance rate for the latest rolling observation window.
    public var observedAcceptanceRate: Double {
        guard observedDraftTokenCount > 0 else { return 0 }
        return Double(observedAcceptedDraftTokenCount) / Double(observedDraftTokenCount)
    }

    /// Whether the iterator permanently switched to autoregressive generation.
    public var didFallbackToAutoregressive: Bool {
        state == .autoregressive
    }

    fileprivate mutating func recordObservation(
        state: AdaptiveSpeculativeDecodingState,
        observedRoundCount: Int,
        observedDraftTokenCount: Int,
        observedAcceptedDraftTokenCount: Int,
        evaluatedWindow: Bool
    ) {
        self.state = state
        self.observedRoundCount = observedRoundCount
        self.observedDraftTokenCount = observedDraftTokenCount
        self.observedAcceptedDraftTokenCount = observedAcceptedDraftTokenCount
        if evaluatedWindow {
            evaluatedWindowCount += 1
        }
    }

    fileprivate mutating func recordFallback(afterRoundCount roundCount: Int) {
        state = .autoregressive
        fallbackAfterRoundCount = roundCount
        fallbackReason = .acceptanceRateBelowMinimum
    }

    fileprivate mutating func recordAutoregressiveTargetCall() {
        autoregressiveTargetModelCallCount += 1
    }
}

private struct SpeculativeDecodingRoundObservation: Sendable, Equatable {
    let draftedTokenCount: Int
    let acceptedDraftTokenCount: Int
}

/// Rolling policy evaluator kept separate from model execution so the
/// evidence gates can be tested deterministically without hardware.
package struct AdaptiveSpeculativeDecodingController: Sendable {
    package private(set) var telemetry: AdaptiveSpeculativeDecodingTelemetry

    private var totalRoundCount = 0
    private var recentRounds: [SpeculativeDecodingRoundObservation] = []
    private var nextReplacementIndex = 0
    private var recentDraftTokenCount = 0
    private var recentAcceptedDraftTokenCount = 0

    package init(policy: AdaptiveSpeculativeDecodingPolicy) {
        self.telemetry = AdaptiveSpeculativeDecodingTelemetry(policy: policy)
    }

    /// Records a completed speculative round and returns `true` exactly once
    /// when the iterator should permanently switch to autoregressive mode.
    package mutating func recordRound(drafted: Int, accepted: Int) -> Bool {
        guard !telemetry.didFallbackToAutoregressive else { return false }

        totalRoundCount += 1
        let drafted = max(0, drafted)
        let observation = SpeculativeDecodingRoundObservation(
            draftedTokenCount: drafted,
            acceptedDraftTokenCount: min(max(0, accepted), drafted)
        )

        if recentRounds.count < telemetry.policy.observationWindowRounds {
            recentRounds.append(observation)
        } else {
            let replaced = recentRounds[nextReplacementIndex]
            recentDraftTokenCount -= replaced.draftedTokenCount
            recentAcceptedDraftTokenCount -= replaced.acceptedDraftTokenCount
            recentRounds[nextReplacementIndex] = observation
            nextReplacementIndex =
                (nextReplacementIndex + 1) % telemetry.policy.observationWindowRounds
        }
        recentDraftTokenCount += observation.draftedTokenCount
        recentAcceptedDraftTokenCount += observation.acceptedDraftTokenCount

        let state: AdaptiveSpeculativeDecodingState
        let evaluatedWindow: Bool
        if totalRoundCount < telemetry.policy.warmUpRounds {
            state = .warmingUp
            evaluatedWindow = false
        } else if recentRounds.count < telemetry.policy.observationWindowRounds {
            state = .collectingWindow
            evaluatedWindow = false
        } else if recentDraftTokenCount < telemetry.policy.minimumObservedDraftTokens {
            state = .insufficientDraftTokens
            evaluatedWindow = false
        } else {
            state = .monitoring
            evaluatedWindow = true
        }

        telemetry.recordObservation(
            state: state,
            observedRoundCount: recentRounds.count,
            observedDraftTokenCount: recentDraftTokenCount,
            observedAcceptedDraftTokenCount: recentAcceptedDraftTokenCount,
            evaluatedWindow: evaluatedWindow
        )

        guard evaluatedWindow,
            telemetry.observedAcceptanceRate < telemetry.policy.minimumAcceptanceRate
        else {
            return false
        }

        telemetry.recordFallback(afterRoundCount: totalRoundCount)
        return true
    }

    package mutating func recordAutoregressiveTargetCall() {
        guard telemetry.didFallbackToAutoregressive else { return }
        telemetry.recordAutoregressiveTargetCall()
    }
}

/// Runtime counters for a speculative decoding pass.
public struct SpeculativeDecodingTelemetry: Sendable, Equatable {
    /// Number of speculative decoding rounds.
    public private(set) var roundCount: Int

    /// Number of tokens proposed by the draft model.
    public private(set) var draftTokenCount: Int

    /// Number of draft tokens accepted by the target model.
    public private(set) var acceptedDraftTokenCount: Int

    /// Number of target-model calls, including adaptive autoregressive calls.
    public private(set) var targetModelCallCount: Int

    /// Number of draft-model calls.
    public private(set) var draftModelCallCount: Int

    /// Number of token positions evaluated by the target model.
    public private(set) var targetVerifiedTokenCount: Int

    /// Number of tokens emitted from speculative rounds, including correction and bonus tokens.
    public private(set) var emittedTokenCount: Int

    /// Runtime adaptation state, or `nil` when no adaptive policy was requested.
    public private(set) var adaptive: AdaptiveSpeculativeDecodingTelemetry?

    public init(
        roundCount: Int = 0,
        draftTokenCount: Int = 0,
        acceptedDraftTokenCount: Int = 0,
        targetModelCallCount: Int = 0,
        draftModelCallCount: Int = 0,
        targetVerifiedTokenCount: Int = 0,
        emittedTokenCount: Int = 0
    ) {
        self.roundCount = roundCount
        self.draftTokenCount = draftTokenCount
        self.acceptedDraftTokenCount = acceptedDraftTokenCount
        self.targetModelCallCount = targetModelCallCount
        self.draftModelCallCount = draftModelCallCount
        self.targetVerifiedTokenCount = targetVerifiedTokenCount
        self.emittedTokenCount = emittedTokenCount
        self.adaptive = nil
    }

    /// Creates telemetry with an explicit initial adaptive-policy state.
    public init(
        roundCount: Int = 0,
        draftTokenCount: Int = 0,
        acceptedDraftTokenCount: Int = 0,
        targetModelCallCount: Int = 0,
        draftModelCallCount: Int = 0,
        targetVerifiedTokenCount: Int = 0,
        emittedTokenCount: Int = 0,
        adaptive: AdaptiveSpeculativeDecodingTelemetry
    ) {
        self.roundCount = roundCount
        self.draftTokenCount = draftTokenCount
        self.acceptedDraftTokenCount = acceptedDraftTokenCount
        self.targetModelCallCount = targetModelCallCount
        self.draftModelCallCount = draftModelCallCount
        self.targetVerifiedTokenCount = targetVerifiedTokenCount
        self.emittedTokenCount = emittedTokenCount
        self.adaptive = adaptive
    }

    /// Number of draft tokens rejected by the target model.
    public var rejectedDraftTokenCount: Int {
        max(0, draftTokenCount - acceptedDraftTokenCount)
    }

    /// Fraction of drafted tokens accepted by the target model.
    public var acceptanceRate: Double {
        guard draftTokenCount > 0 else { return 0 }
        return Double(acceptedDraftTokenCount) / Double(draftTokenCount)
    }

    /// Mean accepted draft tokens per speculative round.
    public var meanAcceptedDraftTokensPerRound: Double {
        guard roundCount > 0 else { return 0 }
        return Double(acceptedDraftTokenCount) / Double(roundCount)
    }

    /// Mean emitted tokens per target-model call.
    public var meanEmittedTokensPerTargetCall: Double {
        guard targetModelCallCount > 0 else { return 0 }
        return Double(emittedTokenCount) / Double(targetModelCallCount)
    }

    package mutating func recordRound(
        drafted: Int,
        accepted: Int,
        targetVerified: Int,
        draftModelCalls: Int? = nil
    ) {
        roundCount += 1
        draftTokenCount += drafted
        acceptedDraftTokenCount += accepted
        targetModelCallCount += 1
        draftModelCallCount += draftModelCalls ?? drafted
        targetVerifiedTokenCount += targetVerified
    }

    package mutating func recordAutoregressiveTargetCall() {
        targetModelCallCount += 1
        targetVerifiedTokenCount += 1
    }

    package mutating func recordGeneratedToken() {
        emittedTokenCount += 1
    }

    package mutating func discardGeneratedToken() {
        emittedTokenCount = max(0, emittedTokenCount - 1)
    }

    package mutating func updateAdaptive(
        _ adaptive: AdaptiveSpeculativeDecodingTelemetry
    ) {
        self.adaptive = adaptive
    }
}

/// Action to take when speculative decoding exceeds a memory budget.
public enum SpeculativeDecodingMemoryAction: Sendable, Hashable {
    /// Use speculative decoding even if the estimate exceeds the budget.
    case allow

    /// Fall back to regular generation.
    case fallbackToDefault

    /// Throw an error instead of silently falling back.
    case fail
}

/// Result of evaluating speculative decoding against a memory policy.
public struct SpeculativeDecodingMemoryEvaluation: Sendable, Equatable {
    /// Estimated main-model parameter bytes.
    public let mainModelBytes: Int

    /// Estimated draft-model parameter bytes.
    public let draftModelBytes: Int

    /// Additional caller-provided budget for KV cache, workspace, or other resident data.
    public let additionalBytes: Int

    /// Total estimated resident bytes for speculative decoding.
    public var estimatedBytes: Int {
        mainModelBytes + draftModelBytes + additionalBytes
    }

    /// Memory budget used for the decision. `nil` means no budget was available.
    public let limitBytes: Int?

    /// Action selected by the policy.
    public let action: SpeculativeDecodingMemoryAction

    /// Whether speculative decoding is within the available budget.
    public var isWithinBudget: Bool {
        guard let limitBytes else { return true }
        return estimatedBytes <= limitBytes
    }

    /// Whether speculative decoding should be used.
    public var shouldUseSpeculativeDecoding: Bool {
        isWithinBudget || action == .allow
    }
}

/// Policy for gating auxiliary-model speculative decoding by resident memory estimates.
public struct SpeculativeDecodingMemoryPolicy: Sendable, Hashable {
    /// Optional absolute budget in bytes. When nil, no budget is enforced.
    public let limitBytes: Int?

    /// Extra bytes to reserve for KV cache, workspace, or application memory.
    public let additionalBytes: Int

    /// Action to take when the estimate exceeds the budget.
    public let action: SpeculativeDecodingMemoryAction

    public init(
        limitBytes: Int? = nil,
        additionalBytes: Int = 0,
        action: SpeculativeDecodingMemoryAction = .fallbackToDefault
    ) {
        self.limitBytes = limitBytes
        self.additionalBytes = max(0, additionalBytes)
        self.action = action
    }

    /// Default policy using `GPU.maxRecommendedWorkingSetBytes()` when available.
    public static var recommendedWorkingSet: Self {
        Self(limitBytes: defaultSpeculativeDecodingMemoryLimit())
    }

    /// Evaluate explicit byte estimates. This is useful before loading a draft model.
    public func evaluate(
        mainModelBytes: Int,
        draftModelBytes: Int
    ) -> SpeculativeDecodingMemoryEvaluation {
        SpeculativeDecodingMemoryEvaluation(
            mainModelBytes: max(0, mainModelBytes),
            draftModelBytes: max(0, draftModelBytes),
            additionalBytes: additionalBytes,
            limitBytes: limitBytes,
            action: action
        )
    }

    package func evaluate(
        mainModel: any LanguageModel,
        draftModel: any LanguageModel
    ) -> SpeculativeDecodingMemoryEvaluation {
        evaluate(
            mainModelBytes: Self.modelWeightBytes(mainModel),
            draftModelBytes: Self.modelWeightBytes(draftModel)
        )
    }

    package static func modelWeightBytes(_ model: any LanguageModel) -> Int {
        model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
    }
}

public struct SpeculativeDecodingMemoryError: Error, Sendable {
    public let evaluation: SpeculativeDecodingMemoryEvaluation

    public init(evaluation: SpeculativeDecodingMemoryEvaluation) {
        self.evaluation = evaluation
    }
}
