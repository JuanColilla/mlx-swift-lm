// Copyright © 2026 Apple Inc.

import MLX
import MLXLMCommon
import MLXNN
import Testing

@Suite(.serialized)
struct AdaptiveSpeculativeDecodingTests {
    @Test func `Policy rejects invalid evidence gates and thresholds`() {
        expectPolicyError(.invalidWarmUpRounds) {
            _ = try AdaptiveSpeculativeDecodingPolicy(
                warmUpRounds: 0,
                observationWindowRounds: 2,
                minimumObservedDraftTokens: 4,
                minimumAcceptanceRate: 0.5
            )
        }
        expectPolicyError(.invalidObservationWindowRounds) {
            _ = try AdaptiveSpeculativeDecodingPolicy(
                warmUpRounds: 2,
                observationWindowRounds: 0,
                minimumObservedDraftTokens: 4,
                minimumAcceptanceRate: 0.5
            )
        }
        expectPolicyError(.invalidMinimumObservedDraftTokens) {
            _ = try AdaptiveSpeculativeDecodingPolicy(
                warmUpRounds: 2,
                observationWindowRounds: 2,
                minimumObservedDraftTokens: 0,
                minimumAcceptanceRate: 0.5
            )
        }
        for invalidRate in [-Double.infinity, -0.01, 1.01, Double.infinity, .nan] {
            expectPolicyError(.invalidMinimumAcceptanceRate) {
                _ = try AdaptiveSpeculativeDecodingPolicy(
                    warmUpRounds: 2,
                    observationWindowRounds: 2,
                    minimumObservedDraftTokens: 4,
                    minimumAcceptanceRate: invalidRate
                )
            }
        }
    }

    private func expectPolicyError(
        _ expected: AdaptiveSpeculativeDecodingPolicyError,
        performing operation: () throws -> Void
    ) {
        do {
            try operation()
            Issue.record("Expected policy error \(expected)")
        } catch let error as AdaptiveSpeculativeDecodingPolicyError {
            #expect(error == expected)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test func `Controller waits for warmup even when its window is already full`() throws {
        let policy = try AdaptiveSpeculativeDecodingPolicy(
            warmUpRounds: 4,
            observationWindowRounds: 2,
            minimumObservedDraftTokens: 4,
            minimumAcceptanceRate: 0.5
        )
        var controller = AdaptiveSpeculativeDecodingController(policy: policy)

        let firstRoundDidFallback = controller.recordRound(drafted: 2, accepted: 0)
        let secondRoundDidFallback = controller.recordRound(drafted: 2, accepted: 0)
        let thirdRoundDidFallback = controller.recordRound(drafted: 2, accepted: 0)
        #expect(!firstRoundDidFallback)
        #expect(!secondRoundDidFallback)
        #expect(!thirdRoundDidFallback)
        #expect(controller.telemetry.state == .warmingUp)
        #expect(controller.telemetry.evaluatedWindowCount == 0)

        let fourthRoundDidFallback = controller.recordRound(drafted: 2, accepted: 0)
        #expect(fourthRoundDidFallback)
        #expect(controller.telemetry.state == .autoregressive)
        #expect(controller.telemetry.fallbackAfterRoundCount == 4)
    }

    @Test func `Controller requires a full rolling window after warmup`() throws {
        let policy = try AdaptiveSpeculativeDecodingPolicy(
            warmUpRounds: 1,
            observationWindowRounds: 3,
            minimumObservedDraftTokens: 6,
            minimumAcceptanceRate: 0.5
        )
        var controller = AdaptiveSpeculativeDecodingController(policy: policy)

        let firstRoundDidFallback = controller.recordRound(drafted: 2, accepted: 0)
        #expect(!firstRoundDidFallback)
        #expect(controller.telemetry.state == .collectingWindow)
        let secondRoundDidFallback = controller.recordRound(drafted: 2, accepted: 0)
        #expect(!secondRoundDidFallback)
        #expect(controller.telemetry.state == .collectingWindow)
        #expect(controller.telemetry.evaluatedWindowCount == 0)

        let thirdRoundDidFallback = controller.recordRound(drafted: 2, accepted: 0)
        #expect(thirdRoundDidFallback)
        #expect(controller.telemetry.observedRoundCount == 3)
        #expect(controller.telemetry.observedDraftTokenCount == 6)
    }

    @Test func `Controller requires enough drafted-token samples inside the window`() throws {
        let policy = try AdaptiveSpeculativeDecodingPolicy(
            warmUpRounds: 1,
            observationWindowRounds: 3,
            minimumObservedDraftTokens: 6,
            minimumAcceptanceRate: 0.5
        )
        var controller = AdaptiveSpeculativeDecodingController(policy: policy)

        let firstRoundDidFallback = controller.recordRound(drafted: 1, accepted: 0)
        let secondRoundDidFallback = controller.recordRound(drafted: 1, accepted: 0)
        let thirdRoundDidFallback = controller.recordRound(drafted: 1, accepted: 0)
        #expect(!firstRoundDidFallback)
        #expect(!secondRoundDidFallback)
        #expect(!thirdRoundDidFallback)
        #expect(controller.telemetry.state == .insufficientDraftTokens)
        #expect(controller.telemetry.evaluatedWindowCount == 0)

        let fourthRoundDidFallback = controller.recordRound(drafted: 3, accepted: 3)
        #expect(!fourthRoundDidFallback)
        #expect(controller.telemetry.state == .insufficientDraftTokens)
        let fifthRoundDidFallback = controller.recordRound(drafted: 3, accepted: 3)
        #expect(!fifthRoundDidFallback)
        #expect(controller.telemetry.state == .monitoring)
        #expect(controller.telemetry.observedDraftTokenCount == 7)
        #expect(controller.telemetry.observedAcceptedDraftTokenCount == 6)
        #expect(controller.telemetry.evaluatedWindowCount == 1)
    }

    @Test func `Threshold is strict and uses only the recent rolling window`() throws {
        let policy = try AdaptiveSpeculativeDecodingPolicy(
            warmUpRounds: 1,
            observationWindowRounds: 2,
            minimumObservedDraftTokens: 4,
            minimumAcceptanceRate: 0.5
        )
        var controller = AdaptiveSpeculativeDecodingController(policy: policy)

        let firstRoundDidFallback = controller.recordRound(drafted: 2, accepted: 2)
        let secondRoundDidFallback = controller.recordRound(drafted: 2, accepted: 0)
        #expect(!firstRoundDidFallback)
        #expect(!secondRoundDidFallback)
        #expect(controller.telemetry.state == .monitoring)
        #expect(controller.telemetry.observedAcceptanceRate == 0.5)

        let thirdRoundDidFallback = controller.recordRound(drafted: 2, accepted: 0)
        #expect(thirdRoundDidFallback)
        #expect(controller.telemetry.state == .autoregressive)
        #expect(controller.telemetry.observedAcceptanceRate == 0)
        #expect(controller.telemetry.evaluatedWindowCount == 2)
        #expect(controller.telemetry.fallbackReason == .acceptanceRateBelowMinimum)
        #expect(controller.telemetry.fallbackAfterRoundCount == 3)

        // Fallback is sticky and the transition signal is emitted only once.
        let postFallbackDidFallback = controller.recordRound(drafted: 2, accepted: 2)
        #expect(!postFallbackDidFallback)
        #expect(controller.telemetry.fallbackAfterRoundCount == 3)
        #expect(controller.telemetry.evaluatedWindowCount == 2)
    }

    @Test(arguments: [false, true])
    func `Adaptive fallback remains token-equivalent to target-only generation`(
        withLogitProcessor: Bool
    ) async throws {
        let vocabularySize = 101
        let tokenizer = TestTokenizer(vocabularySize: vocabularySize)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "adaptive-transition-test"),
            messageGenerator: DefaultMessageGenerator()
        )
        let normalModel = AdaptiveTransitionLanguageModel(
            vocabularySize: vocabularySize, transitionOffset: 7)
        let adaptiveModel = AdaptiveTransitionLanguageModel(
            vocabularySize: vocabularySize, transitionOffset: 7)
        let mismatchedDraftModel = AdaptiveTransitionLanguageModel(
            vocabularySize: vocabularySize, transitionOffset: 19)
        let input = LMInput(tokens: MLXArray([92, 85, 2, 95, 55, 7, 94, 42]))
        let parameters = GenerateParameters(
            maxTokens: 32,
            temperature: 0,
            repetitionPenalty: withLogitProcessor ? 1.5 : nil,
            presencePenalty: withLogitProcessor ? 0.5 : nil,
            frequencyPenalty: withLogitProcessor ? 0.2 : nil
        )
        let policy = try AdaptiveSpeculativeDecodingPolicy(
            warmUpRounds: 3,
            observationWindowRounds: 2,
            minimumObservedDraftTokens: 4,
            minimumAcceptanceRate: 0.5
        )

        let normalContext = ModelContext(
            configuration: processor.configuration,
            model: normalModel,
            processor: processor,
            tokenizer: tokenizer
        )
        var normalTokens: [Int] = []
        for await generation in try generateTokens(
            input: input, parameters: parameters, context: normalContext
        ) {
            if let token = generation.token { normalTokens.append(token) }
        }

        let adaptiveContext = ModelContext(
            configuration: processor.configuration,
            model: adaptiveModel,
            processor: processor,
            tokenizer: tokenizer
        )
        var adaptiveTokens: [Int] = []
        var completionInfo: GenerateCompletionInfo?
        for await generation in try generateTokens(
            input: input,
            parameters: parameters,
            context: adaptiveContext,
            draftModel: mismatchedDraftModel,
            numDraftTokens: 2,
            adaptivePolicy: policy
        ) {
            if let token = generation.token { adaptiveTokens.append(token) }
            if let info = generation.info { completionInfo = info }
        }

        #expect(adaptiveTokens == normalTokens)
        #expect(adaptiveTokens.count == 32)

        let telemetry = try #require(completionInfo?.speculativeDecodingTelemetry)
        let adaptive = try #require(telemetry.adaptive)
        #expect(adaptive.state == .autoregressive)
        #expect(adaptive.fallbackReason == .acceptanceRateBelowMinimum)
        #expect(adaptive.fallbackAfterRoundCount == 3)
        #expect(adaptive.observedRoundCount == 2)
        #expect(adaptive.observedDraftTokenCount == 4)
        #expect(adaptive.observedAcceptanceRate < policy.minimumAcceptanceRate)
        #expect(telemetry.roundCount == 3)
        #expect(telemetry.draftModelCallCount == 6)
        #expect(mismatchedDraftModel.invocationCount == 6)
        #expect(adaptive.autoregressiveTargetModelCallCount > 0)
        #expect(
            telemetry.targetModelCallCount
                == telemetry.roundCount + adaptive.autoregressiveTargetModelCallCount
        )
        #expect(telemetry.emittedTokenCount == adaptiveTokens.count)
    }

    @Test func `High recent acceptance keeps speculative mode active`() async throws {
        let vocabularySize = 101
        let tokenizer = TestTokenizer(vocabularySize: vocabularySize)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "adaptive-high-acceptance-test"),
            messageGenerator: DefaultMessageGenerator()
        )
        let mainModel = AdaptiveTransitionLanguageModel(
            vocabularySize: vocabularySize, transitionOffset: 7)
        let matchingDraftModel = AdaptiveTransitionLanguageModel(
            vocabularySize: vocabularySize, transitionOffset: 7)
        let context = ModelContext(
            configuration: processor.configuration,
            model: mainModel,
            processor: processor,
            tokenizer: tokenizer
        )
        let parameters = GenerateParameters(maxTokens: 24, temperature: 0)
        let policy = try AdaptiveSpeculativeDecodingPolicy(
            warmUpRounds: 2,
            observationWindowRounds: 2,
            minimumObservedDraftTokens: 8,
            minimumAcceptanceRate: 0.75
        )

        var tokens: [Int] = []
        var completionInfo: GenerateCompletionInfo?
        for await generation in try generateTokens(
            input: LMInput(tokens: MLXArray([3, 8, 13])),
            parameters: parameters,
            context: context,
            draftModel: matchingDraftModel,
            numDraftTokens: 4,
            adaptivePolicy: policy
        ) {
            if let token = generation.token { tokens.append(token) }
            if let info = generation.info { completionInfo = info }
        }

        #expect(tokens.count == 24)
        let telemetry = try #require(completionInfo?.speculativeDecodingTelemetry)
        let adaptive = try #require(telemetry.adaptive)
        #expect(adaptive.state == .monitoring)
        #expect(!adaptive.didFallbackToAutoregressive)
        #expect(adaptive.fallbackReason == nil)
        #expect(adaptive.fallbackAfterRoundCount == nil)
        #expect(adaptive.observedAcceptanceRate == 1)
        #expect(adaptive.evaluatedWindowCount > 0)
    }

    @Test func `Omitting adaptive policy preserves fixed speculative telemetry`() async throws {
        let vocabularySize = 101
        let tokenizer = TestTokenizer(vocabularySize: vocabularySize)
        let processor = TestInputProcessor(
            tokenizer: tokenizer,
            configuration: ModelConfiguration(id: "fixed-speculative-test"),
            messageGenerator: DefaultMessageGenerator()
        )
        let mainModel = AdaptiveTransitionLanguageModel(
            vocabularySize: vocabularySize, transitionOffset: 7)
        let draftModel = AdaptiveTransitionLanguageModel(
            vocabularySize: vocabularySize, transitionOffset: 19)
        let context = ModelContext(
            configuration: processor.configuration,
            model: mainModel,
            processor: processor,
            tokenizer: tokenizer
        )

        var completionInfo: GenerateCompletionInfo?
        for await generation in try generateTokens(
            input: LMInput(tokens: MLXArray([3, 8, 13])),
            parameters: GenerateParameters(maxTokens: 12, temperature: 0),
            context: context,
            draftModel: draftModel,
            numDraftTokens: 2
        ) {
            if let info = generation.info { completionInfo = info }
        }

        let telemetry = try #require(completionInfo?.speculativeDecodingTelemetry)
        #expect(telemetry.adaptive == nil)
        #expect(telemetry.roundCount > 3)
    }
}

/// Deterministic model with high-margin logits. Varying `transitionOffset`
/// produces either a perfectly matching or systematically mismatched draft
/// model while keeping target-only and speculative verification deterministic.
private final class AdaptiveTransitionLanguageModel: Module, LanguageModel,
    KVCacheDimensionProvider
{
    let vocabularySize: Int
    let transitionOffset: Int
    var kvHeads: [Int] { [] }
    private(set) var invocationCount = 0

    init(vocabularySize: Int, transitionOffset: Int) {
        self.vocabularySize = vocabularySize
        self.transitionOffset = transitionOffset
        super.init()
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        invocationCount += 1
        let tokenIds = inputs.asArray(Int.self)
        var logits = Array(
            repeating: Float(-100),
            count: tokenIds.count * vocabularySize
        )
        for (position, token) in tokenIds.enumerated() {
            let nextToken = (token * 31 + transitionOffset) % vocabularySize
            logits[position * vocabularySize + nextToken] = 100
        }
        return MLXArray(logits, [1, tokenIds.count, vocabularySize])
    }
}
