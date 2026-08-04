// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

private let mtpSessionStateKey = LMOutput.Key<Int>("tests.mtp-session-state")

private actor MTPDrafterLoadCounter {
    private(set) var value = 0

    func increment() {
        value += 1
    }
}

private final class MTPChatDrafter: Module, MTPDrafterModel {
    let weight = MLXArray.zeros([4])
    private(set) var callCount = 0

    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray {
        callCount += 1
        return MLXArray(Array(repeating: Int32(1), count: blockSize - 1), [1, blockSize - 1])
    }
}

private final class MTPChatTarget: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize = 100
    var kvHeads: [Int] { [1] }
    private(set) var preparedStateSequence: [Int?] = []
    private(set) var emittedStateCount = 0
    var omitMTPState = false

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        windowSize: Int?
    ) throws -> PrepareResult {
        preparedStateSequence.append(state?[mtpSessionStateKey])
        return .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        let positions = input.tokens.dim(-1)
        if let cache = cache?.first as? MTPTrackingKVCache {
            cache.offset += positions
        }

        let logits = deterministicLogits(positions: positions)
        var nextState = state ?? LMOutput.State()
        nextState[mtpSessionStateKey] = (state?[mtpSessionStateKey] ?? 0) + 1

        if state?[mtpEmitFlagKey] ?? false, !omitMTPState {
            emittedStateCount += 1
            let span = cache?.first?.offset ?? positions
            nextState[mtpLastHiddenStatesKey] = MLXArray.zeros([1, positions, 4])
            nextState[mtpSharedKVStatesKey] = [
                "full_attention": (
                    MLXArray.zeros([1, 1, span, 4]),
                    MLXArray.zeros([1, 1, span, 4])
                )
            ]
        }
        return LMOutput(logits: logits, state: nextState)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        deterministicLogits(positions: inputs.dim(-1))
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [MTPTrackingKVCache()]
    }

    private func deterministicLogits(positions: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: positions * vocabularySize)
        for position in 0 ..< positions {
            values[position * vocabularySize + 1] = 100
        }
        return MLXArray(values, [1, positions, vocabularySize])
    }
}

private final class MTPTrackingKVCache: KVCache {
    var offset = 0
    var maxSize: Int? { nil }
    var isTrimmable: Bool { true }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        offset += keys.dim(-2)
        return (keys, values)
    }

    var state: [MLXArray] {
        get { [] }
        set {}
    }

    func innerState() -> [MLXArray] { [] }

    var metaState: [String] {
        get { [] }
        set {}
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        guard n > 0 else { return 0 }
        let removed = min(n, offset)
        offset -= removed
        return removed
    }

    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache {
        let copy = MTPTrackingKVCache()
        copy.offset = offset
        return copy
    }
}

@Suite(.serialized)
struct ChatSessionMTPTests {
    private static func makeTargetContext(_ target: MTPChatTarget = MTPChatTarget())
        -> ModelContext
    {
        let processor = TestInputProcessor()
        return ModelContext(
            configuration: processor.configuration,
            model: target,
            processor: processor,
            tokenizer: processor.tokenizer
        )
    }

    private static func makeDrafterContainer(_ drafter: MTPChatDrafter = MTPChatDrafter())
        -> MTPDrafterContainer
    {
        MTPDrafterContainer(
            context: MTPDrafterContext(
                configuration: ModelConfiguration(id: "test-mtp-drafter"),
                model: drafter
            )
        )
    }

    private func completionInfo(from session: ChatSession, prompt: String) async throws
        -> GenerateCompletionInfo
    {
        var result: GenerateCompletionInfo?
        for try await generation in session.streamDetails(to: prompt) {
            if let info = generation.info {
                result = info
            }
        }
        return try #require(result)
    }

    @Test
    func configurationRejectsInvalidBlockSizes() throws {
        let drafter = Self.makeDrafterContainer()

        #expect(throws: MTPSpeculativeDecodingConfigError.invalidBlockSize(1)) {
            try MTPSpeculativeDecodingConfig(drafter: drafter, blockSize: 1)
        }
        #expect(throws: MTPSpeculativeDecodingConfigError.invalidBlockSize(0)) {
            try MTPSpeculativeDecodingConfig(drafterBytes: 0, blockSize: 0) {
                drafter
            }
        }

        let normalized = try MTPSpeculativeDecodingConfig(drafterBytes: -1) {
            drafter
        }
        #expect(normalized.estimatedDrafterBytes == 0)
    }

    @Test
    func eagerConfigurationExposesPublicValues() throws {
        let drafter = Self.makeDrafterContainer()
        let policy = SpeculativeDecodingMemoryPolicy(
            limitBytes: 1_024,
            additionalBytes: 128,
            action: .fail
        )
        let configuration = try MTPSpeculativeDecodingConfig(
            drafter: drafter,
            blockSize: 5,
            memoryPolicy: policy
        )

        #expect(configuration.drafter === drafter)
        #expect(configuration.blockSize == 5)
        #expect(configuration.memoryPolicy == policy)
    }

    @Test
    func sessionEmitsMTPTelemetryAndCarriesStateAcrossTurns() async throws {
        let target = MTPChatTarget()
        let drafter = MTPChatDrafter()
        let configuration = try MTPSpeculativeDecodingConfig(
            drafter: Self.makeDrafterContainer(drafter),
            blockSize: 4
        )
        let session = ChatSession(
            Self.makeTargetContext(target),
            mtpSpeculativeDecoding: configuration,
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0)
        )

        #expect(session.speculativeDecoding == nil)
        #expect(session.mtpSpeculativeDecoding?.blockSize == 4)

        let first = try await completionInfo(from: session, prompt: "first")
        let firstCacheOffset = await session.withCache { $0?.first?.offset } ?? nil
        let second = try await completionInfo(from: session, prompt: "second")
        let secondCacheOffset = await session.withCache { $0?.first?.offset } ?? nil

        #expect(first.proposedDraftTokens != nil)
        #expect(first.proposedDraftTokens! > 0)
        #expect(first.acceptedDraftTokens == first.proposedDraftTokens)
        #expect(first.passthroughReason == nil)
        #expect(first.speculativeDecodingTelemetry?.draftModelCallCount ?? 0 > 0)
        #expect(second.proposedDraftTokens != nil)
        #expect(drafter.callCount >= 2)
        #expect(target.preparedStateSequence.count == 2)
        #expect(target.preparedStateSequence[0] == nil)
        #expect(target.preparedStateSequence[1] != nil)
        #expect(firstCacheOffset != nil)
        #expect(secondCacheOffset != nil)
        #expect(secondCacheOffset! > firstCacheOffset!)
    }

    @Test
    func classicSpeculativeSessionAlsoCarriesFinalTargetState() async throws {
        let main = MTPChatTarget()
        let draft = MTPChatTarget()
        let session = ChatSession(
            Self.makeTargetContext(main),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: ModelContainer(context: Self.makeTargetContext(draft)),
                numDraftTokens: 2
            ),
            generateParameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )

        _ = try await completionInfo(from: session, prompt: "first")
        _ = try await completionInfo(from: session, prompt: "second")

        #expect(main.preparedStateSequence.count == 2)
        #expect(main.preparedStateSequence[0] == nil)
        #expect(main.preparedStateSequence[1] != nil)
    }

    @Test
    func quantizedCacheMissingStateReportsExplicitPassthroughReason() throws {
        let target = MTPChatTarget()
        target.omitMTPState = true
        var iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([1, 2, 3])),
            mainModel: target,
            drafter: MTPChatDrafter(),
            mainCache: [QuantizedKVCache(groupSize: 64, bits: 4)],
            parameters: GenerateParameters(maxTokens: 3, temperature: 0),
            blockSize: 4
        )

        while iterator.next() != nil {}

        #expect(iterator.passthroughReason?.contains("after KV cache quantization") == true)
    }

    @Test
    func deferredMemoryFallbackDoesNotLoadDrafterAndReportsReason() async throws {
        let loadCounter = MTPDrafterLoadCounter()
        let configuration = try MTPSpeculativeDecodingConfig(
            drafterBytes: 1,
            memoryPolicy: SpeculativeDecodingMemoryPolicy(
                limitBytes: 0,
                action: .fallbackToDefault
            )
        ) {
            await loadCounter.increment()
            return Self.makeDrafterContainer()
        }
        let session = ChatSession(
            Self.makeTargetContext(),
            mtpSpeculativeDecoding: configuration,
            generateParameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )

        let info = try await completionInfo(from: session, prompt: "fallback")

        let fallbackLoadCount = await loadCounter.value
        #expect(fallbackLoadCount == 0)
        #expect(info.proposedDraftTokens == 0)
        #expect(info.acceptedDraftTokens == 0)
        #expect(info.passthroughReason?.contains("MTP skipped by memory policy") == true)
        #expect(info.speculativeDecodingTelemetry == nil)
    }

    @Test
    func deferredMemoryFailureDoesNotLoadDrafter() async throws {
        let loadCounter = MTPDrafterLoadCounter()
        let configuration = try MTPSpeculativeDecodingConfig(
            drafterBytes: 1,
            memoryPolicy: SpeculativeDecodingMemoryPolicy(limitBytes: 0, action: .fail)
        ) {
            await loadCounter.increment()
            return Self.makeDrafterContainer()
        }
        let session = ChatSession(
            Self.makeTargetContext(),
            mtpSpeculativeDecoding: configuration,
            generateParameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )

        await #expect(throws: SpeculativeDecodingMemoryError.self) {
            for try await _ in session.streamDetails(to: "fail") {}
        }
        let failLoadCount = await loadCounter.value
        #expect(failLoadCount == 0)
    }

    @Test
    func postLoadMemoryGateUsesMeasuredDrafterBytes() async throws {
        let loadCounter = MTPDrafterLoadCounter()
        let configuration = try MTPSpeculativeDecodingConfig(
            drafterBytes: 0,
            memoryPolicy: SpeculativeDecodingMemoryPolicy(
                limitBytes: 0,
                action: .fallbackToDefault
            )
        ) {
            await loadCounter.increment()
            return Self.makeDrafterContainer()
        }
        let session = ChatSession(
            Self.makeTargetContext(),
            mtpSpeculativeDecoding: configuration,
            generateParameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )

        let info = try await completionInfo(from: session, prompt: "post-load fallback")

        let loadCount = await loadCounter.value
        #expect(loadCount == 1)
        #expect(info.proposedDraftTokens == 0)
        #expect(info.acceptedDraftTokens == 0)
        #expect(info.passthroughReason?.contains("MTP skipped by memory policy") == true)
    }

    @Test
    func deferredDrafterLoadsOnceAndPrefillDoesNotLoadIt() async throws {
        let loadCounter = MTPDrafterLoadCounter()
        let drafter = Self.makeDrafterContainer()
        let configuration = try MTPSpeculativeDecodingConfig(drafterBytes: 0) {
            await loadCounter.increment()
            return drafter
        }
        let session = ChatSession(
            Self.makeTargetContext(),
            mtpSpeculativeDecoding: configuration,
            generateParameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )

        try await session.prefill("cached context")
        let prefillLoadCount = await loadCounter.value
        #expect(prefillLoadCount == 0)

        _ = try await completionInfo(from: session, prompt: "first")
        _ = try await completionInfo(from: session, prompt: "second")
        let generationLoadCount = await loadCounter.value
        #expect(generationLoadCount == 1)
    }

    @Test
    func wiredMemoryTicketWrapsMTPGeneration() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let manager = WiredMemoryManager.makeForTesting(
                configuration: .init(
                    policyOnlyWhenUnsupported: true,
                    baselineOverride: 0,
                    useRecommendedWorkingSetWhenUnsupported: false
                )
            )
            let ticket = MLXLMCommon.WiredSumPolicy(cap: 1 << 20).ticket(
                size: 1_024,
                manager: manager
            )
            let eventStream = await manager.events()
            let eventTask = Task {
                var events: [WiredMemoryEvent] = []
                for await event in eventStream {
                    events.append(event)
                    if event.kind == .ticketEnded, event.ticketID == ticket.id {
                        return events
                    }
                }
                return events
            }

            let configuration = try MTPSpeculativeDecodingConfig(
                drafter: Self.makeDrafterContainer()
            )
            let session = ChatSession(
                Self.makeTargetContext(),
                mtpSpeculativeDecoding: configuration,
                generateParameters: GenerateParameters(maxTokens: 2, temperature: 0)
            )
            session.wiredMemoryTicket = ticket
            _ = try await completionInfo(from: session, prompt: "wired")

            let events = await eventTask.value
            #expect(events.contains { $0.kind == .ticketStarted && $0.ticketID == ticket.id })
            #expect(events.contains { $0.kind == .ticketEnded && $0.ticketID == ticket.id })
        }
    }
}
