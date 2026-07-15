// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
import MLXOptimizers
import XCTest

@testable import MLXLMCommon

/// See also ChatSessionIntegrationTests
public class ChatSessionTests: XCTestCase {

    private struct RecordedMessage: Equatable, Sendable {
        var role: Chat.Message.Role
        var content: String
    }

    private struct RecordingMessageGenerator: MessageGenerator {
        let continuation: AsyncStream<[RecordedMessage]>.Continuation

        func generate(messages: [Chat.Message]) -> [Message] {
            continuation.yield(messages.map { .init(role: $0.role, content: $0.content) })

            return DefaultMessageGenerator().generate(messages: messages)
        }
    }

    private struct UnexpectedDraftModelLoadError: Error {}

    private actor DraftModelLoadCounter {
        private var count = 0

        func increment() {
            count += 1
        }

        var value: Int {
            count
        }
    }

    private enum EventCollectionError: Error {
        case timeout
    }

    private static func collectTicketEvents(
        stream: AsyncStream<WiredMemoryEvent>,
        ticketIDs: Set<UUID>
    ) async throws -> [WiredMemoryEvent] {
        try await withThrowingTaskGroup(of: [WiredMemoryEvent].self) { group in
            group.addTask {
                var events: [WiredMemoryEvent] = []
                var ended = Set<UUID>()
                for await event in stream {
                    events.append(event)
                    if event.kind == .ticketEnded, let ticketID = event.ticketID {
                        ended.insert(ticketID)
                    }
                    if ended.isSuperset(of: ticketIDs) {
                        return events
                    }
                }
                return events
            }
            group.addTask {
                try await Task.sleep(for: .seconds(5))
                throw EventCollectionError.timeout
            }

            let events = try await group.next() ?? []
            group.cancelAll()
            return events
        }
    }

    private static func makeModel(processor: TestInputProcessor = TestInputProcessor())
        -> ModelContext
    {
        let config = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64, attentionHeads: 4,
            headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )
        let model = Gemma3TextModel(config)
        quantize(model: model, groupSize: 64, bits: 4)

        // Force evaluation of all model weights before concurrent usage
        // This ensures all weight promises are realized and avoids race conditions
        eval(model)

        return .init(
            configuration: processor.configuration,
            model: model,
            processor: processor,
            tokenizer: processor.tokenizer)
    }

    private func model(processor: TestInputProcessor = TestInputProcessor()) -> ModelContext {
        Self.makeModel(processor: processor)
    }

    private let generationParameters = GenerateParameters(maxTokens: 50)

    private let targetLength = 1

    func testChatSessionSync() async throws {
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        let result1 = try await session.respond(to: "hello")
        XCTAssertGreaterThan(result1.count, targetLength, result1)
        let result2 = try await session.respond(to: "hello again")
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionAsync() async throws {
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        var result1 = ""
        for try await part in session.streamResponse(to: "hello") {
            result1 += part
        }
        XCTAssertGreaterThan(result1.count, targetLength, result1)

        var result2 = ""
        for try await part in session.streamResponse(to: "hello again") {
            result2 += part
        }
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionRespondToMessages() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)

        let result = try await session.respond(to: [
            .user("hello"),
            .assistant("hi"),
            .user("hello again"),
        ])
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testChatSessionStreamResponseToMessages() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)

        var result = ""
        for try await part in session.streamResponse(to: [
            .user("hello"),
            .assistant("hi"),
            .user("hello again"),
        ]) {
            result += part
        }
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testStructuredContinuationAvoidsReplayingHistoryAcrossToolTurns() async throws {
        let (recordedMessages, continuation) = AsyncStream<[RecordedMessage]>.makeStream()
        let processor = TestInputProcessor(
            tokenizer: TestTokenizer(),
            configuration: ModelConfiguration(id: "test"),
            messageGenerator: RecordingMessageGenerator(continuation: continuation))
        let history: [Chat.Message] = (0 ..< 8).flatMap { index in
            [
                .user("question \(index)"),
                .assistant("answer \(index)"),
            ]
        }
        let continuations: [[Chat.Message]] = [
            [.tool("first tool result")],
            [.tool("second tool result")],
            [.user("final answer")],
        ]
        let session = ChatSession(
            model(processor: processor),
            history: history,
            generateParameters: GenerateParameters(maxTokens: 1))

        for messages in continuations {
            _ = try await session.respond(to: messages)
        }
        continuation.finish()

        var calls: [[RecordedMessage]] = []
        for await call in recordedMessages {
            calls.append(call)
        }

        XCTAssertEqual(calls.map(\.count), [history.count + 1, 1, 1])
        XCTAssertEqual(calls[0].map(\.role), history.map(\.role) + [.tool])
        XCTAssertEqual(calls[1].map(\.role), [.tool])
        XCTAssertEqual(calls[2].map(\.role), [.user])

        let actualPreparedMessageCount = calls.reduce(0) { $0 + $1.count }
        let replayedHistoryPreparedMessageCount = continuations.indices.reduce(0) {
            $0 + history.count + $1 + 1
        }
        XCTAssertLessThan(actualPreparedMessageCount, replayedHistoryPreparedMessageCount)
    }

    func testChatSessionAsyncInterrupt() async throws {
        // interrupt the streamResponse and continue with another request
        let model = model()
        let session = ChatSession(model, generateParameters: generationParameters)

        for _ in 0 ..< 10 {
            var result1 = ""
            for try await part in session.streamResponse(to: "hello") {
                result1 += part
                break
            }

            // at this point the performStreaming/generate code may still be running.
            // the next call can corrupt the state if not thread safe

            var result2 = ""
            for try await part in session.streamResponse(to: "hello again") {
                result2 += part
                if result2.count > 100 {
                    break
                }
            }
        }

        // since we are interrupting we need to wait for everything to finish
        // (avoids shutdown issues if this is the last/only test). because the
        // streaming task is not a synchronous shutdown
        await session.synchronize()
    }

    func testChatSessionWithTools() async throws {
        let model = model()
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": [
                        "type": "object",
                        "properties": [
                            "location": [
                                "type": "string",
                                "description": "City name",
                            ] as [String: any Sendable]
                        ] as [String: any Sendable],
                        "required": ["location"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model, generateParameters: generationParameters, tools: tools
        )

        let result = try await session.respond(to: "What is the weather in SF?")
        XCTAssertGreaterThan(result.count, targetLength, result)

        // second turn to verify tools persist through cache
        let result2 = try await session.respond(to: "How about NYC?")
        XCTAssertGreaterThan(result2.count, targetLength, result2)
    }

    func testChatSessionWithToolsStreaming() async throws {
        let model = model()
        let tools: [ToolSpec] = [
            [
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": [
                        "type": "object",
                        "properties": [:] as [String: any Sendable],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ] as ToolSpec
        ]
        let session = ChatSession(
            model, generateParameters: generationParameters, tools: tools
        )

        var result = ""
        for try await part in session.streamResponse(to: "hello") {
            result += part
        }
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testSpeculativeDecodingMemoryPolicyFallbackUsesDefaultGeneration() async throws {
        let draft = ModelContainer(context: model())
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: draft,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fallbackToDefault
                )
            ),
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(
            to: "hello",
            role: .user,
            images: [] as [UserInput.Image],
            videos: [] as [UserInput.Video]
        ) {
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }

        let completionInfo = try XCTUnwrap(info)
        XCTAssertNil(completionInfo.speculativeDecodingTelemetry)
    }

    func testSpeculativeDecodingMemoryPolicyFailThrows() async throws {
        let draft = ModelContainer(context: model())
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModel: draft,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fail
                )
            ),
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        do {
            for try await _ in session.streamDetails(
                to: "hello",
                role: .user,
                images: [] as [UserInput.Image],
                videos: [] as [UserInput.Video]
            ) {}
            XCTFail("expected SpeculativeDecodingMemoryError")
        } catch let error as SpeculativeDecodingMemoryError {
            XCTAssertFalse(error.evaluation.isWithinBudget)
            XCTAssertFalse(error.evaluation.shouldUseSpeculativeDecoding)
        } catch {
            XCTFail("expected SpeculativeDecodingMemoryError, got \(error)")
        }
    }

    func testDeferredSpeculativeDecodingMemoryPolicyFallbackDoesNotLoadDraftModel() async throws {
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 1,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fallbackToDefault
                )
            ) {
                throw UnexpectedDraftModelLoadError()
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        var info: GenerateCompletionInfo?
        for try await generation in session.streamDetails(
            to: "hello",
            role: .user,
            images: [] as [UserInput.Image],
            videos: [] as [UserInput.Video]
        ) {
            if let generationInfo = generation.info {
                info = generationInfo
            }
        }

        let completionInfo = try XCTUnwrap(info)
        XCTAssertNil(completionInfo.speculativeDecodingTelemetry)
    }

    func testDeferredSpeculativeDecodingMemoryPolicyFailDoesNotLoadDraftModel() async throws {
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 1,
                numDraftTokens: 2,
                memoryPolicy: SpeculativeDecodingMemoryPolicy(
                    limitBytes: 0,
                    action: .fail
                )
            ) {
                throw UnexpectedDraftModelLoadError()
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        do {
            for try await _ in session.streamDetails(
                to: "hello",
                role: .user,
                images: [] as [UserInput.Image],
                videos: [] as [UserInput.Video]
            ) {}
            XCTFail("expected SpeculativeDecodingMemoryError")
        } catch is UnexpectedDraftModelLoadError {
            XCTFail("draft model loader should not be called")
        } catch let error as SpeculativeDecodingMemoryError {
            XCTAssertFalse(error.evaluation.isWithinBudget)
            XCTAssertFalse(error.evaluation.shouldUseSpeculativeDecoding)
        } catch {
            XCTFail("expected SpeculativeDecodingMemoryError, got \(error)")
        }
    }

    func testDeferredSpeculativeDecodingLoadsDraftModelOnceAcrossTurns() async throws {
        let loadCounter = DraftModelLoadCounter()
        let session = ChatSession(
            model(),
            speculativeDecoding: SpeculativeDecodingConfig(
                draftModelBytes: 0,
                numDraftTokens: 2
            ) {
                await loadCounter.increment()
                return ModelContainer(context: Self.makeModel())
            },
            generateParameters: GenerateParameters(maxTokens: 4, temperature: 0.0)
        )

        _ = try await session.respond(to: "hello")
        _ = try await session.respond(to: "again")

        let loadCount = await loadCounter.value
        XCTAssertEqual(loadCount, 1)
    }

    func testWiredMemoryTicketCanChangeBetweenTurns() async throws {
        try await Device.withDefaultDevice(.cpu) {
            let manager = WiredMemoryManager.makeForTesting(
                configuration: .init(
                    policyOnlyWhenUnsupported: true,
                    baselineOverride: 0,
                    useRecommendedWorkingSetWhenUnsupported: false
                )
            )
            let policy = MLXLMCommon.WiredSumPolicy(cap: 1 << 20)
            let firstTicket = policy.ticket(size: 1_024, manager: manager)
            let secondTicket = policy.ticket(size: 2_048, manager: manager)
            let eventStream = await manager.events()
            let eventTask = Task {
                try await Self.collectTicketEvents(
                    stream: eventStream,
                    ticketIDs: [firstTicket.id, secondTicket.id]
                )
            }

            let session = ChatSession(
                Self.makeModel(),
                generateParameters: GenerateParameters(maxTokens: 1, temperature: 0)
            )
            session.wiredMemoryTicket = firstTicket
            _ = try await session.respond(to: "first")
            session.wiredMemoryTicket = secondTicket
            _ = try await session.respond(to: "second")

            let events = try await eventTask.value
            let startedIDs = Set(
                events.filter { $0.kind == .ticketStarted }.compactMap(\.ticketID)
            )
            let endedIDs = Set(
                events.filter { $0.kind == .ticketEnded }.compactMap(\.ticketID)
            )
            XCTAssertTrue(startedIDs.isSuperset(of: [firstTicket.id, secondTicket.id]))
            XCTAssertTrue(endedIDs.isSuperset(of: [firstTicket.id, secondTicket.id]))
        }
    }

    func testWiredMemoryTicketPropagatesToSpeculativeGeneration() async throws {
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
                try await Self.collectTicketEvents(stream: eventStream, ticketIDs: [ticket.id])
            }

            let session = ChatSession(
                Self.makeModel(),
                speculativeDecoding: SpeculativeDecodingConfig(
                    draftModel: ModelContainer(context: Self.makeModel()),
                    numDraftTokens: 2
                ),
                generateParameters: GenerateParameters(maxTokens: 2, temperature: 0)
            )
            session.wiredMemoryTicket = ticket
            _ = try await session.respond(to: "speculate")

            let events = try await eventTask.value
            XCTAssertTrue(
                events.contains { $0.kind == .ticketStarted && $0.ticketID == ticket.id })
            XCTAssertTrue(events.contains { $0.kind == .ticketEnded && $0.ticketID == ticket.id })
        }
    }

    // MARK: - KV Cache

    func testCurrentCacheNilBeforeGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    func testCurrentCacheAfterGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
    }

    func testPrefillPopulatesCacheWithoutGeneratingTokens() async throws {
        let session = ChatSession(
            model(),
            generateParameters: GenerateParameters(maxTokens: 50, temperature: 0)
        )

        try await session.prefill("hello")

        await session.withCache { cache in
            XCTAssertEqual(cache?.map(\.offset), Array(repeating: 8, count: 8))
        }
    }

    func testPrefillPreservesLMOutputStateForNextTurn() async throws {
        let processor = TestInputProcessor()
        let stateModel = PrefillStateTrackingModel(vocabularySize: 100)
        let context = ModelContext(
            configuration: processor.configuration,
            model: stateModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )
        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0)
        )

        try await session.prefill("prefill")
        _ = try await session.respond(to: "continue")

        XCTAssertGreaterThanOrEqual(stateModel.receivedStates.count, 2)
        XCTAssertNil(stateModel.receivedStates[0])
        XCTAssertEqual(stateModel.receivedStates[1], 1)
    }

    func testCancelledPrefillKeepsPreviouslyCommittedCache() async throws {
        let processorCounter = PrefillProcessorCounter()
        let processor = CancellablePrefillProcessor(counter: processorCounter)
        let stateModel = PrefillStateTrackingModel(vocabularySize: 100)
        let context = ModelContext(
            configuration: ModelConfiguration(id: "prefill-cancellation-test"),
            model: stateModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )
        let session = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0)
        )
        try await session.prefill("committed")
        let sessionBox = UncheckedSendableBox(session)

        let task = Task {
            try await sessionBox.value.prefill("cancelled")
        }
        try await Task.sleep(for: .milliseconds(50))
        task.cancel()
        do {
            try await task.value
            XCTFail("expected CancellationError")
        } catch is CancellationError {
            // expected
        }

        _ = try await sessionBox.value.respond(to: "continue")
        XCTAssertGreaterThanOrEqual(stateModel.receivedStates.count, 2)
        XCTAssertEqual(stateModel.receivedStates[1], 1)
    }

    func testInitWithKVCache() async throws {
        // build a cache from an initial session
        let container = ModelContainer(context: model())
        let initial = ChatSession(container, generateParameters: generationParameters)
        _ = try await initial.respond(to: "hello")

        try await initial.withCache { [targetLength, generationParameters] cache in
            XCTAssertNotNil(cache)

            if let cache {
                // restore the cache into a new session and verify generation continues
                let restored = ChatSession(
                    container,
                    cache: cache.map { $0.copy() },
                    generateParameters: generationParameters)
                let result = try await restored.respond(to: "hello again")
                XCTAssertGreaterThan(result.count, targetLength, result)
            }
        }
    }

    func testSaveCacheThrowsBeforeGeneration() async throws {
        let session = ChatSession(model(), generateParameters: generationParameters)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        do {
            try await session.saveCache(to: url)
            XCTFail("expected ChatSessionError.noCacheAvailable")
        } catch ChatSessionError.noCacheAvailable {
            // expected
        }
    }

    func testSaveAndRestoreCache() async throws {
        let ctx = model()
        let initial = ChatSession(ctx, generateParameters: generationParameters)
        _ = try await initial.respond(to: "hello")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        try await initial.saveCache(to: url)

        let (loadedCache, _) = try loadPromptCache(url: url)
        let restored = ChatSession(
            ctx, cache: loadedCache, generateParameters: generationParameters)
        let result = try await restored.respond(to: "hello again")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testPrefillSaveLoadValidatesMetadataAndRemainsLowLevelCompatible() async throws {
        let context = model()
        let initial = ChatSession(
            context,
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0)
        )
        try await initial.prefill("persistent context")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        let metadata = ["model": "test", "promptFingerprint": "abc123"]
        try await initial.saveCache(to: url, metadata: metadata)

        let (_, lowLevelMetadata) = try loadPromptCache(url: url)
        XCTAssertEqual(lowLevelMetadata["model"], "test")
        XCTAssertEqual(
            lowLevelMetadata[ChatSession.cacheFormatVersionMetadataKey],
            String(ChatSession.cacheFormatVersion)
        )

        let snapshot = try ChatSession.loadCache(from: url, validating: metadata)
        XCTAssertEqual(snapshot.metadata, metadata)
        XCTAssertEqual(snapshot.formatVersion, ChatSession.cacheFormatVersion)

        let restored = ChatSession(
            context,
            cache: snapshot.cache,
            generateParameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )
        let result = try await restored.respond(to: "continue")
        XCTAssertGreaterThan(result.count, targetLength, result)
    }

    func testCacheMetadataMismatchIsRejected() async throws {
        let session = ChatSession(
            model(),
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0)
        )
        try await session.prefill("persistent context")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try await session.saveCache(to: url, metadata: ["model": "test"])

        XCTAssertThrowsError(
            try ChatSession.loadCache(from: url, validating: ["model": "other"])
        ) { error in
            guard case ChatSessionCacheError.cacheMetadataMismatch(_, _, _) = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testHighLevelCacheLoadRejectsMissingAndUnsupportedVersions() async throws {
        let session = ChatSession(
            model(),
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0)
        )
        try await session.prefill("persistent context")

        let missingVersionURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        let futureVersionURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        defer {
            try? FileManager.default.removeItem(at: missingVersionURL)
            try? FileManager.default.removeItem(at: futureVersionURL)
        }

        try await session.withCache { cache in
            let cache = try XCTUnwrap(cache)
            try savePromptCache(url: missingVersionURL, cache: cache)
            try savePromptCache(
                url: futureVersionURL,
                cache: cache,
                metadata: [ChatSession.cacheFormatVersionMetadataKey: "999"]
            )
        }
        XCTAssertThrowsError(try ChatSession.loadCache(from: missingVersionURL)) { error in
            guard case ChatSessionCacheError.missingCacheFormatVersion = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }

        XCTAssertThrowsError(try ChatSession.loadCache(from: futureVersionURL)) { error in
            guard case ChatSessionCacheError.unsupportedCacheFormatVersion(_, _) = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
    }

    func testReservedCacheMetadataKeyIsRejected() async throws {
        let session = ChatSession(
            model(),
            generateParameters: GenerateParameters(maxTokens: 1, temperature: 0)
        )
        try await session.prefill("persistent context")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")

        do {
            try await session.saveCache(
                to: url,
                metadata: [ChatSession.cacheFormatVersionMetadataKey: "2"]
            )
            XCTFail("expected reservedCacheMetadataKey")
        } catch ChatSessionCacheError.reservedCacheMetadataKey(_) {
            // expected
        }
    }

    func testCurrentCacheNilForHistorySessionBeforeGeneration() async throws {
        // .history state should behave like .empty: no cache until first generation
        let history: [Chat.Message] = [.user("hello"), .assistant("hi")]
        let session = ChatSession(
            model(), history: history, generateParameters: generationParameters)
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    func testCurrentCacheNonNilForHistorySessionAfterGeneration() async throws {
        // after generation from .history state, cache transitions to .kvcache
        let history: [Chat.Message] = [.user("hello"), .assistant("hi")]
        let session = ChatSession(
            model(),
            history: history,
            generateParameters: generationParameters)
        _ = try await session.respond(to: "hello again")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
    }

    func testCurrentCacheNilAfterClear() async throws {
        // clear() resets to .empty; currentCache() should return nil again
        let session = ChatSession(model(), generateParameters: generationParameters)
        _ = try await session.respond(to: "hello")
        await session.withCache { cache in
            XCTAssertNotNil(cache)
        }
        await session.clear()
        await session.withCache { cache in
            XCTAssertNil(cache)
        }
    }

    /// something that looks like a view model
    @MainActor class ChatModel {
        let session: ChatSession

        public var messages = [Chat.Message]()

        private var task: Task<Void, Error>?
        public var isBusy: Bool {
            task != nil
        }

        init(model: ModelContext) {
            self.session = ChatSession(model)
        }

        public func cancel() {
            task?.cancel()
        }

        public func respond(_ message: String) {
            guard task == nil else { return }

            self.messages.append(.init(role: .user, content: message))
            self.messages.append(.init(role: .assistant, content: "..."))
            let lastIndex = self.messages.count - 1

            self.task = Task {
                var first = true
                for try await item in session.streamResponse(to: message) {
                    if first {
                        self.messages[lastIndex].content = item
                        first = false
                    } else {
                        self.messages[lastIndex].content += item
                    }
                }
                self.task = nil
            }
        }
    }

    @MainActor
    func testViewModel() async throws {
        let model = ChatModel(model: model())

        // start producing a response but interrupt it
        // triggers https://github.com/ml-explore/mlx-swift/pull/323
        model.respond("message1")
        try await Task.sleep(for: .milliseconds(50))
        model.cancel()

        // wait for it to finish
        while model.isBusy {
            try await Task.sleep(for: .milliseconds(10))
        }

        // try another message, wait for full completion (but cap the length)
        model.session.generateParameters = self.generationParameters
        model.respond("message2")
        while model.isBusy {
            try await Task.sleep(for: .milliseconds(10))
        }
    }
}

private let prefillStateSequenceKey = LMOutput.Key<Int>("chat-session-prefill-sequence")

/// Test-only transfer wrapper. Tests await the child task before accessing the
/// wrapped single-consumer session again, so no concurrent use is permitted.
private struct UncheckedSendableBox<Value>: @unchecked Sendable {
    let value: Value

    init(_ value: Value) {
        self.value = value
    }
}

private actor PrefillProcessorCounter {
    private var count = 0

    func next() -> Int {
        count += 1
        return count
    }
}

private struct CancellablePrefillProcessor: UserInputProcessor {
    let counter: PrefillProcessorCounter
    let tokenizer = TestTokenizer()

    func prepare(input: UserInput) async throws -> LMInput {
        if await counter.next() == 2 {
            try await Task.sleep(for: .seconds(5))
        }
        return LMInput(tokens: MLXArray(Array(1 ... tokenizer.length)))
    }
}

private final class PrefillStateTrackingModel: Module, LanguageModel, KVCacheDimensionProvider {
    let vocabularySize: Int
    var kvHeads: [Int] { [] }
    private(set) var receivedStates: [Int?] = []

    init(vocabularySize: Int) {
        self.vocabularySize = vocabularySize
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

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        let previous = state?[prefillStateSequenceKey]
        receivedStates.append(previous)

        var nextState = state ?? LMOutput.State()
        nextState[prefillStateSequenceKey] = (previous ?? 0) + 1

        let tokenCount = input.tokens.size
        let logits = MLXArray(
            Array(repeating: Float(0), count: tokenCount * vocabularySize),
            [1, tokenCount, vocabularySize]
        )
        return LMOutput(logits: logits, state: nextState)
    }
}
