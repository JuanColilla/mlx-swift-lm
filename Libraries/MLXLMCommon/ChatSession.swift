// Copyright © 2025 Apple Inc.

import Foundation
import MLX

#if canImport(CoreGraphics)
import CoreGraphics
#endif

/// Target-only iterator used when high-level MTP admission falls back before
/// the drafter is loaded. It preserves normal token semantics while exposing
/// the reason through the existing MTP completion-info surface.
private struct MTPFallbackTokenIterator: TokenIteratorProtocol, MTPStatsCollecting {
    private var base: TokenIterator
    private let sink: ChatSessionStateSink

    let passthroughReason: String?
    var proposedDraftTokens: Int { 0 }
    var acceptedDraftTokens: Int { 0 }
    var maxTokens: Int? { base.maxTokens }
    var tokenCount: Int { base.tokenCount }
    var promptPrefillTime: TimeInterval { base.promptPrefillTime }
    var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? { nil }

    init(base: TokenIterator, reason: String, sink: ChatSessionStateSink) {
        self.base = base
        self.sink = sink
        self.passthroughReason = reason
        sink.store(base.state)
    }

    mutating func next() -> Int? {
        let token = base.next()
        sink.store(base.state)
        return token
    }

    mutating func discardGeneratedToken() {
        base.discardGeneratedToken()
        sink.store(base.state)
    }
}

/// Single-consumer handoff for the final iterator state.
///
/// Generation runs in the unstructured task returned by `generateTask`; the
/// session awaits that task before reading this box and before starting any
/// later turn. The lock documents and enforces the handoff boundary while the
/// unchecked conformance is required because `LMOutput.State` can contain
/// non-Sendable MLX arrays.
private final class ChatSessionStateSink: @unchecked Sendable {
    private let lock = NSLock()
    private var state: LMOutput.State?

    func store(_ state: LMOutput.State?) {
        lock.lock()
        self.state = state
        lock.unlock()
    }

    func load() -> LMOutput.State? {
        lock.lock()
        defer { lock.unlock() }
        return state
    }
}

private struct StateReportingSpeculativeTokenIterator: TokenIteratorProtocol {
    private var base: SpeculativeTokenIterator
    private let sink: ChatSessionStateSink

    var maxTokens: Int? { base.maxTokens }
    var tokenCount: Int { base.tokenCount }
    var promptPrefillTime: TimeInterval { base.promptPrefillTime }
    var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        base.speculativeDecodingTelemetry
    }

    init(base: SpeculativeTokenIterator, sink: ChatSessionStateSink) {
        self.base = base
        self.sink = sink
        sink.store(base.state)
    }

    mutating func next() -> Int? {
        let token = base.next()
        sink.store(base.state)
        return token
    }

    mutating func discardGeneratedToken() {
        base.discardGeneratedToken()
        sink.store(base.state)
    }
}

private struct StateReportingMTPTokenIterator: TokenIteratorProtocol, MTPStatsCollecting {
    private var base: MTPSpeculativeTokenIterator
    private let sink: ChatSessionStateSink

    var maxTokens: Int? { base.maxTokens }
    var tokenCount: Int { base.tokenCount }
    var promptPrefillTime: TimeInterval { base.promptPrefillTime }
    var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        base.speculativeDecodingTelemetry
    }
    var proposedDraftTokens: Int { base.proposedDraftTokens }
    var acceptedDraftTokens: Int { base.acceptedDraftTokens }
    var passthroughReason: String? { base.passthroughReason }

    init(base: MTPSpeculativeTokenIterator, sink: ChatSessionStateSink) {
        self.base = base
        self.sink = sink
        sink.store(base.state)
    }

    mutating func next() -> Int? {
        let token = base.next()
        sink.store(base.state)
        return token
    }

    mutating func discardGeneratedToken() {
        base.discardGeneratedToken()
        sink.store(base.state)
    }
}

/// Configuration for speculative decoding in a `ChatSession`.
///
/// Speculative decoding uses a small draft model to propose candidate tokens
/// that the main model then verifies in a single forward pass, providing a
/// speedup with no quality degradation when both models fit comfortably in memory.
///
/// Both models must share the same tokenizer vocabulary.
///
/// Example usage:
/// ```swift
/// let main  = try await LLMModelFactory.shared.loadContainer(configuration: mainConfig)
/// let draft = try await LLMModelFactory.shared.loadContainer(configuration: draftConfig)
///
/// let session = ChatSession(
///     main,
///     speculativeDecoding: SpeculativeDecodingConfig(draftModel: draft, numDraftTokens: 5)
/// )
/// ```
///
/// To avoid loading a draft model that would exceed a memory policy, pass a
/// byte estimate and a loader closure:
///
/// ```swift
/// let session = ChatSession(
///     main,
///     speculativeDecoding: SpeculativeDecodingConfig(
///         draftModelBytes: estimatedDraftBytes,
///         memoryPolicy: .recommendedWorkingSet
///     ) {
///         try await LLMModelFactory.shared.loadContainer(configuration: draftConfig)
///     }
/// )
/// ```
public struct SpeculativeDecodingConfig: Sendable {

    package enum DraftModelSource: Sendable {
        case loaded(ModelContainer)
        case deferred(bytes: Int, @Sendable () async throws -> ModelContainer)
    }

    package let draftModelSource: DraftModelSource

    /// The lightweight model used to propose candidate tokens, when it was provided eagerly.
    ///
    /// Configurations initialized with a loader closure return `nil` because the
    /// draft model is loaded asynchronously by ``ChatSession`` only when speculation
    /// is admitted by the memory policy.
    public var draftModel: ModelContainer? {
        if case .loaded(let draftModel) = draftModelSource {
            return draftModel
        }
        return nil
    }

    /// Number of tokens proposed by the draft model per verification cycle.
    /// The default value of 5 offers a good balance between speed and accuracy.
    public let numDraftTokens: Int

    /// Optional memory policy used to decide whether auxiliary-model speculation should run.
    /// Pass `.recommendedWorkingSet` to fall back to regular generation when
    /// the combined main and draft model parameters exceed the recommended
    /// working set.
    public let memoryPolicy: SpeculativeDecodingMemoryPolicy?

    public init(
        draftModel: ModelContainer,
        numDraftTokens: Int = 5,
        memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil
    ) {
        self.draftModelSource = .loaded(draftModel)
        self.numDraftTokens = numDraftTokens
        self.memoryPolicy = memoryPolicy
    }

    /// Initialize speculative decoding with a draft model loader.
    ///
    /// When a memory policy is present, `draftModelBytes` lets `ChatSession`
    /// decide whether to use speculative decoding before it loads the draft
    /// model. This is the preferred initializer when the draft model may not fit
    /// comfortably beside the main model.
    ///
    /// - Parameters:
    ///   - draftModelBytes: estimated resident parameter bytes for the draft model
    ///   - numDraftTokens: number of tokens proposed by the draft model per verification cycle
    ///   - memoryPolicy: optional memory policy used before loading the draft model
    ///   - loadDraftModel: closure that loads the draft model only if speculation is admitted
    public init(
        draftModelBytes: Int,
        numDraftTokens: Int = 5,
        memoryPolicy: SpeculativeDecodingMemoryPolicy? = nil,
        loadDraftModel: @escaping @Sendable () async throws -> ModelContainer
    ) {
        self.draftModelSource = .deferred(bytes: max(0, draftModelBytes), loadDraftModel)
        self.numDraftTokens = numDraftTokens
        self.memoryPolicy = memoryPolicy
    }

    package var estimatedDraftModelBytes: Int? {
        guard case .deferred(let bytes, _) = draftModelSource else {
            return nil
        }
        return bytes
    }

    package func loadDraftModel() async throws -> ModelContainer {
        switch draftModelSource {
        case .loaded(let draftModel):
            draftModel
        case .deferred(_, let load):
            try await load()
        }
    }
}

/// A prompt cache loaded and validated through ``ChatSession/loadCache(from:validating:)``.
///
/// The cache arrays remain compatible with ``loadPromptCache(url:)``. `metadata`
/// contains only caller-provided entries; the reserved format-version entry is
/// exposed separately as `formatVersion`.
public struct ChatSessionCacheSnapshot {
    public let cache: [KVCache]
    public let metadata: [String: String]
    public let formatVersion: Int

    package init(cache: [KVCache], metadata: [String: String], formatVersion: Int) {
        self.cache = cache
        self.metadata = metadata
        self.formatVersion = formatVersion
    }
}

/// Simplified API for multi-turn conversations with LLMs and VLMs.
///
/// For example:
///
/// ```swift
/// let modelContainer = try await loadModelContainer(id: "mlx-community/Qwen3-4B-4bit")
/// let session = ChatSession(modelContainer)
/// print(try await session.respond(to: "What are two things to see in San Francisco?"))
/// print(try await session.respond(to: "How about a great place to eat?"))
/// ```
///
/// To enable speculative decoding for faster generation, pass a `SpeculativeDecodingConfig`:
///
/// ```swift
/// let draft = try await LLMModelFactory.shared.loadContainer(configuration: draftConfig)
/// let session = ChatSession(
///     modelContainer,
///     speculativeDecoding: SpeculativeDecodingConfig(draftModel: draft)
/// )
/// ```
///
/// - Note: `ChatSession` is not thread-safe. Each session should be used from a single
///   task/thread at a time. The underlying `ModelContainer` handles thread safety for
///   model operations.
/// - Note: A session retains its structured transcript, including referenced media,
///   until ``clear()`` is called. Long-running VLM sessions should clear or replace
///   the session when that retained context is no longer needed.
public final class ChatSession {

    private struct Conversation {
        var messages: [Chat.Message]
        /// Exact tokens represented by the main KV cache when its count
        /// matches the model-cache progress. An empty value paired with
        /// nonzero progress is an invalidated ledger and forces a rebuild.
        var cachedTokens: [Int]

        /// Tokens returned by the previous generation but not yet represented
        /// by `cachedTokens`. This is normally empty; a speculative round can
        /// leave its final verifier sample here.
        var uncommittedTokens: [Int] = []

        @discardableResult
        mutating func record(
            _ assistant: AssistantGeneration,
            generatedTokens: [Int],
            processedTokenCount: Int
        ) -> Bool {
            guard assistant.shouldRecord else {
                // A cancelled or semantically empty generation has no
                // assistant turn to replay. Its cache may nevertheless
                // contain generated or lookahead tokens, so invalidate
                // the ledger and rebuild from the retained messages.
                cachedTokens.removeAll()
                uncommittedTokens.removeAll()
                return false
            }

            let cachedTokenCount = cachedTokens.count
            if processedTokenCount >= cachedTokenCount,
                processedTokenCount - cachedTokenCount <= generatedTokens.count
            {
                let committedGeneratedTokenCount = processedTokenCount - cachedTokenCount
                cachedTokens.append(
                    contentsOf: generatedTokens.prefix(committedGeneratedTokenCount))
                uncommittedTokens = Array(
                    generatedTokens.dropFirst(committedGeneratedTokenCount))
            } else {
                // The iterator/cache relationship could not be proven
                // (for example, a speculative iterator retained
                // un-emitted lookahead).
                cachedTokens.removeAll()
                uncommittedTokens.removeAll()
            }

            messages.append(
                .assistant(
                    assistant.content,
                    toolCalls: assistant.toolCalls.isEmpty ? nil : assistant.toolCalls))
            return true
        }
    }

    private struct GenerationRun {
        let stream: AsyncStream<Generation>
        let task: Task<[Int], Never>

        init(_ pair: (AsyncStream<Generation>, Task<[Int], Never>)) {
            (stream, task) = pair
        }
    }

    private struct AssistantGeneration {
        var content = ""
        var toolCalls: [ToolCall] = []
        var rejectedToolCalls: [RejectedToolCall] = []
        var stopReason: GenerateStopReason?
        var wasTerminatedByConsumer = false

        var shouldRecord: Bool {
            (!content.isEmpty || !toolCalls.isEmpty)
                && rejectedToolCalls.isEmpty
                && !wasTerminatedByConsumer
                && stopReason != .cancelled
        }

        mutating func consume(_ item: Generation) {
            if let chunk = item.chunk {
                content += chunk
            }
            if let toolCall = item.toolCall {
                toolCalls.append(toolCall)
            }
            if let rejection = item.rejectedToolCall {
                rejectedToolCalls.append(rejection)
            }
            if let info = item.info {
                stopReason = info.stopReason
            }
        }
    }

    /// Current version of the high-level `ChatSession` prompt-cache contract.
    ///
    /// The low-level safetensors representation owned by
    /// ``savePromptCache(url:cache:metadata:)`` is unchanged.
    public static let cacheFormatVersion = 1

    /// Metadata key reserved for ``cacheFormatVersion``.
    public static let cacheFormatVersionMetadataKey =
        "mlx-swift-lm.chat-session-cache.format-version"

    /// All mutable continuation state for one realized model cache.
    ///
    /// `KVCacheStorage` provides shared ownership of the cache array because
    /// dynamic compression replaces array elements. Conversation state lives
    /// at the same boundary so cache progress and its exact token ledger cannot
    /// be accidentally separated when the session is carried across turns.
    private struct RealizedCache {
        let main: KVCacheStorage
        var draft: KVCacheStorage?
        var state: LMOutput.State?
        var conversation: Conversation?

        init(
            cache: consuming [KVCache],
            draft: consuming [KVCache]? = nil,
            state: LMOutput.State? = nil,
            conversation: Conversation? = nil,
            plan: KVCachePlan
        ) {
            self.main = KVCacheStorage(cache, plan: plan)
            self.draft = draft.map { KVCacheStorage($0, plan: plan) }
            self.state = state
            self.conversation = conversation
        }

        init(
            main: KVCacheStorage,
            draft: KVCacheStorage? = nil,
            state: LMOutput.State? = nil,
            conversation: Conversation? = nil
        ) {
            self.main = main
            self.draft = draft
            self.state = state
            self.conversation = conversation
        }

        func requirePlan(_ requested: KVCachePlan) throws {
            guard main.plan == requested,
                draft.map({ $0.plan == requested }) ?? true
            else {
                throw ChatSessionError.kvCacheConfigurationChanged(
                    previous: main.plan.configuration,
                    requested: requested.configuration)
            }
        }
    }

    private enum Cache {
        /// `state` is the per-call model state (e.g. M-RoPE rope deltas)
        /// from the last prefill against this cache. It must survive across
        /// turns: without it, a model that anchors positions on carried
        /// state re-derives them from a cold start on the next turn.
        case empty
        case kvcache(RealizedCache)
        case history([Chat.Message])
    }

    private let model: ModelContainer
    public var instructions: String?
    private let cache: SerialAccessContainer<Cache>
    private let loadedDraftModel: SerialAccessContainer<ModelContainer?>
    private let loadedMTPDrafter: SerialAccessContainer<MTPDrafterContainer?>
    public var processing: UserInput.Processing
    public var generateParameters: GenerateParameters
    /// Optional wired-memory ticket applied to generation work for the next turn.
    ///
    /// `ChatSession` captures this value when a turn starts and forwards that
    /// snapshot to both regular and speculative generation tasks. Because a
    /// session is not thread-safe, callers may replace the ticket between turns.
    public var wiredMemoryTicket: WiredMemoryTicket?

    /// Optional behavioral components (e.g. a custom ``LogitProcessor``) applied
    /// on each generation. The `logitProcessorFactory` is invoked fresh for
    /// every ``respond(to:role:images:videos:audios:)`` so stateful processors
    /// do not leak state across turns.
    public var components: GenerationComponents
    public var additionalContext: [String: any Sendable]?
    public var tools: [ToolSpec]?
    public var toolDispatch: (@Sendable (ToolCall) async throws -> String)?

    /// Speculative decoding configuration, nil if disabled.
    public let speculativeDecoding: SpeculativeDecodingConfig?

    /// MTP speculative-decoding configuration, nil if disabled.
    ///
    /// This is mutually exclusive with ``speculativeDecoding``. Use an
    /// initializer whose required label is `mtpSpeculativeDecoding:` to opt in.
    public private(set) var mtpSpeculativeDecoding: MTPSpeculativeDecodingConfig?

    /// Unified KV / recurrent-cache status for this session.
    ///
    /// If the session owns a realized cache, this reports its actual request,
    /// topology, strategy state, and progress. Otherwise it reports the cache
    /// planned by the current ``generateParameters``.
    public func cacheStatus() async throws -> KVCacheStatus {
        let parameters = generateParameters
        if let realized = await cache.read({ cache -> KVCacheStatus? in
            guard case .kvcache(let stored) = cache else { return nil }
            return KVCacheStatus(
                cache: stored.main.cache,
                plan: stored.main.plan,
                phase: .realized,
                processedTokenCount: stored.main.processedTokenCount)
        }) {
            return realized
        }
        return try await model.cacheStatus(parameters: parameters)
    }

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.empty)
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.loadedMTPDrafter = .init(nil)
        self.processing = processing
        self.generateParameters = generateParameters
        self.wiredMemoryTicket = nil
        self.components = components
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
        self.mtpSpeculativeDecoding = nil
    }

    /// Initialize the `ChatSession`.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.empty)
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.loadedMTPDrafter = .init(nil)
        self.processing = processing
        self.generateParameters = generateParameters
        self.wiredMemoryTicket = nil
        self.components = components
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
        self.mtpSpeculativeDecoding = nil
    }

    /// Initialize the `ChatSession` with an existing message history.
    ///
    /// This enables "Prompt Re-hydration" for persistent chat applications.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session
    ///   - history: The full array of messages to restore (including system prompt)
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        history: consuming [Chat.Message],
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(.history(history))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.loadedMTPDrafter = .init(nil)
        self.processing = processing
        self.generateParameters = generateParameters
        self.wiredMemoryTicket = nil
        self.components = components
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
        self.mtpSpeculativeDecoding = nil
    }

    /// Initialize the `ChatSession` with an existing message history.
    ///
    /// This enables "Prompt Re-hydration" for persistent chat applications.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session
    ///   - history: The full array of messages to restore (including system prompt)
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        history: [Chat.Message],
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(.history(history))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.loadedMTPDrafter = .init(nil)
        self.processing = processing
        self.generateParameters = generateParameters
        self.wiredMemoryTicket = nil
        self.components = components
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
        self.mtpSpeculativeDecoding = nil
    }

    /// Initialize the `ChatSession` with a pre-built KV cache.
    ///
    /// This enables prefix caching: build a KV cache from a long shared context (e.g. a
    /// system prompt and document) once, save it via ``saveCache(to:)``, and restore it
    /// across multiple sessions to avoid re-prefilling the same tokens each time.
    ///
    /// > Important: If the cache was built from a session that already included system
    /// > instructions, do not pass the same `instructions` here — they would be
    /// > re-tokenized on each call to ``respond(to:role:images:videos:audios:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// > Important: Prefer the initializer that takes a `promptCache`. A cache and the model state
    /// > that belongs with it must travel together: passing `cache: snapshot.cache` while dropping
    /// > `snapshot.state` compiles, but positions later tokens as if the cached prefix contained no
    /// > images. Models that carry a position anchor throw ``ContinuationStateError`` rather than
    /// > continue without it.
    ///
    /// > Important: A raw cache does not carry the structured chat transcript.
    /// > This initializer therefore preserves fragment-based continuation behavior.
    /// > Use a history initializer when later turns must re-render the full conversation.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContainer``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]`, matching the given model — from
    ///     ``loadPromptCacheSnapshot(url:)`` if it was saved to disk
    ///   - state: the model state captured with that cache; pass `snapshot.state` whenever the
    ///     cache came from a snapshot
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContainer,
        instructions: String? = nil,
        cache: consuming [KVCache],
        state: LMOutput.State? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = model
        self.instructions = instructions
        self.cache = .init(
            .kvcache(
                .init(
                    cache: cache,
                    state: state,
                    plan: (try? generateParameters.kvCachePlan()) ?? .disabled)))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.loadedMTPDrafter = .init(nil)
        self.processing = processing
        self.generateParameters = generateParameters
        self.wiredMemoryTicket = nil
        self.components = components
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
        self.mtpSpeculativeDecoding = nil
    }

    /// Initialize the `ChatSession` with a pre-built KV cache.
    ///
    /// This enables prefix caching: build a KV cache from a long shared context (e.g. a
    /// system prompt and document) once, save it via ``saveCache(to:)``, and restore it
    /// across multiple sessions to avoid re-prefilling the same tokens each time.
    ///
    /// > Important: If the cache was built from a session that already included system
    /// > instructions, do not pass the same `instructions` here — they would be
    /// > re-tokenized on each call to ``respond(to:role:images:videos:audios:)`` without matching
    /// > KV state, producing incoherent output.
    ///
    /// > Important: Prefer the initializer that takes a `promptCache`. A cache and the model state
    /// > that belongs with it must travel together: passing `cache: snapshot.cache` while dropping
    /// > `snapshot.state` compiles, but positions later tokens as if the cached prefix contained no
    /// > images. Models that carry a position anchor throw ``ContinuationStateError`` rather than
    /// > continue without it.
    ///
    /// > Important: A raw cache does not carry the structured chat transcript.
    /// > This initializer therefore preserves fragment-based continuation behavior.
    /// > Use a history initializer when later turns must re-render the full conversation.
    ///
    /// - Parameters:
    ///   - model: the ``ModelContext``
    ///   - instructions: optional system instructions for the session — leave `nil` if the
    ///     cache already encodes a system prompt
    ///   - cache: a non-empty `[KVCache]`, matching the given model — from
    ///     ``loadPromptCacheSnapshot(url:)`` if it was saved to disk
    ///   - state: the model state captured with that cache; pass `snapshot.state` whenever the
    ///     cache came from a snapshot
    ///   - speculativeDecoding: optional speculative decoding configuration for faster generation
    ///   - generateParameters: parameters that control generation
    ///   - components: optional behavioral components, e.g. a custom ``LogitProcessor``
    ///   - processing: media processing configuration for images/videos
    ///   - tools: optional tool specifications
    ///   - toolDispatch: optional tool dispatch -- required for toolcalls if streaming strings rather than details
    ///   - additionalContext: optional model-specific context
    public init(
        _ model: ModelContext,
        instructions: String? = nil,
        cache: consuming [KVCache],
        state: LMOutput.State? = nil,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.model = ModelContainer(context: model)
        self.instructions = instructions
        self.cache = .init(
            .kvcache(
                .init(
                    cache: cache,
                    state: state,
                    plan: (try? generateParameters.kvCachePlan()) ?? .disabled)))
        self.loadedDraftModel = .init(speculativeDecoding?.draftModel)
        self.loadedMTPDrafter = .init(nil)
        self.processing = processing
        self.generateParameters = generateParameters
        self.wiredMemoryTicket = nil
        self.components = components
        self.tools = tools
        self.toolDispatch = toolDispatch
        self.additionalContext = additionalContext
        self.speculativeDecoding = speculativeDecoding
        self.mtpSpeculativeDecoding = nil
    }

    // MARK: MTP initializers

    /// Initializes a session with high-level MTP speculative decoding.
    ///
    /// The required `mtpSpeculativeDecoding` label keeps this overload
    /// separate from the historical no-config and classic-speculative APIs.
    public convenience init(
        _ model: ModelContainer,
        instructions: String? = nil,
        mtpSpeculativeDecoding: MTPSpeculativeDecodingConfig,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.init(
            model,
            instructions: instructions,
            generateParameters: generateParameters,
            components: components,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: toolDispatch
        )
        self.mtpSpeculativeDecoding = mtpSpeculativeDecoding
    }

    /// Initializes a context-backed session with high-level MTP speculative decoding.
    public convenience init(
        _ model: ModelContext,
        instructions: String? = nil,
        mtpSpeculativeDecoding: MTPSpeculativeDecodingConfig,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.init(
            model,
            instructions: instructions,
            generateParameters: generateParameters,
            components: components,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: toolDispatch
        )
        self.mtpSpeculativeDecoding = mtpSpeculativeDecoding
    }

    /// Initializes a rehydrated-history session with MTP speculative decoding.
    public convenience init(
        _ model: ModelContainer,
        instructions: String? = nil,
        history: consuming [Chat.Message],
        mtpSpeculativeDecoding: MTPSpeculativeDecodingConfig,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.init(
            model,
            instructions: instructions,
            history: history,
            generateParameters: generateParameters,
            components: components,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: toolDispatch
        )
        self.mtpSpeculativeDecoding = mtpSpeculativeDecoding
    }

    /// Initializes a context-backed rehydrated-history session with MTP.
    public convenience init(
        _ model: ModelContext,
        instructions: String? = nil,
        history: [Chat.Message],
        mtpSpeculativeDecoding: MTPSpeculativeDecodingConfig,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.init(
            model,
            instructions: instructions,
            history: history,
            generateParameters: generateParameters,
            components: components,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: toolDispatch
        )
        self.mtpSpeculativeDecoding = mtpSpeculativeDecoding
    }

    /// Initializes a prompt-cache-backed session with MTP speculative decoding.
    public convenience init(
        _ model: ModelContainer,
        instructions: String? = nil,
        cache: consuming [KVCache],
        mtpSpeculativeDecoding: MTPSpeculativeDecodingConfig,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.init(
            model,
            instructions: instructions,
            cache: cache,
            generateParameters: generateParameters,
            components: components,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: toolDispatch
        )
        self.mtpSpeculativeDecoding = mtpSpeculativeDecoding
    }

    /// Initializes a context-backed prompt-cache session with MTP.
    public convenience init(
        _ model: ModelContext,
        instructions: String? = nil,
        cache: consuming [KVCache],
        mtpSpeculativeDecoding: MTPSpeculativeDecodingConfig,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.init(
            model,
            instructions: instructions,
            cache: cache,
            generateParameters: generateParameters,
            components: components,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: toolDispatch
        )
        self.mtpSpeculativeDecoding = mtpSpeculativeDecoding
    }

    /// Initialize the `ChatSession` with a prompt cache and the model state that belongs with it.
    ///
    /// This is the preferred way to start a session from a saved cache: the snapshot keeps the KV
    /// arrays and the model state together, so neither can be dropped. Read one from disk with
    /// ``loadPromptCacheSnapshot(url:)``, or build one in process with
    /// ``PromptCacheSnapshot/init(cache:metadata:state:)``.
    ///
    /// Models that do not carry model state simply ignore it, so this works for any model.
    ///
    /// > Important: A snapshot carries the cache and model state, not the structured chat
    /// > transcript, so this initializer uses fragment-based continuation. A later
    /// > image-bearing turn continues warm from the cached prefix rather than rebuilding and
    /// > re-rendering earlier messages. Positions stay correct, but the prompt differs — and on
    /// > vision encoders that attend across image boundaries (Qwen2-VL today) so do the new
    /// > image's features, since it is encoded alone rather than beside the cached one.
    /// > Use a history initializer when later turns must re-render the full conversation.
    public convenience init(
        _ model: ModelContainer,
        instructions: String? = nil,
        promptCache: consuming PromptCacheSnapshot,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.init(
            model,
            instructions: instructions,
            cache: promptCache.cache,
            state: promptCache.state,
            speculativeDecoding: speculativeDecoding,
            generateParameters: generateParameters,
            components: components,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: toolDispatch)
    }

    /// Initialize the `ChatSession` with a prompt cache and the model state that belongs with it.
    ///
    /// See ``init(_:instructions:promptCache:speculativeDecoding:generateParameters:components:processing:additionalContext:tools:toolDispatch:)-(ModelContainer,_,_,_,_,_,_,_,_,_)``
    /// for the continuation behavior a snapshot implies.
    public convenience init(
        _ model: ModelContext,
        instructions: String? = nil,
        promptCache: consuming PromptCacheSnapshot,
        speculativeDecoding: SpeculativeDecodingConfig? = nil,
        generateParameters: GenerateParameters = .init(),
        components: GenerationComponents = .init(),
        processing: UserInput.Processing = .init(resize: CGSize(width: 512, height: 512)),
        additionalContext: [String: any Sendable]? = nil,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (ToolCall) async throws -> String)? = nil
    ) {
        self.init(
            ModelContainer(context: model),
            instructions: instructions,
            promptCache: promptCache,
            speculativeDecoding: speculativeDecoding,
            generateParameters: generateParameters,
            components: components,
            processing: processing,
            additionalContext: additionalContext,
            tools: tools,
            toolDispatch: toolDispatch)
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: the model's response
    nonisolated(nonsending)
        public func respond(
            to prompt: String,
            role: Chat.Message.Role = .user,
            images: consuming [UserInput.Image],
            videos: consuming [UserInput.Video],
            audios: consuming [UserInput.Audio]
        ) async throws -> String
    {
        var output = ""
        for try await chunk in streamResponse(
            to: prompt, role: role, images: images, videos: videos, audios: audios
        ) {
            output += chunk
        }
        return output
    }

    /// Produces a response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    ///   - audio: optional audio (for use with VLMs)
    /// - Returns: the model's response
    nonisolated(nonsending)
        public func respond(
            to prompt: String,
            role: Chat.Message.Role = .user,
            image: consuming UserInput.Image? = nil,
            video: consuming UserInput.Video? = nil,
            audio: consuming UserInput.Audio? = nil
        ) async throws -> String
    {
        try await respond(
            to: prompt,
            role: role,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? [],
            audios: audio.map { [$0] } ?? []
        )
    }

    /// Produces a response after appending a batch of structured chat messages.
    ///
    /// Use this to continue an existing session with non-user roles, such as one
    /// or more tool results, while preserving the session's KV cache.
    ///
    /// - Important: Initializing a new session from history must prefill that
    ///   history once. Reuse the same session with this method for subsequent
    ///   tool or agent turns to avoid repeatedly pre-filling the accumulated
    ///   transcript.
    ///
    /// - Parameter messages: chat messages to append before generation
    /// - Returns: the model's response
    nonisolated(nonsending)
        public func respond(
            to messages: consuming [Chat.Message]
        ) async throws -> String
    {
        var output = ""
        for try await chunk in streamResponse(to: messages) {
            output += chunk
        }
        return output
    }

    /// Prepares a prompt and commits it to the session cache without generating
    /// or emitting response tokens.
    ///
    /// This is useful for building reusable prompt caches for long system
    /// instructions or RAG context. The prompt is processed with the same media,
    /// tools, and additional context configuration as a normal turn.
    ///
    /// - Parameters:
    ///   - prompt: prompt to add to the cached context
    ///   - role: message role (defaults to `.user`)
    ///   - images: images associated with the prompt
    ///   - videos: videos associated with the prompt
    ///   - audios: audio inputs associated with the prompt
    public func prefill(
        _ prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image] = [],
        videos: consuming [UserInput.Video] = [],
        audios: consuming [UserInput.Audio] = []
    ) async throws {
        try await prefill(messages: [
            .init(role: role, content: prompt, images: images, videos: videos, audios: audios)
        ])
    }

    /// Prepares structured chat messages and commits them to the session cache
    /// without generating or emitting response tokens.
    ///
    /// The operation is transactional with respect to the session: if preparation
    /// throws or is cancelled, the previously committed cache remains available.
    ///
    /// - Parameter messages: messages to add to the cached context
    public func prefill(messages: consuming [Chat.Message]) async throws {
        let inputMessages = SendableBox<[Chat.Message]>(messages)

        // ChatSession is intentionally single-consumer. Snapshot all mutable
        // per-turn configuration before the first suspension so callers may
        // replace it safely before a later turn.
        let instructions = instructions
        let processing = processing
        let tools = tools
        let additionalContext = additionalContext
        let generateParameters = generateParameters
        let wiredMemoryTicket = wiredMemoryTicket
        let modelContainer = model
        let cache = cache
        let speculativeDecoding = speculativeDecoding
        let loadedDraftModel = loadedDraftModel

        let operation = {
            try await cache.update { cache in
                try Task.checkCancellation()

                let processor = await modelContainer.processor
                let model = await modelContainer.perform { context in
                    SendableBox(context.model)
                }.consume()

                let appendedMessages = inputMessages.consume()
                var preparedMessages: [Chat.Message] = []
                if let instructions {
                    preparedMessages.append(.system(instructions))
                }

                var kvCache: [KVCache]
                let storedDraftCache: [KVCache]?
                var state: LMOutput.State?
                var conversation: Conversation?
                switch cache {
                case .empty:
                    kvCache = try model.newCache(parameters: generateParameters)
                    storedDraftCache = nil
                    state = nil
                    conversation = Conversation(messages: [], cachedTokens: [])

                case .kvcache(let stored):
                    kvCache = stored.main.cache.map { $0.copy() }
                    storedDraftCache = stored.draft?.cache.map { $0.copy() }
                    state = stored.state
                    conversation = stored.conversation

                case .history(let history):
                    kvCache = try model.newCache(parameters: generateParameters)
                    storedDraftCache = nil
                    state = nil
                    conversation = Conversation(messages: history, cachedTokens: [])
                }

                if let conversation {
                    preparedMessages.append(contentsOf: conversation.messages)
                }
                preparedMessages.append(contentsOf: appendedMessages)
                let userInput = UserInput(
                    chat: preparedMessages,
                    processing: processing,
                    tools: tools,
                    additionalContext: additionalContext
                )
                let preparedInput = try await processor.prepare(input: userInput)
                try Task.checkCancellation()

                var input = preparedInput
                var reusedCachedPrefix = false
                if var currentConversation = conversation {
                    let fullTokenIds = preparedInput.text.tokens.asArray(Int.self)
                    let cachedTokenIds = currentConversation.cachedTokens
                    let cachesAreAligned =
                        !kvCache.isEmpty
                        && kvCache.allSatisfy { $0.offset == cachedTokenIds.count }
                    let containsNewMedia = appendedMessages.contains {
                        !$0.images.isEmpty || !$0.videos.isEmpty || !$0.audios.isEmpty
                    }
                    let hasPreparedMedia =
                        preparedInput.image != nil || preparedInput.video != nil
                        || preparedInput.audio != nil
                    let canReusePrefix =
                        cachesAreAligned
                        && fullTokenIds.starts(with: cachedTokenIds)
                        && fullTokenIds.count > cachedTokenIds.count
                        && !containsNewMedia
                        && !hasPreparedMedia
                        && preparedInput.text.mask == nil

                    if canReusePrefix {
                        input = LMInput(
                            tokens: MLXArray(fullTokenIds.dropFirst(cachedTokenIds.count)))
                        reusedCachedPrefix = true
                    } else if kvCache.first?.offset != 0 {
                        kvCache = try model.newCache(parameters: generateParameters)
                        state = nil
                    }

                    currentConversation.messages.append(contentsOf: appendedMessages)
                    currentConversation.cachedTokens = fullTokenIds
                    conversation = currentConversation
                }

                let nextState = try Self.prefill(
                    input: input,
                    model: model,
                    cache: kvCache,
                    state: state,
                    parameters: generateParameters
                )

                var nextDraftCache: [KVCache]?
                if let draftContainer = try await Self.draftContainerForPrefill(
                    mainModel: model,
                    speculativeDecoding: speculativeDecoding,
                    loadedDraftModel: loadedDraftModel
                ) {
                    let draftModel = await draftContainer.perform { context in
                        SendableBox(context.model)
                    }.consume()
                    let canReuseDraftCache =
                        reusedCachedPrefix
                        && storedDraftCache?.allSatisfy {
                            $0.offset
                                == preparedInput.text.tokens.size - input.text.tokens.size
                        } == true
                    let draftCache: [KVCache]
                    let draftInput: LMInput
                    if canReuseDraftCache, let storedDraftCache {
                        draftCache = storedDraftCache
                        draftInput = input
                    } else {
                        draftCache = try draftModel.newCache(parameters: generateParameters)
                        draftInput = preparedInput
                    }
                    _ = try Self.prefill(
                        input: draftInput,
                        model: draftModel,
                        cache: draftCache,
                        state: nil,
                        parameters: generateParameters
                    )
                    nextDraftCache = draftCache
                }

                try Task.checkCancellation()
                cache = .kvcache(
                    .init(
                        cache: kvCache, draft: nextDraftCache, state: nextState,
                        conversation: conversation,
                        plan: try generateParameters.kvCachePlan()))
            }
        }

        if let wiredMemoryTicket {
            try await WiredMemoryTicket.withWiredLimit(wiredMemoryTicket, operation)
        } else {
            try await operation()
        }
    }

    private static func prefill(
        input: LMInput,
        model: any LanguageModel,
        cache: [KVCache],
        state: LMOutput.State?,
        parameters: GenerateParameters
    ) throws -> LMOutput.State? {
        try Task.checkCancellation()

        let output: LMOutput
        switch try model.prepare(
            input,
            cache: cache,
            state: state,
            windowSize: parameters.prefillStepSize
        ) {
        case .tokens(let remaining):
            guard remaining.tokens.size > 0 else {
                throw ChatSessionCacheError.emptyPrefillInput
            }
            output = withPreparedCache(cache, lengths: remaining.sequenceLengths) {
                model(remaining[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
            }

        case .logits(let preparedOutput):
            output = preparedOutput
        }

        try Task.checkCancellation()
        eval(output.logits)
        eval(cache)
        return output.state
    }

    private static func draftContainerForPrefill(
        mainModel: any LanguageModel,
        speculativeDecoding: SpeculativeDecodingConfig?,
        loadedDraftModel: SerialAccessContainer<ModelContainer?>
    ) async throws -> ModelContainer? {
        guard let speculativeDecoding else { return nil }

        if let memoryPolicy = speculativeDecoding.memoryPolicy,
            let draftModelBytes = speculativeDecoding.estimatedDraftModelBytes
        {
            let evaluation = memoryPolicy.evaluate(
                mainModelBytes: SpeculativeDecodingMemoryPolicy.modelWeightBytes(mainModel),
                draftModelBytes: draftModelBytes
            )
            if !evaluation.shouldUseSpeculativeDecoding {
                if evaluation.action == .fail {
                    throw SpeculativeDecodingMemoryError(evaluation: evaluation)
                }
                return nil
            }
        }

        let cachedDraftContainer = await loadedDraftModel.read { $0 }
        let draftContainer: ModelContainer
        if let cachedDraftContainer {
            draftContainer = cachedDraftContainer
        } else {
            draftContainer = try await speculativeDecoding.loadDraftModel()
        }
        let draftModel = await draftContainer.perform { context in
            SendableBox(context.model)
        }.consume()

        if let evaluation = speculativeDecoding.memoryPolicy?.evaluate(
            mainModel: mainModel,
            draftModel: draftModel
        ), !evaluation.shouldUseSpeculativeDecoding {
            if evaluation.action == .fail {
                throw SpeculativeDecodingMemoryError(evaluation: evaluation)
            }
            return nil
        }

        if cachedDraftContainer == nil {
            await loadedDraftModel.update { storedDraftModel in
                if storedDraftModel == nil {
                    storedDraftModel = draftContainer
                }
            }
        }
        return draftContainer
    }

    private static func modelWeightBytes(_ model: any BaseLanguageModel) -> Int {
        model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
    }

    private static func mtpMemoryFallbackReason(
        _ evaluation: SpeculativeDecodingMemoryEvaluation
    ) -> String {
        if let limit = evaluation.limitBytes {
            return
                "MTP skipped by memory policy: estimated \(evaluation.estimatedBytes) bytes exceeds \(limit) bytes"
        }
        return "MTP skipped by memory policy"
    }

    private static func mtpMemoryAdmissionFallbackReason(
        policy: SpeculativeDecodingMemoryPolicy?,
        mainModelBytes: Int,
        drafterBytes: Int?
    ) throws -> String? {
        guard let policy, let drafterBytes else { return nil }
        let evaluation = policy.evaluate(
            mainModelBytes: mainModelBytes,
            draftModelBytes: drafterBytes
        )
        guard !evaluation.shouldUseSpeculativeDecoding else { return nil }
        if evaluation.action == .fail {
            throw SpeculativeDecodingMemoryError(evaluation: evaluation)
        }
        return mtpMemoryFallbackReason(evaluation)
    }

    /// Produces a streaming response to a prompt as Strings.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    /// - Throws: ``RejectedToolCallError`` if the model emits a rejected tool
    ///   call, or an error produced during generation.
    public func streamResponse(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image] = [],
        videos: consuming [UserInput.Video] = [],
        audios: consuming [UserInput.Audio] = []
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(
            to: prompt, role: role, images: images, videos: videos, audios: audios,
            failOnRejectedToolCall: true
        ) { $0.chunk }
    }

    /// Produces a streaming response after appending a batch of structured chat messages.
    ///
    /// Use this to continue an existing session with non-user roles, such as one
    /// or more tool results, while preserving the session's KV cache.
    ///
    /// - Parameter messages: chat messages to append before generation
    /// - Returns: a stream of string chunks from the model
    /// - Throws: ``RejectedToolCallError`` if the model emits a rejected tool
    ///   call, or an error produced during generation.
    public func streamResponse(
        to messages: consuming [Chat.Message]
    ) -> AsyncThrowingStream<String, Error> {
        streamMap(messages: messages, failOnRejectedToolCall: true) { $0.chunk }
    }

    /// Produces a streaming response to a prompt as `Generation`.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - role: the message role (defaults to `.user`)
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: a stream of `Generation` from the model
    ///
    /// Rejected calls are emitted as ``Generation/rejectedToolCall(_:)``. When
    /// automatic tool dispatch is configured, the stream instead throws
    /// ``RejectedToolCallError`` before dispatching any call from that turn.
    public func streamDetails(
        to prompt: String,
        role: Chat.Message.Role = .user,
        images: consuming [UserInput.Image] = [],
        videos: consuming [UserInput.Video] = [],
        audios: consuming [UserInput.Audio] = [],
    ) -> AsyncThrowingStream<Generation, Error> {
        streamMap(to: prompt, role: role, images: images, videos: videos, audios: audios) {
            $0
        }
    }

    /// Produces a streaming response after appending a batch of structured chat messages as `Generation`.
    ///
    /// Use this to continue an existing session with non-user roles, such as one
    /// or more tool results, while preserving the session's KV cache.
    ///
    /// - Parameter messages: chat messages to append before generation
    /// - Returns: a stream of `Generation` from the model
    ///
    /// Rejected calls are emitted as ``Generation/rejectedToolCall(_:)``. When
    /// automatic tool dispatch is configured, the stream instead throws
    /// ``RejectedToolCallError`` before dispatching any call from that turn.
    public func streamDetails(
        to messages: consuming [Chat.Message]
    ) -> AsyncThrowingStream<Generation, Error> {
        streamMap(messages: messages) {
            $0
        }
    }

    /// Produces a streaming response to a prompt by transforming the
    /// raw `Generation` values.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - images: list of images (for use with VLMs)
    ///   - videos: list of videos (for use with VLMs)
    ///   - audios: list of audios (for use with VLMs)
    /// - Returns: a stream of transformed values from the model
    private func streamMap<R: Sendable>(
        to prompt: String,
        role: Chat.Message.Role,
        images: consuming [UserInput.Image] = [],
        videos: consuming [UserInput.Video] = [],
        audios: consuming [UserInput.Audio] = [],
        failOnRejectedToolCall: Bool = false,
        transform: @Sendable @escaping (Generation) -> R?
    ) -> AsyncThrowingStream<R, Error> {
        streamMap(
            messages: [
                .init(role: role, content: prompt, images: images, videos: videos, audios: audios)
            ],
            failOnRejectedToolCall: failOnRejectedToolCall,
            transform: transform
        )
    }

    private func streamMap<R: Sendable>(
        messages: consuming [Chat.Message],
        failOnRejectedToolCall: Bool = false,
        transform: @Sendable @escaping (Generation) -> R?
    ) -> AsyncThrowingStream<R, Error> {
        let (stream, continuation) = AsyncThrowingStream<R, Error>.makeStream()

        // images and videos are not Sendable (MLXArray) but they are consumed
        // and are only being sent to the inner async
        let inputMessages = SendableBox<[Chat.Message]>(messages)
        let model = self.model
        let instructions = self.instructions
        let processing = self.processing
        let tools = self.tools
        let toolDispatch = self.toolDispatch
        let additionalContext = self.additionalContext
        let cache = self.cache
        let loadedDraftModel = self.loadedDraftModel
        let generateParameters = self.generateParameters
        let components = self.components
        let speculativeDecoding = self.speculativeDecoding
        let loadedMTPDrafter = self.loadedMTPDrafter
        let mtpSpeculativeDecoding = self.mtpSpeculativeDecoding
        let wiredMemoryTicket = self.wiredMemoryTicket

        // Keep the generation body in an explicitly Sendable operation. Swift 6's
        // region checker cannot analyze the two-stage MTP memory gate when that
        // body is written directly as the Task closure.
        let operation: @Sendable () async -> Void = {
            do {
                try await cache.update { cache in

                    // these are all Sendable
                    let processor = await model.processor
                    let tokenizer = await model.tokenizer
                    let modelConfiguration = await model.configuration

                    var leadingMessages: [Chat.Message] = []
                    if let instructions {
                        leadingMessages.append(.system(instructions))
                    }

                    // prepare the cache, if needed.  note:
                    // this is using the LanguageModel (not Sendable) outside
                    // the protective lock.  Assuming the weights are not
                    // being mutated behind the scenes, this will obey the MLXArray
                    // contract that they be evaluated if used across threads.
                    // This is internal to the implementation and this technique
                    // should not be used in calling code.
                    //
                    // The benefit is that callers can be running multiple
                    // ChatSessions in parallel, as long as the instances
                    // are distinct.  In particular the KVCache cannot
                    // be shared and that is the lock that is held here.

                    let model = await model.perform { context in
                        SendableBox(context.model)
                    }.consume()
                    let kvCachePlan = try generateParameters.kvCachePlan()

                    var kvCache: KVCacheStorage
                    var draftKVCache: KVCacheStorage?
                    // Per-call model state (e.g. M-RoPE rope deltas) carried
                    // across turns alongside the KV cache; updated after each
                    // prefill and stored back at the end of the turn.
                    var lmState: LMOutput.State?
                    var conversation: Conversation?
                    switch cache {
                    case .empty:
                        kvCache = KVCacheStorage(
                            try model.newCache(parameters: generateParameters), plan: kvCachePlan)
                        conversation = Conversation(messages: [], cachedTokens: [])
                        cache = .kvcache(
                            .init(main: kvCache, conversation: conversation))

                    case .kvcache(let stored):
                        let realizedStatus = KVCacheStatus(
                            cache: stored.main.cache,
                            plan: stored.main.plan,
                            phase: .realized,
                            processedTokenCount: stored.main.processedTokenCount)
                        let desiredStatus = try model.cacheStatus(
                            parameters: generateParameters)

                        // The plan captures the typed/legacy strategy and capacity; the
                        // realized layer kinds additionally catch layout changes the plan
                        // alone cannot see.
                        if var restored = stored.conversation,
                            stored.main.plan != kvCachePlan
                                || realizedStatus.layers.map(\.kind)
                                    != desiredStatus.layers.map(\.kind)
                        {
                            // Cache-affecting generation parameters changed. Because a
                            // structured transcript is retained, rebuild the cache from it
                            // rather than silently continuing with the old cache policy.
                            restored.cachedTokens.removeAll()
                            restored.uncommittedTokens.removeAll()
                            kvCache = KVCacheStorage(
                                try model.newCache(parameters: generateParameters),
                                plan: kvCachePlan)
                            draftKVCache = nil
                            lmState = nil
                            conversation = restored
                        } else {
                            // Either the realized policy is unchanged, or this is a
                            // restored raw cache with no transcript to rebuild from.
                            // In the latter case, requirePlan rejects a changed policy.
                            try stored.requirePlan(kvCachePlan)
                            kvCache = stored.main
                            draftKVCache = stored.draft
                            lmState = stored.state
                            conversation = stored.conversation
                        }

                    case .history(let history):
                        // the KVCache is represented by a chat history
                        kvCache = KVCacheStorage(
                            try model.newCache(parameters: generateParameters), plan: kvCachePlan)
                        conversation = Conversation(messages: history, cachedTokens: [])
                        cache = .kvcache(
                            .init(main: kvCache, conversation: conversation))
                    }

                    var pendingMessages = inputMessages.consume()

                    // How this model's generated token stream relates to a chat
                    // template re-render is a property of its response protocol,
                    // not of the session.
                    let promptCachePolicy = PromptCacheReusePolicy(
                        protocolRules: modelConfiguration.toolCallFormat?
                            .promptCacheReuseRules(tokenizer: tokenizer) ?? [])

                    // loop can restart on tool calls
                    restart: while !pendingMessages.isEmpty {
                        let isToolResultContinuation =
                            pendingMessages.contains { $0.role == .tool }
                            && conversation?.messages.last?.tool?.calls?.isEmpty == false
                        let templateMessages: [Chat.Message]
                        let conversationMessageCountBeforePending: Int?
                        if var currentConversation = conversation {
                            conversationMessageCountBeforePending =
                                currentConversation.messages.count
                            currentConversation.messages.append(contentsOf: pendingMessages)
                            templateMessages = leadingMessages + currentConversation.messages
                            conversation = currentConversation
                        } else {
                            conversationMessageCountBeforePending = nil
                            // A cache restored without its transcript cannot be reconciled
                            // against a complete chat template. Preserve the legacy prefix
                            // continuation behavior for this explicitly low-level API.
                            templateMessages = leadingMessages + pendingMessages
                        }
                        let containsNewMedia = pendingMessages.contains {
                            !$0.images.isEmpty || !$0.videos.isEmpty || !$0.audios.isEmpty
                        }
                        let structuredToolCallCount = modelConfiguration.toolCallFormat?
                            .promptCacheStructuredToolCallCount(in: templateMessages)

                        let userInput = UserInput(
                            chat: templateMessages,
                            processing: processing,
                            tools: tools, additionalContext: additionalContext)
                        let preparedInput = try await processor.prepare(input: userInput)
                        var input = preparedInput
                        pendingMessages.removeAll()

                        let speculativeMemoryEvaluation: SpeculativeDecodingMemoryEvaluation?
                        if let speculativeDecoding,
                            let memoryPolicy = speculativeDecoding.memoryPolicy,
                            let draftModelBytes = speculativeDecoding.estimatedDraftModelBytes
                        {
                            speculativeMemoryEvaluation = memoryPolicy.evaluate(
                                mainModelBytes:
                                    SpeculativeDecodingMemoryPolicy.modelWeightBytes(model),
                                draftModelBytes: draftModelBytes)
                        } else {
                            speculativeMemoryEvaluation = nil
                        }
                        let willFallBackBeforeLoadingDraft =
                            speculativeMemoryEvaluation.map {
                                !$0.shouldUseSpeculativeDecoding && $0.action != .fail
                            } ?? false

                        var reusedMainCacheWithoutDraft = false
                        var requiresMainOnlyContinuation = false
                        // Prompt tokens this turn does not prefill because the cache
                        // already represents them. Reported to the caller on `.info`.
                        var cachedPromptTokenCount = 0
                        // Read off the prepared input, not `input`: the latter may be narrowed to
                        // a token-only suffix below, which would hide media the model still sees.
                        let carriesPreparedMedia =
                            preparedInput.image != nil || preparedInput.video != nil
                            || preparedInput.audio != nil
                        if var currentConversation = conversation {
                            let promptTokenIds = input.text.tokens.asArray(Int.self)
                            let cachedTokenIds = currentConversation.cachedTokens
                            assert(
                                kvCache.nativeAttentionOffsetsAreAligned,
                                "Main attention cache offsets diverged from model-cache progress")
                            let mainCacheIsAligned =
                                kvCache.processedTokenCount == cachedTokenIds.count
                            let draftCacheIsAligned: Bool
                            if let draftKVCache {
                                assert(
                                    draftKVCache.nativeAttentionOffsetsAreAligned,
                                    "Draft attention cache offsets diverged from model-cache progress"
                                )
                                draftCacheIsAligned =
                                    draftKVCache.processedTokenCount == cachedTokenIds.count
                            } else {
                                // Tentatively reuse the main cache. If speculative
                                // decoding is admitted below, both caches are rebuilt
                                // from `preparedInput`; a fallback can keep this suffix.
                                draftCacheIsAligned = true
                            }

                            let turn = PromptCacheTurn(
                                promptTokens: promptTokenIds,
                                carriesNewMedia: containsNewMedia,
                                carriesPreparedMedia: carriesPreparedMedia,
                                carriesAttentionMask: input.text.mask != nil,
                                carriesModelState: lmState != nil,
                                isToolResultContinuation: isToolResultContinuation,
                                previousGenerationUncommittedTokens:
                                    currentConversation.uncommittedTokens,
                                structuredToolCallCount: structuredToolCallCount,
                                usesSpeculativeDecoding: speculativeDecoding != nil)
                            let cacheState = PromptCacheState(
                                cachedTokens: cachedTokenIds,
                                processedTokenCount: kvCache.processedTokenCount,
                                mainCacheIsAligned: mainCacheIsAligned,
                                hasDraftCache: draftKVCache != nil,
                                draftCacheIsAligned: draftCacheIsAligned,
                                isTrimmable: canTrimPromptCache(kvCache.cache)
                                    && (draftKVCache.map { canTrimPromptCache($0.cache) } ?? true))

                            var decision = promptCachePolicy.decide(turn: turn, cache: cacheState)

                            // Rewinding is the one decision that can fail while being
                            // applied: a cache may trim fewer tokens than requested.
                            // Verify and downgrade to a rebuild before prefilling.
                            if case .trimToCommonPrefix(let commonPrefixLength, let trimCount) =
                                decision
                            {
                                let mainTrimmed = kvCache.trim(trimCount)
                                let draftTrimmed = draftKVCache.map { $0.trim(trimCount) }
                                let mainTrimIsAligned =
                                    mainTrimmed == trimCount
                                    && kvCache.processedTokenCount == commonPrefixLength
                                let draftTrimIsAligned =
                                    draftKVCache.map { draftCache in
                                        draftTrimmed == trimCount
                                            && draftCache.processedTokenCount == commonPrefixLength
                                    } ?? true

                                if !(mainTrimIsAligned && draftTrimIsAligned) {
                                    decision = .rebuild
                                }
                            }

                            switch decision {
                            case .prefillAll:
                                break

                            case .appendSuffix(let suffixStart, _):
                                input = LMInput(
                                    tokens: MLXArray(Array(promptTokenIds[suffixStart...])))
                                cachedPromptTokenCount = suffixStart

                            case .appendSuffixToMain(let suffixStart, _):
                                input = LMInput(
                                    tokens: MLXArray(Array(promptTokenIds[suffixStart...])))
                                cachedPromptTokenCount = suffixStart
                                // The draft does not represent the same private
                                // Harmony path. Preserve the authoritative main
                                // cache and use it alone for this continuation.
                                draftKVCache = nil
                                requiresMainOnlyContinuation = true

                            case .trimToCommonPrefix(let commonPrefixLength, _):
                                input = LMInput(
                                    tokens: MLXArray(
                                        Array(promptTokenIds.dropFirst(commonPrefixLength))))
                                cachedPromptTokenCount = commonPrefixLength

                            case .rebuild:
                                kvCache = KVCacheStorage(
                                    try model.newCache(parameters: generateParameters),
                                    plan: kvCachePlan)
                                draftKVCache = nil
                                lmState = nil
                            }

                            reusedMainCacheWithoutDraft =
                                decision.reusesCachedPrefix
                                && speculativeDecoding != nil
                                && !willFallBackBeforeLoadingDraft
                                && draftKVCache == nil

                            // Usually the rendered prompt, but a protocol rule may
                            // keep generated tokens the cold render cannot reproduce.
                            switch decision {
                            case .appendSuffix(_, let representedTokens),
                                .appendSuffixToMain(_, let representedTokens):
                                currentConversation.cachedTokens = representedTokens
                            case .prefillAll, .trimToCommonPrefix, .rebuild:
                                currentConversation.cachedTokens = promptTokenIds
                            }
                            currentConversation.uncommittedTokens.removeAll()
                            conversation = currentConversation
                        }

                        guard input.text.tokens.size > 0 else {
                            throw ChatSessionError.emptyPreparedInput
                        }

                        // Select the token iterator based on speculative decoding configuration.
                        let generation: GenerationRun
                        var generationStateSink: ChatSessionStateSink?
                        func defaultGeneration(
                            mtpFallbackReason: String? = nil
                        ) throws -> GenerationRun {
                            // Seed the iterator with the carried state; read
                            // back the post-prefill state (prefill runs in the
                            // iterator's init, and the rope delta does not
                            // change during decode) so the next turn — or the
                            // next tool restart — anchors correctly.
                            let iterator = try TokenIterator(
                                input: input, model: model, cacheStorage: kvCache,
                                state: lmState,
                                parameters: generateParameters, components: components)
                            lmState = iterator.state

                            if let mtpFallbackReason {
                                let stateSink = ChatSessionStateSink()
                                generationStateSink = stateSink
                                return GenerationRun(
                                    MLXLMCommon.generateTaskRecordingTokens(
                                        promptTokenCount: input.text.tokens.size,
                                        modelConfiguration: modelConfiguration,
                                        tokenizer: tokenizer,
                                        iterator: MTPFallbackTokenIterator(
                                            base: iterator,
                                            reason: mtpFallbackReason,
                                            sink: stateSink
                                        ),
                                        wiredMemoryTicket: wiredMemoryTicket,
                                        tools: tools
                                    ))
                            } else {
                                return GenerationRun(
                                    MLXLMCommon.generateTaskRecordingTokens(
                                        promptTokenCount: input.text.tokens.size,
                                        modelConfiguration: modelConfiguration,
                                        tokenizer: tokenizer,
                                        iterator: iterator,
                                        wiredMemoryTicket: wiredMemoryTicket,
                                        tools: tools
                                    ))
                            }
                        }

                        // Carried model state is safe to speculate on: the anchors position from
                        // the cache offset, which a rejected proposal rewinds along with the KV
                        // rows. Media is not — the draft would have to prefill the same images.
                        //
                        // A warm main cache with no draft cache is recoverable only when
                        // `reusedMainCacheWithoutDraft` says both can be rebuilt from the full
                        // input below. Without that ledger — a session restored from a snapshot —
                        // there is nothing to re-prefill the draft from, and handing mismatched
                        // caches to the iterator would throw rather than fall back.
                        let draftCacheIsRecoverable =
                            draftKVCache != nil || kvCache.processedTokenCount == 0
                            || reusedMainCacheWithoutDraft
                        if carriesPreparedMedia || !draftCacheIsRecoverable {
                            // Drop the draft cache as `.appendSuffixToMain` does: retaining it
                            // would leave it at an offset the main cache never visits.
                            draftKVCache = nil
                            requiresMainOnlyContinuation = true
                        }

                        if let mtpSpeculativeDecoding {
                            let mainModelBytes =
                                SpeculativeDecodingMemoryPolicy.modelWeightBytes(model)
                            let preLoadFallbackReason =
                                try Self.mtpMemoryAdmissionFallbackReason(
                                    policy: mtpSpeculativeDecoding.memoryPolicy,
                                    mainModelBytes: mainModelBytes,
                                    drafterBytes: mtpSpeculativeDecoding.estimatedDrafterBytes
                                ) ?? ""
                            if !preLoadFallbackReason.isEmpty {
                                generation = try defaultGeneration(
                                    mtpFallbackReason: preLoadFallbackReason
                                )
                            } else {
                                let cachedDrafter = await loadedMTPDrafter.read { $0 }
                                let drafterContainer =
                                    if let cachedDrafter {
                                        cachedDrafter
                                    } else {
                                        try await mtpSpeculativeDecoding.loadDrafter()
                                    }
                                let drafter = await drafterContainer.perform { context in
                                    SendableBox(context.model)
                                }.consume()

                                let postLoadFallbackReason =
                                    try Self.mtpMemoryAdmissionFallbackReason(
                                        policy: mtpSpeculativeDecoding.memoryPolicy,
                                        mainModelBytes: mainModelBytes,
                                        drafterBytes: Self.modelWeightBytes(drafter)
                                    ) ?? ""
                                if !postLoadFallbackReason.isEmpty {
                                    generation = try defaultGeneration(
                                        mtpFallbackReason: postLoadFallbackReason
                                    )
                                } else {
                                    if cachedDrafter == nil {
                                        await loadedMTPDrafter.update { storedDrafter in
                                            if storedDrafter == nil {
                                                storedDrafter = drafterContainer
                                            }
                                        }
                                    }

                                    let iterator = try MTPSpeculativeTokenIterator(
                                        input: input,
                                        mainModel: model,
                                        drafter: drafter,
                                        mainCacheStorage: kvCache,
                                        mainState: lmState,
                                        parameters: generateParameters,
                                        blockSize: mtpSpeculativeDecoding.blockSize,
                                        components: components
                                    )
                                    lmState = iterator.mainState
                                    let stateSink = ChatSessionStateSink()
                                    generationStateSink = stateSink

                                    generation = GenerationRun(
                                        MLXLMCommon.generateTaskRecordingTokens(
                                            promptTokenCount: input.text.tokens.size,
                                            modelConfiguration: modelConfiguration,
                                            tokenizer: tokenizer,
                                            iterator: StateReportingMTPTokenIterator(
                                                base: iterator,
                                                sink: stateSink
                                            ),
                                            wiredMemoryTicket: wiredMemoryTicket,
                                            tools: tools
                                        )
                                    )
                                }
                            }
                        } else if speculativeDecoding != nil, requiresMainOnlyContinuation {
                            generation = try defaultGeneration()
                        } else if let speculativeDecoding {
                            var shouldFallBackBeforeLoadingDraft = false
                            if let memoryEvaluation = speculativeMemoryEvaluation {
                                if !memoryEvaluation.shouldUseSpeculativeDecoding {
                                    if memoryEvaluation.action == .fail {
                                        throw SpeculativeDecodingMemoryError(
                                            evaluation: memoryEvaluation)
                                    }

                                    shouldFallBackBeforeLoadingDraft = true
                                }
                            }

                            if shouldFallBackBeforeLoadingDraft {
                                generation = try defaultGeneration()
                            } else {
                                let cachedDraftContainer = await loadedDraftModel.read { $0 }
                                let draftContainer: ModelContainer
                                if let cachedDraftContainer {
                                    draftContainer = cachedDraftContainer
                                } else {
                                    draftContainer = try await speculativeDecoding.loadDraftModel()
                                }

                                let draftModel = await draftContainer.perform { context in
                                    SendableBox(context.model)
                                }.consume()
                                let memoryEvaluation = speculativeDecoding.memoryPolicy?.evaluate(
                                    mainModel: model,
                                    draftModel: draftModel)

                                if let memoryEvaluation,
                                    !memoryEvaluation.shouldUseSpeculativeDecoding
                                {
                                    if memoryEvaluation.action == .fail {
                                        throw SpeculativeDecodingMemoryError(
                                            evaluation: memoryEvaluation)
                                    }

                                    generation = try defaultGeneration()
                                } else {
                                    if cachedDraftContainer == nil {
                                        await loadedDraftModel.update { storedDraftModel in
                                            if storedDraftModel == nil {
                                                storedDraftModel = draftContainer
                                            }
                                        }
                                    }

                                    if reusedMainCacheWithoutDraft {
                                        // The main cache took a suffix-only path, but
                                        // speculation was admitted without a matching
                                        // draft cache. Rebuild both from the full input.
                                        kvCache = KVCacheStorage(
                                            try model.newCache(parameters: generateParameters),
                                            plan: kvCachePlan)
                                        draftKVCache = nil
                                        lmState = nil
                                        input = preparedInput
                                        cachedPromptTokenCount = 0
                                    }

                                    // Allocate the draft KV cache once and reuse it across turns,
                                    // exactly like the main model's KV cache.
                                    if draftKVCache == nil {
                                        draftKVCache = KVCacheStorage(
                                            try draftModel.newCache(
                                                parameters: generateParameters),
                                            plan: kvCachePlan)
                                        cache = .kvcache(
                                            .init(
                                                main: kvCache,
                                                draft: draftKVCache,
                                                state: lmState,
                                                conversation: conversation))
                                    }

                                    let iterator = try SpeculativeTokenIterator(
                                        input: input,
                                        mainModel: model,
                                        draftModel: draftModel,
                                        mainCacheStorage: kvCache,
                                        draftCacheStorage: draftKVCache!,
                                        mainState: lmState,
                                        parameters: generateParameters,
                                        numDraftTokens: speculativeDecoding.numDraftTokens,
                                        components: components
                                    )
                                    // Carry the post-prefill state, as the standard path does,
                                    // so the next turn keeps its anchor.
                                    lmState = iterator.state
                                    let stateSink = ChatSessionStateSink()
                                    generationStateSink = stateSink

                                    generation = GenerationRun(
                                        MLXLMCommon.generateTaskRecordingTokens(
                                            promptTokenCount: input.text.tokens.size,
                                            modelConfiguration: modelConfiguration,
                                            tokenizer: tokenizer,
                                            iterator: StateReportingSpeculativeTokenIterator(
                                                base: iterator,
                                                sink: stateSink
                                            ),
                                            wiredMemoryTicket: wiredMemoryTicket,
                                            tools: tools))
                                }
                            }
                        } else {
                            // Standard path for stateful, media, or unsynchronized cache input.
                            generation = try defaultGeneration()
                        }

                        var pendingToolCalls: [ToolCall] = []
                        var assistant = AssistantGeneration()

                        for await item in generation.stream {
                            let item = item.attributingCachedPromptTokens(cachedPromptTokenCount)
                            assistant.consume(item)

                            // collect tool calls for dispatch; if no
                            // toolDispatch the caller handles them via
                            // the transform (streamDetails path)
                            if let toolCall = item.toolCall, toolDispatch != nil {
                                pendingToolCalls.append(toolCall)
                            } else if let value = transform(item) {
                                if case .terminated = continuation.yield(value) {
                                    assistant.wasTerminatedByConsumer = true
                                    generation.task.cancel()
                                    break
                                }
                            }
                        }

                        // The generation task is unstructured, so cancellation of
                        // this task (stream onTermination) does not propagate to
                        // it. Without an explicit cancel, awaiting its value
                        // would wait for the FULL generation while holding the
                        // cache lock — deadlocking the session's next call (e.g.
                        // a caller that cancels mid-stream and immediately asks
                        // again). The generate loop checks Task.isCancelled per
                        // token, so this stops it promptly.
                        if Task.isCancelled {
                            assistant.wasTerminatedByConsumer = true
                            generation.task.cancel()
                        }

                        // wait for the task to complete -- this is important in
                        // the case where we broke the loop early as the generation
                        // work may continue (briefly) and use the KVCache
                        let generatedTokens = await generation.task.value

                        if let generationStateSink {
                            lmState = generationStateSink.load()
                        }

                        if var currentConversation = conversation {
                            let recordedAssistant = currentConversation.record(
                                assistant,
                                generatedTokens: generatedTokens,
                                processedTokenCount: kvCache.processedTokenCount)
                            if !recordedAssistant,
                                let conversationMessageCountBeforePending
                            {
                                // A cancelled or empty generation did not commit an
                                // assistant turn. Roll back this restart's pending input
                                // so a later request cannot produce invalid role sequences
                                // such as user/user on strict chat templates.
                                currentConversation.messages.removeSubrange(
                                    conversationMessageCountBeforePending...)
                            }
                            conversation = currentConversation
                        }

                        if let rejection = assistant.rejectedToolCalls.first,
                            failOnRejectedToolCall || toolDispatch != nil
                        {
                            // The failed turn was rolled back above. Persist the
                            // invalidated token ledger before surfacing the error
                            // so the next request cannot reuse rejected output.
                            cache = .kvcache(
                                .init(
                                    main: kvCache,
                                    draft: draftKVCache,
                                    state: lmState,
                                    conversation: conversation))
                            throw RejectedToolCallError(rejection)
                        }

                        // dispatch all tool calls from this generation pass
                        if let toolDispatch, !pendingToolCalls.isEmpty,
                            !Task.isCancelled
                        {
                            if conversation == nil {
                                pendingMessages.append(
                                    .assistant("", toolCalls: pendingToolCalls))
                            }
                            for toolCall in pendingToolCalls {
                                let toolResult = try await toolDispatch(toolCall)
                                pendingMessages.append(
                                    .tool(
                                        toolResult, id: toolCall.id,
                                        name: toolCall.function.name))
                            }
                            continue restart
                        }
                    }

                    // Store the carried state back alongside the KV cache so
                    // the next turn resumes with correct position anchoring.
                    cache = .kvcache(
                        .init(
                            main: kvCache,
                            draft: draftKVCache,
                            state: lmState,
                            conversation: conversation))

                    continuation.finish()
                }
            } catch {
                continuation.finish(throwing: error)
            }
        }

        let task = Task {
            await operation()
        }

        continuation.onTermination = { _ in
            task.cancel()
        }

        return stream
    }

    /// Produces a streaming response to a prompt.
    ///
    /// - Parameters:
    ///   - prompt: the user prompt
    ///   - image: optional image (for use with VLMs)
    ///   - video: optional video (for use with VLMs)
    ///   - audio: optional audio (for use with VLMs)
    /// - Returns: a stream of string chunks from the model
    public func streamResponse(
        to prompt: String,
        image: consuming UserInput.Image? = nil,
        video: consuming UserInput.Video? = nil,
        audio: consuming UserInput.Audio? = nil
    ) -> AsyncThrowingStream<String, Error> {
        streamResponse(
            to: prompt,
            images: image.map { [$0] } ?? [],
            videos: video.map { [$0] } ?? [],
            audios: audio.map { [$0] } ?? [],
        )
    }

    /// Clear the session history and cache, preserving system instructions.
    public func clear() async {
        await cache.update { cache in
            cache = .empty
        }
    }

    /// Wait for exclusive access to the KVCache.
    ///
    /// This is useful for cases where a program is terminating and wants to ensure that any
    /// async operations are complete.
    public func synchronize() async {
        await cache.read { _ in }
    }

    /// Return the legacy runtime-only view of the configured KV-cache strategy.
    ///
    /// The report is `nil` until a typed or legacy cache configuration exists,
    /// or when the session currently stores history rather than a realized cache.
    @available(*, deprecated, message: "Use cacheStatus() for unified cache diagnostics.")
    public func kvCacheRuntimeReport() async throws -> KVCacheRuntimeReport? {
        let kvCachePlan = try generateParameters.kvCachePlan()
        return try await cache.read { cache in
            guard case .kvcache(let stored) = cache else { return nil }
            try stored.requirePlan(kvCachePlan)
            _ = try stored.main.plan.validated(stored.main.cache)
            return stored.main.plan.report(for: stored.main)
        }
    }

    /// Visit the current cache value, if realized as a `[KVCache]`.
    ///
    /// This method is meant for test support.
    func withCache<R: Sendable>(_ body: @Sendable ([KVCache]?) async throws -> R) async rethrows
        -> R?
    {
        try await cache.read { cache in
            switch cache {
            case .kvcache(let stored):
                return try await body(stored.main.cache)
            default:
                return try await body(nil)
            }
        }
    }

    /// Return model-wide cache progress for test support.
    func cacheProgress() async -> (main: Int, draft: Int?)? {
        await cache.read { cache in
            guard case .kvcache(let stored) = cache else { return nil }
            return (
                main: stored.main.processedTokenCount,
                draft: stored.draft?.processedTokenCount
            )
        }
    }

    /// Saves the current KV cache to disk.
    ///
    /// Use ``loadPromptCacheSnapshot(url:)`` and an initializer that accepts a
    /// `promptCache` to restore the saved cache and model state in a future session.
    ///
    /// > Important: This saves raw KV state only. It does not save the structured
    /// > transcript retained by `ChatSession`. A restored raw cache therefore uses
    /// > fragment-based continuation and is not suitable for conversation-aware
    /// > structured tool continuations. Persist the messages separately and restore
    /// > them with a history initializer when the transcript is required.
    ///
    /// - Parameter url: the file URL to write the cache to
    /// - Throws: ``ChatSessionError/noCacheAvailable`` if no generation or prefill has occurred,
    ///   or any error thrown by the underlying file write
    public func saveCache(to url: URL) async throws {
        try await saveCache(to: url, metadata: [:])
    }

    /// Saves the current KV cache and caller metadata to disk.
    ///
    /// The high-level cache format version is added automatically under
    /// ``cacheFormatVersionMetadataKey``. Caller metadata must not use that
    /// reserved key. Files written by this API remain readable through
    /// ``loadPromptCache(url:)``.
    ///
    /// - Parameters:
    ///   - url: file URL to write
    ///   - metadata: caller-defined metadata such as model revision or prompt fingerprint
    /// - Throws: ``ChatSessionError/noCacheAvailable`` if no generation or prefill has occurred,
    ///   ``ChatSessionCacheError/reservedCacheMetadataKey(_:)`` for a reserved key,
    ///   or any error thrown by the underlying file write
    public func saveCache(to url: URL, metadata: [String: String]) async throws {
        guard metadata[Self.cacheFormatVersionMetadataKey] == nil else {
            throw ChatSessionCacheError.reservedCacheMetadataKey(
                Self.cacheFormatVersionMetadataKey)
        }

        let persistedMetadata: [String: String] = {
            var metadata = metadata
            metadata[Self.cacheFormatVersionMetadataKey] = String(Self.cacheFormatVersion)
            return metadata
        }()

        try await cache.read { cache in
            switch cache {
            case .kvcache(let stored):
                try savePromptCache(
                    url: url, cache: stored.main.cache, metadata: persistedMetadata,
                    state: stored.state)
            default:
                throw ChatSessionError.noCacheAvailable
            }
        }
    }

    /// Loads a `ChatSession` cache, validates its format version, and optionally
    /// checks caller metadata before returning any cache arrays.
    ///
    /// Older or third-party prompt caches without high-level version metadata can
    /// still be loaded through ``loadPromptCache(url:)``.
    ///
    /// - Parameters:
    ///   - url: cache file URL
    ///   - expectedMetadata: caller metadata entries that must match exactly
    /// - Returns: validated cache arrays, user metadata, and format version
    public static func loadCache(
        from url: URL,
        validating expectedMetadata: [String: String] = [:]
    ) throws -> ChatSessionCacheSnapshot {
        guard expectedMetadata[cacheFormatVersionMetadataKey] == nil else {
            throw ChatSessionCacheError.reservedCacheMetadataKey(cacheFormatVersionMetadataKey)
        }

        let (cache, persistedMetadata) = try loadPromptCache(url: url)
        guard let rawVersion = persistedMetadata[cacheFormatVersionMetadataKey] else {
            throw ChatSessionCacheError.missingCacheFormatVersion
        }
        guard let version = Int(rawVersion) else {
            throw ChatSessionCacheError.invalidCacheFormatVersion(rawVersion)
        }
        guard version == cacheFormatVersion else {
            throw ChatSessionCacheError.unsupportedCacheFormatVersion(
                found: version,
                supported: cacheFormatVersion
            )
        }

        var userMetadata = persistedMetadata
        userMetadata.removeValue(forKey: cacheFormatVersionMetadataKey)
        for (key, expectedValue) in expectedMetadata {
            let actualValue = userMetadata[key]
            guard actualValue == expectedValue else {
                throw ChatSessionCacheError.cacheMetadataMismatch(
                    key: key,
                    expected: expectedValue,
                    actual: actualValue
                )
            }
        }

        return ChatSessionCacheSnapshot(
            cache: cache,
            metadata: userMetadata,
            formatVersion: version
        )
    }
}

/// Errors thrown by the original ``ChatSession/saveCache(to:)`` API.
public enum ChatSessionError: LocalizedError {
    /// ``ChatSession/saveCache(to:)`` was called before any generation occurred.
    case noCacheAvailable
    /// The processor produced no tokens for generation.
    case emptyPreparedInput
    /// The cache was realized under a different KV-cache configuration.
    case kvCacheConfigurationChanged(
        previous: KVCacheConfiguration?,
        requested: KVCacheConfiguration?
    )

    public var errorDescription: String? {
        switch self {
        case .noCacheAvailable:
            "No KV cache is available. Call respond() or streamResponse() before saveCache(to:)."
        case .emptyPreparedInput:
            "The chat template produced no uncached tokens for generation."
        case .kvCacheConfigurationChanged:
            "KV-cache configuration changed after the session cache was realized. Clear the session or respond() to rebuild the cache from its retained transcript."
        }
    }
}

/// Validation errors thrown by the additive prefill and versioned-cache APIs.
public enum ChatSessionCacheError: LocalizedError, Equatable, Hashable {
    /// The input processor produced no tokens to prefill.
    case emptyPrefillInput
    /// Caller metadata attempted to replace a key owned by `ChatSession`.
    case reservedCacheMetadataKey(String)
    /// A high-level cache does not declare its format version.
    case missingCacheFormatVersion
    /// A high-level cache declares a non-integer format version.
    case invalidCacheFormatVersion(String)
    /// A high-level cache uses a format version this package cannot read.
    case unsupportedCacheFormatVersion(found: Int, supported: Int)
    /// Persisted caller metadata does not match the expected value.
    case cacheMetadataMismatch(key: String, expected: String, actual: String?)

    public var errorDescription: String? {
        switch self {
        case .emptyPrefillInput:
            "The prepared prompt contains no tokens to prefill."
        case .reservedCacheMetadataKey(let key):
            "Cache metadata key '\(key)' is reserved by ChatSession."
        case .missingCacheFormatVersion:
            "The cache does not contain ChatSession format-version metadata."
        case .invalidCacheFormatVersion(let value):
            "The cache contains an invalid ChatSession format version: '\(value)'."
        case .unsupportedCacheFormatVersion(let found, let supported):
            "Unsupported ChatSession cache format version \(found); this build supports version \(supported)."
        case .cacheMetadataMismatch(let key, let expected, let actual):
            "Cache metadata mismatch for '\(key)': expected '\(expected)', found '\(actual ?? "<missing>")'."
        }
    }
}
