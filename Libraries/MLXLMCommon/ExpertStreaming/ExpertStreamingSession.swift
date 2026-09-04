// FORK(JuanColilla): R-56 expert streaming — configuration and activation.
//
// A model is built for streaming or for residency; there is no runtime flag on
// a live module. The selection happens during construction, and this is how
// the pieces a constructor needs (the offset index, the store, the bank) reach
// it: a scoped activation around model construction and weight loading.

import Foundation

public struct ExpertStreamingConfiguration: Sendable {
    /// Target size of the slot bank. The caller samples the process ceiling
    /// and passes bytes; nothing in this package writes a ceiling constant.
    public var bankCapacityBytes: Int
    /// Concurrent `pread` lanes.
    public var queueDepth: Int
    /// `F_NOCACHE`. Off by default: the kernel page cache is the free L2 tier.
    public var useNoCache: Bool
    /// Prefetch the previous token's experts while the GPU is still busy.
    /// Off until measured (P6).
    public var temporalPrefetch: Bool
    /// Whether the prefill sweep is allowed to install into the bank. Off by
    /// default: a sweep touches nearly every expert of a layer and would
    /// evict the decode working set on every turn.
    public var admitOnSweep: Bool
    /// Cap on experts read in one batch, so staging does not scale with the
    /// number of unique experts in a prefill.
    public var maxLoadBatch: Int
    /// Quantization of the routed experts, read from the checkpoint config.
    public var groupSize: Int
    public var bits: Int

    public init(
        bankCapacityBytes: Int,
        queueDepth: Int = 8,
        useNoCache: Bool = false,
        temporalPrefetch: Bool = false,
        admitOnSweep: Bool = false,
        maxLoadBatch: Int = 64,
        groupSize: Int = 64,
        bits: Int = 4
    ) {
        self.bankCapacityBytes = bankCapacityBytes
        self.queueDepth = queueDepth
        self.useNoCache = useNoCache
        self.temporalPrefetch = temporalPrefetch
        self.admitOnSweep = admitOnSweep
        self.maxLoadBatch = maxLoadBatch
        self.groupSize = groupSize
        self.bits = bits
    }
}

/// Everything a streamed model shares: one index, one store, one bank.
public final class ExpertStreamingSession: @unchecked Sendable {
    public let configuration: ExpertStreamingConfiguration
    public let modelDirectory: URL
    public let store: ExpertResidencyStore
    public let bank: ExpertSlotBank

    public var index: ExpertOffsetIndex { store.index }

    private let failureLock = NSLock()
    private var failureStorage: ExpertStreamingFailure?
    private var failureHandler: (@Sendable (ExpertStreamingFailure) -> Void)?

    public convenience init(
        modelDirectory: URL, configuration: ExpertStreamingConfiguration
    ) throws {
        try self.init(
            index: ExpertOffsetIndex.build(modelDirectory: modelDirectory),
            modelDirectory: modelDirectory,
            configuration: configuration)
    }

    public init(
        index: ExpertOffsetIndex,
        modelDirectory: URL,
        configuration: ExpertStreamingConfiguration
    ) throws {
        // Loud before anything is read: a mismatched `bits`/`groupSize` does
        // not fail, it decodes the weights wrong and generates fluent nonsense.
        try index.validateQuantization(
            groupSize: configuration.groupSize, bits: configuration.bits)

        self.configuration = configuration
        self.modelDirectory = modelDirectory
        self.store = try ExpertResidencyStore(
            index: index,
            modelDirectory: modelDirectory,
            configuration: .init(
                queueDepth: configuration.queueDepth,
                useNoCache: configuration.useNoCache))
        self.bank = ExpertSlotBank(
            store: store,
            configuration: .init(
                capacityBytes: configuration.bankCapacityBytes,
                maxLoadBatch: configuration.maxLoadBatch))
    }

    /// Encourage the bank down a step. This is the "invisible release" a
    /// memory-pressure ladder wants: the model survives, the hit rate drops.
    public func resizeBank(toCapacityBytes bytes: Int) {
        bank.resize(to: bytes)
    }

    // MARK: - Failure (TD-054(a))

    /// The failure that took this session out, or `nil` while it is healthy.
    ///
    /// Latched: a session never recovers. The model has to be unloaded.
    public var failure: ExpertStreamingFailure? {
        failureLock.withLock { failureStorage }
    }

    /// Register the host's cancellation route. Called at most once, on the
    /// forward pass's own thread, from inside `callAsFunction`.
    ///
    /// The handler must cancel the generation **and discard the turn's KV
    /// cache**: the degraded token is already in it. See the note at the top
    /// of `ExpertStreamingFailure.swift`.
    ///
    /// Registering after the session already failed calls the handler
    /// immediately, so a host that installs it late does not miss the event.
    public func onFailure(_ handler: @escaping @Sendable (ExpertStreamingFailure) -> Void) {
        let alreadyFailed: ExpertStreamingFailure? = failureLock.withLock {
            if let failure = failureStorage { return failure }
            failureHandler = handler
            return nil
        }
        if let alreadyFailed { handler(alreadyFailed) }
    }

    /// Latch a failure and notify the host exactly once.
    ///
    /// Returns the recorded failure, which is the *first* one seen — a later
    /// error on the same session does not overwrite the diagnosis.
    @discardableResult
    public func recordFailure(_ error: Error) -> ExpertStreamingFailure {
        let failure = ExpertStreamingFailure(error)
        let handler: (@Sendable (ExpertStreamingFailure) -> Void)? = failureLock.withLock {
            guard failureStorage == nil else { return nil }
            failureStorage = failure
            let handler = failureHandler
            failureHandler = nil
            return handler
        }
        // Outside the lock: the host's handler reaches into its own actors and
        // must not be able to deadlock against a `failure` read from here.
        handler?(failure)
        return failureLock.withLock { failureStorage ?? failure }
    }
}

/// Scoped activation of a streaming session.
///
/// Global rather than a parameter because the construction path that needs it
/// (`ModelFactory` → `Model.init(configuration)` → layer initializers) has no
/// seam to pass one through. The invariant that makes this safe is the same
/// one the host app already enforces: exactly one model loads at a time.
public enum ExpertStreaming {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var session: ExpertStreamingSession?

    public static var activeSession: ExpertStreamingSession? {
        lock.withLock { session }
    }

    /// Run `body` with `session` active. Nested activation is not supported and
    /// is a programming error.
    public static func withSession<R>(
        _ session: ExpertStreamingSession, _ body: () throws -> R
    ) rethrows -> R {
        lock.withLock {
            precondition(
                Self.session == nil,
                "an expert streaming session is already active; loads must be serialized")
            Self.session = session
        }
        defer { lock.withLock { Self.session = nil } }
        return try body()
    }
}
