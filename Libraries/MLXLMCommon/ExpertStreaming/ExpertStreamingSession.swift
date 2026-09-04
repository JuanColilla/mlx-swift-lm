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
    /// P6; see `ExpertPrefetcher` for what is and is not predictable.
    public var temporalPrefetch: Bool
    /// Lanes the prefetch reads on. Two by default because P0 measured mobile
    /// NAND saturating around QD2: past that, speculative reads take
    /// bandwidth from the misses the forward pass is blocked on.
    public var prefetchQueueDepth: Int
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
    /// Let the next layer's router read absorb the install's scatter instead
    /// of forcing an `eval` per install. See `ExpertSlotBank.deferInstallEval`.
    ///
    /// **On by default**, which is a measured decision: it takes a streamed
    /// token from ~81 synchronizations to 41 — the router floor — and buys
    /// 9–19% of decode throughput depending on the bank size, with bit-for-bit
    /// identical output and no change in peak memory.
    public var deferInstallEval: Bool

    public init(
        bankCapacityBytes: Int,
        queueDepth: Int = 8,
        useNoCache: Bool = false,
        temporalPrefetch: Bool = false,
        admitOnSweep: Bool = false,
        maxLoadBatch: Int = 64,
        groupSize: Int = 64,
        bits: Int = 4,
        deferInstallEval: Bool = true,
        prefetchQueueDepth: Int = 2
    ) {
        self.bankCapacityBytes = bankCapacityBytes
        self.queueDepth = queueDepth
        self.useNoCache = useNoCache
        self.temporalPrefetch = temporalPrefetch
        self.admitOnSweep = admitOnSweep
        self.maxLoadBatch = maxLoadBatch
        self.groupSize = groupSize
        self.bits = bits
        self.deferInstallEval = deferInstallEval
        self.prefetchQueueDepth = prefetchQueueDepth
    }
}

/// How many times a run had to bring the GPU back to the host.
///
/// The Phase 1 measurement put 90% of the streamed model's loss here rather
/// than in I/O, and the count was inferred ("about 80 per token") instead of
/// measured. It is measured now: any change that claims to reduce
/// synchronization has to move these.
public struct ExpertStreamingSyncCounters: Sendable, Equatable {
    /// `eval` of the router's choice. One per streamed MoE layer, and
    /// irreducible: no `pread` can be issued before it.
    public var routerEvals: Int = 0
    /// `eval` forced to close a slot install. Removable — that is what
    /// `deferInstallEval` does.
    public var installEvals: Int = 0
    /// Streamed MoE layers entered, so the two counts above can be read per
    /// layer rather than per run.
    public var layerForwards: Int = 0

    public var total: Int { routerEvals + installEvals }
}

/// Everything a streamed model shares: one index, one store, one bank.
public final class ExpertStreamingSession: @unchecked Sendable {
    public let configuration: ExpertStreamingConfiguration
    public let modelDirectory: URL
    public let store: ExpertResidencyStore
    public let bank: ExpertSlotBank

    public var index: ExpertOffsetIndex { store.index }

    private let countersLock = NSLock()
    private var countersStorage = ExpertStreamingSyncCounters()

    /// P6 prediction state. Touched only from the forward pass's own thread,
    /// which is the same single-pass invariant `SingleFlightGuard` enforces.
    private var previousTokenExperts: [Int: [Int]] = [:]
    private var currentTokenExperts: [Int: [Int]] = [:]
    private var previousLayerOrder: [Int] = []
    private var currentLayerOrder: [Int] = []
    private var lastLayerSeen = Int.max
    private let prefetchLock = NSLock()
    private var prefetchEnabled = false

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
                useNoCache: configuration.useNoCache,
                prefetchQueueDepth: configuration.temporalPrefetch
                    ? configuration.prefetchQueueDepth : nil))
        self.bank = ExpertSlotBank(
            store: store,
            configuration: .init(
                capacityBytes: configuration.bankCapacityBytes,
                maxLoadBatch: configuration.maxLoadBatch))
        self.bank.deferInstallEval = configuration.deferInstallEval
        self.prefetchEnabled = configuration.temporalPrefetch
    }

    /// Encourage the bank down a step. This is the "invisible release" a
    /// memory-pressure ladder wants: the model survives, the hit rate drops.
    public func resizeBank(toCapacityBytes bytes: Int) {
        bank.resize(to: bytes)
    }

    // MARK: - Synchronization counters

    /// Router `eval`s plus install `eval`s, and the layers that produced them.
    public var syncCounters: ExpertStreamingSyncCounters {
        var counters = countersLock.withLock { countersStorage }
        counters.installEvals = bank.statistics.installEvals
        return counters
    }

    /// Called by a streamed MoE block right after it reads the router's choice
    /// back to the host. Public because the blocks live in `MLXLLM`.
    public func noteRouterEval() {
        countersLock.withLock {
            countersStorage.routerEvals += 1
            countersStorage.layerForwards += 1
        }
    }

    /// Reset every counter of the session in one call: bank, store and
    /// synchronizations. An A/B that resets two of the three and forgets the
    /// last one reports the previous arm's number.
    public func resetStatistics() {
        bank.resetStatistics()
        store.resetStatistics()
        countersLock.withLock { countersStorage = ExpertStreamingSyncCounters() }
    }

    // MARK: - Temporal prefetch (P6)

    /// Whether the prediction is running.
    ///
    /// Settable so an A/B can switch arms on one loaded model: comparing two
    /// processes is not comparable in this subsystem. Turning it off drops
    /// whatever is in flight; turning it on when the session was built without
    /// a prefetcher does nothing, because the lanes do not exist.
    public var isTemporalPrefetchEnabled: Bool {
        get { prefetchLock.withLock { prefetchEnabled } && store.isPrefetching }
        set {
            prefetchLock.withLock { prefetchEnabled = newValue }
            if !newValue { store.cancelPrefetch() }
        }
    }

    /// Record this layer's choice and issue the prediction for the next one.
    ///
    /// The prediction: at layer L of token N, read what layer L+1 used at token
    /// N−1. The forward pass cannot predict layer L+1 of the *current* token —
    /// its router depends on layer L's output — so the previous token is the
    /// only signal available while the GPU is still busy.
    ///
    /// "The next layer" comes from the order the previous token visited, not
    /// from `layer + 1`: with `mlp_only_layers` or a sparse step, the MoE
    /// layers are not consecutive and not necessarily starting at zero. At the
    /// last MoE layer of a token it wraps to the first, which is free and
    /// covers the boundary between tokens.
    func predictAfter(layer: Int, experts: [Int]) {
        // A layer index that did not move forward means a new token started.
        if layer <= lastLayerSeen {
            previousTokenExperts = currentTokenExperts
            previousLayerOrder = currentLayerOrder
            currentTokenExperts = [:]
            currentLayerOrder = []
        }
        lastLayerSeen = layer
        currentTokenExperts[layer] = experts
        currentLayerOrder.append(layer)

        guard let position = previousLayerOrder.firstIndex(of: layer),
            !previousLayerOrder.isEmpty
        else { return }
        let next = previousLayerOrder[(position + 1) % previousLayerOrder.count]
        guard let predicted = previousTokenExperts[next] else { return }

        let keys = predicted.map { ExpertKey(layer: next, expert: $0) }
        let absent = keys.filter { !bank.isResident($0) }
        store.notePrefetchPredicted(keys: keys.count)
        store.prefetch(keys: absent)
    }

    /// Forget the prediction. A new prompt, or anything that makes the
    /// previous token stop being a predictor of the next one.
    public func resetPrediction() {
        store.cancelPrefetch()
        previousTokenExperts = [:]
        currentTokenExperts = [:]
        previousLayerOrder = []
        currentLayerOrder = []
        lastLayerSeen = Int.max
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
