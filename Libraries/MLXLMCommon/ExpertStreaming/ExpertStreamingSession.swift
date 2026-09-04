// FORK(JuanColilla): R-56 expert streaming — configuration and activation.
//
// A model is built for streaming or for residency; there is no runtime flag on
// a live module. The selection happens during construction, and this is how
// the pieces a constructor needs (the offset index, the store, the banks) reach
// it: a scoped activation around model construction and weight loading.

import Foundation

public struct ExpertStreamingConfiguration: Sendable {
    /// Target size of the slot bank, summed over every expert family. The
    /// caller samples the process ceiling and passes bytes; nothing in this
    /// package writes a ceiling constant. How the bytes are shared between
    /// families is `ExpertStreamingSession.bankCapacitySplit`.
    public var bankCapacityBytes: Int
    /// An explicit per-family bank size that replaces the split of
    /// `bankCapacityBytes` when set. For experiments that need one bank
    /// small enough to evict and the other not; a host that has measured
    /// per-family hit rates can use it too. No floor is applied.
    public var bankCapacityBytesPerFamily: [ExpertFamily: Int]?
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
        prefetchQueueDepth: Int = 2,
        bankCapacityBytesPerFamily: [ExpertFamily: Int]? = nil
    ) {
        self.bankCapacityBytes = bankCapacityBytes
        self.bankCapacityBytesPerFamily = bankCapacityBytesPerFamily
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

/// One streamed step of a forward pass: a layer and the family it is
/// resolving. A MoVA layer has two steps, value first, then MLP.
public struct StreamedExpertStep: Hashable, Sendable, Comparable {
    public let layer: Int
    public let family: ExpertFamily

    public init(layer: Int, family: ExpertFamily) {
        self.layer = layer
        self.family = family
    }

    public static func < (lhs: StreamedExpertStep, rhs: StreamedExpertStep) -> Bool {
        (lhs.layer, lhs.family) < (rhs.layer, rhs.family)
    }
}

/// Everything a streamed model shares: one index, one store, one bank per
/// expert family.
public final class ExpertStreamingSession: @unchecked Sendable {
    public let configuration: ExpertStreamingConfiguration
    public let modelDirectory: URL
    public let store: ExpertResidencyStore
    /// One bank per family the index has. Never empty: the MLP family is
    /// mandatory.
    public let banks: [ExpertFamily: ExpertSlotBank]

    public var index: ExpertOffsetIndex { store.index }

    /// The MLP bank — the only one a single-family checkpoint has, and what
    /// the host's logging and statistics read.
    public var bank: ExpertSlotBank { banks[.mlp]! }

    public func bank(for family: ExpertFamily) -> ExpertSlotBank? { banks[family] }

    /// Hits, misses and evictions summed over every bank.
    public var bankStatistics: ExpertSlotBankStatistics {
        banks.values.reduce(ExpertSlotBankStatistics()) { $0 + $1.statistics }
    }

    private let countersLock = NSLock()
    private var countersStorage = ExpertStreamingSyncCounters()

    /// P6 prediction state. Touched only from the forward pass's own thread,
    /// which is the same single-pass invariant `SingleFlightGuard` enforces.
    private var previousTokenExperts: [StreamedExpertStep: [Int]] = [:]
    private var currentTokenExperts: [StreamedExpertStep: [Int]] = [:]
    private var previousStepOrder: [StreamedExpertStep] = []
    private var currentStepOrder: [StreamedExpertStep] = []
    private var lastStepSeen: StreamedExpertStep?
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
        _ = try index.requireFamily(.mlp)

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

        let split =
            configuration.bankCapacityBytesPerFamily
            ?? Self.bankCapacitySplit(capacityBytes: configuration.bankCapacityBytes, index: index)
        var banks = [ExpertFamily: ExpertSlotBank]()
        for family in index.families {
            let bank = ExpertSlotBank(
                store: store,
                family: family.family,
                configuration: .init(
                    capacityBytes: split[family.family] ?? 0,
                    maxLoadBatch: configuration.maxLoadBatch))
            bank.deferInstallEval = configuration.deferInstallEval
            banks[family.family] = bank
        }
        self.banks = banks
        self.prefetchEnabled = configuration.temporalPrefetch
    }

    // MARK: - Bank capacity

    /// The fewest slots any family's bank is given, whatever its share. A
    /// bank smaller than one token's routing choice cannot serve a decode
    /// step at all, and the session does not know the top-K: sixteen covers
    /// every routed family shipped so far (top-8 MLP, top-4 value) twice over.
    public static let minimumBankSlotsPerFamily = 16

    /// How `bankCapacityBytes` is shared between the families of `index`.
    ///
    /// A single-family checkpoint gets the whole capacity, unchanged from the
    /// one-bank design. With two families the bytes are split in proportion
    /// to each family's routed bytes — the fraction of the checkpoint each
    /// family keeps off the heap — so the value experts of K2-Horizon, 22% of
    /// its routed bytes, get 22% of the bank. That is a proportion, not a
    /// measurement: nothing here knows which family misses more, and a
    /// per-family hit rate is what would justify moving it.
    ///
    /// Every family is floored at ``minimumBankSlotsPerFamily`` so a small
    /// bank never starves the minority family below one decode step.
    public static func bankCapacitySplit(
        capacityBytes: Int, index: ExpertOffsetIndex
    ) -> [ExpertFamily: Int] {
        let families = index.families
        guard families.count > 1 else {
            return Dictionary(uniqueKeysWithValues: families.map { ($0.family, capacityBytes) })
        }
        let total = max(1, index.routedBytes)
        var split = [ExpertFamily: Int]()
        for family in families {
            let share = Double(family.routedBytes) / Double(total)
            let proportional = Int(Double(capacityBytes) * share)
            let floor = minimumBankSlotsPerFamily * family.bytesPerExpert
            split[family.family] = max(proportional, floor)
        }
        return split
    }

    /// Encourage the banks down a step. This is the "invisible release" a
    /// memory-pressure ladder wants: the model survives, the hit rate drops.
    ///
    /// Returns `false` if a forward pass held a bank and it was not resized;
    /// the caller retries between tokens. See `ExpertSlotBank.resize` for why
    /// this refuses instead of trapping. With two banks the resize is applied
    /// to each in turn and the result is whether every one accepted it.
    @discardableResult
    public func resizeBank(toCapacityBytes bytes: Int) -> Bool {
        let split = Self.bankCapacitySplit(capacityBytes: bytes, index: index)
        var resized = true
        for (family, bank) in banks {
            resized = bank.resize(to: split[family] ?? 0) && resized
        }
        return resized
    }

    // MARK: - Synchronization counters

    /// Router `eval`s plus install `eval`s, and the layers that produced them.
    public var syncCounters: ExpertStreamingSyncCounters {
        var counters = countersLock.withLock { countersStorage }
        counters.installEvals = bankStatistics.installEvals
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

    /// Reset every counter of the session in one call: banks, store and
    /// synchronizations. An A/B that resets two of the three and forgets the
    /// last one reports the previous arm's number.
    public func resetStatistics() {
        for bank in banks.values { bank.resetStatistics() }
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

    /// Record this step's choice and issue the prediction for the next one.
    ///
    /// The prediction: at step S of token N, read what step S+1 used at token
    /// N−1. The forward pass cannot predict the next step of the *current*
    /// token — its router depends on this step's output — so the previous
    /// token is the only signal available while the GPU is still busy.
    ///
    /// "The next step" comes from the order the previous token visited, not
    /// from `layer + 1`: with `mlp_only_layers` or a sparse step, the MoE
    /// layers are not consecutive and not necessarily starting at zero, and a
    /// MoVA layer contributes two steps (value, then MLP). At the last step of
    /// a token it wraps to the first, which is free and covers the boundary
    /// between tokens.
    func predictAfter(step: StreamedExpertStep, experts: [Int]) {
        // A step that did not move forward means a new token started.
        if let last = lastStepSeen, step <= last {
            previousTokenExperts = currentTokenExperts
            previousStepOrder = currentStepOrder
            currentTokenExperts = [:]
            currentStepOrder = []
        }
        lastStepSeen = step
        currentTokenExperts[step] = experts
        currentStepOrder.append(step)

        guard let position = previousStepOrder.firstIndex(of: step),
            !previousStepOrder.isEmpty
        else { return }
        let next = previousStepOrder[(position + 1) % previousStepOrder.count]
        guard let predicted = previousTokenExperts[next], let bank = banks[next.family]
        else { return }

        let keys = predicted.map {
            ExpertKey(family: next.family, layer: next.layer, expert: $0)
        }
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
        previousStepOrder = []
        currentStepOrder = []
        lastStepSeen = nil
    }

    // MARK: - Failure (TD-054(a))

    /// The failure that took this session out, or `nil` while it is healthy.
    ///
    /// Latched and read under a lock: a session never recovers, so the host
    /// can poll this between tokens and abort the generation cleanly without
    /// having to register a callback. The model then has to be unloaded.
    public var lastFailure: ExpertStreamingFailure? {
        failureLock.withLock { failureStorage }
    }

    @available(*, deprecated, renamed: "lastFailure")
    public var failure: ExpertStreamingFailure? { lastFailure }

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
        activate(session)
        defer { deactivate() }
        return try body()
    }

    /// The async form, for a load that has to `await`.
    ///
    /// Separate from the synchronous overload rather than replacing it: the
    /// activation is a *global*, not a task-local, so nothing about crossing a
    /// suspension point is free. See `activate(_:)` for what the caller owes.
    public static func withSession<R>(
        _ session: ExpertStreamingSession, _ body: () async throws -> R
    ) async rethrows -> R {
        activate(session)
        defer { deactivate() }
        return try await body()
    }

    /// Make `session` the one a model constructor will pick up, until
    /// `deactivate()`.
    ///
    /// For callers that cannot be expressed as a closure — a factory whose
    /// load is `async throws` and returns a container, typically. **Pair it
    /// with `defer { ExpertStreaming.deactivate() }` at the call site**: a
    /// load that throws between the two would otherwise leave the activation
    /// standing, and the next model built in the process would silently be
    /// constructed against a session that belongs to a load that failed.
    ///
    /// Two invariants the caller owns, because this is a process-wide global
    /// and not a task-local value:
    ///
    ///  * **One load at a time.** Two concurrent loads would interleave across
    ///    their suspension points and each could build against the other's
    ///    session. This is INV-MODEL-01, which the host already enforces; the
    ///    precondition below turns a violation into a crash at the load rather
    ///    than a wrong model at generation time.
    ///  * **Same process, any thread.** The storage is lock-protected, so
    ///    reading it from the construction path is safe wherever that runs.
    public static func activate(_ session: ExpertStreamingSession) {
        lock.withLock {
            precondition(
                Self.session == nil,
                "an expert streaming session is already active; loads must be serialized")
            Self.session = session
        }
    }

    /// End the activation. Idempotent, so a `defer` that runs after an early
    /// return or a thrown error is always correct.
    public static func deactivate() {
        lock.withLock { session = nil }
    }
}
