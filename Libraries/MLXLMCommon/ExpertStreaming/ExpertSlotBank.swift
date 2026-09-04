// FORK(JuanColilla): R-56 expert streaming — the residency tier (L1).
//
// A fixed pool of expert slots shared by *every* layer. The 40 MoE layers of
// a Qwen 3.5 checkpoint have identical expert geometry, so one global bank
// lets a hot layer borrow capacity from a cold one; a per-layer bank cannot.
//
// The bank is deliberately NOT a module parameter. `loadWeights` verifies the
// checkpoint against the module tree with `verify: [.all]`, which fails both
// on unknown keys and on parameters the checkpoint does not provide — so a
// bank registered as a parameter would break the load in both directions at
// once. It is mutable streaming state, not a weight.

import Foundation
import MLX

public struct ExpertSlotBankStatistics: Sendable, Equatable {
    public var hits: Int = 0
    public var misses: Int = 0
    public var evictions: Int = 0
    /// Wall time of the install: the `pread` batch plus the scatter.
    ///
    /// With `deferInstallEval` on, the scatter is only *recorded* inside this
    /// window and executed later, so the number stops being comparable with
    /// the eager one. It measures the read in that mode, nothing more.
    public var installSeconds: Double = 0
    /// `eval` calls the install path forced. One of the two synchronization
    /// sources of a streamed token; the other is the router read, counted on
    /// the session. Zero when `deferInstallEval` is on.
    public var installEvals: Int = 0

    public var hitRate: Double {
        let total = hits + misses
        return total > 0 ? Double(hits) / Double(total) : 0
    }
}

public enum ExpertSlotBankError: Error, CustomStringConvertible {
    case requestExceedsCapacity(requested: Int, slots: Int)
    case noEvictableSlot(slots: Int)
    case concurrentForward

    public var description: String {
        switch self {
        case .requestExceedsCapacity(let requested, let slots):
            """
            \(requested) experts requested at once but the bank has \(slots) slots; \
            a prefill sweep must use the staged path, not the bank
            """
        case .noEvictableSlot(let slots):
            "all \(slots) slots are pinned; the bank is too small for this top-K"
        case .concurrentForward:
            """
            a second forward pass entered the slot bank while one was in flight; \
            the bank holds one pass worth of pinned slots and cannot serve two
            """
        }
    }
}

public final class ExpertSlotBank: @unchecked Sendable {

    public struct Configuration: Sendable {
        /// Target size of the bank. Converted to a slot count on construction;
        /// it is sampled against the process ceiling by the caller, never
        /// written as a constant here.
        public var capacityBytes: Int
        /// Upper bound on experts read in a single `pread` batch, so staging
        /// does not scale with the number of misses.
        public var maxLoadBatch: Int

        public init(capacityBytes: Int, maxLoadBatch: Int = 64) {
            self.capacityBytes = capacityBytes
            self.maxLoadBatch = maxLoadBatch
        }
    }

    public let store: ExpertResidencyStore
    public private(set) var configuration: Configuration

    /// Skip the `eval` that closes an install and let the next mandatory
    /// synchronization absorb the scatter.
    ///
    /// The measurement that justifies the switch: 90% of what a streamed token
    /// costs over a resident one is CPU↔GPU synchronization, not I/O. A
    /// streamed token pays two per MoE layer — the router's `asArray`, which
    /// is unavoidable because a `pread` cannot be issued before the routing
    /// choice is on the host, and this one. Dropping this one halves them,
    /// because the scatter is evaluated anyway at the next layer's router
    /// read: that read depends on this layer's output, which depends on the
    /// scatter. Nothing is deferred beyond one layer, so the staging buffers
    /// are still released layer by layer and the prefill peak does not move.
    ///
    /// Measured on the real 35B checkpoint, 32 decode tokens per arm, all arms
    /// on one loaded model: 12,22 → 13,72 tok/s with a 1 GiB bank, 14,00 →
    /// 15,26 with 3 GiB, 10,35 → 12,29 with 256 MiB. Output bit-for-bit
    /// identical, peak memory unchanged.
    ///
    /// Toggleable at runtime so an A/B can run against one loaded model
    /// instead of two processes, which is the only way to compare numbers in
    /// this subsystem (see the variance warning in the Phase 1 write-up).
    ///
    /// The session sets it from its configuration, where it defaults to on.
    public var deferInstallEval: Bool = false

    private var pools: [MLXArray] = []
    private var slotOfKey: [ExpertKey: Int] = [:]
    private var keyOfSlot: [ExpertKey?] = []
    private var referenced: [Bool] = []
    private var pinned: [Bool] = []
    private var pinnedSlots: [Int] = []
    private var clockHand = 0

    private let lock = NSLock()
    private var statisticsStorage = ExpertSlotBankStatistics()

    /// TD-054(c). The bank's pin set belongs to the pass in flight; a second
    /// concurrent `ensure` would unpin it. Rejected, not serialized.
    private let singleFlight = SingleFlightGuard()

    public init(store: ExpertResidencyStore, configuration: Configuration) {
        self.store = store
        self.configuration = configuration
        allocate(slots: Self.slotCount(for: configuration.capacityBytes, store: store))
    }

    public var slotCount: Int { keyOfSlot.count }

    public var bytesResident: Int { slotCount * store.index.bytesPerExpert }

    public var statistics: ExpertSlotBankStatistics {
        lock.withLock { statisticsStorage }
    }

    public func resetStatistics() {
        lock.withLock { statisticsStorage = ExpertSlotBankStatistics() }
    }

    /// The bank array a projection's quantized matmul indexes with slot ids.
    public func pool(_ piece: ExpertPiece) -> MLXArray { pools[piece.slot] }

    /// Whether the bank already holds `key`.
    ///
    /// The temporal prediction filters through this: prefetching an expert
    /// that is already resident is the one read guaranteed to buy nothing.
    public func isResident(_ key: ExpertKey) -> Bool { slotOfKey[key] != nil }

    public static func slotCount(for capacityBytes: Int, store: ExpertResidencyStore) -> Int {
        max(1, capacityBytes / max(1, store.index.bytesPerExpert))
    }

    // MARK: - Residency

    /// Resolve `keys` to slot ids, reading and installing whatever is missing.
    ///
    /// Every returned slot is pinned until the next call: an eviction must
    /// never take a slot the layer in flight is about to read.
    public func ensure(keys: [ExpertKey]) throws -> [Int] {
        // Before the unpin below, not after: a rejected second pass must not
        // have released the slots the first one is about to read.
        try singleFlight.enter(orThrow: ExpertSlotBankError.concurrentForward)
        defer { singleFlight.leave() }

        guard keys.count <= slotCount else {
            throw ExpertSlotBankError.requestExceedsCapacity(
                requested: keys.count, slots: slotCount)
        }

        for slot in pinnedSlots { pinned[slot] = false }
        pinnedSlots.removeAll(keepingCapacity: true)

        var slots = [Int](repeating: -1, count: keys.count)
        var missing = [(position: Int, key: ExpertKey)]()
        var duplicates = [(position: Int, resolvedBy: Int)]()
        var hits = 0

        for (position, key) in keys.enumerated() {
            if let slot = slotOfKey[key] {
                slots[position] = slot
                referenced[slot] = true
                pin(slot)
                hits += 1
            } else if let existing = missing.first(where: { $0.key == key }) {
                // A repeated key inside one request must map to one slot, and
                // its slot only exists after the install below.
                duplicates.append((position, existing.position))
            } else {
                missing.append((position, key))
            }
        }

        // P6. Claimed on every request, not only when this layer missed:
        // leaving a batch pending would block every later prediction behind
        // one nobody wanted, and the mechanism would quietly stall after its
        // first unclaimed read.
        let claimed = store.takePrefetched(covering: Set(missing.map(\.key)))

        if !missing.isEmpty {
            try install(missing, claimed: claimed, into: &slots)
        }
        for duplicate in duplicates {
            slots[duplicate.position] = slots[duplicate.resolvedBy]
        }

        lock.withLock {
            statisticsStorage.hits += hits
            statisticsStorage.misses += missing.count
        }
        return slots
    }

    private typealias Missing = (position: Int, key: ExpertKey)

    private func install(
        _ missing: [Missing], claimed: ExpertPrefetchClaim?, into slots: inout [Int]
    ) throws {
        var missing = missing

        // Whatever the temporal prediction already read costs no I/O here; the
        // rest goes down the normal path.
        if let claimed {
            let fromPrefetch = missing.filter { claimed.batch.row(of: $0.key) != nil }
            missing = missing.filter { claimed.batch.row(of: $0.key) == nil }
            if !fromPrefetch.isEmpty {
                let rows = MLXArray(fromPrefetch.map { Int32(claimed.batch.row(of: $0.key)!) })
                try installBatch(
                    fromPrefetch, into: &slots,
                    staged: { ExpertPiece.all.map { claimed.arrays[$0.slot][rows] } })
                store.notePrefetchServed(keys: fromPrefetch.count)
            }
            let wasted = claimed.batch.keys.count - fromPrefetch.count
            if wasted > 0 {
                store.notePrefetchWasted(
                    bytes: wasted * store.index.bytesPerExpert)
            }
        }

        var pending = missing[...]
        while !pending.isEmpty {
            let chunk = Array(pending.prefix(configuration.maxLoadBatch))
            pending = pending.dropFirst(chunk.count)
            try installBatch(
                chunk, into: &slots,
                staged: { try self.store.readBatch(keys: chunk.map(\.key)) })
        }
    }

    /// Evict, scatter and record, for a set of experts whose rows `staged`
    /// knows how to produce — from a fresh read or from a prefetched batch.
    ///
    /// `staged` is a closure rather than an array because the eviction has to
    /// happen *before* the read: the slots are pinned as they are taken, and a
    /// read that runs first would leave the bank holding rows nothing points
    /// at if it threw.
    private func installBatch(
        _ entries: [Missing], into slots: inout [Int], staged: () throws -> [MLXArray]
    ) throws {
        var targets = [Int]()
        for entry in entries {
            let slot = try evict()
            targets.append(slot)
            slots[entry.position] = slot
            pin(slot)
        }

        let start = Date.timeIntervalSinceReferenceDate
        let rows = try staged()
        let indices = MLXArray(targets.map { Int32($0) })
        for piece in ExpertPiece.all {
            pools[piece.slot][indices] = rows[piece.slot]
        }
        var evals = 0
        if !deferInstallEval {
            // Evaluating here is what lets MLX donate the pool buffer to
            // the scatter instead of copying the whole bank, and it
            // releases the staging buffers as soon as they are consumed.
            eval(pools)
            evals = 1
        }
        let elapsed = Date.timeIntervalSinceReferenceDate - start

        for (entry, slot) in zip(entries, targets) {
            if let previous = keyOfSlot[slot] { slotOfKey[previous] = nil }
            keyOfSlot[slot] = entry.key
            slotOfKey[entry.key] = slot
            referenced[slot] = true
        }
        lock.withLock {
            statisticsStorage.installSeconds += elapsed
            statisticsStorage.installEvals += evals
        }
    }

    /// CLOCK: sweep, clearing reference bits, until an unpinned slot with a
    /// clear bit is found. Cheaper than exact LRU and does not need a list.
    private func evict() throws -> Int {
        let count = slotCount
        var examined = 0
        while examined < 2 * count {
            let slot = clockHand
            clockHand = (clockHand + 1) % count
            examined += 1
            if pinned[slot] { continue }
            if referenced[slot] {
                referenced[slot] = false
                continue
            }
            if keyOfSlot[slot] != nil {
                lock.withLock { statisticsStorage.evictions += 1 }
            }
            return slot
        }
        throw ExpertSlotBankError.noEvictableSlot(slots: count)
    }

    private func pin(_ slot: Int) {
        if !pinned[slot] {
            pinned[slot] = true
            pinnedSlots.append(slot)
        }
    }

    // MARK: - Elasticity

    /// Grow or shrink the bank.
    ///
    /// Growing preserves the resident experts, replacing one piece at a time
    /// so the transient is one pool, not the whole bank. Shrinking releases
    /// before allocating and restarts cold on purpose: under memory pressure
    /// the point is to give memory back immediately, and the alternative
    /// would need both sizes live at once.
    public func resize(to capacityBytes: Int) {
        let target = Self.slotCount(for: capacityBytes, store: store)
        guard target != slotCount else {
            configuration.capacityBytes = capacityBytes
            return
        }

        // A batch in flight was planned against a residency map that is about
        // to change; a shrink throws the whole map away.
        store.cancelPrefetch()

        if target > slotCount {
            grow(to: target)
        } else {
            shrink(to: target)
        }
        configuration.capacityBytes = capacityBytes
    }

    private func grow(to target: Int) {
        let previous = slotCount
        for piece in ExpertPiece.all {
            let record = store.index.layers[0][piece]
            var pool = MLX.zeros([target] + record.rowShape, dtype: record.dtype.mlxDType)
            pool[0 ..< previous] = pools[piece.slot]
            eval(pool)
            pools[piece.slot] = pool
        }
        keyOfSlot.append(contentsOf: [ExpertKey?](repeating: nil, count: target - previous))
        referenced.append(contentsOf: [Bool](repeating: false, count: target - previous))
        pinned.append(contentsOf: [Bool](repeating: false, count: target - previous))
        clockHand = min(clockHand, target - 1)
    }

    private func shrink(to target: Int) {
        pools.removeAll()
        slotOfKey.removeAll()
        keyOfSlot.removeAll()
        referenced.removeAll()
        pinned.removeAll()
        pinnedSlots.removeAll()
        clockHand = 0
        MLX.GPU.clearCache()
        allocate(slots: target)
    }

    private func allocate(slots: Int) {
        pools = ExpertPiece.all.map { piece in
            let record = store.index.layers[0][piece]
            return MLX.zeros([slots] + record.rowShape, dtype: record.dtype.mlxDType)
        }
        eval(pools)
        keyOfSlot = [ExpertKey?](repeating: nil, count: slots)
        referenced = [Bool](repeating: false, count: slots)
        pinned = [Bool](repeating: false, count: slots)
        pinnedSlots = []
        clockHand = 0
    }
}
