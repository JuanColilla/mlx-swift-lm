// FORK(JuanColilla): R-56 P6 — temporal prefetch.
//
// The constraint that shapes this file, from §9 of the plan: the experts of
// layer L+1 of the *current* token cannot be prefetched, because that router
// depends on layer L's output. Only two predictions exist, and this is the
// cheap one — "the next token reuses the previous token's experts". So while
// the GPU computes layer L of token N, the lanes read what layer L+1 used at
// token N−1.
//
// Three decisions that are not obvious:
//
//  * **Only I/O runs in the background.** The read fills page-aligned buffers
//    and stops there. Building the `MLXArray`s and scattering them into the
//    bank happen on the forward pass's own thread, at the point where it was
//    going to install anyway. MLX's graph is not thread-safe and this is not
//    the place to find out.
//  * **Its own lanes.** The foreground lane pool publishes its job closure
//    while its threads are parked, so a second caller would overwrite the
//    first one's work — that is what the single-flight guard rejects. The
//    prefetcher gets its own pool over the same descriptors: `pread` carries
//    its own offset, so sharing an fd across threads is safe.
//  * **Installed on demand, not eagerly.** A prefetched expert is only written
//    into the bank if the layer actually asks for it. Installing the whole
//    prediction would evict the working set on every mispredicted expert,
//    which is the failure mode `admitOnSweep` exists to avoid in prefill.
//
// A prefetch that fails is discarded and counted. It never fails the session:
// it is speculative, and the foreground read of the same expert is the one
// entitled to an opinion.

import Foundation
import MLX

#if canImport(Darwin)
import Darwin
#endif

/// One `pread`, or one coalesced run of them.
struct ExpertReadJob: @unchecked Sendable {
    let fd: Int32
    let shard: Int
    let destination: UnsafeMutableRawPointer
    let fileOffset: Int64
    let byteCount: Int
    /// How many expert rows this single read covers, for the counters.
    let span: Int
}

/// Staging for a batch of expert rows: allocated and planned, not yet read,
/// and not yet owned by MLX.
///
/// Owns raw memory, so it has exactly one consumer: either `materialize`,
/// which hands the buffers to MLX with a finalizer, or `discard`, which frees
/// them. Doing neither leaks; `deinit` catches that.
final class ExpertReadBatch: @unchecked Sendable {
    let family: ExpertFamily
    let keys: [ExpertKey]
    let jobs: [ExpertReadJob]
    private var buffers: [UnsafeMutableRawPointer]
    private let rowOfKey: [ExpertKey: Int]

    init(
        family: ExpertFamily, keys: [ExpertKey], buffers: [UnsafeMutableRawPointer],
        jobs: [ExpertReadJob]
    ) {
        self.family = family
        self.keys = keys
        self.buffers = buffers
        self.jobs = jobs
        self.rowOfKey = Dictionary(
            keys.enumerated().map { ($0.element, $0.offset) }, uniquingKeysWith: { first, _ in
                first
            })
    }

    deinit { discard() }

    var byteCount: Int { jobs.reduce(0) { $0 + $1.byteCount } }

    /// Which row of the staged arrays holds `key`, if this batch read it.
    func row(of key: ExpertKey) -> Int? { rowOfKey[key] }

    /// `byteCount` rounded up to a whole number of pages: the length MLX is
    /// handed for a staging buffer, so `newBufferWithBytesNoCopy` accepts it.
    static func paddedByteCount(_ byteCount: Int) -> Int {
        let page = ExpertOffsetIndex.pageSize
        return (byteCount + page - 1) / page * page
    }

    /// Hand the buffers to MLX. Ownership of each moves to its `MLXArray`; the
    /// finalizer frees it when MLX releases the last reference.
    ///
    /// A piece whose rows are page multiples is wrapped in its final shape.
    /// One that is not — the K2-Horizon MLP scales, 61.440 bytes per expert —
    /// is wrapped over the *padded* length as a flat array, then sliced to
    /// the exact element count and reshaped. Both the slice (a leading-axis
    /// slice of a contiguous array) and the reshape are views in MLX, so the
    /// bytes are still never copied; wrapping the exact length instead would
    /// make `make_buffer` return null and MLX copy silently.
    func materialize(index: ExpertOffsetIndex) -> [MLXArray] {
        let template = index.family(family)!.template
        let taken = buffers
        buffers = []
        return zip(family.pieces, taken).map { piece, buffer in
            let record = template[piece]
            let shape = [keys.count] + record.rowShape
            let exactBytes = record.rowBytes * keys.count
            let paddedBytes = Self.paddedByteCount(exactBytes)
            if paddedBytes == exactBytes {
                return MLXArray(
                    rawPointer: buffer, shape, dtype: record.dtype.mlxDType,
                    finalizer: { free(buffer) })
            }
            let itemSize = record.dtype.itemSize
            let flat = MLXArray(
                rawPointer: buffer, [paddedBytes / itemSize], dtype: record.dtype.mlxDType,
                finalizer: { free(buffer) })
            return flat[0 ..< (exactBytes / itemSize)].reshaped(shape)
        }
    }

    func discard() {
        for buffer in buffers { free(buffer) }
        buffers = []
    }
}

/// A prefetched batch handed to the forward pass, with its rows already in
/// MLX's hands.
typealias ExpertPrefetchClaim = (batch: ExpertReadBatch, arrays: [MLXArray])

/// Background reader for the temporal prediction.
///
/// One batch in flight per family: issued at one step, consumed at the next
/// step of the same family. A second request for a family while one is
/// pending is dropped rather than queued — the prediction it carries is about
/// to be superseded anyway, and a queue here would only spend NAND bandwidth
/// the foreground needs. The families are kept apart so the MLP bank's claim
/// cannot swallow, and count as wasted, a batch read for the value bank.
final class ExpertPrefetcher: @unchecked Sendable {

    private final class Pending {
        let batch: ExpertReadBatch
        let done = DispatchSemaphore(value: 0)
        var error: Error?
        init(batch: ExpertReadBatch) { self.batch = batch }
    }

    private unowned let store: ExpertResidencyStore
    private let lanes: LanePool
    /// The driver is a GCD queue, not a `Task`: a blocking `pread` must not
    /// depend on the cooperative pool getting a turn (TD-030/031).
    private let driver: DispatchQueue
    private let lock = NSLock()
    private var pending: [ExpertFamily: Pending] = [:]
    private var stopped = false

    init(store: ExpertResidencyStore, queueDepth: Int) {
        self.store = store
        self.lanes = LanePool(lanes: max(1, queueDepth))
        self.driver = DispatchQueue(
            label: "expert-streaming-prefetch", qos: .userInitiated)
    }

    /// Wait for anything in flight, free it, and park the lanes.
    ///
    /// Called before the store closes its descriptors: a lane still reading
    /// from a closed fd is a use-after-free with an error code.
    func shutdown() {
        cancel()
        lock.withLock { stopped = true }
        driver.sync {}
        lanes.shutdown()
    }

    /// Drop whatever is in flight, for every family, waiting for it first.
    func cancel() {
        let claimed: [Pending] = lock.withLock {
            defer { pending = [:] }
            return Array(pending.values)
        }
        for pending in claimed {
            pending.done.wait()
            pending.batch.discard()
        }
    }

    /// Issue the read for `keys` — all of one family — in the background.
    /// Cheap and best effort: a planning failure is counted and forgotten.
    func request(keys: [ExpertKey]) {
        guard let family = keys.first?.family else { return }
        let admitted: Bool = lock.withLock {
            guard !stopped, pending[family] == nil else { return false }
            return true
        }
        guard admitted else { return }

        let batch: ExpertReadBatch
        do {
            batch = try store.plan(keys: keys.sorted())
        } catch {
            store.notePrefetchFailure()
            return
        }

        let entry = Pending(batch: batch)
        lock.withLock { pending[family] = entry }
        store.notePrefetchIssued(keys: keys.count)

        // The store is captured strongly for the duration of the block, not
        // through the `unowned` field: if the store started deallocating with
        // this block still queued, `shutdown()` would be waiting on it while
        // the block loaded an `unowned` reference to an object in `deinit`.
        driver.async { [self, store = self.store] in
            do {
                try store.execute(jobs: entry.batch.jobs, on: lanes, foreground: false)
            } catch {
                entry.error = error
                store.notePrefetchFailure()
            }
            entry.done.signal()
        }
    }

    /// Claim the batch in flight for `family` if it covers any of `wanted`.
    ///
    /// Waits for it rather than skipping it: the bytes are already being read,
    /// and waiting is always cheaper than reissuing the same `pread`. The wait
    /// is reported so a prefetch that is never ready in time can be told apart
    /// from one that is simply mispredicting.
    ///
    /// Returns the batch and its staged arrays; the caller decides which rows
    /// to install.
    func take(family: ExpertFamily, covering wanted: Set<ExpertKey>) -> ExpertPrefetchClaim? {
        guard let pending = lock.withLock({ pending.removeValue(forKey: family) }) else {
            return nil
        }

        let start = Date.timeIntervalSinceReferenceDate
        pending.done.wait()
        store.notePrefetchWait(seconds: Date.timeIntervalSinceReferenceDate - start)

        guard pending.error == nil else {
            pending.batch.discard()
            return nil
        }
        let useful = pending.batch.keys.contains { wanted.contains($0) }
        guard useful else {
            store.notePrefetchWasted(bytes: pending.batch.byteCount)
            pending.batch.discard()
            return nil
        }
        return (pending.batch, pending.batch.materialize(index: store.index))
    }
}
