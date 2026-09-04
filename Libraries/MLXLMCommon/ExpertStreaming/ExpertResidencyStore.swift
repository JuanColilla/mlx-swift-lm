// FORK(JuanColilla): R-56 expert streaming — the read side.
//
// Reads expert rows straight out of the checkpoint shards with `pread` and
// hands them to MLX without a copy. Three constraints shape the whole file:
//
//  * No `mmap`. MLX cannot sparsely materialize a mmap'd tensor; the
//    page-fault route measured 6.1 MB/s in prior art, and on iOS a mapping of
//    a 20 GB checkpoint also spends address space it cannot afford.
//  * No actor on the critical path. This is a `final class` driven by its own
//    threads for the same reason as TD-030/031: a blocking read must not
//    depend on the cooperative pool getting a turn.
//  * Not `DispatchQueue.concurrentPerform`, which is the obvious tool and the
//    wrong one: `dispatch_apply` caps its width at the number of *active*
//    CPUs, so on a six-core phone a requested depth of 8 or 16 would both
//    silently be 6, the configurable queue depth would be fiction, and a P5
//    sweep would flatten out for reasons that have nothing to do with the
//    NAND. Verified by the P0 benchmark harness. Real parked threads make the
//    requested depth the actual depth, and the counters report both.
//  * Page alignment is an invariant, not an optimization. MLX turns a host
//    pointer into an `MTLBuffer` with `newBufferWithBytesNoCopy`, and when
//    that returns null (misaligned pointer or length) `array.cpp` degrades to
//    `malloc` + `std::copy` *silently* — double the memory and a copy per
//    expert, with no error and no log.

import Foundation
import MLX

#if canImport(Darwin)
import Darwin
#endif

/// One routed expert of one layer.
public struct ExpertKey: Hashable, Sendable, Comparable {
    public let layer: Int
    public let expert: Int

    public init(layer: Int, expert: Int) {
        self.layer = layer
        self.expert = expert
    }

    public static func < (lhs: ExpertKey, rhs: ExpertKey) -> Bool {
        (lhs.layer, lhs.expert) < (rhs.layer, rhs.expert)
    }
}

public enum ExpertResidencyError: Error, CustomStringConvertible {
    case cannotOpenShard(String, errno: Int32)
    case allocationFailed(bytes: Int)
    case shortRead(shard: String, offset: Int64, requested: Int, read: Int)
    case readFailed(shard: String, offset: Int64, errno: Int32)
    case unknownLayer(Int)
    case expertOutOfRange(ExpertKey, expertCount: Int)
    case concurrentRead

    public var description: String {
        switch self {
        case .cannotOpenShard(let name, let code):
            "cannot open shard \(name): errno \(code)"
        case .allocationFailed(let bytes):
            "cannot allocate \(bytes) page-aligned bytes for expert staging"
        case .shortRead(let shard, let offset, let requested, let read):
            "short read from \(shard) at \(offset): wanted \(requested), got \(read)"
        case .readFailed(let shard, let offset, let code):
            "read from \(shard) at \(offset) failed: errno \(code)"
        case .unknownLayer(let layer):
            "layer \(layer) has no routed experts in this index"
        case .expertOutOfRange(let key, let count):
            "expert \(key.expert) of layer \(key.layer) is outside 0..<\(count)"
        case .concurrentRead:
            """
            a second forward pass entered the read lanes while one was in \
            flight; the lane pool serves one caller at a time
            """
        }
    }
}

/// Aggregate, identifier-free counters. Never carries an expert id or a layer:
/// routing is a function of the prompt (INV-PRV-01/02).
public struct ExpertResidencyStatistics: Sendable, Equatable {
    public var reads: Int = 0
    public var coalescedReads: Int = 0
    public var bytes: Int = 0
    public var readSeconds: Double = 0
    /// The queue depth the caller asked for.
    public var requestedQueueDepth: Int = 0
    /// The most `pread` calls ever seen in flight at once. Reported next to
    /// the requested depth on purpose: a configurable depth that the runtime
    /// silently caps is a lie, and this is what proves it is not being capped.
    public var peakConcurrentReads: Int = 0

    public var megabytesPerSecond: Double {
        readSeconds > 0 ? Double(bytes) / readSeconds / 1_048_576 : 0
    }
}

public final class ExpertResidencyStore: @unchecked Sendable {

    public struct Configuration: Sendable {
        /// Lanes of concurrent `pread`. Prior art saturates NVMe around 8;
        /// mobile NAND is expected to peak lower (P5 fixes it).
        public var queueDepth: Int
        /// `F_NOCACHE`. Off by default: the kernel page cache is the free L2
        /// tier of the design, and clean file-backed pages do not count
        /// against `phys_footprint`, which is what jetsam measures.
        public var useNoCache: Bool

        public init(queueDepth: Int = 8, useNoCache: Bool = false) {
            self.queueDepth = queueDepth
            self.useNoCache = useNoCache
        }
    }

    public let index: ExpertOffsetIndex
    public let configuration: Configuration

    private let descriptors: [Int32]
    private let lanes: LanePool
    private let statisticsLock = NSLock()
    private var statisticsStorage = ExpertResidencyStatistics()

    /// TD-054(c). `LanePool.run` publishes its job closure while the lanes are
    /// parked; a second caller would overwrite the first one's work.
    private let singleFlight = SingleFlightGuard()

    /// Test seam: called from inside the guarded region of `run(jobs:)`,
    /// before the lanes are woken, so a test can prove that a nested caller is
    /// rejected instead of racing for it with two threads.
    var reentrancyProbe: (() -> Void)?

    public init(
        index: ExpertOffsetIndex,
        modelDirectory: URL,
        configuration: Configuration = Configuration()
    ) throws {
        self.index = index
        self.configuration = configuration

        var descriptors = [Int32]()
        for name in index.shardFiles {
            let path = modelDirectory.appending(path: name).path
            let fd = open(path, O_RDONLY)
            guard fd >= 0 else {
                for open in descriptors { close(open) }
                throw ExpertResidencyError.cannotOpenShard(name, errno: errno)
            }
            if configuration.useNoCache {
                _ = fcntl(fd, F_NOCACHE, 1)
            }
            descriptors.append(fd)
        }
        self.descriptors = descriptors
        self.lanes = LanePool(lanes: max(1, configuration.queueDepth))
        self.statisticsStorage.requestedQueueDepth = max(1, configuration.queueDepth)
    }

    deinit {
        // Order matters: the lanes have to be parked and gone before the
        // descriptors they read from are closed.
        lanes.shutdown()
        for fd in descriptors { close(fd) }
    }

    public var statistics: ExpertResidencyStatistics {
        statisticsLock.withLock { statisticsStorage }
    }

    public func resetStatistics() {
        statisticsLock.withLock {
            statisticsStorage = ExpertResidencyStatistics()
            statisticsStorage.requestedQueueDepth = lanes.lanes
        }
    }

    // MARK: - Reading

    /// Read `keys` into nine freshly allocated page-aligned buffers, delivered
    /// as `[keys.count, …rowShape]` arrays in canonical piece order.
    ///
    /// Ownership of each buffer moves to its `MLXArray`; the finalizer frees
    /// it when MLX releases the last reference.
    ///
    /// Consecutive keys of the same layer are merged into one `pread`, so a
    /// caller that passes a sorted run — the prefill sweep — gets the
    /// coalesced read for free. `readRuns` is that call with the sorting done.
    public func readBatch(keys: [ExpertKey]) throws -> [MLXArray] {
        guard !keys.isEmpty else { return [] }
        for key in keys {
            guard index.records(forLayer: key.layer) != nil else {
                throw ExpertResidencyError.unknownLayer(key.layer)
            }
            guard key.expert >= 0, key.expert < index.expertCount else {
                throw ExpertResidencyError.expertOutOfRange(key, expertCount: index.expertCount)
            }
        }

        let template = index.layers[0]
        var buffers = [UnsafeMutableRawPointer]()
        var jobs = [ReadJob]()

        do {
            for piece in ExpertPiece.all {
                let rowBytes = template[piece].rowBytes
                let total = rowBytes * keys.count
                guard let buffer = allocatePageAligned(total) else {
                    for buffer in buffers { free(buffer) }
                    throw ExpertResidencyError.allocationFailed(bytes: total)
                }
                buffers.append(buffer)

                var row = 0
                while row < keys.count {
                    var span = 1
                    while row + span < keys.count,
                        keys[row + span].layer == keys[row].layer,
                        keys[row + span].expert == keys[row].expert + span
                    {
                        span += 1
                    }
                    let record = index.records(forLayer: keys[row].layer)![piece]
                    jobs.append(
                        ReadJob(
                            fd: descriptors[record.shard],
                            shard: record.shard,
                            destination: buffer.advanced(by: row * rowBytes),
                            fileOffset: record.offset(ofExpert: keys[row].expert),
                            byteCount: rowBytes * span,
                            span: span))
                    row += span
                }
            }
        } catch {
            throw error
        }

        do {
            try run(jobs: jobs)
        } catch {
            for buffer in buffers { free(buffer) }
            throw error
        }

        return buffers.enumerated().map { slot, buffer in
            let piece = ExpertPiece.all[slot]
            let record = template[piece]
            return MLXArray(
                rawPointer: buffer,
                [keys.count] + record.rowShape,
                dtype: record.dtype.mlxDType,
                finalizer: { free(buffer) })
        }
    }

    /// Prefill read: every expert of one layer that the prompt touched, sorted
    /// so consecutive ids collapse into single long reads.
    ///
    /// Returns the arrays plus the expert ids in the row order they occupy, so
    /// the caller can remap its routing indices onto them.
    public func readRuns(layer: Int, experts: [Int]) throws -> (
        arrays: [MLXArray], experts: [Int]
    ) {
        let sorted = Array(Set(experts)).sorted()
        let arrays = try readBatch(keys: sorted.map { ExpertKey(layer: layer, expert: $0) })
        return (arrays, sorted)
    }

    // MARK: - Plumbing

    private struct ReadJob: @unchecked Sendable {
        let fd: Int32
        let shard: Int
        let destination: UnsafeMutableRawPointer
        let fileOffset: Int64
        let byteCount: Int
        /// How many expert rows this single read covers, for the counters.
        let span: Int
    }

    private final class ReadState: @unchecked Sendable {
        let lock = NSLock()
        var error: Error?

        func record(_ error: Error) {
            lock.withLock { if self.error == nil { self.error = error } }
        }
    }

    /// Not reentrant: one forward pass at a time owns the lanes. That is the
    /// same single-pass invariant the slot bank relies on.
    private func run(jobs: [ReadJob]) throws {
        try singleFlight.enter(orThrow: ExpertResidencyError.concurrentRead)
        defer { singleFlight.leave() }
        reentrancyProbe?()

        let state = ReadState()
        let inFlight = InFlightCounter()
        let width = lanes.lanes
        let start = Date.timeIntervalSinceReferenceDate

        lanes.run { lane in
            var jobIndex = lane
            while jobIndex < jobs.count {
                if state.lock.withLock({ state.error != nil }) { return }
                let job = jobs[jobIndex]
                inFlight.enter()
                do {
                    try Self.readFully(job)
                    inFlight.leave()
                } catch {
                    inFlight.leave()
                    state.record(error)
                    return
                }
                jobIndex += width
            }
        }

        let elapsed = Date.timeIntervalSinceReferenceDate - start
        if let error = state.error { throw error }

        let bytes = jobs.reduce(0) { $0 + $1.byteCount }
        let rows = jobs.reduce(0) { $0 + $1.span }
        let peak = inFlight.peak
        statisticsLock.withLock {
            statisticsStorage.reads += rows
            statisticsStorage.coalescedReads += jobs.count
            statisticsStorage.bytes += bytes
            statisticsStorage.readSeconds += elapsed
            statisticsStorage.peakConcurrentReads = max(
                statisticsStorage.peakConcurrentReads, peak)
        }
    }

    /// `pread` is allowed to return short; a loop is the only correct use.
    private static func readFully(_ job: ReadJob) throws {
        var done = 0
        while done < job.byteCount {
            let read = pread(
                job.fd,
                job.destination.advanced(by: done),
                job.byteCount - done,
                off_t(job.fileOffset) + off_t(done))
            if read < 0 {
                if errno == EINTR { continue }
                throw ExpertResidencyError.readFailed(
                    shard: "shard \(job.shard)", offset: job.fileOffset, errno: errno)
            }
            if read == 0 {
                throw ExpertResidencyError.shortRead(
                    shard: "shard \(job.shard)", offset: job.fileOffset,
                    requested: job.byteCount, read: done)
            }
            done += read
        }
    }
}

// MARK: - Lanes

/// A fixed set of parked threads, one per requested queue depth lane.
///
/// See the note at the top of this file for why this is not
/// `DispatchQueue.concurrentPerform`.
///
/// Invariant: `job` is only written while every lane is parked on its own
/// `start` semaphore, i.e. between `run(_:)` calls, so the hot path needs no
/// lock. Lanes only ever touch disjoint jobs.
private final class LanePool: @unchecked Sendable {
    let lanes: Int
    private let start: [DispatchSemaphore]
    private let finished = DispatchSemaphore(value: 0)
    private var job: ((Int) -> Void)?
    private var stopping = false

    init(lanes: Int) {
        self.lanes = lanes
        self.start = (0 ..< lanes).map { _ in DispatchSemaphore(value: 0) }
        for lane in 0 ..< lanes {
            // The thread captures the pool, so the pool outlives its owner
            // until `shutdown()` lets the lanes return. The owner's `deinit`
            // is what calls it.
            let thread = Thread { [self] in
                while true {
                    start[lane].wait()
                    if stopping {
                        finished.signal()
                        return
                    }
                    job?(lane)
                    finished.signal()
                }
            }
            thread.name = "expert-streaming-lane-\(lane)"
            thread.qualityOfService = .userInitiated
            thread.stackSize = 512 << 10
            thread.start()
        }
    }

    func run(_ body: (Int) -> Void) {
        withoutActuallyEscaping(body) { escaping in
            job = escaping
            for semaphore in start { semaphore.signal() }
            for _ in 0 ..< lanes { finished.wait() }
            job = nil
        }
    }

    func shutdown() {
        stopping = true
        for semaphore in start { semaphore.signal() }
        for _ in 0 ..< lanes { finished.wait() }
    }
}

/// Peak concurrent `pread` count, so a run can prove the queue depth it claims.
/// A lock costs ~50 ns against reads that cost tens of microseconds.
private final class InFlightCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var current = 0
    private var peakValue = 0

    func enter() {
        lock.withLock {
            current += 1
            if current > peakValue { peakValue = current }
        }
    }

    func leave() {
        lock.withLock { current -= 1 }
    }

    var peak: Int { lock.withLock { peakValue } }
}

// MARK: - Alignment

/// Page-aligned staging allocation.
///
/// The alignment is what keeps `MLXArray(rawPointer:)` on the zero-copy path.
/// Checked with `assert` so a debug build fails at the allocation rather than
/// at the invisible copy a release build would perform.
func allocatePageAligned(_ byteCount: Int) -> UnsafeMutableRawPointer? {
    precondition(
        byteCount % ExpertOffsetIndex.pageSize == 0,
        "expert staging must be a whole number of 16 KiB pages")
    var pointer: UnsafeMutableRawPointer?
    guard posix_memalign(&pointer, ExpertOffsetIndex.pageSize, byteCount) == 0,
        let pointer
    else {
        return nil
    }
    assert(
        UInt(bitPattern: pointer) % UInt(ExpertOffsetIndex.pageSize) == 0,
        "posix_memalign returned a misaligned pointer")
    return pointer
}

extension ExpertDType {
    var mlxDType: DType {
        switch self {
        case .uint8: .uint8
        case .uint32: .uint32
        case .bfloat16: .bfloat16
        case .float16: .float16
        case .float32: .float32
        }
    }
}
