// FORK(JuanColilla): R-56 — TD-054(a), a read failure in mid-decode.
//
// Before this, a `pread` that failed during generation hit a `fatalError` and
// took the host process with it. The assertion that matters here is not that
// the model keeps producing good tokens — it cannot — but that the process
// survives long enough for the host to cancel the turn, and that the failure
// is published exactly once with a cause the host can act on.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class ExpertStreamingFailureTests: XCTestCase {

    // MARK: - Fixture

    private static let configJSON = StreamedSparseMoeBlockTests.configJSON

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-failure-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func configuration() throws -> Qwen35TextConfiguration {
        try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(Self.configJSON.utf8))
    }

    private func writeCheckpoint() throws -> URL {
        let directory = try temporaryDirectory()
        MLXRandom.seed(56)
        let model = Qwen35TextModel(try configuration())
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)
        let weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1) })
        try save(arrays: weights, url: directory.appending(path: "model.safetensors"))
        return directory
    }

    private func loadStreamed(from directory: URL, bankSlots: Int) throws -> (
        model: Qwen35TextModel, session: ExpertStreamingSession
    ) {
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let session = try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(bankCapacityBytes: bankSlots * index.bytesPerExpert))
        let model = try ExpertStreaming.withSession(session) {
            let model = Qwen35TextModel(try configuration())
            quantize(model: model, groupSize: 64, bits: 4)
            try loadWeights(modelDirectory: directory, model: model)
            return model
        }
        return (model, session)
    }

    /// Cuts the checkpoint off after its header. The descriptors the store
    /// already opened stay valid, so every expert read lands past end of file
    /// and `pread` returns 0 — the deterministic version of "the model file
    /// was deleted or replaced under a running generation".
    private func truncateCheckpoint(_ directory: URL) throws {
        let handle = try FileHandle(forWritingTo: directory.appending(path: "model.safetensors"))
        try handle.truncate(atOffset: 4096)
        try handle.close()
    }

    // MARK: - The failure path

    func testReadFailureDegradesInsteadOfKillingTheProcess() throws {
        let directory = try writeCheckpoint()
        let (model, session) = try loadStreamed(from: directory, bankSlots: 16)
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)

        let healthy = model(tokens, cache: nil)
        eval(healthy)
        XCTAssertNil(session.failure, "the session starts healthy")

        let reported = FailureRecorder()
        session.onFailure { reported.record($0) }

        try truncateCheckpoint(directory)
        // Shrinking restarts the bank cold on purpose (documented in
        // `ExpertSlotBank.resize`), so the next forward has to read, and every
        // read now lands past EOF.
        session.resizeBank(toCapacityBytes: 12 * session.index.bytesPerExpert)

        let degraded = model(tokens, cache: nil)
        eval(degraded)

        let failure = try XCTUnwrap(session.failure, "the failure must be published")
        guard case .shortRead = failure else {
            return XCTFail("expected a shortRead, got \(failure)")
        }
        XCTAssertTrue(
            failure.suggestsCorruptCheckpoint,
            "a truncated checkpoint is the checkpoint's fault, not the caller's")
        XCTAssertEqual(reported.count, 1, "the handler fires once")
        XCTAssertEqual(reported.last, failure)
        XCTAssertEqual(degraded.shape, healthy.shape)
    }

    /// Once failed, the session issues no more reads: the remaining layers of
    /// the token, and every later token, degrade without touching the broken
    /// descriptor.
    func testAFailedSessionStopsReading() throws {
        let directory = try writeCheckpoint()
        let (model, session) = try loadStreamed(from: directory, bankSlots: 16)
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        eval(model(tokens, cache: nil))

        try truncateCheckpoint(directory)
        session.resizeBank(toCapacityBytes: 12 * session.index.bytesPerExpert)
        eval(model(tokens, cache: nil))
        XCTAssertNotNil(session.failure)

        session.store.resetStatistics()
        eval(model(tokens, cache: nil))
        XCTAssertEqual(
            session.store.statistics.reads, 0,
            "a failed session must not keep hammering the descriptor")
    }

    /// A host that installs its handler after the fact still learns about it.
    func testLateHandlerIsCalledImmediately() throws {
        let directory = try writeCheckpoint()
        let (model, session) = try loadStreamed(from: directory, bankSlots: 16)
        let tokens = MLXArray([Int32(1), 2, 3]).reshaped(1, 3)
        eval(model(tokens, cache: nil))

        try truncateCheckpoint(directory)
        session.resizeBank(toCapacityBytes: 12 * session.index.bytesPerExpert)
        eval(model(tokens, cache: nil))

        let reported = FailureRecorder()
        session.onFailure { reported.record($0) }
        XCTAssertEqual(reported.count, 1)
        XCTAssertEqual(reported.last, session.failure)
    }

    /// The diagnosis is the first failure, not the last: a later error on the
    /// same broken session must not overwrite what went wrong.
    func testTheFirstFailureIsTheOneKept() throws {
        let directory = try writeCheckpoint()
        let (_, session) = try loadStreamed(from: directory, bankSlots: 16)

        session.recordFailure(ExpertStreamingFailure.allocationFailed(bytes: 1 << 20))
        session.recordFailure(ExpertStreamingFailure.concurrentForward)

        XCTAssertEqual(session.failure, .allocationFailed(bytes: 1 << 20))
    }

    /// The mapping from the subsystem's own errors, which is what the host
    /// switches on. A `readFailed` must not degrade to `.other`.
    func testErrorsMapToTheirCause() {
        XCTAssertEqual(
            ExpertStreamingFailure(
                ExpertResidencyError.readFailed(shard: "shard 0", offset: 4096, errno: EIO)),
            .readFailed(shard: "shard 0", offset: 4096, errno: EIO))
        XCTAssertEqual(
            ExpertStreamingFailure(ExpertResidencyError.unknownLayer(3)),
            .indexMismatch(ExpertResidencyError.unknownLayer(3).description))
        XCTAssertEqual(
            ExpertStreamingFailure(ExpertSlotBankError.concurrentForward), .concurrentForward)
        XCTAssertEqual(
            ExpertStreamingFailure(ExpertResidencyError.concurrentRead), .concurrentForward)
        XCTAssertEqual(
            ExpertStreamingFailure(
                ExpertSlotBankError.requestExceedsCapacity(requested: 9, slots: 8)),
            .bankExhausted(
                ExpertSlotBankError.requestExceedsCapacity(requested: 9, slots: 8).description))
    }
}

/// The handler is `@Sendable` and fires on the forward pass's thread; this is
/// the smallest thing that can receive it without a data race.
private final class FailureRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var failures: [ExpertStreamingFailure] = []

    func record(_ failure: ExpertStreamingFailure) {
        lock.withLock { failures.append(failure) }
    }

    var count: Int { lock.withLock { failures.count } }
    var last: ExpertStreamingFailure? { lock.withLock { failures.last } }
}
