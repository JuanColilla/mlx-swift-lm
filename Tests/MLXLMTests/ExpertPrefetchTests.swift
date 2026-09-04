// FORK(JuanColilla): R-56 P6 — temporal prefetch.
//
// Two questions, in order of importance: does the prediction change what the
// model computes (it must not), and does it ever get claimed (otherwise the
// flag is a no-op with extra I/O).

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class ExpertPrefetchTests: XCTestCase {

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-prefetch-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func configuration() throws -> Qwen35TextConfiguration {
        try JSONDecoder().decode(
            Qwen35TextConfiguration.self,
            from: Data(StreamedSparseMoeBlockTests.configJSON.utf8))
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

    private func loadStreamed(from directory: URL, bankSlots: Int, prefetch: Bool) throws -> (
        model: Qwen35TextModel, session: ExpertStreamingSession
    ) {
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let session = try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(
                bankCapacityBytes: bankSlots * index.bytesPerExpert,
                temporalPrefetch: prefetch))
        let model = try ExpertStreaming.withSession(session) {
            let model = Qwen35TextModel(try configuration())
            quantize(model: model, groupSize: 64, bits: 4)
            try loadWeights(modelDirectory: directory, model: model)
            return model
        }
        return (model, session)
    }

    /// One decode step per call: `tokenCount == 1` is what puts the request on
    /// the bank path, which is the only path the prediction exists for.
    private func decode(_ model: Qwen35TextModel, steps: Int) -> [Data] {
        (0 ..< steps).map { step in
            let logits = model(MLXArray([Int32(step % 100 + 1)]).reshaped(1, 1), cache: nil)
            eval(logits)
            return logits.asData(access: .copy).data
        }
    }

    // MARK: - Correctness

    /// The prediction may change which slot holds which expert; it must not
    /// change a single bit the model produces. Bit for bit, not a tolerance:
    /// streaming does not change the arithmetic, and neither does prefetching.
    func testPrefetchDoesNotChangeTheOutput() throws {
        let directory = try writeCheckpoint()

        let (plain, plainSession) = try loadStreamed(
            from: directory, bankSlots: 10, prefetch: false)
        XCTAssertFalse(plainSession.isTemporalPrefetchEnabled)
        let expected = decode(plain, steps: 6)

        let (prefetching, session) = try loadStreamed(
            from: directory, bankSlots: 10, prefetch: true)
        XCTAssertTrue(session.isTemporalPrefetchEnabled)
        XCTAssertEqual(decode(prefetching, steps: 6), expected)
    }

    // MARK: - Does it ever get claimed?

    /// With a bank too small to hold both layers' working sets, the previous
    /// token's experts really do get evicted, so the prediction has something
    /// to bring back. If this ever reported zero the flag would be pure cost.
    func testThePredictionIsIssuedAndClaimed() throws {
        let directory = try writeCheckpoint()
        let (model, session) = try loadStreamed(from: directory, bankSlots: 10, prefetch: true)

        // The first token has no previous token to predict from.
        _ = decode(model, steps: 1)
        session.resetStatistics()
        _ = decode(model, steps: 8)

        let statistics = session.store.statistics
        XCTAssertGreaterThan(statistics.prefetchIssued, 0, "no prediction was ever issued")
        XCTAssertGreaterThan(statistics.prefetchServed, 0, "no prediction was ever claimed")
        XCTAssertEqual(statistics.prefetchFailures, 0)
        XCTAssertLessThanOrEqual(statistics.prefetchServed, statistics.prefetchIssued)
    }

    /// Turning it off mid-session has to actually stop it, because that is how
    /// the A/B of the experiments switches arms on one loaded model.
    func testDisablingStopsThePrediction() throws {
        let directory = try writeCheckpoint()
        let (model, session) = try loadStreamed(from: directory, bankSlots: 10, prefetch: true)
        _ = decode(model, steps: 4)

        session.isTemporalPrefetchEnabled = false
        session.resetStatistics()
        _ = decode(model, steps: 4)

        XCTAssertEqual(session.store.statistics.prefetchIssued, 0)
    }

    /// A session built without a prefetcher has no lanes to turn on, and the
    /// flag must say so rather than claim a capability that does not exist.
    func testAFlagWithoutLanesStaysOff() throws {
        let directory = try writeCheckpoint()
        let (_, session) = try loadStreamed(from: directory, bankSlots: 10, prefetch: false)
        session.isTemporalPrefetchEnabled = true
        XCTAssertFalse(session.isTemporalPrefetchEnabled)
        XCTAssertFalse(session.store.isPrefetching)
    }

    // MARK: - A speculative read has no standing

    /// A prefetch that cannot even be planned is counted and forgotten. It
    /// must not latch a failure on the session: the foreground read of the
    /// same expert is the one entitled to an opinion, and killing a turn over
    /// a speculative read would be a self-inflicted outage.
    func testAFailedPrefetchDoesNotFailTheSession() throws {
        let directory = try writeCheckpoint()
        let (model, session) = try loadStreamed(from: directory, bankSlots: 10, prefetch: true)

        session.store.prefetch(keys: [ExpertKey(layer: 0, expert: 9_999)])
        XCTAssertEqual(session.store.statistics.prefetchFailures, 1)
        XCTAssertNil(session.lastFailure)

        // And the model still runs.
        XCTAssertEqual(decode(model, steps: 2).count, 2)
        XCTAssertNil(session.lastFailure)
    }

    /// Shrinking the bank invalidates the residency map the prediction was
    /// planned against, so anything in flight is dropped rather than installed
    /// into slots that no longer mean what they meant.
    func testResizingDropsWhatIsInFlight() throws {
        let directory = try writeCheckpoint()
        let (model, session) = try loadStreamed(from: directory, bankSlots: 12, prefetch: true)
        _ = decode(model, steps: 4)

        session.resizeBank(toCapacityBytes: 8 * session.index.bytesPerExpert)
        session.resetPrediction()
        XCTAssertEqual(decode(model, steps: 2).count, 2)
        XCTAssertNil(session.lastFailure)
    }
}
