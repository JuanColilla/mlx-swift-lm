// FORK(JuanColilla): R-56 Phase 2 — the synchronization counters and the
// deferred install.
//
// Phase 1 measured that ~90% of what a streamed token costs over a resident
// one is CPU↔GPU synchronization, and *inferred* the count ("about eighty per
// token"). These tests fix both: the count is measured, and dropping the
// install's `eval` must not change a single number the model produces.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class ExpertStreamingSyncTests: XCTestCase {

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-sync-\(UUID().uuidString)")
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

    // MARK: - What the counters say

    /// The fixture has two MoE layers, so one forward is two router reads —
    /// that is the irreducible half. The install `eval`s are the removable
    /// half, and they disappear when the flag is on.
    func testDeferringTheInstallRemovesHalfTheSynchronizations() throws {
        let directory = try writeCheckpoint()
        let (model, session) = try loadStreamed(from: directory, bankSlots: 16)
        let tokens = MLXArray([Int32(1)]).reshaped(1, 1)

        // The default is on; this arm is the "before" it replaced.
        session.bank.deferInstallEval = false
        session.resetStatistics()
        eval(model(tokens, cache: nil))
        let eager = session.syncCounters
        XCTAssertEqual(eager.layerForwards, 2)
        XCTAssertEqual(eager.routerEvals, 2)
        XCTAssertGreaterThan(eager.installEvals, 0, "a cold bank has to install")

        // Same loaded model, same bank state reset cold by the shrink/grow.
        session.bank.deferInstallEval = true
        session.resizeBank(toCapacityBytes: 12 * session.index.bytesPerExpert)
        session.resizeBank(toCapacityBytes: 16 * session.index.bytesPerExpert)
        session.resetStatistics()
        eval(model(tokens, cache: nil))
        let deferred = session.syncCounters

        XCTAssertEqual(deferred.routerEvals, eager.routerEvals, "the router read is irreducible")
        XCTAssertGreaterThan(
            session.bank.statistics.misses, 0, "the arms must compare the same work")
        XCTAssertEqual(deferred.installEvals, 0)
        XCTAssertLessThan(deferred.total, eager.total)
    }

    /// The load-bearing assertion: deferring changes when the scatter runs,
    /// never what it writes. Bit for bit, not within a tolerance.
    func testDeferredInstallProducesIdenticalLogits() throws {
        let directory = try writeCheckpoint()
        let tokens = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)

        let (eagerModel, eagerSession) = try loadStreamed(from: directory, bankSlots: 12)
        eagerSession.bank.deferInstallEval = false
        let eager = eagerModel(tokens, cache: nil)
        eval(eager)

        let (deferredModel, deferredSession) = try loadStreamed(from: directory, bankSlots: 12)
        deferredSession.bank.deferInstallEval = true
        let deferred = deferredModel(tokens, cache: nil)
        eval(deferred)

        XCTAssertEqual(
            eager.asData(access: .copy).data, deferred.asData(access: .copy).data,
            "deferring the install changed the model's output")
    }

    /// The bank still holds the right bytes when the scatter is only forced by
    /// a later, unrelated evaluation.
    func testDeferredInstallStillInstallsTheRightRows() throws {
        let directory = try temporaryDirectory()
        let payloads = try SyntheticExpertCheckpoint.writeWellFormed(
            experts: 32, layers: 2, to: directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let store = try ExpertResidencyStore(index: index, modelDirectory: directory)
        let bank = ExpertSlotBank(
            store: store, configuration: .init(capacityBytes: 16 * index.bytesPerExpert))
        bank.deferInstallEval = true

        let keys = [ExpertKey(layer: 0, expert: 11), ExpertKey(layer: 1, expert: 4)]
        let slots = try bank.ensure(keys: keys)
        XCTAssertEqual(bank.statistics.installEvals, 0)

        let piece = ExpertPiece(.up, .weight)
        let rowBytes = index.layers[0][piece].rowBytes
        for (key, slot) in zip(keys, slots) {
            let projection = ExpertProjection.allCases.firstIndex(of: piece.projection)!
            let component = ExpertComponent.allCases.firstIndex(of: piece.component)!
            let position = SyntheticExpertCheckpoint.position(
                layer: key.layer, projection: projection, component: component)
            let expected = payloads[position].subdata(
                in: (key.expert * rowBytes) ..< ((key.expert + 1) * rowBytes))
            let row = bank.pool(piece)[slot].expandedDimensions(axis: 0)
            eval(row)
            XCTAssertEqual(row.asData(access: .copy).data, expected)
        }
    }
}
