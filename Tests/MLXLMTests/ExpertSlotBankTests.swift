// FORK(JuanColilla): R-56 task 1.4 — the global slot bank.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class ExpertSlotBankTests: XCTestCase {

    private func bytes(_ array: MLXArray) -> Data {
        array.asData(access: .copy).data
    }

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-bank-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func makeBank(experts: Int = 32, layers: Int = 2, slots: Int) throws -> (
        bank: ExpertSlotBank, payloads: [Data]
    ) {
        let directory = try temporaryDirectory()
        let payloads = try SyntheticExpertCheckpoint.writeWellFormed(
            experts: experts, layers: layers, to: directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let store = try ExpertResidencyStore(index: index, modelDirectory: directory)
        let bank = ExpertSlotBank(
            store: store,
            configuration: .init(capacityBytes: slots * index.bytesPerExpert))
        return (bank, payloads)
    }

    private func expectedRow(
        payloads: [Data], layer: Int, piece: ExpertPiece, expert: Int, rowBytes: Int
    ) -> Data {
        let projection = ExpertProjection.allCases.firstIndex(of: piece.projection)!
        let component = ExpertComponent.allCases.firstIndex(of: piece.component)!
        let position = SyntheticExpertCheckpoint.position(
            layer: layer, projection: projection, component: component)
        return payloads[position].subdata(
            in: (expert * rowBytes) ..< ((expert + 1) * rowBytes))
    }

    // MARK: - Residency

    func testSlotCountFollowsTheByteBudget() throws {
        let (bank, _) = try makeBank(slots: 16)
        XCTAssertEqual(bank.slotCount, 16)
        XCTAssertEqual(bank.bytesResident, 16 * 9 * 16384)
    }

    func testFirstRequestMissesAndSecondHits() throws {
        let (bank, _) = try makeBank(slots: 16)
        let keys = (0 ..< 8).map { ExpertKey(layer: 0, expert: $0) }

        let first = try bank.ensure(keys: keys)
        XCTAssertEqual(bank.statistics.misses, 8)
        XCTAssertEqual(bank.statistics.hits, 0)

        let second = try bank.ensure(keys: keys)
        XCTAssertEqual(second, first, "a resident expert must keep its slot")
        XCTAssertEqual(bank.statistics.hits, 8)
        XCTAssertEqual(bank.statistics.hitRate, 0.5)
    }

    /// The bank is only useful if the bytes it serves are the expert's bytes:
    /// a slot mapping that is right but installs the wrong row is exactly the
    /// failure this catches.
    func testInstalledSlotsHoldTheExpertsBytes() throws {
        let (bank, payloads) = try makeBank(experts: 16, layers: 2, slots: 8)
        let keys = [
            ExpertKey(layer: 0, expert: 11),
            ExpertKey(layer: 1, expert: 4),
        ]
        let slots = try bank.ensure(keys: keys)

        for piece in ExpertPiece.all {
            let pool = bank.pool(piece)
            let rowBytes = bank.store.index.layers[0][piece].rowBytes
            for (position, key) in keys.enumerated() {
                let installed = bytes(pool[slots[position]].expandedDimensions(axis: 0))
                XCTAssertEqual(
                    installed,
                    expectedRow(
                        payloads: payloads, layer: key.layer, piece: piece,
                        expert: key.expert, rowBytes: rowBytes),
                    "piece \(piece.slot), key \(key)")
            }
        }
    }

    func testATopKThatExactlyFillsTheBankStillResolves() throws {
        let (bank, _) = try makeBank(slots: 8)
        let slots = try bank.ensure(keys: (0 ..< 8).map { ExpertKey(layer: 0, expert: $0) })
        XCTAssertEqual(Set(slots).count, 8, "pinned slots must not be reused within a request")
    }

    func testRequestLargerThanTheBankIsRejected() throws {
        let (bank, _) = try makeBank(slots: 4)
        XCTAssertThrowsError(
            try bank.ensure(keys: (0 ..< 8).map { ExpertKey(layer: 0, expert: $0) })
        ) { error in
            guard case ExpertSlotBankError.requestExceedsCapacity = error else {
                return XCTFail("expected requestExceedsCapacity, got \(error)")
            }
        }
    }

    func testRepeatedKeyInOneRequestMapsToOneSlot() throws {
        let (bank, _) = try makeBank(slots: 8)
        let key = ExpertKey(layer: 0, expert: 3)
        let slots = try bank.ensure(keys: [key, ExpertKey(layer: 0, expert: 4), key])
        XCTAssertEqual(slots[0], slots[2])
        XCTAssertEqual(bank.statistics.misses, 2)
    }

    /// CLOCK's reference bit has to protect an expert that is used every step;
    /// evicting it would be the pathological case a bank exists to avoid.
    func testAlwaysUsedExpertIsNeverEvicted() throws {
        let (bank, _) = try makeBank(experts: 32, layers: 1, slots: 4)
        let hot = ExpertKey(layer: 0, expert: 0)
        _ = try bank.ensure(keys: [hot, ExpertKey(layer: 0, expert: 1)])
        let hotSlot = try bank.ensure(keys: [hot])[0]

        for cold in 2 ..< 12 {
            let slots = try bank.ensure(keys: [hot, ExpertKey(layer: 0, expert: cold)])
            XCTAssertEqual(slots[0], hotSlot, "the hot expert was evicted at cold=\(cold)")
        }
    }

    // MARK: - Elasticity

    func testGrowingPreservesResidency() throws {
        let (bank, payloads) = try makeBank(experts: 32, layers: 1, slots: 8)
        let keys = (0 ..< 8).map { ExpertKey(layer: 0, expert: $0) }
        let before = try bank.ensure(keys: keys)

        bank.resize(to: 32 * bank.store.index.bytesPerExpert)
        XCTAssertEqual(bank.slotCount, 32)

        bank.resetStatistics()
        let after = try bank.ensure(keys: keys)
        XCTAssertEqual(after, before)
        XCTAssertEqual(bank.statistics.hits, 8)
        XCTAssertEqual(bank.statistics.misses, 0)

        let piece = ExpertPiece(.up, .weight)
        let rowBytes = bank.store.index.layers[0][piece].rowBytes
        XCTAssertEqual(
            bytes(bank.pool(piece)[after[3]].expandedDimensions(axis: 0)),
            expectedRow(
                payloads: payloads, layer: 0, piece: piece, expert: 3, rowBytes: rowBytes))
    }

    func testShrinkingStartsCold() throws {
        let (bank, _) = try makeBank(experts: 32, layers: 1, slots: 16)
        let keys = (0 ..< 8).map { ExpertKey(layer: 0, expert: $0) }
        _ = try bank.ensure(keys: keys)

        bank.resize(to: 8 * bank.store.index.bytesPerExpert)
        XCTAssertEqual(bank.slotCount, 8)

        bank.resetStatistics()
        _ = try bank.ensure(keys: keys)
        XCTAssertEqual(bank.statistics.misses, 8, "shrinking must not claim stale residency")
    }

    // MARK: - Which regime a request goes to

    private func makeSession(experts: Int, layers: Int, slots: Int) throws
        -> ExpertStreamingSession
    {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.writeWellFormed(
            experts: experts, layers: layers, to: directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        return try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(
                bankCapacityBytes: slots * index.bytesPerExpert, groupSize: 8, bits: 4))
    }

    /// Decode goes through the bank and prefill through transient staging, and
    /// the two are told apart by the token count — not by how many assignments
    /// arrived, which only equals the token count while top-K is exactly 8.
    func testRegimeFollowsTokenCountNotAssignmentCount() throws {
        let session = try makeSession(experts: 32, layers: 1, slots: 16)

        // A single token with top-16: 16 assignments, still decode.
        let decode = try session.resolve(
            layer: 0, tokenCount: 1, experts: (0 ..< 16).map { $0 })
        XCTAssertNil(decode.pools, "one token must go through the bank whatever top-K is")
        XCTAssertEqual(decode.indices.size, 16)

        // Two tokens with top-4: 8 assignments, still prefill.
        let prefill = try session.resolve(
            layer: 0, tokenCount: 2, experts: [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssertNotNil(
            prefill.pools, "a multi-token sweep must not be admitted into the bank")
        XCTAssertEqual(prefill.indices.size, 8)
    }

    /// With `admitOnSweep` on, prefill is allowed into the bank — the A/B the
    /// design leaves open (P2c).
    func testSweepAdmissionIsConfigurable() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.writeWellFormed(experts: 32, layers: 1, to: directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let session = try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(
                bankCapacityBytes: 16 * index.bytesPerExpert, admitOnSweep: true,
                groupSize: 8, bits: 4))

        let prefill = try session.resolve(
            layer: 0, tokenCount: 4, experts: [1, 2, 3, 4, 5, 6, 7, 8])
        XCTAssertNil(prefill.pools)
    }

    // MARK: - P3: the install must not copy the bank

    /// MLX writes a scatter in place only when it can donate the destination
    /// buffer. If it cannot, every install rewrites the whole bank, and the
    /// cost of bringing in eight experts grows with the bank size — which is
    /// the opposite of what a bank is for. Timing the same eight misses
    /// against two very different bank sizes is the cheap way to tell.
    func testInstallCostDoesNotScaleWithBankSize() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["MLX_R56_BENCHMARKS"] == "1",
            "opt-in benchmark")

        /// With `deferred`, `installSeconds` would time a lazy operation and
        /// report nearly zero, so the clock is taken here instead and the
        /// scatter is forced inside the window. Otherwise the two arms would
        /// not be measuring the same thing.
        func installSeconds(slots: Int, deferred: Bool = false) throws -> Double {
            let (bank, _) = try makeBank(experts: 256, layers: 1, slots: slots)
            bank.deferInstallEval = deferred
            var expert = 0
            // Warm-up, then 20 installs of eight fresh experts each.
            for _ in 0 ..< 2 {
                _ = try bank.ensure(
                    keys: (0 ..< 8).map { ExpertKey(layer: 0, expert: ($0 + expert) % 256) })
                expert += 8
            }
            bank.resetStatistics()
            let start = Date.timeIntervalSinceReferenceDate
            for _ in 0 ..< 20 {
                _ = try bank.ensure(
                    keys: (0 ..< 8).map { ExpertKey(layer: 0, expert: ($0 + expert) % 256) })
                expert += 8
                if deferred { eval(ExpertPiece.all.map { bank.pool($0) }) }
            }
            let elapsed = Date.timeIntervalSinceReferenceDate - start
            return (deferred ? elapsed : bank.statistics.installSeconds) / 20
        }

        let small = try installSeconds(slots: 128)
        let large = try installSeconds(slots: 4096)
        let ratio = large / small
        print(
            """
            P3 install cost: 128 slots \(String(format: "%.3f", small * 1000)) ms/install, \
            4096 slots \(String(format: "%.3f", large * 1000)) ms/install, ratio \
            \(String(format: "%.2f", ratio))x (bank is 32x larger)
            """)

        XCTAssertLessThan(
            ratio, 4.0,
            """
            installing eight experts got \(String(format: "%.1f", ratio))x more expensive \
            on a 32x larger bank: the scatter is copying the bank instead of \
            donating it
            """)

        // The same question again with the install's `eval` deferred. The risk
        // of deferring is precisely here: if holding the pending scatter costs
        // MLX the donation, every install rewrites the whole bank and the cost
        // starts scaling with the bank size — silently, with no error and a
        // doubled transient.
        let smallDeferred = try installSeconds(slots: 128, deferred: true)
        let largeDeferred = try installSeconds(slots: 4096, deferred: true)
        let deferredRatio = largeDeferred / smallDeferred
        print(
            """
            P3 install cost (deferred eval): 128 slots \
            \(String(format: "%.3f", smallDeferred * 1000)) ms/install, 4096 slots \
            \(String(format: "%.3f", largeDeferred * 1000)) ms/install, ratio \
            \(String(format: "%.2f", deferredRatio))x
            """)

        XCTAssertLessThan(
            deferredRatio, 4.0,
            """
            with the install eval deferred the cost scaled \
            \(String(format: "%.1f", deferredRatio))x with a 32x larger bank: \
            deferring cost MLX the buffer donation
            """)
    }
}
