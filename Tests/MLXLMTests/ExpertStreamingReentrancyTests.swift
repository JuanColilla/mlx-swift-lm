// FORK(JuanColilla): R-56 — TD-054(c), the single-forward invariant.
//
// Two concurrent forward passes would corrupt the bank in silence: the second
// one unpins the slots the first is about to read. The invariant held by
// convention (INV-MODEL-01 plus the host actor's queue) and by nothing in the
// types. These tests fix the enforcement, not the convention.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class ExpertStreamingReentrancyTests: XCTestCase {

    private enum Probe: Error { case rejected }

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-reentrancy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func makeBank(experts: Int = 32, layers: Int = 1, slots: Int) throws
        -> ExpertSlotBank
    {
        let directory = try temporaryDirectory()
        _ = try SyntheticExpertCheckpoint.writeWellFormed(
            experts: experts, layers: layers, to: directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let store = try ExpertResidencyStore(index: index, modelDirectory: directory)
        return ExpertSlotBank(
            store: store, configuration: .init(capacityBytes: slots * index.bytesPerExpert))
    }

    // MARK: - The guard itself

    func testGuardAdmitsOneCallerAtATime() throws {
        let guardian = SingleFlightGuard()

        try guardian.enter(orThrow: Probe.rejected)
        XCTAssertTrue(guardian.isOccupied)
        XCTAssertThrowsError(try guardian.enter(orThrow: Probe.rejected)) { error in
            XCTAssertEqual(error as? Probe, .rejected)
        }

        guardian.leave()
        XCTAssertFalse(guardian.isOccupied)
        XCTAssertNoThrow(try guardian.enter(orThrow: Probe.rejected))
        guardian.leave()
    }

    /// A rejected caller must not release the region: the `defer` belongs to
    /// the successful `enter`, and getting that wrong is the failure mode the
    /// guard exists to prevent in the first place.
    func testRejectedCallerDoesNotReleaseTheRegion() throws {
        let guardian = SingleFlightGuard()
        try guardian.enter(orThrow: Probe.rejected)

        func rejectedCaller() {
            do {
                try guardian.enter(orThrow: Probe.rejected)
                defer { guardian.leave() }
                XCTFail("the second caller should have been rejected")
            } catch {}
        }
        rejectedCaller()

        XCTAssertTrue(guardian.isOccupied, "the first caller still owns the region")
    }

    // MARK: - The store's read lanes

    /// The nested call is issued from inside the guarded region through the
    /// test probe, so the rejection is deterministic. Two threads racing for
    /// it would prove nothing: the interleaving that corrupts is exactly the
    /// one a race is not guaranteed to produce.
    func testNestedReadIsRejected() throws {
        let bank = try makeBank(slots: 16)
        let store = bank.store

        var nested: Error?
        store.reentrancyProbe = {
            do {
                _ = try store.readBatch(keys: [ExpertKey(layer: 0, expert: 9)])
                XCTFail("a nested read reached the lanes")
            } catch {
                nested = error
            }
        }

        let outer = try store.readBatch(keys: [ExpertKey(layer: 0, expert: 1)])
        XCTAssertEqual(outer.count, ExpertPiece.all.count, "the outer read must still succeed")
        guard case .concurrentRead? = nested as? ExpertResidencyError else {
            return XCTFail("expected a concurrentRead rejection, got \(String(describing: nested))")
        }
    }

    // MARK: - The bank

    /// The bank's nested call arrives the only way it can in production: down
    /// the `ensure` → `install` → `readBatch` path of a second forward pass.
    func testNestedEnsureIsRejectedAndLeavesThePinSetIntact() throws {
        let bank = try makeBank(slots: 16)
        let keys = (0 ..< 8).map { ExpertKey(layer: 0, expert: $0) }

        var nested: Error?
        bank.store.reentrancyProbe = {
            do {
                _ = try bank.ensure(keys: [ExpertKey(layer: 0, expert: 20)])
                XCTFail("a nested forward pass reached the bank")
            } catch {
                nested = error
            }
        }

        let slots = try bank.ensure(keys: keys)
        bank.store.reentrancyProbe = nil

        guard case .concurrentForward? = nested as? ExpertSlotBankError else {
            return XCTFail(
                "expected a concurrentForward rejection, got \(String(describing: nested))")
        }
        XCTAssertEqual(Set(slots).count, keys.count, "the outer pass kept its own slots")

        // The rejected pass must not have unpinned or evicted anything: the
        // same keys still resolve to the same slots, as hits.
        bank.resetStatistics()
        XCTAssertEqual(try bank.ensure(keys: keys), slots)
        XCTAssertEqual(bank.statistics.misses, 0)
    }

    // MARK: - Resize

    /// `resize` tears down `pools`, `slotOfKey` and `keyOfSlot` and calls
    /// `GPU.clearCache()`. Racing that against an `ensure` is memory
    /// corruption, not a stale read — and the intended caller is a
    /// memory-pressure ladder on its own queue, which is exactly a caller that
    /// cannot see the forward pass.
    func testResizeIsRefusedWhileAForwardHoldsTheBank() throws {
        let bank = try makeBank(slots: 16)
        let keys = (0 ..< 8).map { ExpertKey(layer: 0, expert: $0) }
        let before = bank.slotCount

        var refused: Bool?
        bank.store.reentrancyProbe = {
            refused = bank.resize(to: 4 * bank.store.index.bytesPerExpert) == false
        }
        _ = try bank.ensure(keys: keys)
        bank.store.reentrancyProbe = nil

        XCTAssertEqual(refused, true, "a resize during a forward pass must be refused")
        XCTAssertEqual(bank.slotCount, before, "the refused resize must not have torn anything down")

        // And it works once the pass is over.
        XCTAssertTrue(bank.resize(to: 4 * bank.store.index.bytesPerExpert))
        XCTAssertEqual(bank.slotCount, 4)
    }

    /// Refusing must be a no-op, not a partial teardown: the bank still serves
    /// the experts it held.
    func testARefusedResizeLeavesTheBankUsable() throws {
        let bank = try makeBank(slots: 16)
        let keys = (0 ..< 8).map { ExpertKey(layer: 0, expert: $0) }
        let slots = try bank.ensure(keys: keys)

        bank.store.reentrancyProbe = {
            _ = bank.resize(to: 2 * bank.store.index.bytesPerExpert)
        }
        _ = try bank.ensure(keys: (8 ..< 16).map { ExpertKey(layer: 0, expert: $0) })
        bank.store.reentrancyProbe = nil

        bank.resetStatistics()
        XCTAssertEqual(try bank.ensure(keys: keys), slots)
        XCTAssertEqual(bank.statistics.misses, 0)
    }
}
