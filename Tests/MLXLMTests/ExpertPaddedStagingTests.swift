// FORK(JuanColilla): R-56 — reading expert rows that are not page multiples.
//
// The invariant of the read side is on the staging buffer MLX wraps, not on
// the expert row: `newBufferWithBytesNoCopy` wants a page-aligned pointer
// *and* length, and when it refuses `array.cpp` copies silently and frees the
// staging at once. The first test makes that trap visible so the padding is
// justified by an observation, not by a comment; the rest prove the padded
// path delivers the file's bytes for both families.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class ExpertPaddedStagingTests: XCTestCase {

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-padded-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private final class FinalizerFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var fired = false
        var didFire: Bool { lock.withLock { fired } }
        func fire() { lock.withLock { fired = true } }
    }

    /// Whether wrapping `byteCount` bytes at a page-aligned pointer aliases
    /// the memory (zero-copy) or copies it.
    ///
    /// Decided by observation, not by reading the allocator: the buffer is
    /// written *after* the array is created and before it is evaluated. An
    /// aliasing array sees the write; a copy taken at construction does not.
    /// The finalizer is recorded too — the copy path releases the staging at
    /// construction — because that is the symptom a leak checker would see.
    private func wrapsWithoutCopy(
        byteCount: Int, sliceTo exactBytes: Int? = nil, shape: [Int]
    ) throws -> (aliased: Bool, finalizedAtConstruction: Bool) {
        let flag = FinalizerFlag()
        let buffer = try XCTUnwrap(allocatePageAligned(ExpertReadBatch.paddedByteCount(byteCount)))
        var array: MLXArray? = MLXArray(
            rawPointer: buffer, [byteCount / 4], dtype: .float32, finalizer: { flag.fire() })
        let finalizedAtConstruction = flag.didFire
        var view = array!
        if let exactBytes {
            view = view[0 ..< (exactBytes / 4)]
        }
        view = view.reshaped(shape)
        // The write MLX must observe if it aliased the buffer.
        let floats = buffer.bindMemory(to: Float.self, capacity: byteCount / 4)
        for i in 0 ..< (byteCount / 4) { floats[i] = Float(i % 251) }
        let expected = (0 ..< view.size).map { Float($0 % 251) }
        let seen = view.asArray(Float.self)
        array = nil
        if finalizedAtConstruction { free(buffer) }
        return (seen == expected, finalizedAtConstruction)
    }

    /// The padded path — page-multiple length, sliced to the exact element
    /// count, reshaped — must alias the staging buffer: that is the whole
    /// point of padding. The exact-length wrap is measured next to it and
    /// reported: on this Metal it happens to alias as well, but the API
    /// documents the length as having to be a page multiple, so the padded
    /// path is what the store uses.
    func testPaddedStagingIsWrappedWithoutACopy() throws {
        let exactBytes = 61_440  // K2-Horizon MLP scales per expert, times one
        let padded = ExpertReadBatch.paddedByteCount(exactBytes)
        XCTAssertNotEqual(exactBytes % ExpertOffsetIndex.pageSize, 0)

        let paddedPath = try wrapsWithoutCopy(
            byteCount: padded, sliceTo: exactBytes, shape: [1, 768, 20])
        XCTAssertTrue(paddedPath.aliased, "the padded wrap + slice + reshape copied the staging")
        XCTAssertFalse(paddedPath.finalizedAtConstruction)

        let exactPath = try wrapsWithoutCopy(byteCount: exactBytes, shape: [1, 768, 20])
        print(
            "R56 | exact non-page length wrap: aliased \(exactPath.aliased), "
                + "finalized at construction \(exactPath.finalizedAtConstruction)")

        let alignedPath = try wrapsWithoutCopy(
            byteCount: 4 * ExpertOffsetIndex.pageSize, shape: [4, 4096])
        XCTAssertTrue(alignedPath.aliased)
    }

    func testPaddedByteCountRoundsUpToWholePages() {
        let page = ExpertOffsetIndex.pageSize
        XCTAssertEqual(ExpertReadBatch.paddedByteCount(0), 0)
        XCTAssertEqual(ExpertReadBatch.paddedByteCount(1), page)
        XCTAssertEqual(ExpertReadBatch.paddedByteCount(page), page)
        XCTAssertEqual(ExpertReadBatch.paddedByteCount(page + 1), 2 * page)
        XCTAssertEqual(ExpertReadBatch.paddedByteCount(61_440 * 3), 12 * page)
    }

    /// Every row of both families comes back byte-for-byte from the file,
    /// through the padded path for the pieces that need it and the direct
    /// path for the ones that do not.
    func testReadsBothFamiliesThroughThePaddedStaging() throws {
        let directory = try temporaryDirectory()
        let tensors = SyntheticExpertCheckpoint.movaTensors(
            mlpExperts: 6, valueExperts: 4, denseLayers: 1, totalLayers: 3)
        let payloads = try SyntheticExpertCheckpoint.write(
            tensors, to: directory.appending(path: "model.safetensors"))
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let store = try ExpertResidencyStore(index: index, modelDirectory: directory)

        func payload(named name: String) throws -> Data {
            let position = try XCTUnwrap(tensors.firstIndex { $0.name == name })
            return payloads[position]
        }

        // Three MLP experts of layer 2, one of them out of order so the read
        // is two jobs per piece rather than one coalesced run.
        let mlpKeys = [
            ExpertKey(family: .mlp, layer: 2, expert: 4),
            ExpertKey(family: .mlp, layer: 2, expert: 1),
            ExpertKey(family: .mlp, layer: 2, expert: 2),
        ]
        let mlpArrays = try store.readBatch(keys: mlpKeys)
        XCTAssertEqual(mlpArrays.count, 9)
        for (piece, array) in zip(ExpertFamily.mlp.pieces, mlpArrays) {
            let record = index.mlp.template[piece]
            XCTAssertEqual(array.shape, [3] + record.rowShape)
            let file = try payload(named: "model.layers.2.mlp.switch_mlp.\(piece.name)")
            let bytes = array.asData(access: .copy).data
            XCTAssertEqual(bytes.count, 3 * record.rowBytes)
            for (row, key) in mlpKeys.enumerated() {
                let start = key.expert * record.rowBytes
                XCTAssertEqual(
                    bytes[row * record.rowBytes ..< (row + 1) * record.rowBytes],
                    file[start ..< start + record.rowBytes],
                    "\(piece.name) row \(row) (expert \(key.expert)) does not match the file")
            }
        }

        // Two value experts of layer 1.
        let valueKeys = [
            ExpertKey(family: .value, layer: 1, expert: 3),
            ExpertKey(family: .value, layer: 1, expert: 0),
        ]
        let valueArrays = try store.readBatch(keys: valueKeys)
        XCTAssertEqual(valueArrays.count, 3)
        for (piece, array) in zip(ExpertFamily.value.pieces, valueArrays) {
            let record = index.family(.value)!.template[piece]
            XCTAssertEqual(array.shape, [2] + record.rowShape)
            let file = try payload(named: "model.layers.1.self_attn.switch_v.\(piece.component.rawValue)")
            let bytes = array.asData(access: .copy).data
            for (row, key) in valueKeys.enumerated() {
                let start = key.expert * record.rowBytes
                XCTAssertEqual(
                    bytes[row * record.rowBytes ..< (row + 1) * record.rowBytes],
                    file[start ..< start + record.rowBytes],
                    "switch_v.\(piece.component.rawValue) row \(row) does not match the file")
            }
        }

        let statistics = store.statistics
        // `reads` counts rows per piece: 9 pieces × 3 rows plus 3 pieces × 2 rows.
        XCTAssertEqual(statistics.reads, 9 * 3 + 3 * 2)
        XCTAssertEqual(statistics.bytes, 3 * index.mlp.bytesPerExpert + 2 * index.family(.value)!.bytesPerExpert)
    }

    func testABatchMustHoldOneFamily() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.write(
            SyntheticExpertCheckpoint.movaTensors(
                mlpExperts: 6, valueExperts: 4, denseLayers: 1, totalLayers: 2),
            to: directory.appending(path: "model.safetensors"))
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let store = try ExpertResidencyStore(index: index, modelDirectory: directory)

        XCTAssertThrowsError(
            try store.readBatch(keys: [
                ExpertKey(family: .mlp, layer: 1, expert: 0),
                ExpertKey(family: .value, layer: 1, expert: 0),
            ])
        ) { error in
            guard case ExpertResidencyError.mixedFamilies = error else {
                return XCTFail("expected mixedFamilies, got \(error)")
            }
        }

        // The value family's range is its own: expert 5 exists for the MLP
        // family (6 experts) but not for the value family (4).
        XCTAssertThrowsError(
            try store.readBatch(keys: [ExpertKey(family: .value, layer: 1, expert: 5)])
        ) { error in
            guard case ExpertResidencyError.expertOutOfRange(_, let count) = error else {
                return XCTFail("expected expertOutOfRange, got \(error)")
            }
            XCTAssertEqual(count, 4)
        }
    }

    /// One bank per family, each with its own geometry, and a key of the
    /// wrong family is refused rather than scattered into rows of the wrong
    /// size.
    func testSessionBuildsOneBankPerFamily() throws {
        let directory = try temporaryDirectory()
        try SyntheticExpertCheckpoint.write(
            SyntheticExpertCheckpoint.movaTensors(
                mlpExperts: 6, valueExperts: 4, denseLayers: 1, totalLayers: 3),
            to: directory.appending(path: "model.safetensors"))
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let session = try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(
                bankCapacityBytes: 0, groupSize: 8, bits: 4,
                bankCapacityBytesPerFamily: [
                    .mlp: 8 * index.mlp.bytesPerExpert,
                    .value: 3 * index.family(.value)!.bytesPerExpert,
                ]))

        XCTAssertEqual(Set(session.banks.keys), [.mlp, .value])
        XCTAssertEqual(session.bank.slotCount, 8)
        XCTAssertEqual(session.bank.family, .mlp)
        let value = try XCTUnwrap(session.bank(for: .value))
        XCTAssertEqual(value.slotCount, 3)
        XCTAssertEqual(value.pool(ExpertPiece(.value, .weight)).shape, [3, 48, 64])
        XCTAssertEqual(session.bank.pool(ExpertPiece(.down, .scales)).shape, [8, 64, 64])

        XCTAssertThrowsError(try value.ensure(keys: [ExpertKey(family: .mlp, layer: 1, expert: 0)])) {
            error in
            guard case ExpertSlotBankError.wrongFamily(.value, .mlp) = error else {
                return XCTFail("expected wrongFamily, got \(error)")
            }
        }

        let slots = try value.ensure(keys: [
            ExpertKey(family: .value, layer: 1, expert: 2),
            ExpertKey(family: .value, layer: 2, expert: 2),
        ])
        XCTAssertEqual(Set(slots).count, 2)
        XCTAssertTrue(value.isResident(ExpertKey(family: .value, layer: 1, expert: 2)))
        XCTAssertFalse(session.bank.isResident(ExpertKey(family: .mlp, layer: 1, expert: 2)))
        XCTAssertEqual(session.bankStatistics.misses, 2)

        // The resolution goes to the bank of the family asked for.
        let resolution = try session.resolve(
            layer: 2, family: .value, tokenCount: 1, experts: [2, 0])
        XCTAssertNil(resolution.pools)
        XCTAssertEqual(resolution.indices.shape, [2])
        XCTAssertEqual(session.bankStatistics.hits, 1)
        XCTAssertEqual(session.bankStatistics.misses, 3)
        XCTAssertNil(session.lastFailure)
    }
}
