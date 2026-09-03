// FORK(JuanColilla): R-56 tasks 1.2 and 1.3 — the read side and the proof
// that delivery to MLX does not copy.

import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class ExpertResidencyStoreTests: XCTestCase {

    private final class FinalizerFlag {
        var fired = false
    }

    /// Contiguous bytes in the array's native dtype, the only representation
    /// that lets a `U32` payload and a `BF16` payload be compared the same way.
    private func bytes(_ array: MLXArray) -> Data {
        array.asData(access: .copy).data
    }

    func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func makeStore(experts: Int = 8, layers: Int = 2) throws -> (
        store: ExpertResidencyStore, payloads: [Data], directory: URL
    ) {
        let directory = try temporaryDirectory()
        let payloads = try SyntheticExpertCheckpoint.writeWellFormed(
            experts: experts, layers: layers, to: directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let store = try ExpertResidencyStore(index: index, modelDirectory: directory)
        return (store, payloads, directory)
    }

    private func expectedRow(
        payloads: [Data], layer: Int, piece: ExpertPiece, expert: Int, rowBytes: Int
    ) -> Data {
        let projection = ExpertProjection.allCases.firstIndex(of: piece.projection)!
        let component = ExpertComponent.allCases.firstIndex(of: piece.component)!
        let position = SyntheticExpertCheckpoint.position(
            layer: layer, projection: projection, component: component)
        let payload = payloads[position]
        return payload.subdata(in: (expert * rowBytes) ..< ((expert + 1) * rowBytes))
    }

    // MARK: - Reading

    func testReadBatchDeliversTheExactBytesOfEachExpert() throws {
        let (store, payloads, _) = try makeStore()
        let keys = [
            ExpertKey(layer: 0, expert: 5),
            ExpertKey(layer: 1, expert: 0),
            ExpertKey(layer: 0, expert: 7),
        ]

        let arrays = try store.readBatch(keys: keys)
        XCTAssertEqual(arrays.count, 9)

        for piece in ExpertPiece.all {
            let array = arrays[piece.slot]
            let record = store.index.layers[0][piece]
            XCTAssertEqual(array.shape, [keys.count] + record.rowShape)
            XCTAssertEqual(array.dtype, record.dtype.mlxDType)

            let data = bytes(array)
            XCTAssertEqual(data.count, keys.count * record.rowBytes)
            for (row, key) in keys.enumerated() {
                let actual = data.subdata(
                    in: (row * record.rowBytes) ..< ((row + 1) * record.rowBytes))
                let expected = expectedRow(
                    payloads: payloads, layer: key.layer, piece: piece, expert: key.expert,
                    rowBytes: record.rowBytes)
                XCTAssertEqual(actual, expected, "piece \(piece.slot), row \(row)")
            }
        }
    }

    /// Consecutive experts of the same layer must collapse into one `pread`;
    /// that is the whole point of `readRuns` for prefill.
    func testConsecutiveExpertsCoalesceIntoOneRead() throws {
        let (store, _, _) = try makeStore(experts: 8, layers: 1)
        store.resetStatistics()

        _ = try store.readBatch(
            keys: (0 ..< 8).map { ExpertKey(layer: 0, expert: $0) })

        let statistics = store.statistics
        XCTAssertEqual(statistics.reads, 9 * 8)
        XCTAssertEqual(statistics.coalescedReads, 9)
        XCTAssertEqual(statistics.bytes, 9 * 8 * 16384)
    }

    func testScatteredExpertsDoNotCoalesce() throws {
        let (store, _, _) = try makeStore(experts: 8, layers: 1)
        store.resetStatistics()

        _ = try store.readBatch(
            keys: [0, 2, 4, 6].map { ExpertKey(layer: 0, expert: $0) })

        XCTAssertEqual(store.statistics.coalescedReads, 9 * 4)
    }

    func testReadRunsSortsAndDeduplicates() throws {
        let (store, payloads, _) = try makeStore(experts: 8, layers: 1)
        let (arrays, experts) = try store.readRuns(layer: 0, experts: [5, 1, 5, 2])

        XCTAssertEqual(experts, [1, 2, 5])
        let piece = ExpertPiece(.gate, .weight)
        let record = store.index.layers[0][piece]
        let data = bytes(arrays[piece.slot])
        for (row, expert) in experts.enumerated() {
            XCTAssertEqual(
                data.subdata(in: (row * record.rowBytes) ..< ((row + 1) * record.rowBytes)),
                expectedRow(
                    payloads: payloads, layer: 0, piece: piece, expert: expert,
                    rowBytes: record.rowBytes))
        }
        // 1 and 2 are consecutive, 5 is alone: two reads per piece.
        XCTAssertEqual(store.statistics.coalescedReads, 9 * 2)
    }

    func testRejectsUnknownLayerAndOutOfRangeExpert() throws {
        let (store, _, _) = try makeStore(experts: 4, layers: 1)

        XCTAssertThrowsError(try store.readBatch(keys: [ExpertKey(layer: 9, expert: 0)]))
        XCTAssertThrowsError(try store.readBatch(keys: [ExpertKey(layer: 0, expert: 4)]))
    }

    // MARK: - Zero-copy delivery (task 1.3)

    /// The definitive signal, and the reason this is not measured with
    /// `activeMemory`: both paths charge the allocator the same bytes, so the
    /// counter cannot tell them apart. What differs is *when* the finalizer
    /// runs. `array.cpp` calls `deleter(data)` synchronously inside the
    /// constructor when it had to copy, and defers it to buffer release when
    /// it adopted the pointer.
    func testPageAlignedDeliveryAdoptsTheHostBuffer() throws {
        let bytes = ExpertOffsetIndex.pageSize * 4
        let buffer = try XCTUnwrap(allocatePageAligned(bytes))
        let flag = FinalizerFlag()

        var array: MLXArray? = MLXArray(
            rawPointer: buffer, [bytes], dtype: .uint8,
            finalizer: { flag.fired = true; free(buffer) })
        eval(array!)

        XCTAssertFalse(
            flag.fired,
            "a page-aligned buffer must be adopted by MLX, not copied and freed")

        array = nil
        XCTAssertTrue(flag.fired, "the buffer must be freed when MLX releases it")
    }

    /// Characterization, not a requirement: on this platform the copy
    /// fallback of `array.cpp:95` does **not** trigger for a misaligned
    /// pointer. Measured across offsets of 1, 16 and 4096 bytes and lengths
    /// that are not page multiples, `newBufferWithBytesNoCopy` adopted every
    /// one of them and returned correct data.
    ///
    /// The store still allocates page-aligned staging on purpose: page
    /// alignment is what Apple documents as the contract, this tolerance is
    /// undocumented and macOS-only evidence, and the cost of honouring the
    /// contract is zero. What this test pins down is that the delivered bytes
    /// are correct either way — so a future alignment bug would show up as a
    /// performance regression, not as corruption.
    func testUnalignedDeliveryStillReturnsCorrectBytes() throws {
        let length = ExpertOffsetIndex.pageSize * 2
        let base = try XCTUnwrap(allocatePageAligned(length + ExpertOffsetIndex.pageSize))
        let misaligned = base.advanced(by: 16)
        let raw = misaligned.bindMemory(to: UInt8.self, capacity: length)
        for i in 0 ..< length { raw[i] = UInt8(truncatingIfNeeded: i &* 7 &+ 3) }
        let flag = FinalizerFlag()

        let array = MLXArray(
            rawPointer: misaligned, [length], dtype: .uint8,
            finalizer: { flag.fired = true; free(base) })
        eval(array)

        XCTAssertEqual(array[0].item(UInt8.self), 3)
        XCTAssertEqual(
            array[length - 1].item(UInt8.self),
            UInt8(truncatingIfNeeded: (length - 1) &* 7 &+ 3))
    }

    /// End-to-end: a real `readBatch` must land on the adopting path, so the
    /// buffers it allocated are still alive and owned by MLX afterwards.
    func testReadBatchDeliveryIsZeroCopy() throws {
        let (store, payloads, _) = try makeStore(experts: 4, layers: 1)

        GPU.clearCache()
        let before = GPU.activeMemory
        let arrays = try store.readBatch(
            keys: (0 ..< 4).map { ExpertKey(layer: 0, expert: $0) })
        eval(arrays)
        let after = GPU.activeMemory

        let staged = 9 * 4 * 16384
        // A copy fallback would have both the staging buffer and the MLX copy
        // live at the peak; adoption charges the allocator exactly once.
        XCTAssertGreaterThanOrEqual(after - before, staged)
        XCTAssertLessThan(after - before, 2 * staged)

        let piece = ExpertPiece(.down, .weight)
        XCTAssertEqual(
            bytes(arrays[piece.slot]).prefix(16384),
            expectedRow(
                payloads: payloads, layer: 0, piece: piece, expert: 0, rowBytes: 16384))
    }

    // MARK: - The real checkpoint

    /// Byte-exact against MLX's own loader on the real 20 GB checkpoint. Uses
    /// a `scales` piece (8 MiB for the whole tensor) so the reference load
    /// does not materialize half a gigabyte.
    func testRealCheckpointRowMatchesMLXLoader() throws {
        let directory = try XCTUnwrap(
            ExpertStreamingTestCheckpoint.directory(),
            "no local Qwen 3.5 MoE checkpoint; set MLX_R56_MODEL_DIR to run this")

        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        let store = try ExpertResidencyStore(index: index, modelDirectory: directory)

        let layer = 0
        let expert = 3
        let piece = ExpertPiece(.gate, .scales)
        let record = index.records(forLayer: layer)![piece]

        let arrays = try store.readBatch(keys: [ExpertKey(layer: layer, expert: expert)])
        let mine = arrays[piece.slot]

        let shard = index.shardURL(record.shard, relativeTo: directory)
        let loaded = try loadArrays(url: shard)
        let key = try XCTUnwrap(
            loaded.keys.first {
                $0.contains("layers.\(layer).") && $0.hasSuffix("switch_mlp.gate_proj.scales")
            })
        let reference = loaded[key]![expert].expandedDimensions(axis: 0)
        eval(mine, reference)

        XCTAssertEqual(mine.shape, reference.shape)
        XCTAssertEqual(mine.dtype, reference.dtype)
        XCTAssertEqual(bytes(mine), bytes(reference))
    }
}
