// FORK(JuanColilla): R-56 expert streaming — TD-054(c), the single-forward
// invariant made enforceable.
//
// The slot bank and the read lanes are both written for exactly one forward
// pass at a time: the bank unpins the previous request's slots at the top of
// `ensure` (a second pass would unpin slots the first one is about to read),
// and `LanePool.run` writes its job closure while every lane is parked (a
// second caller would overwrite the first one's work). Both are true today
// because of INV-MODEL-01 and the serialized queue of the host app, and
// neither was expressed in the types — a second concurrent forward (the
// distributed mesh, or a future change in the actor) would corrupt the bank
// in silence.
//
// This throws rather than trapping on purpose. A `precondition` cannot be
// tested without killing the test process, and it turns a caller's scheduling
// mistake into a crash of the host app; the failure channel of
// `ExpertStreamingSession` already exists to carry an unrecoverable streaming
// error out to the generation loop, and a concurrency violation is one.

import Foundation

/// Rejects a second concurrent entry into a region written for one caller.
///
/// Not a mutex: serializing would hide the bug and deadlock nothing while the
/// bank served two passes interleaved state. The second caller is told.
final class SingleFlightGuard: @unchecked Sendable {
    private let lock = NSLock()
    private var occupied = false

    /// Enters the region, or throws `violation` if another caller is inside.
    ///
    /// Must be balanced with `leave()`, and the caller's `defer` has to be
    /// installed only when this call succeeded — otherwise the rejected
    /// caller releases the region the legitimate one still owns.
    func enter<E: Error>(orThrow violation: @autoclosure () -> E) throws {
        try lock.withLock {
            if occupied { throw violation() }
            occupied = true
        }
    }

    func leave() {
        lock.withLock { occupied = false }
    }

    var isOccupied: Bool { lock.withLock { occupied } }
}
