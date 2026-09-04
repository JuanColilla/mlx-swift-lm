// FORK(JuanColilla): R-56 expert streaming — TD-054(a), the error channel.
//
// The problem this solves: a streamed MoE block is a `UnaryLayer`, and
// `callAsFunction` cannot throw. Until now a `pread` that failed mid-decode
// (checkpoint deleted under the app, volume unmounted, a truncated file) hit a
// `fatalError` and killed the host process. That is the wrong trade for a
// phone: the read failure is recoverable *as a product event* — cancel the
// turn, unload the model, tell the user — and unrecoverable only as a
// computation.
//
// So the failure leaves the forward pass sideways instead of downwards:
//
//  1. The block records the failure on the session and returns the *shared*
//     expert's output alone, dropping the routed contribution. That is a
//     degradation, not a recovery: the activations of this token are wrong.
//  2. The session latches the failure — once failed, always failed — and
//     every later `resolve` throws it immediately, so no further reads are
//     issued and every remaining layer degrades the same cheap way instead of
//     hammering a broken descriptor forty times per token.
//  3. The host's handler fires once, on the forward pass's own thread.
//
// The contract the host has to honour, and the reason (1) is safe:
// **cancel the generation *and* discard the turn's KV cache.** The degraded
// token was already written into the cache before the handler could run, so
// continuing from that cache would produce fluent nonsense — the exact failure
// mode the MTP key collision produced in Phase 1, and the one no user can see.
// Stopping the stream without dropping the cache is not enough.
//
// A prefetch read that fails does *not* fail the session: it is speculative by
// construction, and the foreground read of the same expert is what decides.

import Foundation

/// Why a streaming session can no longer serve experts.
///
/// Payloads are primitives on purpose: this crosses into the host app's
/// concurrency domain, so it carries no `any Error` and no MLX type.
public enum ExpertStreamingFailure: Error, Sendable, Equatable, CustomStringConvertible {
    /// A `pread` returned an error. `errno` is the system code.
    case readFailed(shard: String, offset: Int64, errno: Int32)
    /// A read hit end of file: the checkpoint on disk is shorter than the
    /// index says. Truncation or replacement under a live session.
    case shortRead(shard: String, offset: Int64, requested: Int, read: Int)
    case cannotOpenShard(shard: String, errno: Int32)
    case allocationFailed(bytes: Int)
    /// The routing choice does not fit the index: a layer or an expert id the
    /// index does not describe. A model/checkpoint mismatch, not an I/O fault.
    case indexMismatch(String)
    /// The bank cannot serve the request at all — top-K larger than the slot
    /// count, or every slot pinned. A configuration fault.
    case bankExhausted(String)
    /// A second forward pass entered while one was in flight (TD-054(c)).
    case concurrentForward
    case other(String)

    public init(_ error: Error) {
        switch error {
        case let failure as ExpertStreamingFailure:
            self = failure
        case let error as ExpertResidencyError:
            switch error {
            case .cannotOpenShard(let name, let code):
                self = .cannotOpenShard(shard: name, errno: code)
            case .allocationFailed(let bytes):
                self = .allocationFailed(bytes: bytes)
            case .shortRead(let shard, let offset, let requested, let read):
                self = .shortRead(
                    shard: shard, offset: offset, requested: requested, read: read)
            case .readFailed(let shard, let offset, let code):
                self = .readFailed(shard: shard, offset: offset, errno: code)
            case .unknownLayer, .unknownFamily, .expertOutOfRange, .mixedFamilies:
                self = .indexMismatch(error.description)
            case .concurrentRead:
                self = .concurrentForward
            }
        case let error as ExpertSlotBankError:
            switch error {
            case .requestExceedsCapacity, .noEvictableSlot, .wrongFamily:
                self = .bankExhausted(error.description)
            case .concurrentForward:
                self = .concurrentForward
            }
        default:
            self = .other(String(describing: error))
        }
    }

    public var description: String {
        switch self {
        case .readFailed(let shard, let offset, let code):
            "expert read from \(shard) at \(offset) failed: errno \(code)"
        case .shortRead(let shard, let offset, let requested, let read):
            "expert read from \(shard) at \(offset) hit EOF: wanted \(requested), got \(read)"
        case .cannotOpenShard(let shard, let code):
            "cannot open shard \(shard): errno \(code)"
        case .allocationFailed(let bytes):
            "cannot allocate \(bytes) page-aligned bytes for expert staging"
        case .indexMismatch(let detail):
            "the routing choice does not fit the expert index: \(detail)"
        case .bankExhausted(let detail):
            "the slot bank cannot serve the request: \(detail)"
        case .concurrentForward:
            "a second forward pass entered the streaming subsystem while one was in flight"
        case .other(let detail):
            detail
        }
    }

    /// Whether the checkpoint on disk is the suspect, as opposed to the
    /// configuration or the caller. The host uses it to decide between "verify
    /// and redownload the model" and "this is a bug".
    public var suggestsCorruptCheckpoint: Bool {
        switch self {
        case .shortRead, .cannotOpenShard, .indexMismatch: true
        case .readFailed, .allocationFailed, .bankExhausted, .concurrentForward, .other: false
        }
    }
}
