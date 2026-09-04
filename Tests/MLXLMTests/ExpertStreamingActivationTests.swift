// FORK(JuanColilla): R-56 — activating a session around an `async` load.
//
// The construction path that needs the session (`ModelFactory` →
// `Model.init(configuration)` → layer initializers) has no seam to pass one
// through, so the activation is a global. The closure form is enough when the
// load is synchronous; a factory whose `loadContainer` is `async throws` needs
// the pair, because the activation has to survive a suspension point.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import XCTest

@testable import MLXLLM

final class ExpertStreamingActivationTests: XCTestCase {

    private func temporaryDirectory() throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appending(path: "r56-activation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    private func makeSession() throws -> ExpertStreamingSession {
        let directory = try temporaryDirectory()
        _ = try SyntheticExpertCheckpoint.writeWellFormed(
            experts: 16, layers: 1, to: directory)
        let index = try ExpertOffsetIndex.build(modelDirectory: directory)
        return try ExpertStreamingSession(
            index: index, modelDirectory: directory,
            configuration: .init(
                bankCapacityBytes: 8 * index.bytesPerExpert, groupSize: 8, bits: 4))
    }

    override func tearDown() {
        // No test may leave the process with an activation standing.
        ExpertStreaming.deactivate()
        super.tearDown()
    }

    func testActivateAndDeactivate() throws {
        let session = try makeSession()
        XCTAssertNil(ExpertStreaming.activeSession)

        ExpertStreaming.activate(session)
        XCTAssertTrue(ExpertStreaming.activeSession === session)

        ExpertStreaming.deactivate()
        XCTAssertNil(ExpertStreaming.activeSession)
    }

    /// `deactivate` is what a `defer` calls, and a `defer` runs on paths that
    /// already deactivated. It must not care.
    func testDeactivateIsIdempotent() throws {
        let session = try makeSession()
        ExpertStreaming.activate(session)
        ExpertStreaming.deactivate()
        ExpertStreaming.deactivate()
        XCTAssertNil(ExpertStreaming.activeSession)
    }

    /// The point of the pair: the activation survives an `await`.
    func testActivationSurvivesASuspensionPoint() async throws {
        let session = try makeSession()

        ExpertStreaming.activate(session)
        defer { ExpertStreaming.deactivate() }

        try await Task.sleep(nanoseconds: 1_000_000)
        XCTAssertTrue(
            ExpertStreaming.activeSession === session,
            "the activation must still stand after the load's await")
    }

    func testAsyncWithSessionActivatesAndClearsUp() async throws {
        let session = try makeSession()

        let seen = await ExpertStreaming.withSession(session) { () async -> Bool in
            try? await Task.sleep(nanoseconds: 1_000_000)
            return ExpertStreaming.activeSession === session
        }

        XCTAssertTrue(seen)
        XCTAssertNil(ExpertStreaming.activeSession)
    }

    /// A load that throws must not leave the activation standing: the next
    /// model built in the process would be constructed against a session that
    /// belongs to a load that failed.
    func testAThrowingAsyncLoadStillDeactivates() async throws {
        struct LoadFailed: Error {}
        let session = try makeSession()

        do {
            _ = try await ExpertStreaming.withSession(session) { () async throws -> Int in
                try await Task.sleep(nanoseconds: 1_000_000)
                throw LoadFailed()
            }
            XCTFail("the load should have thrown")
        } catch is LoadFailed {}

        XCTAssertNil(ExpertStreaming.activeSession)
    }

    func testSynchronousWithSessionStillWorks() throws {
        let session = try makeSession()
        let seen = ExpertStreaming.withSession(session) {
            ExpertStreaming.activeSession === session
        }
        XCTAssertTrue(seen)
        XCTAssertNil(ExpertStreaming.activeSession)
    }
}
