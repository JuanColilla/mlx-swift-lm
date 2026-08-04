// Copyright © 2026 Apple Inc.

import Foundation
import HuggingFace
import IntegrationTestHelpers
import MLXHuggingFace
import MLXLMCommon
import Testing
import Tokenizers

private let runNetworkConformance =
    ProcessInfo.processInfo.environment["MLX_RUN_CONFORMANCE_NETWORK"] == "1"

private let conformanceModels = IntegrationTestModels(
    downloader: #hubDownloader(),
    tokenizerLoader: #huggingFaceTokenizerLoader()
)

private let smolVLM2Configuration = ModelConfiguration(
    id: "mlx-community/SmolVLM2-500M-Video-Instruct-mlx",
    revision: "fa57db46815177fbdfd65cc85a2b3416a8332268",
    defaultPrompt: "Describe the image in English"
)

private let qwen35LongContextConfiguration = ModelConfiguration(
    id: "mlx-community/Qwen3.5-0.8B-4bit",
    revision: "da28692b5f139cb0ec58a356b437486b7dac7462",
    extraEOSTokens: ["<|im_end|>"]
)

@Suite(.serialized)
struct VLMConformanceIntegrationTests {

    @Test(.enabled(if: runNetworkConformance))
    func smolVLM2ImageGeneration() async throws {
        let container = try await conformanceModels.vlmContainer(for: smolVLM2Configuration)
        try await ChatSessionTests.visionModel(container: container)
    }

    @Test(
        .disabled(
            "Requires a pinned FastVLM checkpoint plus physical-device memory and output validation."
        )
    )
    func fastVLMImageGenerationCheckpointGate() {}

    @Test(
        .disabled(
            "Requires a compatible pinned Idefics3 checkpoint and physical multi-image validation."
        )
    )
    func idefics3MultiImageCheckpointGate() {}

    @Test(
        .disabled(
            "Gemma4 video preprocessing and its physical-device memory ceiling need checkpoint validation."
        )
    )
    func gemma4VideoInputCheckpointGate() {}
}

@Suite(.serialized)
struct LongContextConformanceIntegrationTests {

    @Test(.enabled(if: runNetworkConformance))
    func qwen35LongContextSentinel() async throws {
        let container = try await conformanceModels.llmContainer(
            for: qwen35LongContextConfiguration)
        let session = ChatSession(
            container,
            generateParameters: GenerateParameters(maxTokens: 32, temperature: 0)
        )
        let filler = Array(
            repeating: "This paragraph is context filler and does not change the secret key.",
            count: 900
        ).joined(separator: " ")
        let prompt = """
            Remember that the secret key is cobalt.
            \(filler)
            What is the secret key? Reply with only the key.
            """

        var response = ""
        for try await chunk in session.streamResponse(to: prompt) {
            response += chunk
        }

        #expect(response.lowercased().contains("cobalt"))
    }
}
