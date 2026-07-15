// Copyright © 2026 Apple Inc.

import Foundation
import MLXEmbedders
import MLXLLM
import MLXLMCommon
import MLXVLM
import Testing

/// Not a correctness test -- a generator. Prints the live compatibility
/// matrix (registered model_type identifiers + recommended models) derived
/// directly from LLMTypeRegistry/VLMTypeRegistry/EmbedderTypeRegistry and
/// LLMRegistry/VLMRegistry/EmbedderRegistry, so DOCS/compatibility-matrix.md
/// can be regenerated from source of truth instead of hand-maintained.
///
/// Run with:
/// ```
/// xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
///   -skipPackagePluginValidation \
///   -only-testing:MLXLMTests/CompatibilityMatrixGeneratorTests
/// ```
/// and capture stdout.
///
/// See DOCS/tech-debt-and-research-backlog.md, "Matriz de compatibilidad generada".
struct CompatibilityMatrixGeneratorTests {

    @Test func printCompatibilityMatrix() async {
        let llmTypes = await LLMTypeRegistry.shared.registeredModelTypes
        let vlmTypes = await VLMTypeRegistry.shared.registeredModelTypes
        let embedderTypes = await EmbedderTypeRegistry.shared.registeredModelTypes

        print("### LLM model_type identifiers (\(llmTypes.count) registered)")
        for type in llmTypes {
            print("- \(type)")
        }

        print("\n### VLM model_type identifiers (\(vlmTypes.count) registered)")
        for type in vlmTypes {
            print("- \(type)")
        }

        print("\n### Embedder model_type identifiers (\(embedderTypes.count) registered)")
        for type in embedderTypes {
            print("- \(type)")
        }

        let llmModels = LLMRegistry.shared.models.sorted { $0.name < $1.name }
        print("\n### LLM recommended models (\(llmModels.count) registered)")
        for model in llmModels {
            let toolFormat = model.toolCallFormat?.rawValue ?? "-"
            let eos = model.extraEOSTokens.isEmpty ? "-" : model.extraEOSTokens.sorted().joined(separator: ",")
            print("- \(model.name) | toolCallFormat=\(toolFormat) | extraEOS=\(eos)")
        }

        let vlmModels = VLMRegistry.shared.models.sorted { $0.name < $1.name }
        print("\n### VLM recommended models (\(vlmModels.count) registered)")
        for model in vlmModels {
            print("- \(model.name)")
        }

        let embedderModels = EmbedderRegistry.shared.models.sorted { $0.name < $1.name }
        print("\n### Embedder recommended models (\(embedderModels.count) registered)")
        for model in embedderModels {
            print("- \(model.name)")
        }

        #expect(!llmTypes.isEmpty)
        #expect(!vlmTypes.isEmpty)
        #expect(!embedderTypes.isEmpty)
    }
}
