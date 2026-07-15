// Copyright © 2026 Apple Inc.

import Foundation
import MLXLLM
import Testing

@testable import MLXLMCommon

@Suite struct ThinkingSupportTests {
    @Test(
        "Codable round-trip preserves every thinking-support shape",
        arguments: [
            ThinkingSupport.none,
            .toggleableViaTemplate(contextKey: "enable_thinking"),
            .alwaysOn(startTag: "<think>", endTag: "</think>"),
        ])
    func codableRoundTrip(support: ThinkingSupport) throws {
        let encoded = try JSONEncoder().encode(support)
        let decoded = try JSONDecoder().decode(ThinkingSupport.self, from: encoded)

        #expect(decoded == support)
    }

    @Test func hashableConformancePreservesAssociatedValues() {
        let values: Set<ThinkingSupport> = [
            .toggleableViaTemplate(contextKey: "enable_thinking"),
            .toggleableViaTemplate(contextKey: "thinking"),
            .alwaysOn(startTag: "<think>", endTag: "</think>"),
        ]

        #expect(values.count == 3)
    }

    @Test func infersToggleFromInlineTokenizerTemplate() {
        let data = Data(
            #"{"chat_template":"{% if enable_thinking %}<think>{% endif %}"}"#.utf8)

        #expect(
            ThinkingSupport.infer(tokenizerConfiguration: data)
                == .toggleableViaTemplate(contextKey: "enable_thinking"))
    }

    @Test func infersToggleFromNamedTokenizerTemplates() {
        let data = Data(
            #"{"chat_template":[{"name":"default","template":"{{ messages }}"},{"name":"thinking","template":"{% if enable_thinking %}<think>{% endif %}"}]}"#
                .utf8)

        #expect(
            ThinkingSupport.infer(tokenizerConfiguration: data)
                == .toggleableViaTemplate(contextKey: "enable_thinking"))
    }

    @Test func infersToggleFromStandaloneTemplate() {
        let data = Data("{% set enabled = enable_thinking | default(true) %}".utf8)

        #expect(
            ThinkingSupport.infer(chatTemplate: data)
                == .toggleableViaTemplate(contextKey: "enable_thinking"))
    }

    @Test func ignoresUnstructuredMentionsOutsideChatTemplate() {
        let data = Data(
            #"{"model_note":"enable_thinking","chat_template":"{{ messages }}"}"#.utf8)

        #expect(ThinkingSupport.infer(tokenizerConfiguration: data) == nil)
    }

    @Test func tagsAloneDoNotImplyAlwaysOnThinking() {
        let data = Data("{{ messages }}<think></think>".utf8)

        #expect(ThinkingSupport.infer(chatTemplate: data) == nil)
    }

    @Test func explicitMetadataTakesPrecedenceOverTemplateInference() throws {
        let directory = try temporaryTokenizerDirectory(
            template: "{% if enable_thinking %}<think>{% endif %}")
        defer { try? FileManager.default.removeItem(at: directory) }

        let explicit = ThinkingSupport.alwaysOn(
            startTag: "<reasoning>", endTag: "</reasoning>")
        var configuration = ModelConfiguration(
            id: "test/model", thinkingSupport: explicit
        ).resolved(modelDirectory: directory, tokenizerDirectory: directory)

        configuration.inferThinkingSupportIfNeeded()

        #expect(configuration.thinkingSupport == explicit)
    }

    @Test func templateInferenceFillsMissingMetadata() throws {
        let directory = try temporaryTokenizerDirectory(
            template: "{% if enable_thinking %}<think>{% endif %}")
        defer { try? FileManager.default.removeItem(at: directory) }

        var configuration = ModelConfiguration(id: "test/model").resolved(
            modelDirectory: directory, tokenizerDirectory: directory)

        configuration.inferThinkingSupportIfNeeded()

        #expect(
            configuration.thinkingSupport
                == .toggleableViaTemplate(contextKey: "enable_thinking"))
    }

    @Test func registryDeclaresOnlyLocallyVerifiedModels() {
        let alwaysOn = ThinkingSupport.alwaysOn(
            startTag: "<think>", endTag: "</think>")

        #expect(
            LLMRegistry.qwen3_5_2b_4bit.thinkingSupport
                == .toggleableViaTemplate(contextKey: "enable_thinking"))
        #expect(LLMRegistry.deepSeekR1_7B_4bit.thinkingSupport == alwaysOn)
        #expect(LLMRegistry.deepseek_r1_4bit.thinkingSupport == alwaysOn)

        // These concrete checkpoints have no local evidence of thinking behavior.
        #expect(LLMRegistry.qwen3_6_27b_4bit.thinkingSupport == nil)
        #expect(LLMRegistry.glm4_9b_4bit.thinkingSupport == nil)
        #expect(LLMRegistry.nemotron_labs_diffusion_3b_4bit.thinkingSupport == nil)
    }

    private func temporaryTokenizerDirectory(template: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appending(
            component: "thinking-support-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data(template.utf8).write(
            to: directory.appending(component: "chat_template.jinja"))
        return directory
    }
}
