// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon
import Testing

@Suite struct ReasoningTemplateInferenceTests {
    @Test func infersEnableThinkingFromTokenizerConfiguration() {
        let data = Data(
            #"{"chat_template":"{% if enable_thinking %}<think>{% endif %}"}"#.utf8)

        #expect(
            ReasoningConfig.infer(tokenizerConfiguration: data)
                == .thinkTagsWithEnableThinking)
    }

    @Test func infersEnableThinkingFromStandaloneTemplate() {
        let data = Data("{% set enabled = enable_thinking | default(true) %}".utf8)

        #expect(
            ReasoningConfig.infer(chatTemplate: data)
                == .thinkTagsWithEnableThinking)
    }

    @Test func doesNotInferFromBareThinkingTags() {
        let data = Data("<think>{{ content }}</think>".utf8)

        #expect(ReasoningConfig.infer(chatTemplate: data) == nil)
    }
}
