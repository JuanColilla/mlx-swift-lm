// Copyright © 2026 Apple Inc.

import Foundation

extension ReasoningConfig {
    /// Infers a reasoning protocol from tokenizer templates when neither the
    /// caller, a registered resolver, nor the model itself declared one.
    ///
    /// The inference is deliberately conservative. The presence of
    /// `enable_thinking` identifies the Qwen-style toggle, while bare
    /// `<think>` tags are insufficient to distinguish optional from always-on
    /// reasoning.
    public static func infer(
        tokenizerConfiguration: Data? = nil,
        chatTemplate: Data? = nil
    ) -> ReasoningConfig? {
        var templates: [String] = []

        if let tokenizerConfiguration,
            let object = try? JSONSerialization.jsonObject(with: tokenizerConfiguration),
            let dictionary = object as? [String: Any],
            let chatTemplate = dictionary["chat_template"]
        {
            templates.append(contentsOf: strings(in: chatTemplate))
        }

        if let chatTemplate, let template = String(data: chatTemplate, encoding: .utf8) {
            templates.append(template)
        }

        guard templates.contains(where: { $0.contains("enable_thinking") }) else {
            return nil
        }
        return .thinkTagsWithEnableThinking
    }

    package static func infer(fromTokenizerDirectory directory: URL) -> ReasoningConfig? {
        let tokenizerConfiguration = try? Data(
            contentsOf: directory.appending(component: "tokenizer_config.json"))
        let chatTemplate = try? Data(
            contentsOf: directory.appending(component: "chat_template.jinja"))

        return infer(
            tokenizerConfiguration: tokenizerConfiguration,
            chatTemplate: chatTemplate)
    }

    private static func strings(in value: Any) -> [String] {
        switch value {
        case let string as String:
            [string]
        case let array as [Any]:
            array.flatMap(strings(in:))
        case let dictionary as [String: Any]:
            dictionary.values.flatMap(strings(in:))
        default:
            []
        }
    }
}
