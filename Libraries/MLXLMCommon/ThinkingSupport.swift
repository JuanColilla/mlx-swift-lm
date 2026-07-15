// Copyright © 2026 Apple Inc.

import Foundation

/// Describes how a model exposes extended reasoning, when that behavior is known.
///
/// A `nil` ``ModelConfiguration/thinkingSupport`` means that support is unknown or has not
/// been declared. ``none`` is reserved for models that are explicitly known not to expose
/// thinking mode.
public enum ThinkingSupport: Sendable, Codable, Hashable {
    /// The model is explicitly known not to expose thinking mode.
    case none

    /// Thinking can be enabled or disabled through a chat-template context value.
    case toggleableViaTemplate(contextKey: String)

    /// Reasoning is always emitted between fixed tags and must be handled after decoding.
    case alwaysOn(startTag: String, endTag: String)

    /// Infers thinking support from the actual tokenizer template data shipped by a model.
    ///
    /// The inference is deliberately conservative. It recognizes the concrete
    /// `enable_thinking` Jinja variable used by supported templates, but does not infer
    /// always-on behavior from `<think>` tags alone: tags do not prove whether reasoning is
    /// optional, mandatory, or merely listed as tokenizer metadata.
    ///
    /// - Parameters:
    ///   - tokenizerConfiguration: Contents of `tokenizer_config.json`, when available.
    ///   - chatTemplate: Contents of a standalone `chat_template.jinja`, when available.
    /// - Returns: Inferred support, or `nil` when the available template data is inconclusive.
    public static func infer(
        tokenizerConfiguration: Data? = nil,
        chatTemplate: Data? = nil
    ) -> ThinkingSupport? {
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

        return .toggleableViaTemplate(contextKey: "enable_thinking")
    }

    package static func infer(fromTokenizerDirectory directory: URL) -> ThinkingSupport? {
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
            return [string]
        case let array as [Any]:
            return array.flatMap(strings(in:))
        case let dictionary as [String: Any]:
            return dictionary.values.flatMap(strings(in:))
        default:
            return []
        }
    }
}

extension ResolvedModelConfiguration {
    /// Preserves explicit metadata and only consults downloaded templates as a fallback.
    package mutating func inferThinkingSupportIfNeeded() {
        guard thinkingSupport == nil else { return }
        thinkingSupport = ThinkingSupport.infer(fromTokenizerDirectory: tokenizerDirectory)
    }
}
