// Copyright © 2026 Apple Inc.

import Foundation
import Jinja
import Testing

@testable import MLXLMCommon

/// Renders the real K2-Horizon (IFM) chat template, shipped with every K2
/// checkpoint, the way the host tokenizer does: the raw dictionaries from a
/// `MessageGenerator` plus `additionalContext` keys merged into the Jinja
/// context. These pin the contract the structured API has to satisfy — which
/// fields the template demands on assistant turns, how the reasoning effort
/// and the tool-call dialect are selected — without a model or tokenizer.
@Suite("K2-Horizon chat template rendering")
struct K2HorizonChatTemplateTests {

    // MARK: - Fixture

    private static let bosToken = "<|ifm|begin_of_text|>"

    private static let template: Template = {
        let url = Bundle.module.url(forResource: "K2HorizonChatTemplate", withExtension: "jinja")!
        let source = try! String(contentsOf: url, encoding: .utf8)
        return try! Template(source)
    }()

    /// Mirrors `PreTrainedTokenizer.applyChatTemplate` in swift-transformers:
    /// `messages`, `add_generation_prompt`, optional `tools`, then the caller's
    /// `additionalContext`, then the special tokens from `tokenizer_config`.
    private func render(
        _ messages: [Chat.Message],
        generator: any MessageGenerator = DefaultMessageGenerator(),
        tools: [ToolSpec]? = nil,
        additionalContext: [String: any Sendable] = [:],
        addGenerationPrompt: Bool = true
    ) throws -> String {
        var context: [String: Value] = [
            "messages": try Value(any: generator.generate(messages: messages)),
            "add_generation_prompt": .boolean(addGenerationPrompt),
            "bos_token": .string(Self.bosToken),
        ]
        if let tools {
            context["tools"] = try Value(any: tools)
        }
        for (key, value) in additionalContext {
            context[key] = try Value(any: value)
        }
        return try Self.template.render(context)
    }

    private static let weatherTool: ToolSpec = [
        "type": "function",
        "function": [
            "name": "get_weather",
            "description": "Current weather for a city.",
            "parameters": [
                "type": "object",
                "properties": [
                    "city": ["type": "string", "description": "City name"] as [String: any Sendable]
                ] as [String: any Sendable],
                "required": ["city"],
            ] as [String: any Sendable],
        ] as [String: any Sendable],
    ]

    private static let twoTurns: [Chat.Message] = [
        .system("You are terse."),
        .user("Hi"),
        .assistant("Hello.", reasoningContent: "greeting back"),
        .user("Bye"),
    ]

    // MARK: - Assistant thinking field

    @Test("Prior reasoning renders inside <ifm|think> before the content")
    func priorReasoningIsRendered() throws {
        let prompt = try render(Self.twoTurns)

        #expect(prompt.hasPrefix(Self.bosToken + "<|ifm|im_start|>system\nYou are terse.<|ifm|im_end|>"))
        #expect(prompt.contains("<|ifm|im_start|>user\nHi<|ifm|im_end|>"))
        #expect(
            prompt.contains(
                "<|ifm|im_start|>assistant<ifm|think>\ngreeting back</ifm|think>Hello.<|ifm|im_end|>"))
        #expect(prompt.hasSuffix("<|ifm|im_start|>user\nBye<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n"))
    }

    @Test("An empty reasoning string is accepted and renders an empty thinking block")
    func emptyReasoningStringIsAccepted() throws {
        var turns = Self.twoTurns
        turns[2] = .assistant("Hello.", reasoningContent: "")

        let prompt = try render(turns)

        #expect(prompt.contains("<|ifm|im_start|>assistant<ifm|think>\n</ifm|think>Hello.<|ifm|im_end|>"))
    }

    @Test("The default generator leaves the field undefined and the template raises")
    func missingThinkingFieldRaises() {
        var turns = Self.twoTurns
        turns[2] = .assistant("Hello.")

        #expect {
            try render(turns)
        } throws: { error in
            String(describing: error).contains("missing a thinking field")
        }
    }

    @Test("The K2 generator declares the field on assistant turns that lack it")
    func k2GeneratorDeclaresMissingField() throws {
        var turns = Self.twoTurns
        turns[2] = .assistant("Hello.")

        let prompt = try render(turns, generator: K2HorizonMessageGenerator())

        #expect(prompt.contains("<|ifm|im_start|>assistant<ifm|think>\n</ifm|think>Hello.<|ifm|im_end|>"))
    }

    @Test("The K2 generator keeps caller-provided reasoning and other thinking fields")
    func k2GeneratorKeepsCallerFields() throws {
        var turns = Self.twoTurns
        turns[2] = Chat.Message(
            role: .assistant, content: "Hello.", additionalFields: ["think_fast": "quick"])

        let prompt = try render(turns, generator: K2HorizonMessageGenerator())

        #expect(prompt.contains("<ifm|think_fast>\nquick</ifm|think_fast>Hello."))
        #expect(!prompt.contains("<ifm|think>\n</ifm|think>Hello."))
    }

    @Test("A single user turn without any assistant history renders")
    func singleTurnRenders() throws {
        let prompt = try render([.user("Hi")])

        #expect(prompt == Self.bosToken + "<|ifm|im_start|>user\nHi<|ifm|im_end|><|ifm|im_start|>assistant\n<ifm|think>\n")
    }

    // MARK: - reasoning_effort

    @Test(
        "reasoning_effort selects the seeded thinking tag",
        arguments: [
            ("high", "<ifm|think>"),
            ("medium", "<ifm|think_fast>"),
            ("low", "<ifm|think_faster>"),
        ] as [(String, String)]
    )
    func reasoningEffortSelectsTag(effort: String, tag: String) throws {
        let prompt = try render(
            Self.twoTurns, additionalContext: ["reasoning_effort": effort])

        #expect(prompt.hasSuffix("<|ifm|im_start|>assistant\n" + tag + "\n"))
    }

    @Test("The default effort is high")
    func defaultEffortIsHigh() throws {
        let prompt = try render(Self.twoTurns)

        #expect(prompt.hasSuffix("<|ifm|im_start|>assistant\n<ifm|think>\n"))
    }

    @Test("An unknown reasoning_effort raises")
    func unknownEffortRaises() {
        #expect {
            try render(Self.twoTurns, additionalContext: ["reasoning_effort": "max"])
        } throws: { error in
            String(describing: error).contains("Unsupported reasoning_effort")
        }
    }

    // MARK: - Tools

    @Test("Tools are presented in the system turn and the default call dialect is xml")
    func toolsDefaultToXMLDialect() throws {
        let prompt = try render([.user("Weather?")], tools: [Self.weatherTool])

        #expect(prompt.contains("<|ifm|im_start|>system\n# Tools"))
        #expect(prompt.contains("<ifm|tools>\n## get_weather"))
        #expect(prompt.contains("<ifm|arg_key>$PARAMETER_NAME</ifm|arg_key>"))
        #expect(!prompt.contains(#"{\"name\": <function-name>"#))
    }

    @Test("tool_call_format json switches the call instructions to JSON")
    func toolCallFormatJSONSelectsJSONDialect() throws {
        let prompt = try render(
            [.user("Weather?")], tools: [Self.weatherTool],
            additionalContext: ["tool_call_format": "json"])

        #expect(prompt.contains("<ifm|tool_call>{\"name\": <function-name>, \"arguments\": <args-json-object>}</ifm|tool_call>"))
        #expect(!prompt.contains("<ifm|arg_key>$PARAMETER_NAME"))
    }

    @Test("A prior assistant tool call and its result render in the selected dialect")
    func toolCallRoundTripRenders() throws {
        let call = ToolCall(
            function: .init(name: "get_weather", arguments: ["city": "Paris"]), id: "call_1")
        let turns: [Chat.Message] = [
            .user("Weather in Paris?"),
            .assistant("", toolCalls: [call], reasoningContent: "need the tool"),
            .tool("{\"temp\": 21}", id: "call_1", name: "get_weather"),
        ]

        let xml = try render(turns, tools: [Self.weatherTool])
        #expect(
            xml.contains(
                "<ifm|think>\nneed the tool</ifm|think><ifm|tool_calls>\n<ifm|tool_call>get_weather\n<ifm|arg_key>city</ifm|arg_key>\n<ifm|arg_value>Paris</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls><|ifm|im_end|>"))
        #expect(xml.contains("<|ifm|im_start|>tool\n{\"temp\": 21}<|ifm|im_end|>"))

        let json = try render(
            turns, tools: [Self.weatherTool], additionalContext: ["tool_call_format": "json"])
        #expect(
            json.contains(
                "<ifm|tool_calls>\n<ifm|tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}</ifm|tool_call>\n</ifm|tool_calls><|ifm|im_end|>"))
    }

    @Test("The real template infers the K2 tool-call format")
    func templateInfersK2Format() throws {
        let url = Bundle.module.url(forResource: "K2HorizonChatTemplate", withExtension: "jinja")!
        let source = try String(contentsOf: url, encoding: .utf8)

        #expect(ToolCallFormat.inferred(fromChatTemplate: source) == .k2Horizon)
    }
}
