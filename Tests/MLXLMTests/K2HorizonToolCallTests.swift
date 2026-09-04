// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXLMCommon

@Suite("K2-Horizon tool calls")
struct K2HorizonToolCallTests {

    private static let jsonBlock = """
        <ifm|tool_calls>
        <ifm|tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</ifm|tool_call>
        </ifm|tool_calls>
        """

    private static let twoCallBlock = """
        <ifm|tool_calls>
        <ifm|tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</ifm|tool_call>
        <ifm|tool_call>get_time
        <ifm|arg_key>zone</ifm|arg_key>
        <ifm|arg_value>Europe/Paris</ifm|arg_value>
        <ifm|arg_key>offset</ifm|arg_key>
        <ifm|arg_value>2</ifm|arg_value>
        </ifm|tool_call>
        </ifm|tool_calls>
        """

    private static let weatherTools: [[String: any Sendable]] = [
        [
            "type": "function",
            "function": [
                "name": "get_weather",
                "parameters": [
                    "type": "object",
                    "properties": ["code": ["type": "string"] as [String: any Sendable]]
                        as [String: any Sendable],
                ] as [String: any Sendable],
            ] as [String: any Sendable],
        ]
    ]

    // MARK: - Parser

    @Test("Parses a JSON call inside the block")
    func parsesJSONDialect() throws {
        let parser = K2HorizonToolCallParser()

        let call = try #require(parser.parse(content: Self.jsonBlock, tools: nil))

        #expect(call.function.name == "get_weather")
        #expect(call.function.arguments["city"] == .string("Paris"))
    }

    @Test("Parses the tagged dialect and deserializes non-string values")
    func parsesTaggedDialect() throws {
        let parser = K2HorizonToolCallParser()
        let content = """
            <ifm|tool_call>get_time
            <ifm|arg_key>zone</ifm|arg_key>
            <ifm|arg_value>Europe/Paris</ifm|arg_value>
            <ifm|arg_key>offset</ifm|arg_key>
            <ifm|arg_value>2</ifm|arg_value>
            <ifm|arg_key>tags</ifm|arg_key>
            <ifm|arg_value>["a", "b"]</ifm|arg_value>
            </ifm|tool_call>
            """

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.name == "get_time")
        #expect(call.function.arguments["zone"] == .string("Europe/Paris"))
        #expect(call.function.arguments["offset"] == .int(2))
        #expect(call.function.arguments["tags"] == .array([.string("a"), .string("b")]))
    }

    @Test("A declared string parameter keeps a numeric-looking value as text")
    func declaredStringTypeWins() throws {
        let parser = K2HorizonToolCallParser()
        let content = """
            <ifm|tool_call>get_weather
            <ifm|arg_key>code</ifm|arg_key>
            <ifm|arg_value>75001</ifm|arg_value>
            </ifm|tool_call>
            """

        let call = try #require(parser.parse(content: content, tools: Self.weatherTools))

        #expect(call.function.arguments["code"] == .string("75001"))
    }

    @Test("The xml_typed dialect honors its type hint")
    func typedDialectHonorsHint() throws {
        let parser = K2HorizonToolCallParser()
        let content = """
            <ifm|tool_call>lookup
            <ifm|arg_key>id</ifm|arg_key>
            <ifm|arg_type>string</ifm|arg_type>
            <ifm|arg_value>42</ifm|arg_value>
            <ifm|arg_key>limit</ifm|arg_key>
            <ifm|arg_type>integer</ifm|arg_type>
            <ifm|arg_value>10</ifm|arg_value>
            </ifm|tool_call>
            """

        let call = try #require(parser.parse(content: content, tools: nil))

        #expect(call.function.arguments["id"] == .string("42"))
        #expect(call.function.arguments["limit"] == .int(10))
    }

    @Test("parseAll returns every call of a mixed-dialect block in order")
    func parseAllReturnsEveryCall() {
        let calls = K2HorizonToolCallParser().parseAll(content: Self.twoCallBlock, tools: nil)

        #expect(calls.map(\.function.name) == ["get_weather", "get_time"])
        #expect(calls.last?.function.arguments["offset"] == .int(2))
    }

    @Test("Malformed payloads are not repaired into calls")
    func malformedPayloadsAreRejected() {
        let parser = K2HorizonToolCallParser()

        #expect(parser.parse(content: "<ifm|tool_call></ifm|tool_call>", tools: nil) == nil)
        #expect(parser.parse(content: "<ifm|tool_call>{\"name\": }</ifm|tool_call>", tools: nil) == nil)
        #expect(
            parser.parse(
                content: "<ifm|tool_call>bad name\n<ifm|arg_key>a</ifm|arg_key><ifm|arg_value>1</ifm|arg_value></ifm|tool_call>",
                tools: nil) == nil)
        #expect(parser.parse(content: "<ifm|tool_call>{\"name\": \"f\"}", tools: nil) == nil)
    }

    // MARK: - Streaming processor

    @Test("The block is parsed by the processor without leaking protocol text")
    func processorParsesWholeBlock() throws {
        let processor = ToolCallProcessor(format: .k2Horizon)

        let visible = processor.processChunk("Sure. " + Self.jsonBlock)
        processor.processEOS()

        #expect(visible == "Sure. ")
        #expect(processor.rejectedToolCalls.isEmpty)
        #expect(processor.toolCalls.count == 1)
        #expect(processor.toolCalls.first?.function.name == "get_weather")
    }

    @Test("Character-by-character streaming yields both calls of a block and no rejection")
    func processorStreamsCharacterByCharacter() throws {
        let processor = ToolCallProcessor(format: .k2Horizon)
        var visible = ""
        for character in "Checking. " + Self.twoCallBlock + " Done." {
            visible += processor.processChunk(String(character)) ?? ""
        }
        processor.processEOS()

        #expect(visible == "Checking.  Done.")
        #expect(!visible.contains("ifm|"))
        #expect(processor.rejectedToolCalls.isEmpty)
        #expect(processor.toolCalls.map(\.function.name) == ["get_weather", "get_time"])
        #expect(processor.toolCalls.last?.function.arguments["zone"] == .string("Europe/Paris"))
    }

    @Test("A chunk boundary inside the opening block tag does not produce a rejection")
    func chunkBoundaryInsideOpeningTag() throws {
        let processor = ToolCallProcessor(format: .k2Horizon)
        let block = Self.jsonBlock
        let split = block.index(block.startIndex, offsetBy: "<ifm|tool_call".count)

        var outputs = processor.processChunkOutputs(String(block[..<split]))
        outputs += processor.processChunkOutputs(String(block[split...]))
        outputs += processor.processEOSOutputs()

        #expect(processor.rejectedToolCalls.isEmpty)
        #expect(outputs.count == 1)
        guard case .toolCall(let call)? = outputs.first else {
            Issue.record("expected a tool call, got \(outputs)")
            return
        }
        #expect(call.function.name == "get_weather")
    }

    @Test("Ordered outputs keep text and both calls in source order")
    func orderedOutputsKeepSourceOrder() {
        let processor = ToolCallProcessor(format: .k2Horizon)

        var outputs = processor.processChunkOutputs("Let me check.")
        outputs += processor.processChunkOutputs(Self.twoCallBlock)
        outputs += processor.processEOSOutputs()

        let names = outputs.compactMap { output -> String? in
            if case .toolCall(let call) = output { return call.function.name }
            return nil
        }
        #expect(names == ["get_weather", "get_time"])
        #expect(outputs.first == .response("Let me check."))
        #expect(processor.rejectedToolCalls.isEmpty)
    }

    @Test("An unterminated block at EOS still yields the complete calls inside it")
    func unterminatedBlockAtEOS() {
        let processor = ToolCallProcessor(format: .k2Horizon)
        let truncated = String(Self.twoCallBlock.dropLast("</ifm|tool_calls>".count))

        _ = processor.processChunk(truncated)
        processor.processEOS()

        #expect(processor.toolCalls.map(\.function.name) == ["get_weather", "get_time"])
    }

    @Test("An undeclared tool inside the block is rejected, the declared one accepted")
    func undeclaredToolIsRejected() {
        let processor = ToolCallProcessor(format: .k2Horizon, tools: Self.weatherTools)

        _ = processor.processChunk(Self.twoCallBlock)
        processor.processEOS()

        #expect(processor.toolCalls.map(\.function.name) == ["get_weather"])
        #expect(processor.rejectedToolCalls.map(\.reason) == [.undeclaredTool])
    }

    // MARK: - Format plumbing

    @Test("The format round-trips its raw value and builds the K2 parser")
    func formatPlumbing() {
        #expect(ToolCallFormat.k2Horizon.rawValue == "k2_horizon")
        #expect(ToolCallFormat(rawValue: "k2_horizon") == .k2Horizon)
        #expect(ToolCallFormat.k2Horizon.createParser() is K2HorizonToolCallParser)
        #expect(ToolCallFormat.k2Horizon.createParser().startTag == "<ifm|tool_calls>")
    }

    @Test("A sidecar K2 template selects the K2 format from disk")
    func sidecarTemplateSelectsFormat() throws {
        let directory = try TokenizerFixture.make([
            "tokenizer_config.json": #"{"eos_token": "<|ifm|im_end|>"}"#,
            "chat_template.jinja": "<ifm|tool_calls>\n<ifm|tool_call>{{ name }}</ifm|tool_call>",
        ])
        defer { TokenizerFixture.remove(directory) }

        #expect(ToolCallFormat.resolved(forTokenizerDirectory: directory) == .k2Horizon)
    }

    // MARK: - Chat.Message additional fields

    @Test("Additional fields reach the raw message and cannot override the structured keys")
    func additionalFieldsReachRawMessage() throws {
        var message = Chat.Message.assistant("Hi", reasoningContent: "why")
        message.additionalFields["custom"] = 3
        message.additionalFields["role"] = "user"

        let raw = DefaultMessageGenerator().generate(message: message)

        #expect(raw["role"] as? String == "assistant")
        #expect(raw["content"] as? String == "Hi")
        #expect(raw["reasoning_content"] as? String == "why")
        #expect(raw["custom"] as? Int == 3)
        #expect(message.reasoningContent == "why")
    }

    @Test("A nil reasoning content leaves the key undefined, an empty one defines it")
    func reasoningContentNilVersusEmpty() {
        let undefined = DefaultMessageGenerator().generate(message: .assistant("Hi"))
        #expect(undefined["reasoning_content"] == nil)

        var cleared = Chat.Message.assistant("Hi", reasoningContent: "x")
        cleared.reasoningContent = nil
        #expect(cleared.additionalFields["reasoning_content"] == nil)

        let empty = DefaultMessageGenerator().generate(message: .assistant("Hi", reasoningContent: ""))
        #expect(empty["reasoning_content"] as? String == "")
    }

    @Test("The K2 generator only touches assistant turns")
    func k2GeneratorOnlyTouchesAssistantTurns() {
        let generator = K2HorizonMessageGenerator()

        #expect(generator.generate(message: .user("Hi"))["reasoning_content"] == nil)
        #expect(generator.generate(message: .system("S"))["reasoning_content"] == nil)
        #expect(generator.generate(message: .tool("r"))["reasoning_content"] == nil)
        #expect(generator.generate(message: .assistant("A"))["reasoning_content"] as? String == "")
        #expect(
            generator.generate(message: .assistant("A", reasoningContent: "kept"))["reasoning_content"]
                as? String == "kept")
    }
}
