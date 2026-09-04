// Copyright © 2026 Apple Inc.

import Foundation

/// Parser for the K2-Horizon (IFM) tool-call block.
///
/// The model wraps every call of a turn in one
/// `<ifm|tool_calls>…</ifm|tool_calls>` block, and writes each call inside
/// `<ifm|tool_call>…</ifm|tool_call>` in whichever dialect the prompt's
/// `tool_call_format` selected:
///
/// - `json`: `{"name": "f", "arguments": {...}}`
/// - `xml` / `xml_typed`: `f` followed by `<ifm|arg_key>k</ifm|arg_key>` /
///   `<ifm|arg_value>v</ifm|arg_value>` pairs (`xml_typed` adds an
///   `<ifm|arg_type>` tag per argument, which is honored as a type hint).
///
/// The dialect is decided per request while the parser is picked once per
/// model, so a single parser accepts both. Selection is structural: a payload
/// starting with `{` is JSON, anything else is the tagged dialect.
///
/// The block tags are the ``startTag`` / ``endTag`` so the streaming processor
/// buffers the whole block: a per-call start tag would make the processor read
/// the opening `<ifm|tool_calls>` as a malformed `<ifm|tool_call>` and reject
/// the turn. ``parseAll(content:tools:)`` returns every call in the block.
public struct K2HorizonToolCallParser: ToolCallParser, Sendable {
    public let startTag: String? = "<ifm|tool_calls>"
    public let endTag: String? = "</ifm|tool_calls>"

    static let callStartTag = "<ifm|tool_call>"
    static let callEndTag = "</ifm|tool_call>"

    private let jsonParser = JSONToolCallParser(startTag: callStartTag, endTag: callEndTag)

    public init() {}

    /// The first call in the block; the processor uses ``parseAll(content:tools:)``
    /// for a complete block, this is the single-call contract of ``ToolCallParser``.
    public func parse(content: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        parseAll(content: content, tools: tools).first
    }

    public func parseAll(content: String, tools: [[String: any Sendable]]?) -> [ToolCall] {
        callPayloads(in: content).compactMap { parseCall($0, tools: tools) }
    }

    public func parseEOS(_ toolCallBuffer: String, tools: [[String: any Sendable]]?) -> [ToolCall]
    {
        parseAll(content: toolCallBuffer, tools: tools)
    }

    /// The body of every complete `<ifm|tool_call>…</ifm|tool_call>` in the text,
    /// whether or not the surrounding block tags are present.
    private func callPayloads(in content: String) -> [String] {
        var payloads: [String] = []
        var searchRange = content.startIndex ..< content.endIndex
        while let start = content.range(of: Self.callStartTag, range: searchRange) {
            guard
                let end = content.range(
                    of: Self.callEndTag, range: start.upperBound ..< content.endIndex)
            else { break }
            payloads.append(
                String(content[start.upperBound ..< end.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines))
            searchRange = end.upperBound ..< content.endIndex
        }
        return payloads
    }

    private func parseCall(_ payload: String, tools: [[String: any Sendable]]?) -> ToolCall? {
        guard !payload.isEmpty else { return nil }
        if payload.hasPrefix("{") {
            return jsonParser.parse(content: payload, tools: tools)
        }
        return parseTaggedCall(payload, tools: tools)
    }

    /// `name\n<ifm|arg_key>k</ifm|arg_key>[<ifm|arg_type>t</ifm|arg_type>]<ifm|arg_value>v</ifm|arg_value>…`
    ///
    /// Mirrors the GLM4 rule the template shares: string parameters are
    /// written as plain text and everything else as a JSON literal. A declared
    /// schema type (or the `xml_typed` hint) decides whether a value is kept as
    /// text; without either, a value that parses as JSON is deserialized.
    private func parseTaggedCall(_ payload: String, tools: [[String: any Sendable]]?)
        -> ToolCall?
    {
        let nameEnd = payload.range(of: "<ifm|arg_key>")?.lowerBound ?? payload.endIndex
        let name = String(payload[..<nameEnd]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, !name.contains(where: \.isWhitespace), !name.contains("<") else {
            return nil
        }

        var arguments: [String: any Sendable] = [:]
        var searchRange = nameEnd ..< payload.endIndex
        while let keyStart = payload.range(of: "<ifm|arg_key>", range: searchRange) {
            guard
                let keyEnd = payload.range(
                    of: "</ifm|arg_key>", range: keyStart.upperBound ..< payload.endIndex),
                let valueStart = payload.range(
                    of: "<ifm|arg_value>", range: keyEnd.upperBound ..< payload.endIndex),
                let valueEnd = payload.range(
                    of: "</ifm|arg_value>", range: valueStart.upperBound ..< payload.endIndex)
            else { return nil }

            let key = String(payload[keyStart.upperBound ..< keyEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let value = String(payload[valueStart.upperBound ..< valueEnd.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let typeHint = tagContent(
                "<ifm|arg_type>", "</ifm|arg_type>",
                in: payload[keyEnd.upperBound ..< valueStart.lowerBound])

            arguments[key] = convert(
                value, key: key, function: name, typeHint: typeHint, tools: tools)
            searchRange = valueEnd.upperBound ..< payload.endIndex
        }

        return ToolCall(function: .init(name: name, arguments: arguments))
    }

    private func convert(
        _ value: String, key: String, function: String, typeHint: String?,
        tools: [[String: any Sendable]]?
    ) -> any Sendable {
        let declaredType = getParameterType(funcName: function, paramName: key, tools: tools)
        if (declaredType ?? typeHint) == "string" {
            return value
        }
        return tryParseJSON(value) ?? value
    }

    private func tagContent(_ open: String, _ close: String, in text: Substring) -> String? {
        guard let start = text.range(of: open),
            let end = text.range(of: close, range: start.upperBound ..< text.endIndex)
        else { return nil }
        return String(text[start.upperBound ..< end.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
