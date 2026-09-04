// Copyright © 2026 Apple Inc.

import Foundation
import MLXLMCommon

/// Chat conventions of K2-Horizon (IFM) checkpoints, kept apart from the
/// architecture so the model file stays about the forward pass.
extension K2HorizonModel {
    /// Every K2 template dialect is framed by `<ifm|tool_calls>`; the parser
    /// accepts the JSON and the tagged payloads, so the prompt's
    /// `tool_call_format` (`json`, `xml`, `xml_typed`) is free per request.
    public var toolCallFormat: ToolCallFormat? { .k2Horizon }

    /// K2 always reasons: the generation prompt ends *inside* the thinking block
    /// (`…<|ifm|im_start|>assistant\n<ifm|think>\n`), so the model emits only the
    /// closing delimiter and a consumer waiting for `<ifm|think>` in the stream
    /// never sees one — seed it from the prompt instead.
    ///
    /// `reasoning_effort` is a speed knob, not an off switch, hence `alwaysOn`.
    /// It also picks the tag pair: `high` (the template default) uses
    /// `<ifm|think>`, `medium` `<ifm|think_fast>`, `low` `<ifm|think_faster>`.
    /// ``ReasoningConfig`` holds one pair, so this declares the `high` one. A
    /// host that asks for `medium` or `low` has to build the matching config
    /// itself — the faster tags cannot ride in `implicitEndDelimiters`, which
    /// keeps its markers in the stream as answer text.
    public var reasoningConfig: ReasoningConfig? {
        ReasoningConfig(
            startDelimiter: "<ifm|think>",
            endDelimiter: "</ifm|think>",
            promptStrategy: .alwaysOn,
            isSpecialToken: true)
    }

    /// The template raises on any assistant turn without a thinking field, and
    /// the session records turns without one; this generator declares an empty
    /// `reasoning_content` where the caller did not provide any.
    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        K2HorizonMessageGenerator()
    }
}
