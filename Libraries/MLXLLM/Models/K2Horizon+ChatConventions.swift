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

    /// The template raises on any assistant turn without a thinking field, and
    /// the session records turns without one; this generator declares an empty
    /// `reasoning_content` where the caller did not provide any.
    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        K2HorizonMessageGenerator()
    }
}
