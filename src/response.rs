// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The normalized [`CompletionResponse`] returned by a Provider.

/// Why the Provider stopped generating.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum FinishReason {
    /// The model reached a natural stopping point.
    Stop,
    /// Generation hit the `max_tokens` cap.
    MaxTokens,
    /// A configured stop sequence was produced.
    StopSequence,
    /// The model asked to call a tool.
    ToolUse,
    /// A Provider-specific reason not covered above, carrying its raw label.
    Other(String),
}

/// Token accounting for a completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    /// Tokens consumed by the prompt.
    pub input_tokens: u32,
    /// Tokens produced in the completion.
    pub output_tokens: u32,
}

/// A normalized, non-streaming completion.
///
/// The normalized fields cover the common case; [`CompletionResponse::raw`] is
/// the escape hatch to the Provider's untouched response body.
#[derive(Debug, Clone, PartialEq)]
pub struct CompletionResponse {
    /// The generated text, with any Provider content blocks concatenated.
    pub text: String,
    /// Token accounting.
    pub usage: Usage,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// The Provider's untouched response body.
    pub raw: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_defaults_to_zero() {
        let usage = Usage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
    }
}
