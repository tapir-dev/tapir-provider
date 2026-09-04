// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The normalized [`CompletionResponse`] returned by a Provider.

use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic source for SDK-minted tool-call ids, unique within the process.
static NEXT_CALL_ID: AtomicU64 = AtomicU64::new(0);

/// Mint an SDK-owned tool-call id, always present and unique within the process.
///
/// Providers whose wire protocol carries a native call id record it separately
/// (see [`ToolCall::native_id`]); the minted id gives callers a stable handle
/// even when the Provider omits one.
pub(crate) fn mint_call_id() -> String {
    format!("call_{}", NEXT_CALL_ID.fetch_add(1, Ordering::Relaxed))
}

/// A single tool call the model asked to make.
///
/// Every call carries an SDK-minted [`id`](Self::id) that is always present, so
/// callers have a stable handle to correlate a result back to the call. The
/// Provider's own [`native_id`](Self::native_id) is kept alongside it when the
/// wire protocol supplies one, and left `None` otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// The SDK-minted id, always present.
    pub id: String,
    /// The Provider's native id for this call, if the wire protocol carried one.
    pub native_id: Option<String>,
    /// The name of the tool to invoke.
    pub name: String,
    /// The tool's arguments, as a JSON value.
    pub arguments: serde_json::Value,
}

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
    /// Tool calls the model asked to make, in the order they appeared; empty
    /// when the model produced no tool use.
    pub tool_calls: Vec<ToolCall>,
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

    #[test]
    fn minted_call_ids_are_present_and_distinct() {
        let first = mint_call_id();
        let second = mint_call_id();
        assert!(!first.is_empty());
        assert_ne!(first, second);
    }
}
