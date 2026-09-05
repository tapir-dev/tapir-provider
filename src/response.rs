// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Completion primitives shared across Providers: token [`Usage`], the
//! [`FinishReason`], and the SDK-minted tool-call id.
//!
//! A completed generation is returned as an
//! [`AssistantMessage`](crate::message::AssistantMessage), which is itself a
//! [`Message`](crate::message::Message) variant; these are the small value types
//! it is built from.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic source for SDK-minted tool-call ids, unique within the process.
static NEXT_CALL_ID: AtomicU64 = AtomicU64::new(0);

/// Mint an SDK-owned tool-call id, unique within the process.
///
/// Used only as a fallback: a tool call carries the Provider's native id when
/// the wire protocol supplies one, and a minted id otherwise, so callers always
/// have a stable handle to correlate a result back to the call.
pub(crate) fn mint_call_id() -> String {
    format!("call_{}", NEXT_CALL_ID.fetch_add(1, Ordering::Relaxed))
}

/// Why the Provider stopped generating.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize,
)]
pub struct Usage {
    /// Uncached prompt tokens: input served neither from nor into the cache.
    pub input_tokens: u32,
    /// Tokens produced in the completion.
    pub output_tokens: u32,
    /// Prompt tokens served from the cache, disjoint from `input_tokens`.
    pub cache_read_tokens: u32,
    /// Prompt tokens written into the cache, disjoint from `input_tokens`.
    pub cache_write_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_defaults_to_zero() {
        let usage = Usage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.cache_read_tokens, 0);
        assert_eq!(usage.cache_write_tokens, 0);
    }

    #[test]
    fn minted_call_ids_are_present_and_distinct() {
        let first = mint_call_id();
        let second = mint_call_id();
        assert!(!first.is_empty());
        assert_ne!(first, second);
    }
}
