// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The object-safe [`Provider`] trait.

use crate::error::Error;
use crate::request::CompletionRequest;
use crate::response::CompletionResponse;
use async_trait::async_trait;

/// A pluggable adapter to one LLM backend.
///
/// The trait is object-safe so Providers can be held as `Arc<dyn Provider>` and
/// supplied by third parties. The transport an implementation uses is its own
/// concern, injected on the concrete struct rather than surfaced here.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Produce a single, non-streaming completion for the given request.
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time proof that `Provider` is object-safe.
    const _: fn(&dyn Provider) = |_| {};

    #[test]
    fn provider_is_usable_as_a_trait_object() {
        fn assert_object_safe(_: std::sync::Arc<dyn Provider>) {}
        let _ = assert_object_safe;
    }
}
