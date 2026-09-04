// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The object-safe [`Provider`] trait.

use crate::error::{Error, ErrorKind};
use crate::request::CompletionRequest;
use crate::response::CompletionResponse;
use crate::stream::StreamEvents;
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

    /// Stream a completion as ordered, incremental
    /// [`StreamEvent`](crate::stream::StreamEvent)s.
    ///
    /// Streaming is a Capability a Provider opts into: the default reports it as
    /// unsupported, so a Provider that only completes need not implement it. A
    /// caller who ignores the deltas can fold the stream through a
    /// [`StreamAccumulator`](crate::stream::StreamAccumulator) to recover the
    /// same completion [`complete`](Self::complete) would return.
    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamEvents, Error> {
        let _ = request;
        Err(Error::new(
            ErrorKind::Other,
            "this Provider does not support streaming",
        ))
    }
}

/// Forward through a shared handle so an `Arc<P>` (or `Arc<dyn Provider>`) is
/// itself a Provider, letting a decorator or a test hold onto the inner
/// Provider while it is also driven through the trait.
#[async_trait]
impl<T: Provider + ?Sized> Provider for std::sync::Arc<T> {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, Error> {
        (**self).complete(request).await
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamEvents, Error> {
        (**self).complete_stream(request).await
    }
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
