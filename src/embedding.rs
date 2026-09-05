// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`EmbeddingProvider`] trait and its normalized request and response.
//!
//! Embeddings are a Capability kept off the [`Provider`](crate::provider::Provider)
//! trait: not every backend produces them, and a caller that only completes need
//! not depend on the surface. A Provider that does supply them implements this
//! trait, so it can be held as `Arc<dyn EmbeddingProvider>` alongside the rest.

use crate::error::Error;
use crate::response::Usage;
use async_trait::async_trait;

/// A normalized request for embeddings over one or more input texts.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EmbeddingRequest {
    /// The input texts to embed, each mapped to a vector in the same order.
    pub input: Vec<String>,
}

impl EmbeddingRequest {
    /// Construct a request from a list of input texts.
    pub fn new(input: impl Into<Vec<String>>) -> Self {
        Self {
            input: input.into(),
        }
    }
}

/// A normalized embeddings response.
///
/// The `vectors` are returned in the same order as the request's inputs; the
/// normalized fields cover the common case and [`EmbeddingResponse::raw`] is the
/// escape hatch to the Provider's untouched response body.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingResponse {
    /// One embedding vector per input text, in input order.
    pub vectors: Vec<Vec<f32>>,
    /// Token accounting; embeddings consume input tokens only.
    pub usage: Usage,
    /// The Provider's untouched response body.
    pub raw: serde_json::Value,
}

/// A pluggable adapter that turns input texts into embedding vectors.
///
/// Object-safe, like [`Provider`](crate::provider::Provider): implementations
/// can be held as `Arc<dyn EmbeddingProvider>` and supplied by third parties.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Produce an embedding vector for each input text, in input order.
    async fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, Error>;
}

/// Forward through a shared handle so an `Arc<E>` (or `Arc<dyn EmbeddingProvider>`)
/// is itself an EmbeddingProvider, mirroring the [`Provider`](crate::provider::Provider)
/// blanket impl.
#[async_trait]
impl<T: EmbeddingProvider + ?Sized> EmbeddingProvider for std::sync::Arc<T> {
    async fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, Error> {
        (**self).embed(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time proof that `EmbeddingProvider` is object-safe.
    const _: fn(&dyn EmbeddingProvider) = |_| {};

    #[test]
    fn request_carries_its_inputs_in_order() {
        let request =
            EmbeddingRequest::new(vec!["a".to_owned(), "b".to_owned()]);
        assert_eq!(request.input, vec!["a".to_owned(), "b".to_owned()]);
    }
}
