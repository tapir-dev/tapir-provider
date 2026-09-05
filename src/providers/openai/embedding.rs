// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The OpenAI [`EmbeddingProvider`], talking to the Embeddings API and
//! authenticating with a `Bearer` Credential.

use super::{API_KEY_ENV, DEFAULT_BASE_URL, PROVIDER_KEY, bearer};
use crate::credential::Credential;
use crate::embedding::{
    EmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
};
use crate::error::{Error, ErrorKind};
use crate::http::{HttpClient, HttpRequest, Method};
use crate::response::Usage;
use crate::token_store::{TokenStore, resolve};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A Provider for OpenAI's Embeddings API.
///
/// The transport is injected as the generic `H`. The addressable embedding Model
/// (for example `text-embedding-3-small`) is fixed when the Provider is built.
#[derive(Clone)]
pub struct OpenAIEmbeddingProvider<H> {
    http: H,
    credential: Credential,
    model: String,
    base_url: String,
}

impl<H: fmt::Debug> fmt::Debug for OpenAIEmbeddingProvider<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAIEmbeddingProvider")
            .field("http", &self.http)
            .field("credential", &self.credential)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish()
    }
}

impl<H: HttpClient> OpenAIEmbeddingProvider<H> {
    /// Build an embedding Provider for the given Model, authenticating with
    /// `credential` over the injected transport.
    ///
    /// The explicit `credential` routes through the same [`resolve`] rule every
    /// construction path uses, so this is infallible.
    pub fn new(
        http: H,
        credential: Credential,
        model: impl Into<String>,
    ) -> Self {
        let credential =
            resolve(Some(credential), None, PROVIDER_KEY, API_KEY_ENV)
                .ok()
                .flatten()
                .expect("an explicit Credential always resolves");
        Self {
            http,
            credential,
            model: model.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
        }
    }

    /// Build an embedding Provider by resolving its Credential from a Token
    /// Store, then the `OPENAI_API_KEY` environment variable.
    ///
    /// Precedence follows [`resolve`]: an `explicit` Credential wins, else the
    /// `store` under this Provider's key, else the environment. An empty result
    /// is an [`Authentication`](crate::ErrorKind::Authentication) error.
    pub fn resolve(
        http: H,
        model: impl Into<String>,
        explicit: Option<Credential>,
        store: Option<&dyn TokenStore>,
    ) -> Result<Self, Error> {
        let credential = resolve(explicit, store, PROVIDER_KEY, API_KEY_ENV)?
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Authentication,
                    format!(
                        "no OpenAI Credential: pass one, store it under {PROVIDER_KEY:?}, or set {API_KEY_ENV}"
                    ),
                )
            })?;
        Ok(Self::new(http, credential, model))
    }

    /// Override the base URL (for proxies, gateways, or a test server).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }
}

#[async_trait]
impl<H: HttpClient> EmbeddingProvider for OpenAIEmbeddingProvider<H> {
    async fn embed(
        &self,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse, Error> {
        let body = serde_json::to_vec(&WireRequest {
            model: &self.model,
            input: &request.input,
        })
        .map_err(Error::serialize)?;

        let url =
            format!("{}/v1/embeddings", self.base_url.trim_end_matches('/'));
        let http_request = HttpRequest::new(Method::Post, url)
            .header("content-type", "application/json")
            .header("authorization", bearer(&self.credential))
            .body(body);

        let response = self.http.send(http_request).await?;
        if !response.is_success() {
            return Err(crate::http::error_from_response(&response));
        }

        let wire: WireResponse =
            serde_json::from_slice(&response.body).map_err(Error::decode)?;
        let raw: serde_json::Value =
            serde_json::from_slice(&response.body).map_err(Error::decode)?;

        Ok(wire.into_response(raw))
    }
}

/// The OpenAI embeddings request body as sent on the wire.
#[derive(Debug, Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

/// The OpenAI embeddings response body.
#[derive(Debug, Deserialize)]
struct WireResponse {
    #[serde(default)]
    data: Vec<WireEmbedding>,
    #[serde(default)]
    usage: WireUsage,
}

impl WireResponse {
    fn into_response(mut self, raw: serde_json::Value) -> EmbeddingResponse {
        // The API returns each embedding tagged with its input index; order by
        // it so vectors line up with the request inputs regardless of arrival
        // order.
        self.data.sort_by_key(|item| item.index);
        EmbeddingResponse {
            vectors: self.data.into_iter().map(|item| item.embedding).collect(),
            usage: Usage {
                input_tokens: self.usage.prompt_tokens,
                output_tokens: 0,
            },
            raw,
        }
    }
}

/// One embedding in the OpenAI response, tagged with its input index.
#[derive(Debug, Deserialize)]
struct WireEmbedding {
    #[serde(default)]
    index: usize,
    #[serde(default)]
    embedding: Vec<f32>,
}

/// Token usage in the OpenAI embeddings response; embeddings bill input only.
#[derive(Debug, Default, Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u32,
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use crate::http::MockHttpClient;
    use std::sync::Arc;

    const SAMPLE_RESPONSE: &str = r#"{
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
        ],
        "usage": {"prompt_tokens": 5, "total_tokens": 5}
    }"#;

    #[tokio::test]
    async fn produces_vectors_in_input_order_through_the_double() {
        let mock =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = OpenAIEmbeddingProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "text-embedding-3-small",
        );

        let response = provider
            .embed(EmbeddingRequest::new(vec![
                "first".to_owned(),
                "second".to_owned(),
            ]))
            .await
            .unwrap();

        // Vectors are re-ordered by index to match the inputs.
        assert_eq!(response.vectors, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
        assert_eq!(response.usage.input_tokens, 5);
        assert_eq!(response.raw["model"], "text-embedding-3-small");

        // Bearer auth and the input reached the wire.
        let sent = mock.last_request();
        assert!(sent.url.ends_with("/v1/embeddings"));
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "authorization" && v == "Bearer sk-test")
        );
        let body: serde_json::Value =
            serde_json::from_slice(sent.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["model"], "text-embedding-3-small");
        assert_eq!(body["input"][0], "first");
        assert_eq!(body["input"][1], "second");
    }

    #[tokio::test]
    async fn non_2xx_maps_to_a_typed_error_kind() {
        let mock = Arc::new(MockHttpClient::with_response(
            401,
            r#"{"error":{"message":"invalid key"}}"#,
        ));
        let provider = OpenAIEmbeddingProvider::new(
            mock,
            Credential::api_key("bad"),
            "text-embedding-3-small",
        );

        let err = provider
            .embed(EmbeddingRequest::new(vec!["hi".to_owned()]))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Authentication);
    }
}
