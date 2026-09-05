// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! tapir-provider: a uniform interface over LLM backends.
//!
//! A [`Provider`] is a pluggable adapter to one LLM backend. Every Provider
//! talks to the network through an injected [`HttpClient`], so it can be driven
//! without a network in tests. Requests and responses use a normalized message
//! model, and failures surface as a typed [`Error`] with an [`ErrorKind`].
//!
//! This is the crate spine: it ships the Anthropic and OpenAI Providers (each
//! behind its own feature) doing both a single, non-streaming completion and a
//! streamed one that yields incremental [`StreamEvent`]s, plus an
//! [`EmbeddingProvider`] for turning input texts into vectors. A [`Registry`]
//! selects a Provider by name, registering only the ones compiled in.
//!
//! # Example
//!
//! ```
//! # #[cfg(all(feature = "anthropic", feature = "test-utils"))]
//! # {
//! use std::sync::Arc;
//! use tapir_provider::{
//!     AnthropicProvider, CompletionOptions, Context, Credential, Message,
//!     Provider, http::MockHttpClient,
//! };
//!
//! let rt = tokio::runtime::Builder::new_current_thread()
//!     .enable_all()
//!     .build()
//!     .unwrap();
//! rt.block_on(async {
//!     let body = r#"{"content":[{"type":"text","text":"hi"}],
//!                    "stop_reason":"end_turn",
//!                    "usage":{"input_tokens":1,"output_tokens":1}}"#;
//!     let http = Arc::new(MockHttpClient::with_response(200, body));
//!     let provider =
//!         AnthropicProvider::new(http, Credential::api_key("sk-x"), "claude-3-5-sonnet");
//!
//!     let ctx = Context::new(vec![Message::user("hello")]);
//!     let response = provider.complete(&ctx, &CompletionOptions::default()).await.unwrap();
//!     assert_eq!(response.text_content(), "hi");
//! });
//! # }
//! ```

#![forbid(unsafe_code)]

/// Shared base64 encoder, compiled only for the Providers that inline image bytes.
#[cfg(any(feature = "anthropic", feature = "openai"))]
mod base64;

pub mod auth;
/// The runtime [`ModelRegistry`] holding the Catalog and turning a Model Entry
/// into a live Provider.
#[cfg(feature = "models")]
pub mod catalog;
pub mod credential;
pub mod embedding;
pub mod error;
pub mod http;
pub mod message;
pub mod model;
pub mod provider;
pub mod providers;
pub mod registry;
pub mod request;
pub mod response;
pub mod retry;
pub mod sse;
pub mod stream;
pub mod token_store;
/// Record/replay cassette test layer over the [`HttpClient`] transport seam.
#[cfg(feature = "test-utils")]
pub mod vcr;

pub use auth::{AuthSource, ResolvedAuth};
#[cfg(feature = "models-user-config")]
pub use catalog::user_config_path;
#[cfg(feature = "models")]
pub use catalog::{ModelRegistry, create_provider};
pub use credential::{Credential, OAuthTokens};
pub use embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
pub use error::{Error, ErrorKind};
pub use http::{ByteStream, HttpClient, HttpRequest, HttpResponse, Method};
pub use message::{
    AssistantMessage, ContentPart, ImageSource, MediaType, Message,
    ToolResultMessage,
};
#[cfg(feature = "models")]
pub use model::{
    Api, CompatConfig, Dialect, InputType, Model, ModelCost, ModelEntry,
};
pub use model::{ModelId, ProviderId};
pub use provider::Provider;
pub use registry::{ProviderInfo, Registry};
pub use request::{
    CompletionOptions, Context, HeaderTransform, SystemPrompt, ThinkingLevel,
    ToolChoice, ToolDefinition,
};
pub use response::{FinishReason, Usage};
pub use retry::{Clock, RetryPolicy, RetryProvider, SystemClock};
pub use sse::{SseDecoder, SseEvent};
pub use stream::{StreamAccumulator, StreamEvent, StreamEvents};
#[cfg(feature = "token-store-file")]
pub use token_store::{DEFAULT_REFRESH_WINDOW_SECS, FileTokenStore};
pub use token_store::{InMemoryTokenStore, Refresh, TokenStore, resolve};
#[cfg(feature = "test-utils")]
pub use vcr::{Redactor, VcrClient, VcrMode};

#[cfg(feature = "anthropic")]
pub use providers::anthropic::oauth::{
    AnthropicOAuth, AuthorizationCode, OAuthLogin, OAuthMode, Redirect,
    capture_localhost,
};
#[cfg(feature = "anthropic")]
pub use providers::{AnthropicBuilder, AnthropicProvider};

#[cfg(feature = "openai")]
pub use providers::{OpenAIBuilder, OpenAIEmbeddingProvider, OpenAIProvider};
