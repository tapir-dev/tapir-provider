// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! tapir-provider: a uniform interface over LLM backends.
//!
//! A [`Provider`] is a pluggable adapter to one LLM backend. Every Provider
//! talks to the network through an injected [`HttpClient`], so it can be driven
//! without a network in tests. Requests and responses use a normalized message
//! model, and failures surface as a typed [`Error`] with an [`ErrorKind`].
//!
//! This is the crate spine: today it ships the Anthropic Provider (behind the
//! `anthropic` feature) doing a single, non-streaming completion.
//!
//! # Example
//!
//! ```
//! # #[cfg(all(feature = "anthropic", feature = "test-utils"))]
//! # {
//! use std::sync::Arc;
//! use tapir_provider::{
//!     AnthropicProvider, Credential, CompletionRequest, Message, Provider,
//!     http::MockHttpClient,
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
//!     let request = CompletionRequest::new(vec![Message::user("hello")]);
//!     let response = provider.complete(request).await.unwrap();
//!     assert_eq!(response.text, "hi");
//! });
//! # }
//! ```

#![forbid(unsafe_code)]

pub mod credential;
pub mod error;
pub mod http;
pub mod message;
pub mod provider;
pub mod providers;
pub mod request;
pub mod response;

pub use credential::Credential;
pub use error::{Error, ErrorKind};
pub use http::{HttpClient, HttpRequest, HttpResponse, Method};
pub use message::{Message, Role};
pub use provider::Provider;
pub use request::CompletionRequest;
pub use response::{CompletionResponse, FinishReason, Usage};

#[cfg(feature = "anthropic")]
pub use providers::AnthropicProvider;
