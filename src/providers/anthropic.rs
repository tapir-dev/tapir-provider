// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The Anthropic [`Provider`], authenticating with an `x-api-key` Credential.

use crate::credential::Credential;
use crate::error::{Error, ErrorKind};
use crate::http::{HttpClient, HttpRequest, Method};
use crate::message::Role;
use crate::provider::Provider;
use crate::request::CompletionRequest;
use crate::response::{CompletionResponse, FinishReason, Usage};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Default base URL for the Anthropic API.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Anthropic API version header value pinned by this crate.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Anthropic requires `max_tokens`; use this when the request leaves it unset.
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// A Provider for Anthropic's Messages API.
///
/// The transport is injected as the generic `H`, which erases to
/// `Arc<dyn Provider>` at registration. The addressable Model is fixed when the
/// Provider is built.
#[derive(Debug, Clone)]
pub struct AnthropicProvider<H> {
    http: H,
    credential: Credential,
    model: String,
    base_url: String,
}

impl<H: HttpClient> AnthropicProvider<H> {
    /// Build a Provider for the given Model, authenticating with `credential`
    /// over the injected transport.
    pub fn new(
        http: H,
        credential: Credential,
        model: impl Into<String>,
    ) -> Self {
        Self {
            http,
            credential,
            model: model.into(),
            base_url: DEFAULT_BASE_URL.to_owned(),
        }
    }

    /// Override the base URL (for proxies, gateways, or a test server).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    fn build_http_request(
        &self,
        request: &CompletionRequest,
    ) -> Result<HttpRequest, Error> {
        let api_key = self.credential.as_api_key().ok_or_else(|| {
            Error::new(
                ErrorKind::Authentication,
                "Anthropic Provider requires an API-key Credential",
            )
        })?;

        let body = serde_json::to_vec(&WireRequest::from_request(
            &self.model,
            request,
        ))
        .map_err(|err| {
            Error::new(ErrorKind::Other, err.to_string()).with_source(err)
        })?;

        let url =
            format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        Ok(HttpRequest::new(Method::Post, url)
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .body(body))
    }
}

#[async_trait]
impl<H: HttpClient> Provider for AnthropicProvider<H> {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, Error> {
        let http_request = self.build_http_request(&request)?;
        let response = self.http.send(http_request).await?;

        if !response.is_success() {
            return Err(Error::from_status(
                response.status,
                response.body_string(),
            ));
        }

        let raw: serde_json::Value = serde_json::from_slice(&response.body)
            .map_err(|err| {
                Error::new(ErrorKind::Decode, err.to_string()).with_source(err)
            })?;
        let wire: WireResponse =
            serde_json::from_value(raw.clone()).map_err(|err| {
                Error::new(ErrorKind::Decode, err.to_string()).with_source(err)
            })?;

        Ok(wire.into_response(raw))
    }
}

/// The Anthropic request body as sent on the wire.
#[derive(Debug, Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

impl<'a> WireRequest<'a> {
    fn from_request(model: &'a str, request: &'a CompletionRequest) -> Self {
        let mut messages = Vec::new();
        let mut system_parts = Vec::new();

        for message in &request.messages {
            match message.role {
                Role::System => system_parts.push(message.content.as_str()),
                Role::User => messages.push(WireMessage {
                    role: "user",
                    content: &message.content,
                }),
                Role::Assistant => messages.push(WireMessage {
                    role: "assistant",
                    content: &message.content,
                }),
            }
        }

        let system = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n\n"))
        };

        Self {
            model,
            max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            messages,
            temperature: request.temperature,
            system,
        }
    }
}

/// A single message in the Anthropic request body.
#[derive(Debug, Serialize)]
struct WireMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// The Anthropic response body.
#[derive(Debug, Deserialize)]
struct WireResponse {
    #[serde(default)]
    content: Vec<WireContentBlock>,
    stop_reason: Option<String>,
    #[serde(default)]
    usage: WireUsage,
}

impl WireResponse {
    fn into_response(self, raw: serde_json::Value) -> CompletionResponse {
        let text = self
            .content
            .iter()
            .filter(|block| block.block_type == "text")
            .map(|block| block.text.as_str())
            .collect::<Vec<_>>()
            .concat();

        CompletionResponse {
            text,
            usage: Usage {
                input_tokens: self.usage.input_tokens,
                output_tokens: self.usage.output_tokens,
            },
            finish_reason: map_finish_reason(self.stop_reason),
            raw,
        }
    }
}

/// A content block in the Anthropic response.
#[derive(Debug, Deserialize)]
struct WireContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: String,
}

/// Token usage in the Anthropic response.
#[derive(Debug, Default, Deserialize)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

fn map_finish_reason(stop_reason: Option<String>) -> FinishReason {
    match stop_reason.as_deref() {
        Some("end_turn") => FinishReason::Stop,
        Some("max_tokens") => FinishReason::MaxTokens,
        Some("stop_sequence") => FinishReason::StopSequence,
        Some("tool_use") => FinishReason::ToolUse,
        Some(other) => FinishReason::Other(other.to_owned()),
        None => FinishReason::Other(String::new()),
    }
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use crate::http::MockHttpClient;
    use crate::message::Message;

    const SAMPLE_RESPONSE: &str = r#"{
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "Hello there!"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 5}
    }"#;

    #[tokio::test]
    async fn completes_a_text_prompt_through_the_injected_transport() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet-20241022",
        );

        let request = CompletionRequest::new(vec![Message::user("Hello")]);
        let response = provider.complete(request).await.unwrap();

        assert_eq!(response.text, "Hello there!");
        assert_eq!(response.usage.input_tokens, 12);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.raw["id"], "msg_123");
    }

    #[tokio::test]
    async fn sends_api_key_and_version_headers_with_the_body() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::new(
            mock.clone(),
            Credential::api_key("sk-secret"),
            "claude-3-5-sonnet-20241022",
        );

        let request = CompletionRequest::new(vec![
            Message::system("Be terse."),
            Message::user("Hi"),
        ])
        .with_temperature(0.2)
        .with_max_tokens(64);
        provider.complete(request).await.unwrap();

        let sent = mock.last_request();
        assert_eq!(sent.method, Method::Post);
        assert!(sent.url.ends_with("/v1/messages"));
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "x-api-key" && v == "sk-secret")
        );
        assert!(
            sent.headers.iter().any(
                |(k, v)| k == "anthropic-version" && v == ANTHROPIC_VERSION
            )
        );

        let body: serde_json::Value =
            serde_json::from_slice(sent.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["system"], "Be terse.");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "Hi");
    }

    #[tokio::test]
    async fn non_2xx_maps_to_a_typed_error_kind() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            401,
            r#"{"error":{"message":"invalid key"}}"#,
        ));
        let provider = AnthropicProvider::new(
            mock,
            Credential::api_key("bad"),
            "claude-3-5-sonnet",
        );

        let err = provider
            .complete(CompletionRequest::new(vec![Message::user("hi")]))
            .await
            .unwrap_err();

        assert_eq!(err.kind(), ErrorKind::Authentication);
        assert_eq!(err.status(), Some(401));
    }

    #[tokio::test]
    async fn rate_limit_and_server_errors_classify_by_status() {
        for (status, kind) in [
            (429, ErrorKind::RateLimited),
            (500, ErrorKind::ServerError),
            (529, ErrorKind::Overloaded),
        ] {
            let mock = std::sync::Arc::new(MockHttpClient::with_response(
                status, "{}",
            ));
            let provider = AnthropicProvider::new(
                mock,
                Credential::api_key("k"),
                "claude-3-5-sonnet",
            );
            let err = provider
                .complete(CompletionRequest::new(vec![Message::user("hi")]))
                .await
                .unwrap_err();
            assert_eq!(err.kind(), kind, "status {status}");
        }
    }
}
