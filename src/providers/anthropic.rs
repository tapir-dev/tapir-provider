// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The Anthropic [`Provider`], authenticating with an `x-api-key` Credential.

use crate::credential::Credential;
use crate::error::{Error, ErrorKind};
use crate::http::{ByteStream, HttpClient, HttpRequest, Method};
use crate::message::Role;
use crate::provider::Provider;
use crate::request::{CompletionRequest, ToolChoice, ToolDefinition};
use crate::response::{
    CompletionResponse, FinishReason, ToolCall, Usage, mint_call_id,
};
use crate::sse::{SseDecoder, SseEvent};
use crate::stream::{StreamEvent, StreamEvents};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::pin::Pin;
use std::task::{Context, Poll};

/// Default base URL for the Anthropic API.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Anthropic API version header value pinned by this crate.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Anthropic requires `max_tokens`; use this when the request leaves it unset.
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// Whether a request opts into a streamed (SSE) response.
///
/// A named alternative to a bare `bool` at the call sites that build the wire
/// request, so `Streaming::On` reads for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Streaming {
    /// Request an incremental SSE response.
    On,
    /// Request a single buffered response.
    Off,
}

impl Streaming {
    /// Whether streaming is requested, as the wire `stream` flag.
    const fn enabled(self) -> bool {
        matches!(self, Self::On)
    }
}

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
        streaming: Streaming,
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
            streaming,
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
        let http_request = self.build_http_request(&request, Streaming::Off)?;
        let response = self.http.send(http_request).await?;

        if !response.is_success() {
            return Err(Error::from_status(
                response.status,
                response.body_string(),
            ));
        }

        let wire: WireResponse =
            serde_json::from_slice(&response.body).map_err(Error::decode)?;
        let raw: serde_json::Value =
            serde_json::from_slice(&response.body).map_err(Error::decode)?;

        Ok(wire.into_response(raw))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamEvents, Error> {
        let http_request = self.build_http_request(&request, Streaming::On)?;
        let bytes = self.http.send_stream(http_request).await?;
        Ok(Box::pin(SseEventStream::new(bytes)))
    }
}

/// Adapts a byte stream into ordered [`StreamEvent`]s.
///
/// It owns the pipeline for one streamed completion: the [`SseDecoder`] that
/// reassembles events off the byte chunks, the [`StreamNormalizer`] that maps
/// each Anthropic SSE event into the neutral vocabulary, and a queue holding
/// the events a single chunk expanded into but that have not been yielded yet.
struct SseEventStream {
    /// The response body, streamed as byte chunks.
    bytes: ByteStream,
    /// Reassembles SSE events straddling chunk boundaries.
    decoder: SseDecoder,
    /// Maps Anthropic SSE events to the neutral vocabulary.
    normalizer: StreamNormalizer,
    /// Events decoded but not yet yielded to the caller.
    pending: VecDeque<StreamEvent>,
    /// Whether the byte stream has ended (or errored).
    finished: bool,
}

impl SseEventStream {
    fn new(bytes: ByteStream) -> Self {
        Self {
            bytes,
            decoder: SseDecoder::new(),
            normalizer: StreamNormalizer::default(),
            pending: VecDeque::new(),
            finished: false,
        }
    }
}

impl futures_core::Stream for SseEventStream {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(event) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            if this.finished {
                return Poll::Ready(None);
            }

            match this.bytes.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    for sse in this.decoder.push(&chunk) {
                        this.pending.extend(this.normalizer.normalize(&sse));
                    }
                }
                Poll::Ready(Some(Err(err))) => {
                    this.finished = true;
                    return Poll::Ready(Some(Err(err)));
                }
                Poll::Ready(None) => {
                    this.finished = true;
                    if let Some(sse) = this.decoder.finish() {
                        this.pending.extend(this.normalizer.normalize(&sse));
                    }
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Turns Anthropic's SSE events into the neutral [`StreamEvent`] vocabulary.
///
/// It threads the small amount of state the mapping needs: the running token
/// usage (Anthropic reports input tokens up front and output tokens at the
/// end), the finish reason from `message_delta`, and which content-block
/// indices are tool calls, so a `content_block_stop` on one becomes a
/// [`StreamEvent::ToolCallEnd`].
#[derive(Debug, Default)]
struct StreamNormalizer {
    /// Input tokens, reported in `message_start`.
    input_tokens: u32,
    /// Output tokens, reported cumulatively in `message_delta`.
    output_tokens: u32,
    /// Finish reason, reported in `message_delta`.
    finish_reason: Option<FinishReason>,
    /// Content-block indices that opened as tool calls.
    tool_indices: HashSet<usize>,
}

impl StreamNormalizer {
    /// The token usage seen so far.
    fn usage(&self) -> Usage {
        Usage {
            input_tokens: self.input_tokens,
            output_tokens: self.output_tokens,
        }
    }

    /// Map one Anthropic SSE event to zero or more neutral events.
    fn normalize(&mut self, sse: &SseEvent) -> Vec<StreamEvent> {
        // A payload that will not parse is not fatal; surface it verbatim.
        let json: serde_json::Value = match serde_json::from_str(&sse.data) {
            Ok(json) => json,
            Err(_) => {
                return vec![StreamEvent::Unknown(serde_json::Value::String(
                    sse.data.clone(),
                ))];
            }
        };
        let index = json["index"].as_u64().unwrap_or(0) as usize;

        match sse.event.as_deref() {
            Some("message_start") => {
                self.input_tokens = json["message"]["usage"]["input_tokens"]
                    .as_u64()
                    .unwrap_or(0) as u32;
                vec![StreamEvent::MessageStart]
            }
            Some("content_block_start") => {
                let block = &json["content_block"];
                if block["type"] == "tool_use" {
                    self.tool_indices.insert(index);
                    vec![StreamEvent::ToolCallStart {
                        index,
                        id: block["id"].as_str().unwrap_or_default().to_owned(),
                        name: block["name"]
                            .as_str()
                            .unwrap_or_default()
                            .to_owned(),
                    }]
                } else {
                    // A text or reasoning block opens with no incremental content.
                    vec![]
                }
            }
            Some("content_block_delta") => {
                let delta = &json["delta"];
                match delta["type"].as_str() {
                    Some("text_delta") => vec![StreamEvent::TextDelta {
                        index,
                        text: delta["text"]
                            .as_str()
                            .unwrap_or_default()
                            .to_owned(),
                    }],
                    Some("thinking_delta") => {
                        vec![StreamEvent::ReasoningDelta {
                            index,
                            text: delta["thinking"]
                                .as_str()
                                .unwrap_or_default()
                                .to_owned(),
                        }]
                    }
                    Some("input_json_delta") => {
                        vec![StreamEvent::ToolCallDelta {
                            index,
                            partial_json: delta["partial_json"]
                                .as_str()
                                .unwrap_or_default()
                                .to_owned(),
                        }]
                    }
                    // e.g. `signature_delta`: recognized event, unmodeled delta.
                    _ => vec![StreamEvent::Unknown(json)],
                }
            }
            Some("content_block_stop") => {
                if self.tool_indices.contains(&index) {
                    vec![StreamEvent::ToolCallEnd { index }]
                } else {
                    vec![]
                }
            }
            Some("message_delta") => {
                if let Some(reason) = json["delta"]["stop_reason"].as_str() {
                    self.finish_reason =
                        Some(map_finish_reason(Some(reason.to_owned())));
                }
                if let Some(output) = json["usage"]["output_tokens"].as_u64() {
                    self.output_tokens = output as u32;
                }
                vec![StreamEvent::Usage(self.usage())]
            }
            Some("message_stop") => vec![StreamEvent::Done {
                finish_reason: self
                    .finish_reason
                    .clone()
                    .unwrap_or(FinishReason::Other(String::new())),
                usage: self.usage(),
            }],
            // A heartbeat carries nothing.
            Some("ping") => vec![],
            // Any other event (including `error`) is surfaced, not swallowed.
            _ => vec![StreamEvent::Unknown(json)],
        }
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
}

/// Serde predicate: omit a `false` flag from the request body.
fn is_false(flag: &bool) -> bool {
    !*flag
}

impl<'a> WireRequest<'a> {
    fn from_request(
        model: &'a str,
        request: &'a CompletionRequest,
        streaming: Streaming,
    ) -> Self {
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

        let tools = request.tools.iter().map(WireTool::from).collect();

        Self {
            model,
            max_tokens: request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            messages,
            temperature: request.temperature,
            system,
            tools,
            tool_choice: request.tool_choice.as_ref().map(wire_tool_choice),
            stream: streaming.enabled(),
        }
    }
}

/// A tool definition as sent in the Anthropic request body.
#[derive(Debug, Serialize)]
struct WireTool<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: &'a serde_json::Value,
}

impl<'a> From<&'a ToolDefinition> for WireTool<'a> {
    fn from(tool: &'a ToolDefinition) -> Self {
        Self {
            name: &tool.name,
            description: &tool.description,
            input_schema: &tool.input_schema,
        }
    }
}

/// Map a neutral [`ToolChoice`] to Anthropic's `tool_choice` object.
fn wire_tool_choice(choice: &ToolChoice) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::Any => serde_json::json!({"type": "any"}),
        ToolChoice::Tool(name) => {
            serde_json::json!({"type": "tool", "name": name})
        }
        ToolChoice::None => serde_json::json!({"type": "none"}),
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

        let tool_calls = self
            .content
            .into_iter()
            .filter(|block| block.block_type == "tool_use")
            .map(|block| ToolCall {
                id: mint_call_id(),
                native_id: block.id,
                name: block.name,
                arguments: block.input,
            })
            .collect();

        CompletionResponse {
            text,
            tool_calls,
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
    /// The native tool-call id, present on `tool_use` blocks.
    #[serde(default)]
    id: Option<String>,
    /// The tool name, present on `tool_use` blocks.
    #[serde(default)]
    name: String,
    /// The tool arguments, present on `tool_use` blocks.
    #[serde(default)]
    input: serde_json::Value,
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
    use crate::stream::StreamAccumulator;
    use futures_util::StreamExt;

    /// A full Anthropic message stream: a text block, then usage and stop.
    const SAMPLE_STREAM: &str = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":7,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: ping\n",
        "data: {\"type\":\"ping\"}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\", world\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );

    async fn collect_stream(
        provider: &AnthropicProvider<std::sync::Arc<MockHttpClient>>,
    ) -> Vec<StreamEvent> {
        let request = CompletionRequest::new(vec![Message::user("hi")]);
        provider
            .complete_stream(request)
            .await
            .unwrap()
            .map(Result::unwrap)
            .collect()
            .await
    }

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
    async fn tool_definitions_and_choice_map_onto_the_wire_body() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );

        let schema = serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        });
        let request = CompletionRequest::new(vec![Message::user("weather?")])
            .with_tools(vec![ToolDefinition::new(
                "get_weather",
                "Look up the weather for a city",
                schema.clone(),
            )])
            .with_tool_choice(ToolChoice::Tool("get_weather".to_owned()));
        provider.complete(request).await.unwrap();

        let body: serde_json::Value = serde_json::from_slice(
            mock.last_request().body.as_deref().unwrap(),
        )
        .unwrap();
        assert_eq!(body["tools"][0]["name"], "get_weather");
        assert_eq!(
            body["tools"][0]["description"],
            "Look up the weather for a city"
        );
        assert_eq!(body["tools"][0]["input_schema"], schema);
        assert_eq!(body["tool_choice"]["type"], "tool");
        assert_eq!(body["tool_choice"]["name"], "get_weather");
    }

    #[tokio::test]
    async fn omits_tools_and_choice_when_unset() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );
        provider
            .complete(CompletionRequest::new(vec![Message::user("hi")]))
            .await
            .unwrap();

        let body: serde_json::Value = serde_json::from_slice(
            mock.last_request().body.as_deref().unwrap(),
        )
        .unwrap();
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[tokio::test]
    async fn tool_use_response_normalizes_with_both_ids() {
        let response = r#"{
            "id": "msg_tool",
            "content": [
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_42", "name": "get_weather", "input": {"city": "Paris"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 9, "output_tokens": 4}
        }"#;
        let mock =
            std::sync::Arc::new(MockHttpClient::with_response(200, response));
        let provider = AnthropicProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );

        let completion = provider
            .complete(CompletionRequest::new(vec![Message::user("weather?")]))
            .await
            .unwrap();

        assert_eq!(completion.text, "Let me check.");
        assert_eq!(completion.finish_reason, FinishReason::ToolUse);
        assert_eq!(completion.tool_calls.len(), 1);
        let call = &completion.tool_calls[0];
        assert_eq!(call.native_id.as_deref(), Some("toolu_42"));
        assert!(!call.id.is_empty());
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, serde_json::json!({"city": "Paris"}));
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

    #[tokio::test]
    async fn streams_ordered_events_ending_in_done() {
        let mock = std::sync::Arc::new(MockHttpClient::with_stream(vec![
            SAMPLE_STREAM.as_bytes().to_vec(),
        ]));
        let provider = AnthropicProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );

        let events = collect_stream(&provider).await;

        assert_eq!(events.first(), Some(&StreamEvent::MessageStart));
        assert_eq!(
            events[1],
            StreamEvent::TextDelta {
                index: 0,
                text: "Hello".to_owned()
            }
        );
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    input_tokens: 7,
                    output_tokens: 5
                }
            })
        ));

        // The request opted into streaming on the wire.
        let body: serde_json::Value = serde_json::from_slice(
            mock.last_request().body.as_deref().unwrap(),
        )
        .unwrap();
        assert_eq!(body["stream"], true);
    }

    #[tokio::test]
    async fn decodes_events_split_across_chunk_boundaries() {
        // Split the raw stream mid-event, at an arbitrary byte offset.
        let raw = SAMPLE_STREAM.as_bytes();
        let mid = raw.len() / 2;
        let chunks = vec![raw[..mid].to_vec(), raw[mid..].to_vec()];
        let mock = std::sync::Arc::new(MockHttpClient::with_stream(chunks));
        let provider = AnthropicProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );

        let events = collect_stream(&provider).await;
        // Same events despite the boundary falling inside an event.
        assert_eq!(events.first(), Some(&StreamEvent::MessageStart));
        assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
    }

    #[tokio::test]
    async fn accumulator_folds_stream_into_the_non_streaming_shape() {
        let mock = std::sync::Arc::new(MockHttpClient::with_stream(vec![
            SAMPLE_STREAM.as_bytes().to_vec(),
        ]));
        let provider = AnthropicProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );

        let events = collect_stream(&provider).await;
        let folded = StreamAccumulator::fold(&events);

        assert_eq!(folded.text, "Hello, world");
        assert_eq!(folded.finish_reason, FinishReason::Stop);
        assert_eq!(folded.usage.input_tokens, 7);
        assert_eq!(folded.usage.output_tokens, 5);
    }

    #[tokio::test]
    async fn unknown_sse_payloads_surface_as_unknown_events() {
        let stream = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n",
            // An event type this Provider does not model.
            "event: some_future_event\n",
            "data: {\"type\":\"some_future_event\",\"payload\":42}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let mock = std::sync::Arc::new(MockHttpClient::with_stream(vec![
            stream.as_bytes().to_vec(),
        ]));
        let provider = AnthropicProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );

        let events = collect_stream(&provider).await;
        let unknown = events
            .iter()
            .find(|e| matches!(e, StreamEvent::Unknown(_)))
            .expect("unmodeled event should surface as Unknown");
        if let StreamEvent::Unknown(value) = unknown {
            assert_eq!(value["payload"], 42);
        }
    }

    #[tokio::test]
    async fn streams_reasoning_deltas_from_thinking_blocks() {
        let stream = concat!(
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"let me think\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let mock = std::sync::Arc::new(MockHttpClient::with_stream(vec![
            stream.as_bytes().to_vec(),
        ]));
        let provider = AnthropicProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );

        let events = collect_stream(&provider).await;
        assert_eq!(
            events[0],
            StreamEvent::ReasoningDelta {
                index: 0,
                text: "let me think".to_owned(),
            }
        );
        // Reasoning never contributes to the folded completion text.
        assert_eq!(StreamAccumulator::fold(&events).text, "");
    }

    #[tokio::test]
    async fn streams_a_tool_call_start_delta_and_end() {
        let stream = concat!(
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let mock = std::sync::Arc::new(MockHttpClient::with_stream(vec![
            stream.as_bytes().to_vec(),
        ]));
        let provider = AnthropicProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );

        let events = collect_stream(&provider).await;
        assert_eq!(
            events[0],
            StreamEvent::ToolCallStart {
                index: 0,
                id: "toolu_1".to_owned(),
                name: "get_weather".to_owned(),
            }
        );
        assert_eq!(
            events[1],
            StreamEvent::ToolCallDelta {
                index: 0,
                partial_json: "{\"city\":".to_owned(),
            }
        );
        assert_eq!(events[2], StreamEvent::ToolCallEnd { index: 0 });
    }

    #[tokio::test]
    async fn folded_tool_stream_yields_a_complete_tool_call() {
        // Arguments arrive split across two input_json_delta fragments.
        let stream = concat!(
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_9\",\"name\":\"get_weather\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Paris\\\"}\"}}\n\n",
            "event: content_block_stop\n",
            "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":6}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        );
        let mock = std::sync::Arc::new(MockHttpClient::with_stream(vec![
            stream.as_bytes().to_vec(),
        ]));
        let provider = AnthropicProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );

        let events = collect_stream(&provider).await;
        let folded = StreamAccumulator::fold(&events);

        assert_eq!(folded.finish_reason, FinishReason::ToolUse);
        assert_eq!(folded.tool_calls.len(), 1);
        let call = &folded.tool_calls[0];
        assert_eq!(call.native_id.as_deref(), Some("toolu_9"));
        assert!(!call.id.is_empty());
        assert_eq!(call.name, "get_weather");
        assert_eq!(call.arguments, serde_json::json!({"city": "Paris"}));
    }
}
