// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The Anthropic [`Provider`], authenticating with an `x-api-key` or an OAuth
//! `Bearer` Credential.

pub mod oauth;

use crate::credential::Credential;
use crate::error::{Error, ErrorKind};
use crate::http::{ByteStream, HttpClient, HttpRequest, Method};
use crate::message::{
    AssistantMessage, ContentPart, ImageSource, Message, ToolResultMessage,
};
use crate::provider::Provider;
use crate::request::{
    CompletionOptions, Context, SystemPrompt, ThinkingLevel, ToolChoice,
    ToolDefinition,
};
use crate::response::{FinishReason, Usage, mint_call_id};
use crate::sse::{SseDecoder, SseEvent};
use crate::stream::{StreamEvent, StreamEvents};
use crate::token_store::{TokenStore, resolve};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::pin::Pin;
use std::task::Poll;

/// Default base URL for the Anthropic API.
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
/// Provider id this Anthropic Provider keys its Credential under in a Token Store.
pub(crate) const PROVIDER_KEY: &str = "anthropic";
/// Environment variable holding an Anthropic API key, the last-resort Credential.
pub(crate) const API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
/// Alternate names that select this Provider in the [`Registry`](crate::Registry).
pub(crate) const ALIASES: &[&str] = &["claude"];

/// This Provider's identity in the [`Registry`](crate::Registry): its canonical
/// id, the alternate names that select it, and the environment variable holding
/// its default API key.
pub const INFO: crate::registry::ProviderInfo = crate::registry::ProviderInfo {
    id: crate::model::ProviderId::from_static(PROVIDER_KEY),
    aliases: ALIASES,
    api_key_env: API_KEY_ENV,
};
/// Anthropic API version header value pinned by this crate.
const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Anthropic requires `max_tokens`; use this when the request leaves it unset.
const DEFAULT_MAX_TOKENS: u32 = 1024;
/// `anthropic-beta` header value the OAuth lane must send; the API rejects an
/// OAuth request without it.
const ANTHROPIC_OAUTH_BETA: &str = "claude-code-20250219,oauth-2025-04-20";
/// System block the OAuth lane prepends ahead of the caller's own prompt; the
/// API rejects an OAuth request whose leading system block is anything else.
const CLAUDE_CODE_IDENTITY: &str =
    "You are Claude Code, Anthropic's official CLI for Claude.";
/// Longest tool name Anthropic accepts; the OAuth lane truncates to this.
const MAX_TOOL_NAME_LEN: usize = 128;

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

/// Which authentication lane a request is shaped for.
///
/// The Credential variant fixes far more than the auth header: the OAuth lane
/// must prepend the Claude Code identity system block and normalize tool names,
/// or the API rejects the request. Threading this into the wire-body builder
/// keeps that fork in one place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthLane {
    /// An `x-api-key` Credential; the body is sent as-is.
    ApiKey,
    /// A `Bearer` OAuth Credential; the body carries the identity block and
    /// normalized tool names.
    OAuth,
}

impl AuthLane {
    /// The lane a Credential is served on.
    const fn for_credential(credential: &Credential) -> Self {
        match credential {
            Credential::OAuth(_) => Self::OAuth,
            Credential::ApiKey { .. } => Self::ApiKey,
        }
    }
}

/// The auth headers a request authenticating with `credential` carries.
///
/// The single source of truth for Anthropic's auth scheme: the api-key lane
/// pins the `anthropic-version` header and the `x-api-key` secret; the OAuth
/// lane carries the same version, a `Bearer` token, and the beta the API
/// requires. Both the request path and the Model Registry's auth inspection go
/// through this, so what one reports is exactly what the other sends.
pub(crate) fn auth_headers(credential: &Credential) -> Vec<(String, String)> {
    let mut headers =
        vec![("anthropic-version".to_owned(), ANTHROPIC_VERSION.to_owned())];
    match credential {
        Credential::ApiKey { key, .. } => {
            headers.push(("x-api-key".to_owned(), key.clone()));
        }
        Credential::OAuth(tokens) => {
            headers.push((
                "authorization".to_owned(),
                format!("Bearer {}", tokens.access_token),
            ));
            headers.push((
                "anthropic-beta".to_owned(),
                ANTHROPIC_OAUTH_BETA.to_owned(),
            ));
        }
    }
    headers
}

/// A Provider for Anthropic's Messages API.
///
/// The transport is injected as the generic `H`, which erases to
/// `Arc<dyn Provider>` at registration. The addressable Model is fixed when the
/// Provider is built.
#[derive(Clone)]
pub struct AnthropicProvider<H> {
    http: H,
    credential: Credential,
    model: String,
    base_url: String,
    /// Caller-supplied headers appended to every request, after the Provider's
    /// own auth and version headers. For proxies and gateways that key off a
    /// bespoke header.
    extra_headers: Vec<(String, String)>,
}

/// Redacts header *values*, keeping names visible: a caller-supplied header may
/// carry a secret (a proxy authorization token), so its value never reaches
/// Debug output — matching the crate's [`Credential`] redaction discipline.
fn redacted_headers(headers: &[(String, String)]) -> Vec<(&str, &str)> {
    headers
        .iter()
        .map(|(name, _)| (name.as_str(), "<redacted>"))
        .collect()
}

impl<H: fmt::Debug> fmt::Debug for AnthropicProvider<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("http", &self.http)
            .field("credential", &self.credential)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("extra_headers", &redacted_headers(&self.extra_headers))
            .finish()
    }
}

impl<H: HttpClient> AnthropicProvider<H> {
    /// Build a Provider for the given Model, authenticating with `credential`
    /// over the injected transport.
    ///
    /// The explicit `credential` is resolution's top tier, so this routes it
    /// through the same [`resolve`] rule every construction path uses: an
    /// explicit argument never consults the store or environment, so this is
    /// infallible.
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
            extra_headers: Vec::new(),
        }
    }

    /// Build a Provider by resolving its Credential from a Token Store, then
    /// the `ANTHROPIC_API_KEY` environment variable.
    ///
    /// Precedence follows [`resolve`]: an `explicit` Credential wins, else the
    /// `store` under this Provider's key, else the environment. When every tier
    /// is empty this is an
    /// [`Authentication`](crate::ErrorKind::Authentication) error rather than a
    /// Provider that cannot authenticate any request.
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
                        "no Anthropic Credential: pass one, store it under {PROVIDER_KEY:?}, or set {API_KEY_ENV}"
                    ),
                )
            })?;
        Ok(Self::new(http, credential, model))
    }

    /// Start a typed [`AnthropicBuilder`] over the injected transport for the
    /// given Model.
    ///
    /// The builder layers the advanced knobs — an explicit Credential, a
    /// base-URL override, and extra headers — over the same Credential
    /// resolution [`resolve`](Self::resolve) uses.
    #[must_use]
    pub fn builder(http: H, model: impl Into<String>) -> AnthropicBuilder<H> {
        AnthropicBuilder::new(http, model)
    }

    /// Override the base URL (for proxies, gateways, or a test server).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Append an extra header sent with every request, after the Provider's own
    /// auth and version headers.
    #[must_use]
    pub fn with_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    /// Append extra headers sent with every request, after the Provider's own
    /// auth and version headers.
    #[must_use]
    pub fn with_headers(
        mut self,
        headers: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        self.extra_headers.extend(headers);
        self
    }

    fn build_http_request(
        &self,
        ctx: &Context,
        opts: &CompletionOptions,
        streaming: Streaming,
    ) -> Result<HttpRequest, Error> {
        // An explicit per-request key overrides the constructed Credential for
        // this call; an explicit key always rides the api-key lane.
        let override_credential =
            opts.api_key.as_deref().map(Credential::api_key);
        let credential =
            override_credential.as_ref().unwrap_or(&self.credential);
        let lane = AuthLane::for_credential(credential);

        let body = serde_json::to_vec(&WireRequest::from_context(
            &self.model,
            ctx,
            opts,
            streaming,
            lane,
        ))
        .map_err(Error::serialize)?;

        let url =
            format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let request = HttpRequest::new(Method::Post, url)
            .header("content-type", "application/json");
        // Auth headers (version, and the lane's secret) come from the one
        // scheme definition, so the wire matches what auth inspection reports.
        let request = auth_headers(credential)
            .into_iter()
            .fold(request, |req, (name, value)| req.header(name, value));
        // Caller headers ride last, so a proxy or gateway can key off them.
        let request = self
            .extra_headers
            .iter()
            .fold(request, |req, (name, value)| req.header(name, value));
        Ok(request.body(body))
    }
}

/// A typed builder for an [`AnthropicProvider`] with advanced configuration.
///
/// It gathers an injected transport, the Model, and the optional knobs — an
/// explicit Credential, a base-URL override, and extra headers — then
/// [`build`](Self::build)s a Provider, resolving the Credential through the same
/// precedence [`AnthropicProvider::resolve`] uses (explicit, then the
/// `ANTHROPIC_API_KEY` environment variable).
#[derive(Clone)]
pub struct AnthropicBuilder<H> {
    http: H,
    model: String,
    credential: Option<Credential>,
    base_url: Option<String>,
    headers: Vec<(String, String)>,
}

impl<H: fmt::Debug> fmt::Debug for AnthropicBuilder<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnthropicBuilder")
            .field("http", &self.http)
            .field("model", &self.model)
            .field("credential", &self.credential)
            .field("base_url", &self.base_url)
            .field("headers", &redacted_headers(&self.headers))
            .finish()
    }
}

impl<H: HttpClient> AnthropicBuilder<H> {
    /// Start a builder over the injected transport for the given Model.
    #[must_use]
    pub fn new(http: H, model: impl Into<String>) -> Self {
        Self {
            http,
            model: model.into(),
            credential: None,
            base_url: None,
            headers: Vec::new(),
        }
    }

    /// Authenticate with an explicit Credential, the top tier of resolution.
    #[must_use]
    pub fn credential(mut self, credential: Credential) -> Self {
        self.credential = Some(credential);
        self
    }

    /// Override the base URL (for proxies, gateways, or a test server).
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Append an extra header sent with every request.
    #[must_use]
    pub fn header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Resolve the Credential and build the Provider.
    ///
    /// Credential precedence follows [`resolve`]: the explicit
    /// [`credential`](Self::credential) if set, else the `ANTHROPIC_API_KEY`
    /// environment variable. An empty result is an
    /// [`Authentication`](crate::ErrorKind::Authentication) error.
    pub fn build(self) -> Result<AnthropicProvider<H>, Error> {
        let mut provider = AnthropicProvider::resolve(
            self.http,
            self.model,
            self.credential,
            None,
        )?;
        if let Some(base_url) = self.base_url {
            provider = provider.with_base_url(base_url);
        }
        provider = provider.with_headers(self.headers);
        Ok(provider)
    }
}

#[async_trait]
impl<H: HttpClient> Provider for AnthropicProvider<H> {
    async fn complete(
        &self,
        ctx: &Context,
        opts: &CompletionOptions,
    ) -> Result<AssistantMessage, Error> {
        let http_request =
            self.build_http_request(ctx, opts, Streaming::Off)?;
        let response = self.http.send(http_request).await?;

        if !response.is_success() {
            return Err(crate::http::error_from_response(&response));
        }

        let wire: WireResponse =
            serde_json::from_slice(&response.body).map_err(Error::decode)?;
        let raw: serde_json::Value =
            serde_json::from_slice(&response.body).map_err(Error::decode)?;

        Ok(wire.into_message(raw))
    }

    async fn complete_stream(
        &self,
        ctx: &Context,
        opts: &CompletionOptions,
    ) -> Result<StreamEvents, Error> {
        let http_request = self.build_http_request(ctx, opts, Streaming::On)?;
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
        cx: &mut std::task::Context<'_>,
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
/// end), the finish reason from `message_delta`, which content-block indices are
/// text, thinking, or tool calls (so a `content_block_stop` becomes the matching
/// `*End` event), and the replay signature accumulated for each thinking block.
#[derive(Debug, Default)]
struct StreamNormalizer {
    /// Input tokens, reported in `message_start`.
    input_tokens: u32,
    /// Output tokens, reported cumulatively in `message_delta`.
    output_tokens: u32,
    /// Finish reason, reported in `message_delta`.
    finish_reason: Option<FinishReason>,
    /// Content-block indices that opened as text.
    text_indices: HashSet<usize>,
    /// Content-block indices that opened as thinking.
    thinking_indices: HashSet<usize>,
    /// Content-block indices that opened as tool calls.
    tool_indices: HashSet<usize>,
    /// Replay signatures accumulated per thinking block index.
    signatures: HashMap<usize, String>,
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
                match block["type"].as_str() {
                    Some("tool_use") => {
                        self.tool_indices.insert(index);
                        vec![StreamEvent::ToolCallStart {
                            index,
                            id: block["id"]
                                .as_str()
                                .unwrap_or_default()
                                .to_owned(),
                            name: block["name"]
                                .as_str()
                                .unwrap_or_default()
                                .to_owned(),
                        }]
                    }
                    Some("text") => {
                        self.text_indices.insert(index);
                        vec![StreamEvent::TextStart { index }]
                    }
                    Some("thinking") => {
                        self.thinking_indices.insert(index);
                        vec![StreamEvent::ThinkingStart { index }]
                    }
                    // Redacted thinking and any other block open with no
                    // incremental content this crate models.
                    _ => vec![],
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
                        vec![StreamEvent::ThinkingDelta {
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
                    Some("signature_delta") => {
                        // Accumulate the replay signature; it surfaces on the
                        // block's `ThinkingEnd`, not as an event of its own.
                        if let Some(sig) = delta["signature"].as_str() {
                            self.signatures
                                .entry(index)
                                .or_default()
                                .push_str(sig);
                        }
                        vec![]
                    }
                    _ => vec![StreamEvent::Unknown(json)],
                }
            }
            Some("content_block_stop") => {
                if self.tool_indices.contains(&index) {
                    vec![StreamEvent::ToolCallEnd { index }]
                } else if self.text_indices.contains(&index) {
                    vec![StreamEvent::TextEnd { index }]
                } else if self.thinking_indices.contains(&index) {
                    vec![StreamEvent::ThinkingEnd {
                        index,
                        signature: self.signatures.remove(&index),
                    }]
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
    system: Option<WireSystem>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<WireThinking>,
}

/// Anthropic's extended-thinking request block: budget-based ("enabled") mode.
#[derive(Debug, Serialize)]
struct WireThinking {
    #[serde(rename = "type")]
    kind: &'static str,
    budget_tokens: u32,
}

/// Serde predicate: omit a `false` flag from the request body.
fn is_false(flag: &bool) -> bool {
    !*flag
}

impl<'a> WireRequest<'a> {
    fn from_context(
        model: &'a str,
        ctx: &'a Context,
        opts: &'a CompletionOptions,
        streaming: Streaming,
        lane: AuthLane,
    ) -> Self {
        let messages = ctx.messages.iter().map(wire_message).collect();
        let system = wire_system(ctx.system_prompt.as_ref(), lane);

        let tools = ctx
            .tools
            .iter()
            .map(|tool| WireTool::from_tool(tool, lane))
            .collect();

        // Extended thinking, when asked for, reasons by token budget derived
        // from the neutral level. It shares the response ceiling with the
        // answer, so bump `max_tokens` to leave room, and it requires an unset
        // temperature, which Anthropic reads as `1.0`.
        let mut max_tokens = opts.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let level = opts.thinking.filter(|level| *level != ThinkingLevel::Off);
        let (thinking, temperature) = match level {
            Some(level) => {
                let budget = level.default_budget();
                if max_tokens <= budget {
                    max_tokens = budget + DEFAULT_MAX_TOKENS;
                }
                (
                    Some(WireThinking {
                        kind: "enabled",
                        budget_tokens: budget,
                    }),
                    Some(1.0),
                )
            }
            None => (None, opts.temperature),
        };

        Self {
            model,
            max_tokens,
            messages,
            temperature,
            system,
            tools,
            tool_choice: opts.tool_choice.as_ref().map(wire_tool_choice),
            stream: streaming.enabled(),
            thinking,
        }
    }
}

/// Map one neutral [`Message`] onto an Anthropic request message.
///
/// A tool result is carried, per the Messages API, as a `user` message whose
/// content is a single `tool_result` block referencing the call by id.
fn wire_message(message: &Message) -> WireMessage<'_> {
    match message {
        Message::User { content } => WireMessage {
            role: "user",
            content: wire_content(content),
        },
        Message::Assistant(assistant) => WireMessage {
            role: "assistant",
            content: wire_content(&assistant.content),
        },
        Message::ToolResult(result) => WireMessage {
            role: "user",
            content: wire_tool_result(result),
        },
    }
}

/// A [`ToolResultMessage`] as a single-block `user` content on the wire.
fn wire_tool_result(result: &ToolResultMessage) -> WireContent<'_> {
    WireContent::Blocks(vec![WireContentPart::ToolResult {
        tool_use_id: &result.tool_call_id,
        content: tool_result_text(&result.content),
        is_error: result.is_error,
    }])
}

/// The text of a tool result's content, borrowing a lone text part and joining
/// several. Non-text parts are dropped.
fn tool_result_text(parts: &[ContentPart]) -> Cow<'_, str> {
    if let [ContentPart::Text(text)] = parts {
        return Cow::Borrowed(text.as_str());
    }
    let joined: String = parts
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    Cow::Owned(joined)
}

/// A message's `system` prompt on the wire.
///
/// The API-key lane sends the caller's prompt as a plain string (or omits it);
/// the OAuth lane sends an array of typed blocks so the Claude Code identity can
/// lead it.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum WireSystem {
    /// A single text prompt.
    Text(String),
    /// Typed system blocks, identity first.
    Blocks(Vec<WireSystemBlock>),
}

/// One typed `system` block in the Anthropic request body.
#[derive(Debug, Serialize)]
struct WireSystemBlock {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: String,
}

impl WireSystemBlock {
    /// A `text`-typed system block.
    fn text(text: impl Into<String>) -> Self {
        Self {
            block_type: "text",
            text: text.into(),
        }
    }
}

/// Shape the caller's system prompt for the chosen lane.
///
/// An empty prompt is dropped, so the API-key lane omits `system` entirely
/// rather than sending an empty string. The OAuth lane always yields at least
/// the identity block — the API rejects an OAuth request without it — so this
/// returns `Some` even when the caller supplied no system prompt.
fn wire_system(
    caller_system: Option<&SystemPrompt>,
    lane: AuthLane,
) -> Option<WireSystem> {
    let caller = caller_system
        .map(SystemPrompt::as_str)
        .filter(|text| !text.is_empty());
    match lane {
        AuthLane::ApiKey => {
            caller.map(|text| WireSystem::Text(text.to_owned()))
        }
        AuthLane::OAuth => {
            let mut blocks = vec![WireSystemBlock::text(CLAUDE_CODE_IDENTITY)];
            if let Some(caller) = caller {
                blocks.push(WireSystemBlock::text(caller));
            }
            Some(WireSystem::Blocks(blocks))
        }
    }
}

/// Normalize a tool name to Anthropic's accepted shape (`[a-zA-Z0-9_-]`, at
/// most [`MAX_TOOL_NAME_LEN`] characters), borrowing when it already conforms.
///
/// The OAuth lane runs every tool name through this; a name carrying a `.`, a
/// space, or other punctuation would otherwise be rejected.
fn normalize_tool_name(name: &str) -> Cow<'_, str> {
    let allowed = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '-';
    if name.chars().count() <= MAX_TOOL_NAME_LEN && name.chars().all(allowed) {
        return Cow::Borrowed(name);
    }
    let normalized: String = name
        .chars()
        .map(|c| if allowed(c) { c } else { '_' })
        .take(MAX_TOOL_NAME_LEN)
        .collect();
    Cow::Owned(normalized)
}

/// A tool definition as sent in the Anthropic request body.
#[derive(Debug, Serialize)]
struct WireTool<'a> {
    name: Cow<'a, str>,
    description: &'a str,
    input_schema: &'a serde_json::Value,
}

impl<'a> WireTool<'a> {
    /// Map a neutral [`ToolDefinition`] onto the wire, normalizing the tool
    /// name on the OAuth lane and passing it through unchanged otherwise.
    fn from_tool(tool: &'a ToolDefinition, lane: AuthLane) -> Self {
        let name = match lane {
            AuthLane::ApiKey => Cow::Borrowed(tool.name.as_str()),
            AuthLane::OAuth => normalize_tool_name(&tool.name),
        };
        Self {
            name,
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
    content: WireContent<'a>,
}

/// A message's content on the wire.
///
/// Anthropic accepts either a plain string or an array of typed blocks. A
/// message that is a single text part serializes as the string form, keeping
/// the common case compact; anything else (images, or multiple parts)
/// serializes as blocks.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum WireContent<'a> {
    Text(&'a str),
    Blocks(Vec<WireContentPart<'a>>),
}

/// One content block in the Anthropic request body.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireContentPart<'a> {
    Text {
        text: &'a str,
    },
    Image {
        source: WireImageSource<'a>,
    },
    ToolUse {
        id: &'a str,
        name: &'a str,
        input: &'a serde_json::Value,
    },
    ToolResult {
        tool_use_id: &'a str,
        content: Cow<'a, str>,
        #[serde(skip_serializing_if = "is_false")]
        is_error: bool,
    },
}

/// An image block's `source` in the Anthropic request body.
///
/// Raw bytes are base64-encoded into the same shape as an inline base64 image,
/// so both carry a `media_type`.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum WireImageSource<'a> {
    Url {
        url: &'a str,
    },
    Base64 {
        media_type: &'static str,
        data: Cow<'a, str>,
    },
}

/// Map neutral content parts onto Anthropic's message content.
///
/// A lone text part serializes as the compact string form; anything else —
/// images, tool calls, or multiple parts — serializes as typed blocks.
fn wire_content(parts: &[ContentPart]) -> WireContent<'_> {
    if let [ContentPart::Text(text)] = parts {
        return WireContent::Text(text.as_str());
    }
    WireContent::Blocks(parts.iter().filter_map(wire_content_part).collect())
}

/// Map one neutral content part onto an Anthropic content block, or `None` for a
/// part with no request-content representation.
fn wire_content_part(part: &ContentPart) -> Option<WireContentPart<'_>> {
    match part {
        ContentPart::Text(text) => Some(WireContentPart::Text { text }),
        ContentPart::Image(source) => Some(WireContentPart::Image {
            source: wire_image_source(source),
        }),
        ContentPart::ToolCall {
            id,
            name,
            arguments,
        } => Some(WireContentPart::ToolUse {
            id,
            name,
            input: arguments,
        }),
        // Thinking is not replayed on the request without its full block shape.
        ContentPart::Thinking { .. } => None,
    }
}

/// Map a neutral image source onto Anthropic's image `source` object.
fn wire_image_source(source: &ImageSource) -> WireImageSource<'_> {
    match source {
        ImageSource::Url(url) => WireImageSource::Url { url },
        ImageSource::Base64 { media_type, data } => WireImageSource::Base64 {
            media_type: media_type.as_wire(),
            data: Cow::Borrowed(data),
        },
        ImageSource::Bytes { media_type, data } => WireImageSource::Base64 {
            media_type: media_type.as_wire(),
            data: Cow::Owned(crate::base64::base64_encode(data)),
        },
    }
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
    fn into_message(self, raw: serde_json::Value) -> AssistantMessage {
        // Preserve wire order so text and tool calls interleave as they arrived.
        let mut content = Vec::new();
        for block in self.content {
            match block.block_type.as_str() {
                "text" if !block.text.is_empty() => {
                    content.push(ContentPart::Text(block.text));
                }
                "tool_use" => content.push(ContentPart::ToolCall {
                    id: block.id.unwrap_or_else(mint_call_id),
                    name: block.name,
                    arguments: block.input,
                }),
                _ => {}
            }
        }

        AssistantMessage {
            content,
            usage: Usage {
                input_tokens: self.usage.input_tokens,
                output_tokens: self.usage.output_tokens,
            },
            finish_reason: map_finish_reason(self.stop_reason),
            raw: Some(raw),
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
    use crate::credential::OAuthTokens;
    use crate::http::MockHttpClient;
    use crate::message::{MediaType, Message};
    use crate::stream::StreamAccumulator;
    use futures_util::StreamExt;
    use std::time::Duration;

    /// The default, empty per-request options.
    fn opts() -> CompletionOptions {
        CompletionOptions::default()
    }

    /// The tool calls in a reply, as `(id, name, arguments)` tuples in order.
    fn tool_calls(
        message: &AssistantMessage,
    ) -> Vec<(&str, &str, &serde_json::Value)> {
        message
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                } => Some((id.as_str(), name.as_str(), arguments)),
                _ => None,
            })
            .collect()
    }

    /// Send one context and return its parsed JSON body.
    async fn sent_body(ctx: Context) -> serde_json::Value {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );
        provider.complete(&ctx, &opts()).await.unwrap();
        serde_json::from_slice(mock.last_request().body.as_deref().unwrap())
            .unwrap()
    }

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
        let ctx = Context::new(vec![Message::user("hi")]);
        provider
            .complete_stream(&ctx, &opts())
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

        let ctx = Context::new(vec![Message::user("Hello")]);
        let response = provider.complete(&ctx, &opts()).await.unwrap();

        assert_eq!(response.text_content(), "Hello there!");
        assert_eq!(response.usage.input_tokens, 12);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.raw.as_ref().unwrap()["id"], "msg_123");
    }

    #[tokio::test]
    async fn resolves_api_key_from_the_token_store_and_completes() {
        use crate::token_store::InMemoryTokenStore;

        let store = InMemoryTokenStore::new();
        store
            .set(PROVIDER_KEY, Credential::api_key("sk-stored"))
            .unwrap();
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::resolve(
            mock.clone(),
            "claude-3-5-sonnet",
            None,
            Some(&store),
        )
        .unwrap();

        let ctx = Context::new(vec![Message::user("Hello")]);
        let response = provider.complete(&ctx, &opts()).await.unwrap();

        // Same completion as the explicit-Credential path, and the resolved
        // key is what reached the wire.
        assert_eq!(response.text_content(), "Hello there!");
        assert!(
            mock.last_request()
                .headers
                .iter()
                .any(|(k, v)| k == "x-api-key" && v == "sk-stored")
        );
    }

    #[test]
    fn wire_system_omits_an_empty_prompt_on_the_api_key_lane() {
        let empty = SystemPrompt::new("");
        assert!(wire_system(Some(&empty), AuthLane::ApiKey).is_none());
        assert!(wire_system(None, AuthLane::ApiKey).is_none());

        // The OAuth lane still sends the identity block, and drops the empty
        // caller prompt from the block list.
        let Some(WireSystem::Blocks(blocks)) =
            wire_system(Some(&empty), AuthLane::OAuth)
        else {
            panic!("expected identity blocks on the OAuth lane");
        };
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, CLAUDE_CODE_IDENTITY);
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

        let ctx =
            Context::new(vec![Message::user("Hi")]).with_system("Be terse.");
        let opts = opts().with_temperature(0.2).with_max_tokens(64);
        provider.complete(&ctx, &opts).await.unwrap();

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
    async fn per_request_api_key_overrides_the_constructed_credential() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::new(
            mock.clone(),
            Credential::api_key("sk-constructed"),
            "claude-3-5-sonnet",
        );

        let ctx = Context::new(vec![Message::user("Hi")]);
        let opts = opts().with_api_key("sk-explicit");
        provider.complete(&ctx, &opts).await.unwrap();

        let sent = mock.last_request();
        // The explicit per-request key reached the wire on the api-key lane...
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "x-api-key" && v == "sk-explicit")
        );
        // ...and the constructed Credential was not sent.
        assert!(!sent.headers.iter().any(|(_, v)| v == "sk-constructed"));
    }

    #[tokio::test]
    async fn thinking_option_sends_a_budget_block_and_forces_temperature() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::new(
            mock.clone(),
            Credential::api_key("sk-secret"),
            "claude-haiku-4-5",
        );

        let ctx = Context::new(vec![Message::user("Hi")]);
        let opts = opts()
            .with_temperature(0.2)
            .with_max_tokens(64)
            .with_thinking(ThinkingLevel::Low);
        provider.complete(&ctx, &opts).await.unwrap();

        let body: serde_json::Value = serde_json::from_slice(
            mock.last_request().body.as_deref().unwrap(),
        )
        .unwrap();
        // Low derives a 2048-token budget in the "enabled" (budget) shape.
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 2048);
        // max_tokens (64) sat below the budget, so it is lifted to leave the
        // answer room beyond the thinking budget.
        assert_eq!(body["max_tokens"], 2048 + 1024);
        // Extended thinking requires temperature 1.0, overriding the caller's.
        assert_eq!(body["temperature"], 1.0);
    }

    #[tokio::test]
    async fn no_thinking_option_omits_the_block_and_keeps_temperature() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::new(
            mock.clone(),
            Credential::api_key("sk-secret"),
            "claude-haiku-4-5",
        );

        let ctx = Context::new(vec![Message::user("Hi")]);
        let opts = opts().with_temperature(0.2).with_max_tokens(64);
        provider.complete(&ctx, &opts).await.unwrap();

        let body: serde_json::Value = serde_json::from_slice(
            mock.last_request().body.as_deref().unwrap(),
        )
        .unwrap();
        assert!(body.get("thinking").is_none());
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["temperature"], 0.2);
    }

    #[tokio::test]
    async fn builder_overrides_base_url_and_appends_extra_headers() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::builder(mock.clone(), "claude-3-5")
            .credential(Credential::api_key("sk-secret"))
            .base_url("https://gateway.internal/anthropic")
            .header("x-tenant", "acme")
            .build()
            .unwrap();

        let ctx = Context::new(vec![Message::user("hi")]);
        provider.complete(&ctx, &opts()).await.unwrap();

        let sent = mock.last_request();
        // The override replaces the default host, keeping the API path.
        assert_eq!(sent.url, "https://gateway.internal/anthropic/v1/messages");
        // The caller header rides alongside the Provider's own auth header.
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "x-tenant" && v == "acme")
        );
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "x-api-key" && v == "sk-secret")
        );
    }

    #[test]
    fn debug_redacts_extra_header_values() {
        let provider = AnthropicProvider::new(
            std::sync::Arc::new(MockHttpClient::new()),
            Credential::api_key("sk-secret"),
            "claude-3-5",
        )
        .with_header("x-proxy-authorization", "super-secret-token");

        let rendered = format!("{provider:?}");
        // A header value may be a secret; the name stays visible, value gone.
        assert!(!rendered.contains("super-secret-token"));
        assert!(rendered.contains("x-proxy-authorization"));
        assert!(rendered.contains("redacted"));
        // The Credential secret is still redacted through the manual Debug.
        assert!(!rendered.contains("sk-secret"));
    }

    #[tokio::test]
    async fn tool_definitions_and_choice_map_onto_the_wire_body() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        });
        let ctx =
            Context::new(vec![Message::user("weather?")]).with_tools(vec![
                ToolDefinition::new(
                    "get_weather",
                    "Look up the weather for a city",
                    schema.clone(),
                ),
            ]);
        let opts =
            opts().with_tool_choice(ToolChoice::Tool("get_weather".to_owned()));
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );
        provider.complete(&ctx, &opts).await.unwrap();

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
        let body = sent_body(Context::new(vec![Message::user("hi")])).await;
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[tokio::test]
    async fn assistant_tool_call_and_tool_result_round_trip_onto_the_wire() {
        let ctx = Context::new(vec![
            Message::user("weather?"),
            Message::Assistant(AssistantMessage {
                content: vec![ContentPart::tool_call(
                    "toolu_7",
                    "get_weather",
                    serde_json::json!({"city": "Paris"}),
                )],
                usage: Usage::default(),
                finish_reason: FinishReason::ToolUse,
                raw: None,
            }),
            Message::tool_result("toolu_7", "get_weather", "sunny"),
        ]);
        let body = sent_body(ctx).await;

        // The assistant tool call serializes as a `tool_use` block.
        let assistant = &body["messages"][1];
        assert_eq!(assistant["role"], "assistant");
        assert_eq!(assistant["content"][0]["type"], "tool_use");
        assert_eq!(assistant["content"][0]["id"], "toolu_7");
        assert_eq!(assistant["content"][0]["name"], "get_weather");
        assert_eq!(assistant["content"][0]["input"]["city"], "Paris");
        // The result is a `user` message carrying a `tool_result` block.
        let result = &body["messages"][2];
        assert_eq!(result["role"], "user");
        assert_eq!(result["content"][0]["type"], "tool_result");
        assert_eq!(result["content"][0]["tool_use_id"], "toolu_7");
        assert_eq!(result["content"][0]["content"], "sunny");
        // A successful result omits the error flag.
        assert!(result["content"][0].get("is_error").is_none());
    }

    #[tokio::test]
    async fn an_errored_tool_result_sets_the_error_flag() {
        let ctx = Context::new(vec![Message::ToolResult(ToolResultMessage {
            tool_call_id: "toolu_1".to_owned(),
            tool_name: "get_weather".to_owned(),
            content: vec![ContentPart::text("boom")],
            is_error: true,
        })]);
        let body = sent_body(ctx).await;
        assert_eq!(body["messages"][0]["content"][0]["is_error"], true);
    }

    /// Build an OAuth-authenticated Provider over the given mock.
    fn oauth_provider(
        mock: std::sync::Arc<MockHttpClient>,
    ) -> AnthropicProvider<std::sync::Arc<MockHttpClient>> {
        AnthropicProvider::new(
            mock,
            Credential::oauth(OAuthTokens::new(
                "access-tok",
                "refresh-tok",
                Some(1),
            )),
            "claude-3-5-sonnet",
        )
    }

    #[tokio::test]
    async fn oauth_credential_authorizes_with_bearer_and_the_beta_header() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = oauth_provider(mock.clone());
        let ctx = Context::new(vec![Message::user("Hi")]);
        provider.complete(&ctx, &opts()).await.unwrap();

        let sent = mock.last_request();
        // The Bearer lane replaces `x-api-key` entirely.
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "authorization" && v == "Bearer access-tok")
        );
        assert!(!sent.headers.iter().any(|(k, _)| k == "x-api-key"));
        assert!(
            sent.headers.iter().any(
                |(k, v)| k == "anthropic-beta" && v == ANTHROPIC_OAUTH_BETA
            )
        );
        // The pinned version header still rides along.
        assert!(
            sent.headers.iter().any(
                |(k, v)| k == "anthropic-version" && v == ANTHROPIC_VERSION
            )
        );
    }

    #[tokio::test]
    async fn oauth_prepends_the_identity_block_ahead_of_the_caller_system() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = oauth_provider(mock.clone());
        let ctx =
            Context::new(vec![Message::user("Hi")]).with_system("Be terse.");
        provider.complete(&ctx, &opts()).await.unwrap();

        let body: serde_json::Value = serde_json::from_slice(
            mock.last_request().body.as_deref().unwrap(),
        )
        .unwrap();
        // System is an array of blocks, identity leading, caller's own next.
        assert_eq!(body["system"][0]["type"], "text");
        assert_eq!(body["system"][0]["text"], CLAUDE_CODE_IDENTITY);
        assert_eq!(body["system"][1]["text"], "Be terse.");
    }

    #[tokio::test]
    async fn oauth_sends_the_identity_block_even_without_a_caller_system() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = oauth_provider(mock.clone());
        let ctx = Context::new(vec![Message::user("Hi")]);
        provider.complete(&ctx, &opts()).await.unwrap();

        let body: serde_json::Value = serde_json::from_slice(
            mock.last_request().body.as_deref().unwrap(),
        )
        .unwrap();
        assert_eq!(body["system"][0]["text"], CLAUDE_CODE_IDENTITY);
        // No caller prompt, so the identity block stands alone.
        assert!(body["system"][1].is_null());
    }

    #[tokio::test]
    async fn oauth_normalizes_tool_names_on_the_wire() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = oauth_provider(mock.clone());
        let schema = serde_json::json!({"type": "object"});
        let ctx =
            Context::new(vec![Message::user("weather?")]).with_tools(vec![
                ToolDefinition::new(
                    "get.weather now!",
                    "Look up the weather",
                    schema,
                ),
            ]);
        provider.complete(&ctx, &opts()).await.unwrap();

        let body: serde_json::Value = serde_json::from_slice(
            mock.last_request().body.as_deref().unwrap(),
        )
        .unwrap();
        assert_eq!(body["tools"][0]["name"], "get_weather_now_");
    }

    #[tokio::test]
    async fn api_key_lane_leaves_tool_names_untouched() {
        let mock = std::sync::Arc::new(MockHttpClient::with_response(
            200,
            SAMPLE_RESPONSE,
        ));
        let provider = AnthropicProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "claude-3-5-sonnet",
        );
        let schema = serde_json::json!({"type": "object"});
        let ctx = Context::new(vec![Message::user("weather?")])
            .with_tools(vec![ToolDefinition::new("get.weather", "d", schema)]);
        provider.complete(&ctx, &opts()).await.unwrap();

        let body: serde_json::Value = serde_json::from_slice(
            mock.last_request().body.as_deref().unwrap(),
        )
        .unwrap();
        // The normalization is OAuth-only; the API-key lane passes the raw name.
        assert_eq!(body["tools"][0]["name"], "get.weather");
    }

    #[test]
    fn normalize_tool_name_borrows_a_conforming_name() {
        assert!(matches!(
            normalize_tool_name("get_weather-2"),
            Cow::Borrowed("get_weather-2")
        ));
    }

    #[test]
    fn normalize_tool_name_replaces_and_truncates() {
        assert_eq!(normalize_tool_name("a.b c/d"), "a_b_c_d");
        // Over-long names are cut to the accepted maximum.
        let long = "x".repeat(MAX_TOOL_NAME_LEN + 10);
        assert_eq!(normalize_tool_name(&long).len(), MAX_TOOL_NAME_LEN);
    }

    #[tokio::test]
    async fn url_image_and_text_mix_within_one_user_message() {
        let ctx = Context::new(vec![
            Message::user("what is this?")
                .with_image(ImageSource::url("https://example.com/cat.png")),
        ]);
        let body = sent_body(ctx).await;

        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "what is this?");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "url");
        assert_eq!(content[1]["source"]["url"], "https://example.com/cat.png");
    }

    #[tokio::test]
    async fn base64_image_carries_its_media_type_to_the_wire() {
        let ctx =
            Context::new(vec![Message::user_parts(vec![ContentPart::image(
                ImageSource::base64(MediaType::Jpeg, "aGk="),
            )])]);
        let body = sent_body(ctx).await;

        let source = &body["messages"][0]["content"][0]["source"];
        assert_eq!(source["type"], "base64");
        assert_eq!(source["media_type"], "image/jpeg");
        assert_eq!(source["data"], "aGk=");
    }

    #[tokio::test]
    async fn raw_bytes_image_is_base64_encoded_with_media_type() {
        let ctx =
            Context::new(vec![Message::user_parts(vec![ContentPart::image(
                ImageSource::bytes(MediaType::Png, b"hi".to_vec()),
            )])]);
        let body = sent_body(ctx).await;

        let source = &body["messages"][0]["content"][0]["source"];
        assert_eq!(source["type"], "base64");
        assert_eq!(source["media_type"], "image/png");
        // "hi" base64-encodes to "aGk=".
        assert_eq!(source["data"], "aGk=");
    }

    #[tokio::test]
    async fn tool_use_response_normalizes_to_a_content_part() {
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

        let ctx = Context::new(vec![Message::user("weather?")]);
        let completion = provider.complete(&ctx, &opts()).await.unwrap();

        assert_eq!(completion.text_content(), "Let me check.");
        assert_eq!(completion.finish_reason, FinishReason::ToolUse);
        let calls = tool_calls(&completion);
        assert_eq!(calls.len(), 1);
        let (id, name, arguments) = calls[0];
        // The native tool-use id becomes the part's stable handle.
        assert_eq!(id, "toolu_42");
        assert_eq!(name, "get_weather");
        assert_eq!(arguments, &serde_json::json!({"city": "Paris"}));
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

        let ctx = Context::new(vec![Message::user("hi")]);
        let err = provider.complete(&ctx, &opts()).await.unwrap_err();

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
            let ctx = Context::new(vec![Message::user("hi")]);
            let err = provider.complete(&ctx, &opts()).await.unwrap_err();
            assert_eq!(err.kind(), kind, "status {status}");
        }
    }

    #[tokio::test]
    async fn retry_after_header_is_attached_to_the_error() {
        let mock = std::sync::Arc::new(MockHttpClient::new());
        mock.push_response(crate::http::HttpResponse {
            status: 429,
            headers: vec![("Retry-After".to_owned(), "7".to_owned())],
            body: b"{}".to_vec(),
        });
        let provider = AnthropicProvider::new(
            mock,
            Credential::api_key("k"),
            "claude-3-5-sonnet",
        );

        let ctx = Context::new(vec![Message::user("hi")]);
        let err = provider.complete(&ctx, &opts()).await.unwrap_err();

        assert_eq!(err.kind(), ErrorKind::RateLimited);
        assert_eq!(err.retry_after(), Some(Duration::from_secs(7)));
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
        assert_eq!(events[1], StreamEvent::TextStart { index: 0 });
        assert_eq!(
            events[2],
            StreamEvent::TextDelta {
                index: 0,
                text: "Hello".to_owned()
            }
        );
        // The text block closes before the terminal Done.
        assert!(events.contains(&StreamEvent::TextEnd { index: 0 }));
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

        assert_eq!(folded.text_content(), "Hello, world");
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
    async fn streams_thinking_lifecycle_and_retains_it() {
        let stream = concat!(
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"let me think\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig-xyz\"}}\n\n",
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
        // The block opens, deltas, and closes; the signature rides on the end.
        assert_eq!(events[0], StreamEvent::ThinkingStart { index: 0 });
        assert_eq!(
            events[1],
            StreamEvent::ThinkingDelta {
                index: 0,
                text: "let me think".to_owned(),
            }
        );
        assert_eq!(
            events[2],
            StreamEvent::ThinkingEnd {
                index: 0,
                signature: Some("sig-xyz".to_owned()),
            }
        );
        // Thinking is not text, but it is retained on the folded completion.
        let folded = StreamAccumulator::fold(&events);
        assert_eq!(folded.text_content(), "");
        assert_eq!(folded.thinking_content(), "let me think");
        assert_eq!(
            folded.content[0],
            ContentPart::Thinking {
                text: "let me think".to_owned(),
                signature: Some("sig-xyz".to_owned()),
            }
        );
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
        let calls = tool_calls(&folded);
        assert_eq!(calls.len(), 1);
        let (id, name, arguments) = calls[0];
        assert_eq!(id, "toolu_9");
        assert_eq!(name, "get_weather");
        assert_eq!(arguments, &serde_json::json!({"city": "Paris"}));
    }
}
