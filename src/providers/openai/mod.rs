// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The OpenAI [`Provider`], talking to the Chat Completions API and
//! authenticating with a `Bearer` Credential.

mod embedding;

pub use embedding::OpenAIEmbeddingProvider;

use crate::credential::Credential;
use crate::error::{Error, ErrorKind};
use crate::http::{ByteStream, HttpClient, HttpRequest, Method};
use crate::message::{
    AssistantMessage, ContentPart, ImageSource, Message, ToolResultMessage,
};
use crate::provider::Provider;
use crate::request::{CompletionOptions, Context, ToolChoice, ToolDefinition};
use crate::response::{FinishReason, Usage};
use crate::sse::{SseDecoder, SseEvent};
use crate::stream::{StreamEvent, StreamEvents};
use crate::token_store::{TokenStore, resolve};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::VecDeque;
use std::fmt;
use std::pin::Pin;
use std::task::Poll;

/// Default base URL for the OpenAI API.
pub(super) const DEFAULT_BASE_URL: &str = "https://api.openai.com";
/// Provider id this OpenAI Provider keys its Credential under in a Token Store.
pub(crate) const PROVIDER_KEY: &str = "openai";
/// Environment variable holding an OpenAI API key, the last-resort Credential.
pub(crate) const API_KEY_ENV: &str = "OPENAI_API_KEY";
/// Alternate names that select this Provider in the [`Registry`](crate::Registry).
pub(crate) const ALIASES: &[&str] = &["gpt"];

/// This Provider's identity in the [`Registry`](crate::Registry): its canonical
/// id, the alternate names that select it, and the environment variable holding
/// its default API key.
pub const INFO: crate::registry::ProviderInfo = crate::registry::ProviderInfo {
    id: crate::model::ProviderId::from_static(PROVIDER_KEY),
    aliases: ALIASES,
    api_key_env: API_KEY_ENV,
};

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

/// The `Authorization` header value for a Credential.
///
/// OpenAI authorizes every request with a `Bearer` token, whether the Credential
/// is a long-lived API key or an OAuth access token.
pub(super) fn bearer(credential: &Credential) -> String {
    let token = match credential {
        Credential::ApiKey { key, .. } => key.as_str(),
        Credential::OAuth(tokens) => tokens.access_token.as_str(),
    };
    format!("Bearer {token}")
}

/// The auth headers a request authenticating with `credential` carries.
///
/// OpenAI's whole auth scheme is one `Authorization: Bearer` header. Both the
/// request path and the Model Registry's auth inspection go through this, so
/// what one reports is exactly what the other sends.
pub(crate) fn auth_headers(credential: &Credential) -> Vec<(String, String)> {
    vec![("authorization".to_owned(), bearer(credential))]
}

/// A Provider for OpenAI's Chat Completions API.
///
/// The transport is injected as the generic `H`, which erases to
/// `Arc<dyn Provider>` at registration. The addressable Model is fixed when the
/// Provider is built.
#[derive(Clone)]
pub struct OpenAIProvider<H> {
    http: H,
    credential: Credential,
    model: String,
    base_url: String,
    /// Caller-supplied headers appended to every request, after the Provider's
    /// own auth header. For proxies and gateways that key off a bespoke header.
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

impl<H: fmt::Debug> fmt::Debug for OpenAIProvider<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAIProvider")
            .field("http", &self.http)
            .field("credential", &self.credential)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("extra_headers", &redacted_headers(&self.extra_headers))
            .finish()
    }
}

impl<H: HttpClient> OpenAIProvider<H> {
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
    /// the `OPENAI_API_KEY` environment variable.
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
                        "no OpenAI Credential: pass one, store it under {PROVIDER_KEY:?}, or set {API_KEY_ENV}"
                    ),
                )
            })?;
        Ok(Self::new(http, credential, model))
    }

    /// Start a typed [`OpenAIBuilder`] over the injected transport for the given
    /// Model.
    ///
    /// The builder layers the advanced knobs — an explicit Credential, a
    /// base-URL override, and extra headers — over the same Credential
    /// resolution [`resolve`](Self::resolve) uses.
    #[must_use]
    pub fn builder(http: H, model: impl Into<String>) -> OpenAIBuilder<H> {
        OpenAIBuilder::new(http, model)
    }

    /// Override the base URL (for proxies, gateways, or a test server).
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Append an extra header sent with every request, after the Provider's own
    /// auth header.
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
    /// auth header.
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
        // this call.
        let override_credential =
            opts.api_key.as_deref().map(Credential::api_key);
        let credential =
            override_credential.as_ref().unwrap_or(&self.credential);

        let body = serde_json::to_vec(&WireRequest::from_context(
            &self.model,
            ctx,
            opts,
            streaming,
        ))
        .map_err(Error::serialize)?;

        let url = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        let request = HttpRequest::new(Method::Post, url)
            .header("content-type", "application/json");
        // Auth headers come from the one scheme definition, so the wire matches
        // what auth inspection reports.
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

/// A typed builder for an [`OpenAIProvider`] with advanced configuration.
///
/// It gathers an injected transport, the Model, and the optional knobs — an
/// explicit Credential, a base-URL override, and extra headers — then
/// [`build`](Self::build)s a Provider, resolving the Credential through the same
/// precedence [`OpenAIProvider::resolve`] uses (explicit, then the
/// `OPENAI_API_KEY` environment variable).
#[derive(Clone)]
pub struct OpenAIBuilder<H> {
    http: H,
    model: String,
    credential: Option<Credential>,
    base_url: Option<String>,
    headers: Vec<(String, String)>,
}

impl<H: fmt::Debug> fmt::Debug for OpenAIBuilder<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpenAIBuilder")
            .field("http", &self.http)
            .field("model", &self.model)
            .field("credential", &self.credential)
            .field("base_url", &self.base_url)
            .field("headers", &redacted_headers(&self.headers))
            .finish()
    }
}

impl<H: HttpClient> OpenAIBuilder<H> {
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
    /// [`credential`](Self::credential) if set, else the `OPENAI_API_KEY`
    /// environment variable. An empty result is an
    /// [`Authentication`](crate::ErrorKind::Authentication) error.
    pub fn build(self) -> Result<OpenAIProvider<H>, Error> {
        let mut provider = OpenAIProvider::resolve(
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
impl<H: HttpClient> Provider for OpenAIProvider<H> {
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
/// each OpenAI SSE event into the neutral vocabulary, and a queue holding the
/// events a single chunk expanded into but that have not been yielded yet.
struct SseEventStream {
    /// The response body, streamed as byte chunks.
    bytes: ByteStream,
    /// Reassembles SSE events straddling chunk boundaries.
    decoder: SseDecoder,
    /// Maps OpenAI SSE events to the neutral vocabulary.
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

/// Turns OpenAI's SSE chunks into the neutral [`StreamEvent`] vocabulary.
///
/// OpenAI streams data-only events: each `data:` line is a chat-completion chunk,
/// and a final `data: [DONE]` closes the stream. The normalizer threads the
/// small amount of state the mapping needs: whether the opening
/// [`StreamEvent::MessageStart`] has been emitted, the running usage (present
/// only in the trailing chunk when `stream_options.include_usage` is set), the
/// finish reason, and which tool-call indices have already opened.
#[derive(Debug, Default)]
struct StreamNormalizer {
    /// Whether the opening `MessageStart` has been emitted.
    started: bool,
    /// The finish reason, reported on the last content chunk.
    finish_reason: Option<FinishReason>,
    /// Running token usage, reported in the trailing usage chunk.
    usage: Usage,
    /// Text indices that have already opened, so the first fragment brackets the
    /// block with a [`StreamEvent::TextStart`] and the stream end closes it.
    text_started: Vec<usize>,
    /// Tool-call indices that have already opened, so a later argument fragment
    /// does not re-emit a [`StreamEvent::ToolCallStart`].
    tool_started: Vec<usize>,
}

impl StreamNormalizer {
    /// Map one OpenAI SSE event to zero or more neutral events.
    fn normalize(&mut self, sse: &SseEvent) -> Vec<StreamEvent> {
        // The sentinel that terminates every OpenAI stream.
        if sse.data.trim() == "[DONE]" {
            let mut events = Vec::new();
            // Close any text and tool blocks that opened, in index order,
            // before the end.
            self.text_started.sort_unstable();
            for index in self.text_started.drain(..) {
                events.push(StreamEvent::TextEnd { index });
            }
            self.tool_started.sort_unstable();
            for index in self.tool_started.drain(..) {
                events.push(StreamEvent::ToolCallEnd { index });
            }
            events.push(StreamEvent::Done {
                finish_reason: self
                    .finish_reason
                    .clone()
                    .unwrap_or(FinishReason::Other(String::new())),
                usage: self.usage,
            });
            return events;
        }

        // A payload that will not parse is not fatal; surface it verbatim.
        let json: serde_json::Value = match serde_json::from_str(&sse.data) {
            Ok(json) => json,
            Err(_) => {
                return vec![StreamEvent::Unknown(serde_json::Value::String(
                    sse.data.clone(),
                ))];
            }
        };

        let mut events = Vec::new();
        if !self.started {
            self.started = true;
            events.push(StreamEvent::MessageStart);
        }

        if let Some(choices) = json["choices"].as_array() {
            for choice in choices {
                let index = choice["index"].as_u64().unwrap_or(0) as usize;
                let delta = &choice["delta"];

                if let Some(text) = delta["content"].as_str()
                    && !text.is_empty()
                {
                    if !self.text_started.contains(&index) {
                        self.text_started.push(index);
                        events.push(StreamEvent::TextStart { index });
                    }
                    events.push(StreamEvent::TextDelta {
                        index,
                        text: text.to_owned(),
                    });
                }

                if let Some(tool_calls) = delta["tool_calls"].as_array() {
                    for call in tool_calls {
                        events.extend(self.normalize_tool_call(call));
                    }
                }

                if let Some(reason) = choice["finish_reason"].as_str() {
                    self.finish_reason = Some(map_finish_reason(Some(reason)));
                }
            }
        }

        // The trailing chunk carries usage and empty choices.
        if let Some(usage) = json.get("usage")
            && usage.is_object()
        {
            self.usage = Usage {
                input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0)
                    as u32,
                output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0)
                    as u32,
            };
            events.push(StreamEvent::Usage(self.usage));
        }

        events
    }

    /// Map one streamed tool-call fragment, opening the call the first time its
    /// index is seen and emitting an argument delta for any arguments carried.
    fn normalize_tool_call(
        &mut self,
        call: &serde_json::Value,
    ) -> Vec<StreamEvent> {
        let index = call["index"].as_u64().unwrap_or(0) as usize;
        let mut events = Vec::new();

        if !self.tool_started.contains(&index) {
            self.tool_started.push(index);
            events.push(StreamEvent::ToolCallStart {
                index,
                id: call["id"].as_str().unwrap_or_default().to_owned(),
                name: call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
            });
        }

        if let Some(arguments) = call["function"]["arguments"].as_str()
            && !arguments.is_empty()
        {
            events.push(StreamEvent::ToolCallDelta {
                index,
                partial_json: arguments.to_owned(),
            });
        }

        events
    }
}

/// The OpenAI Chat Completions request body as sent on the wire.
#[derive(Debug, Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: Vec<WireMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "is_false")]
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<WireStreamOptions>,
}

/// Serde predicate: omit a `false` flag from the request body.
fn is_false(flag: &bool) -> bool {
    !*flag
}

/// Streaming options; asks the API to include a trailing usage chunk.
#[derive(Debug, Serialize)]
struct WireStreamOptions {
    include_usage: bool,
}

impl<'a> WireRequest<'a> {
    fn from_context(
        model: &'a str,
        ctx: &'a Context,
        opts: &'a CompletionOptions,
        streaming: Streaming,
    ) -> Self {
        // The system prompt rides as a leading `system` message; the rest of
        // the conversation follows in order.
        let mut messages = Vec::with_capacity(ctx.messages.len() + 1);
        if let Some(system) = &ctx.system_prompt {
            messages.push(WireMessage::system(system.as_str()));
        }
        messages.extend(ctx.messages.iter().map(wire_message));

        let tools = ctx.tools.iter().map(WireTool::from_tool).collect();

        Self {
            model,
            messages,
            temperature: opts.temperature,
            max_tokens: opts.max_tokens,
            tools,
            tool_choice: opts.tool_choice.as_ref().map(wire_tool_choice),
            stream: streaming.enabled(),
            stream_options: streaming.enabled().then_some(WireStreamOptions {
                include_usage: true,
            }),
        }
    }
}

/// Map one neutral [`Message`] onto an OpenAI request message.
fn wire_message(message: &Message) -> WireMessage<'_> {
    match message {
        Message::User { content } => WireMessage {
            role: "user",
            content: Some(wire_content(content)),
            tool_calls: Vec::new(),
            tool_call_id: None,
        },
        Message::Assistant(assistant) => wire_assistant(assistant),
        Message::ToolResult(result) => wire_tool_result(result),
    }
}

/// Map an [`AssistantMessage`] onto an OpenAI `assistant` message, splitting its
/// content into text and a `tool_calls` array. Content is omitted (sent `null`)
/// when the reply is only tool calls, as OpenAI expects.
fn wire_assistant(assistant: &AssistantMessage) -> WireMessage<'_> {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for part in &assistant.content {
        match part {
            ContentPart::Text(chunk) => text.push_str(chunk),
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => tool_calls.push(WireToolCall {
                id,
                call_type: "function",
                function: WireToolCallFunction {
                    name,
                    arguments: arguments.to_string(),
                },
            }),
            // An assistant image is not representable on the request; skip it.
            ContentPart::Image(_) => {}
            // OpenAI Chat has no thinking block to replay; skip it.
            ContentPart::Thinking { .. } => {}
        }
    }
    let content = if text.is_empty() && !tool_calls.is_empty() {
        None
    } else {
        Some(WireContent::Text(Cow::Owned(text)))
    };
    WireMessage {
        role: "assistant",
        content,
        tool_calls,
        tool_call_id: None,
    }
}

/// Map a [`ToolResultMessage`] onto an OpenAI `tool` message, referencing the
/// call by id. OpenAI has no error flag on a tool message, so
/// [`is_error`](ToolResultMessage::is_error) is not carried on the wire.
fn wire_tool_result(result: &ToolResultMessage) -> WireMessage<'_> {
    WireMessage {
        role: "tool",
        content: Some(wire_content(&result.content)),
        tool_calls: Vec::new(),
        tool_call_id: Some(&result.tool_call_id),
    }
}

/// Map a neutral [`ToolChoice`] to OpenAI's `tool_choice` field.
fn wire_tool_choice(choice: &ToolChoice) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::json!("auto"),
        ToolChoice::Any => serde_json::json!("required"),
        ToolChoice::None => serde_json::json!("none"),
        ToolChoice::Tool(name) => serde_json::json!({
            "type": "function",
            "function": {"name": name},
        }),
    }
}

/// A tool definition as sent in the OpenAI request body.
#[derive(Debug, Serialize)]
struct WireTool<'a> {
    #[serde(rename = "type")]
    tool_type: &'static str,
    function: WireFunction<'a>,
}

/// The `function` object of an OpenAI tool definition.
#[derive(Debug, Serialize)]
struct WireFunction<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a serde_json::Value,
}

impl<'a> WireTool<'a> {
    fn from_tool(tool: &'a ToolDefinition) -> Self {
        Self {
            tool_type: "function",
            function: WireFunction {
                name: &tool.name,
                description: &tool.description,
                parameters: &tool.input_schema,
            },
        }
    }
}

/// A single message in the OpenAI request body.
///
/// The optional fields cover the several message shapes: a `tool` message
/// carries a [`tool_call_id`](Self::tool_call_id), an `assistant` message may
/// carry [`tool_calls`](Self::tool_calls) with `null` content, and every other
/// message carries [`content`](Self::content).
#[derive(Debug, Serialize)]
struct WireMessage<'a> {
    role: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<WireContent<'a>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<WireToolCall<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<&'a str>,
}

impl<'a> WireMessage<'a> {
    /// A `system` message carrying the given prompt.
    fn system(prompt: &'a str) -> Self {
        Self {
            role: "system",
            content: Some(WireContent::Text(Cow::Borrowed(prompt))),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}

/// A tool call in an OpenAI `assistant` message.
#[derive(Debug, Serialize)]
struct WireToolCall<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    call_type: &'static str,
    function: WireToolCallFunction<'a>,
}

/// The `function` object of an OpenAI assistant tool call; `arguments` is a
/// JSON string, as the API expects.
#[derive(Debug, Serialize)]
struct WireToolCallFunction<'a> {
    name: &'a str,
    arguments: String,
}

/// A message's content on the wire.
///
/// OpenAI accepts either a plain string or an array of typed parts. A message
/// that is a single text part serializes as the string form, keeping the common
/// case compact; anything else (images, or multiple parts) serializes as parts.
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum WireContent<'a> {
    Text(Cow<'a, str>),
    Parts(Vec<WireContentPart<'a>>),
}

/// One content part in the OpenAI request body.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum WireContentPart<'a> {
    Text { text: &'a str },
    ImageUrl { image_url: WireImageUrl<'a> },
}

/// An image part's `image_url` object; the URL is either a fetchable link or an
/// inline `data:` URL for base64 and raw-bytes images.
#[derive(Debug, Serialize)]
struct WireImageUrl<'a> {
    url: Cow<'a, str>,
}

/// Map neutral content parts onto OpenAI's message content.
///
/// A tool-call part never appears in the content of a user or tool message, so
/// it is dropped here; assistant tool calls travel in the `tool_calls` array.
fn wire_content(parts: &[ContentPart]) -> WireContent<'_> {
    if let [ContentPart::Text(text)] = parts {
        return WireContent::Text(Cow::Borrowed(text.as_str()));
    }
    WireContent::Parts(parts.iter().filter_map(wire_content_part).collect())
}

/// Map one neutral content part onto an OpenAI content part, or `None` for a
/// part with no request-content representation.
fn wire_content_part(part: &ContentPart) -> Option<WireContentPart<'_>> {
    match part {
        ContentPart::Text(text) => Some(WireContentPart::Text { text }),
        ContentPart::Image(source) => Some(WireContentPart::ImageUrl {
            image_url: WireImageUrl {
                url: wire_image_url(source),
            },
        }),
        ContentPart::ToolCall { .. } => None,
        ContentPart::Thinking { .. } => None,
    }
}

/// Build the `image_url` for an image source: a URL passes through, while base64
/// and raw bytes become an inline `data:` URL carrying the media type.
fn wire_image_url(source: &ImageSource) -> Cow<'_, str> {
    match source {
        ImageSource::Url(url) => Cow::Borrowed(url.as_str()),
        ImageSource::Base64 { media_type, data } => {
            Cow::Owned(format!("data:{};base64,{}", media_type.as_wire(), data))
        }
        ImageSource::Bytes { media_type, data } => Cow::Owned(format!(
            "data:{};base64,{}",
            media_type.as_wire(),
            crate::base64::base64_encode(data)
        )),
    }
}

/// The OpenAI Chat Completions response body.
#[derive(Debug, Deserialize)]
struct WireResponse {
    #[serde(default)]
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: WireUsage,
}

impl WireResponse {
    fn into_message(self, raw: serde_json::Value) -> AssistantMessage {
        let usage = Usage {
            input_tokens: self.usage.prompt_tokens,
            output_tokens: self.usage.completion_tokens,
        };
        // Chat Completions returns a single choice for the default `n`; take the
        // first and leave the rest to the raw escape hatch.
        let Some(choice) = self.choices.into_iter().next() else {
            return AssistantMessage {
                content: Vec::new(),
                usage,
                finish_reason: FinishReason::Other(String::new()),
                raw: Some(raw),
            };
        };

        // Text leads, then tool calls, mirroring how the reply reads back.
        let mut content = Vec::new();
        if let Some(text) = choice.message.content
            && !text.is_empty()
        {
            content.push(ContentPart::Text(text));
        }
        for call in choice.message.tool_calls {
            content.push(ContentPart::ToolCall {
                id: call.id,
                name: call.function.name,
                arguments: parse_arguments(&call.function.arguments),
            });
        }

        AssistantMessage {
            content,
            usage,
            finish_reason: map_finish_reason(choice.finish_reason.as_deref()),
            raw: Some(raw),
        }
    }
}

/// Parse a tool call's JSON-string arguments into a value, defaulting an empty
/// or unparsable payload to an empty object so a caller always gets a value.
fn parse_arguments(arguments: &str) -> serde_json::Value {
    if arguments.trim().is_empty() {
        return serde_json::json!({});
    }
    serde_json::from_str(arguments).unwrap_or_else(|_| serde_json::json!({}))
}

/// One choice in the OpenAI response.
#[derive(Debug, Deserialize)]
struct WireChoice {
    message: WireResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

/// The assistant message in a response choice.
#[derive(Debug, Deserialize)]
struct WireResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<WireResponseToolCall>,
}

/// A tool call in a response choice.
#[derive(Debug, Deserialize)]
struct WireResponseToolCall {
    id: String,
    function: WireResponseFunction,
}

/// The `function` object of a response tool call.
#[derive(Debug, Deserialize)]
struct WireResponseFunction {
    #[serde(default)]
    name: String,
    #[serde(default)]
    arguments: String,
}

/// Token usage in the OpenAI response.
#[derive(Debug, Default, Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

/// Map an OpenAI `finish_reason` to the neutral [`FinishReason`].
fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("stop") => FinishReason::Stop,
        Some("length") => FinishReason::MaxTokens,
        Some("tool_calls" | "function_call") => FinishReason::ToolUse,
        Some(other) => FinishReason::Other(other.to_owned()),
        None => FinishReason::Other(String::new()),
    }
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use crate::http::MockHttpClient;
    use crate::message::{MediaType, Message};
    use crate::stream::StreamAccumulator;
    use futures_util::StreamExt;
    use std::sync::Arc;
    use std::time::Duration;

    const SAMPLE_RESPONSE: &str = r#"{
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there!"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}
    }"#;

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

    /// Send one context/options pair and return its parsed JSON body.
    async fn sent_body(
        ctx: Context,
        opts: CompletionOptions,
    ) -> serde_json::Value {
        let mock =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = OpenAIProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "gpt-4o-mini",
        );
        provider.complete(&ctx, &opts).await.unwrap();
        serde_json::from_slice(mock.last_request().body.as_deref().unwrap())
            .unwrap()
    }

    async fn collect_stream(
        provider: &OpenAIProvider<Arc<MockHttpClient>>,
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

    #[tokio::test]
    async fn completes_a_text_prompt_through_the_injected_transport() {
        let mock =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = OpenAIProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "gpt-4o-mini",
        );

        let ctx = Context::new(vec![Message::user("Hello")]);
        let response = provider.complete(&ctx, &opts()).await.unwrap();

        assert_eq!(response.text_content(), "Hello there!");
        assert_eq!(response.usage.input_tokens, 12);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.raw.as_ref().unwrap()["id"], "chatcmpl-123");
    }

    #[tokio::test]
    async fn resolves_api_key_from_the_token_store_and_completes() {
        use crate::token_store::InMemoryTokenStore;

        let store = InMemoryTokenStore::new();
        store
            .set(PROVIDER_KEY, Credential::api_key("sk-stored"))
            .unwrap();
        let mock =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = OpenAIProvider::resolve(
            mock.clone(),
            "gpt-4o-mini",
            None,
            Some(&store),
        )
        .unwrap();

        let ctx = Context::new(vec![Message::user("Hello")]);
        let response = provider.complete(&ctx, &opts()).await.unwrap();

        assert_eq!(response.text_content(), "Hello there!");
        assert!(
            mock.last_request()
                .headers
                .iter()
                .any(|(k, v)| k == "authorization" && v == "Bearer sk-stored")
        );
    }

    #[tokio::test]
    async fn per_request_api_key_overrides_the_constructed_credential() {
        let mock =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = OpenAIProvider::new(
            mock.clone(),
            Credential::api_key("sk-constructed"),
            "gpt-4o-mini",
        );

        let ctx = Context::new(vec![Message::user("Hi")]);
        let opts = opts().with_api_key("sk-explicit");
        provider.complete(&ctx, &opts).await.unwrap();

        let sent = mock.last_request();
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "authorization" && v == "Bearer sk-explicit")
        );
        assert!(
            !sent
                .headers
                .iter()
                .any(|(_, v)| v == "Bearer sk-constructed")
        );
    }

    #[tokio::test]
    async fn sends_bearer_auth_and_the_body() {
        let mock =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = OpenAIProvider::new(
            mock.clone(),
            Credential::api_key("sk-secret"),
            "gpt-4o-mini",
        );

        let ctx =
            Context::new(vec![Message::user("Hi")]).with_system("Be terse.");
        let opts = opts().with_temperature(0.2).with_max_tokens(64);
        provider.complete(&ctx, &opts).await.unwrap();

        let sent = mock.last_request();
        assert_eq!(sent.method, Method::Post);
        assert!(sent.url.ends_with("/v1/chat/completions"));
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "authorization" && v == "Bearer sk-secret")
        );

        let body: serde_json::Value =
            serde_json::from_slice(sent.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["model"], "gpt-4o-mini");
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["temperature"], 0.2);
        // The system prompt leads as a plain `system` message.
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "Be terse.");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["messages"][1]["content"], "Hi");
        // A non-streaming request omits the stream flags.
        assert!(body.get("stream").is_none());
        assert!(body.get("stream_options").is_none());
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
        let body = sent_body(ctx, opts).await;

        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            body["tools"][0]["function"]["description"],
            "Look up the weather for a city"
        );
        assert_eq!(body["tools"][0]["function"]["parameters"], schema);
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "get_weather");
    }

    #[tokio::test]
    async fn tool_choice_variants_map_to_their_wire_form() {
        for (choice, expected) in [
            (ToolChoice::Auto, serde_json::json!("auto")),
            (ToolChoice::Any, serde_json::json!("required")),
            (ToolChoice::None, serde_json::json!("none")),
        ] {
            let ctx = Context::new(vec![Message::user("hi")]);
            let body = sent_body(ctx, opts().with_tool_choice(choice)).await;
            assert_eq!(body["tool_choice"], expected);
        }
    }

    #[tokio::test]
    async fn omits_tools_and_choice_when_unset() {
        let body =
            sent_body(Context::new(vec![Message::user("hi")]), opts()).await;
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }

    #[tokio::test]
    async fn assistant_tool_call_and_tool_result_round_trip_onto_the_wire() {
        // A full tool turn appended back into the context: the assistant's tool
        // call, then the result answering it.
        let ctx = Context::new(vec![
            Message::user("weather?"),
            Message::Assistant(AssistantMessage {
                content: vec![ContentPart::tool_call(
                    "call_7",
                    "get_weather",
                    serde_json::json!({"city": "Paris"}),
                )],
                usage: Usage::default(),
                finish_reason: FinishReason::ToolUse,
                raw: None,
            }),
            Message::tool_result("call_7", "get_weather", "sunny"),
        ]);
        let body = sent_body(ctx, opts()).await;

        // The assistant message carries the call in `tool_calls`, content null.
        let assistant = &body["messages"][1];
        assert_eq!(assistant["role"], "assistant");
        assert!(assistant["content"].is_null());
        assert_eq!(assistant["tool_calls"][0]["id"], "call_7");
        assert_eq!(assistant["tool_calls"][0]["type"], "function");
        assert_eq!(
            assistant["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        // Arguments ride as a JSON string.
        assert_eq!(
            assistant["tool_calls"][0]["function"]["arguments"],
            "{\"city\":\"Paris\"}"
        );
        // The result is a `tool` message referencing the call by id.
        let result = &body["messages"][2];
        assert_eq!(result["role"], "tool");
        assert_eq!(result["tool_call_id"], "call_7");
        assert_eq!(result["content"], "sunny");
    }

    #[tokio::test]
    async fn tool_call_response_normalizes_to_a_content_part() {
        let response = r#"{
            "id": "chatcmpl-tool",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_42",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4}
        }"#;
        let mock = Arc::new(MockHttpClient::with_response(200, response));
        let provider = OpenAIProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "gpt-4o-mini",
        );

        let ctx = Context::new(vec![Message::user("weather?")]);
        let completion = provider.complete(&ctx, &opts()).await.unwrap();

        assert_eq!(completion.text_content(), "");
        assert_eq!(completion.finish_reason, FinishReason::ToolUse);
        let calls = tool_calls(&completion);
        assert_eq!(calls.len(), 1);
        let (id, name, arguments) = calls[0];
        // The native call id becomes the part's stable handle.
        assert_eq!(id, "call_42");
        assert_eq!(name, "get_weather");
        assert_eq!(arguments, &serde_json::json!({"city": "Paris"}));
    }

    #[tokio::test]
    async fn non_2xx_maps_to_a_typed_error_kind() {
        let mock = Arc::new(MockHttpClient::with_response(
            401,
            r#"{"error":{"message":"invalid key"}}"#,
        ));
        let provider = OpenAIProvider::new(
            mock,
            Credential::api_key("bad"),
            "gpt-4o-mini",
        );

        let ctx = Context::new(vec![Message::user("hi")]);
        let err = provider.complete(&ctx, &opts()).await.unwrap_err();

        assert_eq!(err.kind(), ErrorKind::Authentication);
        assert_eq!(err.status(), Some(401));
    }

    #[tokio::test]
    async fn retry_after_header_is_attached_to_the_error() {
        let mock = Arc::new(MockHttpClient::new());
        mock.push_response(crate::http::HttpResponse {
            status: 429,
            headers: vec![("Retry-After".to_owned(), "7".to_owned())],
            body: b"{}".to_vec(),
        });
        let provider =
            OpenAIProvider::new(mock, Credential::api_key("k"), "gpt-4o-mini");

        let ctx = Context::new(vec![Message::user("hi")]);
        let err = provider.complete(&ctx, &opts()).await.unwrap_err();

        assert_eq!(err.kind(), ErrorKind::RateLimited);
        assert_eq!(err.retry_after(), Some(Duration::from_secs(7)));
    }

    #[tokio::test]
    async fn url_image_and_text_mix_within_one_user_message() {
        let ctx = Context::new(vec![
            Message::user("what is this?")
                .with_image(ImageSource::url("https://example.com/cat.png")),
        ]);
        let body = sent_body(ctx, opts()).await;

        let content = &body["messages"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "what is this?");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(
            content[1]["image_url"]["url"],
            "https://example.com/cat.png"
        );
    }

    #[tokio::test]
    async fn raw_bytes_image_becomes_a_base64_data_url() {
        let ctx =
            Context::new(vec![Message::user_parts(vec![ContentPart::image(
                ImageSource::bytes(MediaType::Png, b"hi".to_vec()),
            )])]);
        let body = sent_body(ctx, opts()).await;

        // "hi" base64-encodes to "aGk=".
        assert_eq!(
            body["messages"][0]["content"][0]["image_url"]["url"],
            "data:image/png;base64,aGk="
        );
    }

    /// A full OpenAI chat stream: two text deltas, a finish chunk, a usage
    /// chunk, then the terminal sentinel.
    const SAMPLE_STREAM: &str = concat!(
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\", world\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":5,\"total_tokens\":12}}\n\n",
        "data: [DONE]\n\n",
    );

    #[tokio::test]
    async fn streams_ordered_events_ending_in_done() {
        let mock = Arc::new(MockHttpClient::with_stream(vec![
            SAMPLE_STREAM.as_bytes().to_vec(),
        ]));
        let provider = OpenAIProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "gpt-4o-mini",
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
        // The synthesized text block closes before the terminal Done.
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

        // The request opted into streaming and asked for a usage chunk.
        let body: serde_json::Value = serde_json::from_slice(
            mock.last_request().body.as_deref().unwrap(),
        )
        .unwrap();
        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[tokio::test]
    async fn accumulator_folds_stream_into_the_non_streaming_shape() {
        let mock = Arc::new(MockHttpClient::with_stream(vec![
            SAMPLE_STREAM.as_bytes().to_vec(),
        ]));
        let provider = OpenAIProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "gpt-4o-mini",
        );

        let events = collect_stream(&provider).await;
        let folded = StreamAccumulator::fold(&events);

        assert_eq!(folded.text_content(), "Hello, world");
        assert_eq!(folded.finish_reason, FinishReason::Stop);
        assert_eq!(folded.usage.input_tokens, 7);
        assert_eq!(folded.usage.output_tokens, 5);
    }

    #[tokio::test]
    async fn decodes_events_split_across_chunk_boundaries() {
        let raw = SAMPLE_STREAM.as_bytes();
        let mid = raw.len() / 2;
        let chunks = vec![raw[..mid].to_vec(), raw[mid..].to_vec()];
        let mock = Arc::new(MockHttpClient::with_stream(chunks));
        let provider = OpenAIProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "gpt-4o-mini",
        );

        let events = collect_stream(&provider).await;
        assert_eq!(events.first(), Some(&StreamEvent::MessageStart));
        assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
    }

    #[tokio::test]
    async fn folded_tool_stream_yields_a_complete_tool_call() {
        // The tool id and name arrive on the first fragment; arguments split
        // across the next two.
        let stream = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_9\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\":\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":6,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n",
        );
        let mock = Arc::new(MockHttpClient::with_stream(vec![
            stream.as_bytes().to_vec(),
        ]));
        let provider = OpenAIProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "gpt-4o-mini",
        );

        let events = collect_stream(&provider).await;
        assert_eq!(
            events[1],
            StreamEvent::ToolCallStart {
                index: 0,
                id: "call_9".to_owned(),
                name: "get_weather".to_owned(),
            }
        );
        assert!(events.contains(&StreamEvent::ToolCallEnd { index: 0 }));

        let folded = StreamAccumulator::fold(&events);
        assert_eq!(folded.finish_reason, FinishReason::ToolUse);
        let calls = tool_calls(&folded);
        assert_eq!(calls.len(), 1);
        let (id, name, arguments) = calls[0];
        assert_eq!(id, "call_9");
        assert_eq!(name, "get_weather");
        assert_eq!(arguments, &serde_json::json!({"city": "Paris"}));
    }

    #[tokio::test]
    async fn unknown_sse_payloads_surface_as_unknown_events() {
        let stream = concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: not json at all\n\n",
            "data: [DONE]\n\n",
        );
        let mock = Arc::new(MockHttpClient::with_stream(vec![
            stream.as_bytes().to_vec(),
        ]));
        let provider = OpenAIProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "gpt-4o-mini",
        );

        let events = collect_stream(&provider).await;
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::Unknown(serde_json::Value::String(_))
        )));
    }

    #[test]
    fn builder_layers_base_url_and_headers() {
        let mock = Arc::new(MockHttpClient::new());
        let provider = OpenAIProvider::builder(mock, "gpt-4o-mini")
            .credential(Credential::api_key("sk-test"))
            .base_url("https://proxy.test")
            .header("x-proxy", "1")
            .build()
            .unwrap();

        let http = provider
            .build_http_request(
                &Context::new(vec![Message::user("hi")]),
                &opts(),
                Streaming::Off,
            )
            .unwrap();
        assert!(
            http.url
                .starts_with("https://proxy.test/v1/chat/completions")
        );
        assert!(http.headers.iter().any(|(k, v)| k == "x-proxy" && v == "1"));
    }

    #[test]
    fn debug_redacts_extra_header_values() {
        let mock = Arc::new(MockHttpClient::new());
        let provider = OpenAIProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "gpt-4o-mini",
        )
        .with_header("x-proxy-token", "super-secret");
        let rendered = format!("{provider:?}");
        assert!(!rendered.contains("super-secret"));
        assert!(rendered.contains("x-proxy-token"));
    }
}
