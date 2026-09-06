// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The DeepSeek [`Provider`], talking to the OpenAI-compatible Chat Completions
//! API and authenticating with a `Bearer` Credential.
//!
//! DeepSeek speaks the OpenAI Chat Completions protocol, so this reuses the
//! OpenAI request builder (`WireRequest`) and the `openai-completions` `Api` tag,
//! supplying only a bespoke response half (per ADR-0011). The response half
//! exists for `reasoning_content`: the model's
//! reasoning, present on the reasoning models, which maps to a
//! [`ContentPart::Thinking`] on the buffered path here, and its disjoint prompt
//! split, which maps onto [`Usage`].

use crate::credential::Credential;
use crate::error::{Error, ErrorKind};
use crate::http::HttpClient;
use crate::message::{AssistantMessage, ContentPart};
use crate::pipeline::{
    CompletionPipeline, Streaming, WireAdapter, redacted_headers,
};
use crate::provider::Provider;
use crate::providers::openai::{OpenAIStreamNormalizer, WireRequest};
use crate::request::{CompletionOptions, Context};
use crate::response::{FinishReason, Usage};
use crate::stream::StreamEvents;
use crate::token_store::{TokenStore, resolve};
use async_trait::async_trait;
use serde::Deserialize;
use std::fmt;

/// Default base URL for the DeepSeek API.
pub(super) const DEFAULT_BASE_URL: &str = "https://api.deepseek.com";
/// Provider id this DeepSeek Provider keys its Credential under in a Token Store.
pub(crate) const PROVIDER_KEY: &str = "deepseek";
/// Environment variable holding a DeepSeek API key, the last-resort Credential.
pub(crate) const API_KEY_ENV: &str = "DEEPSEEK_API_KEY";
/// Alternate names that select this Provider in the [`Registry`](crate::Registry).
pub(crate) const ALIASES: &[&str] = &[];

/// This Provider's identity in the [`Registry`](crate::Registry): its canonical
/// id, the alternate names that select it, and the environment variable holding
/// its default API key.
pub const INFO: crate::registry::ProviderInfo = crate::registry::ProviderInfo {
    id: crate::model::ProviderId::from_static(PROVIDER_KEY),
    aliases: ALIASES,
    api_key_env: API_KEY_ENV,
};

/// The `Authorization` header value for a Credential.
///
/// DeepSeek authorizes every request with a `Bearer` token, whether the
/// Credential is a long-lived API key or an OAuth access token.
fn bearer(credential: &Credential) -> String {
    let token = match credential {
        Credential::ApiKey { key, .. } => key.as_str(),
        Credential::OAuth(tokens) => tokens.access_token.as_str(),
    };
    format!("Bearer {token}")
}

/// The auth headers a request authenticating with `credential` carries.
///
/// DeepSeek's whole auth scheme is one `Authorization: Bearer` header. Both the
/// request path and the Model Registry's auth inspection go through this, so
/// what one reports is exactly what the other sends.
pub(crate) fn auth_headers(credential: &Credential) -> Vec<(String, String)> {
    vec![("authorization".to_owned(), bearer(credential))]
}

/// The DeepSeek wire specifics behind the shared [`CompletionPipeline`].
///
/// It supplies only what varies for DeepSeek: the Chat Completions endpoint, the
/// `Authorization: Bearer` auth header, the request body (reused wholesale from
/// OpenAI's [`WireRequest`]), and the mapping from a decoded [`WireResponse`]
/// into an [`AssistantMessage`]. Streaming rides OpenAI's normalizer until the
/// bespoke reasoning normalizer lands (a later issue); the pipeline owns
/// everything invariant around these.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DeepSeekWire;

impl WireAdapter for DeepSeekWire {
    type Response = WireResponse;
    type Normalizer = OpenAIStreamNormalizer;

    fn endpoint(&self) -> &str {
        "/chat/completions"
    }

    fn auth_headers(&self, credential: &Credential) -> Vec<(String, String)> {
        auth_headers(credential)
    }

    fn request_body(
        &self,
        model: &str,
        ctx: &Context,
        opts: &CompletionOptions,
        streaming: Streaming,
        _credential: &Credential,
    ) -> Result<Vec<u8>, Error> {
        serde_json::to_vec(&WireRequest::from_context(
            model, ctx, opts, streaming,
        ))
        .map_err(Error::serialize)
    }

    fn map_message(
        &self,
        response: WireResponse,
        raw: serde_json::Value,
    ) -> AssistantMessage {
        response.into_message(raw)
    }

    fn normalizer(&self) -> OpenAIStreamNormalizer {
        OpenAIStreamNormalizer::default()
    }
}

/// A Provider for DeepSeek's OpenAI-compatible Chat Completions API.
///
/// A newtype over the shared `CompletionPipeline` driving a `DeepSeekWire`
/// adapter: the pipeline carries the invariant send / success-gate / decode /
/// stream-wrap flow and holds the Provider fields (transport `H`, Credential,
/// Model, base URL, extra headers). It erases to `Arc<dyn Provider>` at
/// registration; the addressable Model is fixed when the Provider is built.
#[derive(Clone, Debug)]
pub struct DeepSeekProvider<H>(CompletionPipeline<H, DeepSeekWire>);

impl<H: HttpClient> DeepSeekProvider<H> {
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
        Self(CompletionPipeline::new(
            http,
            DeepSeekWire,
            credential,
            model,
            DEFAULT_BASE_URL,
        ))
    }

    /// Build a Provider by resolving its Credential from a Token Store, then
    /// the `DEEPSEEK_API_KEY` environment variable.
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
                        "no DeepSeek Credential: pass one, store it under {PROVIDER_KEY:?}, or set {API_KEY_ENV}"
                    ),
                )
            })?;
        Ok(Self::new(http, credential, model))
    }

    /// Start a typed [`DeepSeekBuilder`] over the injected transport for the
    /// given Model.
    ///
    /// The builder layers the advanced knobs — an explicit Credential, a
    /// base-URL override, and extra headers — over the same Credential
    /// resolution [`resolve`](Self::resolve) uses.
    #[must_use]
    pub fn builder(http: H, model: impl Into<String>) -> DeepSeekBuilder<H> {
        DeepSeekBuilder::new(http, model)
    }

    /// Override the base URL (for proxies, gateways, or a test server).
    #[must_use]
    pub fn with_base_url(self, base_url: impl Into<String>) -> Self {
        Self(self.0.with_base_url(base_url))
    }

    /// Append an extra header sent with every request, after the Provider's own
    /// auth header.
    #[must_use]
    pub fn with_header(
        self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self(self.0.with_header(name, value))
    }

    /// Append extra headers sent with every request, after the Provider's own
    /// auth header.
    #[must_use]
    pub fn with_headers(
        self,
        headers: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        Self(self.0.with_headers(headers))
    }
}

/// A typed builder for a [`DeepSeekProvider`] with advanced configuration.
///
/// It gathers an injected transport, the Model, and the optional knobs — an
/// explicit Credential, a base-URL override, and extra headers — then
/// [`build`](Self::build)s a Provider, resolving the Credential through the same
/// precedence [`DeepSeekProvider::resolve`] uses (explicit, then the
/// `DEEPSEEK_API_KEY` environment variable).
#[derive(Clone)]
pub struct DeepSeekBuilder<H> {
    http: H,
    model: String,
    credential: Option<Credential>,
    base_url: Option<String>,
    headers: Vec<(String, String)>,
}

impl<H: fmt::Debug> fmt::Debug for DeepSeekBuilder<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeepSeekBuilder")
            .field("http", &self.http)
            .field("model", &self.model)
            .field("credential", &self.credential)
            .field("base_url", &self.base_url)
            .field("headers", &redacted_headers(&self.headers))
            .finish()
    }
}

impl<H: HttpClient> DeepSeekBuilder<H> {
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
    /// [`credential`](Self::credential) if set, else the `DEEPSEEK_API_KEY`
    /// environment variable. An empty result is an
    /// [`Authentication`](crate::ErrorKind::Authentication) error.
    pub fn build(self) -> Result<DeepSeekProvider<H>, Error> {
        let mut provider = DeepSeekProvider::resolve(
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
impl<H: HttpClient> Provider for DeepSeekProvider<H> {
    async fn complete(
        &self,
        ctx: &Context,
        opts: &CompletionOptions,
    ) -> Result<AssistantMessage, Error> {
        self.0.complete(ctx, opts).await
    }

    async fn complete_stream(
        &self,
        ctx: &Context,
        opts: &CompletionOptions,
    ) -> Result<StreamEvents, Error> {
        self.0.complete_stream(ctx, opts).await
    }
}

/// The DeepSeek Chat Completions response body.
///
/// The bespoke half of the reuse: identical in shape to OpenAI's but for
/// `reasoning_content` on the message and the disjoint prompt split on the
/// usage, both of which DeepSeek adds and this maps.
#[derive(Debug, Deserialize)]
pub(crate) struct WireResponse {
    #[serde(default)]
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: WireUsage,
}

impl WireResponse {
    fn into_message(self, raw: serde_json::Value) -> AssistantMessage {
        let usage = self.usage.into_usage();
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

        // Reasoning leads, then text, then tool calls — the order the reply
        // reads back, and the order DeepSeek streams them (ADR-0011).
        let mut content = Vec::new();
        if let Some(reasoning) = choice.message.reasoning_content
            && !reasoning.is_empty()
        {
            // DeepSeek supplies no replay signature for its reasoning.
            content.push(ContentPart::Thinking {
                text: reasoning,
                signature: None,
            });
        }
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

/// One choice in the DeepSeek response.
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
    /// The model's reasoning, present on the reasoning models.
    #[serde(default)]
    reasoning_content: Option<String>,
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

/// Token usage in the DeepSeek response.
///
/// DeepSeek reports the prompt as a disjoint split — the cache miss and hit
/// counts sum to the prompt total — rather than OpenAI's single `prompt_tokens`.
#[derive(Debug, Default, Deserialize)]
struct WireUsage {
    /// Prompt tokens served neither from nor into the cache.
    #[serde(default)]
    prompt_cache_miss_tokens: u32,
    /// Prompt tokens served from the cache, disjoint from the miss count.
    #[serde(default)]
    prompt_cache_hit_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

impl WireUsage {
    /// Map DeepSeek's disjoint prompt split onto the neutral [`Usage`].
    ///
    /// The cache miss count is the uncached input; the hit count is the cache
    /// read. Caching is automatic and disk-based, with no write cost or write
    /// counter, so `cache_write_tokens` is `0`. `reasoning_tokens` is dropped:
    /// there is no neutral field for it and it is already counted inside
    /// `completion_tokens` (ADR-0011).
    fn into_usage(self) -> Usage {
        Usage {
            input_tokens: self.prompt_cache_miss_tokens,
            output_tokens: self.completion_tokens,
            cache_read_tokens: self.prompt_cache_hit_tokens,
            cache_write_tokens: 0,
        }
    }
}

/// Map a DeepSeek `finish_reason` to the neutral [`FinishReason`].
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
    use crate::http::{Method, MockHttpClient};
    use crate::message::Message;
    use crate::request::{ToolChoice, ToolDefinition};
    use std::sync::Arc;
    use std::time::Duration;

    /// A buffered reply carrying reasoning, text, and the disjoint prompt split.
    const SAMPLE_RESPONSE: &str = r#"{
        "id": "chatcmpl-ds",
        "object": "chat.completion",
        "model": "deepseek-chat",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": "Let me think.",
                "content": "Hello there!"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 12,
            "prompt_cache_miss_tokens": 10,
            "prompt_cache_hit_tokens": 2,
            "completion_tokens": 5,
            "total_tokens": 17
        }
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
        let provider = DeepSeekProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "deepseek-chat",
        );
        provider.complete(&ctx, &opts).await.unwrap();
        serde_json::from_slice(mock.last_request().body.as_deref().unwrap())
            .unwrap()
    }

    #[tokio::test]
    async fn completes_a_text_prompt_capturing_reasoning_and_usage() {
        let mock =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = DeepSeekProvider::new(
            mock.clone(),
            Credential::api_key("sk-test"),
            "deepseek-chat",
        );

        let ctx = Context::new(vec![Message::user("Hello")]);
        let response = provider.complete(&ctx, &opts()).await.unwrap();

        assert_eq!(response.text_content(), "Hello there!");
        // Reasoning is captured on the buffered path as a leading Thinking part.
        assert!(matches!(
            response.content.first(),
            Some(ContentPart::Thinking { text, signature: None })
                if text == "Let me think."
        ));
        // The disjoint prompt split maps onto the neutral usage.
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.cache_read_tokens, 2);
        assert_eq!(response.usage.cache_write_tokens, 0);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.finish_reason, FinishReason::Stop);
        assert_eq!(response.raw.as_ref().unwrap()["id"], "chatcmpl-ds");
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
            "usage": {"prompt_cache_miss_tokens": 9, "completion_tokens": 4}
        }"#;
        let mock = Arc::new(MockHttpClient::with_response(200, response));
        let provider = DeepSeekProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "deepseek-chat",
        );

        let ctx = Context::new(vec![Message::user("weather?")]);
        let completion = provider.complete(&ctx, &opts()).await.unwrap();

        assert_eq!(completion.text_content(), "");
        assert_eq!(completion.finish_reason, FinishReason::ToolUse);
        let calls = tool_calls(&completion);
        assert_eq!(calls.len(), 1);
        let (id, name, arguments) = calls[0];
        assert_eq!(id, "call_42");
        assert_eq!(name, "get_weather");
        assert_eq!(arguments, &serde_json::json!({"city": "Paris"}));
    }

    #[tokio::test]
    async fn sends_bearer_auth_to_the_chat_completions_endpoint() {
        let mock =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = DeepSeekProvider::new(
            mock.clone(),
            Credential::api_key("sk-secret"),
            "deepseek-chat",
        );

        let ctx = Context::new(vec![Message::user("Hi")]);
        provider.complete(&ctx, &opts()).await.unwrap();

        let sent = mock.last_request();
        assert_eq!(sent.method, Method::Post);
        // The chat-completions endpoint under the DeepSeek base URL.
        assert!(sent.url.starts_with("https://api.deepseek.com"));
        assert!(sent.url.ends_with("/chat/completions"));
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "authorization" && v == "Bearer sk-secret")
        );

        // The body is OpenAI's shape, addressed to the DeepSeek Model.
        let body: serde_json::Value =
            serde_json::from_slice(sent.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["model"], "deepseek-chat");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "Hi");
    }

    #[tokio::test]
    async fn reuses_the_openai_request_builder_for_tools() {
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
        assert_eq!(body["tools"][0]["function"]["parameters"], schema);
        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "get_weather");
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
        let provider = DeepSeekProvider::resolve(
            mock.clone(),
            "deepseek-chat",
            None,
            Some(&store),
        )
        .unwrap();

        let ctx = Context::new(vec![Message::user("Hello")]);
        provider.complete(&ctx, &opts()).await.unwrap();

        assert!(
            mock.last_request()
                .headers
                .iter()
                .any(|(k, v)| k == "authorization" && v == "Bearer sk-stored")
        );
    }

    #[tokio::test]
    async fn non_2xx_maps_to_a_typed_error_kind() {
        let mock = Arc::new(MockHttpClient::with_response(
            401,
            r#"{"error":{"message":"invalid key"}}"#,
        ));
        let provider = DeepSeekProvider::new(
            mock,
            Credential::api_key("bad"),
            "deepseek-chat",
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
        let provider = DeepSeekProvider::new(
            mock,
            Credential::api_key("k"),
            "deepseek-chat",
        );

        let ctx = Context::new(vec![Message::user("hi")]);
        let err = provider.complete(&ctx, &opts()).await.unwrap_err();

        assert_eq!(err.kind(), ErrorKind::RateLimited);
        assert_eq!(err.retry_after(), Some(Duration::from_secs(7)));
    }

    #[tokio::test]
    async fn builder_layers_base_url_and_headers() {
        let mock =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = DeepSeekProvider::builder(mock.clone(), "deepseek-chat")
            .credential(Credential::api_key("sk-test"))
            .base_url("https://proxy.test")
            .header("x-proxy", "1")
            .build()
            .unwrap();

        let ctx = Context::new(vec![Message::user("hi")]);
        provider.complete(&ctx, &opts()).await.unwrap();

        let sent = mock.last_request();
        assert!(sent.url.starts_with("https://proxy.test/chat/completions"));
        assert!(sent.headers.iter().any(|(k, v)| k == "x-proxy" && v == "1"));
    }

    #[test]
    fn auth_headers_report_the_bearer_scheme() {
        let headers = auth_headers(&Credential::api_key("sk-live"));
        assert_eq!(
            headers,
            vec![("authorization".to_owned(), "Bearer sk-live".to_owned())]
        );
    }

    #[test]
    fn debug_redacts_extra_header_values() {
        let mock = Arc::new(MockHttpClient::new());
        let provider = DeepSeekProvider::new(
            mock,
            Credential::api_key("sk-test"),
            "deepseek-chat",
        )
        .with_header("x-proxy-token", "super-secret");
        let rendered = format!("{provider:?}");
        assert!(!rendered.contains("super-secret"));
        assert!(rendered.contains("x-proxy-token"));
    }
}
