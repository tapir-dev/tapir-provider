// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The `WireAdapter` seam and the shared `CompletionPipeline` harness.
//!
//! Every Provider talks to its backend the same way: build a request (auth and
//! extra headers, an optional header augmentation, the Header Transform, and a
//! serialized body), send it, gate on a successful status, and either
//! double-decode the buffered body into an [`AssistantMessage`] or wrap the byte
//! stream in the shared `SseEventStream` driver. Only the parts that genuinely
//! vary by Provider — the endpoint path, the auth scheme, the request body, the
//! response mapping, and the `StreamNormalizer` — differ.
//!
//! `WireAdapter` captures exactly those varying parts, and
//! `CompletionPipeline` owns the invariant flow around them, implementing
//! [`Provider`] once for every adapter. A Provider becomes an adapter plus the
//! shared fields the pipeline holds (transport, Credential, Model, base URL, and
//! extra headers).

use crate::credential::Credential;
use crate::error::Error;
use crate::http::{HttpClient, HttpRequest, Method};
use crate::message::AssistantMessage;
use crate::provider::Provider;
use crate::request::{CompletionOptions, Context};
use crate::stream::{SseEventStream, StreamEvents, StreamNormalizer};
use async_trait::async_trait;
use serde::de::DeserializeOwned;
use std::fmt;

/// Whether a request opts into a streamed (SSE) response.
///
/// A named alternative to a bare `bool` at the call sites that build the wire
/// request, so `Streaming::On` reads for itself. It lives with the pipeline
/// rather than per Provider: the pipeline picks the signal for `complete`
/// (`Off`) and `complete_stream` (`On`) and hands it to the adapter's body
/// builder, so no adapter re-decides it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Streaming {
    /// Request an incremental SSE response.
    On,
    /// Request a single buffered response.
    Off,
}

impl Streaming {
    /// Whether streaming is requested, as the wire `stream` flag.
    pub(crate) const fn enabled(self) -> bool {
        matches!(self, Self::On)
    }
}

/// What a Provider must supply on the wire, factored out of the shared flow.
///
/// An adapter names its buffered response type and its `StreamNormalizer`, and
/// declares the handful of steps that genuinely vary by Provider: the endpoint
/// path, the auth headers a [`Credential`] produces, the serialized request
/// body, the mapping from a decoded response into an [`AssistantMessage`], and a
/// fresh normalizer for a streamed response. The
/// [`augment_headers`](Self::augment_headers) hook is defaulted to a no-op, for
/// the Providers that need to fold something onto the assembled headers before
/// the Header Transform runs (a beta flag, say).
///
/// It is monomorphized into `CompletionPipeline`, not held as a trait object,
/// so it carries associated types freely and need not be object-safe.
pub(crate) trait WireAdapter: Send + Sync {
    /// The Provider's buffered response body, decoded from the successful
    /// response before it is mapped into an [`AssistantMessage`].
    type Response: DeserializeOwned;

    /// The `StreamNormalizer` that maps this Provider's SSE events into the
    /// neutral stream vocabulary.
    type Normalizer: StreamNormalizer + Unpin + Send + 'static;

    /// The endpoint path appended to the base URL, leading slash included
    /// (for example `/v1/messages`).
    fn endpoint(&self) -> &str;

    /// The auth headers a request authenticating with `credential` carries.
    ///
    /// The single source of truth for the Provider's auth scheme, shared with
    /// the Model Registry's auth inspection so the wire matches what it reports.
    fn auth_headers(&self, credential: &Credential) -> Vec<(String, String)>;

    /// The serialized request body for this call.
    ///
    /// `credential` is the one this request authenticates with — the per-request
    /// override when set, else the Provider's own — so a Provider whose body
    /// shape forks on the auth lane can read it here.
    fn request_body(
        &self,
        model: &str,
        ctx: &Context,
        opts: &CompletionOptions,
        streaming: Streaming,
        credential: &Credential,
    ) -> Result<Vec<u8>, Error>;

    /// Map a decoded response, plus its raw JSON, into an [`AssistantMessage`].
    ///
    /// `raw` is the same body decoded a second time as an untyped value, so the
    /// settled message can retain the Provider's response verbatim.
    fn map_message(
        &self,
        response: Self::Response,
        raw: serde_json::Value,
    ) -> AssistantMessage;

    /// A fresh `StreamNormalizer` for one streamed response.
    fn normalizer(&self) -> Self::Normalizer;

    /// Fold Provider-specific entries onto the assembled base headers before the
    /// Header Transform runs. Defaulted to a no-op; a Provider overrides it when
    /// a request needs a header derived from the options (a cache beta flag, for
    /// instance).
    fn augment_headers(
        &self,
        headers: &mut Vec<(String, String)>,
        opts: &CompletionOptions,
    ) {
        let _ = (headers, opts);
    }
}

/// The shared completion flow around a `WireAdapter`.
///
/// It holds the fields every Provider carries — the injected transport `H`, the
/// Credential, the Model, the base URL, and the caller-supplied extra headers —
/// plus the adapter `A` that fills in the wire specifics. Implementing
/// [`Provider`] once here gives every adapter both a buffered `complete` and a
/// streamed `complete_stream` without repeating the send, the success gate, the
/// error classification, the double-decode, or the stream-driver wrap.
#[derive(Clone)]
pub(crate) struct CompletionPipeline<H, A> {
    /// The transport all wire I/O flows through.
    http: H,
    /// The adapter supplying the Provider's wire specifics.
    adapter: A,
    /// The Credential requests authenticate with, absent a per-request override.
    credential: Credential,
    /// The addressable Model, fixed when the pipeline is built.
    model: String,
    /// The base URL the endpoint path is appended to.
    base_url: String,
    /// Caller-supplied headers appended to every request, after the adapter's
    /// own auth headers.
    extra_headers: Vec<(String, String)>,
}

/// Redacts header *values*, keeping names visible: a caller-supplied header may
/// carry a secret, so its value never reaches Debug output, matching the crate's
/// [`Credential`] redaction discipline.
pub(crate) fn redacted_headers(
    headers: &[(String, String)],
) -> Vec<(&str, &str)> {
    headers
        .iter()
        .map(|(name, _)| (name.as_str(), "<redacted>"))
        .collect()
}

impl<H: fmt::Debug, A: fmt::Debug> fmt::Debug for CompletionPipeline<H, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompletionPipeline")
            .field("http", &self.http)
            .field("adapter", &self.adapter)
            .field("credential", &self.credential)
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("extra_headers", &redacted_headers(&self.extra_headers))
            .finish()
    }
}

impl<H, A> CompletionPipeline<H, A> {
    /// Build a pipeline over `adapter`, authenticating with `credential` and
    /// addressing `model` at `base_url`, with no extra headers.
    pub(crate) fn new(
        http: H,
        adapter: A,
        credential: Credential,
        model: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            http,
            adapter,
            credential,
            model: model.into(),
            base_url: base_url.into(),
            extra_headers: Vec::new(),
        }
    }

    /// Override the base URL the endpoint path is appended to.
    #[must_use]
    pub(crate) fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Append a single extra header sent with every request, after the adapter's
    /// own auth headers.
    #[must_use]
    pub(crate) fn with_header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    /// Append extra headers sent with every request, after the adapter's own
    /// auth headers.
    #[must_use]
    pub(crate) fn with_headers(
        mut self,
        headers: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        self.extra_headers.extend(headers);
        self
    }
}

impl<H, A: WireAdapter> CompletionPipeline<H, A> {
    /// Assemble the wire request shared by both paths.
    ///
    /// An explicit per-request key overrides the constructed Credential for this
    /// call. The body is built for the chosen `streaming` signal; the headers are
    /// the adapter's auth headers, then the extra headers, then the adapter's
    /// augmentation hook, then the per-request static headers and the Header
    /// Transform with the final say.
    fn build_request(
        &self,
        ctx: &Context,
        opts: &CompletionOptions,
        streaming: Streaming,
    ) -> Result<HttpRequest, Error> {
        let override_credential =
            opts.api_key.as_deref().map(Credential::api_key);
        let credential =
            override_credential.as_ref().unwrap_or(&self.credential);

        let body = self.adapter.request_body(
            &self.model,
            ctx,
            opts,
            streaming,
            credential,
        )?;

        let url = format!(
            "{}{}",
            self.base_url.trim_end_matches('/'),
            self.adapter.endpoint()
        );
        let request = HttpRequest::new(Method::Post, url)
            .header("content-type", "application/json");

        let mut base: Vec<(String, String)> = self
            .adapter
            .auth_headers(credential)
            .into_iter()
            .chain(self.extra_headers.iter().cloned())
            .collect();
        self.adapter.augment_headers(&mut base, opts);
        let request = opts
            .finalize_headers(base)
            .into_iter()
            .fold(request, |req, (name, value)| req.header(name, value));
        Ok(request.body(body))
    }
}

#[async_trait]
impl<H, A> Provider for CompletionPipeline<H, A>
where
    H: HttpClient,
    A: WireAdapter,
{
    async fn complete(
        &self,
        ctx: &Context,
        opts: &CompletionOptions,
    ) -> Result<AssistantMessage, Error> {
        let request = self.build_request(ctx, opts, Streaming::Off)?;
        let response = self.http.send(request).await?;

        if !response.is_success() {
            // The buffered path keeps the server-requested Retry-After delay.
            return Err(crate::http::error_from_response(&response));
        }

        // Decode once into the typed response, once into the raw value the
        // settled message retains verbatim.
        let typed: A::Response =
            serde_json::from_slice(&response.body).map_err(Error::decode)?;
        let raw: serde_json::Value =
            serde_json::from_slice(&response.body).map_err(Error::decode)?;

        Ok(self.adapter.map_message(typed, raw))
    }

    async fn complete_stream(
        &self,
        ctx: &Context,
        opts: &CompletionOptions,
    ) -> Result<StreamEvents, Error> {
        let request = self.build_request(ctx, opts, Streaming::On)?;
        let bytes = self.http.send_stream(request).await?;
        Ok(Box::pin(SseEventStream::new(
            bytes,
            self.adapter.normalizer(),
        )))
    }
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use crate::http::{HttpResponse, MockHttpClient};
    use crate::message::{ContentPart, Message};
    use crate::response::{FinishReason, Usage};
    use crate::sse::SseEvent;
    use crate::stream::StreamEvent;
    use futures_util::StreamExt;
    use serde::Deserialize;
    use std::sync::Arc;
    use std::time::Duration;

    /// The fake Provider's buffered response body.
    #[derive(Debug, Deserialize)]
    struct FakeResponse {
        text: String,
    }

    /// A scripted `StreamNormalizer` mapping each decoded SSE event to a single
    /// text delta carrying its `data`, so a test reads the driven events straight
    /// off the pipeline's stream wrap.
    #[derive(Debug, Default)]
    struct FakeNormalizer;

    impl StreamNormalizer for FakeNormalizer {
        fn normalize(&mut self, event: &SseEvent) -> Vec<StreamEvent> {
            vec![StreamEvent::TextDelta {
                index: 0,
                text: event.data.clone(),
            }]
        }
    }

    /// A minimal `WireAdapter` that drives both pipeline paths end to end: it
    /// pins an endpoint, an `x-fake-key` auth header, a body echoing the Model and
    /// the streaming flag, and a text-only response mapping.
    #[derive(Debug)]
    struct FakeAdapter;

    impl WireAdapter for FakeAdapter {
        type Response = FakeResponse;
        type Normalizer = FakeNormalizer;

        fn endpoint(&self) -> &str {
            "/v1/fake"
        }

        fn auth_headers(
            &self,
            credential: &Credential,
        ) -> Vec<(String, String)> {
            match credential {
                Credential::ApiKey { key, .. } => {
                    vec![("x-fake-key".to_owned(), key.clone())]
                }
                Credential::OAuth(_) => Vec::new(),
            }
        }

        fn request_body(
            &self,
            model: &str,
            _ctx: &Context,
            _opts: &CompletionOptions,
            streaming: Streaming,
            _credential: &Credential,
        ) -> Result<Vec<u8>, Error> {
            serde_json::to_vec(&serde_json::json!({
                "model": model,
                "stream": streaming.enabled(),
            }))
            .map_err(Error::serialize)
        }

        fn map_message(
            &self,
            response: FakeResponse,
            raw: serde_json::Value,
        ) -> AssistantMessage {
            AssistantMessage {
                content: vec![ContentPart::Text(response.text)],
                usage: Usage::default(),
                finish_reason: FinishReason::Stop,
                raw: Some(raw),
            }
        }

        fn normalizer(&self) -> FakeNormalizer {
            FakeNormalizer
        }
    }

    /// A pipeline over the fake adapter, backed by the given mock transport.
    fn pipeline(
        http: Arc<MockHttpClient>,
    ) -> CompletionPipeline<Arc<MockHttpClient>, FakeAdapter> {
        CompletionPipeline::new(
            http,
            FakeAdapter,
            Credential::api_key("sk-fake"),
            "fake-model",
            "https://fake.test",
        )
    }

    #[tokio::test]
    async fn complete_builds_sends_gates_double_decodes_and_maps() {
        let mock = Arc::new(MockHttpClient::with_response(
            200,
            br#"{"text":"hi from fake"}"#.to_vec(),
        ));
        let provider = pipeline(mock.clone());

        let ctx = Context::new(vec![Message::user("hello")]);
        let reply = provider
            .complete(&ctx, &CompletionOptions::default())
            .await
            .unwrap();

        // The double-decode maps the typed body and keeps the raw value.
        assert_eq!(reply.text_content(), "hi from fake");
        assert_eq!(reply.finish_reason, FinishReason::Stop);
        assert_eq!(reply.raw.as_ref().unwrap()["text"], "hi from fake");

        // The build assembled the endpoint, content-type, and auth header, and
        // the body echoes the Model with streaming off.
        let sent = mock.last_request();
        assert_eq!(sent.method, Method::Post);
        assert!(sent.url.ends_with("/v1/fake"));
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "content-type" && v == "application/json")
        );
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "x-fake-key" && v == "sk-fake")
        );
        let body: serde_json::Value =
            serde_json::from_slice(sent.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["model"], "fake-model");
        assert_eq!(body["stream"], false);
    }

    #[tokio::test]
    async fn complete_gate_classifies_a_failure_and_keeps_retry_after() {
        let mock = Arc::new(MockHttpClient::new());
        mock.push_response(HttpResponse {
            status: 429,
            headers: vec![("retry-after".to_owned(), "7".to_owned())],
            body: b"slow down".to_vec(),
        });
        let provider = pipeline(mock);

        let ctx = Context::new(vec![Message::user("hello")]);
        let error = provider
            .complete(&ctx, &CompletionOptions::default())
            .await
            .unwrap_err();

        // The buffered gate classifies the status and enriches with Retry-After.
        assert_eq!(error.retry_after(), Some(Duration::from_secs(7)));
    }

    #[tokio::test]
    async fn complete_stream_wraps_the_byte_stream_in_the_driver() {
        let mock = Arc::new(MockHttpClient::new());
        // The first event straddles two chunks, exercising the shared driver.
        mock.push_stream(vec![
            b"data: on".to_vec(),
            b"e\n\ndata: two\n\n".to_vec(),
        ]);
        let provider = pipeline(mock.clone());

        let ctx = Context::new(vec![Message::user("hello")]);
        let events: Vec<StreamEvent> = provider
            .complete_stream(&ctx, &CompletionOptions::default())
            .await
            .unwrap()
            .map(Result::unwrap)
            .collect()
            .await;

        assert_eq!(
            events,
            vec![
                StreamEvent::TextDelta {
                    index: 0,
                    text: "one".to_owned(),
                },
                StreamEvent::TextDelta {
                    index: 0,
                    text: "two".to_owned(),
                },
            ]
        );
        // The streamed body was built with the streaming flag on.
        let sent = mock.last_request();
        let body: serde_json::Value =
            serde_json::from_slice(sent.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["stream"], true);
    }
}
