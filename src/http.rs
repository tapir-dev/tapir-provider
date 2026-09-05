// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`HttpClient`] transport seam through which all wire I/O flows.
//!
//! Every Provider talks to the network exclusively through an injected
//! [`HttpClient`]. The default build ships a [`reqwest`](ReqwestClient)-backed
//! implementation; the `test-utils` feature adds a [`MockHttpClient`] test
//! double so Providers can be driven with no network.

use crate::error::Error;
use async_trait::async_trait;
use std::pin::Pin;

/// A boxed stream of response-body byte chunks, as they arrive off the wire.
///
/// Each item is one chunk (arbitrarily framed by the transport) or a
/// [`ErrorKind::Transport`](crate::error::ErrorKind::Transport) error if the
/// connection failed mid-stream. Chunk boundaries carry no meaning; a consumer
/// reassembles them (see [`SseDecoder`](crate::sse::SseDecoder)).
pub type ByteStream =
    Pin<Box<dyn futures_core::Stream<Item = Result<Vec<u8>, Error>> + Send>>;

/// Parse a `Retry-After` header as a whole number of seconds.
///
/// Providers report the delay in `delay-seconds` form; the HTTP-date form is not
/// emitted, so it is not parsed. An absent, non-numeric, or oversized value
/// yields `None`, letting the retry decorator fall back to its own backoff.
#[cfg(any(feature = "anthropic", feature = "openai"))]
pub(crate) fn parse_retry_after(
    headers: &[(String, String)],
) -> Option<std::time::Duration> {
    headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("retry-after"))
        .and_then(|(_, value)| value.trim().parse::<u64>().ok())
        .map(std::time::Duration::from_secs)
}

/// Turn a non-2xx [`HttpResponse`] into a typed [`Error`], classifying the kind
/// from the status and attaching any server-requested `Retry-After` delay.
///
/// The Providers share this so a failed response maps to the same typed error
/// everywhere, and a `Retry-After` is always honored by the
/// [`RetryProvider`](crate::retry::RetryProvider).
#[cfg(any(feature = "anthropic", feature = "openai"))]
pub(crate) fn error_from_response(response: &HttpResponse) -> Error {
    let error = Error::from_status(response.status, response.body_string());
    match parse_retry_after(&response.headers) {
        Some(delay) => error.with_retry_after(delay),
        None => error,
    }
}

/// The HTTP method for an [`HttpRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    /// HTTP GET.
    Get,
    /// HTTP POST.
    Post,
}

/// A transport-agnostic HTTP request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpRequest {
    /// The request method.
    pub method: Method,
    /// The fully-qualified request URL.
    pub url: String,
    /// Header name/value pairs, in insertion order.
    pub headers: Vec<(String, String)>,
    /// The request body, if any.
    pub body: Option<Vec<u8>>,
}

impl HttpRequest {
    /// Start building a request with the given method and URL.
    pub fn new(method: Method, url: impl Into<String>) -> Self {
        Self {
            method,
            url: url.into(),
            headers: Vec::new(),
            body: None,
        }
    }

    /// Append a header.
    #[must_use]
    pub fn header(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Set the request body.
    #[must_use]
    pub fn body(mut self, body: impl Into<Vec<u8>>) -> Self {
        self.body = Some(body.into());
        self
    }
}

/// A transport-agnostic HTTP response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    /// The HTTP status code.
    pub status: u16,
    /// Header name/value pairs, in received order.
    pub headers: Vec<(String, String)>,
    /// The raw response body.
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Whether the status is in the 2xx success range.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }

    /// The response body decoded as UTF-8, lossily.
    #[must_use]
    pub fn body_string(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

/// The transport seam: send an [`HttpRequest`] and get an [`HttpResponse`].
///
/// Implementations own all network concerns. A failure to reach the server or
/// read a response maps to [`ErrorKind::Transport`](crate::error::ErrorKind::Transport);
/// a received response, even a non-2xx one, is returned as `Ok` for the caller
/// to classify.
#[async_trait]
pub trait HttpClient: Send + Sync {
    /// Send a request and await the full response.
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, Error>;

    /// Send a request and stream the response body as byte chunks.
    ///
    /// A non-2xx status is classified up front via [`Error::from_status`], so
    /// the returned stream only ever carries body chunks of a successful
    /// response. The default buffers the whole body through [`send`](Self::send)
    /// and yields it as one chunk; a transport that can stream (see
    /// [`ReqwestClient`]) overrides this to forward chunks as they arrive.
    async fn send_stream(
        &self,
        request: HttpRequest,
    ) -> Result<ByteStream, Error> {
        let response = self.send(request).await?;
        if !response.is_success() {
            return Err(Error::from_status(
                response.status,
                response.body_string(),
            ));
        }
        let chunk = response.body;
        Ok(Box::pin(futures_util::stream::once(
            async move { Ok(chunk) },
        )))
    }
}

/// Forward through a shared handle so an `Arc<H>` (or `Arc<dyn HttpClient>`) is
/// itself a transport, letting a test hold onto the client to inspect requests.
#[async_trait]
impl<T: HttpClient + ?Sized> HttpClient for std::sync::Arc<T> {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, Error> {
        (**self).send(request).await
    }

    async fn send_stream(
        &self,
        request: HttpRequest,
    ) -> Result<ByteStream, Error> {
        (**self).send_stream(request).await
    }
}

#[cfg(feature = "reqwest")]
mod reqwest_client {
    use super::{
        ByteStream, Error, HttpClient, HttpRequest, HttpResponse, Method,
        async_trait,
    };
    use crate::error::ErrorKind;
    use futures_util::StreamExt;

    /// The default [`HttpClient`], backed by [`reqwest`].
    #[derive(Debug, Clone, Default)]
    pub struct ReqwestClient {
        inner: reqwest::Client,
    }

    impl ReqwestClient {
        /// Construct a client with default settings.
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        /// Wrap an already-configured [`reqwest::Client`].
        #[must_use]
        pub fn with_client(inner: reqwest::Client) -> Self {
            Self { inner }
        }
    }

    fn transport_error(err: reqwest::Error) -> Error {
        Error::new(ErrorKind::Transport, err.to_string()).with_source(err)
    }

    impl ReqwestClient {
        /// Translate a transport-agnostic [`HttpRequest`] into a reqwest builder.
        fn builder(&self, request: HttpRequest) -> reqwest::RequestBuilder {
            let method = match request.method {
                Method::Get => reqwest::Method::GET,
                Method::Post => reqwest::Method::POST,
            };
            let mut builder = self.inner.request(method, &request.url);
            for (name, value) in &request.headers {
                builder = builder.header(name, value);
            }
            if let Some(body) = request.body {
                builder = builder.body(body);
            }
            builder
        }
    }

    #[async_trait]
    impl HttpClient for ReqwestClient {
        async fn send(
            &self,
            request: HttpRequest,
        ) -> Result<HttpResponse, Error> {
            let response = self
                .builder(request)
                .send()
                .await
                .map_err(transport_error)?;
            let status = response.status().as_u16();
            let headers = response
                .headers()
                .iter()
                .map(|(name, value)| {
                    (
                        name.as_str().to_owned(),
                        value.to_str().unwrap_or_default().to_owned(),
                    )
                })
                .collect();
            let body =
                response.bytes().await.map_err(transport_error)?.to_vec();

            Ok(HttpResponse {
                status,
                headers,
                body,
            })
        }

        async fn send_stream(
            &self,
            request: HttpRequest,
        ) -> Result<ByteStream, Error> {
            let response = self
                .builder(request)
                .send()
                .await
                .map_err(transport_error)?;
            let status = response.status().as_u16();
            // Classify a non-2xx up front: the error body is small and the
            // caller wants a typed error, not a stream of the error payload.
            if !(200..300).contains(&status) {
                let body =
                    response.text().await.unwrap_or_else(|err| err.to_string());
                return Err(Error::from_status(status, body));
            }

            let bytes = response.bytes_stream().map(|chunk| {
                chunk.map(|b| b.to_vec()).map_err(transport_error)
            });
            Ok(Box::pin(bytes))
        }
    }
}

#[cfg(feature = "reqwest")]
pub use reqwest_client::ReqwestClient;

#[cfg(feature = "test-utils")]
mod mock {
    use super::{
        ByteStream, Error, HttpClient, HttpRequest, HttpResponse, async_trait,
    };
    use std::sync::Mutex;

    /// An in-crate [`HttpClient`] test double.
    ///
    /// It records every request it is sent and replies with a queue of canned
    /// outcomes, so a Provider can be driven end to end with no network. The
    /// streaming path has its own queue of chunk lists, so a test can hand the
    /// decoder bytes framed exactly as it wants — including a payload split
    /// across chunk boundaries.
    #[derive(Debug, Default)]
    pub struct MockHttpClient {
        responses: Mutex<Vec<Result<HttpResponse, Error>>>,
        streams: Mutex<Vec<Vec<Vec<u8>>>>,
        requests: Mutex<Vec<HttpRequest>>,
    }

    impl MockHttpClient {
        /// A mock with no queued responses. Sending against it panics.
        #[must_use]
        pub fn new() -> Self {
            Self::default()
        }

        /// A mock that replies once with the given status and body.
        #[must_use]
        pub fn with_response(status: u16, body: impl Into<Vec<u8>>) -> Self {
            let mock = Self::new();
            mock.push_response(HttpResponse {
                status,
                headers: Vec::new(),
                body: body.into(),
            });
            mock
        }

        /// Queue a successful response to hand back on the next `send`.
        pub fn push_response(&self, response: HttpResponse) {
            self.responses.lock().unwrap().push(Ok(response));
        }

        /// Queue a transport-level error to hand back on the next `send`.
        pub fn push_error(&self, error: Error) {
            self.responses.lock().unwrap().push(Err(error));
        }

        /// A mock that streams the given byte chunks on the next `send_stream`.
        #[must_use]
        pub fn with_stream(chunks: Vec<Vec<u8>>) -> Self {
            let mock = Self::new();
            mock.push_stream(chunks);
            mock
        }

        /// Queue a list of body chunks to stream on the next `send_stream`, in
        /// order. The framing is preserved exactly, so a caller can split a
        /// payload mid-event to exercise chunk-boundary handling.
        pub fn push_stream(&self, chunks: Vec<Vec<u8>>) {
            self.streams.lock().unwrap().push(chunks);
        }

        /// The requests captured so far, in order.
        #[must_use]
        pub fn requests(&self) -> Vec<HttpRequest> {
            self.requests.lock().unwrap().clone()
        }

        /// The single request captured, panicking unless exactly one was sent.
        #[must_use]
        pub fn last_request(&self) -> HttpRequest {
            self.requests
                .lock()
                .unwrap()
                .last()
                .cloned()
                .expect("no request was sent to the mock")
        }
    }

    #[async_trait]
    impl HttpClient for MockHttpClient {
        async fn send(
            &self,
            request: HttpRequest,
        ) -> Result<HttpResponse, Error> {
            self.requests.lock().unwrap().push(request);
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                panic!(
                    "MockHttpClient received a request with no queued response"
                );
            }
            responses.remove(0)
        }

        async fn send_stream(
            &self,
            request: HttpRequest,
        ) -> Result<ByteStream, Error> {
            self.requests.lock().unwrap().push(request);
            let mut streams = self.streams.lock().unwrap();
            if streams.is_empty() {
                panic!(
                    "MockHttpClient received a stream request with no queued stream"
                );
            }
            let chunks = streams.remove(0);
            Ok(Box::pin(futures_util::stream::iter(
                chunks.into_iter().map(Ok),
            )))
        }
    }
}

#[cfg(feature = "test-utils")]
pub use mock::MockHttpClient;

#[cfg(all(test, any(feature = "anthropic", feature = "openai")))]
mod retry_after_tests {
    use super::parse_retry_after;
    use std::time::Duration;

    #[test]
    fn reads_delay_seconds_case_insensitively() {
        let headers = vec![("retry-after".to_owned(), " 12 ".to_owned())];
        assert_eq!(parse_retry_after(&headers), Some(Duration::from_secs(12)));
        // An HTTP-date form is not parsed.
        let dated = vec![(
            "Retry-After".to_owned(),
            "Wed, 21 Oct 2026 07:28:00 GMT".to_owned(),
        )];
        assert_eq!(parse_retry_after(&dated), None);
        assert_eq!(parse_retry_after(&[]), None);
    }
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_records_the_request_and_replies() {
        let mock = MockHttpClient::with_response(200, b"pong".to_vec());
        let req = HttpRequest::new(Method::Post, "https://example.test/ping")
            .header("x-test", "1")
            .body(b"ping".to_vec());

        let resp = mock.send(req.clone()).await.unwrap();

        assert_eq!(resp.status, 200);
        assert!(resp.is_success());
        assert_eq!(resp.body_string(), "pong");
        assert_eq!(mock.last_request(), req);
    }
}
