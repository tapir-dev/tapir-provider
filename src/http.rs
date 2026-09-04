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

/// The HTTP method for an [`HttpRequest`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Method {
    /// HTTP GET.
    Get,
    /// HTTP POST.
    Post,
}

impl Method {
    /// The uppercase method token as used on the wire.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
        }
    }
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
}

/// Forward through a shared handle so an `Arc<H>` (or `Arc<dyn HttpClient>`) is
/// itself a transport, letting a test hold onto the client to inspect requests.
#[async_trait]
impl<T: HttpClient + ?Sized> HttpClient for std::sync::Arc<T> {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, Error> {
        (**self).send(request).await
    }
}

#[cfg(feature = "reqwest")]
mod reqwest_client {
    use super::{
        Error, HttpClient, HttpRequest, HttpResponse, Method, async_trait,
    };
    use crate::error::ErrorKind;

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

    #[async_trait]
    impl HttpClient for ReqwestClient {
        async fn send(
            &self,
            request: HttpRequest,
        ) -> Result<HttpResponse, Error> {
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

            let response = builder.send().await.map_err(transport_error)?;
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
    }
}

#[cfg(feature = "reqwest")]
pub use reqwest_client::ReqwestClient;

#[cfg(feature = "test-utils")]
mod mock {
    use super::{Error, HttpClient, HttpRequest, HttpResponse, async_trait};
    use std::sync::Mutex;

    /// An in-crate [`HttpClient`] test double.
    ///
    /// It records every request it is sent and replies with a queue of canned
    /// outcomes, so a Provider can be driven end to end with no network.
    #[derive(Debug, Default)]
    pub struct MockHttpClient {
        responses: Mutex<Vec<Result<HttpResponse, Error>>>,
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
    }
}

#[cfg(feature = "test-utils")]
pub use mock::MockHttpClient;

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
