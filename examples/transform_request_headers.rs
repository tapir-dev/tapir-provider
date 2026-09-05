// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Shaping a request's HTTP headers per request, without touching Provider
//! construction.
//!
//! [`CompletionOptions`] carries two header seams that apply to one request:
//!
//!   - [`with_headers`](CompletionOptions::with_headers): static headers added
//!     after auth and any construction-time headers.
//!   - [`with_transform_headers`](CompletionOptions::with_transform_headers): a
//!     Header Transform, a sync final rewrite of the fully-assembled headers.
//!     It takes the headers by value and returns the headers to send, so it can
//!     add, drop, reorder, or dedup entries; it runs last and has the final say.
//!
//! Assembly order feeding the wire is: auth headers, construction-time headers,
//! per-request static headers, then the Header Transform. Headers are appended,
//! not merged: a per-request header rides after a same-named construction-time
//! one, so a server that takes the last value sees the per-request one win.
//! Collapsing duplicates is the transform's job.
//!
//! This example runs offline over an injected transport, so it is deterministic
//! and touches no network. It adds a static `x-tenant` header per request, then
//! a transform that injects a generated `x-request-id` and drops a debug header
//! the caller staged but does not want on the wire.
//!
//! Run with:
//!
//! ```text
//! cargo run --example transform_request_headers --features anthropic
//! ```

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tapir_provider::{
    AnthropicProvider, CompletionOptions, Context, Credential, Error,
    HttpClient, HttpRequest, HttpResponse, Message, Provider,
};

/// A transport that records the last request it saw and answers every call with
/// a canned Anthropic message, so the example can inspect what reached the wire
/// without a network.
#[derive(Clone, Default)]
struct RecordingTransport {
    last: Arc<Mutex<Option<HttpRequest>>>,
}

impl RecordingTransport {
    /// The header names and values on the last recorded request, in send order.
    fn last_headers(&self) -> Vec<(String, String)> {
        self.last
            .lock()
            .unwrap()
            .as_ref()
            .map(|request| request.headers.clone())
            .unwrap_or_default()
    }
}

#[async_trait]
impl HttpClient for RecordingTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, Error> {
        *self.last.lock().unwrap() = Some(request);
        let body = br#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-3-5-sonnet","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}"#;
        Ok(HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: body.to_vec(),
        })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    let transport = RecordingTransport::default();
    let provider = AnthropicProvider::new(
        transport.clone(),
        Credential::api_key("sk-example"),
        "claude-3-5-sonnet-20241022",
    );

    // Per request: a static tenant header, plus a transform that injects a
    // request id and strips a header the caller staged only for local debugging.
    let request_id = "req-42";
    let opts = CompletionOptions::default()
        .with_headers(vec![
            ("x-tenant".to_owned(), "acme".to_owned()),
            ("x-debug-note".to_owned(), "staged locally".to_owned()),
        ])
        .with_transform_headers(move |mut headers| {
            headers.retain(|(name, _)| name != "x-debug-note");
            headers.push(("x-request-id".to_owned(), request_id.to_owned()));
            headers
        });

    provider
        .complete(&Context::new(vec![Message::user("hello")]), &opts)
        .await?;

    println!("headers on the wire (in send order):");
    for (name, value) in transport.last_headers() {
        // Redact the auth secret; show every other header end to end.
        let shown = if name == "x-api-key" {
            "<redacted>"
        } else {
            &value
        };
        println!("  {name}: {shown}");
    }

    let headers = transport.last_headers();
    let has = |key: &str| headers.iter().any(|(name, _)| name == key);
    println!();
    println!("static per-request header present? {}", has("x-tenant"));
    println!(
        "transform-injected request id present? {}",
        has("x-request-id")
    );
    println!("staged debug header dropped? {}", !has("x-debug-note"));

    Ok(())
}
