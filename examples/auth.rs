// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Inspecting how a Provider resolves auth — without making a request.
//!
//! Every call the SDK makes resolves auth through the owning Provider and merges
//! it into the request by precedence: an explicit per-request key wins, else a
//! stored Credential, else the Provider's API-key environment variable. You can
//! ask what that resolution *would* produce without spending a request:
//! [`ModelRegistry::get_auth`] answers for a Provider, and
//! [`ModelRegistry::get_auth_for`] answers for one Model, layering that Model's
//! own headers and base URL on top.
//!
//! Each inspection returns a [`ResolvedAuth`]: the [`AuthSource`] tier that won,
//! the auth headers a request would carry, and an auth-derived API key. A
//! Provider with no Credential anywhere resolves to `None` — "not configured".
//!
//! This example runs offline over an injected transport, so it is deterministic
//! and touches no network. It shows, in order:
//!
//!   1. A stored key: `get_auth` reports the `stored credential` source and the
//!      auth headers, making no request.
//!   2. How the environment tier would report (in prose — the example does not
//!      mutate the process environment).
//!   3. `get_auth_for`: the same, plus the Model's headers and base URL.
//!   4. An unconfigured Provider resolving to "not configured".
//!   5. A per-request key override winning on the wire over the constructed
//!      Credential.
//!
//! Run with:
//!
//! ```text
//! cargo run --example auth --features models,openai
//! ```

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tapir_provider::{
    CompletionOptions, Context, Credential, Error, HttpClient, HttpRequest,
    HttpResponse, InMemoryTokenStore, Message, ModelRegistry, TokenStore,
};

/// A transport that records the last request it saw and answers every call with
/// a canned OpenAI chat completion, so the example can inspect what reached the
/// wire without a network.
#[derive(Clone, Default)]
struct RecordingTransport {
    last: Arc<Mutex<Option<HttpRequest>>>,
}

impl RecordingTransport {
    /// The `authorization` header value on the last recorded request, if any.
    fn last_authorization(&self) -> Option<String> {
        let guard = self.last.lock().unwrap();
        let request = guard.as_ref()?;
        request
            .headers
            .iter()
            .find(|(name, _)| name == "authorization")
            .map(|(_, value)| value.clone())
    }
}

#[async_trait]
impl HttpClient for RecordingTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, Error> {
        *self.last.lock().unwrap() = Some(request);
        let body = br#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
        Ok(HttpResponse {
            status: 200,
            headers: Vec::new(),
            body: body.to_vec(),
        })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    // Credential OpenAI through a stored key. The store is retained by the
    // registry, so auth inspection re-reads it on demand.
    let store = Arc::new(InMemoryTokenStore::new());
    store.set("openai", Credential::api_key("sk-stored"))?;
    let registry = ModelRegistry::load(Some(store.clone()), None)?;

    // The transport is never contacted by the inspections below — they resolve
    // auth without a request. It only carries the completion in step 5.
    let transport = RecordingTransport::default();

    // 1. Provider-scoped inspection: what would OpenAI send this turn?
    match registry.get_auth("openai", &transport).await? {
        Some(auth) => {
            println!("openai configured via {}", auth.source);
            let names: Vec<&str> =
                auth.headers.iter().map(|(name, _)| name.as_str()).collect();
            println!("  auth header names: {names:?}");
            println!("  api key present? {}", auth.api_key.is_some());
        }
        None => println!("openai not configured"),
    }

    // 2. The environment tier reports the variable name it read from. With no
    //    stored key, `get_auth` would resolve `OPENAI_API_KEY` and report its
    //    source as "OPENAI_API_KEY" rather than "stored credential".
    println!("(with no stored key, the source would read \"OPENAI_API_KEY\")");

    // 3. Model-scoped inspection layers the Model's headers and base URL.
    if let Some(entry) = registry.find("openai", "gpt-4o-mini") {
        let entry = entry.clone();
        if let Some(auth) = registry.get_auth_for(&entry, &transport).await? {
            println!(
                "gpt-4o-mini base url: {}",
                auth.base_url.as_deref().unwrap_or("<none>")
            );
            println!("  total headers (auth + model): {}", auth.headers.len());
        }
    }

    // 4. A Provider with no Credential anywhere resolves to "not configured".
    match registry.get_auth("anthropic", &transport).await? {
        Some(auth) => println!("anthropic configured via {}", auth.source),
        None => println!("anthropic not configured"),
    }

    // 5. A per-request key overrides the constructed Credential for that one
    //    call. Build a live Provider from the stored key, then complete with an
    //    explicit override and see which key reached the wire.
    let provider =
        registry.create_provider("openai", "gpt-4o-mini", transport.clone())?;
    let opts = CompletionOptions::default().with_api_key("sk-explicit");
    provider
        .complete(&Context::new(vec![Message::user("hello")]), &opts)
        .await?;
    println!(
        "on-the-wire authorization: {}",
        transport
            .last_authorization()
            .as_deref()
            .unwrap_or("<none>")
    );

    Ok(())
}
