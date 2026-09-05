// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Provider Config reaching a [`ResolvedAuth`] — without making a request.
//!
//! A stored API-key [`Credential`] can carry a Provider Config: provider-scoped,
//! non-secret values (a gateway's account and gateway ids, say) folded in beside
//! the key. When the SDK resolves how a Provider would authenticate, that config
//! rides along on the [`ResolvedAuth`] beside the [`AuthSource`] and headers, so a
//! Provider has it request-ready and a caller can inspect it without spending a
//! request.
//!
//! This example runs offline over a transport that panics if contacted, proving
//! the whole path makes no network call. It shows, in order:
//!
//!   1. Build an API-key Credential with config and store it under `openai`.
//!   2. Provider-scoped `get_auth`: the config arrives beside `source` and the
//!      auth headers.
//!   3. Model-scoped `get_auth_for`: the same config, since it comes from the
//!      Credential, not the Model.
//!   4. `Debug` renders the non-secret config while the key stays redacted.
//!
//! Run with:
//!
//! ```text
//! cargo run --example credential_store --features models,openai
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use tapir_provider::{
    Credential, Error, HttpClient, HttpRequest, HttpResponse,
    InMemoryTokenStore, ModelRegistry, TokenStore,
};

/// A transport that panics if contacted: auth inspection resolves without a
/// request, so a live call here would be a bug.
struct OfflineTransport;

#[async_trait]
impl HttpClient for OfflineTransport {
    async fn send(&self, _request: HttpRequest) -> Result<HttpResponse, Error> {
        panic!("auth inspection must not make a request");
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    // 1. Build an API-key Credential with a Provider Config and store it. The
    //    store is retained by the registry, so inspection re-reads it on demand.
    let credential = Credential::api_key("sk-stored").with_config([
        ("CLOUDFLARE_ACCOUNT_ID".to_owned(), "acct-123".to_owned()),
        ("CLOUDFLARE_GATEWAY_ID".to_owned(), "gw-prod".to_owned()),
    ]);
    let store = Arc::new(InMemoryTokenStore::new());
    store.set("openai", credential)?;
    let registry = ModelRegistry::load(Some(store.clone()), None)?;

    let transport = OfflineTransport;

    // 2. Provider-scoped inspection: the config arrives on the ResolvedAuth.
    let auth = registry
        .get_auth("openai", &transport)
        .await?
        .expect("openai is configured");
    println!("openai configured via {}", auth.source);
    println!("  api key present? {}", auth.api_key.is_some());
    println!("  provider config:");
    for (name, value) in &auth.config {
        println!("    {name} = {value}");
    }

    // 3. Model-scoped inspection carries the same config: it comes from the
    //    Credential, not the Model.
    if let Some(entry) = registry.find("openai", "gpt-4o-mini") {
        let entry = entry.clone();
        let auth = registry
            .get_auth_for(&entry, &transport)
            .await?
            .expect("openai is configured");
        println!(
            "gpt-4o-mini config entries: {} (base url {})",
            auth.config.len(),
            auth.base_url.as_deref().unwrap_or("<none>")
        );
    }

    // 4. Debug renders the non-secret config; the key and header values stay
    //    redacted.
    println!("{auth:?}");

    Ok(())
}
