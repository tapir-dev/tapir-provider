// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Refreshing a Provider's Model list from the live wire.
//!
//! Most reads over the Catalog are synchronous and offline (see
//! `querying_models`). Some Providers, though, publish their Model list at
//! runtime: a `/v1/models` endpoint answers with ids the compiled-in baseline
//! never carried. Discovering them is an explicit async verb —
//! [`ModelRegistry::refresh`] — and it is best-effort: a Provider is silently
//! skipped when it has no resolved Credential, its wire API exposes no `/models`
//! endpoint (`Api::Custom`), or the request fails. Refresh does not fail for any
//! of those; only a cache-write error surfaces.
//!
//! This example drives the whole arc through an injected fake transport, so it
//! runs offline and deterministically:
//!
//!   1. Load the baseline Catalog with a Token Store that credentials OpenAI —
//!      and not Anthropic — pointed at a scratch `models_path` for the on-disk
//!      cache.
//!   2. Refresh every Provider concurrently. OpenAI fetches and gains a Model the
//!      baseline never carried; Anthropic, uncredentialed, is skipped.
//!   3. Read the now-fresh list synchronously, then prove it persisted: a second
//!      `load` from the same path — no transport, no network — already carries
//!      the discovered Model.
//!
//! Run with:
//!
//! ```text
//! cargo run --example dynamic_providers --features models-fetch,openai,anthropic
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tapir_provider::{
    Credential, Error, HttpClient, HttpRequest, HttpResponse,
    InMemoryTokenStore, ModelRegistry, TokenStore,
};

/// A stand-in transport that answers a `/v1/models` GET with a canned listing
/// and 404s everything else.
///
/// The listing carries one id the baseline Catalog does not have
/// (`gpt-dynamic-preview`) alongside one it already does (`gpt-4o`): a refresh
/// only ever *adds* a new id, so the known one is left untouched and the novel
/// one is adopted.
struct FakeModelsEndpoint;

#[async_trait]
impl HttpClient for FakeModelsEndpoint {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, Error> {
        if request.url.ends_with("/v1/models") {
            let body =
                br#"{"data":[{"id":"gpt-4o"},{"id":"gpt-dynamic-preview"}]}"#;
            Ok(HttpResponse {
                status: 200,
                headers: Vec::new(),
                body: body.to_vec(),
            })
        } else {
            Ok(HttpResponse {
                status: 404,
                headers: Vec::new(),
                body: Vec::new(),
            })
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Error> {
    // A scratch directory for the on-disk cache. The fetched list is written
    // beside `models_path`, as `<stem>.fetched.json`.
    let dir = std::env::temp_dir()
        .join(format!("tapir-dynamic-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let models_path: PathBuf = dir.join("models.toml");

    // Credential OpenAI but not Anthropic: only a Provider with a resolved key is
    // fetched, so Anthropic is the skipped, best-effort no-op.
    let store = Arc::new(InMemoryTokenStore::new());
    store.set("openai", Credential::api_key("sk-example"))?;

    let mut registry =
        ModelRegistry::load(Some(store.clone()), Some(models_path.clone()))?;

    // Before the first refresh the Catalog holds only the compiled-in baseline.
    println!("baseline models: {}", registry.models().len());
    println!(
        "  gpt-dynamic-preview present? {}",
        registry.find("openai", "gpt-dynamic-preview").is_some()
    );

    // The async verb: discover every Provider's live list concurrently,
    // best-effort. OpenAI is fetched; Anthropic, uncredentialed, is skipped.
    registry.refresh(&FakeModelsEndpoint).await?;

    // OpenAI gained the novel Model; the skipped Provider contributed nothing.
    println!("after refresh: {}", registry.models().len());
    match registry.find("openai", "gpt-dynamic-preview") {
        Some(entry) => println!(
            "  discovered {} on provider {}",
            entry.model.id,
            entry.model.provider.as_str()
        ),
        None => println!("  gpt-dynamic-preview was not discovered"),
    }

    // The fresh list persisted next to the models path.
    let cache = models_path.with_extension("fetched.json");
    println!("cache written: {} ({})", cache.display(), cache.exists());

    // A second load from the same path — no transport, no network — already
    // carries the discovered Model: the last-known list, read back off disk.
    let reloaded = ModelRegistry::load(None, Some(models_path.clone()))?;
    println!(
        "reloaded offline, gpt-dynamic-preview present? {}",
        reloaded.find("openai", "gpt-dynamic-preview").is_some()
    );

    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}
