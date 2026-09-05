// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The fetched layer of the Catalog: a live per-Provider `GET /models` discovery
//! over the [`HttpClient`] seam, cached on disk.
//!
//! This is the middle-precedence layer (user > fetched > baseline). A
//! [`refresh`](super::ModelRegistry::refresh) asks each compiled-in Provider for
//! its current Model-id list, merges those ids over the baseline by id — adding a
//! Model the baseline does not carry, never overriding one it does — and persists
//! the result to an on-disk JSON cache tagged [`CACHE_TAG`]. A subsequent
//! [`load`](super::ModelRegistry::load) reads that cache back as the fetched
//! layer, so freshness survives a restart with no network.
//!
//! The fetch is best-effort: a Provider whose wire [`Api`] has no known `/models`
//! endpoint, whose Credential is unresolved, or whose request fails simply does
//! not update, leaving its last-known ids in place. The cache is derived, so it
//! is disposable — a missing, malformed, mistagged, or over-sized cache is
//! skipped on load rather than failing it.
//!
//! # Cache path
//!
//! The cache lives beside the user-override file: given
//! `.../tapir/models.toml`, the cache is `.../tapir/models.fetched.json`. Both
//! layers therefore travel with the one `models_path` a caller hands
//! [`load`](super::ModelRegistry::load).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, ErrorKind};
use crate::http::{HttpClient, HttpRequest, Method};
use crate::model::{Api, InputType, Model, ModelCost, ModelEntry, ModelId};

/// The tag every fetched cache file carries, so a foreign or stale-schema file is
/// recognized and skipped rather than parsed as ours.
const CACHE_TAG: &str = "tapir.models.fetched.v1";

/// The Anthropic API version header the `/models` request pins, mirroring the
/// value the Anthropic Provider sends on every request.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// The largest a cache file may be before it is refused, in bytes. Guards a load
/// against reading an implausibly large (corrupt or hostile) file into memory.
const MAX_CACHE_BYTES: u64 = 1 << 20;

/// The most Providers a cache may describe. Far above the compiled-in Provider
/// count, so it only ever trips on a corrupt or hand-forged file.
const MAX_PROVIDERS: usize = 64;

/// The most Model ids a cache may hold for one Provider. Bounds both the stored
/// file and the entries a load synthesizes from it.
const MAX_MODELS_PER_PROVIDER: usize = 1024;

/// The on-disk fetched cache: a tag plus each Provider's discovered Model ids.
///
/// Unknown fields are rejected so a file that has drifted from this shape is a
/// parse error (and thus skipped), never silently half-read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FetchedCache {
    /// The schema tag; must equal [`CACHE_TAG`].
    tag: String,
    /// Provider canonical id → its discovered Model ids.
    providers: BTreeMap<String, Vec<String>>,
}

impl FetchedCache {
    /// An empty cache carrying the current [`CACHE_TAG`].
    pub(super) fn empty() -> Self {
        Self {
            tag: CACHE_TAG.to_owned(),
            providers: BTreeMap::new(),
        }
    }
}

/// The cache-file path derived from the user-override `models_path`: the same
/// directory, with the file renamed to `<stem>.fetched.json`.
pub(super) fn cache_path(models_path: &Path) -> PathBuf {
    let stem = models_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("models");
    models_path.with_file_name(format!("{stem}.fetched.json"))
}

/// Read and validate the cache at `path`, or `None` when there is nothing usable.
///
/// A missing, over-sized, unreadable, malformed, mistagged, or over-bounds file
/// all yield `None`: the fetched layer is derived and disposable, so a bad cache
/// is skipped, never a load failure.
pub(super) fn read_cache(path: &Path) -> Option<FetchedCache> {
    let metadata = std::fs::metadata(path).ok()?;
    if metadata.len() > MAX_CACHE_BYTES {
        return None;
    }
    let raw = std::fs::read_to_string(path).ok()?;
    parse_cache(&raw).ok()
}

/// Parse and validate a cache document: correct tag, within size bounds, no
/// unknown fields.
///
/// # Errors
///
/// [`Decode`](ErrorKind::Decode) for malformed JSON, an unknown field, a wrong
/// tag, or a document over the [`MAX_PROVIDERS`]/[`MAX_MODELS_PER_PROVIDER`]
/// bounds.
fn parse_cache(raw: &str) -> Result<FetchedCache, Error> {
    let cache: FetchedCache =
        serde_json::from_str(raw).map_err(Error::decode)?;
    if cache.tag != CACHE_TAG {
        return Err(Error::new(
            ErrorKind::Decode,
            format!(
                "fetched model cache has tag {:?}, expected {CACHE_TAG:?}",
                cache.tag
            ),
        ));
    }
    if cache.providers.len() > MAX_PROVIDERS {
        return Err(Error::new(
            ErrorKind::Decode,
            format!(
                "fetched model cache lists {} providers, over the {MAX_PROVIDERS} limit",
                cache.providers.len()
            ),
        ));
    }
    for (provider, ids) in &cache.providers {
        if ids.len() > MAX_MODELS_PER_PROVIDER {
            return Err(Error::new(
                ErrorKind::Decode,
                format!(
                    "fetched model cache lists {} models for provider {provider:?}, over the {MAX_MODELS_PER_PROVIDER} limit",
                    ids.len()
                ),
            ));
        }
    }
    Ok(cache)
}

/// Serialize `cache` and overwrite `path`, creating any missing parent
/// directories.
///
/// # Errors
///
/// [`Other`](ErrorKind::Other) for a serialization or filesystem failure, or when
/// the serialized cache would exceed [`MAX_CACHE_BYTES`].
pub(super) fn write(path: &Path, cache: &FetchedCache) -> Result<(), Error> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|err| io_error(parent, "create directory for", err))?;
    }
    let json = serde_json::to_string_pretty(cache).map_err(Error::serialize)?;
    if json.len() as u64 > MAX_CACHE_BYTES {
        return Err(Error::new(
            ErrorKind::Other,
            format!(
                "fetched model cache is {} bytes, over the {MAX_CACHE_BYTES}-byte limit",
                json.len()
            ),
        ));
    }
    std::fs::write(path, json).map_err(|err| io_error(path, "write", err))
}

/// Wrap a cache filesystem error with the path and the failed operation.
fn io_error(path: &Path, op: &str, err: std::io::Error) -> Error {
    Error::new(
        ErrorKind::Other,
        format!(
            "fetched model cache {op} failed for {}: {err}",
            path.display()
        ),
    )
    .with_source(err)
}

/// How to reach one Provider's `/models` endpoint: its base URL and wire API.
struct Routing {
    /// The base URL the Provider is served from.
    base_url: String,
    /// The wire protocol the Provider speaks, which fixes the auth header.
    api: Api,
}

/// Fetch each Provider present in `entries` and record its Model ids into
/// `cache`, updating only Providers that answer successfully.
///
/// Best-effort: a Provider with no resolved Credential, no known `/models`
/// endpoint, or a failing request is left as it was in `cache`, so a transient
/// failure keeps that Provider's last-known ids.
pub(super) async fn refresh_into<H: HttpClient>(
    cache: &mut FetchedCache,
    entries: &[ModelEntry],
    transport: &H,
) {
    if cache.providers.len() >= MAX_PROVIDERS {
        return;
    }
    for (provider, routing, key) in providers(entries) {
        if let Some(ids) =
            fetch_provider(&routing, key.as_deref(), transport).await
        {
            cache.providers.insert(provider, clamp_ids(ids));
        }
    }
}

/// The distinct Providers in `entries`, each with its routing and a resolved
/// Credential (the first non-empty key seen for that Provider), in id order.
fn providers(entries: &[ModelEntry]) -> Vec<(String, Routing, Option<String>)> {
    let mut out: Vec<(String, Routing, Option<String>)> = Vec::new();
    for entry in entries {
        let id = entry.model.provider.as_str();
        if let Some(slot) = out.iter_mut().find(|(p, ..)| p == id) {
            if slot.2.is_none() {
                slot.2 = entry.api_key.clone();
            }
            continue;
        }
        out.push((
            id.to_owned(),
            Routing {
                base_url: entry.model.base_url.clone(),
                api: entry.model.api.clone(),
            },
            entry.api_key.clone(),
        ));
    }
    out
}

/// Ask one Provider for its Model ids, or `None` on any best-effort miss.
///
/// Returns `None` when the Provider has no resolved Credential, its wire API has
/// no known `/models` endpoint, the request fails, the status is non-2xx, or the
/// body does not parse.
async fn fetch_provider<H: HttpClient>(
    routing: &Routing,
    key: Option<&str>,
    transport: &H,
) -> Option<Vec<String>> {
    let request = models_request(routing, key?)?;
    let response = transport.send(request).await.ok()?;
    if !response.is_success() {
        return None;
    }
    let parsed: ModelsResponse = serde_json::from_slice(&response.body).ok()?;
    Some(parsed.data.into_iter().map(|row| row.id).collect())
}

/// Build the `GET /v1/models` request for a Provider, authenticated for its wire
/// API, or `None` when the API has no known endpoint.
fn models_request(routing: &Routing, key: &str) -> Option<HttpRequest> {
    let url = format!("{}/v1/models", routing.base_url.trim_end_matches('/'));
    let request = HttpRequest::new(Method::Get, url);
    match routing.api {
        Api::OpenAICompletions | Api::OpenAIResponses => {
            Some(request.header("authorization", format!("Bearer {key}")))
        }
        Api::AnthropicMessages => Some(
            request
                .header("x-api-key", key)
                .header("anthropic-version", ANTHROPIC_VERSION),
        ),
        // An API this crate has no adapter for has no known endpoint: do not update.
        Api::Custom(_) => None,
    }
}

/// Normalize a fetched id list: drop empties, dedupe, sort, and cap the length.
fn clamp_ids(mut ids: Vec<String>) -> Vec<String> {
    ids.retain(|id| !id.is_empty());
    ids.sort();
    ids.dedup();
    ids.truncate(MAX_MODELS_PER_PROVIDER);
    ids
}

/// The shape of a `/models` response: a `data` array of rows, each with an `id`.
/// Extra fields on the envelope and the rows are ignored.
#[derive(Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelRow>,
}

/// One row of a `/models` response; only the id is read.
#[derive(Deserialize)]
struct ModelRow {
    id: String,
}

/// Merge `cache` over `entries`: for each Provider, add a fetched Model id the
/// entries do not already carry, deriving its routing from an existing entry of
/// that Provider.
///
/// Fetched ids only ever *add* Models — an id already present (from the baseline
/// or the user layer) is left untouched — so the merge holds the
/// user > fetched > baseline precedence whatever order the layers apply in. A
/// Provider with no entry to derive routing from is skipped.
pub(super) fn apply(entries: &mut Vec<ModelEntry>, cache: &FetchedCache) {
    for (provider, ids) in &cache.providers {
        let Some(template) = entries
            .iter()
            .find(|e| e.model.provider.as_str().eq_ignore_ascii_case(provider))
            .map(|e| e.model.clone())
        else {
            continue;
        };
        let key = provider_api_key(entries, provider);
        for id in ids {
            if id.is_empty() || present(entries, provider, id) {
                continue;
            }
            let Ok(model_id) = ModelId::new(id.clone()) else {
                continue;
            };
            let model = Model {
                id: model_id,
                provider: template.provider.clone(),
                api: template.api.clone(),
                name: id.clone(),
                base_url: template.base_url.clone(),
                reasoning: false,
                input: vec![InputType::Text],
                cost: ModelCost::default(),
                context_window: 0,
                max_tokens: 0,
                headers: template.headers.clone(),
            };
            let mut entry = ModelEntry::new(model);
            entry.api_key = key.clone();
            entries.push(entry);
        }
    }
}

/// Whether `entries` already holds Model `id` under `provider`.
fn present(entries: &[ModelEntry], provider: &str, id: &str) -> bool {
    entries.iter().any(|e| {
        e.model.provider.as_str().eq_ignore_ascii_case(provider)
            && e.model.id.as_str() == id
    })
}

/// The resolved API key of `provider`, taken from the first entry that carries
/// one, so a synthesized entry shares its Provider's Credential.
fn provider_api_key(entries: &[ModelEntry], provider: &str) -> Option<String> {
    entries
        .iter()
        .find(|e| {
            e.model.provider.as_str().eq_ignore_ascii_case(provider)
                && e.api_key.is_some()
        })
        .and_then(|e| e.api_key.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(provider: &str, id: &str) -> Model {
        Model {
            id: ModelId::new(id).unwrap(),
            provider: crate::model::ProviderId::new(provider).unwrap(),
            api: Api::OpenAICompletions,
            name: "seed".to_owned(),
            base_url: "https://api.openai.com".to_owned(),
            reasoning: false,
            input: vec![InputType::Text],
            cost: ModelCost::default(),
            context_window: 8_000,
            max_tokens: 4_000,
            headers: Vec::new(),
        }
    }

    fn cache_of(provider: &str, ids: &[&str]) -> FetchedCache {
        let mut cache = FetchedCache::empty();
        cache.providers.insert(
            provider.to_owned(),
            ids.iter().map(|s| (*s).to_owned()).collect(),
        );
        cache
    }

    #[test]
    fn cache_path_sits_beside_the_user_file() {
        let path = cache_path(Path::new("/x/tapir/models.toml"));
        assert_eq!(path, Path::new("/x/tapir/models.fetched.json"));
    }

    #[test]
    fn parse_rejects_a_wrong_tag() {
        let raw = r#"{"tag":"someone.else.v1","providers":{}}"#;
        let err = parse_cache(raw).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Decode);
    }

    #[test]
    fn parse_rejects_an_unknown_field() {
        let raw =
            r#"{"tag":"tapir.models.fetched.v1","providers":{},"extra":1}"#;
        let err = parse_cache(raw).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Decode);
    }

    #[test]
    fn parse_rejects_too_many_models_for_a_provider() {
        let ids: Vec<String> = (0..=MAX_MODELS_PER_PROVIDER)
            .map(|n| format!("m{n}"))
            .collect();
        let mut cache = FetchedCache::empty();
        cache.providers.insert("openai".to_owned(), ids);
        let raw = serde_json::to_string(&cache).unwrap();
        let err = parse_cache(&raw).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Decode);
    }

    #[test]
    fn parse_round_trips_a_valid_cache() {
        let cache = cache_of("openai", &["gpt-4o", "gpt-new"]);
        let raw = serde_json::to_string(&cache).unwrap();
        assert_eq!(parse_cache(&raw).unwrap(), cache);
    }

    #[test]
    fn apply_adds_only_the_new_ids() {
        let mut entries = vec![ModelEntry::new(model("openai", "gpt-4o-mini"))];
        // gpt-4o-mini exists in the baseline; gpt-fetched does not.
        apply(
            &mut entries,
            &cache_of("openai", &["gpt-4o-mini", "gpt-fetched"]),
        );

        assert_eq!(entries.len(), 2);
        let added = entries
            .iter()
            .find(|e| e.model.id.as_str() == "gpt-fetched")
            .unwrap();
        // The new Model borrows its Provider's routing from the existing entry.
        assert_eq!(added.model.base_url, "https://api.openai.com");
        assert_eq!(added.model.api, Api::OpenAICompletions);
        assert_eq!(added.model.name, "gpt-fetched");
    }

    #[test]
    fn apply_stamps_the_providers_key_on_a_new_entry() {
        let mut seed = ModelEntry::new(model("openai", "gpt-4o-mini"));
        seed.api_key = Some("sk-openai".to_owned());
        let mut entries = vec![seed];
        apply(&mut entries, &cache_of("openai", &["gpt-fetched"]));
        let added = entries
            .iter()
            .find(|e| e.model.id.as_str() == "gpt-fetched")
            .unwrap();
        assert_eq!(added.api_key.as_deref(), Some("sk-openai"));
    }

    #[test]
    fn apply_skips_a_provider_with_no_entry_to_route_from() {
        let mut entries = vec![ModelEntry::new(model("openai", "gpt-4o-mini"))];
        // No entry for "anthropic", so its fetched ids cannot be routed.
        apply(&mut entries, &cache_of("anthropic", &["claude-x"]));
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn clamp_ids_dedupes_sorts_and_drops_empties() {
        let ids = clamp_ids(vec![
            "b".to_owned(),
            "a".to_owned(),
            "b".to_owned(),
            String::new(),
        ]);
        assert_eq!(ids, vec!["a".to_owned(), "b".to_owned()]);
    }

    #[test]
    fn write_then_read_round_trips_through_a_file() {
        let dir = std::env::temp_dir().join(format!(
            "tapir-fetch-rt-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let path = cache_path(&dir.join("models.toml"));
        let cache = cache_of("openai", &["gpt-4o", "gpt-new"]);
        write(&path, &cache).unwrap();
        assert_eq!(read_cache(&path), Some(cache));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_cache_skips_a_missing_file() {
        assert_eq!(
            read_cache(Path::new("/nonexistent/models.fetched.json")),
            None
        );
    }
}
