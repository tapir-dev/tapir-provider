// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`ModelRegistry`]: the runtime holder of the Catalog.
//!
//! Where the [`Registry`] is a zero-sized, compile-time handle over Provider
//! identity, the Model Registry is a stateful, owned object. It
//! [`load`](ModelRegistry::load)s the layers of the Catalog — the compiled-in
//! baseline, behind `models-fetch` a persisted fetched layer merged over it (and
//! kept fresh by [`refresh`](ModelRegistry::refresh)), and behind
//! `models-user-config` a user-override TOML layer merged over both (precedence:
//! user > fetched > baseline) — resolves each entry's Credential through the same
//! Token Store / environment precedence every construction path uses, and turns a
//! [`ModelEntry`] into a live `Arc<dyn Provider>` via [`create_provider`]. The
//! common path is one call:
//! `ModelRegistry::load(auth, None)?.create_provider("openai", "gpt-4o-mini", http)?`.
//!
//! A [`ModelEntry`] carries only an API-key Credential ([`ModelEntry::api_key`]),
//! so `load` folds a resolved API key into the entry and an OAuth Credential is
//! left for a Provider's own construction path. An entry with no resolved key is
//! reported by [`models`](ModelRegistry::models) but excluded from
//! [`available_models`](ModelRegistry::available_models).

mod baseline;
#[cfg(feature = "models-fetch")]
mod fetch;
#[cfg(feature = "models-user-config")]
mod user_config;

#[cfg(feature = "models-user-config")]
pub use user_config::default_path as user_config_path;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::credential::Credential;
use crate::error::{Error, ErrorKind};
use crate::http::HttpClient;
use crate::model::ModelEntry;
use crate::provider::Provider;
use crate::registry::Registry;
use crate::token_store::{self, TokenStore};

/// The runtime holder of the Catalog.
///
/// It owns the merged, Credential-resolved [`ModelEntry`]s and answers lookups
/// over them. See the [module docs](self) for the layering model.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRegistry {
    entries: Vec<ModelEntry>,
    models_path: Option<PathBuf>,
}

impl ModelRegistry {
    /// Load the Catalog and resolve each entry's Credential.
    ///
    /// The compiled-in baseline layer is loaded first. When the `models-fetch`
    /// feature is on and a fetched cache sits beside `models_path`, its
    /// discovered Models are merged over the baseline next (adding Models the
    /// baseline lacks); a missing or unusable cache is simply skipped. Then, when
    /// the `models-user-config` feature is on and `models_path` names an existing
    /// file, its TOML user-override layer is merged over both (user value
    /// winning) before Credentials are resolved — so precedence is
    /// user > fetched > baseline. A `None` path, or a path with neither file,
    /// leaves the baseline untouched. Callers discover the conventional path with
    /// [`user_config_path`].
    ///
    /// Every entry without an API key already set then has one resolved by
    /// precedence: the `auth` Token Store under the entry's Provider id, else
    /// that Provider's default API-key environment variable. Credential
    /// resolution never fails the load — a broken store or an OAuth-only
    /// Credential simply leaves the entry without a key, so it is present in
    /// [`models`](Self::models) but not in
    /// [`available_models`](Self::available_models).
    ///
    /// # Errors
    ///
    /// Only the user-override layer can fail the load, and only when the
    /// `models-user-config` feature is on: an unreadable file, malformed TOML,
    /// an aliased or duplicate Provider key, or a new Model missing a required
    /// field. Without that feature, or with no user file, `load` never fails.
    pub fn load(
        auth: Option<&dyn TokenStore>,
        models_path: Option<PathBuf>,
    ) -> Result<Self, Error> {
        let mut entries = baseline::entries();
        #[cfg(feature = "models-fetch")]
        if let Some(path) = models_path.as_deref()
            && let Some(cache) = fetch::read_cache(&fetch::cache_path(path))
        {
            fetch::apply(&mut entries, &cache);
        }
        #[cfg(feature = "models-user-config")]
        if let Some(path) = models_path.as_deref()
            && let Some(raw) = read_user_config(path)?
        {
            user_config::apply(&mut entries, &raw)?;
        }
        for entry in &mut entries {
            if entry.api_key.is_some() {
                continue;
            }
            entry.api_key = resolve_api_key(entry, auth);
        }
        Ok(Self {
            entries,
            models_path,
        })
    }

    /// Every entry in the Catalog, whether or not it has a resolved Credential.
    #[must_use]
    pub fn models(&self) -> &[ModelEntry] {
        &self.entries
    }

    /// The entry a `provider` (canonical id or alias) and Model `id` select, or
    /// `None` if the Catalog holds no such Model.
    #[must_use]
    pub fn find(&self, provider: &str, id: &str) -> Option<&ModelEntry> {
        let canonical = Registry::resolve(provider)
            .map_or(provider, |info| info.id.as_str());
        self.entries.iter().find(|entry| {
            entry
                .model
                .provider
                .as_str()
                .eq_ignore_ascii_case(canonical)
                && entry.model.id.as_str() == id
        })
    }

    /// The first entry whose Model `id` matches, across all Providers, or `None`.
    #[must_use]
    pub fn find_by_id(&self, id: &str) -> Option<&ModelEntry> {
        self.entries
            .iter()
            .find(|entry| entry.model.id.as_str() == id)
    }

    /// The entries with a resolved Credential — the Models ready to call.
    #[must_use]
    pub fn available_models(&self) -> Vec<&ModelEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.api_key.is_some())
            .collect()
    }

    /// An owned clone of the entries with a resolved Credential.
    #[must_use]
    pub fn get_available(&self) -> Vec<ModelEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.api_key.is_some())
            .cloned()
            .collect()
    }

    /// The path the user-override and fetched layers are read from, if one was
    /// given to [`load`](Self::load).
    #[must_use]
    pub fn models_path(&self) -> Option<&Path> {
        self.models_path.as_deref()
    }

    /// Build a live Provider for the Model that `provider` and `model` select.
    ///
    /// A convenience over [`find`](Self::find) plus [`create_provider`]: the
    /// common one-call path. `provider` is a canonical id or an alias. A
    /// selection the Catalog does not hold is an
    /// [`InvalidRequest`](crate::ErrorKind::InvalidRequest) error; the mapping
    /// rules and the Credential requirement are [`create_provider`]'s.
    ///
    /// # Errors
    ///
    /// [`InvalidRequest`](crate::ErrorKind::InvalidRequest) when no such Model is
    /// in the Catalog; otherwise whatever [`create_provider`] reports.
    pub fn create_provider<H: HttpClient + 'static>(
        &self,
        provider: &str,
        model: &str,
        transport: H,
    ) -> Result<Arc<dyn Provider>, Error> {
        let entry = self
            .find(provider, model)
            .ok_or_else(|| unknown_model(provider, model))?;
        create_provider(entry, transport)
    }

    /// Refresh the fetched layer: discover each Provider's current Models over
    /// the network, merge them in, and persist the on-disk cache.
    ///
    /// For every Provider in the Catalog with a resolved Credential and a known
    /// `GET /models` endpoint, this asks over the injected `transport` (so the
    /// VCR and mock seams still apply) and folds the returned ids into the
    /// fetched layer, adding any Model the Catalog does not already carry. The
    /// merge only ever adds — a Model already present from the user or baseline
    /// layer is left untouched — so precedence stays user > fetched > baseline.
    ///
    /// The fetch is best-effort per Provider: one with no Credential, no known
    /// endpoint, or a failing request keeps its last-known ids rather than
    /// dropping them. When a `models_path` was given to [`load`](Self::load), the
    /// updated cache is written beside it (`models.toml` -> `models.fetched.json`,
    /// within fixed size bounds); with no path the fetched layer updates in
    /// memory only.
    ///
    /// # Errors
    ///
    /// [`Other`](crate::ErrorKind::Other) only when the cache cannot be written
    /// (a filesystem failure or an over-sized cache). A per-Provider fetch
    /// failure is never surfaced.
    #[cfg(feature = "models-fetch")]
    pub async fn refresh<H: HttpClient>(
        &mut self,
        transport: &H,
    ) -> Result<(), Error> {
        let cache_path = self.models_path.as_deref().map(fetch::cache_path);
        // Start from the persisted cache so a transient per-Provider failure
        // leaves that Provider's last-known ids in place.
        let mut cache = cache_path
            .as_deref()
            .and_then(fetch::read_cache)
            .unwrap_or_else(fetch::FetchedCache::empty);
        fetch::refresh_into(&mut cache, &self.entries, transport).await;
        if let Some(path) = cache_path.as_deref() {
            fetch::write(path, &cache)?;
        }
        fetch::apply(&mut self.entries, &cache);
        Ok(())
    }
}

/// Read the user-override file, or `None` when it does not exist.
///
/// A missing file is the common case — most callers have no user layer — so it
/// is not an error; any other I/O failure is.
#[cfg(feature = "models-user-config")]
fn read_user_config(path: &Path) -> Result<Option<String>, Error> {
    match std::fs::read_to_string(path) {
        Ok(raw) => Ok(Some(raw)),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(Error::new(
            ErrorKind::Other,
            format!("read user model config {}: {err}", path.display()),
        )
        .with_source(err)),
    }
}

/// Resolve a baseline entry's API key from the Token Store or environment.
///
/// Returns the key only when resolution yields an API-key Credential; an OAuth
/// Credential (which a [`ModelEntry`] cannot carry) and every failure map to
/// `None`, so Credential resolution never fails a load.
fn resolve_api_key(
    entry: &ModelEntry,
    auth: Option<&dyn TokenStore>,
) -> Option<String> {
    let info = Registry::resolve(entry.model.provider.as_str())?;
    match token_store::resolve(None, auth, info.id.as_str(), info.api_key_env) {
        Ok(Some(Credential::ApiKey { key, .. })) => Some(key),
        _ => None,
    }
}

/// Turn a [`ModelEntry`] into a live Provider over the injected `transport`.
///
/// The entry's [`provider`](crate::Model::provider) selects the adapter, its
/// [`base_url`](crate::Model::base_url) and [`headers`](crate::Model::headers)
/// configure it, and its [`api_key`](ModelEntry::api_key) authenticates it. Only
/// Providers compiled into this build can be constructed.
///
/// # Errors
///
/// [`Authentication`](crate::ErrorKind::Authentication) when the entry carries
/// no API key, or [`InvalidRequest`](crate::ErrorKind::InvalidRequest) when no
/// Provider adapter for the entry's Provider is compiled into this build.
pub fn create_provider<H: HttpClient + 'static>(
    entry: &ModelEntry,
    transport: H,
) -> Result<Arc<dyn Provider>, Error> {
    let provider_id = entry.model.provider.as_str();
    let credential = entry
        .api_key
        .as_deref()
        .map(Credential::api_key)
        .ok_or_else(|| missing_credential(entry))?;
    let model = entry.model.id.as_str().to_owned();

    #[cfg(feature = "anthropic")]
    if provider_id == crate::providers::anthropic::INFO.id.as_str() {
        let provider = crate::providers::AnthropicProvider::new(
            transport, credential, model,
        )
        .with_base_url(entry.model.base_url.clone())
        .with_headers(entry.model.headers.iter().cloned());
        return Ok(Arc::new(provider));
    }

    #[cfg(feature = "openai")]
    if provider_id == crate::providers::openai::INFO.id.as_str() {
        let provider =
            crate::providers::OpenAIProvider::new(transport, credential, model)
                .with_base_url(entry.model.base_url.clone())
                .with_headers(entry.model.headers.iter().cloned());
        return Ok(Arc::new(provider));
    }

    // Every compiled-in Provider has a construction arm above and returns from
    // it; reaching here means the entry names a Provider this build does not
    // compile. The bindings are consumed so a no-Provider build (where the arms
    // vanish) still type-checks without unused-variable warnings.
    let _ = (credential, model, transport);
    Err(no_adapter(provider_id))
}

/// An [`InvalidRequest`](ErrorKind::InvalidRequest) error for a Model the
/// Catalog does not hold.
fn unknown_model(provider: &str, model: &str) -> Error {
    Error::new(
        ErrorKind::InvalidRequest,
        format!("no Model {model:?} for provider {provider:?} in the Catalog"),
    )
}

/// An [`InvalidRequest`](ErrorKind::InvalidRequest) error for an entry whose
/// Provider has no adapter compiled into this build.
fn no_adapter(provider: &str) -> Error {
    Error::new(
        ErrorKind::InvalidRequest,
        format!(
            "no Provider adapter for {provider:?} is compiled into this build"
        ),
    )
}

/// An [`Authentication`](ErrorKind::Authentication) error for an entry with no
/// resolved API key.
fn missing_credential(entry: &ModelEntry) -> Error {
    let provider = entry.model.provider.as_str();
    Error::new(
        ErrorKind::Authentication,
        format!(
            "no Credential for Model {:?}: store one under {provider:?} or set the Provider's API-key environment variable",
            entry.model.id.as_str()
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_carries_the_baseline_entries() {
        let registry = ModelRegistry::load(None, None).unwrap();
        // With no Credential source, entries are present but none is available.
        assert_eq!(registry.models().len(), registry.entries.len());
        assert!(registry.available_models().is_empty());
    }

    #[test]
    fn models_path_round_trips() {
        // A path with no file leaves the baseline untouched and is retained.
        let path = PathBuf::from("/nonexistent/tapir/models.toml");
        let registry = ModelRegistry::load(None, Some(path.clone())).unwrap();
        assert_eq!(registry.models_path(), Some(path.as_path()));
        assert_eq!(
            ModelRegistry::load(None, None).unwrap().models_path(),
            None
        );
    }
}

// The user-override layer, exercised end to end through `load`: a real file on
// disk must surface through `models()`. Needs a Provider to have a baseline to
// override; `openai` supplies one.
#[cfg(all(test, feature = "models-user-config", feature = "openai"))]
mod user_config_tests {
    use super::*;

    /// A models.toml written to a unique temp path, removed on drop.
    struct TempConfig {
        path: PathBuf,
    }

    impl TempConfig {
        fn write(contents: &str) -> Self {
            let name = format!(
                "tapir-models-{}-{:?}.toml",
                std::process::id(),
                std::thread::current().id()
            );
            let path = std::env::temp_dir().join(name);
            std::fs::write(&path, contents).unwrap();
            Self { path }
        }
    }

    impl Drop for TempConfig {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    #[test]
    fn a_user_file_overrides_and_adds_models() {
        let config = TempConfig::write(
            r#"
            [providers.openai]
            base_url = "https://house.internal"

            [[providers.openai.models]]
            id = "gpt-4o-mini"
            max_tokens = 99999

            [[providers.openai.models]]
            id = "gpt-house"
            name = "House Model"
            api = "openai-completions"
            context_window = 64000
            max_tokens = 8192
            "#,
        );

        let registry =
            ModelRegistry::load(None, Some(config.path.clone())).unwrap();

        // The override reached the existing baseline Model.
        let overridden = registry.find("openai", "gpt-4o-mini").unwrap();
        assert_eq!(overridden.model.max_tokens, 99999);
        // The added Model surfaces through `models()`.
        let added = registry.find("openai", "gpt-house").unwrap();
        assert_eq!(added.model.name, "House Model");
        assert!(
            registry
                .models()
                .iter()
                .any(|entry| entry.model.id.as_str() == "gpt-house")
        );
    }

    #[test]
    fn a_broken_user_file_fails_the_load() {
        let config = TempConfig::write("this is = = not toml");
        let err =
            ModelRegistry::load(None, Some(config.path.clone())).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Decode);
    }
}

#[cfg(all(test, feature = "openai"))]
mod openai_tests {
    use super::*;
    use crate::token_store::InMemoryTokenStore;

    #[test]
    fn find_locates_a_baseline_model_by_provider_and_alias() {
        let registry = ModelRegistry::load(None, None).unwrap();
        assert!(registry.find("openai", "gpt-4o-mini").is_some());
        // The `gpt` alias resolves to the same entry.
        assert!(registry.find("gpt", "gpt-4o-mini").is_some());
        // An unknown Model does not match.
        assert!(registry.find("openai", "nope").is_none());
    }

    #[test]
    fn find_by_id_matches_across_providers() {
        let registry = ModelRegistry::load(None, None).unwrap();
        let entry = registry.find_by_id("gpt-4o-mini").unwrap();
        assert_eq!(entry.model.provider.as_str(), "openai");
    }

    #[test]
    fn a_resolved_key_makes_a_model_available() {
        let store = InMemoryTokenStore::new();
        store
            .set("openai", Credential::api_key("sk-openai"))
            .unwrap();
        let registry = ModelRegistry::load(Some(&store), None).unwrap();

        let available = registry.available_models();
        assert!(
            available
                .iter()
                .any(|entry| entry.model.id.as_str() == "gpt-4o-mini")
        );
        // The owned view agrees with the borrowed one.
        assert_eq!(registry.get_available().len(), available.len());
    }
}

#[cfg(all(test, feature = "openai", feature = "test-utils"))]
mod openai_build_tests {
    use super::*;
    use crate::http::MockHttpClient;
    use crate::message::Message;
    use crate::request::CompletionRequest;
    use crate::token_store::InMemoryTokenStore;

    const SAMPLE_RESPONSE: &str = r#"{
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    }"#;

    #[tokio::test]
    async fn create_provider_is_the_one_call_common_path() {
        let store = InMemoryTokenStore::new();
        store
            .set("openai", Credential::api_key("sk-openai"))
            .unwrap();
        let registry = ModelRegistry::load(Some(&store), None).unwrap();
        let http =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));

        let provider = registry
            .create_provider("openai", "gpt-4o-mini", http.clone())
            .unwrap();

        let response = provider
            .complete(CompletionRequest::new(vec![Message::user("hello")]))
            .await
            .unwrap();
        assert_eq!(response.text, "hi");
        // The resolved key reached the wire on the Bearer lane, and the entry's
        // base URL drove the request.
        let sent = http.last_request();
        assert!(sent.url.starts_with("https://api.openai.com"));
        assert!(
            sent.headers
                .iter()
                .any(|(k, v)| k == "authorization" && v == "Bearer sk-openai")
        );
    }

    #[test]
    fn create_provider_without_a_credential_is_an_auth_error() {
        // No Credential source, so the entry resolves no key.
        let registry = ModelRegistry::load(None, None).unwrap();
        let entry = registry.find("openai", "gpt-4o-mini").unwrap();
        let http = Arc::new(MockHttpClient::new());

        // `Arc<dyn Provider>` is not `Debug`, so match rather than `unwrap_err`.
        let Err(err) = create_provider(entry, http) else {
            panic!("an entry with no Credential must not build");
        };
        assert_eq!(err.kind(), ErrorKind::Authentication);
    }

    #[test]
    fn selecting_an_unknown_model_fails_cleanly() {
        let registry = ModelRegistry::load(None, None).unwrap();
        let http = Arc::new(MockHttpClient::new());
        let Err(err) =
            registry.create_provider("openai", "does-not-exist", http)
        else {
            panic!("an unknown Model must not build");
        };
        assert_eq!(err.kind(), ErrorKind::InvalidRequest);
    }
}

// The fetched layer, exercised end to end: a `refresh` fetches over a VCR
// cassette, persists the cache, and a fresh `load` reads it back — first from the
// on-disk cache, then over a replayed refresh that never touches the network.
#[cfg(all(
    test,
    feature = "models-fetch",
    feature = "test-utils",
    feature = "openai"
))]
mod fetch_tests {
    use super::*;
    use crate::http::MockHttpClient;
    use crate::token_store::InMemoryTokenStore;
    use crate::vcr::{VcrClient, VcrMode};

    /// A `/models` response naming one baseline Model and one the baseline lacks.
    const MODELS_RESPONSE: &str = r#"{
        "object": "list",
        "data": [
            {"id": "gpt-4o-mini", "object": "model", "owned_by": "openai"},
            {"id": "gpt-fetched-model", "object": "model", "owned_by": "openai"}
        ]
    }"#;

    /// A unique temp directory, removed with its contents on drop.
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "tapir-fetch-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self { path }
        }

        /// The user-override path (never written); the cache sits beside it.
        fn models_path(&self) -> PathBuf {
            self.path.join("models.toml")
        }

        fn cassette(&self) -> PathBuf {
            self.path.join("cassette.json")
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn openai_store() -> InMemoryTokenStore {
        let store = InMemoryTokenStore::new();
        store
            .set("openai", Credential::api_key("sk-openai"))
            .unwrap();
        store
    }

    #[tokio::test]
    async fn refresh_populates_persists_and_a_later_load_reads_it_back() {
        let dir = TempDir::new("roundtrip");
        let store = openai_store();

        // Record a refresh: the fetch flows through the VCR record path against a
        // fake upstream, populating the fetched layer and writing the cache.
        let mut registry =
            ModelRegistry::load(Some(&store), Some(dir.models_path())).unwrap();
        assert!(registry.find("openai", "gpt-fetched-model").is_none());
        let recorder = VcrClient::new(
            MockHttpClient::with_response(200, MODELS_RESPONSE),
            dir.cassette(),
            VcrMode::Record,
        )
        .unwrap();
        registry.refresh(&recorder).await.unwrap();

        // The discovered Model is now in the Catalog and, sharing the Provider's
        // resolved key, is available to call.
        let fetched = registry.find("openai", "gpt-fetched-model").unwrap();
        assert!(fetched.api_key.is_some());
        assert!(
            registry
                .available_models()
                .iter()
                .any(|e| e.model.id.as_str() == "gpt-fetched-model")
        );
        // The cache was persisted beside the user-override path.
        assert!(fetch::cache_path(&dir.models_path()).exists());

        // A fresh load reads the fetched layer straight from the on-disk cache,
        // with no network at all.
        let reloaded =
            ModelRegistry::load(Some(&store), Some(dir.models_path())).unwrap();
        assert!(reloaded.find("openai", "gpt-fetched-model").is_some());
    }

    #[tokio::test]
    async fn a_replayed_refresh_needs_no_network() {
        let dir = TempDir::new("replay");
        let store = openai_store();

        // Record once so the cassette exists.
        let mut recording =
            ModelRegistry::load(Some(&store), Some(dir.models_path())).unwrap();
        let recorder = VcrClient::new(
            MockHttpClient::with_response(200, MODELS_RESPONSE),
            dir.cassette(),
            VcrMode::Record,
        )
        .unwrap();
        recording.refresh(&recorder).await.unwrap();

        // Replay against a transport that panics if contacted: the refresh is
        // served entirely from the cassette.
        let mut registry =
            ModelRegistry::load(Some(&store), Some(dir.models_path())).unwrap();
        let replayer = VcrClient::new(
            MockHttpClient::new(),
            dir.cassette(),
            VcrMode::Replay,
        )
        .unwrap();
        registry.refresh(&replayer).await.unwrap();
        assert!(registry.find("openai", "gpt-fetched-model").is_some());
    }

    #[tokio::test]
    async fn a_provider_without_a_credential_is_skipped() {
        let dir = TempDir::new("nocred");
        // No Token Store, so no key resolves: the fetch cannot authenticate.
        let mut registry =
            ModelRegistry::load(None, Some(dir.models_path())).unwrap();
        // The transport would panic if a request reached it.
        let http = MockHttpClient::new();
        registry.refresh(&http).await.unwrap();
        assert!(registry.find("openai", "gpt-fetched-model").is_none());
    }
}

// A build with the Catalog on but no Provider feature: the baseline is empty,
// so there is nothing to load, find, or build. Reachable only when every
// Provider feature is off.
#[cfg(all(test, not(any(feature = "anthropic", feature = "openai"))))]
mod empty_tests {
    use super::*;

    #[test]
    fn no_provider_feature_yields_an_empty_catalog() {
        let registry = ModelRegistry::load(None, None).unwrap();
        assert!(registry.models().is_empty());
        assert!(registry.find_by_id("gpt-4o-mini").is_none());
    }
}
