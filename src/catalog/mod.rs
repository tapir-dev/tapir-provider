// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`ModelRegistry`]: the runtime holder of the Catalog.
//!
//! Where the [`Registry`] is a zero-sized, compile-time handle over Provider
//! identity, the Model Registry is a stateful, owned object. It
//! [`load`](ModelRegistry::load)s the layers of the Catalog — today just the
//! compiled-in baseline — resolves each entry's Credential through the same
//! Token Store / environment precedence every construction path uses, and turns
//! a [`ModelEntry`] into a live `Arc<dyn Provider>` via
//! [`create_provider`]. The common path is one call:
//! `ModelRegistry::load(auth, None).create_provider("openai", "gpt-4o-mini", http)?`.
//!
//! A [`ModelEntry`] carries only an API-key Credential ([`ModelEntry::api_key`]),
//! so `load` folds a resolved API key into the entry and an OAuth Credential is
//! left for a Provider's own construction path. An entry with no resolved key is
//! reported by [`models`](ModelRegistry::models) but excluded from
//! [`available_models`](ModelRegistry::available_models).

mod baseline;

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
    /// The compiled-in baseline layer is loaded, then every entry without an
    /// API key already set has one resolved by precedence: the `auth` Token
    /// Store under the entry's Provider id, else that Provider's default API-key
    /// environment variable. Resolution never fails the load — a broken store or
    /// an OAuth-only Credential simply leaves the entry without a key, so it is
    /// present in [`models`](Self::models) but not in
    /// [`available_models`](Self::available_models).
    ///
    /// `models_path` is where the user-override and fetched layers live; those
    /// layers sit behind their own features and are not loaded yet, so the path
    /// is retained ([`models_path`](Self::models_path)) for when they land.
    #[must_use]
    pub fn load(
        auth: Option<&dyn TokenStore>,
        models_path: Option<PathBuf>,
    ) -> Self {
        let mut entries = baseline::entries();
        for entry in &mut entries {
            if entry.api_key.is_some() {
                continue;
            }
            entry.api_key = resolve_api_key(entry, auth);
        }
        Self {
            entries,
            models_path,
        }
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
}

/// Resolve a baseline entry's API key from the Token Store or environment.
///
/// Returns the key only when resolution yields an API-key Credential; an OAuth
/// Credential (which a [`ModelEntry`] cannot carry) and every failure map to
/// `None`, so [`ModelRegistry::load`] stays infallible.
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
        let registry = ModelRegistry::load(None, None);
        // With no Credential source, entries are present but none is available.
        assert_eq!(registry.models().len(), registry.entries.len());
        assert!(registry.available_models().is_empty());
    }

    #[test]
    fn models_path_round_trips() {
        let registry =
            ModelRegistry::load(None, Some(PathBuf::from("/tmp/models.json")));
        assert_eq!(registry.models_path(), Some(Path::new("/tmp/models.json")));
        assert_eq!(ModelRegistry::load(None, None).models_path(), None);
    }
}

#[cfg(all(test, feature = "openai"))]
mod openai_tests {
    use super::*;
    use crate::token_store::InMemoryTokenStore;

    #[test]
    fn find_locates_a_baseline_model_by_provider_and_alias() {
        let registry = ModelRegistry::load(None, None);
        assert!(registry.find("openai", "gpt-4o-mini").is_some());
        // The `gpt` alias resolves to the same entry.
        assert!(registry.find("gpt", "gpt-4o-mini").is_some());
        // An unknown Model does not match.
        assert!(registry.find("openai", "nope").is_none());
    }

    #[test]
    fn find_by_id_matches_across_providers() {
        let registry = ModelRegistry::load(None, None);
        let entry = registry.find_by_id("gpt-4o-mini").unwrap();
        assert_eq!(entry.model.provider.as_str(), "openai");
    }

    #[test]
    fn a_resolved_key_makes_a_model_available() {
        let store = InMemoryTokenStore::new();
        store
            .set("openai", Credential::api_key("sk-openai"))
            .unwrap();
        let registry = ModelRegistry::load(Some(&store), None);

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
        let registry = ModelRegistry::load(Some(&store), None);
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
        let registry = ModelRegistry::load(None, None);
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
        let registry = ModelRegistry::load(None, None);
        let http = Arc::new(MockHttpClient::new());
        let Err(err) =
            registry.create_provider("openai", "does-not-exist", http)
        else {
            panic!("an unknown Model must not build");
        };
        assert_eq!(err.kind(), ErrorKind::InvalidRequest);
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
        let registry = ModelRegistry::load(None, None);
        assert!(registry.models().is_empty());
        assert!(registry.find_by_id("gpt-4o-mini").is_none());
    }
}
