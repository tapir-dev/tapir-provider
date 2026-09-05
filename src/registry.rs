// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`Registry`]: the compiled-in catalog of available [`Provider`]s.
//!
//! Only entries whose Cargo feature is enabled are present, so the default
//! build ships none and building any Provider by name fails cleanly. A caller
//! selects a Provider by name — its canonical id or an alias — and
//! [`Registry::build`] turns that name plus a Model, a Credential, and a
//! transport into an `Arc<dyn Provider>`, resolving the Credential from the
//! argument or the Provider's default API-key environment variable.

use std::sync::Arc;

use crate::credential::Credential;
use crate::error::{Error, ErrorKind};
use crate::http::HttpClient;
use crate::model::ProviderId;
use crate::provider::Provider;

/// A compiled-in Provider's identity in the [`Registry`].
///
/// It names the Provider (its canonical `id`, the same key it stores Credentials
/// under in a Token Store), the alternate `aliases` that also select it, and the
/// `api_key_env` environment variable holding its default API key. Name matching
/// against the id and aliases is ASCII-case-insensitive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderInfo {
    /// The canonical Provider id (also its Token Store key).
    pub id: ProviderId,
    /// Alternate names that select this Provider.
    pub aliases: &'static [&'static str],
    /// The environment variable holding this Provider's default API key.
    pub api_key_env: &'static str,
}

impl ProviderInfo {
    /// Whether `name` selects this Provider — its canonical id or any alias,
    /// compared ASCII-case-insensitively.
    #[must_use]
    pub fn matches(&self, name: &str) -> bool {
        self.id.as_str().eq_ignore_ascii_case(name)
            || self
                .aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(name))
    }
}

/// Every Provider entry compiled into this build.
///
/// Each element is gated on its Provider feature, so a build with no Provider
/// feature yields an empty slice.
const ENTRIES: &[ProviderInfo] = &[
    #[cfg(feature = "anthropic")]
    crate::providers::anthropic::INFO,
    #[cfg(feature = "openai")]
    crate::providers::openai::INFO,
];

/// The compiled-in catalog of available [`Provider`]s.
///
/// A zero-sized handle over the feature-gated entry table; all its operations
/// are associated functions.
#[derive(Debug, Clone, Copy, Default)]
pub struct Registry;

impl Registry {
    /// The identity of every Provider compiled into this build.
    #[must_use]
    pub fn entries() -> &'static [ProviderInfo] {
        ENTRIES
    }

    /// The canonical id of every Provider compiled into this build.
    #[must_use]
    pub fn provider_ids() -> Vec<&'static str> {
        ENTRIES.iter().map(|info| info.id.as_str()).collect()
    }

    /// The [`ProviderInfo`] a `name` selects — its canonical id or an alias — or
    /// `None` when no compiled-in Provider matches.
    #[must_use]
    pub fn resolve(name: &str) -> Option<&'static ProviderInfo> {
        ENTRIES.iter().find(|info| info.matches(name))
    }

    /// Build the Provider selected by `name` over the injected `transport`.
    ///
    /// `name` is a canonical id or an alias. The Credential is resolved by
    /// precedence: the explicit `credential` if `Some`, else the Provider's
    /// default API-key environment variable ([`ProviderInfo::api_key_env`]). A
    /// name that no compiled-in Provider matches is an
    /// [`InvalidRequest`](crate::ErrorKind::InvalidRequest) error; a Provider
    /// that resolves no Credential is an
    /// [`Authentication`](crate::ErrorKind::Authentication) error.
    pub fn build<H: HttpClient + 'static>(
        name: &str,
        model: impl Into<String>,
        credential: Option<Credential>,
        transport: H,
    ) -> Result<Arc<dyn Provider>, Error> {
        let info = Self::resolve(name).ok_or_else(|| unknown_provider(name))?;

        // Resolve the Credential at the selection boundary so the entry's
        // `api_key_env` is what actually names the fallback variable: the
        // explicit argument wins, else that environment variable. A Provider's
        // own `resolve` then receives the already-chosen Credential.
        let credential = crate::token_store::resolve(
            credential,
            None,
            info.id.as_str(),
            info.api_key_env,
        )?;

        #[cfg(feature = "anthropic")]
        if info.id == crate::providers::anthropic::INFO.id {
            let provider = crate::providers::AnthropicProvider::resolve(
                transport, model, credential, None,
            )?;
            return Ok(Arc::new(provider));
        }

        #[cfg(feature = "openai")]
        if info.id == crate::providers::openai::INFO.id {
            let provider = crate::providers::OpenAIProvider::resolve(
                transport, model, credential, None,
            )?;
            return Ok(Arc::new(provider));
        }

        // `resolve` only ever returns a compiled-in entry, and every such entry
        // has a construction arm above; reaching here would be one added without
        // its arm. The bindings are consumed here so a no-Provider build (where
        // the arms above vanish and `resolve` always returns `None`) still type-
        // checks without unused-variable warnings.
        let _ = (info, model, credential, transport);
        Err(unknown_provider(name))
    }
}

/// An [`InvalidRequest`](ErrorKind::InvalidRequest) error naming the Providers
/// this build actually offers, so a caller who selects a missing one — often a
/// Provider whose feature is off — learns what is available.
fn unknown_provider(name: &str) -> Error {
    let available = Registry::provider_ids();
    let message = if available.is_empty() {
        format!(
            "unknown provider {name:?}: no provider features are enabled in this build"
        )
    } else {
        format!(
            "unknown provider {name:?}; available: {}",
            available.join(", ")
        )
    };
    Error::new(ErrorKind::InvalidRequest, message)
}

#[cfg(all(test, feature = "anthropic", feature = "test-utils"))]
mod anthropic_tests {
    use super::*;
    use crate::http::MockHttpClient;
    use crate::message::Message;
    use crate::request::CompletionRequest;

    const SAMPLE_RESPONSE: &str = r#"{
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1}
    }"#;

    #[test]
    fn anthropic_is_compiled_in_and_resolves_by_id_and_alias() {
        assert!(Registry::provider_ids().contains(&"anthropic"));
        assert_eq!(
            Registry::resolve("anthropic").unwrap().id.as_str(),
            "anthropic"
        );
        // The `claude` alias selects the same Provider...
        assert_eq!(
            Registry::resolve("claude").unwrap().id.as_str(),
            "anthropic"
        );
        // ...and matching is case-insensitive.
        assert_eq!(
            Registry::resolve("Anthropic").unwrap().id.as_str(),
            "anthropic"
        );
        // The entry exposes the default API-key environment variable.
        assert_eq!(
            Registry::resolve("anthropic").unwrap().api_key_env,
            "ANTHROPIC_API_KEY"
        );
    }

    #[test]
    fn an_unknown_name_does_not_resolve() {
        assert!(Registry::resolve("does-not-exist").is_none());
    }

    #[tokio::test]
    async fn builds_anthropic_by_name_and_completes() {
        let http =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = Registry::build(
            "anthropic",
            "claude-3-5-sonnet",
            Some(Credential::api_key("sk-test")),
            http,
        )
        .unwrap();

        let response = provider
            .complete(CompletionRequest::new(vec![Message::user("hello")]))
            .await
            .unwrap();
        assert_eq!(response.text, "hi");
    }

    #[tokio::test]
    async fn builds_anthropic_through_its_alias() {
        let http =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = Registry::build(
            "claude",
            "claude-3-5-sonnet",
            Some(Credential::api_key("sk-test")),
            http.clone(),
        )
        .unwrap();

        provider
            .complete(CompletionRequest::new(vec![Message::user("hello")]))
            .await
            .unwrap();
        // The resolved API key reached the wire under the anthropic lane.
        assert!(
            http.last_request()
                .headers
                .iter()
                .any(|(k, v)| k == "x-api-key" && v == "sk-test")
        );
    }

    #[test]
    fn building_an_unknown_provider_fails_cleanly() {
        let http = Arc::new(MockHttpClient::new());
        // A name no compiled-in Provider claims, whatever features are on.
        // `Arc<dyn Provider>` is not `Debug`, so match rather than `unwrap_err`.
        let Err(err) = Registry::build(
            "cohere",
            "command",
            Some(Credential::api_key("sk-test")),
            http,
        ) else {
            panic!("an unknown provider must not build");
        };
        assert_eq!(err.kind(), ErrorKind::InvalidRequest);
        assert!(err.message().contains("cohere"));
    }
}

#[cfg(all(test, feature = "openai", feature = "test-utils"))]
mod openai_tests {
    use super::*;
    use crate::http::MockHttpClient;
    use crate::message::Message;
    use crate::request::CompletionRequest;

    const SAMPLE_RESPONSE: &str = r#"{
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}
    }"#;

    #[test]
    fn openai_is_compiled_in_and_resolves_by_id_and_alias() {
        assert!(Registry::provider_ids().contains(&"openai"));
        assert_eq!(Registry::resolve("openai").unwrap().id.as_str(), "openai");
        // The `gpt` alias selects the same Provider, case-insensitively.
        assert_eq!(Registry::resolve("gpt").unwrap().id.as_str(), "openai");
        assert_eq!(Registry::resolve("OpenAI").unwrap().id.as_str(), "openai");
        assert_eq!(
            Registry::resolve("openai").unwrap().api_key_env,
            "OPENAI_API_KEY"
        );
    }

    #[tokio::test]
    async fn builds_openai_by_name_and_completes() {
        let http =
            Arc::new(MockHttpClient::with_response(200, SAMPLE_RESPONSE));
        let provider = Registry::build(
            "openai",
            "gpt-4o-mini",
            Some(Credential::api_key("sk-test")),
            http.clone(),
        )
        .unwrap();

        let response = provider
            .complete(CompletionRequest::new(vec![Message::user("hello")]))
            .await
            .unwrap();
        assert_eq!(response.text, "hi");
        // The resolved key reached the wire on the Bearer lane.
        assert!(
            http.last_request()
                .headers
                .iter()
                .any(|(k, v)| k == "authorization" && v == "Bearer sk-test")
        );
    }
}

// Enabling both provider features registers both Providers; enabling only one
// registers only that one. The `--all-features` run enables both and exercises
// this; a single-feature run exercises the exclusive arms.
#[cfg(test)]
mod registration_tests {
    use super::*;

    #[test]
    fn each_enabled_provider_feature_is_registered() {
        let ids = Registry::provider_ids();
        assert_eq!(cfg!(feature = "anthropic"), ids.contains(&"anthropic"));
        assert_eq!(cfg!(feature = "openai"), ids.contains(&"openai"));
    }
}

// A build with no Provider feature enabled: the Registry is empty and building
// any Provider by name fails cleanly. Only reachable when every Provider feature
// is off, so the default `--all-features` test run skips it; a
// `--no-default-features` run exercises it.
#[cfg(all(test, not(any(feature = "anthropic", feature = "openai"))))]
mod empty_tests {
    use super::*;

    #[test]
    fn no_provider_feature_yields_an_empty_registry() {
        assert!(Registry::entries().is_empty());
        assert!(Registry::provider_ids().is_empty());
        assert!(Registry::resolve("anthropic").is_none());
    }
}
