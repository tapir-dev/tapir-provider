// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`Registry`]: the compiled-in catalog of available [`Provider`](crate::Provider)s.
//!
//! Only entries whose Cargo feature is enabled are present, so the default
//! build ships none. A caller selects a Provider by name — its canonical id or
//! an alias — and [`Registry::resolve`] answers which [`ProviderInfo`] that name
//! picks. Turning a selection into a live Provider is the Model Registry's job,
//! not the Registry's: the Registry carries only Provider identity and
//! selection.

use crate::model::ProviderId;

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

/// The compiled-in catalog of available [`Provider`](crate::Provider)s.
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
}

#[cfg(all(test, feature = "anthropic"))]
mod anthropic_tests {
    use super::*;

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
}

#[cfg(all(test, feature = "openai"))]
mod openai_tests {
    use super::*;

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

// A build with no Provider feature enabled: the Registry is empty and no name
// resolves. Only reachable when every Provider feature is off, so the default
// `--all-features` test run skips it; a `--no-default-features` run exercises it.
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
