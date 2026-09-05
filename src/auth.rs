// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! [`ResolvedAuth`]: what one auth inspection produces, and the [`AuthSource`]
//! tier it came from.
//!
//! Where a [`Credential`](crate::Credential) is the stored authentication
//! *material*, a Resolved Auth is the request-ready result of asking "how would
//! this Provider authenticate right now?" — its auth headers, an auth-derived
//! API key and base URL when there is one, and the [`AuthSource`] tier that won
//! the precedence. It is computed on demand and never persisted.

use std::collections::BTreeMap;
use std::fmt;

/// Which tier a [`ResolvedAuth`] came from.
///
/// The precedence resolution keeps the winning tier rather than discarding it,
/// so a caller can see how a Provider is configured — an explicit per-request
/// key, a stored Credential, a named environment variable, or an OAuth token.
/// `#[non_exhaustive]` because further tiers may join it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AuthSource {
    /// An explicit key supplied on the request itself.
    Explicit,
    /// A Credential read from the Token Store.
    Stored,
    /// An API key read from the named environment variable.
    Env(String),
    /// A stored OAuth token set.
    OAuth,
}

/// Renders the human-readable label the SDK reports for each tier.
impl fmt::Display for AuthSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Explicit => f.write_str("explicit key"),
            Self::Stored => f.write_str("stored credential"),
            Self::Env(name) => f.write_str(name),
            Self::OAuth => f.write_str("OAuth"),
        }
    }
}

/// The request-ready material a Provider would send this turn.
///
/// One auth inspection produces this: the auth `headers`, an auth-derived
/// `api_key` and `base_url` when there is one, the winning Credential's Provider
/// `config`, and the [`AuthSource`] that won. A provider-scoped inspection
/// carries only the auth headers; a Model-scoped one also layers that Model's
/// headers and base URL. `#[non_exhaustive]` because further resolved fields may
/// join it.
#[derive(Clone)]
#[non_exhaustive]
pub struct ResolvedAuth {
    /// Which tier this resolution came from.
    pub source: AuthSource,
    /// The auth headers a request would carry, in send order. A Model-scoped
    /// inspection appends that Model's own headers after them.
    pub headers: Vec<(String, String)>,
    /// The auth-derived API key, if the winning Credential is an API key.
    pub api_key: Option<String>,
    /// The base URL a Model-scoped inspection resolved; `None` when
    /// provider-scoped.
    pub base_url: Option<String>,
    /// The winning Credential's Provider Config, non-secret. Populated only when
    /// a stored API-key Credential wins; empty for the per-request key,
    /// environment-variable, and OAuth tiers. Present regardless of scope, since
    /// it comes from the Credential, not the Model.
    pub config: BTreeMap<String, String>,
}

/// Redacts header *values* and the `api_key` so a secret never reaches Debug
/// output — matching the crate's [`Credential`](crate::Credential) discipline.
/// Header names, the source, the base URL, and the non-secret Provider config
/// stay visible.
impl fmt::Debug for ResolvedAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let headers: Vec<(&str, &str)> = self
            .headers
            .iter()
            .map(|(name, _)| (name.as_str(), "<redacted>"))
            .collect();
        f.debug_struct("ResolvedAuth")
            .field("source", &self.source)
            .field("headers", &headers)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("base_url", &self.base_url)
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_renders_the_documented_labels() {
        assert_eq!(AuthSource::Explicit.to_string(), "explicit key");
        assert_eq!(AuthSource::Stored.to_string(), "stored credential");
        assert_eq!(
            AuthSource::Env("ANTHROPIC_API_KEY".to_owned()).to_string(),
            "ANTHROPIC_API_KEY"
        );
        assert_eq!(AuthSource::OAuth.to_string(), "OAuth");
    }

    #[test]
    fn debug_redacts_header_values_and_the_api_key() {
        let auth = ResolvedAuth {
            source: AuthSource::Stored,
            headers: vec![(
                "x-api-key".to_owned(),
                "sk-super-secret".to_owned(),
            )],
            api_key: Some("sk-super-secret".to_owned()),
            base_url: Some("https://api.example.com".to_owned()),
            config: BTreeMap::from([(
                "CLOUDFLARE_ACCOUNT_ID".to_owned(),
                "acct-123".to_owned(),
            )]),
        };
        let rendered = format!("{auth:?}");
        assert!(!rendered.contains("sk-super-secret"));
        assert!(rendered.contains("<redacted>"));
        // Non-secret fields stay visible.
        assert!(rendered.contains("x-api-key"));
        assert!(rendered.contains("Stored"));
        assert!(rendered.contains("https://api.example.com"));
        // The Provider Config is non-secret and renders like the base URL.
        assert!(rendered.contains("CLOUDFLARE_ACCOUNT_ID"));
        assert!(rendered.contains("acct-123"));
    }
}
