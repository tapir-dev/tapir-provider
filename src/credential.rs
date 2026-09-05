// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`Credential`] authentication material carried by a Provider.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The authentication material a Provider uses to authorize requests.
///
/// This is one tagged value that round-trips losslessly: either a long-lived
/// API key or an OAuth token set. Both variants keep any fields they did not
/// model (`extra`), so a Credential written by a newer version deserializes and
/// re-serializes here without losing data. The enum is `#[non_exhaustive]`
/// because further authentication schemes may join it.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Credential {
    /// A long-lived API key.
    ApiKey {
        /// The secret key sent to the Provider.
        key: String,
        /// Provider-scoped, non-secret config values carried alongside the key
        /// (a Provider Config) — for example a gateway's account and gateway
        /// ids. Serialized under the `env` wire key; omitted when empty.
        #[serde(
            rename = "env",
            default,
            skip_serializing_if = "BTreeMap::is_empty"
        )]
        config: BTreeMap<String, String>,
        /// Fields present on the wire that this version does not model.
        #[serde(flatten, default)]
        extra: Map<String, Value>,
    },
    /// An OAuth token set (access token, refresh token, expiry).
    #[serde(rename = "oauth")]
    OAuth(OAuthTokens),
}

impl Credential {
    /// Construct an API-key Credential.
    pub fn api_key(key: impl Into<String>) -> Self {
        Self::ApiKey {
            key: key.into(),
            config: BTreeMap::new(),
            extra: Map::new(),
        }
    }

    /// Fold Provider Config entries into an API-key Credential's config.
    ///
    /// A no-op on OAuth Credentials, which carry no Provider Config.
    #[must_use]
    pub fn with_config(
        mut self,
        entries: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        if let Self::ApiKey { config, .. } = &mut self {
            config.extend(entries);
        }
        self
    }

    /// The Provider Config, if this Credential is an API key.
    #[must_use]
    pub fn config(&self) -> Option<&BTreeMap<String, String>> {
        match self {
            Self::ApiKey { config, .. } => Some(config),
            Self::OAuth(_) => None,
        }
    }

    /// Construct an OAuth Credential from a token set.
    #[must_use]
    pub fn oauth(tokens: OAuthTokens) -> Self {
        Self::OAuth(tokens)
    }

    /// The raw API key, if this Credential is an API key.
    #[must_use]
    pub fn as_api_key(&self) -> Option<&str> {
        match self {
            Self::ApiKey { key, .. } => Some(key),
            Self::OAuth(_) => None,
        }
    }

    /// The OAuth token set, if this Credential is an OAuth token set.
    #[must_use]
    pub fn as_oauth(&self) -> Option<&OAuthTokens> {
        match self {
            Self::OAuth(tokens) => Some(tokens),
            Self::ApiKey { .. } => None,
        }
    }
}

/// Redacts the secret so it never leaks into logs or panic messages.
impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiKey { .. } => {
                f.write_str("Credential::ApiKey(<redacted>)")
            }
            Self::OAuth(tokens) => {
                write!(f, "Credential::OAuth({tokens:?})")
            }
        }
    }
}

/// An OAuth token set: the access token used to authorize, the refresh token
/// used to renew it, and when the access token expires.
///
/// Any wire fields this version does not model are kept in `extra` so the token
/// set round-trips losslessly. `#[non_exhaustive]`: construct with
/// [`OAuthTokens::new`].
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OAuthTokens {
    /// The bearer token sent with each request.
    pub access_token: String,
    /// The token used to obtain a fresh access token.
    pub refresh_token: String,
    /// Unix epoch seconds at which the access token expires, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    /// Fields present on the wire that this version does not model.
    #[serde(flatten, default)]
    pub extra: Map<String, Value>,
}

impl OAuthTokens {
    /// Construct an OAuth token set with no unmodeled fields.
    pub fn new(
        access_token: impl Into<String>,
        refresh_token: impl Into<String>,
        expires_at: Option<u64>,
    ) -> Self {
        Self {
            access_token: access_token.into(),
            refresh_token: refresh_token.into(),
            expires_at,
            extra: Map::new(),
        }
    }
}

/// Redacts the tokens; keeps the non-secret expiry visible.
impl fmt::Debug for OAuthTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OAuthTokens")
            .field("access_token", &"<redacted>")
            .field("refresh_token", &"<redacted>")
            .field("expires_at", &self.expires_at)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_round_trips() {
        let cred = Credential::api_key("sk-test-123");
        assert_eq!(cred.as_api_key(), Some("sk-test-123"));
        assert_eq!(cred.as_oauth(), None);
    }

    #[test]
    fn oauth_exposes_its_token_set() {
        let cred =
            Credential::oauth(OAuthTokens::new("acc", "ref", Some(1_700_000)));
        let tokens = cred.as_oauth().expect("oauth token set");
        assert_eq!(tokens.access_token, "acc");
        assert_eq!(tokens.refresh_token, "ref");
        assert_eq!(tokens.expires_at, Some(1_700_000));
        assert_eq!(cred.as_api_key(), None);
    }

    #[test]
    fn debug_never_reveals_the_secret() {
        let cred = Credential::api_key("sk-super-secret");
        let rendered = format!("{cred:?}");
        assert!(!rendered.contains("sk-super-secret"));
        assert!(rendered.contains("redacted"));

        let oauth = Credential::oauth(OAuthTokens::new(
            "access-secret",
            "refresh-secret",
            Some(42),
        ));
        let rendered = format!("{oauth:?}");
        assert!(!rendered.contains("access-secret"));
        assert!(!rendered.contains("refresh-secret"));
        assert!(rendered.contains("redacted"));
        // The non-secret expiry stays visible.
        assert!(rendered.contains("42"));
    }

    #[test]
    fn api_key_serializes_as_a_tagged_value() {
        let json = serde_json::to_value(Credential::api_key("sk-x")).unwrap();
        assert_eq!(json["type"], "api_key");
        assert_eq!(json["key"], "sk-x");
    }

    #[test]
    fn oauth_serializes_as_a_tagged_value() {
        let cred =
            Credential::oauth(OAuthTokens::new("acc", "ref", Some(1_700_000)));
        let json = serde_json::to_value(cred).unwrap();
        assert_eq!(json["type"], "oauth");
        assert_eq!(json["access_token"], "acc");
        assert_eq!(json["refresh_token"], "ref");
        assert_eq!(json["expires_at"], 1_700_000);
    }

    #[test]
    fn credential_round_trips_through_serialization() {
        for cred in [
            Credential::api_key("sk-test"),
            Credential::oauth(OAuthTokens::new("acc", "ref", Some(999))),
            Credential::oauth(OAuthTokens::new("acc", "ref", None)),
        ] {
            let json = serde_json::to_string(&cred).unwrap();
            let back: Credential = serde_json::from_str(&json).unwrap();
            assert_eq!(cred, back);
        }
    }

    #[test]
    fn round_trip_preserves_unknown_fields() {
        // A Credential minted by a newer version carries a field this version
        // does not model; it must survive a decode/encode round-trip.
        let wire = r#"{
            "type": "oauth",
            "access_token": "acc",
            "refresh_token": "ref",
            "expires_at": 1700000,
            "scope": "read write",
            "token_type": "Bearer"
        }"#;
        let cred: Credential = serde_json::from_str(wire).unwrap();
        let tokens = cred.as_oauth().expect("oauth token set");
        assert_eq!(tokens.extra["scope"], "read write");
        assert_eq!(tokens.extra["token_type"], "Bearer");

        let back = serde_json::to_value(&cred).unwrap();
        assert_eq!(back["scope"], "read write");
        assert_eq!(back["token_type"], "Bearer");
        assert_eq!(back["expires_at"], 1_700_000);
    }

    #[test]
    fn round_trip_preserves_unknown_api_key_fields() {
        let wire = r#"{"type":"api_key","key":"sk-x","label":"prod"}"#;
        let cred: Credential = serde_json::from_str(wire).unwrap();
        let back = serde_json::to_value(&cred).unwrap();
        assert_eq!(back["key"], "sk-x");
        assert_eq!(back["label"], "prod");
    }

    #[test]
    fn with_config_folds_entries_and_config_reads_them_back() {
        let cred = Credential::api_key("sk-x")
            .with_config([("CLOUDFLARE_ACCOUNT_ID".into(), "acct".into())]);
        let config = cred.config().expect("api key config");
        assert_eq!(config["CLOUDFLARE_ACCOUNT_ID"], "acct");
    }

    #[test]
    fn with_config_is_a_no_op_on_oauth() {
        let cred = Credential::oauth(OAuthTokens::new("acc", "ref", None))
            .with_config([("K".into(), "v".into())]);
        assert_eq!(cred.config(), None);
    }

    #[test]
    fn config_is_some_and_empty_for_a_bare_api_key() {
        let cred = Credential::api_key("sk-x");
        assert!(cred.config().expect("api key config").is_empty());
    }

    #[test]
    fn config_serializes_as_a_nested_env_object() {
        let cred = Credential::api_key("sk-x")
            .with_config([("CLOUDFLARE_ACCOUNT_ID".into(), "acct".into())]);
        let json = serde_json::to_value(&cred).unwrap();
        assert_eq!(json["type"], "api_key");
        assert_eq!(json["key"], "sk-x");
        assert_eq!(json["env"]["CLOUDFLARE_ACCOUNT_ID"], "acct");
    }

    #[test]
    fn empty_config_is_omitted_from_the_wire() {
        let json = serde_json::to_value(Credential::api_key("sk-x")).unwrap();
        assert!(json.get("env").is_none());
    }

    #[test]
    fn wire_env_deserializes_into_config_not_extra() {
        let wire = r#"{"type":"api_key","key":"sk-x","env":{"ACCT":"a"},"label":"prod"}"#;
        let cred: Credential = serde_json::from_str(wire).unwrap();
        let config = cred.config().expect("api key config");
        assert_eq!(config["ACCT"], "a");

        // The unmodeled field still round-trips; env is not duplicated there.
        let back = serde_json::to_value(&cred).unwrap();
        assert_eq!(back["label"], "prod");
        assert_eq!(back["env"]["ACCT"], "a");
    }

    #[test]
    fn config_round_trips_through_serialization() {
        let cred = Credential::api_key("sk-x").with_config([
            ("ACCT".into(), "a".into()),
            ("GATEWAY".into(), "g".into()),
        ]);
        let json = serde_json::to_string(&cred).unwrap();
        let back: Credential = serde_json::from_str(&json).unwrap();
        assert_eq!(cred, back);
    }
}
