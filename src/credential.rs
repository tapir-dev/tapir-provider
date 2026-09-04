// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`Credential`] authentication material carried by a Provider.

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
            extra: Map::new(),
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
}
