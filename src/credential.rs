// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`Credential`] authentication material carried by a Provider.

use std::fmt;

/// The authentication material a Provider uses to authorize requests.
///
/// This is one tagged value that round-trips losslessly. Today it carries only
/// an API key; the OAuth token set is a future variant, which is why the enum
/// is `#[non_exhaustive]`.
#[derive(Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Credential {
    /// A long-lived API key.
    ApiKey(String),
}

impl Credential {
    /// Construct an API-key Credential.
    pub fn api_key(key: impl Into<String>) -> Self {
        Self::ApiKey(key.into())
    }

    /// The raw API key, if this Credential is an API key.
    #[must_use]
    pub fn as_api_key(&self) -> Option<&str> {
        match self {
            Self::ApiKey(key) => Some(key),
        }
    }
}

/// Redacts the secret so it never leaks into logs or panic messages.
impl fmt::Debug for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ApiKey(_) => f.write_str("Credential::ApiKey(<redacted>)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_round_trips() {
        let cred = Credential::api_key("sk-test-123");
        assert_eq!(cred.as_api_key(), Some("sk-test-123"));
    }

    #[test]
    fn debug_never_reveals_the_secret() {
        let cred = Credential::api_key("sk-super-secret");
        let rendered = format!("{cred:?}");
        assert!(!rendered.contains("sk-super-secret"));
        assert!(rendered.contains("redacted"));
    }
}
