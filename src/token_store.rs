// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`TokenStore`] persistence boundary for [`Credential`]s and the
//! precedence rule that [`resolve`]s one for a Provider.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::credential::Credential;
use crate::error::Error;

/// The persistence boundary for [`Credential`]s.
///
/// The SDK reads and writes Credentials through it, keyed by a Provider's id
/// (e.g. `"anthropic"`); the caller chooses where they live (in memory, a file,
/// an OS keychain). Reads and writes return [`Error`] so backends that touch
/// the filesystem or a keychain can report failure; the in-memory store never
/// does. Object-safe, so it is held as `&dyn TokenStore` / `Arc<dyn TokenStore>`.
pub trait TokenStore: Send + Sync {
    /// The Credential stored for `provider`, or `None` if none is stored.
    fn get(&self, provider: &str) -> Result<Option<Credential>, Error>;

    /// Store `credential` for `provider`, replacing any previous value.
    fn set(&self, provider: &str, credential: Credential) -> Result<(), Error>;
}

/// A [`TokenStore`] that keeps Credentials in memory for the process lifetime.
///
/// Backed by a `Mutex`, so it reads and writes through a shared reference and
/// can be handed out as `Arc<dyn TokenStore>`. Nothing is persisted: a fresh
/// process starts empty.
#[derive(Debug, Default)]
pub struct InMemoryTokenStore {
    entries: Mutex<HashMap<String, Credential>>,
}

impl InMemoryTokenStore {
    /// Construct an empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl TokenStore for InMemoryTokenStore {
    fn get(&self, provider: &str) -> Result<Option<Credential>, Error> {
        Ok(self
            .entries
            .lock()
            .expect("token store mutex poisoned")
            .get(provider)
            .cloned())
    }

    fn set(&self, provider: &str, credential: Credential) -> Result<(), Error> {
        self.entries
            .lock()
            .expect("token store mutex poisoned")
            .insert(provider.to_owned(), credential);
        Ok(())
    }
}

/// Resolve a Provider's [`Credential`] by precedence.
///
/// Highest precedence first: an `explicit` argument, then the `store` under the
/// `provider` key, then an API key read from the `env_var` environment
/// variable. Returns `None` when every tier is empty; a Provider turns that
/// into an [`Authentication`](crate::ErrorKind::Authentication) error at its
/// own boundary.
///
/// A [`TokenStore::get`] failure propagates rather than silently falling
/// through to the environment: a store that is present but broken is a real
/// error, not an absent Credential.
pub fn resolve(
    explicit: Option<Credential>,
    store: Option<&dyn TokenStore>,
    provider: &str,
    env_var: &str,
) -> Result<Option<Credential>, Error> {
    resolve_with(explicit, store, provider, || std::env::var(env_var).ok())
}

/// The precedence rule with the environment read injected, so it can be tested
/// without mutating the process environment (which `#![forbid(unsafe_code)]`
/// disallows under edition 2024). [`resolve`] is the real-environment wrapper.
fn resolve_with(
    explicit: Option<Credential>,
    store: Option<&dyn TokenStore>,
    provider: &str,
    env: impl FnOnce() -> Option<String>,
) -> Result<Option<Credential>, Error> {
    if let Some(credential) = explicit {
        return Ok(Some(credential));
    }
    if let Some(store) = store
        && let Some(credential) = store.get(provider)?
    {
        return Ok(Some(credential));
    }
    Ok(env().map(Credential::api_key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_store_reads_back_what_it_writes() {
        let store = InMemoryTokenStore::new();
        assert!(store.get("anthropic").unwrap().is_none());

        store
            .set("anthropic", Credential::api_key("sk-stored"))
            .unwrap();
        assert_eq!(
            store.get("anthropic").unwrap().unwrap().as_api_key(),
            Some("sk-stored")
        );
        // A different key is untouched.
        assert!(store.get("openai").unwrap().is_none());
    }

    #[test]
    fn in_memory_store_replaces_on_reset() {
        let store = InMemoryTokenStore::new();
        store.set("anthropic", Credential::api_key("old")).unwrap();
        store.set("anthropic", Credential::api_key("new")).unwrap();
        assert_eq!(
            store.get("anthropic").unwrap().unwrap().as_api_key(),
            Some("new")
        );
    }

    /// An environment that always yields `from-env`.
    fn env_present() -> Option<String> {
        Some("from-env".to_owned())
    }

    /// An environment with the variable unset.
    fn env_absent() -> Option<String> {
        None
    }

    #[test]
    fn explicit_wins_over_store_and_env() {
        let store = InMemoryTokenStore::new();
        store
            .set("anthropic", Credential::api_key("from-store"))
            .unwrap();

        let resolved = resolve_with(
            Some(Credential::api_key("explicit")),
            Some(&store),
            "anthropic",
            env_present,
        )
        .unwrap()
        .unwrap();
        assert_eq!(resolved.as_api_key(), Some("explicit"));
    }

    #[test]
    fn store_wins_over_env_when_no_explicit() {
        let store = InMemoryTokenStore::new();
        store
            .set("anthropic", Credential::api_key("from-store"))
            .unwrap();

        let resolved =
            resolve_with(None, Some(&store), "anthropic", env_present)
                .unwrap()
                .unwrap();
        assert_eq!(resolved.as_api_key(), Some("from-store"));
    }

    #[test]
    fn env_is_the_last_resort() {
        let store = InMemoryTokenStore::new();
        let resolved =
            resolve_with(None, Some(&store), "anthropic", env_present)
                .unwrap()
                .unwrap();
        assert_eq!(resolved.as_api_key(), Some("from-env"));
    }

    #[test]
    fn nothing_resolves_to_none() {
        let store = InMemoryTokenStore::new();
        let resolved =
            resolve_with(None, Some(&store), "anthropic", env_absent).unwrap();
        assert!(resolved.is_none());

        // With no store and no env, also None.
        let resolved =
            resolve_with(None, None, "anthropic", env_absent).unwrap();
        assert!(resolved.is_none());
    }
}
