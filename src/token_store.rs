// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`TokenStore`] persistence boundary for [`Credential`]s and the
//! precedence rule that [`resolve`]s one for a Provider.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

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

/// A source that renews an expiring OAuth [`Credential`].
///
/// The Token Store's proactive refresh calls this when a stored OAuth token
/// falls within its pre-expiry window; an OAuth flow (for example
/// `AnthropicOAuth`) implements it. Injecting the refresher keeps refresh
/// coordination testable against a double instead of a live token endpoint.
#[async_trait]
pub trait Refresh: Send + Sync {
    /// Exchange `refresh_token` for a renewed OAuth [`Credential`].
    ///
    /// # Errors
    ///
    /// Whatever the underlying flow reports: a transport failure, a non-2xx
    /// token-endpoint response, or a body that will not decode.
    async fn refresh(&self, refresh_token: &str) -> Result<Credential, Error>;
}

/// Whether `credential` should be proactively refreshed.
///
/// A Credential is stale when it is an OAuth token with a known `expires_at`
/// that falls within `window_secs` of `now` (both Unix epoch seconds). A
/// non-OAuth Credential, or an OAuth token with no known expiry, is never stale
/// — so an API key is never refreshed.
///
/// Only the file-backed store's proactive refresh consults this, so it is gated
/// on `token-store-file`.
#[cfg(feature = "token-store-file")]
fn needs_refresh(credential: &Credential, window_secs: u64, now: u64) -> bool {
    match credential.as_oauth().and_then(|tokens| tokens.expires_at) {
        Some(expires_at) => now.saturating_add(window_secs) >= expires_at,
        None => false,
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

#[cfg(feature = "token-store-file")]
pub use file::{DEFAULT_REFRESH_WINDOW_SECS, FileTokenStore};

/// The file-backed [`TokenStore`] and its locked, double-checked proactive
/// refresh. Gated on `token-store-file`.
#[cfg(feature = "token-store-file")]
mod file {
    use std::collections::HashMap;
    use std::fs::{File, OpenOptions};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    use fs4::FileExt;

    use super::{Refresh, TokenStore, needs_refresh};
    use crate::credential::Credential;
    use crate::error::{Error, ErrorKind};

    /// Default pre-expiry window for [`FileTokenStore::refresh_if_stale`]: an
    /// OAuth token is refreshed once it comes within five minutes of expiring.
    /// It matches the safety margin baked into a minted OAuth Credential.
    pub const DEFAULT_REFRESH_WINDOW_SECS: u64 = 300;

    /// A [`TokenStore`] that persists Credentials to a TOML file, guarded by an
    /// OS advisory file lock so concurrent processes coordinate.
    ///
    /// The whole file is one table of `provider` → [`Credential`]; a read takes
    /// a shared lock and a write an exclusive one, so a fresh process loads the
    /// Credentials an earlier run stored. On Unix the file is created `0600`,
    /// owner-only, since Credentials are sensitive.
    ///
    /// Beyond plain [`get`](TokenStore::get)/[`set`](TokenStore::set), it owns
    /// [`refresh_if_stale`](Self::refresh_if_stale): a locked, double-checked
    /// proactive refresh that renews an expiring OAuth token exactly once even
    /// when several callers race.
    #[derive(Debug, Clone)]
    pub struct FileTokenStore {
        path: PathBuf,
    }

    impl FileTokenStore {
        /// Back the store with the file at `path`. The file and its parent
        /// directory are created on first write, not here.
        #[must_use]
        pub fn new(path: impl Into<PathBuf>) -> Self {
            Self { path: path.into() }
        }

        /// The file the Credentials live in.
        #[must_use]
        pub fn path(&self) -> &Path {
            &self.path
        }

        /// Renew a stored OAuth Credential that is within `window_secs` of
        /// expiring, returning the Credential now in force (renewed, unchanged,
        /// or `None` if none is stored for `provider`).
        ///
        /// The refresh is double-checked under an exclusive file lock: the token
        /// is first read with only a shared lock, and if it looks stale the
        /// exclusive lock is taken and the token re-read. A concurrent caller
        /// that already renewed it is observed on that second read, so the
        /// redundant refresh is skipped and [`Refresh`] is invoked at most once
        /// across the racing callers. A non-OAuth Credential is never refreshed.
        ///
        /// # Errors
        ///
        /// A filesystem or lock failure; a TOML parse failure on a corrupt file;
        /// or whatever `refresher` reports when a refresh is actually performed.
        pub async fn refresh_if_stale<R: Refresh>(
            &self,
            provider: &str,
            refresher: &R,
            window_secs: u64,
        ) -> Result<Option<Credential>, Error> {
            self.refresh_if_stale_at(
                provider,
                refresher,
                window_secs,
                now_unix(),
            )
            .await
        }

        /// [`refresh_if_stale`](Self::refresh_if_stale) with the clock injected,
        /// so staleness is exercised without waiting on real time.
        pub(super) async fn refresh_if_stale_at<R: Refresh>(
            &self,
            provider: &str,
            refresher: &R,
            window_secs: u64,
            now: u64,
        ) -> Result<Option<Credential>, Error> {
            // First check under only a shared lock: the common path is a fresh
            // token, and a shared read keeps concurrent readers from serializing.
            match self.get(provider)? {
                None => return Ok(None),
                Some(current) if !needs_refresh(&current, window_secs, now) => {
                    return Ok(Some(current));
                }
                Some(_) => {}
            }

            // The token looked stale. Take the exclusive lock and re-read: a
            // racing caller may have renewed it while we waited for the lock.
            let mut file = self.open_write()?;
            file.lock().map_err(|err| lock_error(&self.path, err))?;
            let result = self
                .refresh_locked(
                    &mut file,
                    provider,
                    refresher,
                    window_secs,
                    now,
                )
                .await;
            // Drop would also unlock; unlock explicitly so the lock is released
            // before the File closes even if a later step is added.
            let _ = FileExt::unlock(&file);
            result
        }

        /// The refresh body run while holding the exclusive lock: re-read,
        /// re-check, and renew only if still stale.
        async fn refresh_locked<R: Refresh>(
            &self,
            file: &mut File,
            provider: &str,
            refresher: &R,
            window_secs: u64,
            now: u64,
        ) -> Result<Option<Credential>, Error> {
            let mut entries = self.read_all(file)?;
            let Some(current) = entries.get(provider).cloned() else {
                return Ok(None);
            };
            // The double check: a concurrent writer may have renewed the token
            // between the shared read and this exclusive lock.
            if !needs_refresh(&current, window_secs, now) {
                return Ok(Some(current));
            }

            // needs_refresh only returns true for an OAuth Credential, so the
            // refresh token is present.
            let refresh_token = current
                .as_oauth()
                .expect("needs_refresh implies an OAuth Credential")
                .refresh_token
                .clone();
            let renewed = refresher.refresh(&refresh_token).await?;
            entries.insert(provider.to_owned(), renewed.clone());
            self.write_all(file, &entries)?;
            Ok(Some(renewed))
        }

        /// Open the store file for reading, or `None` if it does not exist yet.
        fn open_read(&self) -> Result<Option<File>, Error> {
            match File::open(&self.path) {
                Ok(file) => Ok(Some(file)),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                    Ok(None)
                }
                Err(err) => Err(io_error(&self.path, "open", err)),
            }
        }

        /// Open the store file for read+write, creating it (and any missing
        /// parent directories) `0600` on first use.
        fn open_write(&self) -> Result<File, Error> {
            if let Some(parent) = self.path.parent()
                && !parent.as_os_str().is_empty()
            {
                std::fs::create_dir_all(parent)
                    .map_err(|err| io_error(parent, "create directory", err))?;
            }
            let mut options = OpenOptions::new();
            options.read(true).write(true).create(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            options
                .open(&self.path)
                .map_err(|err| io_error(&self.path, "open", err))
        }

        /// Read and parse the whole table from an open, locked file.
        fn read_all(
            &self,
            file: &mut File,
        ) -> Result<HashMap<String, Credential>, Error> {
            file.seek(SeekFrom::Start(0))
                .map_err(|err| io_error(&self.path, "seek", err))?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)
                .map_err(|err| io_error(&self.path, "read", err))?;
            if contents.trim().is_empty() {
                return Ok(HashMap::new());
            }
            toml::from_str(&contents).map_err(Error::decode)
        }

        /// Serialize the whole table and overwrite the open, locked file with it.
        fn write_all(
            &self,
            file: &mut File,
            entries: &HashMap<String, Credential>,
        ) -> Result<(), Error> {
            let serialized =
                toml::to_string(entries).map_err(Error::serialize)?;
            file.seek(SeekFrom::Start(0))
                .map_err(|err| io_error(&self.path, "seek", err))?;
            file.set_len(0)
                .map_err(|err| io_error(&self.path, "truncate", err))?;
            file.write_all(serialized.as_bytes())
                .map_err(|err| io_error(&self.path, "write", err))?;
            file.flush()
                .map_err(|err| io_error(&self.path, "flush", err))
        }
    }

    impl TokenStore for FileTokenStore {
        fn get(&self, provider: &str) -> Result<Option<Credential>, Error> {
            let Some(mut file) = self.open_read()? else {
                return Ok(None);
            };
            file.lock_shared()
                .map_err(|err| lock_error(&self.path, err))?;
            let entries = self.read_all(&mut file);
            let _ = FileExt::unlock(&file);
            Ok(entries?.remove(provider))
        }

        fn set(
            &self,
            provider: &str,
            credential: Credential,
        ) -> Result<(), Error> {
            let mut file = self.open_write()?;
            file.lock().map_err(|err| lock_error(&self.path, err))?;
            let result = (|| {
                let mut entries = self.read_all(&mut file)?;
                entries.insert(provider.to_owned(), credential);
                self.write_all(&mut file, &entries)
            })();
            let _ = FileExt::unlock(&file);
            result
        }
    }

    /// Current Unix time in seconds, or 0 if the clock predates the epoch.
    fn now_unix() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Wrap a filesystem error with the path and the operation that failed.
    fn io_error(path: &Path, op: &str, err: std::io::Error) -> Error {
        Error::new(
            ErrorKind::Other,
            format!("token store {op} failed for {}: {err}", path.display()),
        )
        .with_source(err)
    }

    /// Wrap a file-lock failure with the path it was locking.
    fn lock_error(path: &Path, err: std::io::Error) -> Error {
        Error::new(
            ErrorKind::Other,
            format!("token store lock failed for {}: {err}", path.display()),
        )
        .with_source(err)
    }
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

#[cfg(all(test, feature = "token-store-file"))]
mod refresh_tests {
    use super::*;
    use crate::credential::OAuthTokens;

    fn oauth(expires_at: Option<u64>) -> Credential {
        Credential::oauth(OAuthTokens::new("acc", "ref", expires_at))
    }

    #[test]
    fn non_oauth_is_never_stale() {
        // An API key has no expiry to reason about; it must never be refreshed.
        assert!(!needs_refresh(&Credential::api_key("sk"), 300, 1_000));
    }

    #[test]
    fn oauth_without_expiry_is_never_stale() {
        assert!(!needs_refresh(&oauth(None), 300, u64::MAX));
    }

    #[test]
    fn oauth_within_the_window_is_stale() {
        // Expires at 1000; with a 300s window, now=800 lands inside it.
        assert!(needs_refresh(&oauth(Some(1_000)), 300, 800));
        // Exactly at the window edge counts as stale.
        assert!(needs_refresh(&oauth(Some(1_000)), 300, 700));
    }

    #[test]
    fn oauth_outside_the_window_is_fresh() {
        assert!(!needs_refresh(&oauth(Some(1_000)), 300, 699));
    }
}

#[cfg(all(test, feature = "token-store-file"))]
mod file_test_support {
    use super::FileTokenStore;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// A unique temp path that deletes its file on drop, so parallel tests do
    /// not collide and nothing leaks into the temp dir.
    pub(super) struct TempStore {
        path: PathBuf,
    }

    impl TempStore {
        pub(super) fn new(tag: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let mut path = std::env::temp_dir();
            path.push(format!(
                "tapir-tokenstore-{tag}-{}-{n}-{nanos}.toml",
                std::process::id()
            ));
            Self { path }
        }

        pub(super) fn store(&self) -> FileTokenStore {
            FileTokenStore::new(self.path.clone())
        }
    }

    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(all(test, feature = "token-store-file"))]
mod file_tests {
    use super::file_test_support::TempStore;
    use super::*;
    use crate::credential::OAuthTokens;

    #[test]
    fn get_on_a_missing_file_is_none() {
        let temp = TempStore::new("missing");
        assert!(temp.store().get("anthropic").unwrap().is_none());
    }

    #[test]
    fn persists_and_reloads_across_store_instances() {
        let temp = TempStore::new("persist");
        // Write through one instance...
        temp.store()
            .set(
                "anthropic",
                Credential::oauth(OAuthTokens::new("acc", "ref", Some(1_700))),
            )
            .unwrap();
        temp.store()
            .set("openai", Credential::api_key("sk-openai"))
            .unwrap();

        // ...and read it back through a fresh one, as a new process would.
        let reloaded = temp.store();
        let anthropic = reloaded.get("anthropic").unwrap().unwrap();
        let tokens = anthropic.as_oauth().unwrap();
        assert_eq!(tokens.access_token, "acc");
        assert_eq!(tokens.refresh_token, "ref");
        assert_eq!(tokens.expires_at, Some(1_700));
        assert_eq!(
            reloaded.get("openai").unwrap().unwrap().as_api_key(),
            Some("sk-openai")
        );
        assert!(reloaded.get("cohere").unwrap().is_none());
    }

    #[test]
    fn set_replaces_and_preserves_unknown_oauth_fields() {
        let temp = TempStore::new("replace");
        let store = temp.store();

        // A Credential minted by a newer version carries an unmodeled field.
        let wire = r#"{"type":"oauth","access_token":"a","refresh_token":"r","expires_at":42,"scope":"read"}"#;
        let cred: Credential = serde_json::from_str(wire).unwrap();
        store.set("anthropic", cred).unwrap();

        let back = store.get("anthropic").unwrap().unwrap();
        assert_eq!(back.as_oauth().unwrap().extra["scope"], "read");

        // A later set overwrites the entry.
        store
            .set("anthropic", Credential::api_key("sk-new"))
            .unwrap();
        assert_eq!(
            store.get("anthropic").unwrap().unwrap().as_api_key(),
            Some("sk-new")
        );
    }
}

#[cfg(all(
    test,
    feature = "token-store-file",
    feature = "anthropic",
    feature = "test-utils"
))]
mod refresh_coordination_tests {
    use super::file_test_support::TempStore;
    use super::*;
    use crate::credential::OAuthTokens;
    use crate::http::MockHttpClient;
    use crate::providers::anthropic::oauth::AnthropicOAuth;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    const TOKEN_BODY: &str = r#"{"token_type":"Bearer","access_token":"new-access","refresh_token":"new-refresh","expires_in":3600}"#;

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    #[tokio::test]
    async fn stale_oauth_is_refreshed_once_then_reused() {
        let temp = TempStore::new("refresh-once");
        let store = temp.store();
        let now = now_secs();
        // Stored token expires now: inside any positive window, so stale.
        store
            .set(
                "anthropic",
                Credential::oauth(OAuthTokens::new(
                    "old-acc",
                    "old-ref",
                    Some(now),
                )),
            )
            .unwrap();

        // The transport hands back exactly one refreshed token. A second call
        // to the endpoint would panic (no queued response), so if the refresh
        // is not deduplicated the test fails loudly.
        let http = Arc::new(MockHttpClient::with_response(200, TOKEN_BODY));
        let flow = AnthropicOAuth::new(http.clone());

        let renewed = store
            .refresh_if_stale_at("anthropic", &flow, 300, now)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(renewed.as_oauth().unwrap().access_token, "new-access");
        assert_eq!(http.requests().len(), 1);

        // The renewed token is now on disk.
        let on_disk = store.get("anthropic").unwrap().unwrap();
        assert_eq!(on_disk.as_oauth().unwrap().access_token, "new-access");

        // A later caller sees the fresh token on its first (shared-lock) read
        // and returns it without refreshing — no further request hits the
        // transport. (The under-lock re-check, which only fires when the token
        // still looks stale at the shared read, is covered by the concurrent
        // `racing_callers_*` test below.)
        let again = store
            .refresh_if_stale_at("anthropic", &flow, 300, now)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(again.as_oauth().unwrap().access_token, "new-access");
        assert_eq!(http.requests().len(), 1);
    }

    #[tokio::test]
    async fn fresh_oauth_is_not_refreshed() {
        let temp = TempStore::new("fresh");
        let store = temp.store();
        let now = now_secs();
        store
            .set(
                "anthropic",
                Credential::oauth(OAuthTokens::new(
                    "acc",
                    "ref",
                    Some(now + 100_000),
                )),
            )
            .unwrap();

        // A mock with no queued response panics if refresh is attempted.
        let http = Arc::new(MockHttpClient::new());
        let flow = AnthropicOAuth::new(http.clone());

        let result = store
            .refresh_if_stale_at("anthropic", &flow, 300, now)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.as_oauth().unwrap().access_token, "acc");
        assert!(http.requests().is_empty());
    }

    #[tokio::test]
    async fn non_oauth_credential_is_never_refreshed() {
        let temp = TempStore::new("apikey");
        let store = temp.store();
        store
            .set("anthropic", Credential::api_key("sk-live"))
            .unwrap();

        let http = Arc::new(MockHttpClient::new());
        let flow = AnthropicOAuth::new(http.clone());

        let result = store
            .refresh_if_stale_at("anthropic", &flow, 300, now_secs())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(result.as_api_key(), Some("sk-live"));
        assert!(http.requests().is_empty());
    }

    #[tokio::test]
    async fn absent_provider_yields_none() {
        let temp = TempStore::new("absent");
        let store = temp.store();
        let http = Arc::new(MockHttpClient::new());
        let flow = AnthropicOAuth::new(http);
        let result = store
            .refresh_if_stale_at("anthropic", &flow, 300, now_secs())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    /// A refresher that mints one fresh token, holding the exclusive lock across
    /// a deliberate stall so the second racing caller is forced to wait for the
    /// lock and re-read under it. It counts its calls so the test can assert the
    /// refresh is not duplicated.
    struct SlowRefresher {
        calls: std::sync::atomic::AtomicUsize,
        fresh_expiry: u64,
    }

    #[async_trait::async_trait]
    impl Refresh for SlowRefresher {
        async fn refresh(
            &self,
            _refresh_token: &str,
        ) -> Result<Credential, Error> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Hold the exclusive lock long enough that the loser is parked
            // waiting on it before this write lands.
            std::thread::sleep(std::time::Duration::from_millis(200));
            Ok(Credential::oauth(OAuthTokens::new(
                "new-access",
                "new-refresh",
                Some(self.fresh_expiry),
            )))
        }
    }

    // Two callers race on the same stale token. The file lock serializes them:
    // the winner refreshes once; the loser, blocked on the lock, re-reads the
    // now-fresh token under it and skips the redundant refresh. Exercises the
    // double-checked under-lock path and lock contention that the sequential
    // tests cannot reach.
    #[test]
    fn racing_callers_refresh_once_via_the_under_lock_recheck() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier};

        let temp = TempStore::new("race");
        let store = temp.store();
        let now = now_secs();
        // Stored token expires now: stale within any positive window.
        store
            .set(
                "anthropic",
                Credential::oauth(OAuthTokens::new(
                    "old",
                    "old-ref",
                    Some(now),
                )),
            )
            .unwrap();

        let refresher = Arc::new(SlowRefresher {
            calls: AtomicUsize::new(0),
            fresh_expiry: now + 100_000,
        });
        // Release both threads together so both complete their shared-lock read
        // (seeing the stale token) before either takes the exclusive lock.
        let gate = Arc::new(Barrier::new(2));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let store = store.clone();
                let refresher = Arc::clone(&refresher);
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || {
                    let rt = tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                        .unwrap();
                    gate.wait();
                    rt.block_on(store.refresh_if_stale(
                        "anthropic",
                        &*refresher,
                        300,
                    ))
                    .unwrap()
                })
            })
            .collect();

        let results: Vec<_> =
            handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Exactly one refresh happened despite two racing callers.
        assert_eq!(refresher.calls.load(Ordering::SeqCst), 1);
        // Both callers end up holding the renewed token.
        for cred in &results {
            let tokens = cred.as_ref().unwrap().as_oauth().unwrap();
            assert_eq!(tokens.access_token, "new-access");
        }
    }
}
