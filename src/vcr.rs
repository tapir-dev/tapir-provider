// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! A record/replay [`HttpClient`] test layer backed by on-disk cassettes.
//!
//! [`VcrClient`] wraps another [`HttpClient`] and plugs into the same transport
//! seam every Provider talks through, so realistic request-to-stream and
//! OAuth-refresh behavior can be exercised against recorded traffic instead of
//! hand-built doubles. A cassette stores each interaction's request and its
//! response as an ordered list of body chunks, so a streamed SSE completion
//! replays chunk-by-chunk exactly as it was recorded.
//!
//! # Modes
//!
//! - [`VcrMode::Record`] forwards every request to the wrapped transport and
//!   appends the (redacted) interaction to the cassette. It needs a real
//!   Credential, since it actually talks to the network.
//! - [`VcrMode::Replay`] serves responses from the cassette and never touches
//!   the wrapped transport, so playback runs with a dummy Credential.
//! - [`VcrMode::Auto`] replays when the cassette file is present and records
//!   when it is missing, resolved once at construction from the file's presence.
//!
//! # Redaction
//!
//! A [`Redactor`] strips secrets from headers (`Authorization`, `x-api-key`)
//! and sensitive JSON body fields (`access_token`, `code_verifier`, ...) both
//! when recording (so those secrets never land on disk) and before matching a
//! live request against a recorded one (so a redacted recording still matches a
//! request that carries the real secret). It does not touch the request URL, so
//! a Provider that carries a secret in a query string should redact it itself or
//! keep it in the body; the Providers here send their secrets in headers and
//! bodies.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, ErrorKind};
use crate::http::{ByteStream, HttpClient, HttpRequest, HttpResponse, Method};

/// The placeholder a redacted header value or body field is replaced with.
const REDACTED: &str = "<redacted>";

/// Which of record or replay a [`VcrClient`] performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum VcrMode {
    /// Forward to the wrapped transport and append each interaction to the
    /// cassette. Requires a real Credential.
    Record,
    /// Serve from the cassette and never touch the wrapped transport. Runs with
    /// a dummy Credential.
    Replay,
    /// Replay when the cassette file exists, record when it does not. Resolved
    /// once, at construction, from the file's presence.
    Auto,
}

/// Redacts secrets from requests and responses before they are stored, and from
/// a live request before it is matched against a recording.
///
/// Header redaction is by name, case-insensitively. Body redaction parses the
/// body as JSON and replaces every field whose key matches — at any depth — with
/// a `<redacted>` placeholder; a body that is not JSON is left untouched.
/// Redacting on both the record and the match side keeps a recording free of
/// secrets while still letting it match a live request carrying the real values.
#[derive(Debug, Clone)]
pub struct Redactor {
    /// Header names to redact, stored lowercase for case-insensitive matching.
    headers: Vec<String>,
    /// JSON body field names to redact wherever they appear.
    body_fields: Vec<String>,
}

impl Default for Redactor {
    /// The default redactor: the auth-bearing headers and the OAuth/token body
    /// fields this crate's Providers send and receive.
    fn default() -> Self {
        Self {
            headers: [
                "authorization",
                "x-api-key",
                "cookie",
                "set-cookie",
                "proxy-authorization",
            ]
            .iter()
            .map(|h| (*h).to_owned())
            .collect(),
            body_fields: [
                "access_token",
                "refresh_token",
                "client_secret",
                "code",
                "code_verifier",
                "state",
                "password",
            ]
            .iter()
            .map(|f| (*f).to_owned())
            .collect(),
        }
    }
}

impl Redactor {
    /// A redactor with the default header and body field sets.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Also redact the header named `name` (matched case-insensitively).
    #[must_use]
    pub fn redact_header(mut self, name: impl Into<String>) -> Self {
        self.headers.push(name.into().to_ascii_lowercase());
        self
    }

    /// Also redact the JSON body field named `field` wherever it appears.
    #[must_use]
    pub fn redact_field(mut self, field: impl Into<String>) -> Self {
        self.body_fields.push(field.into());
        self
    }

    /// Redacted copies of `headers`, replacing any redacted name's value.
    fn headers(&self, headers: &[(String, String)]) -> Vec<(String, String)> {
        headers
            .iter()
            .map(|(name, value)| {
                if self.is_redacted_header(name) {
                    (name.clone(), REDACTED.to_owned())
                } else {
                    (name.clone(), value.clone())
                }
            })
            .collect()
    }

    /// Whether `name` names a header to redact.
    fn is_redacted_header(&self, name: &str) -> bool {
        self.headers.iter().any(|h| h.eq_ignore_ascii_case(name))
    }

    /// A redacted, canonical string form of `body`, or `None` for no body.
    ///
    /// A JSON body has its redacted fields stripped and is re-serialized to a
    /// canonical form, so two equal payloads compare equal regardless of the
    /// key order they arrived in. A non-JSON body is kept as its lossy UTF-8.
    fn body(&self, body: Option<&[u8]>) -> Option<String> {
        body.map(|bytes| self.redact_json_or_text(bytes))
    }

    /// A redacted string form of one response body chunk.
    ///
    /// A JSON chunk (such as a token response) has its secret fields stripped; a
    /// non-JSON chunk (such as an SSE frame) is kept verbatim so a stream still
    /// replays its exact framing.
    fn chunk(&self, chunk: &[u8]) -> String {
        self.redact_json_or_text(chunk)
    }

    /// Redact `bytes` as JSON when they parse, else keep them as lossy UTF-8.
    ///
    /// A JSON payload has its redacted fields stripped and is re-serialized to a
    /// canonical (key-ordered) form, so two equal payloads compare equal
    /// whatever order their keys arrived in. Anything that is not JSON is kept
    /// verbatim, so a stream frame replays with its exact bytes.
    fn redact_json_or_text(&self, bytes: &[u8]) -> String {
        match serde_json::from_slice::<Value>(bytes) {
            Ok(mut value) => {
                self.redact_value(&mut value);
                // A value we just parsed always re-serializes, but fall back to
                // the lossy text rather than panic.
                serde_json::to_string(&value).unwrap_or_else(|_| {
                    String::from_utf8_lossy(bytes).into_owned()
                })
            }
            Err(_) => String::from_utf8_lossy(bytes).into_owned(),
        }
    }

    /// Recursively replace every redacted field within `value` with [`REDACTED`].
    fn redact_value(&self, value: &mut Value) {
        match value {
            Value::Object(map) => {
                for (key, entry) in map.iter_mut() {
                    if self
                        .body_fields
                        .iter()
                        .any(|f| f.eq_ignore_ascii_case(key))
                    {
                        *entry = Value::String(REDACTED.to_owned());
                    } else {
                        self.redact_value(entry);
                    }
                }
            }
            Value::Array(items) => {
                items.iter_mut().for_each(|item| self.redact_value(item));
            }
            _ => {}
        }
    }
}

/// A request as stored in a cassette, already redacted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct RecordedRequest {
    /// The HTTP method, as `"GET"` or `"POST"`.
    method: String,
    /// The fully-qualified request URL.
    url: String,
    /// Redacted header name/value pairs, kept for inspection only.
    headers: Vec<(String, String)>,
    /// The redacted, canonical request body, if any.
    body: Option<String>,
}

/// A response as stored in a cassette, already redacted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct RecordedResponse {
    /// The HTTP status code.
    status: u16,
    /// Redacted response header name/value pairs.
    headers: Vec<(String, String)>,
    /// The response body as ordered chunks, so a stream replays chunk-by-chunk.
    /// A non-streaming response is stored as a single chunk.
    chunks: Vec<String>,
}

/// One recorded request/response pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct Interaction {
    /// The request that was sent.
    request: RecordedRequest,
    /// The response that came back.
    response: RecordedResponse,
}

/// The on-disk cassette: an ordered list of interactions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct Cassette {
    /// The interactions, replayed in order.
    interactions: Vec<Interaction>,
}

/// The mutable state behind a [`VcrClient`]: the interactions and, on replay,
/// how far through them playback has reached.
#[derive(Debug, Default)]
struct State {
    /// The interactions loaded (replay) or accumulated (record).
    interactions: Vec<Interaction>,
    /// The next interaction to replay; unused while recording.
    cursor: usize,
}

/// A record/replay [`HttpClient`] backed by an on-disk cassette.
///
/// Wrap any transport `H`; in [`VcrMode::Record`] requests are forwarded to it
/// and the traffic captured, and in [`VcrMode::Replay`] it is never contacted.
/// See the [module docs](self) for the mode and redaction rules.
#[derive(Debug)]
pub struct VcrClient<H> {
    /// The wrapped transport, contacted only while recording.
    inner: H,
    /// The cassette file interactions are read from and written to.
    path: PathBuf,
    /// Whether this client records (`true`) or replays (`false`), resolved from
    /// the [`VcrMode`] and, for [`VcrMode::Auto`], the cassette's presence.
    recording: bool,
    /// Strips secrets on record and neutralizes them before matching.
    redactor: Redactor,
    /// The interactions and the replay cursor.
    state: Mutex<State>,
}

impl<H: HttpClient> VcrClient<H> {
    /// Wrap `inner`, backing the cassette with the file at `path` in `mode`,
    /// using the default [`Redactor`].
    ///
    /// # Errors
    ///
    /// Surfaces a cassette that exists but cannot be read or parsed.
    pub fn new(
        inner: H,
        path: impl Into<PathBuf>,
        mode: VcrMode,
    ) -> Result<Self, Error> {
        Self::with_redactor(inner, path, mode, Redactor::default())
    }

    /// [`new`](Self::new) with an explicit [`Redactor`].
    ///
    /// # Errors
    ///
    /// Surfaces a cassette that exists but cannot be read or parsed.
    pub fn with_redactor(
        inner: H,
        path: impl Into<PathBuf>,
        mode: VcrMode,
        redactor: Redactor,
    ) -> Result<Self, Error> {
        let path = path.into();
        let exists = path.exists();
        // Auto resolves to replay when the cassette is already there, else record.
        let recording = match mode {
            VcrMode::Record => true,
            VcrMode::Replay => false,
            VcrMode::Auto => !exists,
        };

        // Load an existing cassette when we will replay from it. A recording run
        // starts empty and overwrites whatever was there.
        let interactions = if recording {
            Vec::new()
        } else {
            load_cassette(&path)?.interactions
        };

        Ok(Self {
            inner,
            path,
            recording,
            redactor,
            state: Mutex::new(State {
                interactions,
                cursor: 0,
            }),
        })
    }

    /// The cassette file this client reads from and writes to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Whether this client is recording rather than replaying.
    #[must_use]
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// The redacted [`RecordedRequest`] for `request`, used both to store a
    /// recording and to match a live request against one.
    fn recorded_request(&self, request: &HttpRequest) -> RecordedRequest {
        RecordedRequest {
            method: method_str(request.method).to_owned(),
            url: request.url.clone(),
            headers: self.redactor.headers(&request.headers),
            body: self.redactor.body(request.body.as_deref()),
        }
    }

    /// Append `interaction` to the in-memory log and persist the whole cassette.
    fn record(&self, interaction: Interaction) -> Result<(), Error> {
        let cassette = {
            let mut state = self.lock();
            state.interactions.push(interaction);
            Cassette {
                interactions: state.interactions.clone(),
            }
        };
        write_cassette(&self.path, &cassette)
    }

    /// Take the next recorded response, verifying it answers `request`.
    ///
    /// Interactions replay in the order they were recorded; the next one's
    /// request must match `request` (same method, URL, and redacted body) or the
    /// cassette has drifted from what the code now sends.
    fn next_response(
        &self,
        request: &HttpRequest,
    ) -> Result<RecordedResponse, Error> {
        let want = self.recorded_request(request);
        let mut state = self.lock();
        let Some(interaction) = state.interactions.get(state.cursor).cloned()
        else {
            return Err(self.replay_error(format!(
                "no recorded interaction left for {} {}",
                want.method, want.url
            )));
        };
        if !request_matches(&interaction.request, &want) {
            return Err(self.replay_error(format!(
                "recorded interaction {} does not match {} {}",
                state.cursor, want.method, want.url
            )));
        }
        state.cursor += 1;
        Ok(interaction.response)
    }

    /// Lock the state, treating a poisoned mutex as a programming error.
    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().expect("VCR state mutex poisoned")
    }

    /// A replay-time [`ErrorKind::Other`] error naming the cassette.
    fn replay_error(&self, message: String) -> Error {
        Error::new(
            ErrorKind::Other,
            format!("VCR replay from {}: {message}", self.path.display()),
        )
    }
}

#[async_trait]
impl<H: HttpClient> HttpClient for VcrClient<H> {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, Error> {
        if self.recording {
            // Forward to the real transport, then store a redacted copy while
            // returning the untouched response so the caller can act on it.
            let response = self.inner.send(request.clone()).await?;
            let recorded = RecordedResponse {
                status: response.status,
                headers: self.redactor.headers(&response.headers),
                chunks: vec![self.redactor.chunk(&response.body)],
            };
            self.record(Interaction {
                request: self.recorded_request(&request),
                response: recorded,
            })?;
            return Ok(response);
        }

        let recorded = self.next_response(&request)?;
        Ok(HttpResponse {
            status: recorded.status,
            headers: recorded.headers,
            body: recorded.chunks.concat().into_bytes(),
        })
    }

    async fn send_stream(
        &self,
        request: HttpRequest,
    ) -> Result<ByteStream, Error> {
        if self.recording {
            // Drain the upstream stream so it can be stored, then hand the same
            // chunks back to the caller. Buffering the whole body is fine here:
            // cassettes cover test-sized responses.
            let mut stream = self.inner.send_stream(request.clone()).await?;
            let mut chunks: Vec<Vec<u8>> = Vec::new();
            while let Some(chunk) = stream.next().await {
                chunks.push(chunk?);
            }
            let recorded = RecordedResponse {
                // A streamed body only reaches here on a 2xx; a non-2xx status
                // is surfaced as an error before any stream exists.
                status: 200,
                headers: Vec::new(),
                chunks: chunks
                    .iter()
                    .map(|chunk| self.redactor.chunk(chunk))
                    .collect(),
            };
            self.record(Interaction {
                request: self.recorded_request(&request),
                response: recorded,
            })?;
            return Ok(chunks_stream(chunks));
        }

        let recorded = self.next_response(&request)?;
        // Mirror a real streaming transport: classify a non-2xx up front rather
        // than streaming the error body.
        if !(200..300).contains(&recorded.status) {
            return Err(Error::from_status(
                recorded.status,
                recorded.chunks.concat(),
            ));
        }
        let bytes = recorded
            .chunks
            .into_iter()
            .map(String::into_bytes)
            .collect::<Vec<_>>();
        Ok(chunks_stream(bytes))
    }
}

/// The wire spelling of an [`HttpRequest`] method.
fn method_str(method: Method) -> &'static str {
    match method {
        Method::Get => "GET",
        Method::Post => "POST",
    }
}

/// Whether a recorded request answers `live`: same method, URL, and redacted
/// body. Headers are stored for inspection but not matched, since volatile
/// headers (dates, request ids) would make matching brittle.
fn request_matches(recorded: &RecordedRequest, live: &RecordedRequest) -> bool {
    recorded.method == live.method
        && recorded.url == live.url
        && recorded.body == live.body
}

/// A boxed byte stream that yields each of `chunks` in turn.
fn chunks_stream(chunks: Vec<Vec<u8>>) -> ByteStream {
    Box::pin(futures_util::stream::iter(chunks.into_iter().map(Ok)))
}

/// Read and parse the cassette at `path`, or an empty one if it is absent.
fn load_cassette(path: &Path) -> Result<Cassette, Error> {
    match std::fs::read_to_string(path) {
        Ok(contents) if contents.trim().is_empty() => Ok(Cassette::default()),
        Ok(contents) => serde_json::from_str(&contents).map_err(|err| {
            Error::new(
                ErrorKind::Decode,
                format!("VCR cassette {} is corrupt: {err}", path.display()),
            )
            .with_source(err)
        }),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Ok(Cassette::default())
        }
        Err(err) => Err(cassette_io_error(path, "read", err)),
    }
}

/// Serialize `cassette` as pretty JSON and overwrite `path`, creating any
/// missing parent directories.
fn write_cassette(path: &Path, cassette: &Cassette) -> Result<(), Error> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|err| {
            cassette_io_error(parent, "create directory for", err)
        })?;
    }
    let serialized =
        serde_json::to_string_pretty(cassette).map_err(Error::serialize)?;
    std::fs::write(path, serialized)
        .map_err(|err| cassette_io_error(path, "write", err))
}

/// Wrap a cassette filesystem error with the path and failed operation.
fn cassette_io_error(path: &Path, op: &str, err: std::io::Error) -> Error {
    Error::new(
        ErrorKind::Other,
        format!("VCR cassette {op} failed for {}: {err}", path.display()),
    )
    .with_source(err)
}

/// A unique temp cassette path shared by the test modules, removed on drop so
/// parallel tests do not collide and nothing leaks into the temp dir.
#[cfg(test)]
mod test_support {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// A temp file path that removes its file when dropped.
    pub(super) struct TempCassette {
        /// The path the cassette is written to.
        pub(super) path: PathBuf,
    }

    impl TempCassette {
        /// A fresh path unique to this call, tagged for readability.
        pub(super) fn new(tag: &str) -> Self {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let mut path = std::env::temp_dir();
            path.push(format!(
                "tapir-vcr-{tag}-{}-{n}-{nanos}.json",
                std::process::id()
            ));
            Self { path }
        }
    }

    impl Drop for TempCassette {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::TempCassette;
    use super::*;
    use crate::http::MockHttpClient;

    fn post(url: &str, body: &str) -> HttpRequest {
        HttpRequest::new(Method::Post, url)
            .header("authorization", "Bearer super-secret")
            .body(body.as_bytes().to_vec())
    }

    #[test]
    fn redactor_strips_headers_and_nested_body_fields() {
        let redactor = Redactor::new();
        let headers = redactor.headers(&[
            ("Authorization".to_owned(), "Bearer abc".to_owned()),
            ("content-type".to_owned(), "application/json".to_owned()),
        ]);
        assert_eq!(
            headers[0],
            ("Authorization".to_owned(), REDACTED.to_owned())
        );
        assert_eq!(
            headers[1],
            ("content-type".to_owned(), "application/json".to_owned())
        );

        let body = redactor
            .body(Some(
                br#"{"grant_type":"refresh_token","refresh_token":"r","nested":{"code":"c"}}"#,
            ))
            .unwrap();
        let value: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["grant_type"], "refresh_token");
        assert_eq!(value["refresh_token"], REDACTED);
        assert_eq!(value["nested"]["code"], REDACTED);
    }

    #[test]
    fn redactor_leaves_non_json_bodies_alone() {
        let redactor = Redactor::new();
        let body = redactor.body(Some(b"event: ping\ndata: {}\n\n")).unwrap();
        assert_eq!(body, "event: ping\ndata: {}\n\n");
    }

    #[tokio::test]
    async fn records_then_replays_a_non_streaming_send() {
        let temp = TempCassette::new("send");

        // Record against a fake upstream: the recorder still exercises the
        // record path without a real network.
        let upstream = MockHttpClient::with_response(
            200,
            br#"{"ok":true,"access_token":"secret-abc"}"#.to_vec(),
        );
        let recorder =
            VcrClient::new(upstream, &temp.path, VcrMode::Record).unwrap();
        let response = recorder
            .send(post("https://api.test/v1/x", r#"{"q":"hi"}"#))
            .await
            .unwrap();
        // The caller sees the real, unredacted body while recording.
        assert!(response.body_string().contains("secret-abc"));

        // The stored cassette has the secret redacted.
        let on_disk = std::fs::read_to_string(&temp.path).unwrap();
        assert!(!on_disk.contains("secret-abc"));
        assert!(!on_disk.contains("super-secret"));
        assert!(on_disk.contains(REDACTED));

        // Replay: the wrapped transport has no queued response, so it panics if
        // touched. A dummy Credential would do here too.
        let replayer =
            VcrClient::new(MockHttpClient::new(), &temp.path, VcrMode::Replay)
                .unwrap();
        let replayed = replayer
            .send(post("https://api.test/v1/x", r#"{"q":"hi"}"#))
            .await
            .unwrap();
        assert_eq!(replayed.status, 200);
        // The replayed body carries the redacted placeholder, not the secret.
        assert!(replayed.body_string().contains(REDACTED));
        assert!(!replayed.body_string().contains("secret-abc"));
    }

    #[tokio::test]
    async fn records_then_replays_a_stream_chunk_by_chunk() {
        let temp = TempCassette::new("stream");
        let chunks = vec![
            b"event: message_start\ndata: {\"a\":1}\n\n".to_vec(),
            b"event: content_block_delta\nda".to_vec(),
            b"ta: {\"b\":2}\n\n".to_vec(),
        ];

        let upstream = MockHttpClient::with_stream(chunks.clone());
        let recorder =
            VcrClient::new(upstream, &temp.path, VcrMode::Record).unwrap();
        let mut stream = recorder
            .send_stream(post("https://api.test/v1/stream", "{}"))
            .await
            .unwrap();
        let mut recorded_out = Vec::new();
        while let Some(chunk) = stream.next().await {
            recorded_out.push(chunk.unwrap());
        }
        assert_eq!(recorded_out, chunks);

        // Replay preserves the exact chunk framing, including the event split
        // across the boundary between chunk two and three.
        let replayer =
            VcrClient::new(MockHttpClient::new(), &temp.path, VcrMode::Replay)
                .unwrap();
        let mut stream = replayer
            .send_stream(post("https://api.test/v1/stream", "{}"))
            .await
            .unwrap();
        let mut replayed = Vec::new();
        while let Some(chunk) = stream.next().await {
            replayed.push(chunk.unwrap());
        }
        assert_eq!(replayed, chunks);
    }

    #[tokio::test]
    async fn auto_records_when_missing_then_replays_when_present() {
        let temp = TempCassette::new("auto");
        assert!(!temp.path.exists());

        // Missing cassette: Auto records.
        let recorder = VcrClient::new(
            MockHttpClient::with_response(200, b"recorded".to_vec()),
            &temp.path,
            VcrMode::Auto,
        )
        .unwrap();
        assert!(recorder.is_recording());
        recorder
            .send(post("https://api.test/auto", "{}"))
            .await
            .unwrap();
        assert!(temp.path.exists());

        // Present cassette: Auto replays and never contacts the transport.
        let replayer =
            VcrClient::new(MockHttpClient::new(), &temp.path, VcrMode::Auto)
                .unwrap();
        assert!(!replayer.is_recording());
        let response = replayer
            .send(post("https://api.test/auto", "{}"))
            .await
            .unwrap();
        assert_eq!(response.body_string(), "recorded");
    }

    #[tokio::test]
    async fn replay_without_a_cassette_errors() {
        let temp = TempCassette::new("empty");
        let replayer =
            VcrClient::new(MockHttpClient::new(), &temp.path, VcrMode::Replay)
                .unwrap();
        let err = replayer
            .send(post("https://api.test/missing", "{}"))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Other);
        assert!(err.message().contains("no recorded interaction"));
    }

    #[tokio::test]
    async fn replay_mismatch_errors() {
        let temp = TempCassette::new("mismatch");
        let recorder = VcrClient::new(
            MockHttpClient::with_response(200, b"body".to_vec()),
            &temp.path,
            VcrMode::Record,
        )
        .unwrap();
        recorder
            .send(post("https://api.test/recorded", r#"{"a":1}"#))
            .await
            .unwrap();

        let replayer =
            VcrClient::new(MockHttpClient::new(), &temp.path, VcrMode::Replay)
                .unwrap();
        // A different URL does not match the single recorded interaction.
        let err = replayer
            .send(post("https://api.test/other", r#"{"a":1}"#))
            .await
            .unwrap_err();
        assert!(err.message().contains("does not match"));
    }

    #[tokio::test]
    async fn replay_matches_regardless_of_body_key_order() {
        let temp = TempCassette::new("keyorder");
        let recorder = VcrClient::new(
            MockHttpClient::with_response(200, b"ok".to_vec()),
            &temp.path,
            VcrMode::Record,
        )
        .unwrap();
        recorder
            .send(post("https://api.test/order", r#"{"a":1,"b":2}"#))
            .await
            .unwrap();

        let replayer =
            VcrClient::new(MockHttpClient::new(), &temp.path, VcrMode::Replay)
                .unwrap();
        // Same fields, different order: the canonical redacted form matches.
        let response = replayer
            .send(post("https://api.test/order", r#"{"b":2,"a":1}"#))
            .await
            .unwrap();
        assert_eq!(response.body_string(), "ok");
    }

    #[tokio::test]
    async fn a_custom_redactor_strips_extra_headers_and_fields() {
        let temp = TempCassette::new("custom-redactor");
        let redactor = Redactor::new()
            .redact_header("x-tenant")
            .redact_field("session_id");
        let request = HttpRequest::new(Method::Post, "https://api.test/custom")
            .header("x-tenant", "acme-corp")
            .body(br#"{"session_id":"sess-123","keep":"me"}"#.to_vec());

        let recorder = VcrClient::with_redactor(
            MockHttpClient::with_response(200, b"ok".to_vec()),
            &temp.path,
            VcrMode::Record,
            redactor,
        )
        .unwrap();
        recorder.send(request).await.unwrap();

        // The extra header and field are redacted on disk; the rest survives.
        let on_disk = std::fs::read_to_string(&temp.path).unwrap();
        assert!(!on_disk.contains("acme-corp"));
        assert!(!on_disk.contains("sess-123"));
        assert!(on_disk.contains("keep"));
    }
}

#[cfg(all(test, feature = "anthropic"))]
mod anthropic_tests {
    use super::test_support::TempCassette;
    use super::*;
    use crate::credential::Credential;
    use crate::http::MockHttpClient;
    use crate::message::Message;
    use crate::provider::Provider;
    use crate::providers::AnthropicProvider;
    use crate::providers::anthropic::oauth::AnthropicOAuth;
    use crate::request::CompletionRequest;
    use crate::stream::{StreamAccumulator, StreamEvent};
    use futures_util::StreamExt;
    use std::sync::Arc;

    /// The SSE frames of a minimal Anthropic streamed completion, split so one
    /// event straddles a chunk boundary.
    fn anthropic_stream_chunks() -> Vec<Vec<u8>> {
        vec![
            b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n".to_vec(),
            b"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n".to_vec(),
            b"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"del".to_vec(),
            b"ta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n".to_vec(),
            b"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n".to_vec(),
            b"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_vec(),
        ]
    }

    #[tokio::test]
    async fn streaming_completion_replays_from_a_cassette() {
        let temp = TempCassette::new("stream");

        // Record a streamed completion against a fake upstream stream.
        let upstream = MockHttpClient::with_stream(anthropic_stream_chunks());
        let recorder =
            VcrClient::new(upstream, &temp.path, VcrMode::Record).unwrap();
        let provider =
            AnthropicProvider::builder(recorder, "claude-3-5-sonnet")
                .credential(Credential::api_key("real-key"))
                .build()
                .unwrap();
        let mut stream = provider
            .complete_stream(CompletionRequest::new(vec![Message::user("hi")]))
            .await
            .unwrap();
        while stream.next().await.is_some() {}

        // Replay with a dummy key and no network: the wrapped transport has no
        // queued stream, so it panics if contacted.
        let vcr =
            VcrClient::new(MockHttpClient::new(), &temp.path, VcrMode::Replay)
                .unwrap();
        let provider = AnthropicProvider::builder(vcr, "claude-3-5-sonnet")
            .credential(Credential::api_key("dummy-key"))
            .build()
            .unwrap();
        let mut stream = provider
            .complete_stream(CompletionRequest::new(vec![Message::user("hi")]))
            .await
            .unwrap();
        let mut accumulator = StreamAccumulator::new();
        let mut saw_done = false;
        while let Some(event) = stream.next().await {
            let event = event.unwrap();
            if matches!(event, StreamEvent::Done { .. }) {
                saw_done = true;
            }
            accumulator.push(&event);
        }
        assert!(saw_done, "the replayed stream ended in a Done event");
        let completion = accumulator.finish();
        assert_eq!(completion.text, "Hi");
    }

    /// Record one OAuth token exchange against a fake upstream, so a replay can
    /// read it back from the cassette.
    async fn record_oauth(temp: &TempCassette, status: u16, body: &str) {
        let upstream = Arc::new(MockHttpClient::with_response(
            status,
            body.as_bytes().to_vec(),
        ));
        let vcr = Arc::new(
            VcrClient::new(upstream, &temp.path, VcrMode::Record).unwrap(),
        );
        let flow = AnthropicOAuth::new(vcr);
        // Ignore the outcome; recording captures the interaction either way.
        let _ = flow.refresh("old-refresh-token").await;
    }

    #[tokio::test]
    async fn oauth_refresh_replays_success() {
        let temp = TempCassette::new("oauth-ok");
        record_oauth(
            &temp,
            200,
            r#"{"token_type":"Bearer","access_token":"fresh-access","refresh_token":"fresh-refresh","expires_in":3600}"#,
        )
        .await;

        // The stored token response is redacted on disk.
        let on_disk = std::fs::read_to_string(&temp.path).unwrap();
        assert!(!on_disk.contains("fresh-access"));
        assert!(!on_disk.contains("old-refresh-token"));

        // Replay with a dummy transport that would panic if contacted.
        let vcr = Arc::new(
            VcrClient::new(MockHttpClient::new(), &temp.path, VcrMode::Replay)
                .unwrap(),
        );
        let flow = AnthropicOAuth::new(vcr);
        let credential = flow.refresh("old-refresh-token").await.unwrap();
        // The access token replays as the redacted placeholder.
        let tokens = credential.as_oauth().unwrap();
        assert_eq!(tokens.access_token, REDACTED);
    }

    #[tokio::test]
    async fn oauth_refresh_replays_failure() {
        let temp = TempCassette::new("oauth-fail");
        record_oauth(&temp, 400, r#"{"error":"invalid_grant"}"#).await;

        let vcr = Arc::new(
            VcrClient::new(MockHttpClient::new(), &temp.path, VcrMode::Replay)
                .unwrap(),
        );
        let flow = AnthropicOAuth::new(vcr);
        let err = flow.refresh("old-refresh-token").await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidRequest);
        assert_eq!(err.status(), Some(400));
    }
}
