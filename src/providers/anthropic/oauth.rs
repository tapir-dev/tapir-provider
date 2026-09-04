// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The Anthropic OAuth flow: PKCE login, code exchange, and token refresh.
//!
//! A caller authenticates without an API key by driving three steps over the
//! injected [`HttpClient`]:
//!
//! 1. [`AnthropicOAuth::start`] mints a PKCE verifier and an independent CSRF
//!    state, and returns the [`OAuthLogin`] whose `authorize_url` the user
//!    opens in a browser.
//! 2. The user authorizes and the authorization code comes back — either
//!    captured off a localhost redirect ([`capture_localhost`]) or pasted by
//!    hand ([`AuthorizationCode::parse_pasted`]).
//! 3. [`AnthropicOAuth::exchange`] trades that code for an OAuth
//!    [`Credential`]; [`AnthropicOAuth::refresh`] renews an expiring one.
//!
//! The challenge is S256 (`base64url(sha256(verifier))`) and the state is an
//! independent random value, both hand-rolled to keep the dependency set
//! minimal. Only the OS entropy source and the token exchange touch the outside
//! world; everything else is pure and unit-tested.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use crate::credential::{Credential, OAuthTokens};
use crate::error::{Error, ErrorKind};
use crate::http::{HttpClient, HttpRequest, Method};

/// OAuth client id this crate authenticates as. Overridable via
/// [`CLIENT_ID_ENV`] so a caller can hedge against it changing.
const DEFAULT_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
/// Environment variable overriding [`DEFAULT_CLIENT_ID`].
const CLIENT_ID_ENV: &str = "ANTHROPIC_OAUTH_CLIENT_ID";
/// Token endpoint for code exchange and refresh. Overridable via
/// [`TOKEN_URL_ENV`].
const DEFAULT_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
/// Environment variable overriding [`DEFAULT_TOKEN_URL`].
const TOKEN_URL_ENV: &str = "ANTHROPIC_OAUTH_TOKEN_URL";
/// Authorize endpoint for the subscription flow.
const AUTHORIZE_SUBSCRIPTION: &str = "https://claude.ai/oauth/authorize";
/// Authorize endpoint for the console flow.
const AUTHORIZE_CONSOLE: &str = "https://console.anthropic.com/oauth/authorize";
/// Redirect URI for the manual code-paste flow.
const MANUAL_REDIRECT_URI: &str =
    "https://console.anthropic.com/oauth/code/callback";
/// Space-separated scopes requested by the flow.
const SCOPES: &str = "org:create_api_key user:profile user:inference";
/// Subtracted from a token's real lifetime so proactive refresh fires before
/// the access token is actually stale.
const EXPIRY_SAFETY_MARGIN_SECS: u64 = 300;

/// Which authorize endpoint the flow points the user at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OAuthMode {
    /// Authorize against the subscription (`claude.ai`) endpoint.
    Subscription,
    /// Authorize against the console (`console.anthropic.com`) endpoint.
    Console,
}

impl OAuthMode {
    /// The authorize endpoint for this mode.
    const fn authorize_base(self) -> &'static str {
        match self {
            Self::Subscription => AUTHORIZE_SUBSCRIPTION,
            Self::Console => AUTHORIZE_CONSOLE,
        }
    }
}

/// Where the authorization code is returned after the user authorizes.
///
/// This fixes the `redirect_uri` sent on both the authorize URL and the code
/// exchange; the two must match or the exchange is rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Redirect {
    /// The code is shown to the user to paste back by hand.
    Manual,
    /// The code is delivered to `http://localhost:{port}/callback`, captured by
    /// [`capture_localhost`].
    Localhost {
        /// The localhost port the redirect lands on.
        port: u16,
    },
}

impl Redirect {
    /// The `redirect_uri` this target registers with the authorize request.
    #[must_use]
    pub fn redirect_uri(&self) -> String {
        match self {
            Self::Manual => MANUAL_REDIRECT_URI.to_owned(),
            Self::Localhost { port } => {
                format!("http://localhost:{port}/callback")
            }
        }
    }
}

/// A started OAuth login: the URL to open plus the secrets that complete it.
///
/// `authorize_url` and `state` are the caller's to inspect; the PKCE verifier
/// and the resolved `redirect_uri` are held privately and replayed by
/// [`AnthropicOAuth::exchange`].
#[derive(Clone)]
pub struct OAuthLogin {
    /// The URL the user opens to authorize.
    pub authorize_url: String,
    /// The CSRF state echoed back on the redirect; compared on exchange.
    pub state: String,
    /// The PKCE code verifier, replayed on exchange.
    verifier: String,
    /// The redirect URI sent on authorize; must match on exchange.
    redirect_uri: String,
}

impl std::fmt::Debug for OAuthLogin {
    /// Redacts the PKCE verifier so it never leaks into logs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthLogin")
            .field("authorize_url", &self.authorize_url)
            .field("state", &self.state)
            .field("verifier", &"<redacted>")
            .field("redirect_uri", &self.redirect_uri)
            .finish()
    }
}

/// An authorization code returned by the authorize step, with the CSRF state
/// that came alongside it (when the transport supplied one).
#[derive(Clone, PartialEq, Eq)]
pub struct AuthorizationCode {
    /// The authorization code to exchange.
    pub code: String,
    /// The CSRF state echoed back, if present.
    pub state: Option<String>,
}

impl std::fmt::Debug for AuthorizationCode {
    /// Redacts the exchangeable code; the CSRF state is not secret.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthorizationCode")
            .field("code", &"<redacted>")
            .field("state", &self.state)
            .finish()
    }
}

impl AuthorizationCode {
    /// Parse a hand-pasted code.
    ///
    /// The manual flow presents the code as `code#state`; a bare code with no
    /// `#` is taken as the whole string with no state. Surrounding whitespace
    /// is trimmed.
    #[must_use]
    pub fn parse_pasted(input: &str) -> Self {
        let trimmed = input.trim();
        match trimmed.split_once('#') {
            Some((code, state)) => Self {
                code: code.to_owned(),
                state: Some(state.to_owned()),
            },
            None => Self {
                code: trimmed.to_owned(),
                state: None,
            },
        }
    }

    /// Parse the `code` and `state` from a redirect query string
    /// (`code=...&state=...`), percent-decoding each value.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidRequest`] if the query carries no `code`.
    pub fn from_callback_query(query: &str) -> Result<Self, Error> {
        let mut code = None;
        let mut state = None;
        for pair in query.split('&') {
            if pair.is_empty() {
                continue;
            }
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            match key {
                "code" => code = Some(percent_decode(value)),
                "state" => state = Some(percent_decode(value)),
                _ => {}
            }
        }
        let code = code.ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidRequest,
                "OAuth callback carried no authorization code",
            )
        })?;
        Ok(Self { code, state })
    }
}

/// Drives the Anthropic OAuth flow over an injected [`HttpClient`].
///
/// The client id and token URL are read once at construction, honoring the
/// `ANTHROPIC_OAUTH_CLIENT_ID` and `ANTHROPIC_OAUTH_TOKEN_URL` overrides, then
/// held for the flow's lifetime.
#[derive(Debug, Clone)]
pub struct AnthropicOAuth<H> {
    http: H,
    client_id: String,
    token_url: String,
}

impl<H: HttpClient> AnthropicOAuth<H> {
    /// Build the flow over `http`, reading the client id and token URL from the
    /// environment when overridden, else the baked-in defaults.
    #[must_use]
    pub fn new(http: H) -> Self {
        Self {
            http,
            client_id: std::env::var(CLIENT_ID_ENV)
                .unwrap_or_else(|_| DEFAULT_CLIENT_ID.to_owned()),
            token_url: std::env::var(TOKEN_URL_ENV)
                .unwrap_or_else(|_| DEFAULT_TOKEN_URL.to_owned()),
        }
    }

    /// Start a login: mint a fresh PKCE verifier and CSRF state and build the
    /// authorize URL the user opens.
    #[must_use]
    pub fn start(&self, mode: OAuthMode, redirect: &Redirect) -> OAuthLogin {
        self.build_login(mode, redirect, random_token(), random_token())
    }

    /// The deterministic core of [`start`](Self::start): builds the login from
    /// an explicit verifier and state, so the URL construction is testable.
    fn build_login(
        &self,
        mode: OAuthMode,
        redirect: &Redirect,
        verifier: String,
        state: String,
    ) -> OAuthLogin {
        let challenge = base64url_nopad(&sha256(verifier.as_bytes()));
        let redirect_uri = redirect.redirect_uri();
        let params = [
            ("code", "true"),
            ("client_id", self.client_id.as_str()),
            ("response_type", "code"),
            ("redirect_uri", redirect_uri.as_str()),
            ("scope", SCOPES),
            ("code_challenge", challenge.as_str()),
            ("code_challenge_method", "S256"),
            ("state", state.as_str()),
        ];
        let query = params
            .iter()
            .map(|(key, value)| format!("{key}={}", percent_encode(value)))
            .collect::<Vec<_>>()
            .join("&");
        let authorize_url = format!("{}?{}", mode.authorize_base(), query);
        OAuthLogin {
            authorize_url,
            state,
            verifier,
            redirect_uri,
        }
    }

    /// Exchange an authorization code for an OAuth [`Credential`].
    ///
    /// When the code carries a state it is checked against the login's state
    /// before any request is sent, so a mismatched (forged) redirect never
    /// reaches the token endpoint.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Authentication`] on a state mismatch or a token response
    /// with no refresh token; a transport or non-2xx error from the token
    /// endpoint; [`ErrorKind::Decode`] if the response body will not parse.
    pub async fn exchange(
        &self,
        login: &OAuthLogin,
        code: &AuthorizationCode,
    ) -> Result<Credential, Error> {
        if let Some(returned) = &code.state
            && returned != &login.state
        {
            return Err(Error::new(
                ErrorKind::Authentication,
                "OAuth state mismatch: possible CSRF, discarding the code",
            ));
        }

        // The token endpoint requires the CSRF state echoed back in the exchange
        // body; a request without it is rejected as an invalid request format.
        // Prefer the state returned with the code, falling back to the login's.
        let state = code.state.as_deref().unwrap_or(&login.state);
        let body = serde_json::json!({
            "grant_type": "authorization_code",
            "code": code.code,
            "state": state,
            "client_id": self.client_id,
            "redirect_uri": login.redirect_uri,
            "code_verifier": login.verifier,
        });
        let token = self.post_token(&body).await?;
        credential_from_token(token, None, now_unix())
    }

    /// Refresh an expiring access token, returning a renewed OAuth
    /// [`Credential`].
    ///
    /// A response that omits a fresh refresh token reuses the one passed in, so
    /// the returned Credential always carries a usable refresh token.
    ///
    /// # Errors
    ///
    /// A transport or non-2xx error from the token endpoint, or
    /// [`ErrorKind::Decode`] if the response body will not parse.
    pub async fn refresh(
        &self,
        refresh_token: &str,
    ) -> Result<Credential, Error> {
        let body = serde_json::json!({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        });
        let token = self.post_token(&body).await?;
        credential_from_token(token, Some(refresh_token.to_owned()), now_unix())
    }

    /// POST a JSON body to the token endpoint and parse the token response.
    async fn post_token(
        &self,
        body: &serde_json::Value,
    ) -> Result<TokenResponse, Error> {
        let payload = serde_json::to_vec(body).map_err(Error::serialize)?;
        let request = HttpRequest::new(Method::Post, &self.token_url)
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .body(payload);

        let response = self.http.send(request).await?;
        if !response.is_success() {
            return Err(Error::from_status(
                response.status,
                response.body_string(),
            ));
        }
        serde_json::from_slice(&response.body).map_err(Error::decode)
    }
}

/// Renews a stored OAuth token for the [`TokenStore`](crate::TokenStore)'s
/// proactive refresh, delegating to the inherent
/// [`refresh`](AnthropicOAuth::refresh).
#[async_trait::async_trait]
impl<H: HttpClient> crate::token_store::Refresh for AnthropicOAuth<H> {
    async fn refresh(&self, refresh_token: &str) -> Result<Credential, Error> {
        AnthropicOAuth::refresh(self, refresh_token).await
    }
}

/// Block on `port` until the browser redirect arrives, returning the captured
/// [`AuthorizationCode`].
///
/// Binds `127.0.0.1:port`, accepts one connection, parses the code and state
/// off the request line, replies with a short "you may close this window"
/// page, and returns. Pair the `port` with [`Redirect::Localhost`] so the
/// captured code exchanges against the same `redirect_uri`.
///
/// # Errors
///
/// [`ErrorKind::Transport`] if the port cannot be bound or the connection
/// fails; [`ErrorKind::InvalidRequest`] if the request carries no query or no
/// authorization code.
pub fn capture_localhost(port: u16) -> Result<AuthorizationCode, Error> {
    let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|err| {
        Error::new(
            ErrorKind::Transport,
            format!("could not bind localhost:{port}: {err}"),
        )
        .with_source(err)
    })?;
    let (mut stream, _) = listener.accept().map_err(|err| {
        Error::new(ErrorKind::Transport, err.to_string()).with_source(err)
    })?;

    let mut buf = [0u8; 4096];
    let read = stream.read(&mut buf).map_err(|err| {
        Error::new(ErrorKind::Transport, err.to_string()).with_source(err)
    })?;
    let request = String::from_utf8_lossy(&buf[..read]);
    let query = request_target_query(&request).ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidRequest,
            "OAuth callback request carried no query string",
        )
    })?;
    let code = AuthorizationCode::from_callback_query(query)?;

    let response = "HTTP/1.1 200 OK\r\n\
        Content-Type: text/html\r\n\
        Connection: close\r\n\r\n\
        <html><body>Authentication complete. You may close this window.\
        </body></html>";
    // Best-effort: the code is already captured whether or not this lands.
    let _ = stream.write_all(response.as_bytes());
    Ok(code)
}

/// Extract the query string from an HTTP request's start line
/// (`GET /callback?code=...&state=... HTTP/1.1`).
fn request_target_query(request: &str) -> Option<&str> {
    let start_line = request.lines().next()?;
    let target = start_line.split_whitespace().nth(1)?;
    target.split_once('?').map(|(_, query)| query)
}

/// The Anthropic token endpoint response.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    #[serde(default)]
    expires_in: Option<u64>,
}

impl std::fmt::Debug for TokenResponse {
    /// Redacts the tokens; keeps the non-secret lifetime visible.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenResponse")
            .field("access_token", &"<redacted>")
            .field(
                "refresh_token",
                &self.refresh_token.as_ref().map(|_| "<redacted>"),
            )
            .field("expires_in", &self.expires_in)
            .finish()
    }
}

/// Mint an OAuth [`Credential`] from a token response, baking in the expiry
/// safety margin and falling back to `fallback_refresh` when the response
/// omits a refresh token.
fn credential_from_token(
    token: TokenResponse,
    fallback_refresh: Option<String>,
    now_unix: u64,
) -> Result<Credential, Error> {
    let refresh_token =
        token.refresh_token.or(fallback_refresh).ok_or_else(|| {
            Error::new(
                ErrorKind::Authentication,
                "OAuth token response carried no refresh token",
            )
        })?;
    let expires_at = token.expires_in.map(|secs| {
        now_unix
            .saturating_add(secs)
            .saturating_sub(EXPIRY_SAFETY_MARGIN_SECS)
    });
    Ok(Credential::oauth(OAuthTokens::new(
        token.access_token,
        refresh_token,
        expires_at,
    )))
}

/// Current Unix time in seconds, or 0 if the clock predates the epoch.
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A random 43-character `base64url` token (32 bytes of OS entropy), suitable
/// for both a PKCE verifier and a CSRF state.
fn random_token() -> String {
    let mut bytes = [0u8; 32];
    getrandom::getrandom(&mut bytes)
        .expect("OS entropy source unavailable for OAuth token generation");
    base64url_nopad(&bytes)
}

/// Standard SHA-256 (FIPS 180-4). Hand-rolled to keep the dependency set
/// minimal; the PKCE challenge is the only caller.
#[allow(clippy::many_single_char_names)]
fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().enumerate().take(16) {
            let b = i * 4;
            *word = u32::from_be_bytes([
                block[b],
                block[b + 1],
                block[b + 2],
                block[b + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7)
                ^ w[i - 15].rotate_right(18)
                ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17)
                ^ w[i - 2].rotate_right(19)
                ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];

        for i in 0..64 {
            let s1 =
                e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 =
                a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// URL-safe base64 (RFC 4648 §5) with no padding.
fn base64url_nopad(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(chunk.get(1).copied().unwrap_or(0));
        let b2 = u32::from(chunk.get(2).copied().unwrap_or(0));
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(n & 0x3f) as usize] as char);
        }
    }
    out
}

/// Percent-encode a string for use as a URL query value, escaping everything
/// outside the RFC 3986 unreserved set.
fn percent_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for &byte in input.as_bytes() {
        match byte {
            b'A'..=b'Z'
            | b'a'..=b'z'
            | b'0'..=b'9'
            | b'-'
            | b'.'
            | b'_'
            | b'~' => out.push(byte as char),
            _ => {
                out.push('%');
                out.push(hex_digit(byte >> 4));
                out.push(hex_digit(byte & 0x0f));
            }
        }
    }
    out
}

/// Percent-decode a URL query value, treating `+` as a space.
fn percent_decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                match (hex_value(bytes[i + 1]), hex_value(bytes[i + 2])) {
                    (Some(hi), Some(lo)) => {
                        out.push(hi << 4 | lo);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The uppercase hex digit for a nibble (`0..=15`).
fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        _ => (b'A' + (nibble - 10)) as char,
    }
}

/// The numeric value of an ASCII hex digit, or `None` if it is not one.
fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use super::*;
    use crate::http::MockHttpClient;
    use std::sync::Arc;

    /// RFC 7636 Appendix B PKCE test vector.
    const RFC7636_VERIFIER: &str =
        "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
    const RFC7636_CHALLENGE: &str =
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM";

    fn oauth(http: Arc<MockHttpClient>) -> AnthropicOAuth<Arc<MockHttpClient>> {
        AnthropicOAuth {
            http,
            client_id: "test-client".to_owned(),
            token_url: "https://tokens.test/oauth/token".to_owned(),
        }
    }

    #[test]
    fn sha256_matches_the_abc_vector() {
        // FIPS 180-4 example: sha256("abc").
        let digest = sha256(b"abc");
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_matches_the_empty_vector() {
        let hex: String =
            sha256(b"").iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn pkce_challenge_matches_the_rfc_vector() {
        let challenge = base64url_nopad(&sha256(RFC7636_VERIFIER.as_bytes()));
        assert_eq!(challenge, RFC7636_CHALLENGE);
    }

    #[test]
    fn base64url_is_url_safe_and_unpadded() {
        // Bytes that force both `-` (62) and `_` (63) in the output.
        let encoded = base64url_nopad(&[0xfb, 0xff, 0xfe]);
        assert_eq!(encoded, "-__-");
        assert!(!encoded.contains('='));
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
    }

    #[test]
    fn start_yields_an_s256_challenge_and_independent_state() {
        let flow = oauth(Arc::new(MockHttpClient::new())).build_login(
            OAuthMode::Subscription,
            &Redirect::Manual,
            RFC7636_VERIFIER.to_owned(),
            "csrf-state-value".to_owned(),
        );

        assert!(flow.authorize_url.starts_with(AUTHORIZE_SUBSCRIPTION));
        assert!(flow.authorize_url.contains("code_challenge_method=S256"));
        assert!(
            flow.authorize_url
                .contains(&format!("code_challenge={RFC7636_CHALLENGE}"))
        );
        // The state travels on the URL and is independent of the challenge.
        assert_eq!(flow.state, "csrf-state-value");
        assert!(flow.authorize_url.contains("state=csrf-state-value"));
        assert_ne!(flow.state, RFC7636_CHALLENGE);
    }

    #[test]
    fn start_mints_distinct_random_verifier_and_state_each_call() {
        let flow = oauth(Arc::new(MockHttpClient::new()));
        let a = flow.start(OAuthMode::Subscription, &Redirect::Manual);
        let b = flow.start(OAuthMode::Subscription, &Redirect::Manual);
        assert_ne!(a.state, b.state);
        assert_ne!(a.verifier, b.verifier);
        // A 43-char base64url token (32 bytes) satisfies PKCE length rules.
        assert_eq!(a.verifier.len(), 43);
        assert!(a.state.len() >= 43);
    }

    #[test]
    fn console_mode_points_at_the_console_endpoint() {
        let flow = oauth(Arc::new(MockHttpClient::new())).build_login(
            OAuthMode::Console,
            &Redirect::Manual,
            RFC7636_VERIFIER.to_owned(),
            "s".to_owned(),
        );
        assert!(flow.authorize_url.starts_with(AUTHORIZE_CONSOLE));
    }

    #[test]
    fn localhost_redirect_uri_carries_the_port() {
        let flow = oauth(Arc::new(MockHttpClient::new())).build_login(
            OAuthMode::Subscription,
            &Redirect::Localhost { port: 8484 },
            RFC7636_VERIFIER.to_owned(),
            "s".to_owned(),
        );
        assert_eq!(flow.redirect_uri, "http://localhost:8484/callback");
        // Redirect URI is percent-encoded onto the authorize URL.
        assert!(
            flow.authorize_url.contains(
                "redirect_uri=http%3A%2F%2Flocalhost%3A8484%2Fcallback"
            )
        );
    }

    #[test]
    fn parse_pasted_splits_code_and_state() {
        let parsed = AuthorizationCode::parse_pasted("  the-code#the-state \n");
        assert_eq!(parsed.code, "the-code");
        assert_eq!(parsed.state.as_deref(), Some("the-state"));

        let bare = AuthorizationCode::parse_pasted("just-a-code");
        assert_eq!(bare.code, "just-a-code");
        assert_eq!(bare.state, None);
    }

    #[test]
    fn from_callback_query_decodes_code_and_state() {
        let parsed =
            AuthorizationCode::from_callback_query("code=a%2Bb&state=xyz")
                .unwrap();
        assert_eq!(parsed.code, "a+b");
        assert_eq!(parsed.state.as_deref(), Some("xyz"));
    }

    #[test]
    fn from_callback_query_without_code_is_an_error() {
        let err =
            AuthorizationCode::from_callback_query("state=only").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidRequest);
    }

    fn token_body(refresh: bool) -> String {
        let refresh = if refresh {
            r#""refresh_token":"refresh-xyz","#
        } else {
            ""
        };
        format!(
            r#"{{"token_type":"Bearer","access_token":"access-abc",{refresh}"expires_in":3600}}"#
        )
    }

    #[tokio::test]
    async fn exchange_runs_through_the_transport_and_mints_a_credential() {
        let mock =
            Arc::new(MockHttpClient::with_response(200, token_body(true)));
        let flow = oauth(mock.clone());
        let login = flow.build_login(
            OAuthMode::Subscription,
            &Redirect::Localhost { port: 8484 },
            RFC7636_VERIFIER.to_owned(),
            "the-state".to_owned(),
        );

        let code = AuthorizationCode {
            code: "auth-code".to_owned(),
            state: Some("the-state".to_owned()),
        };
        let credential = flow.exchange(&login, &code).await.unwrap();

        let tokens = credential.as_oauth().expect("oauth credential");
        assert_eq!(tokens.access_token, "access-abc");
        assert_eq!(tokens.refresh_token, "refresh-xyz");
        // Expiry bakes in the safety margin: now + 3600 - 300 (with slack).
        let expires = tokens.expires_at.expect("expiry set");
        let expected = now_unix() + 3600 - EXPIRY_SAFETY_MARGIN_SECS;
        assert!(expected.saturating_sub(expires) <= 2);

        // The request hit the token endpoint with the exchange payload.
        let sent = mock.last_request();
        assert_eq!(sent.method, Method::Post);
        assert_eq!(sent.url, "https://tokens.test/oauth/token");
        let body: serde_json::Value =
            serde_json::from_slice(sent.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["grant_type"], "authorization_code");
        assert_eq!(body["code"], "auth-code");
        // The token endpoint rejects a body without the echoed CSRF state.
        assert_eq!(body["state"], "the-state");
        assert_eq!(body["code_verifier"], RFC7636_VERIFIER);
        assert_eq!(body["redirect_uri"], "http://localhost:8484/callback");
        assert_eq!(body["client_id"], "test-client");
    }

    #[tokio::test]
    async fn exchange_falls_back_to_the_login_state_when_the_code_omits_it() {
        let mock =
            Arc::new(MockHttpClient::with_response(200, token_body(true)));
        let flow = oauth(mock.clone());
        let login = flow.build_login(
            OAuthMode::Subscription,
            &Redirect::Manual,
            RFC7636_VERIFIER.to_owned(),
            "login-state".to_owned(),
        );
        let code = AuthorizationCode {
            code: "auth-code".to_owned(),
            state: None,
        };

        flow.exchange(&login, &code).await.unwrap();

        let sent = mock.last_request();
        let body: serde_json::Value =
            serde_json::from_slice(sent.body.as_deref().unwrap()).unwrap();
        assert_eq!(body["state"], "login-state");
    }

    #[tokio::test]
    async fn exchange_rejects_a_state_mismatch_before_any_request() {
        let mock = Arc::new(MockHttpClient::new());
        let flow = oauth(mock.clone());
        let login = flow.build_login(
            OAuthMode::Subscription,
            &Redirect::Manual,
            RFC7636_VERIFIER.to_owned(),
            "expected-state".to_owned(),
        );
        let code = AuthorizationCode {
            code: "auth-code".to_owned(),
            state: Some("forged-state".to_owned()),
        };

        let err = flow.exchange(&login, &code).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Authentication);
        // No request reached the transport.
        assert!(mock.requests().is_empty());
    }

    #[tokio::test]
    async fn exchange_maps_a_non_2xx_to_a_typed_error() {
        let mock = Arc::new(MockHttpClient::with_response(
            400,
            r#"{"error":"invalid_grant"}"#,
        ));
        let flow = oauth(mock);
        let login = flow.build_login(
            OAuthMode::Subscription,
            &Redirect::Manual,
            RFC7636_VERIFIER.to_owned(),
            "s".to_owned(),
        );
        let code = AuthorizationCode {
            code: "bad".to_owned(),
            state: None,
        };
        let err = flow.exchange(&login, &code).await.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidRequest);
        assert_eq!(err.status(), Some(400));
    }

    #[tokio::test]
    async fn refresh_runs_through_the_transport_and_renews_the_credential() {
        let mock =
            Arc::new(MockHttpClient::with_response(200, token_body(true)));
        let flow = oauth(mock.clone());

        let credential = flow.refresh("old-refresh").await.unwrap();
        let tokens = credential.as_oauth().expect("oauth credential");
        assert_eq!(tokens.access_token, "access-abc");
        assert_eq!(tokens.refresh_token, "refresh-xyz");

        let body: serde_json::Value = serde_json::from_slice(
            mock.last_request().body.as_deref().unwrap(),
        )
        .unwrap();
        assert_eq!(body["grant_type"], "refresh_token");
        assert_eq!(body["refresh_token"], "old-refresh");
    }

    #[tokio::test]
    async fn refresh_reuses_the_old_token_when_the_response_omits_one() {
        let mock =
            Arc::new(MockHttpClient::with_response(200, token_body(false)));
        let flow = oauth(mock);
        let credential = flow.refresh("keep-me").await.unwrap();
        let tokens = credential.as_oauth().unwrap();
        assert_eq!(tokens.refresh_token, "keep-me");
    }

    #[test]
    fn credential_from_token_bakes_in_the_safety_margin() {
        let token = TokenResponse {
            access_token: "acc".to_owned(),
            refresh_token: Some("ref".to_owned()),
            expires_in: Some(3600),
        };
        let credential = credential_from_token(token, None, 1_000_000).unwrap();
        let tokens = credential.as_oauth().unwrap();
        assert_eq!(
            tokens.expires_at,
            Some(1_000_000 + 3600 - EXPIRY_SAFETY_MARGIN_SECS)
        );
    }

    #[test]
    fn credential_from_token_without_a_refresh_token_is_an_error() {
        let token = TokenResponse {
            access_token: "acc".to_owned(),
            refresh_token: None,
            expires_in: Some(10),
        };
        let err = credential_from_token(token, None, 0).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Authentication);
    }

    #[test]
    fn login_debug_redacts_the_verifier() {
        let flow = oauth(Arc::new(MockHttpClient::new())).build_login(
            OAuthMode::Subscription,
            &Redirect::Manual,
            "super-secret-verifier".to_owned(),
            "s".to_owned(),
        );
        let rendered = format!("{flow:?}");
        assert!(!rendered.contains("super-secret-verifier"));
        assert!(rendered.contains("redacted"));
    }

    #[test]
    fn capture_localhost_reads_the_code_off_the_redirect() {
        use std::io::Write as _;
        use std::net::TcpStream;

        // Bind an ephemeral port, then hand it to the capture on this thread
        // while a client thread plays the browser redirect.
        let probe = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = probe.local_addr().unwrap().port();
        drop(probe);

        let client = std::thread::spawn(move || {
            // Give the capture a moment to bind the freed port.
            for _ in 0..50 {
                if let Ok(mut stream) = TcpStream::connect(("127.0.0.1", port))
                {
                    let request = b"GET /callback?code=cap-code&state=cap-state HTTP/1.1\r\nHost: localhost\r\n\r\n";
                    let _ = stream.write_all(request);
                    return;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            panic!("could not connect to the capture listener");
        });

        let captured = capture_localhost(port).unwrap();
        client.join().unwrap();

        assert_eq!(captured.code, "cap-code");
        assert_eq!(captured.state.as_deref(), Some("cap-state"));
    }
}
