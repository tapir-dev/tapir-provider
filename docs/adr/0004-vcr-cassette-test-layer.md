# VCR cassette record/replay test layer over the HttpClient seam

Realistic request-to-stream and OAuth-refresh behavior is exercised against recorded traffic via `VcrClient`, an `HttpClient` that wraps another transport and plugs into the same seam every Provider already uses, rather than through hand-built doubles. We chose this over extending `MockHttpClient` because record/replay is a distinct concern (on-disk cassettes, redaction, matching) and the seam already makes it transport-shaped; a Provider needs no change to be driven from a cassette. Cassettes are JSON (serde_json is always compiled, unlike the feature-gated `toml`), and the whole layer sits behind `test-utils` alongside `MockHttpClient`.

Key decisions:

- **Chunked response storage.** A response is stored as an ordered list of body chunks, so a streamed SSE completion replays chunk-by-chunk with the exact framing it was recorded with (including events split across boundaries). A non-streaming `send` stores a single chunk. `send_stream` records at status 200 only, since a non-2xx is surfaced as an error before any stream exists.
- **Redaction on both record and match.** A `Redactor` strips auth-bearing headers and sensitive JSON body fields (`access_token`, `refresh_token`, `code_verifier`, `state`, ...) when writing a cassette, and applies the same transform to a live request before matching it. A redacted recording therefore still matches a request carrying the real secret, and no header or body secret lands on disk. Request URLs are stored verbatim (not redacted); the Providers here carry secrets in headers and bodies, not query strings.
- **Strict ordered replay, matched on method + URL + redacted body.** Interactions replay in recorded order; the next one's request must match or replay errors. JSON bodies are re-serialized to a canonical form so key order does not affect matching. Headers are stored for inspection but not matched, since volatile headers (dates, request ids) would make matching brittle.
- **Auto resolves by file presence.** `VcrMode::Auto` records when the cassette file is absent and replays when it is present, decided once at construction. Record forwards to the wrapped transport (needing a real Credential); replay never contacts it (a dummy Credential suffices).

## Consequences

- Cassettes are diffable, reviewable JSON, but assume UTF-8 text bodies (true for the JSON/SSE traffic these Providers exchange); a binary body is stored lossily.
- Replayed secret values are the `<redacted>` placeholder, so tests assert on flow and shape, not on token values.
- Strict ordered, whole-file matching means a code change that alters the request sent (or its order) forces the cassette to be re-recorded rather than replaying the wrong response.
