# Single crate with a feature-gated provider registry, no default providers

tapir-provider ships as one crate whose Registry entries are each behind a per-provider Cargo feature, so `default = []` yields a crate with no Providers compiled in and callers opt in with `--features anthropic,openai`. We chose this over a workspace of per-provider crates because our Providers are thin serde-over-HTTP adapters that share one core (traits, transport, retry, errors); a workspace only pays off when providers pull heavy, divergent SDKs, and would otherwise scatter the shared core and multiply release overhead.

## Consequences

- Cross-cutting axes (TLS backend, OAuth, file Token Store) are also Cargo features on the same crate.
- Adding a Provider means a gated module plus a gated Registry entry — no new crate, no new publish.
