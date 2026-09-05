# Runtime layered Model Registry replaces the compile-time build path

A Model's routing (wire API, endpoint, headers), metadata, and Credential are now resolved through a stateful `ModelRegistry` loaded at runtime, which merges a compiled-in baseline Catalog with a user-override layer and a persisted/fetched layer (precedence: user > fetched > baseline) and turns a Model Entry into a live Provider via `create_provider`. We removed `Registry::build` — the previous single entry point that took a provider name plus model string and built a Provider from compile-time data only — because it cannot express Models a caller adds by config or the SDK discovers by fetch. We chose a separate, owned, async-refreshable `ModelRegistry` over extending the existing `Registry` (a ZST over a compile-time table) so Provider identity and selection stay compile-time and cheap while the Catalog, which needs runtime state and I/O, is a distinct object.

## Consequences

- Two registries coexist by design: the Registry says which Providers exist (ids, aliases, Credential source, feature gating); the Model Registry says which Models exist and how to call them. The Registry keeps its identity/selection role and only loses `build`.
- The overlay layers sit behind their own features (`models-user-config`, `models-fetch`) on the umbrella `models` feature, off by default, so the minimal build is unchanged.
- `create_provider` takes the transport (`H: HttpClient`), preserving the VCR/mock seam. The common path stays one call: `ModelRegistry::create_provider(provider, model, transport)`.
- Removing `build` is a breaking change taken now, while pre-1.0 with no external consumers, rather than carried through a deprecation cycle.
