# Model catalog baseline generated from upstream data and committed as embedded data

The baseline Catalog ships as generated JSON embedded with `include_str!` and parsed with `serde_json` (always compiled). An `xtask` binary, run on demand via `just gen-models`, fetches an aggregate upstream model dataset, filters deprecated entries, normalizes and dedupes aliases, maps the result to tapir's Model schema (tagged with a schema version), and writes the committed file. We chose generation-from-upstream over hand-maintained const tables because model metadata — context windows, pricing, new releases — changes weekly across dozens of Providers and hand-curation would rot. We chose committing the generated file over a `build.rs` that fetches at build time because builds must be offline and deterministic and must not depend on a third-party endpoint's availability.

## Consequences

- Aliases and deprecation are resolved entirely at generation time, so the runtime Model carries no alias or deprecated field.
- The Catalog is only as fresh as the last `just gen-models` run and commit; staleness shows up as a review-visible diff, never a silent build-time fetch.
- Runtime freshness, when wanted, comes from the separate fetched/live layer (see ADR-0005), not from the build.
- Baseline data is gated per provider like the Registry entries, so a minimal build embeds nothing.
