# tapir-provider

Library includes models for agentic workflow

## Build & test

`just` drives everything; run `just --list` for the full recipe set. Tests run
under `cargo-nextest`, so `cargo test` misses the config — use the recipes.

- `just check` — full local CI gate; run before calling any change done.
- `just test` — fast tests (nextest, all targets/features).
- `just test-doc` — doctests (nextest doesn't run these).
- `cargo nextest run <name>` — a single test.

## Agent skills

### Issue tracker

Issues and specs live as GitHub issues, managed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical triage roles, each mapped to a same-named label. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.

