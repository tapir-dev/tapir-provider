# Local task runner for tapir. Run `just` to list recipes.
# Bootstrap `just` itself with `cargo install just`; then run `just setup` to
# install the rest of the dev toolchain.

# Test threads for nextest. Override with `just jobs=8 test` or JOBS=8.
jobs := env_var_or_default("JOBS", "16")

# List available recipes.
default:
    @just --list

# Install the dev tools `just check` and `just test-ci` need. Run once per machine.
# The nightly is pinned in rust-toolchain.toml; `rustup show` installs exactly that.
setup:
    rustup show
    cargo binstall -y cargo-nextest cargo-deny cargo-machete cargo-fuzz cargo-llvm-cov typos-cli
    @echo "Also install editorconfig-checker:"
    @echo "  go install github.com/editorconfig-checker/editorconfig-checker/v3/cmd/editorconfig-checker@latest"

# Full local check suite (mirrors the core CI). Roughly the old check.sh.
check: _check-tools editorconfig fmt clippy build doc deny machete typos license fuzz-build test-ci

# Check formatting (rustfmt from the pinned nightly in rust-toolchain.toml).
fmt:
    cargo fmt --all -- --check

# Lint with clippy, denying warnings.
clippy:
    cargo clippy --all-targets --all-features -- -D warnings

# Build all targets in debug mode, all features. `--locked` enforces a fresh lockfile.
build: _lockfile
    cargo build --all-targets --all-features --locked

# Build the documentation, denying warnings (mirrors CI and docs.rs conventions).
doc: _lockfile
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items --all-features --locked

# Check the dependency tree (advisories, licenses, bans, sources).
deny:
    cargo deny check

# Find unused dependencies.
machete:
    cargo machete

# Check spelling.
typos:
    typos

# Check every source file starts with the copyright + SPDX license header.
license:
    #!/usr/bin/env bash
    set -euo pipefail
    fail=0
    while IFS= read -r f; do
        # Python uses '#' line comments; everything else uses '//'.
        case "$f" in
            *.py) c='#' ;;
            *)    c='//' ;;
        esac
        line1="" line2=""
        { IFS= read -r line1; IFS= read -r line2; } < "$f" || true
        if [ "$line1" != "$c SPDX-License-Identifier: ISC" ] \
            || [ "$line2" != "$c SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>" ]; then
            echo "missing/malformed license header: $f" >&2
            fail=1
        fi
    done < <(git ls-files --cached --others --exclude-standard -- '*.rs' '*.py')
    [ "$fail" -eq 0 ] || { echo "Headers must match src/lib.rs (SPDX-License-Identifier + Copyright line)." >&2; exit 1; }

# Lint the GitHub Actions workflows.
actionlint:
    actionlint

# Check that all files follow the editorconfig.
editorconfig:
    editorconfig-checker

# Default test suite: fast tests, all targets/features.
# `--no-tests=pass` keeps a green run while the suite is still empty/small.
test: _lockfile
    RUST_BACKTRACE=1 cargo nextest run --all-targets --all-features \
        --no-tests=pass --test-threads={{ jobs }}

# Documentation tests (nextest doesn't run these).
test-doc: _lockfile
    RUST_BACKTRACE=1 cargo test --doc --all-features

# Ignored-by-default smoke tests, in release mode.
# Skips cleanly until a `tests/smoke_tests.rs` integration target exists.
test-smoke: _lockfile
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f tests/smoke_tests.rs ]; then
        echo "⚠️  tests/smoke_tests.rs not present, skipping smoke tests."
        exit 0
    fi
    RUST_BACKTRACE=1 cargo nextest run --all-features --release \
        --no-tests=pass --test smoke_tests --run-ignored=all

# Performance-sensitive tests that must run sequentially, in release mode.
# Filters that match nothing are tolerated via `--no-tests=pass`; add real
# perf-sensitive test names here as they land.
test-sequential: _lockfile
    RUST_BACKTRACE=1 cargo nextest run --release --jobs=1 --run-ignored=only \
        --no-tests=pass

# What CI runs: fast + doc + smoke + sequential.
test-ci: test test-doc test-smoke test-sequential

# Full slow suite (release ignored-only then fast). May take ~10 minutes.
test-slow:
    cargo nextest run --release --jobs=1 --run-ignored=only --no-tests=pass
    RUST_BACKTRACE=1 cargo nextest run --no-tests=pass

# Run the fast + sequential suite 50 times to surface flaky tests.
test-many:
    #!/usr/bin/env bash
    set -euo pipefail
    for i in $(seq 1 50); do
        echo "Iteration $i/50:"
        if ! { just test && just test-sequential; }; then
            echo "❌ Test failed in one iteration, maybe some tests are flaky."
            exit 1
        fi
    done

# Build all fuzz targets (compile-check only).
# Two workarounds baked in:
#  - `--target "$host"`: cargo-fuzz otherwise defaults to the triple it was itself
#    built for (e.g. musl), whose static libc is incompatible with the sanitizer.
#  - `-s none`: current rustc wires SanitizerCoverage into `-Zsanitizer=address`
#    itself, so cargo-fuzz's extra `-Cpasses=sancov-module` runs sancov twice and
#    leaves `asan.module_dtor` referencing undefined `__sancov_gen_*` symbols
#    (link error). Disabling the sanitizer drops that pass; instrumentation is
#    codegen-only, so this still fully compile-checks the fuzz harness.
# Uses the pinned nightly (rust-toolchain.toml), not a bare `+nightly`.
fuzz-build:
    #!/usr/bin/env bash
    set -euo pipefail
    host=$(rustc -vV | sed -n 's/^host: //p')
    cargo fuzz build --target "$host" -s none

# Run every fuzz target for 30s.
# NOTE: real fuzzing needs the coverage instrumentation, so we can't apply the
# `-s none` workaround from `fuzz-build` here. On a rustc where cargo-fuzz's
# `-Cpasses=sancov-module` breaks the link (undefined `__sancov_gen_*`), pin a
# compatible nightly in rust-toolchain.toml until the upstream fix lands.
fuzz:
    #!/usr/bin/env bash
    set -euo pipefail
    host=$(rustc -vV | sed -n 's/^host: //p')
    targets=$(cargo fuzz list)
    if [ -z "$targets" ]; then
        echo "⚠️  No fuzz targets found, skipping."
        exit 0
    fi
    for target in $targets; do
        echo "Fuzzing $target..."
        cargo fuzz run --target "$host" "$target" -- -max_total_time=30
    done

# Compile-check benchmarks. CI never runs benches.
# `test-utils` unlocks the `slice`/`SlotState` helpers that most benches need.
bench-build: _lockfile
    cargo bench --no-run --locked --features test-utils

# Run benchmarks (divan). For local profiling only.
bench:
    cargo bench --features test-utils

# Generate an HTML coverage report and open it.
coverage:
    cargo llvm-cov --all-features --open nextest

# Revert every SHA-pinned action (@<sha> # vX) to its mutable tag (@vX). Use sparingly.
# Actions ship SHA-pinned for supply-chain safety; run this once if you'd rather
# track tags. The new ref is read from the `# vX` comment. Dependabot updates either form.
unpin:
    #!/usr/bin/env bash
    set -euo pipefail
    find .github -type f \( -name '*.yml' -o -name '*.yaml' \) -print0 \
        | xargs -0 perl -i -pe 's/uses:(\s*)([^@\s]+)@[0-9a-fA-F]{40}\s+#\s+(\S+)/uses:$1$2\@$3/g'
    echo "Unpinned all actions to tag refs. Review with: git diff .github"

# List unfinished work: todo!/unimplemented! macros and TODO-style comments.
todo:
    -rg 'todo!\(\)|unimplemented!\(\)' --iglob='!Justfile'
    -rg 'TODO|XXX|HACK|PERF|FIXME|BUG' --iglob='!Justfile'

# Ensure a Cargo.lock exists before the `--locked` recipes run (hidden helper).
_lockfile:
    [ -f Cargo.lock ] || cargo generate-lockfile

# Report any tools `just check` needs that aren't installed (hidden helper).
_check-tools:
    #!/usr/bin/env bash
    set -euo pipefail
    missing=()
    command -v cargo-nextest         >/dev/null 2>&1 || missing+=("cargo-nextest")
    command -v cargo-deny            >/dev/null 2>&1 || missing+=("cargo-deny")
    command -v cargo-machete         >/dev/null 2>&1 || missing+=("cargo-machete")
    command -v cargo-fuzz            >/dev/null 2>&1 || missing+=("cargo-fuzz")
    command -v typos                 >/dev/null 2>&1 || missing+=("typos-cli")
    command -v editorconfig-checker  >/dev/null 2>&1 || missing+=("editorconfig-checker")
    cargo fmt --version              >/dev/null 2>&1 || missing+=("nightly rustfmt")
    if [ ${#missing[@]} -ne 0 ]; then
        echo "Missing tools: ${missing[*]}" >&2
        echo "Run \`just setup\` to install most of them (editorconfig-checker is a Go binary)." >&2
        exit 1
    fi
