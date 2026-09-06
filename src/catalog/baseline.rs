// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The compiled-in baseline layer of the Catalog.
//!
//! Each Provider contributes its own [`ModelEntry`]s from a generated JSON file
//! embedded with `include_str!` and parsed with `serde_json`, gated on that
//! Provider's Cargo feature exactly like the [`Registry`](crate::Registry)'s
//! entry table — so a build with no Provider feature embeds nothing. The files
//! are produced by `just gen-models` (see ADR-0006); they are generated data, so
//! a parse or schema-version mismatch here is a generator bug, and the loader
//! treats it as unrecoverable.

#[cfg(any(
    feature = "anthropic",
    feature = "openai",
    feature = "deepseek"
))]
use serde::Deserialize;

#[cfg(any(
    feature = "anthropic",
    feature = "openai",
    feature = "deepseek"
))]
use crate::model::{Model, ModelEntry};

/// The schema version this build expects the embedded files to carry. Kept in
/// lockstep with the generator (`xtask`); a mismatch means a stale committed file.
#[cfg(any(feature = "anthropic", feature = "openai", feature = "deepseek"))]
const SCHEMA_VERSION: u32 = 1;

/// The on-disk shape of a generated baseline file. Extra fields (such as the
/// human-facing `provider` tag) are ignored.
#[cfg(any(feature = "anthropic", feature = "openai", feature = "deepseek"))]
#[derive(Deserialize)]
struct GeneratedCatalog {
    schema_version: u32,
    models: Vec<Model>,
}

/// The baseline [`ModelEntry`]s compiled into this build, one Provider's worth
/// appended after another. Empty when no Provider feature is enabled.
pub(super) fn entries() -> Vec<crate::model::ModelEntry> {
    #[allow(unused_mut)]
    let mut entries = Vec::new();
    #[cfg(feature = "anthropic")]
    entries.extend(parse(include_str!("../../assets/models/anthropic.json")));
    #[cfg(feature = "openai")]
    entries.extend(parse(include_str!("../../assets/models/openai.json")));
    #[cfg(feature = "deepseek")]
    entries.extend(parse(include_str!("../../assets/models/deepseek.json")));
    entries
}

/// Parse an embedded baseline file into entries.
///
/// The input is generated, committed data, not runtime input: a parse failure or
/// a schema-version mismatch is a build-the-world bug in the generation pipeline,
/// so this panics rather than degrading to an empty Catalog.
#[cfg(any(feature = "anthropic", feature = "openai", feature = "deepseek"))]
fn parse(raw: &str) -> Vec<ModelEntry> {
    let catalog: GeneratedCatalog = serde_json::from_str(raw)
        .expect("embedded baseline catalog must parse; run `just gen-models`");
    assert_eq!(
        catalog.schema_version, SCHEMA_VERSION,
        "embedded baseline catalog schema version mismatch; run `just gen-models`"
    );
    catalog.models.into_iter().map(ModelEntry::new).collect()
}

#[cfg(all(test, feature = "openai"))]
mod openai_tests {
    use super::*;

    fn openai() -> Vec<ModelEntry> {
        parse(include_str!("../../assets/models/openai.json"))
    }

    #[test]
    fn embedded_openai_baseline_parses_and_is_populated() {
        let entries = openai();
        assert!(!entries.is_empty());
        assert!(
            entries
                .iter()
                .all(|e| e.model.provider.as_str() == "openai")
        );
        // A current Model survives generation...
        assert!(entries.iter().any(|e| e.model.id.as_str() == "gpt-4o-mini"));
        // ...while a deprecated upstream row is dropped at generation time...
        assert!(
            entries
                .iter()
                .all(|e| e.model.id.as_str() != "gpt-3.5-turbo")
        );
        // ...as is a zero-context (non-completion) row.
        assert!(
            entries
                .iter()
                .all(|e| e.model.id.as_str() != "chatgpt-image-latest")
        );
        // Every embedded Model carries a usable context window.
        assert!(entries.iter().all(|e| e.model.context_window > 0));
    }

    #[test]
    fn embedded_openai_baseline_is_sorted_by_id() {
        let ids: Vec<_> = openai()
            .iter()
            .map(|e| e.model.id.as_str().to_owned())
            .collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }
}

#[cfg(all(test, feature = "deepseek"))]
mod deepseek_tests {
    use super::*;

    #[test]
    fn embedded_deepseek_baseline_parses_and_is_populated() {
        let entries = parse(include_str!("../../assets/models/deepseek.json"));
        assert!(!entries.is_empty());
        assert!(
            entries
                .iter()
                .all(|e| e.model.provider.as_str() == "deepseek")
        );
        assert!(
            entries
                .iter()
                .any(|e| e.model.id.as_str() == "deepseek-chat")
        );
        assert!(entries.iter().all(|e| e.model.context_window > 0));
    }
}
