// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Dev tooling for tapir-provider.
//!
//! The one command today is `gen-models`, which regenerates the compiled-in
//! baseline model catalog (see ADR-0006). It fetches an aggregate upstream model
//! dataset over HTTP, keeps only the Providers this crate has an adapter for,
//! drops deprecated entries, dedupes by Model id, maps each row onto the crate's
//! own [`Model`] schema, and writes one deterministic, schema-versioned JSON file
//! per Provider under `assets/models/`. The crate embeds those files with
//! `include_str!`, so builds stay offline: the network is touched here, on
//! demand, never in a build.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use tapir_provider::{Api, InputType, Model, ModelCost, ModelId, ProviderId};

/// The aggregate upstream dataset the baseline is generated from.
const UPSTREAM_URL: &str = "https://models.dev/api.json";

/// The schema version stamped into each generated file. Bump when the on-disk
/// shape changes so a stale file is a loud parse/version mismatch, not silent
/// drift; the crate's loader asserts the same constant.
const SCHEMA_VERSION: u32 = 1;

/// Where the generated per-Provider files are written, relative to the workspace
/// root (the directory `cargo run -p xtask` runs from).
const OUTPUT_DIR: &str = "assets/models";

/// A Provider this crate has an adapter for, and how its upstream rows map onto
/// the [`Model`] schema. The upstream dataset carries no wire API or base URL for
/// these Providers, so both are fixed here rather than read from the data.
struct ProviderSpec {
    /// The provider's key in the upstream dataset.
    upstream: &'static str,
    /// The canonical Provider id this crate uses.
    id: &'static str,
    /// The wire protocol every Model of this Provider is called over.
    api: Api,
    /// The base URL every Model of this Provider is served from.
    base_url: &'static str,
}

/// The Providers whose baseline we generate. Kept in lockstep with the crate's
/// feature-gated Registry entries and baseline modules.
const SUPPORTED: &[ProviderSpec] = &[
    ProviderSpec {
        upstream: "anthropic",
        id: "anthropic",
        api: Api::AnthropicMessages,
        base_url: "https://api.anthropic.com",
    },
    ProviderSpec {
        upstream: "openai",
        id: "openai",
        api: Api::OpenAICompletions,
        base_url: "https://api.openai.com",
    },
];

/// One Provider in the upstream dataset. Only the fields the generator reads are
/// declared; the rest of each object is ignored.
#[derive(Deserialize)]
struct UpstreamProvider {
    #[serde(default)]
    models: BTreeMap<String, UpstreamModel>,
}

/// One Model in the upstream dataset.
#[derive(Deserialize)]
struct UpstreamModel {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    reasoning: Option<bool>,
    /// Lifecycle status; `"deprecated"` rows are dropped at generation time.
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    modalities: Option<UpstreamModalities>,
    #[serde(default)]
    limit: Option<UpstreamLimit>,
    #[serde(default)]
    cost: Option<UpstreamCost>,
}

/// The input/output modalities block; only `input` is mapped.
#[derive(Deserialize)]
struct UpstreamModalities {
    #[serde(default)]
    input: Vec<String>,
}

/// Context-window and output-token limits.
#[derive(Deserialize)]
struct UpstreamLimit {
    #[serde(default)]
    context: u64,
    #[serde(default)]
    output: u64,
}

/// Per-million-token costs, matching the crate's [`ModelCost`] field-for-field.
#[derive(Deserialize)]
struct UpstreamCost {
    #[serde(default)]
    input: f64,
    #[serde(default)]
    output: f64,
    #[serde(default)]
    cache_read: f64,
    #[serde(default)]
    cache_write: f64,
}

/// The on-disk shape of a generated file: a schema version plus the Provider's
/// Models. The crate's loader reads the same two fields.
#[derive(serde::Serialize)]
struct GeneratedCatalog<'a> {
    schema_version: u32,
    provider: &'a str,
    models: Vec<Model>,
}

fn main() -> Result<(), Box<dyn Error>> {
    match std::env::args().nth(1).as_deref() {
        Some("gen-models") => gen_models(),
        other => {
            eprintln!("unknown command: {}", other.unwrap_or("<none>"));
            eprintln!("usage: cargo run -p xtask -- gen-models");
            std::process::exit(2);
        }
    }
}

/// Fetch the upstream dataset and (re)write one baseline file per supported
/// Provider.
fn gen_models() -> Result<(), Box<dyn Error>> {
    eprintln!("fetching {UPSTREAM_URL}");
    let body = reqwest::blocking::get(UPSTREAM_URL)?
        .error_for_status()?
        .text()?;
    let data: BTreeMap<String, UpstreamProvider> = serde_json::from_str(&body)?;

    fs::create_dir_all(OUTPUT_DIR)?;
    for spec in SUPPORTED {
        let provider = data.get(spec.upstream).ok_or_else(|| {
            format!("upstream dataset has no provider {:?}", spec.upstream)
        })?;
        let models = build_models(spec, provider)?;
        let count = models.len();
        let catalog = GeneratedCatalog {
            schema_version: SCHEMA_VERSION,
            provider: spec.id,
            models,
        };
        let path = Path::new(OUTPUT_DIR).join(format!("{}.json", spec.id));
        let mut json = serde_json::to_string_pretty(&catalog)?;
        json.push('\n');
        fs::write(&path, json)?;
        println!("wrote {} ({count} models)", path.display());
    }
    Ok(())
}

/// Map a Provider's upstream Models onto the crate's schema: drop deprecated
/// rows, dedupe by id, and sort by id so the output is deterministic.
fn build_models(
    spec: &ProviderSpec,
    provider: &UpstreamProvider,
) -> Result<Vec<Model>, Box<dyn Error>> {
    let mut models = Vec::new();
    let mut seen = BTreeSet::new();
    for upstream in provider.models.values() {
        if upstream.status.as_deref() == Some("deprecated") {
            continue;
        }
        let model = to_model(spec, upstream)?;
        // A Model with no context window can't be driven as a completion — the
        // upstream reports none for non-text endpoints (image generation, ...),
        // which don't belong in a completion baseline.
        if model.context_window == 0 {
            continue;
        }
        // Dedupe aliases: the first row to claim an id wins.
        if seen.insert(model.id.as_str().to_owned()) {
            models.push(model);
        }
    }
    models.sort_by(|a, b| a.id.as_str().cmp(b.id.as_str()));
    Ok(models)
}

/// Build one [`Model`] from an upstream row, fixing the wire API and base URL
/// from the [`ProviderSpec`].
fn to_model(
    spec: &ProviderSpec,
    upstream: &UpstreamModel,
) -> Result<Model, Box<dyn Error>> {
    let limit = upstream.limit.as_ref();
    let cost = upstream.cost.as_ref();
    Ok(Model {
        id: ModelId::new(upstream.id.clone())?,
        provider: ProviderId::new(spec.id)?,
        api: spec.api.clone(),
        name: upstream.name.clone().unwrap_or_else(|| upstream.id.clone()),
        base_url: spec.base_url.to_owned(),
        reasoning: upstream.reasoning.unwrap_or(false),
        input: map_inputs(upstream.modalities.as_ref()),
        cost: ModelCost {
            input: cost.map_or(0.0, |c| c.input),
            output: cost.map_or(0.0, |c| c.output),
            cache_read: cost.map_or(0.0, |c| c.cache_read),
            cache_write: cost.map_or(0.0, |c| c.cache_write),
        },
        context_window: limit
            .map_or(0, |l| u32::try_from(l.context).unwrap_or(u32::MAX)),
        max_tokens: limit
            .map_or(0, |l| u32::try_from(l.output).unwrap_or(u32::MAX)),
        headers: Vec::new(),
    })
}

/// Map upstream input modalities onto the modalities the schema models, keeping
/// order and dropping duplicates and unmodeled kinds (pdf, audio, ...). A row
/// with no modeled input still accepts text.
fn map_inputs(modalities: Option<&UpstreamModalities>) -> Vec<InputType> {
    let mut inputs = Vec::new();
    if let Some(modalities) = modalities {
        for kind in &modalities.input {
            let mapped = match kind.as_str() {
                "text" => Some(InputType::Text),
                "image" => Some(InputType::Image),
                _ => None,
            };
            if let Some(input) = mapped
                && !inputs.contains(&input)
            {
                inputs.push(input);
            }
        }
    }
    if inputs.is_empty() {
        inputs.push(InputType::Text);
    }
    inputs
}
