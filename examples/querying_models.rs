// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Querying the model Catalog.
//!
//! Reads are synchronous over the compiled-in baseline Catalog: list the
//! providers, list every model, filter to one provider, look up a single model,
//! and narrow on its wire API before acting on API-specific options.
//!
//! Run with:
//!
//! ```text
//! cargo run --example querying_models --features models,anthropic,openai
//! ```

use std::collections::BTreeSet;

use tapir_provider::{Api, InputType, ModelEntry, ModelRegistry};

fn main() -> Result<(), tapir_provider::Error> {
    // Load the baseline Catalog: no Token Store, no user config file.
    let registry = ModelRegistry::load(None, None)?;

    // Registered providers. The registry keys models, not providers, so derive
    // the set from the model rows.
    let providers: BTreeSet<&str> = registry
        .models()
        .iter()
        .map(|entry| entry.model.provider.as_str())
        .collect();
    println!(
        "providers: {}",
        providers.into_iter().collect::<Vec<_>>().join(", ")
    );

    // Every model across providers.
    println!("total models: {}", registry.models().len());

    // Models for one provider.
    let anthropic: Vec<&ModelEntry> = registry
        .models()
        .iter()
        .filter(|entry| entry.model.provider.as_str() == "anthropic")
        .collect();

    for entry in &anthropic {
        let model = &entry.model;
        println!("{}: {}", model.id, model.name);
        println!("  API: {}", model.api.as_str());
        println!("  Context: {} tokens", model.context_window);
        println!("  Vision: {}", model.input.contains(&InputType::Image));
        println!("  Reasoning: {}", model.reasoning);
    }

    // Look up a single model by (provider, id).
    let Some(entry) = registry.find("anthropic", "claude-sonnet-4-5") else {
        println!("claude-sonnet-4-5 not found in this build");
        return Ok(());
    };

    // Narrow on the wire API before reaching for API-specific options — the
    // typed equivalent of guarding a dynamically listed model.
    match &entry.model.api {
        Api::AnthropicMessages => println!(
            "{} speaks anthropic-messages — thinking options available",
            entry.model.id
        ),
        api => println!(
            "{} speaks {} — no thinking options",
            entry.model.id,
            api.as_str()
        ),
    }

    Ok(())
}
