// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Selecting Providers from the compiled-in [`Registry`].
//!
//! The set of Providers a build ships is the set of Provider Cargo features it
//! was compiled with: each entry in the [`Registry`] is gated on its feature, so
//! a default build (no Provider feature) ships none, and this run — built with
//! `anthropic,openai` — ships exactly those two. There is no runtime step to
//! register a Provider; enabling its feature is the registration.
//!
//! Reads are synchronous and touch no network: list the compiled-in Providers,
//! then resolve a name — canonical id, alias, or mixed case — to the
//! [`ProviderInfo`] it selects, and report the environment variable each Provider
//! reads its default API key from. Turning a selection into a live Provider is a
//! separate step (`ModelRegistry::create_provider`) and is out of scope here.
//!
//! Run with:
//!
//! ```text
//! cargo run --example provider_factories --features anthropic,openai
//! ```

use tapir_provider::Registry;

fn main() {
    // The Providers compiled into this build. With `--features anthropic,openai`
    // both are present; drop a feature and its entry disappears from this list.
    println!(
        "compiled-in providers: {}",
        Registry::provider_ids().join(", ")
    );

    // Each entry carries the Provider's identity: its canonical id, the aliases
    // that also select it, and the environment variable holding its default key.
    for info in Registry::entries() {
        println!("- {}", info.id.as_str());
        if !info.aliases.is_empty() {
            println!("    aliases: {}", info.aliases.join(", "));
        }
        println!("    api key env: {}", info.api_key_env);
    }

    // Selecting a Provider is resolving a name against the Registry. The same
    // Provider answers to its canonical id, any alias, and any case.
    show("anthropic"); // canonical id
    show("claude"); // an alias for the same Provider
    show("OpenAI"); // case-insensitive
    show("gpt"); // an alias for OpenAI
    show("does-not-exist"); // no compiled-in Provider matches
}

/// Resolve `name` against the [`Registry`] and print which Provider it selects,
/// or that nothing matches.
fn show(name: &str) {
    match Registry::resolve(name) {
        Some(info) => {
            println!(
                "{name:>14} -> {} (key: {})",
                info.id.as_str(),
                info.api_key_env
            )
        }
        None => println!("{name:>14} -> no compiled-in provider"),
    }
}
