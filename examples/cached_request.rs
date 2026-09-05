// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! A cached multi-turn request against the real Anthropic API.
//!
//! This is the capstone that ties the request-side cache marker to the
//! observed usage on the reply. It sets [`CachePolicy::Standard`] and runs two
//! turns that share one large system prompt:
//!
//!   1. The first turn processes the whole prompt and *writes* the shared
//!      prefix into the cache — its reply reports `cache_write_tokens`.
//!   2. The follow-up turn reuses that exact prefix and *reads* it back rather
//!      than reprocessing it — its reply reports `cache_read_tokens`, and the
//!      cache-read rate is a fraction of the base input rate, so the second
//!      turn costs less.
//!
//! [`CachePolicy`] is Provider-neutral; here the Anthropic Provider turns it
//! into `ephemeral` `cache_control` breakpoints on the last system block and
//! the final message block. The system prompt is the large, stable span both
//! turns share, so it is the prefix the follow-up reads back; the trailing user
//! message is marked too, but it changes each turn. The prompt has to clear
//! Anthropic's minimum cacheable size, so this example inflates it with a large
//! synthetic reference document; below that floor the request is served
//! uncached and both counts stay zero.
//!
//! Needs a real key in `ANTHROPIC_API_KEY` and network access. Run with:
//!
//! ```text
//! cargo run --example cached_request --features models,anthropic
//! ```

use std::sync::Arc;

use tapir_provider::{
    CachePolicy, CompletionOptions, Context, Message, Model, ModelRegistry,
    Provider, Usage,
};

/// Provider and Model this example talks to; caching is a base Anthropic
/// feature, so any current Claude model works.
const PROVIDER: &str = "anthropic";
const MODEL: &str = "claude-haiku-4-5";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("set ANTHROPIC_API_KEY to run this example");
        return Ok(());
    }

    // Resolve the Model from the catalog so its cache rates drive the cost, and
    // build a live Provider for it over the default reqwest transport.
    let registry = ModelRegistry::load(None, None)?;
    let entry = registry
        .find(PROVIDER, MODEL)
        .ok_or("claude-haiku-4-5 not in this build's catalog")?;
    let model = entry.model.clone();
    let http = Arc::new(tapir_provider::http::ReqwestClient::new());
    let provider = registry.create_provider(PROVIDER, MODEL, http)?;

    // The shared prefix. A cache breakpoint only pays off above Anthropic's
    // minimum cacheable size, so pad a real instruction with a large synthetic
    // reference document that both turns carry verbatim.
    let system = format!(
        "You are a support agent for the ACME API. Answer only from the \
        reference below, and cite the rule number you used.\n\n{}",
        reference_document(),
    );

    // Standard caching: the Provider marks the last system block (and the final
    // message block), so the large system prefix above is the stable span the
    // follow-up reads back. The same options drive both turns.
    let opts = CompletionOptions::default()
        .with_cache(CachePolicy::Standard)
        .with_max_tokens(256);

    // Turn one: nothing is cached yet, so the Provider processes the whole
    // prompt and writes the prefix into the cache.
    let mut ctx =
        Context::new(vec![Message::user("What is rule 7, in one sentence?")])
            .with_system(system);

    println!("--- turn 1 (cache write) ---");
    let first = provider.complete(&ctx, &opts).await?;
    report(&model, &first.usage);
    println!("answer: {}", first.text_content());

    // Carry the reply forward and ask a follow-up. The system prefix is
    // unchanged, so this turn reads it from the cache instead of reprocessing.
    ctx.push(Message::Assistant(first.clone()));
    ctx.push(Message::user("And rule 12?"));

    println!("\n--- turn 2 (cache read) ---");
    let second = provider.complete(&ctx, &opts).await?;
    report(&model, &second.usage);
    println!("answer: {}", second.text_content());

    Ok(())
}

/// Print a turn's cache-token counts and its cost, so the write-then-read
/// pattern and the saving it buys are both visible.
fn report(model: &Model, usage: &Usage) {
    println!(
        "tokens: {} input, {} output, {} cache-write, {} cache-read",
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_write_tokens,
        usage.cache_read_tokens,
    );
    let cost = model.calculate_cost(
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
    );
    println!("cost: ${cost:.6}");
}

/// A large synthetic reference document, big enough to clear the cache floor.
///
/// The content is filler; only its size and its byte-for-byte stability across
/// turns matter, since the cache keys on the exact prefix.
fn reference_document() -> String {
    let mut doc = String::from("ACME API support rules:\n");
    for rule in 1..=60 {
        doc.push_str(&format!(
            "Rule {rule}: Requests to endpoint group {rule} must include a \
            valid API key in the Authorization header, respect the group's \
            rate limit, retry idempotent calls with exponential backoff on \
            HTTP 429 or 503, and treat any 4xx other than 429 as a permanent \
            client error that must not be retried.\n",
        ));
    }
    doc
}
