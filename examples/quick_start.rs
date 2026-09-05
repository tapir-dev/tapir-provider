// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Streaming with extended thinking against the real Anthropic API.
//!
//! Asks a reasoning-capable Claude model to think before answering, streams the
//! turn, and prints the thinking and answer as they arrive. The reasoning is
//! bracketed by `ThinkingStart`/`ThinkingEnd` and the answer by
//! `TextStart`/`TextEnd`; the deltas in between are folded back into the settled
//! `AssistantMessage`, which retains the thinking alongside the text. Finally
//! prints token totals and the computed cost.
//!
//! Needs a real key in `ANTHROPIC_API_KEY` and network access. Run with:
//!
//! ```text
//! cargo run --example quick_start --features models,anthropic
//! ```

use std::io::Write;
use std::sync::Arc;

use futures_util::StreamExt;
use tapir_provider::{
    AssistantMessage, CompletionOptions, ContentPart, Context, Message,
    ModelRegistry, Provider, StreamAccumulator, StreamEvent, ThinkingLevel,
};

/// Provider and Model this example talks to; the model must support reasoning.
const PROVIDER: &str = "anthropic";
const MODEL: &str = "claude-haiku-4-5";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("ANTHROPIC_API_KEY").is_err() {
        eprintln!("set ANTHROPIC_API_KEY to run this example");
        return Ok(());
    }

    // Resolve the Model from the catalog so its cost data drives the total, and
    // build a live Provider for it over the default reqwest transport.
    let registry = ModelRegistry::load(None, None)?;
    let entry = registry
        .find(PROVIDER, MODEL)
        .ok_or("claude-haiku-4-5 not in this build's catalog")?;
    let model = entry.model.clone();
    let http = Arc::new(tapir_provider::http::ReqwestClient::new());
    let provider = registry.create_provider(PROVIDER, MODEL, http)?;

    let prompt = "A farmer has 17 sheep. All but 9 run away. \
        How many are left? Think it through, then answer.";
    // The system prompt steers the turn out of band from the messages.
    // `with_system` takes anything `Into<SystemPrompt>`, so a `&str` works as-is.
    let ctx = Context::new(vec![Message::user(prompt)]).with_system(
        "You are a careful reasoner. Answer with a single number.",
    );
    // Ask for extended thinking; a token budget is derived from the level, and
    // the Provider leaves room for the answer beyond it.
    let opts = CompletionOptions::default()
        .with_thinking(ThinkingLevel::Low)
        .with_max_tokens(1024);

    // Stream the turn: print thinking and answer live, then fold the events back
    // into the settled AssistantMessage.
    let reply = stream_turn(&provider, &ctx, &opts).await?;

    // The folded message retains the reasoning as a Thinking content part.
    println!("\n--- settled message ---");
    for part in &reply.content {
        match part {
            ContentPart::Thinking { text, signature } => {
                println!("thinking ({} chars)", text.len());
                if signature.is_some() {
                    println!("  (carries a replay signature)");
                }
            }
            ContentPart::Text(text) => println!("answer: {text}"),
            ContentPart::ToolCall { name, .. } => {
                println!("tool call: {name}");
            }
            ContentPart::Image(_) => println!("image"),
            _ => {}
        }
    }

    let cost = model.calculate_cost(
        reply.usage.input_tokens,
        reply.usage.output_tokens,
        0,
        0,
    );
    println!("--- totals ---");
    println!(
        "tokens: {} in, {} out",
        reply.usage.input_tokens, reply.usage.output_tokens
    );
    println!("cost: ${cost:.6}");

    Ok(())
}

/// A short name for a stream event, for the lifecycle trace.
fn event_name(event: &StreamEvent) -> &'static str {
    match event {
        StreamEvent::MessageStart => "MessageStart",
        StreamEvent::TextStart { .. } => "TextStart",
        StreamEvent::TextDelta { .. } => "TextDelta",
        StreamEvent::TextEnd { .. } => "TextEnd",
        StreamEvent::ThinkingStart { .. } => "ThinkingStart",
        StreamEvent::ThinkingDelta { .. } => "ThinkingDelta",
        StreamEvent::ThinkingEnd { .. } => "ThinkingEnd",
        StreamEvent::ToolCallStart { .. } => "ToolCallStart",
        StreamEvent::ToolCallDelta { .. } => "ToolCallDelta",
        StreamEvent::ToolCallEnd { .. } => "ToolCallEnd",
        StreamEvent::Usage(_) => "Usage",
        StreamEvent::Done { .. } => "Done",
        StreamEvent::Unknown(_) => "Unknown",
        _ => "Other",
    }
}

/// Stream one turn, printing thinking and answer deltas under section headers as
/// they arrive, and return the settled [`AssistantMessage`] the events fold into.
async fn stream_turn(
    provider: &dyn Provider,
    ctx: &Context,
    opts: &CompletionOptions,
) -> Result<AssistantMessage, Box<dyn std::error::Error>> {
    let mut stream = provider.complete_stream(ctx, opts).await?;
    let mut accumulator = StreamAccumulator::new();
    while let Some(event) = stream.next().await {
        let event = event?;
        match &event {
            StreamEvent::ThinkingStart { .. } => print!("\n[thinking]\n"),
            StreamEvent::TextStart { .. } => print!("\n[answer]\n"),
            StreamEvent::ThinkingDelta { text, .. }
            | StreamEvent::TextDelta { text, .. } => {
                print!("{text}");
                std::io::stdout().flush().ok();
            }
            // Trace every non-delta lifecycle event so the block structure is
            // visible; deltas are omitted here since they already stream above.
            other => eprintln!("[event] {}", event_name(other)),
        }
        accumulator.push(&event);
    }
    println!();
    Ok(accumulator.finish())
}
