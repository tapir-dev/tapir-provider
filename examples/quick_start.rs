// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Streaming tool-calling workflow against the real OpenAI API.
//!
//! Offers the model one tool, `get_time`, streams the first turn and prints the
//! text deltas as they arrive, detects the tool call, executes it, pushes the
//! result back into the conversation, then calls the Provider again for the
//! continuation. Finally prints token totals and the computed cost.
//!
//! Needs a real key in `OPENAI_API_KEY` and network access. Run with:
//!
//! ```text
//! cargo run --example quick_start --features models,openai
//! ```

use std::sync::Arc;

use futures_util::StreamExt;
use tapir_provider::{
    AssistantMessage, CompletionOptions, ContentPart, Context, Message,
    ModelRegistry, Provider, StreamAccumulator, ToolChoice, ToolDefinition,
    Usage,
};

/// The one tool this example offers the model.
const TOOL_NAME: &str = "get_time";
/// Provider and Model this example talks to.
const PROVIDER: &str = "openai";
const MODEL: &str = "gpt-4o-mini";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("set OPENAI_API_KEY to run this example");
        return Ok(());
    }

    // Resolve the Model from the catalog so its cost data drives the total, and
    // build a live Provider for it over the default reqwest transport.
    let registry = ModelRegistry::load(None, None)?;
    let entry = registry
        .find(PROVIDER, MODEL)
        .ok_or("gpt-4o-mini not in this build's catalog")?;
    let model = entry.model.clone();
    let http = Arc::new(tapir_provider::http::ReqwestClient::new());
    let provider = registry.create_provider(PROVIDER, MODEL, http)?;

    // One tool the model may call: a no-argument clock.
    let tools = vec![ToolDefinition::new(
        TOOL_NAME,
        "Get the current time as an ISO-8601 string.",
        serde_json::json!({"type": "object", "properties": {}}),
    )];
    let mut ctx = Context::new(vec![Message::user(
        "What time is it? Use the get_time tool.",
    )])
    .with_tools(tools);
    let opts = CompletionOptions::default()
        .with_tool_choice(ToolChoice::Auto)
        .with_max_tokens(256);

    // First turn, streamed: print text deltas live and fold the events back into
    // the settled AssistantMessage.
    println!("--- first turn (streaming) ---");
    let reply = stream_turn(&provider, &ctx, &opts).await?;
    let mut total = reply.usage;

    // Append the model's reply, execute any tool call, and push the result.
    ctx.push(Message::Assistant(reply.clone()));
    let mut made_a_tool_call = false;
    for part in reply.tool_calls() {
        let ContentPart::ToolCall { id, name, .. } = part else {
            continue;
        };
        made_a_tool_call = true;
        let result = execute_tool(name);
        println!("executed {name} -> {result}");
        ctx.push(Message::tool_result(id, name, result));
    }

    // Continue only if the model actually asked for the tool.
    if made_a_tool_call {
        println!("--- continuation (non-streaming) ---");
        let continuation = provider.complete(&ctx, &opts).await?;
        println!("{}", continuation.text_content());
        total = Usage {
            input_tokens: total.input_tokens + continuation.usage.input_tokens,
            output_tokens: total.output_tokens
                + continuation.usage.output_tokens,
        };
    }

    let cost =
        model.calculate_cost(total.input_tokens, total.output_tokens, 0, 0);
    println!("--- totals ---");
    println!(
        "tokens: {} in, {} out",
        total.input_tokens, total.output_tokens
    );
    println!("cost: ${cost:.6}");

    Ok(())
}

/// Stream one turn, printing text deltas as they arrive, and return the settled
/// [`AssistantMessage`] the events fold into.
async fn stream_turn(
    provider: &dyn Provider,
    ctx: &Context,
    opts: &CompletionOptions,
) -> Result<AssistantMessage, Box<dyn std::error::Error>> {
    let mut stream = provider.complete_stream(ctx, opts).await?;
    let mut accumulator = StreamAccumulator::new();
    use tapir_provider::StreamEvent;
    while let Some(event) = stream.next().await {
        let event = event?;
        if let StreamEvent::TextDelta { text, .. } = &event {
            print!("{text}");
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
        accumulator.push(&event);
    }
    println!();
    Ok(accumulator.finish())
}

/// Run the `get_time` tool. The only tool this example offers.
fn execute_tool(name: &str) -> String {
    match name {
        TOOL_NAME => "2026-09-05T12:00:00Z".to_owned(),
        other => format!("unknown tool: {other}"),
    }
}
