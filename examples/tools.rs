// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The full tool-call loop, driven entirely offline.
//!
//! A tool is a Provider-neutral [`ToolDefinition`]: a name, a description, and a
//! JSON Schema for its arguments, carried verbatim. This example shows the two
//! ways to author that schema and then runs a complete turn against a scripted
//! Provider, so the loop is deterministic and needs no network or API key.
//!
//! It walks the whole shape:
//!
//!   1. Define one tool two ways — a hand-written `serde_json::json!` schema and
//!      a `schemars`-derived one from a typed args struct — and compare them. Both
//!      describe the same tool; the derived one renders the enum as `oneOf`, which
//!      is worth knowing (see `tool_schema`).
//!   2. The model asks to call the tool, at first with arguments that miss a
//!      required field.
//!   3. Validating those arguments against the typed struct fails, so the outcome
//!      goes back as a `ToolResult` flagged `is_error`.
//!   4. The model retries with corrected arguments; the tool runs and its result
//!      returns as a successful `ToolResult`.
//!   5. With the result in hand, the model answers in prose and the loop settles.
//!
//! The Provider here is a hand-written script — the point is the loop, not a
//! backend. `quick_start` covers a live Provider over the network. Run with:
//!
//! ```text
//! cargo run --example tools
//! ```

use async_trait::async_trait;
use schemars::JsonSchema;
use schemars::generate::{SchemaGenerator, SchemaSettings};
use serde::Deserialize;
use serde_json::Value;
use tapir_provider::{
    AssistantMessage, CompletionOptions, ContentPart, Context, Error,
    FinishReason, Message, Provider, ToolChoice, ToolDefinition,
    ToolResultMessage, Usage,
};

/// The typed arguments for `get_weather`.
///
/// One struct serves two roles: `Deserialize` makes it the validator for the
/// arguments the model generates (a failed parse *is* a validation failure), and
/// `JsonSchema` derives the tool's parameter schema so the definition and the
/// validation can never drift apart. Doc comments become schema descriptions.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WeatherArgs {
    /// City name or coordinates.
    city: String,
    /// Temperature unit; defaults to Celsius.
    #[serde(default)]
    units: Units,
}

/// The temperature unit a `get_weather` call may request.
#[derive(Debug, Clone, Copy, Default, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
enum Units {
    /// Degrees Celsius.
    #[default]
    Celsius,
    /// Degrees Fahrenheit.
    Fahrenheit,
}

/// Derive a provider-friendly JSON Schema from a typed args struct.
///
/// Uses OpenAPI 3 settings with subschemas inlined (no `$ref`, which several
/// providers reject) and drops the root `$schema`/`title`/`description` noise,
/// leaving a bare object schema the way a hand-written one would look.
///
/// One caveat the printed output makes visible: `schemars` renders a fieldless
/// enum as `oneOf` of single-value schemas, not the flat
/// `{"type":"string","enum":[...]}` the hand-written schema uses. Some providers
/// (notably Google) reject `oneOf`/`anyOf`/`const` in tool schemas, so the flat
/// form is the portable one — reach for a hand-written schema when a derived enum
/// needs to travel that far.
fn tool_schema<T: JsonSchema>() -> Value {
    let settings = SchemaSettings::openapi3().with(|s| {
        s.inline_subschemas = true;
        s.meta_schema = None;
    });
    let mut schema = SchemaGenerator::new(settings).into_root_schema_for::<T>();
    if let Some(object) = schema.as_object_mut() {
        object.remove("title");
        object.remove("description");
    }
    serde_json::to_value(schema).expect("a derived schema serializes to JSON")
}

/// A Provider that follows a fixed script instead of a network backend, so the
/// tool loop runs deterministically offline. It decides its next move purely
/// from the last message in the context.
struct ScriptedModel;

#[async_trait]
impl Provider for ScriptedModel {
    async fn complete(
        &self,
        ctx: &Context,
        _opts: &CompletionOptions,
    ) -> Result<AssistantMessage, Error> {
        let reply = match ctx.messages.last() {
            // The user just asked. Call the tool — but omit the required `city`
            // on purpose, to drive the validation-failure branch below.
            Some(Message::User { .. }) => tool_call_reply(
                "call_1",
                serde_json::json!({ "units": "celsius" }),
            ),
            // The last tool run errored (bad arguments): retry with a corrected
            // call that satisfies the schema.
            Some(Message::ToolResult(result)) if result.is_error => {
                tool_call_reply(
                    "call_2",
                    serde_json::json!({ "city": "Paris", "units": "celsius" }),
                )
            }
            // The tool succeeded: answer the user in prose and stop.
            Some(Message::ToolResult(result)) => AssistantMessage {
                content: vec![ContentPart::text(format!(
                    "The weather report says: {}",
                    first_text(&result.content).unwrap_or("(no text)")
                ))],
                usage: Usage {
                    input_tokens: 30,
                    output_tokens: 15,
                    ..Usage::default()
                },
                finish_reason: FinishReason::Stop,
                raw: None,
            },
            _ => AssistantMessage::text("Nothing to do."),
        };
        Ok(reply)
    }
}

/// A one-tool-call assistant reply asking to run `get_weather` with `arguments`.
fn tool_call_reply(id: &str, arguments: Value) -> AssistantMessage {
    AssistantMessage {
        content: vec![ContentPart::tool_call(id, "get_weather", arguments)],
        usage: Usage {
            input_tokens: 12,
            output_tokens: 8,
            ..Usage::default()
        },
        finish_reason: FinishReason::ToolUse,
        raw: None,
    }
}

/// The first text part of a content list, if any.
fn first_text(content: &[ContentPart]) -> Option<&str> {
    content.iter().find_map(|part| match part {
        ContentPart::Text(text) => Some(text.as_str()),
        _ => None,
    })
}

/// Validate and run a tool call, returning the text a `ToolResult` carries back.
///
/// Validation is a typed `serde_json::from_value`: the model's arguments either
/// deserialize into [`WeatherArgs`] or they do not, and a parse error is the
/// validation error. An `Err` here becomes a `ToolResult` flagged `is_error`, so
/// the model sees what went wrong and can retry.
fn run_tool(name: &str, arguments: &Value) -> Result<String, String> {
    match name {
        "get_weather" => {
            let args: WeatherArgs = serde_json::from_value(arguments.clone())
                .map_err(|e| {
                format!("invalid arguments for get_weather: {e}")
            })?;
            let (temp, unit) = match args.units {
                Units::Celsius => (18, 'C'),
                Units::Fahrenheit => (64, 'F'),
            };
            Ok(format!("It is {temp}°{unit} and clear in {}.", args.city))
        }
        other => Err(format!("unknown tool: {other}")),
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define the same tool two ways. The raw schema spells out the JSON the
    //    provider receives; the derived one generates it from the typed args
    //    struct, so definition and validation stay in lockstep (at the cost of a
    //    `oneOf` enum the hand-written form writes flat).
    let raw_tool = ToolDefinition::new(
        "get_weather",
        "Get current weather for a location",
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name or coordinates"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit; defaults to Celsius"
                }
            },
            "required": ["city"],
            "additionalProperties": false
        }),
    );
    let derived_tool = ToolDefinition::new(
        "get_weather",
        "Get current weather for a location",
        tool_schema::<WeatherArgs>(),
    );

    println!("hand-written json! schema:");
    println!("{}", serde_json::to_string_pretty(&raw_tool.input_schema)?);
    println!("\nschemars-derived schema:");
    println!(
        "{}",
        serde_json::to_string_pretty(&derived_tool.input_schema)?
    );

    // 2. Offer the derived tool to the model and open the turn.
    let provider = ScriptedModel;
    let mut ctx =
        Context::new(vec![Message::user("What's the weather in Paris?")])
            .with_tools(vec![derived_tool]);
    let opts = CompletionOptions::default().with_tool_choice(ToolChoice::Auto);

    // 3. Drive the loop: complete, run any tool calls, feed the results back, and
    //    repeat until the model answers without calling a tool. The turn cap
    //    guards against a model that never settles.
    println!("\n--- tool loop ---");
    for _turn in 0..8 {
        let reply = provider.complete(&ctx, &opts).await?;
        ctx.push(Message::Assistant(reply.clone()));

        let calls: Vec<ContentPart> = reply.tool_calls().cloned().collect();
        if calls.is_empty() {
            println!("\nfinal answer: {}", reply.text_content());
            return Ok(());
        }

        for call in calls {
            let ContentPart::ToolCall {
                id,
                name,
                arguments,
            } = call
            else {
                continue;
            };
            match run_tool(&name, &arguments) {
                Ok(output) => {
                    println!("[tool ok]    {name}({arguments}) -> {output}");
                    ctx.push(Message::tool_result(id, name, output));
                }
                Err(error) => {
                    println!("[tool error] {name}({arguments}): {error}");
                    ctx.push(Message::ToolResult(ToolResultMessage {
                        tool_call_id: id,
                        tool_name: name,
                        content: vec![ContentPart::text(error)],
                        is_error: true,
                    }));
                }
            }
        }
    }

    Err("tool loop did not settle within the turn cap".into())
}
