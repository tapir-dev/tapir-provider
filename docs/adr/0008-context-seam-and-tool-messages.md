# Context/options seam and tool calls & results in the message model

The `Provider` trait now takes a `Context` (system prompt, conversation, tools) plus a separate `CompletionOptions` (temperature, max tokens, tool choice), both by reference, and returns an `AssistantMessage` that is itself a `Message` variant. We split the old `CompletionRequest`/`CompletionResponse` pair this way — retiring both — so a completion drops straight back into `Context.messages` for the next turn, and so tool calls and their results can live in the conversation: `Message` becomes an enum (`User` | `Assistant` | `ToolResult`), `ContentPart` gains a `ToolCall` variant, and a `ToolResult` message carries the result back to the model. Without this the execute-tool -> return-result -> continue loop central to any agentic workflow cannot be expressed at all.

## Considered Options

- **Additive `Context` over the existing `CompletionRequest` seam, tool results faked as user text.** Rejected: it teaches the wrong pattern and leaves the message model unable to represent a tool result honestly.
- **Keep `CompletionResponse` separate with a `From` conversion into `Message`.** Rejected: the response *is* the message you push back; one unified `AssistantMessage` removes the conversion and the second type.

## Consequences

- Breaking change to the `Provider` trait, both Providers, `retry.rs`, `StreamAccumulator::finish()`, and the crate's re-exports. Taken now, pre-1.0, with no external consumers, rather than through a deprecation cycle.
- `Eq` is dropped from `ContentPart`/`Message` because `ToolCall.arguments` is a `serde_json::Value`; `PartialEq` remains, matching the already-`PartialEq`-only tool-call types.
- The `System` role is gone: a system prompt lives only on `Context.system_prompt`, so it cannot be interleaved between turns.
- Reasoning stays stream-only — it is not retained as a stored content block on the settled `AssistantMessage`.
- Both `openai` and `anthropic` carry correct tool-call and tool-result serialization: OpenAI sends assistant `tool_calls` plus a `tool` role message; Anthropic sends `tool_use` blocks plus a `user` message carrying a `tool_result` block. Each is covered by a wire round-trip test.
