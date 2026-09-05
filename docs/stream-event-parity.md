# Streaming events

The streaming vocabulary this crate exposes as `StreamEvent`, and how the
per-block lifecycle folds back into a settled `AssistantMessage`.

## Event vocabulary

| `StreamEvent`   | Carries                     | Meaning                                             |
| --------------- | --------------------------- | -------------------------------------------------- |
| `MessageStart`  | —                           | The message has begun; no content yet.             |
| `TextStart`     | `index`                     | A text block opened at `index`.                    |
| `TextDelta`     | `index`, `text`             | A fragment of that text block.                     |
| `TextEnd`       | `index`                     | The text block at `index` is complete.             |
| `ThinkingStart` | `index`                     | A thinking (reasoning) block opened at `index`.    |
| `ThinkingDelta` | `index`, `text`             | A fragment of that thinking block.                 |
| `ThinkingEnd`   | `index`, `signature`        | Thinking complete, with a replay signature if any. |
| `ToolCallStart` | `index`, `id`, `name`       | A tool call opened at `index`.                     |
| `ToolCallDelta` | `index`, `partial_json`     | A fragment of the tool-call argument JSON.         |
| `ToolCallEnd`   | `index`                     | The tool call at `index` is complete.              |
| `Usage`         | `Usage`                     | Updated token accounting so far.                   |
| `Done`          | `finish_reason`, `usage`    | The stream finished, with terminal reason + usage. |
| `Unknown`       | `serde_json::Value`         | A recognized-but-unmodeled payload, kept raw.      |

## Block lifecycle

Text, thinking, and tool-call blocks each open with a `*Start`, extend through
zero or more `*Delta`s, and close with a `*End`, all sharing one content `index`.
Blocks correlate by `index`, so interleaved blocks and out-of-band events
(`Usage`, `Done`) stay unambiguous. A Provider whose wire protocol lacks explicit
block boundaries has its boundaries synthesized: the first delta opens the block
and the end of the stream closes it.

## Folding into a message

`StreamAccumulator` folds the events into the same `AssistantMessage` the
non-streaming path produces, with content parts ordered by `index`:

- Text deltas concatenate into a `ContentPart::Text`.
- Thinking deltas concatenate into a `ContentPart::Thinking`, carrying the
  replay `signature` from `ThinkingEnd` when one was supplied. Reasoning is
  retained on the settled message, so an extended-thinking reply can be replayed
  on a later turn.
- Tool-call fragments reassemble into a `ContentPart::ToolCall`; a native id
  passes through, and one is minted when the Provider supplies none.
- `Usage` and `Done` update the running token accounting; `Done` also records the
  finish reason. Absent a `Done`, the finish reason falls back to an empty
  `FinishReason::Other`.
- The `*Start`/`*End` brackets add no content of their own; only the deltas
  between them fold in.

## Design notes

- **Errors are not events.** A stream item is `Result<StreamEvent, Error>`, so a
  failure aborts the stream via `Result::Err` rather than a terminal error event.
- **`Usage` is its own event.** Token accounting is surfaced incrementally as a
  `Usage` event and again on `Done`, rather than riding on every event.
- **`Unknown` preserves the unmodeled.** A payload recognized as an event but not
  modeled is kept as raw JSON instead of failing the stream.
