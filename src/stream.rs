// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The streaming vocabulary: incremental [`StreamEvent`]s and the
//! [`StreamAccumulator`] that folds them back into an [`AssistantMessage`].
//!
//! A Provider that streams surfaces a completion as an ordered sequence of
//! [`StreamEvent`]s. Each event is Provider-neutral and carries a content
//! `index` so deltas can be correlated to the content block they belong to. A
//! caller that only wants the final answer can ignore every delta and fold the
//! whole stream through a [`StreamAccumulator`], arriving at the same
//! [`AssistantMessage`] the non-streaming path would produce.

use crate::error::Error;
use crate::http::ByteStream;
use crate::message::{AssistantMessage, ContentPart};
use crate::response::{FinishReason, Usage, mint_call_id};
use crate::sse::{SseDecoder, SseEvent};
use std::collections::{BTreeMap, VecDeque};
use std::pin::Pin;
use std::task::Poll;

/// A boxed, ordered stream of [`StreamEvent`]s returned by a streaming Provider.
///
/// The trait object keeps the [`Provider`](crate::provider::Provider) trait
/// object-safe: the concrete stream type is erased behind a box so heterogeneous
/// Providers can be held together and supplied by third parties.
pub type StreamEvents = Pin<
    Box<dyn futures_core::Stream<Item = Result<StreamEvent, Error>> + Send>,
>;

/// One incremental event in a streamed completion.
///
/// Text, reasoning, and tool-call events carry an `index` identifying the
/// content block they belong to, so out-of-band events (usage, done) and
/// interleaved blocks stay correlated. Payloads a Provider does not model
/// surface as [`StreamEvent::Unknown`] rather than failing the stream.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum StreamEvent {
    /// The message has begun; no content has arrived yet.
    MessageStart,
    /// A text block has opened at `index`; its deltas follow.
    TextStart {
        /// The content block that opened.
        index: usize,
    },
    /// A run of generated text for the block at `index`.
    TextDelta {
        /// The content block this delta extends.
        index: usize,
        /// The text fragment.
        text: String,
    },
    /// The text block at `index` is complete.
    TextEnd {
        /// The content block that finished.
        index: usize,
    },
    /// A thinking block has opened at `index`; its deltas follow.
    ThinkingStart {
        /// The content block that opened.
        index: usize,
    },
    /// A run of model reasoning ("thinking") for the block at `index`.
    ThinkingDelta {
        /// The content block this delta extends.
        index: usize,
        /// The reasoning fragment.
        text: String,
    },
    /// The thinking block at `index` is complete, carrying the Provider's
    /// replay signature when one was supplied.
    ThinkingEnd {
        /// The content block that finished.
        index: usize,
        /// The opaque signature for replaying this reasoning, if any.
        signature: Option<String>,
    },
    /// A tool call has begun at `index`, with its id and tool name.
    ToolCallStart {
        /// The content block this tool call occupies.
        index: usize,
        /// The Provider's id for this tool call.
        id: String,
        /// The name of the tool being invoked.
        name: String,
    },
    /// A fragment of the JSON arguments for the tool call at `index`.
    ToolCallDelta {
        /// The content block this delta extends.
        index: usize,
        /// A fragment of the arguments JSON, to be concatenated in order.
        partial_json: String,
    },
    /// The tool call at `index` is complete.
    ToolCallEnd {
        /// The content block that finished.
        index: usize,
    },
    /// Updated token accounting for the message so far.
    Usage(Usage),
    /// The stream is finished, carrying the finish reason and final usage.
    Done {
        /// Why generation stopped.
        finish_reason: FinishReason,
        /// Final token accounting.
        usage: Usage,
    },
    /// A payload the Provider recognized as an event but does not model, kept as
    /// its raw JSON so nothing is silently dropped.
    Unknown(serde_json::Value),
}

/// Maps a Provider's decoded SSE events into the neutral [`StreamEvent`]
/// vocabulary.
///
/// This is the one seam a streaming Provider supplies. Given a decoded
/// [`SseEvent`], it produces zero or more neutral [`StreamEvent`]s, threading
/// whatever per-stream state the mapping needs (running usage, the finish
/// reason, which content-block indices have opened) across calls via `&mut
/// self`. Everything else in the streaming pipeline — reassembling events off
/// the byte chunks, buffering the ones a single chunk expands into, propagating
/// transport errors — lives in [`SseEventStream`] and does not vary by Provider.
pub(crate) trait StreamNormalizer {
    /// Map one decoded SSE event to zero or more neutral [`StreamEvent`]s.
    fn normalize(&mut self, event: &SseEvent) -> Vec<StreamEvent>;
}

/// Adapts a byte stream into ordered [`StreamEvent`]s by driving a
/// [`StreamNormalizer`] over the events an [`SseDecoder`] reassembles.
///
/// It owns the pipeline for one streamed completion: the [`SseDecoder`] that
/// reassembles events off the byte chunks, the [`StreamNormalizer`] `N` that
/// maps each SSE event into the neutral vocabulary, and a queue holding the
/// events a single chunk expanded into but that have not been yielded yet. The
/// normalizer is the only part that varies by Provider, so it is injected: each
/// Provider hands in its own, and a test can drive the pipeline with a scripted
/// one.
pub(crate) struct SseEventStream<N> {
    /// The response body, streamed as byte chunks.
    bytes: ByteStream,
    /// Reassembles SSE events straddling chunk boundaries.
    decoder: SseDecoder,
    /// Maps SSE events to the neutral vocabulary.
    normalizer: N,
    /// Events decoded but not yet yielded to the caller.
    pending: VecDeque<StreamEvent>,
    /// Whether the byte stream has ended (or errored).
    finished: bool,
}

impl<N: StreamNormalizer> SseEventStream<N> {
    /// Drive `normalizer` over the SSE events decoded from `bytes`.
    pub(crate) fn new(bytes: ByteStream, normalizer: N) -> Self {
        Self {
            bytes,
            decoder: SseDecoder::new(),
            normalizer,
            pending: VecDeque::new(),
            finished: false,
        }
    }
}

impl<N: StreamNormalizer + Unpin> futures_core::Stream for SseEventStream<N> {
    type Item = Result<StreamEvent, Error>;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(event) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(event)));
            }
            if this.finished {
                return Poll::Ready(None);
            }

            match this.bytes.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    for sse in this.decoder.push(&chunk) {
                        this.pending.extend(this.normalizer.normalize(&sse));
                    }
                }
                Poll::Ready(Some(Err(err))) => {
                    this.finished = true;
                    return Poll::Ready(Some(Err(err)));
                }
                Poll::Ready(None) => {
                    this.finished = true;
                    if let Some(sse) = this.decoder.finish() {
                        this.pending.extend(this.normalizer.normalize(&sse));
                    }
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// A tool call being reassembled from streamed fragments.
///
/// The start event fixes the name and native id; the argument JSON arrives as
/// zero or more delta fragments that concatenate in arrival order.
#[derive(Debug, Default)]
struct PartialToolCall {
    /// The Provider's native id, if the start event carried a non-empty one.
    native_id: Option<String>,
    /// The tool name from the start event.
    name: String,
    /// The argument JSON, concatenated from delta fragments.
    json: String,
}

/// Folds a stream of [`StreamEvent`]s into an [`AssistantMessage`].
///
/// The accumulator is Provider-neutral: it concatenates text and thinking deltas
/// per content block, reassembles tool-call fragments keyed by content index,
/// tracks the running usage, and records the finish reason from the terminal
/// [`StreamEvent::Done`]. Content parts (text, thinking, and tool calls) are
/// emitted in `index` order, matching the non-streaming path.
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    /// Text accumulated per content-block index, kept ordered by index.
    text: BTreeMap<usize, String>,
    /// Thinking accumulated per content-block index, kept ordered by index.
    thinking: BTreeMap<usize, String>,
    /// Replay signatures for thinking blocks, keyed by content-block index.
    thinking_signatures: BTreeMap<usize, String>,
    /// Tool calls being reassembled, keyed and ordered by content-block index.
    tool_calls: BTreeMap<usize, PartialToolCall>,
    /// The most recent usage seen, from a `Usage` or `Done` event.
    usage: Usage,
    /// The finish reason from the terminal `Done` event, if seen.
    finish_reason: Option<FinishReason>,
}

impl StreamAccumulator {
    /// A fresh accumulator with no events folded in.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold one event into the running completion.
    pub fn push(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::TextDelta { index, text } => {
                self.text.entry(*index).or_default().push_str(text);
            }
            StreamEvent::ThinkingDelta { index, text } => {
                self.thinking.entry(*index).or_default().push_str(text);
            }
            StreamEvent::ThinkingEnd {
                index,
                signature: Some(signature),
            } => {
                self.thinking_signatures.insert(*index, signature.clone());
            }
            StreamEvent::ToolCallStart { index, id, name } => {
                let call = self.tool_calls.entry(*index).or_default();
                call.native_id = (!id.is_empty()).then(|| id.clone());
                call.name = name.clone();
            }
            StreamEvent::ToolCallDelta {
                index,
                partial_json,
            } => {
                self.tool_calls
                    .entry(*index)
                    .or_default()
                    .json
                    .push_str(partial_json);
            }
            StreamEvent::Usage(usage) => self.usage = *usage,
            StreamEvent::Done {
                finish_reason,
                usage,
            } => {
                self.usage = *usage;
                self.finish_reason = Some(finish_reason.clone());
            }
            // MessageStart, the *Start/*End brackets, ToolCallEnd, and unknown
            // payloads carry nothing the folded completion needs beyond what is
            // handled above.
            _ => {}
        }
    }

    /// Fold every event from an iterator, then finish.
    pub fn fold<'a, I>(iter: I) -> AssistantMessage
    where
        I: IntoIterator<Item = &'a StreamEvent>,
    {
        let mut acc = Self::new();
        for event in iter {
            acc.push(event);
        }
        acc.finish()
    }

    /// Produce the [`AssistantMessage`] the folded events describe.
    ///
    /// Content parts are ordered by their content-block index, so text and tool
    /// calls interleave the way they arrived. Absent a terminal `Done` event,
    /// the finish reason falls back to an empty [`FinishReason::Other`], the
    /// same placeholder the non-streaming path uses for a missing stop reason.
    #[must_use]
    pub fn finish(self) -> AssistantMessage {
        // Merge text, thinking, and tool-call blocks into one index-ordered part
        // list, so the settled content matches the order the blocks streamed in.
        let mut parts: BTreeMap<usize, ContentPart> = BTreeMap::new();
        for (index, text) in self.text {
            parts.insert(index, ContentPart::Text(text));
        }
        let mut signatures = self.thinking_signatures;
        for (index, text) in self.thinking {
            parts.insert(
                index,
                ContentPart::Thinking {
                    text,
                    signature: signatures.remove(&index),
                },
            );
        }
        for (index, call) in self.tool_calls {
            parts.insert(
                index,
                ContentPart::ToolCall {
                    id: call.native_id.unwrap_or_else(mint_call_id),
                    name: call.name,
                    arguments: parse_arguments(&call.json),
                },
            );
        }
        AssistantMessage {
            content: parts.into_values().collect(),
            usage: self.usage,
            finish_reason: self
                .finish_reason
                .unwrap_or(FinishReason::Other(String::new())),
            // The stream has no single raw body; the escape hatch is absent.
            raw: None,
        }
    }
}

/// Parse reassembled tool-call arguments, defaulting an empty or unparsable
/// fragment to an empty JSON object so a caller always gets a value.
fn parse_arguments(json: &str) -> serde_json::Value {
    if json.trim().is_empty() {
        return serde_json::json!({});
    }
    serde_json::from_str(json).unwrap_or_else(|_| serde_json::json!({}))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tool calls in a folded message, as `(id, name, arguments)` tuples in
    /// content order.
    fn tool_calls(
        message: &AssistantMessage,
    ) -> Vec<(&str, &str, &serde_json::Value)> {
        message
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                } => Some((id.as_str(), name.as_str(), arguments)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn folds_ordered_text_deltas_into_one_completion() {
        let events = vec![
            StreamEvent::MessageStart,
            StreamEvent::TextDelta {
                index: 0,
                text: "Hello".to_owned(),
            },
            StreamEvent::TextDelta {
                index: 0,
                text: ", world".to_owned(),
            },
            StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    input_tokens: 3,
                    output_tokens: 4,
                    ..Usage::default()
                },
            },
        ];

        let completion = StreamAccumulator::fold(&events);
        assert_eq!(completion.text_content(), "Hello, world");
        assert_eq!(completion.usage.input_tokens, 3);
        assert_eq!(completion.usage.output_tokens, 4);
        assert_eq!(completion.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn concatenates_multiple_blocks_in_index_order() {
        // Deltas arrive interleaved but must join in index order.
        let events = vec![
            StreamEvent::TextDelta {
                index: 1,
                text: " second".to_owned(),
            },
            StreamEvent::TextDelta {
                index: 0,
                text: "first".to_owned(),
            },
            StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
            },
        ];
        assert_eq!(
            StreamAccumulator::fold(&events).text_content(),
            "first second"
        );
    }

    #[test]
    fn thinking_does_not_shape_the_text_but_is_retained() {
        let events = vec![
            StreamEvent::ThinkingStart { index: 0 },
            StreamEvent::ThinkingDelta {
                index: 0,
                text: "thinking...".to_owned(),
            },
            StreamEvent::ThinkingEnd {
                index: 0,
                signature: Some("sig-1".to_owned()),
            },
            StreamEvent::ToolCallStart {
                index: 1,
                id: "call_1".to_owned(),
                name: "get_weather".to_owned(),
            },
            StreamEvent::ToolCallDelta {
                index: 1,
                partial_json: "{\"city\":".to_owned(),
            },
            StreamEvent::ToolCallEnd { index: 1 },
            StreamEvent::Done {
                finish_reason: FinishReason::ToolUse,
                usage: Usage::default(),
            },
        ];
        let completion = StreamAccumulator::fold(&events);
        // Thinking is not text, but it is retained on the settled message.
        assert_eq!(completion.text_content(), "");
        assert_eq!(completion.thinking_content(), "thinking...");
        assert_eq!(completion.finish_reason, FinishReason::ToolUse);
        // The thinking part comes before the tool call, in index order, and
        // carries the replay signature.
        assert_eq!(
            completion.content[0],
            ContentPart::Thinking {
                text: "thinking...".to_owned(),
                signature: Some("sig-1".to_owned()),
            }
        );
    }

    #[test]
    fn bracketing_events_do_not_add_content() {
        // Text *Start/*End brackets frame the deltas but add no content of
        // their own; only the deltas between them fold into the message.
        let events = vec![
            StreamEvent::TextStart { index: 0 },
            StreamEvent::TextDelta {
                index: 0,
                text: "hi".to_owned(),
            },
            StreamEvent::TextEnd { index: 0 },
            StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage::default(),
            },
        ];
        let completion = StreamAccumulator::fold(&events);
        assert_eq!(
            completion.content,
            vec![ContentPart::Text("hi".to_owned())]
        );
    }

    #[test]
    fn reassembles_partial_tool_call_fragments() {
        // Arguments arrive split across several deltas and must concatenate.
        let events = vec![
            StreamEvent::ToolCallStart {
                index: 0,
                id: "toolu_1".to_owned(),
                name: "get_weather".to_owned(),
            },
            StreamEvent::ToolCallDelta {
                index: 0,
                partial_json: "{\"city\":".to_owned(),
            },
            StreamEvent::ToolCallDelta {
                index: 0,
                partial_json: "\"Paris\"}".to_owned(),
            },
            StreamEvent::ToolCallEnd { index: 0 },
            StreamEvent::Done {
                finish_reason: FinishReason::ToolUse,
                usage: Usage::default(),
            },
        ];
        let completion = StreamAccumulator::fold(&events);
        let calls = tool_calls(&completion);
        assert_eq!(calls.len(), 1);
        let (id, name, arguments) = calls[0];
        // The native id carries straight through as the call's stable handle.
        assert_eq!(id, "toolu_1");
        assert_eq!(name, "get_weather");
        assert_eq!(arguments, &serde_json::json!({"city": "Paris"}));
    }

    #[test]
    fn interleaved_tool_calls_reassemble_in_index_order() {
        let events = vec![
            StreamEvent::ToolCallStart {
                index: 1,
                id: "toolu_b".to_owned(),
                name: "b".to_owned(),
            },
            StreamEvent::ToolCallStart {
                index: 0,
                id: "toolu_a".to_owned(),
                name: "a".to_owned(),
            },
            StreamEvent::ToolCallDelta {
                index: 0,
                partial_json: "{\"x\":1}".to_owned(),
            },
            StreamEvent::ToolCallDelta {
                index: 1,
                partial_json: "{\"y\":2}".to_owned(),
            },
            StreamEvent::Done {
                finish_reason: FinishReason::ToolUse,
                usage: Usage::default(),
            },
        ];
        let completion = StreamAccumulator::fold(&events);
        let calls = tool_calls(&completion);
        let names: Vec<_> = calls.iter().map(|(_, name, _)| *name).collect();
        assert_eq!(names, vec!["a", "b"]);
        assert_eq!(calls[0].2, &serde_json::json!({"x": 1}));
    }

    #[test]
    fn tool_call_without_native_id_still_gets_a_minted_id() {
        // A Provider that omits the native id leaves it empty on the start event.
        let events = vec![
            StreamEvent::ToolCallStart {
                index: 0,
                id: String::new(),
                name: "search".to_owned(),
            },
            StreamEvent::ToolCallDelta {
                index: 0,
                partial_json: "{}".to_owned(),
            },
            StreamEvent::Done {
                finish_reason: FinishReason::ToolUse,
                usage: Usage::default(),
            },
        ];
        let completion = StreamAccumulator::fold(&events);
        let calls = tool_calls(&completion);
        // No native id, so the call still gets a minted, non-empty handle.
        assert!(!calls[0].0.is_empty());
    }

    #[test]
    fn usage_event_updates_running_accounting() {
        let events = vec![
            StreamEvent::Usage(Usage {
                input_tokens: 10,
                output_tokens: 2,
                ..Usage::default()
            }),
            StreamEvent::Done {
                finish_reason: FinishReason::Stop,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 7,
                    ..Usage::default()
                },
            },
        ];
        // Done carries the final, authoritative usage.
        assert_eq!(StreamAccumulator::fold(&events).usage.output_tokens, 7);
    }

    #[test]
    fn missing_done_falls_back_to_an_empty_other_reason() {
        let events = vec![StreamEvent::TextDelta {
            index: 0,
            text: "partial".to_owned(),
        }];
        let completion = StreamAccumulator::fold(&events);
        assert_eq!(completion.text_content(), "partial");
        assert_eq!(
            completion.finish_reason,
            FinishReason::Other(String::new())
        );
    }

    /// A scripted [`StreamNormalizer`] mapping each decoded [`SseEvent`] to a
    /// single [`StreamEvent::TextDelta`] carrying its `data`, so a test reads the
    /// reassembled events straight off the driver's output.
    struct EchoNormalizer;

    impl StreamNormalizer for EchoNormalizer {
        fn normalize(&mut self, event: &SseEvent) -> Vec<StreamEvent> {
            vec![StreamEvent::TextDelta {
                index: 0,
                text: event.data.clone(),
            }]
        }
    }

    /// Drive the pipeline over `chunks`, returning the `data` payloads the driver
    /// surfaced as text deltas.
    async fn drive(chunks: Vec<Result<Vec<u8>, Error>>) -> Vec<String> {
        use futures_util::StreamExt;
        let bytes: ByteStream = futures_util::stream::iter(chunks).boxed();
        SseEventStream::new(bytes, EchoNormalizer)
            .map(|event| match event.unwrap() {
                StreamEvent::TextDelta { text, .. } => text,
                other => panic!("unexpected event: {other:?}"),
            })
            .collect()
            .await
    }

    #[tokio::test]
    async fn driver_reassembles_chunk_split_events_and_flushes_the_final_one() {
        // The first event straddles two chunks; the last carries no terminating
        // blank line, so only the decoder's `finish` surfaces it.
        let texts = drive(vec![
            Ok(b"data: he".to_vec()),
            Ok(b"llo\n\ndata: wor".to_vec()),
            Ok(b"ld\n".to_vec()),
        ])
        .await;
        assert_eq!(texts, vec!["hello".to_owned(), "world".to_owned()]);
    }

    #[tokio::test]
    async fn driver_yields_decoded_events_then_propagates_a_transport_error() {
        use crate::error::ErrorKind;
        use futures_util::StreamExt;
        let bytes: ByteStream = futures_util::stream::iter(vec![
            Ok(b"data: one\n\n".to_vec()),
            Err(Error::new(ErrorKind::Transport, "boom")),
        ])
        .boxed();
        let mut stream = SseEventStream::new(bytes, EchoNormalizer);

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(
            first,
            StreamEvent::TextDelta {
                index: 0,
                text: "one".to_owned(),
            }
        );
        // The error surfaces once, then the stream is done.
        assert!(stream.next().await.unwrap().is_err());
        assert!(stream.next().await.is_none());
    }
}
