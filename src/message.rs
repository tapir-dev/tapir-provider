// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The normalized message model shared across Providers.
//!
//! A conversation is a sequence of [`Message`]s. A [`Message`] is one of a
//! [user](Message::User) turn, an [`AssistantMessage`] the model produced, or a
//! [`ToolResultMessage`] carrying the outcome of a tool call back to the model.
//! Tool calls the model asks to make live as a [`ContentPart::ToolCall`] inside
//! an [`AssistantMessage`], so a completed reply drops straight back into the
//! conversation for the next turn. A system prompt is not a message; it lives on
//! [`Context::system_prompt`](crate::request::Context::system_prompt).

use crate::response::{FinishReason, Usage};
use serde::{Deserialize, Serialize};

/// The media type of an image, as carried to the Provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum MediaType {
    /// JPEG image (`image/jpeg`).
    Jpeg,
    /// PNG image (`image/png`).
    Png,
    /// GIF image (`image/gif`).
    Gif,
    /// WebP image (`image/webp`).
    Webp,
}

impl MediaType {
    /// The MIME type string as carried on the wire.
    #[must_use]
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
        }
    }
}

/// Where an image's bytes come from.
///
/// A URL is fetched by the Provider; base64 and raw bytes carry the image
/// inline and so also carry its [`MediaType`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ImageSource {
    /// A URL the Provider fetches the image from.
    Url(String),
    /// Base64-encoded image data with its media type.
    Base64 {
        /// The image's media type.
        media_type: MediaType,
        /// The base64-encoded image bytes.
        data: String,
    },
    /// Raw image bytes with their media type; encoded when sent.
    Bytes {
        /// The image's media type.
        media_type: MediaType,
        /// The raw image bytes.
        data: Vec<u8>,
    },
}

impl ImageSource {
    /// An image the Provider fetches from a URL.
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url(url.into())
    }

    /// An image carried inline as base64-encoded data.
    pub fn base64(media_type: MediaType, data: impl Into<String>) -> Self {
        Self::Base64 {
            media_type,
            data: data.into(),
        }
    }

    /// An image carried inline as raw bytes, encoded when sent.
    pub fn bytes(media_type: MediaType, data: impl Into<Vec<u8>>) -> Self {
        Self::Bytes {
            media_type,
            data: data.into(),
        }
    }
}

/// A single part of a [`Message`]'s content.
///
/// A message's content is a sequence of parts, so text, images, and tool calls
/// can be mixed within one message. Only [`Eq`] is dropped from this type — a
/// [`ToolCall`](ContentPart::ToolCall)'s `arguments` is a [`serde_json::Value`],
/// which is not `Eq` — but [`PartialEq`] is kept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ContentPart {
    /// A run of text.
    Text(String),
    /// An image, sourced from a URL, base64, or raw bytes.
    Image(ImageSource),
    /// A tool call the model asked to make.
    ToolCall {
        /// A stable handle for the call: the Provider's native id when the wire
        /// protocol supplies one, else an SDK-minted id.
        id: String,
        /// The name of the tool to invoke.
        name: String,
        /// The tool's arguments, as a JSON value.
        arguments: serde_json::Value,
    },
    /// A run of model reasoning ("thinking"), retained so a reply can be
    /// replayed to the Provider on a later turn.
    Thinking {
        /// The reasoning text.
        text: String,
        /// The Provider's opaque signature for replaying this reasoning, when
        /// one was supplied.
        signature: Option<String>,
    },
}

impl ContentPart {
    /// A text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// An image part from the given source.
    pub fn image(source: impl Into<ImageSource>) -> Self {
        Self::Image(source.into())
    }

    /// A tool-call part.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// A thinking part carrying reasoning text and an optional replay signature.
    pub fn thinking(
        text: impl Into<String>,
        signature: Option<String>,
    ) -> Self {
        Self::Thinking {
            text: text.into(),
            signature,
        }
    }
}

/// A reply produced by the model.
///
/// This is what a [`Provider`](crate::provider::Provider) returns and, being a
/// [`Message::Assistant`] variant, also a message in the conversation — so a
/// completion is appended for the next turn without conversion. Its content
/// mixes text, [`ToolCall`](ContentPart::ToolCall), and
/// [`Thinking`](ContentPart::Thinking) parts, so reasoning survives into the
/// settled reply and can be replayed on a later turn. [`raw`](Self::raw) is the
/// escape hatch to the Provider's untouched response body, absent on a streamed
/// reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// The reply's content, as an ordered list of parts.
    pub content: Vec<ContentPart>,
    /// Token accounting for the completion.
    pub usage: Usage,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// The Provider's untouched response body; `None` for a streamed reply,
    /// which has no single body.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<serde_json::Value>,
}

impl AssistantMessage {
    /// A text-only reply that stopped naturally, with zero usage and no raw
    /// body. A convenience for constructing a reply in tests and callers that
    /// synthesize one.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentPart::text(text)],
            usage: Usage::default(),
            finish_reason: FinishReason::Stop,
            raw: None,
        }
    }

    /// The reply's text parts concatenated in order; empty when the reply is
    /// only tool calls.
    #[must_use]
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The reply's thinking parts concatenated in order; empty when the reply
    /// carries no reasoning.
    #[must_use]
    pub fn thinking_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Thinking { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The tool calls the reply asks to make, in order.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ContentPart> {
        self.content
            .iter()
            .filter(|part| matches!(part, ContentPart::ToolCall { .. }))
    }
}

/// The outcome of running a [`ToolCall`](ContentPart::ToolCall), sent back to
/// the model as its own message.
///
/// It references the call by [`tool_call_id`](Self::tool_call_id), names the
/// tool, carries the result as content, and flags whether the run errored.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultMessage {
    /// The id of the [`ToolCall`](ContentPart::ToolCall) this result answers.
    pub tool_call_id: String,
    /// The name of the tool that ran.
    pub tool_name: String,
    /// The result content, as an ordered list of parts.
    pub content: Vec<ContentPart>,
    /// Whether the tool run errored.
    pub is_error: bool,
}

/// A single message in a conversation.
///
/// Only [`Eq`] is dropped (an [`AssistantMessage`] can carry a
/// [`ToolCall`](ContentPart::ToolCall), whose `arguments` is not `Eq`);
/// [`PartialEq`] is kept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Message {
    /// Input from the end user.
    User {
        /// The message content, as an ordered list of parts.
        content: Vec<ContentPart>,
    },
    /// A reply produced by the model.
    Assistant(AssistantMessage),
    /// The outcome of a tool call, carried back to the model.
    ToolResult(ToolResultMessage),
}

impl Message {
    /// Construct a text user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::User {
            content: vec![ContentPart::text(content)],
        }
    }

    /// Construct a user message from an explicit list of content parts.
    pub fn user_parts(parts: impl Into<Vec<ContentPart>>) -> Self {
        Self::User {
            content: parts.into(),
        }
    }

    /// Construct a text tool-result message answering the call with `id`.
    ///
    /// The result is flagged successful; set
    /// [`is_error`](ToolResultMessage::is_error) on the built
    /// [`ToolResultMessage`] to mark a failed run.
    pub fn tool_result(
        tool_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self::ToolResult(ToolResultMessage {
            tool_call_id: tool_call_id.into(),
            tool_name: tool_name.into(),
            content: vec![ContentPart::text(content)],
            is_error: false,
        })
    }

    /// Append a text part to a user message, returning it. A no-op on any other
    /// variant.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        if let Self::User { content } = &mut self {
            content.push(ContentPart::text(text));
        }
        self
    }

    /// Append an image part to a user message, returning it. A no-op on any
    /// other variant.
    #[must_use]
    pub fn with_image(mut self, source: impl Into<ImageSource>) -> Self {
        if let Self::User { content } = &mut self {
            content.push(ContentPart::image(source));
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_constructor_produces_a_single_text_part() {
        assert_eq!(
            Message::user("hi"),
            Message::User {
                content: vec![ContentPart::text("hi")],
            }
        );
    }

    #[test]
    fn builders_mix_text_and_image_parts() {
        let message = Message::user("look at this")
            .with_image(ImageSource::url("https://example.com/a.png"));
        assert_eq!(
            message,
            Message::User {
                content: vec![
                    ContentPart::text("look at this"),
                    ContentPart::image(ImageSource::url(
                        "https://example.com/a.png"
                    )),
                ],
            }
        );
    }

    #[test]
    fn tool_result_references_the_call_and_defaults_to_success() {
        let Message::ToolResult(result) =
            Message::tool_result("call_1", "get_time", "12:00")
        else {
            panic!("expected a tool-result message");
        };
        assert_eq!(result.tool_call_id, "call_1");
        assert_eq!(result.tool_name, "get_time");
        assert_eq!(result.content, vec![ContentPart::text("12:00")]);
        assert!(!result.is_error);
    }

    #[test]
    fn assistant_text_helper_concatenates_text_parts() {
        let message = AssistantMessage {
            content: vec![
                ContentPart::text("Hello"),
                ContentPart::tool_call("call_1", "noop", serde_json::json!({})),
                ContentPart::text(", world"),
            ],
            usage: Usage::default(),
            finish_reason: FinishReason::ToolUse,
            raw: None,
        };
        assert_eq!(message.text_content(), "Hello, world");
        assert_eq!(message.tool_calls().count(), 1);
    }

    #[test]
    fn media_types_map_to_their_mime_strings() {
        assert_eq!(MediaType::Jpeg.as_wire(), "image/jpeg");
        assert_eq!(MediaType::Png.as_wire(), "image/png");
        assert_eq!(MediaType::Gif.as_wire(), "image/gif");
        assert_eq!(MediaType::Webp.as_wire(), "image/webp");
    }
}
