// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The normalized message model shared across Providers.

use serde::{Deserialize, Serialize};

/// Who authored a [`Message`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Role {
    /// Instructions that steer the assistant, not part of the dialogue turns.
    System,
    /// Input from the end user.
    User,
    /// A reply produced by the model.
    Assistant,
}

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
/// A message is a sequence of parts, so text and images can be mixed within
/// one message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ContentPart {
    /// A run of text.
    Text(String),
    /// An image, sourced from a URL, base64, or raw bytes.
    Image(ImageSource),
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
}

/// A single normalized message: a [`Role`] and its content parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Who authored the message.
    pub role: Role,
    /// The message content, as an ordered list of parts.
    pub content: Vec<ContentPart>,
}

impl Message {
    /// Construct a text message with an explicit role.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::text(content)],
        }
    }

    /// Construct a message from an explicit list of content parts.
    pub fn from_parts(role: Role, parts: impl Into<Vec<ContentPart>>) -> Self {
        Self {
            role,
            content: parts.into(),
        }
    }

    /// Construct a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// Construct a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Construct an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// Append a text part, returning the message.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.content.push(ContentPart::text(text));
        self
    }

    /// Append an image part, returning the message.
    #[must_use]
    pub fn with_image(mut self, source: impl Into<ImageSource>) -> Self {
        self.content.push(ContentPart::image(source));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_set_the_expected_role() {
        assert_eq!(Message::system("s").role, Role::System);
        assert_eq!(Message::user("u").role, Role::User);
        assert_eq!(Message::assistant("a").role, Role::Assistant);
    }

    #[test]
    fn text_constructors_produce_a_single_text_part() {
        assert_eq!(Message::user("hi").content, vec![ContentPart::text("hi")]);
    }

    #[test]
    fn builders_mix_text_and_image_parts() {
        let message = Message::user("look at this")
            .with_image(ImageSource::url("https://example.com/a.png"));
        assert_eq!(
            message.content,
            vec![
                ContentPart::text("look at this"),
                ContentPart::image(ImageSource::url(
                    "https://example.com/a.png"
                )),
            ]
        );
    }

    #[test]
    fn media_types_map_to_their_mime_strings() {
        assert_eq!(MediaType::Jpeg.as_wire(), "image/jpeg");
        assert_eq!(MediaType::Png.as_wire(), "image/png");
        assert_eq!(MediaType::Gif.as_wire(), "image/gif");
        assert_eq!(MediaType::Webp.as_wire(), "image/webp");
    }
}
