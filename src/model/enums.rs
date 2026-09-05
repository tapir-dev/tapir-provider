// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The catalog's closed vocabularies: the wire [`Api`] a Model speaks, its
//! [`InputType`] modalities, and the tool-call [`Dialect`] and
//! [`ThinkingLevel`] a [`CompatConfig`](super::CompatConfig) can override.

use std::convert::Infallible;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The wire protocol a Model is called over.
///
/// The named variants are the protocols this crate's Providers speak today; the
/// [`Custom`](Api::Custom) catch-all carries any other protocol name verbatim,
/// so the catalog can describe a Model this build has no adapter for. Its
/// [`Display`](fmt::Display)/[`FromStr`] and serde forms are the lowercase wire
/// strings below; an unrecognized string parses to [`Custom`](Api::Custom), so
/// parsing never fails.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Api {
    /// Anthropic's Messages API.
    AnthropicMessages,
    /// OpenAI's Chat Completions API.
    OpenAICompletions,
    /// OpenAI's Responses API.
    OpenAIResponses,
    /// Any other protocol, named verbatim.
    Custom(String),
}

impl Api {
    const ANTHROPIC_MESSAGES: &'static str = "anthropic-messages";
    const OPENAI_COMPLETIONS: &'static str = "openai-completions";
    const OPENAI_RESPONSES: &'static str = "openai-responses";

    /// The wire string for a named variant, or the carried name for
    /// [`Custom`](Api::Custom).
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::AnthropicMessages => Self::ANTHROPIC_MESSAGES,
            Self::OpenAICompletions => Self::OPENAI_COMPLETIONS,
            Self::OpenAIResponses => Self::OPENAI_RESPONSES,
            Self::Custom(name) => name,
        }
    }
}

impl fmt::Display for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Api {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            Self::ANTHROPIC_MESSAGES => Self::AnthropicMessages,
            Self::OPENAI_COMPLETIONS => Self::OpenAICompletions,
            Self::OPENAI_RESPONSES => Self::OpenAIResponses,
            other => Self::Custom(other.to_owned()),
        })
    }
}

impl Serialize for Api {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Api {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        // Parsing is infallible: an unknown protocol name becomes `Custom`.
        let Ok(api) = Self::from_str(&s);
        Ok(api)
    }
}

/// An input modality a Model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum InputType {
    /// Plain text.
    Text,
    /// Image input.
    Image,
}

/// How a Model wants tool calls encoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum Dialect {
    /// The Provider's native structured tool-call protocol.
    Native,
    /// Tool calls encoded as XML-ish tags in the text stream.
    Xmlish,
    /// The Harmony tool-call encoding.
    Harmony,
}

/// A discrete reasoning-effort level a Model can be asked for.
///
/// Ordered from least to most effort; a [`CompatConfig`](super::CompatConfig)
/// maps the levels a given Model supports onto its wire representation.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    Serialize,
    Deserialize,
)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum ThinkingLevel {
    /// No reasoning effort.
    Off,
    /// Minimal reasoning effort.
    Minimal,
    /// Low reasoning effort.
    Low,
    /// Medium reasoning effort.
    Medium,
    /// High reasoning effort.
    High,
    /// Extra-high reasoning effort.
    XHigh,
    /// The most reasoning effort the Model offers.
    Max,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_roundtrips_through_its_wire_string() {
        for api in [
            Api::AnthropicMessages,
            Api::OpenAICompletions,
            Api::OpenAIResponses,
        ] {
            let wire = api.to_string();
            assert_eq!(wire.parse::<Api>().unwrap(), api);
            let json = serde_json::to_string(&api).unwrap();
            assert_eq!(json, format!("\"{wire}\""));
            assert_eq!(serde_json::from_str::<Api>(&json).unwrap(), api);
        }
    }

    #[test]
    fn an_unknown_api_parses_to_custom() {
        assert_eq!(
            "cohere-chat".parse::<Api>().unwrap(),
            Api::Custom("cohere-chat".to_owned())
        );
        assert_eq!(
            Api::Custom("cohere-chat".to_owned()).as_str(),
            "cohere-chat"
        );
    }

    #[test]
    fn simple_enums_serialize_lowercase() {
        assert_eq!(
            serde_json::to_string(&InputType::Image).unwrap(),
            "\"image\""
        );
        assert_eq!(
            serde_json::to_string(&Dialect::Xmlish).unwrap(),
            "\"xmlish\""
        );
        assert_eq!(
            serde_json::to_string(&ThinkingLevel::XHigh).unwrap(),
            "\"xhigh\""
        );
    }

    #[test]
    fn thinking_levels_are_ordered_off_to_max() {
        assert!(ThinkingLevel::Off < ThinkingLevel::Minimal);
        assert!(ThinkingLevel::High < ThinkingLevel::Max);
    }
}
