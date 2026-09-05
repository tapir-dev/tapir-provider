// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`Model`] spec, its [`ModelCost`], the optional [`CompatConfig`] bag, and
//! the [`ModelEntry`] that pairs a Model with runtime call state.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::enums::{Api, Dialect, InputType, ThinkingLevel};
use super::id::{ModelId, ProviderId};

/// The specification of one addressable Model a Provider exposes.
///
/// Pure description: it carries no Credential. A [`ModelEntry`] pairs it with
/// the Credential and any [`CompatConfig`] overrides needed to actually call it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    /// The Model's id within its Provider.
    pub id: ModelId,
    /// The Provider that exposes this Model.
    pub provider: ProviderId,
    /// The wire protocol this Model is called over.
    pub api: Api,
    /// A human-readable display name.
    pub name: String,
    /// The base URL the Model is served from.
    pub base_url: String,
    /// Whether the Model produces reasoning/thinking output.
    pub reasoning: bool,
    /// The input modalities the Model accepts.
    pub input: Vec<InputType>,
    /// Per-token cost, in USD per million tokens.
    pub cost: ModelCost,
    /// The Model's total context window, in tokens.
    pub context_window: u32,
    /// The most output tokens the Model will produce in one response.
    pub max_tokens: u32,
    /// Static headers to send with every request to this Model, in order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub headers: Vec<(String, String)>,
}

/// Per-token cost for a Model, in USD per million tokens.
///
/// Flat by design: one rate per token class, so a caller multiplies a token
/// count by the matching field. `cache_read`/`cache_write` are the prompt-cache
/// rates; a Model without prompt caching leaves them `0.0`.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ModelCost {
    /// Cost per million input (prompt) tokens.
    pub input: f64,
    /// Cost per million output (completion) tokens.
    pub output: f64,
    /// Cost per million tokens read from the prompt cache.
    pub cache_read: f64,
    /// Cost per million tokens written to the prompt cache.
    pub cache_write: f64,
}

/// Per-model overrides describing how a Model deviates from its Provider's
/// defaults.
///
/// Every field is optional: an absent field means "use the Provider default",
/// so the whole struct is additive-safe — new override knobs can be added
/// without breaking existing serialized entries, and an all-`None` value is
/// equivalent to no Compat at all. Only the fields the current Providers need
/// are populated today; the rest stay `None`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CompatConfig {
    /// Whether the Model supports streaming responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub streaming: Option<bool>,
    /// Whether the Model supports tool calling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<bool>,
    /// Whether the Model accepts multimodal (non-text) input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multimodal: Option<bool>,
    /// How the Model wants tool calls encoded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dialect: Option<Dialect>,
    /// Whether the Model supports a reasoning/thinking mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<bool>,
    /// The reasoning level to request when the caller does not specify one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_default: Option<ThinkingLevel>,
    /// The wire representation for each reasoning level the Model supports.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_level_map: Option<BTreeMap<ThinkingLevel, String>>,
}

/// A [`Model`] paired with what is resolved at runtime to call it.
///
/// This is the unit the Model Registry holds and turns into a live Provider: the
/// `model` describes the target, `api_key` and `auth_header` carry the
/// Credential and how to present it, and `compat` overrides the Provider's
/// defaults when the Model deviates from them.
#[derive(Clone, PartialEq)]
pub struct ModelEntry {
    /// The Model this entry calls.
    pub model: Model,
    /// The API key to authenticate with, if one is configured on the entry.
    pub api_key: Option<String>,
    /// Whether the API key is sent as an `Authorization` header (Bearer) rather
    /// than the Provider's own key header.
    pub auth_header: bool,
    /// Per-model overrides; `None` means the Provider's defaults apply.
    pub compat: Option<CompatConfig>,
}

impl ModelEntry {
    /// Construct an entry from a Model, with no Credential or overrides.
    #[must_use]
    pub fn new(model: Model) -> Self {
        Self {
            model,
            api_key: None,
            auth_header: false,
            compat: None,
        }
    }
}

/// Redacts `api_key` so a secret never leaks into logs or panic messages,
/// showing only whether one is present.
impl fmt::Debug for ModelEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelEntry")
            .field("model", &self.model)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("auth_header", &self.auth_header)
            .field("compat", &self.compat)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_model() -> Model {
        Model {
            id: ModelId::from_static("gpt-4o"),
            provider: ProviderId::from_static("openai"),
            api: Api::OpenAICompletions,
            name: "GPT-4o".to_owned(),
            base_url: "https://api.openai.com".to_owned(),
            reasoning: false,
            input: vec![InputType::Text, InputType::Image],
            cost: ModelCost {
                input: 2.5,
                output: 10.0,
                cache_read: 1.25,
                cache_write: 0.0,
            },
            context_window: 128_000,
            max_tokens: 16_384,
            headers: Vec::new(),
        }
    }

    #[test]
    fn model_roundtrips_through_json() {
        let model = sample_model();
        let json = serde_json::to_string(&model).unwrap();
        let back: Model = serde_json::from_str(&json).unwrap();
        assert_eq!(back, model);
    }

    #[test]
    fn empty_compat_serializes_to_an_empty_object() {
        assert_eq!(
            serde_json::to_string(&CompatConfig::default()).unwrap(),
            "{}"
        );
    }

    #[test]
    fn a_missing_compat_field_stays_none() {
        let compat: CompatConfig =
            serde_json::from_str(r#"{"streaming":true}"#).unwrap();
        assert_eq!(compat.streaming, Some(true));
        assert_eq!(compat.tools, None);
        assert_eq!(compat.dialect, None);
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let mut entry = ModelEntry::new(sample_model());
        entry.api_key = Some("sk-secret".to_owned());
        let rendered = format!("{entry:?}");
        assert!(rendered.contains("<redacted>"));
        assert!(!rendered.contains("sk-secret"));
    }

    #[test]
    fn debug_shows_none_when_no_key() {
        let entry = ModelEntry::new(sample_model());
        let rendered = format!("{entry:?}");
        assert!(rendered.contains("api_key: None"));
    }
}
