// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The compiled-in baseline layer of the Catalog.
//!
//! Each Provider contributes its own [`ModelEntry`]s, gated on that Provider's
//! Cargo feature exactly like the [`Registry`](crate::Registry)'s entry table,
//! so a build with no Provider feature embeds an empty baseline. The entries
//! here are hand-written seeds; the generation pipeline (see ADR-0006) will
//! replace this module with data generated from an upstream dataset.

#[cfg(any(feature = "anthropic", feature = "openai"))]
use crate::model::{
    Api, InputType, Model, ModelCost, ModelEntry, ModelId, ProviderId,
};

/// The baseline [`ModelEntry`]s compiled into this build, one Provider's worth
/// appended after another. Empty when no Provider feature is enabled.
pub(super) fn entries() -> Vec<crate::model::ModelEntry> {
    #[allow(unused_mut)]
    let mut entries = Vec::new();
    #[cfg(feature = "anthropic")]
    entries.extend(anthropic());
    #[cfg(feature = "openai")]
    entries.extend(openai());
    entries
}

/// The baseline Anthropic Models.
#[cfg(feature = "anthropic")]
fn anthropic() -> Vec<ModelEntry> {
    vec![ModelEntry::new(Model {
        id: ModelId::from_static("claude-3-5-sonnet"),
        provider: ProviderId::from_static("anthropic"),
        api: Api::AnthropicMessages,
        name: "Claude 3.5 Sonnet".to_owned(),
        base_url: "https://api.anthropic.com".to_owned(),
        reasoning: false,
        input: vec![InputType::Text, InputType::Image],
        cost: ModelCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
        },
        context_window: 200_000,
        max_tokens: 8_192,
        headers: Vec::new(),
    })]
}

/// The baseline OpenAI Models.
#[cfg(feature = "openai")]
fn openai() -> Vec<ModelEntry> {
    vec![ModelEntry::new(Model {
        id: ModelId::from_static("gpt-4o-mini"),
        provider: ProviderId::from_static("openai"),
        api: Api::OpenAICompletions,
        name: "GPT-4o mini".to_owned(),
        base_url: "https://api.openai.com".to_owned(),
        reasoning: false,
        input: vec![InputType::Text, InputType::Image],
        cost: ModelCost {
            input: 0.15,
            output: 0.6,
            cache_read: 0.075,
            cache_write: 0.0,
        },
        context_window: 128_000,
        max_tokens: 16_384,
        headers: Vec::new(),
    })]
}
