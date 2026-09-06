// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Concrete Providers, each gated behind its own Cargo feature.

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "anthropic")]
pub use anthropic::{AnthropicBuilder, AnthropicProvider};

// The OpenAI module is also compiled for `deepseek`, which reuses its
// `pub(crate)` Chat Completions request builder (ADR-0011); its public
// Provider surface is re-exported only under the `openai` feature.
#[cfg(any(feature = "openai", feature = "deepseek"))]
pub mod openai;

#[cfg(feature = "openai")]
pub use openai::{OpenAIBuilder, OpenAIEmbeddingProvider, OpenAIProvider};

#[cfg(feature = "deepseek")]
pub mod deepseek;

#[cfg(feature = "deepseek")]
pub use deepseek::{DeepSeekBuilder, DeepSeekProvider};
