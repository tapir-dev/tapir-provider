// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Concrete Providers, each gated behind its own Cargo feature.

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;
