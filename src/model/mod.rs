// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The model catalog's type layer: what a Model is and how a Model Entry pairs
//! it with runtime state.
//!
//! A [`Model`] is the pure description of one addressable model — its ids, wire
//! [`Api`], endpoint, input modalities, limits, and [`ModelCost`]. A
//! [`ModelEntry`] pairs a [`Model`] with what is resolved at runtime to call it:
//! a Credential and optional per-model [`CompatConfig`] overrides.
//!
//! The [`ModelId`] and [`ProviderId`] newtypes are always compiled; everything
//! else here sits behind the `models` feature (off by default), so the minimal
//! build carries only the ids the [`Registry`](crate::Registry) already needs.

pub mod id;

pub use id::{ModelId, ProviderId};

#[cfg(feature = "models")]
mod enums;
#[cfg(feature = "models")]
mod spec;

#[cfg(feature = "models")]
pub use enums::{Api, Dialect, InputType, ThinkingLevel};
#[cfg(feature = "models")]
pub use spec::{CompatConfig, Model, ModelCost, ModelEntry};
