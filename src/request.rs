// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The normalized [`CompletionRequest`] sent to a Provider.

use crate::message::Message;

/// A normalized request for a single, non-streaming text completion.
///
/// The addressable Model is chosen when the Provider is built, so it is not
/// part of the request.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompletionRequest {
    /// The conversation so far, in order.
    pub messages: Vec<Message>,
    /// Sampling temperature; `None` leaves the Provider default.
    pub temperature: Option<f32>,
    /// Upper bound on tokens to generate; `None` leaves the Provider default.
    pub max_tokens: Option<u32>,
}

impl CompletionRequest {
    /// Construct a request from a list of messages, leaving sampling knobs at
    /// their Provider defaults.
    pub fn new(messages: impl Into<Vec<Message>>) -> Self {
        Self {
            messages: messages.into(),
            temperature: None,
            max_tokens: None,
        }
    }

    /// Set the sampling temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tokens to generate.
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_set_the_optional_knobs() {
        let req = CompletionRequest::new(vec![Message::user("hi")])
            .with_temperature(0.5)
            .with_max_tokens(256);
        assert_eq!(req.temperature, Some(0.5));
        assert_eq!(req.max_tokens, Some(256));
        assert_eq!(req.messages.len(), 1);
    }
}
