// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The normalized [`CompletionRequest`] sent to a Provider.

use crate::message::Message;

/// A tool the model may call, described in a Provider-neutral shape.
///
/// The `input_schema` is a JSON Schema object describing the tool's arguments;
/// it is carried verbatim to the Provider, which is free to constrain the
/// model's generated arguments to it.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    /// The tool's name, as the model will reference it.
    pub name: String,
    /// A natural-language description of what the tool does.
    pub description: String,
    /// A JSON Schema object describing the tool's arguments.
    pub input_schema: serde_json::Value,
}

impl ToolDefinition {
    /// Describe a tool by name, description, and argument schema.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// How the model is steered toward (or away from) calling a tool.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolChoice {
    /// The model decides whether to call a tool.
    Auto,
    /// The model must call some tool, its choice which.
    Any,
    /// The model must call the named tool.
    Tool(String),
    /// The model must not call any tool.
    None,
}

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
    /// Tools the model may call; empty leaves tool calling off.
    pub tools: Vec<ToolDefinition>,
    /// How the model is steered toward calling a tool; `None` leaves the
    /// Provider default (typically automatic when `tools` is non-empty).
    pub tool_choice: Option<ToolChoice>,
}

impl CompletionRequest {
    /// Construct a request from a list of messages, leaving sampling knobs at
    /// their Provider defaults.
    pub fn new(messages: impl Into<Vec<Message>>) -> Self {
        Self {
            messages: messages.into(),
            ..Self::default()
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

    /// Offer the model a set of tools it may call.
    #[must_use]
    pub fn with_tools(mut self, tools: impl Into<Vec<ToolDefinition>>) -> Self {
        self.tools = tools.into();
        self
    }

    /// Steer the model toward (or away from) calling a tool.
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
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

    #[test]
    fn builders_attach_tools_and_a_choice() {
        let tool = ToolDefinition::new(
            "get_weather",
            "Look up the weather",
            serde_json::json!({"type": "object"}),
        );
        let req = CompletionRequest::new(vec![Message::user("hi")])
            .with_tools(vec![tool.clone()])
            .with_tool_choice(ToolChoice::Tool("get_weather".to_owned()));
        assert_eq!(req.tools, vec![tool]);
        assert_eq!(
            req.tool_choice,
            Some(ToolChoice::Tool("get_weather".to_owned()))
        );
    }

    #[test]
    fn tools_default_to_empty() {
        let req = CompletionRequest::new(vec![Message::user("hi")]);
        assert!(req.tools.is_empty());
        assert_eq!(req.tool_choice, None);
    }
}
