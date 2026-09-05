// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`Context`] and [`CompletionOptions`] a Provider is called with.
//!
//! A [`Context`] is the conversational input — the system prompt, the messages
//! so far, and the tools the model may call — and drops an
//! [`AssistantMessage`](crate::message::AssistantMessage) straight back in for
//! the next turn. [`CompletionOptions`] are the per-request sampling knobs that
//! steer generation rather than describe the conversation. A Provider takes both
//! by reference, so the same options can drive several turns over one context.

use crate::message::Message;
use serde::{Deserialize, Serialize};
use std::ops::Deref;

/// Instructions that steer the model, sent out of band from the messages.
///
/// A transparent wrapper over the prompt text: it carries no structure of its
/// own, so any block or cache shaping stays a Provider concern. It borrows as a
/// `str` via [`Deref`] and [`as_str`](Self::as_str), and is built from a
/// `String` or `&str`, so `.with_system("...")` on a [`Context`] keeps working
/// unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SystemPrompt(String);

impl SystemPrompt {
    /// Wrap the given prompt text.
    pub fn new(text: impl Into<String>) -> Self {
        Self(text.into())
    }

    /// The prompt text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Deref for SystemPrompt {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl From<String> for SystemPrompt {
    fn from(text: String) -> Self {
        Self(text)
    }
}

impl From<&str> for SystemPrompt {
    fn from(text: &str) -> Self {
        Self(text.to_owned())
    }
}

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

/// How hard the model is asked to reason before answering ("extended
/// thinking").
///
/// This is a Provider-neutral effort level, not a raw token budget: a Provider
/// that reasons by token budget derives one from the level via
/// [`default_budget`](Self::default_budget); a Provider without extended
/// thinking ignores the level entirely. [`Off`](Self::Off) leaves thinking
/// disabled.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
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
    /// No extended thinking.
    #[default]
    Off,
    /// The smallest thinking budget.
    Minimal,
    /// A small thinking budget.
    Low,
    /// A moderate thinking budget.
    Medium,
    /// A large thinking budget.
    High,
    /// A very large thinking budget.
    XHigh,
    /// The largest thinking budget.
    Max,
}

impl ThinkingLevel {
    /// The default thinking token budget for this level, for Providers that
    /// reason by budget rather than effort. [`Off`](Self::Off) is zero.
    #[must_use]
    pub const fn default_budget(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Minimal => 1024,
            Self::Low => 2048,
            Self::Medium => 8192,
            Self::High => 16384,
            Self::XHigh => 32768,
            Self::Max => 65536,
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

/// The full conversational input to a Provider.
///
/// It bundles an optional system prompt, the ordered [`Message`]s so far, and
/// the [`ToolDefinition`]s the model may call. An
/// [`AssistantMessage`](crate::message::AssistantMessage) a Provider returns is
/// itself a [`Message`], so it appends straight back into
/// [`messages`](Self::messages) for the next turn. The addressable Model is
/// chosen when the Provider is built, so it is not part of the context.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Context {
    /// Instructions that steer the model, sent out of band from the messages.
    pub system_prompt: Option<SystemPrompt>,
    /// The conversation so far, in order.
    pub messages: Vec<Message>,
    /// Tools the model may call; empty leaves tool calling off.
    pub tools: Vec<ToolDefinition>,
}

impl Context {
    /// Construct a context from a list of messages, with no system prompt and no
    /// tools.
    pub fn new(messages: impl Into<Vec<Message>>) -> Self {
        Self {
            messages: messages.into(),
            ..Self::default()
        }
    }

    /// Set the system prompt.
    #[must_use]
    pub fn with_system(
        mut self,
        system_prompt: impl Into<SystemPrompt>,
    ) -> Self {
        self.system_prompt = Some(system_prompt.into());
        self
    }

    /// Offer the model a set of tools it may call.
    #[must_use]
    pub fn with_tools(mut self, tools: impl Into<Vec<ToolDefinition>>) -> Self {
        self.tools = tools.into();
        self
    }

    /// Append a message to the conversation.
    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }
}

/// The per-request knobs that steer generation.
///
/// These describe how to sample, not what the conversation is: the sampling
/// temperature, the output-token cap, and the tool choice. Passed alongside a
/// [`Context`], so one set of options can drive several turns.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompletionOptions {
    /// Sampling temperature; `None` leaves the Provider default.
    pub temperature: Option<f32>,
    /// Upper bound on tokens to generate; `None` leaves the Provider default.
    pub max_tokens: Option<u32>,
    /// How the model is steered toward calling a tool; `None` leaves the
    /// Provider default (typically automatic when tools are offered).
    pub tool_choice: Option<ToolChoice>,
    /// How hard the model should reason before answering; `None` leaves thinking
    /// off. Providers without extended thinking ignore it.
    pub thinking: Option<ThinkingLevel>,
    /// An explicit API key that authenticates this one request, overriding the
    /// Credential the Provider was built with. `None` leaves the Provider's own
    /// Credential in force; a set value always wins and is sent on the
    /// Provider's api-key lane.
    pub api_key: Option<String>,
}

impl CompletionOptions {
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

    /// Steer the model toward (or away from) calling a tool.
    #[must_use]
    pub fn with_tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Ask the model to reason before answering at the given effort level.
    #[must_use]
    pub fn with_thinking(mut self, level: ThinkingLevel) -> Self {
        self.thinking = Some(level);
        self
    }

    /// Authenticate this one request with an explicit API key, overriding the
    /// Provider's own Credential.
    #[must_use]
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_builders_set_the_system_prompt_and_tools() {
        let tool = ToolDefinition::new(
            "get_weather",
            "Look up the weather",
            serde_json::json!({"type": "object"}),
        );
        let ctx = Context::new(vec![Message::user("hi")])
            .with_system("Be terse.")
            .with_tools(vec![tool.clone()]);
        assert_eq!(ctx.system_prompt.as_deref(), Some("Be terse."));
        assert_eq!(ctx.tools, vec![tool]);
        assert_eq!(ctx.messages.len(), 1);
    }

    #[test]
    fn system_prompt_wraps_transparently_and_borrows_as_str() {
        let prompt = SystemPrompt::new("Be terse.");
        assert_eq!(prompt.as_str(), "Be terse.");
        // Deref lets it stand in for a `str`.
        assert_eq!(&*prompt, "Be terse.");
        // From<&str> and From<String> agree.
        assert_eq!(SystemPrompt::from("Be terse."), prompt);
        assert_eq!(SystemPrompt::from("Be terse.".to_owned()), prompt);
        // Serde is transparent: a bare JSON string.
        assert_eq!(
            serde_json::to_value(&prompt).unwrap(),
            serde_json::json!("Be terse.")
        );
    }

    #[test]
    fn context_defaults_to_no_system_and_no_tools() {
        let ctx = Context::new(vec![Message::user("hi")]);
        assert_eq!(ctx.system_prompt, None);
        assert!(ctx.tools.is_empty());
    }

    #[test]
    fn push_appends_a_turn() {
        let mut ctx = Context::new(vec![Message::user("hi")]);
        ctx.push(Message::user("again"));
        assert_eq!(ctx.messages.len(), 2);
    }

    #[test]
    fn options_builders_set_the_knobs() {
        let opts = CompletionOptions::default()
            .with_temperature(0.5)
            .with_max_tokens(256)
            .with_tool_choice(ToolChoice::Tool("get_weather".to_owned()));
        assert_eq!(opts.temperature, Some(0.5));
        assert_eq!(opts.max_tokens, Some(256));
        assert_eq!(
            opts.tool_choice,
            Some(ToolChoice::Tool("get_weather".to_owned()))
        );
    }

    #[test]
    fn options_default_to_provider_defaults() {
        let opts = CompletionOptions::default();
        assert_eq!(opts.temperature, None);
        assert_eq!(opts.max_tokens, None);
        assert_eq!(opts.tool_choice, None);
        assert_eq!(opts.thinking, None);
    }

    #[test]
    fn with_thinking_sets_the_effort_level() {
        let opts =
            CompletionOptions::default().with_thinking(ThinkingLevel::High);
        assert_eq!(opts.thinking, Some(ThinkingLevel::High));
    }

    #[test]
    fn thinking_levels_map_to_ascending_budgets() {
        assert_eq!(ThinkingLevel::Off.default_budget(), 0);
        assert_eq!(ThinkingLevel::Minimal.default_budget(), 1024);
        assert_eq!(ThinkingLevel::Low.default_budget(), 2048);
        assert_eq!(ThinkingLevel::Medium.default_budget(), 8192);
        assert_eq!(ThinkingLevel::High.default_budget(), 16384);
        assert_eq!(ThinkingLevel::XHigh.default_budget(), 32768);
        assert_eq!(ThinkingLevel::Max.default_budget(), 65536);
        // The default level is Off.
        assert_eq!(ThinkingLevel::default(), ThinkingLevel::Off);
    }
}
