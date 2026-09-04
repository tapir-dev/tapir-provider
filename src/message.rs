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

/// A single normalized message: a [`Role`] and its text content.
///
/// Content is plain text for now; richer content (images, tool calls) is a
/// later extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    /// Who authored the message.
    pub role: Role,
    /// The text content of the message.
    pub content: String,
}

impl Message {
    /// Construct a message with an explicit role.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
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
}
