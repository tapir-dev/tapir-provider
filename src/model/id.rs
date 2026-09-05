// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! Validated, non-empty string identifiers for the model catalog.
//!
//! [`ModelId`] names a Model within its Provider; [`ProviderId`] names a
//! Provider. Both wrap a `Cow<'static, str>` so a `const` entry can borrow a
//! `&'static str` with no allocation while a runtime value owns a `String`.
//! Construction validates non-emptiness: [`from_static`](ModelId::from_static)
//! panics at compile time on an empty literal, and the fallible
//! [`new`](ModelId::new)/[`FromStr`] paths reject an empty string. Both are
//! always compiled — the catalog's `models` feature gates only the richer types
//! built on top of them — because the [`Registry`](crate::Registry)'s
//! `ProviderInfo` carries a [`ProviderId`] in every build.

use std::borrow::Cow;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{Error, ErrorKind};

/// Define a validated, non-empty string-newtype identifier.
///
/// The generated type wraps a `Cow<'static, str>`, is `serde(transparent)` (so
/// it round-trips as a bare string), and offers a `const` borrowing
/// constructor plus fallible owning ones. Validation is non-emptiness only.
macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(Cow<'static, str>);

        impl $name {
            /// Construct from a `&'static str`, borrowing it with no allocation.
            ///
            /// Intended for `const` catalog entries. Panics if `id` is empty;
            /// in a `const` context that panic is a compile-time error.
            #[must_use]
            pub const fn from_static(id: &'static str) -> Self {
                assert!(
                    !id.is_empty(),
                    concat!(stringify!($name), " must not be empty")
                );
                Self(Cow::Borrowed(id))
            }

            /// Construct from an owned or borrowed string, validating that it is
            /// non-empty.
            ///
            /// # Errors
            ///
            /// Returns an [`InvalidRequest`](crate::ErrorKind::InvalidRequest)
            /// error if the string is empty.
            pub fn new(id: impl Into<String>) -> Result<Self, Error> {
                let id = id.into();
                if id.is_empty() {
                    return Err(Error::new(
                        ErrorKind::InvalidRequest,
                        concat!(stringify!($name), " must not be empty"),
                    ));
                }
                Ok(Self(Cow::Owned(id)))
            }

            /// The identifier as a string slice.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, concat!(stringify!($name), "({:?})"), &self.0)
            }
        }

        impl FromStr for $name {
            type Err = Error;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Self::new(s.to_owned())
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

define_id! {
    /// The id of one addressable Model within its Provider (e.g.
    /// `claude-3-5-sonnet`). Non-empty; see the [module docs](self).
    ModelId
}

define_id! {
    /// The canonical id of a Provider (e.g. `anthropic`). Non-empty; see the
    /// [module docs](self). This is the key a Provider stores Credentials under
    /// in a Token Store and the identity the [`Registry`](crate::Registry)
    /// resolves names against.
    ProviderId
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_static_borrows_and_reads_back() {
        const ID: ProviderId = ProviderId::from_static("anthropic");
        assert_eq!(ID.as_str(), "anthropic");
        assert_eq!(ID.to_string(), "anthropic");
    }

    #[test]
    fn new_rejects_the_empty_string() {
        let err = ModelId::new("").unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidRequest);
        assert!(ModelId::new("gpt-4o").is_ok());
    }

    #[test]
    fn from_str_matches_new() {
        assert_eq!(
            "gpt-4o".parse::<ModelId>().unwrap(),
            ModelId::new("gpt-4o").unwrap()
        );
        assert!("".parse::<ModelId>().is_err());
    }

    #[test]
    fn serde_is_transparent() {
        let id = ModelId::new("gpt-4o").unwrap();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"gpt-4o\"");
        let back: ModelId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn debug_names_the_type() {
        let id = ProviderId::from_static("openai");
        assert_eq!(format!("{id:?}"), "ProviderId(\"openai\")");
    }
}
