// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The typed [`Error`] returned across the crate and its [`ErrorKind`] taxonomy.

use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;

/// A coarse classification of what went wrong, stable enough for callers to
/// branch on without parsing messages.
///
/// HTTP status codes map onto these via [`ErrorKind::from_status`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// The Credential was missing, malformed, or rejected (HTTP 401).
    Authentication,
    /// The Credential authenticated but lacks access to the resource (HTTP 403).
    PermissionDenied,
    /// The request itself was malformed or unprocessable (HTTP 400, 404, 422).
    InvalidRequest,
    /// The caller is being rate limited and should back off (HTTP 429).
    RateLimited,
    /// The Provider is temporarily overloaded (HTTP 529).
    Overloaded,
    /// The Provider failed to handle a well-formed request (HTTP 5xx).
    ServerError,
    /// The transport failed before a response was received (connection, DNS, TLS).
    Transport,
    /// A response was received but its body could not be decoded.
    Decode,
    /// Anything not covered above.
    Other,
}

impl ErrorKind {
    /// Classify an HTTP status code into an [`ErrorKind`].
    ///
    /// Only non-2xx codes should be passed here; 2xx is not an error.
    #[must_use]
    pub fn from_status(status: u16) -> Self {
        match status {
            401 => Self::Authentication,
            403 => Self::PermissionDenied,
            400 | 404 | 422 => Self::InvalidRequest,
            429 => Self::RateLimited,
            529 => Self::Overloaded,
            500..=599 => Self::ServerError,
            _ => Self::Other,
        }
    }

    /// A short, stable, human-readable label for this kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Authentication => "authentication",
            Self::PermissionDenied => "permission denied",
            Self::InvalidRequest => "invalid request",
            Self::RateLimited => "rate limited",
            Self::Overloaded => "overloaded",
            Self::ServerError => "server error",
            Self::Transport => "transport",
            Self::Decode => "decode",
            Self::Other => "other",
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The error type returned by every fallible operation in this crate.
///
/// It carries an [`ErrorKind`] for branching, a human-readable message, an
/// optional originating HTTP status, and an optional underlying source error.
pub struct Error {
    kind: ErrorKind,
    message: String,
    status: Option<u16>,
    retry_after: Option<Duration>,
    source: Option<Box<dyn StdError + Send + Sync>>,
}

impl Error {
    /// Construct an error with an explicit kind and message.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            status: None,
            retry_after: None,
            source: None,
        }
    }

    /// Construct an error from a non-2xx HTTP status, classifying the kind via
    /// [`ErrorKind::from_status`].
    pub fn from_status(status: u16, message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::from_status(status),
            message: message.into(),
            status: Some(status),
            retry_after: None,
            source: None,
        }
    }

    /// Attach a server-requested retry delay, as from a `Retry-After` header.
    ///
    /// The [`RetryProvider`](crate::retry::RetryProvider) honors this over its
    /// computed backoff when deciding how long to wait before the next attempt.
    #[must_use]
    pub fn with_retry_after(mut self, retry_after: Duration) -> Self {
        self.retry_after = Some(retry_after);
        self
    }

    /// Attach an underlying source error.
    #[must_use]
    pub fn with_source(
        mut self,
        source: impl StdError + Send + Sync + 'static,
    ) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// Construct a [`ErrorKind::Decode`] error from a deserialization failure.
    pub fn decode(source: impl StdError + Send + Sync + 'static) -> Self {
        Self::new(ErrorKind::Decode, source.to_string()).with_source(source)
    }

    /// Construct a [`ErrorKind::Other`] error from a serialization failure.
    pub fn serialize(source: impl StdError + Send + Sync + 'static) -> Self {
        Self::new(ErrorKind::Other, source.to_string()).with_source(source)
    }

    /// The error's classification.
    #[must_use]
    pub const fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The originating HTTP status, if this error came from a response.
    #[must_use]
    pub const fn status(&self) -> Option<u16> {
        self.status
    }

    /// The server-requested retry delay, if one was attached.
    #[must_use]
    pub const fn retry_after(&self) -> Option<Duration> {
        self.retry_after
    }

    /// The human-readable message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.message)?;
        if let Some(status) = self.status {
            write!(f, " (status {status})")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Error")
            .field("kind", &self.kind)
            .field("message", &self.message)
            .field("status", &self.status)
            .field("retry_after", &self.retry_after)
            .field("source", &self.source)
            .finish()
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        self.source
            .as_ref()
            .map(|s| s.as_ref() as &(dyn StdError + 'static))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_classification_covers_the_documented_cases() {
        assert_eq!(ErrorKind::from_status(401), ErrorKind::Authentication);
        assert_eq!(ErrorKind::from_status(403), ErrorKind::PermissionDenied);
        assert_eq!(ErrorKind::from_status(400), ErrorKind::InvalidRequest);
        assert_eq!(ErrorKind::from_status(404), ErrorKind::InvalidRequest);
        assert_eq!(ErrorKind::from_status(422), ErrorKind::InvalidRequest);
        assert_eq!(ErrorKind::from_status(429), ErrorKind::RateLimited);
        assert_eq!(ErrorKind::from_status(529), ErrorKind::Overloaded);
        assert_eq!(ErrorKind::from_status(500), ErrorKind::ServerError);
        assert_eq!(ErrorKind::from_status(503), ErrorKind::ServerError);
        assert_eq!(ErrorKind::from_status(418), ErrorKind::Other);
    }

    #[test]
    fn from_status_records_the_status_and_kind() {
        let err = Error::from_status(429, "slow down");
        assert_eq!(err.kind(), ErrorKind::RateLimited);
        assert_eq!(err.status(), Some(429));
        assert_eq!(err.message(), "slow down");
    }

    #[test]
    fn display_includes_status_when_present() {
        let err = Error::from_status(500, "boom");
        assert_eq!(err.to_string(), "server error: boom (status 500)");
        let err = Error::new(ErrorKind::Decode, "bad json");
        assert_eq!(err.to_string(), "decode: bad json");
    }
}
