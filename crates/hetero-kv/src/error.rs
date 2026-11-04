//! One error type for every layer.
//!
//! The variant says what the caller may assume about outstanding work:
//!
//! * [`Error::Invalid`], [`Error::Unsupported`], [`Error::State`] and [`Error::Busy`] are
//!   *rejections*: they are raised before anything was submitted and change no state.
//! * [`Error::Transfer`] means an engine reported a failure or an unverifiable completion.
//!   The regions involved are `Uncertain` and the session (if any) is poisoned.
//! * [`Error::Poisoned`] refuses work on a poisoned session until it is recovered.
//! * [`Error::Cancelled`] reports a cancellation completed at a fenced boundary.

/// Result alias used throughout the crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors raised by hetero-kv. `Display` is the bare message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// A malformed argument (geometry, span, descriptor). Nothing was submitted.
    #[error("{0}")]
    Invalid(String),
    /// No route, capability or phase supports this. Nothing was submitted; there is no fallback.
    #[error("{0}")]
    Unsupported(String),
    /// The call does not fit the current lifecycle state (wrong step, stale request).
    #[error("{0}")]
    State(String),
    /// Another request already holds this single-flight object.
    #[error("{0}")]
    Busy(String),
    /// Cancellation completed; any copies were fenced first.
    #[error("{0}")]
    Cancelled(String),
    /// The session is poisoned; `recover` (which quiesces the engine) is required.
    #[error("{0}")]
    Poisoned(String),
    /// An engine failed or reported an unverifiable completion. Buffers are uncertain.
    #[error("{0}")]
    Transfer(String),
}

impl Error {
    /// True when the error guarantees that nothing was submitted and no state changed.
    pub fn is_rejection(&self) -> bool {
        matches!(
            self,
            Error::Invalid(_) | Error::Unsupported(_) | Error::State(_) | Error::Busy(_)
        )
    }
}

pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(Error::Invalid(msg.into()))
}
