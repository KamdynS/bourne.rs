//! Error types for agent operations.
//!
//! This module defines the error hierarchy:
//!
//! - `AgentError`: Top-level errors from agent execution
//! - `LlmError`: Errors from LLM API calls
//!
//! # Error Philosophy
//!
//! Errors are classified by recoverability:
//!
//! - **Retryable**: Rate limits, overloaded servers. The agent should retry.
//! - **Not retryable**: Auth failures, invalid requests. Bubble up immediately.
//! - **Cancellation**: User requested stop. Clean exit, not an error.
//!
//! Tool errors are handled differently—they're not Rust errors at all, but
//! ToolOutput::error() values that get sent back to the model for self-correction.

use std::time::Duration;

/// Errors that can occur during agent execution.
///
/// This is the top-level error type returned by `Agent::run()`. It wraps
/// LLM errors and adds agent-specific failure modes.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// An LLM API error occurred.
    ///
    /// This wraps any error from the LlmClient. The inner LlmError has more
    /// specific information about what went wrong.
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    /// The agent was cancelled via its cancellation token.
    ///
    /// This is a clean shutdown, not an error. The agent stopped at a safe
    /// point (between turns) when cancellation was requested.
    #[error("Agent cancelled")]
    Cancelled,

    /// The agent exceeded the maximum number of turns.
    ///
    /// This prevents runaway agents. If you hit this, either:
    /// - Increase max_turns in AgentBuilder
    /// - The task is too complex for a single agent run
    /// - The model is stuck in a loop (check your tools/prompts)
    #[error("Exceeded maximum turns ({0})")]
    MaxTurnsExceeded(u32),
}

/// Errors from LLM API calls.
///
/// These errors are classified to help with retry decisions:
///
/// | Variant | Retryable? | Action |
/// |---------|-----------|--------|
/// | RateLimit | Yes | Wait for retry_after, then retry |
/// | Overloaded | Yes | Exponential backoff, then retry |
/// | Auth | No | Fix credentials, don't retry |
/// | InvalidRequest | No | Fix the bug in request construction |
/// | Network | Maybe | Depends on the specific error |
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// Rate limited (HTTP 429).
    ///
    /// The API is throttling requests. The retry_after field, if present,
    /// indicates how long to wait before retrying.
    ///
    /// # Handling
    ///
    /// Wait for `retry_after` (or a default like 30s), then retry. Consider
    /// exponential backoff if rate limits persist.
    #[error("Rate limited (retry after {retry_after:?})")]
    RateLimit {
        /// How long to wait before retrying. None if not provided by the API.
        retry_after: Option<Duration>,
    },

    /// Service overloaded (HTTP 529, 503).
    ///
    /// The provider's servers are overwhelmed. This is temporary.
    ///
    /// # Handling
    ///
    /// Use exponential backoff: wait 1s, 2s, 4s, 8s, etc. Give up after
    /// several attempts.
    #[error("Service overloaded")]
    Overloaded,

    /// Invalid request (HTTP 400).
    ///
    /// Our request was malformed. This indicates a bug in our code, not a
    /// transient issue.
    ///
    /// # Handling
    ///
    /// Don't retry—the same request will fail again. Log the error body
    /// for debugging. Common causes:
    /// - Malformed JSON
    /// - Missing required fields
    /// - Invalid parameter values
    /// - Exceeded context length
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Authentication failed (HTTP 401, 403).
    ///
    /// The API key is invalid, expired, or lacks permission.
    ///
    /// # Handling
    ///
    /// Don't retry—credentials won't magically become valid. Check:
    /// - Is the API key correct?
    /// - Has it expired?
    /// - Does it have the required permissions?
    #[error("Authentication failed: {0}")]
    Auth(String),

    /// Network error (connection failed, timeout, etc.)
    ///
    /// Something went wrong at the network level. Could be:
    /// - DNS resolution failure
    /// - Connection refused
    /// - TLS handshake failure
    /// - Read timeout
    /// - Unexpected disconnect
    ///
    /// # Handling
    ///
    /// May be retryable depending on the specific error. Transient network
    /// issues often resolve on retry; persistent issues won't.
    #[error("Network error: {0}")]
    Network(String),
}
