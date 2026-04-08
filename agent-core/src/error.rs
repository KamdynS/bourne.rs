use std::time::Duration;

/// Errors that can occur during agent execution.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    #[error("Agent cancelled")]
    Cancelled,

    #[error("Exceeded maximum turns ({0})")]
    MaxTurnsExceeded(u32),
}

/// Errors from LLM API calls.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    #[error("Rate limited (retry after {retry_after:?})")]
    RateLimit { retry_after: Option<Duration> },

    #[error("Service overloaded")]
    Overloaded,

    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("Network error: {0}")]
    Network(String),
}
