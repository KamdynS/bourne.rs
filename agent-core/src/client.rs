//! LLM client trait.
//!
//! This trait abstracts over LLM providers (Anthropic, OpenAI, Gemini, etc.).
//! The agent loop uses this interface exclusively—it doesn't know or care which
//! provider it's talking to.
//!
//! # Why a Trait?
//!
//! This is one of only two traits in agent-rs (the other is Tool). We use a
//! trait here because:
//!
//! 1. **Multiple implementations exist**: Anthropic, OpenAI, Gemini, Bedrock
//! 2. **Users may add more**: Custom providers, proxies, mock clients
//! 3. **Testing requires it**: MockClient for unit tests
//!
//! # Provider Normalization
//!
//! Each provider has different:
//! - Request/response JSON formats
//! - Authentication methods
//! - Streaming protocols
//! - Error formats
//!
//! The LlmClient implementation handles all translation. See the providers
//! module for implementation examples.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::{LlmError, Request, Response, StreamChunk};

/// Abstraction over LLM providers.
///
/// Implementations translate between our provider-agnostic types (Request,
/// Response, StreamChunk) and the provider's native API format.
///
/// # Two Methods
///
/// - `complete`: Blocking request, returns full response
/// - `complete_stream`: Streaming request, returns chunks as they arrive
///
/// The agent loop primarily uses `complete_stream` for real-time output.
/// `complete` is used for internal operations like summarization.
///
/// # Implementation Requirements
///
/// Implementations must:
/// - Translate Request → provider's JSON format
/// - Translate provider's JSON → Response/StreamChunk
/// - Map HTTP errors to LlmError variants
/// - Handle the provider's streaming format (SSE, chunked, etc.)
///
/// See `providers/anthropic.rs` for a reference implementation.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Make a non-streaming completion request.
    ///
    /// Sends the request and waits for the complete response. Simpler but
    /// slower perceived latency since nothing is shown until completion.
    ///
    /// # When to Use
    ///
    /// - Internal operations (summarization, context compression)
    /// - Simple one-shot queries
    /// - When streaming isn't needed
    ///
    /// # Errors
    ///
    /// Returns LlmError for:
    /// - Network failures (connection, timeout)
    /// - Authentication failures (bad API key)
    /// - Rate limiting (429)
    /// - Invalid requests (malformed JSON, bad parameters)
    async fn complete(&self, request: Request) -> Result<Response, LlmError>;

    /// Make a streaming completion request.
    ///
    /// Returns a Stream that yields chunks as the model generates them.
    /// This enables real-time display of model output.
    ///
    /// # Streaming Protocol
    ///
    /// The returned stream yields StreamChunk variants:
    /// - `Text(String)`: A piece of text output
    /// - `ToolUseStart { id, name }`: Beginning of a tool call
    /// - `ToolUseInput(String)`: Fragment of tool input JSON
    /// - `ToolUseDone`: End of tool call, input JSON complete
    /// - `MessageDone { stop_reason, usage }`: Stream complete
    ///
    /// # Lifetime
    ///
    /// The stream is `'static` and owns all its data. This allows the agent
    /// loop to hold the stream across await points without lifetime issues.
    ///
    /// # Error Handling
    ///
    /// Errors can occur at any point in the stream:
    /// - Connection errors during streaming
    /// - Rate limiting mid-response (rare but possible)
    /// - Parse errors in the streaming format
    ///
    /// Each item in the stream is `Result<StreamChunk, LlmError>`.
    fn complete_stream(
        &self,
        request: Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send + 'static>>;
}
