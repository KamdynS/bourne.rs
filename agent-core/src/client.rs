use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::{LlmError, Request, Response, StreamChunk};

/// Abstraction over LLM providers.
///
/// Each provider (Anthropic, OpenAI, Gemini, Bedrock) implements this trait.
/// The agent loop is provider-agnostic—it only uses this interface.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Make a non-streaming completion request.
    async fn complete(&self, request: Request) -> Result<Response, LlmError>;

    /// Make a streaming completion request.
    fn complete_stream(
        &self,
        request: Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send + 'static>>;
}
