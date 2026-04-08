//! Mock LLM client for testing.
//!
//! This module provides a `MockClient` that implements `LlmClient` with
//! canned responses. Use it to test agent behavior without hitting real APIs.
//!
//! # Example
//!
//! ```ignore
//! let client = MockClient::new(vec![
//!     MockResponse::text("Hello!"),
//! ]);
//!
//! let agent = AgentBuilder::new(Box::new(client))
//!     .build();
//!
//! // Agent will receive "Hello!" from the mock
//! ```

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::Stream;

use crate::{
    ContentBlock, LlmClient, LlmError, Request, Response, StopReason, StreamChunk, TokenUsage,
};

/// A mock LLM client that returns pre-configured responses.
///
/// Responses are returned in order. If more requests are made than
/// responses available, returns an error.
pub struct MockClient {
    responses: Arc<Mutex<Vec<MockResponse>>>,
}

impl MockClient {
    /// Create a mock client with the given responses.
    ///
    /// Responses are consumed in order, one per `complete` or `complete_stream` call.
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
        }
    }
}

/// A canned response for the mock client.
#[derive(Clone)]
pub struct MockResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
}

impl MockResponse {
    /// Create a simple text response.
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text(s.into())],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        }
    }

    /// Create a response with a tool call.
    pub fn tool_call(
        text: impl Into<String>,
        tool_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        let text = text.into();
        let mut content = Vec::new();

        if !text.is_empty() {
            content.push(ContentBlock::Text(text));
        }

        content.push(ContentBlock::ToolUse {
            id: tool_id.into(),
            name: tool_name.into(),
            input,
        });

        Self {
            content,
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage {
                input_tokens: 15,
                output_tokens: 10,
            },
        }
    }

    /// Create a response with just a tool call (no text).
    pub fn tool_only(
        tool_id: impl Into<String>,
        tool_name: impl Into<String>,
        input: serde_json::Value,
    ) -> Self {
        Self::tool_call("", tool_id, tool_name, input)
    }
}

#[async_trait]
impl LlmClient for MockClient {
    async fn complete(&self, _request: Request) -> Result<Response, LlmError> {
        let mut responses = self.responses.lock().unwrap();

        if responses.is_empty() {
            return Err(LlmError::InvalidRequest(
                "MockClient: no more responses".into(),
            ));
        }

        let mock = responses.remove(0);
        Ok(Response {
            content: mock.content,
            stop_reason: mock.stop_reason,
            usage: mock.usage,
        })
    }

    fn complete_stream(
        &self,
        _request: Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send + 'static>> {
        let mut responses = self.responses.lock().unwrap();

        if responses.is_empty() {
            // Return a stream that yields an error
            return Box::pin(futures::stream::once(async {
                Err(LlmError::InvalidRequest(
                    "MockClient: no more responses".into(),
                ))
            }));
        }

        let mock = responses.remove(0);

        // Convert MockResponse to a sequence of StreamChunks
        let chunks = mock_response_to_chunks(mock);

        Box::pin(futures::stream::iter(chunks.into_iter().map(Ok)))
    }
}

/// Convert a MockResponse into StreamChunks.
///
/// This simulates how a real streaming response would arrive:
/// - Text is split into small chunks
/// - Tool calls come as ToolUseStart, ToolUseInput fragments, ToolUseDone
/// - MessageDone arrives at the end
fn mock_response_to_chunks(response: MockResponse) -> Vec<StreamChunk> {
    let mut chunks = Vec::new();

    for block in response.content {
        match block {
            ContentBlock::Text(text) => {
                // Split text into chunks (simulating streaming)
                for chunk in text.chars().collect::<Vec<_>>().chunks(10) {
                    let s: String = chunk.iter().collect();
                    chunks.push(StreamChunk::Text(s));
                }
            }
            ContentBlock::ToolUse { id, name, input } => {
                chunks.push(StreamChunk::ToolUseStart {
                    id: id.clone(),
                    name: name.clone(),
                });

                // Stream the input JSON in fragments
                let json_str = serde_json::to_string(&input).unwrap_or_default();
                for chunk in json_str.chars().collect::<Vec<_>>().chunks(20) {
                    let s: String = chunk.iter().collect();
                    chunks.push(StreamChunk::ToolUseInput(s));
                }

                chunks.push(StreamChunk::ToolUseDone);
            }
            ContentBlock::ToolResult { .. } => {
                // Tool results don't appear in LLM responses, skip
            }
        }
    }

    chunks.push(StreamChunk::MessageDone {
        stop_reason: response.stop_reason,
        usage: response.usage,
    });

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_mock_client_complete() {
        let client = MockClient::new(vec![MockResponse::text("Hello!")]);

        let request = Request {
            system: None,
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
        };

        let response = client.complete(request).await.unwrap();
        assert_eq!(response.content.len(), 1);
        assert!(matches!(&response.content[0], ContentBlock::Text(s) if s == "Hello!"));
        assert_eq!(response.stop_reason, StopReason::EndTurn);
    }

    #[tokio::test]
    async fn test_mock_client_stream() {
        let client = MockClient::new(vec![MockResponse::text("Hi there!")]);

        let request = Request {
            system: None,
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
        };

        let stream = client.complete_stream(request);
        let chunks: Vec<_> = stream.collect().await;

        // Should have text chunks + MessageDone
        assert!(chunks.len() >= 2);

        // Last chunk should be MessageDone
        let last = chunks.last().unwrap().as_ref().unwrap();
        assert!(matches!(last, StreamChunk::MessageDone { .. }));
    }

    #[tokio::test]
    async fn test_mock_client_tool_call() {
        let client = MockClient::new(vec![MockResponse::tool_call(
            "Let me check",
            "tool_1",
            "echo",
            serde_json::json!({"message": "test"}),
        )]);

        let request = Request {
            system: None,
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
        };

        let stream = client.complete_stream(request);
        let chunks: Vec<_> = stream.collect().await;

        // Should have: Text chunks, ToolUseStart, ToolUseInput(s), ToolUseDone, MessageDone
        let has_tool_start = chunks.iter().any(|c| {
            matches!(
                c.as_ref().unwrap(),
                StreamChunk::ToolUseStart { name, .. } if name == "echo"
            )
        });
        let has_tool_done = chunks
            .iter()
            .any(|c| matches!(c.as_ref().unwrap(), StreamChunk::ToolUseDone));

        assert!(has_tool_start);
        assert!(has_tool_done);
    }

    #[tokio::test]
    async fn test_mock_client_exhausted() {
        let client = MockClient::new(vec![]);

        let request = Request {
            system: None,
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
        };

        let result = client.complete(request).await;
        assert!(result.is_err());
    }
}
