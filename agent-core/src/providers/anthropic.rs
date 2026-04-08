//! Anthropic Messages API client.
//!
//! This module implements the LlmClient trait for Anthropic's Claude models.
//! It serves as the reference implementation for how to build an LLM provider,
//! demonstrating request translation, response parsing, and streaming.
//!
//! # API Overview
//!
//! Anthropic's Messages API uses a structured format:
//! - Requests contain messages (user/assistant turns) and optional tools
//! - Responses contain content blocks (text, tool_use) and metadata
//! - Streaming uses Server-Sent Events (SSE) with typed event names
//!
//! # Why Anthropic First?
//!
//! Anthropic's API is the most straightforward to implement because:
//! 1. Content blocks map directly to our types (no translation gymnastics)
//! 2. Tool results stay in the same message (unlike OpenAI's separate messages)
//! 3. SSE events are well-typed (unlike OpenAI's generic delta format)

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::{
    ContentBlock, LlmClient, LlmError, Message, Request, Response, Role, StopReason, StreamChunk,
    TokenUsage, ToolDef,
};

const API_URL: &str = "https://api.anthropic.com/v1/messages";

/// API version header value. Anthropic uses dated versions for stability.
/// This should be updated when new features require a newer API version.
const API_VERSION: &str = "2023-06-01";

/// Client for the Anthropic Messages API.
///
/// Holds authentication credentials and an HTTP client. The HTTP client is
/// reused across requests for connection pooling.
///
/// # Example
///
/// ```ignore
/// let client = AnthropicClient::new("sk-ant-...", "claude-sonnet-4-20250514");
/// let response = client.complete(request).await?;
/// ```
pub struct AnthropicClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl AnthropicClient {
    /// Create a new Anthropic client.
    ///
    /// The API key should start with "sk-ant-". The model should be a valid
    /// model ID like "claude-sonnet-4-20250514" or "claude-3-5-haiku-20241022".
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            http: reqwest::Client::new(),
        }
    }

    /// Build the headers required for Anthropic API requests.
    ///
    /// Anthropic uses custom headers for auth (x-api-key) and versioning
    /// (anthropic-version), unlike OpenAI's Bearer token approach.
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).expect("invalid api key"),
        );
        headers.insert("anthropic-version", HeaderValue::from_static(API_VERSION));
        headers
    }

    /// Convert our Request type to Anthropic's JSON format.
    ///
    /// This is the core of provider normalization: our generic Request type
    /// gets translated to the provider's specific JSON schema. Each provider
    /// has different field names, nesting, and conventions.
    ///
    /// Anthropic's format:
    /// ```json
    /// {
    ///   "model": "claude-...",
    ///   "max_tokens": 1024,
    ///   "system": "optional system prompt",
    ///   "messages": [...],
    ///   "tools": [...],
    ///   "stream": true/false
    /// }
    /// ```
    fn build_request_body(&self, request: &Request, stream: bool) -> Value {
        let mut body = json!({
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": request.messages.iter().map(translate_message).collect::<Vec<_>>(),
        });

        // System prompt is optional. When present, it's a top-level string field.
        // (OpenAI puts it in messages array with role "system" instead.)
        if let Some(system) = &request.system {
            body["system"] = json!(system);
        }

        // Only include tools array if we have tools. Empty array is valid but wasteful.
        if !request.tools.is_empty() {
            body["tools"] = json!(request.tools.iter().map(translate_tool).collect::<Vec<_>>());
        }

        if stream {
            body["stream"] = json!(true);
        }

        body
    }
}

// =============================================================================
// REQUEST TRANSLATION
// =============================================================================
//
// These functions convert our provider-agnostic types to Anthropic's JSON format.
// They're free functions (not methods) because they don't need client state.

/// Convert a Message to Anthropic's message format.
///
/// Anthropic messages have role ("user" or "assistant") and content (array of blocks).
/// Our Role enum maps directly; our ContentBlock enum needs translation.
fn translate_message(msg: &Message) -> Value {
    json!({
        "role": match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        },
        "content": msg.content.iter().map(translate_block).collect::<Vec<_>>(),
    })
}

/// Convert a ContentBlock to Anthropic's content block format.
///
/// Anthropic uses typed content blocks with a "type" discriminator:
/// - Text: `{ "type": "text", "text": "..." }`
/// - Tool use: `{ "type": "tool_use", "id": "...", "name": "...", "input": {...} }`
/// - Tool result: `{ "type": "tool_result", "tool_use_id": "...", "content": "..." }`
///
/// Note: tool_result uses "tool_use_id" not "id" - a common gotcha.
fn translate_block(block: &ContentBlock) -> Value {
    match block {
        ContentBlock::Text(s) => json!({ "type": "text", "text": s }),
        ContentBlock::ToolUse { id, name, input } => json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": input,
        }),
        ContentBlock::ToolResult { id, content, is_error } => json!({
            "type": "tool_result",
            "tool_use_id": id,  // Note: different field name than tool_use
            "content": content,
            "is_error": is_error,
        }),
    }
}

/// Convert a ToolDef to Anthropic's tool format.
///
/// Anthropic's tool schema is straightforward:
/// ```json
/// { "name": "...", "description": "...", "input_schema": {...} }
/// ```
///
/// The input_schema is a JSON Schema object. Anthropic validates inputs against
/// this schema before calling tools (though we validate too, defensively).
fn translate_tool(tool: &ToolDef) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    })
}

// =============================================================================
// RESPONSE PARSING
// =============================================================================
//
// These functions convert Anthropic's JSON responses back to our types.

/// Convert Anthropic's stop_reason string to our StopReason enum.
///
/// Anthropic returns: "end_turn", "tool_use", "max_tokens", or "stop_sequence".
/// We ignore stop_sequence (treated as end_turn) since we don't use that feature.
fn parse_stop_reason(s: &str) -> StopReason {
    match s {
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

/// Parse a complete (non-streaming) Anthropic response.
///
/// Response format:
/// ```json
/// {
///   "content": [...],
///   "stop_reason": "end_turn",
///   "usage": { "input_tokens": 10, "output_tokens": 20 }
/// }
/// ```
fn parse_response(body: Value) -> Result<Response, LlmError> {
    let content = body["content"]
        .as_array()
        .ok_or_else(|| LlmError::InvalidRequest("missing content".into()))?
        .iter()
        .filter_map(parse_content_block)
        .collect();

    let stop_reason = body["stop_reason"]
        .as_str()
        .map(parse_stop_reason)
        .unwrap_or(StopReason::EndTurn);

    let usage = TokenUsage {
        input_tokens: body["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
        output_tokens: body["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
    };

    Ok(Response { content, stop_reason, usage })
}

/// Parse a single content block from Anthropic's response.
///
/// Returns None for unknown block types (defensive parsing).
/// We only care about "text" and "tool_use" in responses.
fn parse_content_block(block: &Value) -> Option<ContentBlock> {
    match block["type"].as_str()? {
        "text" => Some(ContentBlock::Text(block["text"].as_str()?.to_string())),
        "tool_use" => Some(ContentBlock::ToolUse {
            id: block["id"].as_str()?.to_string(),
            name: block["name"].as_str()?.to_string(),
            input: block["input"].clone(),
        }),
        _ => None,
    }
}

// =============================================================================
// ERROR HANDLING
// =============================================================================

/// Map HTTP status codes to our LlmError variants.
///
/// Error classification determines retry behavior:
/// - RateLimit (429): Retry with backoff
/// - Overloaded (529, 503): Retry with backoff
/// - Auth (401, 403): Don't retry, credentials are wrong
/// - InvalidRequest (400): Don't retry, our code has a bug
/// - Network (other): Maybe retry, depends on the error
fn handle_error_status(status: u16, body: &str) -> LlmError {
    match status {
        429 => LlmError::RateLimit { retry_after: None },
        401 | 403 => LlmError::Auth(body.to_string()),
        400 => LlmError::InvalidRequest(body.to_string()),
        529 | 503 => LlmError::Overloaded,
        _ => LlmError::Network(format!("HTTP {status}: {body}")),
    }
}

// =============================================================================
// LLMCLIENT IMPLEMENTATION
// =============================================================================

#[async_trait]
impl LlmClient for AnthropicClient {
    /// Make a non-streaming completion request.
    ///
    /// Used for simple one-shot requests or internal operations like summarization.
    /// For the main agent loop, complete_stream is preferred for real-time output.
    async fn complete(&self, request: Request) -> Result<Response, LlmError> {
        let body = self.build_request_body(&request, false);

        let response = self
            .http
            .post(API_URL)
            .headers(self.headers())
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Network(e.to_string()))?;

        // Read the full response body before checking status.
        // This ensures we get error details for non-2xx responses.
        let status = response.status().as_u16();
        let text = response
            .text()
            .await
            .map_err(|e| LlmError::Network(e.to_string()))?;

        if status >= 400 {
            return Err(handle_error_status(status, &text));
        }

        let json: Value =
            serde_json::from_str(&text).map_err(|e| LlmError::Network(format!("invalid JSON: {e}")))?;

        parse_response(json)
    }

    /// Make a streaming completion request.
    ///
    /// Returns a Stream that yields chunks as the model generates them.
    /// This is the primary interface for the agent loop, enabling real-time
    /// display of model output.
    ///
    /// # Streaming Protocol
    ///
    /// Anthropic uses Server-Sent Events (SSE). The event sequence is:
    /// 1. `message_start` - Contains message metadata
    /// 2. `content_block_start` - Begins a new content block (text or tool_use)
    /// 3. `content_block_delta` - Incremental content (text chunks or JSON fragments)
    /// 4. `content_block_stop` - Ends the current block
    /// 5. `message_delta` - Contains stop_reason and final usage stats
    /// 6. `message_stop` - Stream complete
    ///
    /// For tool calls, the input JSON arrives in fragments via `input_json_delta`
    /// events. We accumulate these and emit ToolUseInput chunks so callers can
    /// also accumulate and parse when ToolUseDone arrives.
    fn complete_stream(
        &self,
        request: Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send + 'static>> {
        let body = self.build_request_body(&request, true);
        let headers = self.headers();
        let http = self.http.clone();

        // Use unfold to create a stream from our stateful StreamState.
        // Each iteration calls state.next() which returns the next chunk.
        Box::pin(futures::stream::unfold(
            StreamState::new(http, headers, body),
            |mut state| async move { state.next().await.map(|chunk| (chunk, state)) },
        ))
    }
}

// =============================================================================
// STREAMING STATE MACHINE
// =============================================================================
//
// Streaming requires managing connection state, buffering partial SSE events,
// and tracking tool call state across multiple events.

/// Internal state for streaming responses.
///
/// This struct manages the lifecycle of a streaming request:
/// 1. Lazy connection (started on first poll)
/// 2. Buffering incoming bytes until we have complete SSE events
/// 3. Parsing events and emitting StreamChunks
/// 4. Tracking tool call state (id, name, input fragments)
///
/// Why a state machine? SSE events can span multiple TCP packets, and tool
/// inputs arrive in fragments. We need to buffer and track state across polls.
struct StreamState {
    /// The HTTP response, once connected. None until first poll.
    response: Option<reqwest::Response>,

    /// Buffer for incomplete SSE event data.
    /// SSE format: "event: <type>\ndata: <json>\n\n"
    buffer: String,

    /// HTTP client for making the initial request.
    http: reqwest::Client,

    /// Headers to send with the request.
    headers: HeaderMap,

    /// Request body to send.
    body: Value,

    /// Whether we've initiated the HTTP request yet.
    started: bool,

    /// Whether the stream has completed (message_stop received or error).
    done: bool,

    /// Current tool call ID (if we're in the middle of a tool_use block).
    current_tool_id: Option<String>,

    /// Current tool call name.
    current_tool_name: Option<String>,

    /// Accumulated tool input JSON fragments.
    input_buffer: String,
}

impl StreamState {
    fn new(http: reqwest::Client, headers: HeaderMap, body: Value) -> Self {
        Self {
            response: None,
            buffer: String::new(),
            http,
            headers,
            body,
            started: false,
            done: false,
            current_tool_id: None,
            current_tool_name: None,
            input_buffer: String::new(),
        }
    }

    /// Get the next chunk from the stream.
    ///
    /// This is the main driver of the streaming state machine:
    /// 1. Initialize connection on first call
    /// 2. Try to parse a complete event from the buffer
    /// 3. If no complete event, read more data from the response
    /// 4. Repeat until we have an event or the stream ends
    async fn next(&mut self) -> Option<Result<StreamChunk, LlmError>> {
        if self.done {
            return None;
        }

        // Lazy initialization: start the HTTP request on first poll.
        // This lets us return a Stream immediately without blocking.
        if !self.started {
            self.started = true;
            let result = self
                .http
                .post(API_URL)
                .headers(self.headers.clone())
                .json(&self.body)
                .send()
                .await;

            match result {
                Ok(resp) => {
                    let status = resp.status().as_u16();
                    if status >= 400 {
                        self.done = true;
                        let text = resp.text().await.unwrap_or_default();
                        return Some(Err(handle_error_status(status, &text)));
                    }
                    self.response = Some(resp);
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(LlmError::Network(e.to_string())));
                }
            }
        }

        loop {
            // Try to parse a complete SSE event from our buffer.
            // Events are delimited by "\n\n".
            if let Some(event) = self.try_parse_event() {
                return Some(event);
            }

            // No complete event yet. Read more data from the response.
            let response = match self.response.as_mut() {
                Some(r) => r,
                None => return None,
            };

            match response.chunk().await {
                Ok(Some(bytes)) => {
                    // Append new data to our buffer. SSE is always UTF-8.
                    self.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                Ok(None) => {
                    // Response body complete. No more data coming.
                    self.done = true;
                    return None;
                }
                Err(e) => {
                    self.done = true;
                    return Some(Err(LlmError::Network(e.to_string())));
                }
            }
        }
    }

    /// Try to parse a complete SSE event from the buffer.
    ///
    /// SSE format:
    /// ```text
    /// event: content_block_delta
    /// data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}
    ///
    /// ```
    ///
    /// Events are separated by double newlines. We look for "\n\n", extract the
    /// event block, parse the event type and data, then process it.
    fn try_parse_event(&mut self) -> Option<Result<StreamChunk, LlmError>> {
        // Look for a complete event (ends with \n\n)
        let double_newline = self.buffer.find("\n\n")?;
        let event_block = self.buffer[..double_newline].to_string();
        self.buffer = self.buffer[double_newline + 2..].to_string();

        // Parse the event type and data from the block
        let mut event_type = None;
        let mut data = None;

        for line in event_block.lines() {
            if let Some(rest) = line.strip_prefix("event: ") {
                event_type = Some(rest.to_string());
            } else if let Some(rest) = line.strip_prefix("data: ") {
                data = Some(rest.to_string());
            }
        }

        let (event_type, data) = match (event_type, data) {
            (Some(e), Some(d)) => (e, d),
            _ => return None, // Malformed event, skip
        };

        let json: Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(_) => return None, // Invalid JSON, skip
        };

        self.process_event(&event_type, &json)
    }

    /// Process a parsed SSE event and emit the appropriate StreamChunk.
    ///
    /// Event types we care about:
    /// - content_block_start: Begin a new text or tool_use block
    /// - content_block_delta: Text chunk or tool input fragment
    /// - content_block_stop: End of current block
    /// - message_delta: Contains stop_reason and usage
    /// - message_stop: Stream complete
    ///
    /// We ignore: message_start, ping
    fn process_event(
        &mut self,
        event_type: &str,
        json: &Value,
    ) -> Option<Result<StreamChunk, LlmError>> {
        match event_type {
            "content_block_start" => {
                let block = &json["content_block"];
                match block["type"].as_str()? {
                    "tool_use" => {
                        // Starting a tool call. Save the id and name for later.
                        let id = block["id"].as_str()?.to_string();
                        let name = block["name"].as_str()?.to_string();
                        self.current_tool_id = Some(id.clone());
                        self.current_tool_name = Some(name.clone());
                        self.input_buffer.clear();
                        Some(Ok(StreamChunk::ToolUseStart { id, name }))
                    }
                    // Text blocks don't emit on start, only on deltas
                    _ => None,
                }
            }
            "content_block_delta" => {
                let delta = &json["delta"];
                match delta["type"].as_str()? {
                    "text_delta" => {
                        // Text content chunk
                        let text = delta["text"].as_str()?.to_string();
                        Some(Ok(StreamChunk::Text(text)))
                    }
                    "input_json_delta" => {
                        // Tool input JSON fragment. Accumulate for later parsing.
                        let partial = delta["partial_json"].as_str()?.to_string();
                        self.input_buffer.push_str(&partial);
                        Some(Ok(StreamChunk::ToolUseInput(partial)))
                    }
                    _ => None,
                }
            }
            "content_block_stop" => {
                // End of a content block. If we were building a tool call, emit ToolUseDone.
                if self.current_tool_id.is_some() {
                    self.current_tool_id = None;
                    self.current_tool_name = None;
                    self.input_buffer.clear();
                    Some(Ok(StreamChunk::ToolUseDone))
                } else {
                    None
                }
            }
            "message_delta" => {
                // Message-level delta with stop_reason and usage.
                let stop_reason = json["delta"]["stop_reason"]
                    .as_str()
                    .map(parse_stop_reason)
                    .unwrap_or(StopReason::EndTurn);
                let usage = TokenUsage {
                    input_tokens: json["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
                    output_tokens: json["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
                };
                Some(Ok(StreamChunk::MessageDone { stop_reason, usage }))
            }
            "message_stop" => {
                // Stream complete. No more events coming.
                self.done = true;
                None
            }
            // Ignore: message_start, ping
            _ => None,
        }
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_message_user() {
        let msg = Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".into())],
        };
        let json = translate_message(&msg);

        assert_eq!(json["role"], "user");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][0]["text"], "Hello");
    }

    #[test]
    fn test_translate_message_assistant_with_tool() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text("I'll help.".into()),
                ContentBlock::ToolUse {
                    id: "tool_1".into(),
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                },
            ],
        };
        let json = translate_message(&msg);

        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"][0]["type"], "text");
        assert_eq!(json["content"][1]["type"], "tool_use");
        assert_eq!(json["content"][1]["id"], "tool_1");
        assert_eq!(json["content"][1]["name"], "bash");
    }

    #[test]
    fn test_translate_tool_result() {
        let block = ContentBlock::ToolResult {
            id: "tool_1".into(),
            content: "file.txt".into(),
            is_error: false,
        };
        let json = translate_block(&block);

        assert_eq!(json["type"], "tool_result");
        assert_eq!(json["tool_use_id"], "tool_1");
        assert_eq!(json["content"], "file.txt");
        assert_eq!(json["is_error"], false);
    }

    #[test]
    fn test_translate_tool_def() {
        let tool = ToolDef {
            name: "bash".into(),
            description: "Run bash commands".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" }
                },
                "required": ["command"]
            }),
        };
        let json = translate_tool(&tool);

        assert_eq!(json["name"], "bash");
        assert_eq!(json["description"], "Run bash commands");
        assert!(json["input_schema"]["properties"]["command"].is_object());
    }

    #[test]
    fn test_build_request_body() {
        let client = AnthropicClient::new("test-key", "claude-3-5-sonnet-20241022");
        let request = Request {
            system: Some("Be helpful".into()),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text("Hi".into())],
            }],
            tools: vec![],
            max_tokens: 1024,
        };

        let body = client.build_request_body(&request, false);

        assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["system"], "Be helpful");
        assert!(body["stream"].is_null());
    }

    #[test]
    fn test_build_request_body_streaming() {
        let client = AnthropicClient::new("test-key", "claude-3-5-sonnet-20241022");
        let request = Request {
            system: None,
            messages: vec![],
            tools: vec![],
            max_tokens: 100,
        };

        let body = client.build_request_body(&request, true);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn test_parse_response() {
        let body = json!({
            "content": [
                { "type": "text", "text": "Hello!" }
            ],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            }
        });

        let response = parse_response(body).unwrap();
        assert_eq!(response.content.len(), 1);
        assert!(matches!(&response.content[0], ContentBlock::Text(s) if s == "Hello!"));
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }

    #[test]
    fn test_parse_response_with_tool_use() {
        let body = json!({
            "content": [
                { "type": "text", "text": "Let me check." },
                {
                    "type": "tool_use",
                    "id": "tool_abc",
                    "name": "bash",
                    "input": { "command": "ls -la" }
                }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 20, "output_tokens": 15 }
        });

        let response = parse_response(body).unwrap();
        assert_eq!(response.content.len(), 2);
        assert_eq!(response.stop_reason, StopReason::ToolUse);

        if let ContentBlock::ToolUse { id, name, input } = &response.content[1] {
            assert_eq!(id, "tool_abc");
            assert_eq!(name, "bash");
            assert_eq!(input["command"], "ls -la");
        } else {
            panic!("Expected ToolUse");
        }
    }

    #[test]
    fn test_parse_stop_reason() {
        assert_eq!(parse_stop_reason("end_turn"), StopReason::EndTurn);
        assert_eq!(parse_stop_reason("tool_use"), StopReason::ToolUse);
        assert_eq!(parse_stop_reason("max_tokens"), StopReason::MaxTokens);
        assert_eq!(parse_stop_reason("unknown"), StopReason::EndTurn);
    }

    #[test]
    fn test_handle_error_status() {
        assert!(matches!(
            handle_error_status(429, ""),
            LlmError::RateLimit { .. }
        ));
        assert!(matches!(
            handle_error_status(401, "bad key"),
            LlmError::Auth(_)
        ));
        assert!(matches!(
            handle_error_status(403, "forbidden"),
            LlmError::Auth(_)
        ));
        assert!(matches!(
            handle_error_status(400, "bad request"),
            LlmError::InvalidRequest(_)
        ));
        assert!(matches!(
            handle_error_status(529, "overloaded"),
            LlmError::Overloaded
        ));
        assert!(matches!(
            handle_error_status(503, "unavailable"),
            LlmError::Overloaded
        ));
        assert!(matches!(
            handle_error_status(500, "error"),
            LlmError::Network(_)
        ));
    }
}
