//! OpenAI Chat Completions API client.
//!
//! This module implements the LlmClient trait for OpenAI's GPT models.
//! Comparing this to the Anthropic implementation shows how provider
//! normalization works - same trait, different API translations.
//!
//! # Key Differences from Anthropic
//!
//! | Aspect | Anthropic | OpenAI |
//! |--------|-----------|--------|
//! | System prompt | Top-level `system` field | Message with role "system" |
//! | Tool calls | `tool_use` content block | Separate `tool_calls` array |
//! | Tool results | `tool_result` content block | Message with role "tool" |
//! | Streaming | Typed SSE events | Generic delta objects |
//! | Auth header | `x-api-key` | `Authorization: Bearer` |
//!
//! # Supported Models
//!
//! - gpt-4o, gpt-4o-mini
//! - gpt-4-turbo, gpt-4
//! - gpt-3.5-turbo

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

use crate::{
    ContentBlock, LlmClient, LlmError, Message, Request, Response, Role, StopReason, StreamChunk,
    TokenUsage, ToolDef,
};

const API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI Chat Completions API client.
pub struct OpenAiClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl OpenAiClient {
    /// Create a new OpenAI client.
    ///
    /// The API key should start with "sk-". The model should be a valid
    /// model ID like "gpt-4o" or "gpt-4-turbo".
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            http: reqwest::Client::new(),
        }
    }

    /// Build headers for OpenAI API requests.
    ///
    /// OpenAI uses standard Bearer token auth, unlike Anthropic's custom header.
    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key)).expect("invalid api key"),
        );
        headers
    }

    /// Convert our Request to OpenAI's JSON format.
    ///
    /// OpenAI's format differs significantly:
    /// ```json
    /// {
    ///   "model": "gpt-4o",
    ///   "messages": [
    ///     {"role": "system", "content": "..."},  // System prompt is a message!
    ///     {"role": "user", "content": "..."},
    ///     {"role": "assistant", "content": "...", "tool_calls": [...]},
    ///     {"role": "tool", "tool_call_id": "...", "content": "..."}
    ///   ],
    ///   "tools": [...],
    ///   "max_tokens": 1024,
    ///   "stream": true
    /// }
    /// ```
    fn build_request_body(&self, request: &Request, stream: bool) -> Value {
        let mut messages = Vec::new();

        // System prompt becomes a message with role "system"
        if let Some(system) = &request.system {
            messages.push(json!({
                "role": "system",
                "content": system
            }));
        }

        // Convert our messages to OpenAI format
        for msg in &request.messages {
            messages.push(translate_message(msg));
        }

        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        });

        if !request.tools.is_empty() {
            body["tools"] = json!(request.tools.iter().map(translate_tool).collect::<Vec<_>>());
        }

        if stream {
            body["stream"] = json!(true);
            // Request usage stats in streaming mode (OpenAI extension)
            body["stream_options"] = json!({"include_usage": true});
        }

        body
    }
}

// =============================================================================
// REQUEST TRANSLATION
// =============================================================================

/// Convert a Message to OpenAI's format.
///
/// OpenAI's message format varies by role:
/// - User: `{"role": "user", "content": "..."}`
/// - Assistant with text: `{"role": "assistant", "content": "..."}`
/// - Assistant with tools: `{"role": "assistant", "content": "...", "tool_calls": [...]}`
/// - Tool result: `{"role": "tool", "tool_call_id": "...", "content": "..."}`
fn translate_message(msg: &Message) -> Value {
    match msg.role {
        Role::User => {
            // Check if this is a tool result message
            let tool_results: Vec<_> = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolResult { id, content, .. } => {
                        Some((id.clone(), content.clone()))
                    }
                    _ => None,
                })
                .collect();

            if !tool_results.is_empty() {
                // OpenAI wants separate messages for each tool result
                // But we can only return one message here, so we'll use the first
                // (In practice, the agent loop should handle this properly)
                let (id, content) = &tool_results[0];
                return json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": content
                });
            }

            // Regular user message
            let text = extract_text(&msg.content);
            json!({
                "role": "user",
                "content": text
            })
        }
        Role::Assistant => {
            let text = extract_text(&msg.content);
            let tool_calls: Vec<_> = msg
                .content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolUse { id, name, input } => Some(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(input).unwrap_or_default()
                        }
                    })),
                    _ => None,
                })
                .collect();

            let mut msg_json = json!({
                "role": "assistant",
                "content": if text.is_empty() { Value::Null } else { json!(text) }
            });

            if !tool_calls.is_empty() {
                msg_json["tool_calls"] = json!(tool_calls);
            }

            msg_json
        }
    }
}

/// Extract text content from content blocks.
fn extract_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text(s) => Some(s.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Convert a ToolDef to OpenAI's tool format.
///
/// OpenAI wraps tools in a "function" type:
/// ```json
/// {
///   "type": "function",
///   "function": {
///     "name": "...",
///     "description": "...",
///     "parameters": {...}  // Note: "parameters" not "input_schema"
///   }
/// }
/// ```
fn translate_tool(tool: &ToolDef) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema
        }
    })
}

// =============================================================================
// RESPONSE PARSING
// =============================================================================

/// Parse OpenAI's stop reason to our enum.
///
/// OpenAI uses "finish_reason": "stop", "tool_calls", "length"
fn parse_finish_reason(s: &str) -> StopReason {
    match s {
        "tool_calls" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

/// Parse a complete OpenAI response.
fn parse_response(body: Value) -> Result<Response, LlmError> {
    let choice = body["choices"]
        .get(0)
        .ok_or_else(|| LlmError::InvalidRequest("no choices in response".into()))?;

    let message = &choice["message"];
    let mut content = Vec::new();

    // Extract text content
    if let Some(text) = message["content"].as_str() {
        if !text.is_empty() {
            content.push(ContentBlock::Text(text.to_string()));
        }
    }

    // Extract tool calls
    if let Some(tool_calls) = message["tool_calls"].as_array() {
        for tc in tool_calls {
            if let (Some(id), Some(name), Some(args)) = (
                tc["id"].as_str(),
                tc["function"]["name"].as_str(),
                tc["function"]["arguments"].as_str(),
            ) {
                let input: Value = serde_json::from_str(args).unwrap_or(Value::Null);
                content.push(ContentBlock::ToolUse {
                    id: id.to_string(),
                    name: name.to_string(),
                    input,
                });
            }
        }
    }

    let stop_reason = choice["finish_reason"]
        .as_str()
        .map(parse_finish_reason)
        .unwrap_or(StopReason::EndTurn);

    let usage = TokenUsage {
        input_tokens: body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        output_tokens: body["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
    };

    Ok(Response {
        content,
        stop_reason,
        usage,
    })
}

/// Map HTTP status to LlmError.
fn handle_error_status(status: u16, body: &str) -> LlmError {
    match status {
        429 => LlmError::RateLimit { retry_after: None },
        401 => LlmError::Auth(body.to_string()),
        400 => LlmError::InvalidRequest(body.to_string()),
        503 | 500 => LlmError::Overloaded,
        _ => LlmError::Network(format!("HTTP {status}: {body}")),
    }
}

// =============================================================================
// LLMCLIENT IMPLEMENTATION
// =============================================================================

#[async_trait]
impl LlmClient for OpenAiClient {
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

    /// Stream completions from OpenAI.
    ///
    /// OpenAI's streaming format uses SSE with a simpler structure than Anthropic:
    /// ```text
    /// data: {"choices":[{"delta":{"content":"Hello"}}]}
    /// data: {"choices":[{"delta":{"tool_calls":[...]}}]}
    /// data: {"choices":[{"finish_reason":"stop"}]}
    /// data: [DONE]
    /// ```
    ///
    /// Tool calls stream differently too - the function arguments come in
    /// fragments across multiple delta objects.
    fn complete_stream(
        &self,
        request: Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send + 'static>> {
        let body = self.build_request_body(&request, true);
        let headers = self.headers();
        let http = self.http.clone();

        Box::pin(futures::stream::unfold(
            StreamState::new(http, headers, body),
            |mut state| async move { state.next().await.map(|chunk| (chunk, state)) },
        ))
    }
}

// =============================================================================
// STREAMING STATE MACHINE
// =============================================================================

struct StreamState {
    response: Option<reqwest::Response>,
    buffer: String,
    http: reqwest::Client,
    headers: HeaderMap,
    body: Value,
    started: bool,
    done: bool,
    // Tool call accumulation (OpenAI streams tool calls differently)
    current_tool_id: Option<String>,
    current_tool_name: Option<String>,
    current_tool_args: String,
    usage: TokenUsage,
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
            current_tool_args: String::new(),
            usage: TokenUsage::default(),
        }
    }

    async fn next(&mut self) -> Option<Result<StreamChunk, LlmError>> {
        if self.done {
            return None;
        }

        // Lazy connection init
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
            if let Some(event) = self.try_parse_event() {
                return Some(event);
            }

            let response = match self.response.as_mut() {
                Some(r) => r,
                None => return None,
            };

            match response.chunk().await {
                Ok(Some(bytes)) => {
                    self.buffer.push_str(&String::from_utf8_lossy(&bytes));
                }
                Ok(None) => {
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

    /// Parse SSE events from the buffer.
    ///
    /// OpenAI format: `data: {...}\n\n` or `data: [DONE]\n\n`
    fn try_parse_event(&mut self) -> Option<Result<StreamChunk, LlmError>> {
        let newline_pos = self.buffer.find("\n\n")?;
        let line = self.buffer[..newline_pos].to_string();
        self.buffer = self.buffer[newline_pos + 2..].to_string();

        let data = line.strip_prefix("data: ")?;

        if data == "[DONE]" {
            self.done = true;
            // Emit final MessageDone
            return Some(Ok(StreamChunk::MessageDone {
                stop_reason: StopReason::EndTurn,
                usage: self.usage,
            }));
        }

        let json: Value = serde_json::from_str(data).ok()?;

        // Check for usage stats (sent at end with stream_options)
        if let Some(usage) = json.get("usage") {
            self.usage = TokenUsage {
                input_tokens: usage["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: usage["completion_tokens"].as_u64().unwrap_or(0) as u32,
            };
        }

        let choice = json["choices"].get(0)?;
        let delta = &choice["delta"];

        // Check for finish_reason
        if let Some(reason) = choice["finish_reason"].as_str() {
            let stop_reason = parse_finish_reason(reason);

            // If we have a pending tool call, emit ToolUseDone first
            if self.current_tool_id.is_some() {
                let chunk = StreamChunk::ToolUseDone;
                self.current_tool_id = None;
                self.current_tool_name = None;
                self.current_tool_args.clear();
                // Queue the MessageDone for next iteration
                // (Actually, we'll emit it when we see [DONE])
                return Some(Ok(chunk));
            }

            // Don't emit MessageDone here - wait for [DONE]
            if stop_reason == StopReason::ToolUse {
                return None; // Continue to [DONE]
            }

            return Some(Ok(StreamChunk::MessageDone {
                stop_reason,
                usage: self.usage,
            }));
        }

        // Text content
        if let Some(content) = delta["content"].as_str() {
            if !content.is_empty() {
                return Some(Ok(StreamChunk::Text(content.to_string())));
            }
        }

        // Tool calls
        if let Some(tool_calls) = delta["tool_calls"].as_array() {
            for tc in tool_calls {
                // New tool call starting
                if let Some(id) = tc["id"].as_str() {
                    // If we had a previous tool, emit ToolUseDone
                    if self.current_tool_id.is_some() {
                        let chunk = StreamChunk::ToolUseDone;
                        self.current_tool_id = Some(id.to_string());
                        self.current_tool_name = tc["function"]["name"].as_str().map(String::from);
                        self.current_tool_args.clear();
                        return Some(Ok(chunk));
                    }

                    self.current_tool_id = Some(id.to_string());
                    self.current_tool_name = tc["function"]["name"].as_str().map(String::from);
                    self.current_tool_args.clear();

                    if let (Some(id), Some(name)) = (&self.current_tool_id, &self.current_tool_name)
                    {
                        return Some(Ok(StreamChunk::ToolUseStart {
                            id: id.clone(),
                            name: name.clone(),
                        }));
                    }
                }

                // Tool arguments fragment
                if let Some(args) = tc["function"]["arguments"].as_str() {
                    self.current_tool_args.push_str(args);
                    return Some(Ok(StreamChunk::ToolUseInput(args.to_string())));
                }
            }
        }

        None
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_user_message() {
        let msg = Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello".into())],
        };
        let json = translate_message(&msg);

        assert_eq!(json["role"], "user");
        assert_eq!(json["content"], "Hello");
    }

    #[test]
    fn test_translate_assistant_with_tool() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentBlock::Text("Let me check".into()),
                ContentBlock::ToolUse {
                    id: "call_123".into(),
                    name: "bash".into(),
                    input: json!({"command": "ls"}),
                },
            ],
        };
        let json = translate_message(&msg);

        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"], "Let me check");
        assert_eq!(json["tool_calls"][0]["id"], "call_123");
        assert_eq!(json["tool_calls"][0]["function"]["name"], "bash");
    }

    #[test]
    fn test_translate_tool_result() {
        let msg = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                id: "call_123".into(),
                content: "file.txt".into(),
                is_error: false,
            }],
        };
        let json = translate_message(&msg);

        assert_eq!(json["role"], "tool");
        assert_eq!(json["tool_call_id"], "call_123");
        assert_eq!(json["content"], "file.txt");
    }

    #[test]
    fn test_translate_tool_def() {
        let tool = ToolDef {
            name: "bash".into(),
            description: "Run commands".into(),
            input_schema: json!({"type": "object"}),
        };
        let json = translate_tool(&tool);

        assert_eq!(json["type"], "function");
        assert_eq!(json["function"]["name"], "bash");
        assert_eq!(json["function"]["description"], "Run commands");
        assert_eq!(json["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn test_build_request_body() {
        let client = OpenAiClient::new("test-key", "gpt-4o");
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

        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["max_tokens"], 1024);
        // System prompt is first message
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "Be helpful");
        // User message is second
        assert_eq!(body["messages"][1]["role"], "user");
    }

    #[test]
    fn test_parse_response() {
        let body = json!({
            "choices": [{
                "message": {
                    "content": "Hello!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        });

        let response = parse_response(body).unwrap();
        assert_eq!(response.content.len(), 1);
        assert!(matches!(&response.content[0], ContentBlock::Text(s) if s == "Hello!"));
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.usage.input_tokens, 10);
    }

    #[test]
    fn test_parse_response_with_tool() {
        let body = json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "function": {
                            "name": "bash",
                            "arguments": "{\"command\":\"ls\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10}
        });

        let response = parse_response(body).unwrap();
        assert_eq!(response.stop_reason, StopReason::ToolUse);

        if let ContentBlock::ToolUse { id, name, input } = &response.content[0] {
            assert_eq!(id, "call_abc");
            assert_eq!(name, "bash");
            assert_eq!(input["command"], "ls");
        } else {
            panic!("Expected ToolUse");
        }
    }

    #[test]
    fn test_parse_finish_reason() {
        assert_eq!(parse_finish_reason("stop"), StopReason::EndTurn);
        assert_eq!(parse_finish_reason("tool_calls"), StopReason::ToolUse);
        assert_eq!(parse_finish_reason("length"), StopReason::MaxTokens);
    }
}
