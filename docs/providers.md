# Multi-Provider Normalization

This document explains how agent-rs supports multiple LLM providers behind a unified interface, where each provider differs, and how to add new providers.

## The Normalization Problem

LLM providers have converged on similar capabilities but diverged on APIs:

| Capability | Anthropic | OpenAI | Gemini | Bedrock |
|------------|-----------|--------|--------|---------|
| Tool calling | Yes | Yes | Yes | Yes |
| Streaming | Yes | Yes | Yes | Yes |
| System prompts | Yes | Yes | Yes | Yes |

But the implementations differ in almost every detail:

| Aspect | Anthropic | OpenAI | Gemini | Bedrock |
|--------|-----------|--------|--------|---------|
| Tool format | `tool_use` blocks | `tool_calls` array | `functionCall` parts | `toolUse` blocks |
| Result format | `tool_result` blocks | `tool` role message | `functionResponse` | `toolResult` blocks |
| System prompt | Top-level field | System role message | `system_instruction` | Top-level field |
| Streaming | SSE with event types | SSE with data lines | SSE | Chunked JSON |
| Auth | Custom header | Bearer token | API key param | AWS SigV4 |

agent-rs normalizes these differences so the agent loop doesn't care which provider it's using.

## Provider-Agnostic Types

### The Normalized Schema

```rust
// ═══════════════════════════════════════════════════════════════════════════
// REQUEST TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// A request to any LLM provider.
///
/// The agent loop constructs these; providers translate to native format.
pub struct Request {
    /// System prompt (optional).
    ///
    /// Instructions that apply to the entire conversation.
    /// Some providers have this as a top-level field (Anthropic, Bedrock),
    /// others as a message with system role (OpenAI), others as a
    /// separate field (Gemini system_instruction).
    pub system: Option<String>,

    /// Conversation messages.
    ///
    /// Alternating user/assistant messages. The provider translates
    /// the Message/ContentBlock structure to its native format.
    pub messages: Vec<Message>,

    /// Available tools.
    ///
    /// Tool definitions the model can call. Each provider has a
    /// different schema format for tool definitions.
    pub tools: Vec<ToolDef>,

    /// Maximum tokens to generate.
    ///
    /// Most providers respect this directly. Some have different
    /// parameter names (max_output_tokens, maxOutputTokens, etc.)
    pub max_tokens: u32,
}

/// A conversation message.
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

pub enum Role {
    User,
    Assistant,
}

/// Content within a message.
///
/// This enum covers all content types across providers. Each provider
/// translates to its native representation.
pub enum ContentBlock {
    /// Plain text.
    ///
    /// All providers support this directly.
    Text(String),

    /// A tool invocation from the assistant.
    ///
    /// Translation varies significantly:
    /// - Anthropic: `{ "type": "tool_use", "id": ..., "name": ..., "input": ... }`
    /// - OpenAI: In `tool_calls` array with `function` object
    /// - Gemini: `functionCall` in message parts
    /// - Bedrock: `{ "toolUse": { "toolUseId": ..., "name": ..., "input": ... } }`
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    /// Result of a tool execution, sent back to the model.
    ///
    /// Translation varies:
    /// - Anthropic: `{ "type": "tool_result", "tool_use_id": ..., "content": ... }`
    /// - OpenAI: Separate message with `role: "tool"`, `tool_call_id`
    /// - Gemini: `functionResponse` in message parts
    /// - Bedrock: `{ "toolResult": { "toolUseId": ..., "content": ... } }`
    ToolResult {
        id: String,
        content: String,
        is_error: bool,
    },
}

/// Tool definition sent to the model.
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

// ═══════════════════════════════════════════════════════════════════════════
// RESPONSE TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// A complete response from the model.
pub struct Response {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
}

pub enum StopReason {
    /// Model finished naturally.
    EndTurn,
    /// Model wants to call tools.
    ToolUse,
    /// Hit token limit.
    MaxTokens,
}

/// Streaming response chunk.
///
/// The agent loop consumes these to build events and accumulate content.
pub enum StreamChunk {
    /// A text fragment.
    Text(String),

    /// Start of a tool call.
    ToolUseStart { id: String, name: String },

    /// Fragment of tool input JSON.
    ///
    /// Accumulated until ToolUseDone, then parsed.
    ToolUseInput(String),

    /// End of current tool call.
    ToolUseDone,

    /// Message complete.
    MessageDone { stop_reason: StopReason, usage: TokenUsage },
}
```

## Provider Implementations

### Anthropic (Messages API)

**Endpoint:** `https://api.anthropic.com/v1/messages`

**Auth:** `x-api-key` header

**Request Translation:**

```rust
impl AnthropicClient {
    fn translate_request(&self, req: Request) -> serde_json::Value {
        json!({
            "model": self.model,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": req.messages.iter().map(|m| self.translate_message(m)).collect::<Vec<_>>(),
            "tools": req.tools.iter().map(|t| json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema
            })).collect::<Vec<_>>()
        })
    }

    fn translate_message(&self, msg: &Message) -> serde_json::Value {
        json!({
            "role": match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant"
            },
            "content": msg.content.iter().map(|b| self.translate_block(b)).collect::<Vec<_>>()
        })
    }

    fn translate_block(&self, block: &ContentBlock) -> serde_json::Value {
        match block {
            ContentBlock::Text(s) => json!({ "type": "text", "text": s }),
            ContentBlock::ToolUse { id, name, input } => json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input
            }),
            ContentBlock::ToolResult { id, content, is_error } => json!({
                "type": "tool_result",
                "tool_use_id": id,
                "content": content,
                "is_error": is_error
            }),
        }
    }
}
```

**Streaming Format:** Server-Sent Events with typed event names

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_...","model":"claude-3-5-sonnet-20241022"}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}

event: message_stop
data: {"type":"message_stop"}
```

### OpenAI (Chat Completions API)

**Endpoint:** `https://api.openai.com/v1/chat/completions`

**Auth:** `Authorization: Bearer <api_key>`

**Request Translation:**

```rust
impl OpenAiClient {
    fn translate_request(&self, req: Request) -> serde_json::Value {
        let mut messages = Vec::new();

        // System prompt becomes a system message
        if let Some(system) = &req.system {
            messages.push(json!({
                "role": "system",
                "content": system
            }));
        }

        // Translate messages
        for msg in &req.messages {
            messages.extend(self.translate_message(msg));
        }

        json!({
            "model": self.model,
            "max_tokens": req.max_tokens,
            "messages": messages,
            "tools": req.tools.iter().map(|t| json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema
                }
            })).collect::<Vec<_>>(),
            "stream": true
        })
    }

    fn translate_message(&self, msg: &Message) -> Vec<serde_json::Value> {
        // OpenAI uses separate messages for tool results
        let mut result = Vec::new();

        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };

        // Separate tool results (they need their own messages)
        let tool_results: Vec<_> = msg.content.iter()
            .filter_map(|b| match b {
                ContentBlock::ToolResult { id, content, .. } => Some((id, content)),
                _ => None
            })
            .collect();

        // Non-tool-result content
        let other_content: Vec<_> = msg.content.iter()
            .filter(|b| !matches!(b, ContentBlock::ToolResult { .. }))
            .collect();

        if !other_content.is_empty() {
            let mut message = json!({ "role": role });

            // Handle tool calls specially
            let tool_calls: Vec<_> = other_content.iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolUse { id, name, input } => Some(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": input.to_string()
                        }
                    })),
                    _ => None
                })
                .collect();

            if !tool_calls.is_empty() {
                message["tool_calls"] = json!(tool_calls);
            }

            // Text content
            let text: String = other_content.iter()
                .filter_map(|b| match b {
                    ContentBlock::Text(s) => Some(s.as_str()),
                    _ => None
                })
                .collect::<Vec<_>>()
                .join("");

            if !text.is_empty() {
                message["content"] = json!(text);
            }

            result.push(message);
        }

        // Tool results as separate messages
        for (id, content) in tool_results {
            result.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": content
            }));
        }

        result
    }
}
```

**Streaming Format:** SSE with `data:` prefix

```
data: {"id":"chatcmpl-...","choices":[{"delta":{"role":"assistant"},"index":0}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_...","function":{"name":"bash","arguments":""}}]},"index":0}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":"}}]},"index":0}]}

data: {"id":"chatcmpl-...","choices":[{"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":50,"completion_tokens":30}}

data: [DONE]
```

### Google Gemini

**Endpoint:** `https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent`

**Auth:** `x-goog-api-key` header or query parameter

**Request Translation:**

```rust
impl GeminiClient {
    fn translate_request(&self, req: Request) -> serde_json::Value {
        json!({
            "system_instruction": req.system.map(|s| json!({
                "parts": [{ "text": s }]
            })),
            "contents": req.messages.iter().map(|m| self.translate_message(m)).collect::<Vec<_>>(),
            "tools": [{
                "function_declarations": req.tools.iter().map(|t| json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema
                })).collect::<Vec<_>>()
            }],
            "generation_config": {
                "maxOutputTokens": req.max_tokens
            }
        })
    }

    fn translate_message(&self, msg: &Message) -> serde_json::Value {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "model",
        };

        let parts: Vec<_> = msg.content.iter().map(|b| match b {
            ContentBlock::Text(s) => json!({ "text": s }),
            ContentBlock::ToolUse { name, input, .. } => json!({
                "functionCall": {
                    "name": name,
                    "args": input
                }
            }),
            ContentBlock::ToolResult { content, .. } => json!({
                "functionResponse": {
                    "name": "result",  // Gemini doesn't use IDs
                    "response": { "result": content }
                }
            }),
        }).collect();

        json!({ "role": role, "parts": parts })
    }
}
```

### AWS Bedrock

**Endpoint:** `https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse-stream`

**Auth:** AWS SigV4 signing

**Request Translation:**

```rust
impl BedrockClient {
    fn translate_request(&self, req: Request) -> serde_json::Value {
        json!({
            "system": req.system.map(|s| vec![json!({ "text": s })]),
            "messages": req.messages.iter().map(|m| self.translate_message(m)).collect::<Vec<_>>(),
            "toolConfig": {
                "tools": req.tools.iter().map(|t| json!({
                    "toolSpec": {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": {
                            "json": t.input_schema
                        }
                    }
                })).collect::<Vec<_>>()
            },
            "inferenceConfig": {
                "maxTokens": req.max_tokens
            }
        })
    }

    fn translate_message(&self, msg: &Message) -> serde_json::Value {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };

        let content: Vec<_> = msg.content.iter().map(|b| match b {
            ContentBlock::Text(s) => json!({ "text": s }),
            ContentBlock::ToolUse { id, name, input } => json!({
                "toolUse": {
                    "toolUseId": id,
                    "name": name,
                    "input": input
                }
            }),
            ContentBlock::ToolResult { id, content, is_error } => json!({
                "toolResult": {
                    "toolUseId": id,
                    "content": [{ "text": content }],
                    "status": if *is_error { "error" } else { "success" }
                }
            }),
        }).collect();

        json!({ "role": role, "content": content })
    }
}
```

## Adding a New Provider

### Step 1: Create the Client Struct

```rust
// src/providers/my_provider.rs

use crate::{LlmClient, Request, Response, StreamChunk, LlmError};

/// Client for MyProvider's LLM API.
///
/// # Example
///
/// ```rust
/// let client = MyProviderClient::new("api-key", "model-name");
/// let agent = AgentBuilder::new(Box::new(client)).build();
/// ```
pub struct MyProviderClient {
    api_key: String,
    model: String,
    base_url: String,
    http: reqwest::Client,
}

impl MyProviderClient {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://api.myprovider.com/v1".into(),
            http: reqwest::Client::new(),
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}
```

### Step 2: Implement Request Translation

```rust
impl MyProviderClient {
    fn translate_request(&self, req: Request) -> serde_json::Value {
        // Convert Request to your provider's JSON format
        json!({
            // Map fields appropriately
        })
    }

    fn translate_message(&self, msg: &Message) -> serde_json::Value {
        // Handle the message structure your provider expects
    }

    fn translate_block(&self, block: &ContentBlock) -> serde_json::Value {
        // Handle tool_use, tool_result, text appropriately
    }
}
```

### Step 3: Implement Response Translation

```rust
impl MyProviderClient {
    fn translate_response(&self, native: MyProviderResponse) -> Response {
        Response {
            content: self.translate_content(&native.content),
            stop_reason: self.translate_stop_reason(&native.stop_reason),
            usage: TokenUsage {
                input_tokens: native.usage.input,
                output_tokens: native.usage.output,
            },
        }
    }

    fn translate_stream_chunk(&self, chunk: MyProviderChunk) -> Option<StreamChunk> {
        // Parse the streaming format and emit appropriate chunks
    }
}
```

### Step 4: Implement LlmClient

```rust
#[async_trait]
impl LlmClient for MyProviderClient {
    async fn complete(&self, request: Request) -> Result<Response, LlmError> {
        let native_request = self.translate_request(request);

        let response = self.http
            .post(format!("{}/chat", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&native_request)
            .send()
            .await
            .map_err(|e| LlmError::Network(e.to_string()))?;

        // Handle errors
        match response.status().as_u16() {
            429 => return Err(LlmError::RateLimit {
                retry_after: parse_retry_after(&response),
            }),
            401 | 403 => return Err(LlmError::Auth(
                response.text().await.unwrap_or_default()
            )),
            400 => return Err(LlmError::InvalidRequest(
                response.text().await.unwrap_or_default()
            )),
            503 | 529 => return Err(LlmError::Overloaded),
            s if s >= 400 => return Err(LlmError::Network(format!("HTTP {s}"))),
            _ => {}
        }

        let native: MyProviderResponse = response.json().await
            .map_err(|e| LlmError::Network(e.to_string()))?;

        Ok(self.translate_response(native))
    }

    fn complete_stream(
        &self,
        request: Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send + 'static>> {
        let native_request = self.translate_request(request);
        let url = format!("{}/chat/stream", self.base_url);
        let api_key = self.api_key.clone();
        let http = self.http.clone();

        Box::pin(async_stream::try_stream! {
            let response = http
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .json(&native_request)
                .send()
                .await
                .map_err(|e| LlmError::Network(e.to_string()))?;

            // Check for errors before streaming
            // ...

            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let bytes = chunk.map_err(|e| LlmError::Network(e.to_string()))?;
                let text = String::from_utf8_lossy(&bytes);

                // Parse your provider's streaming format
                for line in text.lines() {
                    if let Some(chunk) = self.parse_stream_line(line) {
                        yield chunk;
                    }
                }
            }
        })
    }
}
```

### Step 5: Export from Module

```rust
// src/providers/mod.rs
mod anthropic;
mod openai;
mod gemini;
mod bedrock;
mod my_provider;  // Add this

pub use anthropic::AnthropicClient;
pub use openai::OpenAiClient;
pub use gemini::GeminiClient;
pub use bedrock::BedrockClient;
pub use my_provider::MyProviderClient;  // Add this
```

### Step 6: Add Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_translation() {
        let client = MyProviderClient::new("key", "model");
        let request = Request {
            system: Some("Be helpful".into()),
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text("Hello".into())],
            }],
            tools: vec![],
            max_tokens: 100,
        };

        let native = client.translate_request(request);

        // Assert the JSON structure is correct
        assert_eq!(native["model"], "model");
        // ...
    }

    #[tokio::test]
    async fn test_mock_completion() {
        // Use a mock server or recorded responses
    }
}
```

## Common Pitfalls

### 1. Tool Call ID Handling

Some providers generate IDs, others require you to provide them, others don't use them at all:

```rust
// Anthropic: Provider generates IDs
// OpenAI: Provider generates IDs
// Gemini: No IDs, match by name/order
// Bedrock: Provider generates IDs

// When translating TO Gemini:
ContentBlock::ToolResult { id, content, .. } => {
    // Can't use id, Gemini doesn't support it
    json!({
        "functionResponse": {
            "name": /* need to track this separately */,
            "response": { "result": content }
        }
    })
}
```

### 2. Message Structure Differences

OpenAI requires tool results as separate messages; others embed them:

```rust
// Our format:
// Message { role: User, content: [ToolResult, ToolResult] }

// OpenAI needs:
// { role: "tool", tool_call_id: "1", content: "..." }
// { role: "tool", tool_call_id: "2", content: "..." }

// Anthropic/Bedrock accept:
// { role: "user", content: [{ type: "tool_result", ... }, { type: "tool_result", ... }] }
```

### 3. Streaming Event Ordering

Providers emit events differently:

```rust
// Anthropic: message_start → content_block_start → deltas → content_block_stop → message_stop
// OpenAI: Deltas with role → deltas with content → delta with finish_reason → [DONE]
// Gemini: Partial response objects with candidates array
// Bedrock: messageStart → contentBlockStart → contentBlockDelta → contentBlockStop → messageStop
```

### 4. Error Response Formats

Each provider returns errors differently:

```rust
// Anthropic: { "type": "error", "error": { "type": "...", "message": "..." } }
// OpenAI: { "error": { "message": "...", "type": "...", "code": "..." } }
// Gemini: { "error": { "code": 400, "message": "...", "status": "..." } }
// Bedrock: { "message": "..." } with HTTP status
```
