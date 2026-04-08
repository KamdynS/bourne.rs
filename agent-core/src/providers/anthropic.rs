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
const API_VERSION: &str = "2023-06-01";

/// Anthropic Messages API client.
pub struct AnthropicClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl AnthropicClient {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            http: reqwest::Client::new(),
        }
    }

    fn headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).expect("invalid api key"),
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(API_VERSION),
        );
        headers
    }

    fn build_request_body(&self, request: &Request, stream: bool) -> Value {
        let mut body = json!({
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": request.messages.iter().map(|m| translate_message(m)).collect::<Vec<_>>(),
        });

        if let Some(system) = &request.system {
            body["system"] = json!(system);
        }

        if !request.tools.is_empty() {
            body["tools"] = json!(request.tools.iter().map(translate_tool).collect::<Vec<_>>());
        }

        if stream {
            body["stream"] = json!(true);
        }

        body
    }
}

fn translate_message(msg: &Message) -> Value {
    json!({
        "role": match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        },
        "content": msg.content.iter().map(translate_block).collect::<Vec<_>>(),
    })
}

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
            "tool_use_id": id,
            "content": content,
            "is_error": is_error,
        }),
    }
}

fn translate_tool(tool: &ToolDef) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    })
}

fn parse_stop_reason(s: &str) -> StopReason {
    match s {
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

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

fn handle_error_status(status: u16, body: &str) -> LlmError {
    match status {
        429 => LlmError::RateLimit { retry_after: None },
        401 | 403 => LlmError::Auth(body.to_string()),
        400 => LlmError::InvalidRequest(body.to_string()),
        529 | 503 => LlmError::Overloaded,
        _ => LlmError::Network(format!("HTTP {status}: {body}")),
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
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
        let text = response.text().await.map_err(|e| LlmError::Network(e.to_string()))?;

        if status >= 400 {
            return Err(handle_error_status(status, &text));
        }

        let json: Value = serde_json::from_str(&text)
            .map_err(|e| LlmError::Network(format!("invalid JSON: {e}")))?;

        parse_response(json)
    }

    fn complete_stream(
        &self,
        request: Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send + 'static>> {
        let body = self.build_request_body(&request, true);
        let headers = self.headers();
        let http = self.http.clone();

        Box::pin(futures::stream::unfold(
            StreamState::new(http, headers, body),
            |mut state| async move {
                state.next().await.map(|chunk| (chunk, state))
            },
        ))
    }
}

/// Internal state for streaming responses.
struct StreamState {
    response: Option<reqwest::Response>,
    buffer: String,
    http: reqwest::Client,
    headers: HeaderMap,
    body: Value,
    started: bool,
    done: bool,
    // Track current tool being built
    current_tool_id: Option<String>,
    current_tool_name: Option<String>,
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

    async fn next(&mut self) -> Option<Result<StreamChunk, LlmError>> {
        if self.done {
            return None;
        }

        // Initialize connection if needed
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
            // Try to parse a complete event from buffer
            if let Some(event) = self.try_parse_event() {
                return Some(event);
            }

            // Read more data
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

    fn try_parse_event(&mut self) -> Option<Result<StreamChunk, LlmError>> {
        // SSE format: "event: <type>\ndata: <json>\n\n"
        let double_newline = self.buffer.find("\n\n")?;
        let event_block = self.buffer[..double_newline].to_string();
        self.buffer = self.buffer[double_newline + 2..].to_string();

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
            _ => return None,
        };

        let json: Value = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(_) => return None,
        };

        self.process_event(&event_type, &json)
    }

    fn process_event(&mut self, event_type: &str, json: &Value) -> Option<Result<StreamChunk, LlmError>> {
        match event_type {
            "content_block_start" => {
                let block = &json["content_block"];
                match block["type"].as_str()? {
                    "tool_use" => {
                        let id = block["id"].as_str()?.to_string();
                        let name = block["name"].as_str()?.to_string();
                        self.current_tool_id = Some(id.clone());
                        self.current_tool_name = Some(name.clone());
                        self.input_buffer.clear();
                        Some(Ok(StreamChunk::ToolUseStart { id, name }))
                    }
                    _ => None,
                }
            }
            "content_block_delta" => {
                let delta = &json["delta"];
                match delta["type"].as_str()? {
                    "text_delta" => {
                        let text = delta["text"].as_str()?.to_string();
                        Some(Ok(StreamChunk::Text(text)))
                    }
                    "input_json_delta" => {
                        let partial = delta["partial_json"].as_str()?.to_string();
                        self.input_buffer.push_str(&partial);
                        Some(Ok(StreamChunk::ToolUseInput(partial)))
                    }
                    _ => None,
                }
            }
            "content_block_stop" => {
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
                self.done = true;
                None
            }
            _ => None,
        }
    }
}

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
        assert!(matches!(handle_error_status(429, ""), LlmError::RateLimit { .. }));
        assert!(matches!(handle_error_status(401, "bad key"), LlmError::Auth(_)));
        assert!(matches!(handle_error_status(403, "forbidden"), LlmError::Auth(_)));
        assert!(matches!(handle_error_status(400, "bad request"), LlmError::InvalidRequest(_)));
        assert!(matches!(handle_error_status(529, "overloaded"), LlmError::Overloaded));
        assert!(matches!(handle_error_status(503, "unavailable"), LlmError::Overloaded));
        assert!(matches!(handle_error_status(500, "error"), LlmError::Network(_)));
    }
}
