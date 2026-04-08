/// Token usage statistics from an LLM response.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

impl TokenUsage {
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

/// A request to an LLM.
#[derive(Debug, Clone)]
pub struct Request {
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub max_tokens: u32,
}

/// A message in the conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

/// Message sender role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// A block of content within a message.
#[derive(Debug, Clone)]
pub enum ContentBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        id: String,
        content: String,
        is_error: bool,
    },
}

/// Tool definition sent to the LLM.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// A complete response from an LLM.
#[derive(Debug, Clone)]
pub struct Response {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

/// A chunk from a streaming response.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    Text(String),
    ToolUseStart { id: String, name: String },
    ToolUseInput(String),
    ToolUseDone,
    MessageDone { stop_reason: StopReason, usage: TokenUsage },
}
