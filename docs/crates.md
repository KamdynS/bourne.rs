# Crate Structure and Public API

This document details the workspace layout, inter-crate dependencies, and the complete public API surface for each crate.

## Workspace Layout

```
agent-rs/
├── Cargo.toml              # Workspace manifest
├── agent-core/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs          # Public exports
│       ├── agent.rs        # Agent, AgentBuilder, agent loop
│       ├── types.rs        # Request, Response, Message, ContentBlock, etc.
│       ├── client.rs       # LlmClient trait
│       ├── tool.rs         # Tool trait, ToolOutput
│       ├── context.rs      # ContextManager, token budgeting, eviction
│       ├── store.rs        # ContextStore (feature-gated)
│       ├── error.rs        # AgentError, LlmError, ContextError
│       └── providers/
│           ├── mod.rs
│           ├── anthropic.rs
│           ├── openai.rs
│           ├── gemini.rs
│           └── bedrock.rs
├── agent-tools/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs          # Public exports
│       ├── bash.rs         # BashTool
│       ├── file.rs         # FileTool
│       └── recall.rs       # RecallTool
├── agent-session/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs          # Public exports
│       └── session.rs      # Session, JobId, JobResult
└── agent-bin/              # Binary target (not documented here)
    └── ...
```

## Dependency Graph

```
                    ┌─────────────────┐
                    │  agent-session  │
                    │                 │
                    │  [dashmap]      │
                    │  [tokio]        │
                    │  [uuid]         │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │   agent-core    │◀──────────│  agent-tools    │
    │                 │           │                 │
    │  [tokio]        │           │  [tokio]        │
    │  [tokio-stream] │           │  (no additional │
    │  [serde]        │           │   heavy deps)   │
    │  [serde_json]   │           └─────────────────┘
    │  [reqwest]      │
    │  [async-trait]  │
    │  [thiserror]    │
    │  [rusqlite]*    │  * feature-gated
    │  [aws-*]*       │  * feature-gated for bedrock
    └─────────────────┘
```

## Cargo.toml Configurations

### Workspace Root

```toml
[workspace]
resolver = "2"
members = ["agent-core", "agent-tools", "agent-session", "agent-bin"]

[workspace.dependencies]
tokio = { version = "1", features = ["rt-multi-thread", "macros", "sync", "time", "process"] }
tokio-stream = "0.1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
reqwest = { version = "0.11", features = ["json", "stream"] }
async-trait = "0.1"
thiserror = "1"
futures = "0.3"
```

### agent-core/Cargo.toml

```toml
[package]
name = "agent-core"
version = "0.1.0"
edition = "2021"

[features]
default = []
persistence = ["rusqlite"]
bedrock = ["aws-config", "aws-sdk-bedrockruntime", "aws-sigv4"]

[dependencies]
tokio.workspace = true
tokio-stream.workspace = true
serde.workspace = true
serde_json.workspace = true
reqwest.workspace = true
async-trait.workspace = true
thiserror.workspace = true
futures.workspace = true
tokio-util = { version = "0.7", features = ["rt"] }  # CancellationToken
pin-project-lite = "0.2"

# Optional: persistence
rusqlite = { version = "0.31", features = ["bundled", "fts5"], optional = true }

# Optional: AWS Bedrock
aws-config = { version = "1", optional = true }
aws-sdk-bedrockruntime = { version = "1", optional = true }
aws-sigv4 = { version = "1", optional = true }
```

### agent-tools/Cargo.toml

```toml
[package]
name = "agent-tools"
version = "0.1.0"
edition = "2021"

[dependencies]
agent-core = { path = "../agent-core" }
tokio.workspace = true
serde_json.workspace = true
async-trait.workspace = true
```

### agent-session/Cargo.toml

```toml
[package]
name = "agent-session"
version = "0.1.0"
edition = "2021"

[dependencies]
agent-core = { path = "../agent-core" }
tokio.workspace = true
dashmap = "5"
uuid = { version = "1", features = ["v4"] }
```

---

## agent-core Public API

### Core Types

```rust
//! The embeddable agent library. Contains the agent loop, tool trait,
//! multi-provider LLM client abstraction, and context management.

// ============================================================================
// AGENT
// ============================================================================

/// The agent executor. Constructed via `AgentBuilder`, consumed by calling `run()`.
///
/// An agent is a one-shot executor: you build it, run it on a task, and it's consumed.
/// For multiple tasks, create multiple agents (they're cheap to construct).
pub struct Agent { /* private fields */ }

/// Builder for configuring and constructing an Agent.
///
/// Required: `client` (passed to `new`)
/// Optional: everything else has sensible defaults
pub struct AgentBuilder { /* private fields */ }

impl AgentBuilder {
    /// Create a new builder with the given LLM client.
    ///
    /// The client determines which LLM provider handles requests.
    pub fn new(client: Box<dyn LlmClient>) -> Self;

    /// Set the system prompt that instructs the model's behavior.
    ///
    /// If context management is enabled, additional instructions about
    /// the recall tool will be appended automatically.
    ///
    /// Default: None (no system prompt)
    pub fn system_prompt(self, prompt: impl Into<String>) -> Self;

    /// Register tools the agent can use.
    ///
    /// Tools are matched by name when the model requests them.
    /// Duplicate names will cause the later tool to shadow earlier ones.
    ///
    /// Default: empty (no tools)
    pub fn tools(self, tools: Vec<Box<dyn Tool>>) -> Self;

    /// Set the token budget for context management.
    ///
    /// When estimated token usage exceeds 80% of this budget, older turns
    /// are summarized and evicted. Higher budgets allow longer conversations
    /// but increase costs and latency.
    ///
    /// Default: 100,000 tokens
    pub fn token_budget(self, budget: u32) -> Self;

    /// Set the maximum number of turns before the agent stops.
    ///
    /// A "turn" is one round-trip: send messages to LLM, receive response,
    /// execute any tools. This prevents runaway agents.
    ///
    /// Default: 100 turns
    pub fn max_turns(self, turns: u32) -> Self;

    /// Enable persistent context storage with the given store.
    ///
    /// When enabled, evicted turns are stored in SQLite and can be
    /// retrieved via the recall tool. Requires the `persistence` feature.
    ///
    /// Default: None (pure in-memory, evicted content is lost)
    #[cfg(feature = "persistence")]
    pub fn context_store(self, store: ContextStore) -> Self;

    /// Provide a cancellation token for graceful shutdown.
    ///
    /// When the token is cancelled, the agent will stop at the next
    /// safe point (between turns) and return `AgentError::Cancelled`.
    ///
    /// Default: None (no external cancellation)
    pub fn cancellation_token(self, token: CancellationToken) -> Self;

    /// Build the agent. Panics if configuration is invalid.
    pub fn build(self) -> Agent;
}

impl Agent {
    /// Execute the agent on a task, yielding events as they occur.
    ///
    /// This is the primary public interface. The returned stream yields
    /// `AgentEvent` values as the agent progresses. The agent is consumed
    /// by this call.
    ///
    /// The stream completes when:
    /// - The model signals completion (yields `AgentEvent::Done`)
    /// - Max turns is exceeded (yields error)
    /// - Cancellation is requested (yields error)
    /// - An unrecoverable error occurs (yields error)
    ///
    /// # Example
    ///
    /// ```rust
    /// let mut stream = agent.run("What files are in the current directory?");
    /// while let Some(event) = stream.next().await {
    ///     match event {
    ///         Ok(AgentEvent::Text(s)) => print!("{s}"),
    ///         Ok(AgentEvent::Done { final_text, .. }) => break,
    ///         Err(e) => return Err(e),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    pub fn run(self, task: &str) -> impl Stream<Item = Result<AgentEvent, AgentError>> + Send;
}

// ============================================================================
// EVENTS
// ============================================================================

/// Events emitted by the agent during execution.
///
/// Consumers process these to display progress, log activity, or forward
/// to other systems. All events are cheaply cloneable.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A chunk of text streamed from the model.
    ///
    /// Multiple `Text` events form the complete response. Buffer and
    /// display incrementally for real-time output.
    Text(String),

    /// The model has started a tool call.
    ///
    /// Emitted when the tool call is fully parsed (not during streaming).
    /// The actual execution hasn't started yet.
    ToolStart {
        /// Unique identifier for this tool call (from the model)
        id: String,
        /// Name of the tool being called
        name: String,
        /// Parsed input arguments
        input: serde_json::Value,
    },

    /// A tool has finished executing.
    ///
    /// The output will be sent back to the model in the next turn.
    ToolOutput {
        /// Matches the `id` from the corresponding `ToolStart`
        id: String,
        /// The tool's output
        output: ToolOutput,
    },

    /// A complete turn has finished.
    ///
    /// Emitted after the model response is complete and all tools (if any)
    /// have executed. Useful for progress tracking.
    TurnComplete {
        /// Which turn just completed (1-indexed)
        turn: u32,
        /// Token usage for this turn
        usage: TokenUsage,
    },

    /// Context was evicted due to token budget.
    ///
    /// Only emitted when context management triggers eviction.
    ContextEvicted {
        /// LLM-generated summary of evicted content
        summary: String,
        /// Range of turn numbers that were evicted
        turns_evicted: std::ops::Range<u32>,
    },

    /// The agent has completed successfully.
    ///
    /// This is always the last event in a successful run.
    Done {
        /// The complete final response text
        final_text: String,
    },
}

/// Token usage statistics from an LLM response.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
    /// Tokens in the input (prompt + history)
    pub input_tokens: u32,
    /// Tokens in the output (model response)
    pub output_tokens: u32,
}

impl TokenUsage {
    pub fn total(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

// ============================================================================
// TOOL SYSTEM
// ============================================================================

/// A capability the agent can invoke.
///
/// Tools are the agent's interface to the world. Each tool has a name,
/// description, input schema (JSON Schema), and an async execute function.
///
/// # Implementing a Tool
///
/// ```rust
/// use agent_core::{Tool, ToolOutput};
/// use async_trait::async_trait;
/// use serde_json::{json, Value};
///
/// struct EchoTool;
///
/// #[async_trait]
/// impl Tool for EchoTool {
///     fn name(&self) -> &str { "echo" }
///
///     fn description(&self) -> &str {
///         "Echo the input back. Useful for testing."
///     }
///
///     fn input_schema(&self) -> Value {
///         json!({
///             "type": "object",
///             "properties": {
///                 "message": {
///                     "type": "string",
///                     "description": "The message to echo"
///                 }
///             },
///             "required": ["message"]
///         })
///     }
///
///     async fn execute(&self, input: Value) -> ToolOutput {
///         match input.get("message").and_then(|v| v.as_str()) {
///             Some(msg) => ToolOutput::success(msg.to_string()),
///             None => ToolOutput::error("Missing 'message' parameter"),
///         }
///     }
/// }
/// ```
///
/// # Error Handling
///
/// Tools should **never panic**. Catch all errors and return them via
/// `ToolOutput::error()`. The agent will feed errors back to the model,
/// which can then self-correct.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's unique name. Used by the model to invoke the tool.
    ///
    /// Should be lowercase, alphanumeric with underscores. Examples:
    /// "bash", "read_file", "web_search"
    fn name(&self) -> &str;

    /// Human-readable description of what the tool does.
    ///
    /// This is shown to the model to help it decide when to use the tool.
    /// Be specific about capabilities and limitations.
    fn description(&self) -> &str;

    /// JSON Schema defining the tool's input parameters.
    ///
    /// The model uses this to construct valid input. Be precise about
    /// types, required fields, and provide descriptions for each property.
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given input.
    ///
    /// The input is the parsed JSON from the model's tool call. It should
    /// conform to `input_schema()`, but always validate—models occasionally
    /// produce invalid input.
    ///
    /// # Cancellation
    ///
    /// Long-running tools should respect cancellation. Check
    /// `tokio::task::yield_now()` periodically or use `tokio::select!`
    /// with a cancellation signal.
    async fn execute(&self, input: serde_json::Value) -> ToolOutput;
}

/// The result of a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// The output content (text, JSON, error message, etc.)
    pub content: String,
    /// Whether this represents an error
    pub is_error: bool,
}

impl ToolOutput {
    /// Create a successful tool output.
    pub fn success(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: false }
    }

    /// Create an error tool output.
    ///
    /// The model will see this as a failure and can attempt recovery.
    pub fn error(message: impl Into<String>) -> Self {
        Self { content: message.into(), is_error: true }
    }
}

// ============================================================================
// LLM CLIENT ABSTRACTION
// ============================================================================

/// Abstraction over LLM providers.
///
/// Each provider (Anthropic, OpenAI, Gemini, Bedrock) implements this trait.
/// The agent loop is provider-agnostic—it only uses this interface.
///
/// # Adding a New Provider
///
/// See the [Provider Normalization](./providers.md) documentation for
/// detailed guidance on implementing this trait for a new provider.
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Make a non-streaming completion request.
    ///
    /// Used internally for summarization. Most agent operations use
    /// `complete_stream` instead.
    async fn complete(&self, request: Request) -> Result<Response, LlmError>;

    /// Make a streaming completion request.
    ///
    /// Returns a stream of chunks as the model generates its response.
    /// This is the primary interface used by the agent loop.
    fn complete_stream(
        &self,
        request: Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send + 'static>>;
}

// ============================================================================
// REQUEST/RESPONSE TYPES
// ============================================================================

/// A request to an LLM.
///
/// This is the provider-agnostic request format. Each `LlmClient`
/// implementation translates this to the provider's native format.
#[derive(Debug, Clone)]
pub struct Request {
    /// System prompt (optional)
    pub system: Option<String>,
    /// Conversation history
    pub messages: Vec<Message>,
    /// Available tools
    pub tools: Vec<ToolDef>,
    /// Maximum tokens to generate
    pub max_tokens: u32,
}

/// A message in the conversation.
#[derive(Debug, Clone)]
pub struct Message {
    /// Who sent this message
    pub role: Role,
    /// Message content (can be multiple blocks)
    pub content: Vec<ContentBlock>,
}

/// Message sender role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// A block of content within a message.
///
/// Messages can contain multiple blocks. For example, an assistant message
/// might contain text followed by several tool calls.
#[derive(Debug, Clone)]
pub enum ContentBlock {
    /// Plain text content
    Text(String),

    /// A tool call from the assistant
    ToolUse {
        /// Unique ID for this call (generated by the model)
        id: String,
        /// Name of the tool to invoke
        name: String,
        /// Arguments to pass to the tool
        input: serde_json::Value,
    },

    /// A tool result from the user (response to a ToolUse)
    ToolResult {
        /// Matches the `id` of the corresponding ToolUse
        id: String,
        /// The tool's output
        content: String,
        /// Whether the tool execution failed
        is_error: bool,
    },
}

/// Tool definition sent to the LLM.
#[derive(Debug, Clone)]
pub struct ToolDef {
    /// Tool name (matches `Tool::name()`)
    pub name: String,
    /// Tool description (matches `Tool::description()`)
    pub description: String,
    /// JSON Schema for input (matches `Tool::input_schema()`)
    pub input_schema: serde_json::Value,
}

/// A complete (non-streaming) response from an LLM.
#[derive(Debug, Clone)]
pub struct Response {
    /// Response content
    pub content: Vec<ContentBlock>,
    /// Why the model stopped generating
    pub stop_reason: StopReason,
    /// Token usage for this request
    pub usage: TokenUsage,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Model finished its response naturally
    EndTurn,
    /// Model wants to use tools
    ToolUse,
    /// Hit the max_tokens limit
    MaxTokens,
}

/// A chunk from a streaming response.
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// A piece of text content
    Text(String),

    /// Start of a tool call
    ToolUseStart {
        id: String,
        name: String,
    },

    /// A fragment of tool call input JSON
    ///
    /// Multiple `ToolUseInput` chunks form the complete input.
    /// Accumulate and parse when `ToolUseDone` is received.
    ToolUseInput(String),

    /// End of the current tool call
    ToolUseDone,

    /// End of the complete message
    MessageDone {
        stop_reason: StopReason,
        usage: TokenUsage,
    },
}

// ============================================================================
// PROVIDERS
// ============================================================================

/// Anthropic Messages API client.
///
/// Supports Claude models via api.anthropic.com.
pub struct AnthropicClient { /* private */ }

impl AnthropicClient {
    /// Create a new client.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Anthropic API key (starts with "sk-ant-")
    /// * `model` - Model ID (e.g., "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022")
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self;
}

/// OpenAI Chat Completions API client.
///
/// Supports GPT models via api.openai.com. Also works with compatible APIs
/// (Azure OpenAI, local models) via `with_base_url`.
pub struct OpenAiClient { /* private */ }

impl OpenAiClient {
    /// Create a new client for api.openai.com.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self;

    /// Use a custom base URL (for Azure, local models, etc.)
    pub fn with_base_url(self, base_url: impl Into<String>) -> Self;
}

/// Google Gemini API client.
///
/// Supports Gemini models via generativelanguage.googleapis.com.
pub struct GeminiClient { /* private */ }

impl GeminiClient {
    /// Create a new client.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self;
}

/// AWS Bedrock client.
///
/// Supports Anthropic and other models hosted on AWS Bedrock.
/// Uses AWS SDK credentials (environment, IAM role, etc.)
///
/// Requires the `bedrock` feature.
#[cfg(feature = "bedrock")]
pub struct BedrockClient { /* private */ }

#[cfg(feature = "bedrock")]
impl BedrockClient {
    /// Create a new client with default AWS credentials.
    ///
    /// Credentials are resolved via the AWS SDK's default chain:
    /// environment variables, ~/.aws/credentials, IAM role, etc.
    pub async fn new(region: impl Into<String>, model: impl Into<String>) -> Self;
}

/// Mock client for testing.
///
/// Returns predetermined responses. Useful for unit tests.
pub struct MockClient { /* private */ }

impl MockClient {
    /// Create a mock that returns the given responses in sequence.
    pub fn new(responses: Vec<Response>) -> Self;
}

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

/// Persistent storage for evicted conversation context.
///
/// Requires the `persistence` feature. Uses SQLite with FTS5 for
/// full-text search of evicted content.
#[cfg(feature = "persistence")]
pub struct ContextStore { /* private */ }

#[cfg(feature = "persistence")]
impl ContextStore {
    /// Open or create a context store at the given path.
    ///
    /// The database is created if it doesn't exist. WAL mode is enabled
    /// for concurrent read/write access.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, ContextError>;

    /// Create an in-memory context store.
    ///
    /// Useful for testing. Data is lost when the store is dropped.
    pub fn in_memory() -> Result<Self, ContextError>;
}

// ============================================================================
// ERRORS
// ============================================================================

/// Errors that can occur during agent execution.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// An LLM API error occurred.
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    /// The agent was cancelled via its cancellation token.
    #[error("Agent cancelled")]
    Cancelled,

    /// The agent exceeded the maximum number of turns.
    #[error("Exceeded maximum turns ({0})")]
    MaxTurnsExceeded(u32),
}

/// Errors from LLM API calls.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// Rate limited (HTTP 429). Includes retry-after hint if provided.
    #[error("Rate limited (retry after {retry_after:?})")]
    RateLimit { retry_after: Option<std::time::Duration> },

    /// Service overloaded (HTTP 529 or equivalent).
    #[error("Service overloaded")]
    Overloaded,

    /// Invalid request (HTTP 400). Indicates a bug in request construction.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Authentication failed (HTTP 401/403).
    #[error("Authentication failed: {0}")]
    Auth(String),

    /// Network error (connection failed, timeout, etc.)
    #[error("Network error: {0}")]
    Network(String),
}

/// Errors from context store operations.
#[cfg(feature = "persistence")]
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("SQLite error: {0}")]
    Sqlite(String),

    #[error("IO error: {0}")]
    Io(String),
}
```

---

## agent-tools Public API

```rust
//! Built-in tool implementations for common agent capabilities.
//!
//! All tools implement `agent_core::Tool` and can be passed to
//! `AgentBuilder::tools()`.

use agent_core::{Tool, ToolOutput, ContextStore};
use std::path::PathBuf;
use std::time::Duration;

// ============================================================================
// BASH TOOL
// ============================================================================

/// Execute bash commands.
///
/// Supports configurable timeouts, working directory, and sandboxing.
///
/// # Input Schema
///
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "command": {
///       "type": "string",
///       "description": "The bash command to execute"
///     }
///   },
///   "required": ["command"]
/// }
/// ```
///
/// # Output
///
/// Returns stdout and stderr combined. If the command fails (non-zero exit),
/// returns an error output with the exit code and stderr.
///
/// # Security
///
/// By default, commands run with the same permissions as the agent process.
/// Use `with_sandbox()` to restrict filesystem access and block dangerous
/// commands in untrusted environments.
pub struct BashTool { /* private */ }

impl BashTool {
    /// Create a new bash tool with default settings.
    ///
    /// Default timeout: 120 seconds
    /// Default working directory: current directory
    /// Default sandbox: none (full access)
    pub fn new() -> Self;

    /// Set the maximum execution time for commands.
    ///
    /// Commands exceeding this timeout are killed and return an error.
    pub fn with_timeout(self, timeout: Duration) -> Self;

    /// Set the working directory for command execution.
    pub fn with_working_dir(self, dir: PathBuf) -> Self;

    /// Enable sandboxing with the given configuration.
    ///
    /// Sandboxing restricts filesystem access and blocks dangerous commands.
    pub fn with_sandbox(self, config: SandboxConfig) -> Self;
}

impl Default for BashTool {
    fn default() -> Self { Self::new() }
}

/// Sandbox configuration for BashTool.
#[derive(Debug, Clone, Default)]
pub struct SandboxConfig {
    /// Paths the command is allowed to access.
    ///
    /// If empty, no path restrictions are applied.
    pub allowed_paths: Vec<PathBuf>,

    /// Commands that are blocked from execution.
    ///
    /// Matched against the first word of the command. Examples:
    /// "rm", "sudo", "chmod"
    pub denied_commands: Vec<String>,

    /// Environment variables to pass to the command.
    ///
    /// If Some, only these variables are passed. If None, inherits
    /// the agent's environment.
    pub env: Option<Vec<(String, String)>>,
}

// ============================================================================
// FILE TOOL
// ============================================================================

/// Read and write files.
///
/// Supports optional path jailing to restrict access to a directory tree.
///
/// # Input Schema
///
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "operation": {
///       "type": "string",
///       "enum": ["read", "write"],
///       "description": "Whether to read or write"
///     },
///     "path": {
///       "type": "string",
///       "description": "File path (absolute or relative to root)"
///     },
///     "content": {
///       "type": "string",
///       "description": "Content to write (required for write operation)"
///     }
///   },
///   "required": ["operation", "path"]
/// }
/// ```
///
/// # Output
///
/// For read: the file contents
/// For write: confirmation message
pub struct FileTool { /* private */ }

impl FileTool {
    /// Create a new file tool with no path restrictions.
    pub fn new() -> Self;

    /// Restrict file operations to a directory tree.
    ///
    /// All paths are resolved relative to this root. Attempts to escape
    /// via ".." or absolute paths are blocked.
    pub fn with_root(self, root: PathBuf) -> Self;

    /// Set the maximum file size for read operations.
    ///
    /// Files larger than this return an error suggesting the model
    /// read a portion or use a different approach.
    ///
    /// Default: 1 MB
    pub fn with_max_read_size(self, bytes: usize) -> Self;
}

impl Default for FileTool {
    fn default() -> Self { Self::new() }
}

// ============================================================================
// RECALL TOOL
// ============================================================================

/// Search and retrieve evicted conversation context.
///
/// This tool allows the model to access information that was evicted
/// from the active context due to token budget constraints. It queries
/// the SQLite FTS5 index.
///
/// Requires a `ContextStore` from agent-core (with `persistence` feature).
///
/// # Input Schema
///
/// ```json
/// {
///   "type": "object",
///   "properties": {
///     "query": {
///       "type": "string",
///       "description": "Search terms (full-text search)"
///     },
///     "tool_filter": {
///       "type": "string",
///       "description": "Optional: only search outputs from this tool"
///     }
///   },
///   "required": ["query"]
/// }
/// ```
///
/// # Output
///
/// Returns matching snippets with turn numbers for reference. Results are
/// ranked by relevance and limited to the top 5 matches.
#[cfg(feature = "agent-core/persistence")]
pub struct RecallTool { /* private */ }

#[cfg(feature = "agent-core/persistence")]
impl RecallTool {
    /// Create a recall tool backed by the given context store.
    ///
    /// The store should be the same one passed to `AgentBuilder::context_store()`.
    pub fn new(store: agent_core::ContextStore) -> Self;

    /// Set the maximum number of results to return.
    ///
    /// Default: 5
    pub fn with_max_results(self, n: usize) -> Self;
}
```

---

## agent-session Public API

```rust
//! Job orchestration layer for running multiple agents concurrently.
//!
//! The session layer wraps agents in a job queue, allowing you to:
//! - Submit tasks that run in the background
//! - Track progress of running jobs
//! - Cancel jobs gracefully
//! - Collect results when ready
//!
//! This crate is optional. For simpler use cases, use `agent-core` directly.

use agent_core::{Agent, AgentEvent, AgentError};
use std::sync::Arc;

// ============================================================================
// SESSION
// ============================================================================

/// A job queue for running agents concurrently.
///
/// Sessions manage the lifecycle of agent jobs: submission, progress tracking,
/// cancellation, and result collection.
///
/// # Example
///
/// ```rust
/// let session = Session::new();
///
/// // Submit multiple jobs
/// let job1 = session.submit(agent1, "Task 1".into());
/// let job2 = session.submit(agent2, "Task 2".into());
///
/// // Wait for results
/// while let Some(result) = session.next_completed().await {
///     println!("Job {} completed: {:?}", result.id, result.outcome);
/// }
/// ```
///
/// # Thread Safety
///
/// Session is `Clone` and `Send + Sync`. All operations are thread-safe.
/// Multiple tasks can submit jobs, peek at progress, and collect results
/// concurrently.
#[derive(Clone)]
pub struct Session { /* private: Arc<SessionInner> */ }

impl Session {
    /// Create a new session.
    pub fn new() -> Self;

    /// Submit a task to run in the background.
    ///
    /// Returns immediately with a `JobId`. The agent runs in a spawned
    /// tokio task. Use `peek()` to check progress or `next_completed()`
    /// to wait for the result.
    ///
    /// # Arguments
    ///
    /// * `agent` - The agent to run (consumed)
    /// * `task` - The task string to pass to `agent.run()`
    pub fn submit(&self, agent: Agent, task: String) -> JobId;

    /// Get a snapshot of a running job's events so far.
    ///
    /// Returns `None` if the job ID doesn't exist or has already completed
    /// (completed jobs are removed from the active set).
    ///
    /// The returned events are a clone—the job continues running independently.
    pub fn peek(&self, id: &JobId) -> Option<Vec<AgentEvent>>;

    /// Cancel a running job.
    ///
    /// The job's cancellation token is triggered and its task is aborted.
    /// The job will complete with `AgentError::Cancelled`.
    ///
    /// No-op if the job doesn't exist or has already completed.
    pub fn cancel(&self, id: &JobId);

    /// List all currently active (not yet completed) job IDs.
    pub fn active_jobs(&self) -> Vec<JobId>;

    /// Wait for the next completed job.
    ///
    /// Returns `None` if the session is dropped and no more results
    /// will arrive.
    ///
    /// Jobs complete in arbitrary order—not necessarily the order they
    /// were submitted.
    pub async fn next_completed(&self) -> Option<JobResult>;

    /// Non-blocking check for a completed job.
    ///
    /// Returns `None` immediately if no job has completed yet.
    pub fn try_next_completed(&self) -> Option<JobResult>;
}

impl Default for Session {
    fn default() -> Self { Self::new() }
}

// ============================================================================
// JOB TYPES
// ============================================================================

/// Unique identifier for a submitted job.
///
/// Job IDs are UUIDs and are unique across all sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub uuid::Uuid);

impl JobId {
    /// Generate a new random job ID.
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The result of a completed job.
#[derive(Debug)]
pub struct JobResult {
    /// The job's unique identifier
    pub id: JobId,

    /// The outcome: final text on success, error on failure
    pub outcome: Result<String, AgentError>,

    /// Complete event history from the job's execution
    ///
    /// Includes all events emitted during the run, useful for
    /// logging, replay, or detailed analysis.
    pub events: Vec<AgentEvent>,
}
```

---

## Summary: API Surface Size

| Crate | Public Types | Public Traits | Public Functions |
|-------|-------------|---------------|------------------|
| **agent-core** | 21 | 2 | ~15 constructors/methods |
| **agent-tools** | 4 | 0 | ~10 constructors/methods |
| **agent-session** | 3 | 0 | ~8 methods |

The API is deliberately narrow. Most types are data carriers (structs, enums) with minimal methods. The two traits (`LlmClient`, `Tool`) are the only extension points. This keeps the learning curve low and makes the library predictable.
