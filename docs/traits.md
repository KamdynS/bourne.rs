# Trait Catalog

This document provides a complete reference for all public traits in agent-rs. Following the radical simplicity philosophy, there are exactly **two traits** in the entire system. Everything else is concrete types.

## Design Philosophy

Traits exist for two reasons:

1. **Genuine polymorphism:** Multiple implementations with different behavior
2. **External extensibility:** Third parties need to provide their own implementations

agent-rs has exactly two such cases:
- `LlmClient`: Multiple providers (Anthropic, OpenAI, etc.) with different APIs
- `Tool`: Users provide their own capabilities

Everything else is a struct or enum. This isn't accidental—more traits mean more abstraction, more indirection, more cognitive overhead, and more code. Two traits handle all the genuine extension points.

### What Could Have Been Traits (But Isn't)

| Component | Why Not a Trait |
|-----------|-----------------|
| `Agent` | One implementation. No one needs a different agent loop. |
| `ContextStore` | One implementation (SQLite). If you need different storage, fork it. |
| `Session` | One implementation. The abstraction adds nothing. |
| `TokenEstimator` | Could be, but the simple heuristic works. Overengineering. |
| `EventHandler` | Users consume a Stream. They don't need to implement a trait. |

---

## LlmClient

### Purpose

Abstracts over LLM provider APIs. The agent loop calls this trait and is completely unaware of which provider is actually handling requests.

### Definition

```rust
/// Abstraction over LLM provider APIs.
///
/// # Why This Trait Exists
///
/// Different LLM providers have different APIs:
/// - Different authentication mechanisms (API keys, AWS SigV4, OAuth)
/// - Different request/response formats (tool_use vs function_calling)
/// - Different streaming formats (SSE events vs chunked JSON)
/// - Different endpoint structures and rate limiting
///
/// This trait normalizes all of that. The agent loop works with
/// provider-agnostic types (`Request`, `Response`, `StreamChunk`)
/// and each provider translates to/from its native format.
///
/// # For Agent Authors
///
/// You don't implement this trait. You choose a provider:
///
/// ```rust
/// let client = AnthropicClient::new(&api_key, "claude-sonnet-4-20250514");
/// let agent = AgentBuilder::new(Box::new(client)).build();
/// ```
///
/// # For Provider Implementers
///
/// Implement this trait to add support for a new LLM provider.
/// See [Provider Normalization](./providers.md) for detailed guidance.
///
/// # Design Notes
///
/// - `complete` is non-streaming, used for summarization during eviction
/// - `complete_stream` is the primary interface, used for all agent turns
/// - The trait is object-safe (`Box<dyn LlmClient>`) for runtime flexibility
/// - Both methods take owned `Request` to allow providers to modify it
#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Make a non-streaming completion request.
    ///
    /// # When This Is Used
    ///
    /// - Context eviction: generating summaries of evicted turns
    /// - Testing: simpler to work with than streams
    ///
    /// # Implementation Notes
    ///
    /// - Must handle authentication for your provider
    /// - Must translate `Request` to provider-native format
    /// - Must translate response back to `Response`
    /// - Should implement retry logic for transient errors
    ///
    /// # Errors
    ///
    /// Return appropriate `LlmError` variants:
    /// - `RateLimit` for 429 responses (include retry-after if available)
    /// - `Overloaded` for 529/503 responses
    /// - `InvalidRequest` for 400 responses
    /// - `Auth` for 401/403 responses
    /// - `Network` for connection failures, timeouts, etc.
    async fn complete(&self, request: Request) -> Result<Response, LlmError>;

    /// Make a streaming completion request.
    ///
    /// # Why Streaming?
    ///
    /// Streaming provides:
    /// - Real-time display: users see text as it's generated
    /// - Early cancellation: can stop mid-response
    /// - Memory efficiency: don't buffer entire response
    ///
    /// # When This Is Used
    ///
    /// Every agent turn. This is the primary interface.
    ///
    /// # Implementation Notes
    ///
    /// The returned stream must:
    /// - Yield `Text` chunks as they arrive
    /// - Yield `ToolUseStart` when a tool call begins
    /// - Yield `ToolUseInput` chunks as tool arguments stream
    /// - Yield `ToolUseDone` when a tool call is complete
    /// - Yield exactly one `MessageDone` at the end
    /// - Handle SSE or chunked format as appropriate for your provider
    ///
    /// # Error Handling
    ///
    /// - Yield `Err(LlmError)` for recoverable errors (stream terminates)
    /// - The agent loop handles retry logic at a higher level
    ///
    /// # Lifetime
    ///
    /// The returned stream must be `'static` and `Send`. It cannot borrow
    /// from `self`—clone any needed data into the stream.
    fn complete_stream(
        &self,
        request: Request,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send + 'static>>;
}
```

### Implementations

| Implementation | Provider | Auth | Streaming Format |
|----------------|----------|------|------------------|
| `AnthropicClient` | Anthropic Messages API | `x-api-key` header | SSE with typed events |
| `OpenAiClient` | OpenAI Chat Completions | `Authorization: Bearer` | SSE with `data:` lines |
| `GeminiClient` | Google Generative AI | `x-goog-api-key` | SSE |
| `BedrockClient` | AWS Bedrock | AWS SigV4 | Chunked JSON lines |
| `MockClient` | Testing | None | Immediate yield |

### Extension Point

To add a new provider:

1. Create a struct with connection configuration
2. Implement `LlmClient`
3. Handle the translation between `Request`/`Response` and native format

```rust
pub struct MyProviderClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

#[async_trait]
impl LlmClient for MyProviderClient {
    async fn complete(&self, request: Request) -> Result<Response, LlmError> {
        // 1. Translate Request to provider format
        let native_request = self.translate_request(request);

        // 2. Make HTTP request
        let response = self.http
            .post(&self.endpoint)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&native_request)
            .send()
            .await
            .map_err(|e| LlmError::Network(e.to_string()))?;

        // 3. Handle error status codes
        match response.status() {
            s if s == 429 => return Err(LlmError::RateLimit {
                retry_after: parse_retry_after(&response),
            }),
            s if s == 401 || s == 403 => return Err(LlmError::Auth(
                response.text().await.unwrap_or_default()
            )),
            // ... etc
        }

        // 4. Parse and translate response
        let native_response: MyProviderResponse = response.json().await?;
        Ok(self.translate_response(native_response))
    }

    fn complete_stream(&self, request: Request) -> /* ... */ {
        // Similar, but parse streaming format
    }
}
```

---

## Tool

### Purpose

Defines a capability the agent can invoke. Tools are how the agent interacts with the outside world—running commands, reading files, making API calls, etc.

### Definition

```rust
/// A capability the agent can invoke.
///
/// # Why This Trait Exists
///
/// Tools are the agent's "hands." Without tools, an agent can only
/// generate text. With tools, it can:
/// - Execute shell commands
/// - Read and write files
/// - Search the web
/// - Call external APIs
/// - Query databases
/// - Anything you can express in async Rust
///
/// The `Tool` trait is the extension point for adding new capabilities.
///
/// # For Agent Authors
///
/// Use built-in tools from `agent-tools` or implement your own:
///
/// ```rust
/// let agent = AgentBuilder::new(client)
///     .tools(vec![
///         Box::new(BashTool::new()),
///         Box::new(FileTool::new()),
///         Box::new(MyCustomTool::new()),
///     ])
///     .build();
/// ```
///
/// # Design Decisions
///
/// **Why async?** Tools often do I/O—file access, HTTP requests, subprocess
/// execution. Async allows concurrent execution of multiple tools.
///
/// **Why JSON Schema?** It's the standard way to describe structured input.
/// All major LLM providers understand it for tool definitions.
///
/// **Why ToolOutput instead of Result?** Both success and failure are valid
/// outputs that the model should see. `is_error: true` tells the model
/// something went wrong; the model can then self-correct.
///
/// **Why no state?** Tools receive input and return output. Any state
/// (connections, caches) lives in the tool struct, managed by the implementer.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's unique identifier.
    ///
    /// # Requirements
    ///
    /// - Must be unique within the tool registry
    /// - Should be lowercase, alphanumeric, with underscores
    /// - Used by the model to invoke the tool
    ///
    /// # Examples
    ///
    /// - `"bash"` - execute shell commands
    /// - `"read_file"` - read file contents
    /// - `"web_search"` - search the web
    /// - `"sql_query"` - execute database queries
    ///
    /// # What Happens with Duplicates
    ///
    /// If two tools have the same name, the later one shadows the earlier.
    /// This allows overriding built-in tools with custom implementations.
    fn name(&self) -> &str;

    /// Human-readable description of the tool's purpose.
    ///
    /// # Why This Matters
    ///
    /// The model reads this to decide when to use the tool. A good
    /// description helps the model use tools appropriately.
    ///
    /// # Guidelines
    ///
    /// - Be specific about what the tool does
    /// - Mention limitations (e.g., "read-only", "max 1MB")
    /// - Explain when to use vs. not use
    /// - Keep it concise (1-3 sentences)
    ///
    /// # Example
    ///
    /// ```text
    /// "Execute a bash command in the working directory. Supports
    ///  standard Unix commands. Commands timeout after 120 seconds.
    ///  Use for file operations, system queries, and running programs."
    /// ```
    fn description(&self) -> &str;

    /// JSON Schema defining valid input.
    ///
    /// # Purpose
    ///
    /// The schema tells the model what arguments the tool accepts.
    /// Models use this to construct valid tool calls.
    ///
    /// # Requirements
    ///
    /// - Must be valid JSON Schema (draft-07 or compatible)
    /// - Root must be `"type": "object"`
    /// - Include `"properties"` with parameter definitions
    /// - Include `"required"` array for mandatory parameters
    /// - Add `"description"` to each property
    ///
    /// # Example
    ///
    /// ```rust
    /// fn input_schema(&self) -> Value {
    ///     json!({
    ///         "type": "object",
    ///         "properties": {
    ///             "path": {
    ///                 "type": "string",
    ///                 "description": "Path to the file to read"
    ///             },
    ///             "encoding": {
    ///                 "type": "string",
    ///                 "description": "Text encoding (default: utf-8)",
    ///                 "enum": ["utf-8", "ascii", "latin-1"]
    ///             }
    ///         },
    ///         "required": ["path"]
    ///     })
    /// }
    /// ```
    ///
    /// # Validation
    ///
    /// Models occasionally produce invalid input despite the schema.
    /// Always validate in `execute()` and return errors gracefully.
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool with the given input.
    ///
    /// # Arguments
    ///
    /// `input` - Parsed JSON matching `input_schema()`. The model
    /// constructed this based on the schema, but validation isn't
    /// guaranteed—always check required fields.
    ///
    /// # Return Value
    ///
    /// Return `ToolOutput`:
    /// - `ToolOutput::success(content)` - operation succeeded
    /// - `ToolOutput::error(message)` - operation failed
    ///
    /// The model sees both as tool results. Errors trigger self-correction.
    ///
    /// # Error Handling
    ///
    /// **Never panic.** Catch all errors and return `ToolOutput::error()`.
    ///
    /// ```rust
    /// async fn execute(&self, input: Value) -> ToolOutput {
    ///     let path = match input.get("path").and_then(|v| v.as_str()) {
    ///         Some(p) => p,
    ///         None => return ToolOutput::error("Missing 'path' parameter"),
    ///     };
    ///
    ///     match std::fs::read_to_string(path) {
    ///         Ok(content) => ToolOutput::success(content),
    ///         Err(e) => ToolOutput::error(format!("Failed to read file: {e}")),
    ///     }
    /// }
    /// ```
    ///
    /// # Concurrency
    ///
    /// Multiple tools may execute concurrently (when the model issues
    /// multiple tool calls). Ensure your implementation is safe for
    /// concurrent use.
    ///
    /// # Cancellation
    ///
    /// Long-running tools should respect cancellation. Periodically
    /// check `tokio::task::yield_now()` or use `tokio::select!` with
    /// a cancellation signal.
    async fn execute(&self, input: serde_json::Value) -> ToolOutput;
}
```

### Built-in Implementations

From `agent-tools`:

| Tool | Name | Purpose |
|------|------|---------|
| `BashTool` | `"bash"` | Execute shell commands with sandboxing and timeouts |
| `FileTool` | `"read_file"`, `"write_file"` | File system operations with path jailing |
| `RecallTool` | `"recall"` | Search evicted context via FTS5 |

### Extension Point

Implementing a custom tool:

```rust
use agent_core::{Tool, ToolOutput};
use async_trait::async_trait;
use serde_json::{json, Value};

/// A tool that fetches weather data.
///
/// # Why This Tool?
///
/// Demonstrates a typical "API call" tool pattern. The model provides
/// a location, the tool fetches data from an external service.
pub struct WeatherTool {
    api_key: String,
    http: reqwest::Client,
}

impl WeatherTool {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get current weather for a location. Returns temperature, \
         conditions, and humidity. Use for weather-related queries."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates (e.g., 'London' or '51.5,-0.1')"
                }
            },
            "required": ["location"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        // Extract and validate input
        let location = match input.get("location").and_then(|v| v.as_str()) {
            Some(loc) => loc,
            None => return ToolOutput::error("Missing 'location' parameter"),
        };

        // Make API call
        let url = format!(
            "https://api.weather.com/v1/current?q={}&key={}",
            urlencoding::encode(location),
            self.api_key
        );

        let response = match self.http.get(&url).send().await {
            Ok(r) => r,
            Err(e) => return ToolOutput::error(format!("Request failed: {e}")),
        };

        if !response.status().is_success() {
            return ToolOutput::error(format!(
                "Weather API error: {}",
                response.status()
            ));
        }

        // Parse response
        let data: WeatherResponse = match response.json().await {
            Ok(d) => d,
            Err(e) => return ToolOutput::error(format!("Invalid response: {e}")),
        };

        // Format output for the model
        ToolOutput::success(format!(
            "Weather in {}: {}°C, {}, Humidity: {}%",
            location, data.temp, data.conditions, data.humidity
        ))
    }
}
```

---

## Non-Traits: Why Everything Else Is Concrete

### Agent

**Could be a trait for:** Alternative agent loop implementations (chain-of-thought, multi-agent, etc.)

**Why it's not:** The agent loop is the core innovation. If you need different behavior, you're building a different system. The extension points (LlmClient, Tool) provide sufficient flexibility. Making Agent a trait would complicate the API for a non-existent use case.

### ContextStore

**Could be a trait for:** Alternative storage backends (Redis, PostgreSQL, file system)

**Why it's not:** SQLite with FTS5 is the right choice for local, embedded, full-text-searchable storage. The complexity of abstracting storage isn't worth it. If you need different storage, fork the ContextStore implementation.

### Session

**Could be a trait for:** Alternative job management strategies

**Why it's not:** The Session API is thin—it's just a concurrent map and a channel. There's no behavior to swap out. If you need different orchestration, build it yourself using Agent directly.
