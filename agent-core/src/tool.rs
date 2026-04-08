//! Tool trait and output types.
//!
//! Tools are the agent's interface to the world. Without tools, an LLM can only
//! generate text. With tools, it can execute code, read files, search the web,
//! and more.
//!
//! # The Tool Trait
//!
//! This is one of only two traits in agent-rs (the other is LlmClient). We use
//! a trait here because tools are the primary extension point—users will create
//! their own tools for their specific use cases.
//!
//! # Error Philosophy
//!
//! Tools should **never panic**. All errors should be returned via ToolOutput::error().
//! The agent feeds errors back to the model, which can then self-correct:
//!
//! ```text
//! Model: "Run `cat /nonexistent`"
//! Tool: ToolOutput::error("No such file: /nonexistent")
//! Model: "Let me check what files exist first..." (self-corrects)
//! ```
//!
//! This "errors as data" approach is central to robust agent behavior.

use async_trait::async_trait;
use serde_json::Value;

/// A capability the agent can invoke.
///
/// Tools bridge the gap between the model's reasoning and real-world actions.
/// Each tool has:
///
/// - **name**: Unique identifier the model uses to call it
/// - **description**: Helps the model decide when to use it
/// - **input_schema**: JSON Schema defining expected parameters
/// - **execute**: Async function that does the actual work
///
/// # Implementing a Tool
///
/// ```ignore
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
///                 "message": { "type": "string" }
///             },
///             "required": ["message"]
///         })
///     }
///
///     async fn execute(&self, input: Value) -> ToolOutput {
///         match input.get("message").and_then(|v| v.as_str()) {
///             Some(msg) => ToolOutput::success(msg),
///             None => ToolOutput::error("Missing 'message' parameter"),
///         }
///     }
/// }
/// ```
///
/// # Tool Execution
///
/// When the model calls a tool, the agent:
/// 1. Looks up the tool by name
/// 2. Calls `execute()` with the model's JSON input
/// 3. Sends the ToolOutput back to the model as a ToolResult
///
/// Multiple tool calls in the same response are executed in parallel via
/// `futures::future::join_all`.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's unique name.
    ///
    /// This is what the model uses to invoke the tool. Should be lowercase,
    /// alphanumeric with underscores. Examples: "bash", "read_file", "web_search".
    ///
    /// Tool names must be unique within an agent's tool set. If two tools have
    /// the same name, the later one shadows the earlier one.
    fn name(&self) -> &str;

    /// Human-readable description of what the tool does.
    ///
    /// This is shown to the model to help it decide when to use the tool.
    /// Write this for an LLM audience:
    /// - Be specific about what the tool can and cannot do
    /// - Mention any important limitations or requirements
    /// - Give examples of when to use it
    fn description(&self) -> &str;

    /// JSON Schema defining the tool's input parameters.
    ///
    /// The model uses this to construct valid input. The schema should:
    /// - Use "type": "object" at the top level
    /// - Define properties with types and descriptions
    /// - List required fields in "required" array
    ///
    /// Example:
    /// ```json
    /// {
    ///   "type": "object",
    ///   "properties": {
    ///     "command": {
    ///       "type": "string",
    ///       "description": "The bash command to execute"
    ///     },
    ///     "timeout": {
    ///       "type": "integer",
    ///       "description": "Timeout in seconds (default: 30)"
    ///     }
    ///   },
    ///   "required": ["command"]
    /// }
    /// ```
    fn input_schema(&self) -> Value;

    /// Execute the tool with the given input.
    ///
    /// The input is JSON from the model's tool call. It should conform to
    /// `input_schema()`, but always validate—models occasionally produce
    /// invalid input.
    ///
    /// # Guidelines
    ///
    /// - **Never panic**. Return ToolOutput::error() for all failures.
    /// - **Be defensive**. Validate input, handle edge cases.
    /// - **Be informative**. Error messages should help the model self-correct.
    /// - **Be bounded**. Long-running operations should have timeouts.
    ///
    /// # Cancellation
    ///
    /// Long-running tools should respect cancellation. Periodically call
    /// `tokio::task::yield_now()` or use `tokio::select!` with a cancellation
    /// signal to allow graceful shutdown.
    async fn execute(&self, input: Value) -> ToolOutput;
}

/// The result of a tool execution.
///
/// Every tool execution produces a ToolOutput—either success or error.
/// This is sent back to the model as a ToolResult content block.
///
/// # Why String Content?
///
/// Tool output is always a String because:
/// 1. LLMs process text, not structured data
/// 2. It's the lowest common denominator across tools
/// 3. Structured output can be JSON-serialized into the string
///
/// For structured output, serialize to JSON:
/// ```ignore
/// ToolOutput::success(serde_json::to_string(&my_struct).unwrap())
/// ```
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// The output content. Could be command output, file contents, JSON, etc.
    pub content: String,

    /// Whether this represents an error.
    ///
    /// When true, the model knows the tool call failed and can try to recover.
    /// Error content should explain what went wrong clearly enough for the
    /// model to self-correct.
    pub is_error: bool,
}

impl ToolOutput {
    /// Create a successful tool output.
    ///
    /// Use this when the tool executed successfully, regardless of what the
    /// output contains. Even empty output or "not found" results are successes
    /// if the tool ran correctly.
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    /// Create an error tool output.
    ///
    /// Use this when the tool failed to execute properly:
    /// - Invalid input parameters
    /// - Permission denied
    /// - Resource not found
    /// - Timeout exceeded
    ///
    /// The error message should be clear and actionable. The model will see
    /// this and attempt to recover or try a different approach.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: message.into(),
            is_error: true,
        }
    }
}
