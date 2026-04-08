use async_trait::async_trait;
use serde_json::Value;

/// A capability the agent can invoke.
///
/// Tools are the agent's interface to the world. Each tool has a name,
/// description, input schema (JSON Schema), and an async execute function.
///
/// # Error Handling
///
/// Tools should **never panic**. Catch all errors and return them via
/// `ToolOutput::error()`. The agent will feed errors back to the model,
/// which can then self-correct.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool's unique name. Used by the model to invoke the tool.
    fn name(&self) -> &str;

    /// Human-readable description of what the tool does.
    fn description(&self) -> &str;

    /// JSON Schema defining the tool's input parameters.
    fn input_schema(&self) -> Value;

    /// Execute the tool with the given input.
    async fn execute(&self, input: Value) -> ToolOutput;
}

/// The result of a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

impl ToolOutput {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: message.into(),
            is_error: true,
        }
    }
}
