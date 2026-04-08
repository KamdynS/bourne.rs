//! Mock Demo - Educational Example (No API Key Required)
//!
//! This example demonstrates the agent architecture using MockClient.
//! It's for learning and documentation - see `live_simple` for real API testing.
//!
//! # What This Shows
//!
//! 1. **Agent construction**: AgentBuilder configuration
//! 2. **Tool implementation**: The Tool trait in action
//! 3. **Event streaming**: Processing AgentEvents as they occur
//! 4. **The agentic loop**: Tool calls → results → continuation
//!
//! # Running
//!
//! ```bash
//! cargo run --example mock_demo -p agent-core
//! ```
//!
//! # Note
//!
//! This uses scripted responses - behavior is deterministic but not real.
//! For actual integration testing, see `live_simple`.

use agent_core::{
    mock::{MockClient, MockResponse},
    AgentBuilder, AgentEvent, Tool, ToolOutput,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

/// A simple echo tool for demonstration.
///
/// This tool just echoes back its input - useful for testing
/// that the tool calling mechanism works correctly.
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn description(&self) -> &str {
        "Echo back the input message. Use this to repeat or confirm information."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to echo"
                }
            },
            "required": ["message"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        let message = input
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("(no message)");

        ToolOutput::success(format!("Echo: {message}"))
    }
}

/// A simple calculator tool for demonstration.
///
/// Shows how tools can perform actual computation and return results.
struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Perform basic arithmetic. Supports add, subtract, multiply, divide."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First operand"
                },
                "b": {
                    "type": "number",
                    "description": "Second operand"
                }
            },
            "required": ["operation", "a", "b"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        let op = input.get("operation").and_then(|v| v.as_str());
        let a = input.get("a").and_then(|v| v.as_f64());
        let b = input.get("b").and_then(|v| v.as_f64());

        match (op, a, b) {
            (Some(op), Some(a), Some(b)) => {
                let result = match op {
                    "add" => a + b,
                    "subtract" => a - b,
                    "multiply" => a * b,
                    "divide" if b != 0.0 => a / b,
                    "divide" => return ToolOutput::error("Division by zero"),
                    _ => return ToolOutput::error(format!("Unknown operation: {op}")),
                };
                ToolOutput::success(format!("{result}"))
            }
            _ => ToolOutput::error("Missing required fields: operation, a, b"),
        }
    }
}

#[tokio::main]
async fn main() {
    println!("=== Simple Agent Example ===\n");

    // Create a mock client with scripted responses.
    //
    // In a real application, you'd use AnthropicClient or OpenAiClient:
    //   let client = AnthropicClient::new(api_key, "claude-sonnet-4-20250514");
    //
    // The mock lets us demonstrate the flow without an API key.
    let client = MockClient::new(vec![
        // First response: LLM decides to use the calculator tool
        MockResponse::tool_call(
            "Let me calculate that for you.",
            "tool_1",
            "calculator",
            json!({
                "operation": "multiply",
                "a": 6,
                "b": 7
            }),
        ),
        // Second response: LLM uses the result
        MockResponse::text("The answer is 42. That's the meaning of life!"),
    ]);

    // Build the agent with tools.
    //
    // AgentBuilder is the entry point for configuring an agent:
    // - client: The LLM provider
    // - tools: Capabilities the LLM can use
    // - system_prompt: Instructions for the LLM
    // - max_tokens: Response length limit
    // - max_turns: Prevent infinite loops
    let agent = AgentBuilder::new(Box::new(client))
        .tools(vec![
            Box::new(EchoTool),
            Box::new(CalculatorTool),
        ])
        .system_prompt("You are a helpful assistant with access to tools.")
        .max_tokens(1024)
        .max_turns(10)
        .build();

    // Run the agent with a task.
    //
    // The agent returns a stream of events that we can process
    // as they occur. This is useful for:
    // - Showing progress to users
    // - Logging tool calls
    // - Implementing cancellation
    let task = "What is 6 times 7?";
    println!("Task: {task}\n");
    println!("--- Agent Events ---\n");

    // Track tool IDs so we can show tool names in ToolEnd events
    let mut tool_names: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    // Pin the stream so we can iterate over it.
    // The stream from Agent::run() isn't Unpin, so we need tokio::pin!
    let stream = agent.run(task);
    tokio::pin!(stream);

    while let Some(event) = stream.next().await {
        match event {
            Ok(AgentEvent::Text(text)) => {
                // The LLM is producing text output
                print!("{text}");
            }
            Ok(AgentEvent::ToolStart { id, name, input }) => {
                // The LLM is calling a tool
                println!("\n[Tool call: {name}]");
                println!("  Input: {input}");
                tool_names.insert(id, name);
            }
            Ok(AgentEvent::ToolEnd { id, output }) => {
                // A tool has completed
                let name = tool_names.get(&id).map(|s| s.as_str()).unwrap_or("unknown");
                println!("  Output: {}", output.content);
                if output.is_error {
                    println!("  (This was an error)");
                }
                println!("[End {name}]\n");
            }
            Ok(AgentEvent::TurnComplete { turn, usage }) => {
                // An LLM turn has completed
                println!(
                    "\n[Turn {turn} complete - {} input tokens, {} output tokens]\n",
                    usage.input_tokens, usage.output_tokens
                );
            }
            Ok(AgentEvent::Done { final_text }) => {
                // The agent has finished
                println!("\n--- Agent Done ---");
                println!("Final text: {final_text}");
            }
            Err(e) => {
                // An error occurred
                eprintln!("\nError: {e}");
                break;
            }
        }
    }

    println!("\n=== Example Complete ===");
}
