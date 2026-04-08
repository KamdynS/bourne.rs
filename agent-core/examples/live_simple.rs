//! Live Integration Test - Simple Agent
//!
//! This example tests the agent against a real LLM API.
//! It requires an API key to run.
//!
//! # What This Tests
//!
//! - Request serialization to Anthropic format
//! - Real SSE streaming and parsing
//! - Tool call round-trip with actual LLM
//! - Response handling and event emission
//!
//! # Setup
//!
//! Create a `.env` file in the project root:
//! ```text
//! ANTHROPIC_API_KEY=sk-ant-...
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example live_simple -p agent-core
//! ```
//!
//! # Cost
//!
//! This makes real API calls. Expect ~$0.01 per run with claude-sonnet-4-20250514.

use agent_core::{AgentBuilder, AgentEvent, AnthropicClient, Tool, ToolOutput};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

/// A simple calculator tool.
///
/// We use a real tool to verify the full round-trip:
/// 1. Tool definition sent to API
/// 2. LLM decides to call it
/// 3. We execute and return result
/// 4. LLM incorporates result
struct CalculatorTool;

#[async_trait]
impl Tool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Perform basic arithmetic. Supports add, subtract, multiply, divide operations."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
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
    println!("=== Live Integration Test: Simple Agent ===\n");

    // Load environment variables from .env file
    if let Err(e) = dotenvy::dotenv() {
        eprintln!("Note: No .env file found ({e}), checking environment directly");
    }

    // Get API key from environment
    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("ERROR: ANTHROPIC_API_KEY not set");
            eprintln!();
            eprintln!("To run this integration test:");
            eprintln!("  1. Create a .env file in the project root");
            eprintln!("  2. Add: ANTHROPIC_API_KEY=sk-ant-...");
            eprintln!("  3. Run again");
            eprintln!();
            eprintln!("Or set the environment variable directly:");
            eprintln!("  ANTHROPIC_API_KEY=sk-ant-... cargo run --example live_simple -p agent-core");
            std::process::exit(1);
        }
    };

    // Create a real Anthropic client
    let client = AnthropicClient::new(api_key, "claude-sonnet-4-20250514");

    // Build the agent with our calculator tool
    let agent = AgentBuilder::new(Box::new(client))
        .tools(vec![Box::new(CalculatorTool)])
        .system_prompt("You are a helpful assistant. Use the calculator tool when asked to do math.")
        .max_tokens(1024)
        .max_turns(5)
        .build();

    // Run a task that should trigger tool use
    let task = "What is 42 multiplied by 17? Use the calculator to be precise.";
    println!("Task: {task}\n");
    println!("--- Agent Events ---\n");

    let stream = agent.run(task);
    tokio::pin!(stream);

    let mut tool_names: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut tool_called = false;
    let mut got_response = false;

    while let Some(event) = stream.next().await {
        match event {
            Ok(AgentEvent::Text(text)) => {
                print!("{text}");
                got_response = true;
            }
            Ok(AgentEvent::ToolStart { id, name, input }) => {
                println!("\n[Tool: {name}]");
                println!("  Input: {input}");
                tool_names.insert(id, name);
                tool_called = true;
            }
            Ok(AgentEvent::ToolEnd { id, output }) => {
                let name = tool_names.get(&id).map(|s| s.as_str()).unwrap_or("unknown");
                println!("  Result: {}", output.content);
                if output.is_error {
                    println!("  (error)");
                }
                println!("[End {name}]\n");
            }
            Ok(AgentEvent::TurnComplete { turn, usage }) => {
                println!(
                    "\n[Turn {turn}: {} in, {} out tokens]\n",
                    usage.input_tokens, usage.output_tokens
                );
            }
            Ok(AgentEvent::Done { final_text }) => {
                println!("\n--- Done ---");
                if !final_text.is_empty() {
                    println!("Final: {final_text}");
                }
            }
            Err(e) => {
                eprintln!("\nERROR: {e}");
                std::process::exit(1);
            }
        }
    }

    // Verify the integration worked
    println!("\n=== Verification ===");

    if tool_called {
        println!("[PASS] Tool was called");
    } else {
        println!("[FAIL] Tool was NOT called - LLM should have used calculator");
    }

    if got_response {
        println!("[PASS] Got response from LLM");
    } else {
        println!("[FAIL] No response received");
    }

    // The answer should be 714
    println!("\nExpected answer: 714 (42 * 17)");
    println!("\n=== Integration Test Complete ===");
}
