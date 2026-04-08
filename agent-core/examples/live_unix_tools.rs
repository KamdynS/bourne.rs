//! Live Integration Test - Unix Tools
//!
//! This example tests the Unix-style tools (find, rg, cat, head) with a real LLM.
//! It creates test files and asks the LLM to explore them.
//!
//! # What This Tests
//!
//! - Tool composition: LLM using multiple tools together
//! - Find: Locating files by pattern
//! - Ripgrep: Searching file contents
//! - Head/Cat: Reading file contents
//! - Real LLM reasoning about which tools to use
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
//! cargo run --example live_unix_tools -p agent-core
//! ```

use agent_core::{AgentBuilder, AgentEvent, AnthropicClient, Tool, ToolOutput};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// We need to import the tools from agent-tools, but since this is an example
// in agent-core, we'll define minimal versions here. In a real app, you'd
// use agent_tools::{FindTool, RipgrepTool, CatTool, HeadTool}.

/// Find files by pattern.
struct FindTool;

#[async_trait]
impl Tool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Find files by name pattern. Supports glob patterns like *.rs or test_*."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to search in"
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g., *.rs, *.txt)"
                }
            },
            "required": ["path", "pattern"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
        let pattern = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("*");

        let full_pattern = format!("{}/**/{}", path.trim_end_matches('/'), pattern);

        match glob::glob(&full_pattern) {
            Ok(entries) => {
                let files: Vec<String> = entries
                    .filter_map(|e| e.ok())
                    .filter(|p| p.is_file())
                    .take(50)
                    .map(|p| p.display().to_string())
                    .collect();

                if files.is_empty() {
                    ToolOutput::success(format!("No files matching '{}' found in {}", pattern, path))
                } else {
                    ToolOutput::success(files.join("\n"))
                }
            }
            Err(e) => ToolOutput::error(format!("Invalid pattern: {}", e)),
        }
    }
}

/// Search file contents with regex.
struct RipgrepTool;

#[async_trait]
impl Tool for RipgrepTool {
    fn name(&self) -> &str {
        "rg"
    }

    fn description(&self) -> &str {
        "Search file contents using regex. Returns matches in file:line:content format."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search"
                }
            },
            "required": ["pattern", "path"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        let pattern_str = input.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
        let path_str = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let regex = match regex::Regex::new(pattern_str) {
            Ok(r) => r,
            Err(e) => return ToolOutput::error(format!("Invalid regex: {}", e)),
        };

        let path = Path::new(path_str);
        let mut matches = Vec::new();

        if path.is_file() {
            if let Ok(content) = fs::read_to_string(path) {
                for (i, line) in content.lines().enumerate() {
                    if regex.is_match(line) {
                        matches.push(format!("{}:{}:{}", path.display(), i + 1, line));
                    }
                }
            }
        } else if path.is_dir() {
            let pattern = format!("{}/**/*", path_str.trim_end_matches('/'));
            if let Ok(entries) = glob::glob(&pattern) {
                for entry in entries.filter_map(|e| e.ok()).filter(|p| p.is_file()).take(100) {
                    if let Ok(content) = fs::read_to_string(&entry) {
                        for (i, line) in content.lines().enumerate() {
                            if regex.is_match(line) {
                                matches.push(format!("{}:{}:{}", entry.display(), i + 1, line));
                                if matches.len() >= 50 {
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        if matches.is_empty() {
            ToolOutput::success(format!("No matches for '{}' in {}", pattern_str, path_str))
        } else {
            ToolOutput::success(matches.join("\n"))
        }
    }
}

/// Read file contents.
struct CatTool;

#[async_trait]
impl Tool for CatTool {
    fn name(&self) -> &str {
        "cat"
    }

    fn description(&self) -> &str {
        "Read and return the entire contents of a file."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");

        match fs::read_to_string(path) {
            Ok(content) => ToolOutput::success(content),
            Err(e) => ToolOutput::error(format!("Cannot read {}: {}", path, e)),
        }
    }
}

/// Read first N lines.
struct HeadTool;

#[async_trait]
impl Tool for HeadTool {
    fn name(&self) -> &str {
        "head"
    }

    fn description(&self) -> &str {
        "Read the first N lines of a file (default 10)."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file"
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of lines (default 10)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        let path = input.get("path").and_then(|v| v.as_str()).unwrap_or("");
        let n = input.get("lines").and_then(|v| v.as_u64()).unwrap_or(10) as usize;

        match fs::read_to_string(path) {
            Ok(content) => {
                let lines: Vec<&str> = content.lines().take(n).collect();
                ToolOutput::success(lines.join("\n"))
            }
            Err(e) => ToolOutput::error(format!("Cannot read {}: {}", path, e)),
        }
    }
}

/// Create test files for the LLM to explore.
fn setup_test_files(base: &Path) {
    // Create directory structure
    let src = base.join("src");
    let tests = base.join("tests");
    fs::create_dir_all(&src).unwrap();
    fs::create_dir_all(&tests).unwrap();

    // Main source file
    fs::write(
        src.join("main.rs"),
        r#"//! Main entry point for the application.

fn main() {
    println!("Hello, world!");
    let result = calculate(40, 2);
    println!("Result: {}", result);
}

fn calculate(a: i32, b: i32) -> i32 {
    a + b
}
"#,
    )
    .unwrap();

    // Library file
    fs::write(
        src.join("lib.rs"),
        r#"//! Core library functions.

/// Add two numbers.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Multiply two numbers.
pub fn multiply(a: i32, b: i32) -> i32 {
    a * b
}

/// The answer to everything.
pub const ANSWER: i32 = 42;
"#,
    )
    .unwrap();

    // Test file
    fs::write(
        tests.join("test_lib.rs"),
        r#"//! Tests for the library.

#[test]
fn test_add() {
    assert_eq!(add(2, 2), 4);
}

#[test]
fn test_multiply() {
    assert_eq!(multiply(6, 7), 42);
}

#[test]
fn test_answer() {
    assert_eq!(ANSWER, 42);
}
"#,
    )
    .unwrap();

    // Config file
    fs::write(
        base.join("config.toml"),
        r#"[package]
name = "example"
version = "0.1.0"

[dependencies]
# No dependencies yet
"#,
    )
    .unwrap();

    // README
    fs::write(
        base.join("README.md"),
        r#"# Example Project

This is a test project for the Unix tools integration test.

## Features

- Addition
- Multiplication
- The answer to life, the universe, and everything (42)

## Usage

```
cargo run
```
"#,
    )
    .unwrap();
}

#[tokio::main]
async fn main() {
    println!("=== Live Integration Test: Unix Tools ===\n");

    // Load .env
    if let Err(e) = dotenvy::dotenv() {
        eprintln!("Note: No .env file found ({e})");
    }

    // Get API key
    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("ERROR: ANTHROPIC_API_KEY not set");
            eprintln!("Create a .env file with: ANTHROPIC_API_KEY=sk-ant-...");
            std::process::exit(1);
        }
    };

    // Create temp directory with test files
    let temp_dir = std::env::temp_dir().join("agent_unix_test");
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir).unwrap();
    }
    fs::create_dir_all(&temp_dir).unwrap();
    setup_test_files(&temp_dir);

    println!("Created test files in: {}\n", temp_dir.display());

    // Create client and agent
    let client = AnthropicClient::new(api_key, "claude-sonnet-4-20250514");

    let agent = AgentBuilder::new(Box::new(client))
        .tools(vec![
            Box::new(FindTool),
            Box::new(RipgrepTool),
            Box::new(CatTool),
            Box::new(HeadTool),
        ])
        .system_prompt(
            "You are a helpful assistant exploring a codebase. \
             Use the available tools to find and examine files. \
             Be concise in your responses.",
        )
        .max_tokens(1024)
        .max_turns(10)
        .build();

    // Task that requires multiple tools
    let task = format!(
        "Explore the project at {} and answer: \
         1. What Rust files exist? \
         2. What is the value of ANSWER constant? \
         3. What does the main function do?",
        temp_dir.display()
    );

    println!("Task: {}\n", task);
    println!("--- Agent Events ---\n");

    let stream = agent.run(&task);
    tokio::pin!(stream);

    let mut tool_names: HashMap<String, String> = HashMap::new();
    let mut tools_used: Vec<String> = Vec::new();

    while let Some(event) = stream.next().await {
        match event {
            Ok(AgentEvent::Text(text)) => {
                print!("{text}");
            }
            Ok(AgentEvent::ToolStart { id, name, input }) => {
                println!("\n[{name}] {input}");
                tool_names.insert(id, name.clone());
                if !tools_used.contains(&name) {
                    tools_used.push(name);
                }
            }
            Ok(AgentEvent::ToolEnd { id, output }) => {
                let name = tool_names.get(&id).map(|s| s.as_str()).unwrap_or("?");
                let preview: String = output.content.chars().take(100).collect();
                let suffix = if output.content.len() > 100 { "..." } else { "" };
                println!("[/{name}] {preview}{suffix}\n");
            }
            Ok(AgentEvent::TurnComplete { turn, usage }) => {
                println!("[Turn {turn}: {} in, {} out]\n", usage.input_tokens, usage.output_tokens);
            }
            Ok(AgentEvent::Done { .. }) => {
                println!("\n--- Done ---");
            }
            Err(e) => {
                eprintln!("\nERROR: {e}");
                break;
            }
        }
    }

    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();

    // Verification
    println!("\n=== Verification ===");
    println!("Tools used: {:?}", tools_used);

    if tools_used.contains(&"find".to_string()) {
        println!("[PASS] Used find tool");
    } else {
        println!("[FAIL] Did not use find tool");
    }

    if tools_used.contains(&"rg".to_string()) || tools_used.contains(&"cat".to_string()) {
        println!("[PASS] Used rg or cat to examine files");
    } else {
        println!("[FAIL] Did not examine file contents");
    }

    println!("\n=== Integration Test Complete ===");
}
