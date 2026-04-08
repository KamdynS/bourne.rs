//! Bash command execution tool.
//!
//! This module provides `BashTool`, which allows the LLM to execute shell commands.
//! It's one of the most powerful tools an agent can have - it enables file manipulation,
//! process management, network operations, and virtually anything a shell can do.
//!
//! # Security Considerations
//!
//! **This tool executes arbitrary commands with the privileges of the running process.**
//! Use it only in sandboxed environments or when you trust the LLM's judgment.
//!
//! The tool provides some basic protections:
//! - Configurable timeout to prevent runaway processes
//! - Working directory isolation (commands run in a specified directory)
//! - Output truncation to prevent memory exhaustion
//!
//! However, it does NOT provide:
//! - Command whitelisting/blacklisting
//! - Resource limits (CPU, memory, disk)
//! - Network isolation
//!
//! For production use, consider wrapping this in a container or VM.
//!
//! # Example
//!
//! ```ignore
//! let bash = BashTool::new()
//!     .with_working_dir("/tmp/sandbox")
//!     .with_timeout(Duration::from_secs(30));
//!
//! // The LLM can now execute commands like:
//! // {"command": "ls -la", "description": "List files"}
//! ```

use std::path::PathBuf;
use std::time::Duration;

use agent_core::{Tool, ToolOutput};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::process::Command;
use tokio::time::timeout;

/// Maximum output size before truncation (128 KB).
///
/// This prevents memory exhaustion if a command produces massive output.
/// The LLM can always request more specific output with head/tail/grep.
const MAX_OUTPUT_SIZE: usize = 128 * 1024;

/// Tool that executes bash commands.
///
/// # Why a Bash Tool?
///
/// The bash tool is the "escape hatch" that lets an LLM do anything the
/// system allows. It's incredibly powerful but requires trust:
///
/// - **Flexibility**: Instead of implementing tools for every operation,
///   the LLM can compose existing Unix tools.
/// - **Discoverability**: The LLM already knows how to use standard Unix
///   commands, reducing the need for custom tool documentation.
/// - **Composability**: Pipes, redirects, and command chaining work naturally.
///
/// # Design Choices
///
/// **Why not a generic "execute" tool?**
/// We use `bash -c` specifically because:
/// 1. Full bash features: pipes, redirects, subshells, etc.
/// 2. The LLM is trained on bash examples
/// 3. Available on virtually all Unix-like systems (via PATH)
///
/// **Why require a description?**
/// The `description` field in the input schema forces the LLM to explain
/// what it's doing before doing it. This:
/// 1. Helps users understand agent behavior
/// 2. Appears in tool use events for logging
/// 3. Encourages the LLM to think before acting
///
/// **Why truncate output?**
/// LLMs have context limits. A command like `cat /var/log/syslog` could
/// produce megabytes of output, consuming the entire context window.
/// Truncation keeps things manageable; the LLM can use head/tail/grep
/// to get specific portions.
pub struct BashTool {
    /// Working directory for command execution.
    /// Commands run with this as their cwd.
    working_dir: Option<PathBuf>,

    /// Maximum time a command can run before being killed.
    timeout: Duration,
}

impl Default for BashTool {
    fn default() -> Self {
        Self::new()
    }
}

impl BashTool {
    /// Create a new bash tool with default settings.
    ///
    /// Default timeout is 120 seconds. No working directory is set,
    /// so commands inherit the process's cwd.
    pub fn new() -> Self {
        Self {
            working_dir: None,
            timeout: Duration::from_secs(120),
        }
    }

    /// Set the working directory for command execution.
    ///
    /// All commands will run with this directory as their cwd.
    /// This provides some isolation but is NOT a security boundary -
    /// commands can still access files outside this directory.
    pub fn with_working_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set the maximum execution time for commands.
    ///
    /// Commands that exceed this duration are killed with SIGKILL.
    /// The tool returns an error indicating the timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[async_trait]
impl Tool for BashTool {
    /// Returns "bash" - a short, recognizable name.
    ///
    /// The name appears in tool call events and logs. Keep it short
    /// but distinctive enough that users know what tool is running.
    fn name(&self) -> &str {
        "bash"
    }

    /// Describes the tool's purpose for the LLM.
    ///
    /// This description is sent to the LLM as part of the tool definition.
    /// It should be clear enough that the LLM knows when to use this tool
    /// vs. other options. We emphasize:
    /// - What it does (run shell commands)
    /// - What it returns (stdout, stderr, exit code)
    /// - How to handle failures (non-zero exit codes)
    fn description(&self) -> &str {
        "Execute a bash command and return its output. \
         Returns stdout, stderr, and exit code. \
         Use for file operations, running programs, and system tasks. \
         Non-zero exit codes indicate command failure - check stderr for details."
    }

    /// Defines the expected input format.
    ///
    /// We require two fields:
    /// - `command`: The bash command to execute (required)
    /// - `description`: A human-readable explanation of what the command does (required)
    ///
    /// The description requirement is intentional - it forces the LLM to
    /// articulate its intent, which helps with debugging and auditing.
    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute"
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what this command does and why"
                }
            },
            "required": ["command", "description"]
        })
    }

    /// Execute the bash command.
    ///
    /// # Execution Flow
    ///
    /// 1. Parse and validate input
    /// 2. Spawn bash process with the command
    /// 3. Wait for completion (with timeout)
    /// 4. Collect and format output
    ///
    /// # Error Handling
    ///
    /// We return `ToolOutput::error()` for:
    /// - Missing required fields
    /// - Process spawn failures
    /// - Timeout exceeded
    ///
    /// We return `ToolOutput::success()` with output for:
    /// - Command completed (even with non-zero exit code)
    ///
    /// Non-zero exit codes are NOT errors from the tool's perspective.
    /// The tool successfully ran the command; the command just failed.
    /// This lets the LLM see the error output and decide how to proceed.
    async fn execute(&self, input: Value) -> ToolOutput {
        // Parse input - both fields are required
        let command = match input.get("command").and_then(|v| v.as_str()) {
            Some(cmd) => cmd,
            None => return ToolOutput::error("Missing required field: command"),
        };

        // Description is required but we don't use it in execution -
        // it's for logging/auditing purposes
        if input.get("description").and_then(|v| v.as_str()).is_none() {
            return ToolOutput::error("Missing required field: description");
        }

        // Build the command
        //
        // We use `bash -c` and let PATH find bash. This works on standard
        // Linux (/bin/bash), macOS (/bin/bash), and NixOS (in PATH).
        // The command string is passed as a single argument to avoid
        // shell escaping issues.
        let mut cmd = Command::new("bash");
        cmd.arg("-c").arg(command);

        // Set working directory if configured
        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        // Spawn and wait with timeout
        //
        // We capture both stdout and stderr. The timeout wrapper ensures
        // we don't hang indefinitely on commands that never complete.
        let result = timeout(self.timeout, cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                // Command completed - format the output
                //
                // We include exit code even for success (code 0) because
                // it's useful context. For failures, the LLM can check
                // the code and stderr to understand what went wrong.
                let stdout = truncate_output(&output.stdout);
                let stderr = truncate_output(&output.stderr);
                let code = output.status.code().unwrap_or(-1);

                // Format as a clear, parseable structure
                //
                // Using triple-quoted sections makes it easy for the LLM
                // to parse even if the output contains special characters.
                let formatted = format!(
                    "Exit code: {code}\n\
                     \n\
                     === STDOUT ===\n\
                     {stdout}\n\
                     \n\
                     === STDERR ===\n\
                     {stderr}"
                );

                ToolOutput::success(formatted)
            }
            Ok(Err(e)) => {
                // Process spawn failed - permission denied, command not found, etc.
                ToolOutput::error(format!("Failed to execute command: {e}"))
            }
            Err(_) => {
                // Timeout - the command ran too long
                ToolOutput::error(format!(
                    "Command timed out after {} seconds",
                    self.timeout.as_secs()
                ))
            }
        }
    }
}

/// Truncate output to prevent memory exhaustion.
///
/// Large outputs consume context window space and can cause OOM.
/// We truncate with a clear marker so the LLM knows data was lost
/// and can request specific portions with head/tail/grep.
fn truncate_output(bytes: &[u8]) -> String {
    // Convert to string, replacing invalid UTF-8 with replacement char
    let s = String::from_utf8_lossy(bytes);

    if s.len() <= MAX_OUTPUT_SIZE {
        s.into_owned()
    } else {
        // Truncate and add marker
        //
        // We truncate at the end because usually the beginning of output
        // (headers, initial results) is most informative. The LLM can
        // use `tail` if it needs the end.
        let truncated = &s[..MAX_OUTPUT_SIZE];
        format!(
            "{truncated}\n\n[OUTPUT TRUNCATED - {} bytes total, showing first {}]",
            s.len(),
            MAX_OUTPUT_SIZE
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_command() {
        let tool = BashTool::new();

        let output = tool
            .execute(json!({
                "command": "echo hello",
                "description": "Print hello"
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("hello"));
        assert!(output.content.contains("Exit code: 0"));
    }

    #[tokio::test]
    async fn test_command_with_stderr() {
        let tool = BashTool::new();

        let output = tool
            .execute(json!({
                "command": "echo error >&2",
                "description": "Print to stderr"
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("error"));
        assert!(output.content.contains("STDERR"));
    }

    #[tokio::test]
    async fn test_failing_command() {
        let tool = BashTool::new();

        let output = tool
            .execute(json!({
                "command": "exit 42",
                "description": "Exit with code 42"
            }))
            .await;

        // Failing commands are still successful tool executions
        assert!(!output.is_error);
        assert!(output.content.contains("Exit code: 42"));
    }

    #[tokio::test]
    async fn test_missing_command() {
        let tool = BashTool::new();

        let output = tool
            .execute(json!({
                "description": "No command provided"
            }))
            .await;

        assert!(output.is_error);
        assert!(output.content.contains("Missing required field: command"));
    }

    #[tokio::test]
    async fn test_missing_description() {
        let tool = BashTool::new();

        let output = tool
            .execute(json!({
                "command": "echo hello"
            }))
            .await;

        assert!(output.is_error);
        assert!(output.content.contains("Missing required field: description"));
    }

    #[tokio::test]
    async fn test_timeout() {
        let tool = BashTool::new().with_timeout(Duration::from_millis(100));

        let output = tool
            .execute(json!({
                "command": "sleep 10",
                "description": "Sleep for 10 seconds"
            }))
            .await;

        assert!(output.is_error);
        assert!(output.content.contains("timed out"));
    }

    #[tokio::test]
    async fn test_working_directory() {
        let tool = BashTool::new().with_working_dir("/tmp");

        let output = tool
            .execute(json!({
                "command": "pwd",
                "description": "Print working directory"
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("/tmp"));
    }

    #[test]
    fn test_truncate_output() {
        // Short output - no truncation
        let short = b"hello world";
        let result = truncate_output(short);
        assert_eq!(result, "hello world");

        // Long output - truncated
        let long: Vec<u8> = vec![b'x'; MAX_OUTPUT_SIZE + 1000];
        let result = truncate_output(&long);
        assert!(result.contains("[OUTPUT TRUNCATED"));
        assert!(result.len() < long.len());
    }

    #[test]
    fn test_tool_metadata() {
        let tool = BashTool::new();
        assert_eq!(tool.name(), "bash");
        assert!(tool.description().contains("bash"));

        let schema = tool.input_schema();
        assert!(schema.get("properties").is_some());
        assert!(schema["required"].as_array().unwrap().contains(&json!("command")));
    }
}
