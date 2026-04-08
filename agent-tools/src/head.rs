//! Head tool - read first N lines of a file.
//!
//! Like `head -n 20 file.txt`. Useful for:
//! - Previewing large files without reading everything
//! - Checking file format/headers
//! - Sampling log files
//!
//! # Why Head?
//!
//! LLMs have context limits. Reading a 10MB log file wastes tokens.
//! Head lets the model peek at files efficiently, then decide if it
//! needs more (with specific line ranges or grep).
//!
//! # Example
//!
//! ```ignore
//! // LLM calls:
//! // {"path": "/var/log/syslog", "lines": 5}
//! // Returns first 5 lines
//! ```

use std::path::Path;

use agent_core::{Tool, ToolOutput};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::fs::File;
use tokio::io::{AsyncBufReadExt, BufReader};

/// Default number of lines if not specified.
const DEFAULT_LINES: usize = 10;

/// Maximum lines to prevent abuse.
const MAX_LINES: usize = 1000;

/// Read first N lines of a file.
///
/// Efficient for large files - only reads what's needed.
pub struct HeadTool;

impl Default for HeadTool {
    fn default() -> Self {
        Self::new()
    }
}

impl HeadTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for HeadTool {
    fn name(&self) -> &str {
        "head"
    }

    fn description(&self) -> &str {
        "Read the first N lines of a file (default 10). \
         Efficient for previewing large files without loading everything."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of lines to read (default 10, max 1000)"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        let path_str = match input.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolOutput::error("Missing required field: path"),
        };

        let n_lines = input
            .get("lines")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(DEFAULT_LINES)
            .min(MAX_LINES);

        let path = Path::new(path_str);

        // Open file
        let file = match File::open(path).await {
            Ok(f) => f,
            Err(e) => return ToolOutput::error(format!("Cannot open {}: {e}", path.display())),
        };

        // Read lines
        let reader = BufReader::new(file);
        let mut lines_iter = reader.lines();
        let mut result = Vec::with_capacity(n_lines);

        for _ in 0..n_lines {
            match lines_iter.next_line().await {
                Ok(Some(line)) => result.push(line),
                Ok(None) => break, // EOF
                Err(e) => {
                    // Could be binary file or encoding issue
                    if result.is_empty() {
                        return ToolOutput::error(format!("Cannot read {}: {e}", path.display()));
                    }
                    // Return what we got
                    break;
                }
            }
        }

        let total_lines = result.len();
        let content = result.join("\n");

        // Add note if we might have truncated
        if total_lines == n_lines {
            ToolOutput::success(format!(
                "{}\n\n[Showing first {} lines]",
                content, total_lines
            ))
        } else {
            ToolOutput::success(content)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs as std_fs;
    use tempfile::TempDir;

    fn temp_dir() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    #[tokio::test]
    async fn test_head_default() {
        let dir = temp_dir();
        let file = dir.path().join("test.txt");
        let content: String = (1..=20).map(|i| format!("line {i}\n")).collect();
        std_fs::write(&file, &content).unwrap();

        let tool = HeadTool::new();
        let output = tool
            .execute(json!({"path": file.to_str().unwrap()}))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("line 1"));
        assert!(output.content.contains("line 10"));
        assert!(!output.content.contains("line 11"));
    }

    #[tokio::test]
    async fn test_head_custom_lines() {
        let dir = temp_dir();
        let file = dir.path().join("test.txt");
        let content: String = (1..=20).map(|i| format!("line {i}\n")).collect();
        std_fs::write(&file, &content).unwrap();

        let tool = HeadTool::new();
        let output = tool
            .execute(json!({"path": file.to_str().unwrap(), "lines": 3}))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("line 1"));
        assert!(output.content.contains("line 3"));
        assert!(!output.content.contains("line 4"));
    }

    #[tokio::test]
    async fn test_head_short_file() {
        let dir = temp_dir();
        let file = dir.path().join("test.txt");
        std_fs::write(&file, "just two\nlines").unwrap();

        let tool = HeadTool::new();
        let output = tool
            .execute(json!({"path": file.to_str().unwrap(), "lines": 100}))
            .await;

        assert!(!output.is_error);
        // Should NOT have the truncation note
        assert!(!output.content.contains("[Showing"));
    }

    #[tokio::test]
    async fn test_head_nonexistent() {
        let tool = HeadTool::new();
        let output = tool.execute(json!({"path": "/nonexistent/file"})).await;

        assert!(output.is_error);
    }

    #[test]
    fn test_tool_metadata() {
        let tool = HeadTool::new();
        assert_eq!(tool.name(), "head");
    }
}
