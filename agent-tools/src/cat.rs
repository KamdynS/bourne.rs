//! Cat tool - read file contents.
//!
//! The simplest file reading tool. Like `cat file.txt`.
//!
//! # Why Cat When We Have FileTool?
//!
//! FileTool is a multi-operation tool (read, write, list, etc.).
//! CatTool does one thing: read files. This follows the Unix philosophy
//! of small, composable tools.
//!
//! The LLM can choose based on intent:
//! - "Read this file" → CatTool
//! - "Write to this file" → FileTool or BashTool
//!
//! # Example
//!
//! ```ignore
//! // LLM calls:
//! // {"path": "/etc/hostname"}
//! // Returns: "my-computer\n"
//! ```

use std::path::Path;

use agent_core::{Tool, ToolOutput};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::fs;

/// Maximum file size to read (1 MB).
const MAX_SIZE: u64 = 1024 * 1024;

/// Read file contents.
///
/// Simple, focused tool for reading files. For writing or
/// directory operations, use FileTool or BashTool.
pub struct CatTool;

impl Default for CatTool {
    fn default() -> Self {
        Self::new()
    }
}

impl CatTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for CatTool {
    fn name(&self) -> &str {
        "cat"
    }

    fn description(&self) -> &str {
        "Read and return the contents of a file. Like Unix cat."
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
        let path_str = match input.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolOutput::error("Missing required field: path"),
        };

        let path = Path::new(path_str);

        // Check if file exists and get size
        let metadata = match fs::metadata(path).await {
            Ok(m) => m,
            Err(e) => return ToolOutput::error(format!("Cannot access {}: {e}", path.display())),
        };

        if metadata.is_dir() {
            return ToolOutput::error(format!("{} is a directory", path.display()));
        }

        let size = metadata.len();
        if size > MAX_SIZE {
            return ToolOutput::error(format!(
                "{} is too large ({} bytes, max {}). Use head for large files.",
                path.display(),
                size,
                MAX_SIZE
            ));
        }

        // Read the file
        match fs::read_to_string(path).await {
            Ok(content) => ToolOutput::success(content),
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                // Binary file
                ToolOutput::error(format!("{} appears to be a binary file", path.display()))
            }
            Err(e) => ToolOutput::error(format!("Cannot read {}: {e}", path.display())),
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
    async fn test_cat_file() {
        let dir = temp_dir();
        let file = dir.path().join("test.txt");
        std_fs::write(&file, "hello world").unwrap();

        let tool = CatTool::new();
        let output = tool
            .execute(json!({"path": file.to_str().unwrap()}))
            .await;

        assert!(!output.is_error);
        assert_eq!(output.content, "hello world");
    }

    #[tokio::test]
    async fn test_cat_nonexistent() {
        let tool = CatTool::new();
        let output = tool.execute(json!({"path": "/nonexistent/file"})).await;

        assert!(output.is_error);
        assert!(output.content.contains("Cannot access"));
    }

    #[tokio::test]
    async fn test_cat_directory() {
        let dir = temp_dir();

        let tool = CatTool::new();
        let output = tool
            .execute(json!({"path": dir.path().to_str().unwrap()}))
            .await;

        assert!(output.is_error);
        assert!(output.content.contains("directory"));
    }

    #[test]
    fn test_tool_metadata() {
        let tool = CatTool::new();
        assert_eq!(tool.name(), "cat");
    }
}
