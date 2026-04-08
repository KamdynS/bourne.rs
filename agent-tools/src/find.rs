//! Find tool - locate files by name pattern.
//!
//! Like `find /path -name "*.rs"`. Essential for navigating
//! unfamiliar codebases.
//!
//! # Why Find?
//!
//! Before you can read or grep a file, you need to know it exists.
//! Find helps the LLM discover files by:
//! - Name patterns (*.rs, test_*.py)
//! - Directory structure exploration
//! - File type filtering
//!
//! # Example
//!
//! ```ignore
//! // LLM calls:
//! // {"path": "/project", "pattern": "*.rs", "type": "file"}
//! // Returns list of matching paths
//! ```

use std::path::Path;

use agent_core::{Tool, ToolOutput};
use async_trait::async_trait;
use serde_json::{json, Value};

/// Maximum results to prevent overwhelming output.
const MAX_RESULTS: usize = 100;

/// Maximum directory depth to search.
const MAX_DEPTH: usize = 20;

/// Find files by name pattern.
///
/// Supports glob patterns like `*.rs`, `test_*`, `**/*.json`.
/// Returns matching paths, one per line.
pub struct FindTool;

impl Default for FindTool {
    fn default() -> Self {
        Self::new()
    }
}

impl FindTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Find files by name pattern. Supports glob patterns like *.rs or test_*. \
         Returns matching file paths."
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
                    "description": "Glob pattern to match (e.g., *.rs, test_*, **/*.json)"
                },
                "file_type": {
                    "type": "string",
                    "enum": ["file", "dir", "any"],
                    "description": "Type of entries to find (default: file)"
                }
            },
            "required": ["path", "pattern"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        let path_str = match input.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolOutput::error("Missing required field: path"),
        };

        let pattern = match input.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolOutput::error("Missing required field: pattern"),
        };

        let file_type = input
            .get("file_type")
            .and_then(|v| v.as_str())
            .unwrap_or("file");

        let base_path = Path::new(path_str);

        if !base_path.exists() {
            return ToolOutput::error(format!("Path does not exist: {}", base_path.display()));
        }

        if !base_path.is_dir() {
            return ToolOutput::error(format!("Not a directory: {}", base_path.display()));
        }

        // Build glob pattern
        let full_pattern = if pattern.contains("**") {
            // Pattern already has recursive glob
            format!("{}/{}", path_str.trim_end_matches('/'), pattern)
        } else {
            // Add recursive search
            format!("{}/**/{}", path_str.trim_end_matches('/'), pattern)
        };

        // Execute glob
        let entries = match glob::glob(&full_pattern) {
            Ok(paths) => paths,
            Err(e) => return ToolOutput::error(format!("Invalid pattern: {e}")),
        };

        let mut results = Vec::new();
        let mut count = 0;
        let mut truncated = false;

        for entry in entries {
            if count >= MAX_RESULTS {
                truncated = true;
                break;
            }

            let path = match entry {
                Ok(p) => p,
                Err(_) => continue, // Skip permission errors
            };

            // Check depth
            let depth = path
                .strip_prefix(base_path)
                .map(|p| p.components().count())
                .unwrap_or(0);

            if depth > MAX_DEPTH {
                continue;
            }

            // Filter by type
            let matches_type = match file_type {
                "file" => path.is_file(),
                "dir" => path.is_dir(),
                "any" => true,
                _ => path.is_file(),
            };

            if matches_type {
                results.push(path.display().to_string());
                count += 1;
            }
        }

        if results.is_empty() {
            return ToolOutput::success(format!("No matches found for '{}' in {}", pattern, path_str));
        }

        // Sort for consistent output
        results.sort();

        let mut output = results.join("\n");
        if truncated {
            output.push_str(&format!("\n\n[Showing first {} results, more exist]", MAX_RESULTS));
        }

        ToolOutput::success(output)
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
    async fn test_find_files() {
        let dir = temp_dir();
        std_fs::write(dir.path().join("foo.rs"), "").unwrap();
        std_fs::write(dir.path().join("bar.rs"), "").unwrap();
        std_fs::write(dir.path().join("baz.txt"), "").unwrap();

        let tool = FindTool::new();
        let output = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap(),
                "pattern": "*.rs"
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("foo.rs"));
        assert!(output.content.contains("bar.rs"));
        assert!(!output.content.contains("baz.txt"));
    }

    #[tokio::test]
    async fn test_find_nested() {
        let dir = temp_dir();
        let sub = dir.path().join("src");
        std_fs::create_dir(&sub).unwrap();
        std_fs::write(sub.join("main.rs"), "").unwrap();
        std_fs::write(dir.path().join("lib.rs"), "").unwrap();

        let tool = FindTool::new();
        let output = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap(),
                "pattern": "*.rs"
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("main.rs"));
        assert!(output.content.contains("lib.rs"));
    }

    #[tokio::test]
    async fn test_find_dirs() {
        let dir = temp_dir();
        std_fs::create_dir(dir.path().join("src")).unwrap();
        std_fs::create_dir(dir.path().join("tests")).unwrap();
        std_fs::write(dir.path().join("file.txt"), "").unwrap();

        let tool = FindTool::new();
        let output = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap(),
                "pattern": "*",
                "file_type": "dir"
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("src"));
        assert!(output.content.contains("tests"));
    }

    #[tokio::test]
    async fn test_find_no_matches() {
        let dir = temp_dir();
        std_fs::write(dir.path().join("file.txt"), "").unwrap();

        let tool = FindTool::new();
        let output = tool
            .execute(json!({
                "path": dir.path().to_str().unwrap(),
                "pattern": "*.rs"
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("No matches"));
    }

    #[tokio::test]
    async fn test_find_nonexistent_path() {
        let tool = FindTool::new();
        let output = tool
            .execute(json!({
                "path": "/nonexistent/path",
                "pattern": "*"
            }))
            .await;

        assert!(output.is_error);
    }

    #[test]
    fn test_tool_metadata() {
        let tool = FindTool::new();
        assert_eq!(tool.name(), "find");
    }
}
