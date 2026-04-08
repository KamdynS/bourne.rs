//! Ripgrep tool - search file contents with regex.
//!
//! Like `rg "pattern" /path`. The go-to tool for finding code.
//!
//! # Why Ripgrep Over Grep?
//!
//! Ripgrep is just better:
//! - Faster (parallelized, respects .gitignore)
//! - Better defaults (recursive, smart case)
//! - Cleaner output
//!
//! We implement a ripgrep-inspired interface, not the actual rg binary.
//! This gives us control and avoids shelling out.
//!
//! # Example
//!
//! ```ignore
//! // LLM calls:
//! // {"pattern": "fn main", "path": "./src", "file_pattern": "*.rs"}
//! // Returns matches with file:line:content format
//! ```

use std::path::Path;

use agent_core::{Tool, ToolOutput};
use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};
use tokio::fs;

/// Maximum files to search.
const MAX_FILES: usize = 1000;

/// Maximum matches to return.
const MAX_MATCHES: usize = 100;

/// Maximum line length to include in output.
const MAX_LINE_LEN: usize = 500;

/// Search file contents with regex.
///
/// Returns matches in `file:line:content` format.
/// Supports file filtering with glob patterns.
pub struct RipgrepTool;

impl Default for RipgrepTool {
    fn default() -> Self {
        Self::new()
    }
}

impl RipgrepTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for RipgrepTool {
    fn name(&self) -> &str {
        "rg"
    }

    fn description(&self) -> &str {
        "Search file contents using regex patterns. Like ripgrep. \
         Returns matches in file:line:content format. \
         Use file_pattern to filter which files to search."
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
                    "description": "File or directory to search in"
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., *.rs). Default: all files"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Ignore case when matching (default: false)"
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context before/after match (default: 0)"
                }
            },
            "required": ["pattern", "path"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        let pattern_str = match input.get("pattern").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolOutput::error("Missing required field: pattern"),
        };

        let path_str = match input.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolOutput::error("Missing required field: path"),
        };

        let file_pattern = input.get("file_pattern").and_then(|v| v.as_str());
        let case_insensitive = input
            .get("case_insensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let context_lines = input
            .get("context_lines")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Compile regex
        let pattern = if case_insensitive {
            format!("(?i){}", pattern_str)
        } else {
            pattern_str.to_string()
        };

        let regex = match Regex::new(&pattern) {
            Ok(r) => r,
            Err(e) => return ToolOutput::error(format!("Invalid regex: {e}")),
        };

        let base_path = Path::new(path_str);

        // Collect files to search
        let files = match collect_files(base_path, file_pattern).await {
            Ok(f) => f,
            Err(e) => return ToolOutput::error(e),
        };

        if files.is_empty() {
            return ToolOutput::success("No files to search");
        }

        // Search each file
        let mut matches = Vec::new();
        let mut files_searched = 0;
        let mut truncated = false;

        for file_path in files {
            if matches.len() >= MAX_MATCHES {
                truncated = true;
                break;
            }

            if let Ok(content) = fs::read_to_string(&file_path).await {
                let lines: Vec<&str> = content.lines().collect();

                for (line_num, line) in lines.iter().enumerate() {
                    if regex.is_match(line) {
                        // Format match with context
                        let match_output = format_match(
                            &file_path.display().to_string(),
                            line_num + 1,
                            &lines,
                            context_lines,
                        );
                        matches.push(match_output);

                        if matches.len() >= MAX_MATCHES {
                            truncated = true;
                            break;
                        }
                    }
                }
            }

            files_searched += 1;
        }

        if matches.is_empty() {
            return ToolOutput::success(format!(
                "No matches for '{}' in {} files",
                pattern_str, files_searched
            ));
        }

        let mut output = matches.join("\n\n");
        if truncated {
            output.push_str(&format!(
                "\n\n[Showing first {} matches, more exist]",
                MAX_MATCHES
            ));
        }

        ToolOutput::success(output)
    }
}

/// Collect files to search based on path and optional glob pattern.
async fn collect_files(
    base_path: &Path,
    file_pattern: Option<&str>,
) -> Result<Vec<std::path::PathBuf>, String> {
    if !base_path.exists() {
        return Err(format!("Path does not exist: {}", base_path.display()));
    }

    // Single file
    if base_path.is_file() {
        return Ok(vec![base_path.to_path_buf()]);
    }

    // Directory - use glob to find files
    let pattern = match file_pattern {
        Some(p) => format!("{}/**/{}", base_path.display(), p),
        None => format!("{}/**/*", base_path.display()),
    };

    let entries = glob::glob(&pattern).map_err(|e| format!("Invalid pattern: {e}"))?;

    let mut files = Vec::new();
    for entry in entries.take(MAX_FILES) {
        if let Ok(path) = entry {
            if path.is_file() {
                files.push(path);
            }
        }
    }

    Ok(files)
}

/// Format a match with optional context lines.
fn format_match(file: &str, line_num: usize, lines: &[&str], context: usize) -> String {
    if context == 0 {
        // Simple format
        let line = truncate_line(lines[line_num - 1]);
        return format!("{}:{}:{}", file, line_num, line);
    }

    // With context
    let mut parts = Vec::new();
    parts.push(format!("{}:", file));

    let start = line_num.saturating_sub(context + 1);
    let end = (line_num + context).min(lines.len());

    for i in start..end {
        let prefix = if i + 1 == line_num { ">" } else { " " };
        let line = truncate_line(lines[i]);
        parts.push(format!("{}{:4}:{}", prefix, i + 1, line));
    }

    parts.join("\n")
}

/// Truncate very long lines.
fn truncate_line(line: &str) -> String {
    if line.len() <= MAX_LINE_LEN {
        line.to_string()
    } else {
        format!("{}...", &line[..MAX_LINE_LEN])
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
    async fn test_rg_simple() {
        let dir = temp_dir();
        let file = dir.path().join("test.rs");
        std_fs::write(&file, "fn main() {\n    println!(\"hello\");\n}").unwrap();

        let tool = RipgrepTool::new();
        let output = tool
            .execute(json!({
                "pattern": "fn main",
                "path": dir.path().to_str().unwrap()
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("fn main"));
        assert!(output.content.contains(":1:"));
    }

    #[tokio::test]
    async fn test_rg_file_pattern() {
        let dir = temp_dir();
        std_fs::write(dir.path().join("code.rs"), "fn main() {}").unwrap();
        std_fs::write(dir.path().join("readme.md"), "fn main in docs").unwrap();

        let tool = RipgrepTool::new();
        let output = tool
            .execute(json!({
                "pattern": "fn main",
                "path": dir.path().to_str().unwrap(),
                "file_pattern": "*.rs"
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("code.rs"));
        assert!(!output.content.contains("readme.md"));
    }

    #[tokio::test]
    async fn test_rg_case_insensitive() {
        let dir = temp_dir();
        let file = dir.path().join("test.txt");
        std_fs::write(&file, "Hello World\nhello world\nHELLO WORLD").unwrap();

        let tool = RipgrepTool::new();
        let output = tool
            .execute(json!({
                "pattern": "hello",
                "path": file.to_str().unwrap(),
                "case_insensitive": true
            }))
            .await;

        assert!(!output.is_error);
        // Should match all 3 lines
        assert!(output.content.contains(":1:"));
        assert!(output.content.contains(":2:"));
        assert!(output.content.contains(":3:"));
    }

    #[tokio::test]
    async fn test_rg_no_matches() {
        let dir = temp_dir();
        let file = dir.path().join("test.txt");
        std_fs::write(&file, "nothing here").unwrap();

        let tool = RipgrepTool::new();
        let output = tool
            .execute(json!({
                "pattern": "xyz123",
                "path": file.to_str().unwrap()
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("No matches"));
    }

    #[tokio::test]
    async fn test_rg_invalid_regex() {
        let tool = RipgrepTool::new();
        let output = tool
            .execute(json!({
                "pattern": "[invalid",
                "path": "/tmp"
            }))
            .await;

        assert!(output.is_error);
        assert!(output.content.contains("Invalid regex"));
    }

    #[test]
    fn test_tool_metadata() {
        let tool = RipgrepTool::new();
        assert_eq!(tool.name(), "rg");
    }
}
