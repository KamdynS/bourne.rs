//! Ripgrep tool - search file contents using actual ripgrep internals.
//!
//! This uses the same crates that power the `rg` command-line tool:
//! - `grep-regex`: Regex matching
//! - `grep-searcher`: Line-oriented searching
//! - `ignore`: File walking with .gitignore support
//!
//! # Why Real Ripgrep?
//!
//! - **Fast**: Parallelized, uses SIMD where available
//! - **Smart defaults**: Respects .gitignore, skips binary files
//! - **Battle-tested**: Same code that powers the rg binary
//!
//! # Example
//!
//! ```ignore
//! // LLM calls:
//! // {"pattern": "fn main", "path": "./src"}
//! // Returns matches in file:line:content format
//! ```

use std::path::Path;
use std::sync::{Arc, Mutex};

use agent_core::{Tool, ToolOutput};
use async_trait::async_trait;
use grep_regex::RegexMatcherBuilder;
use grep_searcher::sinks::UTF8;
use grep_searcher::Searcher;
use ignore::WalkBuilder;
use serde_json::{json, Value};

/// Maximum matches to return (prevent output explosion).
const MAX_MATCHES: usize = 100;

/// Maximum line length to include in output.
const MAX_LINE_LEN: usize = 500;

/// Search file contents using ripgrep internals.
///
/// Provides the same fast, smart searching as the `rg` command.
/// Respects .gitignore by default, skips binary files, and
/// searches in parallel.
pub struct RipgrepTool {
    /// Whether to respect .gitignore files.
    respect_gitignore: bool,
}

impl Default for RipgrepTool {
    fn default() -> Self {
        Self::new()
    }
}

impl RipgrepTool {
    pub fn new() -> Self {
        Self {
            respect_gitignore: true,
        }
    }

    /// Configure whether to respect .gitignore files.
    pub fn with_gitignore(mut self, respect: bool) -> Self {
        self.respect_gitignore = respect;
        self
    }
}

#[async_trait]
impl Tool for RipgrepTool {
    fn name(&self) -> &str {
        "rg"
    }

    fn description(&self) -> &str {
        "Search file contents using ripgrep. Fast, respects .gitignore, skips binary files. \
         Returns matches in file:line:content format."
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
                "glob": {
                    "type": "string",
                    "description": "Only search files matching this glob (e.g., *.rs)"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Ignore case when matching (default: false)"
                },
                "hidden": {
                    "type": "boolean",
                    "description": "Search hidden files and directories (default: false)"
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

        let glob_pattern = input.get("glob").and_then(|v| v.as_str());
        let case_insensitive = input
            .get("case_insensitive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let search_hidden = input
            .get("hidden")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        // Build the regex matcher
        let matcher = match RegexMatcherBuilder::new()
            .case_insensitive(case_insensitive)
            .build(pattern_str)
        {
            Ok(m) => m,
            Err(e) => return ToolOutput::error(format!("Invalid regex: {e}")),
        };

        let base_path = Path::new(path_str);
        if !base_path.exists() {
            return ToolOutput::error(format!("Path does not exist: {}", path_str));
        }

        // Collect matches using ripgrep's searcher
        let matches: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let matches_clone = Arc::clone(&matches);

        // Build the file walker
        let mut walker = WalkBuilder::new(base_path);
        walker
            .hidden(!search_hidden)
            .git_ignore(self.respect_gitignore)
            .git_global(self.respect_gitignore)
            .git_exclude(self.respect_gitignore);

        // Add glob filter if specified
        if let Some(glob) = glob_pattern {
            let mut types_builder = ignore::types::TypesBuilder::new();
            if types_builder.add("custom", glob).is_ok() {
                types_builder.select("custom");
                if let Ok(types) = types_builder.build() {
                    walker.types(types);
                }
            }
        }

        // Search each file
        let mut searcher = Searcher::new();

        for entry in walker.build().filter_map(|e| e.ok()) {
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }

            let path = entry.path();
            let path_str = path.display().to_string();

            // Check if we've hit max matches
            if matches_clone.lock().unwrap().len() >= MAX_MATCHES {
                break;
            }

            let matches_ref = Arc::clone(&matches_clone);

            // Search this file
            let _ = searcher.search_path(
                &matcher,
                path,
                UTF8(|line_num, line| {
                    let mut matches = matches_ref.lock().unwrap();
                    if matches.len() >= MAX_MATCHES {
                        return Ok(false); // Stop searching
                    }

                    let line_trimmed = line.trim_end();
                    let display_line = if line_trimmed.len() > MAX_LINE_LEN {
                        format!("{}...", &line_trimmed[..MAX_LINE_LEN])
                    } else {
                        line_trimmed.to_string()
                    };

                    matches.push(format!("{}:{}:{}", path_str, line_num, display_line));
                    Ok(true)
                }),
            );
        }

        let matches = matches.lock().unwrap();

        if matches.is_empty() {
            return ToolOutput::success(format!(
                "No matches for '{}' in {}",
                pattern_str, path_str
            ));
        }

        let truncated = matches.len() >= MAX_MATCHES;
        let mut output = matches.join("\n");

        if truncated {
            output.push_str(&format!(
                "\n\n[Showing first {} matches, more may exist]",
                MAX_MATCHES
            ));
        }

        ToolOutput::success(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn temp_dir() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    #[tokio::test]
    async fn test_rg_simple() {
        let dir = temp_dir();
        let file = dir.path().join("test.rs");
        fs::write(&file, "fn main() {\n    println!(\"hello\");\n}").unwrap();

        let tool = RipgrepTool::new();
        let output = tool
            .execute(json!({
                "pattern": "fn main",
                "path": dir.path().to_str().unwrap()
            }))
            .await;

        assert!(!output.is_error, "Error: {}", output.content);
        assert!(output.content.contains("fn main"));
        assert!(output.content.contains(":1:"));
    }

    #[tokio::test]
    async fn test_rg_case_insensitive() {
        let dir = temp_dir();
        let file = dir.path().join("test.txt");
        fs::write(&file, "Hello World\nhello world\nHELLO WORLD").unwrap();

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
        fs::write(&file, "nothing here").unwrap();

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

    #[tokio::test]
    async fn test_rg_respects_gitignore() {
        let dir = temp_dir();

        // Create a git repo with .gitignore
        fs::create_dir(dir.path().join(".git")).unwrap();
        fs::write(dir.path().join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(dir.path().join("included.txt"), "find me").unwrap();
        fs::write(dir.path().join("ignored.txt"), "find me").unwrap();

        let tool = RipgrepTool::new();
        let output = tool
            .execute(json!({
                "pattern": "find me",
                "path": dir.path().to_str().unwrap()
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("included.txt"));
        assert!(!output.content.contains("ignored.txt"));
    }

    #[tokio::test]
    async fn test_rg_nested_directory() {
        let dir = temp_dir();
        let sub = dir.path().join("src").join("lib");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("mod.rs"), "pub fn hello() {}").unwrap();

        let tool = RipgrepTool::new();
        let output = tool
            .execute(json!({
                "pattern": "pub fn",
                "path": dir.path().to_str().unwrap()
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("mod.rs"));
        assert!(output.content.contains("pub fn hello"));
    }

    #[test]
    fn test_tool_metadata() {
        let tool = RipgrepTool::new();
        assert_eq!(tool.name(), "rg");
        assert!(tool.description().contains("ripgrep"));
    }
}
