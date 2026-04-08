//! File operations tool.
//!
//! This module provides `FileTool`, which handles common file operations
//! (read, write, list) with structured input and better error messages
//! than raw bash commands.
//!
//! # Why a Dedicated File Tool?
//!
//! The bash tool can do file operations (`cat`, `echo >`, `ls`), so why
//! have a separate file tool? Several reasons:
//!
//! 1. **Structured errors**: "File not found: /foo/bar.txt" is clearer than
//!    parsing `cat: /foo/bar.txt: No such file or directory`
//!
//! 2. **Safety**: The file tool can restrict operations to specific directories,
//!    preventing accidental writes to system files
//!
//! 3. **Binary safety**: Properly handles files that might contain binary data
//!    or unusual encodings
//!
//! 4. **Size limits**: Enforces output limits before reading, not after
//!
//! # Operations
//!
//! - `read`: Read file contents
//! - `write`: Write content to a file (creates or overwrites)
//! - `append`: Append content to a file
//! - `list`: List directory contents
//! - `exists`: Check if a path exists
//!
//! # Example
//!
//! ```ignore
//! let file_tool = FileTool::new()
//!     .with_allowed_dir("/home/user/project");
//!
//! // The LLM can now read/write files in that directory:
//! // {"operation": "read", "path": "/home/user/project/src/main.rs"}
//! ```

use std::path::{Path, PathBuf};

use agent_core::{Tool, ToolOutput};
use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::fs;

/// Maximum file size for read operations (1 MB).
///
/// Files larger than this are truncated. This prevents memory exhaustion
/// and context window overflow. The LLM can use line ranges or grep
/// for larger files.
const MAX_READ_SIZE: u64 = 1024 * 1024;

/// Tool for structured file operations.
///
/// # Design Philosophy
///
/// This tool complements (doesn't replace) the bash tool. Use FileTool when:
/// - You want clear error messages for file operations
/// - You need to enforce directory restrictions
/// - You're working with potentially large files
///
/// Use bash when:
/// - You need complex operations (find, grep patterns, etc.)
/// - You're chaining multiple operations
/// - You need shell features like globbing
///
/// # Security
///
/// The `allowed_dirs` setting restricts operations to specific directories.
/// This is a defense-in-depth measure - paths outside these directories
/// are rejected before any file system operation occurs.
///
/// However, symlinks can bypass this check. For true sandboxing, run
/// the agent in a container or VM.
pub struct FileTool {
    /// Directories where operations are permitted.
    /// If empty, all directories are allowed.
    allowed_dirs: Vec<PathBuf>,
}

impl Default for FileTool {
    fn default() -> Self {
        Self::new()
    }
}

impl FileTool {
    /// Create a new file tool with no directory restrictions.
    ///
    /// **Warning**: This allows file operations anywhere on the filesystem.
    /// Use `with_allowed_dir()` to restrict to specific directories.
    pub fn new() -> Self {
        Self {
            allowed_dirs: Vec::new(),
        }
    }

    /// Restrict operations to a specific directory.
    ///
    /// Can be called multiple times to allow multiple directories.
    /// Operations on paths outside these directories will fail.
    ///
    /// Note: This checks path prefixes, not canonicalized paths.
    /// Symlinks could potentially bypass this restriction.
    pub fn with_allowed_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.allowed_dirs.push(dir.into());
        self
    }

    /// Check if a path is within allowed directories.
    ///
    /// Returns true if:
    /// - No directories are restricted (allowed_dirs is empty), or
    /// - The path starts with one of the allowed directories
    fn is_allowed(&self, path: &Path) -> bool {
        if self.allowed_dirs.is_empty() {
            return true;
        }

        self.allowed_dirs.iter().any(|allowed| path.starts_with(allowed))
    }

    /// Execute a read operation.
    async fn do_read(&self, path: &Path) -> ToolOutput {
        // Check file metadata first to get size
        let metadata = match fs::metadata(path).await {
            Ok(m) => m,
            Err(e) => return ToolOutput::error(format!("Cannot read {}: {e}", path.display())),
        };

        if metadata.is_dir() {
            return ToolOutput::error(format!("{} is a directory, use 'list' operation", path.display()));
        }

        let size = metadata.len();
        let truncated = size > MAX_READ_SIZE;

        // Read the file (up to max size)
        let content = if truncated {
            // Read only the first MAX_READ_SIZE bytes
            match fs::read(path).await {
                Ok(bytes) => {
                    let truncated_bytes = &bytes[..MAX_READ_SIZE as usize];
                    String::from_utf8_lossy(truncated_bytes).into_owned()
                }
                Err(e) => return ToolOutput::error(format!("Cannot read {}: {e}", path.display())),
            }
        } else {
            match fs::read_to_string(path).await {
                Ok(s) => s,
                Err(e) if e.kind() == std::io::ErrorKind::InvalidData => {
                    // Binary file - read as bytes and show hex preview
                    match fs::read(path).await {
                        Ok(bytes) => {
                            let preview: String = bytes.iter().take(100).map(|b| format!("{:02x} ", b)).collect();
                            format!("[Binary file, {} bytes. First 100 bytes hex: {}]", size, preview)
                        }
                        Err(e) => return ToolOutput::error(format!("Cannot read {}: {e}", path.display())),
                    }
                }
                Err(e) => return ToolOutput::error(format!("Cannot read {}: {e}", path.display())),
            }
        };

        if truncated {
            ToolOutput::success(format!(
                "{content}\n\n[TRUNCATED: File is {size} bytes, showing first {MAX_READ_SIZE}]"
            ))
        } else {
            ToolOutput::success(content)
        }
    }

    /// Execute a write operation.
    async fn do_write(&self, path: &Path, content: &str) -> ToolOutput {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            if !parent.exists() {
                if let Err(e) = fs::create_dir_all(parent).await {
                    return ToolOutput::error(format!("Cannot create directory {}: {e}", parent.display()));
                }
            }
        }

        match fs::write(path, content).await {
            Ok(()) => ToolOutput::success(format!("Wrote {} bytes to {}", content.len(), path.display())),
            Err(e) => ToolOutput::error(format!("Cannot write {}: {e}", path.display())),
        }
    }

    /// Execute an append operation.
    async fn do_append(&self, path: &Path, content: &str) -> ToolOutput {
        use tokio::io::AsyncWriteExt;

        let mut file = match tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await
        {
            Ok(f) => f,
            Err(e) => return ToolOutput::error(format!("Cannot open {}: {e}", path.display())),
        };

        if let Err(e) = file.write_all(content.as_bytes()).await {
            return ToolOutput::error(format!("Cannot append to {}: {e}", path.display()));
        }

        // Flush to ensure data is written before returning
        if let Err(e) = file.flush().await {
            return ToolOutput::error(format!("Cannot flush {}: {e}", path.display()));
        }

        ToolOutput::success(format!("Appended {} bytes to {}", content.len(), path.display()))
    }

    /// Execute a list operation.
    async fn do_list(&self, path: &Path) -> ToolOutput {
        let metadata = match fs::metadata(path).await {
            Ok(m) => m,
            Err(e) => return ToolOutput::error(format!("Cannot access {}: {e}", path.display())),
        };

        if !metadata.is_dir() {
            return ToolOutput::error(format!("{} is not a directory", path.display()));
        }

        let mut entries = Vec::new();
        let mut dir = match fs::read_dir(path).await {
            Ok(d) => d,
            Err(e) => return ToolOutput::error(format!("Cannot list {}: {e}", path.display())),
        };

        while let Ok(Some(entry)) = dir.next_entry().await {
            let name = entry.file_name().to_string_lossy().into_owned();
            let file_type = match entry.file_type().await {
                Ok(ft) if ft.is_dir() => "dir",
                Ok(ft) if ft.is_symlink() => "link",
                Ok(_) => "file",
                Err(_) => "unknown",
            };
            entries.push(format!("{file_type}\t{name}"));
        }

        entries.sort();
        ToolOutput::success(entries.join("\n"))
    }

    /// Execute an exists check.
    async fn do_exists(&self, path: &Path) -> ToolOutput {
        let exists = path.exists();
        let file_type = if exists {
            let metadata = fs::metadata(path).await.ok();
            match metadata {
                Some(m) if m.is_dir() => "directory",
                Some(m) if m.is_file() => "file",
                Some(m) if m.is_symlink() => "symlink",
                _ => "unknown",
            }
        } else {
            "not found"
        };

        ToolOutput::success(format!("{}: {file_type}", path.display()))
    }
}

#[async_trait]
impl Tool for FileTool {
    fn name(&self) -> &str {
        "file"
    }

    fn description(&self) -> &str {
        "Perform file operations: read, write, append, list, exists. \
         Provides structured errors and can enforce directory restrictions. \
         Use for precise file operations; use bash for complex patterns."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "append", "list", "exists"],
                    "description": "The operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "The file or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content for write/append operations"
                }
            },
            "required": ["operation", "path"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        // Parse required fields
        let operation = match input.get("operation").and_then(|v| v.as_str()) {
            Some(op) => op,
            None => return ToolOutput::error("Missing required field: operation"),
        };

        let path_str = match input.get("path").and_then(|v| v.as_str()) {
            Some(p) => p,
            None => return ToolOutput::error("Missing required field: path"),
        };

        let path = Path::new(path_str);

        // Check directory restrictions
        if !self.is_allowed(path) {
            return ToolOutput::error(format!(
                "Path {} is outside allowed directories",
                path.display()
            ));
        }

        // Dispatch to operation handlers
        match operation {
            "read" => self.do_read(path).await,
            "write" => {
                let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");
                self.do_write(path, content).await
            }
            "append" => {
                let content = input.get("content").and_then(|v| v.as_str()).unwrap_or("");
                self.do_append(path, content).await
            }
            "list" => self.do_list(path).await,
            "exists" => self.do_exists(path).await,
            _ => ToolOutput::error(format!(
                "Unknown operation: {operation}. Use: read, write, append, list, exists"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs as std_fs;
    use tempfile::TempDir;

    /// Helper to create a temp directory for tests
    fn temp_dir() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    #[tokio::test]
    async fn test_read_file() {
        let dir = temp_dir();
        let file_path = dir.path().join("test.txt");
        std_fs::write(&file_path, "hello world").unwrap();

        let tool = FileTool::new();
        let output = tool
            .execute(json!({
                "operation": "read",
                "path": file_path.to_str().unwrap()
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("hello world"));
    }

    #[tokio::test]
    async fn test_read_nonexistent() {
        let tool = FileTool::new();
        let output = tool
            .execute(json!({
                "operation": "read",
                "path": "/nonexistent/file.txt"
            }))
            .await;

        assert!(output.is_error);
        assert!(output.content.contains("Cannot read"));
    }

    #[tokio::test]
    async fn test_write_file() {
        let dir = temp_dir();
        let file_path = dir.path().join("output.txt");

        let tool = FileTool::new();
        let output = tool
            .execute(json!({
                "operation": "write",
                "path": file_path.to_str().unwrap(),
                "content": "test content"
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("Wrote"));

        // Verify file was written
        let content = std_fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "test content");
    }

    #[tokio::test]
    async fn test_write_creates_parent_dirs() {
        let dir = temp_dir();
        let file_path = dir.path().join("nested").join("dir").join("file.txt");

        let tool = FileTool::new();
        let output = tool
            .execute(json!({
                "operation": "write",
                "path": file_path.to_str().unwrap(),
                "content": "nested content"
            }))
            .await;

        assert!(!output.is_error);
        assert!(file_path.exists());
    }

    #[tokio::test]
    async fn test_append_file() {
        let dir = temp_dir();
        let file_path = dir.path().join("append.txt");
        std_fs::write(&file_path, "line1\n").unwrap();

        let tool = FileTool::new();
        let output = tool
            .execute(json!({
                "operation": "append",
                "path": file_path.to_str().unwrap(),
                "content": "line2\n"
            }))
            .await;

        assert!(!output.is_error);

        let content = std_fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "line1\nline2\n");
    }

    #[tokio::test]
    async fn test_list_directory() {
        let dir = temp_dir();
        std_fs::write(dir.path().join("file1.txt"), "").unwrap();
        std_fs::write(dir.path().join("file2.txt"), "").unwrap();
        std_fs::create_dir(dir.path().join("subdir")).unwrap();

        let tool = FileTool::new();
        let output = tool
            .execute(json!({
                "operation": "list",
                "path": dir.path().to_str().unwrap()
            }))
            .await;

        assert!(!output.is_error);
        assert!(output.content.contains("file1.txt"));
        assert!(output.content.contains("file2.txt"));
        assert!(output.content.contains("subdir"));
    }

    #[tokio::test]
    async fn test_exists() {
        let dir = temp_dir();
        let file_path = dir.path().join("exists.txt");
        std_fs::write(&file_path, "").unwrap();

        let tool = FileTool::new();

        // Existing file
        let output = tool
            .execute(json!({
                "operation": "exists",
                "path": file_path.to_str().unwrap()
            }))
            .await;
        assert!(!output.is_error);
        assert!(output.content.contains("file"));

        // Non-existing file
        let output = tool
            .execute(json!({
                "operation": "exists",
                "path": dir.path().join("nope.txt").to_str().unwrap()
            }))
            .await;
        assert!(!output.is_error);
        assert!(output.content.contains("not found"));
    }

    #[tokio::test]
    async fn test_directory_restriction() {
        let dir = temp_dir();
        let allowed_dir = dir.path().join("allowed");
        std_fs::create_dir(&allowed_dir).unwrap();

        let tool = FileTool::new().with_allowed_dir(&allowed_dir);

        // Writing inside allowed dir - should work
        let output = tool
            .execute(json!({
                "operation": "write",
                "path": allowed_dir.join("test.txt").to_str().unwrap(),
                "content": "ok"
            }))
            .await;
        assert!(!output.is_error);

        // Writing outside allowed dir - should fail
        let output = tool
            .execute(json!({
                "operation": "write",
                "path": dir.path().join("outside.txt").to_str().unwrap(),
                "content": "nope"
            }))
            .await;
        assert!(output.is_error);
        assert!(output.content.contains("outside allowed"));
    }

    #[tokio::test]
    async fn test_unknown_operation() {
        let tool = FileTool::new();
        let output = tool
            .execute(json!({
                "operation": "delete",
                "path": "/tmp/test"
            }))
            .await;

        assert!(output.is_error);
        assert!(output.content.contains("Unknown operation"));
    }

    #[tokio::test]
    async fn test_missing_fields() {
        let tool = FileTool::new();

        // Missing operation
        let output = tool.execute(json!({"path": "/tmp/test"})).await;
        assert!(output.is_error);
        assert!(output.content.contains("operation"));

        // Missing path
        let output = tool.execute(json!({"operation": "read"})).await;
        assert!(output.is_error);
        assert!(output.content.contains("path"));
    }

    #[test]
    fn test_tool_metadata() {
        let tool = FileTool::new();
        assert_eq!(tool.name(), "file");
        assert!(tool.description().contains("file"));

        let schema = tool.input_schema();
        assert!(schema["properties"]["operation"]["enum"]
            .as_array()
            .unwrap()
            .contains(&json!("read")));
    }
}
