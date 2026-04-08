//! Built-in tools for agent-rs.
//!
//! This crate provides common tools that agents can use to interact with
//! the world. Each tool implements the [`Tool`](agent_core::Tool) trait from
//! agent-core.
//!
//! # Available Tools
//!
//! ## General Purpose
//!
//! - [`BashTool`]: Execute shell commands (the "escape hatch" for any operation)
//! - [`FileTool`]: Structured file operations with safety controls
//!
//! ## Unix-Style Tools
//!
//! Focused, composable tools following the Unix philosophy:
//!
//! - [`CatTool`]: Read file contents
//! - [`HeadTool`]: Read first N lines of a file
//! - [`FindTool`]: Find files by name pattern
//! - [`RipgrepTool`]: Search file contents with regex
//!
//! # Philosophy
//!
//! Two approaches are available:
//!
//! 1. **BashTool**: The escape hatch. Can do anything but requires parsing output.
//! 2. **Unix tools**: Focused tools that do one thing well with structured I/O.
//!
//! The Unix tools compose naturally:
//! - `find` to locate files
//! - `rg` to search contents
//! - `head` or `cat` to read results
//!
//! # Adding Tools
//!
//! To add a new tool:
//!
//! 1. Create a module in `src/` (e.g., `src/mytool.rs`)
//! 2. Implement the [`Tool`](agent_core::Tool) trait
//! 3. Add comprehensive documentation explaining what/why/when
//! 4. Export from this file
//!
//! Remember: tools should be self-contained and never panic.
//! Return [`ToolOutput::error()`](agent_core::ToolOutput::error) for failures.

mod bash;
mod cat;
mod file;
mod find;
mod head;
mod ripgrep;

pub use bash::BashTool;
pub use cat::CatTool;
pub use file::FileTool;
pub use find::FindTool;
pub use head::HeadTool;
pub use ripgrep::RipgrepTool;
