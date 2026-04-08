//! Built-in tools for agent-rs.
//!
//! This crate provides common tools that agents can use to interact with
//! the world. Each tool implements the [`Tool`](agent_core::Tool) trait from
//! agent-core.
//!
//! # Available Tools
//!
//! - [`BashTool`]: Execute shell commands (the "escape hatch" for any operation)
//!
//! # Philosophy
//!
//! We keep the built-in tool set minimal. The bash tool alone can accomplish
//! most tasks through composition of standard Unix utilities. Specialized tools
//! are only added when they provide significant value over bash:
//!
//! - **Better error handling**: Structured errors vs. parsing stderr
//! - **Safety**: Controlled operations vs. arbitrary shell execution
//! - **Convenience**: Common operations that would be verbose in bash
//!
//! # Adding Tools
//!
//! To add a new tool:
//!
//! 1. Create a module in `src/` (e.g., `src/file.rs`)
//! 2. Implement the [`Tool`](agent_core::Tool) trait
//! 3. Add comprehensive documentation explaining what/why/when
//! 4. Export from this file
//!
//! Remember: tools should be self-contained and never panic.
//! Return [`ToolOutput::error()`](agent_core::ToolOutput::error) for failures.

mod bash;
mod file;

pub use bash::BashTool;
pub use file::FileTool;
