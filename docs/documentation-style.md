# Documentation and Comment Style Guide

This document describes the documentation philosophy for agent-rs. The codebase is designed to be **educational**—a reference implementation that teaches people how to build their own LLM agents.

## Philosophy: Code as Teaching Material

This isn't just a library; it's a learning resource. Every function, trait, and design choice should help someone understand:

1. **What** it does
2. **Why** it exists (what problem does it solve?)
3. **What end** it serves (how does it fit into the bigger picture?)

A reader should be able to understand the agent architecture by reading the code and comments, without needing external resources.

## Comment Categories

### 1. Module-Level Documentation

Every module (file) starts with a doc comment explaining:
- What this module is responsible for
- How it fits into the larger system
- Key types/functions it exposes

```rust
//! # Agent Loop
//!
//! This module contains the core agent execution loop. The loop:
//! 1. Sends the current conversation to the LLM
//! 2. Processes the response for tool calls
//! 3. Executes tools in parallel
//! 4. Feeds results back to the LLM
//! 5. Repeats until completion or max turns
//!
//! ## Design Decisions
//!
//! The loop is synchronous from the perspective of turns—we complete
//! all tools before starting the next turn. This simplifies error
//! handling and makes the mental model clearer. See the [concurrency
//! documentation](../docs/concurrency.md) for details.
//!
//! ## Key Types
//!
//! - [`Agent`]: The executor, constructed via [`AgentBuilder`]
//! - [`AgentEvent`]: Events emitted during execution
```

### 2. Type Documentation

Every public type explains what it represents and when to use it:

```rust
/// A complete LLM response (non-streaming).
///
/// # What This Represents
///
/// After sending a request to an LLM, this is what you get back.
/// It contains the model's output (text and/or tool calls), why
/// it stopped generating, and token usage statistics.
///
/// # When You'll See This
///
/// - From `LlmClient::complete()` (non-streaming)
/// - You won't see this directly when using `Agent`—it's internal
///
/// # Why These Fields?
///
/// - `content`: The actual response—text and tool calls
/// - `stop_reason`: Tells us whether to execute tools or finish
/// - `usage`: For token budgeting and cost tracking
pub struct Response {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub usage: TokenUsage,
}
```

### 3. Trait Documentation

Traits explain the abstraction boundary and implementation requirements:

```rust
/// Abstraction over LLM provider APIs.
///
/// # Why This Trait Exists
///
/// Different LLM providers (Anthropic, OpenAI, Google, AWS) have
/// different APIs. This trait normalizes them so the agent loop
/// can work with any provider without knowing the details.
///
/// # For Library Users
///
/// You don't implement this—you choose a provided implementation:
/// ```rust
/// let client = AnthropicClient::new(&api_key, "claude-sonnet-4-20250514");
/// ```
///
/// # For Contributors Adding Providers
///
/// Implement this trait. Your implementation must:
/// 1. Translate our `Request` type to the provider's format
/// 2. Handle authentication (API keys, AWS SigV4, etc.)
/// 3. Parse responses back into our `Response`/`StreamChunk` types
/// 4. Map errors to appropriate `LlmError` variants
///
/// See `providers/anthropic.rs` as a reference implementation.
#[async_trait]
pub trait LlmClient: Send + Sync {
    // ...
}
```

### 4. Function Documentation

Functions explain what they do, why they exist, and how they fit in:

```rust
/// Execute all pending tool calls in parallel.
///
/// # What This Does
///
/// Takes a list of tool calls from the LLM response and executes
/// them all concurrently using `futures::future::join_all`.
///
/// # Why Parallel?
///
/// When the model issues multiple tool calls (e.g., reading several
/// files), they're typically independent. Parallel execution reduces
/// latency from O(sum of tool times) to O(max tool time).
///
/// # Why Not Spawn?
///
/// We use `join_all`, not `tokio::spawn`. This keeps tools scoped
/// to the current turn—they all complete before we continue. Spawning
/// would complicate cancellation and error handling.
///
/// # Error Handling
///
/// Tool errors don't propagate—they're captured in `ToolOutput` with
/// `is_error: true`. The model sees the error and can self-correct.
/// Even panics are caught and converted to error outputs.
///
/// # Arguments
///
/// * `tools` - The tool registry to look up implementations
/// * `calls` - Tool calls parsed from the LLM response
///
/// # Returns
///
/// A vector of (call_id, output) pairs in the same order as input.
async fn execute_tools(
    tools: &[Box<dyn Tool>],
    calls: Vec<ToolCall>,
) -> Vec<(String, ToolOutput)> {
    // Implementation...
}
```

### 5. Inline Comments

Use inline comments for non-obvious code, tricky logic, or design justifications:

```rust
async fn run_turn(&mut self) -> Result<TurnResult, AgentError> {
    // Check cancellation at turn boundaries, not mid-operation.
    // This ensures we never leave things in an inconsistent state.
    if self.cancel_token.is_cancelled() {
        return Err(AgentError::Cancelled);
    }

    // Budget check BEFORE the LLM call, not after.
    // If we're over budget, we evict first so the request succeeds.
    if self.context_manager.should_evict() {
        self.evict_context().await?;
    }

    let request = self.build_request();

    // Use streaming even though we don't always need it.
    // Unified code path is simpler than maintaining two paths.
    let mut stream = self.client.complete_stream(request);

    // Accumulate the response as we stream.
    // We need the full response to process tool calls anyway.
    let mut response = ResponseAccumulator::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;

        // Emit events as we receive chunks for real-time display.
        // The UI can show text appearing character by character.
        if let Some(event) = self.chunk_to_event(&chunk) {
            yield event;
        }

        response.accumulate(chunk);
    }

    // ...
}
```

### 6. "Why Not" Comments

Explain rejected alternatives to prevent future contributors from repeating analysis:

```rust
/// Estimate token count for a string.
///
/// # Why Not Use a Real Tokenizer?
///
/// Real tokenizers (tiktoken, sentencepiece) are:
/// - Provider-specific (different tokenizers for different models)
/// - Heavy dependencies (native code, large vocab files)
/// - Slow for our use case (we check every turn)
///
/// Our heuristic (4 chars ≈ 1 token) is:
/// - Fast (just string length)
/// - Conservative (over-estimates, so we evict early rather than late)
/// - Good enough (within 20% of actual for English text)
fn estimate_tokens(text: &str) -> u32 {
    (text.len() as u32 + 3) / 4
}
```

## Comment Density Guidelines

### High-Comment Areas

These areas need thorough documentation:

1. **Public API** - Everything users touch
2. **Trait definitions** - Contracts that others implement
3. **State machines** - The agent loop, eviction logic
4. **Error handling** - What errors mean and what to do
5. **Concurrency** - What's shared, what's spawned, how to cancel

### Low-Comment Areas

These need less documentation:

1. **Simple getters/setters** - Self-explanatory
2. **Direct translations** - JSON serialization that mirrors the schema
3. **Test code** - Tests are documentation themselves
4. **Internal helpers** - Private functions with obvious behavior

## Doc Comment Format

Use standard Rust doc comment conventions:

```rust
/// Short one-line summary.
///
/// Longer explanation if needed. Can span multiple paragraphs.
///
/// # Section Headers
///
/// Use headers to organize longer docs:
/// - `# Arguments` - Function parameters
/// - `# Returns` - Return value
/// - `# Errors` - When it returns Err
/// - `# Panics` - When it panics (avoid panicking)
/// - `# Examples` - Usage examples
/// - `# Safety` - For unsafe code
///
/// # Custom Headers
///
/// For this project, also use:
/// - `# What This Does` - Concrete description
/// - `# Why This Exists` - Problem it solves
/// - `# Design Decisions` - Choices and trade-offs
/// - `# Why Not X` - Rejected alternatives
///
/// # Examples
///
/// ```rust
/// let result = my_function(arg);
/// assert_eq!(result, expected);
/// ```
pub fn my_function(arg: Type) -> Result<Output, Error> {
    // ...
}
```

## Learning-Oriented Examples

Examples should teach, not just demonstrate:

```rust
/// # Example: Building a Simple Agent
///
/// This example shows the minimal setup for a working agent:
///
/// ```rust
/// use agent_core::{Agent, AgentBuilder, AnthropicClient, AgentEvent};
/// use agent_tools::BashTool;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Step 1: Create an LLM client
///     // This handles authentication and API communication.
///     // We're using Anthropic, but OpenAI/Gemini work the same way.
///     let client = AnthropicClient::new(
///         std::env::var("ANTHROPIC_API_KEY")?,
///         "claude-sonnet-4-20250514"
///     );
///
///     // Step 2: Create tools
///     // Tools are the agent's "hands"—without them, it can only talk.
///     // BashTool lets it execute shell commands.
///     let bash = BashTool::new()
///         .with_timeout(Duration::from_secs(30));
///
///     // Step 3: Build the agent
///     // The builder pattern lets you configure optional features.
///     let agent = AgentBuilder::new(Box::new(client))
///         .system_prompt("You are a helpful assistant.")
///         .tools(vec![Box::new(bash)])
///         .max_turns(10)  // Safety limit
///         .build();
///
///     // Step 4: Run and process events
///     // The agent returns a stream of events as it works.
///     let mut stream = agent.run("What's in the current directory?");
///
///     while let Some(event) = stream.next().await {
///         match event? {
///             AgentEvent::Text(s) => print!("{s}"),
///             AgentEvent::ToolStart { name, .. } => {
///                 println!("\n[Executing: {name}]");
///             }
///             AgentEvent::Done { .. } => break,
///             _ => {}
///         }
///     }
///
///     Ok(())
/// }
/// ```
```

## Cross-References

Link to related documentation:

```rust
/// Context manager for token budgeting and eviction.
///
/// For a complete explanation of the context management system,
/// see the [Context Management documentation](../docs/context-management.md).
///
/// Related types:
/// - [`ContextStore`] - SQLite persistence layer
/// - [`RecallTool`] - Tool for searching evicted context
pub struct ContextManager {
    // ...
}
```

## Commenting Checklist

Before committing code, verify:

- [ ] Every public type has a doc comment explaining what and why
- [ ] Every public function has a doc comment with arguments and returns
- [ ] Every trait has implementation guidance
- [ ] Complex logic has inline comments explaining the reasoning
- [ ] Non-obvious choices have "why not" explanations
- [ ] Examples compile and run
- [ ] Cross-references link to relevant docs

## Anti-Patterns to Avoid

### Don't State the Obvious

```rust
// BAD: This comment adds nothing
/// Returns the name.
fn name(&self) -> &str { &self.name }

// GOOD: Explain what's not obvious
/// Returns the tool's unique identifier.
///
/// Used by the model to invoke this tool. Must be unique within
/// the tool registry—duplicates shadow earlier registrations.
fn name(&self) -> &str { &self.name }
```

### Don't Explain Syntax

```rust
// BAD: Explaining Rust syntax
// Create a new vector and push items to it
let mut v = Vec::new();
v.push(1);

// GOOD: Explain the domain logic
// Collect tool calls in order—we'll execute them in parallel
// but return results in this order for determinism.
let mut calls = Vec::new();
for block in &response.content {
    if let ContentBlock::ToolUse { .. } = block {
        calls.push(block.clone());
    }
}
```

### Don't Let Comments Rot

Comments must be updated when code changes. A wrong comment is worse than no comment. If you change behavior, update the comment. If you see a stale comment, fix it.
