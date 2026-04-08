# CLAUDE.md - Project Context for AI Assistants

This file provides context for AI assistants (like Claude) working on the agent-rs codebase.

## Project Overview

agent-rs is a minimal, educational LLM agent harness written in Rust. It's designed to be:

1. **Radically simple** - ~750 lines total across three crates
2. **Educational** - Every design choice is documented with what, why, and what end
3. **Multi-provider** - Supports Anthropic, OpenAI, Gemini, and AWS Bedrock
4. **Embeddable** - Library-first design, no framework opinions

## Crate Structure

```
agent-rs/
├── agent-core/       # The embeddable library (agent loop, LLM clients, tools, context)
├── agent-tools/      # Built-in tools (bash, file, recall)
├── agent-session/    # Job orchestration layer (optional)
├── agent-bin/        # Desktop GUI binary (not documented, uses egui)
└── docs/             # Architecture documentation
```

## Key Principles to Follow

### 1. Radical Simplicity

- **Don't add abstractions unless they're necessary.** If something can be a function, don't make it a trait.
- **Don't add types unless they're necessary.** If two types can be one, merge them.
- **Target ~400 lines for agent-core, ~200 for agent-tools, ~150 for agent-session.**
- **When in doubt, leave it out.** Features can be added later; bloat is hard to remove.

### 2. Educational Clarity

- **Every public API needs doc comments** explaining what it does, why it exists, and what end it serves.
- **Non-obvious code needs inline comments** explaining the reasoning.
- **Rejected alternatives need "why not" comments** to prevent repeating analysis.
- **See `docs/documentation-style.md`** for detailed guidelines.

### 3. Error Philosophy

- **Tool errors are data, not exceptions.** Return `ToolOutput::error()`, don't panic.
- **LLM errors are classified by recoverability.** Rate limits retry; auth errors bubble.
- **The model self-corrects.** Feed errors to the model, let it adapt.

### 4. Concurrency Model

- **Everything runs on Tokio.** No thread pools, no custom executors.
- **Tools execute in parallel** via `join_all`, not `spawn`.
- **Cancellation uses `CancellationToken`** from tokio-util.
- **Shared state uses DashMap** for lock-free concurrent access.

## Common Tasks

### Adding a New LLM Provider

1. Create `agent-core/src/providers/my_provider.rs`
2. Implement `LlmClient` trait
3. Handle request translation (our types → provider JSON)
4. Handle response translation (provider JSON → our types)
5. Handle streaming format parsing
6. Export from `agent-core/src/providers/mod.rs`
7. See `docs/providers.md` for detailed guidance

### Adding a New Tool

1. Create a struct in `agent-tools/src/`
2. Implement the `Tool` trait from agent-core
3. Provide `name()`, `description()`, `input_schema()`, `execute()`
4. Export from `agent-tools/src/lib.rs`
5. Remember: never panic, return `ToolOutput::error()` instead

### Modifying the Agent Loop

The agent loop is in `agent-core/src/agent.rs`. Key structure:

```
loop {
    1. Check cancellation
    2. Check context budget, evict if needed
    3. Build request
    4. Stream LLM response
    5. If tool_use: execute tools in parallel, loop
    6. If end_turn: done
}
```

Changes should maintain this structure. The loop is the core of the system.

## Architecture Documentation

Comprehensive docs are in the `docs/` folder:

- `README.md` - Overview and index
- `architecture.md` - System design and diagrams
- `crates.md` - API surface for each crate
- `data-flow.md` - End-to-end data flow
- `context-management.md` - SQLite schema and eviction
- `error-handling.md` - Error classification and retry
- `traits.md` - The two public traits
- `concurrency.md` - Spawning, shared state, cancellation
- `providers.md` - Multi-provider normalization
- `documentation-style.md` - Commenting guidelines

## Code Style

- **Prefer static functions over methods** when `self` isn't needed
- **Prefer simple types** - use primitives and standard library types over custom wrappers
- **Prefer composition over inheritance** - embed structs rather than trait hierarchies
- **Prefer explicit over implicit** - no magic, no hidden behavior, clear data flow
- **Minimize generics** - concrete types are easier to read and debug
- **No `unwrap()` in library code** - use `?` or handle errors explicitly

## Testing Strategy

- **Unit tests** for type conversions and utilities (in `#[cfg(test)]` modules)
- **Examples as integration tests** - each example in `examples/` serves dual purpose:
  1. Educational: shows how to use the API
  2. Verification: exercises real code paths end-to-end
- **MockClient** for deterministic testing without API keys
- **No external service tests** in CI - examples use mocks

### Examples Are Integration Tests

Every example should:
1. Be runnable without API keys (use MockClient for LLM calls)
2. Exercise a specific feature or workflow
3. Have clear output showing what's happening
4. Fail visibly if something breaks

Run all examples to verify the system works:
```bash
cargo run --example simple_agent -p agent-core
# Add more examples as they're created
```

See `CONTRIBUTING.md` for guidelines on adding examples.

## Feature Flags

```toml
[features]
default = []
persistence = ["rusqlite"]  # SQLite context storage
bedrock = ["aws-config", "aws-sdk-bedrockruntime"]  # AWS Bedrock support
```

## Important Files

- `agent-core/src/lib.rs` - Public API exports
- `agent-core/src/agent.rs` - Agent and AgentBuilder
- `agent-core/src/client.rs` - LlmClient trait
- `agent-core/src/tool.rs` - Tool trait
- `agent-core/src/types.rs` - Request, Response, Message, etc.
- `agent-core/src/providers/` - Provider implementations
- `agent-core/examples/` - Integration tests (runnable examples)
- `agent-tools/src/` - Built-in tool implementations

## Things to Avoid

- **Don't add frameworks or heavyweight dependencies**
- **Don't add global state or singletons**
- **Don't add configuration files** - everything is explicit at construction
- **Don't add magic or implicit behavior**
- **Don't add traits unless there are multiple implementations**
- **Don't panic in library code**
- **Don't use `async_std`** - this project uses Tokio exclusively

## When Making Changes

1. **Read the relevant architecture doc first**
2. **Keep changes minimal** - do only what's necessary
3. **Update documentation** if behavior changes
4. **Add comments** explaining why, not just what
5. **Run tests and examples** before committing:
   ```bash
   cargo test
   cargo run --example simple_agent -p agent-core
   ```
6. **Keep the line count low** - if a change adds significant lines, question if it's necessary
7. **See `CONTRIBUTING.md`** for detailed guidelines
