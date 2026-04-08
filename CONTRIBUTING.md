# Contributing to agent-rs

This document explains how to contribute to agent-rs while maintaining its educational and quality goals.

## Testing Strategy

We use a two-tier testing approach:

| Type | Location | API Key? | Purpose |
|------|----------|----------|---------|
| Unit tests | `src/**/*.rs` | No | Test internal logic with mocks |
| Integration tests | `examples/live_*.rs` | **Yes** | Test real API round-trips |
| Mock demos | `examples/mock_*.rs` | No | Educational examples |

### Unit Tests (No API Key)

Unit tests use `MockClient` and run with `cargo test`. They verify:

- Type conversions and parsing
- Agent loop state machine
- Error classification
- Tool dispatch logic

```bash
cargo test  # Runs all unit tests, no API key needed
```

Contributors can run unit tests without any setup.

### Integration Tests (API Key Required)

Integration tests (`examples/live_*.rs`) hit real LLM APIs. They verify:

- Request serialization is correct
- SSE streaming works end-to-end
- Tool calls round-trip properly
- Provider-specific quirks are handled

**Setup:**

1. Copy `.env.example` to `.env`
2. Add your API key: `ANTHROPIC_API_KEY=sk-ant-...`
3. Run:

```bash
cargo run --example live_simple -p agent-core
```

**When to run integration tests:**

- Before submitting a PR that touches providers or the agent loop
- Before releases
- When debugging "works in tests, fails in production" issues

### Mock Demos (No API Key)

Mock demos (`examples/mock_*.rs`) are for learning. They show how the API works with scripted responses. Not real integration tests, but useful for:

- Understanding the event flow
- Documentation examples
- Quick iteration during development

```bash
cargo run --example mock_demo -p agent-core
```

## Example Index

| Example | Type | What It Tests |
|---------|------|---------------|
| `mock_demo` | Mock | Agent construction, event streaming |
| `live_simple` | Live | Basic Anthropic API integration, tool calls |

## Adding New Features

### Before You Start

1. Read `CLAUDE.md` for project principles
2. Check if the feature fits the "radical simplicity" goal
3. Consider: can this be done without adding code?

### Implementation Checklist

- [ ] Feature is minimal and focused
- [ ] Public APIs have doc comments (what, why, when to use)
- [ ] Non-obvious code has inline comments
- [ ] Unit tests cover edge cases (with mocks)
- [ ] Integration test verifies real behavior (if touching API code)
- [ ] `cargo test` passes
- [ ] `cargo run --example live_simple` works (if you have a key)

### Adding a New Integration Test

```rust
//! examples/live_feature.rs

use agent_core::{AgentBuilder, AnthropicClient, ...};

#[tokio::main]
async fn main() {
    // Load .env
    dotenvy::dotenv().ok();

    // Get API key or exit with helpful message
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("Set ANTHROPIC_API_KEY in .env");

    // Use real client
    let client = AnthropicClient::new(api_key, "claude-sonnet-4-20250514");

    // Test the feature
    // ...

    // Print verification
    println!("[PASS] Feature works");
}
```

## Commit Messages

Write educational commit messages that explain:

1. **What** changed (first line, imperative mood)
2. **Why** it was done this way
3. **What alternatives** were considered (if relevant)

Example:
```
Add FileTool - structured file operations with safety controls

This commit adds FileTool, demonstrating a different pattern from BashTool:

1. **Structured operations**: Instead of arbitrary shell commands...
2. **Better error messages**: "Cannot read /foo/bar.txt"...
3. **Directory restrictions**: The `with_allowed_dir()` method...
```

## Code Style

- Prefer static functions over methods when `self` isn't needed
- Prefer simple types over custom wrappers
- Prefer explicit over implicit - no magic
- No `unwrap()` in library code
- Comments explain "why", code shows "what"

## Questions?

Open an issue to discuss before making large changes.
