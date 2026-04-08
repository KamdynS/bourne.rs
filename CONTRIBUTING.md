# Contributing to agent-rs

This document explains how to contribute to agent-rs while maintaining its educational and quality goals.

## Examples Are Integration Tests

In agent-rs, **examples serve two purposes**:

1. **Documentation**: Show users how to use the API
2. **Integration testing**: Verify the system works end-to-end

This approach keeps tests close to real usage and ensures our examples always work.

### Writing an Example

Every example should follow this pattern:

```rust
//! Example Title
//!
//! Brief description of what this example demonstrates.
//!
//! # What This Shows
//!
//! - Bullet points of concepts demonstrated
//! - Each point should teach something
//!
//! # Running
//!
//! ```bash
//! cargo run --example example_name -p crate-name
//! ```

// Use MockClient for deterministic behavior without API keys
let client = MockClient::new(vec![
    MockResponse::text("Expected response"),
]);

// Build and run the agent/component being tested
// ...

// Print output that shows the system working
println!("Result: {result}");
```

### Example Guidelines

1. **No API keys required** - Use `MockClient` with scripted responses
2. **Clear output** - Print what's happening so users can follow along
3. **Self-contained** - Each example should work independently
4. **Educational comments** - Explain why, not just what
5. **Error handling** - Show how to handle errors gracefully

### What to Test via Examples

| Feature | Example Name | What It Tests |
|---------|--------------|---------------|
| Basic agent flow | `simple_agent` | Tool calls, event streaming, completion |
| Multiple tools | (future) | Tool selection, parallel execution |
| Error recovery | (future) | Tool errors, LLM retries |
| Cancellation | (future) | Graceful shutdown mid-execution |
| Streaming | (future) | Real-time text output |

### Running All Examples

```bash
# Run each example to verify the system
cargo run --example simple_agent -p agent-core

# Add new examples to this list as they're created
```

## Unit Tests

Unit tests go in `#[cfg(test)]` modules within source files. Use them for:

- Type conversions and parsing
- Edge cases in utility functions
- Error classification logic

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_specific_behavior() {
        // Test one thing clearly
    }
}
```

## Adding New Features

### Before You Start

1. Read `CLAUDE.md` for project principles
2. Check if the feature fits the "radical simplicity" goal
3. Consider: can this be done without adding code?

### Implementation Checklist

- [ ] Feature is minimal and focused
- [ ] Public APIs have doc comments (what, why, when to use)
- [ ] Non-obvious code has inline comments
- [ ] Unit tests cover edge cases
- [ ] Example demonstrates the feature
- [ ] Existing tests still pass (`cargo test`)
- [ ] Existing examples still run

### Commit Messages

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
