# agent-rs Architecture Documentation

This documentation describes the architecture of agent-rs, a headless agentic harness for LLM-driven task execution. The system is designed around two complementary philosophies: **radical simplicity** and **educational clarity**.

## Design Philosophy

### Radical Simplicity

This project draws inspiration from systems like [Pi](https://github.com/anthropics/pi) that demonstrate a capable agent harness can live in just a couple of files. The architecture is thoughtful and well-separated, but the implementation is deliberately tiny.

**Guiding principles:**

1. **If it can be a function, don't make it a trait.** Traits are for genuine polymorphism where multiple implementations exist or external extensibility is required.

2. **If two types can be one type, merge them.** Don't create separate request/response types for internal vs external use unless the shapes genuinely differ.

3. **The design phase is where the time goes.** Once the design is solid, the actual codebase should be surprisingly small.

4. **Aim for "wait, that's all?"** Someone opening this project should be surprised by how little code achieves the functionality.

### Educational Clarity

**This codebase is designed to teach.** Beyond being a functional library, agent-rs serves as a reference implementation for anyone who wants to understand how to build LLM-powered agents.

Every function, trait, and design choice is documented to explain:
- **What** it does
- **Why** it exists (what problem does it solve?)
- **What end** it serves (how does it fit into the bigger picture?)

A reader should be able to understand the complete agent architecture by reading the code and its comments, without needing external resources. This means:

- **Thorough doc comments** on all public APIs
- **Inline comments** explaining non-obvious logic
- **"Why not" comments** explaining rejected alternatives
- **Cross-references** linking to related documentation
- **Working examples** that teach, not just demonstrate

See the [Documentation Style Guide](./docs/documentation-style.md) for our commenting conventions.

### Target Metrics

| Crate | Target Lines of Code | Rationale |
|-------|---------------------|-----------|
| `agent-core` | ~400 | One loop, two traits, four providers, one context store |
| `agent-tools` | ~200 | Three tools at ~50 lines each |
| `agent-session` | ~150 | Thin orchestration over async primitives |

**Total: ~750 lines** for a multi-provider, context-managed, tool-executing agent harness with job orchestration.

## Documentation Index

| Document | Description |
|----------|-------------|
| [Architecture Overview](./docs/architecture.md) | High-level system design, component diagrams, and architectural principles |
| [Crate Structure](./docs/crates.md) | Workspace layout, dependency graph, and complete public API surface |
| [Data Flow](./docs/data-flow.md) | End-to-end flow from task submission through execution to result delivery |
| [Context Management](./docs/context-management.md) | SQLite schema, token budgeting, eviction lifecycle, and recall system |
| [Error Handling](./docs/error-handling.md) | Error classification, retry strategies, and failure propagation |
| [Trait Catalog](./docs/traits.md) | Complete reference for all public traits and extension points |
| [Concurrency Model](./docs/concurrency.md) | Task spawning, shared state, cancellation, and parallel execution |
| [Provider Normalization](./docs/providers.md) | Multi-provider LLM support and how to add new providers |
| [Documentation Style](./docs/documentation-style.md) | Comment conventions and educational documentation guidelines |

## What This Is

- **A learning resource.** Study this to understand how agents work.
- **An embeddable library.** Use `agent-core` in your own applications.
- **A multi-provider harness.** Swap between Anthropic, OpenAI, Gemini, Bedrock.
- **A minimal implementation.** Proof that agents don't need thousands of lines.

## What This Is Not

- **Not a framework.** This is a library. You call it; it doesn't call you.
- **Not opinionated about orchestration.** The session layer is optional. Use the core directly if you prefer.
- **Not opinionated about UI.** The binary target uses egui, but the library layers know nothing about display.
- **Not opinionated about persistence.** SQLite is behind a feature flag. Pure in-memory operation is fully supported.

## Quick Start (Conceptual)

```rust
use agent_core::{Agent, AgentBuilder, AnthropicClient, AgentEvent};
use agent_tools::{BashTool, FileTool};

// 1. Create an LLM client
// This handles authentication and API communication with your chosen provider.
let client = AnthropicClient::new(&api_key, "claude-sonnet-4-20250514");

// 2. Build an agent with tools
// Tools are the agent's capabilities—without them, it can only generate text.
let agent = AgentBuilder::new(Box::new(client))
    .system_prompt("You are a helpful coding assistant.")
    .tools(vec![
        Box::new(BashTool::new()),
        Box::new(FileTool::new()),
    ])
    .token_budget(100_000)  // For context management
    .max_turns(50)          // Safety limit
    .build();

// 3. Run and consume events
// The agent yields events as it works—stream them for real-time display.
let mut stream = agent.run("List all Rust files in the current directory");
while let Some(event) = stream.next().await {
    match event? {
        AgentEvent::Text(s) => print!("{s}"),
        AgentEvent::ToolStart { name, .. } => println!("\n[Running {name}...]"),
        AgentEvent::Done { .. } => break,
        _ => {}
    }
}
```

## Crate Relationships

```
┌─────────────────┐
│  agent-session  │  Orchestration: job queue, cancellation, progress tracking
└────────┬────────┘
         │
         ├──────────────────┐
         │                  │
         ▼                  ▼
┌─────────────────┐  ┌─────────────────┐
│   agent-core    │◀─│  agent-tools    │  Built-in tool implementations
│                 │  └─────────────────┘
│  Agent loop     │
│  LLM clients    │
│  Tool trait     │
│  Context mgmt   │
└─────────────────┘
```

The dependency direction is strict: tools and session depend on core, never the reverse. Core has zero knowledge of what tools exist or how jobs are orchestrated.

## For Learners

If you're here to learn about building agents, start with:

1. **[Architecture Overview](./docs/architecture.md)** - Understand the big picture
2. **[Data Flow](./docs/data-flow.md)** - Follow a task from start to finish
3. **[Trait Catalog](./docs/traits.md)** - Learn the extension points
4. **Read the source** - The code is heavily commented for learning

Key concepts to understand:
- The turn-based agent loop (send → receive → execute tools → repeat)
- Tool calling (how models invoke external capabilities)
- Streaming (real-time response processing)
- Context management (handling conversations that exceed token limits)

## For Contributors

If you're contributing to this project:

1. Read the [Documentation Style Guide](./docs/documentation-style.md)
2. Ensure your code is well-commented (what, why, what end)
3. Keep things minimal—add only what's necessary
4. Update docs when you change behavior
