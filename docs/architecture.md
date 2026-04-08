# Architecture Overview

This document provides a high-level view of the agent-rs system architecture, including component diagrams and the key architectural decisions that shape the design.

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 agent-session                                    │
│                                                                                  │
│  ┌──────────────────┐  ┌─────────────────────┐  ┌────────────────────────────┐  │
│  │    JobQueue      │  │      DashMap        │  │    CompletedChannel        │  │
│  │                  │  │    <JobId, State>   │  │       (mpsc)               │  │
│  │  submit(agent,   │──│                     │──│                            │  │
│  │         task)    │  │  Running:           │  │  next_completed()          │  │
│  │  cancel(id)      │  │   - events[]        │  │    → JobResult             │  │
│  │  peek(id)        │  │   - cancel_token    │  │                            │  │
│  │  active_jobs()   │  │   - join_handle     │  │                            │  │
│  └──────────────────┘  └─────────────────────┘  └────────────────────────────┘  │
│           │                                                                      │
│           │ spawns tokio tasks                                                   │
└───────────┼──────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  agent-core                                      │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                              Agent Loop                                     │ │
│  │                                                                             │ │
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐              │ │
│  │  │   Message     │    │   LLM Client  │    │     Tool      │              │ │
│  │  │   History     │───▶│   (streaming) │───▶│   Executor    │              │ │
│  │  │               │◀───│               │◀───│   (parallel)  │              │ │
│  │  └───────────────┘    └───────────────┘    └───────────────┘              │ │
│  │         │                    │                    │                        │ │
│  │         │                    ▼                    │                        │ │
│  │         │            ┌───────────────┐            │                        │ │
│  │         │            │    Event      │◀───────────┘                        │ │
│  │         │            │    Stream     │                                     │ │
│  │         │            │  (async iter) │                                     │ │
│  │         │            └───────────────┘                                     │ │
│  │         │                    │                                             │ │
│  │         ▼                    │ yields AgentEvent                           │ │
│  │  ┌───────────────┐           │                                             │ │
│  │  │    Context    │           │                                             │ │
│  │  │    Manager    │           │                                             │ │
│  │  │               │           │                                             │ │
│  │  │ budget check  │           │                                             │ │
│  │  │ eviction      │           │                                             │ │
│  │  │ manifest      │           │                                             │ │
│  │  └───────┬───────┘           │                                             │ │
│  │          │                   │                                             │ │
│  └──────────┼───────────────────┼─────────────────────────────────────────────┘ │
│             │                   │                                               │
│             ▼                   │                                               │
│  ┌───────────────────┐          │                                               │
│  │   Context Store   │          │     ┌─────────────────────────────────────┐   │
│  │     (SQLite)      │          │     │          LLM Providers              │   │
│  │                   │          │     │                                     │   │
│  │  evicted_turns    │          │     │  ┌──────────┐  ┌──────────┐        │   │
│  │  tool_executions  │          │     │  │Anthropic │  │  OpenAI  │        │   │
│  │  FTS5 indexes     │          │     │  │  Client  │  │  Client  │        │   │
│  │                   │          │     │  └──────────┘  └──────────┘        │   │
│  │  [feature-gated]  │          │     │  ┌──────────┐  ┌──────────┐        │   │
│  └───────────────────┘          │     │  │  Gemini  │  │ Bedrock  │        │   │
│                                 │     │  │  Client  │  │  Client  │        │   │
│                                 │     │  └──────────┘  └──────────┘        │   │
│                                 │     └─────────────────────────────────────┘   │
│                                 │                                               │
│                                 │     ┌─────────────────────────────────────┐   │
│                                 │     │          Tool Registry              │   │
│                                 │     │       Vec<Box<dyn Tool>>            │   │
│                                 │     └─────────────────────────────────────┘   │
└─────────────────────────────────┼───────────────────────────────────────────────┘
                                  │
            ┌─────────────────────┘
            │ implements Tool trait
            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 agent-tools                                      │
│                                                                                  │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐            │
│  │     BashTool      │  │     FileTool      │  │    RecallTool     │            │
│  │                   │  │                   │  │                   │            │
│  │  - sandboxing     │  │  - read files     │  │  - FTS5 search    │            │
│  │  - timeouts       │  │  - write files    │  │  - tool filtering │            │
│  │  - working dir    │  │  - path jailing   │  │  - turn retrieval │            │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Architectural Principles

### 1. Dependency Inversion

The agent loop is decoupled from both LLM providers and tools through trait abstractions:

```
Agent Loop
    │
    ├── depends on ──▶ LlmClient trait (not AnthropicClient, OpenAiClient, etc.)
    │
    └── depends on ──▶ Tool trait (not BashTool, FileTool, etc.)
```

This allows:
- Swapping providers without touching the agent loop
- Adding tools without modifying core
- Testing with mock implementations

### 2. Configuration Over Convention

All behavior is explicit at construction time. There are no:
- Global configuration
- Environment variable magic
- Default providers or tools
- Implicit persistence

```rust
// Everything is explicit
AgentBuilder::new(client)       // You choose the provider
    .tools(tools)               // You choose the tools
    .token_budget(100_000)      // You choose the budget
    .context_store(store)       // You opt into persistence
    .build()
```

### 3. Errors as Data

The agent loop treats tool failures as information, not exceptions. When a tool returns an error:

1. The error is wrapped in a `ToolResult` with `is_error: true`
2. The result is sent back to the model like any other tool output
3. The model sees the error and can self-correct

This keeps the agent loop simple (no complex error recovery logic) and leverages the LLM's ability to adapt.

### 4. Streaming as Primary Interface

The agent exposes an async `Stream<Item = AgentEvent>` rather than a blocking `run() -> String`. This enables:

- Real-time progress display in UIs
- Incremental processing of results
- Graceful cancellation at any point
- Memory-efficient handling of long conversations

### 5. Feature-Gated Dependencies

Heavy dependencies like SQLite are behind feature flags:

```toml
[features]
default = []
persistence = ["rusqlite"]
```

Without `persistence`, the context manager falls back to pure in-memory summarization. This keeps the dependency tree minimal for consumers who don't need persistence.

## Key Design Decisions

### Decision 1: Turn-Based Loop with Parallel Tool Execution

The agent loop follows a strict turn-based pattern:

```
Turn N:
  1. Send messages to LLM
  2. Receive response (streaming)
  3. If response contains tool calls:
     a. Execute ALL tools in parallel (tokio::join_all)
     b. Collect ALL results
     c. Append assistant message + tool results to history
     d. Go to Turn N+1
  4. If response is end_turn:
     a. We're done
```

**Why parallel tool execution?** Models like Claude often issue multiple independent tool calls in a single response (e.g., reading several files). Sequential execution would waste time. The model has already decided these calls are independent by issuing them together.

**Why not streaming tool execution?** Executing tools as they stream in (before the full response) would complicate cancellation, error handling, and the mental model. The small latency cost of waiting for the full response is worth the simplicity.

### Decision 2: Provider-Agnostic Types with Provider-Specific Translation

The agent loop operates on normalized types (`Request`, `Response`, `ContentBlock`, etc.). Each provider implements translation to/from its native format.

```
Agent Loop                    Provider Implementation
    │                              │
    ├── Request ──────────────────▶├── translate to native JSON
    │                              │
    │                              ├── HTTP request
    │                              │
    │                              ├── HTTP response
    │                              │
    ◀── Response ◀─────────────────├── translate from native JSON
```

**Why not use provider SDKs?** Most provider SDKs are heavyweight, have their own async runtimes, and don't expose streaming well. Raw HTTP with `reqwest` gives us full control and keeps dependencies minimal.

### Decision 3: Hybrid Context Management

When token usage approaches the budget:

1. **Summarize**: Generate a short LLM summary of oldest turns
2. **Persist** (if enabled): Store full content in SQLite with FTS5 indexing
3. **Evict**: Replace turns with a manifest entry (the summary)
4. **Recall**: Provide a tool for the model to search and retrieve evicted content

**Why not just truncate?** Pure truncation loses information permanently. The model might need context from early in the conversation. Summarization preserves the gist; persistence + recall preserves the details.

**Why not keep everything?** Token limits exist. Costs scale with context length. Performance degrades with very long contexts. Active management is necessary.

### Decision 4: Session Layer as Optional Orchestration

The `agent-session` crate wraps agents in a job queue, but it's entirely optional. You can use `agent-core` directly for simpler use cases:

```rust
// Direct use (no session layer)
let agent = AgentBuilder::new(client).tools(tools).build();
let events: Vec<_> = agent.run("task").collect().await;

// With session layer (background execution, multiple concurrent jobs)
let session = Session::new();
let job_id = session.submit(agent, "task".into());
// ... do other work ...
let result = session.next_completed().await;
```

**Why separate crates?** Not everyone needs job orchestration. Keeping it separate means `agent-core` has fewer dependencies and a smaller API surface.

### Decision 5: No Internal Retries for Tool Execution

When a tool fails (returns `is_error: true`), the agent loop does **not** retry. It feeds the error to the model and lets the model decide what to do.

**Why?** The model has context we don't. Maybe the command was wrong and needs different arguments. Maybe the file doesn't exist and an alternative approach is needed. The model can make intelligent decisions about recovery; a retry loop cannot.

**Exception:** LLM API calls **are** retried for transient errors (rate limits, overload, network issues). These are infrastructure failures, not semantic failures.

## Component Responsibilities

| Component | Responsibility | Does NOT Do |
|-----------|---------------|-------------|
| **Agent** | Run the loop, coordinate LLM calls and tool execution, stream events | Persistence, job management, UI |
| **LlmClient** | Translate requests/responses, handle auth, manage HTTP | Retry logic (handled by caller), tool execution |
| **Tool** | Execute one capability, return result or error | Decide what to do with errors, manage state |
| **ContextManager** | Track token usage, trigger eviction, maintain manifest | LLM calls (delegates to Agent for summarization) |
| **ContextStore** | SQLite operations, FTS5 queries | Token counting, eviction decisions |
| **Session** | Job lifecycle, cancellation, progress tracking | Agent execution (delegates to spawned tasks) |

## Extension Points

The architecture has exactly two extension points, both via traits:

1. **`LlmClient`**: Implement to add a new LLM provider
2. **`Tool`**: Implement to add a new agent capability

Everything else is concrete. This is intentional—more extension points mean more abstraction, more indirection, and more code. Two traits cover the genuine needs for extensibility.
