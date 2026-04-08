# Concurrency Model

This document describes the concurrent execution model of agent-rs: what gets spawned, what state is shared, how cancellation works, and how parallel tool execution is implemented.

## Overview

agent-rs uses Tokio for async runtime. The concurrency model is deliberately simple:

- **No thread pools.** Everything runs on the Tokio runtime.
- **No complex synchronization.** DashMap for shared state, channels for communication.
- **No custom executors.** Standard tokio::spawn for background tasks.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Tokio Runtime                                     │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   Job Task 1    │  │   Job Task 2    │  │   Job Task N    │  ...        │
│  │  (Agent::run)   │  │  (Agent::run)   │  │  (Agent::run)   │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           │    ┌───────────────┴───────────────┐    │                       │
│           │    │                               │    │                       │
│           ▼    ▼                               ▼    ▼                       │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                      Shared State                                │       │
│  │                                                                  │       │
│  │  DashMap<JobId, JobState>      mpsc::channel<JobResult>         │       │
│  │  (concurrent read/write)       (completed job delivery)          │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## What Gets Spawned

### Spawn Points

| Location | What's Spawned | Lifetime | Purpose |
|----------|---------------|----------|---------|
| `Session::submit()` | Job execution task | Until agent completes | Run agent in background |
| Agent loop (implicit) | Nothing | N/A | Streaming happens inline |
| Tool execution | Nothing | N/A | `join_all` in current task |

### Job Execution Task

When you call `Session::submit()`, a new Tokio task is spawned:

```rust
impl Session {
    pub fn submit(&self, agent: Agent, task: String) -> JobId {
        let job_id = JobId::new();
        let cancel_token = CancellationToken::new();

        // Create initial job state
        let state = JobState::Running {
            events: Vec::new(),
            cancel_token: cancel_token.clone(),
            handle: None,  // Will be set after spawn
        };

        // Insert into shared map
        self.inner.jobs.insert(job_id, state);

        // Spawn the job task
        let jobs = self.inner.jobs.clone();
        let completed_tx = self.inner.completed_tx.clone();

        let handle = tokio::spawn(async move {
            run_job(agent, task, job_id, jobs, completed_tx, cancel_token).await
        });

        // Store the handle for cancellation
        if let Some(mut entry) = self.inner.jobs.get_mut(&job_id) {
            if let JobState::Running { handle: h, .. } = &mut *entry {
                *h = Some(handle);
            }
        }

        job_id
    }
}
```

### Why Not Spawn Tools?

Tools execute via `futures::future::join_all`, not `tokio::spawn`. This is intentional:

1. **Structured concurrency.** Tools complete before the turn ends. No dangling tasks.
2. **Error propagation.** Errors bubble up naturally without channel coordination.
3. **Simpler mental model.** The turn is atomic: request → response → tools → next turn.

```rust
// This is what we do
async fn execute_tools(tools: &[Box<dyn Tool>], calls: Vec<ToolCall>) -> Vec<ToolOutput> {
    let futures = calls.into_iter().map(|call| async {
        let tool = tools.iter().find(|t| t.name() == call.name);
        match tool {
            Some(t) => t.execute(call.input).await,
            None => ToolOutput::error("Unknown tool"),
        }
    });

    futures::future::join_all(futures).await
}

// NOT this (spawning each tool)
async fn execute_tools_spawned(tools: &[Box<dyn Tool>], calls: Vec<ToolCall>) -> Vec<ToolOutput> {
    let handles = calls.into_iter().map(|call| {
        tokio::spawn(async move { /* ... */ })  // Don't do this
    });

    // Problems:
    // - Tools are 'static, can't borrow from registry
    // - Need to handle JoinError
    // - Cancellation is more complex
}
```

## Shared State

### Session State Structure

```rust
/// Internal session state, wrapped in Arc for sharing.
struct SessionInner {
    /// Active jobs indexed by ID.
    ///
    /// Uses DashMap for lock-free concurrent access. Multiple tasks
    /// can read/write different entries simultaneously.
    jobs: DashMap<JobId, JobState>,

    /// Channel sender for completed jobs.
    ///
    /// Each job task holds a clone. When a job finishes, it sends
    /// the result here.
    completed_tx: mpsc::Sender<JobResult>,

    /// Channel receiver for completed jobs.
    ///
    /// Protected by Mutex because only one consumer can pull at a time.
    /// This is the only Mutex in the system.
    completed_rx: Mutex<mpsc::Receiver<JobResult>>,
}

/// State of an active job.
enum JobState {
    Running {
        /// Events emitted so far (for peek).
        events: Vec<AgentEvent>,

        /// Token to signal cancellation.
        cancel_token: CancellationToken,

        /// Task handle for abort.
        handle: Option<JoinHandle<()>>,
    },
    // Completed jobs are removed from the map and sent to the channel
}
```

### Why DashMap?

DashMap provides concurrent read/write access without global locking:

```rust
// These can happen concurrently from different tasks:

// Task A: peek at job 1
if let Some(entry) = jobs.get(&job_id_1) {
    let events = entry.events.clone();
}

// Task B: update job 2's events
if let Some(mut entry) = jobs.get_mut(&job_id_2) {
    entry.events.push(event);
}

// Task C: insert new job
jobs.insert(job_id_3, state);

// Task D: remove completed job
jobs.remove(&job_id_4);
```

Each shard in DashMap has its own lock. Operations on different keys don't contend.

### What's NOT Shared

| Component | Why Not Shared |
|-----------|----------------|
| Agent | Consumed by run(). One agent, one execution. |
| Message history | Owned by agent. Not visible outside. |
| LLM client | Cloned per-job or used via Arc internally. |
| Tools | Passed to agent at construction. Immutable. |

## Cancellation Flow

### Components

```
┌──────────────────┐
│ CancellationToken│ ←── From tokio_util::sync
│                  │
│ .cancel()        │ ←── Signal cancellation
│ .cancelled()     │ ←── Future that completes when cancelled
│ .is_cancelled()  │ ←── Sync check
└──────────────────┘
```

### Flow Diagram

```
Session::cancel(job_id)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Look up JobState in DashMap                                 │
│                                                                  │
│  if let Some(mut entry) = self.jobs.get_mut(&job_id) {          │
│      if let JobState::Running { cancel_token, handle, .. } = &mut *entry { │
│                                                                  │
│          // Signal the cancellation token                        │
│          cancel_token.cancel();                                  │
│                                                                  │
│          // Also abort the task (belt and suspenders)            │
│          if let Some(h) = handle.take() {                        │
│              h.abort();                                          │
│          }                                                       │
│      }                                                           │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
         │
         │ Token is now cancelled
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Inside agent loop (next turn start):                           │
│                                                                  │
│  loop {                                                         │
│      // Check cancellation at the start of each turn            │
│      if self.cancel_token.is_cancelled() {                      │
│          return Err(AgentError::Cancelled);                     │
│      }                                                          │
│                                                                  │
│      // Or use select! for streaming                            │
│      tokio::select! {                                           │
│          _ = self.cancel_token.cancelled() => {                 │
│              return Err(AgentError::Cancelled);                 │
│          }                                                       │
│          chunk = stream.next() => {                             │
│              // Process chunk                                    │
│          }                                                       │
│      }                                                          │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Agent returns AgentError::Cancelled                            │
│                                                                  │
│  Job task completes:                                            │
│  - Removes itself from DashMap                                  │
│  - Sends JobResult with Err(Cancelled) to channel               │
└─────────────────────────────────────────────────────────────────┘
```

### Why Both Token AND Abort?

```rust
// We do BOTH of these:
cancel_token.cancel();  // Cooperative cancellation
handle.abort();         // Forced task termination

// Why?

// 1. The token allows graceful shutdown between turns.
//    The agent can clean up, emit a final event, etc.

// 2. The abort handles the case where the agent is stuck
//    (e.g., waiting on a slow LLM response that never comes).

// 3. Abort alone is too aggressive—it can leave things inconsistent.
//    Token alone is too slow—might wait for a long operation.

// Together, they provide graceful-when-possible, forceful-when-needed.
```

### Cancellation Points

| Location | How Checked | What Happens |
|----------|-------------|--------------|
| Turn start | `is_cancelled()` | Return immediately |
| During streaming | `select!` with `cancelled()` | Interrupt stream |
| Between tool executions | `is_cancelled()` | Skip remaining tools |

## Parallel Tool Execution

### When Tools Run in Parallel

The model may issue multiple tool calls in a single response:

```json
{
  "content": [
    { "type": "text", "text": "I'll read both files..." },
    { "type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a.txt"} },
    { "type": "tool_use", "id": "call_2", "name": "read_file", "input": {"path": "b.txt"} }
  ]
}
```

These tools execute **concurrently**, not sequentially:

```rust
async fn execute_tool_calls(
    tools: &[Box<dyn Tool>],
    calls: Vec<ToolCall>,
) -> Vec<(String, ToolOutput)> {
    // Create futures for all tool calls
    let futures = calls.into_iter().map(|call| {
        let tool = tools.iter().find(|t| t.name() == call.name).cloned();
        async move {
            let output = match tool {
                Some(t) => {
                    // Catch panics to prevent one tool from crashing others
                    match AssertUnwindSafe(t.execute(call.input))
                        .catch_unwind()
                        .await
                    {
                        Ok(out) => out,
                        Err(_) => ToolOutput::error("Tool panicked"),
                    }
                }
                None => ToolOutput::error(format!("Unknown tool: {}", call.name)),
            };
            (call.id, output)
        }
    });

    // Execute ALL futures concurrently
    futures::future::join_all(futures).await

    // Note: join_all, not join_all with spawn
    // All tools run on the current task's executor
}
```

### Execution Timeline

```
Sequential (NOT what we do):
┌──────────────────────────────────────────────────────────────┐
│ Tool A ████████████                                          │
│ Tool B              ████████████                             │
│ Tool C                          ████████████                 │
│                                                              │
│ Total time: A + B + C                                        │
└──────────────────────────────────────────────────────────────┘

Parallel (what we do):
┌──────────────────────────────────────────────────────────────┐
│ Tool A ████████████                                          │
│ Tool B ████████████                                          │
│ Tool C ████████████                                          │
│                                                              │
│ Total time: max(A, B, C)                                     │
└──────────────────────────────────────────────────────────────┘
```

### Ordering Guarantees

**What's guaranteed:**
- All tool calls from one response complete before the next turn
- Results are returned in the same order as calls (join_all preserves order)

**What's NOT guaranteed:**
- Order of execution (tools may start/finish in any order)
- Interleaving (a faster tool might complete before a slower one starts)

### Implications for Tool Authors

If your tool has side effects that depend on order, you need to handle this yourself:

```rust
// PROBLEM: Two file writes to the same file
// Tool calls: write_file(path="x", content="A"), write_file(path="x", content="B")
// Result: Race condition! Could be A or B depending on timing.

// SOLUTION 1: Document it
fn description(&self) -> &str {
    "Write content to a file. WARNING: Concurrent writes to the same \
     file have undefined behavior. Issue one write at a time."
}

// SOLUTION 2: Lock in the tool
struct FileTool {
    locks: DashMap<PathBuf, Mutex<()>>,
}

async fn execute(&self, input: Value) -> ToolOutput {
    let path = parse_path(&input)?;

    // Acquire lock for this specific file
    let lock = self.locks.entry(path.clone()).or_default();
    let _guard = lock.lock().await;

    // Now safe to write
    std::fs::write(&path, content)?;
    ToolOutput::success("Written")
}
```

## Memory Model

### What Lives Where

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Stack / Task-Local                              │
│                                                                              │
│  - Agent struct (during execution)                                          │
│  - Message history Vec                                                       │
│  - Current turn's tool calls                                                │
│  - Stream state                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              Arc-Shared (Heap)                              │
│                                                                              │
│  - SessionInner (jobs map, channels)                                        │
│  - LlmClient (if shared across agents)                                      │
│  - ContextStore (if shared)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              Box<dyn Trait> (Heap)                          │
│                                                                              │
│  - Tools in registry                                                        │
│  - LlmClient in agent                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Cloning Strategy

| Type | Clone Behavior | Why |
|------|---------------|-----|
| `Session` | Cheap (Arc clone) | Multiple handles to same session |
| `JobId` | Copy | Just a UUID |
| `AgentEvent` | Deep clone | Events are small, need copies for peek |
| `Agent` | Not Clone | Consumed by run() |
| `Box<dyn Tool>` | Not Clone | Tools may have state |

## Performance Characteristics

### Scalability

| Operation | Complexity | Bottleneck |
|-----------|-----------|------------|
| Job submit | O(1) | DashMap insert |
| Job peek | O(n events) | Clone event vec |
| Job cancel | O(1) | DashMap lookup |
| next_completed | O(1) | Channel recv |
| Parallel tools | O(max tool time) | Slowest tool |

### Contention Points

1. **DashMap shards**: Different jobs rarely contend (different keys)
2. **completed_rx Mutex**: Only one consumer at a time, but recv is fast
3. **Tool execution**: No shared state in the framework; tool-dependent

### Typical Throughput

With a well-configured Tokio runtime:
- Hundreds of concurrent agents (limited by LLM rate limits, not framework)
- Thousands of tool executions per second (limited by tool I/O)
- Negligible framework overhead (~1% of total time)
