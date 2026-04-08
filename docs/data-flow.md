# Data Flow

This document traces the complete path of data through the agent-rs system, from task submission through the agent loop to result delivery. It covers both the happy path and the context eviction/recall cycle.

## Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Submit    │────▶│  Agent Loop │────▶│   Events    │────▶│   Results   │
│   Task      │     │  (turns)    │     │  (stream)   │     │  (collect)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │   Context   │
                    │  Management │
                    └─────────────┘
```

## Phase 1: Task Submission

### Direct Usage (agent-core only)

When using the core library directly, task submission and execution are synchronous from the caller's perspective:

```rust
// 1. Build the agent
let agent = AgentBuilder::new(Box::new(client))
    .system_prompt("You are helpful.")
    .tools(vec![Box::new(BashTool::new())])
    .build();

// 2. Start execution - returns a Stream
let stream = agent.run("List files in current directory");

// 3. Consume events as they arrive
pin_mut!(stream);
while let Some(event) = stream.next().await {
    // Process event
}
```

**Data at this stage:**
- Input: `task: &str` - the user's request
- Output: `impl Stream<Item = Result<AgentEvent, AgentError>>`

### Session Layer Usage (agent-session)

The session layer adds background execution:

```
submit(agent, task)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  Session::submit()                                            │
│                                                               │
│  1. Generate JobId (UUID v4)                                  │
│                                                               │
│  2. Create job state:                                         │
│     JobState::Running {                                       │
│         events: Vec::new(),                                   │
│         cancel_token: CancellationToken::new(),               │
│         handle: None,  // filled below                        │
│     }                                                         │
│                                                               │
│  3. Insert into DashMap<JobId, JobState>                      │
│                                                               │
│  4. Spawn tokio task:                                         │
│     tokio::spawn(run_job(agent, task, job_id, state, tx))     │
│                                                               │
│  5. Store JoinHandle in job state                             │
│                                                               │
│  6. Return JobId immediately                                  │
└───────────────────────────────────────────────────────────────┘
        │
        ▼
    JobId returned to caller (task runs in background)
```

**Data at this stage:**
- Input: `agent: Agent`, `task: String`
- Output: `JobId`
- Side effects: Job registered in DashMap, tokio task spawned

## Phase 2: Agent Loop Execution

The agent loop is the heart of the system. It runs as a state machine with the following structure:

```
                    ┌─────────────────┐
                    │  Initialize     │
                    │  - Add system   │
                    │    prompt       │
                    │  - Add task as  │
                    │    first user   │
                    │    message      │
                    └────────┬────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                      TURN LOOP                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. PRE-TURN CHECKS                                      │  │
│  │     □ Check cancellation token                           │  │
│  │     □ Check turn count < max_turns                       │  │
│  │     □ Check token budget, trigger eviction if needed     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2. BUILD REQUEST                                        │  │
│  │     Request {                                            │  │
│  │         system: system_prompt + context_instructions,    │  │
│  │         messages: message_history,                       │  │
│  │         tools: tool_definitions,                         │  │
│  │         max_tokens: remaining_budget,                    │  │
│  │     }                                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3. LLM STREAMING CALL                                   │  │
│  │     for chunk in client.complete_stream(request):        │  │
│  │         match chunk:                                     │  │
│  │             Text(s) →                                    │  │
│  │                 accumulate to response_text              │  │
│  │                 yield AgentEvent::Text(s)                │  │
│  │             ToolUseStart { id, name } →                  │  │
│  │                 start accumulating tool call             │  │
│  │             ToolUseInput(json_fragment) →                │  │
│  │                 append to current tool call input        │  │
│  │             ToolUseDone →                                │  │
│  │                 parse accumulated input                  │  │
│  │                 yield AgentEvent::ToolStart              │  │
│  │             MessageDone { stop_reason, usage } →         │  │
│  │                 record usage and stop_reason             │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                   │
│              ┌──────────────┴──────────────┐                   │
│              ▼                              ▼                   │
│     stop_reason == EndTurn        stop_reason == ToolUse       │
│              │                              │                   │
│              ▼                              ▼                   │
│  ┌─────────────────────┐      ┌────────────────────────────┐  │
│  │  4a. COMPLETION     │      │  4b. TOOL EXECUTION        │  │
│  │                     │      │                            │  │
│  │  yield Done {       │      │  // Execute all in parallel│  │
│  │    final_text       │      │  let futures = tool_calls  │  │
│  │  }                  │      │      .map(|call| {         │  │
│  │                     │      │          tool.execute(     │  │
│  │  return Ok(())      │      │              call.input    │  │
│  │                     │      │          )                 │  │
│  └─────────────────────┘      │      });                   │  │
│                               │  let results =             │  │
│                               │      join_all(futures);    │  │
│                               │                            │  │
│                               │  for result in results:    │  │
│                               │      yield ToolOutput      │  │
│                               └────────────┬───────────────┘  │
│                                            │                   │
│                                            ▼                   │
│                               ┌────────────────────────────┐  │
│                               │  5. UPDATE HISTORY         │  │
│                               │                            │  │
│                               │  Append assistant message: │  │
│                               │    role: Assistant         │  │
│                               │    content: [              │  │
│                               │      Text(response_text),  │  │
│                               │      ToolUse { ... },      │  │
│                               │      ToolUse { ... },      │  │
│                               │    ]                       │  │
│                               │                            │  │
│                               │  Append user message:      │  │
│                               │    role: User              │  │
│                               │    content: [              │  │
│                               │      ToolResult { ... },   │  │
│                               │      ToolResult { ... },   │  │
│                               │    ]                       │  │
│                               │                            │  │
│                               │  yield TurnComplete        │  │
│                               │                            │  │
│                               │  turn_count += 1           │  │
│                               │  → back to TURN LOOP       │  │
│                               └────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Data Structures During Execution

**Message History (grows each turn):**
```rust
messages: Vec<Message> = [
    // Turn 1
    Message { role: User, content: [Text("List files...")] },
    Message { role: Assistant, content: [Text("I'll use bash"), ToolUse { name: "bash", ... }] },
    Message { role: User, content: [ToolResult { content: "file1.rs\nfile2.rs", ... }] },

    // Turn 2
    Message { role: Assistant, content: [Text("Here are the files: ...")] },
    // ... (if more tool calls, continues)
]
```

**Accumulated Tool Calls (reset each turn):**
```rust
pending_tool_calls: Vec<(String, String, Value)> = [
    ("call_abc123", "bash", {"command": "ls"}),
    ("call_def456", "read_file", {"path": "README.md"}),
]
```

## Phase 3: Context Management

Context management operates within the agent loop but has its own data flow.

### Budget Checking (Before Each Turn)

```
┌─────────────────────────────────────────────────────────────────┐
│  estimate_context_tokens()                                      │
│                                                                  │
│  tokens = 0                                                     │
│  for msg in messages:                                           │
│      for block in msg.content:                                  │
│          tokens += estimate_block_tokens(block)                 │
│                                                                  │
│  // Simple heuristic: 1 token ≈ 4 characters                    │
│  fn estimate_block_tokens(block) -> u32 {                       │
│      match block {                                              │
│          Text(s) => s.len() / 4,                                │
│          ToolUse { input, .. } => input.to_string().len() / 4,  │
│          ToolResult { content, .. } => content.len() / 4,       │
│      }                                                          │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  tokens > budget * 0.8 ?
                     │            │
                    Yes          No
                     │            │
                     ▼            └──▶ Continue to LLM request
            Trigger Eviction
```

### Eviction Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  select_turns_for_eviction()                                    │
│                                                                  │
│  // Keep: system prompt (implicit), last 2 turns (recency)      │
│  // Evict: oldest turns until under budget                      │
│                                                                  │
│  evict_count = 0                                                │
│  while estimate_tokens() > budget * 0.6:  // Target 60%         │
│      evict_count += 1                                           │
│                                                                  │
│  // Minimum: evict at least 2 turns if triggered                │
│  evict_count = max(evict_count, 2)                              │
│                                                                  │
│  return messages[0..evict_count]                                │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  generate_summary()                                             │
│                                                                  │
│  // Use the same LLM client (non-streaming, small request)      │
│  summary_request = Request {                                    │
│      system: None,                                              │
│      messages: [                                                │
│          Message {                                              │
│              role: User,                                        │
│              content: [Text(                                    │
│                  "Summarize this conversation excerpt in 2-3    │
│                   sentences, focusing on key decisions and      │
│                   information:\n\n{turns_content}"              │
│              )]                                                 │
│          }                                                      │
│      ],                                                         │
│      tools: [],                                                 │
│      max_tokens: 200,                                           │
│  };                                                             │
│                                                                  │
│  response = client.complete(summary_request).await?;            │
│  return extract_text(response);                                 │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  persist_and_evict() [if persistence enabled]                   │
│                                                                  │
│  // 1. Store full content in SQLite                             │
│  store.insert_evicted_turns(                                    │
│      session_id,                                                │
│      turn_start,                                                │
│      turn_end,                                                  │
│      summary,                                                   │
│      full_content,  // serialized turns                         │
│  )?;                                                            │
│                                                                  │
│  // 2. Store tool executions separately                         │
│  for turn in evicted_turns:                                     │
│      for tool_use, tool_result in turn.tool_calls():            │
│          store.insert_tool_execution(                           │
│              session_id,                                        │
│              turn_number,                                       │
│              tool_use.name,                                     │
│              tool_use.input,                                    │
│              tool_result.content,                               │
│              tool_result.is_error,                              │
│          )?;                                                    │
│                                                                  │
│  // 3. Replace evicted turns with manifest entry                │
│  messages = [                                                   │
│      Message {                                                  │
│          role: User,                                            │
│          content: [Text(format!(                                │
│              "[Evicted turns {start}-{end}: {summary}. "        │
│              "Use the recall tool to retrieve details.]"        │
│          ))]                                                    │
│      },                                                         │
│      ...remaining_messages                                      │
│  ];                                                             │
│                                                                  │
│  // 4. Emit event                                               │
│  yield ContextEvicted { summary, turns_evicted: start..end };   │
└─────────────────────────────────────────────────────────────────┘
```

### Recall Flow (Model Invokes Tool)

```
Model decides to call: recall(query="auth implementation")
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  RecallTool::execute({ "query": "auth implementation" })        │
│                                                                  │
│  // 1. Parse input                                              │
│  query = input["query"].as_str()?;                              │
│  tool_filter = input["tool_filter"].as_str();  // optional      │
│                                                                  │
│  // 2. Build FTS5 query                                         │
│  fts_query = sanitize_for_fts5(query);                          │
│                                                                  │
│  // 3. Execute search                                           │
│  if let Some(tool) = tool_filter {                              │
│      results = store.query(                                     │
│          "SELECT turn, input, output FROM tool_executions       │
│           WHERE tool_name = ?1                                  │
│           AND tool_executions_fts MATCH ?2                      │
│           ORDER BY rank LIMIT 5",                               │
│          [tool, fts_query]                                      │
│      )?;                                                        │
│  } else {                                                       │
│      results = store.query(                                     │
│          "SELECT turn_start, turn_end, summary, full_content    │
│           FROM evicted_turns                                    │
│           WHERE evicted_turns_fts MATCH ?1                      │
│           ORDER BY rank LIMIT 5",                               │
│          [fts_query]                                            │
│      )?;                                                        │
│  }                                                              │
│                                                                  │
│  // 4. Format results                                           │
│  output = format_recall_results(results);                       │
│  return ToolOutput::success(output);                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
Result returned to agent loop → sent to model as ToolResult
```

## Phase 4: Result Collection

### Direct Usage

For direct usage, the caller consumes the stream to completion:

```rust
let mut final_text = String::new();
let mut all_events = Vec::new();

while let Some(event) = stream.next().await {
    let event = event?;  // Handle AgentError
    all_events.push(event.clone());

    match event {
        AgentEvent::Done { final_text: text } => {
            final_text = text;
            break;
        }
        _ => {}
    }
}

// final_text and all_events now available
```

### Session Layer

The session layer collects events internally and delivers them as `JobResult`:

```
┌─────────────────────────────────────────────────────────────────┐
│  run_job() [inside spawned task]                                │
│                                                                  │
│  let mut events = Vec::new();                                   │
│  let mut final_result = Ok(String::new());                      │
│                                                                  │
│  let stream = agent.run(&task);                                 │
│  pin_mut!(stream);                                              │
│                                                                  │
│  while let Some(result) = stream.next().await {                 │
│      match result {                                             │
│          Ok(event) => {                                         │
│              // Update shared state for peek()                  │
│              if let Some(mut state) = jobs.get_mut(&job_id) {   │
│                  if let JobState::Running { events: e, .. } = &mut *state { │
│                      e.push(event.clone());                     │
│                  }                                              │
│              }                                                  │
│              events.push(event.clone());                        │
│                                                                  │
│              if let AgentEvent::Done { final_text } = event {   │
│                  final_result = Ok(final_text);                 │
│              }                                                  │
│          }                                                      │
│          Err(e) => {                                            │
│              final_result = Err(e);                             │
│              break;                                             │
│          }                                                      │
│      }                                                          │
│  }                                                              │
│                                                                  │
│  // Remove from active jobs                                     │
│  jobs.remove(&job_id);                                          │
│                                                                  │
│  // Send to completed channel                                   │
│  let _ = completed_tx.send(JobResult {                          │
│      id: job_id,                                                │
│      outcome: final_result,                                     │
│      events,                                                    │
│  }).await;                                                      │
└─────────────────────────────────────────────────────────────────┘
```

Consumer retrieves results:

```rust
// Blocking wait for next result
let result = session.next_completed().await;

// Or non-blocking check
if let Some(result) = session.try_next_completed() {
    // Process result
}

// Or with timeout
match tokio::time::timeout(Duration::from_secs(30), session.next_completed()).await {
    Ok(Some(result)) => { /* process */ }
    Ok(None) => { /* session dropped */ }
    Err(_) => { /* timeout */ }
}
```

## Data Flow Summary

| Phase | Input | Output | Side Effects |
|-------|-------|--------|--------------|
| **Submit** | Agent + task | JobId | Task spawned, state registered |
| **Turn Start** | Message history | - | Token budget checked, possible eviction |
| **LLM Call** | Request | StreamChunks | HTTP request to provider |
| **Tool Execution** | Tool calls | Tool outputs | Arbitrary (file I/O, subprocess, etc.) |
| **History Update** | Response + tool outputs | - | Messages appended to history |
| **Eviction** | Oldest turns | Summary | SQLite writes (if persistence) |
| **Recall** | Query | Matching snippets | SQLite reads |
| **Completion** | Final response | Done event | Job removed from active set |
| **Collection** | JobId | JobResult | Result delivered to caller |

## Invariants

The following invariants are maintained throughout execution:

1. **Message alternation**: Messages strictly alternate User → Assistant → User → ...
2. **Tool result correspondence**: Every ToolUse in an assistant message has exactly one corresponding ToolResult in the following user message
3. **Event ordering**: Events are emitted in causal order (ToolStart before ToolOutput, TurnComplete after all tool outputs)
4. **Single consumption**: Agents are consumed by `run()` — no reuse
5. **Cancellation safety**: Cancellation is checked between turns, never mid-operation
6. **Eviction minimum**: At least the last 2 turns are always kept in context
