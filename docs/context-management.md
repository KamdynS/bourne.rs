# Context Management

This document covers the SQLite schema, token budgeting, eviction lifecycle, and recall system that enable agents to handle conversations that exceed the token limit.

## Overview

LLMs have finite context windows. As conversations grow, token usage approaches the limit, causing requests to fail or responses to be truncated. Context management addresses this through a hybrid approach:

1. **Token Tracking**: Estimate usage before each request
2. **Eviction**: When approaching the limit, summarize and remove older turns
3. **Persistence**: Store evicted content in SQLite for later retrieval
4. **Recall**: Provide a tool for the model to search and retrieve evicted content

This allows arbitrarily long conversations while maintaining coherence and preserving important information.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Loop                               │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                  ContextManager                             │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │ │
│  │  │   Token     │  │  Eviction   │  │     Manifest        │ │ │
│  │  │   Budget    │  │  Logic      │  │  (summaries in      │ │ │
│  │  │  Tracking   │  │             │  │   message history)  │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘ │ │
│  │         │                │                    │             │ │
│  │         └────────────────┼────────────────────┘             │ │
│  │                          │                                  │ │
│  └──────────────────────────┼──────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    ContextStore                             │ │
│  │                    (SQLite + FTS5)                          │ │
│  │                                                             │ │
│  │  ┌─────────────────┐  ┌─────────────────────────────────┐  │ │
│  │  │ evicted_turns   │  │     tool_executions             │  │ │
│  │  │ (full content)  │  │  (input/output by tool name)    │  │ │
│  │  └─────────────────┘  └─────────────────────────────────┘  │ │
│  │                                                             │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │              FTS5 Virtual Tables                    │   │ │
│  │  │         (full-text search indexes)                  │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  │                                                             │ │
│  │                   [feature-gated: "persistence"]            │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## SQLite Schema

### Database Initialization

```sql
-- Enable WAL mode for concurrent access
-- WAL allows readers and writers simultaneously
PRAGMA journal_mode = WAL;

-- Synchronous NORMAL is safe with WAL and much faster than FULL
PRAGMA synchronous = NORMAL;

-- Increase page size for better FTS5 performance
PRAGMA page_size = 4096;
```

### Core Tables

```sql
-- =============================================================================
-- EVICTED TURNS
-- =============================================================================
-- Stores complete conversation turns that have been evicted from active context.
-- Each row represents a contiguous range of turns.

CREATE TABLE evicted_turns (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Session identifier (UUID as text)
    -- Allows multiple agent sessions to share one database
    session_id TEXT NOT NULL,

    -- Turn range (inclusive)
    -- e.g., turn_start=3, turn_end=7 means turns 3,4,5,6,7
    turn_start INTEGER NOT NULL,
    turn_end INTEGER NOT NULL,

    -- LLM-generated summary of the evicted content
    -- Used as a "table of contents" entry in the manifest
    summary TEXT NOT NULL,

    -- Complete serialized content of evicted turns
    -- JSON array of Message objects
    full_content TEXT NOT NULL,

    -- When this eviction occurred
    evicted_at INTEGER NOT NULL DEFAULT (unixepoch()),

    -- Indexes for common queries
    UNIQUE(session_id, turn_start, turn_end)
);

-- Index for session-specific queries
CREATE INDEX idx_evicted_turns_session ON evicted_turns(session_id, evicted_at);

-- =============================================================================
-- EVICTED TURNS FULL-TEXT SEARCH
-- =============================================================================
-- FTS5 virtual table for searching evicted content.
-- Uses the full_content column which contains all turn text.

CREATE VIRTUAL TABLE evicted_turns_fts USING fts5(
    -- Searchable content
    full_content,

    -- Reference to source table
    content='evicted_turns',
    content_rowid='id',

    -- Use porter stemmer for English text
    tokenize='porter unicode61'
);

-- Triggers to keep FTS index synchronized with source table

CREATE TRIGGER evicted_turns_ai AFTER INSERT ON evicted_turns BEGIN
    INSERT INTO evicted_turns_fts(rowid, full_content)
    VALUES (new.id, new.full_content);
END;

CREATE TRIGGER evicted_turns_ad AFTER DELETE ON evicted_turns BEGIN
    INSERT INTO evicted_turns_fts(evicted_turns_fts, rowid, full_content)
    VALUES ('delete', old.id, old.full_content);
END;

CREATE TRIGGER evicted_turns_au AFTER UPDATE ON evicted_turns BEGIN
    INSERT INTO evicted_turns_fts(evicted_turns_fts, rowid, full_content)
    VALUES ('delete', old.id, old.full_content);
    INSERT INTO evicted_turns_fts(rowid, full_content)
    VALUES (new.id, new.full_content);
END;

-- =============================================================================
-- TOOL EXECUTIONS
-- =============================================================================
-- Stores individual tool executions for filtered recall.
-- Allows searching "all bash commands" or "all file reads" specifically.

CREATE TABLE tool_executions (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Session identifier
    session_id TEXT NOT NULL,

    -- Which turn this execution occurred in
    turn INTEGER NOT NULL,

    -- Tool name (e.g., "bash", "read_file")
    tool_name TEXT NOT NULL,

    -- Input arguments (JSON)
    input TEXT NOT NULL,

    -- Output content
    output TEXT NOT NULL,

    -- Whether this was an error
    is_error INTEGER NOT NULL DEFAULT 0,

    -- When this execution occurred
    executed_at INTEGER NOT NULL DEFAULT (unixepoch())
);

-- Index for tool-filtered queries
CREATE INDEX idx_tool_executions_tool ON tool_executions(session_id, tool_name);

-- Index for turn-based retrieval
CREATE INDEX idx_tool_executions_turn ON tool_executions(session_id, turn);

-- =============================================================================
-- TOOL EXECUTIONS FULL-TEXT SEARCH
-- =============================================================================
-- FTS5 for searching tool inputs and outputs.

CREATE VIRTUAL TABLE tool_executions_fts USING fts5(
    input,
    output,
    content='tool_executions',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER tool_executions_ai AFTER INSERT ON tool_executions BEGIN
    INSERT INTO tool_executions_fts(rowid, input, output)
    VALUES (new.id, new.input, new.output);
END;

CREATE TRIGGER tool_executions_ad AFTER DELETE ON tool_executions BEGIN
    INSERT INTO tool_executions_fts(tool_executions_fts, rowid, input, output)
    VALUES ('delete', old.id, old.input, old.output);
END;

CREATE TRIGGER tool_executions_au AFTER UPDATE ON tool_executions BEGIN
    INSERT INTO tool_executions_fts(tool_executions_fts, rowid, input, output)
    VALUES ('delete', old.id, old.input, old.output);
    INSERT INTO tool_executions_fts(rowid, input, output)
    VALUES (new.id, new.input, new.output);
END;
```

### Schema Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        evicted_turns                            │
├─────────────────────────────────────────────────────────────────┤
│ id           INTEGER PRIMARY KEY                                │
│ session_id   TEXT NOT NULL                                      │
│ turn_start   INTEGER NOT NULL                                   │
│ turn_end     INTEGER NOT NULL                                   │
│ summary      TEXT NOT NULL                                      │
│ full_content TEXT NOT NULL                ──────┐               │
│ evicted_at   INTEGER NOT NULL                   │               │
└─────────────────────────────────────────────────┼───────────────┘
                                                  │ FTS5 index
                                                  ▼
                              ┌────────────────────────────────────┐
                              │       evicted_turns_fts            │
                              │       (FTS5 virtual table)         │
                              │                                    │
                              │  full_content → tokenized index    │
                              └────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       tool_executions                           │
├─────────────────────────────────────────────────────────────────┤
│ id           INTEGER PRIMARY KEY                                │
│ session_id   TEXT NOT NULL                                      │
│ turn         INTEGER NOT NULL                                   │
│ tool_name    TEXT NOT NULL     ──────┐ filtered by              │
│ input        TEXT NOT NULL     ──────┼─────┐                    │
│ output       TEXT NOT NULL     ──────┼─────┼────┐               │
│ is_error     INTEGER NOT NULL        │     │    │ FTS5 index    │
│ executed_at  INTEGER NOT NULL        │     │    │               │
└──────────────────────────────────────┼─────┼────┼───────────────┘
                                       │     │    │
                                       │     ▼    ▼
                              ┌────────┼──────────────────────────┐
                              │        │  tool_executions_fts     │
                              │        │  (FTS5 virtual table)    │
                              │        │                          │
                              │        │  input → tokenized index │
                              │        │  output → tokenized index│
                              └────────┼──────────────────────────┘
                                       │
                                idx_tool_executions_tool
                                (for filtered queries)
```

## Token Budgeting

### Estimation Algorithm

Token counting is approximate. Exact counts would require tokenizing with each provider's tokenizer, which is slow and provider-specific. Instead, we use a conservative heuristic:

```rust
/// Estimate tokens in a string.
///
/// Uses 4 characters per token as a conservative estimate.
/// This over-estimates for English (typically ~4.5 chars/token)
/// but is safer than under-estimating.
fn estimate_tokens(text: &str) -> u32 {
    (text.len() as u32 + 3) / 4  // Ceiling division
}

/// Estimate tokens for a content block.
fn estimate_block_tokens(block: &ContentBlock) -> u32 {
    match block {
        ContentBlock::Text(s) => estimate_tokens(s),
        ContentBlock::ToolUse { name, input, .. } => {
            // Account for JSON structure overhead
            estimate_tokens(name) + estimate_tokens(&input.to_string()) + 20
        }
        ContentBlock::ToolResult { content, .. } => {
            estimate_tokens(content) + 10
        }
    }
}

/// Estimate total tokens in the message history.
fn estimate_context_tokens(messages: &[Message], system: Option<&str>) -> u32 {
    let mut total = 0;

    // System prompt
    if let Some(s) = system {
        total += estimate_tokens(s);
    }

    // Messages
    for msg in messages {
        // Role overhead (user/assistant markers)
        total += 4;

        for block in &msg.content {
            total += estimate_block_tokens(block);
        }
    }

    // Protocol overhead (conservative)
    total += 100;

    total
}
```

### Budget Thresholds

The context manager uses multiple thresholds:

| Threshold | % of Budget | Action |
|-----------|-------------|--------|
| **Warning** | 70% | Log warning (optional) |
| **Eviction** | 80% | Trigger eviction before next LLM call |
| **Target** | 60% | Evict until below this level |
| **Minimum** | 40% | Always keep at least this much headroom for response |

```rust
struct ContextBudget {
    total: u32,
}

impl ContextBudget {
    fn eviction_threshold(&self) -> u32 {
        (self.total as f32 * 0.8) as u32
    }

    fn target_after_eviction(&self) -> u32 {
        (self.total as f32 * 0.6) as u32
    }

    fn should_evict(&self, current: u32) -> bool {
        current > self.eviction_threshold()
    }
}
```

## Eviction Lifecycle

### Phase 1: Detection

Before each LLM request, the context manager checks the budget:

```rust
async fn prepare_request(&mut self) -> Result<Request, AgentError> {
    let current_tokens = estimate_context_tokens(&self.messages, self.system.as_deref());

    if self.budget.should_evict(current_tokens) {
        self.evict().await?;
    }

    // Build and return request
    // ...
}
```

### Phase 2: Selection

Determine which turns to evict:

```rust
fn select_turns_for_eviction(&self) -> Range<usize> {
    let target = self.budget.target_after_eviction();
    let mut current = estimate_context_tokens(&self.messages, self.system.as_deref());

    // Never evict the last 2 turns (need recent context)
    let max_evict = self.messages.len().saturating_sub(4);  // 2 turns = 4 messages

    let mut evict_count = 0;
    while current > target && evict_count < max_evict {
        // Each "turn" is 2 messages (assistant + user with tool results)
        // But the first message is just the user task
        let msg_idx = evict_count;
        if msg_idx < self.messages.len() {
            current -= estimate_message_tokens(&self.messages[msg_idx]);
            evict_count += 1;
        } else {
            break;
        }
    }

    // Minimum eviction of 2 messages if triggered
    evict_count = evict_count.max(2).min(max_evict);

    0..evict_count
}
```

### Phase 3: Summarization

Generate a summary of evicted content using the LLM:

```rust
async fn generate_summary(&self, turns: &[Message]) -> Result<String, LlmError> {
    // Serialize turns to text
    let content = turns.iter()
        .map(|m| format_message_for_summary(m))
        .collect::<Vec<_>>()
        .join("\n\n");

    let request = Request {
        system: None,
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text(format!(
                "Summarize this conversation excerpt in 2-3 sentences. \
                 Focus on key decisions, important information, and any \
                 unresolved questions:\n\n{}",
                content
            ))],
        }],
        tools: vec![],
        max_tokens: 200,
    };

    let response = self.client.complete(request).await?;

    Ok(extract_text(&response))
}
```

### Phase 4: Persistence

Store evicted content in SQLite (if persistence is enabled):

```rust
#[cfg(feature = "persistence")]
fn persist_evicted_turns(
    &self,
    store: &ContextStore,
    turn_range: Range<u32>,
    summary: &str,
    messages: &[Message],
) -> Result<(), ContextError> {
    // Serialize messages to JSON
    let full_content = serde_json::to_string(messages)
        .map_err(|e| ContextError::Sqlite(e.to_string()))?;

    // Insert evicted turns
    store.insert_evicted_turns(
        &self.session_id,
        turn_range.start,
        turn_range.end,
        summary,
        &full_content,
    )?;

    // Insert tool executions separately for filtered search
    for (turn_idx, msg) in messages.iter().enumerate() {
        let turn_num = turn_range.start + turn_idx as u32;

        for block in &msg.content {
            if let ContentBlock::ToolUse { name, input, id } = block {
                // Find corresponding tool result
                let result = self.find_tool_result(messages, id);

                if let Some((content, is_error)) = result {
                    store.insert_tool_execution(
                        &self.session_id,
                        turn_num,
                        name,
                        &input.to_string(),
                        content,
                        is_error,
                    )?;
                }
            }
        }
    }

    Ok(())
}
```

### Phase 5: Manifest Update

Replace evicted turns with a summary placeholder:

```rust
fn update_manifest(&mut self, turn_range: Range<usize>, summary: &str) {
    // Create manifest entry
    let manifest_entry = Message {
        role: Role::User,
        content: vec![ContentBlock::Text(format!(
            "[Context note: Turns {}-{} have been summarized and archived. \
             Summary: {}. Use the recall tool with relevant search terms \
             to retrieve specific details if needed.]",
            turn_range.start + 1,  // 1-indexed for display
            turn_range.end,
            summary
        ))],
    };

    // Remove evicted messages
    let remaining: Vec<_> = self.messages.drain(turn_range.clone()).collect();

    // Insert manifest entry at the start
    self.messages.insert(0, manifest_entry);

    // Note: remaining messages are now owned by persist_evicted_turns
    // if persistence is enabled
}
```

### Phase 6: Event Emission

Notify consumers about the eviction:

```rust
// In the agent loop
if eviction_occurred {
    yield AgentEvent::ContextEvicted {
        summary: summary.clone(),
        turns_evicted: (turn_range.start as u32)..(turn_range.end as u32),
    };
}
```

## Recall System

### System Prompt Addition

When persistence is enabled, the system prompt is augmented:

```rust
fn build_system_prompt(&self) -> String {
    let mut prompt = self.base_system_prompt.clone().unwrap_or_default();

    if self.store.is_some() {
        prompt.push_str("\n\n## Context Management\n\n");
        prompt.push_str(
            "This conversation uses automatic context management. When the \
             conversation grows long, older turns are summarized and archived. \
             You'll see markers like:\n\n\
             [Context note: Turns 5-12 have been summarized...]\n\n\
             If you need specific details from archived turns, use the `recall` \
             tool to search:\n\
             - recall(query=\"search terms\") - search all archived content\n\
             - recall(query=\"error\", tool_filter=\"bash\") - search only bash outputs\n\n\
             The recall tool uses full-text search. Use specific, relevant terms."
        );
    }

    prompt
}
```

### Recall Tool Implementation

```rust
pub struct RecallTool {
    store: ContextStore,
    max_results: usize,
}

#[async_trait]
impl Tool for RecallTool {
    fn name(&self) -> &str {
        "recall"
    }

    fn description(&self) -> &str {
        "Search and retrieve archived conversation context. Use when you need \
         information from earlier in the conversation that's no longer visible. \
         Supports full-text search with optional tool filtering."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms for full-text search"
                },
                "tool_filter": {
                    "type": "string",
                    "description": "Optional: only search outputs from this tool (e.g., 'bash', 'read_file')"
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: Value) -> ToolOutput {
        let query = match input.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => return ToolOutput::error("Missing required 'query' parameter"),
        };

        let tool_filter = input.get("tool_filter").and_then(|v| v.as_str());

        match self.search(query, tool_filter) {
            Ok(results) => ToolOutput::success(results),
            Err(e) => ToolOutput::error(format!("Recall search failed: {e}")),
        }
    }
}

impl RecallTool {
    fn search(&self, query: &str, tool_filter: Option<&str>) -> Result<String, ContextError> {
        // Sanitize query for FTS5 (escape special characters)
        let fts_query = sanitize_fts5_query(query);

        let results = match tool_filter {
            Some(tool) => self.search_tool_executions(&fts_query, tool)?,
            None => self.search_evicted_turns(&fts_query)?,
        };

        if results.is_empty() {
            Ok("No matching content found in archived context.".into())
        } else {
            Ok(format_recall_results(&results))
        }
    }

    fn search_evicted_turns(&self, query: &str) -> Result<Vec<RecallResult>, ContextError> {
        let sql = r#"
            SELECT
                et.turn_start,
                et.turn_end,
                et.summary,
                snippet(evicted_turns_fts, 0, '>>>', '<<<', '...', 64) as snippet
            FROM evicted_turns_fts
            JOIN evicted_turns et ON et.id = evicted_turns_fts.rowid
            WHERE evicted_turns_fts MATCH ?1
            ORDER BY rank
            LIMIT ?2
        "#;

        self.store.query(sql, [query, &self.max_results.to_string()])
    }

    fn search_tool_executions(&self, query: &str, tool: &str) -> Result<Vec<RecallResult>, ContextError> {
        let sql = r#"
            SELECT
                te.turn,
                te.tool_name,
                te.input,
                snippet(tool_executions_fts, 1, '>>>', '<<<', '...', 64) as output_snippet
            FROM tool_executions_fts
            JOIN tool_executions te ON te.id = tool_executions_fts.rowid
            WHERE tool_executions_fts MATCH ?1
              AND te.tool_name = ?2
            ORDER BY rank
            LIMIT ?3
        "#;

        self.store.query(sql, [query, tool, &self.max_results.to_string()])
    }
}

fn format_recall_results(results: &[RecallResult]) -> String {
    let mut output = String::new();
    output.push_str("Found the following archived content:\n\n");

    for (i, result) in results.iter().enumerate() {
        output.push_str(&format!("{}. ", i + 1));
        output.push_str(&result.format());
        output.push_str("\n\n");
    }

    output
}

/// Sanitize a string for FTS5 query syntax.
///
/// FTS5 uses special characters for operators. This escapes them
/// to ensure the query is treated as literal text.
fn sanitize_fts5_query(query: &str) -> String {
    // Wrap in quotes to treat as phrase, escape internal quotes
    format!("\"{}\"", query.replace('"', "\"\""))
}
```

## Without Persistence

When the `persistence` feature is disabled, context management still works but with reduced capability:

1. **Eviction still occurs** at the same thresholds
2. **Summaries are still generated** via LLM
3. **Manifest entries are still created** in the message history
4. **But full content is discarded** — the recall tool is not available

This provides basic context management for memory-constrained environments or when persistence isn't needed.

```rust
#[cfg(not(feature = "persistence"))]
fn persist_evicted_turns(&self, _: Range<u32>, _: &str, _: &[Message]) -> Result<(), ContextError> {
    // No-op without persistence
    Ok(())
}
```

## Performance Considerations

### FTS5 Optimization

- **WAL mode**: Allows concurrent readers during writes
- **Porter stemmer**: Matches word variations (run/running/ran)
- **Snippet function**: Returns only relevant portions, not full content
- **Rank ordering**: Most relevant results first

### Memory Usage

- **Lazy loading**: Full content only loaded on recall, not kept in memory
- **Streaming results**: Query results are iterated, not fully materialized
- **Connection pooling**: Single connection reused across operations

### Eviction Frequency

Eviction is triggered at 80% capacity and targets 60%. This provides:
- **20% headroom** for the next request+response
- **Batch eviction** — multiple turns at once, reducing LLM summarization calls
- **Stability** — won't thrash with evict/fill cycles
