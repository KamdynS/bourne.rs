# Error Handling Strategy

This document describes how errors are classified, handled, and propagated throughout the agent-rs system. The guiding principle is: **errors are data, not exceptions**.

## Error Philosophy

Traditional error handling treats errors as exceptional conditions that should interrupt normal flow. In an agentic system, this is counterproductive:

1. **Tool failures are expected.** Commands fail, files don't exist, APIs time out. These aren't bugs—they're normal operating conditions.

2. **LLMs can self-correct.** When a tool fails, the model can try a different approach, fix its input, or gracefully degrade.

3. **Infrastructure failures are different from semantic failures.** A network timeout is different from "file not found." The former should be retried; the latter should be reported to the model.

The agent-rs error strategy distinguishes between these cases and handles each appropriately.

## Error Classification

### Taxonomy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              All Errors                                     │
├───────────────────────────────────┬─────────────────────────────────────────┤
│                                   │                                         │
│    Infrastructure Errors          │         Semantic Errors                 │
│    (handled by system)            │         (handled by model)              │
│                                   │                                         │
│    ┌───────────────────────┐      │      ┌───────────────────────┐         │
│    │ Retryable             │      │      │ Tool Failures         │         │
│    │ - Rate limits         │      │      │ - Command failed      │         │
│    │ - Service overload    │      │      │ - File not found      │         │
│    │ - Network timeouts    │      │      │ - Invalid input       │         │
│    └───────────────────────┘      │      │ - Permission denied   │         │
│                                   │      └───────────────────────┘         │
│    ┌───────────────────────┐      │                                         │
│    │ Fatal                 │      │                                         │
│    │ - Auth failures       │      │                                         │
│    │ - Invalid requests    │      │                                         │
│    │ - Max turns exceeded  │      │                                         │
│    │ - Cancellation        │      │                                         │
│    └───────────────────────┘      │                                         │
│                                   │                                         │
└───────────────────────────────────┴─────────────────────────────────────────┘
```

### Detailed Classification

| Error | Category | HTTP Status | Action | Rationale |
|-------|----------|-------------|--------|-----------|
| **Rate Limited** | Infrastructure/Retryable | 429 | Retry with backoff | Temporary, will succeed |
| **Overloaded** | Infrastructure/Retryable | 529, 503 | Retry with longer backoff | Provider-side congestion |
| **Network Error** | Infrastructure/Retryable | N/A | Retry up to 3 times | Connection issues |
| **Auth Failure** | Infrastructure/Fatal | 401, 403 | Bubble up immediately | Configuration error |
| **Invalid Request** | Infrastructure/Fatal | 400 | Bubble up immediately | Bug in request construction |
| **Max Turns** | Infrastructure/Fatal | N/A | Bubble up | Safety limit reached |
| **Cancelled** | Infrastructure/Fatal | N/A | Bubble up | Explicit user action |
| **Tool Error** | Semantic | N/A | Feed to model | Model can self-correct |
| **Tool Panic** | Semantic | N/A | Catch, feed to model | Convert to error result |

## Error Types

### AgentError

The top-level error type returned from `Agent::run()`:

```rust
/// Errors that terminate agent execution.
///
/// These are errors that cannot be recovered from within the agent loop.
/// They bubble up to the caller for handling.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// An LLM API error that couldn't be recovered via retry.
    ///
    /// This includes auth failures, invalid requests, and retryable
    /// errors that exceeded the retry limit.
    #[error("LLM error: {0}")]
    Llm(#[from] LlmError),

    /// The agent was cancelled via its cancellation token.
    ///
    /// This is a clean shutdown, not an error condition. The agent
    /// stopped at a safe point between turns.
    #[error("Agent cancelled")]
    Cancelled,

    /// The agent exceeded the configured maximum number of turns.
    ///
    /// This is a safety limit to prevent runaway execution. The
    /// conversation can be continued by creating a new agent with
    /// the existing message history.
    #[error("Exceeded maximum turns ({0})")]
    MaxTurnsExceeded(u32),
}
```

### LlmError

Errors from LLM API calls:

```rust
/// Errors from LLM provider APIs.
///
/// These are classified by recoverability:
/// - RateLimit, Overloaded, Network: Retryable by the agent
/// - InvalidRequest, Auth: Fatal, bubble up immediately
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// Rate limited by the provider (HTTP 429).
    ///
    /// The `retry_after` hint comes from the response headers.
    /// If not provided, exponential backoff is used.
    #[error("Rate limited (retry after {retry_after:?})")]
    RateLimit {
        retry_after: Option<Duration>,
    },

    /// Provider is overloaded (HTTP 529 or 503).
    ///
    /// More severe than rate limiting—indicates provider-wide issues.
    /// Longer backoff intervals are appropriate.
    #[error("Service overloaded")]
    Overloaded,

    /// The request was invalid (HTTP 400).
    ///
    /// This indicates a bug in request construction. The request
    /// will never succeed and should not be retried.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Authentication failed (HTTP 401 or 403).
    ///
    /// The API key is invalid, expired, or lacks permissions.
    /// Cannot be recovered without configuration changes.
    #[error("Authentication failed: {0}")]
    Auth(String),

    /// Network-level error (connection failed, timeout, DNS, etc.)
    ///
    /// Transient and worth retrying a few times.
    #[error("Network error: {0}")]
    Network(String),
}
```

### ToolOutput

Tool results (not errors in the Rust sense):

```rust
/// The result of executing a tool.
///
/// Note: This is NOT a Result type. Both success and failure are
/// valid outputs that get sent back to the model. The `is_error`
/// flag tells the model whether the tool succeeded.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// The output content.
    ///
    /// For success: the tool's output (file contents, command output, etc.)
    /// For error: an error message describing what went wrong
    pub content: String,

    /// Whether this represents an error.
    ///
    /// When true, the model will see this as a failed tool call and
    /// can attempt recovery (retry with different input, try alternative
    /// approach, or report the failure to the user).
    pub is_error: bool,
}
```

## Retry Logic

### Implementation

```rust
/// Execute an async operation with exponential backoff retry.
///
/// Retries are attempted for:
/// - RateLimit: Uses retry_after header, falls back to exponential backoff
/// - Overloaded: Always uses longer initial delay (30s)
/// - Network: Standard exponential backoff
///
/// Other errors are returned immediately without retry.
async fn with_retry<F, Fut, T>(mut operation: F) -> Result<T, LlmError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, LlmError>>,
{
    const MAX_RETRIES: u32 = 5;
    const INITIAL_DELAY: Duration = Duration::from_secs(1);
    const MAX_DELAY: Duration = Duration::from_secs(60);
    const OVERLOAD_DELAY: Duration = Duration::from_secs(30);

    let mut attempt = 0;
    let mut delay = INITIAL_DELAY;

    loop {
        match operation().await {
            Ok(result) => return Ok(result),

            Err(LlmError::RateLimit { retry_after }) => {
                if attempt >= MAX_RETRIES {
                    return Err(LlmError::RateLimit { retry_after });
                }

                let wait = retry_after.unwrap_or(delay);
                tracing::warn!(
                    attempt = attempt + 1,
                    wait_secs = wait.as_secs(),
                    "Rate limited, retrying"
                );

                tokio::time::sleep(wait).await;
                delay = (delay * 2).min(MAX_DELAY);
                attempt += 1;
            }

            Err(LlmError::Overloaded) => {
                if attempt >= 3 {  // Fewer retries for overload
                    return Err(LlmError::Overloaded);
                }

                tracing::warn!(
                    attempt = attempt + 1,
                    wait_secs = OVERLOAD_DELAY.as_secs(),
                    "Service overloaded, retrying"
                );

                tokio::time::sleep(OVERLOAD_DELAY).await;
                attempt += 1;
            }

            Err(LlmError::Network(msg)) => {
                if attempt >= 3 {
                    return Err(LlmError::Network(msg));
                }

                tracing::warn!(
                    attempt = attempt + 1,
                    wait_secs = delay.as_secs(),
                    error = %msg,
                    "Network error, retrying"
                );

                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(MAX_DELAY);
                attempt += 1;
            }

            // Non-retryable errors return immediately
            Err(e) => return Err(e),
        }
    }
}
```

### Retry Decision Tree

```
LLM Request Failed
        │
        ▼
┌───────────────────┐
│ What type of      │
│ error?            │
└─────────┬─────────┘
          │
    ┌─────┴─────┬──────────────┬──────────────┐
    ▼           ▼              ▼              ▼
RateLimit   Overloaded     Network      Auth/Invalid
    │           │              │              │
    ▼           ▼              ▼              ▼
┌─────────┐ ┌─────────┐  ┌─────────┐    ┌─────────┐
│Attempts │ │Attempts │  │Attempts │    │ Return  │
│  < 5?   │ │  < 3?   │  │  < 3?   │    │  Error  │
└────┬────┘ └────┬────┘  └────┬────┘    └─────────┘
     │           │            │
   Yes/No      Yes/No       Yes/No
     │           │            │
    ┌┴──┐       ┌┴──┐        ┌┴──┐
   Yes  No     Yes  No      Yes  No
    │    │      │    │       │    │
    ▼    ▼      ▼    ▼       ▼    ▼
 Wait  Return Wait Return  Wait Return
 and    Error  30s  Error  exp.  Error
 Retry        Retry        back.
  │            │           Retry
  │            │             │
  └────────────┴─────────────┘
              │
              ▼
        Retry Request
```

## Tool Error Flow

### The Self-Correction Pattern

When a tool fails, the error is fed back to the model as a normal tool result:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Turn N                                                                     │
│                                                                              │
│  Model: "I'll read the config file"                                         │
│  ToolUse: read_file(path="/etc/config.json")                                │
│                                                                              │
│  Tool executes...                                                           │
│  Result: ToolOutput {                                                       │
│      content: "Error: File not found: /etc/config.json",                    │
│      is_error: true,                                                        │
│  }                                                                          │
│                                                                              │
│  ToolResult sent to model with is_error=true                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Turn N+1                                                                   │
│                                                                              │
│  Model sees the error and self-corrects:                                    │
│  "The config file doesn't exist at that path. Let me search for it..."      │
│  ToolUse: bash(command="find /etc -name 'config.json'")                     │
│                                                                              │
│  Or acknowledges the limitation:                                            │
│  "I couldn't find the config file. Could you provide the correct path?"     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation in Agent Loop

```rust
async fn execute_tools(&self, tool_calls: Vec<ToolCall>) -> Vec<(String, ToolOutput)> {
    let futures = tool_calls.into_iter().map(|call| {
        let tools = &self.tools;
        async move {
            // Find the tool
            let tool = tools.iter().find(|t| t.name() == call.name);

            let output = match tool {
                Some(t) => {
                    // Execute with panic catching
                    match std::panic::AssertUnwindSafe(t.execute(call.input.clone()))
                        .catch_unwind()
                        .await
                    {
                        Ok(output) => output,
                        Err(_) => ToolOutput::error(format!(
                            "Tool '{}' panicked during execution",
                            call.name
                        )),
                    }
                }
                None => ToolOutput::error(format!(
                    "Unknown tool: '{}'. Available tools: {}",
                    call.name,
                    tools.iter().map(|t| t.name()).collect::<Vec<_>>().join(", ")
                )),
            };

            (call.id, output)
        }
    });

    futures::future::join_all(futures).await
}
```

### What Tools Should Return

Tools should handle their own errors and return appropriate `ToolOutput`:

```rust
impl Tool for BashTool {
    async fn execute(&self, input: Value) -> ToolOutput {
        let command = match input.get("command").and_then(|v| v.as_str()) {
            Some(c) => c,
            None => return ToolOutput::error("Missing required 'command' parameter"),
        };

        // Check sandbox restrictions
        if let Some(ref sandbox) = self.sandbox {
            if let Some(denied) = sandbox.check_denied(command) {
                return ToolOutput::error(format!(
                    "Command '{}' is not allowed by sandbox policy",
                    denied
                ));
            }
        }

        // Execute with timeout
        let result = tokio::time::timeout(
            self.timeout,
            tokio::process::Command::new("bash")
                .arg("-c")
                .arg(command)
                .output()
        ).await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);

                if output.status.success() {
                    ToolOutput::success(format!("{}{}", stdout, stderr))
                } else {
                    ToolOutput::error(format!(
                        "Command failed with exit code {}:\n{}{}",
                        output.status.code().unwrap_or(-1),
                        stdout,
                        stderr
                    ))
                }
            }
            Ok(Err(e)) => ToolOutput::error(format!("Failed to execute command: {e}")),
            Err(_) => ToolOutput::error(format!(
                "Command timed out after {} seconds",
                self.timeout.as_secs()
            )),
        }
    }
}
```

## Propagation Summary

| Error Source | Initial Handler | Propagation |
|--------------|-----------------|-------------|
| LLM rate limit | `with_retry()` | Retry → bubble if exceeded |
| LLM overload | `with_retry()` | Retry → bubble if exceeded |
| LLM network | `with_retry()` | Retry → bubble if exceeded |
| LLM auth | `complete_stream()` | Immediate bubble |
| LLM invalid | `complete_stream()` | Immediate bubble |
| Tool error | `execute_tools()` | Feed to model |
| Tool panic | `execute_tools()` | Catch, feed to model |
| Max turns | `agent_loop()` | Bubble as `MaxTurnsExceeded` |
| Cancellation | `agent_loop()` | Bubble as `Cancelled` |

## Design Rationale

### Why Not Retry Tool Failures?

The agent loop doesn't retry failed tools because:

1. **Context matters.** The model knows why it called the tool and what alternatives exist.
2. **Input might be wrong.** Retrying with the same input will fail again.
3. **Side effects.** Some tools have side effects—blind retry could cause issues.
4. **The model is smarter.** An LLM can make intelligent recovery decisions; a retry loop cannot.

### Why Catch Panics?

Tools are user-provided code. A panic in a tool shouldn't crash the agent:

1. **Isolation.** Tool bugs shouldn't propagate.
2. **Graceful degradation.** The model can work around a broken tool.
3. **Debugging.** The error message helps identify the problem.

### Why Distinguish Retryable vs Fatal?

Infrastructure errors have different characteristics:

- **Retryable:** Temporary, will succeed eventually, no human intervention needed
- **Fatal:** Permanent, requires configuration change or bug fix

Treating them the same wastes time (retrying auth failures) or loses opportunities (giving up on rate limits).
