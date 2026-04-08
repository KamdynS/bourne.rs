//! Agent executor and builder.
//!
//! This module contains the heart of agent-rs: the agent loop. The loop
//! coordinates LLM calls, tool execution, and event streaming.
//!
//! # Agent Lifecycle
//!
//! ```text
//! AgentBuilder::new(client)
//!     .system_prompt("...")
//!     .tools(vec![...])
//!     .build()
//!         │
//!         ▼
//!     Agent (configured, ready to run)
//!         │
//!         │ .run("task")
//!         ▼
//!     Stream<AgentEvent> (consumed, agent gone)
//! ```
//!
//! Agents are single-use: you build one, run it, and it's consumed.
//! For multiple tasks, create multiple agents (they're cheap).
//!
//! # The Turn Loop
//!
//! Each turn:
//! 1. Check cancellation
//! 2. Build request from message history
//! 3. Stream LLM response, emitting Text events
//! 4. If stop_reason is ToolUse: execute tools in parallel, loop
//! 5. If stop_reason is EndTurn: emit Done, finish

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;

use crate::{
    AgentError, ContentBlock, LlmClient, LlmError, Message, Request, Role, StopReason,
    StreamChunk, Tool, ToolDef, ToolOutput, TokenUsage,
};

// =============================================================================
// AGENT EVENTS
// =============================================================================

/// Events emitted by the agent during execution.
///
/// Consumers process these to display progress, log activity, or forward
/// to other systems. The stream of events tells the story of what the
/// agent is doing.
///
/// # Event Sequence
///
/// A typical successful run:
/// ```text
/// Text("Let me ") → Text("check...") → ToolStart { bash } → ToolEnd { output }
/// → TurnComplete { turn: 1 } → Text("The files are...") → Done { final_text }
/// ```
///
/// Events are emitted in causal order: ToolStart before ToolEnd,
/// TurnComplete after all tools finish.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A chunk of text streamed from the model.
    ///
    /// Multiple Text events form the complete response. Buffer and
    /// display incrementally for real-time output.
    Text(String),

    /// The model is calling a tool.
    ///
    /// Emitted when the tool call is fully parsed from the stream.
    /// Execution hasn't started yet.
    ToolStart {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    /// A tool has finished executing.
    ToolEnd {
        id: String,
        output: ToolOutput,
    },

    /// A complete turn has finished (after tool execution, if any).
    TurnComplete {
        turn: u32,
        usage: TokenUsage,
    },

    /// The agent has completed successfully.
    Done {
        final_text: String,
    },
}

// =============================================================================
// AGENT BUILDER
// =============================================================================

/// Builder for configuring and constructing an Agent.
///
/// # Example
///
/// ```ignore
/// let agent = AgentBuilder::new(Box::new(client))
///     .system_prompt("You are a helpful assistant.")
///     .tools(vec![Box::new(BashTool::new())])
///     .max_turns(50)
///     .build();
/// ```
pub struct AgentBuilder {
    client: Box<dyn LlmClient>,
    system_prompt: Option<String>,
    tools: Vec<Box<dyn Tool>>,
    max_turns: u32,
    max_tokens: u32,
    cancellation_token: Option<CancellationToken>,
}

impl AgentBuilder {
    /// Create a new builder with the given LLM client.
    pub fn new(client: Box<dyn LlmClient>) -> Self {
        Self {
            client,
            system_prompt: None,
            tools: Vec::new(),
            max_turns: 100,
            max_tokens: 4096,
            cancellation_token: None,
        }
    }

    /// Set the system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Register tools the agent can use.
    pub fn tools(mut self, tools: Vec<Box<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Set maximum turns (default: 100).
    pub fn max_turns(mut self, turns: u32) -> Self {
        self.max_turns = turns;
        self
    }

    /// Set max tokens per response (default: 4096).
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Provide a cancellation token for graceful shutdown.
    pub fn cancellation_token(mut self, token: CancellationToken) -> Self {
        self.cancellation_token = Some(token);
        self
    }

    /// Build the agent.
    pub fn build(self) -> Agent {
        let mut tool_map = HashMap::new();
        let mut tool_defs = Vec::new();

        for tool in self.tools {
            tool_defs.push(ToolDef {
                name: tool.name().to_string(),
                description: tool.description().to_string(),
                input_schema: tool.input_schema(),
            });
            tool_map.insert(tool.name().to_string(), tool);
        }

        Agent {
            client: self.client,
            system_prompt: self.system_prompt,
            tools: Arc::new(tool_map),
            tool_defs,
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
            cancellation_token: self.cancellation_token,
        }
    }
}

// =============================================================================
// AGENT
// =============================================================================

/// The agent executor.
///
/// Agents are single-use: build, run, consume. The `run` method returns
/// a stream of events that drives the agent loop.
pub struct Agent {
    client: Box<dyn LlmClient>,
    system_prompt: Option<String>,
    tools: Arc<HashMap<String, Box<dyn Tool>>>,
    tool_defs: Vec<ToolDef>,
    max_turns: u32,
    max_tokens: u32,
    cancellation_token: Option<CancellationToken>,
}

impl Agent {
    /// Execute the agent on a task, yielding events as they occur.
    ///
    /// The agent is consumed by this call. The stream completes when the
    /// model signals done, max turns is reached, or an error occurs.
    pub fn run(self, task: &str) -> impl Stream<Item = Result<AgentEvent, AgentError>> + Send {
        let task = task.to_string();
        futures::stream::unfold(AgentState::new(self, task), |mut state| async move {
            match state.next_event().await {
                Some(result) => Some((result, state)),
                None => None,
            }
        })
    }
}

// =============================================================================
// INTERNAL STATE MACHINE
// =============================================================================

/// Internal state for the agent execution loop.
///
/// Implements the state machine that yields events. The complexity here
/// allows us to return a simple Stream from `Agent::run`.
struct AgentState {
    // Configuration (moved from Agent)
    client: Box<dyn LlmClient>,
    system_prompt: Option<String>,
    tools: Arc<HashMap<String, Box<dyn Tool>>>,
    tool_defs: Vec<ToolDef>,
    max_turns: u32,
    max_tokens: u32,
    cancellation_token: Option<CancellationToken>,

    // Conversation state
    messages: Vec<Message>,
    turn: u32,
    done: bool,

    // Current turn state
    stream: Option<Pin<Box<dyn Stream<Item = Result<StreamChunk, LlmError>> + Send>>>,
    text_buffer: String,
    tool_calls: Vec<ToolCall>,
    current_tool: Option<ToolCallBuilder>,
    turn_usage: TokenUsage,

    // Event queue for yielding multiple events from one transition
    pending_events: Vec<Result<AgentEvent, AgentError>>,
}

struct ToolCall {
    id: String,
    name: String,
    input: serde_json::Value,
}

struct ToolCallBuilder {
    id: String,
    name: String,
    input_json: String,
}

impl AgentState {
    fn new(agent: Agent, task: String) -> Self {
        Self {
            client: agent.client,
            system_prompt: agent.system_prompt,
            tools: agent.tools,
            tool_defs: agent.tool_defs,
            max_turns: agent.max_turns,
            max_tokens: agent.max_tokens,
            cancellation_token: agent.cancellation_token,
            messages: vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text(task)],
            }],
            turn: 0,
            done: false,
            stream: None,
            text_buffer: String::new(),
            tool_calls: Vec::new(),
            current_tool: None,
            turn_usage: TokenUsage::default(),
            pending_events: Vec::new(),
        }
    }

    /// Get the next event, driving the state machine forward.
    async fn next_event(&mut self) -> Option<Result<AgentEvent, AgentError>> {
        loop {
            // 1. Drain any pending events first
            if let Some(event) = self.pending_events.pop() {
                return Some(event);
            }

            if self.done {
                return None;
            }

            // 2. If we have tool calls waiting and no stream, execute them
            if !self.tool_calls.is_empty() && self.stream.is_none() {
                return Some(self.execute_tools().await);
            }

            // 3. If no stream, start a new turn
            if self.stream.is_none() {
                if let Err(e) = self.start_turn() {
                    self.done = true;
                    return Some(Err(e));
                }
            }

            // 4. Process the stream
            // If process_stream returns None, it means the stream ended but we
            // might have more work (tool execution). Loop back to check.
            if let Some(event) = self.process_stream().await {
                return Some(event);
            }
            // process_stream returned None - loop back to handle tool execution
            // or start next turn
        }
    }

    /// Start a new turn: validate, build request, begin streaming.
    fn start_turn(&mut self) -> Result<(), AgentError> {
        self.turn += 1;

        // Check cancellation
        if let Some(token) = &self.cancellation_token {
            if token.is_cancelled() {
                return Err(AgentError::Cancelled);
            }
        }

        // Check turn limit
        if self.turn > self.max_turns {
            return Err(AgentError::MaxTurnsExceeded(self.max_turns));
        }

        // Build and send request
        let request = Request {
            system: self.system_prompt.clone(),
            messages: self.messages.clone(),
            tools: self.tool_defs.clone(),
            max_tokens: self.max_tokens,
        };

        self.stream = Some(self.client.complete_stream(request));
        self.text_buffer.clear();
        self.tool_calls.clear();
        self.current_tool = None;
        self.turn_usage = TokenUsage::default();

        Ok(())
    }

    /// Process chunks from the LLM stream.
    async fn process_stream(&mut self) -> Option<Result<AgentEvent, AgentError>> {
        loop {
            // Check if stream exists - if not, we need to continue to next phase
            let stream = match self.stream.as_mut() {
                Some(s) => s,
                None => {
                    // Stream was consumed (e.g., after MessageDone with ToolUse)
                    // Return None to let next_event handle the next phase
                    return None;
                }
            };

            match stream.next().await {
                Some(Ok(chunk)) => {
                    if let Some(result) = self.handle_chunk(chunk) {
                        return Some(result);
                    }
                    // handle_chunk returned None, loop to get next chunk
                    // (but stream might now be None, so we'll check at top of loop)
                }
                Some(Err(e)) => {
                    self.done = true;
                    return Some(Err(AgentError::Llm(e)));
                }
                None => {
                    // Stream ended unexpectedly
                    self.done = true;
                    return None;
                }
            }
        }
    }

    /// Handle a single chunk, possibly producing an event.
    fn handle_chunk(&mut self, chunk: StreamChunk) -> Option<Result<AgentEvent, AgentError>> {
        match chunk {
            StreamChunk::Text(text) => {
                self.text_buffer.push_str(&text);
                Some(Ok(AgentEvent::Text(text)))
            }

            StreamChunk::ToolUseStart { id, name } => {
                self.current_tool = Some(ToolCallBuilder {
                    id,
                    name,
                    input_json: String::new(),
                });
                None
            }

            StreamChunk::ToolUseInput(fragment) => {
                if let Some(builder) = &mut self.current_tool {
                    builder.input_json.push_str(&fragment);
                }
                None
            }

            StreamChunk::ToolUseDone => {
                if let Some(builder) = self.current_tool.take() {
                    let input: serde_json::Value = serde_json::from_str(&builder.input_json)
                        .unwrap_or(serde_json::Value::Null);

                    let event = AgentEvent::ToolStart {
                        id: builder.id.clone(),
                        name: builder.name.clone(),
                        input: input.clone(),
                    };

                    self.tool_calls.push(ToolCall {
                        id: builder.id,
                        name: builder.name,
                        input,
                    });

                    return Some(Ok(event));
                }
                None
            }

            StreamChunk::MessageDone { stop_reason, usage } => {
                self.turn_usage = usage;
                self.stream = None;

                match stop_reason {
                    StopReason::EndTurn | StopReason::MaxTokens => {
                        self.finish();
                        self.pending_events.pop()
                    }
                    StopReason::ToolUse => {
                        // Tools will be executed on next next_event() call
                        None
                    }
                }
            }
        }
    }

    /// Execute all pending tool calls in parallel.
    async fn execute_tools(&mut self) -> Result<AgentEvent, AgentError> {
        // 1. Add assistant message to history
        let mut assistant_content = Vec::new();
        if !self.text_buffer.is_empty() {
            assistant_content.push(ContentBlock::Text(self.text_buffer.clone()));
        }
        for call in &self.tool_calls {
            assistant_content.push(ContentBlock::ToolUse {
                id: call.id.clone(),
                name: call.name.clone(),
                input: call.input.clone(),
            });
        }
        self.messages.push(Message {
            role: Role::Assistant,
            content: assistant_content,
        });

        // 2. Execute tools in parallel
        let tools = self.tools.clone();
        let calls = std::mem::take(&mut self.tool_calls);

        let futures = calls.into_iter().map(|call| {
            let tools = tools.clone();
            async move {
                let output = match tools.get(&call.name) {
                    Some(tool) => tool.execute(call.input).await,
                    None => ToolOutput::error(format!("Unknown tool: {}", call.name)),
                };
                (call.id, output)
            }
        });

        let results: Vec<_> = futures::future::join_all(futures).await;

        // 3. Build user message with results
        let mut user_content = Vec::new();
        for (id, output) in &results {
            user_content.push(ContentBlock::ToolResult {
                id: id.clone(),
                content: output.content.clone(),
                is_error: output.is_error,
            });
        }
        self.messages.push(Message {
            role: Role::User,
            content: user_content,
        });

        // 4. Queue ToolEnd events and TurnComplete
        // (Push in reverse order since we pop from end)
        self.pending_events.push(Ok(AgentEvent::TurnComplete {
            turn: self.turn,
            usage: self.turn_usage,
        }));

        for (id, output) in results.into_iter().rev() {
            self.pending_events.push(Ok(AgentEvent::ToolEnd { id, output }));
        }

        // 5. Return the first ToolEnd event
        self.pending_events.pop().unwrap()
    }

    /// Finish the conversation: update history, queue final events.
    fn finish(&mut self) {
        // Add final assistant message if we have content
        if !self.text_buffer.is_empty() || !self.tool_calls.is_empty() {
            let mut content = Vec::new();
            if !self.text_buffer.is_empty() {
                content.push(ContentBlock::Text(self.text_buffer.clone()));
            }
            for call in &self.tool_calls {
                content.push(ContentBlock::ToolUse {
                    id: call.id.clone(),
                    name: call.name.clone(),
                    input: call.input.clone(),
                });
            }
            self.messages.push(Message {
                role: Role::Assistant,
                content,
            });
        }

        // Queue final events (reverse order for popping)
        self.pending_events.push(Ok(AgentEvent::Done {
            final_text: self.text_buffer.clone(),
        }));
        self.pending_events.push(Ok(AgentEvent::TurnComplete {
            turn: self.turn,
            usage: self.turn_usage,
        }));

        self.done = true;
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::{MockClient, MockResponse};
    use async_trait::async_trait;
    use futures::StreamExt;
    use serde_json::{json, Value};

    // -------------------------------------------------------------------------
    // Test Tool: Echo
    // -------------------------------------------------------------------------

    /// A simple tool that echoes its input. Used for testing.
    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn description(&self) -> &str {
            "Echo the message back"
        }

        fn input_schema(&self) -> Value {
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"]
            })
        }

        async fn execute(&self, input: Value) -> ToolOutput {
            match input.get("message").and_then(|v| v.as_str()) {
                Some(msg) => ToolOutput::success(format!("Echo: {}", msg)),
                None => ToolOutput::error("Missing 'message' parameter"),
            }
        }
    }

    // -------------------------------------------------------------------------
    // Basic Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_agent_event_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<AgentEvent>();
    }

    #[test]
    fn test_tool_output_constructors() {
        let success = ToolOutput::success("output");
        assert!(!success.is_error);
        assert_eq!(success.content, "output");

        let error = ToolOutput::error("failed");
        assert!(error.is_error);
        assert_eq!(error.content, "failed");
    }

    // -------------------------------------------------------------------------
    // Agent Loop Tests
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn test_agent_simple_response() {
        // Mock returns a simple text response
        let client = MockClient::new(vec![MockResponse::text("Hello, world!")]);

        let agent = AgentBuilder::new(Box::new(client))
            .system_prompt("Be helpful")
            .build();

        let events: Vec<_> = agent
            .run("Say hello")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        // Should have Text events, TurnComplete, and Done
        let has_text = events
            .iter()
            .any(|e| matches!(e, AgentEvent::Text(s) if s.contains("Hello")));
        let has_done = events
            .iter()
            .any(|e| matches!(e, AgentEvent::Done { final_text } if final_text.contains("Hello")));
        let has_turn_complete = events
            .iter()
            .any(|e| matches!(e, AgentEvent::TurnComplete { turn: 1, .. }));

        assert!(has_text, "Should have text events");
        assert!(has_done, "Should have Done event");
        assert!(has_turn_complete, "Should have TurnComplete event");
    }

    #[tokio::test]
    async fn test_agent_with_tool_call() {
        // Mock returns a tool call, then a final response
        let client = MockClient::new(vec![
            MockResponse::tool_call("Let me echo that", "tool_1", "echo", json!({"message": "test"})),
            MockResponse::text("I echoed your message."),
        ]);

        let agent = AgentBuilder::new(Box::new(client))
            .tools(vec![Box::new(EchoTool)])
            .build();

        let events: Vec<_> = agent
            .run("Echo test")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        // Should have ToolStart, ToolEnd events
        let has_tool_start = events.iter().any(|e| {
            matches!(e, AgentEvent::ToolStart { name, .. } if name == "echo")
        });
        let has_tool_end = events.iter().any(|e| {
            matches!(e, AgentEvent::ToolEnd { output, .. } if output.content.contains("Echo: test"))
        });

        assert!(has_tool_start, "Should have ToolStart event");
        assert!(has_tool_end, "Should have ToolEnd with echo output");

        // Should complete with two turns
        let turn_completes: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                AgentEvent::TurnComplete { turn, .. } => Some(*turn),
                _ => None,
            })
            .collect();
        assert_eq!(turn_completes, vec![1, 2], "Should have 2 turns");
    }

    #[tokio::test]
    async fn test_agent_unknown_tool() {
        // Mock calls a tool that doesn't exist
        let client = MockClient::new(vec![
            MockResponse::tool_only("tool_1", "nonexistent", json!({})),
            MockResponse::text("Tool failed, sorry."),
        ]);

        let agent = AgentBuilder::new(Box::new(client))
            .tools(vec![]) // No tools registered
            .build();

        let events: Vec<_> = agent
            .run("Use unknown tool")
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();

        // Should have ToolEnd with error
        let has_error = events.iter().any(|e| {
            matches!(e, AgentEvent::ToolEnd { output, .. } if output.is_error)
        });

        assert!(has_error, "Unknown tool should produce error output");
    }

    #[tokio::test]
    async fn test_agent_max_turns() {
        // Mock keeps returning tool calls forever
        let responses: Vec<_> = (0..10)
            .map(|i| MockResponse::tool_only(format!("tool_{}", i), "echo", json!({"message": "loop"})))
            .collect();

        let client = MockClient::new(responses);

        let agent = AgentBuilder::new(Box::new(client))
            .tools(vec![Box::new(EchoTool)])
            .max_turns(3)
            .build();

        let results: Vec<_> = agent.run("Loop forever").collect().await;

        // Should have an error for exceeding max turns
        let has_max_turns_error = results.iter().any(|r| {
            matches!(r, Err(AgentError::MaxTurnsExceeded(3)))
        });

        assert!(has_max_turns_error, "Should error on max turns exceeded");
    }

    #[tokio::test]
    async fn test_agent_cancellation() {
        let client = MockClient::new(vec![MockResponse::text("Should not see this")]);

        let cancel_token = CancellationToken::new();
        cancel_token.cancel(); // Cancel immediately

        let agent = AgentBuilder::new(Box::new(client))
            .cancellation_token(cancel_token)
            .build();

        let results: Vec<_> = agent.run("Cancelled task").collect().await;

        let has_cancelled_error = results
            .iter()
            .any(|r| matches!(r, Err(AgentError::Cancelled)));

        assert!(has_cancelled_error, "Should error on cancellation");
    }
}
