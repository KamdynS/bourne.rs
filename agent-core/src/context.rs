//! Context management for long-running conversations.
//!
//! LLMs have finite context windows. When conversations grow too long,
//! we need strategies to keep them within budget. This module provides:
//!
//! - **EvictionStrategy**: How to reduce context size (drop old, summarize, etc.)
//! - **ContextStore**: Where to persist messages (memory, SQLite, etc.)
//! - **ContextManager**: Combines storage + strategy for complete management
//!
//! # Design Philosophy
//!
//! Storage and eviction are orthogonal concerns:
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────────┐
//! │ ContextStore    │     │ EvictionStrategy │
//! │                 │     │                  │
//! │ • InMemory      │  +  │ • DropOldest     │
//! │ • Sqlite        │     │ • Summarize      │
//! └─────────────────┘     └──────────────────┘
//!          │                       │
//!          └───────────┬───────────┘
//!                      ▼
//!              ┌───────────────┐
//!              │ContextManager │
//!              └───────────────┘
//! ```
//!
//! Users can mix and match. Want in-memory with summarization? Sure.
//! Want SQLite with simple drop-oldest? Also fine.
//!
//! # Token Counting
//!
//! We use a simple heuristic: ~4 characters per token. This is imprecise
//! but avoids depending on tokenizer libraries. For exact counts, you'd
//! need the model's actual tokenizer.
//!
//! # Example
//!
//! ```ignore
//! use agent_core::context::{ContextManager, InMemoryStore, DropOldest};
//!
//! let manager = ContextManager::new(
//!     InMemoryStore::new(),
//!     DropOldest,
//!     100_000, // max tokens
//! );
//!
//! // Add messages, manager handles eviction automatically
//! manager.add_message(message);
//! let messages = manager.get_messages();
//! ```

use crate::{ContentBlock, LlmClient, Message, Request, Role};

// =============================================================================
// TOKEN ESTIMATION
// =============================================================================

/// Estimate token count for a message.
///
/// Uses ~4 chars per token heuristic. Not exact, but good enough
/// for budget management without tokenizer dependencies.
pub fn estimate_tokens(message: &Message) -> usize {
    let char_count: usize = message
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text(s) => s.len(),
            ContentBlock::ToolUse { input, .. } => {
                // Tool name + JSON input
                input.to_string().len() + 50
            }
            ContentBlock::ToolResult { content, .. } => content.len() + 20,
        })
        .sum();

    // ~4 chars per token, minimum 1
    (char_count / 4).max(1)
}

/// Estimate total tokens for a message list.
pub fn estimate_total_tokens(messages: &[Message]) -> usize {
    messages.iter().map(estimate_tokens).sum()
}

// =============================================================================
// EVICTION STRATEGY
// =============================================================================

/// Strategy for reducing context size when over budget.
///
/// Implementations decide which messages to remove or transform
/// when the context exceeds `max_tokens`.
///
/// # Design Notes
///
/// This is a trait (not an enum) to allow custom strategies.
/// Users might want domain-specific eviction (e.g., keep all tool
/// results, summarize only assistant messages).
pub trait EvictionStrategy: Send + Sync {
    /// Reduce messages to fit within target token budget.
    ///
    /// Called when `estimate_total_tokens(messages) > target_tokens`.
    /// Implementation should mutate `messages` in place.
    ///
    /// # Important
    ///
    /// - Never remove the last message (it's the current turn)
    /// - Never remove the first user message (sets context)
    /// - Tool results should stay paired with their tool uses
    fn evict(&self, messages: &mut Vec<Message>, target_tokens: usize);
}

// =============================================================================
// DROP OLDEST STRATEGY
// =============================================================================

/// Simple eviction: drop oldest messages first.
///
/// This is the simplest strategy. It removes messages from the
/// beginning of the conversation until we're under budget.
///
/// # Behavior
///
/// - Preserves the first message (initial context)
/// - Preserves the last message (current turn)
/// - Removes from index 1 forward
///
/// # Trade-offs
///
/// **Pros:**
/// - Simple, predictable
/// - No LLM calls needed
/// - Fast
///
/// **Cons:**
/// - Loses context abruptly
/// - No summary of what was discussed
/// - Can break tool use pairs if not careful
#[derive(Debug, Clone, Default)]
pub struct DropOldest;

impl EvictionStrategy for DropOldest {
    fn evict(&self, messages: &mut Vec<Message>, target_tokens: usize) {
        // Keep removing from index 1 until under budget
        // (preserve first and last messages)
        while messages.len() > 2 && estimate_total_tokens(messages) > target_tokens {
            messages.remove(1);
        }
    }
}

// =============================================================================
// SLIDING WINDOW STRATEGY
// =============================================================================

/// Keep only the N most recent messages.
///
/// A fixed-size window that always keeps the last N messages,
/// regardless of token count. Simple and predictable.
///
/// # Behavior
///
/// - Always keeps the first message (initial context)
/// - Keeps the last `window_size - 1` messages
/// - Evicts everything in between when window is exceeded
///
/// # Trade-offs
///
/// **Pros:**
/// - Predictable memory usage
/// - Simple to reason about
/// - Preserves recent context
///
/// **Cons:**
/// - May cut off mid-conversation
/// - Doesn't consider token counts
/// - First message might become stale
#[derive(Debug, Clone)]
pub struct SlidingWindow {
    /// Maximum number of messages to keep.
    window_size: usize,
}

impl SlidingWindow {
    /// Create a sliding window strategy.
    ///
    /// # Arguments
    ///
    /// - `window_size`: Max messages to keep (minimum 2: first + last)
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size.max(2),
        }
    }
}

impl EvictionStrategy for SlidingWindow {
    fn evict(&self, messages: &mut Vec<Message>, _target_tokens: usize) {
        if messages.len() <= self.window_size {
            return;
        }

        // Remove messages from index 1 until we're at window_size
        let remove_count = messages.len() - self.window_size;
        for _ in 0..remove_count {
            messages.remove(1);
        }
    }
}

// =============================================================================
// ASYNC SUMMARIZATION
// =============================================================================

/// Summarize a batch of messages using an LLM.
///
/// This is an explicit operation, not an automatic eviction strategy,
/// because it requires async LLM calls. Call this when you want to
/// compact a conversation while preserving context.
///
/// # How It Works
///
/// 1. Takes messages from index 1 to `end_index`
/// 2. Sends them to the LLM with a summarization prompt
/// 3. Replaces them with a single "summary" message
///
/// # Example
///
/// ```ignore
/// // Before: [user, assistant, user, assistant, user, assistant]
/// // After:  [user, summary_of_middle, assistant]
/// summarize_messages(&client, &mut messages, 4).await?;
/// ```
pub async fn summarize_messages(
    client: &dyn LlmClient,
    messages: &mut Vec<Message>,
    end_index: usize,
) -> Result<(), crate::LlmError> {
    if messages.len() < 3 || end_index < 2 {
        // Nothing to summarize
        return Ok(());
    }

    let end = end_index.min(messages.len() - 1);

    // Extract messages to summarize (indices 1..end)
    let to_summarize: Vec<_> = messages[1..end].to_vec();

    if to_summarize.is_empty() {
        return Ok(());
    }

    // Build summarization prompt
    let conversation_text = format_messages_for_summary(&to_summarize);

    let summary_request = Request {
        system: Some(
            "You are a helpful assistant. Summarize the following conversation \
             concisely, preserving key information, decisions, and context. \
             Be brief but complete."
                .to_string(),
        ),
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text(format!(
                "Please summarize this conversation:\n\n{}",
                conversation_text
            ))],
        }],
        tools: vec![],
        max_tokens: 500,
    };

    // Get summary from LLM
    let response = client.complete(summary_request).await?;

    let summary_text = response
        .content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(s) => Some(s.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Create summary message
    let summary_message = Message {
        role: Role::User,
        content: vec![ContentBlock::Text(format!(
            "[Previous conversation summary: {}]",
            summary_text
        ))],
    };

    // Replace summarized messages with summary
    // Remove messages 1..end, insert summary at index 1
    for _ in 1..end {
        messages.remove(1);
    }
    messages.insert(1, summary_message);

    Ok(())
}

/// Format messages for the summarization prompt.
fn format_messages_for_summary(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
            };
            let content: String = msg
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text(s) => Some(s.as_str()),
                    ContentBlock::ToolUse { name, .. } => Some(name.as_str()),
                    ContentBlock::ToolResult { content, .. } => Some(content.as_str()),
                })
                .collect::<Vec<_>>()
                .join(" ");
            format!("{}: {}", role, content)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// =============================================================================
// CONTEXT STORE
// =============================================================================

/// Storage backend for conversation messages.
///
/// Implementations handle where messages are stored between turns.
/// This enables persistence (SQLite) or simple in-memory storage.
///
/// # Session IDs
///
/// Stores are keyed by session ID, allowing multiple conversations.
/// For single-conversation use, just use a constant session ID.
pub trait ContextStore: Send + Sync {
    /// Load messages for a session.
    ///
    /// Returns empty vec if session doesn't exist.
    fn load(&self, session_id: &str) -> Vec<Message>;

    /// Save messages for a session.
    ///
    /// Overwrites any existing messages for this session.
    fn save(&self, session_id: &str, messages: &[Message]);

    /// Clear a session's messages.
    fn clear(&self, session_id: &str);

    /// List all session IDs.
    fn list_sessions(&self) -> Vec<String>;
}

// =============================================================================
// IN-MEMORY STORE
// =============================================================================

use std::collections::HashMap;
use std::sync::RwLock;

/// Simple in-memory storage for messages.
///
/// Messages are stored in a HashMap, keyed by session ID.
/// No persistence - everything is lost when the process exits.
///
/// # Thread Safety
///
/// Uses RwLock for concurrent access. Multiple readers allowed,
/// writers get exclusive access.
///
/// # When to Use
///
/// - Short-lived agents
/// - Testing and development
/// - When you don't need persistence
#[derive(Debug, Default)]
pub struct InMemoryStore {
    sessions: RwLock<HashMap<String, Vec<Message>>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ContextStore for InMemoryStore {
    fn load(&self, session_id: &str) -> Vec<Message> {
        self.sessions
            .read()
            .unwrap()
            .get(session_id)
            .cloned()
            .unwrap_or_default()
    }

    fn save(&self, session_id: &str, messages: &[Message]) {
        self.sessions
            .write()
            .unwrap()
            .insert(session_id.to_string(), messages.to_vec());
    }

    fn clear(&self, session_id: &str) {
        self.sessions.write().unwrap().remove(session_id);
    }

    fn list_sessions(&self) -> Vec<String> {
        self.sessions.read().unwrap().keys().cloned().collect()
    }
}

// =============================================================================
// SQLITE STORE (Feature: persistence)
// =============================================================================

/// SQLite-backed storage for messages.
///
/// Messages are stored in a SQLite database, persisting across process
/// restarts. Each session is a row with JSON-serialized messages.
///
/// # Schema
///
/// ```sql
/// CREATE TABLE IF NOT EXISTS sessions (
///     session_id TEXT PRIMARY KEY,
///     messages TEXT NOT NULL,  -- JSON array
///     updated_at INTEGER NOT NULL
/// );
/// ```
///
/// # Thread Safety
///
/// Uses a Mutex-wrapped connection. SQLite itself handles concurrency
/// at the file level.
///
/// # When to Use
///
/// - Long-running agents that need to resume after restart
/// - Multi-session applications (chat servers, etc.)
/// - When you need an audit trail of conversations
///
/// # Feature Flag
///
/// Requires `persistence` feature:
/// ```toml
/// agent-core = { version = "0.1", features = ["persistence"] }
/// ```
#[cfg(feature = "persistence")]
pub struct SqliteStore {
    conn: std::sync::Mutex<rusqlite::Connection>,
}

#[cfg(feature = "persistence")]
impl SqliteStore {
    /// Open or create a SQLite database for message storage.
    ///
    /// Creates the sessions table if it doesn't exist.
    ///
    /// # Arguments
    ///
    /// - `path`: Path to the SQLite database file
    ///
    /// # Errors
    ///
    /// Returns error if database cannot be opened or schema cannot be created.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, rusqlite::Error> {
        let conn = rusqlite::Connection::open(path)?;

        // Create schema
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                messages TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )",
            [],
        )?;

        Ok(Self {
            conn: std::sync::Mutex::new(conn),
        })
    }

    /// Create an in-memory SQLite database.
    ///
    /// Useful for testing - data is lost when dropped.
    pub fn in_memory() -> Result<Self, rusqlite::Error> {
        let conn = rusqlite::Connection::open_in_memory()?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                messages TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )",
            [],
        )?;

        Ok(Self {
            conn: std::sync::Mutex::new(conn),
        })
    }
}

#[cfg(feature = "persistence")]
impl ContextStore for SqliteStore {
    fn load(&self, session_id: &str) -> Vec<Message> {
        let conn = self.conn.lock().unwrap();

        let result: Result<String, _> = conn.query_row(
            "SELECT messages FROM sessions WHERE session_id = ?",
            [session_id],
            |row| row.get(0),
        );

        match result {
            Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    fn save(&self, session_id: &str, messages: &[Message]) {
        let conn = self.conn.lock().unwrap();
        let json = serde_json::to_string(messages).unwrap_or_else(|_| "[]".to_string());
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Upsert: insert or replace
        let _ = conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, messages, updated_at) VALUES (?, ?, ?)",
            rusqlite::params![session_id, json, now],
        );
    }

    fn clear(&self, session_id: &str) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute("DELETE FROM sessions WHERE session_id = ?", [session_id]);
    }

    fn list_sessions(&self) -> Vec<String> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = match conn.prepare("SELECT session_id FROM sessions ORDER BY updated_at DESC") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = stmt.query_map([], |row| row.get(0));

        match rows {
            Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
            Err(_) => Vec::new(),
        }
    }
}

// =============================================================================
// CONTEXT MANAGER
// =============================================================================

/// Manages conversation context with storage and eviction.
///
/// The ContextManager ties together:
/// - A storage backend (where messages live)
/// - An eviction strategy (how to shrink when full)
/// - A token budget (when to trigger eviction)
///
/// # Usage Pattern
///
/// ```ignore
/// // Setup
/// let manager = ContextManager::new(store, strategy, max_tokens);
///
/// // Each turn
/// manager.add_message(session, user_message);
/// let context = manager.get_messages(session); // For LLM request
/// manager.add_message(session, assistant_response);
/// ```
///
/// # Automatic Eviction
///
/// When `add_message` would exceed `max_tokens`, the eviction
/// strategy is automatically invoked before adding the message.
pub struct ContextManager<S: ContextStore, E: EvictionStrategy> {
    store: S,
    strategy: E,
    max_tokens: usize,
}

impl<S: ContextStore, E: EvictionStrategy> ContextManager<S, E> {
    /// Create a new context manager.
    ///
    /// # Arguments
    ///
    /// - `store`: Where to persist messages
    /// - `strategy`: How to evict when over budget
    /// - `max_tokens`: Token budget (eviction triggers above this)
    pub fn new(store: S, strategy: E, max_tokens: usize) -> Self {
        Self {
            store,
            strategy,
            max_tokens,
        }
    }

    /// Add a message to a session.
    ///
    /// If adding would exceed the token budget, eviction is
    /// triggered first to make room.
    pub fn add_message(&self, session_id: &str, message: Message) {
        let mut messages = self.store.load(session_id);
        messages.push(message);

        // Check if we need to evict
        if estimate_total_tokens(&messages) > self.max_tokens {
            self.strategy.evict(&mut messages, self.max_tokens);
        }

        self.store.save(session_id, &messages);
    }

    /// Get all messages for a session.
    pub fn get_messages(&self, session_id: &str) -> Vec<Message> {
        self.store.load(session_id)
    }

    /// Get current token count for a session.
    pub fn get_token_count(&self, session_id: &str) -> usize {
        estimate_total_tokens(&self.store.load(session_id))
    }

    /// Clear a session.
    pub fn clear(&self, session_id: &str) {
        self.store.clear(session_id);
    }

    /// Get remaining token budget for a session.
    pub fn remaining_tokens(&self, session_id: &str) -> usize {
        let used = self.get_token_count(session_id);
        self.max_tokens.saturating_sub(used)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_message(role: Role, text: &str) -> Message {
        Message {
            role,
            content: vec![ContentBlock::Text(text.to_string())],
        }
    }

    #[test]
    fn test_estimate_tokens() {
        let msg = make_message(Role::User, "Hello, world!"); // 13 chars
        let tokens = estimate_tokens(&msg);
        assert_eq!(tokens, 3); // 13 / 4 = 3
    }

    #[test]
    fn test_in_memory_store() {
        let store = InMemoryStore::new();

        // Initially empty
        assert!(store.load("session1").is_empty());

        // Save and load
        let messages = vec![make_message(Role::User, "Hello")];
        store.save("session1", &messages);
        assert_eq!(store.load("session1").len(), 1);

        // Different sessions are independent
        assert!(store.load("session2").is_empty());

        // Clear
        store.clear("session1");
        assert!(store.load("session1").is_empty());

        // List sessions
        store.save("a", &messages);
        store.save("b", &messages);
        let sessions = store.list_sessions();
        assert!(sessions.contains(&"a".to_string()));
        assert!(sessions.contains(&"b".to_string()));
    }

    #[test]
    fn test_drop_oldest_strategy() {
        // Use longer messages to get meaningful token counts
        // Each message ~50 chars = ~12 tokens
        let mut messages = vec![
            make_message(Role::User, "This is the first message in the conversation here"),
            make_message(Role::Assistant, "This is the second message, a response from assistant"),
            make_message(Role::User, "This is the third message from the user again"),
            make_message(Role::Assistant, "This is the fourth and final message in sequence"),
        ];

        let initial_tokens = estimate_total_tokens(&messages);
        assert!(initial_tokens > 40, "Should have >40 tokens, got {}", initial_tokens);

        // Force eviction to budget that requires dropping messages
        // Target ~25 tokens, should keep first + last (~24 tokens)
        DropOldest.evict(&mut messages, 25);

        // Should keep first and last
        assert_eq!(messages.len(), 2, "Expected 2 messages after eviction");
        assert!(matches!(&messages[0].content[0], ContentBlock::Text(s) if s.contains("first")));
        assert!(matches!(&messages[1].content[0], ContentBlock::Text(s) if s.contains("fourth")));
    }

    #[test]
    fn test_context_manager_basic() {
        let manager = ContextManager::new(InMemoryStore::new(), DropOldest, 1000);

        manager.add_message("s1", make_message(Role::User, "Hello"));
        manager.add_message("s1", make_message(Role::Assistant, "Hi there!"));

        let messages = manager.get_messages("s1");
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_context_manager_eviction() {
        // Budget that can hold ~2 messages (~25 tokens)
        let manager = ContextManager::new(InMemoryStore::new(), DropOldest, 30);

        // Add messages that will exceed budget (each ~12 tokens)
        manager.add_message("s1", make_message(Role::User, "This is the first message in our conversation"));
        manager.add_message("s1", make_message(Role::Assistant, "This is the second message as a response"));
        manager.add_message("s1", make_message(Role::User, "This is the third message from user"));
        manager.add_message("s1", make_message(Role::Assistant, "This is the fourth message final one"));

        let messages = manager.get_messages("s1");

        // Should have evicted some messages to stay under 30 tokens
        assert!(messages.len() < 4, "Expected <4 messages, got {}", messages.len());
        // But should still have at least 2 (first + last)
        assert!(messages.len() >= 2, "Expected >=2 messages, got {}", messages.len());
    }

    #[test]
    fn test_remaining_tokens() {
        let manager = ContextManager::new(InMemoryStore::new(), DropOldest, 100);

        assert_eq!(manager.remaining_tokens("s1"), 100);

        manager.add_message("s1", make_message(Role::User, "Hello world")); // ~3 tokens

        assert!(manager.remaining_tokens("s1") < 100);
        assert!(manager.remaining_tokens("s1") > 90);
    }

    #[test]
    fn test_sliding_window_strategy() {
        let mut messages = vec![
            make_message(Role::User, "First message - this is the initial context"),
            make_message(Role::Assistant, "Second message - assistant response one"),
            make_message(Role::User, "Third message - user follow up question"),
            make_message(Role::Assistant, "Fourth message - assistant response two"),
            make_message(Role::User, "Fifth message - another user question"),
        ];

        // Window of 3: keep first + last 2
        let window = SlidingWindow::new(3);
        window.evict(&mut messages, 0); // target_tokens ignored by SlidingWindow

        assert_eq!(messages.len(), 3);
        // First message preserved
        assert!(matches!(&messages[0].content[0], ContentBlock::Text(s) if s.contains("First")));
        // Last two messages preserved
        assert!(matches!(&messages[1].content[0], ContentBlock::Text(s) if s.contains("Fourth")));
        assert!(matches!(&messages[2].content[0], ContentBlock::Text(s) if s.contains("Fifth")));
    }

    #[test]
    fn test_sliding_window_under_limit() {
        let mut messages = vec![
            make_message(Role::User, "First message"),
            make_message(Role::Assistant, "Second message"),
        ];

        // Window of 5, but only 2 messages - nothing should be evicted
        let window = SlidingWindow::new(5);
        window.evict(&mut messages, 0);

        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_sliding_window_minimum_size() {
        // Window size < 2 should clamp to 2
        let window = SlidingWindow::new(1);

        let mut messages = vec![
            make_message(Role::User, "First"),
            make_message(Role::Assistant, "Second"),
            make_message(Role::User, "Third"),
        ];

        window.evict(&mut messages, 0);

        // Should keep first + last = 2 messages
        assert_eq!(messages.len(), 2);
    }

    #[test]
    fn test_context_manager_with_sliding_window() {
        // Window of 3 messages
        // Use small token budget (~12 tokens) to trigger eviction after 3-4 messages
        // Each message is ~11 chars = ~3 tokens
        let manager = ContextManager::new(InMemoryStore::new(), SlidingWindow::new(3), 12);

        manager.add_message("s1", make_message(Role::User, "Message one"));
        manager.add_message("s1", make_message(Role::Assistant, "Message two"));
        manager.add_message("s1", make_message(Role::User, "Message three"));
        manager.add_message("s1", make_message(Role::Assistant, "Message four"));
        manager.add_message("s1", make_message(Role::User, "Message five"));

        let messages = manager.get_messages("s1");

        // Should have evicted to window size (once token budget exceeded)
        assert_eq!(messages.len(), 3);
    }

    #[tokio::test]
    async fn test_summarize_messages() {
        use crate::mock::{MockClient, MockResponse};

        // Mock client that returns a summary
        let client = MockClient::new(vec![MockResponse::text(
            "The user asked about Rust and the assistant explained ownership.",
        )]);

        let mut messages = vec![
            make_message(Role::User, "Tell me about Rust programming"),
            make_message(Role::Assistant, "Rust is a systems programming language focused on safety."),
            make_message(Role::User, "What about ownership?"),
            make_message(Role::Assistant, "Ownership is Rust's key feature for memory safety."),
            make_message(Role::User, "Thanks! Now let's talk about something else."),
        ];

        // Summarize messages 1..4 (middle 3 messages)
        summarize_messages(&client, &mut messages, 4).await.unwrap();

        // Should have: [first message, summary, last message]
        assert_eq!(messages.len(), 3);

        // First message unchanged
        assert!(matches!(&messages[0].content[0], ContentBlock::Text(s) if s.contains("Rust programming")));

        // Second message is summary
        assert!(matches!(&messages[1].content[0], ContentBlock::Text(s) if s.contains("summary")));

        // Last message unchanged
        assert!(matches!(&messages[2].content[0], ContentBlock::Text(s) if s.contains("something else")));
    }

    #[tokio::test]
    async fn test_summarize_messages_too_short() {
        use crate::mock::MockClient;

        // Client should not be called - empty responses would error if called
        let client = MockClient::new(vec![]);

        let mut messages = vec![
            make_message(Role::User, "Hello"),
            make_message(Role::Assistant, "Hi"),
        ];

        // Try to summarize, but too few messages
        let result = summarize_messages(&client, &mut messages, 2).await;

        assert!(result.is_ok());
        // Messages unchanged
        assert_eq!(messages.len(), 2);
    }

    // =========================================================================
    // SQLITE TESTS (Feature: persistence)
    // =========================================================================

    #[cfg(feature = "persistence")]
    mod sqlite_tests {
        use super::*;

        #[test]
        fn test_sqlite_store_basic() {
            let store = SqliteStore::in_memory().unwrap();

            // Initially empty
            assert!(store.load("session1").is_empty());

            // Save and load
            let messages = vec![make_message(Role::User, "Hello from SQLite")];
            store.save("session1", &messages);

            let loaded = store.load("session1");
            assert_eq!(loaded.len(), 1);
            assert!(matches!(&loaded[0].content[0], ContentBlock::Text(s) if s == "Hello from SQLite"));
        }

        #[test]
        fn test_sqlite_store_multiple_sessions() {
            let store = SqliteStore::in_memory().unwrap();

            store.save("a", &vec![make_message(Role::User, "Session A")]);
            store.save("b", &vec![make_message(Role::User, "Session B")]);

            assert_eq!(store.load("a").len(), 1);
            assert_eq!(store.load("b").len(), 1);

            // Sessions are independent
            let sessions = store.list_sessions();
            assert!(sessions.contains(&"a".to_string()));
            assert!(sessions.contains(&"b".to_string()));
        }

        #[test]
        fn test_sqlite_store_overwrite() {
            let store = SqliteStore::in_memory().unwrap();

            store.save("s", &vec![make_message(Role::User, "First")]);
            store.save("s", &vec![make_message(Role::User, "Second")]);

            let loaded = store.load("s");
            assert_eq!(loaded.len(), 1);
            assert!(matches!(&loaded[0].content[0], ContentBlock::Text(s) if s == "Second"));
        }

        #[test]
        fn test_sqlite_store_clear() {
            let store = SqliteStore::in_memory().unwrap();

            store.save("s", &vec![make_message(Role::User, "Test")]);
            assert_eq!(store.load("s").len(), 1);

            store.clear("s");
            assert!(store.load("s").is_empty());
        }

        #[test]
        fn test_sqlite_store_with_context_manager() {
            let store = SqliteStore::in_memory().unwrap();
            let manager = ContextManager::new(store, DropOldest, 1000);

            manager.add_message("s1", make_message(Role::User, "Hello"));
            manager.add_message("s1", make_message(Role::Assistant, "Hi there!"));

            let messages = manager.get_messages("s1");
            assert_eq!(messages.len(), 2);
        }

        #[test]
        fn test_sqlite_store_file_persistence() {
            use tempfile::NamedTempFile;

            let temp_file = NamedTempFile::new().unwrap();
            let path = temp_file.path().to_path_buf();

            // Write to file
            {
                let store = SqliteStore::open(&path).unwrap();
                store.save("s1", &vec![make_message(Role::User, "Persisted message")]);
            }

            // Read from file (new connection)
            {
                let store = SqliteStore::open(&path).unwrap();
                let loaded = store.load("s1");
                assert_eq!(loaded.len(), 1);
                assert!(matches!(&loaded[0].content[0], ContentBlock::Text(s) if s == "Persisted message"));
            }
        }
    }
}
