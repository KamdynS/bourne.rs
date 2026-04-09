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

use crate::{ContentBlock, Message, Role};

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
}
