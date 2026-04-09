//! Context management demonstration.
//!
//! Shows how to use ContextManager with different stores and strategies.
//!
//! Run with: cargo run --example context_demo

use agent_core::{
    ContentBlock, ContextManager, DropOldest, InMemoryStore, Message, Role, SlidingWindow,
};

fn main() {
    println!("=== Context Management Demo ===\n");

    // ---------------------------------------------------------------------
    // Example 1: DropOldest strategy
    // Removes oldest messages when over token budget
    // ---------------------------------------------------------------------
    println!("1. DropOldest Strategy");
    println!("   Removes oldest messages to stay under token budget.\n");

    // Small budget to trigger eviction quickly (each message ~8 tokens)
    let manager = ContextManager::new(InMemoryStore::new(), DropOldest, 25);

    // Simulate a conversation
    let messages = [
        (Role::User, "Hello, I need help with Rust."),
        (Role::Assistant, "I'd be happy to help! What's your question?"),
        (Role::User, "How do I use iterators?"),
        (Role::Assistant, "Iterators in Rust are lazy and chainable..."),
        (Role::User, "Can you show me map and filter?"),
    ];

    for (role, text) in messages {
        manager.add_message(
            "session1",
            Message {
                role,
                content: vec![ContentBlock::Text(text.to_string())],
            },
        );
    }

    let remaining = manager.get_messages("session1");
    println!("   Added {} messages, {} remain after eviction", messages.len(), remaining.len());
    println!("   Token budget: 25, used: {}\n", manager.get_token_count("session1"));

    // ---------------------------------------------------------------------
    // Example 2: SlidingWindow strategy
    // Keeps only the N most recent messages
    // ---------------------------------------------------------------------
    println!("2. SlidingWindow Strategy");
    println!("   Keeps first message + last N-1 messages.\n");

    // Window of 3: first message + last 2
    // Note: Eviction triggers when token budget exceeded, so we set a low budget
    let manager = ContextManager::new(InMemoryStore::new(), SlidingWindow::new(3), 20);

    for (role, text) in messages {
        manager.add_message(
            "session2",
            Message {
                role,
                content: vec![ContentBlock::Text(text.to_string())],
            },
        );
    }

    let remaining = manager.get_messages("session2");
    println!("   Window size: 3");
    println!("   Added {} messages, {} remain\n", messages.len(), remaining.len());

    // Show which messages were kept
    println!("   Kept messages:");
    for msg in &remaining {
        let text = match &msg.content[0] {
            ContentBlock::Text(s) => s.as_str(),
            _ => "...",
        };
        let role = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
        };
        println!("   - {}: {}...", role, &text[..text.len().min(40)]);
    }

    // ---------------------------------------------------------------------
    // Example 3: Multiple sessions
    // Each session has independent context
    // ---------------------------------------------------------------------
    println!("\n3. Multiple Sessions");
    println!("   Each session is independent.\n");

    let manager = ContextManager::new(InMemoryStore::new(), DropOldest, 1000);

    manager.add_message(
        "alice",
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hi, I'm Alice!".to_string())],
        },
    );

    manager.add_message(
        "bob",
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text("Hello, I'm Bob!".to_string())],
        },
    );

    println!("   Alice's session: {} message(s)", manager.get_messages("alice").len());
    println!("   Bob's session: {} message(s)", manager.get_messages("bob").len());

    // ---------------------------------------------------------------------
    // Example 4: SQLite persistence (feature-gated)
    // ---------------------------------------------------------------------
    #[cfg(feature = "persistence")]
    {
        use agent_core::SqliteStore;

        println!("\n4. SQLite Persistence");
        println!("   Messages survive restarts.\n");

        let store = SqliteStore::in_memory().expect("Failed to create SQLite store");
        let manager = ContextManager::new(store, DropOldest, 1000);

        manager.add_message(
            "persistent",
            Message {
                role: Role::User,
                content: vec![ContentBlock::Text("This would persist to disk!".to_string())],
            },
        );

        println!("   Stored message in SQLite (in-memory for demo)");
        println!("   For real persistence: SqliteStore::open(\"agent.db\")");
    }

    #[cfg(not(feature = "persistence"))]
    {
        println!("\n4. SQLite Persistence");
        println!("   (Enable with: --features persistence)");
    }

    println!("\n=== Demo Complete ===");
}
