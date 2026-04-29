//! Integration tests for the `.r8` Brain format.

use axel::r8::Brain;
use rusqlite::{Connection, OpenFlags};
use tempfile::TempDir;

fn fresh_brain() -> (TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.r8");
    (dir, path)
}

#[test]
fn create_brain_and_verify_tables() {
    let (_dir, path) = fresh_brain();
    let brain = Brain::create(&path, Some("agent")).unwrap();

    let tables: Vec<String> = {
        let mut stmt = brain
            .conn()
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap();
        stmt.query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
    };

    // Brain creates only Memkoshi + Axel tables.
    // VelociRAG tables (documents, nodes, edges, etc.) are added later
    // via velocirag::db::Database::from_connection().
    let expected = [
        "brain_meta",
        "context_data",
        "events",
        "memories",
        "memory_access",
        "patterns",
        "sessions",
        "staged_memories",
    ];

    for t in expected {
        assert!(
            tables.iter().any(|x| x == t),
            "missing table: {t} (have: {tables:?})"
        );
    }
}

#[test]
fn insert_memory_and_retrieve() {
    let (_dir, path) = fresh_brain();
    let brain = Brain::create(&path, None).unwrap();

    brain.conn().execute(
        "INSERT INTO memories (id, category, topic, title, content, importance, confidence, created)
         VALUES ('mem_aabbccdd', 'events', 'rust', 'Shipped Brain Tests', 'integration test memory body', 0.75, 'high', datetime('now'))",
        [],
    ).unwrap();

    let (id, category, title, importance, confidence): (String, String, String, f64, String) =
        brain
            .conn()
            .query_row(
                "SELECT id, category, title, importance, confidence FROM memories WHERE id = 'mem_aabbccdd'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
            )
            .unwrap();

    assert_eq!(id, "mem_aabbccdd");
    assert_eq!(category, "events");
    assert_eq!(title, "Shipped Brain Tests");
    assert!((importance - 0.75).abs() < f64::EPSILON);
    assert_eq!(confidence, "high");
}

#[test]
fn brain_metadata_persists() {
    let (_dir, path) = fresh_brain();

    let created_at;
    let dim;
    let model;
    {
        let b = Brain::create(&path, Some("persist-agent")).unwrap();
        created_at = b.meta().created.clone();
        dim = b.meta().embedding_dim;
        model = b.meta().embedder_model.clone();
    }

    let b2 = Brain::open(&path).unwrap();
    assert_eq!(b2.meta().agent_name.as_deref(), Some("persist-agent"));
    assert_eq!(b2.meta().created, created_at);
    assert_eq!(b2.meta().embedding_dim, dim);
    assert_eq!(b2.meta().embedder_model, model);
}

#[test]
fn touch_updates_memory_count() {
    let (_dir, path) = fresh_brain();
    let mut brain = Brain::create(&path, None).unwrap();

    for i in 0..3 {
        brain.conn().execute(
            "INSERT INTO memories (id, category, topic, title, content, created)
             VALUES (?1, 'events', 'topic', 'Title', 'content body', datetime('now'))",
            rusqlite::params![format!("mem_{i:08x}")],
        ).unwrap();
    }

    assert_eq!(brain.meta().memory_count, 0);
    brain.touch().unwrap();
    assert_eq!(brain.meta().memory_count, 3);

    // Verify persisted
    drop(brain);
    let reopened = Brain::open(&path).unwrap();
    assert_eq!(reopened.meta().memory_count, 3);
}

#[test]
fn concurrent_read_after_write() {
    let (_dir, path) = fresh_brain();
    let writer = Brain::create(&path, None).unwrap();

    writer.conn().execute(
        "INSERT INTO memories (id, category, topic, title, content, created)
         VALUES ('mem_shared01', 'events', 'test', 'Shared', 'shared content', datetime('now'))",
        [],
    ).unwrap();

    // Open a second, read-only connection. WAL mode allows this concurrently.
    let reader = Connection::open_with_flags(
        &path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )
    .unwrap();

    let count: i64 = reader
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);

    let content: String = reader
        .query_row(
            "SELECT content FROM memories WHERE id = 'mem_shared01'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(content, "shared content");
}

#[test]
fn empty_brain_stats() {
    let (_dir, path) = fresh_brain();
    let brain = Brain::create(&path, None).unwrap();

    let mems: i64 = brain
        .conn()
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(mems, 0);
    assert_eq!(brain.meta().document_count, 0);
    assert_eq!(brain.meta().memory_count, 0);
}

#[test]
fn context_data_crud() {
    let (_dir, path) = fresh_brain();
    let brain = Brain::create(&path, None).unwrap();

    brain.conn().execute(
        "INSERT INTO context_data (key, value, layer) VALUES ('cwd', '/tmp', 'session')",
        [],
    ).unwrap();

    let v: String = brain.conn().query_row(
        "SELECT value FROM context_data WHERE key = 'cwd' AND layer = 'session'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(v, "/tmp");

    // Update
    brain.conn().execute(
        "UPDATE context_data SET value = '/home' WHERE key = 'cwd' AND layer = 'session'",
        [],
    ).unwrap();
    let v: String = brain.conn().query_row(
        "SELECT value FROM context_data WHERE key = 'cwd' AND layer = 'session'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(v, "/home");

    // Different layer — composite PK allows same key in different layers
    brain.conn().execute(
        "INSERT INTO context_data (key, value, layer) VALUES ('cwd', '/etc', 'boot')",
        [],
    ).unwrap();
    let count: i64 = brain.conn().query_row(
        "SELECT COUNT(*) FROM context_data WHERE key = 'cwd'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(count, 2);
}

#[test]
fn event_logging() {
    let (_dir, path) = fresh_brain();
    let brain = Brain::create(&path, None).unwrap();
    let conn = brain.conn();

    conn.execute("INSERT INTO events (event_type, target_id, query) VALUES ('search', NULL, 'rust')", []).unwrap();
    conn.execute("INSERT INTO events (event_type, target_id, query) VALUES ('search', NULL, 'sqlite')", []).unwrap();
    conn.execute("INSERT INTO events (event_type, target_id, query) VALUES ('access', 'mem_1', NULL)", []).unwrap();

    let searches: i64 = conn.query_row("SELECT COUNT(*) FROM events WHERE event_type = 'search'", [], |r| r.get(0)).unwrap();
    assert_eq!(searches, 2);

    let accesses: i64 = conn.query_row("SELECT COUNT(*) FROM events WHERE event_type = 'access'", [], |r| r.get(0)).unwrap();
    assert_eq!(accesses, 1);
}

#[test]
fn staged_memory_lifecycle() {
    let (_dir, path) = fresh_brain();
    let brain = Brain::create(&path, None).unwrap();
    let conn = brain.conn();

    conn.execute(
        "INSERT INTO staged_memories
            (id, category, topic, title, content, created, staged_at, review_status)
         VALUES
            ('mem_stage01', 'events', 'topic', 'Stage Title', 'staged content body', datetime('now'), datetime('now'), 'pending')",
        [],
    ).unwrap();

    let staged_count: i64 = conn.query_row("SELECT COUNT(*) FROM staged_memories", [], |r| r.get(0)).unwrap();
    assert_eq!(staged_count, 1);

    // Approve: copy to memories, delete from staged
    conn.execute(
        "INSERT INTO memories (id, category, topic, title, content, created)
         SELECT id, category, topic, title, content, created
         FROM staged_memories WHERE id = 'mem_stage01'",
        [],
    ).unwrap();
    conn.execute("DELETE FROM staged_memories WHERE id = 'mem_stage01'", []).unwrap();

    let staged_count: i64 = conn.query_row("SELECT COUNT(*) FROM staged_memories", [], |r| r.get(0)).unwrap();
    assert_eq!(staged_count, 0);

    let in_memories: i64 = conn.query_row("SELECT COUNT(*) FROM memories WHERE id = 'mem_stage01'", [], |r| r.get(0)).unwrap();
    assert_eq!(in_memories, 1);
}
