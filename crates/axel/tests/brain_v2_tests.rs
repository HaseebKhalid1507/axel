//! Section 1 of the v2 spec, "The brain": tests 1 to 5, 7, 8, 17 (the function half),
//! 18 to 20 and 23, and `degrade_on` (4.2). Every brain is in a temp folder and every
//! path is passed in: nothing here resolves the default path.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use axel::brain_v2::{self, OpenOpts, Role};
use axel::error::AxelError;
use axel::ext::types::{DegradedReason, Mode, Status};
use axel::project_memory::{self, MemoryScope, Retention, Sensitivity};
use axel::r8::{Brain, DEFAULT_EMBEDDER};
use axel::search::BrainSearch;
use axel_memkoshi::memory::{Memory, MemoryCategory};
use axel_memkoshi::storage::MemoryStorage;
use rusqlite::{params, Connection};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use velocirag::db::Database;
use velocirag::embedder::Embedder;

const SCHEMA_V2_SQL: &str = include_str!("fixtures/schema_v2.sql");
const KV_GUARD_MESSAGE: &str = "this brain belongs to Axel v2; use the axel-v2 binary";
const DELETE_GUARD_MESSAGE: &str = "this brain belongs to Axel v2; delete with axel-v2 forget";
const PROJECT: &str = "p1111111111111111";

const V2_TABLES: [&str; 12] = [
    "axel_migrations",
    "memory_ext",
    "axel_purge",
    "archive_index",
    "projects",
    "note_files",
    "chat_pauses",
    "forgotten_turns",
    "forum_posts",
    "forum_tombstones",
    "recall_log",
    "import_log",
];

// ── Helpers ─────────────────────────────────────────────────────────────

fn opts(upgrade_v1: bool) -> OpenOpts<'static> {
    OpenOpts { agent: Some("test"), model: "hash-384", dim: 384, upgrade_v1 }
}

/// SHA-256 of the main file only (B-7.4: a read-only open may make side files).
fn main_hash(path: &Path) -> String {
    hex::encode(Sha256::digest(std::fs::read(path).unwrap()))
}

fn scope() -> MemoryScope {
    MemoryScope {
        project_key: PROJECT.to_string(),
        sensitivity: Sensitivity::Normal,
        retention: Retention::Standard,
        provenance: "test".to_string(),
    }
}

fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |r| r.get(0)).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn count_id(conn: &Connection, sql: &str, id: &str) -> i64 {
    conn.query_row(sql, [id], |r| r.get(0)).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn word_hits(conn: &Connection, word: &str) -> Vec<String> {
    let mut stmt = conn
        .prepare("SELECT mem_id FROM memories_fts WHERE memories_fts MATCH ?1 ORDER BY mem_id")
        .unwrap();
    let rows = stmt.query_map([word], |r| r.get(0)).unwrap();
    rows.map(|r| r.unwrap()).collect()
}

fn table_counts(conn: &Connection) -> Vec<(String, i64)> {
    let names: Vec<String> = {
        let mut stmt = conn
            .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
            .unwrap();
        let rows = stmt.query_map([], |r| r.get(0)).unwrap();
        rows.map(|r| r.unwrap()).collect()
    };
    names
        .into_iter()
        .map(|n| {
            let c = count(conn, &format!("SELECT COUNT(*) FROM \"{n}\""));
            (n, c)
        })
        .collect()
}

fn checkpoint_truncate(conn: &Connection, path: &Path) {
    let busy: i64 = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0)).unwrap();
    assert_eq!(busy, 0, "the checkpoint must complete");
    let wal = std::fs::metadata(format!("{}-wal", path.display())).map(|m| m.len()).unwrap_or(0);
    assert_eq!(wal, 0, "TRUNCATE must leave an empty log");
}

/// A version 1 brain as `Brain::create` makes it: no rows, no velocirag tables.
fn v1_empty(dir: &TempDir, name: &str) -> PathBuf {
    let path = dir.path().join(name);
    drop(Brain::create(&path, Some("old")).unwrap());
    path
}

/// A memory written the way `AxelBrain::remember` stores it: the `memories` row, no word row.
fn remember_plain(path: &Path, title: &str, content: &str) -> Memory {
    let memory = Memory::new(MemoryCategory::Events, "test", title, content);
    let storage = MemoryStorage::open_existing(path).unwrap();
    storage.store_memory(&memory).unwrap();
    memory
}

/// A legacy-shaped file: `Brain::create` and velocirag's tables, 3 documents (two of them
/// the shadows of the 2 memories, each with its own vector), no `memories_fts`.
fn legacy_file(dir: &TempDir) -> (PathBuf, Memory, Memory) {
    let path = v1_empty(dir, "axel.r8");
    let a = remember_plain(&path, "harbour note", "the quokka crossed the harbour at dusk");
    let b = remember_plain(&path, "lantern note", "a narwhal carried the lantern home");
    let db = Database::from_connection(Connection::open(&path).unwrap(), path.clone())
        .unwrap_or_else(|e| panic!("velocirag schema: {e}"));
    let meta = serde_json::json!({});
    let dim = velocirag::EMBEDDING_DIM;
    for (doc_id, content, vector, file) in [
        (a.id.as_str(), "shadow of a", vec![0.25f32; dim], None),
        (b.id.as_str(), "shadow of b", vec![-0.5f32; dim], None),
        ("note-1", "a note file about tides", vec![0.0f32; dim], Some("/notes/tides.md")),
    ] {
        db.insert_document(doc_id, content, &meta, &vector, file)
            .unwrap_or_else(|e| panic!("insert {doc_id}: {e}"));
    }
    assert_eq!(count(db.conn(), "SELECT COUNT(*) FROM sqlite_master WHERE name = 'memories_fts'"), 0);
    drop(db);
    (path, a, b)
}

// ── 1 ───────────────────────────────────────────────────────────────────

#[test]
fn t01_open_v2_on_a_missing_path_creates_a_v2_brain() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("made").join("brain.r8");
    let brain = brain_v2::open_v2(&path, &opts(false)).unwrap();
    let conn = &brain.conn;

    assert_eq!(count(conn, "PRAGMA user_version"), 1);
    assert_eq!(brain_v2::LATEST, 1);
    for table in V2_TABLES {
        assert_eq!(
            count_id(conn, "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1", table),
            1,
            "table {table}"
        );
    }
    let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600, "file mode {mode:o}");
    let dir_mode = std::fs::metadata(path.parent().unwrap()).unwrap().permissions().mode() & 0o777;
    assert_eq!(dir_mode, 0o700, "directory mode {dir_mode:o}");

    let (version, model): (i64, String) = conn
        .query_row(
            "SELECT json_extract(value, '$.schema_version'), json_extract(value, '$.embedder_model')
             FROM brain_meta WHERE key = 'meta'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(version, 2);
    assert_eq!(model, "hash-384");
    assert_eq!(brain.embedder_model, "hash-384");
    assert_eq!(brain.path, path);
    assert!(brain.signer.is_some());
    assert!(brain_v2::signer(conn).unwrap().is_some(), "the front thread's signer");

    // The rest of migration 1.
    for column in ["source_mtime_ms", "hidden"] {
        assert_eq!(
            count_id(conn, "SELECT COUNT(*) FROM pragma_table_info('documents') WHERE name = ?1", column),
            1,
            "documents.{column}"
        );
    }
    for table in ["memories_fts_config", "chunks_fts_config"] {
        assert_eq!(count(conn, &format!("SELECT v FROM {table} WHERE k = 'secure-delete'")), 1, "{table}");
    }
    assert_eq!(count(conn, "PRAGMA secure_delete"), 1);
    let kv: String =
        conn.query_row("SELECT value FROM kv WHERE key = 'schema_version'", [], |r| r.get(0)).unwrap();
    assert_eq!(kv, "3");
    drop(brain);

    // So a hash-384 embedder is accepted on it, and the old reader opens it.
    if let Err(e) = BrainSearch::open_with_embedder(&path, Embedder::hashing()) {
        panic!("the hash embedder must be accepted on a brain made with its model: {e}");
    }
    assert_eq!(Brain::open(&path).unwrap().meta().schema_version, 2);
    assert!(brain_v2::require_v2(&path).is_ok());
}

// ── 2 ───────────────────────────────────────────────────────────────────

#[test]
fn t02_open_v2_twice_migrates_once_and_changes_no_count() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.r8");
    let first = brain_v2::open_v2(&path, &opts(false)).unwrap();
    let memory = Memory::new(MemoryCategory::Events, "test", "kept", "a row that both opens see");
    project_memory::store_scoped(&first.conn, &memory, &scope()).unwrap();
    assert_eq!(brain_v2::adopt_orphans(&first.conn, "cli").unwrap(), 1);
    let before = table_counts(&first.conn);
    drop(first);

    let second = brain_v2::open_v2(&path, &opts(false)).unwrap();
    assert_eq!(count(&second.conn, "SELECT COUNT(*) FROM axel_migrations"), 1);
    assert_eq!(count(&second.conn, "SELECT version FROM axel_migrations"), 1);
    assert_eq!(count(&second.conn, "PRAGMA user_version"), 1);
    assert_eq!(table_counts(&second.conn), before);
}

// ── 3 ───────────────────────────────────────────────────────────────────

#[test]
fn t03_a_newer_or_unreadable_file_is_refused_unwritten() {
    let dir = TempDir::new().unwrap();

    let newer = v1_empty(&dir, "newer.r8");
    let conn = Connection::open(&newer).unwrap();
    conn.execute_batch("PRAGMA user_version = 99").unwrap();
    drop(conn);
    let hash = main_hash(&newer);
    match brain_v2::open_v2(&newer, &opts(false)) {
        Err(AxelError::BrainTooNew { file: 99, binary: 1 }) => {}
        Err(e) => panic!("expected BrainTooNew, got {e:?}"),
        Ok(_) => panic!("expected BrainTooNew, got a brain"),
    }
    assert!(matches!(brain_v2::require_v2(&newer), Err(AxelError::BrainTooNew { file: 99, binary: 1 })));
    assert!(matches!(brain_v2::connect(&newer, Role::Cli), Err(AxelError::BrainNotReady)));
    assert_eq!(main_hash(&newer), hash);

    // Random bytes: a fixed generator, so the file is the same on every run.
    let noise = dir.path().join("noise.r8");
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let bytes: Vec<u8> = (0..8192)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u8
        })
        .collect();
    std::fs::write(&noise, &bytes).unwrap();
    let hash = main_hash(&noise);
    match brain_v2::open_v2(&noise, &opts(false)) {
        Err(AxelError::BrainDamaged(_)) => {}
        Err(e) => panic!("expected BrainDamaged, got {e:?}"),
        Ok(_) => panic!("expected BrainDamaged, got a brain"),
    }
    assert!(matches!(brain_v2::open_v2(&noise, &opts(true)), Err(AxelError::BrainDamaged(_))));
    assert!(matches!(brain_v2::require_v2(&noise), Err(AxelError::BrainDamaged(_))));
    assert_eq!(main_hash(&noise), hash);
    assert_eq!(std::fs::read(&noise).unwrap(), bytes);
}

// ── 4 ───────────────────────────────────────────────────────────────────

#[test]
fn t04_upgrade_v1_keeps_every_row_and_adopts_the_memories_with_their_vectors() {
    let dir = TempDir::new().unwrap();
    let (path, a, b) = legacy_file(&dir);

    let brain = brain_v2::open_v2(&path, &opts(true)).unwrap();
    let conn = &brain.conn;
    assert_eq!(brain.embedder_model, DEFAULT_EMBEDDER, "the model the old brain recorded");
    assert_eq!(count(conn, "PRAGMA user_version"), 1);
    assert_eq!(count(conn, "SELECT json_extract(value, '$.schema_version') FROM brain_meta WHERE key = 'meta'"), 2);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM documents"), 3);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM chunks_fts"), 3);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memories"), 2);
    for table in V2_TABLES {
        assert_eq!(
            count_id(conn, "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1", table),
            1,
            "table {table}"
        );
    }

    for (memory, word) in [(&a, "quokka"), (&b, "narwhal")] {
        let content: String = conn
            .query_row("SELECT content FROM memories WHERE id = ?1", [&memory.id], |r| r.get(0))
            .unwrap();
        assert_eq!(content, memory.content);

        let (origin, origin_ref, hash, created_ms, time_source, model, same_vector, bytes): (
            String,
            String,
            String,
            i64,
            String,
            Option<String>,
            bool,
            i64,
        ) = conn
            .query_row(
                "SELECT e.origin_kind, e.origin_ref, e.content_hash, e.created_ms, e.time_source,
                        e.embed_model, e.embedding = d.embedding, length(e.embedding)
                 FROM memory_ext e JOIN documents d ON d.doc_id = e.mem_id WHERE e.mem_id = ?1",
                [&memory.id],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                        r.get(7)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(origin, "legacy");
        assert_eq!(origin_ref, "");
        assert_eq!(hash, hex::encode(Sha256::digest(memory.content.as_bytes())));
        assert_eq!(created_ms, memory.created.timestamp_millis());
        assert_eq!(time_source, "exact");
        assert_eq!(model.as_deref(), Some(DEFAULT_EMBEDDER));
        assert!(same_vector, "the vector of the shadow document");
        assert_eq!(bytes as usize, velocirag::EMBEDDING_DIM * 4);

        assert_eq!(word_hits(conn, word), vec![memory.id.clone()], "word {word}");
    }
    // The two vectors are different ones: each memory got its own.
    assert_eq!(count(conn, "SELECT COUNT(DISTINCT embedding) FROM memory_ext"), 2);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memories_fts"), 2);
}

// ── Review finding 1: migration 1 adopts what it finds, as `legacy` ─────

fn origins(conn: &Connection) -> Vec<(String, i64)> {
    let mut stmt = conn
        .prepare("SELECT origin_kind, COUNT(*) FROM memory_ext GROUP BY origin_kind ORDER BY origin_kind")
        .unwrap();
    let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    rows.map(|r| r.unwrap()).collect()
}

#[test]
fn f1a_what_migration_1_finds_is_legacy_whoever_the_opener_is() {
    // `migrate init`: step 7 is given `legacy`.
    let dir = TempDir::new().unwrap();
    let (path, _, _) = legacy_file(&dir);
    let brain = brain_v2::open_v2(&path, &opts(true)).unwrap();
    assert_eq!(origins(&brain.conn), vec![("legacy".to_string(), 2)]);
    drop(brain);

    // An ordinary opener: step 7 is given `cli`. The only file with memories that an
    // ordinary `open_v2` migrates is one whose `brain_meta` already says 2 while
    // `user_version` is 0; the test makes it by hand.
    let dir = TempDir::new().unwrap();
    let (path, _, _) = legacy_file(&dir);
    let conn = Connection::open(&path).unwrap();
    conn.execute(
        "UPDATE brain_meta SET value = json_set(value, '$.schema_version', 2) WHERE key = 'meta'",
        [],
    )
    .unwrap();
    assert_eq!(count(&conn, "PRAGMA user_version"), 0);
    drop(conn);
    let brain = brain_v2::open_v2(&path, &opts(false)).unwrap();
    assert_eq!(count(&brain.conn, "SELECT COUNT(*) FROM axel_migrations"), 1);
    assert_eq!(origins(&brain.conn), vec![("legacy".to_string(), 2)]);
    assert_eq!(brain_v2::adopt_orphans(&brain.conn, "cli").unwrap(), 0, "nothing is left for step 7");

    // A row written later without a `memory_ext` row is still step 7's, and the others stay.
    let later = Memory::new(MemoryCategory::Events, "test", "later note", "an ibis stood on the weir");
    project_memory::store_scoped(&brain.conn, &later, &scope()).unwrap();
    assert_eq!(brain_v2::adopt_orphans(&brain.conn, "cli").unwrap(), 1);
    assert_eq!(origins(&brain.conn), vec![("cli".to_string(), 1), ("legacy".to_string(), 2)]);
}

#[test]
fn f1b_a_failure_inside_migration_1_leaves_a_version_1_brain_and_a_second_run_completes() {
    let dir = TempDir::new().unwrap();
    let (path, a, b) = legacy_file(&dir);

    // The injected failure: migration 1 updates `brain_meta` after its schema and after the
    // adoption, so both must be undone with it.
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
        "CREATE TRIGGER test_injected_failure BEFORE UPDATE ON brain_meta
         BEGIN SELECT RAISE(ABORT, 'injected failure'); END;",
    )
    .unwrap();
    drop(conn);
    match brain_v2::open_v2(&path, &opts(true)) {
        Err(e) => assert!(e.to_string().contains("injected failure"), "{e}"),
        Ok(_) => panic!("migration 1 passed the injected failure"),
    }

    let conn = Connection::open(&path).unwrap();
    assert_eq!(count(&conn, "PRAGMA user_version"), 0);
    assert_eq!(count(&conn, "SELECT json_extract(value, '$.schema_version') FROM brain_meta WHERE key = 'meta'"), 1);
    for table in V2_TABLES {
        assert_eq!(
            count_id(&conn, "SELECT COUNT(*) FROM sqlite_master WHERE name = ?1", table),
            0,
            "table {table} survived the rollback"
        );
    }
    assert_eq!(
        count(
            &conn,
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'idx_memory_ext%' OR name IN
             ('memories_ad_ext', 'memories_bd_guard', 'kv_v2_guard_insert', 'kv_v2_guard_update')"
        ),
        0,
        "an index or a trigger of migration 1 survived the rollback"
    );
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM memories"), 2);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM documents"), 3);
    assert!(conn.is_autocommit());
    conn.execute_batch("DROP TRIGGER test_injected_failure").unwrap();
    drop(conn);
    assert!(old_brain_open(&path).is_ok(), "the old binary's check still passes");
    assert!(matches!(brain_v2::require_v2(&path), Err(AxelError::V1Brain(_))));

    let brain = brain_v2::open_v2(&path, &opts(true)).unwrap();
    let conn = &brain.conn;
    assert_eq!(count(conn, "PRAGMA user_version"), 1);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM axel_migrations"), 1);
    assert_eq!(origins(conn), vec![("legacy".to_string(), 2)]);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memories_fts"), 2);
    for memory in [&a, &b] {
        let content: String =
            conn.query_row("SELECT content FROM memories WHERE id = ?1", [&memory.id], |r| r.get(0)).unwrap();
        assert_eq!(content, memory.content);
    }
}

#[test]
fn f1c_a_new_brain_has_no_legacy_row() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.r8");
    let brain = brain_v2::open_v2(&path, &opts(false)).unwrap();
    assert_eq!(count(&brain.conn, "SELECT COUNT(*) FROM memory_ext"), 0);
    let memory = Memory::new(MemoryCategory::Events, "test", "first", "written by version 2");
    project_memory::store_scoped(&brain.conn, &memory, &scope()).unwrap();
    drop(brain);

    let brain = brain_v2::open_v2(&path, &opts(false)).unwrap();
    assert_eq!(origins(&brain.conn), vec![("cli".to_string(), 1)]);
    assert_eq!(count(&brain.conn, "SELECT COUNT(*) FROM memory_ext WHERE origin_kind = 'legacy'"), 0);
}

// ── 5 ───────────────────────────────────────────────────────────────────

#[test]
fn t05_connect_refuses_version_0_and_begin_write_sees_a_changed_version() {
    let dir = TempDir::new().unwrap();

    let old = v1_empty(&dir, "v0.r8");
    let hash = main_hash(&old);
    for role in [Role::Indexer, Role::Front, Role::Cli] {
        assert!(matches!(brain_v2::connect(&old, role), Err(AxelError::BrainNotReady)), "{role:?}");
    }
    assert_eq!(main_hash(&old), hash, "connect wrote to a brain it refused");

    // A missing file is an error and is not created.
    let missing = dir.path().join("missing.r8");
    assert!(matches!(brain_v2::connect(&missing, Role::Cli), Err(AxelError::NoBrain(_))));
    assert!(!missing.exists());

    let path = dir.path().join("brain.r8");
    drop(brain_v2::open_v2(&path, &opts(false)).unwrap());
    let conn = brain_v2::connect(&path, Role::Indexer).unwrap();
    assert_eq!(count(&conn, "PRAGMA secure_delete"), 1);
    assert_eq!(count(&conn, "PRAGMA foreign_keys"), 1);
    assert_eq!(count(&conn, "PRAGMA synchronous"), 1, "Indexer: NORMAL");
    assert_eq!(count(&brain_v2::connect(&path, Role::Front).unwrap(), "PRAGMA synchronous"), 2, "Front: FULL");
    assert_eq!(count(&brain_v2::connect(&path, Role::Cli).unwrap(), "PRAGMA synchronous"), 2, "Cli: FULL");
    assert_eq!(count(&brain_v2::connect(&path, Role::Front).unwrap(), "PRAGMA busy_timeout"), 1_500);
    assert_eq!(count(&conn, "PRAGMA busy_timeout"), 5_000);

    // A write works while the version is ours.
    brain_v2::begin_write(&conn).unwrap();
    assert!(!conn.is_autocommit());
    conn.execute_batch("COMMIT").unwrap();

    // Another binary migrates the file under this process.
    let other = Connection::open(&path).unwrap();
    other.execute_batch("PRAGMA user_version = 2").unwrap();
    assert!(matches!(brain_v2::begin_write(&conn), Err(AxelError::BrainChanged)));
    assert!(conn.is_autocommit(), "begin_write left its transaction open");
    // It holds no lock: a connection with no busy timeout takes the write lock at once.
    other.execute_batch("BEGIN IMMEDIATE; ROLLBACK;").unwrap();

    // And the front's guard clears the flag when begin_write refuses.
    let writing = AtomicBool::new(false);
    assert!(matches!(brain_v2::begin_write_front(&conn, &writing), Err(AxelError::BrainChanged)));
    assert!(!writing.load(Ordering::SeqCst));
}

// ── 7 ───────────────────────────────────────────────────────────────────

/// The old binary's `Brain::open` check: version 1 and nothing else.
fn old_brain_open(path: &Path) -> Result<(), AxelError> {
    const OLD_SCHEMA_VERSION: i32 = 1;
    let conn = Connection::open(path)?;
    let json: String = conn.query_row("SELECT value FROM brain_meta WHERE key = 'meta'", [], |r| r.get(0))?;
    let meta: serde_json::Value = serde_json::from_str(&json)?;
    let file = meta["schema_version"].as_i64().unwrap() as i32;
    if file != OLD_SCHEMA_VERSION {
        return Err(AxelError::SchemaMismatch { file, expected: OLD_SCHEMA_VERSION });
    }
    Ok(())
}

#[test]
fn t07_the_two_locks_keep_the_old_binary_out() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.r8");
    drop(brain_v2::open_v2(&path, &opts(false)).unwrap());

    // The kv lock, with the old binary's constant ('2') and the new one ('3').
    let conn = Connection::open(&path).unwrap();
    let write = |value: &str| {
        conn.execute("INSERT OR REPLACE INTO kv (key, value) VALUES ('schema_version', ?1)", [value])
    };
    let refused = write("2").unwrap_err();
    assert!(refused.to_string().contains(KV_GUARD_MESSAGE), "{refused}");
    assert_eq!(write("3").unwrap(), 1);
    let refused = conn.execute("UPDATE kv SET value = '2' WHERE key = 'schema_version'", []).unwrap_err();
    assert!(refused.to_string().contains(KV_GUARD_MESSAGE), "{refused}");
    drop(conn);

    // The brain_meta lock.
    match old_brain_open(&path) {
        Err(AxelError::SchemaMismatch { file: 2, expected: 1 }) => {}
        other => panic!("expected SchemaMismatch, got {other:?}"),
    }

    // The new binary's openers pass both.
    assert!(Brain::open(&path).is_ok());
    if let Err(e) = Database::from_connection(Connection::open(&path).unwrap(), path.clone()) {
        panic!("the new Database must open a v2 brain: {e}");
    }

    // And a version 1 brain still opens with the old check.
    let old = v1_empty(&dir, "axel.r8");
    assert!(old_brain_open(&old).is_ok());
}

// ── 8 ───────────────────────────────────────────────────────────────────

#[test]
fn t08_a_delete_takes_the_memory_ext_row_and_a_replace_does_not() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.r8");
    let brain = brain_v2::open_v2(&path, &opts(false)).unwrap();
    let conn = &brain.conn;
    let memory = Memory::new(MemoryCategory::Events, "test", "kept", "a row that is replaced then deleted");
    project_memory::store_scoped(conn, &memory, &scope()).unwrap();
    assert_eq!(brain_v2::adopt_orphans(conn, "cli").unwrap(), 1);
    let ext = |conn: &Connection| count_id(conn, "SELECT COUNT(*) FROM memory_ext WHERE mem_id = ?1", &memory.id);
    assert_eq!(ext(conn), 1);

    // INSERT OR REPLACE of the same id (memkoshi's store_memory).
    let storage = MemoryStorage::open_existing(&path).unwrap();
    storage.store_memory(&memory).unwrap();
    assert_eq!(ext(conn), 1, "INSERT OR REPLACE must not delete the memory_ext row");

    // A delete (allowed by the purge row) does.
    conn.execute_batch("BEGIN IMMEDIATE; INSERT INTO axel_purge(one) VALUES (1);").unwrap();
    assert_eq!(conn.execute("DELETE FROM memories WHERE id = ?1", [&memory.id]).unwrap(), 1);
    conn.execute_batch("DELETE FROM axel_purge; COMMIT;").unwrap();
    assert_eq!(ext(conn), 0);
}

// ── 17, the function half ───────────────────────────────────────────────

#[test]
fn t17_a_version_1_brain_is_refused() {
    let dir = TempDir::new().unwrap();

    // Version 1 with one memory.
    let old = v1_empty(&dir, "axel.r8");
    remember_plain(&old, "kept", "the one memory of the old brain");
    let hash = main_hash(&old);
    match brain_v2::open_v2(&old, &opts(false)) {
        Err(AxelError::V1Brain(p)) => assert_eq!(p, old),
        Err(e) => panic!("expected V1Brain, got {e:?}"),
        Ok(_) => panic!("expected V1Brain, got a brain"),
    }
    match brain_v2::require_v2(&old) {
        Err(e @ AxelError::V1Brain(_)) => {
            assert!(e.to_string().starts_with("this is a version 1 brain (axel.r8)"), "{e}");
        }
        other => panic!("expected V1Brain, got {other:?}"),
    }
    assert_eq!(main_hash(&old), hash);
    assert!(Brain::open(&old).is_ok(), "the old brain still opens as version 1");
    assert_eq!(Brain::open(&old).unwrap().meta().schema_version, 1);

    // A missing path.
    let missing = dir.path().join("none").join("brain.r8");
    match brain_v2::require_v2(&missing) {
        Err(e @ AxelError::NoBrain(_)) => {
            let text = e.to_string();
            assert!(text.contains("axel-v2 init") && text.contains("axel-v2 migrate init"), "{text}");
        }
        other => panic!("expected NoBrain, got {other:?}"),
    }
    assert!(!missing.exists() && !missing.parent().unwrap().exists());

    // An empty version 1 file is upgraded without the flag.
    let empty = v1_empty(&dir, "empty.r8");
    assert!(matches!(brain_v2::require_v2(&empty), Err(AxelError::V1Brain(_))));
    let brain = brain_v2::open_v2(&empty, &opts(false)).unwrap();
    assert_eq!(count(&brain.conn, "PRAGMA user_version"), 1);
    drop(brain);
    assert!(brain_v2::require_v2(&empty).is_ok());

    // A v2 brain.
    let path = dir.path().join("brain.r8");
    drop(brain_v2::open_v2(&path, &opts(false)).unwrap());
    assert!(brain_v2::require_v2(&path).is_ok());
}

// ── 18 ──────────────────────────────────────────────────────────────────

#[test]
fn t18_the_indexer_stands_aside_for_the_front() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.r8");
    drop(brain_v2::open_v2(&path, &opts(false)).unwrap());

    let writing = AtomicBool::new(false);
    let stop = AtomicBool::new(false);
    let put = "INSERT OR REPLACE INTO context_data(key, value, layer) VALUES (?1, ?2, 'axel')";

    std::thread::scope(|s| {
        let indexer = s.spawn(|| {
            let conn = brain_v2::connect(&path, Role::Indexer).unwrap();
            let mut rounds = 0u64;
            while !stop.load(Ordering::SeqCst) {
                brain_v2::stand_aside(&writing);
                brain_v2::begin_write(&conn).unwrap();
                conn.execute(put, params!["indexer", rounds.to_string()]).unwrap();
                conn.execute_batch("COMMIT").unwrap();
                rounds += 1;
            }
            rounds
        });

        let front = brain_v2::connect(&path, Role::Front).unwrap();
        // Let the indexer get going first.
        std::thread::sleep(Duration::from_millis(50));
        for i in 0..20 {
            let start = Instant::now();
            let guard = brain_v2::begin_write_front(&front, &writing)
                .unwrap_or_else(|e| panic!("front write {i}: {e}"));
            assert!(writing.load(Ordering::SeqCst));
            front.execute(put, params![format!("front-{i}"), "x"]).unwrap();
            front.execute_batch("COMMIT").unwrap();
            drop(guard);
            let took = start.elapsed();
            assert!(took < Duration::from_millis(1_500), "front write {i} took {took:?}");
            assert!(!writing.load(Ordering::SeqCst), "the flag after write {i}");
        }

        // The closure returned an error: the guard still clears the flag.
        let failed: Result<(), AxelError> = (|| {
            let _guard = brain_v2::begin_write_front(&front, &writing)?;
            let result = front.execute("INSERT INTO no_such_table VALUES (1)", []).map(|_| ());
            front.execute_batch("ROLLBACK")?;
            Ok(result?)
        })();
        assert!(failed.is_err());
        assert!(!writing.load(Ordering::SeqCst), "the flag after a failed write");

        stop.store(true, Ordering::SeqCst);
        let rounds = indexer.join().unwrap();
        assert!(rounds > 0, "the indexer never wrote");
        println!("test 18: the indexer made {rounds} writes beside 20 front writes");
        assert_eq!(count(&front, "SELECT COUNT(*) FROM context_data WHERE key LIKE 'front-%'"), 20);
    });

    // stand_aside gives up after a second when nobody clears the flag.
    let stuck = AtomicBool::new(true);
    let start = Instant::now();
    brain_v2::stand_aside(&stuck);
    let waited = start.elapsed();
    assert!(waited >= Duration::from_millis(990) && waited < Duration::from_millis(1_500), "{waited:?}");
    let start = Instant::now();
    brain_v2::stand_aside(&writing);
    assert!(start.elapsed() < Duration::from_millis(100));
}

// ── 19 ──────────────────────────────────────────────────────────────────

#[test]
fn t19_the_delete_guard() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.r8");
    let brain = brain_v2::open_v2(&path, &opts(false)).unwrap();
    let memory = Memory::new(MemoryCategory::Events, "test", "guarded", "the pangolin row nobody may delete");
    project_memory::store_scoped(&brain.conn, &memory, &scope()).unwrap();
    assert_eq!(brain_v2::adopt_orphans(&brain.conn, "cli").unwrap(), 1);
    drop(brain);

    // A plain connection, as the old `forget <id>` has.
    let conn = Connection::open(&path).unwrap();
    let rows = |conn: &Connection| {
        (
            count_id(conn, "SELECT COUNT(*) FROM memories WHERE id = ?1", &memory.id),
            count_id(conn, "SELECT COUNT(*) FROM memories_fts WHERE mem_id = ?1", &memory.id),
            count_id(conn, "SELECT COUNT(*) FROM memory_ext WHERE mem_id = ?1", &memory.id),
        )
    };
    assert_eq!(rows(&conn), (1, 1, 1));
    let refused = conn.execute("DELETE FROM memories WHERE id = ?1", [&memory.id]).unwrap_err();
    assert!(refused.to_string().contains(DELETE_GUARD_MESSAGE), "{refused}");
    assert_eq!(rows(&conn), (1, 1, 1));

    // The old binary's own call.
    let storage = MemoryStorage::open_existing(&path).unwrap();
    let refused = storage.delete_memory(&memory.id).unwrap_err();
    assert!(refused.to_string().contains(DELETE_GUARD_MESSAGE), "{refused}");
    assert_eq!(rows(&conn), (1, 1, 1));

    // INSERT OR REPLACE of an existing id is not stopped.
    storage.store_memory(&memory).unwrap();
    assert_eq!(rows(&conn), (1, 1, 1));

    // Inside a transaction that first inserts the axel_purge row the delete passes.
    conn.execute_batch("BEGIN IMMEDIATE; INSERT INTO axel_purge(one) VALUES (1);").unwrap();
    assert_eq!(conn.execute("DELETE FROM memories WHERE id = ?1", [&memory.id]).unwrap(), 1);
    conn.execute_batch("DELETE FROM axel_purge; COMMIT;").unwrap();
    assert_eq!(rows(&conn), (0, 1, 0), "the memory and its memory_ext row are gone; the word row is purge_many's");
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM axel_purge"), 0);
}

// ── 20 ──────────────────────────────────────────────────────────────────

#[test]
fn t20_the_word_index_fill_inserts_only_the_missing_rows() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.r8");
    let brain = brain_v2::open_v2(&path, &opts(false)).unwrap();
    let conn = &brain.conn;

    // Three memories with word rows.
    let with_words: Vec<Memory> = ["heron", "otter", "badger"]
        .iter()
        .map(|w| {
            let m = Memory::new(MemoryCategory::Events, "test", "with words", format!("the {w} row has a word row"));
            project_memory::store_scoped(conn, &m, &scope()).unwrap();
            m
        })
        .collect();
    let rowid = |id: &str| count_id(conn, "SELECT rowid FROM memories_fts WHERE mem_id = ?1", id);
    let rowids: Vec<i64> = with_words.iter().map(|m| rowid(&m.id)).collect();

    // One the way `AxelBrain::remember` writes it, and a secret one.
    let fourth = remember_plain(&path, "no word row", "the wombat row came from the command line");
    let secret = remember_plain(&path, "secret title", "the axolotl password is hunter2");
    conn.execute("UPDATE memories SET sensitivity = 'secret' WHERE id = ?1", [&secret.id]).unwrap();
    let words = |id: &str| count_id(conn, "SELECT COUNT(*) FROM memories_fts WHERE mem_id = ?1", id);
    assert_eq!(words(&fourth.id), 0);
    assert!(word_hits(conn, "wombat").is_empty());

    // No row here has a memory_ext row yet: all five are adopted.
    assert_eq!(brain_v2::adopt_orphans(conn, "cli").unwrap(), 5);

    assert_eq!(words(&fourth.id), 1);
    assert_eq!(word_hits(conn, "wombat"), vec![fourth.id.clone()]);
    for (m, before) in with_words.iter().zip(&rowids) {
        assert_eq!(rowid(&m.id), *before, "a word row was rebuilt");
        assert_eq!(words(&m.id), 1);
    }
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memories_fts"), 5);

    // The secret row: a title-only row.
    let (title, abstract_text, content): (String, String, String) = conn
        .query_row("SELECT title, abstract, content FROM memories_fts WHERE mem_id = ?1", [&secret.id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!((title.as_str(), abstract_text.as_str(), content.as_str()), ("secret title", "", ""));
    assert!(word_hits(conn, "axolotl").is_empty());
    assert_eq!(word_hits(conn, "secret"), vec![secret.id.clone()]);

    // Origin `cli`; nothing hidden; no vector, because no shadow document.
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memory_ext WHERE origin_kind = 'cli'"), 5);
    assert_eq!(
        count(conn, "SELECT COUNT(*) FROM memory_ext WHERE hidden_ms IS NULL AND embedding IS NULL AND embed_model IS NULL"),
        5
    );

    // Run again: nothing to do, nothing changed.
    assert_eq!(brain_v2::adopt_orphans(conn, "cli").unwrap(), 0);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memories_fts"), 5);
    assert_eq!(rowid(&with_words[0].id), rowids[0]);
    assert!(brain_v2::adopt_orphans(conn, "tool").is_err(), "only legacy and cli are origins of adoption");

    // `legacy` when adopted under upgrade_v1.
    let (old, a, b) = legacy_file(&dir);
    let upgraded = brain_v2::open_v2(&old, &opts(true)).unwrap();
    for m in [&a, &b] {
        let origin: String = upgraded
            .conn
            .query_row("SELECT origin_kind FROM memory_ext WHERE mem_id = ?1", [&m.id], |r| r.get(0))
            .unwrap();
        assert_eq!(origin, "legacy");
    }
    // A row the command line adds later is `cli`, also on a brain that was upgraded.
    let later = remember_plain(&old, "later", "the ibis row was added after the import");
    assert_eq!(brain_v2::adopt_orphans(&upgraded.conn, "cli").unwrap(), 1);
    let origin: String = upgraded
        .conn
        .query_row("SELECT origin_kind FROM memory_ext WHERE mem_id = ?1", [&later.id], |r| r.get(0))
        .unwrap();
    assert_eq!(origin, "cli");
    assert_eq!(word_hits(&upgraded.conn, "ibis"), vec![later.id.clone()]);
}

// ── 23 ──────────────────────────────────────────────────────────────────

fn master_rows(conn: &Connection, sql: &str) -> Vec<(String, String, Option<String>)> {
    let mut stmt = conn.prepare(sql).unwrap();
    let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).unwrap();
    rows.map(|r| r.unwrap()).collect()
}

#[test]
fn t23_migration_1_is_the_block() {
    // A scratch database with the old tables the block needs: its triggers are on
    // `memories` and `kv`, and it names neither word index.
    assert!(!SCHEMA_V2_SQL.contains("memories_fts") && !SCHEMA_V2_SQL.contains("chunks_fts"));
    let scratch = Connection::open_in_memory().unwrap();
    scratch
        .execute_batch(
            "CREATE TABLE memories(id TEXT PRIMARY KEY);
             CREATE TABLE kv(key TEXT PRIMARY KEY, value TEXT);",
        )
        .unwrap();
    scratch.execute_batch(SCHEMA_V2_SQL).unwrap();
    let block = master_rows(
        &scratch,
        "SELECT type, name, sql FROM sqlite_master
         WHERE type = 'trigger' OR tbl_name NOT IN ('memories', 'kv') ORDER BY type, name",
    );
    assert!(block.len() > 30, "the block's names: {}", block.len());

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.r8");
    let brain = brain_v2::open_v2(&path, &opts(false)).unwrap();
    let conn = &brain.conn;
    for (kind, name, sql) in &block {
        let found: Option<(String, Option<String>)> = conn
            .query_row("SELECT type, sql FROM sqlite_master WHERE name = ?1", [name], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .ok();
        assert_eq!(found, Some((kind.clone(), sql.clone())), "{kind} {name}");
    }

    let sql_of = |name: &str| -> String {
        conn.query_row("SELECT sql FROM sqlite_master WHERE name = ?1", [name], |r| r.get(0)).unwrap()
    };
    assert!(sql_of("forgotten_turns").contains("UNIQUE (logical, source, from_ms, turn_key, archive_id)"));
    assert_eq!(count(conn, "SELECT COUNT(*) FROM sqlite_master WHERE name = 'suppressed_turns'"), 0);
    for column in ["forget_units", "keep_forget_turns", "unmatched_hidden", "win_no", "part_no"] {
        assert_eq!(
            count_id(conn, "SELECT COUNT(*) FROM pragma_table_info('archive_index') WHERE name = ?1", column),
            1,
            "archive_index.{column}"
        );
    }
}

const MEMORY_MARKER: &str = "jwqresiduemem7731xv48";
const CHUNK_MARKER: &str = "hzkresiduedoc2296bn53";

/// Hits of the two markers in the file and its log. The tail of each marker is counted
/// too: the word index stores a term without the prefix it shares with its neighbour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Hits {
    memory: usize,
    chunk: usize,
}

impl Hits {
    fn in_file(path: &Path) -> Hits {
        let mut bytes = std::fs::read(path).unwrap();
        if let Ok(wal) = std::fs::read(format!("{}-wal", path.display())) {
            bytes.extend(wal);
        }
        let occurrences = |needle: &str| {
            let needle = needle.as_bytes();
            bytes.windows(needle.len()).filter(|w| *w == needle).count()
        };
        let hits = |m: &str| occurrences(m) + occurrences(&m[9..]);
        Hits { memory: hits(MEMORY_MARKER), chunk: hits(CHUNK_MARKER) }
    }
}

/// A version 1 brain made with the old openers (as the probe's `legacy_brain`), six
/// memories and six chunks; the third of each carries a marker and is deleted with the
/// old code's statements before any `secure-delete` option exists. Then `open_v2` with
/// `upgrade_v1` and a checkpoint. Returns the hits after the delete and after the upgrade.
fn old_residue(pragma_for_the_delete: bool) -> (Hits, Hits) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("axel.r8");
    let brain = Brain::create(&path, Some("old")).unwrap();
    project_memory::migrate(brain.conn()).unwrap();
    drop(brain);
    let db = Database::from_connection(Connection::open(&path).unwrap(), path.clone())
        .unwrap_or_else(|e| panic!("velocirag schema: {e}"));
    let conn = db.conn();

    let zeros = vec![0.0f32; velocirag::EMBEDDING_DIM];
    let meta = serde_json::json!({});
    let mut doomed = String::new();
    for i in 0..6 {
        let content = if i == 2 {
            format!("the word is {MEMORY_MARKER} today")
        } else {
            format!("kept row {i} about harbours and lanterns")
        };
        let memory = Memory::new(MemoryCategory::Events, "test", if i == 2 { "doomed" } else { "kept" }, content);
        project_memory::store_scoped(conn, &memory, &scope()).unwrap();
        if i == 2 {
            doomed = memory.id.clone();
        }
    }
    for i in 0..6 {
        let (doc_id, content) = if i == 2 {
            ("doomed".to_string(), format!("the word is {CHUNK_MARKER} today"))
        } else {
            (format!("kept-{i}"), format!("kept chunk {i} about harbours and lanterns"))
        };
        db.insert_document(&doc_id, &content, &meta, &zeros, Some("/notes/file.md"))
            .unwrap_or_else(|e| panic!("insert {doc_id}: {e}"));
    }
    checkpoint_truncate(conn, &path);
    let stored = Hits::in_file(&path);
    assert!(stored.memory > 0 && stored.chunk > 0, "each marker must be in the file first: {stored:?}");

    if pragma_for_the_delete {
        conn.execute_batch("PRAGMA secure_delete = ON").unwrap();
        assert_eq!(count(conn, "PRAGMA secure_delete"), 1);
    }
    // The old code's statements: the table row and its word-index row.
    conn.execute("DELETE FROM memories WHERE id = ?1", [&doomed]).unwrap();
    conn.execute("DELETE FROM memories_fts WHERE mem_id = ?1", [&doomed]).unwrap();
    conn.execute("DELETE FROM documents WHERE doc_id = 'doomed'", []).unwrap();
    conn.execute("DELETE FROM chunks_fts WHERE doc_id = 'doomed'", []).unwrap();
    checkpoint_truncate(conn, &path);
    let deleted = Hits::in_file(&path);
    drop(db);

    let brain = brain_v2::open_v2(&path, &opts(true)).unwrap();
    assert_eq!(count(&brain.conn, "SELECT COUNT(*) FROM memories"), 5);
    assert_eq!(count(&brain.conn, "SELECT COUNT(*) FROM memory_ext WHERE origin_kind = 'legacy'"), 5);
    checkpoint_truncate(&brain.conn, &path);
    (deleted, Hits::in_file(&path))
}

#[test]
fn t23_migration_1_removes_the_old_residue_of_the_word_indexes() {
    // Asserted: `PRAGMA secure_delete` on for the delete, so the table rows are wiped and
    // only the word indexes' segments can still hold the markers.
    let (deleted, upgraded) = old_residue(true);
    assert!(
        deleted.memory > 0 && deleted.chunk > 0,
        "each marker must still be in the file after the old delete, or this proves nothing: {deleted:?}"
    );
    assert_eq!(upgraded, Hits { memory: 0, chunk: 0 }, "migration 1's optimize left old residue");

    // Printed, not asserted: the pragma off for that delete, as the old binary ran.
    let (_, upgraded) = old_residue(false);
    println!("old-binary residue: memory {}, chunk {}", upgraded.memory, upgraded.chunk);
}

// ── degrade_on (4.2) ────────────────────────────────────────────────────

fn sqlite_error(code: std::os::raw::c_int) -> AxelError {
    AxelError::Db(rusqlite::Error::SqliteFailure(rusqlite::ffi::Error::new(code), None))
}

#[test]
fn degrade_on_sets_the_reason_then_the_mode_for_a_brain_error_only() {
    let cases = [
        (AxelError::BrainChanged, DegradedReason::BrainChanged),
        (AxelError::BrainDamaged("x".to_string()), DegradedReason::BrainDamaged),
        (sqlite_error(rusqlite::ffi::SQLITE_NOTADB), DegradedReason::BrainDamaged),
        (sqlite_error(rusqlite::ffi::SQLITE_CORRUPT), DegradedReason::BrainDamaged),
    ];
    for (err, reason) in cases {
        let (status, mode) = (Status::default(), AtomicU8::new(Mode::Ready as u8));
        assert!(brain_v2::degrade_on(&err, &status, &mode), "{err:?}");
        assert_eq!(DegradedReason::from_u8(status.reason.load(Ordering::SeqCst)), reason, "{err:?}");
        assert_eq!(Mode::from_u8(mode.load(Ordering::SeqCst)), Some(Mode::Degraded), "{err:?}");
    }

    let others = [
        AxelError::Other("x".to_string()),
        sqlite_error(rusqlite::ffi::SQLITE_BUSY),
        sqlite_error(rusqlite::ffi::SQLITE_FULL),
        AxelError::Db(rusqlite::Error::QueryReturnedNoRows),
        AxelError::BrainNotReady,
    ];
    for err in others {
        let (status, mode) = (Status::default(), AtomicU8::new(Mode::Ready as u8));
        assert!(!brain_v2::degrade_on(&err, &status, &mode), "{err:?}");
        assert_eq!(status.reason.load(Ordering::SeqCst), 0, "{err:?}");
        assert_eq!(mode.load(Ordering::SeqCst), Mode::Ready as u8, "{err:?}");
    }
}
