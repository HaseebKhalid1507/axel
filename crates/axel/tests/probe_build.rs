//! Build probe for the v2 rewrite: assumptions the plan rests on, settled
//! here before anything depends on them. No production code is exercised
//! that this file adds; the tables come from the existing definitions
//! (`r8.rs`, `project_memory.rs`, velocirag's `db.rs`) and the v2 statements
//! are copied from the spec as text.
//!
//! | # | assumption |
//! |---|---|
//! | a | `Embedder` and `rusqlite::Connection` are `Send` |
//! | b | `SQLITE_OPEN_NOFOLLOW` exists and a `?mode=ro&immutable=1` URI opens |
//! | c | recall generators a and c run as written on the existing tables |
//! | d | FTS5 `secure-delete` + `PRAGMA secure_delete` leave no trace of a deleted row |
//! | e | the `kv` guard triggers stop `INSERT OR REPLACE … '2'`, and let `'3'` through |
//! | f | `AFTER DELETE ON memories` fires on a delete, not on `INSERT OR REPLACE` |
//! | g | `json_set` and `json_extract` exist in the bundled SQLite |
//! | h | `instr(supersedes, '"<id>"')` finds an id in the arrays the store writes |
//! | i | the cost of an exact scan of 25,000 vectors (`#[ignore]`, prints ms) |
//! | j | the suite runs with `HOME` moved: not a test here, it is the gate itself (`scripts/dirty-home.sh`) |
//!
//! Every file is in a temp directory.

use std::path::{Path, PathBuf};

use axel::project_memory::{self, MemoryScope, RecordLinks, Retention, Sensitivity};
use axel::r8::Brain;
use axel_memkoshi::memory::{Memory, MemoryCategory};
use axel_memkoshi::storage::MemoryStorage;
use rusqlite::{named_params, params, Connection, ErrorCode, OpenFlags};
use tempfile::TempDir;
use velocirag::db::Database;
use velocirag::embedder::Embedder;

// ── v2 statements, as the spec has them ─────────────────────────────────

const MEMORY_EXT_SQL: &str = "
CREATE TABLE IF NOT EXISTS memory_ext(
  vid          INTEGER PRIMARY KEY,
  mem_id       TEXT NOT NULL UNIQUE,
  origin_kind  TEXT NOT NULL,
  origin_ref   TEXT NOT NULL DEFAULT '',
  unit_pos     TEXT,
  session_id   TEXT,
  logical      TEXT,
  content_hash TEXT NOT NULL,
  created_ms   INTEGER NOT NULL,
  time_source  TEXT NOT NULL DEFAULT 'exact',
  hidden_ms    INTEGER,
  hidden_by    TEXT,
  embed_model  TEXT,
  embedding    BLOB
);
CREATE TRIGGER IF NOT EXISTS memories_ad_ext AFTER DELETE ON memories
  BEGIN DELETE FROM memory_ext WHERE mem_id = OLD.id; END;
";

const DOCUMENTS_V2_SQL: &str = "
ALTER TABLE documents ADD COLUMN source_mtime_ms INTEGER;
ALTER TABLE documents ADD COLUMN hidden TEXT;
";

const KV_GUARD_MESSAGE: &str = "this brain belongs to Axel v2; use the axel-v2 binary";

const KV_GUARD_SQL: &str = "
CREATE TRIGGER IF NOT EXISTS kv_v2_guard_insert BEFORE INSERT ON kv
  WHEN NEW.key = 'schema_version' AND CAST(NEW.value AS INTEGER) < 3
  BEGIN SELECT RAISE(ABORT, 'this brain belongs to Axel v2; use the axel-v2 binary'); END;
CREATE TRIGGER IF NOT EXISTS kv_v2_guard_update BEFORE UPDATE ON kv
  WHEN NEW.key = 'schema_version' AND CAST(NEW.value AS INTEGER) < 3
  BEGIN SELECT RAISE(ABORT, 'this brain belongs to Axel v2; use the axel-v2 binary'); END;
";

/// Section 3, generator a: memory, by words.
const GENERATOR_A: &str = "
SELECT m.id, m.project_key, m.title, m.content, m.category, m.tags, m.created,
       m.sensitivity, m.retention, m.provenance, m.supersedes, m.contradicts,
       m.confirmed_by, m.invalidated_at, m.superseded_by,
       e.vid, e.origin_kind, e.origin_ref, e.session_id, e.content_hash,
       e.created_ms, e.time_source
FROM memories_fts f JOIN memories m ON m.id = f.mem_id
                    JOIN memory_ext e ON e.mem_id = m.id
WHERE memories_fts MATCH :expr
  AND e.hidden_ms IS NULL
  AND (m.project_key = :scope_dir
       OR (:global AND (m.project_key = 'p0000000000000000' OR m.project_key IS NULL)))
  AND (m.expires_at IS NULL OR m.expires_at > :now)
ORDER BY bm25(memories_fts) LIMIT 48;
";

/// Section 3, generator c: note files, by words.
const GENERATOR_C: &str = "
SELECT d.id, d.doc_id, d.content, d.file_path, d.metadata, d.created, d.source_mtime_ms
FROM chunks_fts f JOIN documents d ON d.doc_id = f.doc_id
WHERE chunks_fts MATCH 'content : (' || :expr || ')' AND d.hidden IS NULL
ORDER BY bm25(chunks_fts) LIMIT 24;
";

const GLOBAL_KEY: &str = "p0000000000000000";
const PROJECT: &str = "p1111111111111111";
const OTHER_PROJECT: &str = "p2222222222222222";

// ── Helpers ─────────────────────────────────────────────────────────────

/// A brain file whose tables come from the existing definitions:
/// `Brain::create` (memories, brain_meta), `project_memory::migrate` (the
/// scope columns, `memories_fts`, tombstones) and velocirag's `Database`
/// (documents, `chunks_fts`, `kv`). The returned handle's connection is the
/// only one open on the file.
fn legacy_brain(dir: &TempDir) -> (PathBuf, Database) {
    let path = dir.path().join("probe.r8");
    let brain = Brain::create(&path, Some("probe")).unwrap();
    project_memory::migrate(brain.conn()).unwrap();
    drop(brain);
    let conn = Connection::open(&path).unwrap();
    let db = match Database::from_connection(conn, path.clone()) {
        Ok(db) => db,
        Err(e) => panic!("velocirag schema on a fresh brain: {e}"),
    };
    (path, db)
}

fn scope(project_key: &str) -> MemoryScope {
    MemoryScope {
        project_key: project_key.to_string(),
        sensitivity: Sensitivity::Normal,
        retention: Retention::Standard,
        provenance: "probe".to_string(),
    }
}

fn store(conn: &Connection, project_key: &str, title: &str, content: &str) -> String {
    let memory = Memory::new(MemoryCategory::Events, "probe", title, content);
    project_memory::store_scoped(conn, &memory, &scope(project_key)).unwrap();
    memory.id
}

fn adopt(conn: &Connection, mem_id: &str, hidden_ms: Option<i64>) {
    conn.execute(
        "INSERT INTO memory_ext (mem_id, origin_kind, content_hash, created_ms, hidden_ms)
         VALUES (?1, 'tool', 'probe', 1, ?2)",
        params![mem_id, hidden_ms],
    )
    .unwrap();
}

fn ext_rows(conn: &Connection, mem_id: &str) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM memory_ext WHERE mem_id = ?1",
        [mem_id],
        |r| r.get(0),
    )
    .unwrap()
}

fn ids(conn: &Connection, sql: &str, params: &[(&str, &dyn rusqlite::ToSql)], column: usize) -> Vec<String> {
    let mut stmt = conn.prepare(sql).unwrap();
    let rows = stmt
        .query_map(params, |r| r.get::<_, String>(column))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    rows
}

fn sorted(mut v: Vec<String>) -> Vec<String> {
    v.sort();
    v
}

/// The bytes of the brain file and of its write-ahead log, if there is one.
fn file_bytes(path: &Path) -> Vec<u8> {
    let mut bytes = std::fs::read(path).unwrap();
    if let Ok(wal) = std::fs::read(format!("{}-wal", path.display())) {
        bytes.extend(wal);
    }
    bytes
}

fn occurrences(haystack: &[u8], needle: &str) -> usize {
    let needle = needle.as_bytes();
    haystack.windows(needle.len()).filter(|w| *w == needle).count()
}

fn checkpoint_truncate(conn: &Connection, path: &Path) {
    let busy: i64 = conn
        .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0))
        .unwrap();
    assert_eq!(busy, 0, "the checkpoint must complete");
    let wal = std::fs::metadata(format!("{}-wal", path.display())).map(|m| m.len()).unwrap_or(0);
    assert_eq!(wal, 0, "TRUNCATE must leave an empty log");
}

// ── a ───────────────────────────────────────────────────────────────────

fn is_send<T: Send>() {}

#[test]
fn a_embedder_and_connection_are_send() {
    // Settled when this file compiles.
    is_send::<Embedder>();
    is_send::<Connection>();
}

// ── b ───────────────────────────────────────────────────────────────────

#[test]
fn b_nofollow_flag_and_immutable_uri_open() {
    let dir = TempDir::new().unwrap();
    // NOFOLLOW refuses a symlink anywhere in the path, so resolve the temp
    // directory itself first.
    let root = std::fs::canonicalize(dir.path()).unwrap();
    let path = root.join("b.db");
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE t(v TEXT NOT NULL);
             INSERT INTO t VALUES ('seen');",
        )
        .unwrap();
    }
    let read = |conn: &Connection| -> String {
        conn.query_row("SELECT v FROM t", [], |r| r.get(0)).unwrap()
    };
    let ro_nofollow = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NOFOLLOW;

    // The flag exists, and a regular file opens with it.
    let conn = Connection::open_with_flags(&path, ro_nofollow).unwrap();
    assert_eq!(read(&conn), "seen");
    drop(conn);

    // The URI form opens, reads, refuses writes, and leaves no -wal or -shm
    // beside a file that is in WAL mode.
    for name in ["b.db-wal", "b.db-shm"] {
        let _ = std::fs::remove_file(root.join(name));
    }
    let uri = format!("file:{}?mode=ro&immutable=1", path.display());
    let ro_uri = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI;
    let conn = Connection::open_with_flags(&uri, ro_uri).unwrap();
    assert_eq!(read(&conn), "seen");
    assert!(conn.execute("INSERT INTO t VALUES ('no')", []).is_err());
    assert!(!root.join("b.db-wal").exists(), "immutable open created a -wal file");
    assert!(!root.join("b.db-shm").exists(), "immutable open created a -shm file");
    drop(conn);

    // Both at once.
    let conn =
        Connection::open_with_flags(&uri, ro_uri | OpenFlags::SQLITE_OPEN_NOFOLLOW).unwrap();
    assert_eq!(read(&conn), "seen");
    drop(conn);

    #[cfg(unix)]
    {
        // What the flag is for: a symlink to the file is refused, and opens
        // without the flag.
        let link = root.join("link.db");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        let refused = Connection::open_with_flags(&link, ro_nofollow).unwrap_err();
        assert_eq!(refused.sqlite_error_code(), Some(ErrorCode::CannotOpen), "{refused}");
        let conn = Connection::open_with_flags(&link, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
        assert_eq!(read(&conn), "seen");
        drop(conn);

        // It is stricter than open(2)'s O_NOFOLLOW: a symlinked DIRECTORY on
        // the way to a regular file is refused too. A brain whose folder is
        // reached through a symlink cannot be opened with this flag.
        let linked_dir = root.join("linked-dir");
        std::os::unix::fs::symlink(&root, &linked_dir).unwrap();
        let refused =
            Connection::open_with_flags(linked_dir.join("b.db"), ro_nofollow).unwrap_err();
        assert_eq!(refused.sqlite_error_code(), Some(ErrorCode::CannotOpen), "{refused}");
    }
}

// ── c ───────────────────────────────────────────────────────────────────

#[test]
fn c_generators_a_and_c_run_on_the_existing_tables() {
    let dir = TempDir::new().unwrap();
    let (_path, db) = legacy_brain(&dir);
    let conn = db.conn();
    conn.execute_batch(MEMORY_EXT_SQL).unwrap();
    conn.execute_batch(DOCUMENTS_V2_SQL).unwrap();

    // Generator a, five rows.
    let both = store(conn, PROJECT, "both words", "alpha beta in one project row");
    let one = store(conn, PROJECT, "one word", "alpha alone in a project row");
    let hidden = store(conn, PROJECT, "hidden", "alpha beta in a hidden row");
    let global = store(conn, GLOBAL_KEY, "global", "beta alone in a global row");
    let other = store(conn, OTHER_PROJECT, "elsewhere", "alpha beta in another project");
    adopt(conn, &both, None);
    adopt(conn, &one, None);
    adopt(conn, &hidden, Some(1));
    adopt(conn, &global, None);
    adopt(conn, &other, None);

    let expr = "\"alpha\" OR \"beta\"";
    let now = chrono::Utc::now().to_rfc3339();
    let run_a = |scope_dir: &str, with_global: bool| {
        ids(
            conn,
            GENERATOR_A,
            named_params! {
                ":expr": expr,
                ":scope_dir": scope_dir,
                ":global": with_global,
                ":now": now,
            },
            0,
        )
    };
    assert_eq!(conn.prepare(GENERATOR_A).unwrap().column_count(), 22);

    let found = run_a(PROJECT, true);
    assert_eq!(found[0], both, "bm25 puts the row with both words first");
    assert_eq!(
        sorted(found),
        sorted(vec![both.clone(), one.clone(), global.clone()]),
        "the project's rows and the global row; not the hidden one, not the other project's"
    );
    assert_eq!(run_a(PROJECT, false), vec![both.clone(), one.clone()]);
    assert_eq!(run_a("no row has this", true), vec![global.clone()]);
    assert!(run_a("no row has this", false).is_empty());

    // Generator c, five rows. chunks_fts indexes doc_id, content and
    // file_path; the column filter must keep the match to content.
    let zeros = vec![0.0f32; velocirag::EMBEDDING_DIM];
    let meta = serde_json::json!({});
    for (doc_id, content, file_path) in [
        ("doc-1", "alpha beta in a note", "/notes/one.md"),
        ("doc-2", "alpha alone in a note", "/notes/two.md"),
        ("doc-3", "neither word in the body", "/notes/alpha.md"),
        ("doc-4", "alpha beta in a hidden note", "/notes/four.md"),
        ("doc-5", "nothing to find here", "/notes/five.md"),
    ] {
        db.insert_document(doc_id, content, &meta, &zeros, Some(file_path))
            .unwrap_or_else(|e| panic!("insert {doc_id}: {e}"));
    }
    conn.execute("UPDATE documents SET hidden = 'missing_source' WHERE doc_id = 'doc-4'", [])
        .unwrap();

    assert_eq!(conn.prepare(GENERATOR_C).unwrap().column_count(), 7);
    let found = ids(conn, GENERATOR_C, named_params! { ":expr": expr }, 1);
    assert_eq!(found, vec!["doc-1".to_string(), "doc-2".to_string()]);

    // The filter is what keeps doc-3 out: without it the path matches.
    let unfiltered = ids(
        conn,
        "SELECT d.doc_id FROM chunks_fts f JOIN documents d ON d.doc_id = f.doc_id
          WHERE chunks_fts MATCH :expr AND d.hidden IS NULL",
        named_params! { ":expr": expr },
        0,
    );
    assert!(unfiltered.contains(&"doc-3".to_string()));
}

// ── d ───────────────────────────────────────────────────────────────────

const MEMORY_MARKER: &str = "zqxprobemem8841kd27";
const CHUNK_MARKER: &str = "zqxprobedoc5539wt64";

/// Write a memory and a note chunk that each carry a marker, delete both,
/// checkpoint, and count the markers in the file before and after. The tail
/// of each marker is counted too, because the word index stores a term
/// without the prefix it shares with its neighbour.
fn marker_counts_around_a_purge(fts_secure_delete: bool) -> (usize, usize) {
    let dir = TempDir::new().unwrap();
    let (path, db) = legacy_brain(&dir);
    let conn = db.conn();

    conn.execute_batch("PRAGMA secure_delete = ON").unwrap();
    let on: i64 = conn.query_row("PRAGMA secure_delete", [], |r| r.get(0)).unwrap();
    assert_eq!(on, 1);

    if fts_secure_delete {
        // FTS5 accepts the option on both word indexes, and keeps it.
        conn.execute("INSERT INTO memories_fts(memories_fts, rank) VALUES('secure-delete', 1)", [])
            .unwrap();
        conn.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('secure-delete', 1)", [])
            .unwrap();
        for table in ["memories_fts_config", "chunks_fts_config"] {
            let kept: i64 = conn
                .query_row(
                    &format!("SELECT v FROM {table} WHERE k = 'secure-delete'"),
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(kept, 1, "{table}");
        }
    }

    let zeros = vec![0.0f32; velocirag::EMBEDDING_DIM];
    let meta = serde_json::json!({});
    for i in 0..4 {
        store(conn, PROJECT, "kept", &format!("kept row {i} about harbours and lanterns"));
        db.insert_document(
            &format!("kept-{i}"),
            &format!("kept chunk {i} about harbours and lanterns"),
            &meta,
            &zeros,
            Some("/notes/kept.md"),
        )
        .unwrap_or_else(|e| panic!("insert kept-{i}: {e}"));
    }
    let doomed = store(conn, PROJECT, "doomed", &format!("the word is {MEMORY_MARKER} today"));
    db.insert_document(
        "doomed",
        &format!("the word is {CHUNK_MARKER} today"),
        &meta,
        &zeros,
        Some("/notes/doomed.md"),
    )
    .unwrap_or_else(|e| panic!("insert doomed: {e}"));

    let count = |bytes: &[u8]| {
        occurrences(bytes, MEMORY_MARKER)
            + occurrences(bytes, CHUNK_MARKER)
            + occurrences(bytes, &MEMORY_MARKER[9..])
            + occurrences(bytes, &CHUNK_MARKER[9..])
    };

    checkpoint_truncate(conn, &path);
    let before = count(&file_bytes(&path));

    conn.execute("DELETE FROM memories WHERE id = ?1", [&doomed]).unwrap();
    conn.execute("DELETE FROM memories_fts WHERE mem_id = ?1", [&doomed]).unwrap();
    conn.execute("DELETE FROM documents WHERE doc_id = 'doomed'", []).unwrap();
    conn.execute("DELETE FROM chunks_fts WHERE doc_id = 'doomed'", []).unwrap();
    checkpoint_truncate(conn, &path);
    let after = count(&file_bytes(&path));

    (before, after)
}

#[test]
fn d_secure_delete_leaves_no_marker_in_the_file() {
    // FTS5's 'secure-delete' option needs SQLite 3.42.
    assert!(rusqlite::version_number() >= 3_042_000, "bundled SQLite is {}", rusqlite::version());

    let (before, after) = marker_counts_around_a_purge(true);
    assert!(before > 0, "the markers must be findable before the delete");
    assert_eq!(after, 0, "a deleted row's marker is still in the file");

    // The option is what does it. With only PRAGMA secure_delete the tables
    // are clean but the word indexes still hold the deleted words.
    let (before, after) = marker_counts_around_a_purge(false);
    assert!(before > 0);
    assert!(after > 0, "without FTS5 secure-delete the word index was expected to keep the marker");
}

// ── e ───────────────────────────────────────────────────────────────────

#[test]
fn e_kv_guard_refuses_schema_version_2_even_with_or_replace() {
    let dir = TempDir::new().unwrap();
    let (path, db) = legacy_brain(&dir);
    let conn = db.conn();
    let version = |conn: &Connection| -> String {
        conn.query_row("SELECT value FROM kv WHERE key = 'schema_version'", [], |r| r.get(0))
            .unwrap()
    };
    // What today's code wrote when it opened the file (db.rs, init_schema).
    assert_eq!(version(conn), "2");

    conn.execute_batch(KV_GUARD_SQL).unwrap();
    let write = |conn: &Connection, value: &str| {
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES ('schema_version', ?1)",
            [value],
        )
    };

    // '2' fails with the trigger's message although the statement says OR REPLACE.
    let refused = write(conn, "2").unwrap_err();
    assert!(refused.to_string().contains(KV_GUARD_MESSAGE), "{refused}");
    assert_eq!(refused.sqlite_error_code(), Some(ErrorCode::ConstraintViolation));
    assert_eq!(version(conn), "2");

    // '3' passes.
    assert_eq!(write(conn, "3").unwrap(), 1);
    assert_eq!(version(conn), "3");

    // And from then on '2' cannot come back, by insert or by update.
    let refused = write(conn, "2").unwrap_err();
    assert!(refused.to_string().contains(KV_GUARD_MESSAGE), "{refused}");
    let refused = conn
        .execute("UPDATE kv SET value = '2' WHERE key = 'schema_version'", [])
        .unwrap_err();
    assert!(refused.to_string().contains(KV_GUARD_MESSAGE), "{refused}");
    assert_eq!(version(conn), "3");

    // Other keys are not touched by the guard.
    conn.execute("INSERT OR REPLACE INTO kv (key, value) VALUES ('other', '1')", [])
        .unwrap();

    // The lock itself: the call BrainSearch::open makes (search.rs) runs
    // that same statement with '2', so the old code cannot open this file.
    let second = Connection::open(&path).unwrap();
    let refused = Database::from_connection(second, path.clone())
        .err()
        .expect("today's Database::from_connection must fail on a guarded brain");
    assert!(refused.to_string().contains(KV_GUARD_MESSAGE), "{refused}");
    assert_eq!(version(conn), "3");
}

// ── f ───────────────────────────────────────────────────────────────────

#[test]
fn f_after_delete_fires_on_delete_and_not_on_or_replace() {
    let dir = TempDir::new().unwrap();
    let (path, db) = legacy_brain(&dir);
    let conn = db.conn();
    conn.execute_batch(MEMORY_EXT_SQL).unwrap();

    let recursive: i64 = conn.query_row("PRAGMA recursive_triggers", [], |r| r.get(0)).unwrap();
    assert_eq!(recursive, 0, "recursive triggers are off by default in this build");

    let memory = Memory::new(MemoryCategory::Events, "probe", "kept", "a row that is replaced");
    project_memory::store_scoped(conn, &memory, &scope(PROJECT)).unwrap();
    adopt(conn, &memory.id, None);

    // The store's own INSERT OR REPLACE (memkoshi storage.rs) and its own
    // DELETE, on its own connection.
    let storage = MemoryStorage::open_existing(&path).unwrap();
    storage.store_memory(&memory).unwrap();
    assert_eq!(ext_rows(conn, &memory.id), 1, "INSERT OR REPLACE must not fire the delete trigger");

    assert!(storage.delete_memory(&memory.id).unwrap());
    assert_eq!(ext_rows(conn, &memory.id), 0, "a plain DELETE must fire it");

    // The first half holds only while recursive triggers stay off: a
    // connection that turns them on loses the memory_ext row on a replace.
    let other = Memory::new(MemoryCategory::Events, "probe", "lost", "a row that is replaced");
    project_memory::store_scoped(conn, &other, &scope(PROJECT)).unwrap();
    adopt(conn, &other.id, None);
    conn.execute_batch("PRAGMA recursive_triggers = ON").unwrap();
    conn.execute(
        "INSERT OR REPLACE INTO memories (id, category, topic, title, content, created)
         VALUES (?1, 'events', 'probe', 'lost', 'replaced', '2026-01-01T00:00:00Z')",
        [&other.id],
    )
    .unwrap();
    assert_eq!(ext_rows(conn, &other.id), 0);
}

// ── g ───────────────────────────────────────────────────────────────────

#[test]
fn g_json_set_and_json_extract_exist() {
    let dir = TempDir::new().unwrap();
    let (_path, db) = legacy_brain(&dir);
    let conn = db.conn();

    // Migration 1's statement, on the row Brain::create wrote.
    let changed = conn
        .execute(
            "UPDATE brain_meta SET value = json_set(value, '$.schema_version', 2) WHERE key = 'meta'",
            [],
        )
        .unwrap();
    assert_eq!(changed, 1);
    let (version, model): (i64, String) = conn
        .query_row(
            "SELECT json_extract(value, '$.schema_version'), json_extract(value, '$.embedder_model')
               FROM brain_meta WHERE key = 'meta'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(version, 2);
    assert_eq!(model, "all-MiniLM-L6-v2", "json_set keeps the rest of the object");
}

// ── h ───────────────────────────────────────────────────────────────────

#[test]
fn h_instr_finds_an_id_in_the_supersedes_array() {
    let dir = TempDir::new().unwrap();
    let (_path, db) = legacy_brain(&dir);
    let conn = db.conn();

    let old = store(conn, PROJECT, "old", "the first statement of a fact");
    let unrelated = store(conn, PROJECT, "unrelated", "a row that names nobody");
    let newer = Memory::new(MemoryCategory::Events, "probe", "new", "the corrected statement");
    let links = RecordLinks { supersedes: vec![old.clone()], ..Default::default() };
    project_memory::store_scoped_typed_with_links(conn, &newer, &scope(PROJECT), "fact", &links)
        .unwrap();

    let stored: String = conn
        .query_row("SELECT supersedes FROM memories WHERE id = ?1", [&newer.id], |r| r.get(0))
        .unwrap();
    assert_eq!(stored, format!("[\"{old}\"]"), "the array as the store writes it");

    // The scrub's statement (spec section 2, delete_row), ?2 = the id in double quotes.
    let naming = |id: &str| {
        sorted(ids(
            conn,
            "SELECT id FROM memories WHERE superseded_by = :id OR instr(supersedes, :quoted)
                 OR instr(contradicts, :quoted) OR instr(confirmed_by, :quoted)",
            named_params! { ":id": id, ":quoted": format!("\"{id}\"") },
            0,
        ))
    };
    assert_eq!(naming(&old), vec![newer.id.clone()]);
    assert_eq!(naming(&newer.id), vec![old.clone()], "the reverse edge the store keeps");
    assert!(naming(&unrelated).is_empty());
    // The quotes are what make it exact: a shorter id that is a prefix of a
    // stored one is not found.
    assert!(naming(&old[..old.len() - 1]).is_empty());
}

// ── i ───────────────────────────────────────────────────────────────────

/// Dot product of `query` with every row of `data`; the best `k`, best first.
fn exact_scan(data: &[f32], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = data
        .chunks_exact(query.len())
        .enumerate()
        .map(|(i, row)| (i, row.iter().zip(query).map(|(a, b)| a * b).sum::<f32>()))
        .collect();
    scored.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored
}

#[test]
#[ignore = "timing probe: run once with --ignored --nocapture"]
fn i_exact_scan_of_25000_vectors() {
    const ROWS: usize = 25_000;
    const DIM: usize = 384;
    const TOP: usize = 64;
    const TARGET: usize = 17_321;

    // xorshift64: the same vectors on every run.
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let mut data = vec![0.0f32; ROWS * DIM];
    for row in data.chunks_exact_mut(DIM) {
        for value in row.iter_mut() {
            *value = next();
        }
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        for value in row.iter_mut() {
            *value /= norm;
        }
    }
    let query = data[TARGET * DIM..(TARGET + 1) * DIM].to_vec();

    let mut millis = Vec::new();
    let mut top = Vec::new();
    for _ in 0..9 {
        let started = std::time::Instant::now();
        top = exact_scan(&data, &query, TOP);
        millis.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    millis.sort_by(|a, b| a.total_cmp(b));

    assert_eq!(top.len(), TOP);
    assert_eq!(top[0].0, TARGET);
    assert!((top[0].1 - 1.0).abs() < 1e-4);

    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    println!(
        "probe i: exact scan of {ROWS} x {DIM} f32, top {TOP}: median {:.2} ms, best {:.2} ms, worst {:.2} ms over {} runs ({profile} build)",
        millis[millis.len() / 2],
        millis[0],
        millis[millis.len() - 1],
        millis.len(),
    );
}
