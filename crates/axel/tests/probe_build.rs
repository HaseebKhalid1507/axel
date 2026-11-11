//! Build probe for the v2 rewrite: assumptions the plan rests on, settled
//! here before anything depends on them. No production code is exercised
//! that this file adds; the old tables come from the existing definitions
//! (`r8.rs`, `project_memory.rs`, velocirag's `db.rs`) and the v2 tables from
//! `fixtures/schema_v2.sql`, the spec's block marked "migration 1" pasted
//! whole. The few other v2 statements are copied from the spec as text.
//!
//! | # | assumption |
//! |---|---|
//! | a | `Embedder` and `rusqlite::Connection` are `Send` |
//! | b | `SQLITE_OPEN_NOFOLLOW` exists and a `?mode=ro&immutable=1` URI opens |
//! | c | recall generators a and c run as written on the existing tables |
//! | d | FTS5 `secure-delete` + `PRAGMA secure_delete` leave no trace of a deleted row, option first or rows first, small or bulk writes; text deleted before the option goes only with `optimize` |
//! | e | the `kv` guard triggers stop `INSERT OR REPLACE … '2'`, and let `'3'` through |
//! | f | on the store's own connection: its `INSERT OR REPLACE` fires neither delete trigger; a plain delete is stopped by the guard unless `axel_purge` has its row |
//! | g | `json_set` and `json_extract` exist in the bundled SQLite |
//! | h | the link scrub `purge_many` runs (`<> '[]'`, filtered in Rust), and the probe table's `instr(supersedes, '"<id>"')`, find an id in the arrays the store writes |
//! | i | the cost of an exact scan of 25,000 vectors (`#[ignore]`, prints ms) |
//! | j | the suite runs with `HOME` moved: not a test here, it is the gate itself (`scripts/dirty-home.sh`) |
//! | k | `VACUUM` / `VACUUM INTO` on an old brain: what residue goes, whether every table's rowids survive (measured and printed) |
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

/// Migration 1's definitions: the spec's block, pasted whole (the one copy).
/// It runs on a brain that already has `memories`, `kv`, `documents` and the
/// two word indexes; `legacy_brain` makes them.
const SCHEMA_V2_SQL: &str = include_str!("fixtures/schema_v2.sql");

const DOCUMENTS_V2_SQL: &str = "
ALTER TABLE documents ADD COLUMN source_mtime_ms INTEGER;
ALTER TABLE documents ADD COLUMN hidden TEXT;
";

const KV_GUARD_MESSAGE: &str = "this brain belongs to Axel v2; use the axel-v2 binary";
const DELETE_GUARD_MESSAGE: &str = "this brain belongs to Axel v2; delete with axel-v2 forget";

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
    conn.execute_batch(SCHEMA_V2_SQL).unwrap();
    conn.execute_batch(DOCUMENTS_V2_SQL).unwrap();

    // Generator a, eight rows: every arm of its WHERE runs.
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

    // expires_at in the past, in the future; a legacy row with no project_key.
    let now_utc = chrono::Utc::now();
    let store_expiring = |title: &str, content: &str, expires_at| {
        let mut memory = Memory::new(MemoryCategory::Events, "probe", title, content);
        memory.expires_at = Some(expires_at);
        project_memory::store_scoped(conn, &memory, &scope(PROJECT)).unwrap();
        memory.id
    };
    let expired = store_expiring("expired", "alpha in an expired row", now_utc - chrono::Duration::days(1));
    let future = store_expiring("future", "alpha in a row that expires later", now_utc + chrono::Duration::days(1));
    let unkeyed = store(conn, GLOBAL_KEY, "unkeyed", "beta in a legacy row with no project");
    conn.execute("UPDATE memories SET project_key = NULL WHERE id = ?1", [&unkeyed]).unwrap();
    for id in [&expired, &future] {
        let set: Option<String> = conn
            .query_row("SELECT expires_at FROM memories WHERE id = ?1", [id], |r| r.get(0))
            .unwrap();
        assert!(set.is_some(), "the store wrote expires_at");
    }
    let key: Option<String> = conn
        .query_row("SELECT project_key FROM memories WHERE id = ?1", [&unkeyed], |r| r.get(0))
        .unwrap();
    assert_eq!(key, None);
    adopt(conn, &expired, None);
    adopt(conn, &future, None);
    adopt(conn, &unkeyed, None);

    let expr = "\"alpha\" OR \"beta\"";
    let now = now_utc.to_rfc3339();
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
        sorted(vec![both.clone(), one.clone(), future.clone(), global.clone(), unkeyed.clone()]),
        "the project's live rows, the global row and the unkeyed one; not the hidden, the expired or the other project's"
    );
    let found = run_a(PROJECT, false);
    assert_eq!(found[0], both);
    assert_eq!(sorted(found), sorted(vec![both.clone(), one.clone(), future.clone()]));
    assert_eq!(
        sorted(run_a("no row has this", true)),
        sorted(vec![global.clone(), unkeyed.clone()]),
        "with the global flag: the global row and the row with a NULL project_key"
    );
    assert!(run_a("no row has this", false).is_empty());
    // Without the expiry arm the expired row would be found: the arm is what
    // keeps it out.
    let unfiltered = ids(
        conn,
        "SELECT m.id FROM memories_fts f JOIN memories m ON m.id = f.mem_id
          WHERE memories_fts MATCH :expr AND m.project_key = :scope_dir",
        named_params! { ":expr": expr, ":scope_dir": PROJECT },
        0,
    );
    assert!(unfiltered.contains(&expired));

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
/// Case "old residue": the row deleted before the option is on.
const EARLY_MEMORY_MARKER: &str = "vkjearlymem4417rpq";
const EARLY_CHUNK_MARKER: &str = "wbnearlydoc9023hxm";

/// Hits of one row's markers in the file, one count per word index. The tail
/// of each marker is counted too, because the word index stores a term
/// without the prefix it shares with its neighbour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Hits {
    memory: usize,
    chunk: usize,
}

impl Hits {
    const NONE: Hits = Hits { memory: 0, chunk: 0 };

    fn in_file(path: &Path, memory_marker: &str, chunk_marker: &str) -> Hits {
        let bytes = file_bytes(path);
        let count = |m: &str| occurrences(&bytes, m) + occurrences(&bytes, &m[9..]);
        Hits { memory: count(memory_marker), chunk: count(chunk_marker) }
    }

    fn in_both(self) -> bool {
        self.memory > 0 && self.chunk > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FtsOption {
    Never,
    BeforeRows,
    AfterRows,
}

fn secure_delete_pragma_on(conn: &Connection) {
    conn.execute_batch("PRAGMA secure_delete = ON").unwrap();
    let on: i64 = conn.query_row("PRAGMA secure_delete", [], |r| r.get(0)).unwrap();
    assert_eq!(on, 1);
}

/// Migration 1's two statements. FTS5 accepts the option on both word
/// indexes, and keeps it.
fn fts_secure_delete_on(conn: &Connection) {
    conn.execute("INSERT INTO memories_fts(memories_fts, rank) VALUES('secure-delete', 1)", [])
        .unwrap();
    conn.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('secure-delete', 1)", [])
        .unwrap();
    for table in ["memories_fts_config", "chunks_fts_config"] {
        let kept: i64 = conn
            .query_row(&format!("SELECT v FROM {table} WHERE k = 'secure-delete'"), [], |r| r.get(0))
            .unwrap();
        assert_eq!(kept, 1, "{table}");
    }
}

/// `rows` memories and `rows` note chunks. Each `(i, memory marker, chunk
/// marker)` in `marked` makes row `i` of both carry the markers; its chunk is
/// `doomed-<k>` for the k-th entry. With `one_transaction` all memories go in
/// one transaction and all chunks in another (one fresh segment per word
/// index); otherwise every row is its own transaction. Returns the marked
/// memories' ids, in the order of `marked`.
fn write_rows(db: &Database, rows: usize, one_transaction: bool, marked: &[(usize, &str, &str)]) -> Vec<String> {
    let conn = db.conn();
    let mark = |i: usize| marked.iter().position(|m| m.0 == i);
    let memories: Vec<Memory> = (0..rows)
        .map(|i| match mark(i) {
            Some(k) => Memory::new(
                MemoryCategory::Events,
                "probe",
                "doomed",
                format!("the word is {} today", marked[k].1),
            ),
            None => Memory::new(
                MemoryCategory::Events,
                "probe",
                "kept",
                format!("kept row {i} about harbours and lanterns"),
            ),
        })
        .collect();
    if one_transaction {
        let items: Vec<(Memory, MemoryScope)> =
            memories.iter().map(|m| (m.clone(), scope(PROJECT))).collect();
        project_memory::store_scoped_batch(conn, &items).unwrap();
    } else {
        for memory in &memories {
            project_memory::store_scoped(conn, memory, &scope(PROJECT)).unwrap();
        }
    }

    let zeros = vec![0.0f32; velocirag::EMBEDDING_DIM];
    let meta = serde_json::json!({});
    if one_transaction {
        conn.execute_batch("BEGIN IMMEDIATE").unwrap();
    }
    for i in 0..rows {
        let (doc_id, content, file_path) = match mark(i) {
            Some(k) => (format!("doomed-{k}"), format!("the word is {} today", marked[k].2), "/notes/doomed.md"),
            None => (format!("kept-{i}"), format!("kept chunk {i} about harbours and lanterns"), "/notes/kept.md"),
        };
        db.insert_document(&doc_id, &content, &meta, &zeros, Some(file_path))
            .unwrap_or_else(|e| panic!("insert {doc_id}: {e}"));
    }
    if one_transaction {
        conn.execute_batch("COMMIT").unwrap();
    }
    marked.iter().map(|m| memories[m.0].id.clone()).collect()
}

/// The memory row, its word-index row, the chunk and its word-index row.
fn purge(conn: &Connection, mem_id: &str, doc_id: &str) {
    conn.execute("DELETE FROM memories WHERE id = ?1", [mem_id]).unwrap();
    conn.execute("DELETE FROM memories_fts WHERE mem_id = ?1", [mem_id]).unwrap();
    conn.execute("DELETE FROM documents WHERE doc_id = ?1", [doc_id]).unwrap();
    conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?1", [doc_id]).unwrap();
}

/// Write `rows` rows, the middle one carrying the markers, delete it,
/// checkpoint, and return the markers' hits before and after. `PRAGMA
/// secure_delete` is on throughout, as on every v2 connection; `option` says
/// when FTS5's `secure-delete` is switched on, relative to the writes.
fn marker_hits_around_a_purge(option: FtsOption, rows: usize, one_transaction: bool) -> (Hits, Hits) {
    let dir = TempDir::new().unwrap();
    let (path, db) = legacy_brain(&dir);
    let conn = db.conn();
    secure_delete_pragma_on(conn);

    if option == FtsOption::BeforeRows {
        fts_secure_delete_on(conn);
    }
    let doomed = write_rows(&db, rows, one_transaction, &[(rows / 2, MEMORY_MARKER, CHUNK_MARKER)]);
    if option == FtsOption::AfterRows {
        fts_secure_delete_on(conn);
    }

    checkpoint_truncate(conn, &path);
    let before = Hits::in_file(&path, MEMORY_MARKER, CHUNK_MARKER);
    purge(conn, &doomed[0], "doomed-0");
    checkpoint_truncate(conn, &path);
    let after = Hits::in_file(&path, MEMORY_MARKER, CHUNK_MARKER);
    println!("probe d: option {option:?}, {rows} rows, one transaction {one_transaction}: before {before:?}, after {after:?}");
    (before, after)
}

fn fts5_secure_delete_exists() {
    // FTS5's 'secure-delete' option needs SQLite 3.42.
    assert!(rusqlite::version_number() >= 3_042_000, "bundled SQLite is {}", rusqlite::version());
}

#[test]
fn d_secure_delete_leaves_no_marker_in_the_file() {
    fts5_secure_delete_exists();

    let (before, after) = marker_hits_around_a_purge(FtsOption::BeforeRows, 5, false);
    assert!(before.in_both(), "each marker must be findable before the delete: {before:?}");
    assert_eq!(after, Hits::NONE, "a deleted row's marker is still in the file");

    // The option is what does it. With only PRAGMA secure_delete the tables
    // are clean but EACH word index still holds its deleted word.
    let (before, after) = marker_hits_around_a_purge(FtsOption::Never, 5, false);
    assert!(before.in_both(), "{before:?}");
    assert!(after.in_both(), "without FTS5 secure-delete each word index was expected to keep its marker: {after:?}");
}

#[test]
fn d_rows_first_then_the_option_then_the_delete() {
    // Migration 1's order: the word indexes already hold the rows when the
    // option is switched on.
    fts5_secure_delete_exists();
    let (before, after) = marker_hits_around_a_purge(FtsOption::AfterRows, 5, false);
    assert!(before.in_both(), "{before:?}");
    assert_eq!(after, Hits::NONE, "rows first, then the option: the marker is still in the file");
}

#[test]
fn d_rows_first_in_big_segments_then_the_option() {
    fts5_secure_delete_exists();
    // One 6,000-row transaction: the marker sits inside one large segment.
    let (before, after) = marker_hits_around_a_purge(FtsOption::AfterRows, 6_000, true);
    assert!(before.in_both(), "{before:?}");
    assert_eq!(after, Hits::NONE, "6,000 rows in one segment: the marker is still in the file");

    // 400 transactions: an index built by many small writes and merges.
    let (before, after) = marker_hits_around_a_purge(FtsOption::AfterRows, 400, false);
    assert!(before.in_both(), "{before:?}");
    assert_eq!(after, Hits::NONE, "400 transactions: the marker is still in the file");
}

/// Delete a row with the option off, switch it on, securely delete another
/// row, checkpoint; then `optimize` both word indexes and checkpoint again.
/// Returns the first row's hits after the secure delete, the second row's
/// hits then, and the first row's hits after `optimize`.
/// `pragma_for_early_delete`: whether `PRAGMA secure_delete` is already on
/// for the first delete. With it on, the table rows (`memories`, `documents`,
/// the word indexes' `_content`) are zeroed, so what is left is the word
/// indexes' segments: the question V-1 asks. With it off, as the old binary
/// ran, free pages of ordinary tables may keep text too (spec 2.7 says
/// nothing in version 1 removes that).
fn old_residue(pragma_for_early_delete: bool) -> (Hits, Hits, Hits) {
    let dir = TempDir::new().unwrap();
    let (path, db) = legacy_brain(&dir);
    let conn = db.conn();
    if pragma_for_early_delete {
        secure_delete_pragma_on(conn);
    }
    let ids = write_rows(
        &db,
        6,
        false,
        &[(2, EARLY_MEMORY_MARKER, EARLY_CHUNK_MARKER), (4, MEMORY_MARKER, CHUNK_MARKER)],
    );
    checkpoint_truncate(conn, &path);
    assert!(Hits::in_file(&path, EARLY_MEMORY_MARKER, EARLY_CHUNK_MARKER).in_both());
    assert!(Hits::in_file(&path, MEMORY_MARKER, CHUNK_MARKER).in_both());

    purge(conn, &ids[0], "doomed-0");
    secure_delete_pragma_on(conn);
    fts_secure_delete_on(conn);
    purge(conn, &ids[1], "doomed-1");
    checkpoint_truncate(conn, &path);
    let early = Hits::in_file(&path, EARLY_MEMORY_MARKER, EARLY_CHUNK_MARKER);
    let late = Hits::in_file(&path, MEMORY_MARKER, CHUNK_MARKER);

    conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('optimize')", []).unwrap();
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')", []).unwrap();
    checkpoint_truncate(conn, &path);
    let early_optimized = Hits::in_file(&path, EARLY_MEMORY_MARKER, EARLY_CHUNK_MARKER);
    println!(
        "probe d, old residue (PRAGMA secure_delete on for the first delete: {pragma_for_early_delete}): \
         first row {early:?}, second row {late:?}, first row after optimize {early_optimized:?}"
    );
    (early, late, early_optimized)
}

#[test]
fn d_text_deleted_before_the_option_stays_until_optimize() {
    fts5_secure_delete_exists();
    let (early, late, early_optimized) = old_residue(true);
    assert_eq!(late, Hits::NONE, "the row deleted with the option on must be gone");
    assert!(early.in_both(), "the row deleted before the option was expected to stay in each word index: {early:?}");
    assert_eq!(early_optimized, Hits::NONE, "after optimize the old residue must be gone (spec 2.7, V-1)");

    // As the old binary ran: PRAGMA secure_delete off for the first delete.
    // Measured and printed only; spec 2.7 makes no claim about free pages.
    old_residue(false);
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

    conn.execute_batch(SCHEMA_V2_SQL).unwrap();
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
fn f_store_replace_passes_the_delete_guard_and_a_plain_delete_does_not() {
    let dir = TempDir::new().unwrap();
    let (path, db) = legacy_brain(&dir);
    drop(db);

    // ONE connection for everything below: the store's own. The pragma is
    // read, the block runs and every write is made on it.
    let storage = MemoryStorage::open_existing(&path).unwrap();
    let conn = storage.conn();
    let recursive: i64 = conn.query_row("PRAGMA recursive_triggers", [], |r| r.get(0)).unwrap();
    assert_eq!(recursive, 0, "recursive triggers are off on the store's connection");
    conn.execute_batch(SCHEMA_V2_SQL).unwrap();

    let purge_rows = |conn: &Connection| -> i64 {
        conn.query_row("SELECT COUNT(*) FROM axel_purge", [], |r| r.get(0)).unwrap()
    };
    let memory_rows = |conn: &Connection, id: &str| -> i64 {
        conn.query_row("SELECT COUNT(*) FROM memories WHERE id = ?1", [id], |r| r.get(0)).unwrap()
    };

    let memory = Memory::new(MemoryCategory::Events, "probe", "kept", "a row that is replaced");
    project_memory::store_scoped(conn, &memory, &scope(PROJECT)).unwrap();
    adopt(conn, &memory.id, None);

    // 1. The store's INSERT OR REPLACE (memkoshi storage.rs:182) over an
    //    existing row, with no axel_purge row: it succeeds, the guard does
    //    not fire, and neither does memories_ad_ext.
    assert_eq!(purge_rows(conn), 0);
    storage.store_memory(&memory).unwrap();
    assert_eq!(memory_rows(conn, &memory.id), 1);
    assert_eq!(ext_rows(conn, &memory.id), 1, "INSERT OR REPLACE must not fire the delete triggers");
    assert_eq!(purge_rows(conn), 0, "the replace leaves no axel_purge row");

    // 2. A plain DELETE without the row aborts with the guard's message, and
    //    changes nothing.
    let delete = |conn: &Connection, id: &str| conn.execute("DELETE FROM memories WHERE id = ?1", [id]);
    let refused = delete(conn, &memory.id).unwrap_err();
    assert!(refused.to_string().contains(DELETE_GUARD_MESSAGE), "{refused}");
    assert_eq!(refused.sqlite_error_code(), Some(ErrorCode::ConstraintViolation));
    assert_eq!(memory_rows(conn, &memory.id), 1);
    assert_eq!(ext_rows(conn, &memory.id), 1);
    // The store's own delete is the same statement and is stopped the same way.
    let refused = storage.delete_memory(&memory.id).unwrap_err();
    assert!(refused.to_string().contains(DELETE_GUARD_MESSAGE), "{refused}");

    // 3. The same delete passes with the row, and memories_ad_ext takes the
    //    memory_ext row with it.
    conn.execute("INSERT INTO axel_purge(one) VALUES (1)", []).unwrap();
    assert_eq!(delete(conn, &memory.id).unwrap(), 1);
    assert_eq!(memory_rows(conn, &memory.id), 0);
    assert_eq!(ext_rows(conn, &memory.id), 0, "a plain DELETE must fire memories_ad_ext");
    conn.execute("DELETE FROM axel_purge", []).unwrap();

    // Part 1 holds only while recursive triggers stay off: with them on, the
    // replace's implicit delete fires the guard and the store's write aborts.
    let other = Memory::new(MemoryCategory::Events, "probe", "lost", "a row that is replaced");
    project_memory::store_scoped(conn, &other, &scope(PROJECT)).unwrap();
    adopt(conn, &other.id, None);
    conn.execute_batch("PRAGMA recursive_triggers = ON").unwrap();
    let refused = storage.store_memory(&other).unwrap_err();
    assert!(refused.to_string().contains(DELETE_GUARD_MESSAGE), "{refused}");
    assert_eq!(ext_rows(conn, &other.id), 1);
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
fn h_link_scrub_finds_an_id_in_the_arrays_the_store_writes() {
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
    let empty: (String, String, String) = conn
        .query_row(
            "SELECT supersedes, contradicts, confirmed_by FROM memories WHERE id = ?1",
            [&unrelated],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(empty, ("[]".into(), "[]".into(), "[]".into()), "an empty link is the text []");

    // The spec has two forms. The one purge_many runs (section 2.7, step 2):
    // one read over the purge set, filtered in Rust to the rows whose arrays
    // name an id of the set.
    conn.execute_batch("CREATE TEMP TABLE purge_ids(id TEXT PRIMARY KEY)").unwrap();
    let scrub = |id: &str| -> (Vec<String>, Vec<String>) {
        conn.execute("DELETE FROM purge_ids", []).unwrap();
        conn.execute("INSERT INTO purge_ids(id) VALUES (?1)", [id]).unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT id, supersedes, contradicts, confirmed_by FROM memories
   WHERE superseded_by IN purge_ids OR supersedes <> '[]' OR contradicts <> '[]' OR
   confirmed_by <> '[]'",
            )
            .unwrap();
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, [r.get::<_, String>(1)?, r.get(2)?, r.get(3)?])))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        let read = sorted(rows.iter().map(|(id, _)| id.clone()).collect());
        let naming = sorted(
            rows.into_iter()
                .filter(|(_, arrays)| {
                    arrays.iter().any(|a| {
                        serde_json::from_str::<Vec<String>>(a).unwrap().iter().any(|x| x == id)
                    })
                })
                .map(|(id, _)| id)
                .collect(),
        );
        (read, naming)
    };
    let (read, naming) = scrub(&old);
    assert_eq!(read, vec![newer.id.clone()], "rows with only empty arrays are not read");
    assert_eq!(naming, vec![newer.id.clone()]);
    // The reverse edge: the read returns the row by superseded_by, but its
    // select list has no superseded_by, so the filter on arrays drops it.
    let (read, naming) = scrub(&newer.id);
    assert!(read.contains(&old), "superseded_by IN purge_ids reads the reverse edge");
    assert!(naming.is_empty(), "the arrays of the superseded row name nothing");
    let (read, naming) = scrub(&unrelated);
    assert_eq!(read, vec![newer.id.clone()]);
    assert!(naming.is_empty());
    // A shorter id that is a prefix of a stored one is not named.
    assert!(scrub(&old[..old.len() - 1]).1.is_empty());

    // The form the probe table names (spec section 9, row h): instr over the
    // arrays, ?2 = the id in double quotes.
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
// Release run on the build machine (bbuild, 5dc78ff): median 5.37 ms, best 5.07 ms,
// worst 6.79 ms over 9 runs. The debug profile measured 89 ms median.
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

// ── k ───────────────────────────────────────────────────────────────────

/// One table of the file and what identifies its rows.
struct TableKeys {
    name: String,
    has_rowid: bool,
    /// One `INTEGER` column is the whole primary key: an alias of the rowid.
    integer_pk: bool,
    /// `(rowid, quote(first column))` in rowid order; for a WITHOUT ROWID
    /// table `(None, quote(first column))` in key order.
    keys: Vec<(Option<i64>, String)>,
}

/// Every table in the file, virtual and shadow tables included.
fn table_keys(conn: &Connection) -> Vec<TableKeys> {
    let names = ids(conn, "SELECT name FROM sqlite_master WHERE type IN ('table') ORDER BY name", &[], 0);
    names
        .into_iter()
        .map(|name| {
            let mut info = conn.prepare(&format!("PRAGMA table_info(\"{name}\")")).unwrap();
            let columns: Vec<(String, String, i64)> = info
                .query_map([], |r| Ok((r.get(1)?, r.get(2)?, r.get(5)?)))
                .unwrap()
                .collect::<rusqlite::Result<_>>()
                .unwrap();
            let pk: Vec<_> = columns.iter().filter(|c| c.2 > 0).collect();
            let first = &columns[0].0;
            let by_rowid = conn.prepare(&format!(
                "SELECT rowid, quote(\"{first}\") FROM \"{name}\" ORDER BY rowid"
            ));
            let has_rowid = by_rowid.is_ok();
            let keys = match by_rowid {
                Ok(mut stmt) => stmt
                    .query_map([], |r| Ok((Some(r.get(0)?), r.get(1)?)))
                    .unwrap()
                    .collect::<rusqlite::Result<Vec<_>>>()
                    .unwrap(),
                Err(_) => conn
                    .prepare(&format!("SELECT quote(\"{first}\") FROM \"{name}\""))
                    .unwrap()
                    .query_map([], |r| Ok((None, r.get(0)?)))
                    .unwrap()
                    .collect::<rusqlite::Result<Vec<_>>>()
                    .unwrap(),
            };
            let integer_pk = has_rowid && pk.len() == 1 && pk[0].1.eq_ignore_ascii_case("INTEGER");
            TableKeys { name, has_rowid, integer_pk, keys }
        })
        .collect()
}

/// One query per word index on rows that were not deleted: `rowid:id`, in
/// rank order.
fn word_queries(conn: &Connection) -> (Vec<String>, Vec<String>) {
    (
        ids(
            conn,
            "SELECT rowid || ':' || mem_id FROM memories_fts WHERE memories_fts MATCH 'harbours' ORDER BY rank",
            &[],
            0,
        ),
        ids(
            conn,
            "SELECT rowid || ':' || doc_id FROM chunks_fts WHERE chunks_fts MATCH 'harbours' ORDER BY rank",
            &[],
            0,
        ),
    )
}

/// `PRAGMA integrity_check`, and FTS5's own check of each word index.
fn integrity(conn: &Connection) -> (String, String, String) {
    let fts = |table: &str| match conn
        .execute(&format!("INSERT INTO {table}({table}) VALUES('integrity-check')"), [])
    {
        Ok(_) => "ok".to_string(),
        Err(e) => e.to_string(),
    };
    (
        ids(conn, "PRAGMA integrity_check", &[], 0).join("; "),
        fts("memories_fts"),
        fts("chunks_fts"),
    )
}

/// An old brain with the old binary's residue, migration 1's block (and, with
/// `optimize`, the option and the two `optimize` statements of spec section 1),
/// then `VACUUM` in place or `VACUUM INTO` a second file. Asserts only that
/// the fixture is what the probe needs; prints what it measures.
fn vacuum_probe(optimize: bool, into: bool) {
    const ROWS: usize = 40;
    let dir = TempDir::new().unwrap();
    let (path, db) = legacy_brain(&dir);
    let conn = db.conn();
    let pragma: i64 = conn.query_row("PRAGMA secure_delete", [], |r| r.get(0)).unwrap();
    assert_eq!(pragma, 0, "the old binary's connection has PRAGMA secure_delete off");

    // The marked row and two plain ones, deleted as the old binary deletes:
    // gaps in every rowid sequence.
    let doomed = write_rows(&db, ROWS, false, &[(20, MEMORY_MARKER, CHUNK_MARKER)]);
    purge(conn, &doomed[0], "doomed-0");
    for i in [5, 30] {
        let id: String = conn
            .query_row(
                "SELECT id FROM memories WHERE content = ?1",
                [format!("kept row {i} about harbours and lanterns")],
                |r| r.get(0),
            )
            .unwrap();
        purge(conn, &id, &format!("kept-{i}"));
    }
    checkpoint_truncate(conn, &path);
    let residue = Hits::in_file(&path, MEMORY_MARKER, CHUNK_MARKER);
    assert!(residue.in_both(), "the old binary's delete must leave the markers: {residue:?}");
    for table in ["memories", "documents"] {
        let n: i64 = conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0)).unwrap();
        assert_eq!(n as usize, ROWS - 3, "{table}");
    }

    conn.execute_batch(SCHEMA_V2_SQL).unwrap();
    conn.execute(
        "INSERT INTO memory_ext (mem_id, origin_kind, content_hash, created_ms)
         SELECT id, 'legacy', 'probe', 1 FROM memories",
        [],
    )
    .unwrap();
    if optimize {
        fts_secure_delete_on(conn);
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('optimize')", []).unwrap();
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')", []).unwrap();
    }
    checkpoint_truncate(conn, &path);
    let before = Hits::in_file(&path, MEMORY_MARKER, CHUNK_MARKER);
    let keys_before = table_keys(conn);
    let queries_before = word_queries(conn);
    assert_eq!(queries_before.0.len(), ROWS - 3);
    assert_eq!(queries_before.1.len(), ROWS - 3);
    let size_before = std::fs::metadata(&path).unwrap().len();

    let source_bytes = file_bytes(&path);
    let target = dir.path().join("into.r8");
    let started = std::time::Instant::now();
    if into {
        conn.execute_batch(&format!("VACUUM INTO '{}'", target.display())).unwrap();
    } else {
        conn.execute_batch("VACUUM").unwrap();
    }
    let millis = started.elapsed().as_secs_f64() * 1000.0;
    let source_unchanged = into.then(|| file_bytes(&path) == source_bytes);
    if !into {
        checkpoint_truncate(conn, &path);
    }
    let checked = if into { target } else { path.clone() };
    // Read before anything opens the new file.
    let after = Hits::in_file(&checked, MEMORY_MARKER, CHUNK_MARKER);
    let size_after = std::fs::metadata(&checked).unwrap().len();

    let opened;
    let check: &Connection = if into {
        opened = Connection::open(&checked).unwrap();
        &opened
    } else {
        conn
    };
    let keys_after = table_keys(check);
    let queries_same = word_queries(check) == queries_before;
    let (integrity_check, fts_memories, fts_chunks) = integrity(check);

    let label = format!(
        "{} after {}",
        if into { "VACUUM INTO" } else { "VACUUM" },
        if optimize { "block + secure-delete + optimize" } else { "block only" }
    );
    println!("probe k, {label}:");
    println!("  markers: old binary's delete {residue:?}, before the vacuum {before:?}, after {after:?}");
    println!("  size: {size_before} -> {size_after} bytes; vacuum took {millis:.1} ms");
    println!("  word queries same ids, same order: {queries_same}");
    println!("  integrity_check: {integrity_check}; fts integrity-check memories_fts: {fts_memories}, chunks_fts: {fts_chunks}");
    if let Some(unchanged) = source_unchanged {
        println!("  source file's bytes unchanged: {unchanged}");
    }
    for table in &keys_before {
        let verdict = match keys_after.iter().find(|t| t.name == table.name) {
            None => "MISSING after".to_string(),
            Some(t) if t.keys == table.keys => "same".to_string(),
            Some(t) => {
                let moved = table.keys.iter().zip(&t.keys).filter(|(a, b)| a != b).count();
                format!("CHANGED: {} -> {} rows, {moved} pairs differ", table.keys.len(), t.keys.len())
            }
        };
        println!(
            "  {:<28} rowid {:<5} integer pk {:<5} {:>4} rows  {verdict}",
            table.name,
            table.has_rowid,
            table.integer_pk,
            table.keys.len()
        );
    }
    for table in keys_after.iter().filter(|t| !keys_before.iter().any(|b| b.name == t.name)) {
        println!("  {:<28} NEW after the vacuum, {} rows", table.name, table.keys.len());
    }
}

#[test]
fn k_what_vacuum_removes_and_whether_rowids_survive() {
    fts5_secure_delete_exists();
    for optimize in [false, true] {
        for into in [false, true] {
            vacuum_probe(optimize, into);
        }
    }
}
