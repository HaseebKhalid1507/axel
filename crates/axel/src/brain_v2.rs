//! The migrations ledger, `open_v2`, `require_v2`, `connect`, `begin_write`,
//! `begin_write_front`, `stand_aside`, `signer`, adoption and the word-index fill.

use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use axel_memkoshi::security::MemorySigner;
use chrono::{DateTime, NaiveDateTime, Utc};
use rusqlite::types::Value;
use rusqlite::{params, Connection, ErrorCode, OpenFlags, OptionalExtension};
use sha2::{Digest, Sha256};
use velocirag::db::Database;

use crate::error::{AxelError, Result};
use crate::ext::types::{DegradedReason, Mode, Status};
use crate::r8::{Brain, SCHEMA_VERSION, V2_SCHEMA_VERSION};

pub const LATEST: i32 = 1;

/// The definitions of migration 1: the spec's block, byte for byte (V-2).
const MIGRATION_1_DEFINITIONS: &str = include_str!("../tests/fixtures/schema_v2.sql");

pub struct OpenOpts<'a> {
    pub agent: Option<&'a str>,
    pub model: &'a str,
    pub dim: usize,
    /// `migrate init` only.
    pub upgrade_v1: bool,
}

pub struct V2Brain {
    pub conn: Connection,
    pub signer: Option<MemorySigner>,
    pub embedder_model: String,
    pub path: PathBuf,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Indexer,
    Front,
    Cli,
}

/// The front's write. The caller commits. When the guard is dropped it rolls back a
/// transaction that is still open, then clears the `writing` flag (R-6).
// RULING: wave2-c4-review finding 2: the spec's guard only clears the flag. An early return
// between `begin_write_front` and `COMMIT` would leave the front connection holding the
// write lock, so the guard holds the connection too and ends what the caller did not.
pub struct WriteGuard<'a> {
    conn: &'a Connection,
    writing: &'a AtomicBool,
    /// `begin_write` succeeded: the open transaction is this guard's. A transaction that
    /// was already open when `begin_write` refused is the caller's and is left alone.
    began: bool,
}

impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        if self.began && !self.conn.is_autocommit() {
            let _ = self.conn.execute_batch("ROLLBACK");
        }
        self.writing.store(false, Ordering::SeqCst);
    }
}

/// `COMMIT`; when it fails the transaction is rolled back, so the connection is never left
/// holding the write lock.
fn commit(conn: &Connection) -> Result<()> {
    match conn.execute_batch("COMMIT") {
        Ok(()) => Ok(()),
        Err(e) => {
            if !conn.is_autocommit() {
                let _ = conn.execute_batch("ROLLBACK");
            }
            Err(AxelError::Db(e))
        }
    }
}

// ── Reading a file without writing to it ────────────────────────────────────

/// `NotADatabase` and `DatabaseCorrupt` say the file is damaged; anything else stays what
/// it is (busy, no permission).
fn damaged(e: rusqlite::Error) -> AxelError {
    match e.sqlite_error_code() {
        Some(ErrorCode::NotADatabase) | Some(ErrorCode::DatabaseCorrupt) => {
            AxelError::BrainDamaged(e.to_string())
        }
        _ => AxelError::Db(e),
    }
}

/// `file:<path>?<query>`, with the three characters a URI gives a meaning to escaped.
fn file_uri(path: &Path, query: &str) -> String {
    let mut out = String::from("file:");
    for c in path.to_string_lossy().chars() {
        match c {
            '%' => out.push_str("%25"),
            '?' => out.push_str("%3f"),
            '#' => out.push_str("%23"),
            c => out.push(c),
        }
    }
    out.push('?');
    out.push_str(query);
    out
}

fn user_version(conn: &Connection) -> rusqlite::Result<i32> {
    conn.query_row("PRAGMA user_version", [], |r| r.get(0))
}

fn table_exists(conn: &Connection, name: &str) -> rusqlite::Result<bool> {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        [name],
        |r| r.get(0),
    )
}

/// What a read-only look at the file says.
struct Peek {
    user_version: i32,
    /// `brain_meta.meta.schema_version`; `None` when there is no such table or row.
    meta_version: Option<i64>,
    /// A row in `memories` or `documents`.
    has_rows: bool,
}

fn peek(path: &Path) -> Result<Peek> {
    let conn = Connection::open_with_flags(
        file_uri(path, "mode=ro"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(damaged)?;
    let version = user_version(&conn).map_err(damaged)?;
    let meta_version = if table_exists(&conn, "brain_meta").map_err(damaged)? {
        conn.query_row(
            "SELECT json_extract(value, '$.schema_version') FROM brain_meta WHERE key = 'meta'",
            [],
            |r| r.get::<_, Option<i64>>(0),
        )
        .optional()
        .map_err(damaged)?
        .flatten()
    } else {
        None
    };
    let mut has_rows = false;
    for table in ["memories", "documents"] {
        if table_exists(&conn, table).map_err(damaged)? {
            let any: bool = conn
                .query_row(&format!("SELECT EXISTS(SELECT 1 FROM {table})"), [], |r| r.get(0))
                .map_err(damaged)?;
            has_rows = has_rows || any;
        }
    }
    Ok(Peek { user_version: version, meta_version, has_rows })
}

fn too_new(file: i32) -> AxelError {
    AxelError::BrainTooNew { file: i64::from(file), binary: i64::from(LATEST) }
}

// ── Connections ─────────────────────────────────────────────────────────────

/// The pragmas of `connect`. With `check`, `user_version` is read before any pragma is
/// set and anything but `LATEST` is refused.
// SPEC: 955 reads the version after the pragmas. It is read first here: `journal_mode=WAL`
// rewrites the header of a file that is not in WAL mode, and a refused file must not be
// written. The answer is the same.
fn connect_raw(path: &Path, role: Role, check: bool) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(damaged)?;
    let (busy_ms, synchronous) = match role {
        Role::Indexer => (5_000, "NORMAL"),
        Role::Front => (1_500, "FULL"),
        Role::Cli => (5_000, "FULL"),
    };
    conn.busy_timeout(Duration::from_millis(busy_ms))?;
    if check && user_version(&conn).map_err(damaged)? != LATEST {
        return Err(AxelError::BrainNotReady);
    }
    conn.execute_batch(&format!(
        "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA secure_delete=ON; \
         PRAGMA synchronous={synchronous};"
    ))
    .map_err(damaged)?;
    Ok(conn)
}

/// Pragmas only; never creates, never migrates.
// SPEC: 922 says of a missing file only "error, nothing created": `NoBrain`, whose message
// names the two commands that make one.
pub fn connect(path: &Path, role: Role) -> Result<Connection> {
    if !path.exists() {
        return Err(AxelError::NoBrain(path.to_path_buf()));
    }
    connect_raw(path, role, true)
}

/// A source of `migrate synaps`: read-only, immutable, and the file itself may not be a
/// symbolic link. The folder is resolved first (P-1).
// SPEC: 1172 refuses a non-empty `-wal` or `-journal` beside the source without saying
// where. It is refused here, before the open: `immutable=1` would read past it.
pub fn open_source(path: &Path) -> Result<Connection> {
    let name = path
        .file_name()
        .ok_or_else(|| AxelError::Other(format!("not a file: {}", path.display())))?;
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let resolved = parent.canonicalize()?.join(name);
    for suffix in ["-wal", "-journal"] {
        let mut side = resolved.clone().into_os_string();
        side.push(suffix);
        if std::fs::metadata(&side).map(|m| m.len() > 0).unwrap_or(false) {
            return Err(AxelError::Other(format!(
                "{} has a non-empty {} beside it; checkpoint it or give a copy without one",
                path.display(),
                suffix
            )));
        }
    }
    let conn = Connection::open_with_flags(
        file_uri(&resolved, "mode=ro&immutable=1"),
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NOFOLLOW
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    Ok(conn)
}

// ── Opening ─────────────────────────────────────────────────────────────────

/// May create, may migrate.
pub fn open_v2(path: &Path, opts: &OpenOpts) -> Result<V2Brain> {
    // 1. Look, read-only. Nothing is written to the main file when this refuses.
    let exists = path.exists();
    if exists {
        let seen = peek(path)?;
        if seen.user_version > LATEST {
            return Err(too_new(seen.user_version));
        }
        match seen.meta_version {
            // SPEC: 960-964 names no answer for a file SQLite reads that is not an Axel
            // brain (no `brain_meta` row). `Brain::open` would set WAL on it and then fail:
            // refused here, unwritten.
            None => {
                return Err(AxelError::BrainDamaged(format!(
                    "{} has no brain_meta row; it is not an Axel brain",
                    path.display()
                )))
            }
            Some(v)
                if seen.user_version == 0
                    && v == i64::from(SCHEMA_VERSION)
                    && seen.has_rows
                    && !opts.upgrade_v1 =>
            {
                return Err(AxelError::V1Brain(path.to_path_buf()))
            }
            // RULING: wave2-c4-review finding 3: the spec leaves this file to `Brain::open`,
            // which sets `journal_mode=WAL` before it reads the version and then answers
            // `SchemaMismatch`. A `brain_meta` version that is neither 1 nor 2 is refused
            // here, before any pragma, as `BrainDamaged` so that `degrade_on` acts on it.
            Some(v) if v != i64::from(SCHEMA_VERSION) && v != i64::from(V2_SCHEMA_VERSION) => {
                return Err(AxelError::BrainDamaged(format!(
                    "{} says brain_meta schema version {v}; this binary reads {SCHEMA_VERSION} and {V2_SCHEMA_VERSION}",
                    path.display()
                )))
            }
            Some(_) => {}
        }
    }

    // 2. The old tables, the signer and the model name.
    let brain = if exists {
        Brain::open(path)?
    } else {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::DirBuilder::new().recursive(true).mode(0o700).create(parent)?;
            }
        }
        let brain = Brain::create_with_model(path, opts.agent, opts.model, opts.dim)?;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        brain
    };
    let signer = brain.signer();
    let embedder_model = brain.meta().embedder_model.clone();
    drop(brain);

    // 3 to 5. Our connection, JR's layer, velocirag's tables.
    let conn = connect_raw(path, Role::Indexer, false)?;
    crate::project_memory::migrate(&conn)?;
    drop(Database::from_connection(connect_raw(path, Role::Indexer, false)?, path.to_path_buf())?);

    // 6. The ledger.
    for n in (user_version(&conn)? + 1)..=LATEST {
        migrate_step(&conn, n)?;
    }
    let now = user_version(&conn)?;
    if now != LATEST {
        return Err(too_new(now));
    }

    // 7. Adoption and the word index, for rows written after migration 1 without a
    // `memory_ext` row. What migration 1 found it adopted itself, as `legacy`.
    adopt_orphans(&conn, if opts.upgrade_v1 { "legacy" } else { "cli" })?;

    Ok(V2Brain { conn, signer, embedder_model, path: path.to_path_buf() })
}

/// One migration in one transaction. Another process may have won: the version is read
/// again under the write lock.
fn migrate_step(conn: &Connection, n: i32) -> Result<()> {
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let run = (|| -> Result<bool> {
        if user_version(conn)? >= n {
            return Ok(false);
        }
        let name = match n {
            1 => {
                migration_1(conn)?;
                "brain v2"
            }
            _ => return Err(AxelError::Other(format!("no migration {n}"))),
        };
        conn.execute(
            "INSERT INTO axel_migrations(version, name, applied_at) VALUES (?1, ?2, ?3)",
            params![n, name, Utc::now().to_rfc3339()],
        )?;
        conn.execute_batch(&format!("PRAGMA user_version = {n}"))?;
        Ok(true)
    })();
    match run {
        Ok(true) => commit(conn),
        Ok(false) => Ok(conn.execute_batch("ROLLBACK")?),
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let names = stmt.query_map([], |r| r.get::<_, String>(1))?;
    for name in names {
        if name? == column {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The block, the statements that depend on what the file already has, and the adoption of
/// the memories that are there. The caller holds the transaction.
fn migration_1(conn: &Connection) -> Result<()> {
    conn.execute_batch(MIGRATION_1_DEFINITIONS)?;
    for (column, ddl) in [
        ("source_mtime_ms", "ALTER TABLE documents ADD COLUMN source_mtime_ms INTEGER"),
        ("hidden", "ALTER TABLE documents ADD COLUMN hidden TEXT"),
    ] {
        if !column_exists(conn, "documents", column)? {
            conn.execute_batch(ddl)?;
        }
    }
    // RULING: wave2-c4-review finding 1: the spec adopts in step 7, after migration 1 has
    // committed. Every memory that exists when migration 1 runs predates version 2, whoever
    // the opener is: it is adopted here, in migration 1's transaction, as `legacy`. A run
    // that stops before the commit leaves a version 1 brain; one that stops after it leaves
    // nothing for step 7 to label `cli`.
    adopt_in_transaction(conn, "legacy")?;
    conn.execute(
        "UPDATE brain_meta SET value = json_set(value, '$.schema_version', ?1) WHERE key = 'meta'",
        [V2_SCHEMA_VERSION],
    )?;
    conn.execute_batch(
        "INSERT INTO memories_fts(memories_fts, rank) VALUES('secure-delete', 1);
         INSERT INTO chunks_fts(chunks_fts, rank)   VALUES('secure-delete', 1);
         INSERT INTO memories_fts(memories_fts) VALUES('optimize');
         INSERT INTO chunks_fts(chunks_fts)     VALUES('optimize');",
    )?;
    Ok(())
}

// ── The gate ────────────────────────────────────────────────────────────────

/// Read-only; the command line's gate (R-2).
// SPEC: 909 lists `V1Brain`, `BrainTooNew` and `BrainNotReady`; a file SQLite cannot read
// is none of them and answers `BrainDamaged`, as `open_v2` does (961).
pub fn require_v2(path: &Path) -> Result<()> {
    if !path.exists() {
        return Err(AxelError::NoBrain(path.to_path_buf()));
    }
    let seen = peek(path)?;
    if seen.user_version == LATEST {
        Ok(())
    } else if seen.user_version > LATEST {
        Err(too_new(seen.user_version))
    } else if seen.meta_version == Some(i64::from(SCHEMA_VERSION)) {
        Err(AxelError::V1Brain(path.to_path_buf()))
    } else {
        Err(AxelError::BrainNotReady)
    }
}

// ── Writing ─────────────────────────────────────────────────────────────────

/// `BEGIN IMMEDIATE` and the version check.
pub fn begin_write(conn: &Connection) -> Result<()> {
    conn.execute_batch("BEGIN IMMEDIATE")?;
    match user_version(conn) {
        Ok(v) if v == LATEST => Ok(()),
        Ok(_) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(AxelError::BrainChanged)
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(damaged(e))
        }
    }
}

/// Sets the flag, then `begin_write`. The guard clears the flag when dropped (R-6), also
/// when `begin_write` fails, and rolls back a transaction the caller did not commit.
pub fn begin_write_front<'a>(conn: &'a Connection, writing: &'a AtomicBool) -> Result<WriteGuard<'a>> {
    writing.store(true, Ordering::SeqCst);
    let mut guard = WriteGuard { conn, writing, began: false };
    begin_write(conn)?;
    guard.began = true;
    Ok(guard)
}

/// The indexer, before each transaction: waits in 2 ms steps while the flag is set, at
/// most 1 s.
pub fn stand_aside(writing: &AtomicBool) {
    let start = Instant::now();
    while writing.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(1) {
        std::thread::sleep(Duration::from_millis(2));
    }
}

/// For the front thread (B-2d); never logs the key. Built as `Brain::signer` builds it.
pub fn signer(conn: &Connection) -> Result<Option<MemorySigner>> {
    let key: Option<String> = conn
        .query_row(
            "SELECT json_extract(value, '$.signing_key') FROM brain_meta WHERE key = 'meta'",
            [],
            |r| r.get::<_, Option<String>>(0),
        )
        .optional()?
        .flatten();
    Ok(key.map(|hex_key| {
        let key = hex::decode(&hex_key).unwrap_or_else(|_| hex_key.as_bytes().to_vec());
        MemorySigner::new(&key)
    }))
}

// ── Adoption and the word index ─────────────────────────────────────────────

/// `memories.created` as milliseconds, and the `time_source` that says how sure it is.
// SPEC: 1134 takes `created_ms` from `m.created` and says nothing of a value that is not
// a time. The row is still adopted (verify needs a `memory_ext` row per memory), with the
// time of adoption and `observed`.
fn created_ms(created: &Value) -> (i64, &'static str) {
    if let Value::Text(s) = created {
        if let Ok(t) = DateTime::parse_from_rfc3339(s) {
            return (t.timestamp_millis(), "exact");
        }
        for format in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"] {
            if let Ok(t) = NaiveDateTime::parse_from_str(s, format) {
                return (t.and_utc().timestamp_millis(), "exact");
            }
        }
    }
    (Utc::now().timestamp_millis(), "observed")
}

/// `origin` is "legacy" or "cli"; also fills the word index (B-1b). Returns the number of
/// memories adopted. Takes the write lock only when there is something to do.
pub fn adopt_orphans(conn: &Connection, origin: &str) -> Result<usize> {
    if origin != "legacy" && origin != "cli" {
        return Err(AxelError::Other(format!("adopt_orphans: origin {origin:?}")));
    }
    let pending: i64 = conn.query_row(
        "SELECT (SELECT COUNT(*) FROM memories WHERE id NOT IN (SELECT mem_id FROM memory_ext))
              + (SELECT COUNT(*) FROM memories WHERE id NOT IN (SELECT mem_id FROM memories_fts))",
        [],
        |r| r.get(0),
    )?;
    if pending == 0 {
        return Ok(0);
    }
    begin_write(conn)?;
    // RULING: wave2-c4-review finding 2: any error after the `BEGIN`, a failed `COMMIT`
    // included, rolls back: this is the indexer's long-lived connection.
    match adopt_in_transaction(conn, origin) {
        Ok(n) => commit(conn).map(|()| n),
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

/// A `memories` row with no `memory_ext` row, and the vector of its shadow document.
struct Orphan {
    id: String,
    content: String,
    created: Value,
    shadow: bool,
    embedding: Option<Vec<u8>>,
}

fn adopt_in_transaction(conn: &Connection, origin: &str) -> Result<usize> {
    let model: Option<String> = conn
        .query_row(
            "SELECT json_extract(value, '$.embedder_model') FROM brain_meta WHERE key = 'meta'",
            [],
            |r| r.get::<_, Option<String>>(0),
        )
        .optional()?
        .flatten();

    // The vector comes from the shadow `documents` row `index_memory` wrote; it stays.
    let orphans: Vec<Orphan> = {
        let mut stmt = conn.prepare(
            "SELECT m.id, m.content, m.created, d.doc_id IS NOT NULL, d.embedding
             FROM memories m LEFT JOIN documents d ON d.doc_id = m.id
             WHERE m.id NOT IN (SELECT mem_id FROM memory_ext)",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(Orphan {
                id: r.get(0)?,
                content: r.get(1)?,
                created: r.get(2)?,
                shadow: r.get(3)?,
                embedding: r.get(4)?,
            })
        })?;
        rows.collect::<rusqlite::Result<_>>()?
    };

    let mut adopted = 0;
    {
        let mut insert = conn.prepare(
            "INSERT OR IGNORE INTO memory_ext(mem_id, origin_kind, origin_ref, content_hash, created_ms,
                                              time_source, embed_model, embedding)
             VALUES (?1, ?2, '', ?3, ?4, ?5, ?6, ?7)",
        )?;
        for o in &orphans {
            // SPEC: 639 says "sha256 of the normalised body" and names no function:
            // `archive::normalise` (1468) of `memories.content`.
            let hash = hex::encode(Sha256::digest(crate::archive::normalise(&o.content).as_bytes()));
            let (ms, time_source) = created_ms(&o.created);
            let embed_model = if o.shadow { model.as_deref() } else { None };
            adopted += insert.execute(params![o.id, origin, hash, ms, time_source, embed_model, o.embedding])?;
        }
    }

    // The word rows that are missing, and only those (B-3, R-9k). Never a rebuild.
    conn.execute(
        "INSERT INTO memories_fts(mem_id, title, abstract, content)
         SELECT m.id, m.title,
                CASE WHEN m.sensitivity = 'secret' THEN '' ELSE m.abstract_text END,
                CASE WHEN m.sensitivity = 'secret' THEN '' ELSE m.content END
         FROM memories m WHERE m.id NOT IN (SELECT mem_id FROM memories_fts)",
        [],
    )?;
    Ok(adopted)
}

// ── Degrading ───────────────────────────────────────────────────────────────

/// True when `err` says the brain is damaged or was changed by another binary; then the
/// reason is stored first and the mode after it. Any other error: false, nothing set.
pub fn degrade_on(err: &AxelError, status: &Status, mode: &AtomicU8) -> bool {
    let reason = match err {
        AxelError::BrainChanged => DegradedReason::BrainChanged,
        AxelError::BrainDamaged(_) => DegradedReason::BrainDamaged,
        AxelError::Db(e)
            if matches!(
                e.sqlite_error_code(),
                Some(ErrorCode::NotADatabase) | Some(ErrorCode::DatabaseCorrupt)
            ) =>
        {
            DegradedReason::BrainDamaged
        }
        _ => return false,
    };
    status.reason.store(reason as u8, Ordering::SeqCst);
    mode.store(Mode::Degraded as u8, Ordering::SeqCst);
    true
}
