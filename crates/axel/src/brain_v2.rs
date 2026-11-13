//! The migrations ledger, `open_v2`, `require_v2`, `connect`, `begin_write`,
//! `begin_write_front`, `stand_aside`, `signer`, adoption and the word-index fill.

use std::os::unix::fs::DirBuilderExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use axel_memkoshi::security::MemorySigner;
use chrono::{DateTime, NaiveDateTime, Utc};
use rusqlite::types::{Value, ValueRef};
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
    /// Memories this open could not adopt because their `id` or `content` is not readable
    /// text (finding 5). They are still in `memories`, untouched.
    pub unreadable_memories: usize,
    /// The device and inode numbers of the file `conn` opened, as recorded for `replaced`.
    // RULING: v2 r5.
    pub file: Option<(u64, u64)>,
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
    /// Nothing in `sqlite_master`: a crash before the creation committed leaves such a file.
    empty: bool,
}

fn peek(path: &Path) -> Result<Peek> {
    let conn = Connection::open_with_flags(
        file_uri(path, "mode=ro"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(damaged)?;
    // Another process may be creating or recovering the file at this moment (finding 8).
    crate::busy::set(&conn, Duration::from_secs(5))?;
    let version = user_version(&conn).map_err(damaged)?;
    // `open_v2` and `require_v2` look here first, before anything is written.
    if version == LATEST {
        check_shape(&conn, path)?;
    }
    let empty: bool = conn
        .query_row("SELECT NOT EXISTS(SELECT 1 FROM sqlite_master)", [], |r| r.get(0))
        .map_err(damaged)?;
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
    Ok(Peek { user_version: version, meta_version, has_rows, empty })
}

fn too_new(file: i32) -> AxelError {
    AxelError::BrainTooNew { file: i64::from(file), binary: i64::from(LATEST) }
}

// ── The file this process opened ────────────────────────────────────────────

// RULING: v2 r5. A brain file replaced or gone is never written (section 7), through any
// connection. `open_v2` records the device and inode numbers of the file it opened, under
// the name SQLite gives that file; a later connection to the name, and every `begin_write`
// on it, is refused with `BrainChanged` when the name no longer leads to that file. The
// status tick alone came too late: the front thread connects by path once the indexer
// reports the brain open, and a connection made after a rename was the only one on the new
// file, so its close checkpointed the old process's `-wal` into it.
static OPENED: Mutex<Vec<(String, (u64, u64))>> = Mutex::new(Vec::new());

fn opened() -> MutexGuard<'static, Vec<(String, (u64, u64))>> {
    OPENED.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The device and inode numbers of the file at `path`. Unix only: elsewhere there is no
/// number to compare, nothing is recorded, and nothing is refused.
#[cfg(unix)]
pub fn file_id(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path).ok().map(|m| (m.dev(), m.ino()))
}

#[cfg(not(unix))]
pub fn file_id(_path: &Path) -> Option<(u64, u64)> {
    None
}

/// True when `path` certainly no longer names the file `id`: a stat that gives another file,
/// or none (`NotFound`).
// RULING: v2 r5 round 2, item 1. Any other failed stat (EIO, ESTALE, EACCES: a network or
// FUSE file system, a parent made unreadable) says nothing about the file: it is not taken
// as replaced, and is looked at again next time. Said once in the log.
#[cfg(unix)]
pub fn moved(path: &Path, id: (u64, u64)) -> bool {
    use std::os::unix::fs::MetadataExt;
    static SAID: AtomicBool = AtomicBool::new(false);
    match std::fs::metadata(path) {
        Ok(m) => (m.dev(), m.ino()) != id,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
        Err(e) => {
            if !SAID.swap(true, Ordering::Relaxed) {
                crate::ext::server::log_line(&format!(
                    "brain file could not be looked at ({:?}): not taken as replaced",
                    e.kind()
                ));
            }
            false
        }
    }
}

#[cfg(not(unix))]
pub fn moved(_path: &Path, _id: (u64, u64)) -> bool {
    false
}

/// What `open_v2` recorded for the file of `conn`; `None` when it opened no such file.
fn recorded(conn: &Connection) -> Option<(u64, u64)> {
    let name = conn.path().filter(|name| !name.is_empty())?;
    opened().iter().find(|(n, _)| n == name).map(|(_, id)| *id)
}

/// True when `open_v2` opened the file of `conn` in this process and its path now names
/// another file, or none.
pub fn replaced(conn: &Connection) -> bool {
    match (recorded(conn), conn.path()) {
        (Some(id), Some(name)) => moved(Path::new(name), id),
        _ => false,
    }
}

/// Gives up a connection. One to a brain file this process has seen replaced or gone, or
/// whose path now names another file or none, is leaked, not closed.
// RULING: v2 r5 round 2, item 2. SQLite's close checkpoints and then unlinks `<path>-wal`
// and `<path>-shm` by name when it holds the last connection; those names now belong to
// the brain at the path. SQLite 3.46 skips both when its own stat of the path shows another
// inode (`databaseIsUnmoved`), but the rule does not lean on that: the connection is never
// closed, and the process ends with it open.
pub fn release(conn: Option<Connection>, status: &Status) {
    let Some(conn) = conn else { return };
    if status.brain_changed() || replaced(&conn) {
        std::mem::forget(conn);
    }
}

// ── Connections ─────────────────────────────────────────────────────────────

/// The pragmas of `connect`. With `check`, `user_version` is read before any pragma is
/// set and anything but `LATEST` is refused. With `record` (`open_v2`'s own connection) the
/// file it opened is recorded; without, a file that is no longer the recorded one is
/// refused.
// SPEC: 955 reads the version after the pragmas. It is read first here: `journal_mode=WAL`
// rewrites the header of a file that is not in WAL mode, and a refused file must not be
// written. The answer is the same.
fn connect_raw(path: &Path, role: Role, check: bool, record: bool) -> Result<Connection> {
    let before = if record { file_id(path) } else { None };
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(damaged)?;
    // RULING: v2 r5. Here, before anything is read: a connection that has read nothing
    // closes without touching the file or its `-wal`.
    if record {
        // RULING: v2 r5 round 2, item 1.
        if before.is_some_and(|id| moved(path, id)) {
            return Err(AxelError::BrainChanged);
        }
        let name = conn.path().unwrap_or_default().to_string();
        let mut opened = opened();
        opened.retain(|(n, _)| *n != name);
        if let (Some(id), false) = (before, name.is_empty()) {
            opened.push((name, id));
        }
    } else if replaced(&conn) {
        return Err(AxelError::BrainChanged);
    }
    let (busy_ms, synchronous) = match role {
        Role::Indexer => (5_000, "NORMAL"),
        Role::Front => (1_500, "FULL"),
        Role::Cli => (5_000, "FULL"),
    };
    crate::busy::set(&conn, Duration::from_millis(busy_ms))?;
    if check && user_version(&conn).map_err(damaged)? != LATEST {
        return Err(AxelError::BrainNotReady);
    }
    if check {
        check_shape(&conn, path)?;
    }
    conn.execute_batch(&format!(
        "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA secure_delete=ON; \
         PRAGMA synchronous={synchronous};"
    ))
    .map_err(damaged)?;
    Ok(conn)
}

/// What `check_shape` says of a brain it refuses; the extension's degraded line and
/// `status` say it too (round 5, item 2).
pub fn old_shape() -> String {
    format!(
        "was made by an older build of Axel and lacks tables this build writes; it is not changed. Move it aside and make it again (\"{cmd} init\", or \"{cmd} migrate init --from-axel <axel.r8>\")",
        cmd = crate::cmd_name()
    )
}

/// Whether `e` is the refusal of `check_shape`.
pub fn is_old_shape(e: &AxelError) -> bool {
    matches!(e, AxelError::Other(m) if m.contains(&old_shape()))
}

/// The unique key of `forgotten_turns` that every upsert of a mark names (its conflict
/// target, which matches a key of these columns in any order).
const MARK_KEY: [&str; 6] = ["logical", "source", "from_ms", "turn_key", "archive_id", "text_hash"];

/// A version-1 v2 brain made before the round-3 shape: `forgotten_turns` without
/// `text_hash` in its unique key, or `archive_index` without `cli_units` and
/// `cli_unproven`. Its first mark would fail ("does not match any UNIQUE constraint").
// RULING: v2 c18b review round 4, item 6. No version bump: no v2 brain is in use yet; one
// made by an older build is refused, unchanged, and made again.
// RULING: v2 c18b review round 5, item 3: the columns of each unique index are read row by
// row and compared as sorted sets; nothing leans on the order of an aggregate.
fn check_shape(conn: &Connection, path: &Path) -> Result<()> {
    let columns: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('archive_index') WHERE name IN ('cli_units', 'cli_unproven')",
            [],
            |r| r.get(0),
        )
        .map_err(damaged)?;
    let mut want: Vec<&str> = MARK_KEY.to_vec();
    want.sort_unstable();
    let mut list = conn.prepare("SELECT name FROM pragma_index_list('forgotten_turns') WHERE \"unique\" = 1").map_err(damaged)?;
    let indexes: Vec<String> =
        list.query_map([], |r| r.get(0)).map_err(damaged)?.collect::<rusqlite::Result<_>>().map_err(damaged)?;
    let mut info = conn.prepare("SELECT name FROM pragma_index_info(?1)").map_err(damaged)?;
    let mut keyed = false;
    for index in &indexes {
        let mut names: Vec<Option<String>> =
            info.query_map([index], |r| r.get(0)).map_err(damaged)?.collect::<rusqlite::Result<_>>().map_err(damaged)?;
        names.sort_unstable();
        keyed |= names.iter().map(Option::as_deref).eq(want.iter().map(|w| Some(*w)));
    }
    if columns == 2 && keyed {
        return Ok(());
    }
    Err(AxelError::Other(format!("the brain at {} {}.", path.display(), old_shape())))
}

/// Pragmas only; never creates, never migrates.
// SPEC: 922 says of a missing file only "error, nothing created": `NoBrain`, whose message
// names the two commands that make one.
pub fn connect(path: &Path, role: Role) -> Result<Connection> {
    if !path.exists() {
        return Err(AxelError::NoBrain(path.to_path_buf()));
    }
    connect_raw(path, role, true, false)
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
        // RULING: wave2-c4-review finding 8: only "not found" says there is no log. Any
        // other answer (no permission, a link that loops, something that is not a file) is
        // a log this cannot rule out, and the source is refused.
        match std::fs::metadata(&side) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Ok(m) if m.is_file() && m.len() == 0 => {}
            Ok(_) => {
                return Err(AxelError::Other(format!(
                    "{} has a non-empty {} beside it; checkpoint it or give a copy without one",
                    path.display(),
                    suffix
                )))
            }
            Err(e) => {
                return Err(AxelError::Other(format!(
                    "{}: cannot look at the {} beside it: {e}",
                    path.display(),
                    suffix
                )))
            }
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
    let mut fresh = !path.exists();
    if !fresh {
        let seen = peek(path)?;
        if seen.user_version > LATEST {
            return Err(too_new(seen.user_version));
        }
        match seen.meta_version {
            // RULING: wave2-c4-review finding 4: a SQLite file with no tables at all is what
            // a creation that did not finish leaves, or what another process is creating at
            // this moment. It is treated as missing, not as damaged.
            None if seen.empty => fresh = true,
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
    let brain = if fresh {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::DirBuilder::new().recursive(true).mode(0o700).create(parent)?;
            }
        }
        // The file is 0600 before the signing key is written (finding 7, in `r8.rs`).
        Brain::create_or_join(path, opts.agent, opts.model, opts.dim)?
    } else {
        Brain::open(path)?
    };
    let signer = brain.signer();
    let embedder_model = brain.meta().embedder_model.clone();
    drop(brain);

    // 3 to 5. Our connection, JR's layer, velocirag's tables.
    // RULING: v2 r5. This connection records the file: what this process writes from now on.
    let conn = connect_raw(path, Role::Indexer, false, true)?;
    // RULING: wave2-c4-review finding 4: both layers look for a column and add it later, so
    // of two processes on a file that lacks the columns one fails with "duplicate column
    // name". Steps 4 and 5 run again, once, when they fail. Velocirag's statements run in
    // one transaction of ours, so a second opener waits for the whole of step 5 and a
    // second run never meets a step 5 that is half done.
    let layers = || -> Result<()> {
        crate::project_memory::migrate(&conn)?;
        let own = connect_raw(path, Role::Indexer, false, false)?;
        own.execute_batch("BEGIN IMMEDIATE")?;
        let db = Database::from_connection(own, path.to_path_buf())?;
        commit(db.conn())
    };
    if layers().is_err() {
        layers()?;
    }

    // 6. The ledger.
    // RULING: wave2-c4-review finding 8: a migration commits with `synchronous = FULL`, so
    // a power cut after `open_v2` returned cannot undo it. The connection is the indexer's
    // afterwards and goes back to `NORMAL`.
    conn.execute_batch("PRAGMA synchronous=FULL")?;
    for n in (user_version(&conn)? + 1)..=LATEST {
        migrate_step(&conn, n)?;
    }
    conn.execute_batch("PRAGMA synchronous=NORMAL")?;
    let now = user_version(&conn)?;
    if now != LATEST {
        return Err(too_new(now));
    }

    // 7. Adoption and the word index, for rows written after migration 1 without a
    // `memory_ext` row. What migration 1 found it adopted itself, as `legacy`.
    let unreadable_memories =
        adopt_orphans_counted(&conn, if opts.upgrade_v1 { "legacy" } else { "cli" })?.skipped;

    // RULING: v2 r5.
    let file = recorded(&conn);
    Ok(V2Brain { conn, signer, embedder_model, path: path.to_path_buf(), unreadable_memories, file })
}

/// One migration in one transaction. Another process may have won: the version is read
/// again under the write lock. True when this call ran the migration.
fn migrate_step(conn: &Connection, n: i32) -> Result<bool> {
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
        Ok(true) => commit(conn).map(|()| true),
        Ok(false) => {
            conn.execute_batch("ROLLBACK")?;
            Ok(false)
        }
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

/// `BEGIN IMMEDIATE`, the version check, and the file: one replaced or gone since
/// `open_v2` opened it is `BrainChanged`.
pub fn begin_write(conn: &Connection) -> Result<()> {
    conn.execute_batch("BEGIN IMMEDIATE")?;
    match user_version(conn) {
        // RULING: v2 r5. Under the write lock, as the version is.
        Ok(v) if v == LATEST && replaced(conn) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(AxelError::BrainChanged)
        }
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
// RULING: v2 c19 round, foreman item 1: `pub(crate)`, so that `import`'s catch-up reads
// `created` by this function and not by a copy of it.
pub(crate) fn created_ms(created: &Value) -> (i64, &'static str) {
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
    adopt_orphans_counted(conn, origin).map(|a| a.adopted)
}

/// What one adoption did.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Adoption {
    /// Memories that got their `memory_ext` row in this call.
    pub adopted: usize,
    /// Memories without a `memory_ext` row whose `id` or `content` is not text, or not valid
    /// UTF-8. They are left as they are, and counted again by every call.
    pub skipped: usize,
}

/// `adopt_orphans`, with the number of rows it could not read.
// RULING: wave2-c4-review finding 5: one unreadable `memories` row no longer blocks the
// adoption of all the others; it is skipped and counted. The same holds inside migration 1.
// Such a row stays pending, so a brain that has one takes the write lock at every call.
pub fn adopt_orphans_counted(conn: &Connection, origin: &str) -> Result<Adoption> {
    if origin != "legacy" && origin != "cli" {
        return Err(AxelError::Other(format!("adopt_orphans: origin {origin:?}")));
    }
    let pending: i64 = conn.query_row(
        "SELECT (SELECT COUNT(*) FROM memories m
                 WHERE NOT EXISTS (SELECT 1 FROM memory_ext e WHERE e.mem_id = m.id))
              + (SELECT COUNT(*) FROM memories WHERE id NOT IN (SELECT mem_id FROM memories_fts))",
        [],
        |r| r.get(0),
    )?;
    if pending == 0 {
        return Ok(Adoption::default());
    }
    begin_write(conn)?;
    // RULING: wave2-c4-review finding 2: any error after the `BEGIN`, a failed `COMMIT`
    // included, rolls back: this is the indexer's long-lived connection.
    match adopt_in_transaction(conn, origin) {
        Ok(done) => commit(conn).map(|()| done),
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

/// The column as a string; `None` when it is not text or not valid UTF-8.
pub(crate) fn text_at(row: &rusqlite::Row, index: usize) -> rusqlite::Result<Option<String>> {
    Ok(match row.get_ref(index)? {
        ValueRef::Text(bytes) => std::str::from_utf8(bytes).ok().map(str::to_string),
        _ => None,
    })
}

/// `memories.created` as `created_ms` takes it: a text that is not valid UTF-8 is `Null`,
/// every other value as SQLite holds it.
// RULING: v2 c19 round, foreman item 1: read with `get_ref`, never `get::<Value>`: rusqlite
// 0.32 panics on a text that is not UTF-8. Such a `created` is not a time: `created_ms`
// gives it the time of adoption and `observed`.
pub(crate) fn created_at(row: &rusqlite::Row, index: usize) -> rusqlite::Result<Value> {
    Ok(match row.get_ref(index)? {
        ValueRef::Text(bytes) => match std::str::from_utf8(bytes) {
            Ok(s) => Value::Text(s.to_string()),
            Err(_) => Value::Null,
        },
        other => Value::from(other),
    })
}

fn adopt_in_transaction(conn: &Connection, origin: &str) -> Result<Adoption> {
    let model: Option<String> = conn
        .query_row(
            "SELECT json_extract(value, '$.embedder_model') FROM brain_meta WHERE key = 'meta'",
            [],
            |r| r.get::<_, Option<String>>(0),
        )
        .optional()?
        .flatten();

    // The vector comes from the shadow `documents` row `index_memory` wrote; it stays.
    let mut orphans: Vec<Orphan> = Vec::new();
    let mut skipped = 0;
    {
        let mut stmt = conn.prepare(
            "SELECT m.id, m.content, m.created, d.doc_id IS NOT NULL, d.embedding
             FROM memories m LEFT JOIN documents d ON d.doc_id = m.id
             WHERE NOT EXISTS (SELECT 1 FROM memory_ext e WHERE e.mem_id = m.id)",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let (Some(id), Some(content)) = (text_at(r, 0)?, text_at(r, 1)?) else {
                skipped += 1;
                continue;
            };
            // A shadow vector that is not a blob is not a vector: the row is adopted without.
            let embedding = match r.get_ref(4)? {
                ValueRef::Blob(bytes) => Some(bytes.to_vec()),
                _ => None,
            };
            // RULING: v2 c19 round, foreman item 1: `created_at`, not `r.get(2)`.
            orphans.push(Orphan { id, content, created: created_at(r, 2)?, shadow: r.get(3)?, embedding });
        }
    }

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
    // RULING: wave4 R-12b item 3. The same test as the adoption above (`text_at` on `id` and
    // `content`), row by row: a row that is not adopted gets no word row either. A title or
    // an abstract that cannot be read is indexed as empty.
    let mut words: Vec<(String, String, String, String)> = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT m.id, m.content, m.title, m.abstract_text, COALESCE(m.sensitivity = 'secret', 0)
             FROM memories m WHERE m.id NOT IN (SELECT mem_id FROM memories_fts)",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let (Some(id), Some(content)) = (text_at(r, 0)?, text_at(r, 1)?) else { continue };
            let title = text_at(r, 2)?.unwrap_or_default();
            let secret: bool = r.get(4)?;
            let (abstract_text, content) =
                if secret { (String::new(), String::new()) } else { (text_at(r, 3)?.unwrap_or_default(), content) };
            words.push((id, title, abstract_text, content));
        }
    }
    {
        let mut insert =
            conn.prepare("INSERT INTO memories_fts(mem_id, title, abstract, content) VALUES (?1, ?2, ?3, ?4)")?;
        for (id, title, abstract_text, content) in &words {
            insert.execute(params![id, title, abstract_text, content])?;
        }
    }
    Ok(Adoption { adopted, skipped })
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
    // RULING: v2 r5 round 2, item 2.
    if reason == DegradedReason::BrainChanged {
        status.brain_replaced.store(true, Ordering::SeqCst);
    }
    status.reason.store(reason as u8, Ordering::SeqCst);
    mode.store(Mode::Degraded as u8, Ordering::SeqCst);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The "another process won the migration" branch, set up directly: the version this
    /// connection read before it asked for the write lock is no longer the file's.
    #[test]
    fn migrate_step_does_nothing_when_another_opener_won() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.r8");
        let opts = OpenOpts { agent: None, model: "hash-384", dim: 384, upgrade_v1: false };
        let brain = open_v2(&path, &opts).unwrap();
        let conn = &brain.conn;
        let ledger = |c: &Connection| -> (i64, String) {
            c.query_row("SELECT COUNT(*), MAX(applied_at) FROM axel_migrations", [], |r| Ok((r.get(0)?, r.get(1)?)))
                .unwrap()
        };
        let before = ledger(conn);
        assert_eq!(before.0, 1);

        assert!(!migrate_step(conn, 1).unwrap(), "migration 1 ran a second time");
        assert!(conn.is_autocommit());
        assert_eq!(ledger(conn), before);
        assert_eq!(user_version(conn).unwrap(), 1);
        // It left no lock behind.
        let other = Connection::open(&path).unwrap();
        crate::busy::set(&other, Duration::ZERO).unwrap();
        other.execute_batch("BEGIN IMMEDIATE; ROLLBACK;").unwrap();
    }
}
