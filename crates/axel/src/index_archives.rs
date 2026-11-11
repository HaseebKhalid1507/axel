//! Scan, index one archive, forget one archive, the forget marks (`mark_forgotten_turn`,
//! `chat_files`, `near_stretch`). Commits 13a and 13b write this file.
//!
//! Commit 13a: the scan, the job of `on_context_archived`, `index_archive`,
//! `forget_archive`, and the log after a purge. Commit 13b: the forget marks and what they
//! leave out (steps 3 and 3b of section 2.3), the twins of hidden captures (3a), and the
//! live call (`mark_forgotten_turn`). Nothing here writes, locks, renames or chmods
//! anything under `context-archives`.

use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

use rusqlite::{params, Connection, ErrorCode, OptionalExtension};

use crate::archive::{self, ArchiveEntry, ArchiveFile, ArchivePolicy, LiveArchive, SkipReason, Start, Unit, UnitKind};
use crate::brain_v2;
use crate::error::{AxelError, Result};
use crate::ext::server::log_line;
use crate::ext::types::{IndexerCtx, Job, Mode};
use crate::indexer_start;
use crate::project;
use crate::project_memory::{RecordLinks, Sensitivity};
use crate::store::{self, Inserted, NewRow};
use crate::vindex::{VecDelta, VecKey};

/// Kept in each `archive_index` row. Version 1 does not re-extract old rows.
pub const INDEXER_VERSION: i64 = 1;
/// Archives of one chat within this many milliseconds of each other are the parts of one
/// seal and share a window.
pub const PARTS_SPAN_MS: i64 = 10_000;

/// A long scan writes the status row once this much time has passed since the last one.
const STATUS_EVERY_MS: i64 = 30_000;
/// How long the idle checkpoint waits for a reader before it gives up until the next turn.
const CHECKPOINT_WAIT_MS: u64 = 100;
/// How often the tick tries the checkpoint while a purge mark exists (the idle turn tries
/// it every second; a loop that is never idle has only the tick).
pub const CHECKPOINT_EVERY_MS: i64 = 5_000;

// ── The log after a purge ───────────────────────────────────────────────────

/// A purge in this transaction forgot text: the write-ahead log still holds the frames
/// written when the row was inserted, and an automatic checkpoint does not shorten it.
/// Called INSIDE the transaction of every purge that forgets, on any thread, before its
/// `COMMIT`; the indexer then truncates the log at its next idle turn
/// ([`checkpoint_after_purge`]). A row and not a field, so that a purge made by the front
/// thread or the command line, or just before a crash, is seen too (section 2.7).
// RULING: wave4 R-13 round 3, item 3. The value changes with every purge: the time in
// nanoseconds, and one more than the mark already there when the clock gives no more. The
// checkpoint deletes the mark only when it still holds the value it read before the
// pragma, so a purge committed after the pragma keeps its mark.
pub fn mark_purged(conn: &Connection) -> Result<()> {
    let now_ns = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
    conn.execute(
        "INSERT OR REPLACE INTO context_data(key, layer, value)
         VALUES('purge_pending', 'axel',
                CAST(max(?1, COALESCE((SELECT CAST(value AS INTEGER) FROM context_data
                                        WHERE key = 'purge_pending' AND layer = 'axel'), 0) + 1) AS TEXT))",
        [now_ns],
    )?;
    Ok(())
}

/// The indexer's idle turn, and its tick every [`CHECKPOINT_EVERY_MS`] (called by
/// `indexer::run` and `indexer::tick` through `guarded`): when a purge is
/// pending, `wal_checkpoint(TRUNCATE)`; when the checkpoint was not blocked, the mark is
/// deleted in one small transaction, whose frame holds no text, if no purge wrote it again
/// since it was read. When it was blocked
/// (`busy` is 1) the mark stays and the next idle turn tries again (section 2.7).
pub fn checkpoint_after_purge(ctx: &mut IndexerCtx) -> Result<()> {
    checkpoint_after_purge_then(ctx, || {})
}

/// [`checkpoint_after_purge`], with `between` run after a checkpoint that was not blocked
/// and before the mark's delete: the seam where a test commits another purge.
#[doc(hidden)]
pub fn checkpoint_after_purge_then(ctx: &mut IndexerCtx, between: impl FnOnce()) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    // The mark as it is before the pragma: only this value is deleted afterwards.
    let mark: Option<String> = conn
        .query_row("SELECT value FROM context_data WHERE key = 'purge_pending' AND layer = 'axel'", [], |r| {
            r.get(0)
        })
        .optional()?;
    let Some(mark) = mark else { return Ok(()) };
    // SPEC: 2560-2564 do not say how long the checkpoint may wait. A `TRUNCATE` checkpoint
    // that is blocked by a reader waits for the connection's busy timeout (5 s on the
    // indexer) and keeps every other writer out while it waits, and an idle turn comes
    // every second. So it waits 100 ms here, and the connection's own timeout is put back
    // before anything else runs.
    let usual: i64 = conn.query_row("PRAGMA busy_timeout", [], |r| r.get(0))?;
    conn.busy_timeout(Duration::from_millis(CHECKPOINT_WAIT_MS))?;
    let answer = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get::<_, i64>(0));
    // RULING: wave4 R-13 item 3. A busy answer is "blocked", in whichever form it comes:
    // the pragma's row with `busy` = 1, an error that says busy or locked, or the lock of
    // the small transaction below not being free. The mark stays, the next turn tries
    // again, and nothing is counted as a write error.
    let done = match answer {
        Ok(0) => {
            between();
            match in_write(conn, || {
                conn.execute(
                    "DELETE FROM context_data WHERE key = 'purge_pending' AND layer = 'axel' AND value = ?1",
                    [&mark],
                )?;
                Ok(())
            }) {
                Err(e) if is_busy(&e) => Ok(()),
                other => other,
            }
        }
        Ok(_) => Ok(()),
        Err(e) => match AxelError::from(e) {
            e if is_busy(&e) => Ok(()),
            e => Err(e),
        },
    };
    conn.busy_timeout(Duration::from_millis(usual.max(0) as u64))?;
    done
}

fn is_busy(e: &AxelError) -> bool {
    matches!(e, AxelError::Db(db)
        if matches!(db.sqlite_error_code(), Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked)))
}

// ── Small things ────────────────────────────────────────────────────────────

/// One write: `begin_write`, the work, `COMMIT`. After any error nothing is committed.
fn in_write<T>(conn: &Connection, work: impl FnOnce() -> Result<T>) -> Result<T> {
    brain_v2::begin_write(conn)?;
    let out = work().and_then(|value| {
        conn.execute_batch("COMMIT")?;
        Ok(value)
    });
    if out.is_err() && !conn.is_autocommit() {
        let _ = conn.execute_batch("ROLLBACK");
    }
    out
}

/// [`in_write`] for work that purges rows: the work pushes the `vid` of every row it
/// purged, and after the `COMMIT` they go to the front thread, whose vector array would
/// otherwise keep them until the next resync.
// RULING: wave4 R-13 item 4. `VecDelta::Remove` after a purge commits.
fn in_write_gone<T>(ctx: &IndexerCtx, conn: &Connection, work: impl FnOnce(&mut Vec<i64>) -> Result<T>) -> Result<T> {
    let mut gone: Vec<i64> = Vec::new();
    let out = in_write(conn, || work(&mut gone))?;
    if !gone.is_empty() {
        let _ = ctx.vec_tx.send(VecDelta::Remove(gone.into_iter().map(VecKey::Mem).collect()));
    }
    Ok(out)
}

/// `store::purge_many`, and the `vid`s of the rows that went are pushed to `gone`.
fn purge(conn: &Connection, ids: &[String], tombstone: bool, forget_log: bool, gone: &mut Vec<i64>) -> Result<usize> {
    if ids.is_empty() {
        return Ok(0);
    }
    let mut vids: Vec<i64> = Vec::new();
    {
        let mut stmt = conn.prepare("SELECT vid FROM memory_ext WHERE mem_id = ?1")?;
        for id in ids {
            vids.extend(stmt.query_row([id], |r| r.get::<_, i64>(0)).optional()?);
        }
    }
    let purged = store::purge_many(conn, ids, tombstone, forget_log)?;
    gone.extend(vids);
    Ok(purged)
}

fn is_ready(ctx: &IndexerCtx) -> bool {
    Mode::from_u8(ctx.mode.load(Ordering::SeqCst)) == Some(Mode::Ready)
}

/// An I/O failure, which says nothing about the file: it is tried again.
fn is_io(reason: &SkipReason) -> bool {
    reason.0.starts_with("cannot ")
}

/// Nothing was read from the path: an I/O failure, or a path that is not a regular file
/// (a directory, a link, a FIFO put where the archive was).
// SPEC: 2242-2246 and test 34 (b) (2946-2949) call a directory of the archive's name an I/O
// failure, `cannot …`, after which the rows stay. As built `read_archive` refuses it at the
// `lstat` with another text, before anything is opened. It is not a file that "can be read
// and is refused", so an indexed archive keeps its rows after it too; `archive.rs` says
// which reasons these are (`SkipReason::nothing_read`).
fn not_read(reason: &SkipReason) -> bool {
    reason.nothing_read()
}

fn archives_root(ctx: &IndexerCtx) -> PathBuf {
    ctx.cfg.base.join("context-archives")
}

fn policy(ctx: &IndexerCtx, keep_forget_turns: bool) -> ArchivePolicy {
    ArchivePolicy {
        event_sources: ctx.cfg.archive_events.clone(),
        forget_tools: vec!["axel_forget".to_string(), "memory_forget".to_string()],
        keep_forget_turns,
    }
}

fn mtime_ms(path: &Path) -> Option<i64> {
    let modified = std::fs::symlink_metadata(path).ok()?.modified().ok()?;
    let since = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
    i64::try_from(since.as_millis()).ok()
}

/// Where a file is: its project directory, that directory's name, its name in it, its
/// `lstat` size.
struct FileAt<'a> {
    dir: &'a Path,
    scope_dir: &'a str,
    rel_name: &'a str,
    size: i64,
    /// The id in a shelf file's name.
    name_id: Option<&'a str>,
}

impl<'a> FileAt<'a> {
    fn of(dir: &'a Path, scope_dir: &'a str, entry: &'a ArchiveEntry) -> FileAt<'a> {
        FileAt {
            dir,
            scope_dir,
            rel_name: &entry.rel_name,
            size: i64::try_from(entry.size).unwrap_or(i64::MAX),
            name_id: entry.id_from_name.as_deref(),
        }
    }

    fn path(&self) -> PathBuf {
        self.dir.join(self.rel_name)
    }

    fn read(&self, expect_id: Option<&str>) -> std::result::Result<ArchiveFile, SkipReason> {
        archive::read_archive(&self.path(), self.scope_dir, expect_id.or(self.name_id))
    }
}

/// What an `on_context_archived` event said about a file. The scan has none, but it
/// knows the new files it will index after this one.
#[derive(Clone, Copy, Default)]
struct Seen<'a> {
    session_id: Option<&'a str>,
    received_ms: Option<i64>,
    window: Option<i64>,
    part: Option<i64>,
    /// The new files of this scan that come after this one in seal order.
    later: &'a [Later],
}

/// A new file of the scan that is not indexed yet, with the archive's own time.
struct Later {
    scope_dir: String,
    logical: String,
    rel_name: String,
    archive_id: String,
    cover_ms: i64,
}

/// Where a file stands in its chat's order when it is not indexed: what its row would say
/// had it been.
// RULING: v2 archives round 4. A file that fails (a panic, a refusal, an error that is not
// a panic) keeps its place in its chat's order: the window of the file after it ends at
// its seal, and a mark of the file before it stops at it, as at any neighbour that is not
// indexed. Nothing of it is taken or shown: `edges_of` gives nothing for it.
struct Held {
    archive_id: String,
    logical: String,
    session_id: Option<String>,
    created_ms: i64,
    time_source: &'static str,
    cover_ms: i64,
    win_no: Option<i64>,
    part_no: Option<i64>,
}

impl Held {
    /// The place the scan knows from the record's own time.
    fn of_later(l: &Later) -> Held {
        Held {
            archive_id: l.archive_id.clone(),
            logical: l.logical.clone(),
            session_id: None,
            created_ms: l.cover_ms,
            time_source: "archive",
            cover_ms: l.cover_ms,
            win_no: None,
            part_no: None,
        }
    }

    /// The place `index_archive` would give a live record read with what an event said;
    /// none for a file whose time is from `mtime` or is no time.
    fn of_live(a: &LiveArchive, seen: &Seen, row: Option<&Known>, at: &FileAt, now: i64) -> Option<Held> {
        let first_seen_ms = row.map_or(now, |r| r.first_seen_ms);
        let (created_ms, time_source, cover_ms) = times(a, seen, row, &at.path(), first_seen_ms);
        if time_source == "mtime" || chrono::DateTime::<chrono::Utc>::from_timestamp_millis(created_ms).is_none() {
            return None;
        }
        Some(Held {
            archive_id: a.id.clone(),
            logical: a.logical.clone(),
            session_id: seen.session_id.or_else(|| row.and_then(|r| r.session_id.as_deref())).map(str::to_string),
            created_ms,
            time_source,
            cover_ms,
            win_no: seen.window,
            part_no: seen.part,
        })
    }
}

impl Held {
    /// The place an event gives a file whose head cannot be read: its chat, from the
    /// session id (`logical` is derived from it, as capture derives it), and its arrival
    /// time, `observed`. None with no event, or an event with no session id.
    fn of_event(seen: &Seen, at: &FileAt) -> Option<Held> {
        let session_id = seen.session_id?;
        let ms = seen.received_ms.filter(|ms| chrono::DateTime::<chrono::Utc>::from_timestamp_millis(*ms).is_some())?;
        Some(Held {
            archive_id: at.name_id.unwrap_or("").to_string(),
            logical: project::logical(session_id),
            session_id: Some(session_id.to_string()),
            created_ms: ms,
            time_source: "observed",
            cover_ms: ms,
            win_no: seen.window,
            part_no: seen.part,
        })
    }
}

/// The `skip_reason`s the indexer writes itself. Any other is the reader's refusal of a
/// file (`write_refused`).
// RULING: v2 archives round 4. `error` is new: a file whose indexing failed with an error
// that is not a panic.
const OWN_REASONS: [&str; 5] = ["paused", "forgotten_elsewhere", "panic", "error", "created_ms is not a time"];

/// The `archive_index` row of one file, as far as this file reads it.
struct Known {
    archive_id: String,
    state: String,
    skip_reason: Option<String>,
    file_size: Option<i64>,
    created_ms: Option<i64>,
    time_source: Option<String>,
    cover_ms: Option<i64>,
    session_id: Option<String>,
    win_no: Option<i64>,
    part_no: Option<i64>,
    forget_units: i64,
    keep_forget_turns: i64,
    indexer_version: i64,
    first_seen_ms: i64,
}

fn known(conn: &Connection, scope_dir: &str, rel_name: &str) -> Result<Option<Known>> {
    Ok(conn
        .query_row(
            "SELECT archive_id, state, skip_reason, file_size, created_ms, time_source, cover_ms,
                    session_id, win_no, part_no, forget_units, keep_forget_turns, indexer_version,
                    first_seen_ms
               FROM archive_index WHERE scope_dir = ?1 AND rel_name = ?2",
            params![scope_dir, rel_name],
            |r| {
                Ok(Known {
                    archive_id: r.get(0)?,
                    state: r.get(1)?,
                    skip_reason: r.get(2)?,
                    file_size: r.get(3)?,
                    created_ms: r.get(4)?,
                    time_source: r.get(5)?,
                    cover_ms: r.get(6)?,
                    session_id: r.get(7)?,
                    win_no: r.get(8)?,
                    part_no: r.get(9)?,
                    forget_units: r.get(10)?,
                    keep_forget_turns: r.get(11)?,
                    indexer_version: r.get(12)?,
                    first_seen_ms: r.get(13)?,
                })
            },
        )
        .optional()?)
}

/// What is written into the `archive_index` row of one file.
#[derive(Clone, Copy)]
struct RowOut<'a> {
    archive_id: &'a str,
    logical: Option<&'a str>,
    digest: Option<&'a str>,
    state: &'a str,
    skip_reason: Option<&'a str>,
    created_ms: Option<i64>,
    time_source: Option<&'a str>,
    cover_ms: Option<i64>,
    session_id: Option<&'a str>,
    win_no: Option<i64>,
    part_no: Option<i64>,
    source_message_count: Option<i64>,
    message_count: Option<i64>,
    rows_indexed: i64,
    forget_units: i64,
    cli_units: i64,
    cli_unproven: i64,
    unmatched_hidden: i64,
    first_seen_ms: i64,
    indexed_ms: Option<i64>,
}

impl<'a> RowOut<'a> {
    fn new(archive_id: &'a str, state: &'a str, first_seen_ms: i64) -> RowOut<'a> {
        RowOut {
            archive_id,
            logical: None,
            digest: None,
            state,
            skip_reason: None,
            created_ms: None,
            time_source: None,
            cover_ms: None,
            session_id: None,
            win_no: None,
            part_no: None,
            source_message_count: None,
            message_count: None,
            rows_indexed: 0,
            forget_units: 0,
            cli_units: 0,
            cli_unproven: 0,
            unmatched_hidden: 0,
            first_seen_ms,
            indexed_ms: None,
        }
    }
}

/// The row of `(scope_dir, rel_name)`. An existing row keeps its `first_seen_ms` and its
/// `keep_forget_turns`, its `unmatched_hidden` unless it is indexed now, and what an event
/// once said (the session
/// id, `win_no`, `part_no`) when this write has none. `indexed_ms`, once set, stays: it is
/// how `forget_archive` knows that the archive was indexed here at some time.
fn upsert_row(conn: &Connection, at: &FileAt, row: &RowOut) -> Result<()> {
    conn.execute(
        "INSERT INTO archive_index(scope_dir, rel_name, archive_id, logical, digest, state, skip_reason,
                                   file_size, created_ms, time_source, cover_ms, session_id, win_no, part_no,
                                   source_message_count, message_count, rows_indexed, forget_units,
                                   indexer_version, first_seen_ms, indexed_ms, unmatched_hidden, cli_units,
                                   cli_unproven)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21,
                 ?22, ?23, ?24)
         ON CONFLICT(scope_dir, rel_name) DO UPDATE SET
            archive_id = excluded.archive_id,
            logical = COALESCE(excluded.logical, archive_index.logical),
            digest = COALESCE(excluded.digest, archive_index.digest),
            state = excluded.state,
            skip_reason = excluded.skip_reason,
            file_size = excluded.file_size,
            created_ms = COALESCE(excluded.created_ms, archive_index.created_ms),
            time_source = COALESCE(excluded.time_source, archive_index.time_source),
            cover_ms = COALESCE(excluded.cover_ms, archive_index.cover_ms),
            session_id = COALESCE(excluded.session_id, archive_index.session_id),
            win_no = COALESCE(excluded.win_no, archive_index.win_no),
            part_no = COALESCE(excluded.part_no, archive_index.part_no),
            source_message_count = excluded.source_message_count,
            message_count = excluded.message_count,
            rows_indexed = excluded.rows_indexed,
            forget_units = excluded.forget_units,
            cli_units = excluded.cli_units,
            cli_unproven = excluded.cli_unproven,
            unmatched_hidden = CASE WHEN excluded.state = 'indexed' THEN excluded.unmatched_hidden
                                    ELSE archive_index.unmatched_hidden END,
            indexer_version = excluded.indexer_version,
            indexed_ms = COALESCE(excluded.indexed_ms, archive_index.indexed_ms)",
        params![
            at.scope_dir,
            at.rel_name,
            row.archive_id,
            row.logical,
            row.digest,
            row.state,
            row.skip_reason,
            at.size,
            row.created_ms,
            row.time_source,
            row.cover_ms,
            row.session_id,
            row.win_no,
            row.part_no,
            row.source_message_count,
            row.message_count,
            row.rows_indexed,
            row.forget_units,
            INDEXER_VERSION,
            row.first_seen_ms,
            row.indexed_ms,
            row.unmatched_hidden,
            row.cli_units,
            row.cli_unproven,
        ],
    )?;
    Ok(())
}

fn ids_of(conn: &Connection, sql: &str, params: impl rusqlite::Params) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map(params, |r| r.get::<_, String>(0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// The rows whose text came from one archive of one directory. They are selected by the
/// directory and the archive id, never by a project key from an event (B2).
fn archive_rows(conn: &Connection, scope_dir: &str, archive_id: &str) -> Result<Vec<String>> {
    ids_of(
        conn,
        "SELECT e.mem_id FROM memory_ext e JOIN memories m ON m.id = e.mem_id
          WHERE e.origin_kind = 'archive' AND e.origin_ref = ?1 AND m.project_key = ?2",
        params![archive_id, scope_dir],
    )
}

/// The captured turns of one chat in one directory whose `created_ms` lies in
/// `(after_ms, up_to_ms]`: `(id, hidden)`.
fn captures(
    conn: &Connection,
    scope_dir: &str,
    logical: &str,
    session_id: Option<&str>,
    after_ms: i64,
    up_to_ms: i64,
) -> Result<Vec<(String, bool)>> {
    let mut stmt = conn.prepare(
        "SELECT e.mem_id, e.hidden_ms IS NOT NULL
           FROM memory_ext e JOIN memories m ON m.id = e.mem_id
          WHERE e.origin_kind = 'capture' AND m.project_key = ?1
            AND ((?2 <> '' AND e.logical = ?2) OR (?3 IS NOT NULL AND e.session_id = ?3))
            AND e.created_ms > ?4 AND e.created_ms <= ?5",
    )?;
    let rows = stmt.query_map(params![scope_dir, logical, session_id, after_ms, up_to_ms], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, bool>(1)?))
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

fn window_start(conn: &Connection, scope_dir: &str, logical: &str, session_id: Option<&str>, t: i64) -> Result<i64> {
    Ok(conn.query_row(
        "SELECT COALESCE(MAX(cover_ms), 0) FROM archive_index
          WHERE scope_dir = ?1
            AND ((?2 <> '' AND logical = ?2) OR (?3 IS NOT NULL AND session_id = ?3))
            AND COALESCE(time_source, '') <> 'mtime'
            AND cover_ms < ?4 - ?5",
        params![scope_dir, logical, session_id, t, PARTS_SPAN_MS],
        |r| r.get(0),
    )?)
}

/// The largest `cover_ms` of an `archive_index` row of the same directory and chat (same
/// `logical`, or same non-null `session_id`), in any state, that is more than
/// `PARTS_SPAN_MS` below `t`; 0 when there is none. A row whose time is from `mtime`
/// bounds no window. The window of archive A is `(window_start_ms(A.cover_ms), A.cover_ms]`.
// SPEC: 2142-2146 give this function no way to fail. When the query fails it answers `t`,
// which makes the window empty: nothing is purged or left out on the strength of a bound
// that could not be read.
pub fn window_start_ms(conn: &Connection, scope_dir: &str, logical: &str, session_id: Option<&str>, t: i64) -> i64 {
    window_start(conn, scope_dir, logical, session_id, t).unwrap_or(t)
}

// ── Writing units: one archive ──────────────────────────────────────────────

fn kind_of(unit: &Unit) -> (char, &'static str) {
    match unit.kind {
        UnitKind::Exchange => ('x', "episodic_turn"),
        UnitKind::Continued => ('c', "episodic_turn"),
        UnitKind::Report => ('r', "tool_outcome"),
        UnitKind::Note => ('n', "conversation_summary"),
    }
}

/// `created_ms`, `time_source`, `cover_ms`: the record's `created_ms` (`archive`); else the
/// arrival time of the event that announced the file (`observed`); else the file's `lstat`
/// modification time (`mtime`), and then `cover_ms` is `first_seen_ms`.
// SPEC: 2134-2140 do not say what a file's times are when it is read a second time and
// its record has none. A row that has times keeps them: the rows written at the first
// reading carry them, and one archive has one time.
fn times(a: &LiveArchive, seen: &Seen, row: Option<&Known>, path: &Path, first_seen_ms: i64) -> (i64, &'static str, i64) {
    if let Some(ms) = a.created_ms {
        return (ms, "archive", ms);
    }
    if let Some(r) = row {
        match (r.created_ms, r.time_source.as_deref(), r.cover_ms) {
            // RULING: v2 archives round 4. The cover time, which an `observed` row writes
            // as its `created_ms` too: a refused file's row holds the file's time there.
            (Some(_), Some("observed"), Some(cover)) => return (cover, "observed", cover),
            (Some(ms), Some("mtime"), Some(cover)) => return (ms, "mtime", cover),
            _ => {}
        }
    }
    if let Some(ms) = seen.received_ms {
        return (ms, "observed", ms);
    }
    (mtime_ms(path).unwrap_or(first_seen_ms), "mtime", first_seen_ms)
}

/// One live archive, in one transaction: indexed whole or not at all. `replace`: the file
/// was indexed before and is another file now; its old rows go first, in the same
/// transaction.
///
/// Steps 1 to 5 of section 2.3: forgotten already, paused, the forget marks and what they
/// leave out (3, 3b), the twins of hidden captures (3a), the covered captures, the row.
fn index_archive(
    ctx: &IndexerCtx,
    at: &FileAt,
    a: &LiveArchive,
    seen: &Seen,
    row: Option<&Known>,
    replace: bool,
) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    // A file this directory knows as forgotten is never indexed, whatever it holds now.
    if row.is_some_and(|r| r.state == "tombstone") {
        return Ok(());
    }
    if ctx.test.panic_archive_id.as_deref() == Some(a.id.as_str()) {
        panic!("test switch: panic_archive_id");
    }
    // RULING: wave4 R-13 round 2, item 6. `keep_forget_turns` was asked for the file that
    // was under this name; a file that replaced it is read with the forget rules on.
    let policy = policy(ctx, !replace && row.is_some_and(|r| r.keep_forget_turns == 1));
    let units = archive::extract_units(a, &policy);
    // One mark for each forget span of the file; none when the row has `keep_forget_turns`.
    let spans = archive::forgotten_keys(a, &policy);
    let forget_spans = spans.len() as i64;

    let now = (ctx.clock)();
    let first_seen_ms = row.map_or(now, |r| r.first_seen_ms);
    let (created_ms, time_source, cover_ms) = times(a, seen, row, &at.path(), first_seen_ms);
    let session_id = seen.session_id.or_else(|| row.and_then(|r| r.session_id.as_deref()));

    let base = RowOut {
        logical: Some(&a.logical),
        session_id,
        win_no: seen.window,
        part_no: seen.part,
        created_ms: Some(created_ms),
        time_source: Some(time_source),
        cover_ms: Some(cover_ms),
        ..RowOut::new(&a.id, "skipped", first_seen_ms)
    };

    // SPEC: 1792 lets any whole number of milliseconds through the reader, and the row
    // writer refuses a `created_ms` that is no time. Such a file is skipped with a reason,
    // not tried and failed at every scan.
    if chrono::DateTime::<chrono::Utc>::from_timestamp_millis(created_ms).is_none() {
        return in_write_gone(ctx, conn, |gone| {
            if replace {
                forget_rows(conn, at.scope_dir, row, gone)?;
            }
            let skipped = RowOut {
                digest: Some(&a.digest),
                skip_reason: Some("created_ms is not a time"),
                created_ms: None,
                time_source: None,
                cover_ms: None,
                ..base
            };
            upsert_row(conn, at, &skipped)
        });
    }

    // A file whose time is from `mtime` takes no part: it has no place in the chat's order,
    // gives and takes no forget mark, and has no twin. A file whose row has
    // `keep_forget_turns` gives no mark and takes only `cli` marks.
    // RULING: v2 c18b review, item 4. A kept archive ignores `call` and `archive` marks and
    // obeys `cli` marks: a turn deleted with `forget --now` does not come back with
    // `restore --forget-turns`.
    let timed = time_source != "mtime";
    let marks_on = timed && !policy.keep_forget_turns;
    let takes_marks = timed;
    let own: &[archive::ForgetMark] = if marks_on { &spans } else { &[] };
    let files: Vec<ChatFile> = if timed {
        // The chat's files with this one at its place, as it will be written: the windows
        // of its neighbours are bounded by it too.
        let mut files = chat_files(conn, at.scope_dir, &a.logical, session_id)?;
        files.retain(|f| f.rel_name != at.rel_name);
        // RULING: wave4 R-13 round 2, item 1a. The new files of this scan that are sealed
        // after this one take their place too, not indexed yet: a far file indexed by an
        // earlier scan is then not "the window after" this one when a file between is here.
        let later = seen.later.iter().filter(|l| {
            l.scope_dir == at.scope_dir
                && !a.logical.is_empty()
                && l.logical == a.logical
                && l.rel_name != at.rel_name
                && !files.iter().any(|f| f.rel_name == l.rel_name)
        });
        let later: Vec<ChatFile> = later
            .map(|l| ChatFile {
                rel_name: l.rel_name.clone(),
                archive_id: l.archive_id.clone(),
                cover_ms: l.cover_ms,
                first_seen_ms: i64::MAX,
                keep: false,
                indexed: false,
                win_no: None,
                part_no: None,
            })
            .collect();
        files.extend(later);
        files.push(ChatFile {
            rel_name: at.rel_name.to_string(),
            archive_id: a.id.clone(),
            cover_ms,
            first_seen_ms,
            keep: policy.keep_forget_turns,
            indexed: false,
            win_no: None,
            part_no: None,
        });
        files.sort_by(|x, y| x.order().cmp(&y.order()));
        files
    } else {
        Vec::new()
    };
    let me = files.iter().position(|f| f.rel_name == at.rel_name);
    let prev = me.and_then(|at_me| at_me.checked_sub(1)).map(|before| &files[before]);
    let next = me.and_then(|at_me| files.get(at_me + 1));
    let window_from = start_of(&files, cover_ms);
    let stretch = stretch_of(&files, cover_ms);
    // RULING: wave4 R-13 round 2, item 1b. A file that is the newest known of its chat has
    // no window after it yet, and files between it and a later mark may simply not be
    // known: it takes marks (and gives up log rows) only up to its cover time plus
    // `PARTS_SPAN_MS`. A later mark reaches it through step 3b of the mark's own file and
    // through the live call, which then know the files between. (A `call` or `cli` mark
    // made before this file was indexed, for a turn after its seal, is the one case that
    // nothing visits again: a same-key turn of this file stays.)
    let mark_stretch =
        (stretch.0, if stretch.1 == i64::MAX { cover_ms.saturating_add(PARTS_SPAN_MS) } else { stretch.1 });
    // Step 3b by position: which unit of the file just before, and of the file just after,
    // this file's own marks leave out. Read from those files, not from their rows: a row
    // another rule already took must not make the one before it "the last".
    let before = if own.iter().any(|m| m.head) { prev.and_then(|f| edges_of(ctx, at, f)) } else { None };
    let after = if own.iter().any(|m| m.tail) { next.and_then(|f| edges_of(ctx, at, f)) } else { None };

    in_write_gone(ctx, conn, |gone| {
        // RULING: wave4 R-13 item 4. A row of the old file that was hidden (by the model,
        // by the command line, by a pause) hands its hide to the new file's row of the same
        // position: replacing a file does not bring back what a person hid. A `twin:` hide
        // is not carried, it is judged again from the hidden captures (step 3a).
        let mut carried: HashMap<String, (i64, String)> = HashMap::new();
        if replace {
            if let Some(old) = row {
                carried = hides_of(conn, at.scope_dir, &old.archive_id)?;
            }
            forget_rows(conn, at.scope_dir, row, gone)?;
            // The marks the old file gave are not the new file's.
            // (What those marks took out of its neighbours does not come back.)
            conn.execute(
                "DELETE FROM forgotten_turns WHERE source = 'archive' AND scope_dir = ?1 AND archive_id = ?2",
                params![at.scope_dir, row.map_or(a.id.as_str(), |r| r.archive_id.as_str())],
            )?;
            conn.execute(
                "UPDATE archive_index SET keep_forget_turns = 0 WHERE scope_dir = ?1 AND rel_name = ?2",
                params![at.scope_dir, at.rel_name],
            )?;
        }

        // 1. Forgotten already? A forget reaches every copy of a digest, and a second file
        // with one id can exist.
        let forgotten: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM archive_index
                            WHERE scope_dir = ?1 AND state = 'tombstone'
                              AND (archive_id = ?2 OR (?3 <> '' AND digest = ?3)))",
            params![at.scope_dir, a.id, a.digest],
            |r| r.get(0),
        )?;
        if forgotten {
            let skipped = RowOut { digest: Some(&a.digest), skip_reason: Some("forgotten_elsewhere"), ..base };
            return upsert_row(conn, at, &skipped);
        }

        // 2. Paused? The chat is the archive's `logical`, or the session the event named.
        let paused: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM chat_pauses
                            WHERE what = 'learning'
                              AND (logical = ?1 OR (?2 IS NOT NULL AND session_id = ?2)))",
            params![a.logical, session_id],
            |r| r.get(0),
        )?;
        if paused {
            // The chat, the times, and nothing else (R5). It purges no capture.
            let skipped = RowOut { skip_reason: Some("paused"), ..base };
            return upsert_row(conn, at, &skipped);
        }

        // 3. The forget marks. This file's own marks are rows of `forgotten_turns` first;
        // then every row of the chat, of any source, says which units of this file are
        // left out.
        let mut out = vec![false; units.len()];
        let mut by_cli = vec![false; units.len()];
        let mut unproven = vec![false; units.len()];
        if takes_marks {
            for mark in own {
                conn.execute(
                    "INSERT INTO forgotten_turns(logical, session_id, source, turn_key, from_ms, to_ms, scope_dir,
                                                 archive_id, head, tail, added_ms)
                     VALUES (?1, ?2, 'archive', ?3, ?4, ?4, ?5, ?6, ?7, ?8, ?9)
                     ON CONFLICT(logical, source, from_ms, turn_key, archive_id, text_hash) DO UPDATE
                        SET head = max(head, excluded.head), tail = max(tail, excluded.tail)",
                    params![
                        a.logical,
                        session_id,
                        mark.turn_key.as_deref().unwrap_or(""),
                        cover_ms,
                        at.scope_dir,
                        a.id,
                        mark.head,
                        mark.tail,
                        now,
                    ],
                )?;
            }
            let mut marks = forgotten_of(conn, &a.logical, session_id)?;
            if !marks_on {
                marks.retain(|m| m.source == "cli");
            }
            let place = Place { scope_dir: at.scope_dir, cover_ms, window_from, stretch: mark_stretch, prev, next };
            // RULING: v2 c18b review round 2, item 2. What `cli` marks leave out is counted
            // apart (`cli_units`): no `restore --forget-turns` brings it back, so it is not
            // in `forget_units`, and a kept archive is not read again for it at every scan.
            let (cli, rest): (Vec<Forgot>, Vec<Forgot>) = marks.into_iter().partition(|m| m.source == "cli");
            by_cli = left_out(&units, &cli, &place);
            unproven = not_proven(&units, &cli, &place, archive::carried_prompt_key(a).as_deref());
            out = left_out(&units, &rest, &place);
            for (left, by_cli) in out.iter_mut().zip(&by_cli) {
                *left |= *by_cli;
            }

            // RULING: wave4 R-13 round 2, item 2. A unit that is left out was never a row,
            // so no purge takes the recall-log row of its turn: the log rows of this chat
            // with the key of one of this file's marks, inside its stretch, go here.
            let mut log_rows = 0;
            for key in own.iter().filter_map(|m| m.turn_key.as_deref()).filter(|k| !k.is_empty()) {
                log_rows += conn.execute(
                    "DELETE FROM recall_log
                      WHERE ?1 <> '' AND logical = ?1 AND query_hash = ?2 AND ts_ms > ?3 AND ts_ms <= ?4",
                    params![a.logical, key, mark_stretch.0, mark_stretch.1],
                )?;
            }
            if log_rows > 0 {
                mark_purged(conn)?;
            }
        }

        // 3a. Twins of hidden captures: a unit whose key is the key of a hidden capture of
        // this chat, made in this file's window, the one before or the one after, is born
        // hidden as that capture's twin.
        let hidden: Vec<HiddenCapture> = if timed {
            hidden_captures(conn, at.scope_dir, &a.logical, session_id, stretch.0, stretch.1)?
        } else {
            Vec::new()
        };
        let twin_of = |unit: &Unit| {
            let key = unit.turn_key.as_deref().filter(|k| !k.is_empty())?;
            let h = hidden.iter().find(|h| h.turn_key.as_deref() == Some(key))?;
            Some((h.hidden_ms, format!("twin:{}", h.id)))
        };

        // Every other unit is a row. `Exists` and `Tombstoned` are "nothing to write".
        let mut tags = vec![format!("ctx:{}", a.id)];
        if let Some(session) = session_id {
            tags.push(format!("session:{session}"));
        }
        let mut rows_indexed = 0i64;
        for (at_unit, (unit, left)) in units.iter().zip(&out).enumerate() {
            if *left {
                continue;
            }
            let (letter, class) = kind_of(unit);
            let pos = format!("{letter}{}.{}", unit.source_index, unit.block_index);
            let content = stored_content(unit);
            let new = NewRow {
                id: archive::unit_id(at.scope_dir, &a.id, unit),
                class,
                project_key: at.scope_dir,
                title: unit.title.clone(),
                abstract_text: String::new(),
                content,
                tags: tags.clone(),
                created_ms,
                time_source,
                sensitivity: Sensitivity::Normal,
                provenance: format!("synaps:archive:{}", a.id),
                expires_ms: None,
                importance: 0.5,
                origin_kind: "archive",
                origin_ref: &a.id,
                unit_pos: Some(pos.clone()),
                session_id,
                logical: Some(&a.logical),
                links: RecordLinks::default(),
                turn_key: unit.turn_key.clone(),
                // RULING: wave4 R-13 round 3, item 2. A carried hide wins over a twin mark:
                // what a person or a model hid stays theirs, and no re-judging of twins and
                // no restore of a capture can clear it.
                // RULING: v2 c18b review round 3, item 1. A unit with the key of a deleted
                // turn that its text does not prove that turn is born hidden: a miss of the
                // hash never brings deleted text back into recall.
                // RULING: v2 c18b review round 4, item 4: a carried hide by a model or a pause
                // gives way to `cli:unproven`.
                hidden: match carried.get(&pos) {
                    Some((_, by)) if unproven[at_unit] && yields_to_unproven(by) => Some((now, UNPROVEN.to_string())),
                    Some(hide) => Some(hide.clone()),
                    None => twin_of(unit).or_else(|| unproven[at_unit].then(|| (now, UNPROVEN.to_string()))),
                },
            };
            match store::insert_row(conn, ctx.signer.as_ref(), &new)? {
                Inserted::New { .. } | Inserted::Exists => rows_indexed += 1,
                Inserted::Tombstoned => {}
            }
        }
        // A unit that is left out now and was a row from an earlier reading goes.
        let left: Vec<String> = units
            .iter()
            .zip(&out)
            .filter(|(_, left)| **left)
            .map(|(unit, _)| archive::unit_id(at.scope_dir, &a.id, unit))
            .collect();
        purge_left_out(conn, at.scope_dir, &left, gone)?;

        // RULING: wave4 R-13 round 2, item 1c. Twins are judged again: this file changed
        // the stretch of the chat's files around it.
        if timed {
            rejudge_twins(conn, at, &a.logical, session_id, &files)?;
        }

        // A hidden capture of this file's own window that no unit of the file matches and
        // that has no twin anywhere is counted, and nothing else is hidden for it.
        let mut unmatched_hidden = 0i64;
        for h in hidden.iter().filter(|h| h.created_ms > window_from && h.created_ms <= cover_ms) {
            let matched = h.turn_key.as_deref().filter(|k| !k.is_empty()).is_some_and(|key| {
                units.iter().any(|unit| unit.turn_key.as_deref() == Some(key))
            });
            let has_twin: bool = conn.query_row(
                // `hidden_ms IS NOT NULL`: the partial index serves it (no walk of the table).
                "SELECT EXISTS(SELECT 1 FROM memory_ext WHERE hidden_ms IS NOT NULL AND hidden_by = ?1)",
                [format!("twin:{}", h.id)],
                |r| r.get(0),
            )?;
            if !matched && !has_twin {
                unmatched_hidden += 1;
            }
        }

        // 3b. What this file's own marks leave out elsewhere, among the rows indexed so
        // far: rows with one of its keys in an archive near this one; for a mark with
        // `head`, the last turn of the file just before; for a mark with `tail`, the
        // leading `Continued` unit and the note of the file just after.
        if !own.is_empty() {
            let mut elsewhere: Vec<String> = Vec::new();
            let near = near_archives(&files, cover_ms, cover_ms);
            for key in own.iter().filter_map(|m| m.turn_key.as_deref()).filter(|k| !k.is_empty()) {
                for (id, archive_id) in key_rows(conn, &a.logical, session_id, key)? {
                    if archive_id != a.id && near.contains(archive_id.as_str()) {
                        elsewhere.push(id);
                    }
                }
            }
            elsewhere.extend(before.as_ref().and_then(|e| e.last.clone()));
            elsewhere.extend(after.as_ref().and_then(|e| e.lead.clone()));
            elsewhere.extend(after.as_ref().and_then(|e| e.note.clone()));
            purge_left_out(conn, at.scope_dir, &elsewhere, gone)?;
        }

        // 4. Covered live captures go: they were stand-ins for turns that are rows of this
        // archive now. Replaced, not forgotten: no tombstone, and their log rows stay. A
        // hidden capture is left alone, and so is one older than the window. A file whose
        // time is from `mtime` purges no capture.
        if time_source != "mtime" {
            let start = window_start(conn, at.scope_dir, &a.logical, session_id, cover_ms)?;
            let covered: Vec<String> = captures(conn, at.scope_dir, &a.logical, session_id, start, cover_ms)?
                .into_iter()
                .filter(|(_, hidden)| !hidden)
                .map(|(id, _)| id)
                .collect();
            purge(conn, &covered, false, false, gone)?;
        }

        // 5. The row.
        let indexed = RowOut {
            digest: Some(&a.digest),
            state: "indexed",
            source_message_count: Some(i64::from(a.source_message_count)),
            message_count: Some(a.messages.len() as i64),
            rows_indexed,
            forget_units: forget_spans + out.iter().zip(&by_cli).filter(|(left, cli)| **left && !**cli).count() as i64,
            cli_units: by_cli.iter().filter(|cli| **cli).count() as i64,
            cli_unproven: unproven.iter().zip(&out).filter(|(u, left)| **u && !**left).count() as i64,
            unmatched_hidden,
            indexed_ms: Some(now),
            ..base
        };
        upsert_row(conn, at, &indexed)
    })
}

/// The rows of a file that is another file now (a person replaced it): the mirror follows
/// the file. Inside the caller's transaction.
// SPEC: 2240-2241 say "purge the archive's rows without tombstones" and 2413-2416 do not
// list this purge under either value of `forget_log`. The text these rows held is in no
// file any more, so it is a forget: `forget_log = true`, and the log is truncated.
fn forget_rows(conn: &Connection, scope_dir: &str, row: Option<&Known>, gone: &mut Vec<i64>) -> Result<()> {
    let Some(row) = row else { return Ok(()) };
    let ids = archive_rows(conn, scope_dir, &row.archive_id)?;
    if purge(conn, &ids, false, true, gone)? > 0 {
        mark_purged(conn)?;
    }
    Ok(())
}

/// The hides of an archive's rows that are not twin marks, by `unit_pos`.
fn hides_of(conn: &Connection, scope_dir: &str, archive_id: &str) -> Result<HashMap<String, (i64, String)>> {
    let mut stmt = conn.prepare(
        "SELECT e.unit_pos, e.hidden_ms, e.hidden_by FROM memory_ext e JOIN memories m ON m.id = e.mem_id
          WHERE e.origin_kind = 'archive' AND e.origin_ref = ?1 AND m.project_key = ?2
            AND e.unit_pos IS NOT NULL AND e.hidden_ms IS NOT NULL AND e.hidden_by IS NOT NULL
            AND substr(e.hidden_by, 1, 5) <> 'twin:'",
    )?;
    let rows = stmt.query_map(params![archive_id, scope_dir], |r| Ok((r.get(0)?, (r.get(1)?, r.get(2)?))))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Twins judged again for the chat's files whose stretch the file at `at` changed, and for
/// that file itself: a row hidden as the twin of a capture that lies outside its archive's
/// stretch as now known becomes the twin of a hidden capture with its key inside the
/// stretch, or, with none, visible. A twin whose capture is gone is left as it is.
// A row made visible here has its vector back in the front's array at the next resync.
fn rejudge_twins(
    conn: &Connection,
    at: &FileAt,
    logical: &str,
    session_id: Option<&str>,
    files: &[ChatFile],
) -> Result<()> {
    let now: Vec<i64> = files.iter().map(|f| f.cover_ms).collect();
    let before: Vec<i64> = files.iter().filter(|f| f.rel_name != at.rel_name).map(|f| f.cover_ms).collect();
    for file in files {
        let stretch = stretch_in(&now, file.cover_ms);
        if file.rel_name != at.rel_name && stretch_in(&before, file.cover_ms) == stretch {
            continue;
        }
        let stale: Vec<(String, Option<String>)> = {
            let mut stmt = conn.prepare(
                "SELECT e.mem_id, e.turn_key FROM memory_ext e JOIN memories m ON m.id = e.mem_id
                  WHERE e.origin_kind = 'archive' AND e.origin_ref = ?1 AND m.project_key = ?2
                    AND e.hidden_ms IS NOT NULL AND substr(COALESCE(e.hidden_by, ''), 1, 5) = 'twin:'
                    AND EXISTS (SELECT 1 FROM memory_ext c
                                 WHERE c.mem_id = substr(e.hidden_by, 6)
                                   AND (c.created_ms <= ?3 OR c.created_ms > ?4))",
            )?;
            let rows = stmt
                .query_map(params![file.archive_id, at.scope_dir, stretch.0, stretch.1], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        if stale.is_empty() {
            continue;
        }
        let hidden = hidden_captures(conn, at.scope_dir, logical, session_id, stretch.0, stretch.1)?;
        for (id, key) in stale {
            let inside = key
                .as_deref()
                .filter(|k| !k.is_empty())
                .and_then(|k| hidden.iter().find(|h| h.turn_key.as_deref() == Some(k)));
            conn.execute(
                "UPDATE memory_ext SET hidden_ms = ?1, hidden_by = ?2 WHERE mem_id = ?3",
                params![inside.map(|h| h.hidden_ms), inside.map(|h| format!("twin:{}", h.id)), id],
            )?;
        }
    }
    Ok(())
}

/// What becomes of a file that is not known, or that is known as `skipped` and is read
/// again: a live record is indexed, a tombstone is a `tombstone` row and a forget, a file
/// that can be read and is refused is `skipped` with the reader's reason. An I/O failure
/// writes nothing and the file is tried again.
// SPEC: 2234-2237 say "not known → read it" and name no row for a file that could not be
// opened. None is written: a `skipped` row is read again only when the size changes, and
// a failure to open says nothing about the file.
fn settle(
    ctx: &IndexerCtx,
    at: &FileAt,
    row: Option<&Known>,
    got: std::result::Result<ArchiveFile, SkipReason>,
    seen: &Seen,
) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    let first_seen_ms = row.map_or_else(|| (ctx.clock)(), |r| r.first_seen_ms);
    match got {
        Ok(ArchiveFile::Live(mut a)) => {
            a.start = if seen.window.is_some() || seen.part.is_some() {
                Start::from_event(seen.window, seen.part)
            } else {
                Start::from_event(row.and_then(|r| r.win_no), row.and_then(|r| r.part_no))
            };
            index_archive(ctx, at, &a, seen, row, false)
        }
        Ok(ArchiveFile::Tombstone { id, logical, digest }) => in_write_gone(ctx, conn, |gone| {
            let tombstone = RowOut {
                logical: Some(logical.as_str()).filter(|l| !l.is_empty()),
                digest: Some(digest.as_str()).filter(|d| !d.is_empty()),
                session_id: seen.session_id,
                ..RowOut::new(&id, "tombstone", first_seen_ms)
            };
            upsert_row(conn, at, &tombstone)?;
            forget_archive_in(conn, at.scope_dir, &id, gone)?;
            mark_purged(conn)
        }),
        Err(reason) if is_io(&reason) => Ok(()),
        // RULING: v2 archives round 5, item 2. A file the reader refuses takes its place from
        // the event that named it, when there was one (`Held::of_event`). On the scan path it
        // keeps none: there is no event, and nothing of the file can be read (no chat, no
        // time). Core writes no file over its own limits (16 MiB, 4,096 messages, a note of
        // 8,192 bytes: a window over them is cut into parts), so a refused file is a damaged
        // one, one of a version this reader does not know, or one changed by hand: not
        // expected from core at all.
        Err(reason) => {
            let held = Held::of_event(seen, at);
            in_write(conn, || write_refused(conn, at, row, &reason, first_seen_ms, held.as_ref()))
        }
    }
}

/// The `skipped` row of a file that can be read and is refused, with the reader's reason.
// RULING: wave4 R-13 item 2. Such a file is read again when its size or its time changes.
// `archive_index` has no column for a file's time and the schema is not this commit's, so
// the row holds the `lstat` modification time in `created_ms`; its reason, one that is not
// in `OWN_REASONS`, is what tells it from another `skipped` row (`read_again`).
// RULING: v2 archives round 4. A row that has a place in its chat's order (a file indexed
// before and refused now) keeps it: its `time_source` and `cover_ms` stay. A row with none
// says `mtime`, with a null `cover_ms`, as before.
// RULING: v2 archives round 5, item 2. A row with no place takes `held`, the place an event
// gave, when there is one.
fn write_refused(
    conn: &Connection,
    at: &FileAt,
    row: Option<&Known>,
    reason: &SkipReason,
    first_seen_ms: i64,
    held: Option<&Held>,
) -> Result<()> {
    let archive_id = at.name_id.or(row.map(|r| r.archive_id.as_str())).unwrap_or("");
    let file_ms = mtime_ms(&at.path());
    let placed = row.is_some_and(|r| {
        r.cover_ms.is_some() && matches!(r.time_source.as_deref(), Some("archive" | "observed"))
    });
    let held = held.filter(|_| !placed);
    let skipped = RowOut {
        skip_reason: Some(&reason.0),
        logical: held.map(|h| h.logical.as_str()),
        session_id: held.and_then(|h| h.session_id.as_deref()),
        created_ms: file_ms,
        time_source: match held {
            _ if placed => None,
            Some(h) => Some(h.time_source),
            None => file_ms.map(|_| "mtime"),
        },
        cover_ms: held.map(|h| h.cover_ms),
        win_no: held.and_then(|h| h.win_no),
        part_no: held.and_then(|h| h.part_no),
        ..RowOut::new(archive_id, "skipped", first_seen_ms)
    };
    upsert_row(conn, at, &skipped)?;
    if !placed && held.is_none() {
        conn.execute(
            "UPDATE archive_index SET cover_ms = NULL WHERE scope_dir = ?1 AND rel_name = ?2",
            params![at.scope_dir, at.rel_name],
        )?;
    }
    Ok(())
}

/// An indexed file that can be read and is refused now (not valid JSON, another version,
/// over the size limit): it loses its own rows and its row says `skipped` with the reason.
// RULING: wave4 R-13 item 2 (spec 2.5 changes). It is not a tombstone: nothing is done by
// digest, no capture is purged, no recall-log row is deleted, and the file is read again
// when its size or time changes. A second file of the same id that is still indexed in
// this directory owns the rows, and then they stay.
fn refuse_indexed(ctx: &IndexerCtx, at: &FileAt, row: &Known, reason: &SkipReason) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    in_write_gone(ctx, conn, |gone| {
        let shared: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM archive_index
                            WHERE scope_dir = ?1 AND archive_id = ?2 AND rel_name <> ?3 AND state = 'indexed')",
            params![at.scope_dir, row.archive_id, at.rel_name],
            |r| r.get(0),
        )?;
        if !shared {
            let ids = archive_rows(conn, at.scope_dir, &row.archive_id)?;
            // The text of these rows is in no file any more: the log is truncated.
            if purge(conn, &ids, false, false, gone)? > 0 {
                mark_purged(conn)?;
            }
            conn.execute(
                "DELETE FROM forgotten_turns WHERE source = 'archive' AND scope_dir = ?1 AND archive_id = ?2",
                params![at.scope_dir, row.archive_id],
            )?;
        }
        write_refused(conn, at, Some(row), reason, row.first_seen_ms, None)
    })
}

/// An indexed file that is a tombstone now: its rows go, and the row says `tombstone` and
/// has the file's new size.
fn forget_indexed(ctx: &IndexerCtx, at: &FileAt, archive_id: &str) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    in_write_gone(ctx, conn, |gone| {
        forget_archive_in(conn, at.scope_dir, archive_id, gone)?;
        conn.execute(
            "UPDATE archive_index SET file_size = ?1 WHERE scope_dir = ?2 AND rel_name = ?3",
            params![at.size, at.scope_dir, at.rel_name],
        )?;
        mark_purged(conn)
    })
}

/// Called by `indexer_start::open` before the mode is `Ready`: every `indexed` file that
/// is a tombstone now is forgotten, as `scan_file` does. A forget made while no extension
/// ran (or whose job was dropped) is applied before recall or a tool can read the brain,
/// not at the first scan, which comes after the notes pass and may be long. Only the files
/// `archive_index` holds as `indexed` are looked at: `lstat`, and a read for a size that
/// changed.
// RULING: v2 real-binaries round, item 2.
pub fn forget_tombstoned(ctx: &IndexerCtx) -> Result<usize> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(0) };
    let rows: Vec<(String, String, String, Option<i64>)> = conn
        .prepare("SELECT scope_dir, rel_name, archive_id, file_size FROM archive_index WHERE state = 'indexed'")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let root = archives_root(ctx);
    let mut listed: HashMap<String, Vec<ArchiveEntry>> = HashMap::new();
    let mut forgotten = 0;
    for (scope_dir, rel_name, archive_id, size) in rows {
        let dir = root.join(&scope_dir);
        let entries = listed.entry(scope_dir.clone()).or_insert_with(|| archive::list_project(&dir));
        let Some(entry) = entries.iter().find(|e| e.rel_name == rel_name) else { continue };
        let at = FileAt::of(&dir, &scope_dir, entry);
        if size == Some(at.size) {
            continue;
        }
        if let Ok(Ok(ArchiveFile::Tombstone { .. })) = catch_unwind(AssertUnwindSafe(|| at.read(None))) {
            forget_indexed(ctx, &at, &archive_id)?;
            forgotten += 1;
        }
    }
    Ok(forgotten)
}

// ── The fast path: on_context_archived ──────────────────────────────────────

/// `Job::Archive` (called by `indexer::dispatch`). Any other job is nothing to it. With no
/// connection, a mode that is not `Ready`, or `archives = off`, it does nothing and the
/// scan finds the file later.
pub fn on_job(ctx: &mut IndexerCtx, job: &Job) -> Result<()> {
    let Job::Archive { archive_id, path, project_key, cwd, session_id, received_ms, window, part } = job else {
        return Ok(());
    };
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    if !is_ready(ctx) || !ctx.cfg.archives {
        return Ok(());
    }
    // The event carries the key, so the directory never comes from `cwd` (2.1).
    let scope_dir = project::project_dir(project_key);
    let dir = archives_root(ctx).join(&scope_dir);

    // For display only; never used to decide scope.
    let now = (ctx.clock)();
    in_write(conn, || {
        conn.execute(
            "INSERT INTO projects(scope_dir, project_key, root_hint, first_seen_ms, last_seen_ms)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(scope_dir) DO UPDATE SET
                project_key = excluded.project_key,
                root_hint = COALESCE(excluded.root_hint, projects.root_hint),
                last_seen_ms = excluded.last_seen_ms",
            params![scope_dir, project_key, cwd, now],
        )?;
        Ok(())
    })?;

    // `path` only when it lies in the project's directory and names an archive file of
    // this id; otherwise the id is looked for in the directory's shelves.
    let entries = archive::list_project(&dir);
    let named = path
        .as_deref()
        .and_then(|p| Path::new(p).strip_prefix(&dir).ok())
        .and_then(Path::to_str)
        .and_then(|rel| entries.iter().find(|e| e.rel_name == rel))
        .filter(|e| e.id_from_name.is_none() || e.id_from_name.as_deref() == Some(archive_id.as_str()));
    let Some(entry) =
        named.or_else(|| entries.iter().find(|e| e.id_from_name.as_deref() == Some(archive_id.as_str())))
    else {
        return Ok(());
    };
    let at = FileAt::of(&dir, &scope_dir, entry);
    let seen = Seen {
        session_id: session_id.as_deref(),
        received_ms: Some(*received_ms),
        window: *window,
        part: *part,
        later: &[],
    };

    let row = known(conn, &scope_dir, &entry.rel_name)?;
    // RULING: v2 archives round 4. A file whose indexing failed with an error is read again
    // as a file with no row is.
    let failed = row.as_ref().is_some_and(|r| r.state == "skipped" && r.skip_reason.as_deref() == Some("error"));
    match row {
        None => event_file(ctx, &at, None, &seen, archive_id),
        Some(row) if failed => event_file(ctx, &at, Some(&row), &seen, archive_id),
        // The scan got there first: the job writes what the event said, and reads the file
        // again only where the event changes how it is read: the event says `Whole`, the
        // record has no `carried`, and the first `source_index` is above 0. Ids are
        // positions, so the second reading only adds the units the first left out.
        Some(row) if row.state == "indexed" && row.win_no.is_none() => {
            in_write(conn, || {
                conn.execute(
                    "UPDATE archive_index
                        SET win_no = COALESCE(?1, win_no), part_no = COALESCE(?2, part_no),
                            session_id = COALESCE(session_id, ?3)
                      WHERE scope_dir = ?4 AND rel_name = ?5",
                    params![window, part, session_id, scope_dir, entry.rel_name],
                )?;
                Ok(())
            })?;
            if Start::from_event(*window, *part) != Start::Whole {
                return Ok(());
            }
            match at.read(Some(archive_id)) {
                Ok(ArchiveFile::Live(mut a))
                    if a.carried.is_none() && a.messages.first().is_some_and(|m| m.source_index > 0) =>
                {
                    a.start = Start::Whole;
                    index_archive(ctx, &at, &a, &seen, Some(&row), false)
                }
                _ => Ok(()),
            }
        }
        // Indexed with what an event said, skipped, or forgotten: the same event twice
        // indexes once, and what a skipped file becomes is the scan's to decide.
        Some(_) => Ok(()),
    }
}

/// A file an event names that is not indexed yet: read and settled.
// RULING: v2 archives round 4. When indexing it fails (a panic, an error that is not a
// panic), its `skipped` row keeps its place: the chat and the time `index_archive` would
// have given it, from the record's head and what the event said (the session id, the
// arrival time for a record with no time of its own).
// RULING: v2 archives round 5, item 2. A file the reader refuses has no head to read: it
// takes its place from the event alone (`settle`, `Held::of_event`).
fn event_file(ctx: &IndexerCtx, at: &FileAt, row: Option<&Known>, seen: &Seen, archive_id: &str) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    let got = at.read(Some(archive_id));
    // A slot file's id is inside it: one that is not this archive is the scan's.
    if at.name_id.is_none() && got.is_err() {
        return Ok(());
    }
    let held = match &got {
        Ok(ArchiveFile::Live(a)) => Held::of_live(a, seen, row, at, (ctx.clock)()),
        _ => None,
    };
    let out = catch_unwind(AssertUnwindSafe(|| settle(ctx, at, row, got, seen)));
    if !matches!(out, Ok(Ok(()))) && !conn.is_autocommit() {
        let _ = conn.execute_batch("ROLLBACK");
    }
    match out {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => {
            if !brain_v2::degrade_on(&e, &ctx.status, &ctx.mode) {
                let _ = write_failed(conn, at, "error", || held);
            }
            Err(e)
        }
        // `guarded` counts the panic.
        Err(payload) => {
            let _ = write_failed(conn, at, "panic", || held);
            std::panic::resume_unwind(payload)
        }
    }
}

// ── Forgetting ──────────────────────────────────────────────────────────────

/// `Job::Forgotten` with `kind = "archive"` (called by `indexer::dispatch`): `begin_write`,
/// `forget_archive`, `mark_purged`, `COMMIT`. The event's key is used as it is. When no
/// file and no row exists for the id, nothing. A forget is applied whatever `archives`
/// says.
pub fn on_forgotten(ctx: &mut IndexerCtx, id: &str, project_key: &str) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    let scope_dir = project::project_dir(project_key);
    let dir = archives_root(ctx).join(&scope_dir);
    let now = (ctx.clock)();

    // A file of this id that has no row yet gets one, so that the forget has a digest to
    // reach by and the file is never indexed afterwards, whatever it still holds.
    struct Fresh {
        entry: ArchiveEntry,
        logical: String,
        digest: String,
        created_ms: Option<i64>,
    }
    let mut fresh = Vec::new();
    for entry in archive::list_project(&dir) {
        if entry.id_from_name.as_deref() != Some(id) || known(conn, &scope_dir, &entry.rel_name)?.is_some() {
            continue;
        }
        let (logical, digest, created_ms) =
            match archive::read_archive(&dir.join(&entry.rel_name), &scope_dir, Some(id)) {
                Ok(ArchiveFile::Tombstone { logical, digest, .. }) => (logical, digest, None),
                Ok(ArchiveFile::Live(a)) => (a.logical, a.digest, a.created_ms),
                Err(_) => (String::new(), String::new(), None),
            };
        fresh.push(Fresh { entry, logical, digest, created_ms });
    }
    let has_row: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM archive_index WHERE scope_dir = ?1 AND archive_id = ?2)",
        params![scope_dir, id],
        |r| r.get(0),
    )?;
    if fresh.is_empty() && !has_row {
        return Ok(());
    }

    in_write_gone(ctx, conn, |gone| {
        for f in &fresh {
            let at = FileAt::of(&dir, &scope_dir, &f.entry);
            let tombstone = RowOut {
                logical: Some(f.logical.as_str()).filter(|l| !l.is_empty()),
                digest: Some(f.digest.as_str()).filter(|d| !d.is_empty()),
                created_ms: f.created_ms,
                time_source: f.created_ms.map(|_| "archive"),
                cover_ms: f.created_ms,
                ..RowOut::new(id, "tombstone", now)
            };
            upsert_row(conn, &at, &tombstone)?;
        }
        forget_archive_in(conn, &scope_dir, id, gone)?;
        mark_purged(conn)
    })
}

/// Inside the caller's transaction, which calls [`mark_purged`] before its `COMMIT`.
/// Returns the number of rows purged. Section 2.7:
///
/// 1. the targets are every `archive_index` row of this directory with this id or with the
///    digest of a row that has this id;
/// 2. one purge, without tombstones, of every row whose text came from a target;
/// 3. the target rows say `tombstone`, which is what keeps the files from being indexed;
/// 4. the captures the archive covered: hidden ones always, visible ones when no target was
///    ever indexed here;
/// 5. the chat's recall log up to the archive's cover time.
// The caller that owns a vector array removes the purged rows from it; the indexer's own
// calls go through `forget_archive_in`, which hands back their `vid`s.
pub fn forget_archive(conn: &Connection, scope_dir: &str, archive_id: &str) -> Result<usize> {
    forget_archive_in(conn, scope_dir, archive_id, &mut Vec::new())
}

fn forget_archive_in(conn: &Connection, scope_dir: &str, archive_id: &str, gone: &mut Vec<i64>) -> Result<usize> {
    struct Target {
        rel_name: String,
        archive_id: String,
        logical: String,
        session_id: Option<String>,
        cover_ms: Option<i64>,
        from_mtime: bool,
        was_indexed: bool,
    }
    // 1.
    let targets: Vec<Target> = {
        let mut stmt = conn.prepare(
            "SELECT rel_name, archive_id, COALESCE(logical, ''), session_id, cover_ms,
                    COALESCE(time_source, '') = 'mtime', indexed_ms IS NOT NULL
               FROM archive_index
              WHERE scope_dir = ?1
                AND (archive_id = ?2
                     OR digest IN (SELECT digest FROM archive_index
                                    WHERE scope_dir = ?1 AND archive_id = ?2 AND COALESCE(digest, '') <> ''))",
        )?;
        let rows = stmt.query_map(params![scope_dir, archive_id], |r| {
            Ok(Target {
                rel_name: r.get(0)?,
                archive_id: r.get(1)?,
                logical: r.get(2)?,
                session_id: r.get(3)?,
                cover_ms: r.get(4)?,
                from_mtime: r.get(5)?,
                was_indexed: r.get(6)?,
            })
        })?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    if targets.is_empty() {
        return Ok(0);
    }

    // 2.
    let mut ids: Vec<String> = Vec::new();
    let mut seen_ids: HashSet<&str> = HashSet::new();
    for target in &targets {
        if seen_ids.insert(target.archive_id.as_str()) {
            ids.extend(archive_rows(conn, scope_dir, &target.archive_id)?);
        }
    }
    let mut purged = purge(conn, &ids, false, true, gone)?;

    // 3.
    for target in &targets {
        conn.execute(
            "UPDATE archive_index SET state = 'tombstone', rows_indexed = 0, skip_reason = NULL
              WHERE scope_dir = ?1 AND rel_name = ?2",
            params![scope_dir, target.rel_name],
        )?;
    }

    // 4. and 5.
    let ever_indexed = targets.iter().any(|t| t.was_indexed);
    let now = chrono::Utc::now().timestamp_millis();
    let mut stand_ins: Vec<String> = Vec::new();
    for target in &targets {
        let session_id = target.session_id.as_deref();
        if target.logical.is_empty() && session_id.is_none() {
            continue;
        }
        // SPEC: 2594 bounds the captures by "up to now" when the archive's `cover_ms` is
        // not known (a file first seen as a tombstone): every capture of the chat in this
        // directory up to now, with no lower bound. A row whose time is from `mtime`
        // "purges no capture" (2150) and has no window: its captures are left alone.
        if !target.from_mtime {
            let (after_ms, up_to_ms) = match target.cover_ms {
                Some(cover) => (window_start(conn, scope_dir, &target.logical, session_id, cover)?, cover),
                None => (i64::MIN, now),
            };
            for (id, hidden) in captures(conn, scope_dir, &target.logical, session_id, after_ms, up_to_ms)? {
                if (hidden || !ever_indexed) && !stand_ins.contains(&id) {
                    stand_ins.push(id);
                }
            }
        }
        conn.execute(
            "DELETE FROM recall_log
              WHERE ((?1 <> '' AND logical = ?1) OR (?2 IS NOT NULL AND session_id = ?2)) AND ts_ms <= ?3",
            params![target.logical, session_id, target.cover_ms.unwrap_or(now)],
        )?;
    }
    purged += purge(conn, &stand_ins, false, true, gone)?;
    Ok(purged)
}

// ── The truth: the catch-up scan ────────────────────────────────────────────

/// Every 600 s (called by `indexer::tick`), and at start. Lists
/// `<base>/context-archives/`, compares it with `archive_index` by `(scope_dir, rel_name)`
/// and reads only what is new or has another size. With no connection it returns `Ok(())`.
pub fn scan(ctx: &mut IndexerCtx) -> Result<()> {
    if ctx.conn.is_none() {
        return Ok(());
    }
    if ctx.status.scan.load(Ordering::Relaxed) != 2 {
        ctx.status.scan.store(1, Ordering::Relaxed);
    }

    // 1. and 2. `lstat` only: no file is opened.
    let root = archives_root(ctx);
    let mut dirs: Vec<String> = match std::fs::read_dir(&root) {
        Ok(entries) => entries
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.file_name().into_string().ok())
            .filter(|name| name.len() == 64 && name.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
            .filter(|name| std::fs::symlink_metadata(root.join(name)).is_ok_and(|meta| meta.file_type().is_dir()))
            .collect(),
        Err(_) => Vec::new(),
    };
    dirs.sort();
    let mut files: Vec<(String, ArchiveEntry)> = Vec::new();
    for scope_dir in dirs {
        for entry in archive::list_project(&root.join(&scope_dir)) {
            files.push((scope_dir.clone(), entry));
        }
    }
    let listed: HashSet<(&str, &str)> =
        files.iter().map(|(scope_dir, entry)| (scope_dir.as_str(), entry.rel_name.as_str())).collect();

    // 3. and 4. The files `archive_index` knows first, in the order of their names; then
    // the new ones.
    // RULING: wave4 R-13 round 2, item 1a. New files are indexed in SEAL order, oldest
    // first: "near" and "the file just before" are judged from the files known at that
    // moment, and names are ids, not times. The time is the archive's own `created_ms`,
    // not the file's (an import writes a hundred files within seconds, in any order), and
    // it is not to be had without reading the record: with more than one new file, each is
    // read once for its time and once more to be indexed. The file's `lstat` time, then
    // its place in the listing, only break ties. A tombstone or a file that is refused has
    // no seal time and goes first: a forget lands before anything is indexed. Each new
    // file is indexed knowing the new files of the scan sealed after it, so a far file
    // indexed by an earlier scan is not taken for "the window after" when a file between
    // is in this scan.
    let mut fresh: Vec<usize> = Vec::new();
    for (at, (scope_dir, entry)) in files.iter().enumerate() {
        if !between_files(ctx)? {
            return Ok(());
        }
        // RULING: wave4 R-13 round 3, item 4. A known `skipped` file that is read again is
        // indexed in seal order with the new files, knowing the files sealed after it.
        let dir = root.join(scope_dir);
        let is_new = ctx.cfg.archives
            && ctx.conn.as_ref().is_some_and(|conn| match known(conn, scope_dir, &entry.rel_name) {
                Ok(None) => true,
                Ok(Some(row)) => row.state == "skipped" && read_again(&row, &FileAt::of(&dir, scope_dir, entry)),
                Err(_) => false,
            });
        if is_new {
            fresh.push(at);
        } else {
            one_file(ctx, &dir, scope_dir, entry, &[])?;
        }
    }
    // Each new file is indexed knowing the new files sealed after it (`Seen::later`).
    let mut places: Vec<Option<Later>> = Vec::new();
    if fresh.len() > 1 {
        let mut timed: Vec<(i64, i64, usize, Option<Later>)> = Vec::with_capacity(fresh.len());
        for at in fresh {
            if !between_files(ctx)? {
                return Ok(());
            }
            let (scope_dir, entry) = &files[at];
            let (seal_ms, file_ms, place) = seal_time(&root.join(scope_dir), scope_dir, entry);
            timed.push((seal_ms, file_ms, at, place));
        }
        timed.sort_unstable_by_key(|t| (t.0, t.1, t.2));
        fresh = timed.iter().map(|t| t.2).collect();
        places = timed.into_iter().map(|t| t.3).collect();
    }
    // `after[i]`: where the places of the files after the i-th begin in `later`.
    let mut after: Vec<usize> = Vec::with_capacity(places.len());
    for place in &places {
        after.push(after.last().copied().unwrap_or(0) + usize::from(place.is_some()));
    }
    let later: Vec<Later> = places.into_iter().flatten().collect();
    for (i, at) in fresh.into_iter().enumerate() {
        if !between_files(ctx)? {
            return Ok(());
        }
        let (scope_dir, entry) = &files[at];
        let after_this = after.get(i).map_or(&[][..], |from| &later[*from..]);
        one_file(ctx, &root.join(scope_dir), scope_dir, entry, after_this)?;
    }

    // Gone files are looked at after new files, so a move seen whole by one scan is
    // settled in that scan.
    if !between_files(ctx)? {
        return Ok(());
    }
    settle_moved(ctx, &listed)?;

    ctx.status.scan.store(2, Ordering::Relaxed);
    Ok(())
}

/// `(seal time, lstat time, place)` of a file the scan has not indexed yet: the record's
/// own `created_ms`, or the file's time when the record gives none; `i64::MIN` for a
/// tombstone and for a file that cannot be read as a record. `place` is the file as one
/// sealed after another, for a record that gives its time and its chat. Nothing is written.
fn seal_time(dir: &Path, scope_dir: &str, entry: &ArchiveEntry) -> (i64, i64, Option<Later>) {
    let path = dir.join(&entry.rel_name);
    let file_ms = mtime_ms(&path).unwrap_or(0);
    let read = catch_unwind(AssertUnwindSafe(|| archive::read_archive(&path, scope_dir, entry.id_from_name.as_deref())));
    match read {
        Ok(Ok(ArchiveFile::Live(a))) => {
            // RULING: wave4 R-13 round 3, item 4. The reader's test of a time (as in
            // `index_archive`): a record whose `created_ms` is no time sorts as one that
            // gives none, and stands as no file's neighbour.
            let created_ms =
                a.created_ms.filter(|ms| chrono::DateTime::<chrono::Utc>::from_timestamp_millis(*ms).is_some());
            let place = created_ms.filter(|_| !a.logical.is_empty()).map(|cover_ms| Later {
                scope_dir: scope_dir.to_string(),
                logical: a.logical.clone(),
                rel_name: entry.rel_name.clone(),
                archive_id: a.id.clone(),
                cover_ms,
            });
            (created_ms.unwrap_or(file_ms), file_ms, place)
        }
        _ => (i64::MIN, file_ms, None),
    }
}

/// Between two files: false when the scan must stop (no connection, or a mode that is not
/// `Ready`: a brain that went `Degraded` in the middle of a scan is not written again).
/// Once 30 s have passed since the status row was written, it is written now.
fn between_files(ctx: &mut IndexerCtx) -> Result<bool> {
    let now = (ctx.clock)();
    if ctx.conn.is_none() || !is_ready(ctx) {
        return Ok(false);
    }
    if now - ctx.status.status_written_ms.load(Ordering::Relaxed) >= STATUS_EVERY_MS {
        if let Err(e) = indexer_start::write_status(ctx) {
            if brain_v2::degrade_on(&e, &ctx.status, &ctx.mode) {
                return Err(e);
            }
            ctx.status.write_errors.fetch_add(1, Ordering::Relaxed);
        }
    }
    Ok(true)
}

/// One file, inside `catch_unwind`. A panic marks the file `skipped`, `panic`; an `Err` is
/// rolled back, marks it `skipped`, `error`, and is counted; either way the scan goes on,
/// unless the error says the brain is damaged or was changed, and then the scan returns it
/// at once.
fn one_file(ctx: &mut IndexerCtx, dir: &Path, scope_dir: &str, entry: &ArchiveEntry, later: &[Later]) -> Result<()> {
    let got = catch_unwind(AssertUnwindSafe(|| scan_file(ctx, dir, scope_dir, entry, later)));
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    if !matches!(got, Ok(Ok(()))) && !conn.is_autocommit() {
        let _ = conn.execute_batch("ROLLBACK");
    }
    // RULING: v2 archives round 4. The failed file keeps its place: the record's own time
    // and chat, read from its head once more as the sorted pass reads them (`seal_time`),
    // so that a file that is the one new file of its scan has it too.
    let at = FileAt::of(dir, scope_dir, entry);
    let held = || seal_time(dir, scope_dir, entry).2.as_ref().map(Held::of_later);
    let failed = match got {
        Ok(Ok(())) => return Ok(()),
        Ok(Err(e)) => {
            if !brain_v2::degrade_on(&e, &ctx.status, &ctx.mode) {
                let _ = write_failed(conn, &at, "error", held);
            }
            Err(e)
        }
        Err(_) => {
            ctx.status.panics.fetch_add(1, Ordering::Relaxed);
            // SPEC: 2273 says the scan "logs the location": the file, by its directory and
            // name. The panic's own message and source line go to stderr.
            log_line(&format!("archive scan: panic in {scope_dir}/{}", entry.rel_name));
            write_failed(conn, &at, "panic", held)
        }
    };
    match failed {
        Ok(()) => Ok(()),
        // `guarded` counts the error the scan returns.
        Err(e) if brain_v2::degrade_on(&e, &ctx.status, &ctx.mode) => Err(e),
        Err(e) => {
            ctx.status.write_errors.fetch_add(1, Ordering::Relaxed);
            log_line(&format!("archive scan: {scope_dir}/{}: {e}", entry.rel_name));
            Ok(())
        }
    }
}

/// `skipped` with `reason` (`panic` or `error`) and the binary's `INDEXER_VERSION`, in its
/// own small transaction, with the file's place when it is known. A row that is not
/// `skipped` is left as it is.
// SPEC: 2272-2274 do not say what becomes of a file that is `indexed` and panics when it
// is read again. Its row is left as it is (its rows are still what the index holds) and
// the next scan tries it again.
// RULING: v2 archives round 4. An error that is not a panic writes a row too, `error`, and
// it is tried again at every scan (`read_again`); the place is what keeps the file after it
// from taking its window. A place the row has already stays when none is known now.
fn write_failed(conn: &Connection, at: &FileAt, reason: &str, held: impl FnOnce() -> Option<Held>) -> Result<()> {
    let row = known(conn, at.scope_dir, at.rel_name)?;
    if row.as_ref().is_some_and(|r| r.state != "skipped") {
        return Ok(());
    }
    let held = held();
    let held = held.as_ref();
    let now = chrono::Utc::now().timestamp_millis();
    let first_seen_ms = row.as_ref().map_or(now, |r| r.first_seen_ms);
    let archive_id = at
        .name_id
        .or(held.map(|h| h.archive_id.as_str()))
        .or(row.as_ref().map(|r| r.archive_id.as_str()))
        .unwrap_or("");
    in_write(conn, || {
        let skipped = RowOut {
            skip_reason: Some(reason),
            logical: held.map(|h| h.logical.as_str()),
            session_id: held.and_then(|h| h.session_id.as_deref()),
            created_ms: held.map(|h| h.created_ms),
            time_source: held.map(|h| h.time_source),
            cover_ms: held.map(|h| h.cover_ms),
            win_no: held.and_then(|h| h.win_no),
            part_no: held.and_then(|h| h.part_no),
            ..RowOut::new(archive_id, "skipped", first_seen_ms)
        };
        upsert_row(conn, at, &skipped)
    })
}

/// Step 3 of the scan for one listed file.
fn scan_file(ctx: &IndexerCtx, dir: &Path, scope_dir: &str, entry: &ArchiveEntry, later: &[Later]) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    let at = FileAt::of(dir, scope_dir, entry);
    let on = ctx.cfg.archives;
    let no_event = Seen { later, ..Seen::default() };
    let Some(row) = known(conn, scope_dir, &entry.rel_name)? else {
        // Not known. With `archives = off` nothing is indexed and no row is written for it.
        if !on {
            return Ok(());
        }
        return settle(ctx, &at, None, at.read(None), &no_event);
    };
    let changed = row.file_size != Some(at.size);
    match row.state.as_str() {
        // A live archive never changes and a tombstone is always shorter: "forgotten since".
        "indexed" if changed => match at.read(None) {
            Ok(ArchiveFile::Tombstone { .. }) => forget_indexed(ctx, &at, &row.archive_id),
            // Still live: a person replaced it. The mirror follows the file.
            Ok(ArchiveFile::Live(mut a)) => {
                if !on {
                    return Ok(());
                }
                a.start = Start::from_event(row.win_no, row.part_no);
                index_archive(ctx, &at, &a, &no_event, Some(&row), true)
            }
            // The rows stay, the row keeps its old size, the next scan tries again.
            Err(reason) if not_read(&reason) => Ok(()),
            // It can be read and is refused: it loses its own rows and is `skipped`.
            Err(reason) => refuse_indexed(ctx, &at, &row, &reason),
        },
        // What `restore --forget-turns` asks for: read again with the forget rules off.
        "indexed" if on && row.keep_forget_turns == 1 && row.forget_units > 0 => match at.read(None) {
            Ok(ArchiveFile::Live(mut a)) => {
                a.start = Start::from_event(row.win_no, row.part_no);
                index_archive(ctx, &at, &a, &no_event, Some(&row), false)
            }
            _ => Ok(()),
        },
        "skipped" => {
            let reason = row.skip_reason.as_deref();
            if !read_again(&row, &at) {
                return Ok(());
            }
            let got = at.read(None);
            // SPEC: 2254-2257 say a `paused` row is not read again, and 2292 that a forget
            // is never held back. So a `paused` file whose size changed is read, and only a
            // tombstone does anything; the same with `archives = off`.
            if (reason == Some("paused") || !on) && !matches!(got, Ok(ArchiveFile::Tombstone { .. })) {
                return Ok(());
            }
            settle(ctx, &at, Some(&row), got, &no_event)
        }
        // Known and the same size, or a tombstone: no open, no read.
        _ => Ok(()),
    }
}

/// Is a `skipped` file read again: its size changed, it is a refused file (see
/// `write_refused`) whose time is another now, its indexing failed with an error, or it
/// panicked under an older indexer.
// RULING: v2 archives round 4. A refused file is known by its reason, not by a null
// `cover_ms`: it keeps its place and is still read again when its time changes. A file
// whose indexing failed with an error is tried again at every scan, as a file with no row.
fn read_again(row: &Known, at: &FileAt) -> bool {
    let reason = row.skip_reason.as_deref();
    let changed = row.file_size != Some(at.size);
    let refused = reason.is_some_and(|r| !OWN_REASONS.contains(&r));
    let touched = refused && row.created_ms.is_some() && mtime_ms(&at.path()) != row.created_ms;
    changed || touched || reason == Some("error") || (reason == Some("panic") && INDEXER_VERSION > row.indexer_version)
}

/// Known `indexed` files that are gone. One whose archive is live in another project
/// directory (an `indexed` row there, with the same id or digest, whose file this scan
/// listed) was moved: the rows of the gone copy and the visible captures it covered are
/// purged without tombstones, and its row is deleted, so nothing reads as "forgotten". A
/// gone file with no such copy keeps its rows.
fn settle_moved(ctx: &mut IndexerCtx, listed: &HashSet<(&str, &str)>) -> Result<()> {
    struct Gone {
        scope_dir: String,
        rel_name: String,
        archive_id: String,
        digest: String,
        logical: String,
        session_id: Option<String>,
        cover_ms: Option<i64>,
        from_mtime: bool,
    }
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    let indexed: Vec<Gone> = {
        let mut stmt = conn.prepare(
            "SELECT scope_dir, rel_name, archive_id, COALESCE(digest, ''), COALESCE(logical, ''), session_id,
                    cover_ms, COALESCE(time_source, '') = 'mtime'
               FROM archive_index WHERE state = 'indexed'",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(Gone {
                scope_dir: r.get(0)?,
                rel_name: r.get(1)?,
                archive_id: r.get(2)?,
                digest: r.get(3)?,
                logical: r.get(4)?,
                session_id: r.get(5)?,
                cover_ms: r.get(6)?,
                from_mtime: r.get(7)?,
            })
        })?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    for moved in indexed.iter().filter(|g| !listed.contains(&(g.scope_dir.as_str(), g.rel_name.as_str()))) {
        if !between_files(ctx)? {
            return Ok(());
        }
        let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
        let copies: Vec<(String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT scope_dir, rel_name FROM archive_index
                  WHERE state = 'indexed' AND scope_dir <> ?1
                    AND (archive_id = ?2 OR (?3 <> '' AND digest = ?3))",
            )?;
            let rows =
                stmt.query_map(params![moved.scope_dir, moved.archive_id, moved.digest], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        if !copies.iter().any(|(scope_dir, rel_name)| listed.contains(&(scope_dir.as_str(), rel_name.as_str()))) {
            continue;
        }
        in_write_gone(ctx, conn, |gone| {
            // A second file of the same id in the old directory still owns the rows.
            let shared: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM archive_index
                                WHERE scope_dir = ?1 AND archive_id = ?2 AND rel_name <> ?3 AND state = 'indexed')",
                params![moved.scope_dir, moved.archive_id, moved.rel_name],
                |r| r.get(0),
            )?;
            if !shared {
                let ids = archive_rows(conn, &moved.scope_dir, &moved.archive_id)?;
                purge(conn, &ids, false, false, gone)?;
            }
            if let (Some(cover), false) = (moved.cover_ms, moved.from_mtime) {
                let session_id = moved.session_id.as_deref();
                let start = window_start(conn, &moved.scope_dir, &moved.logical, session_id, cover)?;
                let covered: Vec<String> =
                    captures(conn, &moved.scope_dir, &moved.logical, session_id, start, cover)?
                        .into_iter()
                        .filter(|(_, hidden)| !hidden)
                        .map(|(id, _)| id)
                        .collect();
                purge(conn, &covered, false, false, gone)?;
            }
            conn.execute(
                "DELETE FROM archive_index WHERE scope_dir = ?1 AND rel_name = ?2",
                params![moved.scope_dir, moved.rel_name],
            )?;
            Ok(())
        })?;
    }
    Ok(())
}

// ── The order of a chat's files, "near", and what a forget mark leaves out ───

/// One file of a chat in one directory, as `archive_index` knows it.
struct ChatFile {
    rel_name: String,
    archive_id: String,
    cover_ms: i64,
    first_seen_ms: i64,
    /// `keep_forget_turns = 1`: no purge by a forget mark touches its rows.
    keep: bool,
    indexed: bool,
    win_no: Option<i64>,
    part_no: Option<i64>,
}

impl ChatFile {
    fn order(&self) -> (i64, i64, &str) {
        (self.cover_ms, self.first_seen_ms, &self.rel_name)
    }
}

/// The chat's `archive_index` rows in this directory whose time is not from `mtime`, in
/// any state, ordered by `(cover_ms, first_seen_ms, rel_name)`. "The file just before" and
/// "just after" are the neighbours in this order.
// SPEC: 2153-2155 do not say where a row with no `cover_ms` stands (a file first seen as a
// tombstone has no time at all). It has no place in the order and is left out.
fn chat_files(conn: &Connection, scope_dir: &str, logical: &str, session_id: Option<&str>) -> Result<Vec<ChatFile>> {
    let mut stmt = conn.prepare(
        "SELECT rel_name, archive_id, cover_ms, first_seen_ms, keep_forget_turns = 1, state = 'indexed',
                win_no, part_no
           FROM archive_index
          WHERE scope_dir = ?1
            AND ((?2 <> '' AND logical = ?2) OR (?3 IS NOT NULL AND session_id = ?3))
            AND COALESCE(time_source, '') <> 'mtime' AND cover_ms IS NOT NULL
          ORDER BY cover_ms, first_seen_ms, rel_name",
    )?;
    let rows = stmt.query_map(params![scope_dir, logical, session_id], |r| {
        Ok(ChatFile {
            rel_name: r.get(0)?,
            archive_id: r.get(1)?,
            cover_ms: r.get(2)?,
            first_seen_ms: r.get(3)?,
            keep: r.get(4)?,
            indexed: r.get(5)?,
            win_no: r.get(6)?,
            part_no: r.get(7)?,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// `window_start_ms` over a list of the chat's files: the largest cover time more than
/// `PARTS_SPAN_MS` below `t`, or 0.
fn start_of(files: &[ChatFile], t: i64) -> i64 {
    start_in(&covers(files), t)
}

fn covers(files: &[ChatFile]) -> Vec<i64> {
    files.iter().map(|f| f.cover_ms).collect()
}

fn start_in(covers: &[i64], t: i64) -> i64 {
    let below = t.saturating_sub(PARTS_SPAN_MS);
    covers.iter().copied().filter(|cover| *cover < below).max().unwrap_or(0)
}

/// The stretch of an archive sealed at `cover_ms`: from the start of the window before its
/// own (exclusive) to the end of the window after it (inclusive). It has no end when its
/// window is the newest known: the turn may be in the window that is still open.
// SPEC: 2156-2162 end the stretch at "the end of the window after X's". That window ends
// with the next seal, and the parts of one seal lie within `PARTS_SPAN_MS` of each other:
// the end is the first cover time more than that above X's, plus that span.
fn stretch_of(files: &[ChatFile], cover_ms: i64) -> (i64, i64) {
    stretch_in(&covers(files), cover_ms)
}

fn stretch_in(covers: &[i64], cover_ms: i64) -> (i64, i64) {
    let from = start_in(covers, start_in(covers, cover_ms));
    let above = cover_ms.saturating_add(PARTS_SPAN_MS);
    let to = covers
        .iter()
        .copied()
        .filter(|cover| *cover > above)
        .min()
        .map_or(i64::MAX, |next| next.saturating_add(PARTS_SPAN_MS));
    (from, to)
}

/// The archives of `files` that a mark with the interval `[from_ms, to_ms]` is near: the
/// interval meets their stretch.
fn near_archives(files: &[ChatFile], from_ms: i64, to_ms: i64) -> HashSet<&str> {
    files
        .iter()
        .filter(|f| {
            let (start, end) = stretch_of(files, f.cover_ms);
            to_ms > start && from_ms <= end
        })
        .map(|f| f.archive_id.as_str())
        .collect()
}

/// The stretch a `forgotten_turns` row must meet to be **near** the archive of this chat
/// sealed at `cover_ms`: `(start, end)`, where a row is near when `to_ms > start` and
/// `from_ms <= end`. The end is `i64::MAX` when the archive's window is the newest known.
// SPEC: 2161 gives this function no way to fail. When the chat's files cannot be read it
// answers an empty stretch, `(cover_ms, cover_ms)` with nothing in it: no unit is left out
// or purged on the strength of a neighbourhood that could not be read.
pub fn near_stretch(
    conn: &Connection,
    scope_dir: &str,
    logical: &str,
    session_id: Option<&str>,
    cover_ms: i64,
) -> (i64, i64) {
    match chat_files(conn, scope_dir, logical, session_id) {
        Ok(files) => stretch_of(&files, cover_ms),
        Err(_) => (cover_ms, cover_ms),
    }
}

/// One row of `forgotten_turns`.
struct Forgot {
    source: String,
    turn_key: String,
    from_ms: i64,
    to_ms: i64,
    scope_dir: String,
    archive_id: String,
    head: bool,
    tail: bool,
    /// `cli`: the deleted turn's text, hashed.
    text: TurnText,
}

impl Forgot {
    /// True when this row's key rule takes this unit: any row with the key, except a `cli`
    /// row, which takes only the turn it stands for.
    fn takes_unit(&self, unit: &Unit) -> bool {
        self.source != "cli" || self.text.takes(&unit.body) || self.text.takes_stored(&stored_content(unit))
    }
}

/// The content of an archived unit's row: its body, then its tool lines under a heading.
fn stored_content(unit: &Unit) -> String {
    if unit.aux.is_empty() {
        unit.body.clone()
    } else {
        format!("{}{TOOLS_PART}{}", unit.body, unit.aux)
    }
}

/// Every `forgotten_turns` row of a chat, of any source.
fn forgotten_of(conn: &Connection, logical: &str, session_id: Option<&str>) -> Result<Vec<Forgot>> {
    let mut stmt = conn.prepare(
        "SELECT source, turn_key, from_ms, to_ms, scope_dir, archive_id, head, tail, text_hash, cut_hash
           FROM forgotten_turns
          WHERE (?1 <> '' AND logical = ?1) OR (?2 IS NOT NULL AND session_id = ?2)",
    )?;
    let rows = stmt.query_map(params![logical, session_id], |r| {
        Ok(Forgot {
            source: r.get(0)?,
            turn_key: r.get(1)?,
            from_ms: r.get(2)?,
            to_ms: r.get(3)?,
            scope_dir: r.get(4)?,
            archive_id: r.get(5)?,
            head: r.get(6)?,
            tail: r.get(7)?,
            text: TurnText { whole: r.get(8)?, cut: r.get(9)? },
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// The bytes an archived unit's body is cut at (`archive::BODY_BYTES`).
const UNIT_BODY_BYTES: usize = 4096;

/// What comes between an archived row's body and the tool lines its unit adds.
const TOOLS_PART: &str = "\n\n[tools]\n";

/// hex(sha256("axel/turntext/1" || 0 || normalise(text)))[..32].
pub fn turn_text_hash(text: &str) -> String {
    hash_in(b"axel/turntext/1\0", text)
}

/// hex(sha256("axel/turnrow/1" || 0 || normalise(content)))[..32]: of an archived row's
/// content as it is stored. Another domain than `turn_text_hash`, so that the two never meet.
fn stored_hash(content: &str) -> String {
    hash_in(b"axel/turnrow/1\0", content)
}

fn hash_in(domain: &[u8], text: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(archive::normalise(text).as_bytes());
    hex::encode(hasher.finalize())[..32].to_string()
}

/// What a `cli` mark keeps of the turn it stands for: hashes of its text, never the text.
// RULING: v2 c18b review round 2, item 1. A `cli` mark takes ONLY the deleted turn, never
// another turn with the same prompt: a unit (or an archived row) with the mark's key is
// taken only when its own text, after `archive::normalise`, hashes to the mark's `whole`,
// or, when the unit's body was cut at its size limit, to the mark's `cut` (the hash of the
// deleted text's first 4096 bytes).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TurnText {
    pub whole: String,
    /// `''` when the text is not over 4096 bytes.
    pub cut: String,
}

impl TurnText {
    /// Of a turn's text as a capture holds it ("User: …\nAssistant: …").
    pub fn of(text: &str) -> TurnText {
        let cut = if text.len() > UNIT_BODY_BYTES {
            let mut at = UNIT_BODY_BYTES;
            while !text.is_char_boundary(at) {
                at -= 1;
            }
            turn_text_hash(&text[..at])
        } else {
            String::new()
        };
        TurnText { whole: turn_text_hash(text), cut }
    }

    /// Of an archived row's content, exactly as it is stored (its body, then its tool lines
    /// under their heading): it takes a unit whose row would hold the same content.
    // RULING: v2 c18b review round 3, item 4. Not split at a heading: a body or a tool line
    // can hold the heading's text, and a split there misses the turn.
    pub fn of_archived(content: &str) -> TurnText {
        TurnText { whole: stored_hash(content), cut: String::new() }
    }

    /// True when an archived row's stored content (or a unit's, `stored_content`) is the
    /// row this was made `of_archived`.
    pub fn takes_stored(&self, content: &str) -> bool {
        !self.whole.is_empty() && stored_hash(content) == self.whole
    }

    /// True when a unit's body is this turn.
    pub fn takes(&self, body: &str) -> bool {
        if self.whole.is_empty() {
            return false;
        }
        let hash = turn_text_hash(body);
        // `archive::cut` steps back at most 3 bytes to a character boundary.
        hash == self.whole || (!self.cut.is_empty() && body.len() + 4 > UNIT_BODY_BYTES && hash == self.cut)
    }

    /// True when an archived row's content is this turn: its body is whatever comes before
    /// one of its tools headings, or all of it.
    pub fn takes_row(&self, content: &str) -> bool {
        self.takes_stored(content)
            || content.match_indices(TOOLS_PART).map(|(at, _)| at).chain([content.len()]).any(|end| self.takes(&content[..end]))
    }
}

fn is_turn(unit: &Unit) -> bool {
    matches!(unit.kind, UnitKind::Exchange | UnitKind::Continued)
}

/// The last `Exchange` or `Continued` unit by position.
fn last_turn(units: &[Unit]) -> Option<usize> {
    units
        .iter()
        .enumerate()
        .filter(|(_, unit)| is_turn(unit))
        .max_by_key(|(_, unit)| (unit.source_index, unit.block_index))
        .map(|(at, _)| at)
}

/// The leading `Continued` unit: a `Continued` unit with no `Exchange` or `Continued` unit
/// before it.
fn leading_continued(units: &[Unit]) -> Option<usize> {
    units
        .iter()
        .enumerate()
        .filter(|(_, unit)| is_turn(unit))
        .min_by_key(|(_, unit)| (unit.source_index, unit.block_index))
        .filter(|(_, unit)| unit.kind == UnitKind::Continued)
        .map(|(at, _)| at)
}

/// Where an archive stands among its chat's files.
struct Place<'a> {
    scope_dir: &'a str,
    cover_ms: i64,
    /// The start of its window.
    window_from: i64,
    stretch: (i64, i64),
    prev: Option<&'a ChatFile>,
    next: Option<&'a ChatFile>,
}

/// Step 3 of section 2.3: which units of an archive the chat's `forgotten_turns` rows
/// leave out. *key*: the unit's key is the key of a row near the archive. *head*: the
/// archive is the file just before the file of an `archive` row with `head`, and the unit
/// is its last turn. *tail*: it is the file just after the file of an `archive` row with
/// `tail`, and the unit is its leading `Continued` unit or its note. *time*: a `call` row
/// holds the archive's cover time (its last turn) or the start of its window (its leading
/// `Continued` unit).
fn left_out(units: &[Unit], marks: &[Forgot], place: &Place) -> Vec<bool> {
    let mut out = vec![false; units.len()];
    let near = |m: &Forgot| m.to_ms > place.stretch.0 && m.from_ms <= place.stretch.1;
    for (unit, left) in units.iter().zip(out.iter_mut()) {
        if let Some(key) = unit.turn_key.as_deref().filter(|k| !k.is_empty()) {
            *left = marks.iter().any(|m| m.turn_key == key && near(m) && m.takes_unit(unit));
        }
    }
    let of_file = |file: Option<&ChatFile>, head: bool| {
        file.is_some_and(|f| {
            marks.iter().any(|m| {
                m.source == "archive"
                    && m.scope_dir == place.scope_dir
                    && m.archive_id == f.archive_id
                    && if head { m.head } else { m.tail }
            })
        })
    };
    let call_holds = |t: i64| marks.iter().any(|m| m.source == "call" && m.from_ms <= t && t <= m.to_ms);

    if of_file(place.next, true) || call_holds(place.cover_ms) {
        if let Some(at) = last_turn(units) {
            out[at] = true;
        }
    }
    let after_tail = of_file(place.prev, false);
    if after_tail || (place.window_from > 0 && call_holds(place.window_from)) {
        if let Some(at) = leading_continued(units) {
            out[at] = true;
        }
    }
    if after_tail {
        for (unit, left) in units.iter().zip(out.iter_mut()) {
            if unit.kind == UnitKind::Note {
                *left = true;
            }
        }
    }
    out
}

/// `hidden_by` of a row with the key of a turn deleted with `forget --now`, near it, whose
/// text does not prove it that turn.
pub const UNPROVEN: &str = "cli:unproven";

/// Which units of an archive the `cli` marks near it hold back without proof: a unit with a
/// mark's key whose text is not that turn's (`left_out` takes the proven ones), and the
/// leading `Continued` unit when the prompt of the carried head (`carried_key`) is a
/// mark's key: the rest of a turn whose window was split.
// RULING: v2 c18b review round 3, item 1 (fail closed). The hash is exact one way only: it
// never takes a turn that was not deleted, but the archive's text of the deleted turn can
// differ from the capture's (a worker's report kept the span going, a harness note, a
// window split). Such a unit is not left out (it may be another turn) and not visible (it
// may be the deleted one): it is born hidden, `cli:unproven`.
fn not_proven(units: &[Unit], marks: &[Forgot], place: &Place, carried_key: Option<&str>) -> Vec<bool> {
    let near = |m: &Forgot| m.source == "cli" && m.to_ms > place.stretch.0 && m.from_ms <= place.stretch.1;
    let marked = |key: &str| marks.iter().any(|m| near(m) && m.turn_key == key);
    let mut out: Vec<bool> = units
        .iter()
        .map(|unit| unit.turn_key.as_deref().filter(|k| !k.is_empty()).is_some_and(marked))
        .collect();
    if let (Some(at), Some(key)) = (leading_continued(units), carried_key.filter(|k| !k.is_empty())) {
        out[at] |= marked(key);
    }
    out
}

/// The ids of the units at the edges of a neighbouring file, read from the file itself.
struct Edges {
    last: Option<String>,
    lead: Option<String>,
    note: Option<String>,
}

/// `None` when the neighbour has no rows a mark may take (not indexed, or
/// `keep_forget_turns`), or cannot be read as a live record now.
fn edges_of(ctx: &IndexerCtx, at: &FileAt, file: &ChatFile) -> Option<Edges> {
    if !file.indexed || file.keep {
        return None;
    }
    let Ok(ArchiveFile::Live(mut a)) = archive::read_archive(&at.dir.join(&file.rel_name), at.scope_dir, None) else {
        return None;
    };
    a.start = Start::from_event(file.win_no, file.part_no);
    let units = archive::extract_units(&a, &policy(ctx, false));
    let id = |at_unit: usize| archive::unit_id(at.scope_dir, &a.id, &units[at_unit]);
    Some(Edges {
        last: last_turn(&units).map(id),
        lead: leading_continued(&units).map(id),
        note: units.iter().position(|unit| unit.kind == UnitKind::Note).map(id),
    })
}

/// A hidden captured turn of a chat.
struct HiddenCapture {
    id: String,
    turn_key: Option<String>,
    hidden_ms: i64,
    created_ms: i64,
}

/// The hidden capture rows of a chat in a directory whose `created_ms` lies in
/// `(after_ms, up_to_ms]`, oldest first. A row that is itself marked as a twin is not one.
fn hidden_captures(
    conn: &Connection,
    scope_dir: &str,
    logical: &str,
    session_id: Option<&str>,
    after_ms: i64,
    up_to_ms: i64,
) -> Result<Vec<HiddenCapture>> {
    let mut stmt = conn.prepare(
        "SELECT e.mem_id, e.turn_key, e.hidden_ms, e.created_ms
           FROM memory_ext e JOIN memories m ON m.id = e.mem_id
          WHERE e.origin_kind = 'capture' AND e.hidden_ms IS NOT NULL AND m.project_key = ?1
            AND ((?2 <> '' AND e.logical = ?2) OR (?3 IS NOT NULL AND e.session_id = ?3))
            AND e.created_ms > ?4 AND e.created_ms <= ?5
            AND substr(COALESCE(e.hidden_by, ''), 1, 5) <> 'twin:'
          ORDER BY e.created_ms, e.vid",
    )?;
    let rows = stmt.query_map(params![scope_dir, logical, session_id, after_ms, up_to_ms], |r| {
        Ok(HiddenCapture { id: r.get(0)?, turn_key: r.get(1)?, hidden_ms: r.get(2)?, created_ms: r.get(3)? })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// The archive rows of a chat with one turn key: `(id, archive id)`. The chat is what it
/// is everywhere else (`chat_files`, the `mtime` files of `mark_turn`): the `logical`, or
/// the session an event named for the file. By `logical`, one lookup in
/// `idx_memory_ext_turn`; by session, the chat's `archive_index` rows, then each file's
/// rows by `idx_memory_ext_origin`.
// RULING: v2 c18b review round 5, item 5. `logical OR session_id` for both, not `logical`
// alone: a row's `logical` is its record's, and a file the event tied to the session may
// carry another; what a cli mark finds there is hidden or proven, never missed.
fn key_rows(conn: &Connection, logical: &str, session_id: Option<&str>, turn_key: &str) -> Result<Vec<(String, String)>> {
    let mut stmt = conn.prepare(
        "SELECT mem_id, origin_ref FROM memory_ext
          WHERE ?1 <> '' AND logical = ?1 AND turn_key = ?3 AND origin_kind = 'archive'
         UNION
         SELECT e.mem_id, e.origin_ref FROM archive_index a
           JOIN memory_ext e ON e.origin_kind = 'archive' AND e.origin_ref = a.archive_id
           JOIN memories m ON m.id = e.mem_id AND m.project_key = a.scope_dir
          WHERE ?2 IS NOT NULL AND a.session_id = ?2 AND e.turn_key = ?3",
    )?;
    let rows = stmt.query_map(params![logical, session_id, turn_key], |r| Ok((r.get(0)?, r.get(1)?)))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// The `Exchange` and `Continued` rows of one archive that are in the brain now, by
/// position: `(id, kind letter)`. Kind and place are read from `memory_ext.unit_pos`.
fn turn_rows(conn: &Connection, scope_dir: &str, archive_id: &str) -> Result<Vec<(String, char)>> {
    let mut stmt = conn.prepare(
        "SELECT e.mem_id, e.unit_pos FROM memory_ext e JOIN memories m ON m.id = e.mem_id
          WHERE e.origin_kind = 'archive' AND e.origin_ref = ?1 AND m.project_key = ?2
            AND e.unit_pos IS NOT NULL",
    )?;
    let rows = stmt.query_map(params![archive_id, scope_dir], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    let mut turns: Vec<((u32, u32), String, char)> = Vec::new();
    for row in rows {
        let (id, pos) = row?;
        let mut chars = pos.chars();
        let Some(letter @ ('x' | 'c')) = chars.next() else { continue };
        let Some((message, block)) = chars.as_str().split_once('.') else { continue };
        if let (Ok(message), Ok(block)) = (message.parse(), block.parse()) {
            turns.push(((message, block), id, letter));
        }
    }
    turns.sort();
    Ok(turns.into_iter().map(|(_, id, letter)| (id, letter)).collect())
}

/// The one function through which a `forgotten_turns` row takes rows out: no tombstone,
/// `forget_log = true`, `mark_purged`, and `forget_units` of the archives raised by what
/// went. **No row of an archive whose `keep_forget_turns` is 1 is touched** (the exclusion
/// is in the `WHERE`), and none that is not an archive row of this directory. Inside the
/// caller's transaction. Returns the ids that went.
fn purge_left_out(conn: &Connection, scope_dir: &str, ids: &[String], gone: &mut Vec<i64>) -> Result<Vec<String>> {
    purge_left_out_by(conn, scope_dir, ids, false, gone)
}

/// `purge_left_out`; with `cli`, what went is counted in `cli_units`, not `forget_units`
/// (v2 c18b review round 2, item 2), and the rows of a kept archive go too (round 3, item 5).
fn purge_left_out_by(
    conn: &Connection,
    scope_dir: &str,
    ids: &[String],
    cli: bool,
    gone: &mut Vec<i64>,
) -> Result<Vec<String>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut going: Vec<String> = Vec::new();
    let mut per_archive: Vec<(String, i64)> = Vec::new();
    {
        // RULING: v2 c18b review round 3, item 5. A kept archive obeys a `cli` mark (round 1,
        // item 4): its proven copy of a deleted turn goes too.
        let mut stmt = conn.prepare(
            "SELECT e.origin_ref FROM memory_ext e JOIN memories m ON m.id = e.mem_id
              WHERE e.mem_id = ?1 AND e.origin_kind = 'archive' AND m.project_key = ?2
                AND (?3 OR NOT EXISTS (SELECT 1 FROM archive_index a
                                        WHERE a.scope_dir = ?2 AND a.archive_id = e.origin_ref
                                          AND a.keep_forget_turns = 1))",
        )?;
        for id in ids {
            if going.contains(id) {
                continue;
            }
            let archive_id: Option<String> = stmt.query_row(params![id, scope_dir, cli], |r| r.get(0)).optional()?;
            let Some(archive_id) = archive_id else { continue };
            going.push(id.clone());
            match per_archive.iter_mut().find(|(known, _)| *known == archive_id) {
                Some((_, n)) => *n += 1,
                None => per_archive.push((archive_id, 1)),
            }
        }
    }
    if going.is_empty() {
        return Ok(going);
    }
    purge(conn, &going, false, true, gone)?;
    for (archive_id, n) in &per_archive {
        conn.execute(
            if cli {
                "UPDATE archive_index SET cli_units = cli_units + ?1
                  WHERE scope_dir = ?2 AND archive_id = ?3 AND state = 'indexed'"
            } else {
                "UPDATE archive_index SET forget_units = forget_units + ?1
                  WHERE scope_dir = ?2 AND archive_id = ?3 AND state = 'indexed'"
            },
            params![n, scope_dir, archive_id],
        )?;
    }
    mark_purged(conn)?;
    Ok(going)
}

/// The leading `Continued` row of the file after `archive_id` among `files`, when the row
/// `id` is the last turn of `archive_id`: the rest of its turn, when a compaction split it.
fn rest_of(conn: &Connection, scope_dir: &str, files: &[ChatFile], archive_id: &str, id: &str) -> Result<Option<String>> {
    let Some(at) = files.iter().position(|f| f.archive_id == archive_id) else { return Ok(None) };
    let Some(next) = files.get(at + 1) else { return Ok(None) };
    if turn_rows(conn, scope_dir, archive_id)?.pop().is_none_or(|(last, _)| last != id) {
        return Ok(None);
    }
    let lead = turn_rows(conn, scope_dir, &next.archive_id)?.into_iter().next();
    Ok(lead.filter(|(_, letter)| *letter == 'c').map(|(id, _)| id))
}

/// `rest_of` an archived row, read from the brain: call it before the row is deleted.
pub fn rest_of_turn(conn: &Connection, id: &str) -> Result<Option<String>> {
    let row: Option<(String, Option<String>, Option<String>, String)> = conn
        .query_row(
            "SELECT e.origin_ref, e.logical, e.session_id, m.project_key FROM memory_ext e JOIN memories m ON m.id = e.mem_id
              WHERE e.mem_id = ?1 AND e.origin_kind = 'archive'",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .optional()?;
    let Some((archive_id, logical, session_id, scope_dir)) = row else { return Ok(None) };
    let files = chat_files(conn, &scope_dir, logical.as_deref().unwrap_or(""), session_id.as_deref())?;
    rest_of(conn, &scope_dir, &files, &archive_id, id)
}

/// Whether a hide gives way to `cli:unproven`: one by a model or a pause, which
/// `restore --by-model` or `resume` would lift without a word about the deleted turn.
// RULING: v2 c18b review round 4, item 4.
pub fn yields_to_unproven(hidden_by: &str) -> bool {
    hidden_by == "pause" || hidden_by.starts_with("model:")
}

/// Hides as `cli:unproven` each of `ids` that is an archive row, visible or hidden by a
/// model or a pause (`yields_to_unproven`), and is not in `taken`; counts it in its
/// archive's `cli_unproven`, and adds it to `hidden`.
pub fn hide_unproven(conn: &Connection, ids: &[String], taken: &[String], hidden: &mut Vec<String>) -> Result<()> {
    for id in ids {
        if taken.contains(id) || hidden.contains(id) {
            continue;
        }
        let row: Option<(String, String, Option<String>)> = conn
            .query_row(
                "SELECT e.origin_ref, m.project_key, CASE WHEN e.hidden_ms IS NULL THEN NULL ELSE COALESCE(e.hidden_by, '') END
                   FROM memory_ext e JOIN memories m ON m.id = e.mem_id
                  WHERE e.mem_id = ?1 AND e.origin_kind = 'archive'",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        let Some((archive_id, scope_dir, by)) = row else { continue };
        let done = match by {
            None => store::hide(conn, id, UNPROVEN)?,
            // Already hidden, its shadow already marked: only who hid it, and when, change.
            Some(by) if yields_to_unproven(&by) => conn.execute(
                "UPDATE memory_ext SET hidden_by = ?1, hidden_ms = ?2 WHERE mem_id = ?3 AND hidden_by = ?4",
                params![UNPROVEN, chrono::Utc::now().timestamp_millis(), id, by],
            )?,
            Some(_) => 0,
        };
        if done > 0 {
            conn.execute(
                "UPDATE archive_index SET cli_unproven = cli_unproven + 1
                  WHERE scope_dir = ?1 AND archive_id = ?2 AND state = 'indexed'",
                params![scope_dir, archive_id],
            )?;
            hidden.push(id.clone());
        }
    }
    Ok(())
}

/// Section 4, "the live call": the row of `forgotten_turns` for a turn in which a forget
/// tool was called (source `call`), or for a hidden turn that `forget --now` removed
/// (source `cli`), and what that row leaves out among the rows indexed so far. Returns the
/// ids it purged.
///
/// **It opens no transaction**: the caller holds one (`begin_write_front` on the front
/// thread) and commits; a connection in autocommit is refused. **It is bounded**: the
/// chat's `archive_index` rows, one lookup by `(logical, turn_key)`, and the rows of the
/// archives sealed inside the interval; no scan of `memories`.
///
/// - The row: `ON CONFLICT … DO UPDATE SET to_ms = max(to_ms, excluded.to_ms)`, so two
///   forgets in one turn are one row.
/// - *key*: archive rows of the chat with this key, in archives the row is near.
/// - *time* (`call` only): an archive whose cover time lies in `[from_ms, to_ms]` loses
///   its last `Exchange` or `Continued` row, and one whose window starts there its leading
///   `Continued` row.
/// - No row of an archive whose `keep_forget_turns` is 1 is touched.
/// - The turn's own trace leaves the recall log: the chat's rows in `[from_ms, to_ms]`,
///   its rows with the prompt's key, and for a `call` with no pending turn the chat's
///   newest row.
/// - It may be called late and more than once for one mark: a second call changes nothing
///   but `to_ms`. The rows of one call with other keys (same `from_ms`) are the same call
///   for the time rule.
/// - `mark_purged`, always: the log row deleted here held the prompt's first characters.
///
/// `mark_forgotten_turn_vids` is the same call, and also returns the `vid`s of the rows it
/// purged: the front thread removes their vectors from its array once its `COMMIT` holds.
pub fn mark_forgotten_turn(
    conn: &Connection,
    logical: &str,
    session_id: Option<&str>,
    source: &str,
    from_ms: i64,
    to_ms: i64,
    turn_key: Option<&str>,
) -> Result<Vec<String>> {
    mark_forgotten_turn_vids(conn, logical, session_id, source, from_ms, to_ms, turn_key).map(|(ids, _)| ids)
}

/// `mark_forgotten_turn`, returning `(ids, vids)` of the rows it purged.
// SPEC: 3644-3647 bound the function to indexed lookups, so the time rule reads an
// archive's edge from the rows that are in the brain, not from its file (step 3b of the
// indexer reads the file). When another rule has already taken that archive's edge unit
// (a key mark of a neighbouring window with the same prompt), the row before it is "the
// last" and goes too: one unit too many, which `restore --forget-turns` brings back. An
// archive the same row already covered at an earlier call (`to_ms` only grows) is not
// visited again.
pub fn mark_forgotten_turn_vids(
    conn: &Connection,
    logical: &str,
    session_id: Option<&str>,
    source: &str,
    from_ms: i64,
    to_ms: i64,
    turn_key: Option<&str>,
) -> Result<(Vec<String>, Vec<i64>)> {
    mark_turn(conn, logical, session_id, source, from_ms, to_ms, turn_key, &TurnText::default(), &mut Vec::new())
}

/// The `cli` row of a hidden turn that `forget --now` deleted, with the hashes of its text
/// (`TurnText::of` of a capture's content, `TurnText::of_archived` of an archived row's),
/// and what it leaves out among the rows indexed so far: the archived rows of the chat
/// with its key, near it, whose own text is that turn's. Returns the ids it purged. As
/// `mark_forgotten_turn` otherwise; `mark_forgotten_turn` with `cli` and no text takes no
/// row by its key (nothing proves one the same turn).
///
/// The rows with its key, near it, that their text does not prove the turn, and the
/// leading `Continued` row of the file after one whose last turn has the key, are hidden
/// as `cli:unproven` when they are visible (v2 c18b review round 3, item 2).
pub fn mark_deleted_turn(
    conn: &Connection,
    logical: &str,
    session_id: Option<&str>,
    at_ms: i64,
    turn_key: Option<&str>,
    text: &TurnText,
) -> Result<DeletedTurn> {
    let mut unproven = Vec::new();
    let (taken, _) = mark_turn(conn, logical, session_id, "cli", at_ms, at_ms, turn_key, text, &mut unproven)?;
    Ok(DeletedTurn { taken, unproven })
}

/// What the `cli` mark of one deleted turn did.
#[derive(Debug, Default)]
pub struct DeletedTurn {
    /// The archived rows it took: their text is the turn's.
    pub taken: Vec<String>,
    /// The visible rows it hid as `cli:unproven`.
    pub unproven: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn mark_turn(
    conn: &Connection,
    logical: &str,
    session_id: Option<&str>,
    source: &str,
    from_ms: i64,
    to_ms: i64,
    turn_key: Option<&str>,
    text: &TurnText,
    unproven: &mut Vec<String>,
) -> Result<(Vec<String>, Vec<i64>)> {
    if conn.is_autocommit() {
        return Err(crate::error::AxelError::Other(
            "mark_forgotten_turn: no transaction is open; the caller begins and commits".into(),
        ));
    }
    // SPEC: 3609-3611 name three sources. `archive` rows are written by the indexer with
    // their file and their `head` and `tail`; this function writes the other two.
    if source != "call" && source != "cli" {
        return Err(crate::error::AxelError::Other(format!(
            "mark_forgotten_turn: source {source:?} is neither call nor cli"
        )));
    }
    let key = turn_key.unwrap_or("");
    let to_ms = to_ms.max(from_ms);

    let text_hash = if source == "cli" { text.whole.as_str() } else { "" };
    let covered_to: Option<i64> = conn
        .query_row(
            "SELECT to_ms FROM forgotten_turns
              WHERE logical = ?1 AND source = ?2 AND from_ms = ?3 AND turn_key = ?4 AND archive_id = '' AND text_hash = ?5",
            params![logical, source, from_ms, key, text_hash],
            |r| r.get(0),
        )
        .optional()?;
    // RULING: v2 archives round 5, item 4. Capture writes one keyed row per key of a turn,
    // all with the turn's `from_ms`. The time rule runs once for the call, not once per key:
    // an archive any row of this call already reached is not visited again (`purge` leaves
    // no tombstone, so a second run would take the unit before the last one). The key purge
    // and the log steps stay per key.
    let call_covered_to: Option<i64> = conn.query_row(
        "SELECT MAX(to_ms) FROM forgotten_turns
          WHERE logical = ?1 AND source = ?2 AND from_ms = ?3 AND archive_id = ''",
        params![logical, source, from_ms],
        |r| r.get(0),
    )?;
    // RULING: v2 c18b review round 3, item 4. The text's hash is in the conflict key: two
    // archived turns of one file share the chat, the key and the moment (the file's time),
    // and each deleted turn keeps a row of its own.
    conn.execute(
        "INSERT INTO forgotten_turns(logical, session_id, source, turn_key, from_ms, to_ms, added_ms, text_hash, cut_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
         ON CONFLICT(logical, source, from_ms, turn_key, archive_id, text_hash) DO UPDATE
            SET to_ms = max(to_ms, excluded.to_ms),
                session_id = COALESCE(forgotten_turns.session_id, excluded.session_id)",
        params![
            logical,
            session_id,
            source,
            key,
            from_ms,
            to_ms,
            chrono::Utc::now().timestamp_millis(),
            text_hash,
            if source == "cli" { text.cut.as_str() } else { "" },
        ],
    )?;
    let to_ms = covered_to.map_or(to_ms, |old| old.max(to_ms));
    let seen_before = |t: i64| call_covered_to.is_some_and(|old| t <= old);

    let dirs: Vec<String> = ids_of(
        conn,
        "SELECT DISTINCT scope_dir FROM archive_index
          WHERE (?1 <> '' AND logical = ?1) OR (?2 IS NOT NULL AND session_id = ?2)",
        params![logical, session_id],
    )?;
    let mut keyed = if key.is_empty() { Vec::new() } else { key_rows(conn, logical, session_id, key)? };
    // Every row with the key, proven or not: the files whose last turn may go on in the next.
    let all_keyed = keyed.clone();
    // RULING: v2 c18b review round 2, item 1: a `cli` row takes only the rows whose own
    // text is the deleted turn's.
    let mut not_proven: Vec<(String, String)> = Vec::new();
    if source == "cli" {
        let mut proven = Vec::new();
        for (id, archive_id) in keyed {
            let content: Option<String> = conn
                .query_row("SELECT content FROM memories WHERE id = ?1", [&id], |r| brain_v2::text_at(r, 0))
                .optional()?
                .flatten();
            if content.is_some_and(|c| text.takes_row(&c)) {
                proven.push((id, archive_id));
            } else {
                not_proven.push((id, archive_id));
            }
        }
        keyed = proven;
    }
    let mut purged: Vec<String> = Vec::new();
    // The front thread owns the vector array and removes these from it after its commit.
    let mut vids: Vec<i64> = Vec::new();
    // Where the window before the turn's own begins, in the directory where that is
    // earliest: the lower bound of the log rows that go by key.
    let mut key_from = if dirs.is_empty() { 0 } else { i64::MAX };
    for scope_dir in &dirs {
        let files = chat_files(conn, scope_dir, logical, session_id)?;
        key_from = key_from.min(start_of(&files, start_of(&files, from_ms)));
        let mut targets: Vec<String> = Vec::new();
        let near = near_archives(&files, from_ms, to_ms);
        targets.extend(keyed.iter().filter(|(_, archive_id)| near.contains(archive_id.as_str())).map(|(id, _)| id.clone()));
        if source == "call" {
            for file in files.iter().filter(|f| f.indexed && !f.keep) {
                let sealed_inside = from_ms <= file.cover_ms && file.cover_ms <= to_ms && !seen_before(file.cover_ms);
                let window_from = start_of(&files, file.cover_ms);
                let starts_inside =
                    window_from > 0 && from_ms <= window_from && window_from <= to_ms && !seen_before(window_from);
                if !sealed_inside && !starts_inside {
                    continue;
                }
                let turns = turn_rows(conn, scope_dir, &file.archive_id)?;
                if sealed_inside {
                    targets.extend(turns.last().map(|(id, _)| id.clone()));
                }
                if starts_inside {
                    targets.extend(turns.first().filter(|(_, letter)| *letter == 'c').map(|(id, _)| id.clone()));
                }
            }
        }
        // RULING: v2 c18b review round 3, item 2 (fail closed). What has the key, is near and
        // is not proven may be the deleted turn as the archive holds it: hidden, in this
        // transaction. So is the leading `Continued` row of the file after one whose last
        // turn has the key: the rest of a turn whose window was split.
        // RULING: v2 c18b review round 4, item 1: the rests are read before the purge; once
        // a proven last turn is gone, the turn before it is "the last" and no rest is found.
        let mut doubt: Vec<String> = Vec::new();
        if source == "cli" {
            doubt.extend(
                not_proven.iter().filter(|(_, archive_id)| near.contains(archive_id.as_str())).map(|(id, _)| id.clone()),
            );
            for (id, archive_id) in all_keyed.iter().filter(|(_, archive_id)| near.contains(archive_id.as_str())) {
                doubt.extend(rest_of(conn, scope_dir, &files, archive_id, id)?);
            }
            // RULING: v2 c18b review round 4, item 5. A file timed by its `mtime` has no place
            // among the chat's files (`chat_files`) and no mark reaches it: its rows with the
            // key, proven or not, are hidden, never taken.
            let untimed: HashSet<String> = ids_of(
                conn,
                "SELECT archive_id FROM archive_index
                  WHERE scope_dir = ?1 AND state = 'indexed' AND time_source = 'mtime'
                    AND ((?2 <> '' AND logical = ?2) OR (?3 IS NOT NULL AND session_id = ?3))",
                params![scope_dir, logical, session_id],
            )?
            .into_iter()
            .collect();
            doubt.extend(all_keyed.iter().filter(|(_, archive_id)| untimed.contains(archive_id)).map(|(id, _)| id.clone()));
        }
        purged.extend(purge_left_out_by(conn, scope_dir, &targets, source == "cli", &mut vids)?);
        if source == "cli" {
            hide_unproven(conn, &doubt, &purged, unproven)?;
        }
    }

    // The turn's own trace in the recall log: its row was written at the start of the turn,
    // with the prompt's first characters, and no stored row names it.
    // SPEC: 3652-3655 give the delete for `call` only, with no upper bound.
    // RULING: wave4 R-13 round 2, item 3. `ts_ms <= to_ms` for every source: the front
    // thread keeps a mark it could not write and writes it minutes later, and an open end
    // would then take every later log row of the chat. (A log job that arrives late is
    // refused by `recall` when its time lies in a `call` mark.)
    let mut log_rows = conn.execute(
        "DELETE FROM recall_log
          WHERE ((?1 <> '' AND logical = ?1) OR (?2 IS NOT NULL AND session_id = ?2))
            AND ts_ms >= ?3 AND ts_ms <= ?4",
        params![logical, session_id, from_ms, to_ms],
    )?;
    // RULING: wave4 R-13 round 2, item 2. By key too: the chat's log rows with the hash of
    // this prompt, from the window before the turn's own up to the mark's end.
    if !key.is_empty() {
        log_rows += conn.execute(
            "DELETE FROM recall_log
              WHERE ?1 <> '' AND logical = ?1 AND query_hash = ?2 AND ts_ms > ?3 AND ts_ms <= ?4",
            params![logical, key, key_from, to_ms],
        )?;
    }
    // The same ruling: with no pending turn the front gives `from_ms` = the time of the
    // call, and the turn's log row, written at its start, is older than that. A `call`
    // mark with no key that found no log row in its interval takes the chat's newest log
    // row at or before `to_ms`. Once for a mark (a retry of the same mark finds its row),
    // and not when an earlier `call` mark, with a key or without, already stands between
    // that log row and this one: that mark took the turn's row, and what is newest now is
    // an older turn's.
    // RULING: wave4 R-13 round 3, item 1. A forgotten turn that wrote no log row at all (a
    // prompt too short for the log, a hook that was never forwarded) gives up the row of
    // the turn before it: ACCEPTED as it is. Only a trace of that turn is lost, and it errs
    // towards forgetting.
    if source == "call" && turn_key.is_none() && covered_to.is_none() && log_rows == 0 {
        log_rows += conn.execute(
            "DELETE FROM recall_log
              WHERE id = (SELECT id FROM recall_log
                           WHERE ((?1 <> '' AND logical = ?1) OR (?2 IS NOT NULL AND session_id = ?2))
                             AND ts_ms <= ?4
                           ORDER BY ts_ms DESC, id DESC LIMIT 1)
                AND NOT EXISTS (SELECT 1 FROM forgotten_turns f
                                 WHERE f.logical = ?1 AND f.source = 'call'
                                   AND f.from_ms >= recall_log.ts_ms AND f.from_ms < ?3)",
            params![logical, session_id, from_ms, to_ms],
        )?;
    }
    if log_rows > 0 {
        log_line(&format!("forget mark: {log_rows} recall-log rows of the turn deleted"));
    }
    mark_purged(conn)?;
    Ok((purged, vids))
}
