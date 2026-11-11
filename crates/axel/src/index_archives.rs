//! Scan, index one archive, forget one archive, the forget marks (`mark_forgotten_turn`,
//! `chat_files`, `near_stretch`). Commits 13a and 13b write this file.
//!
//! Commit 13a: the scan, the job of `on_context_archived`, `index_archive`,
//! `forget_archive`, and the log after a purge. Commit 13b: the forget marks and what they
//! leave out (steps 3 and 3b of section 2.3), the twins of hidden captures (3a), and the
//! live call (`mark_forgotten_turn`). Nothing here writes, locks, renames or chmods
//! anything under `context-archives`.

use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

use rusqlite::{params, Connection, OptionalExtension};

use crate::archive::{self, ArchiveEntry, ArchiveFile, ArchivePolicy, LiveArchive, SkipReason, Start, Unit, UnitKind};
use crate::brain_v2;
use crate::error::Result;
use crate::ext::server::log_line;
use crate::ext::types::{IndexerCtx, Job, Mode};
use crate::indexer_start;
use crate::project;
use crate::project_memory::{RecordLinks, Sensitivity};
use crate::store::{self, Inserted, NewRow};

/// Kept in each `archive_index` row. Version 1 does not re-extract old rows.
pub const INDEXER_VERSION: i64 = 1;
/// Archives of one chat within this many milliseconds of each other are the parts of one
/// seal and share a window.
pub const PARTS_SPAN_MS: i64 = 10_000;

/// A long scan writes the status row once this much time has passed since the last one.
const STATUS_EVERY_MS: i64 = 30_000;
/// How long the idle checkpoint waits for a reader before it gives up until the next turn.
const CHECKPOINT_WAIT_MS: u64 = 100;

// ── The log after a purge ───────────────────────────────────────────────────

/// A purge in this transaction forgot text: the write-ahead log still holds the frames
/// written when the row was inserted, and an automatic checkpoint does not shorten it.
/// Called INSIDE the transaction of every purge that forgets, on any thread, before its
/// `COMMIT`; the indexer then truncates the log at its next idle turn
/// ([`checkpoint_after_purge`]). A row and not a field, so that a purge made by the front
/// thread or the command line, or just before a crash, is seen too (section 2.7).
pub fn mark_purged(conn: &Connection) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO context_data(key, layer, value) VALUES('purge_pending', 'axel', '1')",
        [],
    )?;
    Ok(())
}

/// The indexer's idle turn (called by `indexer::run` through `guarded`): when a purge is
/// pending, `wal_checkpoint(TRUNCATE)`; when the checkpoint was not blocked, the mark is
/// deleted in one small transaction, whose frame holds no text. When it was blocked
/// (`busy` is 1) the mark stays and the next idle turn tries again (section 2.7).
pub fn checkpoint_after_purge(ctx: &mut IndexerCtx) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    let pending: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM context_data WHERE key = 'purge_pending' AND layer = 'axel')",
        [],
        |r| r.get(0),
    )?;
    if !pending {
        return Ok(());
    }
    // SPEC: 2560-2564 do not say how long the checkpoint may wait. A `TRUNCATE` checkpoint
    // that is blocked by a reader waits for the connection's busy timeout (5 s on the
    // indexer) and keeps every other writer out while it waits, and an idle turn comes
    // every second. So it waits 100 ms here, and the connection's own timeout is put back
    // before anything else runs.
    let usual: i64 = conn.query_row("PRAGMA busy_timeout", [], |r| r.get(0))?;
    conn.busy_timeout(Duration::from_millis(CHECKPOINT_WAIT_MS))?;
    let answer = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get::<_, i64>(0));
    conn.busy_timeout(Duration::from_millis(usual.max(0) as u64))?;
    if answer? != 0 {
        return Ok(());
    }
    in_write(conn, || {
        conn.execute("DELETE FROM context_data WHERE key = 'purge_pending' AND layer = 'axel'", [])?;
        Ok(())
    })
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

/// What an `on_context_archived` event said about a file. The scan has none.
#[derive(Clone, Copy, Default)]
struct Seen<'a> {
    session_id: Option<&'a str>,
    received_ms: Option<i64>,
    window: Option<i64>,
    part: Option<i64>,
}

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
                                   indexer_version, first_seen_ms, indexed_ms, unmatched_hidden)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21,
                 ?22)
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
            (Some(ms), Some("observed"), Some(cover)) => return (ms, "observed", cover),
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
    let policy = policy(ctx, row.is_some_and(|r| r.keep_forget_turns == 1));
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
        return in_write(conn, || {
            if replace {
                forget_rows(conn, at.scope_dir, row)?;
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
    // `keep_forget_turns` gives and takes no mark (step 3 is skipped whole for both).
    let timed = time_source != "mtime";
    let marks_on = timed && !policy.keep_forget_turns;
    let own: &[archive::ForgetMark] = if marks_on { &spans } else { &[] };
    let files: Vec<ChatFile> = if timed {
        // The chat's files with this one at its place, as it will be written: the windows
        // of its neighbours are bounded by it too.
        let mut files = chat_files(conn, at.scope_dir, &a.logical, session_id)?;
        files.retain(|f| f.rel_name != at.rel_name);
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
    // Step 3b by position: which unit of the file just before, and of the file just after,
    // this file's own marks leave out. Read from those files, not from their rows: a row
    // another rule already took must not make the one before it "the last".
    let before = if own.iter().any(|m| m.head) { prev.and_then(|f| edges_of(ctx, at, f)) } else { None };
    let after = if own.iter().any(|m| m.tail) { next.and_then(|f| edges_of(ctx, at, f)) } else { None };

    in_write(conn, || {
        if replace {
            forget_rows(conn, at.scope_dir, row)?;
            // The marks the old file gave are not the new file's.
            conn.execute(
                "DELETE FROM forgotten_turns WHERE source = 'archive' AND scope_dir = ?1 AND archive_id = ?2",
                params![at.scope_dir, row.map_or(a.id.as_str(), |r| r.archive_id.as_str())],
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
        if marks_on {
            for mark in own {
                conn.execute(
                    "INSERT INTO forgotten_turns(logical, session_id, source, turn_key, from_ms, to_ms, scope_dir,
                                                 archive_id, head, tail, added_ms)
                     VALUES (?1, ?2, 'archive', ?3, ?4, ?4, ?5, ?6, ?7, ?8, ?9)
                     ON CONFLICT(logical, source, from_ms, turn_key, archive_id) DO UPDATE
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
            let marks = forgotten_of(conn, &a.logical, session_id)?;
            let place = Place { scope_dir: at.scope_dir, cover_ms, window_from, stretch, prev, next };
            out = left_out(&units, &marks, &place);
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
        for (unit, left) in units.iter().zip(&out) {
            if *left {
                continue;
            }
            let (letter, class) = kind_of(unit);
            let content = if unit.aux.is_empty() {
                unit.body.clone()
            } else {
                format!("{}\n\n[tools]\n{}", unit.body, unit.aux)
            };
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
                unit_pos: Some(format!("{letter}{}.{}", unit.source_index, unit.block_index)),
                session_id,
                logical: Some(&a.logical),
                links: RecordLinks::default(),
                turn_key: unit.turn_key.clone(),
                hidden: twin_of(unit),
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
        purge_left_out(conn, at.scope_dir, &left)?;

        // A hidden capture of this file's own window that no unit of the file matches and
        // that has no twin anywhere is counted, and nothing else is hidden for it.
        let mut unmatched_hidden = 0i64;
        for h in hidden.iter().filter(|h| h.created_ms > window_from && h.created_ms <= cover_ms) {
            let matched = h.turn_key.as_deref().filter(|k| !k.is_empty()).is_some_and(|key| {
                units.iter().any(|unit| unit.turn_key.as_deref() == Some(key))
            });
            let has_twin: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM memory_ext WHERE hidden_by = ?1)",
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
                for (id, archive_id) in key_rows(conn, &a.logical, key)? {
                    if archive_id != a.id && near.contains(archive_id.as_str()) {
                        elsewhere.push(id);
                    }
                }
            }
            elsewhere.extend(before.as_ref().and_then(|e| e.last.clone()));
            elsewhere.extend(after.as_ref().and_then(|e| e.lead.clone()));
            elsewhere.extend(after.as_ref().and_then(|e| e.note.clone()));
            purge_left_out(conn, at.scope_dir, &elsewhere)?;
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
            store::purge_many(conn, &covered, false, false)?;
        }

        // 5. The row.
        let indexed = RowOut {
            digest: Some(&a.digest),
            state: "indexed",
            source_message_count: Some(i64::from(a.source_message_count)),
            message_count: Some(a.messages.len() as i64),
            rows_indexed,
            forget_units: forget_spans + out.iter().filter(|left| **left).count() as i64,
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
fn forget_rows(conn: &Connection, scope_dir: &str, row: Option<&Known>) -> Result<()> {
    let Some(row) = row else { return Ok(()) };
    let ids = archive_rows(conn, scope_dir, &row.archive_id)?;
    if store::purge_many(conn, &ids, false, true)? > 0 {
        mark_purged(conn)?;
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
        Ok(ArchiveFile::Tombstone { id, logical, digest }) => in_write(conn, || {
            let tombstone = RowOut {
                logical: Some(logical.as_str()).filter(|l| !l.is_empty()),
                digest: Some(digest.as_str()).filter(|d| !d.is_empty()),
                session_id: seen.session_id,
                ..RowOut::new(&id, "tombstone", first_seen_ms)
            };
            upsert_row(conn, at, &tombstone)?;
            forget_archive(conn, at.scope_dir, &id)?;
            mark_purged(conn)
        }),
        Err(reason) if is_io(&reason) => Ok(()),
        Err(reason) => in_write(conn, || {
            let archive_id = at.name_id.or(row.map(|r| r.archive_id.as_str())).unwrap_or("");
            let skipped =
                RowOut { skip_reason: Some(&reason.0), ..RowOut::new(archive_id, "skipped", first_seen_ms) };
            upsert_row(conn, at, &skipped)
        }),
    }
}

/// An indexed file that is a tombstone now, or that can be read and is no longer a live
/// record: its rows go, and the row says `tombstone` and has the file's new size.
fn forget_indexed(conn: &Connection, at: &FileAt, archive_id: &str) -> Result<()> {
    in_write(conn, || {
        forget_archive(conn, at.scope_dir, archive_id)?;
        conn.execute(
            "UPDATE archive_index SET file_size = ?1 WHERE scope_dir = ?2 AND rel_name = ?3",
            params![at.size, at.scope_dir, at.rel_name],
        )?;
        mark_purged(conn)
    })
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
    };

    match known(conn, &scope_dir, &entry.rel_name)? {
        None => {
            let got = at.read(Some(archive_id));
            // A slot file's id is inside it: one that is not this archive is the scan's.
            if entry.id_from_name.is_none() && got.is_err() {
                return Ok(());
            }
            settle(ctx, &at, None, got, &seen)
        }
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

    in_write(conn, || {
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
        forget_archive(conn, &scope_dir, id)?;
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
pub fn forget_archive(conn: &Connection, scope_dir: &str, archive_id: &str) -> Result<usize> {
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
    let mut purged = store::purge_many(conn, &ids, false, true)?;

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
    purged += store::purge_many(conn, &stand_ins, false, true)?;
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

    // 3. and 4.
    for (scope_dir, entry) in &files {
        if !between_files(ctx)? {
            return Ok(());
        }
        one_file(ctx, &root.join(scope_dir), scope_dir, entry)?;
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
/// rolled back and counted; either way the scan goes on, unless the error says the brain
/// is damaged or was changed, and then the scan returns it at once.
fn one_file(ctx: &mut IndexerCtx, dir: &Path, scope_dir: &str, entry: &ArchiveEntry) -> Result<()> {
    let got = catch_unwind(AssertUnwindSafe(|| scan_file(ctx, dir, scope_dir, entry)));
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    if !matches!(got, Ok(Ok(()))) && !conn.is_autocommit() {
        let _ = conn.execute_batch("ROLLBACK");
    }
    let failed = match got {
        Ok(Ok(())) => return Ok(()),
        Ok(Err(e)) => Err(e),
        Err(_) => {
            ctx.status.panics.fetch_add(1, Ordering::Relaxed);
            // SPEC: 2273 says the scan "logs the location": the file, by its directory and
            // name. The panic's own message and source line go to stderr.
            log_line(&format!("archive scan: panic in {scope_dir}/{}", entry.rel_name));
            mark_panic(conn, dir, scope_dir, entry)
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

/// `skipped`, `panic`, with the binary's `INDEXER_VERSION`: its own small transaction.
// SPEC: 2272-2274 do not say what becomes of a file that is `indexed` and panics when it
// is read again. Its row is left as it is (its rows are still what the index holds) and
// the next scan tries it again.
fn mark_panic(conn: &Connection, dir: &Path, scope_dir: &str, entry: &ArchiveEntry) -> Result<()> {
    let row = known(conn, scope_dir, &entry.rel_name)?;
    if row.as_ref().is_some_and(|r| r.state != "skipped") {
        return Ok(());
    }
    let at = FileAt::of(dir, scope_dir, entry);
    let now = chrono::Utc::now().timestamp_millis();
    let first_seen_ms = row.as_ref().map_or(now, |r| r.first_seen_ms);
    let archive_id = at.name_id.or(row.as_ref().map(|r| r.archive_id.as_str())).unwrap_or("");
    in_write(conn, || {
        let skipped = RowOut { skip_reason: Some("panic"), ..RowOut::new(archive_id, "skipped", first_seen_ms) };
        upsert_row(conn, &at, &skipped)
    })
}

/// Step 3 of the scan for one listed file.
fn scan_file(ctx: &IndexerCtx, dir: &Path, scope_dir: &str, entry: &ArchiveEntry) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    let at = FileAt::of(dir, scope_dir, entry);
    let on = ctx.cfg.archives;
    let no_event = Seen::default();
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
            Ok(ArchiveFile::Tombstone { .. }) => forget_indexed(conn, &at, &row.archive_id),
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
            // It can be read and is no longer the same live record: it loses its rows.
            Err(_) => forget_indexed(conn, &at, &row.archive_id),
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
            if !(changed || (reason == Some("panic") && INDEXER_VERSION > row.indexer_version)) {
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
    for gone in indexed.iter().filter(|g| !listed.contains(&(g.scope_dir.as_str(), g.rel_name.as_str()))) {
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
                stmt.query_map(params![gone.scope_dir, gone.archive_id, gone.digest], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        if !copies.iter().any(|(scope_dir, rel_name)| listed.contains(&(scope_dir.as_str(), rel_name.as_str()))) {
            continue;
        }
        in_write(conn, || {
            // A second file of the same id in the old directory still owns the rows.
            let shared: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM archive_index
                                WHERE scope_dir = ?1 AND archive_id = ?2 AND rel_name <> ?3 AND state = 'indexed')",
                params![gone.scope_dir, gone.archive_id, gone.rel_name],
                |r| r.get(0),
            )?;
            if !shared {
                let ids = archive_rows(conn, &gone.scope_dir, &gone.archive_id)?;
                store::purge_many(conn, &ids, false, false)?;
            }
            if let (Some(cover), false) = (gone.cover_ms, gone.from_mtime) {
                let session_id = gone.session_id.as_deref();
                let start = window_start(conn, &gone.scope_dir, &gone.logical, session_id, cover)?;
                let covered: Vec<String> =
                    captures(conn, &gone.scope_dir, &gone.logical, session_id, start, cover)?
                        .into_iter()
                        .filter(|(_, hidden)| !hidden)
                        .map(|(id, _)| id)
                        .collect();
                store::purge_many(conn, &covered, false, false)?;
            }
            conn.execute(
                "DELETE FROM archive_index WHERE scope_dir = ?1 AND rel_name = ?2",
                params![gone.scope_dir, gone.rel_name],
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
    let below = t.saturating_sub(PARTS_SPAN_MS);
    files.iter().map(|f| f.cover_ms).filter(|cover| *cover < below).max().unwrap_or(0)
}

/// The stretch of an archive sealed at `cover_ms`: from the start of the window before its
/// own (exclusive) to the end of the window after it (inclusive). It has no end when its
/// window is the newest known: the turn may be in the window that is still open.
// SPEC: 2156-2162 end the stretch at "the end of the window after X's". That window ends
// with the next seal, and the parts of one seal lie within `PARTS_SPAN_MS` of each other:
// the end is the first cover time more than that above X's, plus that span.
fn stretch_of(files: &[ChatFile], cover_ms: i64) -> (i64, i64) {
    let from = start_of(files, start_of(files, cover_ms));
    let above = cover_ms.saturating_add(PARTS_SPAN_MS);
    let to = files
        .iter()
        .map(|f| f.cover_ms)
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
}

/// Every `forgotten_turns` row of a chat, of any source.
fn forgotten_of(conn: &Connection, logical: &str, session_id: Option<&str>) -> Result<Vec<Forgot>> {
    let mut stmt = conn.prepare(
        "SELECT source, turn_key, from_ms, to_ms, scope_dir, archive_id, head, tail FROM forgotten_turns
          WHERE logical = ?1 OR (?2 IS NOT NULL AND session_id = ?2)",
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
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
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
            *left = marks.iter().any(|m| m.turn_key == key && near(m));
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

/// The archive rows of a chat with one turn key: `(id, archive id)`. One lookup in
/// `idx_memory_ext_turn`.
fn key_rows(conn: &Connection, logical: &str, turn_key: &str) -> Result<Vec<(String, String)>> {
    let mut stmt = conn.prepare(
        "SELECT mem_id, origin_ref FROM memory_ext
          WHERE logical = ?1 AND turn_key = ?2 AND origin_kind = 'archive'",
    )?;
    let rows = stmt.query_map(params![logical, turn_key], |r| Ok((r.get(0)?, r.get(1)?)))?;
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
fn purge_left_out(conn: &Connection, scope_dir: &str, ids: &[String]) -> Result<Vec<String>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mut going: Vec<String> = Vec::new();
    let mut per_archive: Vec<(String, i64)> = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT e.origin_ref FROM memory_ext e JOIN memories m ON m.id = e.mem_id
              WHERE e.mem_id = ?1 AND e.origin_kind = 'archive' AND m.project_key = ?2
                AND NOT EXISTS (SELECT 1 FROM archive_index a
                                 WHERE a.scope_dir = ?2 AND a.archive_id = e.origin_ref
                                   AND a.keep_forget_turns = 1)",
        )?;
        for id in ids {
            if going.contains(id) {
                continue;
            }
            let archive_id: Option<String> = stmt.query_row(params![id, scope_dir], |r| r.get(0)).optional()?;
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
    store::purge_many(conn, &going, false, true)?;
    for (archive_id, n) in &per_archive {
        conn.execute(
            "UPDATE archive_index SET forget_units = forget_units + ?1
              WHERE scope_dir = ?2 AND archive_id = ?3 AND state = 'indexed'",
            params![n, scope_dir, archive_id],
        )?;
    }
    mark_purged(conn)?;
    Ok(going)
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
/// - The turn's own trace leaves the recall log: the chat's rows from `from_ms` on for a
///   `call`, the rows of the mark's one moment for a `cli` mark.
/// - `mark_purged`, always: the log row deleted here held the prompt's first characters.
// SPEC: 3644-3647 bound the function to indexed lookups, so the time rule reads an
// archive's edge from the rows that are in the brain, not from its file (step 3b of the
// indexer reads the file). When another rule has already taken that archive's edge unit
// (a key mark of a neighbouring window with the same prompt), the row before it is "the
// last" and goes too: one unit too many, which `restore --forget-turns` brings back. An
// archive the same row already covered at an earlier call (`to_ms` only grows) is not
// visited again.
pub fn mark_forgotten_turn(
    conn: &Connection,
    logical: &str,
    session_id: Option<&str>,
    source: &str,
    from_ms: i64,
    to_ms: i64,
    turn_key: Option<&str>,
) -> Result<Vec<String>> {
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

    let covered_to: Option<i64> = conn
        .query_row(
            "SELECT to_ms FROM forgotten_turns
              WHERE logical = ?1 AND source = ?2 AND from_ms = ?3 AND turn_key = ?4 AND archive_id = ''",
            params![logical, source, from_ms, key],
            |r| r.get(0),
        )
        .optional()?;
    conn.execute(
        "INSERT INTO forgotten_turns(logical, session_id, source, turn_key, from_ms, to_ms, added_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(logical, source, from_ms, turn_key, archive_id) DO UPDATE
            SET to_ms = max(to_ms, excluded.to_ms),
                session_id = COALESCE(forgotten_turns.session_id, excluded.session_id)",
        params![logical, session_id, source, key, from_ms, to_ms, chrono::Utc::now().timestamp_millis()],
    )?;
    let to_ms = covered_to.map_or(to_ms, |old| old.max(to_ms));
    let seen_before = |t: i64| covered_to.is_some_and(|old| t <= old);

    let dirs: Vec<String> = ids_of(
        conn,
        "SELECT DISTINCT scope_dir FROM archive_index
          WHERE (?1 <> '' AND logical = ?1) OR (?2 IS NOT NULL AND session_id = ?2)",
        params![logical, session_id],
    )?;
    let keyed = if key.is_empty() { Vec::new() } else { key_rows(conn, logical, key)? };
    let mut purged: Vec<String> = Vec::new();
    for scope_dir in &dirs {
        let files = chat_files(conn, scope_dir, logical, session_id)?;
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
        purged.extend(purge_left_out(conn, scope_dir, &targets)?);
    }

    // The turn's own trace in the recall log: its row was written at the start of the turn,
    // with the prompt's first characters, and no stored row names it.
    // SPEC: 3652-3655 give the delete for `call` only, with no upper bound: the turn is
    // still running, and its `to_ms` grows. A `cli` mark is one moment (the hidden turn's
    // time) and takes the log rows of that moment (ruling on the review of commit 15).
    let log_to = if source == "call" { i64::MAX } else { to_ms };
    let log_rows = conn.execute(
        "DELETE FROM recall_log
          WHERE ((?1 <> '' AND logical = ?1) OR (?2 IS NOT NULL AND session_id = ?2))
            AND ts_ms >= ?3 AND ts_ms <= ?4",
        params![logical, session_id, from_ms, log_to],
    )?;
    if log_rows > 0 {
        log_line(&format!("forget mark: {log_rows} recall-log rows of the turn deleted"));
    }
    mark_purged(conn)?;
    Ok(purged)
}
