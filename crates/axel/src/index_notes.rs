//! The notes pass (section 2.6): the notes saved with `memory_store` are core's JSONL
//! files; this keeps the brain's `note_file` rows in step with them.
//!
//! What decides here: a note the owner forgot leaves the brain (a tombstone line always
//! deletes), and a note he did not forget never leaves it because a file or a folder could
//! not be read. Only a file that was read, and says so, removes a row.

use std::collections::HashSet;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::sync::atomic::Ordering;

use chrono::DateTime;
use rusqlite::{params, Connection, OptionalExtension};

use crate::error::Result;
use crate::ext::types::{IndexerCtx, Mode};
use crate::notes::{self, NoteRecord, NoteSensitivity, NotesError, NotesFile};
use crate::project_memory::{RecordLinks, Sensitivity};
use crate::store::{self, Inserted, NewRow};
use crate::{brain_v2, index_archives, project};

const DAY_MS: i64 = 86_400_000;
const TITLE_WORDS: usize = 12;
// SPEC: 2350-2352: "`title` = the first 12 words of the content" and "a secret note: the
// body is not word-indexed". The word index holds a secret row's title, so a title made
// from the body would index its first 12 words. A secret note gets this fixed title.
const SECRET_TITLE: &str = "Secret note";

/// What one pass did, for its tests. Counts of this call only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassReport {
    /// Files that were read and whose transaction committed.
    pub files_read: usize,
    /// Files whose `(size, mtime_ns)` equals their `note_files` row, which says `ok = 1`.
    pub files_unchanged: usize,
    /// Files the reader refused (`TooLarge`, `Io`): `ok = 0`, the mirror as it was. Such a
    /// file is read again at every pass until it can be read.
    pub files_refused: usize,
    /// Files that changed under the read: nothing written.
    pub files_changed: usize,
    /// `note_files` rows whose file is gone, removed with their rows.
    pub files_gone: usize,
    pub inserted: usize,
    pub purged: usize,
    /// Notes whose record changed under the same id: the old row purged, the new one
    /// inserted. Counted here only, not in `inserted` or `purged`.
    pub rewritten: usize,
    /// False: `<base>/memory` could not be listed, and nothing was changed.
    pub listed: bool,
}

/// Every 5 s (called by `indexer::tick`). With no connection, or a mode that is not
/// `Ready`, nothing.
pub fn pass(ctx: &mut IndexerCtx) -> Result<()> {
    pass_counted(ctx).map(|_| ())
}

/// [`pass`], and what it did.
pub fn pass_counted(ctx: &mut IndexerCtx) -> Result<PassReport> {
    let mut report = PassReport::default();
    if ctx.conn.is_none() || !ready(ctx) {
        return Ok(report);
    }
    let dir = ctx.cfg.base.join("memory");
    // A folder that cannot be listed (a mount that is away, no permission, not there)
    // changes nothing: no file is read and no row is taken as gone.
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Ok(report);
    };
    let mut names: Vec<String> = Vec::new();
    let mut listed = true;
    for entry in entries {
        match entry {
            Ok(entry) => {
                if let Some(name) = entry.file_name().to_str().filter(|n| key_of(n).is_some()) {
                    names.push(name.to_string());
                }
            }
            // The listing broke off: the files seen are read, nothing is taken as gone.
            Err(_) => listed = false,
        }
    }
    names.sort();
    names.dedup();

    for name in &names {
        // Rule 2 of wave 4: the mode is read between two files.
        if !ready(ctx) {
            return Ok(report);
        }
        let Some(conn) = ctx.conn.as_ref() else { return Ok(report) };
        let now = (ctx.clock)();
        one_file(conn, ctx.signer.as_ref(), &dir, name, now, &mut report)?;
    }

    // Step 4: a notes file that is gone, only when the folder was listed without error.
    if listed {
        if !ready(ctx) {
            return Ok(report);
        }
        let Some(conn) = ctx.conn.as_ref() else { return Ok(report) };
        gone_files(conn, &dir, &names, &mut report)?;
        report.listed = true;
    }
    Ok(report)
}

/// `Job::Forgotten` with `kind = "note"` (called by `indexer::dispatch`): the row with that
/// id leaves when its origin is `note_file` and its `project_key` is
/// `project_dir(project_key)`. No tombstone; the event's key is used as it is (2.1).
pub fn on_forgotten(ctx: &mut IndexerCtx, id: &str, project_key: &str) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    let scope_dir = project::project_dir(project_key);
    const IS_NOTE: &str = "SELECT 1 FROM memory_ext e JOIN memories m ON m.id = e.mem_id
                            WHERE e.mem_id = ?1 AND e.origin_kind = 'note_file' AND m.project_key = ?2";
    let found = |conn: &Connection| -> Result<bool> {
        Ok(conn.query_row(IS_NOTE, params![id, scope_dir], |_| Ok(())).optional()?.is_some())
    };
    // Looked at first without the write lock: most ids of this event are not mirrored yet.
    if !found(conn)? {
        return Ok(());
    }
    brain_v2::begin_write(conn)?;
    let work = || -> Result<()> {
        if found(conn)? && store::purge_many(conn, &[id.to_string()], false, true)? > 0 {
            index_archives::mark_purged(conn)?;
        }
        conn.execute_batch("COMMIT")?;
        Ok(())
    };
    work().inspect_err(|_| rollback(conn))
}

/// Core's test of a note id (`valid_import_id`): 1 to 128 bytes, no white space, no
/// control character.
pub fn valid_note_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= 128 && !id.chars().any(|c| c.is_whitespace() || c.is_control())
}

fn ready(ctx: &IndexerCtx) -> bool {
    ctx.mode.load(Ordering::SeqCst) == Mode::Ready as u8
}

fn rollback(conn: &Connection) {
    if !conn.is_autocommit() {
        let _ = conn.execute_batch("ROLLBACK");
    }
}

/// The project key in `project-p[0-9a-f]{16}.jsonl`.
fn key_of(name: &str) -> Option<&str> {
    let key = name.strip_prefix("project-")?.strip_suffix(".jsonl")?;
    let hex = key.strip_prefix('p')?;
    (hex.len() == 16 && hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))).then_some(key)
}

fn mtime_ns(meta: &std::fs::Metadata) -> i64 {
    meta.mtime().saturating_mul(1_000_000_000).saturating_add(meta.mtime_nsec())
}

struct Stored {
    size: i64,
    mtime_ns: i64,
    ok: bool,
    reason: Option<String>,
    bad_lines: Option<String>,
}

fn stored_row(conn: &Connection, path: &str) -> Result<Option<Stored>> {
    Ok(conn
        .query_row("SELECT size, mtime_ns, ok, reason, bad_lines FROM note_files WHERE path = ?1", [path], |r| {
            Ok(Stored {
                size: r.get(0)?,
                mtime_ns: r.get(1)?,
                ok: r.get::<_, i64>(2)? == 1,
                reason: r.get(3)?,
                bad_lines: r.get(4)?,
            })
        })
        .optional()?)
}

fn one_file(
    conn: &Connection,
    signer: Option<&axel_memkoshi::security::MemorySigner>,
    dir: &Path,
    name: &str,
    now: i64,
    report: &mut PassReport,
) -> Result<()> {
    let Some(key) = key_of(name) else { return Ok(()) };
    let path = dir.join(name);
    let path_text = path.to_string_lossy().into_owned();
    // SPEC: 2318-2319 "regular files only": a listed name that cannot be `lstat`ed, or is
    // not a regular file (a link, a folder), is left alone, and so are its rows: it was
    // listed, so it is not "gone", and it was not read, so it says nothing.
    let Ok(meta) = std::fs::symlink_metadata(&path) else { return Ok(()) };
    if !meta.file_type().is_file() {
        return Ok(());
    }
    let seen = (i64::try_from(meta.len()).unwrap_or(i64::MAX), mtime_ns(&meta));
    let stored = stored_row(conn, &path_text)?;
    // RULING: wave 4, R-14 item 1: only a file that was read is skipped as unchanged. A file
    // that was refused (`ok = 0`) is tried again at every pass: a failed open says nothing
    // about what the file holds, and a tombstone line may be waiting in it.
    let same = stored.as_ref().is_some_and(|s| (s.size, s.mtime_ns) == seen);
    if same && stored.as_ref().is_some_and(|s| s.ok) {
        report.files_unchanged += 1;
        return Ok(());
    }

    let file = match notes::read_file(&path, key) {
        Ok(file) => file,
        Err(NotesError::Changed) => {
            report.files_changed += 1;
            return Ok(());
        }
        Err(refused) => {
            // `ok = 0` with the reason and the size and time seen; the mirror, the live
            // count and the bad-line set stay exactly as they were. A file that stays
            // refused for the same reason writes nothing.
            let reason = refused.to_string();
            report.files_refused += 1;
            if same && stored.as_ref().is_some_and(|s| !s.ok && s.reason.as_deref() == Some(reason.as_str())) {
                return Ok(());
            }
            brain_v2::begin_write(conn)?;
            let work = || -> Result<()> {
                conn.execute(
                    "INSERT INTO note_files(path, project_key, size, mtime_ns, ok, reason, live_count, scanned_ms)
                     VALUES (?1, ?2, ?3, ?4, 0, ?5, 0, ?6)
                     ON CONFLICT(path) DO UPDATE SET size = excluded.size, mtime_ns = excluded.mtime_ns,
                            ok = 0, reason = excluded.reason, scanned_ms = excluded.scanned_ms",
                    params![path_text, key, seen.0, seen.1, reason, now],
                )?;
                conn.execute_batch("COMMIT")?;
                Ok(())
            };
            work().inspect_err(|_| rollback(conn))?;
            return Ok(());
        }
    };

    brain_v2::begin_write(conn)?;
    let done = mirror(conn, signer, &path_text, name, key, &file, stored.as_ref(), now)
        .and_then(|done| {
            conn.execute_batch("COMMIT")?;
            Ok(done)
        })
        .inspect_err(|_| rollback(conn))?;
    report.files_read += 1;
    report.inserted += done.inserted;
    report.purged += done.purged;
    report.rewritten += done.rewritten;
    Ok(())
}

/// What [`mirror`] did to the rows of one file.
#[derive(Default)]
struct Mirrored {
    inserted: usize,
    purged: usize,
    rewritten: usize,
}

/// The row an id already has, whatever made it.
struct Existing {
    origin_kind: Option<String>,
    origin_ref: Option<String>,
    hidden_ms: Option<i64>,
    hidden_by: Option<String>,
    title: String,
    content: String,
    sensitivity: String,
    tags: String,
    expires_at: Option<String>,
}

fn existing_row(conn: &Connection, id: &str) -> Result<Option<Existing>> {
    Ok(conn
        .query_row(
            "SELECT e.origin_kind, e.origin_ref, e.hidden_ms, e.hidden_by,
                    m.title, m.content, m.sensitivity, m.tags, m.expires_at
               FROM memories m LEFT JOIN memory_ext e ON e.mem_id = m.id WHERE m.id = ?1",
            [id],
            |r| {
                Ok(Existing {
                    origin_kind: r.get(0)?,
                    origin_ref: r.get(1)?,
                    hidden_ms: r.get(2)?,
                    hidden_by: r.get(3)?,
                    title: r.get(4)?,
                    content: r.get(5)?,
                    sensitivity: r.get(6)?,
                    tags: r.get(7)?,
                    expires_at: r.get(8)?,
                })
            },
        )
        .optional()?)
}

/// Is the stored row what `insert_row` would store for `row` today? The text as the row
/// writer stores it (title and content through `screen::redact`), the sensitivity, the
/// expiry and the tags. A stored value that cannot be read back is a difference.
fn same_as_stored(old: &Existing, row: &NewRow) -> bool {
    let expires_ms = match old.expires_at.as_deref() {
        None => Some(None),
        Some(text) => DateTime::parse_from_rfc3339(text).ok().map(|t| Some(t.timestamp_millis())),
    };
    old.content == crate::screen::redact(&row.content).text
        && old.title == crate::screen::redact(&row.title).text
        && old.sensitivity == row.sensitivity.as_str()
        && serde_json::from_str::<Vec<String>>(&old.tags).is_ok_and(|tags| tags == row.tags)
        && expires_ms == Some(row.expires_ms)
}

/// Step 3, inside the caller's transaction.
#[allow(clippy::too_many_arguments)]
fn mirror(
    conn: &Connection,
    signer: Option<&axel_memkoshi::security::MemorySigner>,
    path_text: &str,
    name: &str,
    key: &str,
    file: &NotesFile,
    stored: Option<&Stored>,
    now: i64,
) -> Result<Mirrored> {
    let scope_dir = project::project_dir(key);
    let mirrored = rows_of(conn, name)?;
    let live: HashSet<&str> = file.live.iter().map(|n| n.id.as_str()).collect();
    let tombstoned: HashSet<&str> = file.tombstoned.iter().map(String::as_str).collect();

    // Deletion by a tombstone line: always, whatever else in the file was skipped.
    let mut leave: Vec<String> =
        mirrored.iter().filter(|id| tombstoned.contains(id.as_str())).cloned().collect();

    // Deletion by absence: held back for one pass when the bad lines changed. A file with
    // no `note_files` row yet has no stored set (NULL).
    let mut hashes: Vec<&str> = file.bad.iter().map(|b| b.hash.as_str()).collect();
    hashes.sort_unstable();
    let now_bad: Option<String> = (!hashes.is_empty()).then(|| hashes.join(","));
    let bad_same = stored.and_then(|s| s.bad_lines.as_deref()) == now_bad.as_deref();
    if bad_same {
        leave.extend(
            mirrored
                .iter()
                .filter(|id| !live.contains(id.as_str()) && !tombstoned.contains(id.as_str()))
                .cloned(),
        );
    }
    let purged = store::purge_many(conn, &leave, false, true)?;

    // Inserts, and notes whose record changed under the same id.
    let mut inserted = 0usize;
    let mut rewritten = 0usize;
    let mut ids_skipped = 0usize;
    for note in &file.live {
        if !valid_note_id(&note.id) {
            ids_skipped += 1;
            continue;
        }
        let mut row = new_row(note, &scope_dir, name);
        let mut rewrote = false;
        if let Some(old) = existing_row(conn, &note.id)? {
            // A row of another origin or of another notes file is not mirrored and not
            // touched.
            if old.origin_kind.as_deref() != Some("note_file") || old.origin_ref.as_deref() != Some(name) {
                ids_skipped += 1;
                continue;
            }
            if same_as_stored(&old, &row) {
                continue;
            }
            // RULING: wave 4, R-14 item 2: the record changed under the same id (the owner
            // edited the note, or made it secret). The old row is forgotten, with its word
            // rows and its vector, and the new one is written in this transaction. A hide
            // someone set on the row is carried over: an edit never unhides.
            store::purge_many(conn, std::slice::from_ref(&note.id), false, true)?;
            rewritten += 1;
            rewrote = true;
            row.hidden = old.hidden_ms.map(|ms| (ms, old.hidden_by.unwrap_or_default()));
        }
        // SPEC: 2346 "`created_ms` = `timestamp_ms`": a time the row writer cannot store
        // (the reader saturates core's u64 at i64::MAX) would fail the whole transaction
        // at every pass, and with it every forget in this file. Such a record is skipped
        // and counted with the skipped ids.
        if DateTime::from_timestamp_millis(note.timestamp_ms).is_none() {
            ids_skipped += 1;
            continue;
        }
        match store::insert_row(conn, signer, &row)? {
            Inserted::New { .. } if rewrote => {}
            Inserted::New { .. } => inserted += 1,
            // SPEC: 2356-2361 names `Exists` only. An id with a `memory_tombstones` row (an
            // id Axel itself forgot for good) is not mirrored either, and is counted.
            Inserted::Exists | Inserted::Tombstoned => ids_skipped += 1,
        }
    }

    if purged > 0 || rewritten > 0 {
        index_archives::mark_purged(conn)?;
    }

    // When the bad set changed: it is stored with the time, and `mtime_ns` is written as 0
    // so that the next pass reads the file again whatever its time, finds the set
    // unchanged, and purges by absence.
    let reason = (ids_skipped > 0).then(|| format!("ids_skipped: {ids_skipped}"));
    let mtime = if bad_same { file.mtime_ns } else { 0 };
    let size = i64::try_from(file.size).unwrap_or(i64::MAX);
    let live_count = i64::try_from(file.live.len()).unwrap_or(i64::MAX);
    conn.execute(
        "INSERT INTO note_files(path, project_key, size, mtime_ns, ok, reason, live_count, scanned_ms,
                                bad_lines, bad_lines_changed_ms)
         VALUES (?1, ?2, ?3, ?4, 1, ?5, ?6, ?7, ?8, ?9)
         ON CONFLICT(path) DO UPDATE SET project_key = excluded.project_key, size = excluded.size,
                mtime_ns = excluded.mtime_ns, ok = 1, reason = excluded.reason,
                live_count = excluded.live_count, scanned_ms = excluded.scanned_ms,
                bad_lines = excluded.bad_lines,
                bad_lines_changed_ms = COALESCE(excluded.bad_lines_changed_ms, note_files.bad_lines_changed_ms)",
        params![path_text, key, size, mtime, reason, live_count, now, now_bad, (!bad_same).then_some(now)],
    )?;
    // `projects`: display only.
    conn.execute(
        "INSERT INTO projects(scope_dir, project_key, root_hint, first_seen_ms, last_seen_ms)
         VALUES (?1, ?2, NULL, ?3, ?3)
         ON CONFLICT(scope_dir) DO UPDATE SET project_key = COALESCE(projects.project_key, excluded.project_key),
                last_seen_ms = MAX(projects.last_seen_ms, excluded.last_seen_ms)",
        params![scope_dir, key, now],
    )?;
    Ok(Mirrored { inserted, purged, rewritten })
}

/// The ids this file has mirrored: `origin_kind = 'note_file'`, `origin_ref` = its name.
fn rows_of(conn: &Connection, name: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT mem_id FROM memory_ext WHERE origin_kind = 'note_file' AND origin_ref = ?1 ORDER BY mem_id",
    )?;
    let rows = stmt.query_map([name], |r| r.get::<_, String>(0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

fn new_row<'a>(note: &NoteRecord, scope_dir: &'a str, name: &'a str) -> NewRow<'a> {
    let mut tags = note.tags.clone();
    let sensitivity = match note.sensitivity {
        NoteSensitivity::Normal => Sensitivity::Normal,
        NoteSensitivity::Sensitive => {
            if !tags.iter().any(|t| t == "sensitive") {
                tags.push("sensitive".to_string());
            }
            Sensitivity::Normal
        }
        NoteSensitivity::Secret => Sensitivity::Secret,
    };
    let title = match sensitivity {
        Sensitivity::Secret => SECRET_TITLE.to_string(),
        Sensitivity::Normal => {
            note.content.split_whitespace().take(TITLE_WORDS).collect::<Vec<_>>().join(" ")
        }
    };
    // SPEC: 2348-2349: an expiry past what a time can hold is no expiry.
    let expires_ms = note
        .max_age_days
        .and_then(|days| note.timestamp_ms.checked_add(i64::from(days).checked_mul(DAY_MS)?))
        .filter(|ms| DateTime::from_timestamp_millis(*ms).is_some());
    NewRow {
        id: note.id.clone(),
        class: "note",
        project_key: scope_dir,
        title,
        abstract_text: String::new(),
        content: note.content.clone(),
        tags,
        created_ms: note.timestamp_ms,
        time_source: "exact",
        sensitivity,
        provenance: format!("synaps:note:{name}"),
        expires_ms,
        // SPEC: 2344-2350 names no importance; 0.5, as the archive and capture rows (2050, 3446).
        importance: 0.5,
        origin_kind: "note_file",
        origin_ref: name,
        unit_pos: None,
        session_id: None,
        logical: None,
        links: RecordLinks::default(),
        turn_key: None,
        hidden: None,
    }
}

/// Step 4. `names` are the file names the listing gave, whatever kind of entry they are.
fn gone_files(conn: &Connection, dir: &Path, names: &[String], report: &mut PassReport) -> Result<()> {
    let known: Vec<String> = {
        let mut stmt = conn.prepare("SELECT path FROM note_files ORDER BY path")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    for path_text in known {
        let path = Path::new(&path_text);
        // SPEC: 2382-2385 "each `note_files` row whose path was not listed": a row whose
        // path is not a file of this folder was not looked for here, and is left alone.
        if path.parent() != Some(dir) {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        if key_of(name).is_none() || names.iter().any(|n| n == name) {
            continue;
        }
        brain_v2::begin_write(conn)?;
        let work = || -> Result<usize> {
            let purged = store::purge_many(conn, &rows_of(conn, name)?, false, true)?;
            if purged > 0 {
                index_archives::mark_purged(conn)?;
            }
            conn.execute("DELETE FROM note_files WHERE path = ?1", [&path_text])?;
            conn.execute_batch("COMMIT")?;
            Ok(purged)
        };
        report.purged += work().inspect_err(|_| rollback(conn))?;
        report.files_gone += 1;
    }
    Ok(())
}
