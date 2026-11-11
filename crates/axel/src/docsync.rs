//! The owner's note folders (section 2.8). Every 900 s the folders `load_sources` names are
//! compared with their `documents` rows: a changed or new file is read, embedded and written
//! in place (`velocirag::db::upsert_document`, same row id), a row whose file is gone is
//! hidden as `missing_source`, and a hidden row whose file is back is shown again. Nothing
//! is ever deleted here, and nothing outside the named folders is read: the walk follows no
//! link, and a file is read only through a handle that is the regular file the walk saw.
//! The files are stored whole, as the command line stores them; they are screened when shown.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};

use crate::brain_v2;
use crate::consolidate::{load_sources, SourceDir};
use crate::error::Result;
use crate::ext::types::{IndexerCtx, Mode};
use crate::vindex::{VecDelta, VecItem, VecKey, VecScope};

/// Files embedded per tick (B-2e).
const FILES_PER_TICK: usize = 8;
/// The deadline of one file's `embed_wait`.
const EMBED_DEADLINE: Duration = Duration::from_secs(5);
/// Smaller files are not indexed, as today.
const MIN_BYTES: u64 = 50;
/// Larger files are never opened (wave 4 ruling R-22.1): only the first 256 tokens are
/// embedded, and a huge file would be held whole and written in one transaction.
const MAX_BYTES: u64 = 2 * 1024 * 1024;
/// The one mark this file writes and clears.
const MISSING: &str = "missing_source";

/// What one sync did. `waiting` is the number of files that need a vector and did not get
/// one this tick (no model, `embed_wait` gave nothing, or past the 8 of this tick).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SyncReport {
    pub indexed: usize,
    pub waiting: usize,
    pub hidden: usize,
    pub restored: usize,
    /// Files that could not be read, or whose write failed; their rows are untouched.
    pub skipped: usize,
    /// Files whose time changed and whose text did not: only `source_mtime_ms` was set.
    pub retimed: usize,
    /// New files that took over a hidden row of the same text (a rename or a move).
    pub moved: usize,
}

/// A file the walk found that needs its row written.
struct ToIndex {
    path: PathBuf,
    doc_id: String,
    /// Device and inode the walk saw; the handle that is read must be the same file.
    file_id: Option<(u64, u64)>,
    /// The row under `doc_id`, when there is one.
    row: Option<i64>,
    /// No row under `doc_id` and no visible row under the path: it may take over a hidden
    /// row of its source (`doc_id` prefix `<source>::`, path under `root`).
    fresh: Option<Fresh>,
}

struct Fresh {
    /// The source's folder, ending in `/`.
    root: String,
    /// `<source name>::`.
    ids: String,
}

/// The cheap half: what the folders say, before anything is read or written.
struct Plan {
    index: Vec<ToIndex>,
    /// Visible rows whose file is not there (the name is not found).
    gone: Vec<i64>,
    /// Rows hidden as `missing_source` whose file the walk found again.
    back: Vec<i64>,
    /// Files over `MAX_BYTES`: not opened, their rows untouched.
    too_big: usize,
}

/// Every 900 s (called by `indexer::tick`).
pub fn tick(ctx: &mut IndexerCtx) -> Result<()> {
    let report = sync(ctx)?;
    // What `status` shows as documents waiting: the files this sync left without a vector.
    ctx.status.docs_waiting.store(report.waiting as u64, Ordering::Relaxed);
    Ok(())
}

/// One whole sync: plan, hide, restore, then at most 8 files embedded and written, each in
/// its own transaction. Returns `Ok` with nothing done when there is no brain, and stops
/// between two items as soon as the mode is not `Ready`.
pub fn sync(ctx: &mut IndexerCtx) -> Result<SyncReport> {
    let mut report = SyncReport::default();
    if ctx.conn.is_none() || !ready(ctx) {
        return Ok(report);
    }
    // SPEC: 2676: sources the configuration cannot give (no file and no HOME for the default
    // six) are no folders at all: nothing is planned, so nothing is hidden.
    let Ok(sources) = load_sources(ctx.cfg.sources.as_deref()) else { return Ok(report) };
    let plan = match ctx.conn.as_ref() {
        Some(conn) => plan(conn, &sources)?,
        None => return Ok(report),
    };
    let model_loaded = ctx.status.model_loaded.load(Ordering::Relaxed);
    report.skipped += plan.too_big;

    for id in plan.gone {
        if !ready(ctx) {
            return Ok(report);
        }
        let now = (ctx.clock)();
        match write(ctx, |conn| set_hidden(conn, id, Some(now))) {
            Some(true) => {
                report.hidden += 1;
                let _ = ctx.vec_tx.send(VecDelta::Remove(vec![VecKey::Doc(id)]));
            }
            Some(false) => {}
            None => report.skipped += 1,
        }
    }

    for id in plan.back {
        if !ready(ctx) {
            return Ok(report);
        }
        match write(ctx, |conn| set_hidden(conn, id, None)) {
            Some(true) => {
                report.restored += 1;
                // RULING: R-22.3 (SH-B3): the vector is stored; it goes back whatever the model.
                send_stored(ctx, id);
            }
            Some(false) => {}
            None => report.skipped += 1,
        }
    }

    let mut embedded = 0usize;
    for file in &plan.index {
        if !ready(ctx) {
            return Ok(report);
        }
        // RULING: W3-12b.9: with no usable model `embed_wait` is not called at all.
        let no_slot = !model_loaded || embedded >= FILES_PER_TICK;
        // RULING: R-22.2: a file is read without a slot only when its text may spare the
        // embedding: it has a row, or it is new and its source has rows hidden by this sync.
        let movable = match &file.fresh {
            Some(fresh) => ctx.conn.as_ref().is_some_and(|c| has_hidden(c, fresh).unwrap_or(false)),
            None => false,
        };
        if no_slot && file.row.is_none() && !movable {
            report.waiting += 1;
            continue;
        }
        let Some((text, mtime_ms)) = read_file(file) else {
            report.skipped += 1;
            continue;
        };
        let path = file.path.to_string_lossy().into_owned();
        // RULING: R-22.2: the same text under a new time: the time only, no embedding, no slot.
        if let Some(id) = file.row {
            let same = ctx.conn.as_ref().is_some_and(|c| same_text(c, id, &text).unwrap_or(false));
            if same {
                match write(ctx, |conn| retime(conn, id, &text, mtime_ms)) {
                    Some(true) => report.retimed += 1,
                    Some(false) => {}
                    None => report.skipped += 1,
                }
                continue;
            }
        }
        // RULING: R-22.2: a new file with the text of a row this sync hid in its source takes
        // that row over (a rename or a move): same row id, no embedding, its vector sent back.
        if let (Some(fresh), true) = (&file.fresh, movable) {
            match write(ctx, |conn| take_over(conn, fresh, &file.doc_id, &path, &text, mtime_ms)) {
                Some(Some(id)) => {
                    report.moved += 1;
                    send_stored(ctx, id);
                    continue;
                }
                Some(None) => {}
                None => {
                    report.skipped += 1;
                    continue;
                }
            }
        }
        if no_slot {
            report.waiting += 1;
            continue;
        }
        embedded += 1;
        let vector = ctx
            .embed
            .embed_wait(vec![text.clone()], Instant::now() + EMBED_DEADLINE)
            .and_then(|mut v| v.pop());
        let Some(vector) = vector else {
            report.waiting += 1;
            continue;
        };
        // RULING: W3-12b.8: the mode is read again after the wait, before the write.
        if !ready(ctx) {
            return Ok(report);
        }
        let written = write(ctx, |conn| {
            let id = velocirag::db::upsert_document(conn, &file.doc_id, &text, &serde_json::json!({}), &vector, Some(&path))?;
            conn.execute(
                "UPDATE documents SET source_mtime_ms = ?2,
                        hidden = CASE WHEN hidden = ?3 THEN NULL ELSE hidden END
                  WHERE id = ?1",
                params![id, mtime_ms, MISSING],
            )?;
            let hidden: Option<String> = conn.query_row("SELECT hidden FROM documents WHERE id = ?1", [id], |r| r.get(0))?;
            Ok((id, hidden.is_none()))
        });
        match written {
            Some((id, visible)) => {
                report.indexed += 1;
                if visible {
                    let item = VecItem { key: VecKey::Doc(id), scope: VecScope::Documents, v: vector };
                    let _ = ctx.vec_tx.send(VecDelta::Upsert(vec![item]));
                }
            }
            None => report.skipped += 1,
        }
    }
    Ok(report)
}

fn ready(ctx: &IndexerCtx) -> bool {
    ctx.mode.load(Ordering::SeqCst) == Mode::Ready as u8 && !ctx.stop.load(Ordering::SeqCst)
}

/// One transaction: stand aside, `begin_write`, `f`, `COMMIT`. Any error rolls back, is
/// counted in `write_errors` and may degrade the process; the caller goes on with the next
/// item (`None`), and the next mode check stops it if the process degraded.
fn write<T>(ctx: &IndexerCtx, f: impl FnOnce(&Connection) -> Result<T>) -> Option<T> {
    let conn = ctx.conn.as_ref()?;
    brain_v2::stand_aside(&ctx.writing);
    let done = brain_v2::begin_write(conn).and_then(|()| {
        let value = f(conn)?;
        conn.execute_batch("COMMIT")?;
        Ok(value)
    });
    match done {
        Ok(value) => Some(value),
        Err(e) => {
            if !conn.is_autocommit() {
                let _ = conn.execute_batch("ROLLBACK");
            }
            ctx.status.write_errors.fetch_add(1, Ordering::Relaxed);
            brain_v2::degrade_on(&e, &ctx.status, &ctx.mode);
            None
        }
    }
}

/// Hides a visible row as `missing_source` (`Some(now)`, in ms), or clears that mark
/// (`None`). Any other mark is left alone. True when the row changed.
// SPEC: R-22.2 takes "the row hidden most recently" and the schema has no time for a
// hide: it is kept in the row's metadata as `missing_since_ms` while the mark is set.
fn set_hidden(conn: &Connection, id: i64, hide: Option<i64>) -> Result<bool> {
    let n = match hide {
        Some(now) => conn.execute(
            "UPDATE documents SET hidden = ?2,
                    metadata = CASE WHEN json_valid(metadata) THEN json_set(metadata, '$.missing_since_ms', ?3) ELSE metadata END
              WHERE id = ?1 AND hidden IS NULL",
            params![id, MISSING, now],
        )?,
        None => conn.execute(
            &format!("UPDATE documents SET hidden = NULL, metadata = {UNSTAMP} WHERE id = ?1 AND hidden = ?2"),
            params![id, MISSING],
        )?,
    };
    Ok(n > 0)
}

/// `metadata` without the time of the hide.
const UNSTAMP: &str =
    "CASE WHEN json_valid(metadata) THEN json_remove(metadata, '$.missing_since_ms') ELSE metadata END";

/// The row's stored text is this text.
fn same_text(conn: &Connection, id: i64, text: &str) -> Result<bool> {
    let same: Option<bool> = conn
        .query_row("SELECT content = ?2 FROM documents WHERE id = ?1", params![id, text], |r| r.get(0))
        .optional()?;
    Ok(same == Some(true))
}

/// Sets the time of a row whose text is still `text`. A row someone else hid is not touched.
fn retime(conn: &Connection, id: i64, text: &str, mtime_ms: i64) -> Result<bool> {
    let n = conn.execute(
        "UPDATE documents SET source_mtime_ms = ?2
          WHERE id = ?1 AND content = ?3 AND (hidden IS NULL OR hidden = ?4)",
        params![id, mtime_ms, text, MISSING],
    )?;
    Ok(n > 0)
}

/// The rows this sync hid in a source, with the given text, most recently hidden first.
const HIDDEN_SQL: &str = "FROM documents
    WHERE hidden = 'missing_source' AND file_path IS NOT NULL
      AND substr(file_path, 1, length(?1)) = ?1 AND substr(doc_id, 1, length(?2)) = ?2
      AND COALESCE(json_extract(metadata, '$.type'), '') <> 'memory'";

fn has_hidden(conn: &Connection, fresh: &Fresh) -> Result<bool> {
    let n: i64 = conn.query_row(
        &format!("SELECT EXISTS(SELECT 1 {HIDDEN_SQL})"),
        params![fresh.root, fresh.ids],
        |r| r.get(0),
    )?;
    Ok(n == 1)
}

/// A new file takes over the most recently hidden row of its source with the same text,
/// whose file is still not found: new `doc_id`, new path, the time, the mark cleared. The
/// row id, the embedding and every other column are kept. `None` when no row qualifies.
fn take_over(conn: &Connection, fresh: &Fresh, doc_id: &str, path: &str, text: &str, mtime_ms: i64) -> Result<Option<i64>> {
    // Under the write lock again: someone may have written this doc_id since the plan.
    let taken: bool = conn.query_row("SELECT EXISTS(SELECT 1 FROM documents WHERE doc_id = ?1)", [doc_id], |r| r.get(0))?;
    if taken {
        return Ok(None);
    }
    let mut stmt = conn.prepare(&format!(
        "SELECT id, doc_id, file_path {HIDDEN_SQL} AND content = ?3
          ORDER BY COALESCE(json_extract(metadata, '$.missing_since_ms'), -1) DESC, id DESC"
    ))?;
    let rows = stmt
        .query_map(params![fresh.root, fresh.ids, text], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    // A row whose file is back (its restore failed) is that file's, not this one's.
    let Some((id, old_id, _)) = rows.into_iter().find(|(_, _, p)| is_gone(Path::new(p))) else {
        return Ok(None);
    };
    conn.execute(
        &format!(
            "UPDATE documents SET doc_id = ?2, file_path = ?3, source_mtime_ms = ?4, hidden = NULL,
                    metadata = {UNSTAMP}
              WHERE id = ?1 AND hidden = ?5"
        ),
        params![id, doc_id, path, mtime_ms, MISSING],
    )?;
    conn.execute(
        "UPDATE chunks_fts SET doc_id = ?2, file_path = ?3 WHERE doc_id = ?1",
        params![old_id, doc_id, path],
    )?;
    Ok(Some(id))
}

/// A restored row's stored vector goes back to the front thread.
fn send_stored(ctx: &IndexerCtx, id: i64) {
    let Some(conn) = ctx.conn.as_ref() else { return };
    let blob: Option<Vec<u8>> = conn
        .query_row("SELECT embedding FROM documents WHERE id = ?1", [id], |r| r.get(0))
        .optional()
        .ok()
        .flatten();
    if let Some(blob) = blob {
        let v: Vec<f32> = blob.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        if v.len() == velocirag::EMBEDDING_DIM {
            let item = VecItem { key: VecKey::Doc(id), scope: VecScope::Documents, v };
            let _ = ctx.vec_tx.send(VecDelta::Upsert(vec![item]));
        }
    }
}

/// Walks every source and compares it with the brain. Reads no file's content.
fn plan(conn: &Connection, sources: &[SourceDir]) -> Result<Plan> {
    // SPEC: 2649-2654 do not say what a source that is missing, unreadable or not a folder
    // gives. It gives nothing: its rows are neither hidden nor restored, so a folder whose
    // mount point is gone loses nothing and shows nothing new.
    let mut roots: Vec<(PathBuf, &SourceDir)> = Vec::new();
    for source in sources {
        let Ok(root) = std::fs::canonicalize(&source.path) else { continue };
        if !root.is_dir() || roots.iter().any(|(r, _)| *r == root) {
            continue;
        }
        roots.push((root, source));
    }
    let all_roots: HashSet<PathBuf> = roots.iter().map(|(r, _)| r.clone()).collect();

    let mut seen: HashSet<String> = HashSet::new();
    let mut index: BTreeMap<String, ToIndex> = BTreeMap::new();
    let mut too_big = 0usize;
    for (root, source) in &roots {
        let walk = walkdir::WalkDir::new(root).follow_links(false).into_iter().filter_entry(|e| {
            // A folder that is another source's root is walked as that source only.
            e.depth() == 0 || !(e.file_type().is_dir() && all_roots.contains(e.path()))
        });
        // An entry the walk cannot read is skipped; what is under it is not "gone" for that
        // (absence is decided by the name alone, below).
        for entry in walk.filter_map(|e| e.ok()) {
            // `file_type` is the entry's own: a link is a link, never what it points at.
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            if !path.extension().is_some_and(|ext| ext == "md" || ext == "txt") {
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            let abs = path.to_string_lossy().into_owned();
            seen.insert(abs.clone());
            if meta.len() < MIN_BYTES {
                continue;
            }
            if meta.len() > MAX_BYTES {
                too_big += 1;
                continue;
            }
            let doc_id = doc_id(source, root, path);
            if let Some(need) = needs_index(conn, &doc_id, &abs, mtime_ms(&meta))? {
                let fresh = need.fresh.then(|| {
                    let mut root = root.to_string_lossy().into_owned();
                    if !root.ends_with('/') {
                        root.push('/');
                    }
                    Fresh { root, ids: format!("{}::", source.name) }
                });
                let file = ToIndex { path: path.to_path_buf(), doc_id, file_id: file_id(&meta), row: need.row, fresh };
                index.insert(abs, file);
            }
        }
    }

    let mut gone = BTreeSet::new();
    let mut back = BTreeSet::new();
    let mut stmt = conn.prepare(
        "SELECT id, file_path, hidden FROM documents
          WHERE file_path IS NOT NULL AND substr(file_path, 1, length(?1)) = ?1
            AND (hidden IS NULL OR hidden = ?2)
            AND COALESCE(json_extract(metadata, '$.type'), '') <> 'memory'",
    )?;
    for (root, _) in &roots {
        let mut prefix = root.to_string_lossy().into_owned();
        if !prefix.ends_with('/') {
            prefix.push('/');
        }
        let rows = stmt
            .query_map(params![prefix, MISSING], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, Option<String>>(2)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for (id, file_path, hidden) in rows {
            if seen.contains(&file_path) {
                if hidden.is_some() {
                    back.insert(id);
                }
            } else if hidden.is_none() && is_gone(Path::new(&file_path)) {
                gone.insert(id);
            }
        }
    }
    Ok(Plan { index: index.into_values().collect(), gone: gone.into_iter().collect(), back: back.into_iter().collect(), too_big })
}

/// Only "not found" says a file is gone. Any other answer (no permission, an I/O error, a
/// link where the file was) keeps the row as it is.
fn is_gone(path: &Path) -> bool {
    matches!(std::fs::symlink_metadata(path), Err(e) if e.kind() == std::io::ErrorKind::NotFound)
}

/// `doc_id` as the command line makes it: `<source name>::<path under the source>`, with
/// `/` as `::` and `.md` and `.txt` taken out.
fn doc_id(source: &SourceDir, root: &Path, path: &Path) -> String {
    let relative = path
        .strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('/', "::")
        .replace(".md", "")
        .replace(".txt", "");
    format!("{}::{}", source.name, relative)
}

/// A file's row: id, mark, `source_mtime_ms`, `indexed_at` in seconds.
type RowState = (i64, Option<String>, Option<i64>, Option<f64>);

/// What the walk knows of a file that needs its row written.
struct Need {
    row: Option<i64>,
    fresh: bool,
}

/// The file is newer than its row, or has no row (`None`: nothing to do).
fn needs_index(conn: &Connection, doc_id: &str, abs: &str, mtime_ms: i64) -> Result<Option<Need>> {
    let row: Option<RowState> = conn
        .query_row(
            "SELECT id, hidden, source_mtime_ms, CAST(strftime('%s', indexed_at) AS REAL) FROM documents WHERE doc_id = ?1",
            [doc_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .optional()?;
    let has_row = |id: i64, yes: bool| yes.then_some(Need { row: Some(id), fresh: false });
    match row {
        // A row someone else hid (not this file's mark) is left as it is.
        Some((_, Some(mark), _, _)) if mark != MISSING => Ok(None),
        // RULING: R-22.3 (SH-B4): any other time, an older one too (a file restored from a
        // backup keeps its old time).
        Some((id, _, Some(stored), _)) => Ok(has_row(id, mtime_ms != stored)),
        // As today: `indexed_at` has whole seconds, so half a second of slack.
        Some((id, _, None, indexed)) => Ok(has_row(id, mtime_ms as f64 / 1000.0 > indexed.unwrap_or(0.0) + 0.5)),
        None => {
            // SPEC: 2652-2653 compare with "documents.source_mtime_ms": the row of this
            // doc_id. A file the command line indexed under another doc_id has a visible
            // row with its path; when that row is as new as the file, no second row is made.
            let newest: Option<f64> = conn.query_row(
                "SELECT MAX(COALESCE(source_mtime_ms / 1000.0, CAST(strftime('%s', indexed_at) AS REAL) + 0.5))
                   FROM documents WHERE file_path = ?1 AND hidden IS NULL",
                [abs],
                |r| r.get(0),
            )?;
            Ok(match newest {
                Some(t) => (mtime_ms as f64 / 1000.0 > t).then_some(Need { row: None, fresh: false }),
                None => Some(Need { row: None, fresh: true }),
            })
        }
    }
}

/// The file's content and modification time, read through a handle that is the regular
/// file the walk saw (no link is followed, a FIFO does not block). `None` skips the file.
fn read_file(file: &ToIndex) -> Option<(String, i64)> {
    let handle = crate::fs_open::open_no_wait(&file.path).ok()?;
    let meta = handle.metadata().ok()?;
    if !meta.file_type().is_file() || file_id(&meta) != file.file_id || !(MIN_BYTES..=MAX_BYTES).contains(&meta.len()) {
        return None;
    }
    let mut bytes = Vec::with_capacity(meta.len() as usize);
    (&handle).take(meta.len()).read_to_end(&mut bytes).ok()?;
    let text = String::from_utf8(bytes).ok()?;
    if (text.len() as u64) < MIN_BYTES {
        return None;
    }
    Some((text, mtime_ms(&meta)))
}

fn mtime_ms(meta: &std::fs::Metadata) -> i64 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_millis() as i64)
}

#[cfg(unix)]
fn file_id(meta: &std::fs::Metadata) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    Some((meta.dev(), meta.ino()))
}

#[cfg(not(unix))]
fn file_id(_meta: &std::fs::Metadata) -> Option<(u64, u64)> {
    None
}
