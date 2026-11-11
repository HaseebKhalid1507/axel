//! Opening the brain with retry, the base guard, the first vector load, the embedding
//! backlog, the status row. All of it runs on the indexer thread.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rusqlite::{params, Connection, ErrorCode, OpenFlags, OptionalExtension};
use serde_json::json;

use crate::brain_v2::{self, OpenOpts};
use crate::embed::{BatchOut, BulkRow, EmbedService};
use crate::error::{AxelError, Result};
use crate::ext::config::{base_config_value, ExtConfig};
use crate::ext::server::{degrade, log_line, recover, JOB_QUEUE};
use crate::ext::types::{Boot, DegradedReason, IndexerCtx, Job, Mode};
use crate::r8::{DEFAULT_EMBEDDER, EMBEDDING_DIM};
use crate::vindex::{VecDelta, VecItem, VecKey, VecScope};

/// The indexer retries `open_v2` this often.
const OPEN_RETRY_MS: u64 = 60_000;
/// Rows given to the embed thread at a time, and vectors written per transaction.
const BATCH: usize = 64;
/// The project directory of global rows.
const GLOBAL_DIR: &str = "p0000000000000000";

const DOCS_WHERE: &str =
    "hidden IS NULL AND COALESCE(json_extract(metadata, '$.type'), '') <> 'memory'";

/// The one constructor of an `IndexerCtx` for tests. It hands back the other ends of the
/// channels it makes (B3-3a). The brain is opened (created, migrated) before it returns, as
/// the indexer thread does at the top of its loop: look at `ctx.mode` for how that went.
pub fn open_ctx(cfg: ExtConfig, boot: Boot) -> (IndexerCtx, Receiver<Job>, Receiver<VecDelta>) {
    let (jobs, jobs_rx) = mpsc::sync_channel::<Job>(JOB_QUEUE);
    let (vec_tx, vec_rx) = mpsc::channel::<VecDelta>();
    let embed = EmbedService::start(boot.embedder, jobs.clone(), boot.test.clone());
    let mut ctx = IndexerCtx {
        cfg,
        conn: None,
        signer: None,
        embed,
        vec_tx,
        jobs,
        status: boot.status,
        mode: Arc::new(AtomicU8::new(Mode::Starting as u8)),
        last_run: Default::default(),
        open_tried: None,
        vectors_sent: None,
        stop: Arc::new(AtomicBool::new(false)),
        brain_file: None,
        clock: boot.clock,
        test: boot.test,
        writing: Arc::new(AtomicBool::new(false)),
    };
    ensure_open(&mut ctx);
    (ctx, jobs_rx, vec_rx)
}

fn mode_of(ctx: &IndexerCtx) -> Option<Mode> {
    Mode::from_u8(ctx.mode.load(Ordering::SeqCst))
}

/// At the top of every turn of the indexer's loop. Sets the mode; returns nothing.
///
/// With a connection: nothing, unless the mode has become `Degraded`, and then the
/// connection is given up, so no handler writes again. Without one: `open_v2`, at once and
/// then every 60 s.
// SPEC: 3866 says "the indexer retries `open_v2` every 60 s" of every reason. A brain that
// was changed under this process (3880: "restart Synaps"), one that is damaged (3878: the
// file is not touched) and a stopped indexer are not retried: a restart of Synaps does that.
pub fn ensure_open(ctx: &mut IndexerCtx) {
    let degraded = mode_of(ctx) == Some(Mode::Degraded);
    if ctx.conn.is_some() {
        if degraded {
            ctx.conn = None;
            ctx.signer = None;
            ctx.brain_file = None;
        }
        return;
    }
    if degraded {
        match DegradedReason::from_u8(ctx.status.reason.load(Ordering::SeqCst)) {
            // RULING: wave3-c12b ruling 2. A process whose front thread stopped opens nothing.
            DegradedReason::BrainChanged
            | DegradedReason::BrainDamaged
            | DegradedReason::IndexerStopped
            | DegradedReason::FrontStopped => return,
            _ => {}
        }
    }
    // RULING: wave3-c12b ruling 10. A monotonic clock, not the wall clock.
    let retry = Duration::from_millis(ctx.test.open_retry_ms.unwrap_or(OPEN_RETRY_MS));
    if ctx.open_tried.is_some_and(|tried| tried.elapsed() < retry) {
        return;
    }
    ctx.open_tried = Some(Instant::now());

    match open(ctx) {
        Ok(()) => {
            log_line("brain open");
            // The row is there early: before the first scan (C-5).
            if write_status(ctx).is_err() {
                ctx.status.write_errors.fetch_add(1, Ordering::Relaxed);
            }
            if send_all(ctx).is_err() {
                ctx.status.write_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
        Err(e) => {
            ctx.conn = None;
            ctx.signer = None;
            let reason = reason_of(&e);
            log_line(&format!("brain not open: {}", reason.as_str()));
            degrade(&ctx.status, &ctx.mode, reason, None);
        }
    }
}

fn reason_of(e: &AxelError) -> DegradedReason {
    match e {
        AxelError::NoBrain(_) | AxelError::Io(_) => DegradedReason::NoBrain,
        AxelError::V1Brain(_) => DegradedReason::V1Brain,
        AxelError::BrainTooNew { .. } => DegradedReason::BrainTooNew,
        AxelError::BrainDamaged(_) => DegradedReason::BrainDamaged,
        AxelError::BrainChanged => DegradedReason::BrainChanged,
        AxelError::OtherBase(_) => DegradedReason::OtherBase,
        AxelError::Db(e)
            if matches!(
                e.sqlite_error_code(),
                Some(ErrorCode::NotADatabase) | Some(ErrorCode::DatabaseCorrupt)
            ) =>
        {
            DegradedReason::BrainDamaged
        }
        _ => DegradedReason::BrainNotReady,
    }
}

fn canonical_base(base: &Path) -> String {
    base.canonicalize().unwrap_or_else(|_| base.to_path_buf()).to_string_lossy().into_owned()
}

const BASE_SQL: &str = "SELECT value FROM context_data WHERE layer = 'axel' AND key = 'synaps_base'";

/// The base an existing file records, read without writing to it. `Ok(None)` when there is
/// no file, or the file has no such table or row: `open_v2` says what it is.
// RULING: wave3-c12b ruling 6. A look that fails (the file is locked, or cannot be read) is
// an error, never "no base recorded": `open_v2` must not run on a brain that may belong to
// another base. `reason_of` makes it `BrainNotReady`, which is tried again, or
// `BrainDamaged` for a file that is not a database.
fn recorded_base_unopened(path: &Path) -> Result<Option<String>> {
    if !path.is_file() {
        return Ok(None);
    }
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(Duration::from_millis(1_000))?;
    let has_table: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'context_data')",
        [],
        |r| r.get(0),
    )?;
    if !has_table {
        return Ok(None);
    }
    Ok(conn.query_row(BASE_SQL, [], |r| r.get::<_, String>(0)).optional()?)
}

// RULING: wave3-c12b ruling 5. The file this process opened is known by its device and
// inode numbers. A brain restored over the path, or a path that is gone, leaves the open
// connections on a file nobody will read again. Unix only: elsewhere there is no number to
// compare, nothing is kept, and the check never fires.
#[cfg(unix)]
fn file_id(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path).ok().map(|m| (m.dev(), m.ino()))
}

#[cfg(not(unix))]
fn file_id(_path: &Path) -> Option<(u64, u64)> {
    None
}

/// True when the path no longer names the file that was opened.
fn brain_file_changed(ctx: &IndexerCtx) -> bool {
    match ctx.brain_file {
        Some(opened) => file_id(&ctx.cfg.brain) != Some(opened),
        None => false,
    }
}

/// `open_v2`, with the base guard on both sides of it: a brain recorded for another Synaps
/// base is refused before anything can write to it.
fn open(ctx: &mut IndexerCtx) -> Result<()> {
    let path = ctx.cfg.brain.clone();
    let base = canonical_base(&ctx.cfg.base);
    if let Some(recorded) = recorded_base_unopened(&path)? {
        if recorded != base {
            return Err(AxelError::OtherBase(PathBuf::from(recorded)));
        }
    }

    let model = ctx.embed.model_id().unwrap_or(DEFAULT_EMBEDDER).to_string();
    let opts = OpenOpts { agent: None, model: &model, dim: EMBEDDING_DIM, upgrade_v1: false };
    let brain = brain_v2::open_v2(&path, &opts)?;

    // Again under the write lock (a brain made a moment ago has none), then record ours.
    brain_v2::begin_write(&brain.conn)?;
    let recorded = (|| -> Result<Option<String>> {
        let recorded = brain.conn.query_row(BASE_SQL, [], |r| r.get::<_, String>(0)).optional()?;
        if recorded.is_none() {
            brain.conn.execute(
                "INSERT INTO context_data(key, layer, value) VALUES('synaps_base', 'axel', ?1)",
                [&base],
            )?;
        }
        Ok(recorded)
    })();
    match recorded {
        Ok(Some(recorded)) if recorded != base => {
            let _ = brain.conn.execute_batch("ROLLBACK");
            return Err(AxelError::OtherBase(PathBuf::from(recorded)));
        }
        Ok(_) => brain.conn.execute_batch("COMMIT")?,
        Err(e) => {
            let _ = brain.conn.execute_batch("ROLLBACK");
            return Err(e);
        }
    }

    // RULING: wave4 R-12b item 2. What `open_v2` could not adopt is said in `status`.
    ctx.status.unreadable_memories.store(brain.unreadable_memories as u64, Ordering::Relaxed);
    ctx.conn = Some(brain.conn);
    ctx.signer = brain.signer;
    ctx.brain_file = file_id(&path);
    let usable = model_usable(ctx).unwrap_or(false);
    ctx.status.model_loaded.store(usable, Ordering::Relaxed);
    if ctx.embed.model_id().is_some() && !usable {
        log_line("model differs from the brain's: words only");
    }
    recover(&ctx.status, &ctx.mode);
    Ok(())
}

/// Section 1 (1006): the running model and its dimension are compared with `brain_meta`.
/// On a mismatch, with no model, or when the embed thread has ended, the process runs on
/// words alone: no vector is loaded, asked for or written.
fn model_usable(ctx: &IndexerCtx) -> Result<bool> {
    let (Some(conn), Some(model)) = (ctx.conn.as_ref(), ctx.embed.model_id()) else { return Ok(false) };
    // RULING: wave3-c12b ruling 9. A thread hung inside the model is no model.
    if ctx.embed.stopped() || ctx.embed.stuck() {
        return Ok(false);
    }
    Ok(model_matches(&brain_model(conn)?, model))
}

/// The model and the dimension `brain_meta` records.
fn brain_model(conn: &Connection) -> Result<(Option<String>, Option<i64>)> {
    let recorded: Option<(Option<String>, Option<i64>)> = conn
        .query_row(
            "SELECT json_extract(value, '$.embedder_model'), json_extract(value, '$.embedding_dim')
             FROM brain_meta WHERE key = 'meta'",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()?;
    Ok(recorded.unwrap_or((None, None)))
}

fn model_matches(recorded: &(Option<String>, Option<i64>), model: &str) -> bool {
    matches!(recorded, (Some(m), Some(d)) if m == model && *d == EMBEDDING_DIM as i64)
}

fn blob_to_vec(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn scope_of(project_key: Option<String>) -> VecScope {
    match project_key {
        None => VecScope::Global,
        Some(dir) if dir == GLOBAL_DIR => VecScope::Global,
        Some(dir) => VecScope::Project(dir),
    }
}

/// Count and highest id of the visible embedded rows: documents, then memory rows.
fn signature(conn: &Connection, model: &str) -> Result<[i64; 4]> {
    let (docs, doc_max): (i64, i64) = conn.query_row(
        &format!("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM documents WHERE {DOCS_WHERE}"),
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    let (mems, mem_max): (i64, i64) = conn.query_row(
        "SELECT COUNT(*), COALESCE(MAX(vid), 0) FROM memory_ext
         WHERE embedding IS NOT NULL AND embed_model = ?1 AND hidden_ms IS NULL",
        [model],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    Ok([docs, doc_max, mems, mem_max])
}

/// The stored vectors as one `VecDelta::Replace` (section 3): no hidden row, and no shadow
/// document of a memory, whose vector is in the array as the memory's own row (R-8).
fn send_all(ctx: &mut IndexerCtx) -> Result<()> {
    if !model_usable(ctx)? {
        return Ok(());
    }
    let (Some(conn), Some(model)) = (ctx.conn.as_ref(), ctx.embed.model_id()) else { return Ok(()) };
    let sig = signature(conn, model)?;
    let mut items = Vec::new();
    {
        let mut stmt = conn.prepare(&format!("SELECT id, embedding FROM documents WHERE {DOCS_WHERE}"))?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Option<Vec<u8>>>(1)?)))?;
        for row in rows {
            let (id, blob) = row?;
            if let Some(blob) = blob.filter(|b| !b.is_empty()) {
                items.push(VecItem { key: VecKey::Doc(id), scope: VecScope::Documents, v: blob_to_vec(&blob) });
            }
        }
        let mut stmt = conn.prepare(
            "SELECT e.vid, m.project_key, e.embedding
             FROM memory_ext e JOIN memories m ON m.id = e.mem_id
             WHERE e.embedding IS NOT NULL AND e.embed_model = ?1 AND e.hidden_ms IS NULL",
        )?;
        let rows = stmt.query_map([model], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, Vec<u8>>(2)?))
        })?;
        for row in rows {
            let (vid, project_key, blob) = row?;
            items.push(VecItem { key: VecKey::Mem(vid), scope: scope_of(project_key), v: blob_to_vec(&blob) });
        }
    }
    // Recorded only when it was sent: a front thread that is gone has been sent nothing.
    if ctx.vec_tx.send(VecDelta::Replace(items)).is_ok() {
        ctx.vectors_sent = Some(sig);
    }
    Ok(())
}

/// Every 60 s (called by `indexer::tick`): when the count or the highest id of visible
/// embedded rows differs from what this process has sent to its front thread, the whole set
/// is sent again. That is how another process's vectors, hides and restores arrive.
pub fn resync_vectors(ctx: &mut IndexerCtx) -> Result<()> {
    if !model_usable(ctx)? {
        return Ok(());
    }
    let (Some(conn), Some(model)) = (ctx.conn.as_ref(), ctx.embed.model_id()) else { return Ok(()) };
    let now = signature(conn, model)?;
    if ctx.vectors_sent == Some(now) {
        return Ok(());
    }
    send_all(ctx)
}

/// Every 1 s (called by `indexer::tick`): up to 64 rows with `embedding IS NULL`, newest
/// first, never a `secret` row, go to the embed thread's bulk queue when no bulk request is
/// out. What is embedded for a row is `title + "\n" + body`.
pub fn pump_embeddings(ctx: &mut IndexerCtx) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    let waiting: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memory_ext e JOIN memories m ON m.id = e.mem_id
         WHERE e.embedding IS NULL AND m.sensitivity <> 'secret'",
        [],
        |r| r.get(0),
    )?;
    ctx.status.vectors_waiting.store(waiting.max(0) as u64, Ordering::Relaxed);
    let usable = model_usable(ctx)?;
    ctx.status.model_loaded.store(usable, Ordering::Relaxed);
    // The embed thread's bulk queue takes whatever it is given: rows it still holds are
    // not offered a second time.
    // RULING: wave3-c12b ruling 11. `backlog()`, not a flag of this thread's: it is 0 again
    // when the embed thread has ended or has dropped a batch, and (wave4 R-12b item 5) when
    // `on_embedded` has finished with the job, not when the job was queued.
    if !usable || ctx.embed.backlog() > 0 || waiting == 0 || mode_of(ctx) != Some(Mode::Ready) {
        return Ok(());
    }
    let mut stmt = conn.prepare(
        "SELECT e.vid, m.title, m.content, e.mem_id, e.content_hash
         FROM memory_ext e JOIN memories m ON m.id = e.mem_id
         WHERE e.embedding IS NULL AND m.sensitivity <> 'secret'
         ORDER BY e.created_ms DESC, e.vid DESC LIMIT ?1",
    )?;
    // RULING: wave4 R-12b item 3. A title or a body that is not text, or not valid UTF-8,
    // is embedded as empty: read as a string it failed the whole query, every second, and
    // no older row was ever embedded.
    let items = stmt
        .query_map([BATCH as i64], |r| {
            let title = brain_v2::text_at(r, 1)?.unwrap_or_default();
            let body = brain_v2::text_at(r, 2)?.unwrap_or_default();
            // RULING: wave4 R-12b item 4. The batch says which row each number stood for.
            Ok(BulkRow { vid: r.get(0)?, mem_id: r.get(3)?, content_hash: r.get(4)?, text: format!("{title}\n{body}") })
        })?
        .collect::<rusqlite::Result<Vec<BulkRow>>>()?;
    drop(stmt);
    ctx.embed.submit_rows(items);
    Ok(())
}

/// `Job::Embedded` (called by `indexer::dispatch`): the vectors are written, 64 per
/// transaction, and the same vectors go to the front thread as a `VecDelta::Upsert`.
///
/// A vector is written only to the row its text was read from: the number, the `mem_id` and
/// the `content_hash` must all be what `out` recorded when the batch was sent, and the
/// memory must not be `secret` now. A row purged while its text was with the embed thread
/// may have given its number to a new row, which must not get the purged text's vector.
// RULING: wave4 R-12b item 4.
pub fn on_embedded(
    ctx: &mut IndexerCtx,
    model: &str,
    vectors: Vec<(VecKey, Vec<f32>)>,
    out: Option<&BatchOut>,
) -> Result<()> {
    if mode_of(ctx) != Some(Mode::Ready) || ctx.embed.model_id() != Some(model) || !model_usable(ctx)? {
        return Ok(());
    }
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };

    let mut items: Vec<VecItem> = Vec::new();
    let mut failed = None;
    for (n, chunk) in vectors.chunks(BATCH).enumerate() {
        if n > 0 {
            brain_v2::stand_aside(&ctx.writing);
        }
        if let Err(e) = brain_v2::begin_write(conn) {
            failed = Some(e);
            break;
        }
        let mut written = Vec::new();
        let wrote = (|| -> Result<()> {
            for (key, v) in chunk {
                let VecKey::Mem(vid) = key else { continue };
                if v.len() != EMBEDDING_DIM {
                    continue;
                }
                let Some((mem_id, content_hash)) = out.and_then(|out| out.row(*vid)) else { continue };
                let changed = conn.execute(
                    "UPDATE memory_ext SET embedding = ?1, embed_model = ?2
                     WHERE vid = ?3 AND mem_id = ?4 AND content_hash = ?5 AND embedding IS NULL
                       AND EXISTS (SELECT 1 FROM memories m
                                   WHERE m.id = memory_ext.mem_id AND m.sensitivity <> 'secret')",
                    params![vec_to_blob(v), model, vid, mem_id, content_hash],
                )?;
                if changed != 1 {
                    continue;
                }
                let row: Option<(Option<String>, Option<i64>)> = conn
                    .query_row(
                        "SELECT m.project_key, e.hidden_ms FROM memory_ext e JOIN memories m ON m.id = e.mem_id
                         WHERE e.vid = ?1",
                        [vid],
                        |r| Ok((r.get(0)?, r.get(1)?)),
                    )
                    .optional()?;
                if let Some((project_key, None)) = row {
                    written.push(VecItem { key: VecKey::Mem(*vid), scope: scope_of(project_key), v: v.clone() });
                }
            }
            Ok(())
        })();
        let ended = match wrote {
            Ok(()) => conn.execute_batch("COMMIT").map_err(AxelError::from),
            Err(e) => Err(e),
        };
        match ended {
            Ok(()) => items.extend(written),
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                failed = Some(e);
                break;
            }
        }
    }

    // What was committed is sent, and counted as sent, whatever happened to a later batch.
    if !items.is_empty() {
        let count = items.len() as i64;
        let highest = items.iter().map(|i| if let VecKey::Mem(v) = i.key { v } else { 0 }).max().unwrap_or(0);
        if ctx.vec_tx.send(VecDelta::Upsert(items)).is_ok() {
            if let Some(sent) = ctx.vectors_sent.as_mut() {
                sent[2] += count;
                sent[3] = sent[3].max(highest);
            }
        }
    }
    match failed {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Every 30 s (called by `indexer::tick`, by a long scan between two archives, and once as
/// soon as the brain is open): the row `context_data`, layer `axel`, key `status`. It
/// begins with `begin_write`, which re-reads the brain's version. A process that is not
/// `Ready` writes nothing.
pub fn write_status(ctx: &mut IndexerCtx) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    let mode = mode_of(ctx);
    if mode != Some(Mode::Ready) {
        return Ok(());
    }
    // RULING: wave3-c12b ruling 5. Before the row, every 30 s.
    if brain_file_changed(ctx) {
        log_line("brain file replaced or gone");
        degrade(&ctx.status, &ctx.mode, DegradedReason::BrainChanged, None);
        return Ok(());
    }
    // RULING: wave3-c12b ruling 9. What the brain was made for, beside what is running:
    // `model_matches` is null when no model is running, so there is nothing to compare.
    let recorded = brain_model(conn)?;
    let matches = ctx.embed.model_id().map(|model| model_matches(&recorded, model));
    // RULING: wave4 R-12b item 8. How long a purge has waited for its log to be truncated:
    // the age of the mark `index_archives::mark_purged` leaves, which every later purge
    // writes again. Null when there is no mark; 0 when the mark's time cannot be read.
    let purge_pending_ms: Option<i64> = conn
        .query_row(
            "SELECT CAST((julianday('now') - julianday(updated)) * 86400000 AS INTEGER)
             FROM context_data WHERE key = 'purge_pending' AND layer = 'axel'",
            [],
            |r| r.get::<_, Option<i64>>(0),
        )
        .optional()?
        .map(|age| age.unwrap_or(0).max(0));
    let s = &ctx.status;
    let now = (ctx.clock)();
    let n = |a: &std::sync::atomic::AtomicU64| a.load(Ordering::Relaxed);
    let row = json!({
        "pid": std::process::id(),
        "version": env!("CARGO_PKG_VERSION"),
        "mode": "ready",
        "reason": DegradedReason::from_u8(s.reason.load(Ordering::SeqCst)).as_str(),
        "reason_at": s.reason_place(),
        "written_ms": now,
        "started_ms": s.started_ms.load(Ordering::Relaxed),
        "scan": match s.scan.load(Ordering::Relaxed) { 0 => "pending", 1 => "running", _ => "done" },
        "indexer_running": true,
        "embed_stopped": ctx.embed.stopped(),
        "embed_stuck": ctx.embed.stuck(),
        "model_loaded": s.model_loaded.load(Ordering::Relaxed),
        "model": ctx.embed.model_id(),
        "brain_model": recorded.0,
        "model_matches": matches,
        "model_dir": ctx.cfg.model_dir.to_string_lossy(),
        "embedder": ctx.cfg.embedder.to_string(),
        "vectors_waiting": n(&s.vectors_waiting),
        "docs_waiting": n(&s.docs_waiting),
        "unreadable_memories": n(&s.unreadable_memories),
        "purge_pending_ms": purge_pending_ms,
        "breaker_open": s.breaker_open.load(Ordering::Relaxed),
        "breaker_trips": n(&s.breaker_trips),
        "hook_misses": n(&s.hook_misses),
        "sessions_dropped": n(&s.sessions_dropped),
        "jobs_dropped": n(&s.jobs_dropped),
        "tokens_blanked": n(&s.tokens_blanked),
        "turns_withheld": n(&s.turns_withheld),
        "turns_dropped": n(&s.turns_dropped),
        "write_errors": n(&s.write_errors),
        "panics": n(&s.panics),
        "forum": {
            "declared": ctx.cfg.forum_on(),
            "core_owns": base_config_value(&ctx.cfg.base, "memory.backend").as_deref() == Some("axel"),
        },
        "base": canonical_base(&ctx.cfg.base),
        "warnings": ctx.cfg.warnings,
    });

    brain_v2::begin_write(conn)?;
    let wrote = conn
        .execute(
            "INSERT INTO context_data(key, layer, value, updated) VALUES('status', 'axel', ?1, datetime('now'))
             ON CONFLICT(key, layer) DO UPDATE SET value = excluded.value, updated = excluded.updated",
            [row.to_string()],
        )
        .and_then(|_| conn.execute_batch("COMMIT"));
    if let Err(e) = wrote {
        let _ = conn.execute_batch("ROLLBACK");
        return Err(e.into());
    }
    s.status_written_ms.store(now, Ordering::Relaxed);
    Ok(())
}
