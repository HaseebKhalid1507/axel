//! The indexer thread's loop, the `Job` match, the tick order. Final in commit 3.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::error::Result;
use crate::ext::types::{IndexerCtx, Job, Mode};
use crate::{brain_v2, capture, docsync, index_archives, index_notes, indexer_start, recall};

/// Returns on `Job::Shutdown`, when `ctx.stop` is set, when every sender is gone, and when
/// `ctx.test.indexer_stop_after_jobs` jobs have been taken.
pub fn run(mut ctx: IndexerCtx, rx: Receiver<Job>) {
    let mut jobs: u64 = 0;
    loop {
        // RULING: wave3-c12b ruling 10. `Job::Shutdown` goes to the back of the queue and is
        // dropped when the queue is full; the flag is seen at the next turn either way.
        if ctx.stop.load(Ordering::SeqCst) {
            return;
        }
        indexer_start::ensure_open(&mut ctx);
        match rx.recv_timeout(Duration::from_secs(1)) {
            Ok(Job::Shutdown) => return,
            Ok(job) => {
                dispatch(&mut ctx, job);
                jobs += 1;
                if ctx.test.indexer_stop_after_jobs.is_some_and(|n| jobs >= n) {
                    return;
                }
            }
            // No job for a second: the idle turn. A purge that forgot text left a mark;
            // the log is truncated now (section 2.7; the one edit commit 13a's part makes
            // in this file).
            Err(RecvTimeoutError::Timeout) => guarded(&mut ctx, index_archives::checkpoint_after_purge),
            Err(RecvTimeoutError::Disconnected) => return,
        }
        tick(&mut ctx);
    }
}

/// One call per arm, inside `catch_unwind`.
pub fn dispatch(ctx: &mut IndexerCtx, job: Job) {
    if matches!(job, Job::Tick | Job::Shutdown) {
        // A tick is the loop's own next step; `run` takes `Shutdown` before it gets here.
        return;
    }
    // Before a job, outside any transaction.
    if ctx.test.indexer_sleep_ms > 0 {
        std::thread::sleep(Duration::from_millis(ctx.test.indexer_sleep_ms));
    }
    guarded(ctx, |ctx| match job {
        Job::Tick | Job::Shutdown => Ok(()),
        Job::Archive { .. } => index_archives::on_job(ctx, &job),
        Job::Forgotten { id, kind, project_key } => {
            if ctx.test.panic_on_forgotten_job {
                panic!("test switch: panic_on_forgotten_job");
            }
            match kind.as_str() {
                "archive" => index_archives::on_forgotten(ctx, &id, &project_key),
                "note" => index_notes::on_forgotten(ctx, &id, &project_key),
                _ => Ok(()),
            }
        }
        Job::Capture(turn) => match ctx.conn.as_ref() {
            Some(conn) => capture::write_turn(conn, ctx.signer.as_ref(), &ctx.cfg, &turn).map(|_| ()),
            None => Ok(()),
        },
        Job::LogRecall(trace) => recall::write_log(ctx, &trace),
        Job::Embedded { model, vectors, out } => indexer_start::on_embedded(ctx, &model, vectors, out.as_deref()),
    });
}

const ADOPT_MS: i64 = 60_000;
const NOTES_MS: i64 = 5_000;
const SCAN_MS: i64 = 600_000;
const DOCSYNC_MS: i64 = 900_000;
const PUMP_MS: i64 = 1_000;
const RESYNC_MS: i64 = 60_000;
const TRIM_MS: i64 = 3_600_000;
const STATUS_MS: i64 = 30_000;

/// The eight calls, in this order, each at most as often as its interval and each inside
/// `catch_unwind`. There is no sweep: nothing is deleted by time.
pub fn tick(ctx: &mut IndexerCtx) {
    if due(ctx, "adopt_orphans", ADOPT_MS) {
        // RULING: wave4 R-12b item 2. The rows the adoption could not read are counted.
        guarded(ctx, |ctx| {
            let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
            let done = brain_v2::adopt_orphans_counted(conn, "cli")?;
            ctx.status.unreadable_memories.store(done.skipped as u64, Ordering::Relaxed);
            Ok(())
        });
    }
    if due(ctx, "index_notes::pass", NOTES_MS) {
        guarded(ctx, index_notes::pass);
    }
    if due(ctx, "index_archives::scan", SCAN_MS) {
        guarded(ctx, index_archives::scan);
    }
    if due(ctx, "docsync::tick", DOCSYNC_MS) {
        guarded(ctx, docsync::tick);
    }
    if due(ctx, "pump_embeddings", PUMP_MS) {
        guarded(ctx, indexer_start::pump_embeddings);
    }
    if due(ctx, "resync_vectors", RESYNC_MS) {
        guarded(ctx, indexer_start::resync_vectors);
    }
    if due(ctx, "recall::trim_log", TRIM_MS) {
        guarded(ctx, recall::trim_log);
    }
    let status_ms = ctx.test.status_tick_ms.map_or(STATUS_MS, |ms| ms as i64);
    if due(ctx, "write_status", status_ms) {
        guarded(ctx, indexer_start::write_status);
    }
}

/// True, and the time is recorded, when `name` has never run or last ran at least
/// `every_ms` ago. The time is recorded before the call, so a call that panics waits its
/// interval like any other.
fn due(ctx: &mut IndexerCtx, name: &'static str, every_ms: i64) -> bool {
    // RULING: wave3-c12b ruling 10. A monotonic clock: the wall clock set back an hour
    // would stop every call for an hour.
    let every = Duration::from_millis(every_ms.max(0) as u64);
    match ctx.last_run.get(name) {
        Some(last) if last.elapsed() < every => false,
        _ => {
            ctx.last_run.insert(name, Instant::now());
            true
        }
    }
}

// SPEC: 4798-4800 say every job and every tick call runs inside `catch_unwind` and that
// the indexer calls `stand_aside` "before each transaction". The transactions are the
// handlers' own, so they call it too; this file calls it once before each job and each
// tick call. An `Err` is counted in `Status.write_errors` and handed to
// `brain_v2::degrade_on` (4809-4817); what becomes of a panic is not written: it is
// counted in `Status.panics`. Either way the work is dropped.
/// How every job and every tick call is run. Public for its tests.
pub fn guarded(ctx: &mut IndexerCtx, f: impl FnOnce(&mut IndexerCtx) -> Result<()>) {
    // RULING: wave3-c12b ruling 8. A degraded process leaves the brain alone: after the
    // first error that degrades, the rest of the tick and every queued job do nothing.
    if ctx.mode.load(Ordering::SeqCst) == Mode::Degraded as u8 {
        return;
    }
    // RULING: wave3-c12b ruling 12. After `shutdown` no handler begins: what a job would
    // have written is found again at the next start.
    if ctx.stop.load(Ordering::SeqCst) {
        return;
    }
    brain_v2::stand_aside(&ctx.writing);
    let failed = match catch_unwind(AssertUnwindSafe(|| f(ctx))) {
        Ok(Ok(())) => false,
        Ok(Err(e)) => {
            ctx.status.write_errors.fetch_add(1, Ordering::Relaxed);
            brain_v2::degrade_on(&e, &ctx.status, &ctx.mode);
            true
        }
        Err(_) => {
            ctx.status.panics.fetch_add(1, Ordering::Relaxed);
            true
        }
    };
    // RULING: wave3-c11 ruling 2. An error or a panic between `begin_write` and `COMMIT`
    // must not leave its transaction open on this long-lived connection: the next handler
    // would write inside it, and whoever committed would commit the failed work too.
    if failed {
        if let Some(conn) = ctx.conn.as_ref().filter(|conn| !conn.is_autocommit()) {
            let _ = conn.execute_batch("ROLLBACK");
        }
    }
}
