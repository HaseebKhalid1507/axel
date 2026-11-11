//! Scan, index one archive, forget one archive, the forget marks (`mark_forgotten_turn`,
//! `chat_files`, `near_stretch`). Commits 13a and 13b write this file.

use rusqlite::Connection;

use crate::error::Result;
use crate::ext::types::{IndexerCtx, Job};

// SPEC: 4789-4792 name `index_archives::on_job`, `on_forgotten` and `scan` without
// signatures (1780 writes `index_archives::scan(base)`, but a scan writes rows and needs
// the connection too). They take the `IndexerCtx` and return `Result<()>`. `on_job` takes
// the whole job, which is a `Job::Archive`: its eight fields as eight arguments would be
// one more list to keep in step with `ext/types.rs`.

/// `Job::Archive` (called by `indexer::dispatch`). Any other job is nothing to it.
pub fn on_job(ctx: &mut IndexerCtx, job: &Job) -> Result<()> {
    let _ = (ctx, job);
    Ok(())
}

/// `Job::Forgotten` with `kind = "archive"` (called by `indexer::dispatch`).
pub fn on_forgotten(ctx: &mut IndexerCtx, id: &str, project_key: &str) -> Result<()> {
    let _ = (ctx, id, project_key);
    Ok(())
}

/// Every 600 s (called by `indexer::tick`).
pub fn scan(ctx: &mut IndexerCtx) -> Result<()> {
    let _ = ctx;
    Ok(())
}

/// The ids it purged. The stub returns an empty list.
pub fn mark_forgotten_turn(
    conn: &Connection,
    logical: &str,
    session_id: Option<&str>,
    source: &str,
    from_ms: i64,
    to_ms: i64,
    turn_key: Option<&str>,
) -> Result<Vec<String>> {
    let _ = (conn, logical, session_id, source, from_ms, to_ms, turn_key);
    Ok(Vec::new())
}

// SPEC: the three functions below are written with their arguments and without types
// (`forget_archive(conn, scope_dir, archive_id)` 1955, no result;
// `window_start_ms(conn, scope_dir, logical, session_id, t) -> i64` 1714;
// `near_stretch(conn, scope_dir, logical, session_id, cover_ms) -> (i64, i64)` 1733). The
// types are assumed. `index_archive` (1593) and `chat_files` (1725) are not declared: the
// document gives the first no arguments and the second no result, and only this file
// calls them.

pub fn forget_archive(conn: &Connection, scope_dir: &str, archive_id: &str) -> Result<usize> {
    let _ = (conn, scope_dir, archive_id);
    Ok(0)
}

pub fn window_start_ms(
    conn: &Connection,
    scope_dir: &str,
    logical: &str,
    session_id: Option<&str>,
    t: i64,
) -> i64 {
    let _ = (conn, scope_dir, logical, session_id, t);
    0
}

pub fn near_stretch(
    conn: &Connection,
    scope_dir: &str,
    logical: &str,
    session_id: Option<&str>,
    cover_ms: i64,
) -> (i64, i64) {
    let _ = (conn, scope_dir, logical, session_id, cover_ms);
    (0, 0)
}
