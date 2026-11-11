//! The migrations ledger, `open_v2`, `require_v2`, `connect`, `begin_write`,
//! `begin_write_front`, `stand_aside`, `signer`, adoption and the word-index fill.
//! Commit 4 writes this file.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use axel_memkoshi::security::MemorySigner;
use rusqlite::Connection;

use crate::error::{AxelError, Result};
use crate::ext::types::Status;

pub const LATEST: i32 = 1;

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

// SPEC: 910 returns `WriteGuard<'a>` and no line defines it. Taken from the comment at 911:
// "the guard clears the flag when dropped (R-6)".
pub struct WriteGuard<'a> {
    writing: &'a AtomicBool,
}

impl Drop for WriteGuard<'_> {
    fn drop(&mut self) {
        self.writing.store(false, Ordering::Relaxed);
    }
}

/// May create, may migrate. The stub returns `Err(BrainNotReady)` (B3-5.6).
pub fn open_v2(path: &Path, opts: &OpenOpts) -> Result<V2Brain> {
    let _ = (path, opts);
    Err(AxelError::BrainNotReady)
}

/// Pragmas only; never creates, never migrates.
// SPEC: the stub rule (4747-4758) gives `connect` no neutral answer and a stub has no
// connection to hand back: `Err(BrainNotReady)`, as `open_v2`.
pub fn connect(path: &Path, role: Role) -> Result<Connection> {
    let _ = (path, role);
    Err(AxelError::BrainNotReady)
}

/// `BEGIN IMMEDIATE` and the version check.
pub fn begin_write(conn: &Connection) -> Result<()> {
    let _ = conn;
    Ok(())
}

/// Sets the flag, then `begin_write`; the guard clears the flag when dropped (R-6).
pub fn begin_write_front<'a>(conn: &Connection, writing: &'a AtomicBool) -> Result<WriteGuard<'a>> {
    let _ = conn;
    Ok(WriteGuard { writing })
}

/// The indexer, before each transaction: waits in 2 ms steps while the flag is set, at
/// most 1 s.
pub fn stand_aside(writing: &AtomicBool) {
    let _ = writing;
}

/// Read-only; the command line's gate (R-2).
pub fn require_v2(path: &Path) -> Result<()> {
    let _ = path;
    Ok(())
}

/// For the front thread (B-2d); never logs the key.
pub fn signer(conn: &Connection) -> Result<Option<MemorySigner>> {
    let _ = conn;
    Ok(None)
}

/// `origin` is "legacy" or "cli"; also fills the word index (B-1b).
pub fn adopt_orphans(conn: &Connection, origin: &str) -> Result<usize> {
    let _ = (conn, origin);
    Ok(0)
}

/// True when `err` says the brain is damaged or was changed by another binary; then the
/// reason is stored first and the mode after it. The stub returns false and sets nothing.
pub fn degrade_on(err: &AxelError, status: &Status, mode: &AtomicU8) -> bool {
    let _ = (err, status, mode);
    false
}

// SPEC: 1138 names `brain_v2::open_source(path)` without a signature. A connection is
// assumed. Only `import.rs` (commit 19) calls it.
pub fn open_source(path: &Path) -> Result<Connection> {
    let _ = path;
    Err(AxelError::BrainNotReady)
}
