//! The notes pass. Commit 14 writes this file.

use crate::error::Result;
use crate::ext::types::IndexerCtx;

// SPEC: 4789-4792 name `index_notes::pass` and `index_notes::on_forgotten` without
// signatures. They take the `IndexerCtx` (4491: "its handlers take an `IndexerCtx`") and
// return `Result<()>`; `on_forgotten` takes the two other fields of `Job::Forgotten`.

/// Every 5 s (called by `indexer::tick`).
pub fn pass(ctx: &mut IndexerCtx) -> Result<()> {
    let _ = ctx;
    Ok(())
}

/// `Job::Forgotten` with `kind = "note"` (called by `indexer::dispatch`).
pub fn on_forgotten(ctx: &mut IndexerCtx, id: &str, project_key: &str) -> Result<()> {
    let _ = (ctx, id, project_key);
    Ok(())
}
