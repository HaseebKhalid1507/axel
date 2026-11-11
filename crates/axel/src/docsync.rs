//! The owner's note folders. Commit 22 writes this file.

use crate::error::Result;
use crate::ext::types::IndexerCtx;

// SPEC: 4792 names `docsync::tick` without a signature; like every indexer handler the
// document gives no signature for, it takes the `IndexerCtx` and returns `Result<()>`.
// `docsync::plan(conn, sources) -> Plan` (2014) is not declared: `Plan` is defined nowhere
// and only this file calls it.

/// Every 900 s (called by `indexer::tick`).
pub fn tick(ctx: &mut IndexerCtx) -> Result<()> {
    let _ = ctx;
    Ok(())
}
