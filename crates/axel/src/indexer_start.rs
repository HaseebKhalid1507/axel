//! Opening the brain with retry, the base guard, the first vector load, the embedding
//! backlog, the status row. Commit 12b writes this file.

use std::sync::mpsc::Receiver;

use crate::error::Result;
use crate::ext::config::ExtConfig;
use crate::ext::types::{Boot, IndexerCtx, Job};
use crate::vindex::{VecDelta, VecKey};

/// Hands back the other ends of the channels it makes (B3-3a).
pub fn open_ctx(cfg: ExtConfig, boot: Boot) -> (IndexerCtx, Receiver<Job>, Receiver<VecDelta>) {
    let _ = (cfg, boot);
    unimplemented!("commit 12b")
}

// SPEC: 4787-4797 name the five functions below without signatures. They take the
// `IndexerCtx`; `ensure_open` returns nothing (it sets the mode), the others `Result<()>`;
// `on_embedded` takes the two fields of `Job::Embedded`.

/// At the top of every turn of the indexer's loop.
pub fn ensure_open(ctx: &mut IndexerCtx) {
    let _ = ctx;
}

/// `Job::Embedded` (called by `indexer::dispatch`).
pub fn on_embedded(ctx: &mut IndexerCtx, model: &str, vectors: Vec<(VecKey, Vec<f32>)>) -> Result<()> {
    let _ = (ctx, model, vectors);
    Ok(())
}

/// Every 1 s (called by `indexer::tick`).
pub fn pump_embeddings(ctx: &mut IndexerCtx) -> Result<()> {
    let _ = ctx;
    Ok(())
}

/// Every 60 s (called by `indexer::tick`).
pub fn resync_vectors(ctx: &mut IndexerCtx) -> Result<()> {
    let _ = ctx;
    Ok(())
}

/// Every 30 s (called by `indexer::tick`, and by a long scan between two archives).
pub fn write_status(ctx: &mut IndexerCtx) -> Result<()> {
    let _ = ctx;
    Ok(())
}
