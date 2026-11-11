//! `EmbedService`: the embed thread, which owns the one ONNX session. Commit 9 writes this
//! file.

use std::sync::mpsc::SyncSender;
use std::sync::Arc;
use std::time::Instant;

use velocirag::embedder::Embedder;

use crate::ext::types::{Job, TestHooks};
use crate::vindex::VecKey;

/// A field of both `FrontState` and `IndexerCtx`, so it is `Clone` (B-2c).
#[derive(Clone)]
pub struct EmbedService {
    _stub: (),
}

impl EmbedService {
    /// Spawns the thread; with `test.panic_embed` it panics on its first request.
    /// The stub spawns nothing and drops its arguments.
    pub fn start(embedder: Option<Embedder>, jobs: SyncSender<Job>, test: Arc<TestHooks>) -> Self {
        let _ = (embedder, jobs, test);
        EmbedService { _stub: () }
    }

    pub fn model_id(&self) -> Option<&str> {
        None
    }

    pub fn embed_query(&self, text: &str, deadline: Instant) -> Option<Vec<f32>> {
        let _ = (text, deadline);
        None
    }

    /// Memory rows; results arrive as `Job::Embedded`.
    pub fn submit_bulk(&self, items: Vec<(VecKey, String)>) {
        let _ = items;
    }

    /// Through the bulk queue, blocking the caller; the indexer thread only (section 2.8).
    pub fn embed_wait(&self, texts: Vec<String>, deadline: Instant) -> Option<Vec<Vec<f32>>> {
        let _ = (texts, deadline);
        None
    }

    /// The embed thread has ended (section 7).
    // SPEC: the stub rule (4747-4758) gives `stopped` no neutral answer. `false` is chosen:
    // `true` would make `status` say `embedding: stopped` (3811).
    pub fn stopped(&self) -> bool {
        false
    }
}
