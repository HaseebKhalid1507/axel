//! Candidates, adapters into JR's types, floors, the trace. Commit 15 writes this file.

use std::time::Instant;

use rusqlite::Connection;

use crate::embed::EmbedService;
use crate::error::Result;
use crate::ext::types::{FrontState, HookEvent, HookOut, IndexerCtx};
use crate::project::ProjectRef;
use crate::project_memory::ScopedDescriptor;
use crate::retrieval::{RankReason, ScoreParts};
use crate::vindex::ExactIndex;

/// What this caller may see (section 5 fills it).
#[derive(Clone, Debug)]
pub struct View {
    /// Rows filed under this project's directory.
    pub project: Option<ProjectRef>,
    /// Global rows and the old unscoped rows.
    pub global: bool,
    /// The owner's note files.
    pub documents: bool,
    /// Live captures of this chat are still in its window.
    pub own_session: Option<String>,
}

// SPEC: 2322 uses `Kind` and no line defines it. The variants are the five values of the
// search tool's `kinds` (3148): file, note, chat, worker_report, saved.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    File,
    Note,
    Chat,
    WorkerReport,
    Saved,
}

pub struct RecallRequest<'a> {
    pub query: &'a str,
    pub view: &'a View,
    pub kinds: Option<&'a [Kind]>,
    pub since_ms: Option<i64>,
    pub max_items: usize,
    pub deadline: Instant,
    /// The hook: floors apply; nothing relevant means nothing.
    pub ambient: bool,
}

// SPEC: 2328 uses `Origin` and no line defines it. The variants are the six values of
// `memory_ext.origin_kind` (603) and one for a row of `documents`, which has no origin_kind.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Origin {
    Archive,
    NoteFile,
    Capture,
    Tool,
    Cli,
    Legacy,
    Document,
}

// SPEC: 2329 uses `Outcome` and no line defines it. It is the text of `recall_log.outcome`
// (731), for example `skipped:<why>` (2499).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome(pub String);

// SPEC: 2332 uses `RecallConfig` and no line defines it; its one named field is `now`
// (2565, "a fixed clock in `RecallConfig.now`"), assumed to be milliseconds.
#[derive(Clone, Debug)]
pub struct RecallConfig {
    pub now: i64,
}

#[derive(Clone, Debug)]
pub struct Pick {
    pub n: usize,
    pub r#ref: String,
    pub descriptor: ScopedDescriptor,
    pub reasons: Vec<RankReason>,
    pub parts: ScoreParts,
    pub origin: Origin,
    pub date: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct RecallOutcome {
    pub picks: Vec<Pick>,
    pub outcome: Outcome,
    pub trace: RecallTrace,
}

// SPEC: 4742 says `RecallTrace` is "the fields of the `detail` JSON of section 3 plus the
// `recall_log` columns" and gives no Rust. The columns are 728-733; the JSON is 2530-2539.
// The names of the five nested structs are commit 3's.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecallTrace {
    // the recall_log columns
    pub ts_ms: i64,
    pub session_id: Option<String>,
    pub logical: Option<String>,
    pub scope_dir: Option<String>,
    pub query_preview: Option<String>,
    pub query_hash: String,
    pub outcome: String,
    pub total_ms: f64,
    pub embed_ms: Option<f64>,
    pub candidates: i64,
    pub selected: i64,
    // the detail JSON
    pub query_tokens: Vec<String>,
    pub embed: TraceEmbed,
    pub thresholds: TraceThresholds,
    pub generators: TraceGenerators,
    pub picks: Vec<TracePick>,
    pub dropped: Vec<TraceDropped>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TraceEmbed {
    pub used: bool,
    pub ms: Option<f64>,
    pub fallback: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TraceThresholds {
    pub min_sem: f64,
    pub min_lex: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TraceGenerators {
    pub mem_words: usize,
    pub doc_words: usize,
    pub by_meaning: usize,
    pub past_floor: usize,
    pub after_dedupe: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TracePick {
    pub n: usize,
    pub r#ref: String,
    pub kind: String,
    pub origin: String,
    pub scope: String,
    pub created: Option<String>,
    pub parts: ScoreParts,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TraceDropped {
    pub r#ref: String,
    pub why: String,
}

/// The one entry point, called on the front thread by the hook and by `axel_search`.
pub fn recall(
    conn: &Connection,
    index: &ExactIndex,
    embed: &EmbedService,
    cfg: &RecallConfig,
    req: &RecallRequest,
) -> RecallOutcome {
    let _ = (conn, index, embed, cfg, req);
    RecallOutcome {
        picks: Vec::new(),
        outcome: Outcome("skipped:not_built".to_string()),
        trace: RecallTrace::default(),
    }
}

/// The `before_message` hook (called by `ext::hooks::handle`).
pub fn on_before_message(st: &mut FrontState, ev: &HookEvent, recall_allowed: bool) -> HookOut {
    let _ = (st, ev, recall_allowed);
    HookOut::Continue
}

// SPEC: 4790 and 4796 name `recall::write_log` and `recall::trim_log` without signatures.
// Like every indexer handler the document gives no signature for, they take the
// `IndexerCtx` (4491: "its handlers take an `IndexerCtx`") and return `Result<()>`.

/// `Job::LogRecall` (called by `indexer::dispatch`).
pub fn write_log(ctx: &mut IndexerCtx, trace: &RecallTrace) -> Result<()> {
    let _ = (ctx, trace);
    Ok(())
}

/// Every 3,600 s (called by `indexer::tick`).
pub fn trim_log(ctx: &mut IndexerCtx) -> Result<()> {
    let _ = ctx;
    Ok(())
}
