//! `ExtConfig` from `initialize.params`. Commit 12a writes this file.

use std::path::{Path, PathBuf};

use serde_json::Value;

// SPEC: 4355 and 4358 name `ForumMode` and `EmbedderKind` only in comments
// (`/* Auto | On | Off */`, `/* Onnx | Off | Hash */`); the variants are those, in that order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ForumMode {
    Auto,
    On,
    Off,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EmbedderKind {
    Onnx,
    Off,
    Hash,
}

#[derive(Clone, Debug)]
pub struct ExtConfig {
    pub brain: PathBuf,
    pub base: PathBuf,
    pub data_dir: PathBuf,
    pub recall: bool,
    pub recall_max_items: usize,
    pub recall_max_chars: usize,
    pub recall_budget_ms: u64,
    pub recall_min_sem: f64,
    pub recall_min_lex: f64,
    pub recall_on_events: bool,
    pub recall_exclude_paths: Vec<PathBuf>,
    pub capture: bool,
    pub archives: bool,
    pub forum: ForumMode,
    pub worker_scope_all: bool,
    pub forget_now_min_hidden_hours: u32,
    pub handoff_on_start: bool,
    pub sources: Option<PathBuf>,
    pub archive_events: Vec<String>,
    pub embedder: EmbedderKind,
    pub model_dir: PathBuf,
    pub log_queries: bool,
    pub warnings: Vec<String>,
}

impl ExtConfig {
    /// Never fails.
    pub fn from_initialize(params: &Value, home: &Path) -> ExtConfig {
        let _ = (params, home);
        unimplemented!("commit 12a")
    }

    /// Auto: reads `<base>/config` once (section 6).
    pub fn forum_on(&self) -> bool {
        false
    }
}
