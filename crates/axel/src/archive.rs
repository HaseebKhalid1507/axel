//! Archive file reader, the carried head, classification, unit extraction,
//! `forgotten_keys` (pure functions, no database). Commit 6 writes this file.

use std::path::Path;

#[derive(Clone, Debug)]
pub enum ArchiveFile {
    Live(LiveArchive),
    Tombstone { id: String, logical: String, digest: String },
}

#[derive(Clone, Debug)]
pub struct LiveArchive {
    pub id: String,
    pub scope: String,
    pub logical: String,
    pub digest: String,
    pub created_ms: Option<i64>,
    pub source_message_count: u32,
    pub note: String,
    pub messages: Vec<ArchivedMessage>,
    pub file_size: u64,
    /// 4.1: the record's own top-level `carried`, when core wrote it.
    pub carried: Option<u32>,
    /// 4.1: what an event said about this file; `read_archive` sets `Unknown`, the indexer
    /// fills it.
    pub start: Start,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Start {
    #[default]
    Unknown,
    Whole,
    Carried,
}

impl Start {
    /// `part_no > 1` → `Whole`; `part_no = 1` and `win_no > 2` → `Carried`; anything else →
    /// `Unknown`. The stub returns `Unknown`.
    pub fn from_event(win_no: Option<i64>, part_no: Option<i64>) -> Start {
        let _ = (win_no, part_no);
        Start::Unknown
    }
}

// SPEC: 1401 uses `Role` for a message and no line defines it. The two roles of a stored
// conversation are assumed. (`brain_v2::Role` is another type.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Clone, Debug)]
pub struct ArchivedMessage {
    pub source_index: u32,
    pub role: Role,
    pub blocks: Vec<Block>,
}

#[derive(Clone, Debug)]
pub enum Block {
    Text(String),
    ToolUse { name: String, input_head: String },
    ToolResult { is_error: bool },
    Other,
}

#[derive(Clone, Debug)]
pub struct ArchiveEntry {
    pub rel_name: String,
    pub id_from_name: Option<String>,
    pub size: u64,
}

/// Id letters x, c, r, n.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnitKind {
    Exchange,
    Continued,
    Report,
    Note,
}

#[derive(Clone, Debug)]
pub struct Unit {
    pub kind: UnitKind,
    pub source_index: u32,
    pub block_index: u32,
    pub title: String,
    pub body: String,
    pub aux: String,
    pub turn_key: Option<String>,
}

// SPEC: 1411 returns `SkipReason` and no line defines it; 641 says the reason is stored as
// text in `archive_index.skip_reason`. A wrapped string is assumed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkipReason(pub String);

#[derive(Clone, Debug)]
pub struct ArchivePolicy {
    pub event_sources: Vec<String>,
    /// `["axel_forget", "memory_forget"]`.
    pub forget_tools: Vec<String>,
    /// `restore --forget-turns`: rule 9 and the marks are off for this file.
    pub keep_forget_turns: bool,
}

impl ArchivePolicy {
    /// An entry, or an entry + '_' + digits. The stub returns `false`.
    pub fn is_forget_tool(&self, name: &str) -> bool {
        let _ = name;
        false
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForgetMark {
    pub turn_key: Option<String>,
    pub head: bool,
    pub tail: bool,
}

/// `lstat` only; opens nothing.
pub fn list_project(dir: &Path) -> Vec<ArchiveEntry> {
    let _ = dir;
    Vec::new()
}

pub fn read_archive(
    path: &Path,
    scope_dir: &str,
    expect_id: Option<&str>,
) -> Result<ArchiveFile, SkipReason> {
    let _ = (path, scope_dir, expect_id);
    Err(SkipReason("not built yet".to_string()))
}

pub fn extract_units(a: &LiveArchive, policy: &ArchivePolicy) -> Vec<Unit> {
    let _ = (a, policy);
    Vec::new()
}

pub fn unit_id(scope_dir: &str, archive_id: &str, u: &Unit) -> String {
    let _ = (scope_dir, archive_id, u);
    String::new()
}

/// hex(sha256("axel/turn/3" || 0 || normalise(prompt)))[..16]. Callers pass the WHOLE prompt
/// after `screen::redact`, before any cut (B-2f).
pub fn turn_key(prompt: &str) -> String {
    let _ = prompt;
    String::new()
}

/// Rule 10 of section 2.2. The stub returns an empty list.
pub fn forgotten_keys(a: &LiveArchive, policy: &ArchivePolicy) -> Vec<ForgetMark> {
    let _ = (a, policy);
    Vec::new()
}

/// The position in `a.messages` of the last message that yields nothing (the carried tail);
/// `None` when the file does not start with copies. Decided from `a.carried`, then
/// `a.start`, then the first `source_index` (4.1). The stub returns `None`.
pub fn carried_head(a: &LiveArchive) -> Option<usize> {
    let _ = a;
    None
}

/// Trim, then every run of whitespace becomes one space. The stub returns its input.
pub fn normalise(s: &str) -> String {
    s.to_string()
}
