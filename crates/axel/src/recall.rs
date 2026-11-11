//! Recall (section 3): the candidates, the adapters into JR's types, the floors, the trace,
//! the `before_message` hook and the recall log.
//!
//! Two rules stand above everything else here. A caller gets rows of its own view only: the
//! scope is in the SQL of every generator, inside the scan of the vector array, and checked
//! once more on every row read. And nothing is shown from memory: every key the array
//! returns is resolved against its row, and a key with no visible row is dropped.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use chrono::{DateTime, NaiveDateTime, SecondsFormat, Utc};
use rusqlite::{named_params, params, Connection, Row};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::chat_history::{
    class_weight, recency_score, session_tag, RecallCandidate, CANDIDATES_RETAINED_CAP,
    CLASS_CONVERSATION_SUMMARY, CLASS_EPISODIC_TURN, CLASS_TOOL_OUTCOME,
};
use crate::embed::EmbedService;
use crate::error::Result;
use crate::ext::config::ExtConfig;
use crate::ext::types::{Caller, FrontState, HookEvent, HookOut, IndexerCtx, Job, Mode};
use crate::project::ProjectRef;
use crate::project_memory::{RecordLinks, Retention, ScopedDescriptor, Sensitivity};
use crate::retrieval::{
    rank_and_select_hybrid, score_parts, EmbeddingRankError, EmbeddingRanking, LocalEmbeddingRanker,
    RankReason, ScoreParts,
};
use crate::store::GLOBAL_KEY;
use crate::vindex::{ExactIndex, VecKey, VecScope};
use crate::{archive, brain_v2, inject, project, screen};

/// What this caller may see (section 5 fills it). Built by `front::view_for` from
/// `session_id` and `cwd` alone.
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

/// The five values of the search tool's `kinds`.
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

/// The six values of `memory_ext.origin_kind`, and one for a row of `documents`.
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

impl Origin {
    /// The value of `memory_ext.origin_kind`; `document` for a row of `documents`.
    pub fn as_str(self) -> &'static str {
        match self {
            Origin::Archive => "archive",
            Origin::NoteFile => "note_file",
            Origin::Capture => "capture",
            Origin::Tool => "tool",
            Origin::Cli => "cli",
            Origin::Legacy => "legacy",
            Origin::Document => "document",
        }
    }

    fn parse(origin_kind: &str) -> Option<Origin> {
        Some(match origin_kind {
            "archive" => Origin::Archive,
            "note_file" => Origin::NoteFile,
            "capture" => Origin::Capture,
            "tool" => Origin::Tool,
            "cli" => Origin::Cli,
            "legacy" => Origin::Legacy,
            _ => return None,
        })
    }
}

// SPEC: 3215 and 3304-3310 name `skipped:<why>`, `nothing_relevant` and `timeout`; no line
// lists the rest. `recall` answers `selected` (it has picks), `nothing_relevant`, `timeout`
// or `error` (a read of the brain failed: no picks). The hook turns `selected` into
// `injected`, or into `over_budget` when no item fits the block, and its skips are
// `skipped:off`, `skipped:breaker`, `skipped:not_ready`, `skipped:event`,
// `skipped:continuation`, `skipped:short` and `skipped:paused`.
/// The text of `recall_log.outcome`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome(pub String);

/// The clock, the floors and the limits of one recall. Built with `from_ext`, never as a
/// literal (commits 18a and 18b).
#[derive(Clone, Debug)]
pub struct RecallConfig {
    /// Milliseconds.
    pub now: i64,
    /// The floors of an ambient recall (config `recall_min_sem`, `recall_min_lex`).
    pub min_sem: f64,
    pub min_lex: f64,
    /// The limits of the injection block (config `recall_max_items`, `recall_max_chars`).
    pub max_items: usize,
    pub max_chars: usize,
    /// Config `recall_exclude_paths`: a document under one of these is never a candidate.
    pub exclude_paths: Vec<PathBuf>,
    /// `Status.model_loaded`, set by the caller. `from_ext` leaves it false, and with it
    /// false no `embed_query` is made: recall runs on words alone.
    pub model_loaded: bool,
    // SPEC: 3181 shows a file's folder as `~/Jawz/mikoshi/Notes`. `ExtConfig` does not keep
    // the home folder it was read with, so this is `$HOME` when the process has one; with
    // none (the host starts the extension with an empty environment) the folder is shown
    // as its absolute path. Display only: nothing is found or scoped by it.
    /// What `~` stands for in a label.
    pub home: Option<PathBuf>,
}

impl RecallConfig {
    pub fn from_ext(cfg: &ExtConfig, now_ms: i64) -> RecallConfig {
        RecallConfig {
            now: now_ms,
            min_sem: cfg.recall_min_sem,
            min_lex: cfg.recall_min_lex,
            max_items: cfg.recall_max_items,
            max_chars: cfg.recall_max_chars,
            exclude_paths: cfg.recall_exclude_paths.clone(),
            model_loaded: false,
            home: std::env::var_os("HOME").filter(|h| !h.is_empty()).map(PathBuf::from),
        }
    }
}

/// One selected item.
///
/// `descriptor.id` is the row: the memory's id, or `doc:<doc_id>`. `r#ref` is what a label
/// shows: `doc:<doc_id>`, `ctx-<archive id>` for a row read from an archive, the note's own
/// `mem-…` id for a Synaps note, `m:<id>` for everything else. `descriptor.title` and
/// `descriptor.snippet` have passed `screen::show`; every other string of the descriptor is
/// as stored and must pass `show` before a model reads it (`inject::who`, `inject::label`
/// and `inject::format_recall` do that).
#[derive(Clone, Debug)]
pub struct Pick {
    pub n: usize,
    pub r#ref: String,
    pub descriptor: ScopedDescriptor,
    pub reasons: Vec<RankReason>,
    pub parts: ScoreParts,
    pub origin: Origin,
    /// Milliseconds; `None` when the row's time is only a file's modification time, or
    /// not known.
    pub date: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct RecallOutcome {
    pub picks: Vec<Pick>,
    pub outcome: Outcome,
    pub trace: RecallTrace,
}

/// The `recall_log` columns, then the fields of the `detail` JSON of section 3.
///
/// `recall` leaves `query_preview` and `query_tokens` empty: whether a prompt's words may be
/// kept is the caller's to decide (`on_before_message` does, by the rule of section 3).
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

/// `r#ref` is always `m:<memory id>` or `doc:<doc_id>`, whatever the label shows:
/// `store::purge_many` finds the log rows of a purged memory by that text.
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

// ── The numbers of section 3 ────────────────────────────────────────────────

/// The query is embedded from its first 1,000 chars.
const EMBED_CHARS: usize = 1_000;
const MAX_TOKENS: usize = 12;
const MIN_TOKEN_CHARS: usize = 3;
/// How long the front thread waits for the query's vector.
const EMBED_WAIT: Duration = Duration::from_millis(80);
const MEM_WORDS_LIMIT: i64 = 48;
const DOC_WORDS_LIMIT: i64 = 24;
const BY_MEANING: usize = 64;
/// A passage, and `inject`'s cut of it.
pub const PASSAGE_CHARS: usize = 240;
const DROPPED_KEPT: usize = 20;
const PREVIEW_CHARS: usize = 120;
/// The front thread works to a deadline this much before the rpc thread's.
const HEADROOM: Duration = Duration::from_millis(20);
const LOG_ROWS: i64 = 1_000;
const LOG_AGE_MS: i64 = 30 * 24 * 60 * 60 * 1000;
/// How long the answer to "where are this chat's rows filed" is kept, per session.
const FILED_KEEP_MS: i64 = 60_000;
const FILED_SESSIONS: usize = 256;

// SPEC: 3136-3140, 3160: a note file is stored whole and nothing bounds what recall reads
// of one. The words and the passage are looked for in the first 64 KiB of a body, so that
// a few very large files cannot take the budget.
const BODY_SCAN_BYTES: usize = 64 * 1024;

// SPEC: 3081 says "a 60-word stop list" and gives no words. These are the sixty.
const STOP_WORDS: [&str; 60] = [
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was", "one",
    "our", "out", "has", "him", "his", "how", "its", "may", "now", "did", "get", "got", "let", "who",
    "why", "use", "with", "that", "this", "from", "have", "will", "your", "what", "when", "where",
    "which", "there", "their", "them", "then", "they", "been", "were", "would", "could", "should",
    "about", "into", "than", "does", "just", "also", "some", "such", "please",
];

// ── The query ───────────────────────────────────────────────────────────────

/// The query tokens of a text that has ALREADY passed `screen::redact`: lowercase
/// alphanumeric runs of 3 or more chars, minus the stop list, de-duplicated, at most 12,
/// longest first (equal lengths in the order they were written).
pub fn tokens(redacted: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for run in redacted.split(|c: char| !c.is_alphanumeric()) {
        if run.chars().count() < MIN_TOKEN_CHARS {
            continue;
        }
        let token = run.to_lowercase();
        // What `redact` writes in place of a credential is not a word of the prompt.
        if token == "redacted" || STOP_WORDS.contains(&token.as_str()) || out.contains(&token) {
            continue;
        }
        out.push(token);
    }
    out.sort_by_key(|t| std::cmp::Reverse(t.chars().count()));
    out.truncate(MAX_TOKENS);
    out
}

/// `"t1" OR "t2" OR …`: alphanumeric tokens in double quotes and nothing else.
fn fts_expr(tokens: &[String]) -> String {
    tokens
        .iter()
        .filter(|t| t.chars().all(char::is_alphanumeric))
        .map(|t| format!("\"{t}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn first_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((at, _)) => &s[..at],
        None => s,
    }
}

fn first_bytes(s: &str, n: usize) -> &str {
    if s.len() <= n {
        return s;
    }
    let mut at = n;
    while !s.is_char_boundary(at) {
        at -= 1;
    }
    &s[..at]
}

fn ms_between(from: Instant, to: Instant) -> f64 {
    to.saturating_duration_since(from).as_secs_f64() * 1000.0
}

fn time_of(ms: i64) -> Option<DateTime<Utc>> {
    DateTime::<Utc>::from_timestamp_millis(ms)
}

// ── Rows ────────────────────────────────────────────────────────────────────

const MEM_COLUMNS: &str = "m.id, m.project_key, m.title, m.content, m.category, m.tags, m.created,
            m.sensitivity, m.retention, m.provenance, m.supersedes, m.contradicts,
            m.confirmed_by, m.invalidated_at, m.superseded_by,
            e.vid, e.origin_kind, e.origin_ref, e.session_id, e.content_hash,
            e.created_ms, e.time_source";

/// The conditions of generator a that are not the word match: visible, in the caller's
/// scope, not expired. With no project `:scope_dir` is NULL, which equals no row's key.
const MEM_VISIBLE: &str = "e.hidden_ms IS NULL
       AND (m.project_key = :scope_dir
            OR (:global AND (m.project_key = 'p0000000000000000' OR m.project_key IS NULL)))
       AND (m.expires_at IS NULL OR m.expires_at > :now)";

const DOC_COLUMNS: &str = "d.id, d.doc_id, d.content, d.file_path, d.metadata, d.created, d.source_mtime_ms";

struct MemRow {
    id: String,
    project_key: Option<String>,
    title: String,
    content: String,
    category: String,
    tags: Vec<String>,
    sensitivity: Sensitivity,
    retention: Retention,
    provenance: String,
    links: RecordLinks,
    vid: i64,
    origin: Origin,
    origin_ref: String,
    session_id: Option<String>,
    content_hash: String,
    created_ms: i64,
    time_source: String,
}

struct DocRow {
    id: i64,
    doc_id: String,
    content: String,
    file_path: String,
    metadata: String,
    created: Option<String>,
    source_mtime_ms: Option<i64>,
}

fn list(json: Option<String>) -> Vec<String> {
    json.and_then(|j| serde_json::from_str(&j).ok()).unwrap_or_default()
}

/// A row that cannot be read as these types is `None`: it is skipped, not guessed at.
fn mem_row(r: &Row) -> Option<MemRow> {
    fn read(r: &Row) -> rusqlite::Result<MemRow> {
        let sensitivity: String = r.get(7)?;
        let retention: String = r.get(8)?;
        let invalidated_at: Option<String> = r.get(13)?;
        let origin_kind: String = r.get(16)?;
        Ok(MemRow {
            id: r.get(0)?,
            project_key: r.get(1)?,
            title: r.get(2)?,
            content: r.get(3)?,
            category: r.get(4)?,
            tags: list(r.get(5)?),
            // An unknown value is read as the closed one, as `project_memory` reads it.
            sensitivity: Sensitivity::parse(&sensitivity).unwrap_or(Sensitivity::Secret),
            retention: Retention::parse(&retention).unwrap_or(Retention::LocalOnly),
            provenance: r.get(9)?,
            links: RecordLinks {
                supersedes: list(r.get(10)?),
                contradicts: list(r.get(11)?),
                confirmed_by: list(r.get(12)?),
                // A value that is there and cannot be read still means "invalidated".
                invalidated_at: invalidated_at.map(|v| {
                    DateTime::parse_from_rfc3339(&v).map(|t| t.with_timezone(&Utc)).unwrap_or(DateTime::UNIX_EPOCH)
                }),
                superseded_by: r.get(14)?,
            },
            vid: r.get(15)?,
            origin: Origin::parse(&origin_kind).ok_or(rusqlite::Error::InvalidQuery)?,
            origin_ref: r.get(17)?,
            session_id: r.get(18)?,
            content_hash: r.get(19)?,
            created_ms: r.get(20)?,
            time_source: r.get(21)?,
        })
    }
    read(r).ok()
}

fn doc_row(r: &Row) -> Option<DocRow> {
    fn read(r: &Row) -> rusqlite::Result<DocRow> {
        Ok(DocRow {
            id: r.get(0)?,
            doc_id: r.get(1)?,
            content: r.get(2)?,
            file_path: r.get::<_, Option<String>>(3)?.unwrap_or_default(),
            metadata: r.get(4)?,
            created: r.get(5)?,
            source_mtime_ms: r.get(6)?,
        })
    }
    read(r).ok()
}

/// What the scope conditions are bound to.
struct Scope<'a> {
    /// The view's project directory; `None` binds NULL.
    dir: Option<&'a str>,
    global: bool,
    /// `:now`, as `memories.expires_at` is written.
    now: String,
}

/// Generator a: memory, by words.
fn mem_by_words(conn: &Connection, expr: &str, scope: &Scope) -> rusqlite::Result<Vec<MemRow>> {
    let sql = format!(
        "SELECT {MEM_COLUMNS}
     FROM memories_fts f JOIN memories m ON m.id = f.mem_id
                         JOIN memory_ext e ON e.mem_id = m.id
     WHERE memories_fts MATCH :expr
       AND {MEM_VISIBLE}
     ORDER BY bm25(memories_fts) LIMIT {MEM_WORDS_LIMIT}"
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let mut rows = stmt.query(named_params! {
        ":expr": expr, ":scope_dir": scope.dir, ":global": scope.global, ":now": scope.now,
    })?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        out.extend(mem_row(r));
    }
    Ok(out)
}

/// A key of the vector array, resolved to its row with the conditions of generator a.
/// `None`: the row is gone, hidden, expired or not in the caller's scope.
fn mem_by_vid(conn: &Connection, vid: i64, scope: &Scope) -> rusqlite::Result<Option<MemRow>> {
    let sql = format!(
        "SELECT {MEM_COLUMNS}
     FROM memories m JOIN memory_ext e ON e.mem_id = m.id
     WHERE e.vid = :vid
       AND {MEM_VISIBLE}"
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let mut rows = stmt.query(named_params! {
        ":vid": vid, ":scope_dir": scope.dir, ":global": scope.global, ":now": scope.now,
    })?;
    Ok(match rows.next()? {
        Some(r) => mem_row(r),
        None => None,
    })
}

/// Generator c: note files, by words.
fn docs_by_words(conn: &Connection, expr: &str) -> rusqlite::Result<Vec<DocRow>> {
    let sql = format!(
        "SELECT {DOC_COLUMNS}
     FROM chunks_fts f JOIN documents d ON d.doc_id = f.doc_id
     WHERE chunks_fts MATCH 'content : (' || :expr || ')' AND d.hidden IS NULL
     ORDER BY bm25(chunks_fts) LIMIT {DOC_WORDS_LIMIT}"
    );
    let mut stmt = conn.prepare_cached(&sql)?;
    let mut rows = stmt.query(named_params! { ":expr": expr })?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        out.extend(doc_row(r));
    }
    Ok(out)
}

fn doc_by_id(conn: &Connection, id: i64) -> rusqlite::Result<Option<DocRow>> {
    let sql = format!("SELECT {DOC_COLUMNS} FROM documents d WHERE d.id = ?1 AND d.hidden IS NULL");
    let mut stmt = conn.prepare_cached(&sql)?;
    let mut rows = stmt.query(params![id])?;
    Ok(match rows.next()? {
        Some(r) => doc_row(r),
        None => None,
    })
}

// ── Candidates ──────────────────────────────────────────────────────────────

enum Source {
    Mem(Box<MemRow>),
    Doc(DocRow),
}

struct Cand {
    source: Source,
    /// `ScopedDescriptor.id`: the memory's id, or `doc:<doc_id>`.
    id: String,
    /// Share of the query tokens found in the title and body.
    lexical: f64,
    /// Dot product of the query and the row's vector, 0..1; 0 without either.
    semantic: f64,
    /// Milliseconds; `None` for a time that is only a file's modification time.
    date: Option<i64>,
}

impl Cand {
    /// What duplicates and the cut at 128 are decided by.
    fn base(&self) -> f64 {
        2.0 * self.lexical + 2.5 * self.semantic
    }

    /// `m:<id>` or `doc:<doc_id>`.
    fn log_ref(&self) -> String {
        match &self.source {
            Source::Mem(m) => format!("m:{}", m.id),
            Source::Doc(_) => self.id.clone(),
        }
    }

    /// Documents by `file_path` (else by a hash of the content), memory rows by
    /// `content_hash`.
    fn duplicate_key(&self) -> String {
        match &self.source {
            Source::Mem(m) if m.content_hash.is_empty() => format!("i:{}", m.id),
            Source::Mem(m) => format!("m:{}", m.content_hash),
            Source::Doc(d) if d.file_path.is_empty() => {
                format!("h:{}", hex::encode(Sha256::digest(d.content.as_bytes())))
            }
            Source::Doc(d) => format!("f:{}", d.file_path),
        }
    }
}

/// Best first: the higher `base`, then the newer, then the id.
fn best_first(a: &Cand, b: &Cand) -> std::cmp::Ordering {
    b.base().total_cmp(&a.base()).then_with(|| b.date.cmp(&a.date)).then_with(|| a.id.cmp(&b.id))
}

/// `documents.created` is SQLite's `CURRENT_TIMESTAMP` (UTC, no zone) or an RFC 3339 time.
fn parse_created(created: &str) -> Option<i64> {
    if let Ok(t) = DateTime::parse_from_rfc3339(created) {
        return Some(t.timestamp_millis());
    }
    NaiveDateTime::parse_from_str(created, "%Y-%m-%d %H:%M:%S").ok().map(|t| t.and_utc().timestamp_millis())
}

fn doc_title(doc_id: &str) -> &str {
    doc_id.rsplit("::").next().unwrap_or(doc_id)
}

/// The share of `tokens` that occur in the lowercased title and body.
fn lexical_share(tokens: &[String], title: &str, body: &str) -> f64 {
    if tokens.is_empty() {
        return 0.0;
    }
    let mut hay = title.to_lowercase();
    hay.push('\n');
    hay.push_str(&first_bytes(body, BODY_SCAN_BYTES).to_lowercase());
    let found = tokens.iter().filter(|t| hay.contains(t.as_str())).count();
    found as f64 / tokens.len() as f64
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum()
}

/// The whole of one request that every generator needs.
struct Search<'a> {
    cfg: &'a RecallConfig,
    req: &'a RecallRequest<'a>,
    /// The view's project directory, when it has one that can name rows.
    project: Option<&'a str>,
    tokens: &'a [String],
    q_vec: Option<&'a [f32]>,
    index: &'a ExactIndex,
}

impl Search<'_> {
    /// The scope once more, on the row as read: a row of another project is never a
    /// candidate, whatever returned it.
    fn in_scope(&self, project_key: Option<&str>) -> bool {
        match project_key {
            None => self.req.view.global,
            Some(key) if key == GLOBAL_KEY => self.req.view.global,
            Some(key) => self.project == Some(key),
        }
    }

    fn kind_wanted(&self, of: impl Fn(Kind) -> bool) -> bool {
        match self.req.kinds {
            None => true,
            Some(kinds) => kinds.iter().any(|k| of(*k)),
        }
    }

    fn since_ok(&self, date: Option<i64>) -> bool {
        match self.req.since_ms {
            None => true,
            // A row whose time is not known is not "since" anything.
            Some(since) => date.is_some_and(|d| d >= since),
        }
    }

    fn semantic(&self, key: VecKey) -> f64 {
        match (self.q_vec, self.index.get(&key)) {
            (Some(q), Some(v)) => {
                let s = dot(q, v);
                if s.is_finite() {
                    s.clamp(0.0, 1.0)
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// A memory row as a candidate; `None` for a row that is never one (section 3, step 3).
    fn mem(&self, m: MemRow) -> Option<Cand> {
        if !self.in_scope(m.project_key.as_deref()) {
            return None;
        }
        // JR's gate: secret rows, and rows whose retention is not model-visible.
        if m.sensitivity != Sensitivity::Normal || !m.retention.model_visible() {
            return None;
        }
        if m.links.superseded_by.is_some() || m.links.invalidated_at.is_some() {
            return None;
        }
        if self.req.ambient && m.tags.iter().any(|t| t == "sensitive") {
            return None;
        }
        let wanted = self.kind_wanted(|k| match k {
            Kind::File => false,
            Kind::Note => m.origin == Origin::NoteFile,
            Kind::Chat => m.category == CLASS_EPISODIC_TURN || m.category == CLASS_CONVERSATION_SUMMARY,
            Kind::WorkerReport => m.category == CLASS_TOOL_OUTCOME,
            Kind::Saved => matches!(m.origin, Origin::Tool | Origin::Cli | Origin::Legacy),
        });
        let date = (m.time_source != "mtime").then_some(m.created_ms);
        if !wanted || !self.since_ok(date) {
            return None;
        }
        Some(Cand {
            id: m.id.clone(),
            lexical: lexical_share(self.tokens, &m.title, &m.content),
            semantic: self.semantic(VecKey::Mem(m.vid)),
            date,
            source: Source::Mem(Box::new(m)),
        })
    }

    /// A row of `documents` as a candidate; `None` for a shadow of a memory, a file under
    /// `recall_exclude_paths`, and when the view has no documents.
    fn doc(&self, d: DocRow) -> Option<Cand> {
        if !self.req.view.documents {
            return None;
        }
        // The shadow of a memory: the memory row is the record. Metadata that cannot be
        // read cannot say it is not one.
        let metadata: serde_json::Value = serde_json::from_str(&d.metadata).ok()?;
        if metadata.get("type").and_then(|t| t.as_str()) == Some("memory") {
            return None;
        }
        let excluded = self.cfg.exclude_paths.iter().any(|p| {
            let p = p.to_string_lossy();
            !p.is_empty() && d.file_path.starts_with(p.as_ref())
        });
        let date = d.source_mtime_ms.or_else(|| d.created.as_deref().and_then(parse_created));
        if excluded || !self.kind_wanted(|k| k == Kind::File) || !self.since_ok(date) {
            return None;
        }
        Some(Cand {
            id: format!("doc:{}", d.doc_id),
            lexical: lexical_share(self.tokens, doc_title(&d.doc_id), &d.content),
            semantic: self.semantic(VecKey::Doc(d.id)),
            date,
            source: Source::Doc(d),
        })
    }
}

// ── Passages ────────────────────────────────────────────────────────────────

fn has_token(text: &str, tokens: &[String]) -> usize {
    let lower = text.to_lowercase();
    tokens.iter().filter(|t| lower.contains(t.as_str())).count()
}

/// The text from the first line that contains a query token, when there is one.
fn from_first_match<'a>(body: &'a str, tokens: &[String]) -> &'a str {
    let mut at = 0;
    for line in body.split_inclusive('\n') {
        if has_token(line, tokens) > 0 {
            return &body[at..];
        }
        at += line.len();
    }
    body
}

/// A document: the 240-char window with the most query tokens, sliding by lines; ties to
/// the earliest.
fn doc_passage(body: &str, tokens: &[String]) -> String {
    let body = first_bytes(body, BODY_SCAN_BYTES);
    let (mut best_count, mut best_at) = (0, 0);
    let mut at = 0;
    for line in body.split_inclusive('\n') {
        if !line.trim().is_empty() {
            let count = has_token(first_chars(&body[at..], PASSAGE_CHARS), tokens);
            if count > best_count {
                (best_count, best_at) = (count, at);
                if count == tokens.len() {
                    break;
                }
            }
        }
        at += line.len();
    }
    first_chars(body[best_at..].trim_start(), PASSAGE_CHARS).to_string()
}

/// A memory row: the first 240 chars of the body, from the first line that contains a
/// query token when there is one. An exchange keeps both halves: `User: … | Assistant: …`.
fn mem_passage(category: &str, body: &str, tokens: &[String]) -> String {
    let body = first_bytes(body, BODY_SCAN_BYTES);
    if category == CLASS_EPISODIC_TURN {
        const REPLY: &str = "\nAssistant:";
        if let (Some(rest), Some(at)) = (body.strip_prefix("User:"), body.find(REPLY)) {
            let prompt = rest[..at - "User:".len()].trim();
            let reply = from_first_match(body[at + REPLY.len()..].trim(), tokens);
            // SPEC: 3183 shows both halves in one item of 240 chars and does not say how
            // they share them: the prompt gets 100, the reply the rest.
            return format!("User: {} | Assistant: {}", first_chars(prompt, 100), first_chars(reply.trim(), 120));
        }
    }
    first_chars(from_first_match(body, tokens).trim_start(), PASSAGE_CHARS).to_string()
}

// ── The ranker's adapter ────────────────────────────────────────────────────

/// `LocalEmbeddingRanker` over scores that are already computed: it does no inference.
struct Precomputed(Vec<f64>);

impl LocalEmbeddingRanker for Precomputed {
    fn is_available(&self) -> bool {
        true
    }

    fn semantic_scores(
        &mut self,
        _query: &str,
        candidates: &[RecallCandidate],
        _timeout: Duration,
    ) -> std::result::Result<Vec<f64>, EmbeddingRankError> {
        if self.0.len() == candidates.len() {
            Ok(self.0.clone())
        } else {
            Err(EmbeddingRankError::Failed)
        }
    }
}

fn reason_text(reason: &RankReason) -> String {
    match reason {
        RankReason::LexicalMatch => "lexical_match".to_string(),
        RankReason::SemanticMatch => "semantic_match".to_string(),
        RankReason::ExactTagMatch => "exact_tag_match".to_string(),
        RankReason::RecentSession => "recent_session".to_string(),
        RankReason::RecencyWindow => "recency_window".to_string(),
        RankReason::ExplicitPreference => "explicit_preference".to_string(),
        RankReason::ExplicitDecision => "explicit_decision".to_string(),
        RankReason::UnresolvedTaskState => "unresolved_task_state".to_string(),
        RankReason::ClassBoost(class) => format!("class_boost:{class}"),
    }
}

/// JR's candidate for one of ours, and whether it has this chat's affinity.
fn adapt(c: &Cand, own_session: Option<&str>, now: DateTime<Utc>) -> RecallCandidate {
    // A row whose time is a file's modification time ranks as if it had no date.
    let created = c.date.and_then(time_of).map(|t| t.to_rfc3339()).unwrap_or_default();
    let recency = if created.is_empty() { 0.0 } else { recency_score(&created, now) };
    let (descriptor, session_affinity) = match &c.source {
        Source::Mem(m) => {
            let tag = own_session.map(session_tag);
            let affinity = own_session.is_some()
                && (m.session_id.as_deref() == own_session || m.tags.iter().any(|t| Some(t) == tag.as_ref()));
            let descriptor = ScopedDescriptor {
                id: m.id.clone(),
                project_key: m.project_key.clone().unwrap_or_default(),
                title: m.title.clone(),
                snippet: None,
                category: m.category.clone(),
                tags: m.tags.clone(),
                created,
                sensitivity: m.sensitivity,
                retention: m.retention,
                provenance: m.provenance.clone(),
                links: m.links.clone(),
            };
            (descriptor, affinity)
        }
        Source::Doc(d) => {
            let source = d.doc_id.split("::").next().unwrap_or_default();
            let descriptor = ScopedDescriptor {
                id: c.id.clone(),
                project_key: GLOBAL_KEY.to_string(),
                title: doc_title(&d.doc_id).to_string(),
                snippet: None,
                category: "document".to_string(),
                tags: vec![format!("source:{source}")],
                created,
                sensitivity: Sensitivity::Normal,
                retention: Retention::Standard,
                provenance: d.file_path.clone(),
                links: RecordLinks::default(),
            };
            (descriptor, false)
        }
    };
    RecallCandidate {
        recency,
        session_affinity,
        class_weight: class_weight(&descriptor.category),
        lexical_match: c.lexical > 0.0,
        recency_fallback: false,
        lexical_score: Some(c.lexical),
        descriptor,
    }
}

fn is_archive_id(s: &str) -> bool {
    s.len() == 32 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// What a label shows for this row.
fn shown_ref(c: &Cand) -> String {
    match &c.source {
        Source::Doc(_) => c.id.clone(),
        Source::Mem(m) => match m.origin {
            Origin::Archive if is_archive_id(&m.origin_ref) => format!("ctx-{}", m.origin_ref),
            Origin::NoteFile => m.id.clone(),
            _ => format!("m:{}", m.id),
        },
    }
}

// ── The entry point ─────────────────────────────────────────────────────────

fn finish(mut trace: RecallTrace, started: Instant, picks: Vec<Pick>, outcome: &str) -> RecallOutcome {
    trace.outcome = outcome.to_string();
    trace.selected = picks.len() as i64;
    trace.total_ms = ms_between(started, Instant::now());
    RecallOutcome { picks, outcome: Outcome(outcome.to_string()), trace }
}

/// The one entry point, called on the front thread by the hook and by `axel_search`.
///
/// It reads and never writes; it waits for nothing but the query's vector, and for that at
/// most 80 ms; it looks at `req.deadline` between its stages and inside the loop over the
/// array's hits, and past it answers `timeout` with no picks: never a part of an answer.
pub fn recall(
    conn: &Connection,
    index: &ExactIndex,
    embed: &EmbedService,
    cfg: &RecallConfig,
    req: &RecallRequest,
) -> RecallOutcome {
    let started = Instant::now();
    let view = req.view;
    let late = || Instant::now() >= req.deadline;

    // 1. The query. Redacted first: a pasted key is never a token, embedded or logged.
    let redacted = screen::redact(req.query);
    let query: &str = &redacted.text;
    let q_tokens = tokens(query);
    // A directory that could also name the global rows names no project.
    let project = view.project.as_ref().map(|p| p.scope_dir.as_str()).filter(|d| !d.is_empty() && *d != GLOBAL_KEY);
    let mut trace = RecallTrace {
        ts_ms: cfg.now,
        session_id: view.own_session.clone(),
        logical: view.own_session.as_deref().map(project::logical),
        scope_dir: project.map(str::to_string),
        query_hash: archive::turn_key(query),
        thresholds: TraceThresholds { min_sem: cfg.min_sem, min_lex: cfg.min_lex },
        ..RecallTrace::default()
    };
    if late() {
        return finish(trace, started, Vec::new(), "timeout");
    }

    // 2. The query's vector. Never asked for without a loaded model of the brain's own.
    let q_embed = first_chars(query, EMBED_CHARS);
    let mut q_vec: Option<Vec<f32>> = None;
    if !cfg.model_loaded {
        trace.embed.fallback = Some("model_unavailable".to_string());
    } else if !q_embed.trim().is_empty() {
        let asked = Instant::now();
        let until = req.deadline.min(asked + EMBED_WAIT);
        q_vec = embed.embed_query(q_embed, until);
        let answered = Instant::now();
        let ms = ms_between(asked, answered);
        trace.embed.ms = Some(ms);
        trace.embed_ms = Some(ms);
        match &q_vec {
            Some(_) => trace.embed.used = true,
            None if answered >= until => trace.embed.fallback = Some("timed_out".to_string()),
            None => trace.embed.fallback = Some("model_unavailable".to_string()),
        }
        if late() {
            return finish(trace, started, Vec::new(), "timeout");
        }
    }

    // 3. Candidates, each generator bounded.
    let search = Search { cfg, req, project, tokens: &q_tokens, q_vec: q_vec.as_deref(), index };
    let scope = Scope {
        dir: project,
        global: view.global,
        now: time_of(cfg.now).unwrap_or(DateTime::UNIX_EPOCH).to_rfc3339(),
    };
    let expr = fts_expr(&q_tokens);
    let mut cands: Vec<Cand> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut add = |cands: &mut Vec<Cand>, c: Option<Cand>| {
        if let Some(c) = c {
            if seen.insert(c.id.clone()) {
                cands.push(c);
            }
        }
    };

    // a. Memory, by words.
    if !expr.is_empty() {
        let rows = match mem_by_words(conn, &expr, &scope) {
            Ok(rows) => rows,
            Err(_) => return finish(trace, started, Vec::new(), "error"),
        };
        trace.generators.mem_words = rows.len();
        for m in rows {
            add(&mut cands, search.mem(m));
        }
        if late() {
            return finish(trace, started, Vec::new(), "timeout");
        }
    }

    // b. By meaning. The scope filter is inside the scan; and because the array can hold
    // the vector of a row that is hidden or gone, every key is resolved to its row.
    if let Some(q) = q_vec.as_deref() {
        let allow = |s: &VecScope| match s {
            VecScope::Project(dir) => project == Some(dir.as_str()),
            VecScope::Global => view.global,
            VecScope::Documents => view.documents,
        };
        for (key, score) in index.search(q, BY_MEANING, &allow) {
            // A hit at or below 0 is not a hit (a zero query returns every row at 0.0).
            if score.is_nan() || score <= 0.0 {
                continue;
            }
            if late() {
                return finish(trace, started, Vec::new(), "timeout");
            }
            let found = match key {
                VecKey::Mem(vid) => mem_by_vid(conn, vid, &scope).map(|m| m.and_then(|m| search.mem(m))),
                VecKey::Doc(id) if view.documents => doc_by_id(conn, id).map(|d| d.and_then(|d| search.doc(d))),
                VecKey::Doc(_) => Ok(None),
            };
            match found {
                Ok(Some(c)) => {
                    trace.generators.by_meaning += 1;
                    add(&mut cands, Some(c));
                }
                // No such row: the key is dropped.
                Ok(None) => {}
                Err(_) => return finish(trace, started, Vec::new(), "error"),
            }
        }
    }

    // c. Note files, by words.
    if view.documents && !expr.is_empty() {
        let rows = match docs_by_words(conn, &expr) {
            Ok(rows) => rows,
            Err(_) => return finish(trace, started, Vec::new(), "error"),
        };
        trace.generators.doc_words = rows.len();
        for d in rows {
            add(&mut cands, search.doc(d));
        }
    }
    if late() {
        return finish(trace, started, Vec::new(), "timeout");
    }
    trace.candidates = cands.len() as i64;

    let mut dropped: Vec<(f64, TraceDropped)> = Vec::new();
    let mut note_drop = |c: &Cand, why: &str| {
        dropped.push((c.base(), TraceDropped { r#ref: c.log_ref(), why: why.to_string() }));
    };

    // 4. Still in the window: this chat's own live captures are in its context already.
    if req.ambient && view.own_session.is_some() {
        cands.retain(|c| {
            let own = matches!(&c.source, Source::Mem(m)
                if m.origin == Origin::Capture && m.session_id.as_deref() == view.own_session.as_deref());
            if own {
                note_drop(c, "in_window");
            }
            !own
        });
    }

    // 6. The floor (step 5, the signals, was made with each candidate).
    if req.ambient {
        cands.retain(|c| {
            let keep = c.semantic >= cfg.min_sem || c.lexical >= cfg.min_lex;
            if !keep {
                note_drop(c, "below_floor");
            }
            keep
        });
    }
    trace.generators.past_floor = cands.len();

    // 7. Collapse duplicates: best first, so the first of each key is the one that stays.
    cands.sort_by(best_first);
    let mut keys: HashSet<String> = HashSet::new();
    cands.retain(|c| {
        let first = keys.insert(c.duplicate_key());
        if !first {
            note_drop(c, "duplicate");
        }
        first
    });
    trace.generators.after_dedupe = cands.len();

    // 8. Adapt and rank: at most 128 go in, best first.
    for c in cands.iter().skip(CANDIDATES_RETAINED_CAP) {
        note_drop(c, "not_selected");
    }
    cands.truncate(CANDIDATES_RETAINED_CAP);
    let now = time_of(cfg.now).unwrap_or(DateTime::UNIX_EPOCH);
    let own_session = view.own_session.as_deref();
    let adapted: Vec<RecallCandidate> = cands.iter().map(|c| adapt(c, own_session, now)).collect();
    let by_id: HashMap<&str, usize> = cands.iter().enumerate().map(|(i, c)| (c.id.as_str(), i)).collect();
    let mut scores = Precomputed(cands.iter().map(|c| c.semantic).collect());
    let selection = rank_and_select_hybrid(
        adapted.clone(),
        own_session,
        req.max_items,
        query,
        EmbeddingRanking::enabled(&mut scores),
    );
    if late() {
        return finish(trace, started, Vec::new(), "timeout");
    }

    // 9. The picks, each with its passage.
    let mut picks: Vec<Pick> = Vec::new();
    let mut picked: HashSet<usize> = HashSet::new();
    for (descriptor, reasons) in selection.records {
        let Some(&i) = by_id.get(descriptor.id.as_str()) else { continue };
        if !picked.insert(i) {
            continue;
        }
        let c = &cands[i];
        let passage = match &c.source {
            Source::Mem(m) => mem_passage(&m.category, &m.content, &q_tokens),
            Source::Doc(d) => doc_passage(&d.content, &q_tokens),
        };
        let parts = score_parts(&adapted[i], adapted[i].session_affinity, Some(c.semantic));
        let origin = match &c.source {
            Source::Mem(m) => m.origin,
            Source::Doc(_) => Origin::Document,
        };
        let n = picks.len() + 1;
        trace.picks.push(TracePick {
            n,
            r#ref: c.log_ref(),
            kind: descriptor.category.clone(),
            origin: origin.as_str().to_string(),
            scope: match &c.source {
                Source::Doc(_) => "documents",
                Source::Mem(m) if m.project_key.as_deref().is_some_and(|k| k != GLOBAL_KEY) => "project",
                Source::Mem(_) => "global",
            }
            .to_string(),
            created: c.date.and_then(time_of).map(|t| t.to_rfc3339_opts(SecondsFormat::Secs, true)),
            parts,
            reasons: reasons.iter().map(reason_text).collect(),
        });
        picks.push(Pick {
            n,
            r#ref: shown_ref(c),
            descriptor: ScopedDescriptor {
                title: screen::show(&descriptor.title, PASSAGE_CHARS),
                snippet: Some(screen::show(&passage, PASSAGE_CHARS)),
                ..descriptor
            },
            reasons,
            parts,
            origin,
            date: c.date,
        });
    }
    for (i, c) in cands.iter().enumerate() {
        if !picked.contains(&i) {
            note_drop(c, "not_selected");
        }
    }

    // The trace keeps at most 20 of what was dropped, best first.
    dropped.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.r#ref.cmp(&b.1.r#ref)));
    dropped.truncate(DROPPED_KEPT);
    trace.dropped = dropped.into_iter().map(|(_, d)| d).collect();

    if late() {
        trace.picks.clear();
        return finish(trace, started, Vec::new(), "timeout");
    }
    let outcome = if picks.is_empty() { "nothing_relevant" } else { "selected" };
    finish(trace, started, picks, outcome)
}

// ── The hook ────────────────────────────────────────────────────────────────

/// The pauses of one chat: one indexed read of `chat_pauses`. `(learning, recall)`.
fn pauses(conn: &Connection, logical: &str) -> rusqlite::Result<(bool, bool)> {
    let mut stmt = conn.prepare_cached("SELECT what FROM chat_pauses WHERE logical = ?1")?;
    let mut rows = stmt.query(params![logical])?;
    let (mut learning, mut recall) = (false, false);
    while let Some(r) = rows.next()? {
        match r.get::<_, String>(0)?.as_str() {
            "learning" => learning = true,
            "recall" => recall = true,
            _ => {}
        }
    }
    Ok((learning, recall))
}

/// The project directories that already hold rows or archives of this chat (section 4,
/// "A chat filed elsewhere").
fn filed_dirs(conn: &Connection, logical: &str) -> rusqlite::Result<Vec<String>> {
    let mut stmt = conn.prepare_cached(
        "SELECT DISTINCT scope_dir FROM archive_index WHERE logical = ?1
         UNION
         SELECT DISTINCT m.project_key FROM memory_ext e JOIN memories m ON m.id = e.mem_id WHERE e.logical = ?1",
    )?;
    let rows = stmt.query_map(params![logical], |r| r.get::<_, Option<String>>(0))?;
    let mut out = Vec::new();
    for dir in rows {
        out.extend(dir?);
    }
    Ok(out)
}

/// True when this chat's rows are filed under another project than `scope_dir`: recall
/// then runs with no project. The read is kept per session for 60 s. A read that failed
/// counts as "elsewhere": global material only.
fn filed_elsewhere(st: &mut FrontState, session_id: &str, logical: &str, scope_dir: &str, now: i64) -> bool {
    let fresh = |at: i64| now >= at && now - at < FILED_KEEP_MS;
    let known = st.filed.get(session_id).filter(|(at, _)| fresh(*at)).map(|(_, dirs)| dirs.clone());
    let dirs = match known {
        Some(dirs) => dirs,
        None => {
            let dirs = st.conn.as_ref().and_then(|conn| filed_dirs(conn, logical).ok());
            if st.filed.len() >= FILED_SESSIONS {
                st.filed.retain(|_, (at, _)| fresh(*at));
                if st.filed.len() >= FILED_SESSIONS {
                    st.filed.clear();
                }
            }
            st.filed.insert(session_id.to_string(), (now, dirs.clone()));
            dirs
        }
    };
    match dirs {
        None => true,
        Some(dirs) => !dirs.is_empty() && !dirs.iter().any(|d| d == scope_dir),
    }
}

// SPEC: 3218: "fewer than 3 word tokens". A word token is read as an alphanumeric run of
// any length, not as a query token: "what is the limit" has four.
fn word_count(text: &str) -> usize {
    text.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).count()
}

/// The `before_message` hook (called by `ext::hooks::handle`): recall, then one
/// `Job::LogRecall`. It writes nothing itself.
pub fn on_before_message(st: &mut FrontState, ev: &HookEvent, recall_allowed: bool) -> HookOut {
    let started = Instant::now();
    let now = (st.clock)();
    let budget = Duration::from_millis(st.cfg.recall_budget_ms).saturating_sub(HEADROOM);
    let deadline = started + budget;

    let prompt = ev.message.as_deref().unwrap_or("");
    let redacted = screen::redact(prompt);
    let text: &str = &redacted.text;
    let session_id = ev.session_id.as_deref();
    let logical = session_id.map(project::logical);
    let ready = st.conn.is_some() && st.mode.load(Ordering::SeqCst) == Mode::Ready as u8;

    // The chat's pauses. A read that fails is read as both: nothing is recalled into the
    // chat and nothing of its prompt is kept.
    let (learning_paused, recall_paused) = match (st.conn.as_ref(), logical.as_deref()) {
        (Some(conn), Some(logical)) => pauses(conn, logical).unwrap_or((true, true)),
        _ => (false, false),
    };

    let skip = if !st.cfg.recall {
        Some("skipped:off")
    } else if !recall_allowed {
        Some("skipped:breaker")
    } else if !ready {
        Some("skipped:not_ready")
    } else if prompt.starts_with("<event ") && !st.cfg.recall_on_events {
        Some("skipped:event")
    } else if prompt.starts_with("[Context continuation") || prompt.starts_with("[Host context advisory") {
        Some("skipped:continuation")
    } else if prompt.chars().count() < 12 || word_count(text) < 3 {
        Some("skipped:short")
    } else if recall_paused {
        Some("skipped:paused")
    } else {
        None
    };

    let mut view = crate::ext::front::view_for(
        &Caller { session_id: ev.session_id.clone(), cwd: ev.cwd.clone(), author: None },
        &st.cfg,
    );
    let (mut out, mut trace) = (HookOut::Continue, None);
    if let Some(why) = skip {
        // SPEC: 3250 says every recall becomes a log row and 3215 gives a skip an outcome,
        // so a skip is logged; it keeps no preview and no tokens (nothing was looked for).
        trace = Some(RecallTrace {
            ts_ms: now,
            session_id: ev.session_id.clone(),
            logical: logical.clone(),
            scope_dir: view.project.as_ref().map(|p| p.scope_dir.clone()),
            query_hash: archive::turn_key(text),
            outcome: why.to_string(),
            thresholds: TraceThresholds { min_sem: st.cfg.recall_min_sem, min_lex: st.cfg.recall_min_lex },
            ..RecallTrace::default()
        });
    } else {
        // A chat whose rows are filed under another project gets global material only.
        if let (Some(session), Some(logical), Some(p)) = (session_id, logical.as_deref(), view.project.clone()) {
            if filed_elsewhere(st, session, logical, &p.scope_dir, now) {
                view.project = None;
            }
        }
        let mut cfg = RecallConfig::from_ext(&st.cfg, now);
        cfg.model_loaded = st.status.model_loaded.load(Ordering::Relaxed);
        let req = RecallRequest {
            query: prompt,
            view: &view,
            kinds: None,
            since_ms: None,
            max_items: cfg.max_items,
            deadline,
            ambient: true,
        };
        if let Some(conn) = st.conn.as_ref() {
            let mut found = recall(conn, &st.index, &st.embed, &cfg, &req);
            if !found.picks.is_empty() {
                let block = inject::format_recall(&found.picks, &cfg);
                let shown = block.as_deref().map_or(0, inject::items_in);
                // What was selected and did not fit the block.
                // They were picks, so they are the best of what was dropped: they go first.
                let over: Vec<TraceDropped> = found
                    .trace
                    .picks
                    .drain(shown.min(found.trace.picks.len())..)
                    .map(|p| TraceDropped { r#ref: p.r#ref, why: "over_budget".to_string() })
                    .collect();
                found.trace.dropped.splice(0..0, over);
                found.trace.dropped.truncate(DROPPED_KEPT);
                found.trace.selected = shown as i64;
                // Past the deadline nothing is injected: never a late block.
                match block {
                    Some(_) if Instant::now() >= deadline => {
                        found.trace.outcome = "timeout".to_string();
                        found.trace.picks.clear();
                        found.trace.selected = 0;
                    }
                    Some(block) => {
                        found.trace.outcome = "injected".to_string();
                        out = HookOut::Inject(block);
                    }
                    None => found.trace.outcome = "over_budget".to_string(),
                }
            }
            // The log never holds text the brain would not hold anyway: the prompt's start
            // and its words are kept only when this turn would be captured.
            if st.cfg.log_queries && st.cfg.capture && !learning_paused && !screen::core_withholds(prompt) {
                found.trace.query_preview = Some(first_chars(text, PREVIEW_CHARS).to_string());
                found.trace.query_tokens = tokens(text);
            }
            trace = Some(found.trace);
        }
    }

    if let Some(mut trace) = trace {
        trace.total_ms = ms_between(started, Instant::now());
        crate::ext::front::send_job(st, Job::LogRecall(Box::new(trace)));
    }
    out
}

// ── The log ─────────────────────────────────────────────────────────────────

/// The `detail` column of one `recall_log` row (section 3, "Seeing why"): what `why` reads.
pub fn detail_json(trace: &RecallTrace) -> String {
    json!({
        "v": 1,
        "query_tokens": trace.query_tokens,
        "embed": { "used": trace.embed.used, "ms": trace.embed.ms, "fallback": trace.embed.fallback },
        "thresholds": { "min_sem": trace.thresholds.min_sem, "min_lex": trace.thresholds.min_lex },
        "generators": {
            "mem_words": trace.generators.mem_words,
            "doc_words": trace.generators.doc_words,
            "by_meaning": trace.generators.by_meaning,
            "past_floor": trace.generators.past_floor,
            "after_dedupe": trace.generators.after_dedupe,
        },
        "picks": trace.picks.iter().map(|p| json!({
            "n": p.n, "ref": p.r#ref, "kind": p.kind, "origin": p.origin, "scope": p.scope,
            "created": p.created, "parts": p.parts, "reasons": p.reasons,
        })).collect::<Vec<_>>(),
        "dropped": trace.dropped.iter().map(|d| json!({ "ref": d.r#ref, "why": d.why })).collect::<Vec<_>>(),
    })
    .to_string()
}

/// Inside the caller's transaction. False: the row falls inside a forget call and is dropped.
fn insert_log_row(conn: &Connection, trace: &RecallTrace) -> Result<bool> {
    // A turn in which a forget tool was called leaves no trace: a log job that arrives
    // after the call's mark is not written at all (section 4, "the live call").
    if trace.logical.is_some() || trace.session_id.is_some() {
        let inside: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM forgotten_turns
                            WHERE source = 'call' AND (logical = ?1 OR session_id = ?2)
                              AND from_ms <= ?3 AND to_ms >= ?3)",
            params![trace.logical, trace.session_id, trace.ts_ms],
            |r| r.get(0),
        )?;
        if inside {
            return Ok(false);
        }
    }
    conn.execute(
        "INSERT INTO recall_log(ts_ms, session_id, logical, scope_dir, query_preview, query_hash, outcome,
                                total_ms, embed_ms, candidates, selected, detail)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            trace.ts_ms,
            trace.session_id,
            trace.logical,
            trace.scope_dir,
            trace.query_preview,
            trace.query_hash,
            trace.outcome,
            trace.total_ms,
            trace.embed_ms,
            trace.candidates,
            trace.selected,
            detail_json(trace),
        ],
    )?;
    Ok(true)
}

/// `begin_write` … `COMMIT` around `work`; any error rolls back and is returned.
fn in_write<T>(conn: &Connection, work: impl FnOnce() -> Result<T>) -> Result<T> {
    brain_v2::begin_write(conn)?;
    let done = work().and_then(|v| {
        conn.execute_batch("COMMIT")?;
        Ok(v)
    });
    if done.is_err() && !conn.is_autocommit() {
        let _ = conn.execute_batch("ROLLBACK");
    }
    done
}

/// `Job::LogRecall` (called by `indexer::dispatch`): one `recall_log` row, in its own
/// transaction. A row that falls inside a `call` mark of its chat is dropped.
pub fn write_log(ctx: &mut IndexerCtx, trace: &RecallTrace) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    in_write(conn, || insert_log_row(conn, trace)).map(|_| ())
}

/// Every 3,600 s (called by `indexer::tick`): the table keeps the newest 1,000 rows and
/// none older than 30 days. One transaction.
pub fn trim_log(ctx: &mut IndexerCtx) -> Result<()> {
    let Some(conn) = ctx.conn.as_ref() else { return Ok(()) };
    let oldest = (ctx.clock)() - LOG_AGE_MS;
    in_write(conn, || {
        conn.execute(
            "DELETE FROM recall_log
              WHERE ts_ms < ?1
                 OR id NOT IN (SELECT id FROM recall_log ORDER BY id DESC LIMIT ?2)",
            params![oldest, LOG_ROWS],
        )?;
        Ok(())
    })
}
