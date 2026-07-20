//! Candidate generation and typed episodic storage (continuous-memory
//! spec §8.1, §9.1, §9.5, §11.2, §16.2 — Tasks B2/C2).
//!
//! This module builds **no parallel storage layer**: it reuses the proven
//! project-scoped, sensitivity/retention enforced store in
//! [`crate::project_memory`] end to end. Legacy `.r8` files are migrated by
//! that layer before the first typed capture.
//!
//! * **Record classes (spec §8.1)** reuse the existing free-text
//!   `memories.category` column via the fixed lowercase-snake-case constants
//!   below ([`CLASS_EPISODIC_TURN`] … [`CLASS_CORRECTION`]); no new table is
//!   required.
//! * **Session affinity (spec §9.1 point 3)**: the existing schema has no
//!   session-identifying column, but `memories.tags` is an existing,
//!   queryable column (JSON array text, filtered in SQL by
//!   `search_scoped`). Session identity is therefore encoded as a plain tag
//!   `"session:<session_id>"` (see [`session_tag`]). Affinity detection
//!   happens on the parsed tag vector — an exact string comparison — so
//!   session ids containing `_`/`%` are unaffected by the LIKE-escaping in
//!   the SQL tag filter.
//! * **Bounds (spec §16.2)**: candidates retained are capped at
//!   [`CANDIDATES_RETAINED_CAP`] **by construction**, never by post-hoc
//!   truncation: the cap is passed into `ScopedQuery.limit`, which
//!   `search_scoped` clamps and binds as the SQL `LIMIT`
//!   (see `SEARCH_LIMIT_CAP`); no unbounded `Vec` ever exists. The two
//!   generators (lexical, recency fallback) are mutually exclusive, so their
//!   bounds never add up beyond the cap.
//! * **Fallback (spec §9.5)**: when strict lexical matching yields no
//!   useful (model-visible) record, a *small, fixed* recency window is used
//!   — at most [`RECENCY_FALLBACK_LIMIT`] records from the last
//!   [`RECENCY_FALLBACK_WINDOW_HOURS`] hours. Arbitrary large recent
//!   history is never injected.
//! * **Disclosure**: records that are `Sensitivity::Secret` or whose
//!   retention class is not [`crate::project_memory::Retention::model_visible`]
//!   are excluded *while building candidates* — they are never wrapped in a
//!   [`RecallCandidate`], so downstream ranking can never inspect them.

use chrono::{DateTime, Duration, Utc};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use axel_memkoshi::memory::{Confidence, Memory, MemoryCategory};

use crate::error::{AxelError, Result};
use crate::project_memory::{
    search_scoped, store_scoped_typed_with_links, MemoryScope, RecordLinks, Retention,
    ScopedDescriptor, ScopedQuery, Sensitivity,
};

// ── Record-class category constants (spec §8.1) ─────────────────────────

/// One logical user/assistant turn.
pub const CLASS_EPISODIC_TURN: &str = "episodic_turn";
/// Compaction summary with source range digest.
pub const CLASS_CONVERSATION_SUMMARY: &str = "conversation_summary";
/// Explicit project decision and rationale.
pub const CLASS_DECISION: &str = "decision";
/// Explicit user preference.
pub const CLASS_PREFERENCE: &str = "preference";
/// Open work and last known state.
pub const CLASS_UNRESOLVED_TASK: &str = "unresolved_task";
/// Project entity and relationship.
pub const CLASS_ENTITY_FACT: &str = "entity_fact";
/// Bounded summary of a meaningful tool action.
pub const CLASS_TOOL_OUTCOME: &str = "tool_outcome";
/// A later statement that supersedes older memory.
pub const CLASS_CORRECTION: &str = "correction";

/// All eight chat-history record classes (spec §8.1).
pub const RECORD_CLASSES: [&str; 8] = [
    CLASS_EPISODIC_TURN,
    CLASS_CONVERSATION_SUMMARY,
    CLASS_DECISION,
    CLASS_PREFERENCE,
    CLASS_UNRESOLVED_TASK,
    CLASS_ENTITY_FACT,
    CLASS_TOOL_OUTCOME,
    CLASS_CORRECTION,
];

/// Is `category` one of the eight chat-history record classes?
pub fn is_chat_history_class(category: &str) -> bool {
    RECORD_CLASSES.contains(&category)
}

// ── Typed episodic records and durable capture (spec §8.1) ─────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicTurn {
    pub content: String,
    pub source_session_id: Option<String>,
    pub source_turn_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSummary {
    pub summary: String,
    pub source_range_digest: String,
    pub source_session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decision {
    pub decision: String,
    pub rationale: Option<String>,
    pub source_session_id: Option<String>,
    pub source_turn_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preference {
    pub preference: String,
    pub source_session_id: Option<String>,
    pub source_turn_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnresolvedTask {
    pub task: String,
    pub last_known_state: Option<String>,
    pub source_session_id: Option<String>,
    pub source_turn_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityFact {
    pub entity: String,
    pub relationship: String,
    pub source_session_id: Option<String>,
    pub source_turn_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutcome {
    pub tool: String,
    pub outcome: String,
    pub source_session_id: Option<String>,
    pub source_turn_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correction {
    pub correction: String,
    /// Stable 64-bit numeric id of the record this corrects, when supplied.
    pub supersedes_id: Option<u64>,
    pub source_session_id: Option<String>,
    pub source_turn_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "class", content = "record", rename_all = "snake_case")]
pub enum EpisodicRecord {
    Turn(EpisodicTurn),
    Summary(ConversationSummary),
    Decision(Decision),
    Preference(Preference),
    UnresolvedTask(UnresolvedTask),
    EntityFact(EntityFact),
    ToolOutcome(ToolOutcome),
    Correction(Correction),
}

impl EpisodicRecord {
    pub fn class(&self) -> &'static str {
        match self {
            Self::Turn(_) => CLASS_EPISODIC_TURN,
            Self::Summary(_) => CLASS_CONVERSATION_SUMMARY,
            Self::Decision(_) => CLASS_DECISION,
            Self::Preference(_) => CLASS_PREFERENCE,
            Self::UnresolvedTask(_) => CLASS_UNRESOLVED_TASK,
            Self::EntityFact(_) => CLASS_ENTITY_FACT,
            Self::ToolOutcome(_) => CLASS_TOOL_OUTCOME,
            Self::Correction(_) => CLASS_CORRECTION,
        }
    }

    fn source_session_id(&self) -> Option<&str> {
        match self {
            Self::Turn(v) => v.source_session_id.as_deref(),
            Self::Summary(v) => v.source_session_id.as_deref(),
            Self::Decision(v) => v.source_session_id.as_deref(),
            Self::Preference(v) => v.source_session_id.as_deref(),
            Self::UnresolvedTask(v) => v.source_session_id.as_deref(),
            Self::EntityFact(v) => v.source_session_id.as_deref(),
            Self::ToolOutcome(v) => v.source_session_id.as_deref(),
            Self::Correction(v) => v.source_session_id.as_deref(),
        }
    }

    fn primary_text(&self) -> &str {
        match self {
            Self::Turn(v) => &v.content,
            Self::Summary(v) => &v.summary,
            Self::Decision(v) => &v.decision,
            Self::Preference(v) => &v.preference,
            Self::UnresolvedTask(v) => &v.task,
            Self::EntityFact(v) => &v.relationship,
            Self::ToolOutcome(v) => &v.outcome,
            Self::Correction(v) => &v.correction,
        }
    }

    fn explicit_entity(&self) -> Option<&str> {
        match self {
            Self::EntityFact(v) => Some(&v.entity),
            Self::ToolOutcome(v) => Some(&v.tool),
            _ => None,
        }
    }

    fn title_prefix(&self) -> &'static str {
        match self {
            Self::Turn(_) => "Turn",
            Self::Summary(_) => "Summary",
            Self::Decision(_) => "Decision",
            Self::Preference(_) => "Preference",
            Self::UnresolvedTask(_) => "Task",
            Self::EntityFact(_) => "Entity",
            Self::ToolOutcome(_) => "Tool outcome",
            Self::Correction(_) => "Correction",
        }
    }
}

/// Host capture envelope. Provenance is copied verbatim: this layer neither
/// verifies it nor upgrades it to a stronger claim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureRequest {
    pub project_key: String,
    pub sensitivity: Sensitivity,
    pub retention: Retention,
    pub provenance: String,
    pub record: EpisodicRecord,
    pub title: Option<String>,
    pub tags: Vec<String>,
    pub entities: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureAck {
    /// Random stable 64-bit id, represented numerically at this API boundary.
    pub id: u64,
    pub class: String,
    pub provenance: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeuristicEnrichment {
    pub title: String,
    pub tags: Vec<String>,
    pub entities: Vec<String>,
}

fn bounded_words(text: &str, count: usize) -> String {
    text.split_whitespace()
        .take(count)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Deterministic, offline enrichment. No model, GLiNER, embedding, download,
/// or network path is present here.
pub fn enrich_heuristically(text: &str) -> HeuristicEnrichment {
    let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let title = bounded_words(cleaned.split(['.', '\n']).next().unwrap_or(""), 12);
    let lower = cleaned.to_ascii_lowercase();
    let mut tags = BTreeSet::new();
    for (tag, needles) in [
        ("decision", &["decided", "decision", "we will"][..]),
        ("preference", &["prefer", "preference"][..]),
        ("task", &["todo", "fixme", "unresolved", "need to"][..]),
        ("tool", &["cargo ", "git ", "tool "][..]),
        ("correction", &["correction", "actually", "instead"][..]),
    ] {
        if needles.iter().any(|needle| lower.contains(needle)) {
            tags.insert(tag.to_string());
        }
    }
    let mut entities = BTreeSet::new();
    for raw in cleaned.split_whitespace() {
        let token = raw.trim_matches(|c: char| !c.is_alphanumeric() && !"_/-".contains(c));
        let file_like = token.contains('/')
            || [".rs", ".toml", ".md", ".json", ".yaml", ".yml"]
                .iter()
                .any(|ext| token.ends_with(ext));
        let named = token.chars().next().is_some_and(char::is_uppercase)
            && token.chars().skip(1).any(char::is_lowercase);
        if token.len() > 2 && (file_like || named) {
            entities.insert(token.to_string());
        }
    }
    HeuristicEnrichment {
        title,
        tags: tags.into_iter().collect(),
        entities: entities.into_iter().collect(),
    }
}

fn storage_id(id: u64) -> String {
    format!("mem_{id:016x}")
}

fn random_id() -> u64 {
    loop {
        let generated = Memory::generate_id();
        if let Some(hex) = generated.strip_prefix("mem_") {
            if let Ok(id) = u64::from_str_radix(hex, 16) {
                if id != 0 {
                    return id;
                }
            }
        }
    }
}

/// Store and acknowledge one capture. The scoped store performs migration and
/// a `synchronous=FULL` transaction commit before this function returns `Ok`.
pub fn store_capture(conn: &Connection, request: &CaptureRequest) -> Result<CaptureAck> {
    store_capture_with_id(conn, random_id(), request)
}

/// Deterministic-id variant for replay/import and tombstone compatibility.
pub fn store_capture_with_id(
    conn: &Connection,
    id: u64,
    request: &CaptureRequest,
) -> Result<CaptureAck> {
    if id == 0 {
        return Err(AxelError::Memory("capture id must be non-zero".into()));
    }
    let body = request.record.primary_text();
    if body.trim().is_empty() {
        return Err(AxelError::Memory("capture body must not be empty".into()));
    }
    let enrichment = enrich_heuristically(body);
    let mut tags: BTreeSet<String> = request.tags.iter().cloned().collect();
    tags.extend(enrichment.tags);
    if let Some(session) = request.record.source_session_id() {
        tags.insert(session_tag(session));
    }
    let mut entities: BTreeSet<String> = request.entities.iter().cloned().collect();
    entities.extend(enrichment.entities);
    if let Some(entity) = request.record.explicit_entity() {
        entities.insert(entity.to_string());
    }
    let title = request
        .title
        .as_deref()
        .filter(|title| !title.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("{}: {}", request.record.title_prefix(), enrichment.title));
    let content = serde_json::to_string(&request.record)?;
    let scope = MemoryScope {
        project_key: request.project_key.clone(),
        sensitivity: request.sensitivity,
        retention: request.retention,
        provenance: request.provenance.clone(),
    };
    let mut memory = Memory::new(
        MemoryCategory::Events,
        request.record.class(),
        title,
        content,
    );
    memory.id = storage_id(id);
    memory.abstract_text = bounded_words(body, 40);
    memory.confidence = Confidence::Low;
    memory.trust_level = 0.0;
    memory.source_sessions = request
        .record
        .source_session_id()
        .map(|session| vec![session.to_string()])
        .unwrap_or_default();
    memory.tags = tags.into_iter().collect();
    memory.related_topics = entities.into_iter().collect();
    let links = match &request.record {
        EpisodicRecord::Correction(correction) => RecordLinks {
            supersedes: correction
                .supersedes_id
                .map(storage_id)
                .into_iter()
                .collect(),
            ..RecordLinks::default()
        },
        _ => RecordLinks::default(),
    };
    store_scoped_typed_with_links(conn, &memory, &scope, request.record.class(), &links)?;
    Ok(CaptureAck {
        id,
        class: request.record.class().to_string(),
        provenance: request.provenance.clone(),
    })
}

// ── Bounds (spec §9.5, §16.2) ───────────────────────────────────────────

/// Hard cap on candidates retained per recall (spec §16.2: ≤128).
pub const CANDIDATES_RETAINED_CAP: usize = 128;
/// Maximum records the recency fallback may contribute (spec §9.5:
/// "small recency candidate set", never arbitrary large recent history).
pub const RECENCY_FALLBACK_LIMIT: usize = 12;
/// Fixed recency-fallback window, in hours (spec §9.5).
pub const RECENCY_FALLBACK_WINDOW_HOURS: i64 = 48;

// ── Session affinity via existing tags column (spec §9.1 point 3) ───────

/// Prefix for the session-identity tag stored in the existing `tags` column.
pub const SESSION_TAG_PREFIX: &str = "session:";

/// Canonical session-affinity tag for `session_id`.
pub fn session_tag(session_id: &str) -> String {
    format!("{SESSION_TAG_PREFIX}{session_id}")
}

// ── Derived signals ─────────────────────────────────────────────────────

/// Class weight in `[0.0, 1.0]` used as a ranking signal (spec §9.2
/// "record class"). Corrections and explicit decisions/preferences carry
/// the most authority; raw episodic turns the least.
pub fn class_weight(category: &str) -> f64 {
    match category {
        CLASS_CORRECTION => 1.0,
        CLASS_DECISION => 0.9,
        CLASS_PREFERENCE => 0.85,
        CLASS_UNRESOLVED_TASK => 0.8,
        CLASS_CONVERSATION_SUMMARY => 0.7,
        CLASS_ENTITY_FACT => 0.6,
        CLASS_TOOL_OUTCOME => 0.5,
        CLASS_EPISODIC_TURN => 0.4,
        // Non-chat-history categories still rank, just with low class weight.
        _ => 0.3,
    }
}

/// Recency score in `(0.0, 1.0]` with a 7-day half-life. Unparseable
/// timestamps score `0.0` (fail closed: no artificial freshness).
pub fn recency_score(created_rfc3339: &str, now: DateTime<Utc>) -> f64 {
    const HALF_LIFE_SECS: f64 = 7.0 * 24.0 * 3600.0;
    match DateTime::parse_from_rfc3339(created_rfc3339) {
        Ok(t) => {
            let age_secs = (now - t.with_timezone(&Utc)).num_seconds().max(0) as f64;
            0.5_f64.powf(age_secs / HALF_LIFE_SECS)
        }
        Err(_) => 0.0,
    }
}

// ── Candidate model ─────────────────────────────────────────────────────

/// One model-visible recall candidate: a [`ScopedDescriptor`] plus derived
/// ranking signals. Secret / withheld-retention records are never wrapped
/// in this type — construction is the disclosure gate.
#[derive(Debug, Clone, Serialize)]
pub struct RecallCandidate {
    pub descriptor: ScopedDescriptor,
    /// Recency score in `[0.0, 1.0]` (see [`recency_score`]).
    pub recency: f64,
    /// True when the record carries the current session's tag.
    pub session_affinity: bool,
    /// Class weight (see [`class_weight`]).
    pub class_weight: f64,
    /// Produced by the lexical FTS generator.
    pub lexical_match: bool,
    /// Produced by the bounded recency-window fallback generator.
    pub recency_fallback: bool,
}

fn model_visible(d: &ScopedDescriptor) -> bool {
    d.sensitivity == Sensitivity::Normal && d.retention.model_visible()
}

/// Convert bounded search hits into candidates, dropping every record the
/// model must not see *before* it becomes a candidate. `out` never grows
/// past `cap` (the inputs are already SQL-bounded; this is belt-and-braces).
fn push_candidates(
    out: &mut Vec<RecallCandidate>,
    hits: Vec<ScopedDescriptor>,
    session_id: Option<&str>,
    lexical_match: bool,
    recency_fallback: bool,
    now: DateTime<Utc>,
    cap: usize,
) {
    let session_tag = session_id.map(session_tag);
    for d in hits {
        if out.len() >= cap {
            break;
        }
        // Disclosure gate: secret / non-model-visible records never become
        // candidates, so ranking can never inspect them.
        if !model_visible(&d) {
            continue;
        }
        let session_affinity = session_tag
            .as_deref()
            .is_some_and(|t| d.tags.iter().any(|tag| tag == t));
        out.push(RecallCandidate {
            recency: recency_score(&d.created, now),
            session_affinity,
            class_weight: class_weight(&d.category),
            lexical_match,
            recency_fallback,
            descriptor: d,
        });
    }
}

/// Generate a bounded, project-scoped, model-visible candidate set for one
/// recall request (spec §9.1, §9.5, §16.2).
///
/// Generators, in order:
/// 1. **Lexical FTS** via [`search_scoped`] when `query_text` is `Some` and
///    non-empty — SQL-bounded by the candidate cap.
/// 2. **Recency-window fallback** (only when the lexical pass yielded no
///    useful record, or there was no query text): `text: None` with
///    `since = now − RECENCY_FALLBACK_WINDOW_HOURS`, SQL-bounded by
///    [`RECENCY_FALLBACK_LIMIT`].
///
/// The candidate cap ([`CANDIDATES_RETAINED_CAP`]) is enforced by passing
/// it into the SQL `LIMIT` (via `ScopedQuery.limit`, which `search_scoped`
/// clamps further to `SEARCH_LIMIT_CAP`), never by truncating an unbounded
/// vector after the fact.
pub fn generate_candidates(
    conn: &Connection,
    project_key: &str,
    session_id: Option<&str>,
    query_text: Option<&str>,
    limit_candidates: usize,
) -> Result<Vec<RecallCandidate>> {
    let cap = limit_candidates.clamp(1, CANDIDATES_RETAINED_CAP);
    let now = Utc::now();
    let mut out: Vec<RecallCandidate> = Vec::with_capacity(cap.min(RECENCY_FALLBACK_LIMIT.max(1)));

    // 1. Lexical FTS pass — bound goes into the SQL LIMIT.
    let text = query_text.map(str::trim).filter(|t| !t.is_empty());
    if let Some(text) = text {
        let hits = search_scoped(
            conn,
            &ScopedQuery {
                project_key: project_key.to_string(),
                text: Some(text.to_string()),
                tags: Vec::new(),
                since: None,
                until: None,
                limit: cap,
            },
        )?;
        push_candidates(&mut out, hits, session_id, true, false, now, cap);
    }

    // 2. Recency-window fallback (spec §9.5): only when lexical matching
    //    produced no useful (model-visible) record. Small and fixed —
    //    never an arbitrary large recent-history dump.
    if out.is_empty() {
        let hits = search_scoped(
            conn,
            &ScopedQuery {
                project_key: project_key.to_string(),
                text: None,
                tags: Vec::new(),
                since: Some(now - Duration::hours(RECENCY_FALLBACK_WINDOW_HOURS)),
                until: None,
                limit: cap.min(RECENCY_FALLBACK_LIMIT),
            },
        )?;
        push_candidates(&mut out, hits, session_id, false, true, now, cap);
    }

    debug_assert!(out.len() <= CANDIDATES_RETAINED_CAP);
    Ok(out)
}

#[cfg(test)]
mod episodic_storage_tests {
    use super::*;
    use crate::project_memory::{migrate, search_scoped, ScopedQuery};
    use rusqlite::Connection;
    use tempfile::TempDir;

    fn create_legacy_r8(path: &std::path::Path) -> Connection {
        let conn = Connection::open(path).unwrap();
        conn.execute_batch(
            "CREATE TABLE memories (
                 id TEXT PRIMARY KEY,
                 category TEXT NOT NULL,
                 topic TEXT NOT NULL,
                 title TEXT NOT NULL,
                 abstract_text TEXT NOT NULL,
                 content TEXT NOT NULL,
                 confidence TEXT NOT NULL,
                 importance REAL NOT NULL,
                 source_sessions TEXT NOT NULL,
                 tags TEXT NOT NULL,
                 related_topics TEXT NOT NULL,
                 created TEXT NOT NULL,
                 updated TEXT,
                 trust_level INTEGER NOT NULL,
                 signature TEXT
             );",
        )
        .unwrap();
        conn
    }

    fn capture_request(content: &str, provenance: &str) -> CaptureRequest {
        CaptureRequest {
            project_key: "project-a".into(),
            sensitivity: Sensitivity::Normal,
            retention: crate::project_memory::Retention::Standard,
            provenance: provenance.into(),
            record: EpisodicRecord::Turn(EpisodicTurn {
                content: content.into(),
                source_session_id: Some("session-7".into()),
                source_turn_id: Some("turn-9".into()),
            }),
            title: None,
            tags: vec!["caller-tag".into()],
            entities: vec![],
        }
    }

    #[test]
    fn kill_after_store_reopen_recovers_every_acknowledged_capture() {
        const CHILD_ENV: &str = "AXEL_C2_KILL_AFTER_STORE_PATH";
        const TEST_NAME: &str =
            "chat_history::episodic_storage_tests::kill_after_store_reopen_recovers_every_acknowledged_capture";
        const FIRST_ID: u64 = 0xc200_0000_0000_0001;
        const SECOND_ID: u64 = 0xc200_0000_0000_0002;

        if let Some(path) = std::env::var_os(CHILD_ENV) {
            let conn = Connection::open(path).unwrap();
            store_capture_with_id(
                &conn,
                FIRST_ID,
                &capture_request(
                    "Implemented crates/axel/src/chat_history.rs for ProjectAxel",
                    "host:turn",
                ),
            )
            .unwrap();
            println!("ACK {FIRST_ID}");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            store_capture_with_id(
                &conn,
                SECOND_ID,
                &CaptureRequest {
                    record: EpisodicRecord::Decision(Decision {
                        decision: "Keep embeddings disabled".into(),
                        rationale: Some("Offline operation".into()),
                        source_session_id: None,
                        source_turn_id: None,
                    }),
                    ..capture_request("unused", "host:decision")
                },
            )
            .unwrap();
            println!("ACK {SECOND_ID}");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
            std::process::abort();
        }

        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("memory.r8");
        drop(create_legacy_r8(&path));
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST_NAME, "--nocapture"])
            .env(CHILD_ENV, &path)
            .output()
            .unwrap();
        assert!(!output.status.success(), "child must terminate abruptly");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains(&format!("ACK {FIRST_ID}")));
        assert!(stdout.contains(&format!("ACK {SECOND_ID}")));

        let reopened = Connection::open(&path).unwrap();
        migrate(&reopened).unwrap();
        for (id, class, provenance) in [
            (FIRST_ID, CLASS_EPISODIC_TURN, "host:turn"),
            (SECOND_ID, CLASS_DECISION, "host:decision"),
        ] {
            let stored: (String, String) = reopened
                .query_row(
                    "SELECT category, provenance FROM memories WHERE id = ?1",
                    [storage_id(id)],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap();
            assert_eq!(stored.0, class);
            assert_eq!(stored.1, provenance);
        }
    }

    #[test]
    fn tombstoned_id_never_resurfaces_after_reopen() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("memory.r8");
        let conn = create_legacy_r8(&path);
        migrate(&conn).unwrap();
        let acknowledged =
            store_capture(&conn, &capture_request("Forget this turn", "host")).unwrap();
        crate::project_memory::forget_scoped(&conn, &storage_id(acknowledged.id), "project-a")
            .unwrap();
        drop(conn);

        let reopened = Connection::open(&path).unwrap();
        migrate(&reopened).unwrap();
        let hits = search_scoped(
            &reopened,
            &ScopedQuery {
                project_key: "project-a".into(),
                text: None,
                tags: vec![],
                since: None,
                until: None,
                limit: 50,
            },
        )
        .unwrap();
        assert!(hits.iter().all(|hit| hit.id != storage_id(acknowledged.id)));
        assert!(store_capture_with_id(
            &reopened,
            acknowledged.id,
            &capture_request("Try to revive it", "host")
        )
        .is_err());
    }

    #[test]
    fn old_r8_opens_and_migrates_before_capture() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("old.r8");
        let conn = create_legacy_r8(&path);
        let acknowledged = store_capture(
            &conn,
            &capture_request("Legacy migration capture", "legacy-host"),
        )
        .unwrap();
        drop(conn);

        let reopened = Connection::open(path).unwrap();
        let project_key: String = reopened
            .query_row(
                "SELECT project_key FROM memories WHERE id = ?1",
                [storage_id(acknowledged.id)],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(project_key, "project-a");
    }

    #[test]
    fn provenance_is_preserved_without_upgrading_caller_claim() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("memory.r8");
        let conn = create_legacy_r8(&path);
        let supplied = "unverified:caller-assertion";
        let acknowledged =
            store_capture(&conn, &capture_request("Caller supplied only", supplied)).unwrap();

        assert_eq!(acknowledged.provenance, supplied);
        let (provenance, trust_level): (String, i64) = conn
            .query_row(
                "SELECT provenance, trust_level FROM memories WHERE id = ?1",
                [storage_id(acknowledged.id)],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(provenance, supplied);
        assert_eq!(trust_level, 0, "capture must not invent verified trust");
    }

    #[test]
    fn model_free_enrichment_adds_title_tags_and_entities_without_models() {
        let enrichment = enrich_heuristically(
            "Decided to update crates/axel/src/chat_history.rs for ProjectAxel. TODO: add reopen tests.",
        );
        assert!(!enrichment.title.is_empty());
        assert!(enrichment.tags.iter().any(|tag| tag == "decision"));
        assert!(enrichment.tags.iter().any(|tag| tag == "task"));
        assert!(enrichment
            .entities
            .iter()
            .any(|entity| entity == "crates/axel/src/chat_history.rs"));
        assert!(enrichment
            .entities
            .iter()
            .any(|entity| entity == "ProjectAxel"));
    }

    #[test]
    fn all_typed_record_classes_have_the_required_storage_class() {
        let records = vec![
            EpisodicRecord::Turn(EpisodicTurn {
                content: "turn".into(),
                source_session_id: None,
                source_turn_id: None,
            }),
            EpisodicRecord::Summary(ConversationSummary {
                summary: "summary".into(),
                source_range_digest: "sha256:abc".into(),
                source_session_id: None,
            }),
            EpisodicRecord::Decision(Decision {
                decision: "decision".into(),
                rationale: None,
                source_session_id: None,
                source_turn_id: None,
            }),
            EpisodicRecord::Preference(Preference {
                preference: "preference".into(),
                source_session_id: None,
                source_turn_id: None,
            }),
            EpisodicRecord::UnresolvedTask(UnresolvedTask {
                task: "task".into(),
                last_known_state: None,
                source_session_id: None,
                source_turn_id: None,
            }),
            EpisodicRecord::EntityFact(EntityFact {
                entity: "Axel".into(),
                relationship: "is a crate".into(),
                source_session_id: None,
                source_turn_id: None,
            }),
            EpisodicRecord::ToolOutcome(ToolOutcome {
                tool: "cargo".into(),
                outcome: "passed".into(),
                source_session_id: None,
                source_turn_id: None,
            }),
            EpisodicRecord::Correction(Correction {
                correction: "corrected".into(),
                supersedes_id: Some(7),
                source_session_id: None,
                source_turn_id: None,
            }),
        ];
        assert_eq!(
            records
                .iter()
                .map(EpisodicRecord::class)
                .collect::<Vec<_>>(),
            RECORD_CLASSES
        );
    }
}
