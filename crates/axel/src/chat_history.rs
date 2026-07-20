//! Chat-history recall: bounded candidate generation (continuous-memory
//! spec §8.1, §9.1, §9.5, §16.2 — Task B2).
//!
//! This module deliberately builds **no new storage layer and no schema
//! migration**: it reuses the proven project-scoped, sensitivity/retention
//! enforced store in [`crate::project_memory`] end to end.
//!
//! * **Record classes (spec §8.1)** are represented with the *existing*
//!   free-text `memories.category` column, using the fixed
//!   lowercase-snake-case constants below ([`CLASS_EPISODIC_TURN`] …
//!   [`CLASS_CORRECTION`]). No new enum column, no migration.
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
use serde::Serialize;

use crate::error::Result;
use crate::project_memory::{search_scoped, ScopedDescriptor, ScopedQuery, Sensitivity};

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
