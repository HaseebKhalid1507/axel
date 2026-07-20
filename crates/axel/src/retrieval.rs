//! Bounded chat-history ranking with machine-readable rank reasons
//! (continuous-memory spec §9.2, §10.4, §16.2 — Task B2).
//!
//! * **Disclosure before ranking**: every secret / withheld-retention
//!   record is excluded *before* any score is computed (via the existing
//!   [`crate::project_memory::Retention::model_visible`] model). Ranking
//!   never inspects a record it must not disclose. (Candidate generation
//!   already enforces the same gate; this is defence in depth for
//!   hand-constructed candidates.)
//! * **Bounded selection**: output is capped at [`MAX_SELECTED_CAP`]
//!   (spec §16.2: ≤8 selected records), exact duplicate ids are deduped,
//!   and exact (case/whitespace-normalised) duplicate titles are dropped
//!   as a cheap near-identical guard. The full MMR diversity pass is
//!   deliberately out of scope here (Task E2).
//! * **Explainability (spec §10.4)**: every selected record carries at
//!   least one machine-readable [`RankReason`] — never free text.

use std::cmp::Ordering;
use std::collections::HashSet;

use serde::Serialize;

use crate::chat_history::{
    session_tag, RecallCandidate, CLASS_CONVERSATION_SUMMARY, CLASS_CORRECTION, CLASS_DECISION,
    CLASS_ENTITY_FACT, CLASS_EPISODIC_TURN, CLASS_PREFERENCE, CLASS_TOOL_OUTCOME,
    CLASS_UNRESOLVED_TASK,
};
use crate::project_memory::{ScopedDescriptor, Sensitivity};

/// Hard cap on selected records per recall (spec §16.2: ≤8).
pub const MAX_SELECTED_CAP: usize = 8;

// Ranking weights (spec §9.2). Session affinity must dominate small
// recency/class differences so same-session records outrank foreign-session
// records at equal lexical relevance.
const W_LEXICAL: f64 = 2.0;
const W_SESSION: f64 = 1.0;
const W_CLASS: f64 = 0.75;
const W_RECENCY: f64 = 0.5;

/// Machine-readable reason a record was selected (spec §10.4). Every
/// selected record carries at least one. Never free text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RankReason {
    /// Matched the lexical FTS query.
    LexicalMatch,
    /// Matched an explicitly requested tag/entity (reserved for the exact
    /// tag generator; not emitted by the B2 generators).
    ExactTagMatch,
    /// Carries the current session's affinity tag.
    RecentSession,
    /// Entered via the bounded recency-window fallback (spec §9.5).
    RecencyWindow,
    /// Record class `preference` (explicit user preference).
    ExplicitPreference,
    /// Record class `decision` (explicit project decision).
    ExplicitDecision,
    /// Record class `unresolved_task` (open work state).
    UnresolvedTaskState,
    /// Boosted by its record class (the class constant is attached).
    ClassBoost(&'static str),
}

fn model_visible(d: &ScopedDescriptor) -> bool {
    d.sensitivity == Sensitivity::Normal && d.retention.model_visible()
}

fn score(c: &RecallCandidate, affinity: bool) -> f64 {
    let lexical = if c.lexical_match { 1.0 } else { 0.0 };
    let session = if affinity { 1.0 } else { 0.0 };
    W_LEXICAL * lexical + W_SESSION * session + W_CLASS * c.class_weight + W_RECENCY * c.recency
}

fn reasons_for(c: &RecallCandidate, affinity: bool) -> Vec<RankReason> {
    let mut reasons = Vec::with_capacity(3);
    if c.lexical_match {
        reasons.push(RankReason::LexicalMatch);
    }
    if affinity {
        reasons.push(RankReason::RecentSession);
    }
    if c.recency_fallback {
        reasons.push(RankReason::RecencyWindow);
    }
    match c.descriptor.category.as_str() {
        CLASS_PREFERENCE => reasons.push(RankReason::ExplicitPreference),
        CLASS_DECISION => reasons.push(RankReason::ExplicitDecision),
        CLASS_UNRESOLVED_TASK => reasons.push(RankReason::UnresolvedTaskState),
        CLASS_CORRECTION => reasons.push(RankReason::ClassBoost(CLASS_CORRECTION)),
        CLASS_CONVERSATION_SUMMARY => {
            reasons.push(RankReason::ClassBoost(CLASS_CONVERSATION_SUMMARY))
        }
        CLASS_ENTITY_FACT => reasons.push(RankReason::ClassBoost(CLASS_ENTITY_FACT)),
        CLASS_TOOL_OUTCOME => reasons.push(RankReason::ClassBoost(CLASS_TOOL_OUTCOME)),
        CLASS_EPISODIC_TURN => reasons.push(RankReason::ClassBoost(CLASS_EPISODIC_TURN)),
        _ => {}
    }
    if reasons.is_empty() {
        // Unreachable via generate_candidates (every candidate records its
        // generator provenance), but spec §10.4 demands ≥1 machine-readable
        // reason for hand-constructed candidates too.
        reasons.push(RankReason::ClassBoost("unclassified"));
    }
    reasons
}

fn known_stale(c: &RecallCandidate) -> bool {
    c.descriptor.links.superseded_by.is_some() || c.descriptor.links.invalidated_at.is_some()
}

/// Measurable stale/superseded selection counters for quality evaluation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct SelectionMetrics {
    pub candidates: usize,
    pub stale_candidates: usize,
    pub selected: usize,
    pub stale_selected: usize,
}

impl SelectionMetrics {
    /// Fraction of selected records known stale. Empty selections report 0.
    pub fn stale_selected_rate(&self) -> f64 {
        if self.selected == 0 {
            0.0
        } else {
            self.stale_selected as f64 / self.selected as f64
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RankedSelection {
    pub records: Vec<(ScopedDescriptor, Vec<RankReason>)>,
    pub metrics: SelectionMetrics,
}

/// Rank with measurable stale-selection counters. Known superseded or
/// invalidated records receive the strongest possible penalty: exclusion, so
/// they can never be presented as current without an explicit conflict label.
pub fn rank_and_select_with_metrics(
    candidates: Vec<RecallCandidate>,
    session_id: Option<&str>,
    max_selected: usize,
) -> RankedSelection {
    let mut metrics = SelectionMetrics {
        candidates: candidates.len(),
        stale_candidates: candidates
            .iter()
            .filter(|candidate| known_stale(candidate))
            .count(),
        ..SelectionMetrics::default()
    };
    let records = rank_current(candidates, session_id, max_selected);
    metrics.selected = records.len();
    metrics.stale_selected = records
        .iter()
        .filter(|(descriptor, _)| {
            descriptor.links.superseded_by.is_some() || descriptor.links.invalidated_at.is_some()
        })
        .count();
    RankedSelection { records, metrics }
}

/// Rank the candidate set and select at most
/// `min(max_selected, MAX_SELECTED_CAP)` records (spec §9.2, §16.2).
///
/// Order: weighted lexical + session-affinity + class + recency score,
/// tie-broken by creation time (newest first) then id for determinism.
/// Exact duplicate ids and exact normalised-title duplicates are dropped.
/// Every returned record carries ≥1 [`RankReason`].
pub fn rank_and_select(
    candidates: Vec<RecallCandidate>,
    session_id: Option<&str>,
    max_selected: usize,
) -> Vec<(ScopedDescriptor, Vec<RankReason>)> {
    rank_and_select_with_metrics(candidates, session_id, max_selected).records
}

fn rank_current(
    candidates: Vec<RecallCandidate>,
    session_id: Option<&str>,
    max_selected: usize,
) -> Vec<(ScopedDescriptor, Vec<RankReason>)> {
    let cap = max_selected.min(MAX_SELECTED_CAP);
    if cap == 0 {
        return Vec::new();
    }
    let session_tag = session_id.map(session_tag);

    // Disclosure gate FIRST (defence in depth): a withheld record is never
    // scored, never sorted, never inspected by the ranker.
    let mut scored: Vec<(RecallCandidate, bool, f64)> = candidates
        .into_iter()
        .filter(|c| model_visible(&c.descriptor))
        .filter(|c| !known_stale(c))
        .map(|c| {
            let affinity = c.session_affinity
                || session_tag
                    .as_deref()
                    .is_some_and(|t| c.descriptor.tags.iter().any(|tag| tag == t));
            let s = score(&c, affinity);
            (c, affinity, s)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(Ordering::Equal)
            .then_with(|| b.0.descriptor.created.cmp(&a.0.descriptor.created))
            .then_with(|| a.0.descriptor.id.cmp(&b.0.descriptor.id))
    });

    let mut seen_ids: HashSet<String> = HashSet::with_capacity(cap);
    let mut seen_titles: HashSet<String> = HashSet::with_capacity(cap);
    let mut selected: Vec<(ScopedDescriptor, Vec<RankReason>)> = Vec::with_capacity(cap);
    for (candidate, affinity, _) in scored {
        if selected.len() >= cap {
            break;
        }
        if !seen_ids.insert(candidate.descriptor.id.clone()) {
            continue; // exact duplicate id
        }
        let norm_title = candidate.descriptor.title.trim().to_lowercase();
        if !norm_title.is_empty() && !seen_titles.insert(norm_title) {
            continue; // exact (normalised) duplicate title
        }
        let reasons = reasons_for(&candidate, affinity);
        debug_assert!(!reasons.is_empty(), "spec §10.4: ≥1 rank reason required");
        selected.push((candidate.descriptor, reasons));
    }
    selected
}
