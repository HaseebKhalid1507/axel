//! Bounded chat-history ranking with machine-readable rank reasons
//! (continuous-memory spec §9.2, §10.4, §16.2 — Task B2).
//!
//! * **Disclosure before ranking**: every secret / withheld-retention
//!   record is excluded *before* any score is computed (via the existing
//!   [`crate::project_memory::Retention::model_visible`] model). Ranking
//!   never inspects a record it must not disclose. (Candidate generation
//!   already enforces the same gate; this is defence in depth for
//!   hand-constructed candidates.)
//! * **Bounded diverse selection**: after ranking, a bounded MMR-style
//!   pass mixes useful record classes and penalises near-identical titles.
//!   Output remains capped at [`MAX_SELECTED_CAP`] (spec §16.2: ≤8 selected
//!   records), so downstream byte-bound packing cannot be expanded.
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

/// The five useful classes called out by spec §9.3. Related episodic and fact
/// categories share a class so the pass does not spend its small budget on
/// cosmetic category differences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DiversityClass {
    Decision,
    Preference,
    Episode,
    UnresolvedTask,
    Fact,
}

impl DiversityClass {
    fn of(category: &str) -> Option<Self> {
        match category {
            CLASS_DECISION => Some(Self::Decision),
            CLASS_PREFERENCE => Some(Self::Preference),
            CLASS_EPISODIC_TURN | CLASS_CONVERSATION_SUMMARY | CLASS_TOOL_OUTCOME => {
                Some(Self::Episode)
            }
            CLASS_UNRESOLVED_TASK => Some(Self::UnresolvedTask),
            CLASS_ENTITY_FACT | CLASS_CORRECTION => Some(Self::Fact),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct DiversityKey {
    class: Option<DiversityClass>,
    score: f64,
    title_tokens: HashSet<String>,
}

impl DiversityKey {
    fn new(class: Option<DiversityClass>, score: f64, title: &str) -> Self {
        const STOP_WORDS: &[&str] = &["a", "an", "and", "are", "for", "is", "of", "the", "to"];
        let title_tokens = title
            .split(|character: char| !character.is_alphanumeric())
            .filter(|token| !token.is_empty())
            .map(str::to_lowercase)
            .filter(|token| !STOP_WORDS.contains(&token.as_str()))
            .collect();
        Self {
            class,
            score,
            title_tokens,
        }
    }
}

fn title_similarity(left: &DiversityKey, right: &DiversityKey) -> f64 {
    if left.title_tokens.is_empty() || right.title_tokens.is_empty() {
        return 0.0;
    }
    let intersection = left.title_tokens.intersection(&right.title_tokens).count();
    let union = left.title_tokens.union(&right.title_tokens).count();
    intersection as f64 / union as f64
}

/// Select indices from an already-ranked, bounded window. Rank relevance
/// decays gently while a one-time class novelty bonus promotes the spread in
/// spec §9.3. Near-identical titles are excluded rather than merely reordered.
fn diverse_selection_indices(ranked: &[DiversityKey], cap: usize) -> Vec<usize> {
    const CLASS_NOVELTY_BONUS: f64 = 0.4;
    const DUPLICATE_PENALTY: f64 = 0.75;
    const NEAR_DUPLICATE_THRESHOLD: f64 = 0.72;

    let cap = cap.min(MAX_SELECTED_CAP).min(ranked.len());
    let mut selected = Vec::with_capacity(cap);
    let mut selected_classes = HashSet::with_capacity(cap);

    while selected.len() < cap {
        let mut best: Option<(usize, f64)> = None;
        for (index, key) in ranked.iter().enumerate() {
            if selected.contains(&index) {
                continue;
            }
            let similarity = selected
                .iter()
                .map(|&selected_index| title_similarity(key, &ranked[selected_index]))
                .fold(0.0, f64::max);
            if similarity >= NEAR_DUPLICATE_THRESHOLD {
                continue;
            }

            let rank_relevance = key.score;
            let class_novelty = key
                .class
                .filter(|class| !selected_classes.contains(class))
                .map_or(0.0, |_| CLASS_NOVELTY_BONUS);
            let mmr_score = rank_relevance + class_novelty - DUPLICATE_PENALTY * similarity;
            if best.is_none_or(|(best_index, best_score)| {
                mmr_score > best_score || (mmr_score == best_score && index < best_index)
            }) {
                best = Some((index, mmr_score));
            }
        }

        let Some((index, _)) = best else {
            break;
        };
        if let Some(class) = ranked[index].class {
            selected_classes.insert(class);
        }
        selected.push(index);
    }
    selected
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
    // Four alternatives per output slot is enough room for useful class
    // mixing while keeping the diversity pass strictly bounded (≤32 rows).
    let diversity_window = cap.saturating_mul(4).min(MAX_SELECTED_CAP * 4);
    let mut ranked = Vec::with_capacity(diversity_window);
    for (candidate, affinity, candidate_score) in scored {
        if ranked.len() >= diversity_window {
            break;
        }
        if !seen_ids.insert(candidate.descriptor.id.clone()) {
            continue; // exact duplicate id
        }
        let norm_title = candidate.descriptor.title.trim().to_lowercase();
        if !norm_title.is_empty() && !seen_titles.insert(norm_title) {
            continue; // exact (normalised) duplicate title
        }
        ranked.push((candidate, affinity, candidate_score));
    }

    let diversity_keys: Vec<_> = ranked
        .iter()
        .map(|(candidate, _, score)| {
            DiversityKey::new(
                DiversityClass::of(&candidate.descriptor.category),
                *score,
                &candidate.descriptor.title,
            )
        })
        .collect();
    diverse_selection_indices(&diversity_keys, cap)
        .into_iter()
        .map(|index| {
            let (candidate, affinity, _) = &ranked[index];
            let reasons = reasons_for(candidate, *affinity);
            debug_assert!(!reasons.is_empty(), "spec §10.4: ≥1 rank reason required");
            (candidate.descriptor.clone(), reasons)
        })
        .collect()
}

#[cfg(test)]
mod diversity_tests {
    use super::*;

    fn key(class: DiversityClass, title: &str) -> DiversityKey {
        DiversityKey::new(Some(class), 1.0, title)
    }

    #[test]
    fn near_duplicate_fixture_keeps_at_most_one_representative() {
        let ranked = vec![
            key(DiversityClass::Decision, "Deploy Rust service"),
            key(DiversityClass::Decision, "Deploy the Rust service"),
            key(DiversityClass::Fact, "Production region is eu-west-1"),
        ];

        let selected = diverse_selection_indices(&ranked, MAX_SELECTED_CAP);
        let duplicate_representatives = selected.iter().filter(|&&index| index < 2).count();

        assert!(duplicate_representatives <= 1);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn diversity_stays_within_selection_and_prebounded_input() {
        let ranked: Vec<_> = (0..12)
            .map(|index| key(DiversityClass::Fact, &format!("unique fact number {index}")))
            .collect();

        let selected = diverse_selection_indices(&ranked, MAX_SELECTED_CAP);

        assert!(selected.len() <= MAX_SELECTED_CAP);
        assert!(selected.iter().all(|&index| index < ranked.len()));

        // A byte-bounded upstream slice cannot be expanded by the pass.
        let byte_bounded = &ranked[..3];
        let selected = diverse_selection_indices(byte_bounded, MAX_SELECTED_CAP);
        assert!(selected.len() <= byte_bounded.len());
    }

    #[test]
    fn class_mixing_prefers_spread_over_many_same_class_records() {
        let ranked = vec![
            key(DiversityClass::Fact, "database uses postgres"),
            key(DiversityClass::Fact, "api transport uses tonic"),
            key(DiversityClass::Fact, "cache uses redis"),
            key(DiversityClass::Fact, "frontend uses react"),
            key(DiversityClass::Fact, "queue uses nats"),
            key(DiversityClass::Decision, "ship on Tuesdays"),
            key(DiversityClass::Preference, "prefer concise output"),
            key(DiversityClass::Episode, "recent deploy failed"),
            key(DiversityClass::UnresolvedTask, "repair deploy credentials"),
        ];

        let selected = diverse_selection_indices(&ranked, 5);
        let classes: HashSet<_> = selected
            .iter()
            .filter_map(|&index| ranked[index].class)
            .collect();

        assert_eq!(classes.len(), 5);
    }
}
