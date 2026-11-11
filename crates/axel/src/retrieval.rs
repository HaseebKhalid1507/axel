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
use std::time::Duration;

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
const W_SEMANTIC: f64 = 2.5;
const W_SESSION: f64 = 1.0;
const W_CLASS: f64 = 0.75;
const W_RECENCY: f64 = 0.5;

/// Optional-local-embedding p95 target (spec §16.2). Implementations receive
/// this strict budget and must return [`EmbeddingRankError::TimedOut`] when it
/// is exhausted.
pub const LOCAL_EMBEDDING_TIMEOUT: Duration = Duration::from_millis(250);

/// What one candidate scored, signal by signal, before the weights; `total` is the weighted
/// sum the ranker sorted by. The recall log shows these numbers.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
pub struct ScoreParts {
    pub lexical: f64,
    pub semantic: f64,
    pub session: f64,
    pub class: f64,
    pub recency: f64,
    pub total: f64,
}

/// Machine-readable reason a record was selected (spec §10.4). Every
/// selected record carries at least one. Never free text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RankReason {
    /// Matched the lexical FTS query.
    LexicalMatch,
    /// Optional local embeddings contributed to hybrid ranking.
    SemanticMatch,
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

/// The lexical signal: the caller's share of matched words when it gave one, else 1 for a
/// lexical match and 0 for none (as before v2).
fn lexical_signal(c: &RecallCandidate) -> f64 {
    c.lexical_score.unwrap_or(if c.lexical_match { 1.0 } else { 0.0 })
}

fn score(c: &RecallCandidate, affinity: bool) -> f64 {
    let lexical = lexical_signal(c);
    let session = if affinity { 1.0 } else { 0.0 };
    W_LEXICAL * lexical + W_SESSION * session + W_CLASS * c.class_weight + W_RECENCY * c.recency
}

// SPEC: 5918 writes `score_parts(c: &Candidate, affinity: f64, semantic: Option<f64>)`; 3174
// writes `score_parts(c, affinity, semantic)`. There is no type `Candidate`, and the ranker's
// affinity is a `bool` (`score` above): this takes `&RecallCandidate` and a `bool`.
/// The parts of the ranker's score for one candidate, from the same constants as `score`:
/// `total` is exactly what `rank_and_select_hybrid` sorted this candidate by, given the same
/// `affinity` (its own flag or the session tag) and the same semantic score (`None` is 0;
/// clamped to `0..=1` as the ranker clamps it).
pub fn score_parts(c: &RecallCandidate, affinity: bool, semantic: Option<f64>) -> ScoreParts {
    let lexical = lexical_signal(c);
    let session = if affinity { 1.0 } else { 0.0 };
    let semantic = semantic.unwrap_or(0.0).clamp(0.0, 1.0);
    ScoreParts {
        lexical,
        semantic,
        session,
        class: c.class_weight,
        recency: c.recency,
        total: W_LEXICAL * lexical
            + W_SESSION * session
            + W_CLASS * c.class_weight
            + W_RECENCY * c.recency
            + W_SEMANTIC * semantic,
    }
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

/// Metadata-only reason that explicitly enabled embeddings fell back to the
/// lexical ranker. No query, candidate, path, or model content is retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingFallbackNote {
    ModelUnavailable,
    TimedOut,
    Failed,
    InvalidScores,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingRankError {
    TimedOut,
    Failed,
}

/// Adapter boundary for an already-enabled, already-provisioned local model.
/// It deliberately has no download or network operation.
pub trait LocalEmbeddingRanker {
    fn is_available(&self) -> bool;

    fn semantic_scores(
        &mut self,
        query: &str,
        candidates: &[RecallCandidate],
        timeout: Duration,
    ) -> Result<Vec<f64>, EmbeddingRankError>;
}

/// Closed opt-in state: default/off contains no ranker and cannot infer or
/// provision embeddings. A caller must explicitly construct `Enabled`.
#[derive(Default)]
pub enum EmbeddingRanking<'a> {
    #[default]
    Off,
    Enabled(&'a mut dyn LocalEmbeddingRanker),
}

impl<'a> EmbeddingRanking<'a> {
    pub fn enabled(ranker: &'a mut dyn LocalEmbeddingRanker) -> Self {
        Self::Enabled(ranker)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct EmbeddingRankingMetadata {
    pub embeddings_enabled: bool,
    pub embeddings_used: bool,
    pub fallback_note: Option<EmbeddingFallbackNote>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HybridRankedSelection {
    pub records: Vec<(ScopedDescriptor, Vec<RankReason>)>,
    pub metadata: EmbeddingRankingMetadata,
}

/// Rank lexically by default; only an explicit `Enabled` ranker can add the
/// bounded local semantic signal. Any unavailable, failed, malformed, or timed
/// out embedding result automatically returns the unchanged lexical ranking.
pub fn rank_and_select_hybrid(
    candidates: Vec<RecallCandidate>,
    session_id: Option<&str>,
    max_selected: usize,
    query: &str,
    embeddings: EmbeddingRanking<'_>,
) -> HybridRankedSelection {
    let lexical = || rank_and_select(candidates.clone(), session_id, max_selected);
    let EmbeddingRanking::Enabled(ranker) = embeddings else {
        return HybridRankedSelection {
            records: lexical(),
            metadata: EmbeddingRankingMetadata::default(),
        };
    };

    let mut metadata = EmbeddingRankingMetadata {
        embeddings_enabled: true,
        ..EmbeddingRankingMetadata::default()
    };
    if !ranker.is_available() {
        metadata.fallback_note = Some(EmbeddingFallbackNote::ModelUnavailable);
        return HybridRankedSelection {
            records: lexical(),
            metadata,
        };
    }

    let semantic_scores = match ranker.semantic_scores(query, &candidates, LOCAL_EMBEDDING_TIMEOUT)
    {
        Ok(scores)
            if scores.len() == candidates.len() && scores.iter().all(|score| score.is_finite()) =>
        {
            scores
        }
        Ok(_) => {
            metadata.fallback_note = Some(EmbeddingFallbackNote::InvalidScores);
            return HybridRankedSelection {
                records: lexical(),
                metadata,
            };
        }
        Err(EmbeddingRankError::TimedOut) => {
            metadata.fallback_note = Some(EmbeddingFallbackNote::TimedOut);
            return HybridRankedSelection {
                records: lexical(),
                metadata,
            };
        }
        Err(EmbeddingRankError::Failed) => {
            metadata.fallback_note = Some(EmbeddingFallbackNote::Failed);
            return HybridRankedSelection {
                records: lexical(),
                metadata,
            };
        }
    };

    metadata.embeddings_used = true;
    HybridRankedSelection {
        records: rank_current_with_semantic(
            candidates,
            session_id,
            max_selected,
            Some(&semantic_scores),
        ),
        metadata,
    }
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
            CLASS_PREFERENCE | "preferences" => Some(Self::Preference),
            CLASS_EPISODIC_TURN | CLASS_CONVERSATION_SUMMARY | CLASS_TOOL_OUTCOME | "events" | "cases" => {
                Some(Self::Episode)
            }
            CLASS_UNRESOLVED_TASK => Some(Self::UnresolvedTask),
            CLASS_ENTITY_FACT | CLASS_CORRECTION | "note" | "document" | "entities" => Some(Self::Fact),
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
    rank_current_with_semantic(candidates, session_id, max_selected, None)
}

fn rank_current_with_semantic(
    candidates: Vec<RecallCandidate>,
    session_id: Option<&str>,
    max_selected: usize,
    semantic_scores: Option<&[f64]>,
) -> Vec<(ScopedDescriptor, Vec<RankReason>)> {
    let cap = max_selected.min(MAX_SELECTED_CAP);
    if cap == 0 {
        return Vec::new();
    }
    let session_tag = session_id.map(session_tag);

    // Disclosure gate FIRST (defence in depth): a withheld record is never
    // scored, never sorted, never inspected by the ranker.
    let mut scored: Vec<(RecallCandidate, bool, f64, bool)> = candidates
        .into_iter()
        .enumerate()
        .filter(|(_, c)| model_visible(&c.descriptor))
        .filter(|(_, c)| !known_stale(c))
        .map(|(index, c)| {
            let affinity = c.session_affinity
                || session_tag
                    .as_deref()
                    .is_some_and(|t| c.descriptor.tags.iter().any(|tag| tag == t));
            let semantic = semantic_scores
                .and_then(|scores| scores.get(index))
                .copied()
                .unwrap_or(0.0)
                .clamp(0.0, 1.0);
            let s = score(&c, affinity) + W_SEMANTIC * semantic;
            (c, affinity, s, semantic > 0.0)
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
    for (candidate, affinity, candidate_score, semantic_match) in scored {
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
        ranked.push((candidate, affinity, candidate_score, semantic_match));
    }

    let diversity_keys: Vec<_> = ranked
        .iter()
        .map(|(candidate, _, score, _)| {
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
            let (candidate, affinity, _, semantic_match) = &ranked[index];
            let mut reasons = reasons_for(candidate, *affinity);
            if *semantic_match {
                reasons.push(RankReason::SemanticMatch);
            }
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

#[cfg(test)]
mod local_embedding_tests {
    use std::time::Duration;

    use super::*;
    use crate::project_memory::{RecordLinks, Retention};

    fn candidate(id: &str, title: &str) -> RecallCandidate {
        RecallCandidate {
            descriptor: ScopedDescriptor {
                id: id.into(),
                project_key: "project".into(),
                title: title.into(),
                snippet: Some(format!("{title} details")),
                category: CLASS_ENTITY_FACT.into(),
                tags: Vec::new(),
                created: "2026-01-01T00:00:00Z".into(),
                sensitivity: Sensitivity::Normal,
                retention: Retention::Standard,
                provenance: "test".into(),
                links: RecordLinks::default(),
            },
            recency: 0.5,
            session_affinity: false,
            class_weight: 0.5,
            lexical_match: true,
            recency_fallback: false,
            lexical_score: None,
        }
    }

    struct FakeRanker {
        available: bool,
        result: Result<Vec<f64>, EmbeddingRankError>,
        calls: usize,
        observed_timeout: Option<Duration>,
    }

    impl LocalEmbeddingRanker for FakeRanker {
        fn is_available(&self) -> bool {
            self.available
        }

        fn semantic_scores(
            &mut self,
            _query: &str,
            _candidates: &[RecallCandidate],
            timeout: Duration,
        ) -> Result<Vec<f64>, EmbeddingRankError> {
            self.calls += 1;
            self.observed_timeout = Some(timeout);
            self.result.clone()
        }
    }

    #[test]
    fn embeddings_are_off_by_default_and_never_inferred_by_recall() {
        let candidates = vec![candidate("lexical", "lexical winner")];
        let selected = rank_and_select_hybrid(
            candidates.clone(),
            None,
            8,
            "query",
            EmbeddingRanking::default(),
        );
        let capture_selected = rank_and_select_hybrid(
            candidates.clone(),
            None,
            8,
            "captured turn",
            EmbeddingRanking::default(),
        );
        let search_selected = rank_and_select_hybrid(
            candidates,
            None,
            8,
            "search terms",
            EmbeddingRanking::default(),
        );

        for result in [&selected, &capture_selected, &search_selected] {
            assert!(!result.metadata.embeddings_enabled);
            assert!(!result.metadata.embeddings_used);
            assert_eq!(result.metadata.fallback_note, None);
            assert_eq!(result.records[0].0.id, "lexical");
        }
    }

    #[test]
    fn enabled_but_missing_model_falls_back_with_metadata_only_note() {
        let mut ranker = FakeRanker {
            available: false,
            result: Ok(vec![1.0]),
            calls: 0,
            observed_timeout: None,
        };
        let candidates = vec![candidate("lexical", "unaltered title")];
        let selected = rank_and_select_hybrid(
            candidates,
            None,
            8,
            "query",
            EmbeddingRanking::enabled(&mut ranker),
        );

        assert_eq!(ranker.calls, 0, "missing model must not attempt inference");
        assert!(selected.metadata.embeddings_enabled);
        assert!(!selected.metadata.embeddings_used);
        assert_eq!(
            selected.metadata.fallback_note,
            Some(EmbeddingFallbackNote::ModelUnavailable)
        );
        assert_eq!(selected.records[0].0.title, "unaltered title");
    }

    #[test]
    fn timeout_falls_back_to_lexical_and_records_target_in_metadata() {
        let mut ranker = FakeRanker {
            available: true,
            result: Err(EmbeddingRankError::TimedOut),
            calls: 0,
            observed_timeout: None,
        };
        let selected = rank_and_select_hybrid(
            vec![candidate("lexical", "lexical fallback")],
            None,
            8,
            "query",
            EmbeddingRanking::enabled(&mut ranker),
        );

        assert_eq!(ranker.calls, 1);
        assert_eq!(ranker.observed_timeout, Some(LOCAL_EMBEDDING_TIMEOUT));
        assert_eq!(
            selected.metadata.fallback_note,
            Some(EmbeddingFallbackNote::TimedOut)
        );
        assert_eq!(selected.records[0].0.id, "lexical");
    }

    #[test]
    fn available_embeddings_add_a_hybrid_ranking_signal() {
        let mut ranker = FakeRanker {
            available: true,
            result: Ok(vec![0.0, 1.0]),
            calls: 0,
            observed_timeout: None,
        };
        let selected = rank_and_select_hybrid(
            vec![
                candidate("lexical-first", "first lexical candidate"),
                candidate("semantic-best", "second semantic candidate"),
            ],
            None,
            2,
            "query",
            EmbeddingRanking::enabled(&mut ranker),
        );

        assert!(selected.metadata.embeddings_used);
        assert_eq!(selected.metadata.fallback_note, None);
        assert_eq!(selected.records[0].0.id, "semantic-best");
        assert!(selected.records[0].1.contains(&RankReason::SemanticMatch));
    }
}

#[cfg(test)]
mod score_parts_tests {
    use super::*;
    use crate::chat_history::class_weight;
    use crate::project_memory::{RecordLinks, Retention};

    fn candidate(id: &str, category: &str, lexical_match: bool, lexical_score: Option<f64>, recency: f64) -> RecallCandidate {
        RecallCandidate {
            descriptor: ScopedDescriptor {
                id: id.into(),
                project_key: "project".into(),
                // Titles that share no word, so the diversity pass drops none of them.
                title: format!("title{id}"),
                snippet: None,
                category: category.into(),
                tags: Vec::new(),
                created: "2026-01-01T00:00:00Z".into(),
                sensitivity: Sensitivity::Normal,
                retention: Retention::Standard,
                provenance: "test".into(),
                links: RecordLinks::default(),
            },
            recency,
            session_affinity: false,
            class_weight: class_weight(category),
            lexical_match,
            recency_fallback: false,
            lexical_score,
        }
    }

    fn mixed() -> Vec<RecallCandidate> {
        vec![
            candidate("a", CLASS_DECISION, true, None, 0.9),
            candidate("b", CLASS_EPISODIC_TURN, false, None, 0.2),
            candidate("c", "note", true, Some(0.67), 0.5),
            candidate("d", "document", false, Some(0.0), 0.0),
            candidate("e", "entities", true, Some(0.25), 1.0),
            candidate("f", "no_such_class", true, Some(1.0), 0.01),
            candidate("g", CLASS_TOOL_OUTCOME, false, Some(0.5), 0.33),
        ]
    }

    /// Section 3, test 11: `score_parts` is the ranker's own score, bit for bit.
    #[test]
    fn score_parts_equals_the_rankers_score_for_every_candidate_of_a_mixed_set() {
        for c in mixed() {
            for affinity in [false, true] {
                for semantic in [None, Some(0.0_f64), Some(0.37), Some(1.0), Some(1.7), Some(-0.4)] {
                    // What `rank_current_with_semantic` computes for one candidate.
                    let ranker = score(&c, affinity) + W_SEMANTIC * semantic.unwrap_or(0.0).clamp(0.0, 1.0);
                    let parts = score_parts(&c, affinity, semantic);
                    assert_eq!(parts.total, ranker, "{} {affinity} {semantic:?}", c.descriptor.id);
                    assert_eq!(parts.lexical, c.lexical_score.unwrap_or(if c.lexical_match { 1.0 } else { 0.0 }));
                    assert_eq!(parts.session, if affinity { 1.0 } else { 0.0 });
                    assert_eq!(parts.class, c.class_weight);
                    assert_eq!(parts.recency, c.recency);
                    assert!((0.0..=1.0).contains(&parts.semantic));
                }
            }
        }
    }

    /// And the ranker orders by it: with one class the diversity pass keeps the score order.
    #[test]
    fn the_ranker_returns_one_class_in_the_order_of_score_parts() {
        let semantic = [0.1, 0.9, 0.4, 0.0, 0.75];
        let set: Vec<RecallCandidate> = [(true, 0.2, 0.9), (false, 0.0, 0.1), (true, 1.0, 0.5), (true, 0.6, 0.0), (false, 0.3, 0.7)]
            .iter()
            .enumerate()
            .map(|(i, (m, l, r))| candidate(&format!("n{i}"), "note", *m, Some(*l), *r))
            .collect();
        let mut expected: Vec<(f64, String)> = set
            .iter()
            .zip(semantic)
            .map(|(c, s)| (score_parts(c, false, Some(s)).total, c.descriptor.id.clone()))
            .collect();
        expected.sort_by(|a, b| b.0.total_cmp(&a.0));
        let got = rank_current_with_semantic(set, None, 8, Some(&semantic));
        let got: Vec<String> = got.into_iter().map(|(d, _)| d.id).collect();
        assert_eq!(got, expected.into_iter().map(|(_, id)| id).collect::<Vec<_>>());
    }

    /// The four edits leave an existing caller's score as it was.
    #[test]
    fn a_candidate_without_a_lexical_score_scores_as_before() {
        let with = candidate("a", CLASS_DECISION, true, None, 0.5);
        let without = candidate("b", CLASS_DECISION, false, None, 0.5);
        assert!((score(&with, false) - score(&without, false) - W_LEXICAL).abs() < 1e-12);
        assert_eq!(class_weight("note"), 0.85);
        assert_eq!(class_weight("preferences"), 0.85);
        assert_eq!(class_weight("entities"), 0.6);
        assert_eq!(class_weight("events"), 0.5);
        assert_eq!(class_weight("cases"), 0.5);
        assert_eq!(class_weight("patterns"), 0.5);
        assert_eq!(class_weight("document"), 0.45);
        assert_eq!(DiversityClass::of("note"), Some(DiversityClass::Fact));
        assert_eq!(DiversityClass::of("document"), Some(DiversityClass::Fact));
        assert_eq!(DiversityClass::of("entities"), Some(DiversityClass::Fact));
        assert_eq!(DiversityClass::of("preferences"), Some(DiversityClass::Preference));
        assert_eq!(DiversityClass::of("events"), Some(DiversityClass::Episode));
        assert_eq!(DiversityClass::of("cases"), Some(DiversityClass::Episode));
        assert_eq!(DiversityClass::of("patterns"), None);
    }
}
