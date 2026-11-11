//! Task B2 contract tests: bounded chat-history candidate generation and
//! explainable ranking (spec §8.1, §9.1–9.2, §9.5, §10.4, §16.2).
//!
//! Connection setup mirrors the existing project_memory tests: a real
//! temp-file sqlite `.r8` (TempDir + `r8::Brain::create` + the same
//! `project_memory::migrate` that `AxelBrain::open*` runs).

use axel::chat_history::{
    generate_candidates, session_tag, store_capture_with_id, CaptureRequest, Correction,
    EpisodicRecord, EpisodicTurn, RecallCandidate, CANDIDATES_RETAINED_CAP, CLASS_DECISION,
    CLASS_EPISODIC_TURN, CLASS_PREFERENCE, CLASS_UNRESOLVED_TASK, RECENCY_FALLBACK_LIMIT,
    RECENCY_FALLBACK_WINDOW_HOURS,
};
use axel::project_memory::{
    self, MemoryScope, RecordLinks, Retention, ScopedDescriptor, Sensitivity,
};
use axel::r8::Brain;
use axel::retrieval::{
    rank_and_select, rank_and_select_with_metrics, RankReason, MAX_SELECTED_CAP,
};
use axel_memkoshi::memory::{Memory, MemoryCategory};
use chrono::{DateTime, Duration, Utc};
use rusqlite::Connection;
use tempfile::TempDir;

const PROJECT: &str = "proj_chat";

fn setup() -> (TempDir, Brain) {
    let dir = TempDir::new().unwrap();
    let brain = Brain::create(&dir.path().join("b.r8"), Some("chat-history-tests")).unwrap();
    project_memory::migrate(brain.conn()).unwrap();
    (dir, brain)
}

#[allow(clippy::too_many_arguments)]
fn store_record(
    conn: &Connection,
    class: &str,
    title: &str,
    content: &str,
    tags: Vec<String>,
    sensitivity: Sensitivity,
    retention: Retention,
    created: Option<DateTime<Utc>>,
) -> String {
    let mut m = Memory::new(
        MemoryCategory::Events,
        "chat-history-tests",
        title.to_string(),
        content.to_string(),
    );
    m.tags = tags;
    if let Some(t) = created {
        m.created = t;
    }
    let id = m.id.clone();
    let scope = MemoryScope {
        project_key: PROJECT.to_string(),
        sensitivity,
        retention,
        provenance: "test-suite".to_string(),
    };
    project_memory::store_scoped(conn, &m, &scope).unwrap();
    // Record classes live in the EXISTING free-text `category` column
    // (spec §8.1) — no schema change, just the fixed class constant.
    conn.execute(
        "UPDATE memories SET category = ?1 WHERE id = ?2",
        rusqlite::params![class, id],
    )
    .unwrap();
    id
}

fn normal_record(conn: &Connection, class: &str, title: &str, content: &str) -> String {
    store_record(
        conn,
        class,
        title,
        content,
        Vec::new(),
        Sensitivity::Normal,
        Retention::Standard,
        None,
    )
}

// ── Bounds: candidates ≤128, selection ≤8 (spec §16.2) ──────────────────

#[test]
fn hundred_records_stay_within_candidate_and_selection_caps() {
    let (_dir, brain) = setup();
    let conn = brain.conn();
    for i in 0..100 {
        normal_record(
            conn,
            CLASS_EPISODIC_TURN,
            &format!("Needle record number {i:03}"),
            &format!("needle shared body content for bounded retrieval test row {i:03}"),
        );
    }

    // Even an absurd requested limit is capped by construction.
    let candidates = generate_candidates(conn, PROJECT, None, Some("needle"), 10_000).unwrap();
    assert!(!candidates.is_empty());
    assert!(
        candidates.len() <= CANDIDATES_RETAINED_CAP,
        "candidates_retained {} exceeds cap {CANDIDATES_RETAINED_CAP}",
        candidates.len()
    );

    let selected = rank_and_select(candidates, None, 100);
    assert!(!selected.is_empty());
    assert!(
        selected.len() <= MAX_SELECTED_CAP,
        "selected {} exceeds cap {MAX_SELECTED_CAP}",
        selected.len()
    );
    // Explainability (spec §10.4): every selected record carries ≥1
    // machine-readable rank reason.
    for (desc, reasons) in &selected {
        assert!(
            !reasons.is_empty(),
            "record {} selected without a rank reason",
            desc.id
        );
        assert!(reasons.contains(&RankReason::LexicalMatch));
    }
    // Exact duplicate ids are impossible in the selection.
    let mut ids: Vec<&str> = selected.iter().map(|(d, _)| d.id.as_str()).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), selected.len());
}

// ── Disclosure: secret / withheld records never surface ─────────────────

#[test]
fn secret_record_is_never_a_candidate_or_selected_even_on_lexical_match() {
    let (_dir, brain) = setup();
    let conn = brain.conn();
    // Secret titles ARE fts-indexed (title-only), so this would lexically
    // match — it must still never become a candidate.
    let secret_id = store_record(
        conn,
        CLASS_EPISODIC_TURN,
        "needle credential secret title",
        "SECRET-SENTINEL-BODY should never be seen by ranking",
        Vec::new(),
        Sensitivity::Secret,
        Retention::Standard,
        None,
    );
    // Withheld retention class: model_visible() == false — same exclusion.
    let local_only_id = store_record(
        conn,
        CLASS_DECISION,
        "needle local only decision",
        "needle body that is persisted locally but never model-visible",
        Vec::new(),
        Sensitivity::Normal,
        Retention::LocalOnly,
        None,
    );
    let normal_id = normal_record(
        conn,
        CLASS_DECISION,
        "needle normal decision",
        "needle body that is perfectly fine to disclose to the model",
    );

    let candidates = generate_candidates(conn, PROJECT, None, Some("needle"), 50).unwrap();
    assert!(candidates.iter().any(|c| c.descriptor.id == normal_id));
    assert!(
        candidates.iter().all(|c| c.descriptor.id != secret_id),
        "secret record leaked into candidates"
    );
    assert!(
        candidates.iter().all(|c| c.descriptor.id != local_only_id),
        "non-model-visible retention record leaked into candidates"
    );

    let selected = rank_and_select(candidates, None, 8);
    assert!(selected
        .iter()
        .all(|(d, _)| d.id != secret_id && d.id != local_only_id));

    // Defence in depth: even a hand-constructed secret candidate is
    // excluded BEFORE ranking.
    let smuggled = RecallCandidate {
        descriptor: ScopedDescriptor {
            id: "mem_smuggled_secret".to_string(),
            project_key: PROJECT.to_string(),
            title: "smuggled secret".to_string(),
            snippet: None,
            category: CLASS_DECISION.to_string(),
            tags: Vec::new(),
            created: Utc::now().to_rfc3339(),
            sensitivity: Sensitivity::Secret,
            retention: Retention::Standard,
            provenance: "test".to_string(),
            links: RecordLinks::default(),
        },
        recency: 1.0,
        session_affinity: true,
        class_weight: 1.0,
        lexical_match: true,
        recency_fallback: false,
        lexical_score: None,
    };
    let selected = rank_and_select(vec![smuggled], None, 8);
    assert!(
        selected.is_empty(),
        "hand-built secret candidate was ranked"
    );
}

// ── Fallback: only on zero lexical rows, small and bounded (spec §9.5) ──

#[test]
fn recency_fallback_triggers_only_on_empty_lexical_and_stays_bounded() {
    let (_dir, brain) = setup();
    let conn = brain.conn();
    for i in 0..30 {
        normal_record(
            conn,
            CLASS_EPISODIC_TURN,
            &format!("Recent turn number {i:02}"),
            &format!("recent conversational body text for fallback test row {i:02}"),
        );
    }
    // A record older than the fallback window must never enter the fallback.
    let stale_id = store_record(
        conn,
        CLASS_EPISODIC_TURN,
        "Stale ancient turn",
        "ancient body text well outside the recency fallback window",
        Vec::new(),
        Sensitivity::Normal,
        Retention::Standard,
        Some(Utc::now() - Duration::hours(RECENCY_FALLBACK_WINDOW_HOURS + 24)),
    );

    // Lexical hit ⇒ no fallback candidates at all.
    let candidates = generate_candidates(conn, PROJECT, None, Some("conversational"), 128).unwrap();
    assert!(!candidates.is_empty());
    assert!(
        candidates.iter().all(|c| !c.recency_fallback),
        "fallback ran despite lexical hits"
    );
    assert!(candidates.iter().all(|c| c.lexical_match));

    // Zero lexical rows ⇒ fallback, capped at RECENCY_FALLBACK_LIMIT.
    let candidates =
        generate_candidates(conn, PROJECT, None, Some("zzzznomatchquery"), 128).unwrap();
    assert!(!candidates.is_empty(), "fallback produced nothing");
    assert!(
        candidates.len() <= RECENCY_FALLBACK_LIMIT,
        "fallback returned {} > bound {RECENCY_FALLBACK_LIMIT}",
        candidates.len()
    );
    assert!(candidates
        .iter()
        .all(|c| c.recency_fallback && !c.lexical_match));
    assert!(
        candidates.iter().all(|c| c.descriptor.id != stale_id),
        "record outside the recency window entered the fallback"
    );
    // Fallback selections are explainable too.
    let selected = rank_and_select(candidates, None, 8);
    assert!(!selected.is_empty());
    for (_, reasons) in &selected {
        assert!(reasons.contains(&RankReason::RecencyWindow));
    }
}

// ── Explainability: class-specific machine-readable reasons ─────────────

#[test]
fn every_selected_record_has_nonempty_class_aware_rank_reasons() {
    let (_dir, brain) = setup();
    let conn = brain.conn();
    let decision_id = normal_record(
        conn,
        CLASS_DECISION,
        "needle decision about storage",
        "needle we decided to reuse the project memory store entirely",
    );
    let preference_id = normal_record(
        conn,
        CLASS_PREFERENCE,
        "needle preference about style",
        "needle the user prefers bounded explainable retrieval always",
    );
    let task_id = normal_record(
        conn,
        CLASS_UNRESOLVED_TASK,
        "needle unresolved migration task",
        "needle the diversity MMR pass is still open work for task E2",
    );

    let candidates = generate_candidates(conn, PROJECT, None, Some("needle"), 128).unwrap();
    let selected = rank_and_select(candidates, None, 8);
    assert_eq!(selected.len(), 3);
    for (desc, reasons) in &selected {
        assert!(!reasons.is_empty(), "record {} has no rank reason", desc.id);
        if desc.id == decision_id {
            assert!(reasons.contains(&RankReason::ExplicitDecision));
        } else if desc.id == preference_id {
            assert!(reasons.contains(&RankReason::ExplicitPreference));
        } else if desc.id == task_id {
            assert!(reasons.contains(&RankReason::UnresolvedTaskState));
        }
    }
}

// ── Session affinity outranks at equal lexical relevance (spec §9.1/9.2) ─

#[test]
fn session_tagged_records_outrank_untagged_at_equal_lexical_relevance() {
    let (_dir, brain) = setup();
    let conn = brain.conn();
    let session = "sess-alpha-42";

    let mut tagged: Vec<String> = Vec::new();
    // Identical token structure/length ⇒ equal bm25 lexical relevance.
    for i in 0..3 {
        tagged.push(store_record(
            conn,
            CLASS_EPISODIC_TURN,
            &format!("Needle turn tagged {i:02}"),
            &format!("needle shared context body row {i:02}"),
            vec![session_tag(session)],
            Sensitivity::Normal,
            Retention::Standard,
            None,
        ));
    }
    // Untagged rows stored LAST (newer) so recency slightly favours them —
    // session affinity must still win.
    for i in 3..6 {
        normal_record(
            conn,
            CLASS_EPISODIC_TURN,
            &format!("Needle turn untagg {i:02}"),
            &format!("needle shared context body row {i:02}"),
        );
    }

    let candidates =
        generate_candidates(conn, PROJECT, Some(session), Some("needle"), 128).unwrap();
    assert_eq!(candidates.len(), 6);
    assert_eq!(candidates.iter().filter(|c| c.session_affinity).count(), 3);

    let selected = rank_and_select(candidates, Some(session), 3);
    assert_eq!(selected.len(), 3);
    for (desc, reasons) in &selected {
        assert!(
            tagged.contains(&desc.id),
            "untagged record {} outranked session-affine records",
            desc.id
        );
        assert!(reasons.contains(&RankReason::RecentSession));
    }
}

#[test]
fn correction_supersedes_target_and_stale_rate_is_computable() {
    let (_dir, brain) = setup();
    let conn = brain.conn();
    let old_id = 0xe100_0000_0000_0001;
    let correction_id = 0xe100_0000_0000_0002;
    let request = |record| CaptureRequest {
        project_key: PROJECT.into(),
        sensitivity: Sensitivity::Normal,
        retention: Retention::Standard,
        provenance: "test-suite".into(),
        record,
        title: Some("needle database location".into()),
        tags: vec![],
        entities: vec![],
    };
    store_capture_with_id(
        conn,
        old_id,
        &request(EpisodicRecord::Turn(EpisodicTurn {
            content: "needle database is in east".into(),
            source_session_id: None,
            source_turn_id: None,
        })),
    )
    .unwrap();
    store_capture_with_id(
        conn,
        correction_id,
        &request(EpisodicRecord::Correction(Correction {
            correction: "needle database is in west".into(),
            supersedes_id: Some(old_id),
            source_session_id: None,
            source_turn_id: None,
        })),
    )
    .unwrap();

    let persisted: (String, String) = conn
        .query_row(
            "SELECT supersedes, confirmed_by FROM memories WHERE id = ?1",
            [format!("mem_{correction_id:016x}")],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(persisted.0, format!("[\"mem_{old_id:016x}\"]"));
    assert_eq!(persisted.1, "[]");

    let candidates = generate_candidates(conn, PROJECT, None, Some("needle database"), 16).unwrap();
    let selected = rank_and_select_with_metrics(candidates, None, 8);
    assert!(selected
        .records
        .iter()
        .any(|(d, _)| d.id == format!("mem_{correction_id:016x}")));
    assert!(selected
        .records
        .iter()
        .all(|(d, _)| d.id != format!("mem_{old_id:016x}")));
    assert_eq!(selected.metrics.stale_candidates, 1);
    assert_eq!(selected.metrics.stale_selected, 0);
    assert_eq!(selected.metrics.stale_selected_rate(), 0.0);
}

#[test]
fn tombstoned_id_does_not_resurface_through_links_after_index_rebuild() {
    let (_dir, brain) = setup();
    let conn = brain.conn();
    let old_id = 0xe100_0000_0000_0011;
    let correction_id = 0xe100_0000_0000_0012;
    let request = |record| CaptureRequest {
        project_key: PROJECT.into(),
        sensitivity: Sensitivity::Normal,
        retention: Retention::Standard,
        provenance: "test-suite".into(),
        record,
        title: Some("needle cache policy".into()),
        tags: vec![],
        entities: vec![],
    };
    store_capture_with_id(
        conn,
        old_id,
        &request(EpisodicRecord::Turn(EpisodicTurn {
            content: "needle cache is unbounded".into(),
            source_session_id: None,
            source_turn_id: None,
        })),
    )
    .unwrap();
    store_capture_with_id(
        conn,
        correction_id,
        &request(EpisodicRecord::Correction(Correction {
            correction: "needle cache is bounded".into(),
            supersedes_id: Some(old_id),
            source_session_id: None,
            source_turn_id: None,
        })),
    )
    .unwrap();
    let old_storage_id = format!("mem_{old_id:016x}");
    project_memory::forget_scoped(conn, &old_storage_id, PROJECT).unwrap();
    project_memory::rebuild_search_index(conn).unwrap();

    let candidates = generate_candidates(conn, PROJECT, None, Some("needle cache"), 16).unwrap();
    assert!(candidates.iter().all(|c| c.descriptor.id != old_storage_id));
    assert!(candidates
        .iter()
        .flat_map(|c| c.descriptor.links.all_ids())
        .all(|id| id != old_storage_id));
}

// ── Dedup: exact duplicate ids collapse before the cap applies ──────────

#[test]
fn exact_duplicate_candidate_ids_are_deduped_in_selection() {
    let (_dir, brain) = setup();
    let conn = brain.conn();
    normal_record(
        conn,
        CLASS_DECISION,
        "needle duplicated decision",
        "needle body for the duplicate id dedup contract check",
    );
    let mut candidates = generate_candidates(conn, PROJECT, None, Some("needle"), 16).unwrap();
    assert_eq!(candidates.len(), 1);
    let dup = candidates[0].clone();
    candidates.push(dup.clone());
    candidates.push(dup);

    let selected = rank_and_select(candidates, None, 8);
    assert_eq!(selected.len(), 1, "duplicate ids must collapse to one");
}
