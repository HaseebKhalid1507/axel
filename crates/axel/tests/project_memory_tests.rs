//! T32–T36 project-scoped memory contract tests.
//!
//! Covers: additive transactional migration, project scoping (fail closed),
//! sensitivity/retention (disclosure) classes, tombstoned forget, bounded
//! offline lexical search, crash-safe reopen, legacy `.r8` compatibility.

use axel::project_memory::{
    MemoryScope, Retention, ScopedQuery, Sensitivity, BODY_MAX_CHARS, SEARCH_LIMIT_CAP,
};
use axel::AxelBrain;
use axel_memkoshi::memory::{Memory, MemoryCategory};
use chrono::{Duration, Utc};
use tempfile::TempDir;

fn mk_memory(title: &str, content: &str) -> Memory {
    // Pipeline minimums: content >= 50 chars, title >= 10, topic >= 5.
    let mut m = Memory::new(
        MemoryCategory::Events,
        "project-memory-tests",
        format!("{title} ........."),
        format!("{content} — padded to satisfy the fifty character content validation minimum."),
    );
    m.tags = vec!["t32".to_string()];
    m
}

fn scope(project: &str) -> MemoryScope {
    MemoryScope {
        project_key: project.to_string(),
        sensitivity: Sensitivity::Normal,
        retention: Retention::Standard,
        provenance: "test-suite".to_string(),
    }
}

fn text_query(project: &str, text: &str, limit: usize) -> ScopedQuery {
    ScopedQuery {
        project_key: project.to_string(),
        text: Some(text.to_string()),
        tags: Vec::new(),
        since: None,
        until: None,
        limit,
    }
}

// ── Migration ───────────────────────────────────────────────────────────

#[test]
fn migration_adds_scope_columns_and_tables() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.r8");
    let _brain = AxelBrain::open_or_create(&path, Some("test")).unwrap();

    let conn = rusqlite::Connection::open(&path).unwrap();
    let mut cols: Vec<String> = Vec::new();
    let mut stmt = conn.prepare("PRAGMA table_info(memories)").unwrap();
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(1))
        .unwrap();
    for c in rows {
        cols.push(c.unwrap());
    }
    for expected in [
        "project_key",
        "sensitivity",
        "retention",
        "provenance",
        "expires_at",
    ] {
        assert!(cols.iter().any(|c| c == expected), "missing column {expected}");
    }
    // Tombstone + FTS tables exist.
    for table in ["memory_tombstones", "memories_fts"] {
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = ?1",
                [table],
                |r| r.get(0),
            )
            .unwrap();
        assert!(n >= 1, "missing table {table}");
    }
}

#[test]
fn legacy_r8_opens_migrates_and_old_rows_stay_out_of_project_scope() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("legacy.r8");

    // Build a "legacy" brain: raw Brain::create (pre-migration schema) plus a
    // hand-inserted legacy memory row using only the original column list.
    {
        let brain = axel::r8::Brain::create(&path, Some("legacy")).unwrap();
        brain
            .conn()
            .execute(
                "INSERT INTO memories (id, category, topic, title, abstract_text, content,
                     confidence, importance, source_sessions, tags, related_topics,
                     created, trust_level)
                 VALUES ('mem_legacy1', 'events', 'legacy-topic', 'Legacy title here',
                     '', 'Legacy content body that was written before project scoping existed at all.',
                     'medium', 0.5, '[]', '[]', '[]', ?1, 1.0)",
                [Utc::now().to_rfc3339()],
            )
            .unwrap();
    }

    // Opening with the new code must migrate without error…
    let mut brain = AxelBrain::open(&path).unwrap();
    // …keep the legacy row readable through the legacy API…
    assert!(brain.memories(10).unwrap().iter().any(|m| m.id == "mem_legacy1"));
    // …and NOT leak it into any project scope (fail closed by default).
    let hits = brain
        .search_scoped(&text_query("proj_a", "Legacy", 10))
        .unwrap();
    assert!(hits.is_empty(), "legacy rows must not cross into project scope");
    assert!(brain.fetch_scoped("mem_legacy1", "proj_a").unwrap().is_none());
}

// ── Store / fetch ───────────────────────────────────────────────────────

#[test]
fn store_and_fetch_are_project_scoped_and_fail_closed() {
    let dir = TempDir::new().unwrap();
    let mut brain = AxelBrain::open_or_create(dir.path().join("b.r8"), Some("t")).unwrap();

    let id = brain
        .store_scoped(mk_memory("Alpha decision", "We decided to use FTS5"), &scope("proj_a"))
        .unwrap();

    // Same project: full body, signature verified.
    let view = brain.fetch_scoped(&id, "proj_a").unwrap().expect("found");
    assert!(view.body.as_deref().unwrap().contains("FTS5"));
    assert_eq!(view.descriptor.project_key, "proj_a");
    assert_eq!(view.verified, Some(true));
    assert!(view.body_withheld_reason.is_none());

    // Cross-project fetch: fail closed.
    assert!(brain.fetch_scoped(&id, "proj_b").unwrap().is_none());
    // Unknown id: None, not error.
    assert!(brain.fetch_scoped("mem_nope", "proj_a").unwrap().is_none());
}

#[test]
fn never_persist_is_rejected_and_never_enters_the_r8() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("b.r8");
    let mut brain = AxelBrain::open_or_create(&path, Some("t")).unwrap();

    let sentinel = "NEVERPERSIST-SENTINEL-73cd21";
    let mut s = scope("proj_a");
    s.retention = Retention::NeverPersist;
    let err = brain.store_scoped(mk_memory("Ephemeral secret", sentinel), &s);
    assert!(err.is_err(), "NeverPersist must be rejected at the persistence boundary");
    drop(brain);

    let bytes = std::fs::read(&path).unwrap();
    assert!(
        !bytes
            .windows(sentinel.len())
            .any(|w| w == sentinel.as_bytes()),
        "NeverPersist sentinel leaked into the .r8 file"
    );
}

#[test]
fn secret_sensitivity_has_no_snippet_and_body_is_withheld() {
    let dir = TempDir::new().unwrap();
    let mut brain = AxelBrain::open_or_create(dir.path().join("b.r8"), Some("t")).unwrap();

    let sentinel = "SECRET-BODY-SENTINEL-a1b2c3";
    let mut s = scope("proj_a");
    s.sensitivity = Sensitivity::Secret;
    let id = brain
        .store_scoped(mk_memory("Credential note", sentinel), &s)
        .unwrap();

    // Search: descriptor visible, but no body-derived snippet.
    let hits = brain
        .search_scoped(&text_query("proj_a", "Credential", 10))
        .unwrap();
    let hit = hits.iter().find(|h| h.id == id).expect("descriptor listed");
    assert!(hit.snippet.is_none(), "secret memories must not expose snippets");
    let ser = serde_json::to_string(&hits).unwrap();
    assert!(!ser.contains(sentinel), "secret body leaked through search");

    // Search by secret body text must not match (body not FTS-indexed).
    let by_body = brain
        .search_scoped(&text_query("proj_a", sentinel, 10))
        .unwrap();
    assert!(by_body.is_empty(), "secret body must not be lexically indexed");

    // Fetch: body withheld with a reason.
    let view = brain.fetch_scoped(&id, "proj_a").unwrap().unwrap();
    assert!(view.body.is_none());
    assert!(view.body_withheld_reason.is_some());
}

#[test]
fn restricted_retention_classes_withhold_body_at_model_boundary() {
    let dir = TempDir::new().unwrap();
    let mut brain = AxelBrain::open_or_create(dir.path().join("b.r8"), Some("t")).unwrap();

    for retention in [
        Retention::LocalOnly,
        Retention::VisibleAfterConsent,
        Retention::PersistNeverTransmit,
    ] {
        let mut s = scope("proj_a");
        s.retention = retention;
        let id = brain
            .store_scoped(mk_memory("Restricted note", "restricted body content"), &s)
            .unwrap();
        let view = brain.fetch_scoped(&id, "proj_a").unwrap().unwrap();
        assert!(view.body.is_none(), "{retention:?} body must be withheld");
        assert!(view.body_withheld_reason.is_some());
    }
}

// ── Search ──────────────────────────────────────────────────────────────

#[test]
fn search_supports_text_tag_time_filters_and_hard_limit_cap() {
    let dir = TempDir::new().unwrap();
    let mut brain = AxelBrain::open_or_create(dir.path().join("b.r8"), Some("t")).unwrap();

    for i in 0..10 {
        let mut m = mk_memory(&format!("Widget note {i}"), &format!("widget body number {i}"));
        m.tags = vec![if i % 2 == 0 { "even" } else { "odd" }.to_string()];
        brain.store_scoped(m, &scope("proj_a")).unwrap();
    }
    // Different project — must never appear.
    brain
        .store_scoped(mk_memory("Widget foreign", "widget body foreign"), &scope("proj_b"))
        .unwrap();

    // Text search, bounded.
    let hits = brain.search_scoped(&text_query("proj_a", "widget", 4)).unwrap();
    assert_eq!(hits.len(), 4);
    assert!(hits.iter().all(|h| h.project_key == "proj_a"));

    // Tag filter.
    let q = ScopedQuery {
        tags: vec!["even".to_string()],
        ..text_query("proj_a", "widget", 20)
    };
    let hits = brain.search_scoped(&q).unwrap();
    assert_eq!(hits.len(), 5);

    // Timestamp filter: everything is recent, so until=past excludes all.
    let q = ScopedQuery {
        until: Some(Utc::now() - Duration::days(1)),
        ..text_query("proj_a", "widget", 20)
    };
    assert!(brain.search_scoped(&q).unwrap().is_empty());

    // Limit is hard-capped.
    let q = text_query("proj_a", "widget", 10_000);
    let hits = brain.search_scoped(&q).unwrap();
    assert!(hits.len() <= SEARCH_LIMIT_CAP);

    // No-text listing works (project-scoped, recency ordered).
    let q = ScopedQuery {
        text: None,
        ..text_query("proj_a", "", 3)
    };
    assert_eq!(brain.search_scoped(&q).unwrap().len(), 3);
}

#[test]
fn search_snippets_are_bounded() {
    let dir = TempDir::new().unwrap();
    let mut brain = AxelBrain::open_or_create(dir.path().join("b.r8"), Some("t")).unwrap();
    let long_body = "needle ".repeat(2000);
    brain
        .store_scoped(mk_memory("Long body doc", &long_body), &scope("proj_a"))
        .unwrap();
    let hits = brain.search_scoped(&text_query("proj_a", "needle", 5)).unwrap();
    let snip = hits[0].snippet.as_deref().unwrap();
    assert!(snip.chars().count() <= axel::project_memory::SNIPPET_MAX_CHARS);
}

#[test]
fn fetch_body_is_bounded() {
    let dir = TempDir::new().unwrap();
    let mut brain = AxelBrain::open_or_create(dir.path().join("b.r8"), Some("t")).unwrap();
    let long_body = "x".repeat(BODY_MAX_CHARS * 3);
    let id = brain
        .store_scoped(mk_memory("Huge body doc", &long_body), &scope("proj_a"))
        .unwrap();
    let view = brain.fetch_scoped(&id, "proj_a").unwrap().unwrap();
    assert!(view.body.as_deref().unwrap().chars().count() <= BODY_MAX_CHARS);
    assert!(view.body_truncated);
}

#[test]
fn expired_memories_are_excluded() {
    let dir = TempDir::new().unwrap();
    let mut brain = AxelBrain::open_or_create(dir.path().join("b.r8"), Some("t")).unwrap();
    let mut m = mk_memory("Expiring note", "expiring body text");
    m.expires_at = Some(Utc::now() - Duration::hours(1));
    let id = brain.store_scoped(m, &scope("proj_a")).unwrap();
    assert!(brain.search_scoped(&text_query("proj_a", "expiring", 10)).unwrap().is_empty());
    assert!(brain.fetch_scoped(&id, "proj_a").unwrap().is_none());
}

// ── Forget ──────────────────────────────────────────────────────────────

#[test]
fn forget_is_exact_project_scoped_tombstoned_and_search_excludes() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("b.r8");
    let mut brain = AxelBrain::open_or_create(&path, Some("t")).unwrap();

    let id = brain
        .store_scoped(mk_memory("Forget me note", "forgettable body text"), &scope("proj_a"))
        .unwrap();
    // Wrong project cannot forget (fail closed).
    assert!(!brain.forget_scoped(&id, "proj_b").unwrap());
    // Right project can.
    assert!(brain.forget_scoped(&id, "proj_a").unwrap());
    // Search + fetch exclude it.
    assert!(brain.search_scoped(&text_query("proj_a", "forgettable", 10)).unwrap().is_empty());
    assert!(brain.fetch_scoped(&id, "proj_a").unwrap().is_none());
    // Tombstone recorded.
    let conn = rusqlite::Connection::open(&path).unwrap();
    let n: i64 = conn
        .query_row("SELECT COUNT(*) FROM memory_tombstones WHERE id = ?1", [&id], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 1);
    // Re-storing under a tombstoned id is refused.
    let mut m = mk_memory("Zombie note", "zombie body text for tombstone check");
    m.id = id.clone();
    assert!(brain.store_scoped(m, &scope("proj_a")).is_err());
}

// ── Durability / offline ────────────────────────────────────────────────

#[test]
fn stored_memories_survive_reopen() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("b.r8");
    let id = {
        let mut brain = AxelBrain::open_or_create(&path, Some("t")).unwrap();
        brain
            .store_scoped(mk_memory("Durable note", "durable body text"), &scope("proj_a"))
            .unwrap()
    };
    let brain = AxelBrain::open(&path).unwrap();
    let view = brain.fetch_scoped(&id, "proj_a").unwrap().expect("survives reopen");
    assert!(view.body.as_deref().unwrap().contains("durable"));
}

#[test]
fn lexical_search_is_offline_and_never_touches_the_model_cache() {
    // Point the model cache at an empty temp dir; a lexical store+search run
    // must not create anything under it (no download, no embedder init).
    let cache = TempDir::new().unwrap();
    std::env::set_var("XDG_CACHE_HOME", cache.path());
    let dir = TempDir::new().unwrap();
    let mut brain = AxelBrain::open_or_create(dir.path().join("b.r8"), Some("t")).unwrap();
    brain
        .store_scoped(mk_memory("Offline note", "offline lexical body"), &scope("proj_a"))
        .unwrap();
    let hits = brain.search_scoped(&text_query("proj_a", "offline", 5)).unwrap();
    assert_eq!(hits.len(), 1);
    let velocirag_dir = cache.path().join("velocirag");
    assert!(
        !velocirag_dir.exists(),
        "lexical search must not initialise the embedding model cache"
    );
}

#[test]
fn batch_store_is_transactional_and_searchable() {
    let dir = TempDir::new().unwrap();
    let mut brain = AxelBrain::open_or_create(dir.path().join("b.r8"), Some("t")).unwrap();
    let items: Vec<(Memory, MemoryScope)> = (0..20)
        .map(|i| (mk_memory(&format!("Batch note {i}"), &format!("batch body {i}")), scope("proj_a")))
        .collect();
    let ids = brain.store_scoped_batch(items).unwrap();
    assert_eq!(ids.len(), 20);
    let hits = brain.search_scoped(&text_query("proj_a", "batch", 50)).unwrap();
    assert_eq!(hits.len(), 20);
}

#[test]
fn retention_stats_are_project_scoped() {
    let dir = TempDir::new().unwrap();
    let mut brain = AxelBrain::open_or_create(dir.path().join("b.r8"), Some("t")).unwrap();
    brain.store_scoped(mk_memory("Standard one", "standard body one"), &scope("proj_a")).unwrap();
    let mut s = scope("proj_a");
    s.retention = Retention::LocalOnly;
    brain.store_scoped(mk_memory("Local only one", "local only body"), &s).unwrap();
    brain.store_scoped(mk_memory("Other project", "other project body"), &scope("proj_b")).unwrap();

    let stats = brain.retention_stats_scoped("proj_a").unwrap();
    let total: u64 = stats.iter().map(|(_, n)| n).sum();
    assert_eq!(total, 2);
    assert!(stats.iter().any(|(k, n)| k == "standard" && *n == 1));
    assert!(stats.iter().any(|(k, n)| k == "local_only" && *n == 1));
}
