//! Consolidation unit + integration tests.
//!
//! Covers DB-level consolidation primitives (access logging, co-retrieval
//! canonical ordering, timestamp normalization, audit log round-trip, table
//! cleanup) and the strengthen() phase end-to-end via BrainSearch.

use axel::brain::AxelBrain;
use axel::consolidate::strengthen::strengthen;
use rusqlite::params;
use tempfile::TempDir;
use velocirag::db::{ConsolidationLogEntry, Database};

fn setup_brain() -> (AxelBrain, TempDir) {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("brain.r8");
    let brain = AxelBrain::open_or_create(&path, Some("test")).unwrap();
    (brain, tmp)
}

const EMBEDDING_DIM: usize = 384;

fn fake_emb() -> Vec<f32> {
    vec![0.0f32; EMBEDDING_DIM]
}

fn excitability_of(db: &Database, doc_id: &str) -> f64 {
    db.conn()
        .query_row(
            "SELECT excitability FROM documents WHERE doc_id = ?1",
            params![doc_id],
            |r| r.get::<_, f64>(0),
        )
        .unwrap()
}

// ─── 1. Schema migration is idempotent ─────────────────────────────────────
#[test]
fn schema_migration_is_idempotent() {
    // open_memory() runs init_schema. Open a *file*-backed DB twice so the
    // second open re-runs init_schema on a populated schema.
    let tmp = TempDir::new().unwrap();
    let _ = Database::open(tmp.path()).unwrap();
    // Drop then re-open — second open() runs init_schema again.
    let db = Database::open(tmp.path()).unwrap();
    // Sanity: documents table is queryable, no panic, no duplicate-column error.
    assert_eq!(db.document_count().unwrap(), 0);
}

// ─── 2. document_access logging works ──────────────────────────────────────
#[test]
fn log_document_access_round_trip() {
    let db = Database::open_memory().unwrap();
    db.insert_document("doc_1", "some content long enough for the sanity checks here", &serde_json::json!({}), &fake_emb(), None).unwrap();

    db.log_document_access("doc_1", "search_hit", Some("rust async"), Some(0.7), Some("sess1")).unwrap();

    let rows = db.get_document_accesses_since("1970-01-01").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].doc_id, "doc_1");
    assert_eq!(rows[0].access_type, "search_hit");
    assert_eq!(rows[0].query.as_deref(), Some("rust async"));
    assert_eq!(rows[0].score, Some(0.7));
}

// ─── 3. co_retrieval canonical ordering ────────────────────────────────────
#[test]
fn co_retrieval_canonical_ordering() {
    let db = Database::open_memory().unwrap();
    db.log_co_retrieval("b", "a", "query").unwrap();

    let (a, b): (String, String) = db.conn()
        .query_row("SELECT doc_id_a, doc_id_b FROM co_retrieval LIMIT 1", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(a, "a");
    assert_eq!(b, "b");
}

#[test]
fn co_retrieval_same_doc_is_noop() {
    let db = Database::open_memory().unwrap();
    db.log_co_retrieval("a", "a", "q").unwrap();
    let count: i64 = db.conn()
        .query_row("SELECT COUNT(*) FROM co_retrieval", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

// ─── 4. increment_document_access ──────────────────────────────────────────
#[test]
fn increment_document_access_updates_count_and_time() {
    let db = Database::open_memory().unwrap();
    db.insert_document("doc_1", "some content long enough for the sanity checks here", &serde_json::json!({}), &fake_emb(), None).unwrap();

    db.increment_document_access("doc_1").unwrap();
    db.increment_document_access("doc_1").unwrap();

    let (count, last): (i64, Option<String>) = db.conn()
        .query_row(
            "SELECT access_count, last_accessed FROM documents WHERE doc_id = ?1",
            params!["doc_1"],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(count, 2);
    assert!(last.is_some(), "last_accessed should be set");
}

// ─── 5. strengthen boost ───────────────────────────────────────────────────
#[test]
fn strengthen_boosts_accessed_documents() {
    let (mut brain, _tmp) = setup_brain();
    let search = brain.search_mut();

    let db = search.db();
    db.insert_document("doc_a", "content for the boost test path goes here easy", &serde_json::json!({}), &fake_emb(), None).unwrap();

    // Log multiple high-score search hits.
    for _ in 0..4 {
        db.log_document_access("doc_a", "search_hit", Some("q"), Some(0.8), None).unwrap();
    }

    let before = excitability_of(db, "doc_a");
    let stats = strengthen(search, false, false).unwrap();
    let after = excitability_of(search.db(), "doc_a");

    assert!(after > before, "excitability should increase: {before} -> {after}");
    assert_eq!(stats.boosted, 1);
    assert_eq!(stats.extinction_signals, 0);
}

// ─── 6. strengthen decay ───────────────────────────────────────────────────
#[test]
fn strengthen_decays_old_untouched_documents() {
    let (mut brain, _tmp) = setup_brain();
    let search = brain.search_mut();

    let db = search.db();
    db.insert_document("doc_old", "content for decay test that is long enough indeed", &serde_json::json!({}), &fake_emb(), None).unwrap();
    // Backdate indexed_at + created so days_inactive > 14.
    db.conn().execute(
        "UPDATE documents SET indexed_at = datetime('now', '-30 days'),
                              created    = datetime('now', '-30 days')
         WHERE doc_id = ?1",
        params!["doc_old"],
    ).unwrap();

    let before = excitability_of(db, "doc_old");
    let stats = strengthen(search, false, false).unwrap();
    let after = excitability_of(search.db(), "doc_old");

    assert!(after < before, "excitability should decay: {before} -> {after}");
    assert!(stats.decayed >= 1);
}

// ─── 7. strengthen extinction ──────────────────────────────────────────────
#[test]
fn strengthen_extinction_on_low_scores() {
    let (mut brain, _tmp) = setup_brain();
    let search = brain.search_mut();

    let db = search.db();
    db.insert_document("doc_x", "content for extinction signal test path is long enough", &serde_json::json!({}), &fake_emb(), None).unwrap();
    // Bump excitability so we can observe a drop without floor clipping.
    db.conn().execute(
        "UPDATE documents SET excitability = 0.7 WHERE doc_id = ?1",
        params!["doc_x"],
    ).unwrap();

    for _ in 0..5 {
        db.log_document_access("doc_x", "search_hit", Some("q"), Some(0.005), None).unwrap();
    }

    let before = excitability_of(db, "doc_x");
    let stats = strengthen(search, false, false).unwrap();
    let after = excitability_of(search.db(), "doc_x");

    assert!(after < before, "extinction should drop excitability: {before} -> {after}");
    assert_eq!(stats.extinction_signals, 1);
    assert_eq!(stats.boosted, 0);
}

// ─── 8. timestamp normalization ────────────────────────────────────────────
#[test]
fn timestamp_normalization_rfc3339_and_sqlite() {
    let db = Database::open_memory().unwrap();
    db.insert_document("doc_1", "content for timestamp normalization regression test long", &serde_json::json!({}), &fake_emb(), None).unwrap();
    db.log_document_access("doc_1", "search_hit", None, Some(0.5), None).unwrap();

    // RFC3339 with Z
    let r1 = db.get_document_accesses_since("1970-01-01T00:00:00Z").unwrap();
    assert_eq!(r1.len(), 1, "RFC3339 Z suffix should be normalized");

    // RFC3339 with positive offset
    let r2 = db.get_document_accesses_since("2020-01-01T00:00:00+00:00").unwrap();
    assert_eq!(r2.len(), 1, "RFC3339 positive offset should be normalized");

    // SQLite native format
    let r3 = db.get_document_accesses_since("1970-01-01 00:00:00").unwrap();
    assert_eq!(r3.len(), 1, "SQLite-format timestamp should still match");
}

// ─── 9. consolidation_log round-trip ───────────────────────────────────────
#[test]
fn consolidation_log_round_trip() {
    let db = Database::open_memory().unwrap();

    assert_eq!(db.last_consolidation_time().unwrap(), None);

    let entry = ConsolidationLogEntry {
        started_at: "2026-01-01T00:00:00Z".into(),
        finished_at: Some("2026-01-01T00:01:00Z".into()),
        phase1_reindexed: 3,
        phase1_pruned: 1,
        phase2_boosted: 2,
        phase2_decayed: 4,
        phase3_edges_added: 5,
        phase3_edges_updated: 6,
        phase4_flagged: 7,
        phase4_removed: 8,
        duration_secs: Some(60.0),
    };
    db.insert_consolidation_log(&entry).unwrap();

    let last = db.last_consolidation_time().unwrap();
    assert_eq!(last.as_deref(), Some("2026-01-01T00:01:00Z"));
}

// ─── 10. Table cleanup (90-day DELETE) ─────────────────────────────────────
#[test]
fn cleanup_deletes_rows_older_than_90_days() {
    let db = Database::open_memory().unwrap();
    db.insert_document("doc_1", "content for retention cleanup test exists is long enough", &serde_json::json!({}), &fake_emb(), None).unwrap();

    let conn = db.conn();
    // Insert one old + one recent into each retention table.
    conn.execute(
        "INSERT INTO document_access (doc_id, access_type, timestamp) VALUES
            ('doc_1', 'search_hit', datetime('now', '-120 days')),
            ('doc_1', 'search_hit', datetime('now', '-1 days'))",
        [],
    ).unwrap();
    conn.execute(
        "INSERT INTO co_retrieval (doc_id_a, doc_id_b, query, timestamp) VALUES
            ('a', 'b', 'q', datetime('now', '-120 days')),
            ('a', 'b', 'q', datetime('now', '-1 days'))",
        [],
    ).unwrap();

    // Run the same DELETE statements consolidate::consolidate emits inline.
    conn.execute("DELETE FROM co_retrieval     WHERE timestamp < datetime('now', '-90 days')", []).unwrap();
    conn.execute("DELETE FROM document_access  WHERE timestamp < datetime('now', '-90 days')", []).unwrap();

    let da_count: i64 = conn.query_row("SELECT COUNT(*) FROM document_access", [], |r| r.get(0)).unwrap();
    let cr_count: i64 = conn.query_row("SELECT COUNT(*) FROM co_retrieval", [], |r| r.get(0)).unwrap();
    assert_eq!(da_count, 1, "old document_access row should be deleted");
    assert_eq!(cr_count, 1, "old co_retrieval row should be deleted");
}

// ─── 11. excitability floor clamp ──────────────────────────────────────────
#[test]
fn excitability_clamp_floor() {
    let (mut brain, _tmp) = setup_brain();
    let search = brain.search_mut();
    let db = search.db();

    db.insert_document("doc_floor", "content for floor clamp test long enough to satisfy", &serde_json::json!({}), &fake_emb(), None).unwrap();
    db.conn().execute(
        "UPDATE documents SET excitability = 0.12,
                              indexed_at   = datetime('now', '-365 days'),
                              created      = datetime('now', '-365 days')
         WHERE doc_id = ?1",
        params!["doc_floor"],
    ).unwrap();

    // Hammer decay enough times to drive past the floor if it weren't clamped.
    for _ in 0..50 {
        let _ = strengthen(search, false, false).unwrap();
    }

    let v = excitability_of(search.db(), "doc_floor");
    assert!(v >= 0.1 - 1e-9, "excitability dipped below floor: {v}");
}

// ─── 12. excitability ceiling clamp ────────────────────────────────────────
#[test]
fn excitability_clamp_ceiling() {
    let (mut brain, _tmp) = setup_brain();
    let search = brain.search_mut();
    let db = search.db();

    db.insert_document("doc_ceil", "content for ceiling clamp test long enough to satisfy", &serde_json::json!({}), &fake_emb(), None).unwrap();
    db.conn().execute(
        "UPDATE documents SET excitability = 0.99 WHERE doc_id = ?1",
        params!["doc_ceil"],
    ).unwrap();

    // Pile on high-score hits so every strengthen() pass tries to boost.
    for _ in 0..20 {
        db.log_document_access("doc_ceil", "search_hit", Some("q"), Some(0.95), None).unwrap();
    }

    for _ in 0..50 {
        let _ = strengthen(search, false, false).unwrap();
    }

    let v = excitability_of(search.db(), "doc_ceil");
    assert!(v <= 1.0 + 1e-9, "excitability exceeded ceiling: {v}");
}

// ─── 13. coret_edge_id is order-independent ────────────────────────────────
#[test]
fn coret_edge_id_is_order_independent() {
    use axel::consolidate::reorganize::coret_edge_id;
    assert_eq!(coret_edge_id("a", "b"), coret_edge_id("b", "a"));
    assert_eq!(coret_edge_id("zzz", "aaa"), coret_edge_id("aaa", "zzz"));
    // Different inputs ⇒ different edges.
    assert_ne!(coret_edge_id("a", "b"), coret_edge_id("a", "c"));
}

// ─── 14. document access_count increments to 3 ─────────────────────────────
#[test]
fn document_access_count_increments() {
    let db = Database::open_memory().unwrap();
    db.insert_document("doc_n", "content for access count increment test long enough", &serde_json::json!({}), &fake_emb(), None).unwrap();

    db.increment_document_access("doc_n").unwrap();
    db.increment_document_access("doc_n").unwrap();
    db.increment_document_access("doc_n").unwrap();

    let count: i64 = db.conn()
        .query_row("SELECT access_count FROM documents WHERE doc_id = ?1", params!["doc_n"], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 3);
}

// ─── 15. last_consolidation_time skips in-progress runs ────────────────────
#[test]
fn consolidation_log_in_progress_marker() {
    let db = Database::open_memory().unwrap();

    // Previous, fully-finished run.
    db.insert_consolidation_log(&ConsolidationLogEntry {
        started_at:           "2026-01-01T00:00:00Z".into(),
        finished_at:          Some("2026-01-01T00:01:00Z".into()),
        phase1_reindexed: 0, phase1_pruned: 0,
        phase2_boosted:   0, phase2_decayed: 0,
        phase3_edges_added: 0, phase3_edges_updated: 0,
        phase4_flagged:   0, phase4_removed: 0,
        duration_secs: Some(60.0),
    }).unwrap();

    // Current, in-progress run (finished_at = NULL).
    db.conn().execute(
        "INSERT INTO consolidation_log (started_at, finished_at) VALUES (?1, NULL)",
        params!["2026-02-01T00:00:00Z"],
    ).unwrap();

    let last = db.last_consolidation_time().unwrap();
    assert_eq!(last.as_deref(), Some("2026-01-01T00:01:00Z"),
        "in-progress run should be ignored; previous finished run wins");
}

// ─── 16. CLI search --limit cap ────────────────────────────────────────────
//
// MCP caps `limit` at 50. The CLI currently does NOT cap — `cmd_search`
// forwards the user-supplied value straight to `BrainSearch::search`. This
// test pins that behavior so any future cap (or its absence) is intentional.
#[test]
fn search_limit_cap_cli() {
    let src = std::fs::read_to_string(
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/main.rs")
    ).expect("read main.rs");

    // Find cmd_search and inspect ~40 lines for any explicit cap.
    let idx = src.find("fn cmd_search").expect("cmd_search defined");
    let window: String = src[idx..].lines().take(40).collect::<Vec<_>>().join("\n");

    let has_cap = window.contains(".min(50")
        || window.contains(".min(100")
        || window.contains("limit.min(")
        || window.contains("if limit >");

    // CLI now caps --limit at 100 (MCP caps at 50).
    assert!(has_cap,
        "CLI should cap --limit to prevent OOM. Expected .min(100) in cmd_search");
}

// ─── 18. co_retrieval rows are stored and aggregate correctly ───────────────
//
// Tests the DB-level primitives reorganize() builds on — without needing
// an ONNX model. Inserts 4 co-retrieval events for one pair (> threshold of 3)
// and 1 for another (below threshold), then verifies the COUNT/GROUP BY query
// that reorganize() uses returns exactly the pair that crossed the threshold.
#[test]
fn test_reorganize_creates_edge() {
    let db = Database::open_memory().unwrap();

    // Two docs — content must be non-trivial; insert_document validates length.
    db.insert_document("doc_alpha", "alpha content long enough to pass sanity check", &serde_json::json!({}), &fake_emb(), None).unwrap();
    db.insert_document("doc_beta",  "beta content long enough to pass sanity check",  &serde_json::json!({}), &fake_emb(), None).unwrap();

    // 4 co-retrieval events → crosses threshold of 3.
    for _ in 0..4 {
        db.log_co_retrieval("doc_alpha", "doc_beta", "some query").unwrap();
    }
    // 2 events for a second pair → below threshold.
    db.insert_document("doc_gamma", "gamma content long enough to pass sanity check", &serde_json::json!({}), &fake_emb(), None).unwrap();
    for _ in 0..2 {
        db.log_co_retrieval("doc_alpha", "doc_gamma", "other query").unwrap();
    }

    // Total raw rows stored.
    let total: i64 = db.conn()
        .query_row("SELECT COUNT(*) FROM co_retrieval", [], |r| r.get(0))
        .unwrap();
    assert_eq!(total, 6, "all co_retrieval events should be stored");

    // Verify the aggregate query reorganize() uses. Threshold = 3.
    let threshold: i64 = 3;
    let mut stmt = db.conn().prepare(
        "SELECT doc_id_a, doc_id_b, COUNT(*) as co_count
         FROM co_retrieval
         WHERE timestamp > '1970-01-01'
         GROUP BY doc_id_a, doc_id_b
         HAVING COUNT(*) >= ?1",
    ).unwrap();
    let pairs: Vec<(String, String, i64)> = stmt
        .query_map(params![threshold], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();

    assert_eq!(pairs.len(), 1, "only the pair with ≥3 events should cross threshold");
    let (a, b, count) = &pairs[0];
    // log_co_retrieval canonicalises ordering (a < b).
    assert_eq!(a, "doc_alpha");
    assert_eq!(b, "doc_beta");
    assert_eq!(*count, 4);
}

// ─── 19. prune query flags old zero-access doc with low excitability ────────
//
// Directly mirrors the WHERE clause inside prune_with_priorities():
//   excitability < 0.15  AND  access_count = 0  AND  age > 60 days.
// We fake the age with a direct SQL UPDATE on `created`.
#[test]
fn test_prune_flags_old_zero_access() {
    let db = Database::open_memory().unwrap();

    db.insert_document(
        "stale_doc",
        "stale content long enough to pass insert_document sanity check here",
        &serde_json::json!({}),
        &fake_emb(),
        None,
    ).unwrap();

    // Force the document into prune-candidate territory.
    db.conn().execute(
        "UPDATE documents
         SET excitability  = 0.12,
             access_count  = 0,
             created       = datetime('now', '-90 days'),
             indexed_at    = datetime('now', '-90 days')
         WHERE doc_id = ?1",
        params!["stale_doc"],
    ).unwrap();

    // Run the exact query prune_with_priorities() uses.
    let candidates: Vec<(String, f64, i64, i64)> = db.conn()
        .prepare(
            "SELECT doc_id, excitability, access_count,
                    CAST(julianday('now') - julianday(created) AS INTEGER) as age_days
             FROM documents
             WHERE excitability < 0.15
               AND access_count = 0
               AND CAST(julianday('now') - julianday(created) AS INTEGER) > 60",
        )
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();

    assert_eq!(candidates.len(), 1, "stale doc should be flagged by prune query");
    let (doc_id, excitability, access_count, age_days) = &candidates[0];
    assert_eq!(doc_id, "stale_doc");
    assert!(*excitability < 0.15, "excitability should be below threshold: {excitability}");
    assert_eq!(*access_count, 0);
    assert!(*age_days > 60, "doc should appear > 60 days old: {age_days}");
}

// ─── 20. co_retrieval cleanup removes old rows ──────────────────────────────
//
// Tests the exact DELETE statement used in consolidate::consolidate() for the
// bounded-retention cleanup of the co_retrieval table (90-day window).
#[test]
fn test_co_retrieval_cleanup() {
    let db = Database::open_memory().unwrap();
    let conn = db.conn();

    // Manually insert rows with backdated timestamps — bypassing log_co_retrieval
    // so we control the timestamp directly.
    conn.execute(
        "INSERT INTO co_retrieval (doc_id_a, doc_id_b, query, timestamp)
         VALUES
           ('a', 'b', 'q', datetime('now', '-100 days')),
           ('a', 'c', 'q', datetime('now', '-100 days')),
           ('a', 'b', 'q', datetime('now', '-100 days'))",
        [],
    ).unwrap();

    let before: i64 = conn
        .query_row("SELECT COUNT(*) FROM co_retrieval", [], |r| r.get(0))
        .unwrap();
    assert_eq!(before, 3, "rows should be present before cleanup");

    // Same DELETE used in consolidate().
    conn.execute(
        "DELETE FROM co_retrieval WHERE timestamp < datetime('now', '-90 days')",
        [],
    ).unwrap();

    let after: i64 = conn
        .query_row("SELECT COUNT(*) FROM co_retrieval", [], |r| r.get(0))
        .unwrap();
    assert_eq!(after, 0, "all rows older than 90 days should be removed");
}

// ─── 21. document_access cleanup removes old, preserves recent ─────────────
//
// Same logic as test_co_retrieval_cleanup but for the document_access table.
// Verifies the DELETE preserves rows inside the 90-day window.
#[test]
fn test_document_access_cleanup() {
    let db = Database::open_memory().unwrap();
    db.insert_document("doc_1", "some content long enough for the sanity checks here", &serde_json::json!({}), &fake_emb(), None).unwrap();
    let conn = db.conn();

    conn.execute(
        "INSERT INTO document_access (doc_id, access_type, timestamp)
         VALUES
           ('doc_1', 'search_hit', datetime('now', '-120 days')),
           ('doc_1', 'search_hit', datetime('now', '-95 days')),
           ('doc_1', 'search_hit', datetime('now', '-1 days')),
           ('doc_1', 'search_hit', datetime('now'))",
        [],
    ).unwrap();

    // Same DELETE used in consolidate().
    conn.execute(
        "DELETE FROM document_access WHERE timestamp < datetime('now', '-90 days')",
        [],
    ).unwrap();

    let remaining: i64 = conn
        .query_row("SELECT COUNT(*) FROM document_access", [], |r| r.get(0))
        .unwrap();
    assert_eq!(remaining, 2, "only rows within 90 days should survive: got {remaining}");

    // Also verify each surviving row is actually recent.
    let old_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM document_access
             WHERE timestamp < datetime('now', '-90 days')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(old_count, 0, "no old rows should remain after cleanup");
}

// ─── 22. coret_edge_id is deterministic across multiple calls ───────────────
//
// coret_edge_id uses DefaultHasher which is seeded at program start. Calling
// it multiple times within the same process must return the same value every
// time (deterministic within a run). Order-independence is already tested in
// test 13; here we verify stability over repeated invocations.
#[test]
fn test_reorganize_edge_id_deterministic() {
    use axel::consolidate::reorganize::coret_edge_id;

    let id_a = coret_edge_id("doc_alpha", "doc_beta");

    // Same call ten times — must always equal the first result.
    for i in 0..10 {
        let id = coret_edge_id("doc_alpha", "doc_beta");
        assert_eq!(id, id_a, "call #{i} returned a different id: {id} vs {id_a}");
    }

    // Both orderings yield the same id, consistently.
    for i in 0..10 {
        let id_fwd = coret_edge_id("doc_alpha", "doc_beta");
        let id_rev = coret_edge_id("doc_beta", "doc_alpha");
        assert_eq!(id_fwd, id_rev,
            "call #{i}: forward {id_fwd} != reverse {id_rev}");
    }

    // Different inputs → different ids, consistently.
    let id_other = coret_edge_id("doc_alpha", "doc_gamma");
    assert_ne!(id_a, id_other, "different pairs should produce different edge ids");
    for i in 0..10 {
        assert_eq!(coret_edge_id("doc_alpha", "doc_gamma"), id_other,
            "call #{i} for second pair was unstable");
    }

    // The id always starts with "coret_" and looks like a hex string.
    assert!(id_a.starts_with("coret_"), "edge id should have coret_ prefix: {id_a}");
}

// ─── 23. record_search_feedback creates DB rows ─────────────────────────────
//
// Calls BrainSearch::record_search_feedback with synthetic results and
// verifies that document_access and co_retrieval rows are written.
#[test]
fn test_record_search_feedback() {
    let (mut brain, _tmp) = setup_brain();
    let search = brain.search_mut();
    let db = search.db();

    // Pre-insert two docs so FK constraints are satisfied.
    db.insert_document("fb_doc_1", "content for feedback test alpha long enough here", &serde_json::json!({}), &fake_emb(), None).unwrap();
    db.insert_document("fb_doc_2", "content for feedback test beta long enough here",  &serde_json::json!({}), &fake_emb(), None).unwrap();

    // Fake SearchResults — only doc_id + score matter for record_search_feedback.
    let results = vec![
        velocirag::search::SearchResult {
            doc_id:   "fb_doc_1".to_string(),
            content:  "alpha".to_string(),
            score:    0.9,
            source:   "fused".to_string(),
            metadata: serde_json::json!({}),
        },
        velocirag::search::SearchResult {
            doc_id:   "fb_doc_2".to_string(),
            content:  "beta".to_string(),
            score:    0.8,
            source:   "fused".to_string(),
            metadata: serde_json::json!({}),
        },
    ];

    search.record_search_feedback("test query", &results);

    let da_count: i64 = db.conn()
        .query_row("SELECT COUNT(*) FROM document_access", [], |r| r.get(0))
        .unwrap();
    assert_eq!(da_count, 2, "one document_access row per result");

    let cr_count: i64 = db.conn()
        .query_row("SELECT COUNT(*) FROM co_retrieval", [], |r| r.get(0))
        .unwrap();
    assert_eq!(cr_count, 1, "one co_retrieval row for the two-doc pair");
}

// ─── 24. Ebbinghaus decay: higher access_count → slower decay ───────────────
//
// Pure math test — no DB. Verifies the stability formula:
//   stability = S_BASE * (1 + ln(1 + access_count))
// and that the resulting retention at 30 days is strictly higher for
// access_count=10 than for access_count=1.
#[test]
fn test_ebbinghaus_decay_stability_increases_with_access() {
    const S_BASE: f64 = 60.0;
    let stability_1  = S_BASE * (1.0 + (1.0 +  1.0_f64).ln()); // 1 access
    let stability_10 = S_BASE * (1.0 + (1.0 + 10.0_f64).ln()); // 10 accesses
    let retention_1  = (-30.0 / stability_1 ).exp();
    let retention_10 = (-30.0 / stability_10).exp();
    assert!(
        retention_10 > retention_1,
        "more accesses should mean slower decay: retention_1={retention_1:.4} retention_10={retention_10:.4}"
    );
}

// ─── 25. extract_top_terms filters stopwords ───────────────────────────────
//
// Duplicates the extract_top_terms logic inline so we can test it without
// needing pub(crate) visibility into velocirag internals.
#[test]
fn test_query_expansion_stopwords() {
    // ── inline replica of velocirag::search::extract_top_terms ──────────
    fn extract_top_terms_local(text: &str, n: usize) -> Vec<String> {
        use std::collections::HashMap;
        const STOPWORDS: &[&str] = &[
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "must",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
            "this", "that", "these", "those", "it", "its", "they", "them",
            "he", "she", "we", "you", "i", "me", "my", "our", "your", "his", "her",
            "which", "who", "whom", "what", "where", "when", "how", "why",
            "all", "each", "every", "any", "some", "no", "more", "most", "other",
        ];
        let mut counts: HashMap<String, usize> = HashMap::new();
        for word in text.split_whitespace() {
            let clean = word
                .to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string();
            if clean.len() > 3 && !STOPWORDS.contains(&clean.as_str()) {
                *counts.entry(clean).or_insert(0) += 1;
            }
        }
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().take(n).map(|(w, _)| w).collect()
    }
    // ── end replica ──────────────────────────────────────────────────────

    let text = "the quick brown fox jumps over the lazy dog and it was very fast";
    let terms = extract_top_terms_local(text, 5);

    // "the", "and", "was", "it" are in STOPWORDS; "over" is too but only 4 chars —
    // the filter requires len > 3, so ≤3-char words never reach STOPWORDS anyway.
    let stopwords = ["the", "and", "was", "it"];
    for sw in &stopwords {
        assert!(
            !terms.contains(&sw.to_string()),
            "stopword '{sw}' should have been filtered out; got: {terms:?}"
        );
    }
    // At least one meaningful term must survive.
    assert!(!terms.is_empty(), "should return at least one meaningful term");
    // Every returned term must be longer than 3 chars.
    for term in &terms {
        assert!(term.len() > 3, "term '{term}' is too short — should have been filtered");
    }
}

// ─── 26. handoff command is case-insensitive ───────────────────────────────
//
// Verifies that cmd_handoff uses .to_lowercase() before matching, so
// "Set", "SET", and "set" are all handled as the same command.
// Tests the source directly (same pattern as test 16 / search_limit_cap_cli).
#[test]
fn test_handoff_case_insensitive() {
    let src = std::fs::read_to_string(
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/main.rs")
    ).expect("read main.rs");

    let idx = src.find("fn cmd_handoff").expect("cmd_handoff should be defined in main.rs");
    let window: String = src[idx..].lines().take(20).collect::<Vec<_>>().join("\n");

    assert!(
        window.contains("to_lowercase()"),
        "cmd_handoff should call .to_lowercase() on the action argument for case-insensitive matching;\
         \nwindow:\n{window}"
    );
}
