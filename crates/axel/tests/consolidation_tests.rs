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
