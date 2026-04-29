//! Integration tests for `decay.rs` — importance decay and access-driven boosting.

use axel_memkoshi::decay::{decay_and_boost, DecayStats};
use axel_memkoshi::memory::{Confidence, Memory, MemoryCategory};
use axel_memkoshi::storage::MemoryStorage;
use tempfile::TempDir;

// ------------------------------------------------------------------ helpers

fn tmp_storage() -> (TempDir, MemoryStorage) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("decay_test.db");
    let storage = MemoryStorage::open(&path).unwrap();
    (dir, storage)
}

fn make_memory(importance: f64) -> Memory {
    let mut m = Memory::new(
        MemoryCategory::Events,
        "test-topic",
        "Test Memory Title For Decay",
        "This is the body content of the test memory used in decay tests.",
    );
    m.importance = importance;
    m.confidence = Confidence::Medium;
    m.abstract_text = "Short abstract for decay test memory.".to_string();
    m
}

/// Backdate the `created` column of a memory in the DB by `days_ago` days.
fn backdate_memory(storage: &MemoryStorage, id: &str, days_ago: i64) {
    let conn = storage.conn();
    let ts = (chrono::Utc::now() - chrono::Duration::days(days_ago))
        .to_rfc3339();
    conn.execute(
        "UPDATE memories SET created = ?1 WHERE id = ?2",
        rusqlite::params![ts, id],
    )
    .unwrap();
}

/// Insert a `memory_access` row with a timestamp `days_ago` days in the past.
fn insert_access_at(storage: &MemoryStorage, memory_id: &str, days_ago: i64) {
    let conn = storage.conn();
    let ts = (chrono::Utc::now() - chrono::Duration::days(days_ago))
        .to_rfc3339();
    conn.execute(
        "INSERT INTO memory_access (memory_id, access_type, timestamp) VALUES (?1, 'read', ?2)",
        rusqlite::params![memory_id, ts],
    )
    .unwrap();
}

/// Read the current importance of a memory directly from the DB.
fn get_importance(storage: &MemoryStorage, id: &str) -> f64 {
    storage
        .conn()
        .query_row(
            "SELECT importance FROM memories WHERE id = ?1",
            rusqlite::params![id],
            |r| r.get(0),
        )
        .unwrap()
}

// ------------------------------------------------------------------ tests

#[test]
fn boost_increases_importance_for_recently_accessed() {
    let (_dir, storage) = tmp_storage();
    let mut mem = make_memory(0.5);
    storage.store_memory(&mem).unwrap();

    // Record an access that happened 2 days ago (well within the 7-day boost window).
    insert_access_at(&storage, &mem.id, 2);

    let stats = decay_and_boost(&storage).unwrap();

    let new_importance = get_importance(&storage, &mem.id);
    assert!(
        new_importance > 0.5,
        "importance should have been boosted above 0.5, got {new_importance}"
    );
    assert_eq!(stats.boosted, 1, "expected exactly 1 boosted memory");
    assert_eq!(stats.decayed, 0);
}

#[test]
fn decay_decreases_importance_for_old_unaccessed() {
    let (_dir, storage) = tmp_storage();
    let mem = make_memory(0.8);
    storage.store_memory(&mem).unwrap();

    // Create an access record that is 30 days old — beyond the 14-day decay threshold.
    // The access record triggers the query; its timestamp puts it in the decay branch.
    insert_access_at(&storage, &mem.id, 30);

    let stats = decay_and_boost(&storage).unwrap();

    let new_importance = get_importance(&storage, &mem.id);
    assert!(
        new_importance < 0.8,
        "importance should have decayed below 0.8, got {new_importance}"
    );
    assert_eq!(stats.decayed, 1, "expected exactly 1 decayed memory");
    assert_eq!(stats.boosted, 0);
}

#[test]
fn decay_floors_at_minimum() {
    let (_dir, storage) = tmp_storage();
    // Start near 0 so any penalty would push below the floor.
    let mem = make_memory(0.11);
    storage.store_memory(&mem).unwrap();

    // 200 days ago → weeks ≈ 28.6, raw penalty ≈ 0.572, capped at 0.3.
    // 0.11 − 0.3 = −0.19 → should floor at 0.1.
    insert_access_at(&storage, &mem.id, 200);

    decay_and_boost(&storage).unwrap();

    let new_importance = get_importance(&storage, &mem.id);
    assert!(
        new_importance >= 0.1 - f64::EPSILON,
        "importance must not drop below the 0.1 floor, got {new_importance}"
    );
    assert!(
        new_importance < 0.11,
        "importance should still have dropped from 0.11, got {new_importance}"
    );
}

#[test]
fn no_change_for_recent_unaccessed() {
    let (_dir, storage) = tmp_storage();
    let mem = make_memory(0.5);
    storage.store_memory(&mem).unwrap();

    // No access records at all → the boost/decay query over memory_access returns
    // nothing, so this memory is never touched.
    let stats = decay_and_boost(&storage).unwrap();

    let new_importance = get_importance(&storage, &mem.id);
    assert!(
        (new_importance - 0.5).abs() < f64::EPSILON,
        "importance should be unchanged at 0.5, got {new_importance}"
    );
    assert_eq!(stats.boosted, 0);
    assert_eq!(stats.decayed, 0);
}

#[test]
fn boost_caps_at_max() {
    let (_dir, storage) = tmp_storage();
    let mem = make_memory(0.5);
    storage.store_memory(&mem).unwrap();

    // Insert 1 000 access records within the last day.
    // bump = min(0.2, 0.05 * log2(1000 + 1)) ≈ min(0.2, 0.499) = 0.2
    // So importance should be exactly 0.5 + 0.2 = 0.7 (clamped to 1.0 at most).
    for _ in 0..1_000 {
        insert_access_at(&storage, &mem.id, 0);
    }

    decay_and_boost(&storage).unwrap();

    let new_importance = get_importance(&storage, &mem.id);
    // The bump formula is min(0.2, …) so it can never add more than 0.2.
    assert!(
        new_importance <= 0.7 + 1e-9,
        "boost should be capped; importance must not exceed 0.7 (0.5 + 0.2), got {new_importance}"
    );
    assert!(
        new_importance > 0.5,
        "importance should have risen from 0.5, got {new_importance}"
    );
}

#[test]
fn returns_correct_stats() {
    let (_dir, storage) = tmp_storage();

    // mem_a — boosted (accessed 1 day ago)
    let mem_a = make_memory(0.4);
    storage.store_memory(&mem_a).unwrap();
    insert_access_at(&storage, &mem_a.id, 1);

    // mem_b — decayed (accessed 60 days ago)
    let mem_b = make_memory(0.9);
    storage.store_memory(&mem_b).unwrap();
    insert_access_at(&storage, &mem_b.id, 60);

    // mem_c — no access at all → ignored
    let mem_c = make_memory(0.5);
    storage.store_memory(&mem_c).unwrap();

    let stats: DecayStats = decay_and_boost(&storage).unwrap();

    assert_eq!(stats.boosted, 1, "expected 1 boosted, got {}", stats.boosted);
    assert_eq!(stats.decayed, 1, "expected 1 decayed, got {}", stats.decayed);
}
