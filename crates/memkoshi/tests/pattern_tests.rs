//! Integration tests for `patterns.rs` — pattern detection over the event log.

use axel_memkoshi::memory::{Memory, MemoryCategory};
use axel_memkoshi::patterns::{PatternDetector, PatternType};
use axel_memkoshi::storage::MemoryStorage;
use tempfile::TempDir;

// ------------------------------------------------------------------ helpers

fn tmp_storage() -> (TempDir, MemoryStorage) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("pattern_test.db");
    let storage = MemoryStorage::open(&path).unwrap();
    (dir, storage)
}

/// Insert a raw event row with a fully-controlled timestamp (RFC-3339).
/// Use this when you need deterministic weekday values for temporal tests.
fn insert_event_at(
    storage: &MemoryStorage,
    event_type: &str,
    target_id: Option<&str>,
    query: Option<&str>,
    metadata: Option<&str>,
    timestamp: &str,
) {
    storage
        .conn()
        .execute(
            "INSERT INTO events (event_type, target_id, query, metadata, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![event_type, target_id, query, metadata, timestamp],
        )
        .unwrap();
}

// ------------------------------------------------------------------ tests

#[test]
fn frequency_detects_repeated_searches() {
    let (_dir, storage) = tmp_storage();

    // 5 search events targeting the same memory id — above the threshold of 3.
    for _ in 0..5 {
        storage
            .record_event("search", Some("mem_target01"), None, None)
            .unwrap();
    }

    let detector = PatternDetector::new(&storage);
    let patterns = detector.detect().unwrap();

    let freq: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::Frequency)
        .collect();

    assert_eq!(freq.len(), 1, "expected exactly 1 frequency pattern");
    assert_eq!(freq[0].name, "mem_target01");
    assert_eq!(freq[0].sample_size, 5);
    assert!(
        freq[0].confidence > 0.0 && freq[0].confidence <= 1.0,
        "confidence out of range: {}",
        freq[0].confidence
    );
}

#[test]
fn frequency_ignores_low_count() {
    let (_dir, storage) = tmp_storage();

    // Only 2 search events — below the HAVING n >= 3 threshold.
    for _ in 0..2 {
        storage
            .record_event("search", Some("mem_rare"), None, None)
            .unwrap();
    }

    let detector = PatternDetector::new(&storage);
    let patterns = detector.detect().unwrap();

    let freq: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::Frequency)
        .collect();

    assert!(
        freq.is_empty(),
        "should not detect frequency pattern for only 2 events, got: {freq:?}"
    );
}

#[test]
fn gap_detects_search_with_no_results() {
    let (_dir, storage) = tmp_storage();

    // 4 `search_complete` events for the same query with results_count = 0.
    for _ in 0..4 {
        storage
            .record_event(
                "search_complete",
                None,
                Some("quantum entanglement"),
                Some(serde_json::json!({"results_count": 0})),
            )
            .unwrap();
    }

    // One event with results — should not affect the gap count.
    storage
        .record_event(
            "search_complete",
            None,
            Some("quantum entanglement"),
            Some(serde_json::json!({"results_count": 3})),
        )
        .unwrap();

    let detector = PatternDetector::new(&storage);
    let patterns = detector.detect().unwrap();

    let gaps: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::Gap)
        .collect();

    assert_eq!(gaps.len(), 1, "expected exactly 1 gap pattern");
    assert_eq!(gaps[0].name, "quantum entanglement");
    assert_eq!(gaps[0].sample_size, 4);
    assert!(gaps[0].description.contains("knowledge gap"));
}

#[test]
fn temporal_detects_day_of_week_pattern() {
    let (_dir, storage) = tmp_storage();

    // Pin all events to the same weekday (Monday = '1' in SQLite strftime('%w',...)).
    // We use 2024-01-08 (a Monday), 2024-01-15, 2024-01-22 — three Mondays.
    let monday_timestamps = [
        "2024-01-08T10:00:00+00:00",
        "2024-01-15T10:00:00+00:00",
        "2024-01-22T10:00:00+00:00",
    ];
    for ts in &monday_timestamps {
        insert_event_at(&storage, "search", None, Some("rust async"), None, ts);
    }

    let detector = PatternDetector::new(&storage);
    let patterns = detector.detect().unwrap();

    let temporal: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::Temporal)
        .collect();

    assert!(
        !temporal.is_empty(),
        "expected at least one temporal pattern"
    );

    let target = temporal
        .iter()
        .find(|p| p.name.contains("rust async"))
        .expect("no temporal pattern for 'rust async'");

    assert_eq!(target.sample_size, 3);
    assert!(
        target.name.starts_with("Monday:"),
        "expected pattern name to start with 'Monday:', got '{}'",
        target.name
    );
    assert!(target.description.contains("Monday"));
}

#[test]
fn empty_events_returns_empty() {
    let (_dir, storage) = tmp_storage();

    let detector = PatternDetector::new(&storage);
    let patterns = detector.detect().unwrap();

    assert!(
        patterns.is_empty(),
        "fresh storage should yield no patterns, got {patterns:?}"
    );
}

#[test]
fn confidence_scales_with_count() {
    let (_dir, storage) = tmp_storage();

    // 3 hits for target_a (minimum threshold).
    for _ in 0..3 {
        storage
            .record_event("search", Some("target_a"), None, None)
            .unwrap();
    }

    // 10 hits for target_b (much higher).
    for _ in 0..10 {
        storage
            .record_event("search", Some("target_b"), None, None)
            .unwrap();
    }

    let detector = PatternDetector::new(&storage);
    let patterns = detector.detect().unwrap();

    let freq: Vec<_> = patterns
        .iter()
        .filter(|p| p.pattern_type == PatternType::Frequency)
        .collect();

    // We expect patterns for both targets.
    assert_eq!(freq.len(), 2, "expected 2 frequency patterns, got {freq:?}");

    let conf_a = freq
        .iter()
        .find(|p| p.name == "target_a")
        .expect("no pattern for target_a")
        .confidence;
    let conf_b = freq
        .iter()
        .find(|p| p.name == "target_b")
        .expect("no pattern for target_b")
        .confidence;

    assert!(
        conf_b > conf_a,
        "higher count (target_b={conf_b}) should yield higher confidence than lower count (target_a={conf_a})"
    );

    // Verify the formula: n / (n + 3.0)
    let expected_a = 3.0_f64 / (3.0 + 3.0);
    let expected_b = 10.0_f64 / (10.0 + 3.0);
    assert!(
        (conf_a - expected_a).abs() < 1e-9,
        "conf_a: expected {expected_a}, got {conf_a}"
    );
    assert!(
        (conf_b - expected_b).abs() < 1e-9,
        "conf_b: expected {expected_b}, got {conf_b}"
    );
}
