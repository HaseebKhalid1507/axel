# Axel Consolidation — Testing Gap Analysis

**Reviewed by:** Yoru (subagent)  
**Date:** 2025-07-09  
**Scope:** `crates/axel/src/consolidate/`, `crates/velocirag/src/db.rs` (consolidation methods), `crates/axel/tests/brain_tests.rs`

---

## Executive Summary

The consolidation system — four phases, three DB tables, one audit log — has **zero test coverage**. Not one test for any consolidation function exists. The existing tests cover `AxelBrain::remember/update_memory`, basic DB inserts, and FTS sanitization. The consolidation pipeline, which is the most complex and stateful part of the system, is completely dark.

Coverage breakdown:

| Component | Functions | Tested |
|-----------|-----------|--------|
| `consolidate::consolidate()` (orchestrator) | 1 | ❌ 0 |
| `wants()` (phase gating) | 1 | ❌ 0 |
| `reindex::reindex_source()` | 1 | ❌ 0 |
| `reindex::allocate_new_doc()` | 1 | ❌ 0 |
| `strengthen::strengthen()` | 1 | ❌ 0 |
| `reorganize::reorganize()` | 1 | ❌ 0 |
| `reorganize::coret_edge_id()` | 1 | ❌ 0 |
| `prune::prune()` | 1 | ❌ 0 |
| `prune::prune_with_priorities()` | 1 | ❌ 0 |
| `prune::source_of()` | 1 | ❌ 0 |
| `db::log_document_access()` | 1 | ❌ 0 |
| `db::increment_document_access()` | 1 | ❌ 0 |
| `db::log_co_retrieval()` | 1 | ❌ 0 |
| `db::get_document_accesses_since()` | 1 | ❌ 0 |
| `db::last_consolidation_time()` | 1 | ❌ 0 |
| `db::insert_consolidation_log()` | 1 | ❌ 0 |
| `db::indexed_files_under()` | 1 | ❌ 0 |
| `db::delete_documents_by_file()` | 1 | ❌ 0 |
| `db::invalidate_edge()` | 1 | ❌ 0 |

---

## 1. Missing Unit Tests

### 1.1 `db::log_document_access` + `get_document_accesses_since`

The most critical pair — this is the data pipeline that feeds Phase 2 (strengthen). The timestamp normalization fix lives here.

**Tests needed:**

```
test_log_and_retrieve_document_access
  - Insert one access event
  - Call get_document_accesses_since("1970-01-01T00:00:00Z")
  - Assert: returned vec len == 1, fields match inserted values

test_get_accesses_since_filters_correctly
  - Insert 3 events: one old (mocked via direct SQL), two recent
  - Call get_document_accesses_since with timestamp between old and new
  - Assert: only 2 events returned (the old one is excluded)

test_get_accesses_since_empty_table
  - Call on fresh DB with no events
  - Assert: returns Ok(vec![])

test_get_accesses_access_types
  - Insert events with access_type = "search_hit" and "explicit"
  - Assert: both come back, fields preserved correctly

test_get_accesses_optional_score
  - Insert event with score = None
  - Assert: score field in returned DocumentAccess is None (not 0.0, not panic)
```

### 1.2 Timestamp Normalization Bug — Regression Tests

The fix in `get_document_accesses_since` strips `T`, `Z`, and `+offset` from RFC3339 timestamps so they compare correctly against SQLite's `datetime('now')` format (which produces `"YYYY-MM-DD HH:MM:SS"` with no suffix). Without these tests, the bug can silently re-enter.

```
test_timestamp_normalization_rfc3339_utc_z
  - Insert access event
  - Query with since = "2020-01-01T00:00:00Z"
  - Assert: event is returned (T and Z are stripped, comparison works)

test_timestamp_normalization_rfc3339_with_positive_offset
  - Insert access event
  - Query with since = "2020-01-01T00:00:00+05:30"
  - Assert: event is returned (offset is stripped, not treated as garbage)

test_timestamp_normalization_rfc3339_with_negative_offset
  - Query with since = "2020-01-01T00:00:00-08:00"
  - Assert: event is returned (negative offset doesn't break split logic)
  - NOTE: current impl uses split('+') which WON'T strip negative offsets.
    This is a bug in the normalization — the test will catch it.

test_timestamp_normalization_already_sqlite_format
  - Query with since = "2020-01-01 00:00:00" (no T, no Z)
  - Assert: works correctly (no double-stripping corruption)

test_timestamp_normalization_epoch_string
  - Query with since = "1970-01-01T00:00:00Z" (the default fallback)
  - Assert: returns all events (this is the "first consolidation" path)
```

> ⚠️ **Known gap in fix:** The current normalization uses `split('+')` to strip positive timezone offsets, but this silently fails for negative offsets like `-08:00`. A test will surface this. The fix is to use a proper RFC3339 parser or regex: strip everything after the last `T...` seconds portion.

### 1.3 `db::log_co_retrieval`

```
test_log_co_retrieval_canonical_ordering
  - Call log_co_retrieval("doc_b", "doc_a", "query")
  - Query co_retrieval table directly
  - Assert: doc_id_a = "doc_a", doc_id_b = "doc_b" (a < b always)

test_log_co_retrieval_same_doc_id_noop
  - Call log_co_retrieval("doc_a", "doc_a", "query")
  - Assert: returns Ok, no row inserted (early-return guard)

test_log_co_retrieval_multiple_pairs
  - Log the same pair 5 times
  - Assert: 5 rows in co_retrieval (no dedup — counts are aggregated at query time)
```

### 1.4 `db::last_consolidation_time`

```
test_last_consolidation_time_empty
  - Fresh DB, no log entries
  - Assert: returns Ok(None)

test_last_consolidation_time_single_entry
  - Insert one consolidation_log row with finished_at = "2024-01-15 10:00:00"
  - Assert: returns Ok(Some("2024-01-15 10:00:00"))

test_last_consolidation_time_returns_most_recent
  - Insert two entries: one at 2024-01-01, one at 2024-06-01
  - Assert: returns the 2024-06-01 one (ORDER BY DESC LIMIT 1)

test_last_consolidation_time_null_finished_at_excluded
  - Insert one entry with finished_at = NULL (mid-run crash scenario)
  - Assert: returns Ok(None) (WHERE finished_at IS NOT NULL filters it)
```

### 1.5 `db::insert_consolidation_log`

```
test_insert_consolidation_log_roundtrip
  - Insert a ConsolidationLogEntry with known field values
  - Query the DB directly and assert all 11 columns match
  - Assert duration_secs is stored as REAL (not truncated to integer)

test_insert_consolidation_log_null_finished_at
  - Insert with finished_at = None (would model an in-progress run)
  - Assert: row exists, finished_at IS NULL in DB
```

### 1.6 `db::indexed_files_under`

```
test_indexed_files_under_prefix_match
  - Insert documents with file_paths: "/brain/a.md", "/brain/b.md", "/other/c.md"
  - Call indexed_files_under("/brain/")
  - Assert: returns 2 entries, not the /other/ one

test_indexed_files_under_empty_result
  - Call with prefix that matches nothing
  - Assert: returns Ok(vec![])

test_indexed_files_under_returns_indexed_at_seconds
  - Insert document, check returned f64 is non-zero unix timestamp

test_indexed_files_under_like_special_chars
  - Insert file_path = "/brain/notes_%archive/file.md"
  - Assert: escape_like() prevents wildcard from matching unintended rows
  - (Tests the escape_like helper indirectly through this public surface)
```

### 1.7 `db::delete_documents_by_file`

```
test_delete_documents_by_file_removes_doc_and_fts
  - Insert document with file_path = "x.md"
  - Call delete_documents_by_file("x.md")
  - Assert: document_count() == 0
  - Assert: keyword_search for any term in the document returns empty
    (verifies the FTS cascading DELETE in the function body)

test_delete_documents_by_file_nonexistent_path
  - Call on path not in DB
  - Assert: returns Ok(0) (no panic, no error)

test_delete_documents_by_file_only_targets_matching_path
  - Insert two documents with different file_paths
  - Delete one
  - Assert: the other remains (document_count == 1)
```

### 1.8 `db::invalidate_edge`

```
test_invalidate_edge_sets_valid_to
  - Insert an edge (via insert_edge with valid_to = None)
  - Call invalidate_edge(&edge_id)
  - Assert: returns Ok(true)
  - Assert: querying the edge shows valid_to IS NOT NULL

test_invalidate_edge_nonexistent_id
  - Call invalidate_edge("ghost_id")
  - Assert: returns Ok(false) (rows_affected == 0)
```

### 1.9 `prune::source_of` (private helper)

This is private but testable via `prune_with_priorities`. However, the logic is worth isolating in a unit test inside the module:

```
// Inside prune.rs #[cfg(test)] block:
test_source_of_standard_doc_id
  - source_of("obsidian::notes::rust") == "obsidian"

test_source_of_no_separator
  - source_of("nodoublecolon") == "nodoublecolon"

test_source_of_empty_string
  - source_of("") == ""
```

### 1.10 `reorganize::coret_edge_id` (private helper)

```
// Inside reorganize.rs #[cfg(test)] block:
test_coret_edge_id_order_independent
  - coret_edge_id("doc_a", "doc_b") == coret_edge_id("doc_b", "doc_a")

test_coret_edge_id_different_pairs_different_ids
  - coret_edge_id("doc_a", "doc_b") != coret_edge_id("doc_a", "doc_c")

test_coret_edge_id_format
  - result starts with "coret_"
  - result is ASCII hex after the prefix
```

---

## 2. Missing Integration Tests

These live in `crates/axel/tests/` and require a full `BrainSearch` / `Database` setup (in-memory or tempdir). They validate multi-function flows.

### 2.1 Phase 2 (Strengthen) — Boost Path

```
test_strengthen_boosts_accessed_documents
  Setup:
    - Create in-memory BrainSearch with 2 documents
    - Set both excitabilities to 0.5 (default)
    - Log 3 search_hit accesses for doc_a with score = 0.8
    - Log 0 accesses for doc_b
    - Set last_consolidation_time to "1970-01-01T00:00:00Z" (epoch)
  Run:
    - strengthen::strengthen(&search, false)
  Assert:
    - doc_a excitability > 0.5 (boosted)
    - doc_b excitability <= 0.5 (untouched within grace period, or decayed if old enough)
    - stats.boosted == 1
```

### 2.2 Phase 2 (Strengthen) — Extinction Path

```
test_strengthen_extinction_signal_on_low_score
  Setup:
    - Document with excitability = 0.7
    - Log 5 accesses with score = 0.005 (below SCORE_EXTINCTION_THRESHOLD = 0.02)
  Run:
    - strengthen(&search, false)
  Assert:
    - excitability decreased from 0.7 by EXTINCTION_PENALTY (0.05)
    - stats.extinction_signals == 1
    - stats.boosted == 0
```

### 2.3 Phase 2 (Strengthen) — Decay Path

```
test_strengthen_decays_untouched_old_documents
  Setup:
    - Insert document with indexed_at set to 30+ days ago via direct SQL
    - No access log entries for it
    - Last consolidation time = epoch
  Run:
    - strengthen(&search, false)
  Assert:
    - excitability < 0.5 (decayed below default)
    - stats.decayed >= 1

test_strengthen_does_not_decay_within_grace_period
  Setup:
    - Document indexed today (fresh, default indexed_at = now)
    - No accesses
  Run:
    - strengthen(&search, false)
  Assert:
    - excitability unchanged (still 0.5)
    - stats.decayed == 0
```

### 2.4 Phase 2 (Strengthen) — Dry Run

```
test_strengthen_dry_run_no_db_mutation
  Setup:
    - Document with excitability = 0.5
    - Log accesses that would trigger boost
  Run:
    - strengthen(&search, dry_run: true)
  Assert:
    - stats.boosted > 0 (counted)
    - DB excitability still 0.5 (not written)
```

### 2.5 Phase 3 (Reorganize) — Edge Creation

```
test_reorganize_creates_edge_above_threshold
  Setup:
    - Insert 2 documents: doc_a, doc_b
    - Log CO_RETRIEVAL_THRESHOLD (3) co_retrieval pairs for (doc_a, doc_b)
    - Set last_consolidation_time to epoch so the since window covers all pairs
  Run:
    - reorganize::reorganize(&search, false)
  Assert:
    - stats.edges_added == 1
    - stats.co_retrieval_pairs == 1
    - A "co_retrieved" edge exists between doc_a and doc_b in edges table

test_reorganize_does_not_create_edge_below_threshold
  Setup:
    - Log only 2 co_retrieval pairs (below threshold of 3)
  Assert:
    - stats.edges_added == 0
    - No co_retrieved edge inserted
```

### 2.6 Phase 3 (Reorganize) — Edge Update and Decay

```
test_reorganize_updates_existing_edge_weight
  Setup:
    - Pre-insert a co_retrieved edge with weight = 0.3
    - Log 3 more co_retrieval pairs (above threshold)
  Run:
    - reorganize(&search, false)
  Assert:
    - stats.edges_updated == 1, stats.edges_added == 0
    - Edge weight > 0.3 (bumped), capped at 1.0

test_reorganize_decays_stale_edges
  Setup:
    - Insert co_retrieved edge via DB directly (valid_to = NULL)
    - No recent co_retrieval entries for that pair
  Run:
    - reorganize(&search, false)
  Assert:
    - stats.edges_decayed == 1
    - Edge weight < original (decayed by EDGE_DECAY_FACTOR = 0.8)

test_reorganize_removes_edge_below_removal_threshold
  Setup:
    - Insert co_retrieved edge with weight = 0.1 (below EDGE_REMOVAL_THRESHOLD = 0.2)
    - No recent activity
  Run:
    - reorganize(&search, false)
  Assert:
    - stats.edges_removed == 1
    - Edge valid_to IS NOT NULL (invalidated)
```

### 2.7 Phase 4 (Prune) — Stale Document Handling

```
test_prune_flags_stale_doc_from_non_low_source
  Setup:
    - Insert document: excitability = 0.10, access_count = 0
    - Set created to 61+ days ago via direct SQL
    - source_priorities does NOT contain its source as Low
  Run:
    - prune_with_priorities(&search, &priorities, false)
  Assert:
    - stats.flagged == 1, stats.removed == 0
    - PruneCandidate returned with reason = "stale_low_excitability"

test_prune_auto_removes_stale_doc_from_low_source
  Setup:
    - Same staleness criteria
    - source_priorities maps the doc's source → Priority::Low
  Run:
    - prune_with_priorities(&search, &priorities, false)
  Assert:
    - stats.removed == 1
    - document no longer exists in DB

test_prune_does_not_remove_doc_with_accesses
  Setup:
    - Stale doc (low excitability, 61 days old), but access_count = 5
  Assert:
    - stats.flagged == 0, stats.removed == 0 (WHERE access_count = 0 filter)

test_prune_does_not_remove_doc_above_excitability_threshold
  Setup:
    - Old doc, access_count = 0, but excitability = 0.5
  Assert:
    - Not flagged or removed (WHERE excitability < 0.15 filter)
```

### 2.8 Phase 4 (Prune) — Misaligned Embeddings

```
test_prune_flags_misaligned_embedding
  Setup:
    - Insert document doc_a
    - Log 5 search_hit accesses with score = 0.005 (below 0.015 avg threshold)
  Run:
    - prune_with_priorities(&search, &{}, false)
  Assert:
    - stats.misaligned == 1
    - PruneCandidate with reason = "misaligned_embedding" returned

test_prune_ignores_doc_with_fewer_than_5_hits
  Setup:
    - 4 search_hits with score = 0.005
  Assert:
    - stats.misaligned == 0

test_prune_ignores_doc_with_acceptable_avg_score
  Setup:
    - 5 hits with score = 0.5 avg
  Assert:
    - stats.misaligned == 0
```

### 2.9 Phase 4 (Prune) — Dry Run

```
test_prune_dry_run_does_not_delete
  Setup:
    - Stale doc in a Low-priority source
  Run:
    - prune_with_priorities(&search, &priorities, dry_run: true)
  Assert:
    - stats.removed == 1 (counted)
    - Document still exists in DB
```

### 2.10 Phase 1 (Reindex) — Core File Tracking

```
test_reindex_indexes_new_files
  Setup:
    - TempDir with 2 .md files (content > 50 chars)
    - Fresh BrainSearch (no prior indexed docs)
  Run:
    - reindex_source(&search, &src, false)
  Assert:
    - stats.new_files == 2, stats.reindexed == 2
    - DB document_count() == 2

test_reindex_skips_unchanged_files
  Setup:
    - Index a file once (sets indexed_at)
    - Run reindex_source again without modifying the file
  Assert:
    - stats.reindexed == 0 (mtime <= indexed_at + 0.5)

test_reindex_reindexes_modified_files
  Setup:
    - Index a file, then modify it (touch the mtime via std::fs::File write)
  Assert:
    - stats.reindexed == 1, stats.new_files == 0

test_reindex_prunes_deleted_files
  Setup:
    - Index a file, then delete it from disk
    - Run reindex_source again
  Assert:
    - stats.pruned == 1
    - DB document_count() == 0

test_reindex_skips_short_files
  Setup:
    - Create .md file with content shorter than 50 chars
  Assert:
    - stats.reindexed == 0 (content.len() < 50 guard)

test_reindex_nonexistent_source_dir_returns_error
  - Pass SourceDir with path that doesn't exist
  - Assert: returns Err(AxelError::Other)

test_reindex_dry_run_no_db_writes
  Setup:
    - TempDir with 1 .md file
    - dry_run = true
  Assert:
    - stats.reindexed == 1 (counted)
    - DB document_count() == 0 (nothing written)
```

### 2.11 Phase 1 (Reindex) — Source Priority Ordering

```
test_reindex_processes_high_priority_sources_first
  - This is a property of the orchestrator (mod.rs), not reindex_source directly.
  - Create 3 SourceDirs with Mixed/Low/High priorities
  - Mock or instrument to track insertion order
  - Assert: High-priority source's docs appear first in DB (by rowid / created_at)
  NOTE: May require refactoring to make order observable, or check via
        the audit log's phase1_reindexed values per source.
```

### 2.12 Full Orchestrator — `consolidate::consolidate()`

```
test_consolidate_all_phases_writes_audit_log
  Setup:
    - BrainSearch with some documents and access events
    - ConsolidateOptions with phases = HashSet::new() (run all)
    - dry_run = false
  Run:
    - consolidate(&mut search, &opts)
  Assert:
    - returns Ok(stats)
    - last_consolidation_time() returns Some(_) (log was written)
    - The log row has correct field counts (phase1_reindexed matches stats)

test_consolidate_dry_run_does_not_write_audit_log
  Run with dry_run = true
  Assert:
    - last_consolidation_time() returns None (no log entry)
    - Stats are still returned (non-zero if data exists)

test_consolidate_single_phase_selection
  - opts.phases = {Phase::Strengthen}
  - Assert: only strengthen runs (stats.reindex unchanged at 0, etc.)
  - This tests the `wants()` phase-gate function end-to-end

test_consolidate_stats_duration_is_positive
  - Run any consolidate pass
  - Assert: stats.duration_secs > 0.0
```

---

## 3. Missing Edge Case Tests

### 3.1 Empty Brain (No Documents)

```
test_strengthen_empty_brain
  - strengthen() on DB with no documents, no accesses
  - Assert: Ok(stats), all counts = 0, no panic

test_reorganize_empty_brain
  - reorganize() on DB with no documents, no co_retrieval rows
  - Assert: Ok(stats), all counts = 0

test_prune_empty_brain
  - prune() on DB with no documents
  - Assert: Ok(stats), removed = 0, flagged = 0

test_reindex_empty_source_dir
  - SourceDir pointing to an empty directory (no .md or .txt files)
  - Assert: Ok(stats), checked = 0, reindexed = 0
```

### 3.2 First Consolidation (No Prior Log Entry)

This is the "since = epoch" path. All three phases that depend on `last_consolidation_time()` must handle `None → "1970-01-01T00:00:00Z"` correctly.

```
test_strengthen_first_run_uses_epoch_as_since
  - No consolidation_log entries
  - Log an access event before running
  - Assert: access event IS picked up (epoch since covers all history)

test_reorganize_first_run_uses_epoch_as_since
  - No prior log, co_retrieval pairs exist
  - Assert: pairs are found (since = epoch)

test_consolidate_first_run_creates_first_log_entry
  - Run consolidate on fresh brain
  - Assert: consolidation_log has exactly 1 row
  - Assert: started_at and finished_at are both non-null
```

### 3.3 Repeated Consolidation (Incremental Windowing)

```
test_strengthen_second_run_only_sees_new_accesses
  - Run consolidate pass 1 (writes log entry with finished_at = now)
  - Log new access events after pass 1
  - Run consolidate pass 2
  - Assert: pass 2 stats only reflect the new accesses
  (Tests that last_consolidation_time() correctly gates the since window)
```

### 3.4 Excitability Boundary Conditions

```
test_strengthen_boost_capped_at_ceiling
  - Document with excitability = 0.99
  - Log many high-score accesses
  - Assert: excitability <= 1.0 after boost (ceiling enforced)

test_strengthen_decay_floored_at_minimum
  - Document with excitability = 0.11 (just above EXCITABILITY_FLOOR = 0.1)
  - Force decay conditions
  - Assert: excitability >= 0.1 after decay (floor enforced)

test_strengthen_extinction_floored_at_minimum
  - Document with excitability = 0.10 (at floor)
  - Log low-score accesses (extinction signal)
  - Assert: excitability stays at 0.1, doesn't go below
```

### 3.5 Edge Weight Boundary Conditions

```
test_reorganize_edge_weight_capped_at_1_0
  - Existing co_retrieved edge with weight = 0.95
  - High co-count pair (bump would exceed 1.0)
  - Assert: weight == 1.0 (not > 1.0)
```

### 3.6 Concurrent / Re-entrant Safety (DB-level)

SQLite allows only one writer at a time. These aren't true concurrency tests but guard against logic that assumes exclusive access.

```
test_log_co_retrieval_with_same_doc_both_orderings
  - log_co_retrieval("z", "a", "q") → inserts as (a, z)
  - log_co_retrieval("a", "z", "q") → inserts as (a, z)
  - Assert: 2 rows in co_retrieval, both with doc_id_a = "a"
  (Validates the canonical ordering is consistent across both call orderings)
```

---

## 4. Missing Regression Tests

### 4.1 Timestamp Normalization Bug (Already Fixed, Zero Regression Coverage)

**Context:** `get_document_accesses_since()` receives RFC3339 timestamps (from `Utc::now().to_rfc3339()` stored in `consolidation_log.finished_at`), but SQLite stores `document_access.timestamp` as `datetime('now')` which produces `"YYYY-MM-DD HH:MM:SS"` format. Without normalization, the ISO 8601 `T` separator and `Z` suffix cause the `>=` comparison to silently return 0 rows — meaning strengthen always sees no accesses and never boosts anything.

**Regression tests (MUST NOT be deleted):**

```rust
// In crates/velocirag/src/db.rs #[cfg(test)] mod tests

#[test]
fn regression_timestamp_normalization_rfc3339_z_suffix() {
    // Regression for: get_document_accesses_since ignoring events when
    // called with RFC3339 "Z" suffix timestamps from consolidation_log.
    let db = Database::open_memory().unwrap();
    db.insert_document("doc_1", "content for access log test that is representative", 
        &serde_json::json!({}), &vec![0.1f32; EMBEDDING_DIM], None).unwrap();
    db.log_document_access("doc_1", "search_hit", Some("query"), Some(0.5), None).unwrap();

    // Use an RFC3339 timestamp with Z suffix — the format consolidation_log stores
    let since = "1970-01-01T00:00:00Z";
    let accesses = db.get_document_accesses_since(since).unwrap();
    assert_eq!(accesses.len(), 1,
        "REGRESSION: get_document_accesses_since failed to find events when \
         called with RFC3339 Z-suffix timestamp. Normalization must strip T and Z.");
}

#[test]
fn regression_timestamp_normalization_positive_offset() {
    let db = Database::open_memory().unwrap();
    db.insert_document("doc_1", "content for timezone offset regression test here",
        &serde_json::json!({}), &vec![0.1f32; EMBEDDING_DIM], None).unwrap();
    db.log_document_access("doc_1", "search_hit", None, None, None).unwrap();

    let since = "1970-01-01T00:00:00+00:00";
    let accesses = db.get_document_accesses_since(since).unwrap();
    assert_eq!(accesses.len(), 1,
        "REGRESSION: +00:00 offset not stripped. split('+') must handle this.");
}

#[test]
fn regression_timestamp_normalization_negative_offset() {
    // This test DOCUMENTS A KNOWN REMAINING BUG in the normalization.
    // split('+') does not handle "-08:00" style negative offsets.
    // The since string "-08:00" suffix is NOT stripped, leaving a corrupted 
    // comparison string. Fix: use a proper RFC3339 strip or chrono parse.
    let db = Database::open_memory().unwrap();
    db.insert_document("doc_1", "content for negative timezone offset test case",
        &serde_json::json!({}), &vec![0.1f32; EMBEDDING_DIM], None).unwrap();
    db.log_document_access("doc_1", "search_hit", None, None, None).unwrap();

    let since = "1970-01-01T00:00:00-08:00";
    let accesses = db.get_document_accesses_since(since).unwrap();
    // Currently FAILS — accesses.len() == 0 because "-08:00" is not stripped.
    // When this test passes, the bug is fully fixed.
    assert_eq!(accesses.len(), 1,
        "BUG: negative timezone offsets in RFC3339 timestamps are not normalized. \
         The -HH:MM suffix remains after split('+'), corrupting the SQLite comparison.");
}

#[test]
fn regression_strengthen_sees_accesses_after_first_consolidation() {
    // Regression for the full pipeline version of the timestamp bug:
    // strengthen() feeds last_consolidation_time() (an RFC3339 timestamp) into
    // get_document_accesses_since(). Before normalization, strengthen would
    // always return 0 boosted even with active accesses.
    //
    // This is an integration-level regression — requires BrainSearch or direct
    // DB setup with access events + consolidation_log entry.
    //
    // Tagged INTEGRATION — move to crates/axel/tests/ when BrainSearch test
    // infra is available.
}
```

### 4.2 `prune::source_of` — Regression for Edge Cases

The `source_of` function is a simple string slicer. One future refactor risk: doc_ids with no `::` separator. Capture this forever:

```rust
// In prune.rs #[cfg(test)]
#[test]
fn regression_source_of_no_separator_returns_full_string() {
    // If doc_id format ever changes and loses the :: separator, source_of must
    // not panic or return an empty string — it returns the full doc_id as source.
    assert_eq!(source_of("plaintextid"), "plaintextid");
}
```

### 4.3 `reorganize::coret_edge_id` — Commutativity Regression

```rust
#[test]
fn regression_coret_edge_id_is_commutative() {
    // If the sort logic (a <= b) is ever changed without updating the UPSERT
    // query's bidirectional OR condition, duplicate edges can be created.
    // This is the property that ties together coret_edge_id and the SQL query.
    assert_eq!(
        coret_edge_id("second_doc", "first_doc"),
        coret_edge_id("first_doc", "second_doc"),
        "REGRESSION: coret_edge_id must be order-independent. Changing sort logic \
         without updating the edges UPSERT query creates duplicate co_retrieved edges."
    );
}
```

---

## 5. Test Infrastructure Gaps

Beyond individual test cases, two structural pieces are missing:

### 5.1 No Shared Test Fixtures for Consolidation State

Every consolidation test currently needs to manually set up: DB schema, documents, access events, and co_retrieval rows. A test helper module is needed:

```rust
// crates/axel/tests/consolidation_fixtures.rs  (or in tests/common/mod.rs)

pub fn setup_brain_with_documents(n: usize) -> (BrainSearch, TempDir) { ... }
pub fn seed_access_events(db: &Database, doc_id: &str, count: usize, score: f64) { ... }
pub fn seed_co_retrieval_pairs(db: &Database, a: &str, b: &str, count: i64) { ... }
pub fn set_document_age_days(db: &Database, doc_id: &str, days: i64) { ... }
pub fn set_document_excitability(db: &Database, doc_id: &str, value: f64) { ... }
```

Without these, test setup code will be duplicated across 30+ tests and become a maintenance burden.

### 5.2 No `Database::open_memory()` Equivalent for `BrainSearch`

The `db.rs` tests use `Database::open_memory()` which is clean and fast. The consolidation tests need `BrainSearch` (which wraps DB + embedding model). Check if `BrainSearch` can be constructed with a mock/stub embedder for tests — if not, add that path. Consolidation tests that don't touch embeddings (strengthen, reorganize, prune) should not require a real embedding model to run.

---

## Priority Order

Write tests in this sequence — highest ROI first:

1. **Regression tests** (section 4) — prevent re-introducing fixed bugs
2. **`get_document_accesses_since` unit tests** (1.1, 1.2) — the heart of strengthen
3. **`last_consolidation_time` unit tests** (1.4) — the windowing gate
4. **`strengthen` integration tests** (2.1–2.4) — Phase 2 is the highest-value phase
5. **`prune_with_priorities` integration tests** (2.7–2.9) — deletion is highest-risk
6. **`reorganize` integration tests** (2.5–2.6) — edge lifecycle
7. **`reindex_source` integration tests** (2.10) — filesystem interaction
8. **Empty brain / first consolidation edge cases** (section 3)
9. **Full orchestrator tests** (2.12) — end-to-end smoke
10. **Remaining unit tests** (1.3, 1.5–1.10) — completeness
