# Correctness Review — Axel Consolidation System

**Reviewer:** Shady  
**Date:** 2025-07-10  
**Scope:** `consolidate/{mod,reindex,strengthen,reorganize,prune}.rs` · `velocirag/src/db.rs` (consolidation-related functions) · `axel/src/mcp.rs` (axel_consolidate, logging)

---

## Findings

---

### 1. `strengthen.rs` · Lines 97–98 · **CRITICAL** — Decay query double-filters by the same column, misses docs that have `last_accessed` but are still stale

**File:** `crates/axel/src/consolidate/strengthen.rs`  
**Lines:** 97–98  
**Severity:** Critical

**What's wrong:**  
The decay `WHERE` clause is:
```sql
WHERE (last_accessed IS NULL OR last_accessed < datetime('now', ?1))
  AND (indexed_at    IS NULL OR indexed_at    < datetime('now', ?1))
```
Both conditions must be true simultaneously. That means any document that has `last_accessed = NULL` but a recent `indexed_at` (i.e., freshly indexed but never searched) gets captured. Fine. But any document that *has* been accessed recently but was *indexed* long ago is also excluded from decay. The intent was clearly OR — "untouched by either measure" — but AND means a doc can escape decay just because it was re-indexed recently, even if it hasn't been accessed in months. This is the *exact opposite* of the biological model where recency of *access* is what matters.

**Suggested fix:**  
```sql
WHERE (last_accessed IS NULL OR last_accessed < datetime('now', ?1))
```
Just filter on `last_accessed`. `indexed_at` is not a proxy for retrieval; re-indexing a stale file should not reset its decay clock.

---

### 2. `strengthen.rs` · Lines 101–102 · **HIGH** — `GRACE_DAYS` truncated to integer in format string

**File:** `crates/axel/src/consolidate/strengthen.rs`  
**Lines:** 101–102  
**Severity:** High

**What's wrong:**  
```rust
let grace_clause = format!("-{} days", GRACE_DAYS as i64);
```
`GRACE_DAYS` is `14.0_f64`. Casting to `i64` works fine now, but this is a footgun: if someone bumps `GRACE_DAYS` to `14.5` for any reason, the cast silently truncates to `14` and you get different behavior than you declared. More importantly, the SQL `datetime('now', '-14 days')` already handles fractional days — you're throwing away precision for no reason.

**Suggested fix:**  
```rust
let grace_clause = format!("-{} days", GRACE_DAYS);
```
SQLite accepts `'-14.0 days'`. Or better, make `GRACE_DAYS` an `i64` constant since you never use the fractional part anyway.

---

### 3. `strengthen.rs` · Lines 113–115 · **HIGH** — Redundant `days_inactive < GRACE_DAYS` guard is dead code AND wrong direction

**File:** `crates/axel/src/consolidate/strengthen.rs`  
**Lines:** 113–115  
**Severity:** High

**What's wrong:**  
The SQL `WHERE` clause already filters to docs older than the grace period. Then in Rust you check again:
```rust
if days_inactive < GRACE_DAYS { continue; }
```
This is dead code — SQLite already excluded those rows. BUT: if there's ever a clock skew or the SQL uses `?1` incorrectly, this guard only catches the case where `days_inactive` is less than the threshold, meaning you still process them if the SQL let them through incorrectly. It's not a safety net — it's a redundant no-op that gives false confidence.

The real risk: the SQL uses `?1` twice (for `last_accessed` and `indexed_at`) but only one param is bound. Wait — actually the query uses `params![grace_clause]` with a single `?1` referenced twice. SQLite reuses the same bound value for both `?1` references. That part works. But the duplicate Rust guard is still dead code cluttering the logic.

**Suggested fix:**  
Remove the Rust guard. Trust the SQL. Or add a comment explaining why it's a belt-and-suspenders check — but it isn't, so remove it.

---

### 4. `strengthen.rs` · Lines 70–76 · **HIGH** — Boost formula uses `count + 1` making single-access logarithm non-zero, double-counting

**File:** `crates/axel/src/consolidate/strengthen.rs`  
**Lines:** 74–76  
**Severity:** High

**What's wrong:**  
```rust
let boost = (BOOST_SCALE * ((*count as f64) + 1.0).ln()).min(BOOST_CAP);
```
`count` is the number of access events since last consolidation. Adding `1.0` before `ln()` means even a doc accessed exactly once gets `ln(2.0) ≈ 0.693` units of boost scale, not `ln(1.0) = 0`. This is mathematically intentional *only if* you want "accessed at least once" to give a nonzero boost. Fine. But `count` is already the raw count of `document_access` rows, which also drive `increment_document_access()` — the `access_count` column on `documents` is incremented separately. So a single search hit causes:
1. A row in `document_access`
2. `access_count` bumped on `documents`
3. Strengthen reads `document_access` rows and gets `count=1`, boosting with `ln(2) * 0.05 ≈ 0.035`

That's correct, but `count + 1` is not documented anywhere as intentional. If it IS intentional (treating 0 accesses as base case), then `count=0` should never appear in `grouped` since we only insert to `grouped` from actual access rows. So `count` is always >= 1, making `+1` effectively mean "treat 1 access like 2 accesses." It's an off-by-one-in-log-space bug. Use `(*count as f64).ln().max(0.0)` or just `(*count as f64)` linearly if the log is unnecessary.

---

### 5. `reorganize.rs` · Lines 56–59 · **HIGH** — `last_consolidation_time` returns `finished_at`, not `started_at`; new co-retrievals logged during the run get missed next time

**File:** `crates/axel/src/consolidate/reorganize.rs`  
**Lines:** 56–59  
**Severity:** High

**What's wrong:**  
```rust
let since: String = db
    .last_consolidation_time()?
    .unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string());
```
`last_consolidation_time()` returns `finished_at` of the previous run. Co-retrieval events logged *during* the previous consolidation run (between `started_at` and `finished_at`) will have timestamps between those two values. Because `since = finished_at`, the query `WHERE timestamp > ?1` will exclude those events permanently — they fall in the dead zone between the previous run's start and finish and are never processed.

This is a small window (consolidation is fast) but it's a real gap. Over time co-retrieval pairs from busy consolidation windows evaporate into the void.

**Suggested fix:**  
Store and use `started_at` as the `since` cutoff, not `finished_at`. Add a `last_consolidation_started_at()` helper. Or use `>=` and accept minor double-counting (idempotent for counting purposes).

---

### 6. `reorganize.rs` · Lines 86–89 · **HIGH** — `bump` calculation uses raw count divided by 10, no normalization; first edge from high-frequency pair gets weight > 1.0 clamped silently

**File:** `crates/axel/src/consolidate/reorganize.rs`  
**Lines:** 86–89  
**Severity:** High

**What's wrong:**  
```rust
let bump = (*count as f64) / 10.0;
```
`CO_RETRIEVAL_THRESHOLD` is 3. So the minimum `bump` for a qualifying pair is `0.3`. Maximum is unbounded — if a pair was co-retrieved 50 times since last consolidation, `bump = 5.0`. For a new edge:
```rust
let weight = bump.min(1.0);
```
It clamps to 1.0. Fine. But for an existing edge:
```rust
let new_w = (w + bump).min(1.0);
```
Also clamps. So a hot pair hits max weight immediately and stays there forever (or until decay). The `bump` value carries no meaningful information at high counts because it's always clamped. This makes the system unable to distinguish "co-retrieved 5 times" from "co-retrieved 500 times" for established edges. The weight ceiling kills signal resolution.

**Suggested fix:**  
Consider `bump = (count as f64).ln() / 10.0` to compress the range, or use a smaller divisor and rely on the cap less. Or track `co_count` on the edge itself.

---

### 7. `reorganize.rs` · Lines 42–48 · **MEDIUM** — `DefaultHasher` is non-deterministic across Rust versions; edge IDs will break between upgrades

**File:** `crates/axel/src/consolidate/reorganize.rs`  
**Lines:** 42–48  
**Severity:** Medium

**What's wrong:**  
```rust
use std::collections::hash_map::DefaultHasher;
```
`DefaultHasher` is explicitly documented as *not* stable across Rust versions or even across program invocations with `-Z randomize-layout`. The hash of `"a::b"` could change between Rust 1.X and 1.Y, producing a different `edge_id`. When you upgrade Rust, every existing `coret_*` edge ID becomes an orphan — the upsert logic generates a new ID, inserts a duplicate edge, and the old one decays and dies. You now have ghost edges.

**Suggested fix:**  
Use a stable hasher — `FxHasher` from `rustc-hash`, or better, just `format!("coret_{}_{}", lo, hi)` directly. The pair is already canonicalized by string comparison. A deterministic string ID is cleaner and debuggable.

---

### 8. `prune.rs` · Lines 49–53 · **HIGH** — Public `prune()` always uses empty priority map; auto-removal is permanently disabled from `consolidate()`

**File:** `crates/axel/src/consolidate/prune.rs`  
**Lines:** 49–53  
**Severity:** High

**What's wrong:**  
```rust
pub fn prune(search: &BrainSearch, dry_run: bool) -> Result<PruneStats> {
    let priorities: HashMap<String, Priority> = HashMap::new();
    let (stats, _candidates) = prune_with_priorities(search, &priorities, dry_run)?;
    Ok(stats)
}
```
The top-level `consolidate()` in `mod.rs` calls `prune::prune()`. This function always passes an *empty* priority map. `prune_with_priorities` checks:
```rust
let is_low = matches!(source_priorities.get(&source), Some(Priority::Low));
```
With an empty map, `is_low` is always `false`. Therefore **nothing is ever auto-removed** — every stale document is flagged instead. The auto-removal path (`stats.removed`) will always be zero in production.

The MCP `axel_consolidate` handler builds a proper `sources` vec with priorities, but then calls `consolidate::consolidate()` which ignores those sources for the prune phase and calls the neutered `prune()` instead.

This is a silent correctness failure. Prune looks like it's working (non-zero `flagged`), but auto-removal never fires.

**Suggested fix:**  
`consolidate()` in `mod.rs` needs to pass the source priority map to `prune_with_priorities`. Add it to `ConsolidateOptions` or derive a `HashMap<String, Priority>` from `opts.sources` before calling prune:
```rust
let prio_map: HashMap<String, Priority> = opts.sources.iter()
    .map(|s| (s.name.clone(), s.priority))
    .collect();
let (prune_stats, _) = prune::prune_with_priorities(search, &prio_map, opts.dry_run)?;
stats.prune = prune_stats;
```

---

### 9. `prune.rs` · Lines 111–115 · **MEDIUM** — Dry-run prune counts are inflated: double-counts via both `seen_on_disk` check AND `Path::new().exists()` check

**Wait — that's `reindex.rs` lines 110–119.** Let me be precise:

**File:** `crates/axel/src/consolidate/reindex.rs`  
**Lines:** 110–119  
**Severity:** Medium

**What's wrong:**  
```rust
if !seen_on_disk.contains(file_path) && !std::path::Path::new(file_path).exists() {
```
In dry-run mode, `stats.pruned += 1` fires for any DB file not in `seen_on_disk` AND not on disk. `seen_on_disk` was populated by WalkDir over the source directory. If a file is in the DB but not found by WalkDir (e.g., outside the walked tree, permission denied on a subdir), `seen_on_disk` won't contain it. Then `Path::new(file_path).exists()` is a second existence check. This is a redundant check — if WalkDir missed it, `exists()` might still return true (e.g., symlinks, permission issues). The logic is: "not seen by walk AND not on disk" — but if permissions blocked the walk, you'll prune files that are still there.

More critically: WalkDir is scoped to `target_abs`, but `indexed_files_under` returns files under the *prefix* which is the canonicalized path. These should match, but `unwrap_or_else(|_| source.path.clone())` in the fallback means if `canonicalize()` fails on either the source dir or a file path, you get mismatched path formats — one may be relative, one absolute — and `seen_on_disk` never matches anything in the DB, causing every indexed file to look like it's been deleted.

**Suggested fix:**  
Remove the redundant `!Path::new(file_path).exists()` check — `seen_on_disk` is already the ground truth for what WalkDir found. Add a log warning if `canonicalize()` fails instead of silently falling back to a potentially wrong path.

---

### 10. `db.rs` · Lines 949–956 · **MEDIUM** — Timestamp normalization strips negative UTC offsets incorrectly

**File:** `crates/velocirag/src/db.rs`  
**Lines:** 950–956  
**Severity:** Medium

**What's wrong:**  
```rust
let normalized = since
    .replace('T', " ")
    .split('+').next().unwrap_or(since)
    .split('Z').next().unwrap_or(since)
    .to_string();
```
This splits on `'+'` to strip timezone offsets. But negative UTC offsets look like `"2025-07-10T12:00:00-07:00"` — splitting on `'+'` does nothing, and the `-07:00` suffix remains. The normalized string becomes `"2025-07-10 12:00:00-07:00"` which SQLite will not compare correctly to its `datetime('now')` format.

Additionally, `unwrap_or(since)` references the pre-`replace` `since` value (before the `'T'→' '` replacement) — so the fallback isn't even using the partially-normalized version. This is a `.split().next()` chain on a temporary; the fallback is the original string, not the partially-processed one.

**Suggested fix:**  
Use a proper RFC3339 parser, or at minimum handle the negative offset:
```rust
let normalized = since
    .replace('T', " ")
    .split(['+', 'Z']).next()
    // strip any trailing negative offset:
    .map(|s| if let Some(i) = s.rfind('-').filter(|&i| i > 10) { &s[..i] } else { s })
    .unwrap_or(since)
    .to_string();
```
Or just store and compare timestamps in the same format everywhere (RFC3339 with `Z`) and use `CAST` in SQLite.

---

### 11. `prune.rs` · Lines 125–142 · **MEDIUM** — Misaligned-embedding query counts `search_hit` rows in `document_access`, not unique queries; hot documents penalized for being popular

**File:** `crates/axel/src/consolidate/prune.rs`  
**Lines:** 125–142  
**Severity:** Medium

**What's wrong:**  
```sql
SELECT da.doc_id, COUNT(*) as hits, AVG(da.score) as avg_score
FROM document_access da
WHERE da.access_type = 'search_hit'
GROUP BY da.doc_id
HAVING hits >= 5 AND avg_score < 0.015
```
A document is flagged as "misaligned" if it was returned in at least 5 searches with an average score below 0.015. But scores in the embedding system are cosine similarities that can range widely. A document that's a weak match for *many different queries* might have a low average score purely because it's retrieved for a broad range of topics (low individual relevance, high breadth). This isn't necessarily a bad embedding — it might just be a cross-domain document.

More importantly: if the `score` column in `document_access` is NULL for some access types (it's `Option<f64>`), `AVG()` in SQLite ignores NULLs. So `avg_score` is only over non-NULL scores. A document with 10 logged accesses but only 1 having a score (0.01) gets flagged — `hits=10 >= 5`, `avg_score=0.01 < 0.015` — even though 9 of those accesses had no score context at all.

**Suggested fix:**  
Add `AND da.score IS NOT NULL` to the WHERE clause and `HAVING COUNT(da.score) >= 5` instead of `COUNT(*) >= 5`. You want 5 actual scored hits, not 5 rows of which some have NULL scores.

---

### 12. `prune.rs` · Lines 94–122 / `reindex.rs` · Lines 100–106 · **MEDIUM** — Dry-run stat inflation: stats count operations that don't happen

**File:** `crates/axel/src/consolidate/prune.rs` lines 111, 120; `reindex.rs` line 105  
**Severity:** Medium

**What's wrong:**  
In `prune_with_priorities`, when `is_low && dry_run`:
```rust
if !dry_run {
    // ... delete
}
stats.removed += 1;  // counted regardless of dry_run
```
Similarly in `reindex_source`:
```rust
if !dry_run {
    search.index_document(...)?;
    if is_new { newly_indexed.push(...); }
}
stats.reindexed += 1;    // always
if is_new { stats.new_files += 1; }  // always
```
This is the intended behavior — dry-run shows what *would* happen. That's fine and arguably correct. **But** the `ConsolidationLogEntry` is only written when `!dry_run` (checked in `mod.rs` line 115), so the log is never inflated. The stats themselves are used only for display. This is consistent and probably correct behavior.

Actually I'm taking this one back — this is intentional and correct. Dry-run stats should reflect *what would happen*. Not a bug. Removing from severity. ~~Medium~~ → Non-issue.

---

### 13. `reindex.rs` · Lines 93–99 · **MEDIUM** — `doc_id` construction strips `.md` and `.txt` anywhere in path, not just the extension

**File:** `crates/axel/src/consolidate/reindex.rs`  
**Lines:** 93–99  
**Severity:** Medium

**What's wrong:**  
```rust
let relative_id = file_path
    .strip_prefix(&target_abs).unwrap_or(file_path)
    .to_string_lossy()
    .replace('/', "::")
    .replace(".md", "")
    .replace(".txt", "");
```
`.replace(".md", "")` is a string replace, not an extension strip. It removes `.md` from *anywhere* in the path. A file at `notes/readme.md.backup/file.txt` becomes `notes::readme.backup::file` — the `.md` inside a directory name gets eaten. A file named `update.md_draft.txt` becomes `update_draft` — the `.md` in the middle of the stem is consumed.

More practically: a directory named `cmd.md_notes/` (unlikely but valid) would corrupt all doc IDs under it.

**Suggested fix:**  
Strip only the final extension:
```rust
let stem = file_path.file_stem()
    .unwrap_or(file_path.as_os_str())
    .to_string_lossy();
let relative_dir = file_path.parent()
    .and_then(|p| p.strip_prefix(&target_abs).ok())
    ...
```
Or use `Path::with_extension("")` before converting to string.

---

### 14. `reorganize.rs` · Lines 141–157 · **LOW** — Full `live_edges` table scan on every consolidation run with no time-based filter

**File:** `crates/axel/src/consolidate/reorganize.rs`  
**Lines:** 141–157  
**Severity:** Low

**What's wrong:**  
```sql
SELECT id, source_id, target_id, weight FROM edges
WHERE type = 'co_retrieved'
  AND (valid_to IS NULL OR valid_to > datetime('now'))
```
This loads **every live `co_retrieved` edge** into memory on every consolidation run, then for each one fires a separate `SELECT COUNT(*)` against `co_retrieval` to check for recent activity. That's N+1 queries where N is the total number of co-retrieved edges ever created. As the graph grows this becomes expensive and slow.

**Suggested fix:**  
Do the decay in SQL in one shot:
```sql
UPDATE edges SET weight = weight * 0.8
WHERE type = 'co_retrieved'
  AND (valid_to IS NULL OR valid_to > datetime('now'))
  AND id NOT IN (
      SELECT DISTINCT e.id FROM edges e
      JOIN co_retrieval cr ON (
          (e.source_id = cr.doc_id_a AND e.target_id = cr.doc_id_b)
          OR (e.source_id = cr.doc_id_b AND e.target_id = cr.doc_id_a)
      )
      WHERE cr.timestamp > datetime('now', '-30 days')
  )
```
This is one query, no N+1, uses indexes.

---

### 15. `db.rs` · Lines 511–522 · **LOW** — `delete_documents_by_file` doesn't clean `document_access` or `co_retrieval` for deleted doc IDs; orphan access logs accumulate

**File:** `crates/velocirag/src/db.rs`  
**Lines:** 511–522  
**Severity:** Low

**What's wrong:**  
When a document is deleted (during prune or reindex), the `document_access` and `co_retrieval` tables retain rows for that `doc_id`. Future `strengthen` runs will try to look up excitability for those orphaned doc IDs:
```rust
let current: f64 = match conn.query_row(
    "SELECT excitability FROM documents WHERE doc_id = ?1", ...
) {
    Ok(v) => v,
    Err(_) => continue,  // silently skipped
};
```
The `Err(_) => continue` handles it gracefully. But the orphan rows still accumulate in `document_access` and `co_retrieval` indefinitely. A pruned doc that was frequently accessed will keep contributing to co-retrieval pair counts in `reorganize` — you'll keep strengthening and maintaining edges for documents that no longer exist.

**Suggested fix:**  
Add to `delete_documents_by_file`:
```sql
DELETE FROM document_access WHERE doc_id = ?1;
DELETE FROM co_retrieval WHERE doc_id_a = ?1 OR doc_id_b = ?1;
```
Or add a `ON DELETE CASCADE`-equivalent trigger. At minimum, `reorganize`'s pair query should JOIN back to `documents` to filter out orphan doc IDs.

---

### 16. `mcp.rs` · Lines 200–207 · **LOW** — Co-retrieval logged for ALL top-5 pairs per search, regardless of result count; panics on index if fewer than 5 results

Actually — `top_ids.len()` is capped by `take(5)` and the loops use `top_ids.len()`, not hardcoded 5. So no panic. But: co-retrieval is logged for every search regardless of whether the results are actually relevant (even low-score junk results create co-retrieval pairs). A garbage query that returns 5 unrelated low-score results will create 10 co-retrieval pairs that eventually generate graph edges. There's no score threshold on what qualifies as a meaningful co-retrieval signal.

**File:** `crates/axel/src/mcp.rs`  
**Lines:** 200–207  
**Severity:** Low

**Suggested fix:**  
Filter `top_ids` to only include results above a minimum score threshold before logging co-retrieval pairs. Something like `results.results.iter().filter(|r| r.score > 0.1).take(5)`.

---

## Summary Table

| # | File | Lines | Severity | Issue |
|---|------|--------|----------|-------|
| 1 | `strengthen.rs` | 97–98 | 🔴 Critical | Decay `WHERE` uses `AND` instead of `OR` — re-indexed docs escape decay |
| 2 | `strengthen.rs` | 101–102 | 🟠 High | `GRACE_DAYS as i64` truncates float constant in format string |
| 3 | `strengthen.rs` | 113–115 | 🟠 High | Redundant Rust guard is dead code after SQL already filters |
| 4 | `strengthen.rs` | 74–76 | 🟠 High | `count + 1` in log formula means 1 access treated as 2 |
| 5 | `reorganize.rs` | 56–59 | 🟠 High | `since` uses `finished_at`; events during run permanently lost |
| 6 | `reorganize.rs` | 86–89 | 🟠 High | `bump = count/10` loses resolution at max-weight ceiling |
| 7 | `reorganize.rs` | 42–48 | 🟡 Medium | `DefaultHasher` non-deterministic; edge IDs break across Rust upgrades |
| 8 | `prune.rs` | 49–53 | 🟠 High | `prune()` always uses empty priority map; auto-removal permanently disabled |
| 9 | `reindex.rs` | 110–119 | 🟡 Medium | Path canonicalize failure → mismatch → all files look deleted |
| 10 | `db.rs` | 950–956 | 🟡 Medium | Timestamp normalization misses negative UTC offsets |
| 11 | `prune.rs` | 125–142 | 🟡 Medium | Misaligned query counts NULL-score rows in `hits >= 5` |
| 12 | `reindex.rs` | 93–99 | 🟡 Medium | `.replace(".md", "")` strips extension from anywhere in path |
| 13 | `reorganize.rs` | 141–157 | 🔵 Low | N+1 query pattern for edge decay loop |
| 14 | `db.rs` | 511–522 | 🔵 Low | Deleted docs leave orphan rows in `document_access`/`co_retrieval` |
| 15 | `mcp.rs` | 200–207 | 🔵 Low | Co-retrieval logged for low-score junk results |

---

## Overall Verdict

**4/10.** The architecture is solid and the biological analogy is actually clever. The error handling posture (degrade gracefully on missing tables, soft errors on allocation) is appropriate. Parameterized queries throughout — no SQL injection. Good.

But there are three issues that genuinely break the system's core purpose:

**Issue #8 is the worst.** Auto-removal is silently disabled in production because `prune()` passes an empty priority map. Your prune phase has never actually pruned anything automatically. It just flags everything and calls it a day. That's not a prune, that's a to-do list generator.

**Issue #1 is a close second.** The decay `AND` vs `OR` condition means freshly re-indexed documents that haven't been accessed in months escape decay. You're protecting stale content from the very mechanism designed to clean it up.

**Issue #5 means your co-retrieval window has a gap on every run.** Small, but it compounds.

Fix #8 first. Fix #1 second. Then handle the `DefaultHasher` time bomb before you upgrade Rust and wonder why all your co-retrieval edges doubled.
