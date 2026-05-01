# Axel — Error Handling & Robustness Review

**Scope:** `crates/axel/src/consolidate/*.rs` + `crates/velocirag/src/search.rs`
**Reviewer:** Gojo
**Verdict:** Solid foundation, but there are meaningful gaps. Nothing catastrophic. Several things that will silently eat your data or lie to you in production.

---

## Executive Summary

The good news: the code has a real error model (`Result<T>`), uses `?` propagation properly in most places, and the consolidation pipeline has a `partial_failure` flag that lets one phase fail without killing the others. The codebase isn't the kind of cowboy code that panics everywhere and hopes for the best.

The bad news: **silent failure is endemic in `search.rs`**. The search engine swallows DB errors with `.ok()` throughout its hot paths — vector layer, excitability boost, MMR diversity — in ways that produce subtly wrong results without any signal to the caller. The consolidation code is cleaner but has its own issues with `unwrap_or` hiding real errors and a DB-locked scenario that could produce ghost state.

Below are the specific findings, from most to least severe.

---

## 1. Silent Error Swallowing

### 1a. Vector Layer — DB Hit Silently Dropped (`search.rs:148–163`)

```rust
let ranked: Vec<RankedResult> = vector_results
    .into_iter()
    .filter_map(|vr| {
        let doc = self.db.conn().query_row(
            "SELECT doc_id, content, metadata FROM documents WHERE id = ?1",
            [vr.id as i64],
            |row| { ... },
        ).ok()?;   // ← swallowed
        Some(RankedResult { ... })
    })
    .collect();
```

**What happens:** If `query_row` fails (DB locked, schema mismatch, corrupt row), `ok()` turns the error into `None`, and `filter_map` silently drops that vector hit. The vector layer returns fewer results than it should, silently. The caller gets `stats.vector_candidates` that undercounts. No log, no warning, no error propagation.

**Risk:** DB contention during consolidation (consolidation writes while search reads) is a realistic scenario here. You'd get degraded search quality with no indication why.

**Fix:**
```rust
let ranked: Vec<RankedResult> = vector_results
    .into_iter()
    .filter_map(|vr| {
        match self.db.conn().query_row(...) {
            Ok(doc) => Some(RankedResult { ... }),
            Err(e) => {
                tracing::warn!("vector layer: failed to load doc for id {}: {e}", vr.id);
                None
            }
        }
    })
    .collect();
```

---

### 1b. Excitability Boost — Silent Degradation (`search.rs:~290–340`)

```rust
let mut stmt = self.db.conn().prepare(&sql).ok();  // ← swallowed
// ...
if let Some(ref mut s) = stmt {
    if let Ok(rows) = s.query_map(...) {       // ← swallowed
        for row in rows.flatten() {            // ← per-row errors swallowed
```

Three nested silent failures. If `prepare` fails (e.g. DB locked), `stmt` is `None` and the entire excitability boost block is silently skipped. Every result gets the fallback excitability of `0.5` and a uniform boost of `0.9 + (0.5 * 0.2) = 1.0`. Results look plausible — wrong, but plausible. No signal to the caller.

The `rows.flatten()` at the innermost level swallows individual row errors too.

**Risk:** Any DB issue during this phase produces quietly incorrect ranking. This is exactly the kind of bug that's invisible in testing but degrades results in production.

**Fix:** Propagate the error or at minimum log it:
```rust
let mut stmt = match self.db.conn().prepare(&sql) {
    Ok(s) => Some(s),
    Err(e) => {
        tracing::warn!("excitability boost: prepare failed: {e}");
        None
    }
};
```

---

### 1c. Graph Boost — Cached Statement Prep Swallowed (`search.rs:~395–410`)

```rust
let mut fwd = conn.prepare_cached(...).ok();  // ← swallowed
let mut rev = conn.prepare_cached(...).ok();  // ← swallowed
```

Same pattern. If both fail, the entire graph spreading activation pass silently does nothing. The gate check (`has_coret`) passed, so the engine *thinks* it ran graph boost. It didn't.

**Fix:** Same as above — at minimum `tracing::warn!`.

---

### 1d. MMR Block — Embedding Load Swallowed (`search.rs:~450–480`)

```rust
if let Ok(mut stmt) = self.db.conn().prepare(&sql) {
    // ...
    if let Ok(rows) = stmt.query_map(...) {
        for r in rows.flatten() {
```

If `prepare` fails, `emb_map` stays empty. Every candidate has no embedding. Every `cosine()` call gets two empty slices and returns `0.0`. MMR degrades to pure relevance ranking (λ=1 mode). Results aren't wrong, but you've silently lost diversity re-ranking.

---

### 1e. Consolidation Retention Deletes (`consolidate/mod.rs:~220–228`)

```rust
let _ = conn.execute(
    "DELETE FROM co_retrieval WHERE timestamp < datetime('now', '-90 days')",
    [],
);
let _ = conn.execute(
    "DELETE FROM document_access WHERE timestamp < datetime('now', '-90 days')",
    [],
);
```

`let _ =` discards the `Result`. If these silently fail, the tables grow without bound. This is a maintenance operation — failure should at minimum be logged. The comment says "grow without limit otherwise," which means the author knows why this matters. Log the failure.

**Fix:**
```rust
if let Err(e) = conn.execute("DELETE FROM co_retrieval ...", []) {
    eprintln!("⚠ failed to trim co_retrieval: {e}");
}
```

---

## 2. Panics and Unsafe Operations

### 2a. `unwrap()` in sort comparators

Two instances in `search.rs`:

```rust
// After excitability boost re-sort:
fused.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));

// After graph boost re-sort:
fused.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));
```

These actually use `unwrap_or` which is fine — NaN produces `Equal`. Non-panicking. ✓

### 2b. `reranker.as_mut().unwrap()` (`search.rs`, `rerank()` method)

```rust
fn rerank(&mut self, ...) -> Result<Vec<SearchResult>> {
    let reranker = self.reranker.as_mut().unwrap();
```

This panics if `reranker` is `None`. The caller (`search()`) gates this with:
```rust
if opts.use_reranker && self.reranker.is_some() {
```

So in normal code paths this is safe — `is_some()` check before calling `rerank()`. But `rerank()` is a `&mut self` method on a public-ish type. If anyone calls it directly (or the guard check ever gets refactored away), this panics. It's a hidden invariant.

**Fix:** Return `Result` or use `if let`:
```rust
let reranker = self.reranker.as_mut()
    .ok_or_else(|| AxelError::Other("rerank called with no reranker".into()))?;
```

### 2c. `unwrap_or` on `serde_json::from_str` (`search.rs`, vector layer)

```rust
metadata: serde_json::from_str(&doc.2).unwrap_or_default(),
```

Non-panicking, but silently replaces corrupt metadata with `{}`. Fine in practice since metadata is display-only. Low risk. ✓ (document for correctness, not a real bug)

### 2d. `unwrap_or_else` in `default_sources()` (`consolidate/mod.rs:69`)

```rust
let home = std::env::var("HOME").unwrap_or_else(|_| "/home/haseeb".to_string());
```

Hardcoded fallback username. On any machine that isn't Haseeb's this produces wrong paths silently. Not a panic, but a silent misconfiguration. Should be `dirs::home_dir()` or a proper error.

### 2e. `unwrap_or_else` in path canonicalization (`reindex.rs:55, 66`)

```rust
let target_abs = std::fs::canonicalize(&source.path)
    .unwrap_or_else(|_| source.path.clone());
// ...
let abs = std::fs::canonicalize(file_path).unwrap_or_else(|_| file_path.to_path_buf());
```

If canonicalization fails (path doesn't exist), fallback to the raw path. Then `seen_on_disk` tracking and `indexed` map lookups may not match. Symptom: files could be re-indexed on every run without a clear error, or pruned incorrectly. The earlier check `if !source.path.is_dir()` provides some guard, but only for the source root — not for individual files.

---

## 3. DB Locked Scenario

There's no explicit DB connection management in `search.rs` — it calls `self.db.conn()` repeatedly in a single search call across multiple phases. SQLite in WAL mode tolerates concurrent readers, but if consolidation holds a write transaction (Phase 2 or 4 run a `conn.unchecked_transaction()`), a concurrent search could hit `SQLITE_BUSY`.

**What happens under lock:**

| Location | Failure mode |
|---|---|
| Vector layer `query_row` | `.ok()?` → drops hits silently |
| Excitability `prepare` | `.ok()` → skips entire boost block silently |
| Graph boost `prepare_cached` | `.ok()` → skips spreading activation silently |
| MMR `prepare` | `if let Ok` → emb_map empty, MMR degrades silently |
| `graph_search` `prepare` | `?` → propagates error, search fails loudly ✓ |

**The pattern is clear:** code that was written defensively uses `?` and surfaces errors. Code added later (excitability, graph boost, MMR) uses `.ok()` which silently degrades. The hot paths from a feature-complexity standpoint are exactly the ones that fail silently.

**Fix options:**
1. Set `busy_timeout` on the SQLite connection (e.g. 5s) so reads retry instead of immediately erroring. Simple, effective.
2. Use a read connection pool for search and a dedicated write connection for consolidation.
3. At minimum, change `.ok()` to `match`/`.unwrap_or_else(|e| { log; fallback })` so you know when DB is the problem.

---

## 4. Empty Brain — No Documents Scenario

**`search.rs`:** Vector search calls `self.index.search(&query_embedding, k)`. If the index is empty, this returns `Ok(vec![])`. The ranked list is empty. All layers that produce results come back empty. `results_lists` is empty or has only empty vecs. `rrf::reciprocal_rank_fusion` on empty input returns empty. Final result: `SearchResponse { results: [], stats: { ... } }`. **Graceful.** ✓

The excitability boost block gates on `if !fused.is_empty()`. The graph boost block gates on `if !fused.is_empty()`. MMR block gates on `fused.len() > opts.limit`. All safe with empty fused list. ✓

**`consolidate/strengthen.rs`:** `get_document_accesses_since` returns empty vec. `grouped` is empty. `doc_ids` is empty, skips bulk query. `boost_updates` is empty. Decay query runs but returns no rows (no documents exist). Commits nothing. Returns `StrengthenStats::default()`. **Graceful.** ✓

**`consolidate/reorganize.rs`:** `co_retrieval` table empty or missing. The graceful-degrade `match conn.prepare(...) { Err(_) => return Ok(stats) }` handles missing table. **Graceful.** ✓

**`consolidate/prune.rs`:** Schema-gated queries return `Vec::new()` on error. `dead` is empty. **Graceful.** ✓

**Verdict:** Empty brain is handled well everywhere. No panics, no errors, just empty results. ✓

---

## 5. Partial Failure Handling

### Consolidation pipeline (`consolidate/mod.rs`)

```rust
match reindex::reindex_source(search, src, opts.dry_run) {
    Ok(s) => { stats.reindex.checked += s.checked; ... }
    Err(e) => {
        partial_failure = true;
        eprintln!("⚠ reindex source [{}] failed: {e}", src.name);
    }
}
```

All four phases follow this pattern. One source directory failing doesn't abort others. One phase failing doesn't abort later phases. `partial_failure` is set but currently **only used for an `eprintln!` at the end** — it's not reflected in the return value or in the audit log. The caller gets `Ok(stats)` even if three of four phases failed.

**This is the most actionable finding in consolidation.** The function signature is `-> Result<ConsolidateStats>`. Partial failure should be surfaced. Options:

1. Return `Err` if `partial_failure` (strict — may be too harsh since phases are independent)
2. Add `partial_failure: bool` field to `ConsolidateStats` (good balance)
3. Add a `Vec<String>` of failure messages to `ConsolidateStats` (best for observability)

Currently the audit log entry doesn't record partial failure either — `ConsolidationLogEntry` has no field for it. The log shows a "successful" consolidation that may have done almost nothing.

### Reindex per-source (`reindex.rs`)

```rust
if let Err(e) = search.index_document(&doc_id, &content, None, Some(&abs_str)) {
    eprintln!("⚠ index_document failed for {abs_str}: {e}");
    continue;
}
```

One bad document doesn't crash the source. `stats.reindexed` is only incremented on success. **Good.** ✓

But the `stats.skipped` counter is only incremented for unreadable files, not for `index_document` failures. A document that fails embedding would not be counted in `skipped` or `reindexed` — it falls through the cracks of the stats entirely.

### Prune per-document

```rust
if let Some(fp) = file_path.as_deref() {
    db.delete_documents_by_file(fp)?;   // ← propagates, stops loop
} else {
    conn.execute(...)?;                  // ← propagates, stops loop
}
```

One delete failing in Phase 4 aborts the entire prune loop via `?`. This is inconsistent with the per-phase resilience elsewhere. If a document deletion fails (e.g. briefly locked), all subsequent candidates are skipped without being processed. Should use the same `match`/`continue` pattern as reindex.

---

## 6. Additional Issues

### Query expansion: direct index access without bounds check (`search.rs:~180`)

```rust
let top_content = &results_lists[0][0].content;
```

This is gated by `!results_lists.is_empty() && !results_lists[0].is_empty()`, so `[0][0]` is safe — the bounds are checked before access. ✓

### `unchecked_transaction` usage (`strengthen.rs`)

```rust
let tx = conn.unchecked_transaction()
    .map_err(|e| AxelError::Search(format!("begin tx: {e}")))?;
```

`unchecked_transaction` bypasses rusqlite's borrow-checker safety around transactions. It's used here because the connection is borrowed through a shared reference. It works, but it means the compiler won't catch you if you accidentally nest transactions or use the connection outside the TX scope. Low immediate risk but worth noting as a footgun.

### `strip_prefix(...).unwrap_or(file_path)` (`reindex.rs:~81`)

```rust
let relative_id = file_path
    .strip_prefix(&target_abs).unwrap_or(file_path)
    ...
```

`strip_prefix` can fail if `file_path` isn't under `target_abs` (e.g. symlinks resolved differently). The fallback uses the full `file_path`, which means `doc_id` will contain the full absolute path. This is a data quality issue — the doc_id format becomes inconsistent. Documents re-indexed after the symlink resolves differently get a new doc_id and are treated as new documents, duplicating content.

---

## Summary Table

| Issue | Location | Severity | Type |
|---|---|---|---|
| Vector hits dropped silently on DB error | `search.rs` vector layer | **High** | Silent failure |
| Excitability boost silently skipped on DB error | `search.rs` excitability | **High** | Silent failure |
| Graph boost silently skipped on DB error | `search.rs` graph boost | **Medium** | Silent failure |
| MMR diversity silently skipped on DB error | `search.rs` MMR | **Medium** | Silent failure |
| `partial_failure` not surfaced in return value | `consolidate/mod.rs` | **Medium** | Missing signal |
| `partial_failure` not in audit log | `consolidate/mod.rs` | **Medium** | Missing signal |
| Prune loop aborts on first delete failure | `prune.rs` | **Medium** | Partial failure |
| Retention deletes not logged on failure | `consolidate/mod.rs` | **Low** | Silent failure |
| `reranker.unwrap()` hidden panic | `search.rs` | **Low** | Panic risk |
| Hardcoded `/home/haseeb` fallback | `consolidate/mod.rs` | **Low** | Misconfiguration |
| Failed index_document not in skipped stats | `reindex.rs` | **Low** | Observability |
| `unchecked_transaction` footgun | `strengthen.rs` | **Low** | Safety |
| Symlink/strip_prefix doc_id inconsistency | `reindex.rs` | **Low** | Data quality |

---

## Recommended Fixes — Priority Order

**Do these first:**

1. **`search.rs` — replace all `.ok()` with `.unwrap_or_else(|e| { tracing::warn!(...); None/default })`** in the excitability, graph boost, and vector layer blocks. This converts silent degradation into observable degradation.

2. **`consolidate/mod.rs` — add `partial_failure: bool` to `ConsolidateStats`** and set it from the `partial_failure` local. Surface it in the audit log entry.

3. **`prune.rs` — change `?` to `match`/`continue`** in the document deletion loop for consistency with the per-phase resilience model.

4. **SQLite busy_timeout** — set a 5–10 second busy timeout on the database connection so reads retry under write contention rather than immediately erroring. This fixes the DB-locked scenario at the root.

**Do these second:**

5. Replace `unwrap_or_else(|_| "/home/haseeb")` with a proper `dirs::home_dir()` call.
6. Log the retention delete failures instead of discarding with `let _ =`.
7. Track failed `index_document` calls in `stats.skipped`.
8. Make `rerank()`'s `unwrap()` an explicit `?`-propagated error.
