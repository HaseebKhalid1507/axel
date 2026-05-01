# Performance Review: Axel Consolidation System

**Reviewed:** `crates/axel/src/consolidate/` + `crates/velocirag/src/db.rs` (document_access + co_retrieval sections)
**Date:** 2025-07-03

---

## Executive Summary

The consolidation pipeline has **4 confirmed N+1 query patterns**, **5 missing indexes** on columns that are actively filtered in consolidation queries, **2 unbounded append-only tables** with no cleanup path, and **1 embedding call per new file** with no batching or cap. None of these are catastrophic at small scale — but they're time bombs. Each scales poorly in a different dimension.

---

## Finding 1 — N+1 in `strengthen.rs`: Per-Doc Excitability Lookup

**File:** `consolidate/strengthen.rs:59–66`
**Severity:** High

```rust
for (doc_id, (count, score_sum, score_n)) in &grouped {
    let current: f64 = match conn.query_row(
        "SELECT excitability FROM documents WHERE doc_id = ?1",
        params![doc_id],
        ...
    ) { ... };
    // then UPDATE documents SET excitability = ?1 WHERE doc_id = ?2
}
```

One `SELECT` + one `UPDATE` per accessed doc, serialized. With 100 documents in the access log window, that's 200 round-trips. The SELECT hits `idx_doc_doc_id`, so it's fast per-query — but the loop overhead is pure waste.

**Fix:** Fold the SELECT into a single `WITH` CTE or `JOIN`, compute new excitability in SQL, and issue a bulk `UPDATE` (or a `CASE WHEN` batch). The UPDATE can stay per-row since SQLite has no `UPDATE ... FROM` — but the SELECT definitely doesn't need to be in the loop:

```sql
-- Compute all in one shot
SELECT doc_id, excitability FROM documents
WHERE doc_id IN (/* the grouped keyset */)
```

Then do the math in Rust over the returned map, then issue per-row UPDATEs (still N writes, but zero extra reads). Alternatively, wrap the batch in a single `BEGIN`/`COMMIT` transaction to eliminate N fsync-equivalent serialization points (see Finding 6).

---

## Finding 2 — N+1 in `reorganize.rs` Upsert Loop: Per-Pair Edge Lookup

**File:** `consolidate/reorganize.rs:91–103`
**Severity:** High

```rust
for (a, b, count) in &pairs {
    let existing: Option<(String, f64)> = conn.query_row(
        "SELECT id, weight FROM edges
         WHERE type = 'co_retrieved'
           AND ((source_id = ?1 AND target_id = ?2)
                OR (source_id = ?2 AND target_id = ?1))
           AND (valid_to IS NULL OR valid_to > datetime('now'))
         LIMIT 1",
        params![a, b], ...
    ).ok();
```

One edge lookup per co-retrieval pair. For a busy brain with 50 pairs above threshold, that's 50 queries. Worse, this query uses an `OR` on `source_id`/`target_id` which prevents SQLite from using either `idx_edges_source` or `idx_edges_target` efficiently — it has to union two index scans or fall back.

**Fix 1:** Use the deterministic `coret_edge_id(a, b)` to look up by primary key instead:
```sql
SELECT id, weight FROM edges WHERE id = ?1
  AND (valid_to IS NULL OR valid_to > datetime('now'))
```
`coret_edge_id` is already computed right above this query — use it. Primary key lookup is O(log n) with no OR ambiguity.

**Fix 2:** Pre-load all `co_retrieved` edges once (the decay step already does this), build a `HashMap<String, (String, f64)>` keyed by edge_id, and avoid the per-pair query entirely.

---

## Finding 3 — N+1 in `reorganize.rs` Decay Loop: Per-Edge Co-Retrieval Count

**File:** `consolidate/reorganize.rs:159–175`
**Severity:** High

```rust
for (id, src, tgt, weight) in live_edges {
    let recent: i64 = conn.query_row(
        "SELECT COUNT(*) FROM co_retrieval
         WHERE timestamp > datetime('now', ?1)
           AND ((doc_id_a = ?2 AND doc_id_b = ?3)
                OR (doc_id_a = ?3 AND doc_id_b = ?2))",
        ...
    ).unwrap_or(0);
```

One query into `co_retrieval` per live `co_retrieved` edge. If there are 200 co_retrieved edges, that's 200 queries. Each of those hits an unbounded, never-cleaned table (see Finding 5). The `OR` reversal also prevents index use on `idx_coret_pair(doc_id_a, doc_id_b)` for the `(b, a)` direction.

**Fix:** Pull all recent co-retrieval activity in one query before the loop:
```sql
SELECT doc_id_a, doc_id_b, COUNT(*) FROM co_retrieval
WHERE timestamp > datetime('now', '-30 days')
GROUP BY doc_id_a, doc_id_b
```
Build a `HashSet<(String, String)>` (canonical order, smaller of the two first), then do `set.contains(&(lo, hi))` per edge. Zero extra queries in the loop.

---

## Finding 4 — N+1 in `prune.rs` Misaligned Enrichment Loop

**File:** `consolidate/prune.rs:147–162`
**Severity:** Medium

```rust
for (doc_id, hits, _avg_score) in misaligned {
    let meta: Option<(f64, i64, i64)> = conn.query_row(
        "SELECT excitability, access_count,
                CAST(julianday('now') - julianday(created) AS INTEGER)
         FROM documents WHERE doc_id = ?1",
        params![doc_id], ...
    ).ok();
```

One `documents` lookup per misaligned doc. Minor — the misaligned set is typically small (docs with ≥5 hits and avg score < 0.015 is a tight filter). But it's unnecessary: the data can be joined directly in the `misaligned` query itself:

```sql
SELECT da.doc_id, COUNT(*) as hits, AVG(da.score) as avg_score,
       d.excitability, d.access_count,
       CAST(julianday('now') - julianday(d.created) AS INTEGER) as age_days
FROM document_access da
JOIN documents d ON d.doc_id = da.doc_id
WHERE da.access_type = 'search_hit'
GROUP BY da.doc_id
HAVING hits >= 5 AND avg_score < 0.015
```

One query, no loop, all the data.

---

## Finding 5 — N+1 in `reindex.rs` Competitive Allocation: Per-Hit Excitability

**File:** `consolidate/reindex.rs:154–160`
**Severity:** Medium

```rust
for hit in response.results.iter().filter(...).take(ALLOCATION_K) {
    let excitability: f64 = match search.db().conn().query_row(
        "SELECT excitability FROM documents WHERE doc_id = ?1",
        params![&hit.doc_id], ...
    ) { ... };
```

ALLOCATION_K is 5, so this is at most 5 queries per new file — not terrible in isolation. But this function is called once per new file in the reindex loop, compounding with Finding 7 (one full embedding search per new file). Fix is the same pattern: pull excitability for all K hits in a single `WHERE doc_id IN (...)` query before the loop.

---

## Finding 6 — Missing Indexes on Consolidation-Critical Columns

**File:** `velocirag/src/db.rs` (schema)
**Severity:** High

The `documents` table has two indexes: `idx_doc_doc_id(doc_id)` and `idx_doc_file_path(file_path)`. The consolidation phases add four filtered columns via migration (no indexes added for them):

| Column | Used in | Query Type | Index? |
|---|---|---|---|
| `excitability` | `prune.rs:75–77` | Range filter `< 0.15` | ❌ None |
| `access_count` | `prune.rs:77` | Equality `= 0` | ❌ None |
| `last_accessed` | `strengthen.rs:97` | Range filter `< datetime(...)` | ❌ None |
| `indexed_at` | `strengthen.rs:98` | Range filter `< datetime(...)` | ❌ None |

The prune stale-document query (`WHERE excitability < 0.15 AND access_count = 0 AND ...age > 60`) does a **full table scan** on every consolidation run. The strengthen decay query (`WHERE last_accessed < datetime(...)`) does another full table scan across all documents.

Additionally, `co_retrieval` has `idx_coret_pair(doc_id_a, doc_id_b)` but **no index on `timestamp`**. The reorganize aggregation query filters on `WHERE timestamp > ?1` — this is a full scan of an ever-growing table.

**Recommended indexes to add (in schema init or a migration):**

```sql
-- For prune stale query
CREATE INDEX IF NOT EXISTS idx_doc_excitability ON documents(excitability);
CREATE INDEX IF NOT EXISTS idx_doc_access_count ON documents(access_count);

-- For strengthen decay query
CREATE INDEX IF NOT EXISTS idx_doc_last_accessed ON documents(last_accessed);

-- For reorganize aggregation (timestamp filter)
CREATE INDEX IF NOT EXISTS idx_coret_timestamp ON co_retrieval(timestamp);

-- Compound for reorganize/decay edge lookup (eliminates OR scan)
CREATE INDEX IF NOT EXISTS idx_edges_type_source_target ON edges(type, source_id, target_id);
```

The `idx_doc_excitability` index also enables a future optimization: SQLite can seek directly to `excitability < 0.15` rather than scanning and filtering.

---

## Finding 7 — Unbounded `co_retrieval` Table: O(n²) Growth Per Search

**File:** `main.rs:539–542`, `mcp.rs:201–207`, `velocirag/src/db.rs:927–945`
**Severity:** High (long-term)

Every search logs co-retrieval pairs for the top-5 results. With k=5 results, that's C(5,2) = **10 rows inserted per search**. There is **no DELETE, TRUNCATE, or archival path** anywhere in the codebase for `co_retrieval`. The reorganize phase reads from it but never writes back cleanup.

At 100 searches/day, that's 1,000 rows/day → 365,000 rows/year. This table grows indefinitely. The `co_retrieval` timestamp filter in reorganize (`WHERE timestamp > ?1`) already partially mitigates this by ignoring old data — but the table still accumulates, bloating every `GROUP BY` aggregation and every per-edge decay count query.

**Fix:** Add a cleanup step to `reorganize()` after processing:

```rust
// After upsert loop — delete co_retrieval rows that have been processed
// (older than STALE_DAYS, all pairs below threshold already handled)
if !dry_run {
    conn.execute(
        "DELETE FROM co_retrieval WHERE timestamp < datetime('now', ?1)",
        params![format!("-{} days", STALE_DAYS)],
    )?;
}
```

This is safe: the reorganize phase only cares about the window since the last consolidation (for upsert) and the last 30 days (for decay). Anything older has already been promoted to an edge or decayed away.

Similarly, `document_access` has no cleanup either. The strengthen phase only reads since the last consolidation — old rows accumulate. The misaligned query in prune (`WHERE access_type = 'search_hit'` with no time bound) scans the **entire history** of document_access. Add a similar bounded DELETE after the strengthen phase.

---

## Finding 8 — Embedding Cost: One Full Vector Search Per New File

**File:** `consolidate/reindex.rs:136–186`
**Severity:** Medium

`allocate_new_doc` calls `search.search(&content, ALLOCATION_K + 1)` for **every new file** discovered during reindex. This is a full ANN search (cosine similarity over all embeddings in the brain). At a brain with 10,000 documents and a reindex pass that discovers 50 new files, that's 50 embedding calls + 50 ANN scans.

The embedding model call itself is the expensive part: if using an external API (OpenAI, Cohere), this is 50 HTTP round-trips that block the reindex loop synchronously. If local (ONNX/llama.cpp), it's CPU-intensive and sequential.

**Issues:**
1. **No batch embedding:** Each `search.search()` presumably embeds the query content once per call. There's no batching of new-doc embeddings.
2. **Content re-read from DB:** `allocate_new_doc` reads `content` back from `documents` via `SELECT content FROM documents WHERE doc_id = ?1` — the content was just indexed moments ago in `reindex_source`. It could be passed directly.
3. **No cap on allocation per pass:** A reindex of a large new source could trigger hundreds of embedding calls in one pass.

**Fix options:**
- Pass `content` directly into `allocate_new_doc` (eliminates the DB re-read).
- Collect all `(new_doc_id, content)` pairs first, then batch-embed and batch-search.
- Add a per-pass cap: `newly_indexed.truncate(MAX_ALLOCATION_PER_PASS)` with a configurable `MAX_ALLOCATION_PER_PASS` constant (e.g., 20) so a large initial import doesn't hammer the embedding model.
- Defer allocation to a separate `--allocate-only` pass that can be rate-limited independently.

---

## Finding 9 — No Transaction Wrapping on Batch Writes

**File:** `consolidate/strengthen.rs`, `consolidate/reorganize.rs`
**Severity:** Medium

All the `UPDATE` statements in the strengthen boost loop, the decay update loop, and the reorganize edge-update loop are executed as individual auto-commit statements. SQLite in WAL mode handles this reasonably, but each auto-commit still requires a fsync-equivalent barrier.

For strengthen: if 200 documents need excitability updates, that's 200 separate commits. Wrapping in a transaction cuts this to 1 commit:

```rust
let tx = conn.transaction()?;
for (doc_id, new_excitability) in updates {
    tx.execute("UPDATE documents SET excitability = ?1 WHERE doc_id = ?2", ...)?;
}
tx.commit()?;
```

Same applies to the reorganize decay loop (`UPDATE edges SET weight`) and the prune auto-remove loop (`DELETE FROM documents`). These are the most write-heavy sections of the pipeline.

**Expected speedup:** 10–50x on spinning disk; 2–5x on SSD. SQLite's own documentation puts this as the single highest-impact optimization for batch writes.

---

## Finding 10 — `get_document_accesses_since` Loads All Rows into Memory

**File:** `velocirag/src/db.rs:949–976`
**Severity:** Low-Medium

```rust
pub fn get_document_accesses_since(&self, since: &str) -> Result<Vec<DocumentAccess>> {
    // ... query ...
    let mut out = Vec::new();
    for r in rows { out.push(r?); }
    Ok(out)
```

The full result set is collected into a `Vec<DocumentAccess>` and returned. `strengthen()` then iterates over it to build a `HashMap`. If the window since last consolidation is large (first run after a long gap, or consolidation was skipped for weeks), this can be a large in-memory load — each `DocumentAccess` carries `doc_id` (String), `query` (Option<String> — potentially large), `timestamp` (String).

More critically: the `query` field is stored and returned for every access event even though `strengthen()` never uses it (only `doc_id` and `score` are consumed). This is wasted allocation.

**Fix:** Either stream the rows (process in the iterator without collecting), or add a targeted query that pre-aggregates on the DB side:

```sql
SELECT doc_id, COUNT(*) as access_count,
       AVG(score) as avg_score, SUM(CASE WHEN score IS NOT NULL THEN 1 ELSE 0 END) as score_n
FROM document_access
WHERE timestamp >= ?1
GROUP BY doc_id
```

This returns one row per doc instead of one row per access event, eliminates the `HashMap` grouping in Rust, and never loads query text into memory.

---

## Finding 11 — `last_consolidation_time` Has No Index

**File:** `velocirag/src/db.rs:980–991`
**Severity:** Low

```sql
SELECT finished_at FROM consolidation_log
WHERE finished_at IS NOT NULL
ORDER BY finished_at DESC LIMIT 1
```

`consolidation_log` has no index on `finished_at`. This table will stay tiny (one row per consolidation run), so this is genuinely negligible — but worth noting if the table is ever queried in hot paths. No action required unless query patterns change.

---

## Summary Table

| # | Finding | File | Severity | Fix Complexity |
|---|---|---|---|---|
| 1 | N+1: excitability SELECT per accessed doc | `strengthen.rs:59` | High | Low |
| 2 | N+1: edge lookup per co-retrieval pair (bad OR query) | `reorganize.rs:91` | High | Low |
| 3 | N+1: co_retrieval COUNT per live edge in decay loop | `reorganize.rs:159` | High | Medium |
| 4 | N+1: documents lookup per misaligned doc | `prune.rs:147` | Medium | Low |
| 5 | N+1: excitability lookup per ANN hit in allocation | `reindex.rs:154` | Medium | Low |
| 6 | Missing indexes on `excitability`, `access_count`, `last_accessed`, `co_retrieval.timestamp` | `db.rs` schema | High | Low |
| 7 | `co_retrieval` + `document_access` grow unbounded, never cleaned | `reorganize.rs`, `db.rs` | High | Low |
| 8 | One embedding search per new file, content re-read from DB, no cap | `reindex.rs:136` | Medium | Medium |
| 9 | Batch writes not wrapped in transactions | `strengthen.rs`, `reorganize.rs` | Medium | Low |
| 10 | Full access log loaded into memory; query field fetched but unused | `db.rs:949` | Low-Medium | Medium |
| 11 | No index on `consolidation_log.finished_at` | `db.rs:980` | Low | Negligible |

---

## Recommended Fix Order

1. **Transaction wrapping** (Finding 9) — 5 lines of code, potentially 10–50x write speedup. Do this first.
2. **Missing indexes** (Finding 6) — Add to schema migration. Zero logic change.
3. **co_retrieval cleanup** (Finding 7) — Add DELETE to end of reorganize. Prevents indefinite growth.
4. **Reorganize N+1s** (Findings 2 & 3) — Replace OR-based edge lookup with PK lookup; pre-load recent co-retrieval set before decay loop.
5. **Strengthen N+1** (Finding 1) — Pre-load excitability for accessed docs in one query.
6. **Access log aggregation** (Finding 10) — Push grouping into SQL, eliminate large Vec allocation.
7. **Allocation content re-read** (Finding 8, partial) — Pass content through instead of re-querying DB.
8. **Allocation per-pass cap** (Finding 8) — Add `MAX_ALLOCATION_PER_PASS` constant.
9. **Prune join** (Finding 4) — Minor; clean it up when touching prune anyway.
