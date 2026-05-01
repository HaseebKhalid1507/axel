# Performance Review — velocirag/src/search.rs + axel/src/consolidate/

**Reviewed:** `crates/velocirag/src/search.rs`, `crates/axel/src/consolidate/{mod,reindex,strengthen,reorganize,prune}.rs`
**Focus:** N+1 queries, unnecessary allocations, algorithmic complexity, query batching, `prepare` vs `prepare_cached`

---

## Executive Summary

The search pipeline has been partially hardened — the excitability batch-load and the graph-boost `prepare_cached` pair are good. But there are still sharp edges that compound under load: a genuine **N+1 in the vector layer**, a real **O(n²) inner loop in MMR** with a redundant full-table batch-load feeding it, **2×N `prepare()` calls per search call in the metadata layer**, and a slow gate query that fires unconditionally on every search. Total minimum SQL round-trips per `search()` call with all layers active comes out to **≥ 6 + 2W + 5K** (where W = query word count, K = graph-search word count). Details below.

---

## search.rs — Issues by Layer

### 1. LAYER 1 (Vector): Classic N+1 — one SELECT per vector hit

**Location:** `search.rs:152–164`

```rust
let ranked: Vec<RankedResult> = vector_results
    .into_iter()
    .filter_map(|vr| {
        let doc = self.db.conn().query_row(
            "SELECT doc_id, content, metadata FROM documents WHERE id = ?1",
            [vr.id as i64],
            |row| { ... },
        ).ok()?;
        ...
    })
    .collect();
```

**Problem:** For `limit=10` and `VECTOR_CANDIDATES_MULTIPLIER=3` this fires **30 individual SELECT statements** — one per vector result. Each is a prepared-then-discarded statement (no `prepare_cached`).

**Fix:** Collect all rowids first, then run a single `WHERE id IN (...)` query:

```rust
let ids: Vec<i64> = vector_results.iter().map(|vr| vr.id as i64).collect();
// build placeholders, single query, then join back by id for ordering/score
let placeholders = ids.iter().enumerate()
    .map(|(i, _)| format!("?{}", i + 1))
    .collect::<Vec<_>>().join(",");
let sql = format!(
    "SELECT id, doc_id, content, metadata FROM documents WHERE id IN ({})",
    placeholders
);
// Then reconstruct RankedResult ordered by original vector score
```

**Impact:** 30 round-trips → 1. Most expensive single fix available.

---

### 2. LAYER 3 (Graph Search): N+1 `prepare()` inside word loop

**Location:** `search.rs:584–596` (`graph_search` method)

```rust
for word in &words {
    let pattern = format!("%{}%", word.to_lowercase());
    let mut stmt = self.db.conn().prepare(
        "SELECT id, title, content FROM nodes WHERE LOWER(title) LIKE ?1 LIMIT 10",
    )?;
    ...
}
```

**Problem:** `prepare()` is called once per query word. For a five-word query that's five statement compiles. The same SQL string is prepared repeatedly.

**Fix:** Call `prepare_cached()` instead of `prepare()`. The statement cache keyed on the SQL string will return the pre-compiled statement on the second and subsequent words:

```rust
let mut stmt = self.db.conn().prepare_cached(
    "SELECT id, title, content FROM nodes WHERE LOWER(title) LIKE ?1 LIMIT 10",
)?;
```

**Impact:** Eliminates K−1 redundant compiles per search (K = word count). Low cost, zero-risk change.

---

### 3. GRAPH BOOST GATE: Unconditional `query_row` before every boost pass

**Location:** `search.rs:353–359`

```rust
let has_coret: bool = self.db.conn()
    .query_row(
        "SELECT EXISTS(SELECT 1 FROM edges WHERE type = 'co_retrieved'
                       AND (valid_to IS NULL OR valid_to > datetime('now')) LIMIT 1)",
        [],
        |row| row.get::<_, i64>(0).map(|n| n != 0),
    )
    .unwrap_or(false);
```

**Problem:** This fires on *every single search call*, even when a search is repeated microseconds apart and the graph hasn't changed. For a system doing many searches per session this is pure overhead.

**Fix:** Cache the result in `SearchEngine` state, invalidated after N seconds or after the engine is reconstructed:

```rust
// In SearchEngine struct:
coret_exists_cache: Option<(bool, std::time::Instant)>,

// In search():
let has_coret = match self.coret_exists_cache {
    Some((v, t)) if t.elapsed() < Duration::from_secs(30) => v,
    _ => { /* query + store */ }
};
```

Or simpler: move the gate check to a once-per-engine `new()` call if the engine is constructed per-session.

**Impact:** Eliminates 1 round-trip per search call. Low effort.

---

### 4. MMR: O(n²) inner loop + redundant batch load

**Location:** `search.rs:512–538` (inner MMR loop)

```rust
while selected.len() < opts.limit && !remaining.is_empty() {
    for (i, cand) in remaining.iter().enumerate() {         // O(remaining)
        let max_sim = if let Some(ce) = emb_map.get(&cand.doc_id) {
            selected_embs
                .iter()
                .map(|se| cosine(ce, se))                   // O(selected)
                .fold(f32::MIN, f32::max)
        }
    }
}
```

**Problem:** Standard greedy MMR is O(n·m) per iteration and O(n²·m) total where n = `remaining` size and m = `selected` size. With `fused.len()` potentially being several × `limit` and embeddings being 384-dimensional, this burns CPU.

The batch embedding load immediately above (`SELECT doc_id, embedding FROM documents WHERE doc_id IN (...)`) is also wasted when `fused.len() <= opts.limit` — the condition `if fused.len() > opts.limit` guards the MMR block but the SQL is built and executed before checking this for the **embedding load** specifically. Actually this is guarded — re-read: the entire block starting at line ~451 is inside `if fused.len() > opts.limit`. So the load is gated. ✓

The O(n²) loop itself remains. For `limit=10` and `fused` of 30 docs it's manageable, but as the fused pool grows (e.g., `limit=50`) it becomes painful.

**Fix options:**
- Accept the complexity for typical `limit` values (≤20). Document the known bound.
- For larger limits, precompute pairwise similarities into a triangular matrix before the selection loop (trades memory for iteration speed), or use max-heap on MMR scores and update incrementally.
- More impactful: avoid hitting MMR at all for short-content results with identical embeddings by deduplicating at the RRF stage by `doc_id` prefix (same file, different chunks).

---

### 5. MMR: `prepare()` instead of `prepare_cached()` for embedding batch load

**Location:** `search.rs:461`

```rust
if let Ok(mut stmt) = self.db.conn().prepare(&sql) {
```

**Problem:** The SQL string is dynamically generated with N placeholders each time. `prepare_cached()` can't help here because the key changes with N. This is unavoidable for the IN-list pattern. Not a bug — noted for awareness.

The excitability batch load at line 307 has the same structure and the same caveat:
```rust
let mut stmt = self.db.conn().prepare(&sql).ok();
```
Both are fine as-is; the dynamic IN-list precludes caching.

---

### 6. LAYER 4 (Metadata): 2×W `prepare()` calls per search

**Location:** `search.rs:252–273`

```rust
for word in &words {
    if let Ok(tag_docs) = self.db.search_by_tag(word, 10) { ... }
    if let Ok(ref_docs) = self.db.search_by_cross_ref(word, 10) { ... }
}
```

**`db.rs:839`** and **`db.rs:855`**: each method calls `self.conn.prepare(...)` internally, not `prepare_cached()`:

```rust
pub fn search_by_tag(&self, tag_pattern: &str, limit: usize) -> Result<Vec<(String, String)>> {
    let mut stmt = self.conn.prepare(   // ← should be prepare_cached
        "SELECT d.doc_id, d.content ..."
    )?;
```

**Problem:** For a 4-word query, this compiles the tag-search SQL 4 times and the cross-ref SQL 4 times = **8 statement compiles** for what should be 0 (after the first).

**Fix:** Change both `db::search_by_tag` and `db::search_by_cross_ref` to use `prepare_cached()`. The SQL is static so caching is safe and beneficial.

**Impact:** 2W compiles → 2 (cached on first use). Easy win.

---

### 7. LAYER 4 (Metadata): Could be a single batched query

**Location:** `search.rs:248–284`

**Problem:** The metadata layer issues 2×W queries (tag + cross-ref per word) sequentially. For a 5-word query that's 10 queries.

**Alternative design:** Batch all words into a single IN-list query for each type:

```sql
-- Tags: one query instead of W queries
SELECT d.doc_id, d.content
FROM documents d
JOIN document_tags dt ON d.id = dt.doc_id
JOIN tags t ON dt.tag_id = t.id
WHERE LOWER(t.name) IN (?, ?, ?)   -- all words at once
LIMIT ?

-- Cross-refs: same pattern
SELECT d.doc_id, d.content
FROM documents d JOIN cross_refs cr ON d.id = cr.doc_id
WHERE LOWER(cr.ref_target) IN (?, ?, ?)
LIMIT ?
```

**Impact:** W+W queries → 2 queries regardless of query length. For a 5-word query: 10 → 2.

---

### 8. `graph_search`: O(W × N) repeated `to_lowercase()` in scoring loop

**Location:** `search.rs:631–641`

```rust
for word in &words {
    let w = word.to_lowercase();              // allocates per word per neighbor
    if title_lower.contains(&w) { ... }
    if content_lower.contains(&w) { ... }
}
```

**Problem:** `word.to_lowercase()` allocates a new `String` per word per neighbor node. With `GRAPH_MAX_RESULTS=20` matched nodes and potentially 5 words, that's 100 short-lived allocations per `graph_search` call. The words were already lowercased at the outer loop level (`word.to_lowercase()` at line 585 for the pattern) but not preserved for reuse here.

**Fix:** Lower-case the query words once before the neighbor-scoring loop:

```rust
let words_lower: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
// Then use &words_lower[i] in the inner loop — zero extra allocations
```

---

### 9. `graph_search`: BFS via recursive CTE called once per matched node

**Location:** `search.rs:613` / `db.rs:737`

```rust
for (node_id, title, _content) in &matched_nodes {
    let neighbors = self.db.get_neighbors(node_id, GRAPH_DEPTH, GRAPH_MAX_RESULTS)?;
```

**Problem:** `get_neighbors` prepares and executes a heavy recursive CTE once per matched node. With K=5 query words and 10 nodes per word the outer loop could have up to 50 matched nodes (after dedup, fewer). Each fires a recursive CTE traversal.

**Existing mitigation:** `seen_ids` deduplicates matched nodes, so in practice it's bounded by the number of distinct title-matching nodes. Acceptable for typical queries.

**Note:** `get_neighbors` itself uses `prepare()` not `prepare_cached()`. For the graph_search hot path this query is prepared on every call. Change to `prepare_cached()` in `db::get_neighbors`.

---

## Total SQL Round-trips Per `search()` Call (all layers active, limit=10)

| Step | Queries | Note |
|---|---|---|
| Vector layer: embed | 0 (in-process) | |
| Vector layer: N+1 fetches | **30** | `limit × 3 = 30` individual SELECTs — the big one |
| Keyword search | 1 | `keyword_search()` |
| Graph search node match | K | one per query word (should be `prepare_cached`) |
| Graph search BFS | ≤ K×10 | one CTE per matched node (typically 2–5 in practice) |
| Metadata: tag search | W | one per word |
| Metadata: cross-ref search | W | one per word |
| Excitability batch | 1 | good, already batched |
| Graph boost gate | 1 | fires unconditionally |
| Graph boost fwd edges | up to 5 | `prepare_cached` ✓ |
| Graph boost rev edges | up to 5 | `prepare_cached` ✓ |
| Graph boost new-node fetch | 0–N | only if neighbors not in fused set |
| MMR embedding batch | 1 | good, already batched |
| **TOTAL (typical 4-word query)** | **≈ 47+** | dominated by vector N+1 |

After fixing issue #1 alone, typical total drops to **≈ 18**.
After fixing issues #1, #6, and #7 together, typical total drops to **≈ 8–10**.

---

## consolidate/ — Issues

### C1. `strengthen.rs`: `prepare()` inside boost update loop

**Location:** `strengthen.rs:120–127`

```rust
let mut stmt = tx.prepare(
    "UPDATE documents SET excitability = ?1 WHERE doc_id = ?2"
).map_err(...)? ;
for (doc_id, new_excit) in &boost_updates {
    stmt.execute(params![new_excit, doc_id])?;
}
```

**This is fine** — `stmt` is prepared once outside the loop and reused. ✓

Similarly for the decay loop at line 177. ✓

**The N+1 that existed here has already been fixed** (bulk excitability preload at lines 63–82). Good.

---

### C2. `strengthen.rs`: Decay query scans all documents

**Location:** `strengthen.rs:132–153`

```rust
let mut stmt = conn.prepare(
    "SELECT doc_id, excitability, COALESCE(access_count, 0), ... AS days_inactive
     FROM documents
     WHERE (last_accessed IS NULL OR last_accessed < datetime('now', ?1))
       AND (indexed_at  IS NULL OR indexed_at  < datetime('now', ?1))",
)?;
```

**Problem:** This is a full table scan against a compound OR predicate on nullable columns. The index `idx_doc_last_accessed` exists but the `IS NULL OR col < ?` pattern typically prevents SQLite from using it efficiently — it must check every row.

**Fix:** Split into two queries and UNION, or restructure to allow index usage:

```sql
-- Last accessed is non-null and old:
SELECT ... FROM documents WHERE last_accessed < datetime('now', ?)
UNION ALL
-- Never accessed, but indexed and old:
SELECT ... FROM documents
WHERE last_accessed IS NULL AND indexed_at < datetime('now', ?)
UNION ALL
-- Never accessed, never indexed (edge case):
SELECT ... FROM documents
WHERE last_accessed IS NULL AND indexed_at IS NULL
```

Each branch can use its index. This matters at scale (tens of thousands of documents).

**Use `prepare_cached()`** here too — this query is run every consolidation pass.

---

### C3. `prune.rs`: N+1 in misaligned embeddings phase

**Location:** `prune.rs:145–162`

```rust
for (doc_id, hits, _avg_score) in misaligned {
    let meta: Option<(f64, i64, i64)> = conn
        .query_row(
            "SELECT excitability, access_count,
                    CAST(julianday('now') - julianday(created) AS INTEGER)
             FROM documents WHERE doc_id = ?1",
            params![doc_id],
            |row| { ... },
        )
        .ok();
```

**Problem:** One `query_row` per misaligned document. If there are 50 misaligned docs, that's 50 round-trips.

**Fix:** Join the metadata into the aggregation query directly:

```sql
SELECT da.doc_id,
       COUNT(*) as hits,
       AVG(da.score) as avg_score,
       d.excitability,
       d.access_count,
       CAST(julianday('now') - julianday(d.created) AS INTEGER) as age_days
FROM document_access da
JOIN documents d ON d.doc_id = da.doc_id
WHERE da.access_type = 'search_hit'
GROUP BY da.doc_id, d.excitability, d.access_count, d.created
HAVING COUNT(*) >= 5 AND AVG(da.score) < 0.015
```

One query replaces N+1. The `doc_id` field in `document_access` has an index (`idx_docaccess_doc_id`) so the join should be efficient.

---

### C4. `reorganize.rs`: `co_retrieved` edges loaded twice

**Location:** `reorganize.rs:94–113` and `reorganize.rs:198–212`

The `live_edges` query at line 198 fetches all `co_retrieved` edges a second time — the same set was already fetched into `existing_edges` at line 95. They're loaded for different purposes (upsert lookup vs. decay iteration) but the data overlap is near-total.

**Fix:** Compute both uses from a single load. Load once into a `Vec`, then build `existing_edges` HashMap from that vec for upsert lookups, and reuse the vec for the decay iteration:

```rust
let all_coret_edges: Vec<(String, String, String, f64)> = /* single SELECT */;
let existing_edges: HashMap<String, (String, f64)> = all_coret_edges.iter()
    .map(|(id, src, tgt, w)| (coret_edge_id(src, tgt), (id.clone(), *w)))
    .collect();
// ... upsert loop using existing_edges ...
// ... decay loop using all_coret_edges directly ...
```

**Impact:** Halves the edge-table reads during Phase 3. Noticeable on large graphs.

---

### C5. `reorganize.rs`: Per-edge `invalidate_edge` and `UPDATE` calls in decay loop

**Location:** `reorganize.rs:215–235`

```rust
for (id, src, tgt, weight) in live_edges {
    if weight < EDGE_REMOVAL_THRESHOLD {
        if !dry_run {
            db.invalidate_edge(&id)?;       // one UPDATE per edge
        }
    } else {
        if !dry_run {
            conn.execute(
                "UPDATE edges SET weight = ?1 WHERE id = ?2",
                params![new_w, id],
            )?;                              // one UPDATE per edge
        }
    }
}
```

**Problem:** Each invalidation/update is a separate statement execute. For a large graph with many stale edges this could be hundreds of individual UPDATEs.

**Fix:** Batch both operations in a transaction (already done implicitly via the connection, but could be wrapped explicitly) and use a prepared statement reused across iterations — which `invalidate_edge` does NOT do (it calls `conn.execute()` which re-prepares each time):

```rust
// db::invalidate_edge should be: self.conn.prepare_cached("UPDATE edges SET valid_to = ?1 WHERE id = ?2")
```

Also consider batch-invalidating with `WHERE id IN (...)` for the removal case.

---

### C6. `reindex.rs`: `allocate_new_doc` does a full search + N `query_row` lookups per new file

**Location:** `reindex.rs:146–195`

```rust
fn allocate_new_doc(search: &mut BrainSearch, new_doc_id: &str) -> Result<()> {
    let content: Option<String> = search.db().conn()
        .query_row("SELECT content FROM documents WHERE doc_id = ?1", ...)?;
    let response = search.search(&content, ALLOCATION_K + 1)?;  // full 4-layer search!

    for hit in response.results.iter()... {
        let excitability: f64 = match search.db().conn().query_row(
            "SELECT excitability FROM documents WHERE doc_id = ?1",  // N+1
            ...
        )
```

**Problem:** For every newly indexed file, this runs a full 4-layer fusion search (itself doing all the queries above) plus K additional `query_row` calls for excitability lookups.

The excitability N+1 is the easy fix: batch-load excitability for all `hit.doc_id`s from the search response in a single query before the loop.

The deeper issue — running a full search per new document during consolidation — is expensive but intentional (competitive allocation). At least avoid the inner N+1.

---

## Summary Table

| # | File | Issue | Severity | Fix Complexity |
|---|---|---|---|---|
| 1 | `search.rs:152` | N+1 vector fetch (30 SELECTs) | 🔴 Critical | Medium |
| 2 | `search.rs:586` | `prepare()` per word in graph_search | 🟡 Medium | Trivial |
| 3 | `search.rs:353` | Unconditional gate query per search | 🟡 Medium | Low |
| 4 | `search.rs:512` | O(n²) MMR loop | 🟡 Medium | Medium |
| 5 | `db.rs:839,855` | `prepare()` in search_by_tag/cross_ref | 🟡 Medium | Trivial |
| 6 | `search.rs:252` | 2×W queries for metadata layer | 🟡 Medium | Medium |
| 7 | `search.rs:631` | `to_lowercase()` alloc per word per neighbor | 🟢 Low | Trivial |
| 8 | `db.rs:737` | `prepare()` in get_neighbors hot path | 🟡 Medium | Trivial |
| C1 | `strengthen.rs:132` | Full table scan in decay query | 🟡 Medium | Medium |
| C2 | `prune.rs:145` | N+1 in misaligned embedding phase | 🟡 Medium | Medium |
| C3 | `reorganize.rs:94,198` | `co_retrieved` edges loaded twice | 🟡 Medium | Low |
| C4 | `reorganize.rs:215` | Per-edge UPDATE in decay loop | 🟡 Medium | Low |
| C5 | `reindex.rs:164` | N+1 excitability lookup in allocate_new_doc | 🟡 Medium | Low |

---

## Recommended Fix Order

1. **Issue #1** (vector N+1) — highest impact, single change eliminates 29 queries per search
2. **Issues #2, #5, #8** (`prepare` → `prepare_cached` in hot paths) — all trivial, do together
3. **Issue #6 + #5 combined** — replace metadata layer with 2 batched IN-list queries
4. **Issue C2** (prune N+1 JOIN) — eliminates N+1 in consolidation phase 4
5. **Issue C3** (double edge load in reorganize) — halves Phase 3 DB reads
6. **Issue #3** (gate query caching) — quick win, 1 query/search saved
7. **Issues C4, C5** (batch updates) — consolidation perf, lower urgency
8. **Issue #4** (MMR O(n²)) — accept for now unless `limit` regularly exceeds 20
