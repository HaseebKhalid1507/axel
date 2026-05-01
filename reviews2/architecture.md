# Architecture Review: Search + Consolidation System
**Scope:** `velocirag/src/search.rs`, `axel/src/consolidate/mod.rs`, `axel/src/main.rs`, `axel/src/mcp.rs`
**Date:** 2025-07

---

## 1. Is `search()` Too Long? Should Layers Be Extracted?

### Verdict: Yes — it has grown past the point of comfortable comprehension.

`SearchEngine::search()` runs from line 125 to line 568 — **443 lines** in a single method body. The file
itself is 764 lines total, meaning that one function accounts for 58% of the entire file.

### What's Inside That Single Function

The method does six distinct algorithmic jobs:

| Job | Lines | Description |
|-----|-------|-------------|
| Vector layer | ~33 | Embed query, HNSW search, DB join for doc rows |
| Query expansion (PRF) | ~15 | Extract top terms from top vector hit, rewrite query |
| Keyword layer | ~25 | BM25/FTS5 search, score normalisation |
| Graph layer | ~9 | Delegates to `graph_search()` helper |
| Metadata layer | ~42 | Tag + cross-ref matching, dedup with HashSet |
| RRF fusion | ~3 | Delegates to `rrf::reciprocal_rank_fusion` |
| Excitability boost | ~43 | Dynamic SQL with positional placeholders, Ebbinghaus decay, recency boost |
| Graph boost (spreading activation) | ~95 | `has_coret` gate, parent snapshot, fwd+rev neighbor queries, boost accumulator, optional candidate injection |
| MMR diversity | ~90 | Batch embedding load, cosine fn defined inside the method, greedy selection loop |
| Rerank | ~18 | Delegates to `rerank()` helper |

The graph boost block alone (lines ~347–446) contains three nested `HashMap` allocations, two
`prepare_cached` statements, a borrow-juggling comment (`// drop cached stmts so we can re-borrow conn`),
and an inner loop that conditionally pushes new `FusedResult` entries into the vector it is iterating
over (via an index map). This is the hardest section to reason about in the entire codebase.

### What Should Be Extracted

Each of the first five retrievers is already a conceptual layer. They should be private methods:

```
fn retrieve_vector(&mut self, query: &str, limit: usize) -> Result<(Vec<RankedResult>, f64)>
fn retrieve_keyword(&self, query: &str, limit: usize) -> Result<(Vec<RankedResult>, f64)>
fn retrieve_graph(&self, query: &str) -> Result<(Vec<RankedResult>, f64)>
fn retrieve_metadata(&self, query: &str) -> Result<(Vec<RankedResult>, f64)>
fn apply_excitability_boost(&self, fused: &mut Vec<FusedResult>) -> Result<()>
fn apply_graph_boost(&self, fused: &mut Vec<FusedResult>) -> Result<()>
fn apply_mmr(&self, fused: Vec<FusedResult>, limit: usize) -> Vec<FusedResult>
```

`search()` then becomes a ~60-line orchestration function whose shape matches the comments already in
the code. Readability, testability, and future layer additions all improve.

### Minor but Noteworthy: `cosine` defined inside `search()`

```rust
fn cosine(a: &[f32], b: &[f32]) -> f32 { ... }
```

This inner function is defined on line 478 inside the body of `search()`. It belongs in a module-level
`mod math` or alongside the existing `rrf` module. As written, it is invisible to any future caller
who needs cosine similarity (and would likely duplicate it).

---

## 2. Code Duplication Between CLI and MCP

### Verdict: Significant — three distinct blocks are copy-pasted verbatim.

#### 2a. Post-search access logging (most critical duplicate)

`cmd_search` in `main.rs` (lines 604–614):

```rust
let db = search.db();
for r in &response.results {
    let _ = db.log_document_access(&r.doc_id, "search_hit", Some(&query), Some(r.score), None);
    let _ = db.increment_document_access(&r.doc_id);
}
let top_ids: Vec<&str> = response.results.iter().take(5).map(|r| r.doc_id.as_str()).collect();
for i in 0..top_ids.len() {
    for j in (i+1)..top_ids.len() {
        let _ = db.log_co_retrieval(top_ids[i], top_ids[j], &query);
    }
}
```

`axel_search` handler in `mcp.rs` (lines 188–207) — **identical logic**, different variable names. These
are 15 lines of feedback-loop logic that need to stay in sync. If the `take(5)` threshold, the event
name `"search_hit"`, or the `log_co_retrieval` call signature changes, both files must be updated
independently.

**Fix:** Move into `BrainSearch` as a method:

```rust
impl BrainSearch {
    pub fn record_search_feedback(&self, results: &[SearchResult], query: &str) { ... }
}
```

Both callers collapse to one line.

#### 2b. Consolidation output formatting

`cmd_consolidate` (main.rs ~lines 464–479) and `axel_consolidate` handler (mcp.rs lines 398–412) both
format a `ConsolidateStats` summary. The field names match, the structure matches, the numbers are the
same — but one writes to stdout with `println!` and the other builds a `String`. They will diverge when
a new phase field is added.

**Fix:** Add a `Display` impl or `fn summarise(stats: &ConsolidateStats) -> String` on `ConsolidateStats`
in `consolidate/mod.rs`. Each formatter calls that. The CLI pretty-prints it; the MCP wraps it in
`tool_text`.

#### 2c. Phase-string → `HashSet<Phase>` parsing

`cmd_consolidate` (main.rs lines 411–417):

```rust
let phases = match phase {
    Some("reindex") => [Phase::Reindex].into_iter().collect(),
    ...
    _ => std::collections::HashSet::new(),
};
```

`axel_consolidate` handler (mcp.rs lines 385–391) — same match, same arms, same semantics. Should be
`impl FromStr for Phase` or a free function `parse_phases(s: Option<&str>) -> HashSet<Phase>` in
`consolidate/mod.rs`.

---

## 3. Tight Coupling Between Components

### 3a. `search.rs` reaches directly into the DB schema

The excitability boost block (lines 298–340) contains a raw 4-column SQL query with `julianday`
arithmetic built directly into `search.rs`:

```rust
let sql = format!(
    "SELECT doc_id, COALESCE(excitability, 0.5),
            COALESCE((julianday('now') - julianday(last_accessed)), 0),
            COALESCE((julianday('now') - julianday(indexed_at)), 30)
     FROM documents WHERE doc_id IN ({})",
    ...
);
```

This means `search.rs` knows the `documents` table's column names and temporal semantics. The graph
boost block (lines 350–446) similarly contains raw SQL over `edges` with `valid_to` and `type =
'co_retrieved'` hardcoded. `Database` already encapsulates the schema — these queries should live
there as named methods (`fn get_excitabilities(&self, doc_ids: &[&str]) -> Result<HashMap<String,
ExcitabilityRecord>>`, `fn get_co_retrieval_neighbors(...)`), not inline in the search pipeline.

The borrow juggling comment at line 411 — `// drop cached stmts so we can re-borrow conn` — is a
direct symptom of this: the `Connection` is being borrowed through the graph-boost block in two
different ways because the queries aren't behind a method boundary.

### 3b. `main.rs` bypasses `BrainSearch` to query the DB

`cmd_excitability` (lines 1006–1107) opens `Brain::open()` and calls `brain.conn()` directly to run
raw SQL. Similarly `cmd_stats` (lines 831–926) does five separate `conn.query_row()` calls.
`cmd_suggest` (lines 929–1000) calls `search.db().conn()` to build its own prepared statement.

There is an architectural boundary implied by `BrainSearch` — it is supposed to be the gateway.
When CLI commands bypass it, that boundary erodes. If the schema changes (e.g., `excitability`
column is renamed or the edge validity model changes), fixes must be applied in the DB module
*and* the CLI *and* any MCP handler that also does direct queries.

### 3c. Consolidation `mod.rs` performs retention cleanup inline

Lines 221–231 of `consolidate/mod.rs`:

```rust
let conn = search.db().conn();
let _ = conn.execute("DELETE FROM co_retrieval WHERE timestamp < datetime('now', '-90 days')", []);
let _ = conn.execute("DELETE FROM document_access WHERE timestamp < datetime('now', '-90 days')", []);
```

This is schema-level SQL (with the 90-day magic number hardcoded) executing in the orchestration
layer. The `Database` struct should own a `fn prune_old_events(&self) -> Result<usize>` method.
The consolidation orchestrator calls it; the constant and the SQL live where the schema lives.

---

## 4. Constants: Scattered vs. Centralised

### Current State

Constants are spread across at least 8 files with no single module of record:

| Constant | File | Notes |
|----------|------|-------|
| `EMBEDDING_DIM = 384` | `velocirag/src/db.rs` line 23 | Schema-level |
| `EMBEDDING_DIM = 384` | `velocirag/src/embedder.rs` line 21 | **Duplicate** |
| `EMBEDDING_DIM: usize = 384` | `axel/src/r8.rs` line 42 | **Third copy** |
| `EMBEDDING_DIM: usize = 384` | `axel/tests/consolidation_tests.rs` line 20 | **Fourth copy** |
| `DEFAULT_EMBEDDER = "all-MiniLM-L6-v2"` | `axel/src/r8.rs` line 39 | |
| `MODEL_NAME = "all-MiniLM-L6-v2"` | `velocirag/src/embedder.rs` line 19 | **Same thing, different name** |
| `GRAPH_BOOST_FACTOR = 0.3` | `velocirag/src/search.rs` line 347 | Inside fn body |
| `GRAPH_SOURCES_TOPK = 5` | `velocirag/src/search.rs` line 348 | Inside fn body |
| `LAMBDA = 0.7` | `velocirag/src/search.rs` line 452 | Inside fn body |
| `DEFAULT_RRF_K = 60` | `velocirag/src/search.rs` line 26 | Module-level |
| `CO_RETRIEVAL_THRESHOLD = 3` | `axel/src/consolidate/reorganize.rs` line 36 | |
| `EDGE_REMOVAL_THRESHOLD = 0.2` | `axel/src/consolidate/reorganize.rs` line 37 | |

`EMBEDDING_DIM` being defined four times is the most dangerous instance. If the model changes (e.g.,
to `all-MiniLM-L12-v2` at 384d or `bge-small-en` at 384d or a future 768d model), it must be updated
in four places. One miss produces a silent mismatch between the stored binary blob size and the
expected vector dimension, which corrupts search results with no error.

`GRAPH_BOOST_FACTOR`, `GRAPH_SOURCES_TOPK`, and `LAMBDA` being defined inside the function body
means they cannot be overridden, referenced from tests, or surfaced in documentation without reading
443 lines of code to find them.

### Recommendation

Two files should own all tuneable constants:

- `velocirag/src/constants.rs` — model dimensions, search defaults, algorithm hyperparameters
- `axel/src/constants.rs` — retention windows, CLI limits, tier budgets

`EMBEDDING_DIM` should appear exactly once, in `velocirag/src/constants.rs`, and be `pub use`'d by
everything that needs it. The three algorithm constants currently buried in `search()` should move to
the module-level constants block at the top of `search.rs`.

---

## 5. Is the Feature Ordering in `search()` Correct?

### Current Order
```
1. Vector search
2. Query expansion (PRF from top vector hit)
3. Keyword search (using expanded query)
4. Graph layer (knowledge graph traversal)
5. Metadata layer (tags + cross-refs)
6. RRF fusion
7. Excitability boost
8. Graph boost (spreading activation via co-retrieval edges)
9. MMR diversity
10. Rerank
```

### Assessment: Mostly correct. One sequencing issue and one comment mismatch.

**What is correct:**

The fundamental ordering — retrieval → fusion → post-fusion scoring → diversity → rerank — is
sound. Retrieval stages must all run before RRF so the fusion has all candidates. Excitability boost
must follow RRF because it operates on the fused score. Graph boost must follow excitability because
it reads the final `rrf_score` to compute `parent_score * weight * GRAPH_BOOST_FACTOR`. MMR must
follow graph boost because it needs the final relevance scores to compute the λ·rel term. Rerank
must be last because it's the most expensive operation and should operate on an already-trimmed,
already-diversified candidate list.

**The sequencing issue: query expansion is between vector and keyword, not before both.**

Query expansion at position 2 is *intentionally* placed after vector search because it performs
Pseudo-Relevance Feedback: it pulls terms from the top vector hit. This is documented in the
comment at line 181. However, the file-level module docstring (lines 1–11) describes a "four-layer
fusion" without mentioning query expansion at all. The docstring should be updated to reflect the
actual pipeline:

```
1. Vector similarity → seed for PRF
2. Query expansion (PRF)
3. BM25 keyword (with expanded query)
4. Knowledge graph traversal
5. Metadata (tags + cross-refs)
6. RRF fusion
7. Excitability + recency boost
8. Spreading activation (graph boost)
9. MMR diversity
10. Cross-encoder rerank
```

**The comment mismatch at line 1:**

The header says "Four-layer fusion" but the pipeline now has five retrieval layers (vector, keyword,
graph, metadata, and graph boost). The docstring was accurate when written but has been outpaced
by the implementation. This is a minor issue but misleads anyone reading the file top-down.

**The metadata layer has no feature flag.**

Layers 1, 2, and 3 are gated by `opts.layers.vector`, `opts.layers.keyword`, and `opts.layers.graph`
respectively. The metadata layer (tags + cross-refs) always runs unconditionally — there is no
`opts.layers.metadata`. This is inconsistent. If a caller wants a pure vector search (e.g., for
benchmarking or debugging), they can disable keyword and graph but cannot disable metadata. Add
`metadata: bool` to `SearchLayers`.

---

## 6. Additional Observations

### 6a. `extract_top_terms` stopword list is frozen in code

The stopword list at line 737 (58 words) is module-level but private and hardcoded. For a system
whose purpose is indexing personal knowledge, domain-specific stopwords (e.g., "haseeb", "jawz",
"note", "see") are common and currently unsuppressable. This is not a critical flaw, but the list
should be extensible.

### 6b. The positional SQL placeholder generation is repeated twice

The pattern of building `?1, ?2, ?3, ...` strings appears in both the excitability boost block (line
299) and the MMR block (line 455):

```rust
let placeholders: Vec<String> = (0..fused.len()).map(|i| format!("?{}", i + 1)).collect();
let sql = format!("... WHERE doc_id IN ({})", placeholders.join(","));
```

This is a SQLite-specific idiom for binding variable-length lists. It belongs in a `Database`
helper method (`fn ids_in_clause(ids: &[&str]) -> String` or simply as a query method that accepts
a slice).

### 6c. `cmd_excitability` is purely diagnostic but queries raw schema

`cmd_excitability` in `main.rs` (lines 1006–1107) is 101 lines of SQL and terminal formatting logic
that lives in `main.rs`. This is the kind of command that could evolve (add percentiles, add
per-source breakdown, add temporal trends) and every evolution requires touching the CLI file.
A `BrainSearch::excitability_report() -> Result<ExcitabilityReport>` struct pattern would separate
the data retrieval from the display.

### 6d. MCP `axel_consolidate` silently omits `verbose` and `new_files`

The CLI output for consolidation includes:
- `new_files` in the reindex phase summary
- per-candidate prune detail when `verbose` is set

The MCP handler's format string (mcp.rs lines 402–410) omits both. If Jawz calls
`axel_consolidate` over MCP and wants to know how many new files were picked up, that information
is lost. The MCP response should either use the same `summarise()` function (see §2b) or expose
`new_files` explicitly.

---

## Summary

| Issue | Severity | File(s) |
|-------|----------|---------|
| `search()` is 443 lines — retrieval layers should be private methods | High | `search.rs` |
| Post-search access logging duplicated CLI ↔ MCP | High | `main.rs`, `mcp.rs` |
| Phase-string parsing duplicated CLI ↔ MCP | Medium | `main.rs`, `mcp.rs` |
| `EMBEDDING_DIM` defined four times | High | `db.rs`, `embedder.rs`, `r8.rs`, tests |
| `cosine()` defined inside function body | Medium | `search.rs` |
| Algorithm constants (`LAMBDA`, `GRAPH_BOOST_FACTOR`, etc.) buried in fn body | Medium | `search.rs` |
| Module docstring says "four-layer" — pipeline is now ten stages | Low | `search.rs` |
| Raw SQL with schema knowledge in `search.rs` (excitability + graph boost) | Medium | `search.rs`, `db.rs` |
| CLI commands bypass `BrainSearch` to query DB directly | Medium | `main.rs` |
| Inline retention cleanup SQL in consolidation orchestrator | Low | `consolidate/mod.rs` |
| Metadata layer has no feature flag in `SearchLayers` | Low | `search.rs` |
| Consolidation summary formatting duplicated CLI ↔ MCP | Medium | `main.rs`, `mcp.rs` |
| MCP consolidate response missing `new_files` field | Low | `mcp.rs` |
