# FIXES — Critical & High Severity Only

> Extracted from `architecture.md`, `performance.md`, `research.md`. One finding per line.

---

## 🔴 Critical

- **[performance.md #1]** `search.rs:152` — N+1 vector fetch: 30 individual `SELECT` statements fired per search call (one per vector hit); batch into a single `WHERE id IN (...)` query. *(30 round-trips → 1; highest-impact fix available)*

---

## 🔴 High

- **[architecture.md §2a]** `main.rs:604`, `mcp.rs:188` — Post-search access logging (`log_document_access`, `increment_document_access`, `log_co_retrieval`) duplicated verbatim in CLI and MCP; divergence risk on every signature change. Move to `BrainSearch::record_search_feedback()`.
- **[architecture.md §4]** `db.rs`, `embedder.rs`, `r8.rs`, tests — `EMBEDDING_DIM = 384` defined four times; a one-miss model change silently corrupts stored vector blobs with no error. Consolidate to a single definition in `velocirag/src/constants.rs`.
- **[architecture.md §1]** `search.rs:125–568` — `SearchEngine::search()` is 443 lines (58% of the file) doing ten distinct algorithmic jobs; must be decomposed into private methods (`retrieve_vector`, `retrieve_keyword`, `retrieve_graph`, `retrieve_metadata`, `apply_excitability_boost`, `apply_graph_boost`, `apply_mmr`) for testability and maintainability.
- **[research.md §4]** `search.rs:183–190` — Query expansion fires unconditionally on the top result regardless of relevance score; PRF with no confidence gate is the highest-risk precision regression path. Add `if results_lists[0][0].score > 0.75` guard before expansion.
- **[research.md §2]** `search.rs:322`, `strengthen.rs:162` — Two independent, uncoordinated decay systems operate on the same `excitability` field (search-time hardcodes 60-day half-life; consolidation uses dynamic `S_BASE=30`); produces non-monotonic excitability behavior for low-access documents. Unify under a single `STABILITY_BASE_DAYS` constant.
- **[research.md §2]** `strengthen.rs:95` — `S_BASE=30` days is calibrated for Ebbinghaus nonsense-syllable experiments, not long-form knowledge notes; causes a 500-note cold-start corpus to decay to ~5% retention within 90 days. Raise to 60–90 days or add a new-document protection window.
- **[research.md §1]** `search.rs:~451` — MMR activation guard `fused.len() > opts.limit` is off-by-one; when RRF returns exactly `limit` results, MMR is silently skipped and near-duplicate results are never diversified. Fix to `>= opts.limit` or run MMR unconditionally.
