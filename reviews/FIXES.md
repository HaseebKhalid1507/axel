# Axel Consolidation — Fix List (Critical & High Only)

Synthesized from: architecture.md, performance.md, robustness.md, security.md, testing.md, ux.md  
Severity order: CRITICAL → HIGH. Within each tier, sorted by file.

---

## CRITICAL

| File | Severity | Finding |
|------|----------|---------|
| robustness.md | CRITICAL | `DateTime::parse_from_rfc3339(...).unwrap()` in `db.rs:770–771` panics on any malformed edge timestamp, crashing the entire consolidation run |
| robustness.md | CRITICAL | Phase failure aborts all remaining phases via `?` in `mod.rs`; audit log never written, so `last_consolidation_time` cursor doesn't advance and the next run double-processes |
| robustness.md | CRITICAL | No in-progress log marker: a mid-run crash leaves no `consolidation_log` row, so the next run re-processes the full backlog and double-boosts excitability / double-counts edges |

---

## HIGH

| File | Severity | Finding |
|------|----------|---------|
| architecture.md | HIGH | `consolidate()` calls `prune::prune()` (empty priority map) instead of `prune_with_priorities()`; Phase 4 auto-removal is silently dead code — no Low-priority documents are ever auto-pruned |
| architecture.md | HIGH | Default source list is duplicated verbatim in `main.rs` and `mcp.rs`; `mcp.rs` never calls `load_sources()`, so editing `sources.toml` has no effect on MCP-triggered consolidation runs |
| architecture.md | HIGH | Boost formula divergence: `strengthen.rs` uses `ln` (natural log), `memkoshi/decay.rs` uses `log2`; consolidation excitability grows ~1.44× slower than memkoshi for identical access patterns |
| performance.md | HIGH | N+1 in `strengthen.rs:59`: one `SELECT excitability` + one `UPDATE` per accessed document; 100 accessed docs = 200 serialised round-trips |
| performance.md | HIGH | N+1 in `reorganize.rs:91`: one edge lookup per co-retrieval pair using an `OR` on `source_id`/`target_id` that prevents index use; replace with PK lookup via the already-computed `coret_edge_id` |
| performance.md | HIGH | N+1 in `reorganize.rs:159`: one `COUNT(*)` query into `co_retrieval` per live edge in the decay loop; pre-load all recent pairs into a `HashSet` before the loop |
| performance.md | HIGH | Five missing indexes on consolidation-critical columns: `documents.excitability`, `documents.access_count`, `documents.last_accessed`, `co_retrieval.timestamp`, and a compound `edges(type, source_id, target_id)` — all filtered/scanned on every run |
| performance.md | HIGH | `co_retrieval` and `document_access` tables grow unbounded (no DELETE path); the misaligned-doc query in `prune.rs` scans the entire lifetime of `document_access` with no time bound |
| robustness.md | HIGH | `upsert_document` and `delete_documents_by_file` in `db.rs:425–522` execute multiple SQL statements without a wrapping transaction; a panic between statements leaves `documents` and `chunks_fts` out of sync |
| robustness.md | HIGH | `let _ = self.search.index_memory(...)` in `brain.rs:241,286` silently discards indexing errors; memory is stored in SQLite but invisible to search with no warning to the caller |
| robustness.md | HIGH | `Err(_) => continue` on `fs::read_to_string` in `reindex.rs:87–90`: unreadable files are silently skipped with no counter and no log entry, and may subsequently be pruned as "deleted" |
| robustness.md | HIGH | A single failing source in the `reindex_source` loop aborts all remaining sources; if the failing source already ran its prune step, those documents are deleted but never re-added |
| robustness.md | HIGH | Phase events only emitted under `opts.verbose`; MCP-triggered consolidation runs produce zero structured logging — hangs and failures are completely dark |
| robustness.md | HIGH | An embedding error on file N in `reindex_source` propagates via `?`, leaving files N+1 through end-of-batch unprocessed with no indication in stats or logs |
| security.md | HIGH | `axel_search` MCP tool accepts unbounded `limit` parameter; `limit=10_000_000` requests 30M HNSW candidates, causing OOM; no server-side cap anywhere in the call stack |
| ux.md | HIGH | `axel_consolidate` MCP handler hardcodes the default source list instead of calling `load_sources()`; editing `sources.toml` is silently ignored for all MCP-triggered consolidation runs (duplicates architecture finding — same root cause, fix once) |
| ux.md | HIGH | Prune phase flagged-for-review candidates are not printed anywhere; the output shows only a count, making the "flag for human review" safety mechanism non-actionable |

---

## Notes

- The **`prune()` / `load_sources()` duplication** is flagged by both architecture.md and ux.md; it is one fix.
- The **`co_retrieval` unbounded growth** fix (add a bounded `DELETE` after `reorganize()`) also resolves the full-history scan in `prune.rs` misalignment query.
- **Transaction wrapping** (performance.md Finding 9, robustness.md Finding 5b) overlap: both call for `BEGIN`/`COMMIT` around batch writes in `upsert_document`, `strengthen`, and `reorganize`; fix together.
- Testing.md findings were excluded (no tests = not a bug/corruption/security risk in itself); the regression test specifications there should be consulted when implementing the fixes above.
