# Axel — Test Coverage Gap Analysis

**Scope:** 22 tests in `consolidation_tests.rs` + 5 in `brain_tests.rs` vs. full implementation  
**Files analysed:** `consolidate/{mod,reindex,strengthen,reorganize,prune}.rs`, `velocirag/src/search.rs`, `brain.rs`  
**Method:** Manual path-tracing. Every function mapped to existing test, or flagged uncovered.

---

## Coverage Map (what IS tested)

| Area | Tests | Coverage quality |
|---|---|---|
| `strengthen()` boost/decay/extinction | 5, 6, 7 | Solid — happy path only |
| Excitability floor/ceiling clamps | 11, 12 | Good |
| `log_document_access` round-trip | 2 | Basic |
| `log_co_retrieval` canonical order / self-noop | 3, 4 | Good |
| `increment_document_access` | 4, 14 | Good |
| Timestamp normalisation (RFC3339 variants) | 8 | Good |
| `consolidation_log` round-trip + in-progress skip | 9, 15 | Good |
| Table cleanup DELETE (90-day) | 10, 20, 21 | Good |
| `coret_edge_id` order-independence + determinism | 13, 22 | Good |
| `prune_with_priorities` stale query filter | 19 | DB-level only |
| `co_retrieval` aggregate threshold | 18 | DB-level only |
| `get_memory_with_verification` | brain_tests 1, 2 | Basic |
| `update_memory` content/importance/nonexistent | brain_tests 3–5 | Good |
| CLI `--limit` cap behaviour | 16 | Source-scan only |

---

## Critical Gaps (zero coverage, ordered by risk)

---

### 1. `reindex_source()` — entire Phase 1 is untested  
**Risk: HIGH**

`reindex.rs` has ~140 lines of logic: directory walk, mtime delta detection, `doc_id` construction, `index_document` call, stale-file pruning, and the competitive allocation (`allocate_new_doc`). None of this has a single test.

**What can break silently:**
- `mtime > indexed_at + 0.5` tolerance — off-by-one causes either missed updates or infinite re-indexing
- `doc_id` construction (`source::relative::path`) — any `/` → `::` conversion bug silently creates orphan entries that can never be pruned
- Stale-file pruning: `seen_on_disk` miss → docs for deleted files live forever in the index
- `allocate_new_doc` CREB-analog edges — never exercised; edge creation failure is currently swallowed via `tracing::warn!`

**Test surface available without ONNX:** All of this is testable with `TempDir` + real `.md` files using `fake_emb()` already in scope. The `allocate_new_doc` path can be gated by pre-inserting a high-excitability neighbor.

---

### 2. `reorganize()` end-to-end — Phase 3 execution path  
**Risk: HIGH**

Test 18 only validates the raw SQL aggregate query. The actual `reorganize()` function — edge upsert, edge weight bump on revisit, edge decay, edge removal below `EDGE_REMOVAL_THRESHOLD (0.2)` — has zero execution coverage.

**What can break silently:**
- Edge upsert when nodes don't exist yet (the `upsert_node` pre-flight before `insert_edge`)
- Weight bump cap at `1.0` — never verified
- Edge decay: `weight * 0.8` per pass, removal at `< 0.2` — the feedback loop that keeps the graph clean is unvalidated
- The `dry_run` branch counting `edges_updated` / `edges_added` without writes — could silently miscount
- `since` timestamp normalisation in `reorganize()` (strips `T` and `+offset`) — distinct from the one in `strengthen()` and untested

**Minimum test:** insert two docs, log 4 co-retrievals, call `reorganize()`, assert `edges_added == 1`. Then call again without new events and assert `edges_decayed` increments.

---

### 3. `prune_with_priorities()` auto-remove (Low-priority source path)  
**Risk: HIGH**

Test 19 validates the candidate-selection query only. The branch that actually **deletes** a document (`delete_documents_by_file` / direct `DELETE`) when source priority is `Low` has never been executed in a test. 

**What can break silently:**
- `source_of(doc_id)` prefix extraction — if `doc_id` lacks `::`, the whole source resolution returns the full `doc_id` as the source name, never matching any priority key; stale Low-priority docs are flagged instead of removed
- `delete_documents_by_file` cascade — cascades chunks/edges; a schema mismatch would silently leave orphaned rows
- `misaligned_embedding` detection path (`hits >= 5 AND avg_score < 0.015`) — zero test coverage; this is the "Arc analog" the spec emphasises

**Minimum test:** insert a doc with `file_path`, backdate it, set excitability to `0.12`, call `prune_with_priorities` with `{"slack-diary": Priority::Low}`, assert `stats.removed == 1` and the doc is gone from the DB.

---

### 4. `consolidate()` orchestration (mod.rs)  
**Risk: MEDIUM-HIGH**

The top-level `consolidate()` function — phase sequencing, partial-failure handling, the in-progress log marker, and the audit log write — is never called in tests. Individual phases are tested in isolation, but the wiring between them is not.

**Specific untested paths:**
- Phase error recovery: a failing phase sets `partial_failure = true` and continues; this resilience is architectural but unverified
- `start_consolidation_log` / `update_consolidation_log` round-trip (the two-step "started → finished" audit trail) — only single-insert `insert_consolidation_log` is tested
- `wants()` gate: when `phases` is non-empty, only selected phases run — no test exercises a subset invocation
- The 90-day retention DELETE **inside** `consolidate()` — tests 10/20/21 run the DELETE directly, bypassing the `!dry_run` guard in `consolidate()`; the guard itself is untested

---

### 5. `boot_context()` hot-doc injection  
**Risk: MEDIUM**

`boot_context()` in `brain.rs` has a second path: after loading memories, it queries `documents WHERE excitability > 0.6 ORDER BY excitability DESC LIMIT 3` and injects them as `InjectionEntry { category: "hot_document" }`. This is the consolidation feedback loop completing — hot docs surface in the system prompt.

Zero tests cover:
- Hot doc injection when docs with `excitability > 0.6` exist
- `seen_memory_ids` dedup — a doc already in `seen_memory_ids` must NOT be re-injected (the guard `if !self.seen_memory_ids.contains(&row.0)` is untested)
- `contextual_recall()` — the mid-session injection path — is never tested at all

---

### 6. `strengthen()` dry-run and multi-doc batch path  
**Risk: MEDIUM**

All five `strengthen()` tests pass `dry_run: false`. The `dry_run: true` path has separate counting logic (`stats.decayed += updates.len()` instead of the transaction path). It's reachable, distinct from the live path, and untested.

Additionally: all existing tests use a single document. The bulk-excitability pre-load path (chunked IN clause, 500-doc limit) that was specifically added to kill N+1 queries is never exercised with multiple documents.

---

### 7. `SearchEngine` new layers: excitability boost, graph boost (spreading activation), MMR diversity  
**Risk: MEDIUM**

`velocirag/src/search.rs` contains three post-fusion stages added after the original vector+BM25 design:

| Stage | What it does | Test coverage |
|---|---|---|
| Excitability boost | Batch-loads excitability + decay factor, scales `rrf_score` by `0.9–1.1×` | **None** |
| Graph boost (spreading activation) | Propagates relevance along `co_retrieved` edges to neighbors | **None** |
| MMR diversity | Penalises near-duplicates via cosine similarity | **None** |

These are search-quality features; silent breakage would degrade result ranking without any failing test. The excitability boost directly depends on the consolidation system's output — making it the primary integration point between the two halves of Axel.

---

### 8. `load_sources()` TOML parsing  
**Risk: LOW-MEDIUM**

`load_sources()` has three resolution paths: override path → `AXEL_SOURCES` env → `~/.config/axel/sources.toml` → fallback. The TOML parsing branch (`content.parse::<toml::Value>()`) — including the `~` expansion, empty-result fallback, and priority string mapping — is untested. A malformed sources.toml falls back to `default_sources()` silently; wrong priority string maps to `Medium` silently.

---

## Summary Table

| Gap | Risk | Lines affected | Testable without infra? |
|---|---|---|---|
| `reindex_source()` + `allocate_new_doc` | 🔴 HIGH | ~140 | Yes — TempDir + fake_emb |
| `reorganize()` edge lifecycle | 🔴 HIGH | ~90 | Yes — DB-only |
| `prune_with_priorities` auto-remove + misaligned path | 🔴 HIGH | ~80 | Yes — DB-only |
| `consolidate()` orchestration + audit log | 🟠 MEDIUM-HIGH | ~140 | Yes — DB-only |
| `boot_context` hot-doc injection + dedup | 🟠 MEDIUM | ~30 | Yes — DB + AxelBrain |
| `strengthen()` dry-run + multi-doc bulk path | 🟠 MEDIUM | ~20 | Yes — existing harness |
| Search engine post-fusion stages | 🟠 MEDIUM | ~150 | Partial — needs ONNX or mock |
| `load_sources()` TOML parsing | 🟡 LOW-MEDIUM | ~35 | Yes — TempDir + file write |

---

## Recommended Test Order

1. **`reorganize()` end-to-end** — highest bang-for-buck; pure DB, no ONNX needed, covers the entire Phase 3 execution path in ~20 lines
2. **`prune_with_priorities` auto-remove** — one test, proves the deletion branch and `source_of()` logic
3. **`reindex_source()` delta detection + pruning** — needs TempDir + real files; covers the most code with the least mock setup
4. **`consolidate()` full pass** — integration test wiring all four phases; exposes phase-ordering bugs
5. **`boot_context()` hot-doc injection** — unit test for the consolidation→prompt feedback loop
6. **`strengthen()` dry-run + 500+ doc bulk path** — quick; uses existing harness

---

## Watch Next

The `consolidation_tests.rs` tests pin DB-layer primitives well. The gap is entirely at the **function level** — every `pub fn` in the four phase modules has some internal logic that's exercised only indirectly (or not at all). The risk is not in the primitives; it's in the orchestration. A bug in `reindex_source` doc_id construction, for example, would produce valid-looking DB rows that silently fail to prune — detectable only if you test the full round-trip.

The search engine post-fusion stages are the secondary concern: they affect ranking quality, not correctness, so breakage is harder to notice without a ranking-quality assertion.
