# Architecture Review: Axel Consolidation System

**Reviewed by:** Chrollo (subagent)
**Date:** 2026-05-01
**Scope:** Spec compliance, module structure, API design, extensibility, missing abstractions, technical debt

---

## Files Retrieved

| Path | What's There |
|------|-------------|
| `docs/CONSOLIDATION.md` | Full spec — biology basis, schema changes, phase algorithms, CLI interface, implementation order |
| `crates/axel/src/consolidate/mod.rs` | Orchestrator — `Phase`, `SourceDir`, `Priority`, `ConsolidateOptions`, `ConsolidateStats`, `consolidate()` entry point |
| `crates/axel/src/consolidate/reindex.rs` | Phase 1 — walkdir, mtime compare, index_document, prune deleted, competitive allocation |
| `crates/axel/src/consolidate/strengthen.rs` | Phase 2 — access log grouping, boost/extinction math, grace-period decay |
| `crates/axel/src/consolidate/reorganize.rs` | Phase 3 — co_retrieval aggregation, edge upsert, stale edge decay |
| `crates/axel/src/consolidate/prune.rs` | Phase 4 — stale doc flagging, auto-removal for Low-priority, misalignment detection |
| `crates/velocirag/src/db.rs` (selected sections) | `document_access`, `co_retrieval`, `consolidation_log` DDL; migration guards; `log_document_access`, `increment_document_access`, `log_co_retrieval`, `get_document_accesses_since`, `last_consolidation_time`, `insert_consolidation_log` |
| `crates/axel/src/mcp.rs` (selected sections) | `axel_consolidate` tool registration + handler; search-hit logging inside `axel_search` |
| `crates/axel/src/main.rs` (selected sections) | `Consolidate` subcommand, `load_sources()`, `cmd_consolidate()`, `cmd_consolidate_history()`, `cmd_index_sync()` |
| `crates/memkoshi/src/decay.rs` | Parallel decay/boost implementation for memories |

---

## 1. Spec Compliance

### ✅ What Is Implemented

| Spec Feature | Status | Notes |
|---|---|---|
| Four-phase orchestrator | ✅ Implemented | `mod.rs` runs all four in order with `wants()` guard |
| Priority-ordered reindex | ✅ Implemented | Sorted by `High=0, Medium=1, Low=2` before loop |
| Competitive allocation (CREB analog) | ✅ Implemented | `allocate_new_doc()` in `reindex.rs`, threshold=0.6, K=5 |
| Access log → excitability boost | ✅ Implemented | `strengthen.rs` groups events, applies log-scaled boost |
| Extinction signal for low-score hits | ✅ Implemented | avg_score < 0.02 → subtract 0.05 |
| Grace-period decay | ✅ Implemented | 14-day grace, DECAY_RATE_PER_WEEK=0.02 |
| Co-retrieval logging in axel_search | ✅ Implemented | Top-5 pairs, canonical ordering in `log_co_retrieval` |
| Co-retrieval → edge creation/update | ✅ Implemented | Threshold=3, bump=count/10.0, soft confidence=0.7 |
| Stale edge decay | ✅ Implemented | STALE_DAYS=30, factor=0.8, removal below 0.2 |
| Edge invalidation (valid_to) not deletion | ✅ Implemented | `db.invalidate_edge()` used throughout |
| Auto-remove criteria (Low priority only) | ✅ Implemented | `prune_with_priorities()` correctly guards on Low |
| Misalignment detection | ✅ Implemented | ≥5 hits, avg_score < 0.015 |
| Audit log (consolidation_log) | ✅ Implemented | Written at end of pass, skipped on dry_run |
| `--dry-run` flag | ✅ Implemented | All phases check dry_run before writes |
| `--history` flag | ✅ Implemented | `cmd_consolidate_history()` with last-20 display |
| `axel_consolidate` MCP tool | ✅ Implemented | Tool registration + handler in `mcp.rs` |
| Source config via TOML | ✅ Implemented | `load_sources()` with 4-level fallback chain |
| Schema migrations | ✅ Implemented | PRAGMA-guarded ALTER TABLE for all new columns |
| `--phase` single-phase flag | ✅ Implemented | CLI and MCP both support it |

### ❌ Spec Features Missing from Implementation

**1. Context prefix embedding (Phase 1, spec §"Re-embed with context prefix")**

The spec calls for prepending `[source:mikoshi][path:3. Learning/...]` to content before embedding. This is described as a "grid cell orthogonalization" feature — pushing similar documents from different sources apart in embedding space. **Not present in `reindex.rs`.** `index_document()` receives raw content with no prefix.

*Impact:* Medium. Documents from different sources with overlapping content will cluster more tightly than intended. Co-retrieval signal will be noisier. The spec itself flags this in Open Questions as needing a one-time re-embed, which may be why it was deferred — but the gap should be explicit.

**2. Interleaved (round-robin) processing in Phase 3 (spec §"Interleaving")**

The spec says Phase 3 should process co-retrieval pairs in round-robin order across source namespaces (the Kudrimoti principle) to prevent graph domination by the largest source. **`reorganize.rs` processes pairs sequentially**, in the order SQLite returns them from `GROUP BY`. No interleaving is implemented.

*Impact:* Low in current state (2,238 docs, modest co-retrieval volume). Will become a correctness concern if one source dominates access patterns — e.g., mikoshi generating 10× more pairs than all other sources combined.

**3. `prune` does not receive source priorities from `opts.sources`**

The spec requires Phase 4 to use source priorities: Low-priority sources are auto-removed, High/Medium are flagged. The implementation has two entry points:

```rust
// consolidate/mod.rs — called from the orchestrator:
stats.prune = prune::prune(search, opts.dry_run)?;
// This creates an empty priorities HashMap → nothing is ever auto-removed.

// The full version that would work:
prune::prune_with_priorities(search, &priorities, opts.dry_run)
```

`prune_with_priorities()` exists and is correct. But `consolidate()` calls the stub `prune()` which passes an empty `HashMap<String, Priority>` — so the auto-remove path is **effectively dead code** in both CLI and MCP invocations. The only way to get actual deletion is to call `prune_with_priorities()` directly, which nothing does.

*Impact:* High. The spec's safety guarantee ("Low-priority sources are auto-pruned, High are only flagged") cannot trigger at all. Phase 4 auto-removal is silently disabled.

**4. Shared decay math (spec §Implementation Order, Step 3)**

The spec explicitly says: *"Port decay math from `memkoshi/decay.rs` into shared utility."* The math is identical:

| Parameter | `memkoshi/decay.rs` | `consolidate/strengthen.rs` |
|-----------|--------------------|-----------------------------|
| Boost formula | `0.05 * log2(count + 1)`, cap 0.2 | `0.05 * ln(count + 1)`, cap 0.2 |
| Decay rate | 0.02/week, cap 0.3, floor 0.1 | 0.02/week, cap 0.3, floor 0.1 |
| Grace period | 14 days | 14 days |

The math was duplicated, not unified. There is also a subtle **divergence in the boost formula**: memkoshi uses `log2`, consolidation uses `ln` (natural log). These are not equivalent — ln grows ~1.44× slower than log2. The spec says *"Matches memkoshi decay.rs"* for BOOST_SCALE — implying log2 was intended.

*Impact:* Low for correctness (both are monotonic, similar shape), but represents the exact kind of drift the spec warned against. If the formula is tuned in one place, the other silently diverges.

**5. Verbose per-document output**

The spec shows verbose output with per-document detail lines. The `verbose` flag is threaded through `ConsolidateOptions` and the orchestrator uses it for phase progress messages (`⟳ reindex [mikoshi]`), but individual phase functions (`reindex_source`, `strengthen`, `reorganize`, `prune`) do not use `opts.verbose` to emit per-document lines. The `verbose` field is never passed down into phase functions.

*Impact:* Low. `--verbose` exists as a CLI flag but produces only slightly more output than the default. No regression — it just doesn't deliver the per-document detail the spec implies.

**6. Backup warning before first consolidation**

The spec (§Safety, point 6): *"The CLI should warn (not block) if no recent backup exists."* Not implemented.

---

## 2. Separation of Concerns

The module structure is clean and the phase boundaries are correctly drawn. A detailed assessment:

### What Works Well

**`mod.rs` is a pure orchestrator.** It sorts sources, calls phase functions, accumulates stats, writes the audit log. It makes no decisions about how any phase works. This is correct — the orchestrator should not know about excitability formulas or edge weights.

**Each phase owns its own stats type.** `ReindexStats`, `StrengthenStats`, `ReorganizeStats`, `PruneStats` live in their respective files. `ConsolidateStats` in `mod.rs` composes them. Clean hierarchy.

**DB access is correctly localized.** All SQL is in either `velocirag/src/db.rs` (schema + access methods) or the phase modules themselves (inline queries for phase-specific logic). No SQL leaks into `mod.rs`, `main.rs`, or `mcp.rs`.

**`SourceDir` and `Priority` are defined once in `mod.rs`.** Both CLI and MCP import them from there. No duplication of the type definitions.

### What Needs Attention

**`reindex.rs` reaches into raw SQL for two operations** that arguably belong in `db.rs`:

```rust
// In allocate_new_doc():
"SELECT content FROM documents WHERE doc_id = ?1"
"SELECT excitability FROM documents WHERE doc_id = ?1"
```

These are ad-hoc queries using `db.conn()` directly. They're not complex, but they set a precedent for bypassing the `db.rs` abstraction layer. If the `documents` schema changes, these are invisible to anyone maintaining `db.rs`.

**`strengthen.rs` uses `db.conn()` for all of its SQL.** The strengthen phase does all its reads and writes via raw `conn.prepare()` / `conn.execute()`. Only `last_consolidation_time()` and `get_document_accesses_since()` go through the DB abstraction. Queries like:

```rust
"SELECT excitability FROM documents WHERE doc_id = ?1"
"UPDATE documents SET excitability = ?1 WHERE doc_id = ?2"
```

appear in-file rather than as named methods on `Database`. This is a consistency issue — the DB layer defines `log_document_access` and `increment_document_access` as proper methods, but `update_excitability` is not.

**`reorganize.rs` raw SQL for edge operations.** Similar pattern:
```rust
"SELECT id, weight FROM edges WHERE type = 'co_retrieved' ..."
"UPDATE edges SET weight = ?1 WHERE id = ?2"
```

`db.update_edge_weight()` does not exist as a method. `db.invalidate_edge()` does exist and is correctly used. The inconsistency is: some edge operations go through the abstraction, others bypass it.

**The `prune()` / `prune_with_priorities()` split is an unclear API.** The module exports two public functions for the same operation with materially different behavior. The default `prune()` is safe but inert (no auto-removal). The "real" `prune_with_priorities()` is not called from the orchestrator. The intent appears to be a safe default for direct calls, but the result is that the orchestrator never gets full Phase 4 behavior. This should either be one function that accepts priorities (with an empty map as the safe default), or the orchestrator should build the priority map from `opts.sources`.

---

## 3. API Surface

### Public Interface Assessment

```rust
// consolidate/mod.rs — the primary public API
pub fn consolidate(search: &mut BrainSearch, opts: &ConsolidateOptions) -> Result<ConsolidateStats>

pub enum Phase { Reindex, Strengthen, Reorganize, Prune }
pub struct SourceDir { pub path, pub name, pub priority }
pub enum Priority { High, Medium, Low }
pub struct ConsolidateOptions { pub sources, pub phases, pub dry_run, pub verbose }
pub struct ConsolidateStats { pub reindex, pub strengthen, pub reorganize, pub prune, pub duration_secs }
```

**Strengths:**
- `ConsolidateOptions` is a well-designed parameter object. All behavior is controlled through it — no hidden globals.
- The `phases: HashSet<Phase>` encoding of "empty = all" is correct and idiomatic. The `wants()` helper makes the intent readable.
- Stats are structured hierarchically. Callers can inspect any phase individually.
- `Phase` derives `Hash + Eq` which is required for `HashSet` — no footgun there.

**Issues:**

**`consolidate()` takes `&mut BrainSearch` not `&mut AxelBrain`.**  
The spec (§"New public API on `AxelBrain`") defines the entry point as:
```rust
impl AxelBrain {
    pub fn consolidate(&mut self, opts: ConsolidateOptions) -> Result<ConsolidateStats>;
}
```
Instead, the implementation is a free function on `BrainSearch`. The CLI creates its own `BrainSearch::open()` and passes it directly. `AxelBrain` has no `consolidate()` method. The MCP handler accesses `brain.search_mut()` to get a `BrainSearch` to pass through. This inversion means the consolidation system is not reachable through the `AxelBrain` abstraction layer, which was the specified integration point.

**`Priority` has no `Default`.** Minor, but `SourceDir` construction always requires an explicit priority. Since all source lists include this, it's not a practical problem — but a `#[derive(Default)]` with `Medium` as the default would be ergonomic.

**`ConsolidateStats` derives `Default` but the sub-stats do too** — this is correct and allows the zero-init pattern in `consolidate()`.

**Phase sub-stats types are `pub` but their fields lack documentation.** `ReindexStats::new_files` vs `reindexed` — what's the difference? (`reindexed` = total files re-embedded including updates; `new_files` = subset that were brand new.) This distinction is not documented in the struct. The spec is clear about it; the code is not.

**`load_sources()` in `main.rs` has the right fallback chain** (explicit path → env var → `~/.config/axel/sources.toml` → hardcoded defaults). This should be a method on a `SourceConfig` type in the library crate, not a free function in `main.rs`. Currently it's not reachable from MCP, which re-implements the source list inline.

---

## 4. Extensibility

### Adding a New Phase

The current design supports adding phases, but not without changes to multiple files:

1. Add variant to `Phase` enum in `mod.rs` ✅ (one line)
2. Create `new_phase.rs` in `consolidate/` ✅ (isolated)
3. Add `pub mod new_phase` to `mod.rs` ✅
4. Add stats field to `ConsolidateStats` ✅
5. Add `wants(&opts.phases, Phase::NewPhase)` block in `consolidate()` ✅
6. Add `phase5_*` columns to `consolidation_log` in `db.rs` ⚠️ (requires schema migration)
7. Add mapping in `ConsolidationLogEntry` ⚠️
8. Update CLI `--phase` value parser in `main.rs` ⚠️
9. Update MCP `enum` in `inputSchema` ⚠️
10. Update `ConsolidationLogEntry` → `insert_consolidation_log()` ⚠️

Points 6-10 are unavoidable schema and integration concerns, but they're not abstracted in any way. The `consolidation_log` table has hardcoded columns per phase (`phase1_reindexed`, `phase2_boosted`, etc.) — a new phase requires a schema migration and DDL change. This is fine for the current scale of four phases, but it means extensibility is "possible but not smooth."

**The `Phase` enum cannot be extended from outside the crate.** It's non-exhaustive only informally (no `#[non_exhaustive]` attribute). Match arms in `main.rs` and `mcp.rs` would panic or fail to compile on an unknown phase string, rather than gracefully degrading.

### The Spec's Step-by-Step Order Is Honored

The implementation order from the spec (Schema → Reindex → Strengthen → Reorganize → Prune → Polish) was followed. Each step builds on the previous. The foundation (Step 1: schema + search logging) is solid, and the remaining phases correctly depend on it.

---

## 5. Missing Abstractions

### 5.1 Excitability Update Method

Every phase that modifies excitability does so with raw SQL:

```rust
// strengthen.rs
conn.execute("UPDATE documents SET excitability = ?1 WHERE doc_id = ?2", params![new_exc, id])?;

// reindex.rs (allocate_new_doc) — reads excitability directly from conn
conn.query_row("SELECT excitability FROM documents WHERE doc_id = ?1", ...)
```

`db.rs` should define:

```rust
pub fn get_excitability(&self, doc_id: &str) -> Result<f64>
pub fn set_excitability(&self, doc_id: &str, value: f64) -> Result<()>
```

These would centralize range clamping, make the abstraction consistent, and remove the `EXCITABILITY_FLOOR` / `EXCITABILITY_CEILING` constants from phase modules where they'd logically live in the DB layer.

### 5.2 Shared Decay Math

As noted in §1, `memkoshi/decay.rs` and `consolidate/strengthen.rs` implement the same decay curve with a subtle log-base divergence. The spec explicitly called for extraction into a shared utility. This could live in `velocirag` (since it's a pure numeric function with no dependencies) or in a new `axel-common` crate:

```rust
// Proposed: velocirag/src/decay_math.rs or common/src/decay.rs
pub fn excitability_boost(access_count: usize) -> f64 {
    (BOOST_SCALE * ((access_count as f64) + 1.0).log2()).min(BOOST_CAP)
}

pub fn excitability_decay(weeks_inactive: f64) -> f64 {
    (DECAY_RATE_PER_WEEK * weeks_inactive).min(DECAY_CAP)
}
```

The correct base is `log2` (matching memkoshi). The consolidation code uses `ln`. Fix the implementation when unifying.

### 5.3 Phase String ↔ Enum Conversion

Phase string parsing is duplicated in three places:

```rust
// main.rs cmd_consolidate():
Some("reindex") => [Phase::Reindex].into_iter().collect(),

// mcp.rs axel_consolidate handler:
"reindex" => [Phase::Reindex].into(),

// main.rs Consolidate subcommand:
#[arg(long, value_parser = ["reindex", "strengthen", "reorganize", "prune"])]
```

`Phase` should implement `FromStr`:

```rust
impl std::str::FromStr for Phase {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> { ... }
}
```

This lets clap parse directly, eliminates the manual match arms, and means a new phase only needs one registration point.

### 5.4 `load_sources()` as a Library Function

Source loading belongs in the `axel` library crate (probably `consolidate/mod.rs` or a new `sources.rs`), not in `main.rs`. Currently MCP bypasses it entirely and maintains a second hardcoded source list (see §6 below). The function should be:

```rust
// In axel crate:
pub fn load_sources(override_path: Option<&Path>) -> Result<Vec<SourceDir>>
```

Then both `main.rs` and `mcp.rs` import it from one place.

---

## 6. Technical Debt

### 6.1 Hardcoded Source Lists in Two Places (Critical)

The default source list exists twice, byte-for-byte:

**`main.rs`, lines 183-188:**
```rust
SourceDir { path: PathBuf::from(format!("{home}/Jawz/mikoshi/Notes/")), name: "mikoshi".into(), priority: Priority::High },
SourceDir { path: PathBuf::from(format!("{home}/Jawz/data/context/")), name: "context".into(), priority: Priority::High },
SourceDir { path: PathBuf::from(format!("{home}/Jawz/notes/")), name: "notes".into(), priority: Priority::Medium },
SourceDir { path: PathBuf::from(format!("{home}/Jawz/slack/diary/")), name: "slack-diary".into(), priority: Priority::Low },
SourceDir { path: PathBuf::from(format!("{home}/Jawz/data/context/memories/permanent/")), name: "memories-legacy".into(), priority: Priority::Medium },
SourceDir { path: PathBuf::from(format!("{home}/.stelline/memkoshi/exports/")), name: "memories".into(), priority: Priority::Medium },
```

**`mcp.rs`, lines ~387-393:** *Identical block.*

`load_sources()` exists in `main.rs` with a proper fallback chain and TOML support. The MCP handler never calls it. Adding a source to one does not add it to the other. This will cause silent behavioral divergence between CLI and MCP consolidation runs.

**Fix:** Move `load_sources()` into the `axel` library crate. Have `mcp.rs` call it.

### 6.2 `prune()` / `prune_with_priorities()` Split Leaves Auto-Removal Dead (High)

As detailed in §1 (spec gap #3), the orchestrator calls `prune::prune(search, opts.dry_run)` which creates an empty priority map. `prune_with_priorities()` exists with the correct implementation but is unreachable from normal execution paths. Phase 4 auto-removal never fires.

The fix is a one-line change in `mod.rs`:

```rust
// Current (broken):
stats.prune = prune::prune(search, opts.dry_run)?;

// Fixed:
let source_priorities: HashMap<String, Priority> = opts.sources.iter()
    .map(|s| (s.name.clone(), s.priority))
    .collect();
let (prune_stats, _candidates) = prune::prune_with_priorities(search, &source_priorities, opts.dry_run)?;
stats.prune = prune_stats;
```

The public `prune()` stub can remain for direct callers who don't have source context — but the orchestrator should not be one of them.

### 6.3 `cmd_index_sync` Duplication (Medium)

The reindex logic in `consolidate/reindex.rs` (`reindex_source()`) is a near-exact duplicate of `cmd_index_sync()` in `main.rs`. The spec acknowledged this: *"Already built: `cmd_index_sync()` in main.rs. Mirrors the logic of `cmd_index_sync` in main.rs."* The intention was for `cmd_index_sync` to delegate to `reindex_source`.

Current state: they run independently. `cmd_index_sync` does not call `reindex_source`. Any bug fixed in one won't propagate to the other.

Differences between them:
- `reindex_source` adds competitive allocation; `cmd_index_sync` does not
- `cmd_index_sync` respects a `--no-prune` flag; `reindex_source` always prunes
- `cmd_index_sync` has a `source` override for doc_id prefix; `reindex_source` uses `SourceDir.name`
- `cmd_index_sync` takes a single directory path; `reindex_source` takes a `SourceDir`

The most direct fix: `cmd_index_sync` should build a `SourceDir` from its arguments and call `reindex_source()`. The `--no-prune` flag would need threading through (perhaps a flag on `SourceDir` or `reindex_source`'s signature).

### 6.4 Magic Numbers in `prune.rs` SQL (Low)

```rust
// prune.rs
WHERE excitability < 0.15       // prune threshold — no named constant
AND access_count = 0
AND ... > 60                    // age threshold — no named constant
HAVING hits >= 5 AND avg_score < 0.015  // misalignment thresholds — no named constants
```

`strengthen.rs` and `reorganize.rs` correctly define all their thresholds as named `const` values at the top of the file. `prune.rs` embeds its critical thresholds directly in SQL strings. They are invisible to documentation, cannot be changed without modifying SQL strings, and are easy to miss in a review.

**Fix:**
```rust
const PRUNE_EXCITABILITY_THRESHOLD: f64 = 0.15;
const PRUNE_AGE_DAYS: i64 = 60;
const MISALIGNMENT_MIN_HITS: i64 = 5;
const MISALIGNMENT_MAX_AVG_SCORE: f64 = 0.015;
```

### 6.5 Log Normalization Fragility (Low)

`get_document_accesses_since()` in `db.rs` normalizes timestamps by string manipulation:

```rust
let normalized = since
    .replace('T', " ")
    .split('+').next().unwrap_or(since)
    .split('Z').next().unwrap_or(since)
    .to_string();
```

This strips timezone info to get SQLite-comparable datetime strings. It works for UTC timestamps but silently corrupts positive-offset timezone strings (e.g., `+05:30` becomes unusable after `.split('+').next()`). Since `consolidation_log.finished_at` is written as `Utc::now().to_rfc3339()` (always Z-suffixed), this is currently safe. But the function is brittle to any caller who passes a non-UTC or non-Z-terminated timestamp. A proper `chrono::DateTime<Utc>` parameter type would prevent this class of bug at the type level.

### 6.6 `session_id` Not Propagated in MCP (Low)

The MCP handler for `axel_search` logs document accesses but passes `None` for `session_id`:

```rust
let _ = db.log_document_access(&r.doc_id, "search_hit", Some(query), Some(r.score), None);
```

The `session_id` field exists in the schema for exactly this use case — correlating which MCP session triggered which accesses. MCP sessions have stable connection identities. This is a missed observability opportunity, not a correctness issue.

---

## Summary

The implementation is **substantially complete and structurally sound.** The module boundaries are correct, the orchestration logic is clean, and the biological model maps faithfully to code in four of the six areas the spec describes. The DB migration pattern is safe and idempotent. The audit log works. The MCP integration is present.

**Three issues require immediate attention before Phase 4 can be considered functional:**

1. **`prune()` does not pass source priorities to `prune_with_priorities()`** — auto-removal is silently disabled. One-line fix in `mod.rs`.
2. **Source list duplication in `mcp.rs`** — MCP consolidation uses a different (unconfigurable) source list than the CLI. Move `load_sources()` to the library crate.
3. **`log2` vs `ln` in boost formula** — consolidation decays more slowly than memkoshi for the same access pattern, producing inconsistent excitability growth across the two subsystems.

**Two medium-term improvements for robustness:**

4. **Extract `update_excitability()` into `db.rs`** — phases should not reach into raw SQL for a core domain operation.
5. **`cmd_index_sync` should delegate to `reindex_source()`** — eliminate the duplicate walk/mtime/prune logic before they diverge further.

**One low-priority but spec-mandated gap:**

6. **Context prefix embedding** (grid cell orthogonalization) is not implemented. The spec explicitly describes this as important for separating documents from different sources in embedding space. It was likely deferred due to the re-embed cost, but should be tracked as an open item.

The remaining gaps (interleaving in Phase 3, verbose per-document output, backup warning) are cosmetic and do not affect correctness. The spec's Open Questions remain open and are appropriately deferred.
