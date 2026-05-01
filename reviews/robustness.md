# Robustness Review — Axel Consolidation Engine
**Scope:** `crates/axel/src/consolidate/` + `crates/velocirag/src/db.rs` (DB methods) + `brain.rs`
**Reviewer:** Gojo (subagent)
**Date:** 2025-07

---

## Executive Summary

The consolidation engine is well-structured and clearly bio-inspired. The error-type hierarchy
is solid, `thiserror` is used correctly, and most error paths propagate cleanly. However there
are **six production-risk issues** ranging from data-loss on crash to silent file-skip silencing,
plus several lower-severity hygiene items.

**Risk tiers used in this doc:**
- 🔴 **CRITICAL** — can silently corrupt or lose data in production
- 🟠 **HIGH** — operational problem, likely noticed but hard to diagnose
- 🟡 **MEDIUM** — latent bug / future foot-gun
- 🟢 **LOW** — hygiene / nice-to-have

---

## 1. Unwrap / Expect Calls

### 1a. 🔴 `parse_from_rfc3339(...).unwrap()` — panics on bad DB timestamps
**Location:** `velocirag/src/db.rs:770-771`
```rust
valid_from: row.get::<_, Option<String>>(14)?.map(|s| chrono::DateTime::parse_from_rfc3339(&s).unwrap().into()),
valid_to:   row.get::<_, Option<String>>(15)?.map(|s| chrono::DateTime::parse_from_rfc3339(&s).unwrap().into()),
```
**Risk:** If any edge row ever gets a timestamp written in the wrong format (e.g., SQLite
`datetime('now')` returns `"2025-07-06 12:00:00"` — **not** RFC3339), this panics mid-query.
`reorganize.rs` reads all live edges with this query path (`get_neighbors`-style traversal).
A single bad timestamp row crashes the entire consolidation run and takes down the process.

**Fix:**
```rust
valid_from: row.get::<_, Option<String>>(14)?
    .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
    .map(Into::into),
```

---

### 1b. 🟡 `unwrap_or_else` fallbacks in `reindex.rs` — safe but audit-worthy
**Location:** `reindex.rs:44, 69`
```rust
let target_abs = std::fs::canonicalize(&source.path).unwrap_or_else(|_| source.path.clone());
let abs = std::fs::canonicalize(file_path).unwrap_or_else(|_| file_path.to_path_buf());
```
These are fine — the fallback to the original path is the right behavior for
symlinks or paths that can't be canonicalized. Not a panic risk.
No action required, but consider logging a `tracing::debug!` on the error branch
so permissions issues are at least visible.

---

### 1c. 🟡 `unwrap_or` on mtime in `reindex.rs:78`
```rust
.unwrap_or(0.0)
```
Falls back to Unix epoch (1970). That means a file whose metadata can't be read
will always appear "older than indexed_at" — so it will always be re-indexed every
run. Harmless correctness-wise, but wastes work. A `tracing::debug!` here would
make this diagnosable.

---

### 1d. 🟡 `unwrap_or` inside `prune.rs` misaligned-embedding enrichment (lines 155-163)
```rust
row.get::<_, f64>(0).unwrap_or(0.5),
row.get::<_, i64>(1).unwrap_or(hits),
row.get::<_, i64>(2).unwrap_or(0),
...
let (excitability, access_count, age_days) = meta.unwrap_or((0.5, hits, 0));
```
These are inside the enrichment query for the **report only** — they don't affect
what gets deleted. The fallbacks are reasonable. Low risk.

---

## 2. Silent Failures

### 2a. 🟠 `let _ = self.search.index_memory(...)` — silent indexing failure
**Location:** `brain.rs:241` and `brain.rs:286`
```rust
let _ = self.search.index_memory(&memory);
```
Called in `remember()` and `update_memory()`. If the HNSW index add fails or the
embedder fails, the memory is stored in SQLite but **silently not searchable**.
The user gets back `Ok(id)` — no indication the memory won't appear in search results.

This is a real UX and data-consistency trap. The memory exists but is invisible
until a flush/rebuild re-indexes it (if that ever happens).

**Fix:** Propagate the error, or at minimum warn:
```rust
if let Err(e) = self.search.index_memory(&memory) {
    tracing::warn!("Memory stored but not indexed (search miss until next consolidation): {e}");
}
```
The warn approach is more forgiving for CLI users — they don't lose the write,
but operators/logs will catch the degradation.

---

### 2b. 🟠 `Err(_) => continue` on `fs::read_to_string` — unreadable files are invisible
**Location:** `reindex.rs:87-90`
```rust
let content = match std::fs::read_to_string(file_path) {
    Ok(c) => c,
    Err(_) => continue,
};
```
If a file can't be read (permissions, encoding, etc.) it is silently skipped.
The stats counter (`stats.reindexed`) is not incremented, so there's no way to
know how many files were skipped. The file won't appear in the DB, so if it was
previously indexed and the mtime changed, it could also be deleted by the prune
step below (double loss).

**Fix:** Add an `errors` or `skipped` counter to `ReindexStats` and record it:
```rust
Err(e) => {
    tracing::warn!("skipping unreadable file {}: {e}", file_path.display());
    stats.skipped += 1;
    continue;
}
```

---

### 2c. 🟡 `let _ = self.search.flush()` in `Drop` — flush errors vanish on process exit
**Location:** `brain.rs:399`
```rust
fn drop(&mut self) {
    let _ = self.search.flush();
}
```
This is acceptable — panicking in `Drop` is worse. But the HNSW cache could fail
to persist silently (disk full, permissions). The `BrainSearch::drop` already has
a `tracing::warn!` for this case (`search.rs:~253`), so this is actually fine as
implemented — just noting it for completeness.

---

### 2d. 🟡 `filter_map(|e| e.ok())` on WalkDir — permission errors silently dropped
**Location:** `reindex.rs:64`
```rust
for entry in walkdir::WalkDir::new(&target_abs)
    .into_iter()
    .filter_map(|e| e.ok())
```
Subdirectories with permission errors are silently skipped. For Haseeb's vault
this is probably never an issue, but in a general-purpose tool it can hide
partially-indexed source trees.

---

## 3. Error Propagation

### 3a. 🟢 Generally good
The `?` operator is used consistently throughout all four phase files.
`AxelError` covers all the right variants (`Db`, `Io`, `Search`, etc.).
The consolidation log insert in `mod.rs:129-131` correctly maps through:
```rust
search.db().insert_consolidation_log(&entry)
    .map_err(|e| crate::error::AxelError::Search(format!("log insert failed: {e}")))?;
```

### 3b. 🟡 `reorganize.rs` — bare `?` on rusqlite errors loses context
**Location:** `reorganize.rs:77, 78, 119, 135`
```rust
let rows = stmt.query_map(...)?;
rows.collect::<rusqlite::Result<Vec<_>>>()?
...
conn.execute("UPDATE edges SET weight = ...", ...)?;
```
These propagate `rusqlite::Error` directly via the `From<rusqlite::Error>` impl on
`AxelError::Db`. That works correctly — but the error message at the user level
just says "Database error: ..." with no phase context. Compare with `strengthen.rs`
which wraps with context strings like `"update excitability: {e}"`. Reorganize
should do the same.

---

## 4. Graceful Degradation Between Phases

### 4a. 🔴 A single-phase failure aborts ALL remaining phases — no phase isolation
**Location:** `consolidate/mod.rs:85, 96, 103, 108`
```rust
let s = reindex::reindex_source(search, src, opts.dry_run)?;   // Phase 1
stats.strengthen = strengthen::strengthen(search, opts.dry_run)?;  // Phase 2
stats.reorganize = reorganize::reorganize(search, opts.dry_run)?;  // Phase 3
stats.prune = prune::prune(search, opts.dry_run)?;             // Phase 4
```
Every phase is connected with `?`. If Phase 2 (Strengthen) hits a DB error,
Phases 3 and 4 never run. Since the audit log insert also never runs, the
`last_consolidation_time()` cursor doesn't advance — so the next run may
re-process a huge backlog it was supposed to skip.

Strengthen failures (excitability update) are especially risky: a DB lock
contention mid-strengthen leaves all the boosted docs in a partial state.

**Recommended fix:** Collect phase results independently, log per-phase errors,
and continue to subsequent phases:

```rust
#[derive(Debug)]
pub enum PhaseOutcome<T> {
    Ok(T),
    Err(AxelError),
    Skipped,
}

// Then in consolidate():
let strengthen_result = if wants(&opts.phases, Phase::Strengthen) {
    match strengthen::strengthen(search, opts.dry_run) {
        Ok(s) => { stats.strengthen = s; PhaseOutcome::Ok(()) }
        Err(e) => {
            tracing::error!("Phase 2 (Strengthen) failed: {e}");
            PhaseOutcome::Err(e)
        }
    }
} else { PhaseOutcome::Skipped };
// ... continue to Phase 3 regardless
```

The audit log should still be written at the end with whatever partial stats
were collected, so the time cursor advances and the next run doesn't re-process.

---

### 4b. 🟡 Single failing source in `reindex_source` loop aborts remaining sources
**Location:** `consolidate/mod.rs:85`
```rust
for src in sources {
    let s = reindex::reindex_source(search, src, opts.dry_run)?;
    ...
}
```
If source[0] (`mikoshi`) hits an error, sources[1..n] never run. The prune
step for source[0] may have already fired, so files from that source got
deleted from the DB but never re-added.

Same fix as 4a — collect errors, continue loop, report at the end.

---

## 5. Recovery After Mid-Run Crash

### 5a. 🔴 No in-progress marker — crashed run is undetectable; time cursor may be stale
The consolidation_log schema only records a row **after** the run completes
(both `started_at` and `finished_at` are written atomically in a single INSERT
at the end of `consolidate()`). If the process crashes mid-run, **no row is written**.

That means:
1. `last_consolidation_time()` returns the timestamp of the *previous* successful run.
2. Phase 2 (Strengthen) will re-process all access events since that older timestamp —
   including ones it already processed and boosted last run. Docs get double-boosted.
3. Phase 3 (Reorganize) re-processes all co-retrieval pairs from the full window.
   Edges may be double-counted and pushed past their intended weight.

**Fix — two-phase log write:**
```rust
// At the START of consolidate(), before any phases run:
search.db().insert_consolidation_log_start(&started_at)?;
// ... phases run ...
// At the END, UPDATE the row with finished_at and stats:
search.db().finalize_consolidation_log(log_id, &stats)?;
```
Then `last_consolidation_time()` can check for rows where `finished_at IS NULL`
as a crash-recovery signal, and `get_document_accesses_since` can use `started_at`
of the in-progress entry as a fence.

---

### 5b. 🟠 `upsert_document` (FTS sync) and `delete_documents_by_file` are not atomic
**Location:** `velocirag/src/db.rs:425-442` and `511-522`

Both functions execute two separate SQL statements without a transaction:
```rust
// upsert_document:
self.conn.execute("INSERT OR REPLACE INTO documents ...")?;
self.conn.execute("DELETE FROM chunks_fts WHERE ...")?;  // if crash here...
self.conn.execute("INSERT INTO chunks_fts ...")?;         // FTS is orphaned

// delete_documents_by_file:
self.conn.execute("DELETE FROM chunks_fts WHERE ...")?;   // if crash here...
self.conn.execute("DELETE FROM documents ...")?;           // doc survives, FTS gone
```
A crash between these two statements leaves `documents` and `chunks_fts` out of
sync. The vector index (HNSW) is a separate file and adds a third desync risk.

WAL mode mitigates OS-level crashes (WAL is checkpointed atomically), but a
Rust panic between the two `execute` calls is not protected by WAL.

**Fix:**
```rust
pub fn upsert_document(...) -> Result<i64> {
    let tx = self.conn.transaction()?;
    tx.execute("INSERT OR REPLACE INTO documents ...", ...)?;
    tx.execute("DELETE FROM chunks_fts WHERE ...", ...)?;
    tx.execute("INSERT INTO chunks_fts ...", ...)?;
    tx.commit()?;
    Ok(self.conn.last_insert_rowid())
}
```

---

## 6. Logging

### 6a. 🟠 Phase-level logging only exists under `opts.verbose` — errors go nowhere otherwise
**Location:** `consolidate/mod.rs:83-95`
```rust
if opts.verbose {
    eprintln!("⟳ reindex [{}] ...", ...);
}
```
In `verbose: false` mode (the default for MCP calls from `mcp.rs`), there is
zero logging of phase start/end, phase durations, or per-source progress.
If MCP consolidation hangs or silently fails mid-phase, there's nothing in
logs to pinpoint where.

**Fix:** Emit `tracing::info!` (not just `eprintln!`) unconditionally for phase
start/end and per-source reindex. The verbose `eprintln!` can stay for the CLI
human-readable output. Structured tracing is orthogonal.

---

### 6b. 🟢 Allocation failures are correctly traced
```rust
tracing::warn!("allocation failed for {new_id}: {e}");
tracing::warn!("insert_edge failed: {e}");
```
Good — these are non-fatal so warn-level + continue is exactly right.

---

### 6c. 🟡 Strengthen: no log of how many docs were processed, only results
`StrengthenStats` tracks `boosted`, `decayed`, `extinction_signals` — but not
`total_processed` or `total_accesses_read`. If boosted+decayed = 0 on a large
brain, it's impossible to tell if strengthen ran correctly or if the access log
was empty vs. something silently failed.

---

## 7. Partial Failures in Batch (1000 files)

### 7a. 🟠 File 500 embedding failure → files 501-1000 never indexed; 500 is lost without notice
**Location:** `reindex.rs:101-103`
```rust
search.index_document(&doc_id, &content, None, Some(&abs_str))?;
```
`index_document` can fail if the embedder fails (OOM, model error, etc.). The
`?` immediately exits `reindex_source`, leaving all subsequent files in the batch
unprocessed. Worse:
- The `stats.reindexed` counter already incremented for file 500 before the DB
  error — so the CLI output claims it was processed.

Wait — actually no, let me re-read:
```rust
if !dry_run {
    search.index_document(&doc_id, &content, None, Some(&abs_str))?;  // if this errors, we exit
    if is_new { newly_indexed.push(doc_id.clone()); }
}
stats.reindexed += 1;  // this line is NEVER reached on error — correct
```
Stats are counted after the `?`, so the count is accurate. However, files 501-1000
are still never processed.

**Fix:** Same as 4b — catch per-file errors, log them, continue the loop:
```rust
match search.index_document(&doc_id, &content, None, Some(&abs_str)) {
    Ok(_) => {
        if is_new { newly_indexed.push(doc_id.clone()); }
        stats.reindexed += 1;
        if is_new { stats.new_files += 1; }
    }
    Err(e) => {
        tracing::warn!("failed to index {doc_id}: {e}");
        stats.errors += 1;
    }
}
```
Files 501-1000 keep processing. Operators see the error count in stats.

---

### 7b. 🟡 Prune loop does NOT have the same problem — each delete is independent
Each iteration of the prune loop calls `db.delete_documents_by_file(fp)?`.
This will bail on the first delete error. Since deletes are independent (no
shared state), the same per-iteration error catch would help here too, but
the impact is lower — a failed delete leaves a stale doc in the DB (no data
loss, just wasted space).

---

## Summary Table

| # | Severity | Location | Issue |
|---|----------|----------|-------|
| 1a | 🔴 CRITICAL | `db.rs:770-771` | `.unwrap()` on RFC3339 parse panics on malformed timestamps |
| 4a | 🔴 CRITICAL | `mod.rs:85-108` | Phase failure aborts all remaining phases; audit log never written |
| 5a | 🔴 CRITICAL | `mod.rs` + `db.rs` | No in-progress log marker; crashed run causes double-processing on next run |
| 5b | 🟠 HIGH | `db.rs:425-442, 511-522` | `upsert_document` and `delete_documents_by_file` lack wrapping transactions |
| 2a | 🟠 HIGH | `brain.rs:241,286` | `let _ = index_memory(...)` silently stores memories that won't be searchable |
| 2b | 🟠 HIGH | `reindex.rs:87-90` | `Err(_) => continue` on file read — no count, no log, invisible skips |
| 4b | 🟠 HIGH | `mod.rs:85` | Single failing source aborts remaining sources |
| 6a | 🟠 HIGH | `mod.rs` | Phase events only logged under `verbose`; MCP calls are dark |
| 7a | 🟠 HIGH | `reindex.rs:102` | Embedding error on file N kills files N+1..end of batch |
| 3b | 🟡 MEDIUM | `reorganize.rs` | Bare `?` on rusqlite loses phase context in error messages |
| 1c | 🟡 MEDIUM | `reindex.rs:78` | mtime=0 fallback always re-indexes unreadable-metadata files |
| 6c | 🟡 MEDIUM | `strengthen.rs` | No `total_processed` counter; 0 stats indistinguishable from no-op |
| 7b | 🟡 MEDIUM | `prune.rs` | Delete errors abort remaining prune items |
| 1b | 🟢 LOW | `reindex.rs:44,69` | `unwrap_or_else` on canonicalize — safe but silent |
| 2c | 🟢 LOW | `brain.rs:399` | `let _ = flush()` in Drop — acceptable, already warned in BrainSearch::drop |
| 2d | 🟢 LOW | `reindex.rs:64` | WalkDir permission errors silently dropped |

---

## Recommended Fix Priority Order

1. **Wrap `upsert_document` and `delete_documents_by_file` in transactions** (5b) — pure
   SQLite change, high leverage, no API change needed.

2. **Fix the `.unwrap()` on RFC3339 parse** (1a) — one-line change, prevents process crash.

3. **Per-file error isolation in `reindex_source`** (2b, 7a) — change `?` to match+warn+continue
   and add `errors: usize` to `ReindexStats`. Prevents batch loss.

4. **Phase isolation in `consolidate()`** (4a, 4b) — wrap each phase in a match, continue on
   error, always write the audit log. Biggest architecture change but highest operational value.

5. **Two-phase consolidation log write** (5a) — insert `started_at` before phases, update
   `finished_at` after. Enables crash detection and prevents double-processing.

6. **Surface `index_memory` errors** (2a) — convert `let _ =` to `warn!`. One-liner.

7. **Unconditional `tracing::info!` for phase events** (6a) — add structured logging alongside
   the existing `verbose` eprintln. Keeps MCP calls observable.
