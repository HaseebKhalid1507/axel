# Axel Consolidation — Security Review

**Reviewed by:** Silverhand  
**Date:** 2025-07-13  
**Scope:** `crates/axel/src/consolidate/` (all phases), `crates/velocirag/src/db.rs`, `crates/axel/src/mcp.rs`  
**Verdict:** No critical show-stoppers, but four real issues worth fixing before this runs unattended.

---

## Summary Table

| # | Finding | Severity | File | Status |
|---|---------|----------|------|--------|
| 1 | Unbounded `limit` in MCP search → integer overflow + OOM | **HIGH** | `mcp.rs` | Open |
| 2 | Path traversal via `sources.toml` — `..` not blocked | **MEDIUM** | `main.rs` / `reindex.rs` | Open |
| 3 | No concurrency lock on consolidation | **MEDIUM** | `consolidate/mod.rs` | Open |
| 4 | Error messages leak internal file paths | **LOW** | `mcp.rs`, all phases | Open |
| 5 | `DefaultHasher` for edge IDs — non-stable, collision-prone | **LOW** | `reorganize.rs` | Open |
| 6 | SQL injection — **not present** (all parameterised) | N/A | `db.rs` | Clean ✓ |
| 7 | FTS5 injection — mitigated by sanitizer | N/A | `db.rs` | Clean ✓ |
| 8 | `grace_clause` string interpolation — safe (const-only) | N/A | `strengthen.rs` | Clean ✓ |
| 9 | Prune auto-delete gated on `Priority::Low` + hard criteria | N/A | `prune.rs` | Clean ✓ |
| 10 | MCP `axel_consolidate` sources hardcoded — no user injection | N/A | `mcp.rs` | Clean ✓ |

---

## Finding 1 — Unbounded `limit` in MCP search

**Severity:** HIGH  
**CWE:** CWE-789 (Uncontrolled Memory Allocation)  
**File:** `crates/axel/src/mcp.rs:179`, `crates/velocirag/src/search.rs:148`

### What's happening

```rust
// mcp.rs:179
let limit = args["limit"].as_u64().unwrap_or(5) as usize;
// … passed directly to brain.search(query, limit)
```

```rust
// velocirag/src/search.rs:148
let k = opts.limit * VECTOR_CANDIDATES_MULTIPLIER;  // × 3
let k = opts.limit * KEYWORD_CANDIDATES_MULTIPLIER; // × 3
```

### Exploit Path

Any MCP client (or local process speaking JSON-RPC to stdin) sends:

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"axel_search","arguments":{"query":"test","limit":9999999999}}}
```

`as_u64()` parses the value as `u64`. Cast to `usize` on a 64-bit system is a no-op. Then `limit * 3` is computed in `search.rs` with no overflow check. With `limit = 9_223_372_036_854_775_808` (2^63) the multiplication wraps to 0 in debug mode and silently produces garbage in release — or hits a panic on overflow checks if compiled with `-C overflow-checks`. More practically, `limit = 10_000_000` causes the HNSW search to request 30 million candidates, allocating gigabytes of Vec before SQLite even runs.

**The input schema says `"type": "integer"` with `"default": 5` but imposes no `maximum`.** There is no server-side cap anywhere in the call stack.

### Impact

- Process OOM-killed → MCP server dies, brain unavailable.
- If panic on overflow → same result.
- Attacker needs only stdin access to the MCP process (i.e., they are the LLM client or a compromised SynapsCLI plugin).

### Fix

```rust
// mcp.rs
let limit = args["limit"].as_u64().unwrap_or(5).min(100) as usize;
```

And tighten the schema:

```json
"limit": {
    "type": "integer",
    "description": "Max results (default 5, max 100)",
    "default": 5,
    "minimum": 1,
    "maximum": 100
}
```

---

## Finding 2 — Path Traversal via `sources.toml`

**Severity:** MEDIUM  
**CWE:** CWE-22 (Improper Limitation of a Pathname)  
**File:** `crates/axel/src/main.rs:162`, `crates/axel/src/consolidate/reindex.rs:36-43`

### What's happening

```rust
// main.rs:162
let expanded = path_str.replace("~", &home);
// … no further sanitisation
result.push(SourceDir {
    path: PathBuf::from(expanded),   // ← raw attacker-controlled path
    ...
});
```

```rust
// reindex.rs:36-43
if !source.path.is_dir() {          // ← is_dir() resolves .. before canonicalize
    return Err(...);
}
let target_abs = std::fs::canonicalize(&source.path)
    .unwrap_or_else(|_| source.path.clone()); // ← fallback skips canonicalize entirely!
```

### Exploit Path

If an attacker controls `~/.config/axel/sources.toml` (e.g. via a misconfigured dotfile sync, a compromised editor plugin writing configs, or just the user being the attacker on a shared system):

```toml
[[source]]
name    = "escape"
path    = "~/../../etc/"
priority = "low"
```

`~/../../etc/` expands to `/home/haseeb/../../etc/` which is a valid directory (`/etc`). `is_dir()` returns `true`. `canonicalize()` resolves it to `/etc`. WalkDir then walks `/etc`, indexing every `.md` and `.txt` file it finds into the brain. Because `Priority::Low` is set, those files become auto-delete candidates on the next prune pass.

**The `unwrap_or_else` fallback on `canonicalize` failure is the real killer** — if `canonicalize` fails (e.g. permission denied resolving a symlink), the raw `../`-containing path is used as the DB prefix for the LIKE query in `indexed_files_under`. That can match existing documents that live in completely unrelated paths.

### Impact

- Unintended files indexed into the brain (information gathering).
- With `Priority::Low`, those indexed files get auto-deleted from the brain on the next prune — data loss without explicit user action.
- Not exploitable via the MCP interface (sources are hardcoded in `mcp.rs`). CLI-only.

### Fix

After `canonicalize`, assert the resolved path is still under an allowed base:

```rust
let target_abs = std::fs::canonicalize(&source.path)
    .map_err(|e| AxelError::Other(format!("canonicalize failed: {e}")))?;
// Refuse to walk outside $HOME
let home = std::env::var("HOME").unwrap_or_default();
if !target_abs.starts_with(&home) {
    return Err(AxelError::Other(format!(
        "source path escapes HOME: {}", target_abs.display()
    )));
}
```

Remove the `unwrap_or_else` fallback — if `canonicalize` fails, that's an error, not a thing to silently paper over.

---

## Finding 3 — No Concurrency Lock on Consolidation

**Severity:** MEDIUM  
**CWE:** CWE-362 (Race Condition / TOCTOU)  
**File:** `crates/axel/src/consolidate/mod.rs`

### What's happening

`consolidate()` runs multiple destructive phases sequentially:

1. **Reindex** — re-embeds and *deletes* stale documents  
2. **Strengthen** — bulk-updates `excitability` across all documents  
3. **Reorganize** — reads `co_retrieval` then upserts/decays edges  
4. **Prune** — deletes documents matching staleness criteria  

There is no advisory lock, no PID file, no `BEGIN EXCLUSIVE TRANSACTION` wrapping the full pass. SQLite WAL mode provides row-level isolation for individual statements, but does not prevent two concurrent consolidation processes from interleaving their phases.

### Exploit Path

Not an external exploit — this is an operational risk. Two paths trigger it today:

1. **Cron + MCP**: a cron job runs `axel consolidate` at the same moment the LLM calls `axel_consolidate` via MCP.
2. **Two MCP clients**: two SynapsCLI instances sharing the same brain file, both issuing `axel_consolidate`.

**Concrete corruption scenario:**

```
Process A: Phase 1 reindex — deletes doc X (stale on disk)
Process B: Phase 2 strengthen — reads access log, queues boost for doc X
Process B: UPDATE documents SET excitability = 0.9 WHERE doc_id = 'X'  → 0 rows affected (X deleted)
Process A: Phase 4 prune — evaluates based on pre-B state → flags wrong docs
Process A: consolidation_log written with merged stats from both runs
```

The `last_consolidation_time()` used by Phase 2 and Phase 3 as their window start will return the timestamp written by whichever process finishes first — causing the *second* process to skip all events that happened before the first run completed, silently dropping access signals.

### Impact

- Excitability scores drift incorrectly → wrong documents get pruned.
- Co-retrieval edges built on stale data → graph integrity degrades.
- `consolidation_log` shows clean runs but the brain is corrupted.

### Fix

Acquire a SQLite exclusive lock at the start of a full consolidation pass and release it at the end:

```rust
// In consolidate() before any phase runs:
search.db().conn().execute_batch("BEGIN EXCLUSIVE")?;
// ... all phases ...
search.db().conn().execute_batch("COMMIT")?;
```

Or for a lighter-weight approach that doesn't block reads for the whole pass, use a dedicated `kv` table entry as an advisory lock:

```sql
INSERT INTO kv (key, value) VALUES ('consolidation_lock', datetime('now'))
-- check and fail if lock is <30min old
```

---

## Finding 4 — Error Messages Leak Internal File Paths

**Severity:** LOW  
**CWE:** CWE-209 (Information Exposure Through an Error Message)  
**File:** `mcp.rs`, `reindex.rs`, `prune.rs`

### What's happening

```rust
// mcp.rs:420
Err(e) => tool_error(&format!("Consolidation failed: {e}")),

// reindex.rs:37-40
return Err(AxelError::Other(format!(
    "reindex source not a directory: {}",
    source.path.display()     // ← absolute path
)));

// reindex.rs:55
.map_err(|e| AxelError::Search(format!("DB read failed: {e}")))?;
// rusqlite errors include the DB file path
```

The full error chain propagates through `tool_error()` directly to the MCP JSON response, which is written to stdout and consumed by the LLM (or any logging middleware).

### Exploit Path

Call `axel_consolidate` via MCP and observe the error response. On a misconfigured brain:

```json
{
  "content": [{"type": "text", "text": "Consolidation failed: reindex source not a directory: /home/haseeb/Jawz/mikoshi/Notes/"}]
}
```

This discloses: username, home directory structure, project layout, and the existence of specific subdirectories. On a DB error, rusqlite may include the full path to `axel.r8`.

### Impact

- Low in a single-user local setup. Elevated in any multi-tenant or networked scenario.
- LLM context injection: if a note file contains crafted content that produces a deliberate error, the path leaks into the LLM's context window where it can be exfiltrated via prompt injection in a subsequent query.

### Fix

Log the full error internally (`tracing::error!`) and return a sanitised message to the caller:

```rust
Err(e) => {
    tracing::error!("consolidation failed: {e}");
    tool_error("Consolidation failed. Check logs for details.")
}
```

---

## Finding 5 — `DefaultHasher` for Co-retrieval Edge IDs

**Severity:** LOW  
**CWE:** CWE-327 (Use of a Broken or Risky Cryptographic Algorithm — hash stability)  
**File:** `crates/axel/src/consolidate/reorganize.rs:8,42-49`

### What's happening

```rust
use std::collections::hash_map::DefaultHasher;

fn coret_edge_id(a: &str, b: &str) -> String {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let mut h = DefaultHasher::new();
    lo.hash(&mut h);
    "::".hash(&mut h);
    hi.hash(&mut h);
    format!("coret_{:x}", h.finish())
}
```

`DefaultHasher` in Rust's standard library:

1. **Is not stable across Rust versions** — the implementation can change between compiler releases. Same input → different hash after a `rustup update`.
2. **Is not stable across processes** — since Rust 1.36, `HashMap` uses per-process random seeds by default. `DefaultHasher::new()` does *not* apply the random seed, but this is an implementation detail that could change.
3. **Has a 64-bit output space** — birthday collision probability with 4 billion edge pairs is ~50% (birthday bound at 2^32 pairs).

### Impact

- After a Rust toolchain upgrade, all existing `coret_*` edge IDs become orphaned. New runs generate different IDs for the same pairs, producing duplicate edges instead of updating existing ones. The brain accumulates stale graph edges that never decay.
- A collision between two different doc_id pairs maps them to the same edge row, causing one pair's weight to silently update the other's relationship.

### Fix

Use a deterministic, stable hash. The simplest drop-in replacement:

```rust
// Use a hex encoding of the canonical pair directly
fn coret_edge_id(a: &str, b: &str) -> String {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    // SHA-256 first 16 hex chars gives 64-bit collision resistance, stably
    use std::collections::hash_map::DefaultHasher; // replace with:
    // format!("coret_{}", &sha256_hex(&format!("{lo}::{hi}"))[..16])
    // Or simply use the raw pair as the ID (IDs are TEXT PRIMARY KEY, length is fine):
    format!("coret_{}::{}", lo, hi)
}
```

Using the raw canonical pair as the ID is simplest and completely collision-free. Edge IDs are TEXT PRIMARY KEY in SQLite — no size constraint.

---

## What's Actually Clean

These were checked and found solid. Credit where it's due.

### SQL Injection — Not Present ✓

Every single SQL statement across `db.rs`, `prune.rs`, `strengthen.rs`, `reorganize.rs` uses `rusqlite::params![]` bound parameters. No string interpolation of user data into query strings was found. The `indexed_files_under` LIKE query correctly uses `escape_like()` to escape `%` and `_` metacharacters before binding. This is the right way to do it.

### FTS5 Injection — Mitigated ✓

`keyword_search` in `db.rs` runs `sanitize_fts5_query()` before binding to the FTS5 `MATCH` operator. The sanitizer strips all FTS5 special characters and wraps each word in double-quotes. The `content:` column filter syntax (which could be used to restrict search to specific FTS columns) is neutralised because `:` is in `FTS5_SPECIAL`. Tested manually — `content:password` becomes `"contentpassword"`.

### `grace_clause` String Interpolation — Safe ✓

```rust
// strengthen.rs:101
let grace_clause = format!("-{} days", GRACE_DAYS as i64);
```

`GRACE_DAYS` is a `const f64 = 14.0`. Cast to `i64` produces `14`. The resulting string `"-14 days"` is a constant derived value, not user input. This is passed as a bound parameter via `params![]`, not interpolated into the SQL string. Safe.

### Prune Auto-Delete Gating — Conservative ✓

Auto-deletion in `prune.rs` requires all three conditions simultaneously: `excitability < 0.15` AND `access_count = 0` AND `age_days > 60`, AND the source must be tagged `Priority::Low` in the caller's map. The default `prune()` entry point passes an empty `HashMap`, meaning **nothing is ever auto-deleted** without the caller explicitly providing source priorities. The spec says "flag for human review" as the safe default, and that's what the code does.

### MCP `axel_consolidate` Sources — Hardcoded ✓

The MCP handler constructs source directories from hardcoded paths under `$HOME`. There is no mechanism for an MCP caller to supply arbitrary source paths via the tool arguments. The `phase` and `dry_run` parameters are the only inputs, and both are strictly validated (enum match for phase, bool for dry_run). The path traversal risk in Finding 2 only applies to the CLI `axel consolidate --sources` path.

---

## Recommendations by Priority

1. **Fix immediately:** Cap `limit` in `axel_search`. One line. No excuse not to.
2. **Fix before unattended operation:** Add a concurrency guard to `consolidate()`. A `kv`-table advisory lock is five lines of SQL.
3. **Fix before exposing to untrusted config:** Add the `starts_with($HOME)` check in `reindex_source` after `canonicalize`. Drop the `unwrap_or_else` fallback.
4. **Fix before any networked deployment:** Sanitise error messages in MCP handlers. Log internally, return generic strings externally.
5. **Fix before next breaking Rust version:** Replace `DefaultHasher` in `coret_edge_id` with the raw canonical pair string as the edge ID.
