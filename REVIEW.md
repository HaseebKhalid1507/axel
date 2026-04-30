# Axel — 8-Agent Review Synthesis (S181)

## Agents Deployed
| Agent | Angle | Status |
|-------|-------|--------|
| Chrollo | Architecture | ⚠️ Partial (timed out mid-report, got dependency + schema analysis) |
| Shady | Security | ✅ Full report — 6 critical/warning findings |
| Spike | Code Quality | ❌ Timed out before report |
| Silverhand | Adversarial | ❌ Timed out before report |
| Yoru | Performance | ✅ Rich findings — 2 high-impact issues |
| Gojo | Test Coverage | ✅ Full coverage map — 5 untested modules found |
| Star-Lord | UX/CLI | ✅ Full report — 11 issues, #1 is broken query parser |

---

## 🔴 CRITICAL (fix before shipping)

### SEC-1: HMAC signing is NEVER called in production (Shady)
`MemorySigner` exists, tested in isolation, but `cmd_remember`, `cmd_extract`, `storage.stage_memory()`, `storage.approve()` — NONE of them call it. Every memory hits the DB with `signature: None`. Direct SQLite edits are fully undetected.

### SEC-2: No key management exists (Shady)
`MemorySigner::new()` takes `&[u8]` but no production code provides a key. No `AXEL_SIGNING_KEY` env var, no config field, no derivation. The signing infrastructure has no foundation.

### SEC-3: Memory tampering via direct SQLite is undetected (Shady)
`sqlite3 axel.r8 "UPDATE memories SET content='POISON'"` → next `axel recall` serves it. No verification on read. `trust_level` is stored but never consulted.

### UX-1: Argument parser is a logic minefield (Star-Lord)
Hand-rolled `Vec<String>` parser with chained iterators. Breaks on: `--limit` before query, numeric words in queries (e.g. "5 reasons"), bad `--limit` values silently swallowed. **Recommendation: adopt `clap` derive — fixes UX-1, UX-2, UX-4, UX-6 in one dep.**

### PERF-1: HNSW index rebuilt on every CLI invocation (Yoru)
`BrainSearch::open()` creates empty index every time. `VectorIndex::save()`/`load()` exist but are never called. Adds ~350-600ms to every search. Fix: persist the HNSW index to disk, load on open, rebuild only when doc count changes.

### PERF-2: Embedding cache not wired (Yoru)
`Embedder::new(None, None, true)` — `cache_dir=None` disables disk cache. Query embeddings are recomputed on every invocation instead of being cached.

---

## 🟠 WARNING (should fix)

### SEC-4: Prompt injection detection is 5 string patterns (Shady)
Bypassed by: unicode homoglyphs, whitespace insertion, synonym substitution, encoding tricks. Title/topic fields not checked at all. Modern jailbreak taxonomy entirely absent.

### SEC-5: LIKE injection in velocirag tag/cross-ref search (Shady)
Unescaped `%` and `_` in user input are live LIKE metacharacters. Search for `%` returns all tags.

### SEC-6: Handoff has no size limit on write (Shady)
`axel handoff set $(python3 -c "print('A'*10000000)")` writes 10MB to SQLite. No cap before DB.

### UX-2: No per-subcommand `--help` (Star-Lord)
`axel search --help` searches for the string "--help". Returns 0 results.

### UX-3: Inconsistent exit codes (Star-Lord)
`forget` on nonexistent ID exits 0. No-results search exits 0. Scripts can't distinguish success from not-found.

### UX-5: Silent category mismatch (Star-Lord)
`--category PREFS` silently falls back to Events. Should be a hard error.

### ARCH-1: Dual schema ownership (Chrollo)
Brain (r8.rs) creates Memkoshi tables. MemoryStorage (storage.rs) ALSO creates tables with different column names. Led to the `abstract` vs `abstract_text` bug. Single source of truth needed.

### TEST-1: decay module is 0% tested (Gojo)
The core "memory gets smarter over time" feature has zero tests. Boost and decay math completely unverified.

### TEST-2: patterns module is 0% tested (Gojo)
Frequency, gap, and temporal pattern detection — all untested. Confidence formula never asserted.

### TEST-3: MemoryStorage context methods untested (Gojo)
`set_context`/`get_context` layer priority logic has zero coverage.

---

## 🟡 NICE TO HAVE

- UX-6: No `--version` flag (Star-Lord)
- UX-7: Silent extract pipeline failures (Star-Lord)
- UX-8: Undocumented handoff shorthand (Star-Lord)
- UX-9: Empty recall prints nothing (Star-Lord)
- UX-10: stdout/stderr mixing (Star-Lord)
- TEST-4: Unicode in memory titles never tested (Gojo)
- TEST-5: Schema version mismatch error path untested (Gojo)
- ARCH-2: memkoshi depends on velocirag but never uses it (Chrollo)

---

## Priority Fix Order

1. **Adopt `clap`** — kills UX-1/2/4/5/6 in one shot
2. **Persist HNSW index** — PERF-1, biggest user-facing improvement
3. **Wire embedding cache** — PERF-2
4. **Fix exit codes** — UX-3
5. **Wire HMAC signing into production paths** — SEC-1/2/3
6. **Expand injection patterns + check title/topic** — SEC-4
7. **Write tests for decay + patterns** — TEST-1/2
8. **Unify schema ownership** — ARCH-1
