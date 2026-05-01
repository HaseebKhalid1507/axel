# Axel Consolidation — UX & Documentation Review

> Reviewed by: Star-Lord (creative/UX subagent)
> Scope: CLI interface, output formatting, error messages, docs, sources config
> Files: `main.rs`, `mcp.rs`, `CONSOLIDATION.md`, `sources.toml`, systemd units

---

## Executive Summary

The consolidation system is **architecturally solid and well-documented**, but
the UX has a cluster of fixable rough edges. The spec is excellent as a design
document and mostly matches the implementation — with one meaningful gap (the
spec's rich per-phase output exists only in the spec). The CLI help is
thin-by-clap-defaults, the output summary is functional but not informative, and
there are a few paper cuts that would frustrate a user trying to understand
what's happening in their brain.

**Verdict:** 7/10 spec quality, 5/10 CLI UX, 4/10 output quality.

---

## 1. CLI Help Text

### What `axel consolidate --help` actually emits

Clap generates help from the doc-comments on the `Consolidate` variant and its
fields. What a user sees today:

```
Run a consolidation pass on the brain

  Reindexes changed files, strengthens accessed documents,
  reorganizes graph edges, and prunes stale content.

Usage: axel consolidate [OPTIONS]

Options:
      --phase <PHASE>    Run specific phase only [possible values: reindex, strengthen, reorganize, prune]
      --dry-run          Preview changes without applying them
  -v, --verbose          Verbose output (per-document details)
      --sources <SOURCES> Path to sources config TOML
      --history          Show consolidation history
  -h, --help             Print help
```

### Issues

**1.1 — `--history` is a mode disguised as a flag**

`--history` completely changes the command's behaviour — it switches from
"run consolidation" to "show past runs". It should be a subcommand or at least
clearly labelled. Right now there's nothing in the help that hints `--history`
produces different output.

**Recommendation:** Either make it `axel consolidate history` (subcommand) or
add a clear note in the flag description:

```rust
/// Show consolidation history (last 20 runs). Does not run consolidation.
#[arg(long)]
history: bool,
```

**1.2 — `--phase` has no descriptions**

The possible values are listed (`reindex`, `strengthen`, `reorganize`, `prune`)
but there's no hint what each does. A new user staring at `--phase reorganize`
has no idea what that means.

**Recommendation:** Use a custom `value_parser` or clap's `value_enum` with
`about` text, OR expand the long_help for this flag:

```rust
/// Run specific phase only.
///
/// Phases:
///   reindex    — re-embed changed/new files, prune deleted ones
///   strengthen — boost/decay excitability from retrieval history
///   reorganize — update graph edges from co-retrieval patterns
///   prune      — remove/flag stale or misaligned documents
#[arg(long, value_parser = [...])]
phase: Option<String>,
```

**1.3 — No mention of env vars or config file in help**

The system supports `AXEL_SOURCES` env var and `~/.config/axel/sources.toml`
auto-discovery, but neither appears in the help output. Users who want to
override sources have to find this in the spec (if they read it).

**Recommendation:** Add to the `--sources` flag description:

```rust
/// Path to sources config TOML.
///
/// Resolution order:
///   1. --sources <path>
///   2. $AXEL_SOURCES env var
///   3. ~/.config/axel/sources.toml (auto-discovered)
///   4. Built-in defaults
#[arg(long)]
sources: Option<PathBuf>,
```

**1.4 — `--verbose` promises per-document details, but does it deliver?**

The flag says "Verbose output (per-document details)" but this depends entirely
on what `consolidate::consolidate()` does with `opts.verbose`. If verbose mode
isn't fully wired through all phases, the flag is lying to the user. Worth
auditing `consolidate/mod.rs` and phase files to verify.

**1.5 — No `--since` flag (spec mentions it, flag doesn't exist)**

`CONSOLIDATION.md` (line ~153) says:
> `consolidate --since last` uses the most recent entry's `finished_at` to
> scope Phase 2 queries.

This flag doesn't exist in the CLI. The history table has `finished_at`, the
spec describes the feature, but there's no `--since` argument. Either document
that it's not implemented, or implement it. Currently it's dead spec prose.

---

## 2. Output Formatting

### Current output (full run)

```
🔍 Dry run — no changes will be written

═══ Consolidation (dry run) ═══
  Phase 1 — Reindex:    47 checked, 3 reindexed (2 new), 0 pruned
  Phase 2 — Strengthen: 12 boosted, 8 decayed, 1 extinction
  Phase 3 — Reorganize: 22 pairs, +3 edges, ~5 updated, -1 removed
  Phase 4 — Prune:      0 removed, 2 flagged, 0 misaligned
  Duration: 6.8s
```

### Issues

**2.1 — One-liner phase summaries lose information**

The spec's example output (lines 748–781 of CONSOLIDATION.md) has per-phase
duration, source counts, `new_files` counts distinct from `reindexed`, and
aggregate brain stats at the end. The actual implementation collapses all of
this into four compact lines that look like log output, not a report.

Specifically, Phase 1 prints `reindexed (N new)` but Phase 3 and 4 lose
granularity the spec promised:
- Phase 3: spec distinguishes edges_created vs edges_strengthened vs
  edges_decayed vs edges_invalidated. Code lumps edges_added + edges_updated
  + edges_removed.
- Phase 4: spec lists `auto_removed` + `flagged` + `misaligned` + `contradictions`.
  Code has `removed + flagged + misaligned` (no contradictions counter).

**2.2 — No pass number or timestamp**

The spec shows `Consolidation pass #47 — 2026-05-01T06:00:00Z`. The actual
header is just `═══ Consolidation complete ═══`. After 50 runs, knowing "which
run was this" and "when did it run" is genuinely useful, especially when cross-
referencing with `--history`.

**Recommendation:** Pull the run ID from `consolidation_log` after writing it:

```
═══ Consolidation #47 — 2026-05-09 06:00 ═══
```

**2.3 — History output is usable but dense**

```
  #3  2026-05-09 06:00  (6.8s)
    reindex: 12 indexed, 2 pruned | strengthen: 28 ↑ 142 ↓
    graph: +7 ~12 | prune: 1 removed, 3 flagged
```

The `↑` and `↓` arrows are cute but non-obvious without context (boosted vs
decayed). First-time readers won't parse them correctly. Also `graph: +7 ~12`
is cryptic — "plus 7 what? approximately 12 what?"

**Recommendation:** Spell it out, even if it wraps:

```
  #3  2026-05-09 06:00:11  (6.8s)
    reindex:    12 reindexed, 2 pruned
    strengthen: 28 boosted, 142 decayed, 3 extinction
    graph:      +7 edges added, ~12 updated
    prune:      1 removed, 3 flagged
```

Or add a legend line at the top:
```
  (reindex / strengthen: boosted↑ decayed↓ / graph: +added ~updated / prune)
```

**2.4 — No brain-state summary at the end**

The spec promises:
```
Total: 6.8s | Brain: 2,239 docs (+2) | Excitability μ=0.47 σ=0.19
```

The implementation just shows duration. The "docs delta" and excitability stats
would be highly informative — they let you see whether the brain is growing,
stable, or shrinking over time. These would require a quick `SELECT COUNT(*)`
and `AVG(excitability)` after the run, which are cheap queries.

**2.5 — Phase 4 flagged items are invisible**

The spec says Phase 4 should print human-readable per-document prune candidates:

```
  Flagged for review: 7 documents
    - mikoshi::Notes::3. Learning::old-class-notes  (excitability: 0.12, 0 accesses, 94 days old)
    - context::memories::rejected::some-old-thing    (excitability: 0.10, 0 accesses, 120 days old)
```

The actual output just says `2 flagged`. If something is flagged for human
review, the human needs to know *what* was flagged. A `--report` flag could
gate the full list, but without it there's no way to act on the output.

**Recommendation (short-term):** Print flagged items inline, gated on
`--verbose`. Add a `--report <file>` flag to write them to a file for longer
lists.

---

## 3. Error Messages

### Issues

**3.1 — `load_sources` fails silently on malformed TOML**

In `load_sources()`, if `sources.toml` is unparseable, the error is propagated
up to `main()` where it prints `Error: <toml parse error>`. That's fine. But
if the TOML parses but has no `[[source]]` entries (e.g. misspelled `[[sources]]`
array key), the code silently falls through to hardcoded defaults with no
warning. A user who typo'd their config will wonder why their custom sources
aren't being used.

```rust
// Current behaviour — silent fallback:
if let Some(sources) = parsed.get("source").and_then(|v| v.as_array()) {
    ...
    if !result.is_empty() {
        return Ok(result);
    }
}
// Falls through to hardcoded defaults. No warning emitted.
```

**Recommendation:** Warn when falling back:

```rust
eprintln!("Warning: {} exists but contains no [[source]] entries. \
           Using built-in defaults. Check your TOML key: [[source]] not [[sources]].",
           config_path.display());
```

**3.2 — Sources with non-existent paths produce no warning**

`load_sources()` builds `SourceDir` structs for every configured path without
checking whether the paths exist. If a source path is wrong (e.g.
`~/Jawz/mikoshi/Notes/` was moved), the consolidation run will just show
`0 checked, 0 reindexed` for that source, with no indication the path is broken.

**Recommendation:** In `cmd_consolidate`, before running, validate sources:

```rust
for src in &sources {
    if !src.path.exists() {
        eprintln!("Warning: source '{}' path does not exist: {}",
                  src.name, src.path.display());
    }
}
```

**3.3 — `ensure_brain` calls `process::exit(1)` instead of returning an error**

Most error paths in `main.rs` return `Err(...)` and let the top-level match
print it. But `ensure_brain()` calls `std::process::exit(1)` directly, which
bypasses any cleanup and produces no error code from `main()`. It also makes
`ensure_brain` impossible to unit-test.

```rust
fn ensure_brain(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        eprintln!("No brain at {}. Run `axel init` first.", path.display());
        std::process::exit(1);  // ← exits process, doesn't return Err
    }
    Ok(())
}
```

**Recommendation:** Return `Err`:

```rust
fn ensure_brain(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        return Err(format!(
            "No brain at {}. Run `axel init` first.",
            path.display()
        ).into());
    }
    Ok(())
}
```

**3.4 — `consolidation_log` SQL query has no error handling context**

`cmd_consolidate_history` runs a raw SQL `prepare()`. If it fails (schema
mismatch, no `consolidation_log` table on an old brain), the error message will
be a raw SQLite error string with no context about what the user should do.

**Recommendation:** Wrap with context:

```rust
let mut stmt = db.conn().prepare(...).map_err(|e| format!(
    "Could not read consolidation history (has this brain run consolidate yet?): {e}"
))?;
```

---

## 4. Missing Commands / Features

**4.1 — No `axel consolidate status` command**

Users want to ask "when did consolidation last run? is the timer working?"
without wading through `--history` output. A simple status line would help:

```
$ axel consolidate status
Last run: 2026-05-09 06:00:11 (#47, 6.8s)
Next run:  scheduled (systemd timer active)
Sources:   6 configured (4 exist, 2 missing)
```

This doesn't exist. `--history` gets you part of the way but it's more output
than needed for a quick status check.

**4.2 — No `axel consolidate sources` command (list/validate configured sources)**

Users should be able to verify what sources are configured and which paths
actually exist:

```
$ axel consolidate sources
Sources (from ~/.config/axel/sources.toml):
  ✓ mikoshi       ~/Jawz/mikoshi/Notes/       [high]    (1,284 files)
  ✓ context       ~/Jawz/data/context/        [high]    (142 files)
  ✓ notes         ~/Jawz/notes/               [medium]  (62 files)
  ✗ slack-diary   ~/Jawz/slack/diary/         [low]     PATH NOT FOUND
  ✓ memories      ~/.stelline/memkoshi/...    [medium]  (23 files)
```

Currently you can only discover what sources are configured by reading the TOML
file or running consolidation and observing which sources produce output.

**4.3 — No `axel consolidate report` command (view flagged documents)**

The prune phase flags documents for human review, but there's no command to see
them. They're buried in the database with no CLI surface. Even `--verbose`
doesn't expose them today.

**4.4 — MCP tool can't use `sources.toml` — hardcodes defaults**

The `axel_consolidate` MCP handler hardcodes the same six paths that are the
fallback defaults, instead of calling `load_sources()`. This means:

1. Editing `sources.toml` affects CLI runs but NOT MCP-triggered consolidations
2. The two paths are now in two places and will drift

```rust
// In mcp.rs — axel_consolidate handler (lines ~380-394):
let sources = vec![
    SourceDir { path: PathBuf::from(format!("{home}/Jawz/mikoshi/Notes/")), ... },
    // ... hardcoded, not reading sources.toml
];
```

This is a significant consistency bug. The fix is to call `load_sources(None)`
from `mcp.rs` (it's already a free function in `main.rs` — move it to
`consolidate/mod.rs` or a shared location).

**4.5 — Systemd service has no `AXEL_SOURCES` env var set**

The `.service` file sets `Environment=HOME=%h` but not `AXEL_SOURCES`. The
`load_sources()` resolution order checks `AXEL_SOURCES` before
`~/.config/axel/sources.toml`, but the service is relying on the file path
fallback. This works, but the spec's example service (CONSOLIDATION.md line 665)
shows `Environment=AXEL_SOURCES=%h/.config/axel/sources.toml` — the actual file
omits this. Minor, but creates a discrepancy between spec and reality.

**4.6 — `--history` is a flag on `consolidate`, not a standalone subcommand**

`axel consolidate --history` looks like it should run consolidation _then_ show
history. The mutually-exclusive intent isn't obvious. The implementation checks
the flag first and early-returns, but the CLI doesn't communicate this. A user
who runs `axel consolidate --dry-run --history` will be confused when the dry
run doesn't happen.

---

## 5. Spec vs Reality

| Spec feature | Implemented? | Notes |
|---|---|---|
| 4-phase consolidation pipeline | ✅ Yes | All 4 phases present |
| `--dry-run` flag | ✅ Yes | Works |
| `--history` flag | ✅ Yes | Works |
| `--sources` flag + `AXEL_SOURCES` env | ✅ Yes | Works |
| `--verbose` flag | ✅ Partial | Flag exists; depth of per-doc output unverified |
| `--since last` flag | ❌ No | Spec mentions it (§consolidation_log); not in CLI |
| `--report <file>` flag for prune output | ❌ No | Spec mentions it; not implemented |
| Run ID + timestamp in output header | ❌ No | Spec shows `#47 — 2026-05-01T06:00:00Z` |
| Per-phase duration in output | ❌ No | Spec shows per-phase timing; only total shown |
| Brain doc count + excitability stats at end | ❌ No | Spec shows `Brain: 2,239 docs (+2) \| μ=0.47` |
| Flagged-for-review list in output | ❌ No | Spec shows per-doc detail; code shows count only |
| `axel_consolidate` MCP tool reads `sources.toml` | ❌ No | Hardcodes defaults instead |
| Systemd service sets `AXEL_SOURCES` | ❌ No | Spec example shows it; actual file omits it |
| `--phase all` as explicit value in CLI | ❌ No | MCP tool accepts `"all"`; CLI only accepts omission |
| Competitive allocation details in output | ❌ No | Spec shows "Allocated: 4 new docs → 11 edges" |
| Phase 3 distinguishes strengthened vs decayed edges | ❌ No | Code lumps `edges_updated`; spec splits further |
| Phase 4 `contradictions` counter | ❌ No | `PruneStats` in spec has it; CLI output omits it |
| Backup warning before first run | ❌ No | Spec says "CLI should warn if no recent backup" |

**Summary:** ~8 of 15 spec features are missing from the CLI output/UX layer.
The backend _may_ compute some of these (e.g. contradictions) but they're
not surfaced.

---

## 6. `sources.toml` Documentation

### Current state

The file has a one-line comment at the top:

```toml
# Axel brain consolidation sources
# Used by: axel consolidate, axel-consolidate.service
```

That's it. No field documentation, no explanation of `priority` semantics, no
examples for alternative configurations.

### Issues

**6.1 — `priority` semantics are undocumented in the file**

What does `priority = "high"` actually do? A user reading the TOML has no idea
that priority controls:
- Processing order within Phase 1 (high sources reindex first)
- Prune thresholds in Phase 4 (high-priority sources never auto-pruned)

This is critical information — setting a source to `low` means its documents
can be auto-deleted if they go stale. Users should know that before they write
the config.

**6.2 — No comment explaining `~` expansion**

The file uses `~` in paths (`~/Jawz/mikoshi/Notes/`). This works because
`load_sources()` manually does `.replace("~", &home)`. But it's not standard
TOML behaviour, and it only expands `~` — it won't expand `$HOME` or other
env vars. A comment clarifying this prevents user confusion.

**6.3 — No indication of what file types are indexed**

Sources walk for `.md` and `.txt` files only. There's no comment in `sources.toml`
or in the `--sources` help text indicating this. A user who points a source at a
directory of `.rst` or `.org` files will get `0 checked` with no explanation.

**6.4 — No `enabled` flag**

To temporarily disable a source, a user must comment out the whole `[[source]]`
block. An `enabled = false` field would be a quality-of-life improvement and is
trivially added to `load_sources()` with:

```rust
let enabled = src.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
if !enabled { continue; }
```

### Recommended `sources.toml` header

```toml
# Axel brain consolidation sources
# Used by: axel consolidate (CLI) and axel-consolidate.service (systemd)
#
# Resolution order:
#   1. axel consolidate --sources <path>
#   2. $AXEL_SOURCES environment variable
#   3. This file (~/.config/axel/sources.toml)
#   4. Built-in defaults (same paths as below)
#
# Fields:
#   name     — identifier used in doc_ids (e.g. "mikoshi::path::to::file")
#   path     — directory to walk. Use ~ for home dir. Expands ~ only (not $HOME).
#   priority — "high" | "medium" | "low"
#              high:   processed first; never auto-pruned by Phase 4
#              medium: standard processing; flagged but not auto-pruned
#              low:    processed last; documents may be auto-pruned if stale
#   enabled  — (optional) set to false to skip this source without removing it
#
# Only .md and .txt files are indexed. Other extensions are ignored.
```

---

## 7. Assorted Nits

| # | Location | Issue |
|---|---|---|
| N1 | `main.rs:110–112` | Dangling comment: `/// Run as a SynapsCLI extension (JSON-RPC over stdio)` sits above the `Consolidate` variant but describes `Extension`. This is a clap doc-comment bug — that doc will appear in `axel consolidate --help`. |
| N2 | `main.rs:454` | Output uses `═══` box-drawing chars. `cmd_stats` also uses them. Consistent — good. But the `axel consolidate --history` header uses the same chars while the consolidation run output also uses them — so piping both could look like two sections of the same table. |
| N3 | `main.rs:455–463` | Phase labels are right-padded with spaces to align the colons. This will break if phase names or stats grow longer. Use `{:>12}` format alignment instead of manual spaces. |
| N4 | `mcp.rs` | The `axel_consolidate` MCP output omits `stats.reindex.new_files`, which exists on the struct. The CLI prints it; MCP doesn't. Minor inconsistency. |
| N5 | `axel-consolidate.service` | No `SyslogIdentifier=axel-consolidate` directive. Without it, `journalctl -u axel-consolidate` will work but the log entries will show the binary name, not the unit name. |
| N6 | `axel-consolidate.timer` | `RandomizedDelaySec=300` (5 min jitter) — good practice, but undocumented. Worth a comment: `# 5-min random delay prevents exact-clock contention with other timers`. |
| N7 | `load_sources()` | `HOME` fallback is hardcoded to `/home/haseeb`. Fine for now, but worth a `todo!()` comment or at least acknowledging it. |

---

## Priority Ranking (what to fix first)

| Priority | Fix | Effort |
|---|---|---|
| 🔴 High | MCP `axel_consolidate` reads hardcoded sources instead of `sources.toml` | Small — extract `load_sources()` to shared location, call it from `mcp.rs` |
| 🔴 High | `--history` flag semantics are confusing (hidden mode switch) | Small — update doc-comment to say "Does not run consolidation" |
| 🔴 High | Flagged prune candidates not printed anywhere | Medium — add `--verbose` or `--report` path to Phase 4 output |
| 🟡 Medium | `load_sources()` silent fallback when config has no entries | Tiny — add one `eprintln!` warning |
| 🟡 Medium | Source path existence not validated before run | Small — loop and `eprintln!` for missing paths |
| 🟡 Medium | `--phase` has no per-value descriptions | Small — expand long_help text |
| 🟡 Medium | Output missing run ID + timestamp | Small — thread through from `consolidation_log` |
| 🟡 Medium | `sources.toml` needs documented field semantics | Tiny — add comments to the file |
| 🟢 Low | `ensure_brain` uses `process::exit` instead of `Err` | Small refactor |
| 🟢 Low | History output uses cryptic `↑↓` and `+/-` without labels | Cosmetic |
| 🟢 Low | Systemd service missing `SyslogIdentifier` | One line |
| 🟢 Low | `--since` is in spec prose but nowhere in the CLI | Delete from spec or implement |
| 🟢 Low | Brain-state summary line missing from consolidation output | Medium — requires post-run query |
