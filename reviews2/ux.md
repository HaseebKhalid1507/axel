# Axel CLI — UX Review
**Reviewer:** Star-Lord (UX Subagent) · **Date:** 2026-05-01  
**Binary:** `target/release/axel` · **Source:** `crates/axel/src/main.rs`

---

## TL;DR

Axel's CLI is *mostly* solid. The fundamentals are there — consistent error handling, machine-readable JSON modes, a sensible subcommand taxonomy. But there are specific rough edges that will frustrate users: a silent footgun in `handoff`, confusing `recall` dual-mode, inconsistent formatting across commands, and zero guidance for new users on onboarding flow. The bar for "ship it" is close. These are fixable in an afternoon.

**Overall Score: 7.5 / 10** — Competent but not delightful.

---

## 1. Help Text — Does it Explain What's Going On?

### ✅ What Works

The top-level `--help` is clean and scannable. Commands are grouped logically enough. The one-liner descriptions are accurate. `consolidate` is the standout — its long-help (`--help` vs `-h`) shows real depth: phases, behavior, what `--verbose` actually adds. That two-tier help system is the right pattern.

```
Run a consolidation pass on the brain
Reindexes changed files, strengthens accessed documents,
reorganizes graph edges, and prunes stale content.
```

Good. Tells you the *what* and *why* in one breath.

### ❌ What's Missing or Misleading

**`recall` has a secret dual-mode that help doesn't explain.**  
The command does two completely different things: no-args = "boot context summary", with args = "semantic search". This is the most confusing UX in the whole binary. The help says:

```
Optional search query (omit for boot context)
```

"Boot context" is jargon. A new user has no idea what that means or why omitting a query changes the entire output format. **Recommendation:** Rename the modes or add an explicit flag:

```
axel recall                    → (boot context — session handoff + recent memories)
axel recall --query <terms>    → (semantic search shortcut)
```

Or just point users at `axel search` for the search case and keep `recall` as boot-context only.

---

**`handoff` treats unknown actions as content silently.**  
This is the most dangerous behavior in the codebase:

```rust
other => {
    // Treat unknown action as "set" with action word included
    let content = std::iter::once(other.to_string())
        .chain(content_parts.iter().cloned())
        ...
    storage.set_context("handoff", &content, "boot")?;
    println!("✓ Handoff set ({} chars)", content.len());
}
```

```bash
$ axel handoff badaction
✓ Handoff set (9 chars)    # <-- SILENTLY STORES "badaction" AS HANDOFF CONTENT
```

Tested and confirmed. A typo like `axel handoff Set "important context"` (capital S) silently stores `"Set important context"` as handoff. You'd never know. **This must reject unknown actions with an error.** The valid actions should also be surfaced in `--help`:

```
[ACTION]  Subcommand: set, get, or clear [default: get]
```

That `[possible values: set, get, clear]` clap annotation is right there on `consolidate --phase`. Use it here too.

---

**`remember` categories are totally opaque.**  
```bash
$ axel remember "some note" --category ???
```
There's no list of valid categories anywhere in the help. You just get a validation error at runtime if you guess wrong. Add `[possible values: ...]` the same way `consolidate --phase` does it.

---

**`index` silently skips files under 50 chars — no feedback.**  
```rust
if content.len() < 50 { continue; }
```
No warning, no count of skipped files. The output just says `✓ Indexed 3 documents` when you had 8. Users will wonder where their files went. The `index-sync` summary has a `skipped=N` field but the basic `index` command doesn't. **Add a skipped count.**

---

**`suggest` with no co-retrieval history is a dead end with no next step.**  
```
No co-retrieval connections found for: some-doc
(Run more searches to build the co-retrieval graph)
```
This is okay but could be better. What searches? How many? Point them at `axel search` explicitly. New users won't understand what "co-retrieval graph" means.

---

## 2. Error Messages — Are They Actionable?

### ✅ Good

| Scenario | Message | Grade |
|---|---|---|
| Brain not found | `No brain at /path/axel.r8. Run \`axel init\` first.` | ✅ Perfect |
| Brain already exists | `Brain already exists. To start fresh, delete it first.` | ✅ Clear |
| Empty search query | `Please provide a search query.` | ✅ |
| Empty memory | `Please provide memory content.` | ✅ |
| Empty handoff set | `Please provide handoff content.` | ✅ |
| No handoff get | `No handoff set.` | ✅ |
| index-sync on file | `index-sync requires a directory, got: /path` | ✅ |
| Memory not found | `Memory not found: mem_xyz` | ✅ |

The "no brain" error with the `init` hint is genuinely good. First-time user experience handled.

### ❌ Needs Work

**`index` on a bad path gives a raw OS error:**
```bash
$ axel index /nonexistent/path
Error: No such file or directory (os error 2)
```
Compare this to every other error in the tool. This one has no path context, no hint. Should be:
```
Error: Path not found: /nonexistent/path
       Check the path exists and is readable.
```

**Validation errors from `remember` surface as bare bullets with no recovery hint:**
```
Validation failed:
  • content too short
```
(Hypothetical — the actual content is unknown without triggering it.) No guidance on what "too short" means or what the minimum is. The error should include the threshold.

**`index-sync` prefix mismatch (canonical vs stored path)** — not surfaced to users at all. If the stored `file_path` was originally indexed with a relative path and `index-sync` canonicalizes to absolute, no files match and nothing gets pruned or re-indexed. Silent divergence. This is a correctness bug with UX consequences (user thinks sync ran, brain is actually stale).

---

## 3. Missing Features Users Would Expect

### 🔴 High Priority

**1. `axel search --min-score` / score threshold filtering**  
Search results include scores as low as `0.013`. That's noise. There's no way to say "only show me results above 0.5." Users piping results to scripts get flooded with garbage. A `--min-score` flag would make the JSON output scriptable. Currently, the 100-result cap is the only filtering mechanism.

**2. `axel forget` should support removing indexed documents too**  
Right now `forget` only removes memories (`mem_*` IDs). If you index a file and want to remove it from the brain, there's no command. You'd have to manually delete the file and wait for `index-sync --prune`. A `axel remove <doc_id>` or `axel forget --doc <doc_id>` would complete the lifecycle.

**3. `axel remember` doesn't echo back what was stored**  
```
✓ Remembered: some note content truncated to (mem_abc123)
  category: Events | topic: general
```
The title truncation to 60 chars means you can't verify the memory stored correctly from the confirmation message alone. Add `axel memories --id mem_abc123` support or show the full content on confirm.

**4. No `axel list` or `axel index list` for indexed documents**  
You can list memories with `axel memories`. You can get stats with `axel stats`. But there's no way to list indexed documents except by running a search. Discoverability gap for users who want to audit what's in the brain.

### 🟡 Medium Priority

**5. `recall` has no `--limit` flag**  
The boot-context path hardcodes `list_memories(5)`. Five is a reasonable default but it should be configurable:
```bash
axel recall --limit 10
```

**6. `consolidate --dry-run` doesn't show *what would change*, only counts**  
Dry run says "3 files would be reindexed" but not *which* files. Combined with `--verbose` it still doesn't show paths. The value of dry run is seeing the specific changes — otherwise it's just a metadata diff.

**7. `handoff` has no `--append` mode**  
Common workflow: multiple agents want to contribute to the handoff. Right now each `set` overwrites. An `append` action would be immediately useful.

**8. `stats` shows top queries but not top documents**  
The most-accessed documents are arguably more interesting than the most-searched queries. `excitability` covers this partially but it's a separate command. A "Top accessed documents" section in `stats` would complete the picture without needing another command.

---

## 4. Output Formatting Consistency

The tool has a personality — `✓`, `🔍`, `═══`, `──` — and mostly applies it consistently. But there are some cracks.

### Inconsistency Table

| Command | Uses `═══` header | Uses `✓` prefix | Has timing | JSON mode |
|---|---|---|---|---|
| `search` | ❌ | ❌ | ✅ (ms) | ✅ |
| `stats` | ✅ | ❌ | ❌ | ❌ |
| `recall` | ❌ | ❌ | ❌ | ❌ |
| `remember` | ❌ | ✅ | ❌ | ❌ |
| `excitability` | ✅ | ❌ | ❌ | ❌ |
| `suggest` | ✅ | ❌ | ❌ | ❌ |
| `consolidate` | ✅ | ❌ | ✅ (s) | ✅ |
| `index` | ❌ | ✅ | ✅ (s) | ❌ |
| `index-sync` | ❌ | ✅ | ✅ (s) | ❌ |
| `memories` | ❌ | ❌ | ❌ | ❌ |

**Issues:**

- `search` and `recall` (query mode) produce nearly identical output formats but look slightly different (`🔍 "query" — N results` vs `🔍 "query" — N results\n` — one has a blank line, one doesn't).
- `memories` has no section header at all. Output just starts. Compare to `excitability` which has `═══ Excitability Distribution ═══`. Pick a lane.
- `consolidate` outputs timing in seconds, `search` in milliseconds. Neither is wrong but it's worth noting the inconsistency for scripting users.
- `recall` (boot context) uses `[Handoff]`, `[Recent Memories]`, `[Stats]` as headers — square bracket style. No other command does this. If that's intentional (machine-parseable markers for agent consumption), make it explicit in the docs. If not, align the style.

### The `recall` content truncation is too aggressive

```rust
let preview: String = clean.chars().take(200).collect();
println!("[{}] {preview}…\n", result.doc_id);
```

`search` previews 120 chars across 3 lines. `recall` (query mode) previews 200 chars in one block. Different commands, different truncation, no consistency. Establish a preview standard and stick to it.

---

## 5. Edge Cases in User Input

| Input | Behavior | Verdict |
|---|---|---|
| `axel search ""` | `Please provide a search query.` + exit 1 | ✅ |
| `axel search` (no args) | Same — `Vec<String>` empty, caught by `.is_empty()` check | ✅ |
| `axel remember` (no content) | `Please provide memory content.` | ✅ |
| `axel handoff badaction` | **Silently stores "badaction" as handoff** | 🔴 Bug |
| `axel handoff set` (no content) | `Please provide handoff content.` | ✅ |
| `axel suggest` (no query) | `Please provide a query or document ID.` | ✅ |
| `axel index /nonexistent` | Raw OS error, no context | 🟡 |
| `axel index-sync /file.txt` | `index-sync requires a directory, got: /file.txt` | ✅ |
| `axel forget mem_doesnotexist` | `Memory not found: mem_doesnotexist` + exit 1 | ✅ |
| `axel search` with 100+ word query | Handled — joins tokens, no crash | ✅ |
| No `AXEL_BRAIN` + no `--brain` | Falls back to `~/.config/axel/axel.r8` | ✅ (but undocumented default) |
| `axel init` when brain exists | Clear error + hint | ✅ |
| `axel remember` very long string | Truncates title to 60 chars silently | 🟡 |

**Most critical edge case — `handoff` unknown action:** tested and confirmed above. This is a real bug, not a nitpick. An agent calling `axel handoff Set "context"` (wrong case) silently corrupts the handoff. Must be fixed.

**Undocumented default brain path:** The `--help` shows `[env: AXEL_BRAIN=]` with an empty value, which is technically accurate but gives zero hint that there's a `~/.config/axel/axel.r8` fallback. A new user who doesn't set the env var and doesn't pass `--brain` has no idea their data is going to a default location. The help should say: `[env: AXEL_BRAIN] [default: ~/.config/axel/axel.r8]`.

---

## 6. Onboarding Flow — The Missing Piece

There's no onboarding. A brand new user sees:

```bash
$ axel --help
```

And gets a list of 14 commands with no suggested starting point. The README presumably covers this, but the CLI itself could do better.

**Suggested quick win:** Add a "Getting started" hint to `axel --help` footer, or make `axel` with no subcommand (currently an error) print a minimal quickstart:

```
New here? Try:
  axel init --name myagent        # create your brain
  axel index ~/notes              # index some documents
  axel search "what I care about" # find things
  axel remember "important fact"  # store a memory
```

Clap supports `after_help` — this is a one-liner to add.

---

## Priority Fix List

| Priority | Issue | Effort |
|---|---|---|
| 🔴 P0 | `handoff` unknown action silently stores content | ~5 min |
| 🔴 P0 | `index` bad path gives raw OS error | ~10 min |
| 🔴 P1 | `remember` categories not listed in help | ~5 min |
| 🟡 P1 | Undocumented default brain path | ~5 min |
| 🟡 P1 | `recall` dual-mode is confusing — needs clearer docs/flags | ~30 min |
| 🟡 P2 | `index` doesn't report skipped files count | ~10 min |
| 🟡 P2 | `recall` hardcoded `--limit 5` | ~5 min |
| 🟡 P2 | `search --min-score` for result filtering | ~20 min |
| 🟢 P3 | CLI onboarding quickstart hint | ~15 min |
| 🟢 P3 | `forget --doc` for removing indexed documents | ~45 min |
| 🟢 P3 | Output formatting consistency pass | ~60 min |
| 🟢 P3 | `consolidate --dry-run` show specific files | ~30 min |

---

## What's Actually Good (Don't Touch)

- The `init` → `index` → `search` path works cleanly end-to-end with helpful feedback at each step.
- Brain-not-found error with `axel init` hint is exactly right.
- `consolidate` is the most polished command — four phases, dry-run, verbose, JSON, history, report file. That's feature-complete.
- `excitability` output with bar charts is genuinely informative and fun to read.
- `stats` top-queries section is a nice touch — the brain telling you what it thinks about most.
- JSON output on `search` and `consolidate` is properly structured for scripting.
- `index-sync` stats line (`checked=N reindexed=N (new=N) pruned=N`) is exactly the right level of detail.

The bones are strong. Fix the handoff bug, clean up the rough edges, and this CLI earns a 9/10.

---

*Review complete. Rocket would've caught the handoff bug in five minutes. Just saying.*
