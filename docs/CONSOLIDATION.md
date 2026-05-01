# Consolidation — Biologically-Inspired Memory Lifecycle for Axel

## Overview

Consolidation is the process by which Axel's brain evolves over time — not just
accumulating data, but actively strengthening, reorganizing, and pruning it.
The design is grounded in neuroscience research on how biological memory works.

Today Axel has two disconnected worlds: **documents** (VelociRAG — 2,238 indexed
files) and **memories** (Memkoshi — structured knowledge extracted by agents).
Documents have no access tracking, no decay, no concept of importance. Memories
have all of that but aren't connected to documents. Consolidation unifies them
under a single lifecycle.

### The Biological Basis

Six lines of neuroscience research inform this design:

| Principle | Source | Implication |
|-----------|--------|-------------|
| **Two-stage consolidation** | Squire & Alvarez 1995, Born & Wilhelm 2011 | Fast encoding (hippocampus) → slow integration (neocortex). Maps to staging → permanent store. |
| **Reconsolidation** | Alberini 2011, Suzuki et al. 2004 | Every retrieval makes a memory temporarily labile. Brief retrieval → strengthening. Prolonged retrieval without action → extinction. |
| **Memory allocation (CREB)** | Zhou et al. 2009, Park et al. 2016 | Neurons compete to encode memories. Winners are the most *excitable* at encoding time. Not random — biased by readiness. |
| **Hierarchical sleep replay** | Staresina et al. 2015, Kudrimoti et al. 1999 | Consolidation during sleep uses nested oscillations (SO → spindle → ripple) to replay memories in interleaved, topic-grouped batches. |
| **IEG pruning (Arc)** | Minatohara et al. 2016 | Active synapses are preserved; inactive ones at the same neuron are weakened. Contrast enhancement through selective pruning. |
| **Selective consolidation** | Born & Wilhelm 2011, Rasch & Born 2013 | The brain preferentially consolidates memories tagged as having future relevance. Not everything is worth keeping. |

Full research notes: `~/Jawz/notes/research/biological-memory-2026-05-01.md`

---

## Architecture

Consolidation is a **single process with four phases**, run periodically. It is
not four separate daemons — the phases feed into each other and share state
within a pass.

```
┌─────────────────────────────────────────────────────────────┐
│                    axel consolidate                          │
│                                                             │
│  Phase 1: REINDEX ──────────────────────────────────────    │
│  │  Walk source dirs, compare mtimes against indexed_at     │
│  │  Re-embed changed files (ripple = individual replay)     │
│  │  Prune documents whose files were deleted                │
│  │  New files get competitive allocation (link to           │
│  │  high-excitability neighbors via graph edges)            │
│  ▼                                                          │
│  Phase 2: STRENGTHEN ───────────────────────────────────    │
│  │  Read document_access log since last consolidation       │
│  │  Boost excitability for recently-accessed docs           │
│  │  Decay excitability for untouched docs                   │
│  │  Extinction signal: retrieved but never acted on         │
│  ▼                                                          │
│  Phase 3: REORGANIZE ───────────────────────────────────    │
│  │  Co-retrieval analysis: docs appearing in same search    │
│  │  results get edge weight bumps                           │
│  │  Interleaved processing across source namespaces         │
│  │  Topic clustering for graph coherence                    │
│  ▼                                                          │
│  Phase 4: PRUNE ────────────────────────────────────────    │
│  │  Apply decay curve to documents (extend memkoshi logic)  │
│  │  Flag never-accessed docs past age threshold             │
│  │  Remove contradicted/superseded documents                │
│  │  Report candidates for human review                      │
│  ▼                                                          │
│  Output: ConsolidateStats + updated brain                   │
└─────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **One process, four phases.** Not four separate tools. Phases share database
   connections and can be ordered optimally (reindex before strengthen, because
   strengthen needs up-to-date documents).

2. **Idempotent.** Running consolidation twice in a row produces the same result.
   The second run is a no-op (modulo clock progression for decay).

3. **Incremental.** Consolidation processes deltas, not the full corpus. The
   `index-sync` mtime comparison pattern extends to all phases.

4. **Non-destructive by default.** Phase 4 (prune) flags candidates but only
   auto-removes documents that meet strict criteria. Human review for ambiguous
   cases.

5. **Observable.** Every phase emits stats. The CLI prints a summary. A `--dry-run`
   flag shows what *would* happen without modifying anything.

---

## Schema Changes

### New: `document_access` table (in VelociRAG)

Mirrors `memory_access` in Memkoshi. Every search hit, every document open,
every reference in a conversation — all are retrieval events that trigger
reconsolidation logic.

```sql
CREATE TABLE IF NOT EXISTS document_access (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL,
    access_type TEXT NOT NULL,   -- 'search_hit' | 'opened' | 'referenced'
    query       TEXT,            -- the query that surfaced this doc (NULL for non-search)
    score       REAL,            -- search relevance score (NULL for non-search)
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    session_id  TEXT             -- which session triggered this access
);
CREATE INDEX IF NOT EXISTS idx_docaccess_doc_id ON document_access(doc_id);
CREATE INDEX IF NOT EXISTS idx_docaccess_timestamp ON document_access(timestamp);
```

**Population point:** `axel_search` in `mcp.rs` inserts a `search_hit` row for
every result returned. This is the primary data source for Phase 2.

### New columns on `documents`

```sql
ALTER TABLE documents ADD COLUMN access_count   INTEGER DEFAULT 0;
ALTER TABLE documents ADD COLUMN last_accessed   TIMESTAMP;
ALTER TABLE documents ADD COLUMN excitability    REAL DEFAULT 0.5;
```

- `access_count`: total retrieval events. Monotonically increasing.
- `last_accessed`: timestamp of most recent retrieval. Used for recency decay.
- `excitability`: the CREB analog. Range [0.0, 1.0]. Higher = more likely to be
  linked to new documents during Phase 1. Boosted by access, decayed by neglect.

**Migration:** Same pattern as `indexed_at` — `ALTER TABLE ADD COLUMN` with
defaults, idempotent, checked via `PRAGMA table_info`.

### New: `consolidation_log` table

```sql
CREATE TABLE IF NOT EXISTS consolidation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    phase1_reindexed  INTEGER DEFAULT 0,
    phase1_pruned     INTEGER DEFAULT 0,
    phase2_boosted    INTEGER DEFAULT 0,
    phase2_decayed    INTEGER DEFAULT 0,
    phase3_edges_added    INTEGER DEFAULT 0,
    phase3_edges_updated  INTEGER DEFAULT 0,
    phase4_flagged    INTEGER DEFAULT 0,
    phase4_removed    INTEGER DEFAULT 0,
    duration_secs     REAL
);
```

Provides audit trail. `consolidate --since last` uses the most recent entry's
`finished_at` to scope Phase 2 queries.

---

## Phase Details

### Phase 1: Reindex (Biological: Ripple-level replay)

**What:** Walk configured source directories, detect changed/new/deleted files,
re-embed as needed.

**Already built:** `cmd_index_sync()` in `main.rs` (S183, Case). Compares file
mtime against `indexed_at` column. Uses `INSERT OR REPLACE` for upserts.
Prunes entries whose `file_path` no longer exists on disk.

**New behavior during consolidation:**

1. **Competitive allocation for new files.** When a new file is indexed, after
   embedding it, find its K nearest neighbors in the existing index. Create
   `similar_to` graph edges to neighbors whose excitability exceeds a threshold
   (default 0.6). This biases new documents toward connection with "hot" areas
   of the brain — the CREB analog.

   ```rust
   // After indexing a new document:
   let neighbors = search.find_nearest(&embedding, 5)?;
   for neighbor in neighbors {
       if get_excitability(neighbor.doc_id) > 0.6 {
           db.insert_edge(
               &new_edge_id(), &new_doc_id, &neighbor.doc_id,
               "similar_to", neighbor.score, 0.8,
               &json!({"source": "consolidation_allocation"}),
               None, None, None,
           )?;
       }
   }
   ```

2. **Re-embed with context prefix.** When re-embedding a changed file, prepend
   a context prefix to the content before embedding. This provides pattern
   separation — same content from different sources produces different vectors.

   ```
   [source:mikoshi][path:3. Learning/CS646/Module 7] <content>
   ```

   This mimics grid cell orthogonalization: similar documents from different
   contexts get pushed apart in embedding space.

3. **Track what changed.** Return a `ReindexResult` with lists of new, updated,
   and pruned doc_ids. Phase 2 and 3 use this to scope their work.

**Source configuration:**

```rust
pub struct SourceDir {
    pub path: PathBuf,
    pub name: String,       // e.g. "mikoshi"
    pub priority: Priority, // High | Medium | Low
}
```

Priority determines processing order within Phase 1 (high-priority sources
reindex first) and feeds into Phase 4 (low-priority sources have stricter
prune thresholds).

### Phase 2: Strengthen (Biological: Reconsolidation)

**What:** Use retrieval history to adjust document excitability. Every search
hit is a reconsolidation event.

**The biological model:**

- Brief retrieval → reconsolidation → **strengthening**
- Retrieval without subsequent action → **extinction signal**
- Younger memories are more susceptible to modification
- Older memories are more resistant (temporal gradient)

**Algorithm:**

```rust
pub fn strengthen(db: &Database, since: DateTime<Utc>) -> StrengthenStats {
    // 1. Get all document_access events since last consolidation
    let accesses = db.get_document_accesses_since(since)?;

    // 2. Group by doc_id
    let grouped: HashMap<String, Vec<Access>> = group_by_doc_id(accesses);

    // 3. For each document with access events:
    for (doc_id, events) in grouped {
        let hit_count = events.len();
        let avg_score = events.iter().map(|e| e.score).sum::<f64>() / hit_count;
        let last_access = events.iter().map(|e| e.timestamp).max();

        // Boost: log-scaled, capped at 0.2 per consolidation cycle
        let boost = (0.05 * (hit_count as f64 + 1.0).log2()).min(0.2);

        // Score penalty: consistently low scores = misaligned embedding
        // (Arc pruning analog — weaken inactive synapses)
        if avg_score < SCORE_EXTINCTION_THRESHOLD {  // e.g. 0.02
            let penalty = 0.05;  // mild extinction signal
            update_excitability(doc_id, -penalty);
            stats.extinction_signals += 1;
        } else {
            update_excitability(doc_id, boost);
            stats.boosted += 1;
        }
    }

    // 4. Decay documents with NO access events since last consolidation
    //    but only if they're older than the grace period (14 days)
    let untouched = db.get_documents_without_access_since(since, grace_days=14)?;
    for doc in untouched {
        let weeks_inactive = doc.days_since_last_access() / 7.0;
        let penalty = (0.02 * weeks_inactive).min(0.3);
        let new_excitability = (doc.excitability - penalty).max(0.1);
        update_excitability(doc.id, new_excitability);
        stats.decayed += 1;
    }
}
```

**Constants (configurable):**

| Constant | Default | Rationale |
|----------|---------|-----------|
| `BOOST_SCALE` | 0.05 | Per-access log-scaled boost. Matches memkoshi decay.rs. |
| `BOOST_CAP` | 0.2 | Max boost per consolidation cycle. Prevents runaway. |
| `DECAY_RATE` | 0.02/week | Gentle decay for untouched docs. Floor at 0.1. |
| `GRACE_DAYS` | 14 | No decay for docs younger than 14 days. Matches memkoshi. |
| `SCORE_EXTINCTION_THRESHOLD` | 0.02 | Below this avg score = extinction signal. |
| `EXCITABILITY_FLOOR` | 0.1 | Never decays below this. Can always be found. |
| `EXCITABILITY_CEILING` | 1.0 | Hard cap. |

### Phase 3: Reorganize (Biological: Spindle-level grouping)

**What:** Maintain the knowledge graph based on co-retrieval patterns.
Documents that appear in the same search results should be connected.

**Co-retrieval tracking:**

When `axel_search` returns results, the set of doc_ids forms a **co-retrieval
group**. Over time, documents that repeatedly co-appear build stronger
associations — like neurons that fire together wiring together.

```sql
-- New table for tracking co-retrieval
CREATE TABLE IF NOT EXISTS co_retrieval (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id_a    TEXT NOT NULL,
    doc_id_b    TEXT NOT NULL,
    query       TEXT,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(doc_id_a, doc_id_b, query)
);
CREATE INDEX IF NOT EXISTS idx_coret_pair ON co_retrieval(doc_id_a, doc_id_b);
```

**Population:** In `axel_search`, after computing results, insert co-retrieval
pairs for the top-K results (K=5 to avoid combinatorial explosion):

```rust
// After search returns results:
let top_ids: Vec<&str> = results.iter().take(5).map(|r| r.doc_id.as_str()).collect();
for i in 0..top_ids.len() {
    for j in (i+1)..top_ids.len() {
        let (a, b) = if top_ids[i] < top_ids[j] {
            (top_ids[i], top_ids[j])
        } else {
            (top_ids[j], top_ids[i])
        };
        db.insert_co_retrieval(a, b, &query)?;
    }
}
```

**Phase 3 algorithm:**

```rust
pub fn reorganize(db: &Database, since: DateTime<Utc>) -> ReorganizeStats {
    // 1. Aggregate co-retrieval counts since last consolidation
    let pairs = db.get_co_retrieval_counts_since(since)?;
    // Returns: Vec<(doc_a, doc_b, count)>

    // 2. For pairs with count >= threshold (e.g. 3):
    for (doc_a, doc_b, count) in pairs.filter(|p| p.2 >= CO_RETRIEVAL_THRESHOLD) {
        let weight = (count as f64 / 10.0).min(1.0); // normalized
        db.upsert_edge(&Edge {
            id: format!("coret_{}_{}", hash(doc_a), hash(doc_b)),
            source_id: doc_a,
            target_id: doc_b,
            edge_type: "co_retrieved".to_string(),
            weight,
            confidence: 0.7, // co-retrieval is a soft signal
            ..Default::default()
        })?;
        stats.edges_updated += 1;
    }

    // 3. Decay old co-retrieval edges that haven't been reinforced
    let stale_edges = db.get_edges_by_type_older_than("co_retrieved", stale_days=30)?;
    for edge in stale_edges {
        if edge.weight < 0.2 {
            db.invalidate_edge(&edge.id)?;
            stats.edges_removed += 1;
        } else {
            // Reduce weight but don't remove
            db.update_edge_weight(&edge.id, edge.weight * 0.8)?;
            stats.edges_decayed += 1;
        }
    }
}
```

**Interleaving (the Kudrimoti principle):**

Phase 3 processes source namespaces in **round-robin** order, not sequentially.
If there are 100 pairs from mikoshi and 50 from context, process them as:
mikoshi[0], context[0], mikoshi[1], context[1], ... This prevents the graph
from becoming dominated by the largest source.

**Constants:**

| Constant | Default | Rationale |
|----------|---------|-----------|
| `CO_RETRIEVAL_THRESHOLD` | 3 | Minimum co-appearances to create an edge. |
| `CO_RETRIEVAL_TOP_K` | 5 | Only top-5 results generate co-retrieval pairs. |
| `EDGE_STALE_DAYS` | 30 | Unreinforced edges older than this get decayed. |
| `EDGE_DECAY_FACTOR` | 0.8 | Multiplicative decay per consolidation cycle. |
| `EDGE_REMOVAL_THRESHOLD` | 0.2 | Below this weight, edges are invalidated. |

### Phase 4: Prune (Biological: Arc-mediated contrast enhancement)

**What:** Remove noise, flag stale content, clean up the brain.

**Pruning criteria (all must be met for auto-removal):**

1. `excitability < 0.15` (well below floor — suggests chronic irrelevance)
2. `access_count == 0` (never retrieved — truly dead)
3. `age > 60 days` (not a new document that hasn't had time to be found)
4. `source.priority == Low` (high-priority sources are never auto-pruned)

Documents meeting criteria 1-3 but from high-priority sources are **flagged**
(added to a `prune_candidates` report) but not removed.

**The Arc principle — inverse tagging:**

Rather than only strengthening accessed documents, Phase 4 actively identifies
documents whose embeddings are *misaligned* — they appear in search results
but consistently score poorly. These documents pollute the search space:

```rust
// Find documents that appear in search results but always score low
let misaligned = db.query(
    "SELECT doc_id, COUNT(*) as hits, AVG(score) as avg_score
     FROM document_access
     WHERE access_type = 'search_hit'
     GROUP BY doc_id
     HAVING hits >= 5 AND avg_score < 0.015"
)?;

for doc in misaligned {
    // Option A: re-embed with updated content (maybe content drifted)
    // Option B: re-chunk into smaller pieces (maybe too broad)
    // Option C: flag for review
    stats.misaligned_flagged += 1;
}
```

**Contradiction resolution:**

Leverage Memkoshi's existing contradiction detection (`pipeline.rs`) extended
to documents. When two documents make contradictory claims and one is newer:
- The newer document's `excitability` gets a boost
- The older document gets an extinction signal
- Both are flagged for human review (auto-resolution is dangerous for documents)

**Output:**

```rust
pub struct PruneStats {
    pub auto_removed: usize,    // met all criteria, gone
    pub flagged: usize,         // met some criteria, needs review
    pub misaligned: usize,      // consistently poor search scores
    pub contradictions: usize,  // detected contradictory documents
}
```

Phase 4 writes a human-readable report to stdout (or a file with `--report`):

```
Phase 4: Prune
  Auto-removed: 3 documents (all low-priority, zero access, excitability < 0.15)
  Flagged for review: 7 documents
    - mikoshi::Notes::3. Learning::old-class-notes  (excitability: 0.12, 0 accesses, 94 days old)
    - context::memories::rejected::some-old-thing    (excitability: 0.10, 0 accesses, 120 days old)
    ...
  Misaligned embeddings: 2 documents (>5 search hits, avg score < 0.015)
  Contradictions detected: 0
```

---

## CLI Interface

```bash
# Full consolidation pass (all 4 phases)
axel --brain ~/.config/axel/axel.r8 consolidate

# Dry run — show what would change without modifying anything
axel --brain ~/.config/axel/axel.r8 consolidate --dry-run

# Run specific phases only
axel --brain ~/.config/axel/axel.r8 consolidate --phase reindex
axel --brain ~/.config/axel/axel.r8 consolidate --phase strengthen
axel --brain ~/.config/axel/axel.r8 consolidate --phase reorganize
axel --brain ~/.config/axel/axel.r8 consolidate --phase prune

# Override source configuration
axel --brain ~/.config/axel/axel.r8 consolidate --sources ./consolidation-sources.toml

# Show consolidation history
axel --brain ~/.config/axel/axel.r8 consolidate --history

# Verbose output (per-document details)
axel --brain ~/.config/axel/axel.r8 consolidate -v
```

**Source configuration file** (`consolidation-sources.toml`):

```toml
# Default: ~/.config/axel/sources.toml
# Overridable via --sources flag or AXEL_SOURCES env var

[[source]]
name = "mikoshi"
path = "~/Jawz/mikoshi/Notes/"
priority = "high"

[[source]]
name = "context"
path = "~/Jawz/data/context/"
priority = "high"

[[source]]
name = "notes"
path = "~/Jawz/notes/"
priority = "medium"

[[source]]
name = "slack-diary"
path = "~/Jawz/slack/diary/"
priority = "low"

[[source]]
name = "memories-legacy"
path = "~/Jawz/data/context/memories/permanent/"
priority = "medium"

[[source]]
name = "memories"
path = "~/.stelline/memkoshi/exports/"
priority = "medium"
```

---

## MCP Integration

### Search hit logging (reconsolidation trigger)

`axel_search` in `mcp.rs` currently returns results and forgets about them.
After this change, every search also records access events:

```rust
// In handle_axel_search(), after computing results:
for result in &response.results {
    db.log_document_access(
        &result.doc_id,
        "search_hit",
        Some(&query),
        Some(result.score),
        session_id.as_deref(),
    )?;
    db.increment_access_count(&result.doc_id)?;
}
```

### New MCP tool: `axel_consolidate`

Expose consolidation as an MCP tool so agents can trigger it:

```json
{
    "name": "axel_consolidate",
    "description": "Run a consolidation pass on the brain. Reindexes changed files, strengthens accessed documents, reorganizes graph edges, and prunes stale content.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "phase": {
                "type": "string",
                "enum": ["all", "reindex", "strengthen", "reorganize", "prune"],
                "description": "Which phase to run. Default: all."
            },
            "dry_run": {
                "type": "boolean",
                "description": "Preview changes without applying them."
            }
        }
    }
}
```

---

## Code Layout

```
axel/src/
├── consolidate/
│   ├── mod.rs              # Consolidator struct, orchestration, ConsolidateStats
│   ├── reindex.rs          # Phase 1 — mtime sync, competitive allocation
│   ├── strengthen.rs       # Phase 2 — retrieval-based reconsolidation
│   ├── reorganize.rs       # Phase 3 — co-retrieval graph maintenance
│   └── prune.rs            # Phase 4 — decay, cleanup, misalignment detection
├── brain.rs                # Add consolidate() method → delegates to module
├── mcp.rs                  # Add search hit logging + axel_consolidate tool
├── main.rs                 # Add Consolidate subcommand
└── ...

velocirag/src/
├── db.rs                   # Add document_access table, co_retrieval table,
│                           # access tracking methods, consolidation_log table
└── ...

memkoshi/src/
├── decay.rs                # Extract shared decay math into reusable functions
│                           # (both memkoshi and consolidator use same curves)
└── ...
```

### New public API on `AxelBrain`

```rust
impl AxelBrain {
    /// Run a full or partial consolidation pass.
    pub fn consolidate(&mut self, opts: ConsolidateOptions) -> Result<ConsolidateStats>;

    /// Get consolidation history.
    pub fn consolidation_history(&self, limit: usize) -> Result<Vec<ConsolidationRun>>;
}

pub struct ConsolidateOptions {
    pub sources: Vec<SourceDir>,
    pub phases: HashSet<Phase>,     // default: all
    pub dry_run: bool,
    pub verbose: bool,
}

pub enum Phase {
    Reindex,
    Strengthen,
    Reorganize,
    Prune,
}

pub struct ConsolidateStats {
    pub reindex: ReindexStats,
    pub strengthen: StrengthenStats,
    pub reorganize: ReorganizeStats,
    pub prune: PruneStats,
    pub duration: Duration,
}
```

---

## Scheduling

Consolidation should run automatically but not continuously. Recommended
triggers:

| Trigger | When | Rationale |
|---------|------|-----------|
| **Boot** | Every `jawz-boot` | Catches overnight file changes. Phase 1 only (~4s). |
| **Systemd timer** | Every 6 hours | Full 4-phase pass. Matches biological sleep cycle frequency. |
| **Manual** | `axel consolidate` | On-demand, any time. |
| **Post-session** | Jawz shutdown | Consolidate documents modified during the session. |
| **Watcher** | On significant file changes | Future — inotify-triggered, debounced. |

**Systemd timer example:**

```ini
# ~/.config/systemd/user/axel-consolidate.timer
[Unit]
Description=Axel brain consolidation

[Timer]
OnCalendar=*-*-* 00,06,12,18:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

```ini
# ~/.config/systemd/user/axel-consolidate.service
[Unit]
Description=Axel brain consolidation pass

[Service]
Type=oneshot
ExecStart=%h/Projects/axel/target/release/axel --brain %h/.config/axel/axel.r8 consolidate
Environment=AXEL_SOURCES=%h/.config/axel/sources.toml
```

---

## Implementation Order

Ordered by impact and dependency chain:

### Step 1: Schema + Search Logging (foundation)
- Add `document_access` table to `db.rs` (migration in `init_schema()`)
- Add `access_count`, `last_accessed`, `excitability` columns to `documents`
- Add `consolidation_log` table
- Hook `axel_search` in `mcp.rs` to log every search hit
- **~100 lines across 2 files. Every search starts generating data immediately.**

### Step 2: Consolidate CLI + Phase 1 (reindex)
- Create `axel/src/consolidate/mod.rs` with `Consolidator` struct
- Extract `cmd_index_sync` logic into `consolidate/reindex.rs`
- Add competitive allocation (link new docs to high-excitability neighbors)
- Add `Consolidate` subcommand to `main.rs`
- Add source config parsing (`sources.toml`)
- **~300 lines. Consolidation becomes a real command.**

### Step 3: Phase 2 (strengthen)
- Implement `consolidate/strengthen.rs`
- Port decay math from `memkoshi/decay.rs` into shared utility
- Excitability boost/decay based on `document_access` data
- **~150 lines. Documents start having different importance levels.**

### Step 4: Phase 3 (reorganize)
- Add `co_retrieval` table to `db.rs`
- Hook co-retrieval logging into `axel_search`
- Implement `consolidate/reorganize.rs`
- Edge creation, decay, interleaved processing
- **~200 lines. The graph becomes dynamic.**

### Step 5: Phase 4 (prune)
- Implement `consolidate/prune.rs`
- Auto-removal for documents meeting all criteria
- Misalignment detection
- Human-readable report output
- **~150 lines. The brain gets self-cleaning.**

### Step 6: Polish
- `axel_consolidate` MCP tool
- `--dry-run` flag
- `--history` flag
- Systemd timer setup
- Integration tests
- **~200 lines. Production-ready.**

**Total estimated: ~1,100 lines of new Rust across ~10 files.**

---

## Safety

1. **No auto-deletion of high-priority sources.** Mikoshi and context files are
   never auto-pruned. Phase 4 only flags them.

2. **Consolidation log provides audit trail.** Every run is recorded with stats.
   If something goes wrong, you can see exactly when.

3. **`--dry-run` is always available.** First consolidation on a brain should
   always be a dry run.

4. **Excitability has a floor of 0.1.** Documents never become completely
   invisible — they can always be found by exact query.

5. **Graph edge invalidation, not deletion.** `valid_to` is set rather than
   row deletion. Edges can be forensically examined.

6. **Backup before first consolidation.** The CLI should warn (not block) if
   no recent backup exists.

---

## Metrics & Observability

```
$ axel --brain axel.r8 consolidate
Consolidation pass #47 — 2026-05-01T06:00:00Z

  Phase 1: Reindex
    Sources:    6 (3 high, 2 medium, 1 low)
    Checked:    1,488 files
    Reindexed:  12 (8 modified, 4 new)
    Pruned:     2 (deleted from disk)
    Allocated:  4 new docs → 11 graph edges created
    Duration:   6.2s

  Phase 2: Strengthen
    Access events:  47 search hits across 31 documents
    Boosted:        28 documents (avg +0.08 excitability)
    Extinction:     3 documents (avg score < 0.015)
    Decayed:        142 documents (no access in 14+ days)
    Duration:       0.3s

  Phase 3: Reorganize
    Co-retrieval pairs:   89
    Edges created:        7 (new co-retrieval links)
    Edges strengthened:   12
    Edges decayed:        4
    Edges invalidated:    1
    Duration:             0.2s

  Phase 4: Prune
    Auto-removed:     1
    Flagged:          3 (see --report for details)
    Misaligned:       0
    Contradictions:   0
    Duration:         0.1s

  Total: 6.8s | Brain: 2,239 docs (+2) | Excitability μ=0.47 σ=0.19
```

---

## Open Questions

1. **Should context prefixes in embeddings be retroactive?** If we add
   `[source:mikoshi]` prefixes to embedding input, all existing embeddings
   become stale. A one-time full re-embed (~6 min) is needed. Worth it?

2. **Should co-retrieval be bidirectional or directed?** Currently proposed as
   bidirectional (A↔B). But if doc A surfaces doc B but not vice versa, that's
   a directed relationship. More complex, more accurate.

3. **Should the extinction signal trigger re-embedding?** A document with
   consistently low search scores might just need a better embedding, not
   removal. Re-embed with current model could fix it without pruning.

4. **Consolidation during active sessions?** If an agent is using the brain
   while consolidation runs, SQLite WAL handles concurrent reads, but write
   contention could slow both. Should consolidation acquire an advisory lock?

5. **Memory ↔ document unification.** Currently memories and documents are
   separate tables with separate access tracking. Long-term, should they
   converge into a single entity type with different "memory kinds"?
