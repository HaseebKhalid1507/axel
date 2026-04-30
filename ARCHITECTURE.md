# Axel — Architectural Decisions

## Decision 1: ONNX Model Storage

### The Problem
The embedding model (all-MiniLM-L6-v2, 87MB ONNX + 700KB tokenizer) and optional reranker 
(TinyBERT, similar size) are required to produce and search embeddings. Where do they live?

### Options Considered

**A) Bundle inside every .r8 file**
- +True portability: one dir = everything
- -Every agent brain costs +90MB minimum
- -5 agents = 450MB of duplicate models
- -Model upgrades require updating every .r8
- -The model is RUNTIME, not DATA. You don't put the SQLite engine inside a .db file.

**B) System-level cache (`~/.cache/axel/models/`)**
- +Install once, all .r8 files share it
- +.r8 files stay lean (data only)
- +Model upgrades don't touch .r8 files
- -Not portable without also copying models
- -First-run requires download

**C) System cache + manifest coupling**
- .r8 manifest records: `{ "embedder": "all-MiniLM-L6-v2", "dim": 384 }`
- On open: validate system has the right model, auto-download if missing
- Model mismatch = refuse to open (embeddings are model-coupled)
- +Lean files, auto-resolving, version-aware
- +Portable in practice (auto-downloads ~90MB on first use on new machine)
- +Clean separation of data vs runtime

### Decision: Option C
Models at `~/.cache/axel/models/`, auto-downloaded from HuggingFace on first use.
velocirag-rs already has `download.rs` with exactly this pattern. Just change the cache path.

The .r8 manifest locks the model version. If you open a .r8 built with model X and 
your system has model Y, Axel downloads model X. Embeddings are sacred — they're 
coupled to the model that produced them.

---

## Decision 2: .r8 as Single File vs Directory

### The Problem
The spec says .r8 is a directory bundle. But "one file = one brain" is the pitch.

### Data

Current Jawz brain (Python VelociRAG):
- store.db: 48MB (7,104 documents with embeddings)
- index.faiss: 11MB (ANN accelerator)
- graph.db: 251MB (3,502 nodes, 1,913 edges — bloated, mostly note content)
- metadata.db: 484KB

The FAISS/USearch index is REBUILDABLE from embeddings stored in SQLite.
For 7,104 docs at 384-dim, rebuild takes ~70ms (USearch does ~100K vectors/sec).

### Options Considered

**A) Directory bundle (.r8/ with store.db + vectors.usearch + manifest.json)**
- +Standard approach, easy to inspect
- -Not actually "one file"
- -Sync issues between SQLite and USearch on crash

**B) Single SQLite file IS the .r8**
- +Truly one file. Copy, backup, move — dead simple.
- +ACID guarantees on everything (no split-brain between db and index)
- +SQLite is the most deployed database engine on Earth. Every machine can read it.
- -Cold start: must rebuild USearch HNSW in memory from stored embeddings
- -For 7K docs: ~70ms rebuild. For 50K docs: ~500ms. Acceptable.
- -USearch index cached at `~/.cache/axel/<sha256>.usearch` for hot starts

**C) Single file with USearch embedded as SQLite blob**
- -USearch needs memory-mapped file access. Can't mmap a blob.
- -Would need to extract to temp file anyway. Pointless.

### Decision: Option B — .r8 IS a single SQLite file

The graph.db / metadata.db / store.db split in Python VelociRAG is historical accident.
In velocirag-rs, it's already ONE SQLite file. Just add the Memkoshi tables to it.

Cold start path: open SQLite → load all embeddings → build HNSW in memory → cache to disk.
Hot start path: open SQLite → mmap cached HNSW → ready in <10ms.

The manifest becomes a `schema_info` table inside SQLite. No separate file needed.

**This is the right call because:**
1. Embeddings of record live in SQLite. USearch is just an accelerator.
2. One file = one brain. The pitch IS the implementation.
3. SQLite WAL gives concurrent readers + single writer. No sync issues.
4. Schema migrations are just SQL. 
5. The file IS the API. `sqlite3 axel.r8 "SELECT * FROM memories"` just works.

---

## Decision 3: Token Budget for Context Injection

### The Problem
Before each message, Axel searches the brain and injects relevant context.
Too much = wasted tokens, higher cost, diluted attention.
Too little = agent has no context, defeats the purpose.

### The Math

Claude's context window: 200K tokens.
Average system prompt: ~2K tokens.
Average conversation after 10 turns: ~5-15K tokens.
Available budget for injection: plenty, but attention degrades with noise.

Research shows LLMs attend best to:
- First ~1K tokens of system prompt (primacy)
- Last ~2K tokens before the query (recency)
- Content semantically similar to the query (relevance)

### Decision: Tiered injection with hard budget

```
Tier 0 — Always (≤200 tokens):
  Handoff note from last session (if exists)
  
Tier 1 — Relevant (≤500 tokens):  
  Top 3-5 memories by search relevance to current message
  Format: "## title\nabstract" (not full content)
  
Tier 2 — On-demand (uncapped):
  Agent calls axel_search explicitly for deep retrieval
  Returns full content, up to 10 results
```

Total automatic injection: ≤700 tokens per turn. ~$0.002 per turn at Sonnet rates.
Agent can pull more via tool calls when it needs depth.

**Key optimization:** Don't re-inject memories already mentioned in the conversation.
Hash memory IDs against conversation history, skip duplicates.

---

## Decision 4: When to Extract Memories

### The Problem
That costs tokens. When do we do it?

### Options

**A) Every session end** — most complete, but wastes tokens on trivial sessions
**B) Threshold-based** — only extract if session > N messages or > M minutes
**C) Hybrid extraction** — regex first (free), LLM only for complex sessions
**D) Agent-initiated** — agent decides what to remember (via axel_remember tool)

### Decision: D + B hybrid

1. **During session:** Agent can call `axel_remember` to commit important things immediately (zero LLM cost — agent already has the context).

2. **On session end:** If session had > 5 user messages, run lightweight regex extraction (free, no API call). Catches obvious things: decisions, names, project references.

3. **Background batch (optional):** Periodically, run full LLM extraction on sessions that only got regex treatment. This is the "deep think" pass — runs when idle, not blocking the user.

This means:
- Short sessions: no extraction cost
- Medium sessions: free regex extraction + agent's own axel_remember calls
- Deep extraction: runs async, doesn't block, can use cheaper models

---

## Decision 5: Concurrency Model

### The Problem
Multiple SynapsCLI sessions might use the same .r8 file.

### Decision: SQLite WAL + write lock

- SQLite WAL mode: unlimited concurrent readers, single writer
- Write operations (commit memory, update context) acquire an exclusive lock
- Read operations (search, recall) never block
- USearch index: rebuilt per-process from SQLite on cold start, read-only after that
- If two sessions write simultaneously: SQLite's built-in locking handles it (SQLITE_BUSY retry)

This is the same model SQLite was designed for. Don't overthink it.

---

## Decision 6: Deduplication Strategy

### The Problem
Same information might be extracted multiple times across sessions.

### Decision: Two-layer dedup

1. **Fast check (on commit):** Title similarity via normalized Levenshtein distance. Threshold 0.85. O(n) against existing memories but n is bounded (few thousand).

2. **Semantic dedup (on search):** If search returns two memories with cosine similarity > 0.92, flag the older one for decay. Don't delete — decay its importance until it falls below the recall threshold.

Why not Jaccard? Jaccard on word sets misses paraphrases. Levenshtein on titles catches exact rephrasing. Embedding similarity catches semantic duplicates. Two layers, two tools.

---

## Decision 7: Graph — Keep or Cut?

### The Data
Current Jawz graph.db is 251MB for 3,502 nodes. That's 71KB per node — insanely bloated 
because it stores full note content in nodes. velocirag-rs clears content before insert 
but the Python version doesn't.

Graph search contributes the LEAST to result quality in the current 4-layer pipeline.
In profiling, graph results rarely appear in top-5 after RRF fusion.

### Decision: Keep, but lightweight

- Graph nodes store only: id, type, title, metadata (no content — content lives in documents)
- Graph edges: source, target, type, weight, confidence
- Build graph lazily: only when memories reference each other (related_topics field)
- Don't run the full 6-analyzer pipeline on every commit — that's batch work
- This should keep the graph under 1MB for typical use

The graph's value is CONNECTIONS, not content storage. Keep it focused.

---

## Decision 8: Reranker — Worth the Cost?

### The Data
TinyBERT reranker: ~90MB ONNX model, adds ~50ms per search.
For a memory store with <5,000 entries, RRF fusion alone gives good results.
Reranker helps most at scale (>10K documents) where initial retrieval has more noise.

### Decision: Optional, off by default

- Default: 4-layer search + RRF fusion (no reranker)
- Config flag: `reranker = true` in axel.toml
- Only download reranker model if enabled
- This saves 90MB download and 50ms per search for typical users
