# Axel

**Portable agent intelligence — search, memory, and self-organizing knowledge in one `.r8` file.**

One file is your agent's entire brain. It searches, remembers, and gets smarter the more you use it.

---

## What It Does

Axel gives AI agents persistent, self-organizing memory. Three systems, one file:

- **VelociRAG** — 4-layer search (vector + BM25 + knowledge graph + metadata), fused with Reciprocal Rank Fusion, enhanced with MMR diversity, graph-boosted retrieval, query expansion, and excitability-aware ranking
- **Memkoshi** — Structured memory with staging, HMAC signing, importance decay/boost, and injection detection
- **Consolidation** — Biologically-inspired memory lifecycle that strengthens accessed documents, decays forgotten ones, wires related docs together, and prunes stale content

Everything lives in a single `.r8` file — a SQLite database that IS your agent's brain.

---

## Quick Start

```bash
# Build
cargo build --release

# Create a brain
axel init --name myagent

# Index your notes
axel index ./notes --source mynotes

# Search
axel search "rust async patterns"

# Remember something
axel remember --category preferences --topic stack "Chose axum over actix for async websocket support"

# Boot context (handoff + recent memories + hot docs)
axel recall

# Run consolidation (or let the timer do it)
axel consolidate
```

---

## Commands

| Command | Description |
|---------|-------------|
| `axel init [--name N]` | Create a new `.r8` brain |
| `axel index <path> [--source S]` | Index a file or directory |
| `axel index-sync <path> [--source S]` | Incremental sync — only re-index changed files |
| `axel search <query> [--limit N] [--json]` | Search with excitability boost, MMR diversity, graph expansion |
| `axel remember <content>` | Store a signed memory |
| `axel recall [query]` | Boot context or query-based recall |
| `axel handoff <set\|get\|clear>` | Session handoff management |
| `axel forget <id>` | Delete a memory |
| `axel stats` | Brain health dashboard — docs, access events, excitability, top queries |
| `axel memories [--limit N]` | List stored memories |
| `axel consolidate` | Run the 4-phase memory lifecycle |
| `axel excitability [--limit N]` | Visualize document importance distribution |
| `axel suggest <query>` | Graph-based document recommendations |
| `axel extension` | Run as SynapsCLI extension (JSON-RPC) |
| `axel mcp` | Run as MCP server (6 tools) |

### Consolidation Flags

```bash
axel consolidate                    # Full 4-phase run
axel consolidate --phase reindex    # Single phase only
axel consolidate --dry-run          # Preview without changes
axel consolidate -v                 # Per-document detail
axel consolidate --history          # Past runs + timer status
axel consolidate --report out.md    # Export prune candidates
axel consolidate --json             # Machine-readable output
axel consolidate --sources cfg.toml # Custom source config
```

---

## Consolidation

The brain self-organizes through a 4-phase cycle that runs automatically every 6 hours:

**Phase 1 — Reindex.** Walk source directories, detect changed files via mtime comparison, re-embed modified content, prune deleted files. New documents get linked to high-excitability neighbors (competitive allocation).

**Phase 2 — Strengthen.** Read the access log since last run. Documents that were searched for get an excitability boost. Documents with consistently low search scores get an extinction signal. Documents untouched for weeks decay following an exponential forgetting curve. More accesses = slower decay.

**Phase 3 — Reorganize.** Documents that keep appearing in the same search results get graph edges between them. Unreinforced edges decay. The knowledge graph wires itself through usage.

**Phase 4 — Prune.** Flag stale documents (low excitability, zero access, old). Auto-remove from low-priority sources. Flag for human review from high-priority sources. Detect misaligned embeddings.

### The Feedback Loop

```
search → access logged → consolidation boosts excitability →
search ranking improves → more access → more boost → repeat
```

Documents you use get stronger. Documents you don't fade away. The brain self-organizes.

### Setup

```bash
# Enable the systemd timer (every 6 hours)
systemctl --user enable --now axel-consolidate.timer

# Configure sources
cat ~/.config/axel/sources.toml
```

```toml
[[source]]
name = "notes"
path = "~/notes/"
priority = "high"

[[source]]
name = "archive"
path = "~/archive/"
priority = "low"
```

---

## Search

Search combines 4 retrieval layers via Reciprocal Rank Fusion, then applies post-processing:

1. **Vector similarity** — ONNX MiniLM-L6-v2 embeddings, USearch HNSW index
2. **BM25 keywords** — FTS5 full-text search with pseudo-relevance query expansion
3. **Knowledge graph** — traverse entity and relationship edges
4. **Metadata** — tags and cross-references
5. **RRF fusion** — merge all layers into a single ranked list
6. **Excitability boost** — recently accessed docs rank higher, with exponential temporal decay
7. **Graph boost** — documents connected via co-retrieval edges get a spreading activation bump
8. **MMR diversity** — penalize near-duplicate results via cosine similarity

```bash
axel search "deployment strategy" --limit 5
axel search "deployment strategy" --json | jq '.results[0].content'
```

---

## The `.r8` Format

A `.r8` file is a **single SQLite database** (WAL mode). One file = one brain.

```
axel.r8  →  SQLite
  ├── documents              — indexed corpus with excitability scores
  ├── documents_fts          — FTS5 full-text search index
  ├── document_access        — search hit log (feeds consolidation)
  ├── co_retrieval           — co-appearance tracking (feeds graph wiring)
  ├── consolidation_log      — audit trail for consolidation runs
  ├── nodes + edges          — knowledge graph (co-retrieval + entity edges)
  ├── memories               — HMAC-signed structured memories
  ├── staged_memories        — memories pending review
  ├── memory_access          — memory retrieval log
  ├── context_data           — session handoff (key/value)
  └── brain_meta             — schema version, agent name, model info
```

```bash
# It's just SQLite
sqlite3 axel.r8 "SELECT doc_id, excitability FROM documents ORDER BY excitability DESC LIMIT 5"
sqlite3 axel.r8 "SELECT COUNT(*) FROM document_access"
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                      axel (crate)                     │
│  CLI · MCP server · SynapsCLI extension · library     │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐   │
│  │ search   │  │ memory   │  │  consolidation    │   │
│  │ (query)  │  │ (store)  │  │  (4-phase cycle)  │   │
│  └────┬─────┘  └────┬─────┘  └────────┬──────────┘   │
│       └──────────────┼────────────────-┘              │
│              ┌───────▼───────┐                        │
│              │   brain.r8    │                        │
│              │   (SQLite)    │                        │
│              └───────────────┘                        │
└──────────────────────────────────────────────────────┘
```

### Crates

| Crate | Role |
|-------|------|
| `velocirag` | 4-layer RAG engine, ONNX embeddings, HNSW index, RRF fusion, MMR, graph boost |
| `axel-memkoshi` | Memory storage, HMAC signing, decay/boost, pattern detection |
| `axel` | CLI, `.r8` format, consolidation engine, MCP server, context injection |

---

## Integration with SynapsCLI

Axel is designed to be the memory layer for [SynapsCLI](https://github.com/HaseebKhalid1507/synaps-cli) — a terminal-native AI agent runtime. Three integration points:

### 1. MCP Server

Axel runs as an MCP (Model Context Protocol) server, exposing 6 tools that any SynapsCLI agent can call:

```bash
axel mcp   # Start the MCP server (configured in ~/.synaps-cli/mcp.json)
```

| Tool | What It Does |
|------|-------------|
| `axel_search` | Search the brain — returns ranked results with scores |
| `axel_remember` | Store a new memory with category, importance, TTL |
| `axel_recall` | Get boot context — handoff + recent memories + hot docs |
| `axel_verify` | Check a memory's HMAC signature and provenance |
| `axel_update` | Edit a memory's content or importance (re-signs) |
| `axel_consolidate` | Trigger a consolidation pass (any phase, dry-run supported) |

```json
// ~/.synaps-cli/mcp.json
{
  "mcpServers": {
    "axel": {
      "command": "/path/to/axel",
      "args": ["mcp"]
    }
  }
}
```

### 2. Native Plugin

The `axel-brain` plugin hooks directly into SynapsCLI's extension system for zero-friction integration:

```
~/.synaps-cli/plugins/axel-brain/
├── .synaps-plugin/
│   └── plugin.json          # Extension manifest
└── main.py                  # Hook handlers
```

**What it does:**

| Hook | When | Action |
|------|------|--------|
| `on_session_start` | Agent boots | Runs Phase 1 (reindex) to catch file changes |
| `before_message` | First user message | Searches brain, injects relevant context into system prompt |
| `on_session_end` | Session closes | Logs session boundary |

The plugin only injects on the **first message** of a session — after that, the agent calls `axel_search` explicitly when it needs context. This keeps token costs minimal.

Results below a confidence threshold (0.025) are filtered out — only genuinely relevant documents get injected.

### 3. Multi-Agent Shared Memory

Every SynapsCLI agent — main session and subagents — can read from the same `.r8` brain. When one agent searches for "CS646" and another later searches for "network security," the access patterns compound:

- Agent A searches → document_access logged → consolidation boosts excitability
- Agent B searches → boosted doc ranks higher → access logged again → stronger boost
- The whole crew builds shared knowledge through a single brain

### Configuration

```bash
AXEL_BRAIN=/path/to/custom.r8   # Override default brain path
```

| Path | Purpose |
|------|---------|
| `~/.config/axel/axel.r8` | Default brain location |
| `~/.config/axel/sources.toml` | Consolidation source directories |
| `~/.cache/axel/models/` | ONNX embedding model cache |
| `~/.synaps-cli/mcp.json` | MCP server configuration |
| `~/.synaps-cli/plugins/axel-brain/` | Native plugin |

---

## License

MIT
