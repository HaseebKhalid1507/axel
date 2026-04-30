# Axel

**Portable agent intelligence — search, memory, and session awareness in one `.r8` file.**

One file is your agent's entire brain. Copy it. Move it. It just works.

---

## What It Is

Axel is a CLI tool and library that gives any AI agent persistent memory. Three engines, one file:

- **VelociRAG** — 4-layer search (vector similarity + BM25 keywords + knowledge graph + metadata), fused with Reciprocal Rank Fusion
- **Memkoshi** — Memory system with staging gates, HMAC signing, importance decay/boost, and injection detection

Everything lives in a single `.r8` file — a SQLite database that IS your agent's brain.

---

## Quick Start

```bash
# Build
cargo build --release

# Create a brain
axel init --name myagent

# Index your notes
axel index ./notes

# Search
axel search "rust async patterns"

# Remember something
axel remember --category preferences --topic stack "Chose axum over actix for async websocket support"

# Boot context (handoff + recent memories)
axel recall
```

---

## Commands

| Command | Description |
|---------|-------------|
| `axel init [--name N]` | Create a new `.r8` brain file |
| `axel index <path>` | Index a file or directory (`.md` + `.txt`) |
| `axel search <query> [--limit N] [--json]` | 4-layer search across documents and memories |
| `axel remember <content> [--category C] [--topic T]` | Store a signed memory |
| `axel recall [query]` | Boot context (handoff + recent memories), or query-based recall |
| `axel extract <file\|text>` | Extract memories from a transcript via regex + quality gate |
| `axel handoff <set\|get\|clear> [content]` | Manage session handoff note (max 4096 chars) |
| `axel forget <id>` | Delete a memory by ID (`mem_xxxxxxxx`) |
| `axel stats` | Brain statistics: documents, memories, graph nodes, file size |
| `axel memories [--limit N]` | List stored memories with signature status |

**Memory categories:** `events` (default) · `preferences` · `entities` · `cases` · `patterns`

**`--json` flag:** `axel search --json "query"` returns structured JSON for scripting/agents.

```bash
# JSON output example
axel search --json "deployment strategy" | jq '.results[0].content'
```

---

## The `.r8` Format

A `.r8` file is a **single SQLite database** (WAL mode). One file = one brain. No directories, no sidecars.

```
axel.r8  →  Single SQLite file
  ├── documents + documents_fts   — indexed corpus (FTS5 full-text search)
  ├── nodes + edges               — knowledge graph (lightweight: title/type only, no content)
  ├── memories + staged_memories  — HMAC-signed memories with review pipeline
  ├── events + memory_access      — behavioral event log
  ├── patterns                    — detected behavioral patterns
  ├── context_data                — session handoff + boot context (key/value)
  └── schema_info                 — schema version, agent name, created timestamp
```

The USearch HNSW index is rebuilt in memory from stored embeddings on open (~70ms for 7K docs), then cached to `~/.cache/axel/<sha256>.usearch` for hot starts. Embeddings are the source of truth; the index is the accelerator.

```bash
# It's just SQLite — inspect it directly
sqlite3 axel.r8 "SELECT title, category FROM memories LIMIT 10"
sqlite3 axel.r8 "SELECT COUNT(*) FROM documents"
```

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  axel (CLI + glue)               │
│                                                  │
│  ┌──────────────┐  ┌───────────┐  ┌───────────┐ │
│  │  (search)    │  │  (memory) │  │  (extract) │ │
│  └──────┬───────┘  └─────┬─────┘  └─────┬─────┘ │
│         └────────────────┼───────────────┘       │
│                  ┌───────▼───────┐               │
│                  │   axel.r8     │               │
│                  │  (SQLite)     │               │
│                  └───────────────┘               │
└─────────────────────────────────────────────────┘
```

### Crates

| Crate | Role | Lines | Tests |
|-------|------|-------|-------|
| `velocirag` | 4-layer RAG search engine, ONNX embeddings (MiniLM-L6-v2), USearch HNSW, RRF fusion | 5,468 | 34 |
| `axel-memkoshi` | Memory staging, HMAC signing, decay/boost, pattern detection | 2,134 | 35 |
| `axel` | CLI, `.r8` format, context injection | 2,138 | 31 |

**Total: ~11,200 lines · 157 tests · 4 crates**

---

## Features

**Search**
- 4-layer retrieval: vector similarity · BM25 keyword · knowledge graph · metadata tags
- Reciprocal Rank Fusion across all layers
- Optional TinyBERT cross-encoder reranker (off by default, enable in config)
- `--json` flag for scripting and agent tool calls

**Memory**
- HMAC-SHA256 signing on every memory — `axel memories` shows `✓` valid / `⚠ TAMPERED` / `⚠ UNSIGNED`
- Prompt injection detection before storage
- Importance decay over time, boost on access
- Stage → validate → approve pipeline
- Pattern detection across memory access history

**Extraction**
- `axel extract` runs regex patterns against transcripts — zero LLM cost
- Quality gate: length check, importance threshold, field validation
- Deduplication against existing memories (Levenshtein on titles + semantic similarity)

**Context Injection**
- Tiered injection budget: 200 tokens handoff + 500 tokens relevant memories per turn
- Agent pulls more depth via `axel search --json` on demand

---

## SynapsCLI Integration

Skill file at `~/.synaps-cli/skills/axel/axel.md` — loaded via `load_skill` to give the agent full command awareness.

The `jawz-axel` tool injects boot context automatically at session start (handoff + recent memories). Agents use `axel search --json` for in-session retrieval and `axel remember` to commit decisions as they happen.

```bash
# Skill gives the agent these patterns:
axel search --json "topic"                        # Before answering anything
axel remember --category entities --topic people "Dr. Smith, security lab, Thursdays 3pm"
axel handoff set "Working on X, blocked on Y, resume with Z"
axel extract session-brief.md                     # End of session
```

---

## Environment

```bash
AXEL_BRAIN=/path/to/custom.r8   # Override default brain path
```

Default brain: `~/.config/axel/axel.r8`  
Model cache: `~/.cache/axel/models/` (ONNX models, auto-downloaded on first use)

---

## License

MIT
