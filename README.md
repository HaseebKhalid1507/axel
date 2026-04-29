# Axel

**Portable Agent Intelligence — Search, memory, and session awareness in one `.r8` file.**

AxelR8 = Accelerate. One file is your agent's entire brain.

## What is this?

Axel is a SynapsCLI plugin that gives any AI agent persistent memory. It combines three capabilities:

- **VelociRAG** — 4-layer search engine (vector similarity + BM25 keywords + knowledge graph + metadata), fused with Reciprocal Rank Fusion
- **Memkoshi** — Agent memory system with staging gates, pattern detection, importance decay/boost, and HMAC signing
- **Stelline** — Session intelligence that reads transcripts, extracts memories, and updates context files

All stored in a single `.r8` file — a SQLite database that IS your agent's brain.

## The `.r8` Format

```
axel.r8  →  Single SQLite file (WAL mode)
  ├── documents + FTS5        — search corpus
  ├── nodes + edges           — knowledge graph
  ├── memories + staged       — agent memory with review gates
  ├── events + patterns       — behavioral pattern detection
  ├── context_data            — session handoff + boot context
  └── brain_meta              — schema version, model info
```

Copy the file. That's it. Your agent's entire knowledge state moves with it.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Axel Plugin                     │
│                                                  │
│  ┌──────────────┐  ┌───────────┐  ┌───────────┐ │
│  │  VelociRAG   │  │  Memkoshi │  │  Stelline  │ │
│  │  (search)    │  │  (memory) │  │  (extract) │ │
│  └──────┬───────┘  └─────┬─────┘  └─────┬─────┘ │
│         └────────┬───────┘───────────────┘       │
│          ┌───────▼───────┐                       │
│          │   axel.r8     │                       │
│          │  (SQLite)     │                       │
│          └───────────────┘                       │
└─────────────────────────────────────────────────┘
```

## Crates

| Crate | Description | Lines |
|-------|-------------|-------|
| `velocirag` | 4-layer RAG search engine with ONNX embeddings | 5,683 |
| `axel-memkoshi` | Memory staging, patterns, evolution, HMAC signing | 1,238 |
| `axel-stelline` | Session parser, regex extraction, quality gate, dedup | 1,001 |
| `axel` | Plugin shell, `.r8` format, context injection | 919 |

## Quick Start

```bash
# Build
cargo build

# Run tests
cargo test

# Create a brain
# (programmatic — plugin integration coming soon)
```

## Key Design Decisions

1. **`.r8` is a single SQLite file** — not a directory bundle. One file = one brain.
2. **ONNX models live in system cache** (`~/.cache/axel/models/`), auto-downloaded on first use. Brain files stay lean.
3. **Token injection budget: 700 tokens/turn** — handoff (200) + relevant memories (500). Agent can pull more via tool calls.
4. **Memory extraction is agent-driven + regex** — `axel_remember` during session (free), regex on session end (free), LLM extraction optional and async.
5. **Graph is lightweight** — nodes have no content (just title/type/metadata). Connections only.
6. **Reranker is optional** — off by default, RRF alone is good enough under 5K documents.

## License

MIT
