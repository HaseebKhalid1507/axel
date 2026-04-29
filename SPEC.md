# Axel — Portable Agent Intelligence

**Plugin for SynapsCLI**
**File format: `.r8`**
**Together: AxelR8 (Accelerate)**

---

## 1. Objective

Axel is a SynapsCLI plugin that gives any agent persistent memory, intelligent search, and session awareness — stored in a single portable `.r8` file.

One file = one agent brain. Drop it in, agent has context. Move it between machines. No setup, no configuration, no separate databases.

**What it replaces:** Three separate Python projects (VelociRAG, Memkoshi, Stelline) consolidated into one Rust plugin with a unified storage format.

### Success Criteria

- Agent boots with relevant context from previous sessions (< 200ms search)
- Session memories are extracted and stored automatically on shutdown
- A single `axel.r8` file contains everything: vectors, documents, graph, memories, metadata, session history
- File is portable — copy to new machine, agent resumes with full context
- Plugin installs with zero configuration: `synaps plugin install axel`
- Embedding model runs locally (no API calls for search)

### Users

- **Primary:** SynapsCLI users who want persistent agent memory
- **Secondary:** Agent developers building on SynapsCLI who need a memory layer

---

## 2. Architecture

### Components (from existing codebases)

```
┌─────────────────────────────────────────────────┐
│                    Axel Plugin                   │
│                                                  │
│  ┌──────────────┐  ┌───────────┐  ┌───────────┐│
│  │  VelociRAG-RS │  │  Memkoshi │  │  Stelline ││
│  │  (search)     │  │  (memory) │  │  (extract)││
│  │              │  │           │  │           ││
│  │ • 4-layer    │  │ • Stage   │  │ • Session ││
│  │   search     │  │ • Review  │  │   parsing ││
│  │ • Embedding  │  │ • Approve │  │ • LLM     ││
│  │ • Graph      │  │ • Pattern │  │   extract ││
│  │ • RRF fusion │  │ • Evolve  │  │ • Context ││
│  │ • Reranker   │  │ • Decay   │  │   updates ││
│  └──────┬───────┘  └─────┬─────┘  └─────┬─────┘│
│         │                │               │      │
│         └────────┬───────┘───────────────┘      │
│                  │                               │
│          ┌───────▼───────┐                       │
│          │   axel.r8     │                       │
│          │  (SQLite +    │                       │
│          │   USearch)    │                       │
│          └───────────────┘                       │
└─────────────────────────────────────────────────┘
```

### The `.r8` Format

A `.r8` file is a **directory bundle** (like `.app` on macOS) containing:

```
axel.r8/
├── store.db          # SQLite WAL — documents, memories, graph, metadata, sessions, patterns
├── vectors.usearch   # USearch HNSW index (ANN accelerator, rebuildable from store.db)
├── manifest.json     # Version, agent name, created, last_modified, stats
└── models/           # Embedded ONNX models (optional, can use system models)
    ├── embedder/     # all-MiniLM-L6-v2 (23MB)
    └── reranker/     # ms-marco-TinyBERT (17MB)
```

**SQLite schema (store.db):**

```sql
-- VelociRAG layer
CREATE TABLE documents (
    id INTEGER PRIMARY KEY,
    doc_id TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    metadata JSON,
    embedding BLOB,              -- f32 × 384, rebuildable
    file_path TEXT,
    created TEXT DEFAULT (datetime('now'))
);
CREATE VIRTUAL TABLE documents_fts USING fts5(content, content=documents);

-- Knowledge Graph
CREATE TABLE nodes (id TEXT PK, node_type TEXT, title TEXT, content TEXT, metadata JSON, source TEXT);
CREATE TABLE edges (id TEXT PK, source_id TEXT, target_id TEXT, edge_type TEXT, weight REAL, confidence REAL, metadata JSON);

-- Memkoshi layer
CREATE TABLE memories (
    id TEXT PRIMARY KEY,          -- mem_XXXXXXXX
    category TEXT NOT NULL,       -- preferences|entities|events|cases|patterns
    topic TEXT NOT NULL,
    title TEXT NOT NULL,
    abstract TEXT,
    content TEXT NOT NULL,
    confidence TEXT DEFAULT 'medium',
    importance REAL DEFAULT 0.5,
    source_sessions JSON,
    tags JSON,
    related_topics JSON,
    created TEXT,
    updated TEXT,
    trust_level REAL DEFAULT 1.0,
    signature TEXT                -- HMAC-SHA256
);
CREATE TABLE staged_memories (...same + review_status, reviewer_notes, staged_at);

-- Metadata layer
CREATE TABLE tags (id INTEGER PK, name TEXT UNIQUE);
CREATE TABLE document_tags (document_id INTEGER, tag_id INTEGER);
CREATE TABLE cross_refs (source_doc TEXT, target_doc TEXT, ref_type TEXT, weight REAL);

-- Session intelligence
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT,
    session_date TEXT,
    model TEXT,
    memory_count INTEGER,
    transcript_chars INTEGER,
    duration_seconds REAL,
    status TEXT
);
CREATE TABLE context_data (key TEXT, value TEXT, layer TEXT, updated TEXT);

-- Patterns & Evolution  
CREATE TABLE patterns (pattern_type TEXT, name TEXT, description TEXT, confidence REAL, sample_size INT, ...);
CREATE TABLE events (event_type TEXT, target_id TEXT, query TEXT, metadata JSON, timestamp TEXT);
CREATE TABLE memory_access (memory_id TEXT, access_type TEXT, timestamp TEXT);

-- Meta
CREATE TABLE schema_info (version INTEGER, created TEXT, agent_name TEXT);
```

### SynapsCLI Integration Points

Axel hooks into SynapsCLI via the extension system:

| Hook | What Axel Does |
|------|---------------|
| `on_session_start` | Load `axel.r8`, warm search index |
| `before_message` | Search .r8 for relevant context, inject into system prompt |
| `after_tool_call` | Index tool outputs into .r8 (if significant) |
| `on_session_end` | Extract memories from session transcript, update context |
| `on_memory_search` | Expose `axel_search` tool to the agent |
| `on_memory_commit` | Expose `axel_remember` tool to the agent |

### Tools Registered

```
axel_search    — Search the agent's brain (query, limit, layers)
axel_remember  — Manually commit a memory (content, category, importance)
axel_recall    — Get boot context (handoff, recent sessions, relevant memories)
axel_forget    — Remove a memory by ID
axel_status    — Stats: memory count, index size, last session, patterns
```

---

## 3. Implementation Phases

### Phase 1: Foundation — velocirag-rs as library crate
- [ ] Move velocirag-rs to `~/Projects/axel/crates/velocirag/`
- [ ] Clean up: remove unused `petgraph` dep, fix warnings
- [ ] Add as workspace member of axel
- [ ] Verify: `cargo test` passes, search pipeline works
- [ ] Remove CLI binary — library only

### Phase 2: `.r8` Format
- [ ] Design `manifest.json` schema
- [ ] Implement `.r8` open/create/validate
- [ ] Extend SQLite schema with Memkoshi tables (memories, staged, patterns, events, context_data)
- [ ] Extend with session tracking tables
- [ ] Migration system for schema versions

### Phase 3: Memory Layer (Memkoshi port)
- [ ] `Memory` struct with all fields from Python model
- [ ] Staging pipeline: stage → validate → dedup → approve/reject
- [ ] Injection detection (prompt injection guard)
- [ ] HMAC signing
- [ ] Pattern detection (frequency, gaps, temporal)
- [ ] Decay & boost (importance over time)

### Phase 4: Session Intelligence (Stelline port)
- [ ] Session transcript parser (SynapsCLI JSONL format)
- [ ] LLM-based memory extraction (via SynapsCLI's own model connection)
- [ ] Quality gate (content length, importance threshold, field validation)
- [ ] Context update system (projects, people, handoff files)
- [ ] Deduplication against existing memories

### Phase 5: Plugin Shell
- [ ] SynapsCLI plugin manifest (`plugin.json`)
- [ ] Hook registrations (session start/end, before_message)
- [ ] Tool registrations (axel_search, axel_remember, etc.)
- [ ] Context injection (search .r8, format results, prepend to system prompt)
- [ ] Config: `~/.synaps-cli/axel.toml` (brain path, auto-extract on/off, injection budget)

### Phase 6: Polish
- [ ] `.r8` export/import (share brains between agents)
- [ ] `.r8` stats CLI (`synaps axel stats`)
- [ ] `.r8` merge (combine two brains)
- [ ] Embedding model auto-download on first run
- [ ] Documentation

---

## 4. Project Structure

```
~/Projects/axel/
├── Cargo.toml              # Workspace root
├── SPEC.md                 # This file
├── crates/
│   ├── velocirag/           # Search engine (existing Rust port)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── db.rs        # SQLite storage
│   │       ├── embedder.rs  # ONNX MiniLM-L6-v2
│   │       ├── index.rs     # USearch HNSW
│   │       ├── search.rs    # 4-layer search engine
│   │       ├── graph.rs     # Knowledge graph queries
│   │       ├── rrf.rs       # Reciprocal rank fusion
│   │       ├── reranker.rs  # TinyBERT cross-encoder
│   │       ├── chunker.rs   # Document chunking
│   │       ├── pipeline.rs  # Index + graph build
│   │       ├── analyzers.rs # 6 graph analyzers
│   │       └── ...
│   ├── memkoshi/            # Memory system
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── memory.rs    # Memory struct + categories
│   │       ├── staging.rs   # Stage → review → approve pipeline
│   │       ├── patterns.rs  # Pattern detection
│   │       ├── evolution.rs # Session scoring
│   │       ├── security.rs  # HMAC signing + injection guard
│   │       └── decay.rs     # Importance decay & boost
│   ├── stelline/            # Session intelligence
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── parser.rs    # Session transcript parser
│   │       ├── extractor.rs # LLM memory extraction
│   │       ├── quality.rs   # Quality gate
│   │       ├── context.rs   # Context file updates
│   │       └── dedup.rs     # Memory deduplication
│   └── axel/                # Plugin + .r8 format
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           ├── r8.rs        # .r8 format: open, create, validate, migrate
│           ├── plugin.rs    # SynapsCLI plugin hooks
│           ├── tools.rs     # Registered tools (search, remember, etc.)
│           ├── inject.rs    # Context injection before messages
│           └── config.rs    # Plugin configuration
└── tests/
    └── integration/
```

---

## 5. Commands

```bash
# Build
cargo build                          # Debug build, all crates
cargo build --release                # Release build

# Test
cargo test                           # All tests
cargo test -p velocirag              # Just search engine
cargo test -p memkoshi               # Just memory layer
cargo test -p stelline               # Just extraction
cargo test -p axel                   # Just plugin + .r8

# Run (standalone, outside SynapsCLI)
cargo run -p axel -- search "query"  # Search a .r8 file
cargo run -p axel -- stats           # Show .r8 stats
cargo run -p axel -- init            # Create empty axel.r8
```

---

## 6. Code Style

Follows SynapsCLI conventions (same as velocirag-rs existing code):

```rust
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub category: MemoryCategory,
    pub topic: String,
    pub title: String,
    pub content: String,
    pub importance: f64,
    pub confidence: Confidence,
    pub tags: Vec<String>,
    pub created: DateTime<Utc>,
    pub updated: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryCategory {
    Preferences,
    Entities,
    Events,
    Cases,
    Patterns,
}

impl Memory {
    pub fn new(category: MemoryCategory, topic: &str, title: &str, content: &str) -> Self {
        Self {
            id: format!("mem_{:08x}", rand::random::<u32>()),
            category,
            topic: topic.to_string(),
            title: title.to_string(),
            content: content.to_string(),
            importance: 0.5,
            confidence: Confidence::Medium,
            tags: Vec::new(),
            created: Utc::now(),
            updated: None,
        }
    }
}
```

---

## 7. Testing Strategy

- **Unit tests:** In each crate's `src/` files (`#[cfg(test)] mod tests`)
- **Integration tests:** `tests/integration/` — create `.r8` in tempdir, run full pipelines
- **Existing velocirag-rs tests:** Preserve and expand
- **Target:** Every public function has at least one test
- **Property:** `.r8` files created on one machine open correctly on another (endianness, path independence)

---

## 8. Boundaries

### Always Do
- Run `cargo test` before commits
- Validate all inputs to public APIs
- Use `thiserror` for error types, never `.unwrap()` in library code
- Keep `.r8` format backward-compatible (schema migrations)
- HMAC-sign memories to detect tampering

### Ask First
- Adding new dependencies
- Changing `.r8` schema (requires migration)
- Changing the embedding model (changes all vectors)
- Exposing new tools to the agent

### Never Do
- Store API keys or credentials in `.r8` files
- Make network calls during search (embeddings are local)
- Break `.r8` backward compatibility without migration
- Skip the quality gate on memory extraction

---

## 9. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage format | SQLite + USearch in a directory bundle | SQLite gives ACID, FTS5, JSON. USearch gives fast ANN. Both are single-file, portable. |
| Embedding model | all-MiniLM-L6-v2 (384-dim) | Small (23MB), fast, good quality. Already proven in velocirag-rs. |
| Reranker | ms-marco-TinyBERT-L-2-v2 | Tiny cross-encoder, proven in velocirag-rs. |
| Language | Rust | Native SynapsCLI integration, performance, single binary. |
| Plugin system | SynapsCLI extensions/hooks | Uses the existing spec at `docs/specs/2026-04-27-extensions-and-hooks.md`. |
| Graph storage | SQLite tables (not petgraph) | velocirag-rs already does this — petgraph is declared but unused. |
| Memory extraction | LLM via SynapsCLI's model connection | No separate API keys needed. Uses whatever model the agent is already connected to. |

---

## 10. Dependencies (from velocirag-rs + new)

```toml
# Existing (proven in velocirag-rs)
rusqlite = { version = "0.32", features = ["bundled", "vtab", "blob"] }
usearch = "2"
ort = { version = "2.0.0-rc.12", features = ["download-binaries"] }
tokenizers = "0.21"
ndarray = "0.17"
sha2 = "0.10"
uuid = { version = "1", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
rayon = "1"
regex = "1"
ordered-float = "4"
walkdir = "2"
pulldown-cmark = "0.12"

# New for Axel
hmac = "0.12"                    # Memory signing
hex = "0.4"                      # HMAC output
rand = "0.8"                     # Memory ID generation
```
