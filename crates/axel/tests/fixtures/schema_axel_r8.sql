-- The tables of an old Axel brain (axel.r8, version 1), as this repository creates them:
-- `Brain::init_tables` (crates/axel/src/r8.rs) and velocirag's `Database::init_schema`
-- (crates/velocirag/src/db.rs) with the columns its migrations add to `documents`.
-- Schema only: no row of any real brain is here. The tests insert synthetic rows, the
-- `brain_meta` row and the `kv` row.

CREATE TABLE brain_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE memories (
    id              TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    topic           TEXT NOT NULL,
    title           TEXT NOT NULL,
    abstract_text   TEXT,
    content         TEXT NOT NULL,
    confidence      TEXT DEFAULT 'medium',
    importance      REAL DEFAULT 0.5,
    source_sessions TEXT DEFAULT '[]',
    tags            TEXT DEFAULT '[]',
    related_topics  TEXT DEFAULT '[]',
    created         TEXT NOT NULL,
    updated         TEXT,
    trust_level     REAL DEFAULT 1.0,
    signature       TEXT
);
CREATE INDEX idx_memories_category ON memories(category);
CREATE INDEX idx_memories_importance ON memories(importance);
CREATE INDEX idx_memories_created ON memories(created);

CREATE TABLE staged_memories (
    id              TEXT PRIMARY KEY,
    category        TEXT NOT NULL,
    topic           TEXT NOT NULL,
    title           TEXT NOT NULL,
    abstract_text   TEXT,
    content         TEXT NOT NULL,
    confidence      TEXT DEFAULT 'medium',
    importance      REAL DEFAULT 0.5,
    source_sessions TEXT DEFAULT '[]',
    tags            TEXT DEFAULT '[]',
    related_topics  TEXT DEFAULT '[]',
    created         TEXT NOT NULL,
    updated         TEXT,
    trust_level     REAL DEFAULT 1.0,
    signature       TEXT,
    review_status   TEXT DEFAULT 'pending',
    reviewer_notes  TEXT,
    staged_at       TEXT NOT NULL
);

CREATE TABLE memory_access (
    id          INTEGER PRIMARY KEY,
    memory_id   TEXT NOT NULL,
    access_type TEXT NOT NULL,
    timestamp   TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_memory_access_id ON memory_access(memory_id);

CREATE TABLE sessions (
    id               TEXT PRIMARY KEY,
    source           TEXT,
    session_date     TEXT,
    model            TEXT,
    memory_count     INTEGER DEFAULT 0,
    transcript_chars INTEGER DEFAULT 0,
    duration_seconds REAL,
    status           TEXT DEFAULT 'processed'
);

CREATE TABLE events (
    id         INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    target_id  TEXT,
    query      TEXT,
    metadata   TEXT DEFAULT '{}',
    timestamp  TEXT DEFAULT (datetime('now'))
);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_timestamp ON events(timestamp);

CREATE TABLE patterns (
    id           INTEGER PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    name         TEXT NOT NULL,
    description  TEXT,
    confidence   REAL DEFAULT 0.5,
    sample_size  INTEGER DEFAULT 0,
    last_triggered TEXT,
    created      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE context_data (
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,
    layer   TEXT NOT NULL DEFAULT 'session',
    updated TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (key, layer)
);

CREATE TABLE documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT UNIQUE NOT NULL,
    content     TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    embedding   BLOB NOT NULL,
    file_path   TEXT,
    created     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    l0_abstract     TEXT,
    l1_overview     TEXT,
    l0_embedding    BLOB,
    l1_embedding    BLOB,
    indexed_at      TIMESTAMP,
    access_count    INTEGER DEFAULT 0,
    last_accessed   TIMESTAMP,
    excitability    REAL DEFAULT 0.5
);
CREATE INDEX idx_doc_doc_id ON documents(doc_id);
CREATE INDEX idx_doc_file_path ON documents(file_path);
CREATE INDEX idx_doc_excitability ON documents(excitability);
CREATE INDEX idx_doc_access_count ON documents(access_count);
CREATE INDEX idx_doc_last_accessed ON documents(last_accessed);

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    doc_id,
    content,
    file_path,
    tokenize='porter unicode61'
);

CREATE TABLE nodes (
    id          TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    title       TEXT NOT NULL,
    content     TEXT,
    metadata    TEXT NOT NULL DEFAULT '{}',
    source_file TEXT,
    source_name TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE edges (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    type        TEXT NOT NULL,
    weight      REAL NOT NULL,
    confidence  REAL NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    source_file TEXT,
    source_name TEXT,
    created_at  TEXT NOT NULL,
    valid_from  TEXT,
    valid_to    TEXT,
    FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES nodes(id) ON DELETE CASCADE
);
CREATE INDEX idx_nodes_type ON nodes(type);
CREATE INDEX idx_nodes_source_file ON nodes(source_file);
CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_edges_type ON edges(type);
CREATE INDEX idx_edges_weight ON edges(weight DESC);
CREATE INDEX idx_edges_source_file ON edges(source_file);
CREATE INDEX idx_edges_type_src_tgt ON edges(type, source_id, target_id);

CREATE TABLE tags (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name    TEXT UNIQUE NOT NULL
);

CREATE TABLE document_tags (
    doc_id  INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    tag_id  INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (doc_id, tag_id)
);

CREATE TABLE cross_refs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    ref_type    TEXT NOT NULL,
    ref_target  TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_tags_name ON tags(name);
CREATE INDEX idx_cross_refs_doc ON cross_refs(doc_id);
CREATE INDEX idx_cross_refs_target ON cross_refs(ref_target);

CREATE TABLE file_cache (
    cache_key       TEXT PRIMARY KEY,
    last_modified   REAL NOT NULL,
    source_name     TEXT DEFAULT '',
    last_indexed    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE file_provenance (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path       TEXT NOT NULL,
    source_name     TEXT NOT NULL DEFAULT '',
    last_modified   REAL NOT NULL,
    content_hash    TEXT,
    node_count      INTEGER DEFAULT 0,
    edge_count      INTEGER DEFAULT 0,
    indexed_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(file_path, source_name)
);
CREATE INDEX idx_provenance_path ON file_provenance(file_path);

CREATE TABLE kv (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);

CREATE TABLE document_access (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL,
    access_type TEXT NOT NULL,
    query       TEXT,
    score       REAL,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    session_id  TEXT
);
CREATE INDEX idx_docaccess_doc_id ON document_access(doc_id);
CREATE INDEX idx_docaccess_timestamp ON document_access(timestamp);

CREATE TABLE co_retrieval (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id_a    TEXT NOT NULL,
    doc_id_b    TEXT NOT NULL,
    query       TEXT,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX idx_coret_pair ON co_retrieval(doc_id_a, doc_id_b);
CREATE INDEX idx_coret_timestamp ON co_retrieval(timestamp);

CREATE TABLE consolidation_log (
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
