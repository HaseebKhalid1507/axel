-- The tables of a Synaps brain.r8, as Synaps core's Axel helper keeps them
-- (sidecars/axel-memory-service): JR's `memories` with the project-memory columns, its
-- tombstones, and the helper's own tables. The shapes are the ones
-- `crates/axel/src/synaps_export.rs` reads and `tests/synaps_export.rs` builds.
-- Schema only: no row of any real brain is here.

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
    signature       TEXT,
    project_key     TEXT,
    sensitivity     TEXT NOT NULL DEFAULT 'normal',
    retention       TEXT NOT NULL DEFAULT 'standard',
    provenance      TEXT NOT NULL DEFAULT '',
    expires_at      TEXT,
    superseded_by   TEXT,
    supersedes      TEXT NOT NULL DEFAULT '[]',
    contradicts     TEXT NOT NULL DEFAULT '[]',
    confirmed_by    TEXT NOT NULL DEFAULT '[]',
    invalidated_at  TEXT
);
CREATE INDEX idx_memories_project ON memories(project_key);

CREATE TABLE memory_tombstones (
    id          TEXT PRIMARY KEY,
    project_key TEXT NOT NULL,
    deleted_at  TEXT NOT NULL
);
CREATE INDEX idx_tombstones_project ON memory_tombstones(project_key);

CREATE TABLE synaps_history (
    project_key   TEXT NOT NULL,
    id            TEXT PRIMARY KEY,
    logical       TEXT NOT NULL,
    digest        TEXT NOT NULL,
    fingerprint   TEXT,
    source_count  INTEGER NOT NULL,
    message_count INTEGER NOT NULL,
    messages      TEXT NOT NULL,
    note          TEXT NOT NULL,
    created_ms    INTEGER NOT NULL,
    tombstoned    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX synaps_history_fp ON synaps_history(project_key, fingerprint, tombstoned);

CREATE TABLE synaps_captures (
    project_key    TEXT NOT NULL,
    capture_id     TEXT PRIMARY KEY,
    note_id        TEXT NOT NULL UNIQUE,
    payload_digest TEXT NOT NULL,
    source_digest  TEXT NOT NULL,
    evidence       TEXT NOT NULL,
    withheld       INTEGER NOT NULL,
    tombstoned     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE synaps_fingerprints (
    project_key TEXT NOT NULL,
    kind        TEXT NOT NULL,
    digest      TEXT NOT NULL,
    PRIMARY KEY (project_key, kind, digest)
);

CREATE TABLE synaps_migrations (
    project_key     TEXT NOT NULL,
    migration_id    TEXT PRIMARY KEY,
    manifest_digest TEXT NOT NULL,
    committed       INTEGER NOT NULL DEFAULT 0,
    counts          TEXT,
    apply_digest    TEXT
);

CREATE TABLE synaps_scope_members (
    member    TEXT PRIMARY KEY,
    canonical TEXT NOT NULL
);
CREATE INDEX synaps_scope_group ON synaps_scope_members(canonical);
