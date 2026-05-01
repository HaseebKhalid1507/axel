//! Unified SQLite storage for VelociRAG.
//!
//! Single database file holds:
//! - Documents + embeddings (chunks)
//! - FTS5 full-text index (BM25 keyword search)
//! - Knowledge graph (nodes + edges)
//! - Metadata (tags, cross-refs, usage log)
//! - File cache (incremental indexing)
//!
//! WAL mode for concurrent read/write performance.

use std::path::{Path, PathBuf};

use chrono::Utc;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::error::{Result, VelociError};

// ── Schema version ──────────────────────────────────────────────────────────

const SCHEMA_VERSION: i32 = 2;
use crate::EMBEDDING_DIM;

// ── Types ───────────────────────────────────────────────────────────────────

/// A document chunk stored in the database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: i64,
    pub doc_id: String,
    pub content: String,
    pub metadata: serde_json::Value,
    pub file_path: Option<String>,
    pub created: String,
}

/// A knowledge graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub node_type: String,
    pub title: String,
    pub content: Option<String>,
    pub metadata: serde_json::Value,
    pub source_file: Option<String>,
}

/// A knowledge graph edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub edge_type: String,
    pub weight: f64,
    pub confidence: f64,
    pub metadata: serde_json::Value,
    pub source_file: Option<String>,
    pub valid_from: Option<chrono::DateTime<chrono::Utc>>,
    pub valid_to: Option<chrono::DateTime<chrono::Utc>>,
}

/// Result from a BM25 keyword search.
#[derive(Debug, Clone)]
pub struct FtsResult {
    pub doc_id: String,
    pub content: String,
    pub file_path: String,
    pub snippet: String,
    pub bm25_rank: f64,
}

/// File cache entry for incremental indexing.
#[derive(Debug, Clone)]
pub struct FileEntry {
    pub cache_key: String,
    pub last_modified: f64,
    pub source_name: String,
}

/// A document access event (search hit, open, reference, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentAccess {
    pub doc_id: String,
    pub access_type: String,
    pub query: Option<String>,
    pub score: Option<f64>,
    pub timestamp: String,
}

/// Stats from a single consolidation run.
#[derive(Debug, Clone, Default)]
pub struct ConsolidationLogEntry {
    pub started_at: String,
    pub finished_at: Option<String>,
    pub phase1_reindexed: i64,
    pub phase1_pruned: i64,
    pub phase2_boosted: i64,
    pub phase2_decayed: i64,
    pub phase3_edges_added: i64,
    pub phase3_edges_updated: i64,
    pub phase4_flagged: i64,
    pub phase4_removed: i64,
    pub duration_secs: Option<f64>,
}

// ── Database ────────────────────────────────────────────────────────────────

pub struct Database {
    conn: Connection,
    path: PathBuf,
}

impl Database {
    /// Open or create the unified database at the given directory.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let path = dir.join("velocirag.db");

        let conn = Connection::open(&path)?;

        // WAL mode for concurrent reads + writes
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        // Larger cache for better read performance
        conn.pragma_update(None, "cache_size", -64_000)?; // 64MB

        let mut db = Self { conn, path };
        db.init_schema()?;
        Ok(db)
    }

    /// Wrap an existing SQLite connection as a VelociRAG Database.
    ///
    /// Runs `init_schema()` with `CREATE TABLE IF NOT EXISTS`, so this
    /// is safe to call on a database that already has the velocirag tables
    /// (e.g., an Axel `.r8` brain file).
    pub fn from_connection(conn: Connection, path: PathBuf) -> Result<Self> {
        let mut db = Self { conn, path };
        db.init_schema()?;
        Ok(db)
    }

    /// Open an in-memory database (for tests).
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let mut db = Self {
            conn,
            path: PathBuf::from(":memory:"),
        };
        db.init_schema()?;
        Ok(db)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    // ── Schema ──────────────────────────────────────────────────────────

    fn init_schema(&mut self) -> Result<()> {
        self.conn.execute_batch(
            "
            -- ═══ DOCUMENTS (chunks + embeddings) ═══
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      TEXT UNIQUE NOT NULL,
                content     TEXT NOT NULL,
                metadata    TEXT NOT NULL DEFAULT '{}',
                embedding   BLOB NOT NULL,
                file_path   TEXT,
                created     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                -- L0/L1 abstract hierarchy
                l0_abstract     TEXT,
                l1_overview     TEXT,
                l0_embedding    BLOB,
                l1_embedding    BLOB
            );

            CREATE INDEX IF NOT EXISTS idx_doc_doc_id ON documents(doc_id);
            CREATE INDEX IF NOT EXISTS idx_doc_file_path ON documents(file_path);

            -- ═══ FTS5 (BM25 keyword search) ═══
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                doc_id,
                content,
                file_path,
                tokenize='porter unicode61'
            );

            -- ═══ KNOWLEDGE GRAPH ═══
            CREATE TABLE IF NOT EXISTS nodes (
                id          TEXT PRIMARY KEY,
                type        TEXT NOT NULL,
                title       TEXT NOT NULL,
                content     TEXT,
                metadata    TEXT NOT NULL DEFAULT '{}',
                source_file TEXT,
                source_name TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS edges (
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

            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
            CREATE INDEX IF NOT EXISTS idx_nodes_source_file ON nodes(source_file);
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type);
            CREATE INDEX IF NOT EXISTS idx_edges_weight ON edges(weight DESC);
            CREATE INDEX IF NOT EXISTS idx_edges_source_file ON edges(source_file);
            CREATE INDEX IF NOT EXISTS idx_edges_type_src_tgt ON edges(type, source_id, target_id);

            -- ═══ METADATA ═══
            CREATE TABLE IF NOT EXISTS tags (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT UNIQUE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS document_tags (
                doc_id  INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                tag_id  INTEGER REFERENCES tags(id) ON DELETE CASCADE,
                PRIMARY KEY (doc_id, tag_id)
            );

            CREATE TABLE IF NOT EXISTS cross_refs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                ref_type    TEXT NOT NULL,
                ref_target  TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name);
            CREATE INDEX IF NOT EXISTS idx_cross_refs_doc ON cross_refs(doc_id);
            CREATE INDEX IF NOT EXISTS idx_cross_refs_target ON cross_refs(ref_target);

            -- ═══ FILE CACHE (incremental indexing) ═══
            CREATE TABLE IF NOT EXISTS file_cache (
                cache_key       TEXT PRIMARY KEY,
                last_modified   REAL NOT NULL,
                source_name     TEXT DEFAULT '',
                last_indexed    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS file_provenance (
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

            CREATE INDEX IF NOT EXISTS idx_provenance_path ON file_provenance(file_path);

            -- ═══ KV STORE (schema version, settings) ═══
            CREATE TABLE IF NOT EXISTS kv (
                key     TEXT PRIMARY KEY,
                value   TEXT NOT NULL
            );

            -- ═══ DOCUMENT ACCESS LOG (consolidation Phase 2 input) ═══
            CREATE TABLE IF NOT EXISTS document_access (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      TEXT NOT NULL,
                access_type TEXT NOT NULL,
                query       TEXT,
                score       REAL,
                timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
                session_id  TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_docaccess_doc_id ON document_access(doc_id);
            CREATE INDEX IF NOT EXISTS idx_docaccess_timestamp ON document_access(timestamp);

            -- ═══ CO-RETRIEVAL PAIRS (consolidation Phase 3 input) ═══
            CREATE TABLE IF NOT EXISTS co_retrieval (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id_a    TEXT NOT NULL,
                doc_id_b    TEXT NOT NULL,
                query       TEXT,
                timestamp   TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_coret_pair ON co_retrieval(doc_id_a, doc_id_b);
            CREATE INDEX IF NOT EXISTS idx_coret_timestamp ON co_retrieval(timestamp);

            -- ═══ CONSOLIDATION RUN LOG ═══
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
            ",
        )?;

        // Set schema version
        self.conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES ('schema_version', ?1)",
            params![SCHEMA_VERSION.to_string()],
        )?;

        // ── Migration: add `indexed_at` to documents if missing ──
        // Used by the incremental indexer to compare file mtime vs. last-indexed time.
        let has_indexed_at: bool = {
            let mut stmt = self.conn.prepare("PRAGMA table_info(documents)")?;
            let cols: Vec<String> = stmt
                .query_map([], |row| row.get::<_, String>(1))?
                .filter_map(|r| r.ok())
                .collect();
            cols.iter().any(|c| c == "indexed_at")
        };
        if !has_indexed_at {
            // CURRENT_TIMESTAMP isn't allowed as a default for ALTER TABLE ADD COLUMN,
            // so use NULL default and backfill from `created`.
            self.conn.execute(
                "ALTER TABLE documents ADD COLUMN indexed_at TIMESTAMP",
                [],
            )?;
            self.conn.execute(
                "UPDATE documents SET indexed_at = COALESCE(created, CURRENT_TIMESTAMP) WHERE indexed_at IS NULL",
                [],
            )?;
        }

        // ── Migration: add consolidation columns to documents ──
        let doc_cols: Vec<String> = {
            let mut stmt = self.conn.prepare("PRAGMA table_info(documents)")?;
            let cols: Vec<String> = stmt
                .query_map([], |row| row.get::<_, String>(1))?
                .filter_map(|r| r.ok())
                .collect();
            cols
        };
        if !doc_cols.iter().any(|c| c == "access_count") {
            self.conn.execute(
                "ALTER TABLE documents ADD COLUMN access_count INTEGER DEFAULT 0",
                [],
            )?;
        }
        if !doc_cols.iter().any(|c| c == "last_accessed") {
            self.conn.execute(
                "ALTER TABLE documents ADD COLUMN last_accessed TIMESTAMP",
                [],
            )?;
        }
        if !doc_cols.iter().any(|c| c == "excitability") {
            self.conn.execute(
                "ALTER TABLE documents ADD COLUMN excitability REAL DEFAULT 0.5",
                [],
            )?;
        }

        // Indexes on migrated columns (must come after ALTER TABLE).
        self.conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_doc_excitability ON documents(excitability);
             CREATE INDEX IF NOT EXISTS idx_doc_access_count ON documents(access_count);
             CREATE INDEX IF NOT EXISTS idx_doc_last_accessed ON documents(last_accessed);"
        )?;

        // ── Migration: add valid_from/valid_to to edges if missing ──
        let edge_cols: Vec<String> = self.conn
            .prepare("PRAGMA table_info(edges)")?
            .query_map([], |row| row.get::<_, String>(1))?
            .filter_map(|r| r.ok())
            .collect();
        if !edge_cols.is_empty() && !edge_cols.iter().any(|c| c == "valid_from") {
            self.conn.execute(
                "ALTER TABLE edges ADD COLUMN valid_from TEXT",
                [],
            )?;
        }
        if !edge_cols.is_empty() && !edge_cols.iter().any(|c| c == "valid_to") {
            self.conn.execute(
                "ALTER TABLE edges ADD COLUMN valid_to TEXT",
                [],
            )?;
        }

        Ok(())
    }

    // ── Document CRUD ───────────────────────────────────────────────────

    /// Insert a document chunk with its embedding.
    pub fn insert_document(
        &self,
        doc_id: &str,
        content: &str,
        metadata: &serde_json::Value,
        embedding: &[f32],
        file_path: Option<&str>,
    ) -> Result<i64> {
        if embedding.len() != EMBEDDING_DIM {
            return Err(VelociError::DimensionMismatch {
                expected: EMBEDDING_DIM,
                got: embedding.len(),
            });
        }

        let metadata_json = serde_json::to_string(metadata)?;
        let embedding_blob = embedding_to_blob(embedding);

        let _id = self.conn.execute(
            "INSERT OR REPLACE INTO documents (doc_id, content, metadata, embedding, file_path, indexed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, CURRENT_TIMESTAMP)",
            params![doc_id, content, metadata_json, embedding_blob, file_path],
        )?;
        let rowid = self.conn.last_insert_rowid();

        // Sync FTS5
        self.conn.execute(
            "DELETE FROM chunks_fts WHERE doc_id = ?1",
            params![doc_id],
        )?;
        self.conn.execute(
            "INSERT INTO chunks_fts (doc_id, content, file_path) VALUES (?1, ?2, ?3)",
            params![doc_id, content, file_path.unwrap_or("")],
        )?;

        Ok(rowid)
    }

    /// Get a document by doc_id.
    pub fn get_document(&self, doc_id: &str) -> Result<Option<Document>> {
        let result = self
            .conn
            .query_row(
                "SELECT id, doc_id, content, metadata, file_path, created
                 FROM documents WHERE doc_id = ?1",
                params![doc_id],
                |row| {
                    Ok(Document {
                        id: row.get(0)?,
                        doc_id: row.get(1)?,
                        content: row.get(2)?,
                        metadata: serde_json::from_str(&row.get::<_, String>(3)?).unwrap_or_default(),
                        file_path: row.get(4)?,
                        created: row.get(5)?,
                    })
                },
            )
            .optional()?;
        Ok(result)
    }

    /// Get embedding for a document by rowid.
    pub fn get_embedding(&self, rowid: i64) -> Result<Option<Vec<f32>>> {
        let result = self
            .conn
            .query_row(
                "SELECT embedding FROM documents WHERE id = ?1",
                params![rowid],
                |row| {
                    let blob: Vec<u8> = row.get(0)?;
                    Ok(blob_to_embedding(&blob))
                },
            )
            .optional()?;
        Ok(result)
    }

    /// Load all embeddings as (rowid, Vec<f32>) for building the vector index.
    pub fn load_all_embeddings(&self) -> Result<Vec<(i64, Vec<f32>)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, embedding FROM documents ORDER BY id")?;
        let rows = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((id, blob_to_embedding(&blob)))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Count of documents.
    pub fn document_count(&self) -> Result<usize> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0))?;
        Ok(count as usize)
    }

    /// Delete all documents for a given file path (for re-indexing).
    pub fn delete_documents_by_file(&self, file_path: &str) -> Result<usize> {
        // Delete FTS entries
        self.conn.execute(
            "DELETE FROM chunks_fts WHERE doc_id IN (SELECT doc_id FROM documents WHERE file_path = ?1)",
            params![file_path],
        )?;
        // Delete documents
        let deleted = self.conn.execute(
            "DELETE FROM documents WHERE file_path = ?1",
            params![file_path],
        )?;
        Ok(deleted)
    }

    /// Return (file_path, indexed_at_unix_seconds) for every indexed file
    /// whose `file_path` starts with `prefix`. Used by the incremental indexer
    /// to decide which files need re-indexing and which to prune.
    pub fn indexed_files_under(&self, prefix: &str) -> Result<Vec<(String, f64)>> {
        let pattern = format!("{}%", escape_like(prefix));
        let mut stmt = self.conn.prepare(
            "SELECT file_path,
                    COALESCE(CAST(strftime('%s', indexed_at) AS REAL),
                             CAST(strftime('%s', created) AS REAL),
                             0)
             FROM documents
             WHERE file_path IS NOT NULL
               AND file_path LIKE ?1 ESCAPE '\\'",
        )?;
        let rows = stmt.query_map(params![pattern], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows { out.push(r?); }
        Ok(out)
    }

    // ── BM25 / FTS5 ────────────────────────────────────────────────────

    /// BM25 keyword search via FTS5.
    pub fn keyword_search(&self, query: &str, limit: usize) -> Result<Vec<FtsResult>> {
        // Sanitize query for FTS5
        let safe_query = sanitize_fts5_query(query);
        if safe_query.is_empty() {
            return Ok(Vec::new());
        }

        let mut stmt = self.conn.prepare(
            "SELECT doc_id,
                    content,
                    file_path,
                    snippet(chunks_fts, 1, '', '', '...', 64) as snippet,
                    rank
             FROM chunks_fts
             WHERE chunks_fts MATCH ?1
             ORDER BY rank
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(params![safe_query, limit as i64], |row| {
            Ok(FtsResult {
                doc_id: row.get(0)?,
                content: row.get(1)?,
                file_path: row.get(2)?,
                snippet: row.get(3)?,
                bm25_rank: row.get(4)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Knowledge Graph ─────────────────────────────────────────────────

    /// Insert or replace a graph node.
    pub fn upsert_node(&self, node: &Node) -> Result<()> {
        let metadata_json = serde_json::to_string(&node.metadata)?;
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "INSERT OR REPLACE INTO nodes (id, type, title, content, metadata, source_file, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                node.id,
                node.node_type,
                node.title,
                node.content,
                metadata_json,
                node.source_file,
                now,
            ],
        )?;
        Ok(())
    }

    /// Insert or replace a graph edge.
    pub fn upsert_edge(&self, edge: &Edge) -> Result<()> {
        let metadata_json = serde_json::to_string(&edge.metadata)?;
        let now = Utc::now().to_rfc3339();
        let valid_from = edge.valid_from.map(|dt| dt.to_rfc3339());
        let valid_to = edge.valid_to.map(|dt| dt.to_rfc3339());
        
        self.conn.execute(
            "INSERT OR REPLACE INTO edges (id, source_id, target_id, type, weight, confidence, metadata, source_file, created_at, valid_from, valid_to)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                edge.id,
                edge.source_id,
                edge.target_id,
                edge.edge_type,
                edge.weight,
                edge.confidence,
                metadata_json,
                edge.source_file,
                now,
                valid_from,
                valid_to,
            ],
        )?;
        Ok(())
    }

    /// Invalidate an edge by setting its valid_to timestamp to now.
    pub fn invalidate_edge(&self, edge_id: &str) -> Result<bool> {
        let now = Utc::now().to_rfc3339();
        let rows_affected = self.conn.execute(
            "UPDATE edges SET valid_to = ?1 WHERE id = ?2",
            params![now, edge_id],
        )?;
        Ok(rows_affected > 0)
    }

    /// Convenience: insert a node from individual fields.
    pub fn insert_node(
        &self,
        id: &str,
        node_type: &str,
        title: &str,
        content: Option<&str>,
        metadata: &serde_json::Value,
        source_file: Option<&str>,
    ) -> Result<()> {
        self.upsert_node(&Node {
            id: id.to_string(),
            node_type: node_type.to_string(),
            title: title.to_string(),
            content: content.map(|s| s.to_string()),
            metadata: metadata.clone(),
            source_file: source_file.map(|s| s.to_string()),
        })
    }

    /// Convenience: insert an edge from individual fields.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_edge(
        &self,
        id: &str,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
        weight: f64,
        confidence: f64,
        metadata: &serde_json::Value,
        source_file: Option<&str>,
        valid_from: Option<chrono::DateTime<chrono::Utc>>,
        valid_to: Option<chrono::DateTime<chrono::Utc>>,
    ) -> Result<()> {
        self.upsert_edge(&Edge {
            id: id.to_string(),
            source_id: source_id.to_string(),
            target_id: target_id.to_string(),
            edge_type: edge_type.to_string(),
            weight,
            confidence,
            metadata: metadata.clone(),
            source_file: source_file.map(|s| s.to_string()),
            valid_from,
            valid_to,
        })
    }

    /// Get a node by ID.
    pub fn get_node(&self, node_id: &str) -> Result<Option<Node>> {
        let result = self.conn.query_row(
            "SELECT id, type, title, content, metadata, source_file FROM nodes WHERE id = ?1",
            [node_id],
            |row| {
                let metadata_str: String = row.get(4)?;
                Ok(Node {
                    id: row.get(0)?,
                    node_type: row.get(1)?,
                    title: row.get(2)?,
                    content: row.get(3)?,
                    metadata: serde_json::from_str(&metadata_str).unwrap_or_default(),
                    source_file: row.get(5)?,
                })
            },
        );

        match result {
            Ok(node) => Ok(Some(node)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get neighbors of a node within a given depth.
    pub fn get_neighbors(
        &self,
        node_id: &str,
        depth: usize,
        max_results: usize,
    ) -> Result<Vec<(Node, Edge)>> {
        // BFS traversal via recursive CTE
        let depth = depth.min(3); // cap at 3
        let mut stmt = self.conn.prepare(
            "WITH RECURSIVE reachable(node_id, depth) AS (
                 SELECT ?1, 0
                 UNION
                 SELECT CASE
                     WHEN e.source_id = reachable.node_id THEN e.target_id
                     ELSE e.source_id
                 END, reachable.depth + 1
                 FROM edges e
                 JOIN reachable ON (e.source_id = reachable.node_id OR e.target_id = reachable.node_id)
                 WHERE reachable.depth < ?2
                   AND (e.valid_to IS NULL OR e.valid_to > datetime('now'))
             )
             SELECT DISTINCT n.id, n.type, n.title, n.content, n.metadata, n.source_file,
                             e.id, e.source_id, e.target_id, e.type, e.weight, e.confidence, e.metadata, e.source_file, e.valid_from, e.valid_to
             FROM reachable r
             JOIN nodes n ON n.id = r.node_id
             JOIN edges e ON (e.source_id = r.node_id OR e.target_id = r.node_id)
             WHERE r.node_id != ?1
               AND (e.valid_to IS NULL OR e.valid_to > datetime('now'))
             ORDER BY e.weight DESC
             LIMIT ?3",
        )?;

        let rows = stmt.query_map(params![node_id, depth as i64, max_results as i64], |row| {
            let node = Node {
                id: row.get(0)?,
                node_type: row.get(1)?,
                title: row.get(2)?,
                content: row.get(3)?,
                metadata: serde_json::from_str(&row.get::<_, String>(4)?).unwrap_or_default(),
                source_file: row.get(5)?,
            };
            let edge = Edge {
                id: row.get(6)?,
                source_id: row.get(7)?,
                target_id: row.get(8)?,
                edge_type: row.get(9)?,
                weight: row.get(10)?,
                confidence: row.get(11)?,
                metadata: serde_json::from_str(&row.get::<_, String>(12)?).unwrap_or_default(),
                source_file: row.get(13)?,
                valid_from: row.get::<_, Option<String>>(14)?.map(|s| chrono::DateTime::parse_from_rfc3339(&s).unwrap().into()),
                valid_to: row.get::<_, Option<String>>(15)?.map(|s| chrono::DateTime::parse_from_rfc3339(&s).unwrap().into()),
            };
            Ok((node, edge))
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Count nodes and edges.
    pub fn graph_stats(&self) -> Result<(usize, usize)> {
        let nodes: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM nodes", [], |row| row.get(0))?;
        let edges: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM edges", [], |row| row.get(0))?;
        Ok((nodes as usize, edges as usize))
    }

    // ── Metadata (tags + cross-refs) ────────────────────────────────────

    /// Insert a tag and return its ID. If it already exists, return existing ID.
    pub fn upsert_tag(&self, tag_name: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT OR IGNORE INTO tags (name) VALUES (?1)",
            params![tag_name],
        )?;
        let id: i64 = self.conn.query_row(
            "SELECT id FROM tags WHERE name = ?1",
            params![tag_name],
            |row| row.get(0),
        )?;
        Ok(id)
    }

    /// Link a document to a tag.
    pub fn tag_document(&self, doc_rowid: i64, tag_id: i64) -> Result<()> {
        self.conn.execute(
            "INSERT OR IGNORE INTO document_tags (doc_id, tag_id) VALUES (?1, ?2)",
            params![doc_rowid, tag_id],
        )?;
        Ok(())
    }

    /// Insert a cross-reference from a document to a target.
    pub fn insert_cross_ref(&self, doc_rowid: i64, ref_type: &str, ref_target: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO cross_refs (doc_id, ref_type, ref_target) VALUES (?1, ?2, ?3)",
            params![doc_rowid, ref_type, ref_target],
        )?;
        Ok(())
    }

    /// Search documents by tag name. Returns matching doc_ids and content.
    pub fn search_by_tag(&self, tag_pattern: &str, limit: usize) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT d.doc_id, d.content
             FROM documents d
             JOIN document_tags dt ON d.id = dt.doc_id
             JOIN tags t ON dt.tag_id = t.id
             WHERE LOWER(t.name) LIKE LOWER(?1) ESCAPE '\\'
             LIMIT ?2"
        )?;
        let rows = stmt.query_map(params![format!("%{}%", escape_like(tag_pattern)), limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Search documents by cross-reference target. Returns matching doc_ids and content.
    pub fn search_by_cross_ref(&self, target_pattern: &str, limit: usize) -> Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT d.doc_id, d.content
             FROM documents d
             JOIN cross_refs cr ON d.id = cr.doc_id
             WHERE LOWER(cr.ref_target) LIKE LOWER(?1) ESCAPE '\\'
             LIMIT ?2"
        )?;
        let rows = stmt.query_map(params![format!("%{}%", escape_like(target_pattern)), limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Get metadata stats.
    pub fn metadata_stats(&self) -> Result<(usize, usize, usize)> {
        let tags: usize = self.conn.query_row("SELECT COUNT(*) FROM tags", [], |r| r.get(0))?;
        let doc_tags: usize = self.conn.query_row("SELECT COUNT(*) FROM document_tags", [], |r| r.get(0))?;
        let cross_refs: usize = self.conn.query_row("SELECT COUNT(*) FROM cross_refs", [], |r| r.get(0))?;
        Ok((tags, doc_tags, cross_refs))
    }

    // ── File cache ──────────────────────────────────────────────────────

    /// Check if a file needs re-indexing.
    pub fn file_needs_update(&self, cache_key: &str, current_mtime: f64) -> Result<bool> {
        let stored: Option<f64> = self
            .conn
            .query_row(
                "SELECT last_modified FROM file_cache WHERE cache_key = ?1",
                params![cache_key],
                |row| row.get(0),
            )
            .optional()?;

        match stored {
            Some(stored_mtime) => Ok((current_mtime - stored_mtime).abs() > 0.001),
            None => Ok(true),
        }
    }

    /// Update file cache entry.
    pub fn update_file_cache(&self, cache_key: &str, mtime: f64, source_name: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO file_cache (cache_key, last_modified, source_name) VALUES (?1, ?2, ?3)",
            params![cache_key, mtime, source_name],
        )?;
        Ok(())
    }

    // ── Consolidation: access logging ───────────────────────────────────

    /// Log a document access event.
    pub fn log_document_access(
        &self,
        doc_id: &str,
        access_type: &str,
        query: Option<&str>,
        score: Option<f64>,
        session_id: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "INSERT INTO document_access (doc_id, access_type, query, score, session_id)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![doc_id, access_type, query, score, session_id],
        )?;
        Ok(())
    }

    /// Increment access_count and update last_accessed on the documents table.
    pub fn increment_document_access(&self, doc_id: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE documents
             SET access_count = COALESCE(access_count, 0) + 1,
                 last_accessed = CURRENT_TIMESTAMP
             WHERE doc_id = ?1",
            params![doc_id],
        )?;
        Ok(())
    }

    /// Log a co-retrieval pair (orders the pair canonically so a < b).
    pub fn log_co_retrieval(
        &self,
        doc_id_a: &str,
        doc_id_b: &str,
        query: &str,
    ) -> Result<()> {
        if doc_id_a == doc_id_b {
            return Ok(());
        }
        let (a, b) = if doc_id_a < doc_id_b {
            (doc_id_a, doc_id_b)
        } else {
            (doc_id_b, doc_id_a)
        };
        self.conn.execute(
            "INSERT INTO co_retrieval (doc_id_a, doc_id_b, query) VALUES (?1, ?2, ?3)",
            params![a, b, query],
        )?;
        Ok(())
    }

    /// Get document access events since a given RFC3339 timestamp.
    pub fn get_document_accesses_since(&self, since: &str) -> Result<Vec<DocumentAccess>> {
        // Normalize: strip any timezone suffix and replace 'T' with ' '
        // so we compare apples-to-apples with SQLite's datetime() format.
        let normalized = since
            .replace('T', " ")
            .split('+').next().unwrap_or(since)
            .split('Z').next().unwrap_or(since)
            .to_string();
        let mut stmt = self.conn.prepare(
            "SELECT doc_id, access_type, query, score, timestamp
             FROM document_access
             WHERE timestamp >= ?1
             ORDER BY timestamp ASC",
        )?;
        let rows = stmt.query_map(params![normalized], |row| {
            Ok(DocumentAccess {
                doc_id: row.get(0)?,
                access_type: row.get(1)?,
                query: row.get(2)?,
                score: row.get(3)?,
                timestamp: row.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r?);
        }
        Ok(out)
    }

    /// Get the most recent consolidation run's finished_at timestamp.
    pub fn last_consolidation_time(&self) -> Result<Option<String>> {
        let result = self
            .conn
            .query_row(
                "SELECT finished_at FROM consolidation_log
                 WHERE finished_at IS NOT NULL
                 ORDER BY finished_at DESC LIMIT 1",
                [],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?;
        Ok(result.flatten())
    }

    /// Insert a consolidation log entry.
    pub fn insert_consolidation_log(&self, log: &ConsolidationLogEntry) -> Result<()> {
        self.conn.execute(
            "INSERT INTO consolidation_log (
                started_at, finished_at,
                phase1_reindexed, phase1_pruned,
                phase2_boosted, phase2_decayed,
                phase3_edges_added, phase3_edges_updated,
                phase4_flagged, phase4_removed,
                duration_secs
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                log.started_at,
                log.finished_at,
                log.phase1_reindexed,
                log.phase1_pruned,
                log.phase2_boosted,
                log.phase2_decayed,
                log.phase3_edges_added,
                log.phase3_edges_updated,
                log.phase4_flagged,
                log.phase4_removed,
                log.duration_secs,
            ],
        )?;
        Ok(())
    }

    /// Insert a consolidation log entry at start (finished_at NULL), return rowid.
    pub fn start_consolidation_log(&self, started_at: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO consolidation_log (started_at, finished_at,
                phase1_reindexed, phase1_pruned,
                phase2_boosted, phase2_decayed,
                phase3_edges_added, phase3_edges_updated,
                phase4_flagged, phase4_removed,
                duration_secs
             ) VALUES (?1, NULL, 0, 0, 0, 0, 0, 0, 0, 0, NULL)",
            params![started_at],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Update an in-progress consolidation log row with final stats.
    pub fn update_consolidation_log(&self, id: i64, log: &ConsolidationLogEntry) -> Result<()> {
        self.conn.execute(
            "UPDATE consolidation_log SET
                finished_at = ?2,
                phase1_reindexed = ?3, phase1_pruned = ?4,
                phase2_boosted = ?5, phase2_decayed = ?6,
                phase3_edges_added = ?7, phase3_edges_updated = ?8,
                phase4_flagged = ?9, phase4_removed = ?10,
                duration_secs = ?11
             WHERE id = ?1",
            params![
                id,
                log.finished_at,
                log.phase1_reindexed,
                log.phase1_pruned,
                log.phase2_boosted,
                log.phase2_decayed,
                log.phase3_edges_added,
                log.phase3_edges_updated,
                log.phase4_flagged,
                log.phase4_removed,
                log.duration_secs,
            ],
        )?;
        Ok(())
    }

    /// Execute a closure inside a transaction.
    pub fn transaction<F, T>(&mut self, f: F) -> Result<T>
    where
        F: FnOnce(&Connection) -> Result<T>,
    {
        let tx = self.conn.transaction()?;
        let result = f(&tx)?;
        tx.commit()?;
        Ok(result)
    }

    /// Raw connection access (for advanced queries).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Escape SQLite LIKE metacharacters so user input cannot alter the pattern.
///
/// Uses `\` as the escape character, which must be declared in the SQL clause
/// with `ESCAPE '\\'`.
fn escape_like(input: &str) -> String {
    input.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
}

/// Convert f32 slice to bytes for BLOB storage.
fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    embedding
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

/// Convert BLOB bytes back to f32 vec.
fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Sanitize a query string for FTS5 MATCH.
/// Uses OR between terms for better recall on multi-word queries.
fn sanitize_fts5_query(query: &str) -> String {
    const FTS5_SPECIAL: &[char] = &['"', '\'', '\\', '(', ')', '{', '}', '[', ']', '*', '^', ':', '~', '@', '#', '$', '%', '&', '|', '<', '>', '!'];

    let words: Vec<String> = query
        .split_whitespace()
        .filter_map(|word| {
            let cleaned: String = word
                .chars()
                .filter(|c| !FTS5_SPECIAL.contains(c))
                .collect();
            let cleaned = cleaned.trim_matches('-').to_string();
            if cleaned.is_empty() || cleaned.len() < 2 {
                None
            } else {
                Some(format!("\"{}\"", cleaned))
            }
        })
        .collect();

    // Use OR for multi-word queries to get partial matches
    // Single word: just quote it. Multi-word: OR them.
    if words.len() <= 1 {
        words.join("")
    } else {
        words.join(" OR ")
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_memory() {
        let db = Database::open_memory().unwrap();
        assert_eq!(db.document_count().unwrap(), 0);
    }

    #[test]
    fn test_insert_and_get_document() {
        let db = Database::open_memory().unwrap();
        let embedding = vec![0.1f32; EMBEDDING_DIM];
        let metadata = serde_json::json!({"source": "test"});

        let id = db
            .insert_document("doc_1", "hello world", &metadata, &embedding, Some("test.md"))
            .unwrap();
        assert!(id > 0);

        let doc = db.get_document("doc_1").unwrap().unwrap();
        assert_eq!(doc.content, "hello world");
        assert_eq!(doc.doc_id, "doc_1");
    }

    #[test]
    fn test_keyword_search() {
        let db = Database::open_memory().unwrap();
        let embedding = vec![0.1f32; EMBEDDING_DIM];
        let meta = serde_json::json!({});

        db.insert_document("d1", "rust programming language", &meta, &embedding, Some("a.md"))
            .unwrap();
        db.insert_document("d2", "python scripting language", &meta, &embedding, Some("b.md"))
            .unwrap();
        db.insert_document("d3", "rust borrow checker", &meta, &embedding, Some("c.md"))
            .unwrap();

        let results = db.keyword_search("rust", 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_dimension_mismatch() {
        let db = Database::open_memory().unwrap();
        let bad_embedding = vec![0.1f32; 128]; // wrong size
        let meta = serde_json::json!({});

        let result = db.insert_document("doc_1", "test", &meta, &bad_embedding, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_operations() {
        let db = Database::open_memory().unwrap();

        let node = Node {
            id: "n1".to_string(),
            node_type: "concept".to_string(),
            title: "Rust".to_string(),
            content: Some("Systems programming language".to_string()),
            metadata: serde_json::json!({}),
            source_file: None,
        };
        db.upsert_node(&node).unwrap();

        let node2 = Node {
            id: "n2".to_string(),
            node_type: "concept".to_string(),
            title: "Memory Safety".to_string(),
            content: None,
            metadata: serde_json::json!({}),
            source_file: None,
        };
        db.upsert_node(&node2).unwrap();

        let edge = Edge {
            id: "e1".to_string(),
            source_id: "n1".to_string(),
            target_id: "n2".to_string(),
            edge_type: "related_to".to_string(),
            weight: 0.9,
            confidence: 0.85,
            metadata: serde_json::json!({}),
            source_file: None,
            valid_from: None,
            valid_to: None,
        };
        db.upsert_edge(&edge).unwrap();

        let (nodes, edges) = db.graph_stats().unwrap();
        assert_eq!(nodes, 2);
        assert_eq!(edges, 1);

        let neighbors = db.get_neighbors("n1", 1, 10).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0.title, "Memory Safety");
    }

    #[test]
    fn test_file_cache() {
        let db = Database::open_memory().unwrap();
        assert!(db.file_needs_update("test.md", 1000.0).unwrap());

        db.update_file_cache("test.md", 1000.0, "").unwrap();
        assert!(!db.file_needs_update("test.md", 1000.0).unwrap());
        assert!(db.file_needs_update("test.md", 2000.0).unwrap());
    }

    #[test]
    fn test_sanitize_fts5() {
        assert_eq!(sanitize_fts5_query("hello world"), "\"hello\" OR \"world\"");
        assert_eq!(sanitize_fts5_query("rust's (cool)"), "\"rusts\" OR \"cool\"");
        assert_eq!(sanitize_fts5_query(""), "");
    }
}
