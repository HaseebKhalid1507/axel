//! The `.r8` brain format — a single SQLite file containing an agent's
//! complete knowledge state: documents, embeddings, memories, graph,
//! metadata, session history, and patterns.
//!
//! # Design
//!
//! A `.r8` file IS a SQLite database. One file = one brain.
//! The USearch HNSW index is an accelerator cached externally,
//! rebuilt from embeddings stored in SQLite on cold start (~70ms for 7K docs).
//!
//! ```text
//! axel.r8  (SQLite, WAL mode)
//!   ├── documents          — chunked text + embedding BLOBs
//!   ├── documents_fts      — FTS5 full-text index
//!   ├── nodes / edges      — knowledge graph
//!   ├── tags / document_tags / cross_refs — metadata layer
//!   ├── memories           — approved agent memories
//!   ├── staged_memories    — pending review
//!   ├── memory_access      — access log for decay/boost
//!   ├── events             — event log for pattern detection
//!   ├── context_data       — key/value context per layer
//!   ├── sessions           — processed session log
//!   ├── patterns           — detected behavioral patterns
//!   └── brain_meta         — schema version, model info, agent name
//! ```

use std::path::Path;

use chrono::Utc;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::error::{AxelError, Result};

/// Current schema version. Bump on any table/column change.
pub const SCHEMA_VERSION: i32 = 1;

/// `brain_meta.schema_version` of a v2 brain. Only migration 1 (`brain_v2.rs`) writes it;
/// `Brain::create*` still writes [`SCHEMA_VERSION`].
pub const V2_SCHEMA_VERSION: i32 = 2;

/// Default embedding model name (recorded in brain_meta).
pub const DEFAULT_EMBEDDER: &str = "all-MiniLM-L6-v2";

/// Embedding dimensionality for the default model.
pub const EMBEDDING_DIM: usize = 384;

/// Brain metadata stored in the `brain_meta` table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainMeta {
    pub schema_version: i32,
    pub embedder_model: String,
    pub embedding_dim: i32,
    pub agent_name: Option<String>,
    pub created: String,
    pub last_modified: String,
    pub document_count: i64,
    pub memory_count: i64,
    /// Hex-encoded HMAC signing key (auto-generated on brain creation).
    /// Used to sign memories and detect tampering.
    #[serde(default)]
    pub signing_key: Option<String>,
}

/// Handle to an open `.r8` brain file.
pub struct Brain {
    conn: Connection,
    meta: BrainMeta,
}

impl Brain {
    /// Create a new `.r8` brain file at the given path, recorded as built
    /// with the default embedding model.
    /// Fails if the file already exists.
    pub fn create(path: &Path, agent_name: Option<&str>) -> Result<Self> {
        Self::create_with_model(path, agent_name, DEFAULT_EMBEDDER, EMBEDDING_DIM)
    }

    /// Create a new `.r8` brain file recorded as built with the embedding
    /// model `model` of `dim` dimensions (`brain_meta.embedder_model`,
    /// `embedding_dim`). Search refuses to open a brain with an embedder of
    /// another model.
    /// Fails if the file already exists.
    pub fn create_with_model(
        path: &Path,
        agent_name: Option<&str>,
        model: &str,
        dim: usize,
    ) -> Result<Self> {
        if path.exists() {
            return Err(AxelError::Other(format!(
                "Brain already exists at {}",
                path.display()
            )));
        }

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(path)?;
        Self::init_schema(&conn)?;

        let meta = Self::new_meta(agent_name, model, dim)?;
        conn.execute(
            "INSERT INTO brain_meta (key, value) VALUES (?1, ?2)",
            params!["meta", serde_json::to_string(&meta)?],
        )?;

        Ok(Self { conn, meta })
    }

    /// `brain_v2::open_v2`'s creation, safe for two processes at once: the tables and the
    /// `brain_meta` row are written in one transaction, so another opener sees a file with
    /// no tables or a whole brain, never a part. The path may be missing or a file with no
    /// tables. The opener that loses the race joins the brain the other one made.
    // RULING: wave2-c4-review finding 4.
    pub(crate) fn create_or_join(
        path: &Path,
        agent_name: Option<&str>,
        model: &str,
        dim: usize,
    ) -> Result<Self> {
        // RULING: wave2-c4-review finding 7: the file is 0600 before the signing key is
        // written. It is created empty with that mode here, and a file that was already
        // there is set to it, before SQLite opens it; `-wal` and `-shm` take the mode of the
        // main file. Not the umask: it belongs to the whole process, and changing it around
        // the create would change the mode of a file another thread makes at that moment.
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            std::fs::OpenOptions::new().append(true).create(true).mode(0o600).open(path)?;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
        }
        let conn = Connection::open(path)?;
        crate::busy::set(&conn, std::time::Duration::from_secs(5))?;
        // Two connections that switch an empty file to WAL at the same moment: one of them
        // gets `busy` at once, without the timeout. It asks again.
        let start = std::time::Instant::now();
        loop {
            match conn.execute_batch("PRAGMA journal_mode=WAL") {
                Err(e)
                    if matches!(
                        e.sqlite_error_code(),
                        Some(rusqlite::ErrorCode::DatabaseBusy) | Some(rusqlite::ErrorCode::DatabaseLocked)
                    ) && start.elapsed() < std::time::Duration::from_secs(5) =>
                {
                    std::thread::sleep(std::time::Duration::from_millis(5))
                }
                other => break other?,
            }
        }
        conn.execute_batch("PRAGMA synchronous=NORMAL; BEGIN IMMEDIATE;")?;
        let made = (|| -> Result<Option<BrainMeta>> {
            let there: bool = conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'brain_meta')",
                [],
                |r| r.get(0),
            )?;
            if there && conn.query_row("SELECT EXISTS(SELECT 1 FROM brain_meta WHERE key = 'meta')", [], |r| r.get(0))? {
                return Ok(None);
            }
            Self::init_tables(&conn)?;
            let meta = Self::new_meta(agent_name, model, dim)?;
            conn.execute(
                "INSERT INTO brain_meta (key, value) VALUES (?1, ?2)",
                params!["meta", serde_json::to_string(&meta)?],
            )?;
            Ok(Some(meta))
        })();
        let meta = match made {
            Ok(Some(meta)) => match conn.execute_batch("COMMIT") {
                Ok(()) => meta,
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    return Err(e.into());
                }
            },
            Ok(None) => {
                conn.execute_batch("ROLLBACK")?;
                let meta = Self::load_meta(&conn)?;
                if meta.schema_version != SCHEMA_VERSION && meta.schema_version != V2_SCHEMA_VERSION {
                    return Err(AxelError::SchemaMismatch { file: meta.schema_version, expected: SCHEMA_VERSION });
                }
                meta
            }
            Err(e) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(e);
            }
        };
        Ok(Self { conn, meta })
    }

    /// The metadata of a new brain, with a fresh signing key.
    fn new_meta(agent_name: Option<&str>, model: &str, dim: usize) -> Result<BrainMeta> {
        let now = Utc::now().to_rfc3339();

        // Generate a random 32-byte signing key for HMAC-SHA256
        let signing_key = {
            let mut key = [0u8; 32];
            use std::io::Read;
            std::fs::File::open("/dev/urandom")
                .and_then(|mut f| f.read_exact(&mut key))
                .map_err(|e| AxelError::Other(format!("Failed to generate signing key: {e}")))?;
            hex::encode(key)
        };

        Ok(BrainMeta {
            schema_version: SCHEMA_VERSION,
            embedder_model: model.to_string(),
            embedding_dim: dim as i32,
            agent_name: agent_name.map(|s| s.to_string()),
            created: now.clone(),
            last_modified: now,
            document_count: 0,
            memory_count: 0,
            signing_key: Some(signing_key),
        })
    }

    /// Open an existing `.r8` brain file.
    pub fn open(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Err(AxelError::NotFound(path.display().to_string()));
        }

        let conn = Connection::open(path)?;

        // Enable WAL mode for concurrent read performance
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;

        let meta = Self::load_meta(&conn)?;

        if meta.schema_version != SCHEMA_VERSION && meta.schema_version != V2_SCHEMA_VERSION {
            return Err(AxelError::SchemaMismatch {
                file: meta.schema_version,
                expected: SCHEMA_VERSION,
            });
        }

        Ok(Self { conn, meta })
    }

    /// Open existing or create new brain at path.
    pub fn open_or_create(path: &Path, agent_name: Option<&str>) -> Result<Self> {
        if path.exists() {
            Self::open(path)
        } else {
            Self::create(path, agent_name)
        }
    }

    /// Get a reference to the brain metadata.
    pub fn meta(&self) -> &BrainMeta {
        &self.meta
    }

    /// Get a MemorySigner using this brain's signing key.
    /// Returns None if the brain has no signing key (legacy brains).
    pub fn signer(&self) -> Option<axel_memkoshi::security::MemorySigner> {
        self.meta.signing_key.as_ref().map(|hex_key| {
            let key = hex::decode(hex_key).unwrap_or_else(|_| hex_key.as_bytes().to_vec());
            axel_memkoshi::security::MemorySigner::new(&key)
        })
    }

    /// Get a reference to the underlying SQLite connection.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Update the last_modified timestamp and counts.
    pub fn touch(&mut self) -> Result<()> {
        let mem_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM memories",
            [],
            |r| r.get(0),
        )?;

        // documents table may not exist if velocirag hasn't been initialized
        let doc_count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='documents'",
            [],
            |r| r.get(0),
        ).and_then(|exists: i64| {
            if exists > 0 {
                self.conn.query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
            } else {
                Ok(0)
            }
        })?;

        // RULING: wave2-c4-review finding 8: only the three fields are written, into the
        // value that is stored now. Writing the whole value from this handle would put back
        // what the handle read when it was opened: `schema_version` 1 on a brain that was
        // upgraded to version 2 since.
        self.conn.execute(
            "UPDATE brain_meta SET value = json_set(value, '$.document_count', ?1,
                 '$.memory_count', ?2, '$.last_modified', ?3) WHERE key = 'meta'",
            params![doc_count, mem_count, Utc::now().to_rfc3339()],
        )?;
        self.meta = Self::load_meta(&self.conn)?;

        Ok(())
    }

    /// Initialize the complete schema for a new brain.
    ///
    /// Creates only the Memkoshi + Axel-specific tables. VelociRAG tables
    /// (documents, FTS, nodes, edges, tags, etc.) are created separately
    /// when a `velocirag::db::Database` is constructed via `from_connection()`.
    fn init_schema(conn: &Connection) -> Result<()> {
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        Self::init_tables(conn)
    }

    /// The tables of `init_schema` without its pragmas: this can run inside a transaction.
    fn init_tables(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "
            -- Brain metadata (key-value, single row for 'meta')
            CREATE TABLE IF NOT EXISTS brain_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- ═══ Memkoshi Layer ═══

            CREATE TABLE IF NOT EXISTS memories (
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
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created);

            CREATE TABLE IF NOT EXISTS staged_memories (
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

            CREATE TABLE IF NOT EXISTS memory_access (
                id          INTEGER PRIMARY KEY,
                memory_id   TEXT NOT NULL,
                access_type TEXT NOT NULL,
                timestamp   TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_memory_access_id ON memory_access(memory_id);

            -- ═══ Session Intelligence ═══

            CREATE TABLE IF NOT EXISTS sessions (
                id               TEXT PRIMARY KEY,
                source           TEXT,
                session_date     TEXT,
                model            TEXT,
                memory_count     INTEGER DEFAULT 0,
                transcript_chars INTEGER DEFAULT 0,
                duration_seconds REAL,
                status           TEXT DEFAULT 'processed'
            );

            -- ═══ Patterns & Events ═══

            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY,
                event_type TEXT NOT NULL,
                target_id  TEXT,
                query      TEXT,
                metadata   TEXT DEFAULT '{}',
                timestamp  TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);

            CREATE TABLE IF NOT EXISTS patterns (
                id           INTEGER PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                name         TEXT NOT NULL,
                description  TEXT,
                confidence   REAL DEFAULT 0.5,
                sample_size  INTEGER DEFAULT 0,
                last_triggered TEXT,
                created      TEXT DEFAULT (datetime('now'))
            );

            -- ═══ Context ═══

            CREATE TABLE IF NOT EXISTS context_data (
                key     TEXT NOT NULL,
                value   TEXT NOT NULL,
                layer   TEXT NOT NULL DEFAULT 'session',
                updated TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (key, layer)
            );
            ",
        )?;

        Ok(())
    }

    /// Load brain metadata from the database.
    fn load_meta(conn: &Connection) -> Result<BrainMeta> {
        let json: String = conn.query_row(
            "SELECT value FROM brain_meta WHERE key = 'meta'",
            [],
            |r| r.get(0),
        )?;
        let meta: BrainMeta = serde_json::from_str(&json)?;
        Ok(meta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tmp_brain() -> (TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.r8");
        (dir, path)
    }

    #[test]
    fn create_and_open_brain() {
        let (_dir, path) = tmp_brain();

        // Create
        let brain = Brain::create(&path, Some("test-agent")).unwrap();
        assert_eq!(brain.meta().schema_version, SCHEMA_VERSION);
        assert_eq!(brain.meta().agent_name.as_deref(), Some("test-agent"));
        assert_eq!(brain.meta().embedder_model, DEFAULT_EMBEDDER);
        assert_eq!(brain.meta().embedding_dim, EMBEDDING_DIM as i32);
        drop(brain);

        // Reopen
        let brain = Brain::open(&path).unwrap();
        assert_eq!(brain.meta().agent_name.as_deref(), Some("test-agent"));
    }

    #[test]
    fn create_with_model_records_the_model_and_its_dimension() {
        let (_dir, path) = tmp_brain();

        let brain = Brain::create_with_model(&path, Some("test-agent"), "hash-384", 384).unwrap();
        assert_eq!(brain.meta().embedder_model, "hash-384");
        assert_eq!(brain.meta().embedding_dim, 384);
        assert_eq!(brain.meta().schema_version, SCHEMA_VERSION);
        drop(brain);

        // It is in the file, not only in the handle.
        let brain = Brain::open(&path).unwrap();
        assert_eq!(brain.meta().embedder_model, "hash-384");
        assert_eq!(brain.meta().embedding_dim, 384);
        assert!(Brain::create_with_model(&path, None, "hash-384", 384).is_err());
    }

    #[test]
    fn create_fails_if_exists() {
        let (_dir, path) = tmp_brain();
        Brain::create(&path, None).unwrap();
        assert!(Brain::create(&path, None).is_err());
    }

    #[test]
    fn open_fails_if_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.r8");
        assert!(Brain::open(&path).is_err());
    }

    #[test]
    fn open_or_create_creates_when_missing() {
        let (_dir, path) = tmp_brain();
        let brain = Brain::open_or_create(&path, Some("agent")).unwrap();
        assert_eq!(brain.meta().agent_name.as_deref(), Some("agent"));
    }

    #[test]
    fn open_or_create_opens_when_exists() {
        let (_dir, path) = tmp_brain();
        Brain::create(&path, Some("first")).unwrap();
        let brain = Brain::open_or_create(&path, Some("second")).unwrap();
        // Should keep original agent name
        assert_eq!(brain.meta().agent_name.as_deref(), Some("first"));
    }

    #[test]
    fn touch_updates_counts() {
        let (_dir, path) = tmp_brain();
        let mut brain = Brain::create(&path, None).unwrap();

        // Insert a memory directly
        brain.conn().execute(
            "INSERT INTO memories (id, category, topic, title, content, created)
             VALUES ('mem_00000001', 'events', 'test', 'Test Memory', 'content', datetime('now'))",
            [],
        ).unwrap();

        brain.touch().unwrap();
        assert_eq!(brain.meta().memory_count, 1);
    }

    #[test]
    fn memory_table_works() {
        let (_dir, path) = tmp_brain();
        let brain = Brain::create(&path, None).unwrap();

        brain.conn().execute(
            "INSERT INTO memories (id, category, topic, title, content, created)
             VALUES ('mem_00000001', 'events', 'test', 'Test Memory', 'content here', datetime('now'))",
            [],
        ).unwrap();

        let count: i64 = brain.conn().query_row(
            "SELECT COUNT(*) FROM memories",
            [],
            |r| r.get(0),
        ).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn schema_has_all_tables() {
        let (_dir, path) = tmp_brain();
        let brain = Brain::create(&path, None).unwrap();

        let tables: Vec<String> = {
            let mut stmt = brain.conn().prepare(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).unwrap();
            stmt.query_map([], |r| r.get(0))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect()
        };

        // Only Memkoshi + Axel tables — velocirag tables are added later
        // via Database::from_connection()
        let expected = vec![
            "brain_meta", "context_data", "events",
            "memories", "memory_access",
            "patterns", "sessions", "staged_memories",
        ];

        for table in &expected {
            assert!(
                tables.iter().any(|t| t == table),
                "Missing table: {}. Got: {:?}",
                table,
                tables
            );
        }
    }
}
