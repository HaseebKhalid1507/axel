//! SQLite-backed persistence for memories, staging, events and context.

use std::path::Path;

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};


use crate::memory::{
    Confidence, Memory, MemoryCategory, ReviewStatus, StagedMemory,
};

/// Schema version written to `schema_info` on first open.
const SCHEMA_VERSION: i64 = 1;

use crate::error::{MemkoshiError, Result};

/// Aggregate counts and bounds returned by [`MemoryStorage::stats`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Number of approved memories.
    pub total_memories: u64,
    /// Number of staged memories (any status).
    pub staged_count: u64,
    /// Number of recorded events.
    pub event_count: u64,
    /// Earliest `created` timestamp across approved memories.
    pub oldest_memory: Option<DateTime<Utc>>,
    /// Latest `created` timestamp across approved memories.
    pub newest_memory: Option<DateTime<Utc>>,
}

/// SQLite-backed memory store.
pub struct MemoryStorage {
    conn: Connection,
}

impl MemoryStorage {
    /// Open or create the database at `path`, enabling WAL mode and
    /// running schema migrations.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let mut storage = Self { conn };
        storage.migrate()?;
        Ok(storage)
    }

    /// Open an existing database WITHOUT running schema migrations.
    ///
    /// Use this when the schema is already managed by a higher-level
    /// component (e.g., Axel's Brain creates all tables). Avoids dual
    /// schema ownership where two components both try to create tables
    /// with potentially different column definitions.
    pub fn open_existing(path: &Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        Ok(Self { conn })
    }

    /// Create the schema if absent and verify the on-disk version.
    pub fn migrate(&mut self) -> Result<()> {
        let tx = self.conn.transaction()?;
        tx.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS schema_info (
                version    INTEGER PRIMARY KEY,
                created    TEXT NOT NULL,
                agent_name TEXT
            );

            CREATE TABLE IF NOT EXISTS memories (
                id              TEXT PRIMARY KEY,
                category        TEXT NOT NULL,
                topic           TEXT NOT NULL,
                title           TEXT NOT NULL,
                abstract_text   TEXT NOT NULL,
                content         TEXT NOT NULL,
                confidence      TEXT NOT NULL,
                importance      REAL NOT NULL,
                source_sessions TEXT NOT NULL,
                tags            TEXT NOT NULL,
                related_topics  TEXT NOT NULL,
                created         TEXT NOT NULL,
                updated         TEXT,
                trust_level     REAL NOT NULL,
                signature       TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memories_topic    ON memories(topic);
            CREATE INDEX IF NOT EXISTS idx_memories_created  ON memories(created);

            CREATE TABLE IF NOT EXISTS staged_memories (
                id              TEXT PRIMARY KEY,
                category        TEXT NOT NULL,
                topic           TEXT NOT NULL,
                title           TEXT NOT NULL,
                abstract_text   TEXT NOT NULL,
                content         TEXT NOT NULL,
                confidence      TEXT NOT NULL,
                importance      REAL NOT NULL,
                source_sessions TEXT NOT NULL,
                tags            TEXT NOT NULL,
                related_topics  TEXT NOT NULL,
                created         TEXT NOT NULL,
                updated         TEXT,
                trust_level     REAL NOT NULL,
                signature       TEXT,
                review_status   TEXT NOT NULL,
                reviewer_notes  TEXT,
                staged_at       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_access (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id   TEXT NOT NULL,
                access_type TEXT NOT NULL,
                timestamp   TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_access_memory_id ON memory_access(memory_id);
            CREATE INDEX IF NOT EXISTS idx_access_timestamp ON memory_access(timestamp);

            CREATE TABLE IF NOT EXISTS events (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                target_id  TEXT,
                query      TEXT,
                metadata   TEXT,
                timestamp  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_type      ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);

            CREATE TABLE IF NOT EXISTS context_data (
                key     TEXT NOT NULL,
                layer   TEXT NOT NULL,
                value   TEXT NOT NULL,
                updated TEXT NOT NULL,
                PRIMARY KEY (key, layer)
            );
            "#,
        )?;

        let current: Option<i64> = tx
            .query_row(
                "SELECT MAX(version) FROM schema_info",
                [],
                |r| r.get::<_, Option<i64>>(0),
            )
            .optional()?
            .flatten();

        match current {
            None => {
                tx.execute(
                    "INSERT INTO schema_info (version, created, agent_name) VALUES (?1, ?2, NULL)",
                    params![SCHEMA_VERSION, Utc::now().to_rfc3339()],
                )?;
            }
            Some(v) if v > SCHEMA_VERSION => {
                return Err(MemkoshiError::SchemaTooNew {
                    found: v,
                    supported: SCHEMA_VERSION,
                });
            }
            Some(_) => {} // current/older — nothing to do for v1
        }

        tx.commit()?;
        Ok(())
    }

    // ---------------------------------------------------------------- memories

    /// Insert (or replace) an approved memory.
    pub fn store_memory(&self, memory: &Memory) -> Result<()> {
        self.conn.execute(
            r#"INSERT OR REPLACE INTO memories
                (id, category, topic, title, abstract_text, content, confidence,
                 importance, source_sessions, tags, related_topics, created,
                 updated, trust_level, signature)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                        ?13, ?14, ?15)"#,
            params![
                memory.id,
                memory.category.as_str(),
                memory.topic,
                memory.title,
                memory.abstract_text,
                memory.content,
                memory.confidence.as_str(),
                memory.importance,
                serde_json::to_string(&memory.source_sessions)?,
                serde_json::to_string(&memory.tags)?,
                serde_json::to_string(&memory.related_topics)?,
                memory.created.to_rfc3339(),
                memory.updated.map(|t| t.to_rfc3339()),
                memory.trust_level,
                memory.signature,
            ],
        )?;
        Ok(())
    }

    /// Fetch a memory by id.
    pub fn get_memory(&self, id: &str) -> Result<Option<Memory>> {
        let mut stmt = self.conn.prepare(
            r#"SELECT id, category, topic, title, abstract_text, content, confidence,
                       importance, source_sessions, tags, related_topics, created,
                       updated, trust_level, signature
                  FROM memories WHERE id = ?1"#,
        )?;
        let row = stmt
            .query_row(params![id], row_to_memory)
            .optional()?;
        row.transpose()
    }

    /// List the most recent `limit` memories by `created` descending.
    pub fn list_memories(&self, limit: usize) -> Result<Vec<Memory>> {
        let mut stmt = self.conn.prepare(
            r#"SELECT id, category, topic, title, abstract_text, content, confidence,
                       importance, source_sessions, tags, related_topics, created,
                       updated, trust_level, signature
                  FROM memories
                 ORDER BY created DESC
                 LIMIT ?1"#,
        )?;
        let rows = stmt.query_map(params![limit as i64], row_to_memory)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r??);
        }
        Ok(out)
    }

    /// Delete a memory by id. Returns `true` if a row was removed.
    pub fn delete_memory(&self, id: &str) -> Result<bool> {
        let n = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", params![id])?;
        Ok(n > 0)
    }

    /// Remove all expired memories and return the count of deleted memories.
    /// A memory is expired if it has an `expires_at` timestamp that is in the past.
    /// Since the database schema doesn't yet support the `expires_at` field,
    /// this implementation checks expiry using the in-memory representation.
    pub fn prune_expired(&self) -> Result<u64> {
        // TODO: This is inefficient as it loads all memories into memory.
        // Once the database schema is updated to include expires_at, 
        // this should be done with a SQL DELETE WHERE expires_at < NOW().
        let all_memories = self.list_memories(10000)?; // Large limit to get all memories
        let mut deleted_count = 0u64;
        
        for memory in all_memories {
            if memory.is_expired() {
                if self.delete_memory(&memory.id)? {
                    deleted_count += 1;
                }
            }
        }
        
        Ok(deleted_count)
    }

    // ------------------------------------------------------------------ staging

    /// Stage `memory` as `Pending` and return the staging envelope.
    pub fn stage_memory(&self, memory: &Memory) -> Result<StagedMemory> {
        let staged = StagedMemory::pending(memory.clone());
        self.conn.execute(
            r#"INSERT OR REPLACE INTO staged_memories
                (id, category, topic, title, abstract_text, content, confidence,
                 importance, source_sessions, tags, related_topics, created,
                 updated, trust_level, signature, review_status, reviewer_notes,
                 staged_at)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                        ?13, ?14, ?15, ?16, ?17, ?18)"#,
            params![
                staged.memory.id,
                staged.memory.category.as_str(),
                staged.memory.topic,
                staged.memory.title,
                staged.memory.abstract_text,
                staged.memory.content,
                staged.memory.confidence.as_str(),
                staged.memory.importance,
                serde_json::to_string(&staged.memory.source_sessions)?,
                serde_json::to_string(&staged.memory.tags)?,
                serde_json::to_string(&staged.memory.related_topics)?,
                staged.memory.created.to_rfc3339(),
                staged.memory.updated.map(|t| t.to_rfc3339()),
                staged.memory.trust_level,
                staged.memory.signature,
                staged.review_status.as_str(),
                staged.reviewer_notes,
                staged.staged_at.to_rfc3339(),
            ],
        )?;
        Ok(staged)
    }

    /// List every staged memory ordered by `staged_at` descending.
    pub fn list_staged(&self) -> Result<Vec<StagedMemory>> {
        let mut stmt = self.conn.prepare(
            r#"SELECT id, category, topic, title, abstract_text, content, confidence,
                       importance, source_sessions, tags, related_topics, created,
                       updated, trust_level, signature, review_status,
                       reviewer_notes, staged_at
                  FROM staged_memories
                 ORDER BY staged_at DESC"#,
        )?;
        let rows = stmt.query_map([], row_to_staged)?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r??);
        }
        Ok(out)
    }

    /// Promote a staged memory to the approved store and remove it from
    /// staging. Returns the promoted [`Memory`].
    pub fn approve(&mut self, id: &str) -> Result<Memory> {
        let tx = self.conn.transaction()?;
        let staged: Option<std::result::Result<StagedMemory, MemkoshiError>> = {
            let mut stmt = tx.prepare(
                r#"SELECT id, category, topic, title, abstract_text, content, confidence,
                           importance, source_sessions, tags, related_topics, created,
                           updated, trust_level, signature, review_status,
                           reviewer_notes, staged_at
                      FROM staged_memories WHERE id = ?1"#,
            )?;
            stmt.query_row(params![id], row_to_staged).optional()?
        };

        let staged = match staged {
            Some(r) => r?,
            None => return Err(MemkoshiError::NotFound(id.to_string())),
        };

        let memory = staged.memory.clone();
        tx.execute(
            r#"INSERT OR REPLACE INTO memories
                (id, category, topic, title, abstract_text, content, confidence,
                 importance, source_sessions, tags, related_topics, created,
                 updated, trust_level, signature)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                        ?13, ?14, ?15)"#,
            params![
                memory.id,
                memory.category.as_str(),
                memory.topic,
                memory.title,
                memory.abstract_text,
                memory.content,
                memory.confidence.as_str(),
                memory.importance,
                serde_json::to_string(&memory.source_sessions)?,
                serde_json::to_string(&memory.tags)?,
                serde_json::to_string(&memory.related_topics)?,
                memory.created.to_rfc3339(),
                memory.updated.map(|t| t.to_rfc3339()),
                memory.trust_level,
                memory.signature,
            ],
        )?;
        tx.execute(
            "DELETE FROM staged_memories WHERE id = ?1",
            params![id],
        )?;
        tx.commit()?;
        Ok(memory)
    }

    /// Mark a staged memory as `Rejected` with a reviewer-supplied reason.
    pub fn reject(&self, id: &str, reason: &str) -> Result<()> {
        let n = self.conn.execute(
            r#"UPDATE staged_memories
                  SET review_status = ?1,
                      reviewer_notes = ?2
                WHERE id = ?3"#,
            params![ReviewStatus::Rejected.as_str(), reason, id],
        )?;
        if n == 0 {
            return Err(MemkoshiError::NotFound(id.to_string()));
        }
        Ok(())
    }

    // --------------------------------------------------------------- access/events

    /// Append a row to `memory_access`.
    pub fn record_access(&self, memory_id: &str, access_type: &str) -> Result<()> {
        self.conn.execute(
            r#"INSERT INTO memory_access (memory_id, access_type, timestamp)
                VALUES (?1, ?2, ?3)"#,
            params![memory_id, access_type, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// Append a row to `events`.
    pub fn record_event(
        &self,
        event_type: &str,
        target_id: Option<&str>,
        query: Option<&str>,
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        let meta_str = match metadata {
            Some(v) => Some(serde_json::to_string(&v)?),
            None => None,
        };
        self.conn.execute(
            r#"INSERT INTO events (event_type, target_id, query, metadata, timestamp)
                VALUES (?1, ?2, ?3, ?4, ?5)"#,
            params![event_type, target_id, query, meta_str, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    // ------------------------------------------------------------------- context

    /// Upsert a context value at the given layer (`boot`/`session`/`archive`).
    pub fn set_context(&self, key: &str, value: &str, layer: &str) -> Result<()> {
        self.conn.execute(
            r#"INSERT INTO context_data (key, layer, value, updated)
                VALUES (?1, ?2, ?3, ?4)
                ON CONFLICT(key, layer) DO UPDATE
                  SET value   = excluded.value,
                      updated = excluded.updated"#,
            params![key, layer, value, Utc::now().to_rfc3339()],
        )?;
        Ok(())
    }

    /// Look up a context value across layers, preferring `session` then
    /// `boot` then `archive`.
    pub fn get_context(&self, key: &str) -> Result<Option<String>> {
        let v: Option<String> = self
            .conn
            .query_row(
                r#"SELECT value FROM context_data
                    WHERE key = ?1
                    ORDER BY CASE layer
                              WHEN 'session' THEN 0
                              WHEN 'boot'    THEN 1
                              WHEN 'archive' THEN 2
                              ELSE 3
                             END
                    LIMIT 1"#,
                params![key],
                |r| r.get(0),
            )
            .optional()?;
        Ok(v)
    }

    // -------------------------------------------------------------------- stats

    /// Aggregate counts and timestamp bounds.
    pub fn stats(&self) -> Result<MemoryStats> {
        let total: u64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?;
        let staged: u64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM staged_memories", [], |r| r.get(0))?;
        let events: u64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM events", [], |r| r.get(0))?;

        let oldest: Option<String> = self
            .conn
            .query_row("SELECT MIN(created) FROM memories", [], |r| r.get(0))
            .optional()?
            .flatten();
        let newest: Option<String> = self
            .conn
            .query_row("SELECT MAX(created) FROM memories", [], |r| r.get(0))
            .optional()?
            .flatten();

        Ok(MemoryStats {
            total_memories: total,
            staged_count: staged,
            event_count: events,
            oldest_memory: oldest.and_then(parse_ts),
            newest_memory: newest.and_then(parse_ts),
        })
    }

    /// Borrow the underlying connection (used by sibling modules).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

// --------------------------------------------------------------------- helpers

fn parse_ts(s: String) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(&s)
        .ok()
        .map(|t| t.with_timezone(&Utc))
}

fn parse_ts_str(s: &str) -> std::result::Result<DateTime<Utc>, MemkoshiError> {
    DateTime::parse_from_rfc3339(s)
        .map(|t| t.with_timezone(&Utc))
        .map_err(|_| MemkoshiError::InvalidEnum {
            field: "timestamp",
            value: s.to_string(),
        })
}

fn row_to_memory(row: &Row<'_>) -> rusqlite::Result<std::result::Result<Memory, MemkoshiError>> {
    let id: String = row.get(0)?;
    let category: String = row.get(1)?;
    let topic: String = row.get(2)?;
    let title: String = row.get(3)?;
    let abstract_text: String = row.get(4)?;
    let content: String = row.get(5)?;
    let confidence: String = row.get(6)?;
    let importance: f64 = row.get(7)?;
    let source_sessions: String = row.get(8)?;
    let tags: String = row.get(9)?;
    let related_topics: String = row.get(10)?;
    let created: String = row.get(11)?;
    let updated: Option<String> = row.get(12)?;
    let trust_level: f64 = row.get(13)?;
    let signature: Option<String> = row.get(14)?;

    Ok((|| -> std::result::Result<Memory, MemkoshiError> {
        Ok(Memory {
            id,
            category: MemoryCategory::parse(&category).ok_or_else(|| {
                MemkoshiError::InvalidEnum {
                    field: "category",
                    value: category.clone(),
                }
            })?,
            topic,
            title,
            abstract_text,
            content,
            confidence: Confidence::parse(&confidence).ok_or_else(|| {
                MemkoshiError::InvalidEnum {
                    field: "confidence",
                    value: confidence.clone(),
                }
            })?,
            importance,
            source_sessions: serde_json::from_str(&source_sessions)?,
            tags: serde_json::from_str(&tags)?,
            related_topics: serde_json::from_str(&related_topics)?,
            created: parse_ts_str(&created)?,
            updated: match updated {
                Some(u) => Some(parse_ts_str(&u)?),
                None => None,
            },
            trust_level,
            signature,
            superseded_by: None, // TODO: Handle this field properly in database schema
            expires_at: None, // TODO: Handle this field properly in database schema
        })
    })())
}

fn row_to_staged(
    row: &Row<'_>,
) -> rusqlite::Result<std::result::Result<StagedMemory, MemkoshiError>> {
    let mem_result = row_to_memory(row)?;
    let review_status: String = row.get(15)?;
    let reviewer_notes: Option<String> = row.get(16)?;
    let staged_at: String = row.get(17)?;

    Ok((|| -> std::result::Result<StagedMemory, MemkoshiError> {
        let memory = mem_result?;
        Ok(StagedMemory {
            memory,
            review_status: ReviewStatus::parse(&review_status).ok_or_else(|| {
                MemkoshiError::InvalidEnum {
                    field: "review_status",
                    value: review_status.clone(),
                }
            })?,
            reviewer_notes,
            staged_at: parse_ts_str(&staged_at)?,
        })
    })())
}
