//! Project-scoped memory layer (T32–T36).
//!
//! Adds an **additive, backward-compatible** schema layer on top of the
//! `.r8` memories table:
//!
//! * `project_key`  — trusted project scope. `NULL` = legacy/unscoped rows,
//!   which are **excluded** from every scoped query (fail closed: old data
//!   never crosses project boundaries by default).
//! * `sensitivity`  — `normal` | `secret`. Secret bodies are never
//!   lexically indexed, never snippeted, and withheld on fetch.
//! * `retention`    — disclosure/retention class (see [`Retention`]).
//!   `never_persist` is rejected at this boundary and never enters `.r8`.
//! * `provenance`   — free-form origin string (who/what stored it).
//! * `expires_at` / `superseded_by` — persisted (previously in-memory only).
//! * `memory_tombstones` — exact, project-scoped deletion markers; a
//!   tombstoned id can never be re-inserted and never re-surfaces in search.
//! * `memories_fts` — FTS5 lexical index (title/abstract/content for normal
//!   rows, title only for secret rows). Works fully offline: this module
//!   never touches the embedding model or the network.
//!
//! The migration runs inside a single transaction on open; legacy `.r8`
//! files migrate in place and remain readable by older Axel versions
//! (columns are additive; `brain_meta.schema_version` is unchanged).
//!
//! Search allocations are proportional to the *requested limit* (hard-capped
//! at [`SEARCH_LIMIT_CAP`]), never to the total match count: bounding happens
//! in SQL `LIMIT`, and snippets/bodies are truncated to
//! [`SNIPPET_MAX_CHARS`] / [`BODY_MAX_CHARS`] before they are materialised.

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::error::{AxelError, Result};
use axel_memkoshi::memory::Memory;

/// Hard cap on scoped search results, regardless of the requested limit.
pub const SEARCH_LIMIT_CAP: usize = 50;
/// Maximum characters in a search snippet.
pub const SNIPPET_MAX_CHARS: usize = 240;
/// Maximum characters returned in a fetched body.
pub const BODY_MAX_CHARS: usize = 16_384;

/// Sensitivity of a stored memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sensitivity {
    /// Ordinary memory — body may be snippeted and fetched.
    Normal,
    /// Secret — body is never indexed, snippeted, or returned to a model.
    Secret,
}

impl Sensitivity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Secret => "secret",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "normal" => Some(Self::Normal),
            "secret" => Some(Self::Secret),
            _ => None,
        }
    }
}

/// Disclosure / retention class.
///
/// Enforced at two boundaries:
/// * **persistence** — [`Retention::NeverPersist`] is rejected by
///   [`store_scoped`] and never enters the `.r8` file;
/// * **model visibility** — every class except [`Retention::Standard`]
///   withholds the body from fetch results (the model-visible channel is
///   treated as potential transmission). Bodies remain locally inspectable
///   via retention inspect/export surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Retention {
    /// Persist and allow model-visible disclosure.
    Standard,
    /// Persist locally; never disclose to a (possibly remote) model.
    LocalOnly,
    /// Persist; disclose only after redaction/consent (not grantable by the
    /// model itself, so tool fetches withhold the body).
    VisibleAfterConsent,
    /// Persist; never transmit anywhere (including model context).
    PersistNeverTransmit,
    /// Never write to disk at all. Rejected by the store boundary.
    NeverPersist,
}

impl Retention {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::LocalOnly => "local_only",
            Self::VisibleAfterConsent => "visible_after_consent",
            Self::PersistNeverTransmit => "persist_never_transmit",
            Self::NeverPersist => "never_persist",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "standard" => Some(Self::Standard),
            "local_only" => Some(Self::LocalOnly),
            "visible_after_consent" => Some(Self::VisibleAfterConsent),
            "persist_never_transmit" => Some(Self::PersistNeverTransmit),
            "never_persist" => Some(Self::NeverPersist),
            _ => None,
        }
    }
    /// May the body be shown through a model-visible channel?
    pub fn model_visible(&self) -> bool {
        matches!(self, Self::Standard)
    }
}

/// Trusted scope attached to a stored memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryScope {
    /// Canonical project key derived from a trusted local source.
    pub project_key: String,
    pub sensitivity: Sensitivity,
    pub retention: Retention,
    /// Origin description, e.g. `"synaps:memory_store"`.
    pub provenance: String,
}

/// Typed links between durable memory records (continuous-memory spec §9.4).
/// IDs use the same stable storage representation as [`ScopedDescriptor::id`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordLinks {
    #[serde(default)]
    pub supersedes: Vec<String>,
    #[serde(default)]
    pub contradicts: Vec<String>,
    #[serde(default)]
    pub confirmed_by: Vec<String>,
    #[serde(default)]
    pub invalidated_at: Option<DateTime<Utc>>,
    /// Reverse edge maintained when a newer record supersedes this one.
    #[serde(default)]
    pub superseded_by: Option<String>,
}

impl RecordLinks {
    /// Every referenced record id, irrespective of relation type.
    pub fn all_ids(&self) -> impl Iterator<Item = &str> {
        self.supersedes
            .iter()
            .chain(&self.contradicts)
            .chain(&self.confirmed_by)
            .map(String::as_str)
            .chain(self.superseded_by.iter().map(String::as_str))
    }
}

/// Filters for a scoped search. All filters are ANDed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScopedQuery {
    pub project_key: String,
    /// Lexical text query (FTS5). `None` = recency listing.
    pub text: Option<String>,
    /// Every tag must be present.
    pub tags: Vec<String>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    /// Requested limit; clamped to `1..=SEARCH_LIMIT_CAP`.
    pub limit: usize,
}

impl Default for MemoryScope {
    fn default() -> Self {
        Self {
            project_key: String::new(),
            sensitivity: Sensitivity::Normal,
            retention: Retention::Standard,
            provenance: String::new(),
        }
    }
}

/// Bounded, stable-ID search descriptor. Never contains a secret body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopedDescriptor {
    pub id: String,
    pub project_key: String,
    pub title: String,
    /// Bounded body-derived snippet; `None` for secret memories.
    pub snippet: Option<String>,
    pub category: String,
    pub tags: Vec<String>,
    pub created: String,
    pub sensitivity: Sensitivity,
    pub retention: Retention,
    pub provenance: String,
    /// Persisted supersession/contradiction graph links, scrubbed of tombstones.
    pub links: RecordLinks,
}

/// Full fetch view. Body may be withheld by sensitivity/retention policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScopedMemoryView {
    pub descriptor: ScopedDescriptor,
    /// Bounded body — `None` when withheld.
    pub body: Option<String>,
    pub body_truncated: bool,
    pub body_withheld_reason: Option<String>,
    /// Signature verification result (`None` = no signer available).
    pub verified: Option<bool>,
}

// ── Migration ───────────────────────────────────────────────────────────

fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let name: String = row.get(1)?;
        if name == column {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Run the additive project-memory migration inside one transaction.
/// Idempotent; safe on legacy and current `.r8` files.
pub fn migrate(conn: &Connection) -> Result<()> {
    let mut alters: Vec<String> = Vec::new();
    for (col, ddl) in [
        (
            "project_key",
            "ALTER TABLE memories ADD COLUMN project_key TEXT",
        ),
        (
            "sensitivity",
            "ALTER TABLE memories ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'normal'",
        ),
        (
            "retention",
            "ALTER TABLE memories ADD COLUMN retention TEXT NOT NULL DEFAULT 'standard'",
        ),
        (
            "provenance",
            "ALTER TABLE memories ADD COLUMN provenance TEXT NOT NULL DEFAULT ''",
        ),
        (
            "expires_at",
            "ALTER TABLE memories ADD COLUMN expires_at TEXT",
        ),
        (
            "superseded_by",
            "ALTER TABLE memories ADD COLUMN superseded_by TEXT",
        ),
        (
            "supersedes",
            "ALTER TABLE memories ADD COLUMN supersedes TEXT NOT NULL DEFAULT '[]'",
        ),
        (
            "contradicts",
            "ALTER TABLE memories ADD COLUMN contradicts TEXT NOT NULL DEFAULT '[]'",
        ),
        (
            "confirmed_by",
            "ALTER TABLE memories ADD COLUMN confirmed_by TEXT NOT NULL DEFAULT '[]'",
        ),
        (
            "invalidated_at",
            "ALTER TABLE memories ADD COLUMN invalidated_at TEXT",
        ),
    ] {
        if !column_exists(conn, "memories", col)? {
            alters.push(ddl.to_string());
        }
    }

    conn.execute_batch("BEGIN IMMEDIATE")?;
    let body = (|| -> Result<()> {
        for ddl in &alters {
            conn.execute_batch(ddl)?;
        }
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memory_tombstones (
                 id          TEXT PRIMARY KEY,
                 project_key TEXT NOT NULL,
                 deleted_at  TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_tombstones_project
                 ON memory_tombstones(project_key);
             CREATE INDEX IF NOT EXISTS idx_memories_project
                 ON memories(project_key);
             CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                 mem_id UNINDEXED, title, abstract, content
             );",
        )?;
        Ok(())
    })();
    match body {
        Ok(()) => {
            conn.execute_batch("COMMIT")?;
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

// ── Store ───────────────────────────────────────────────────────────────

fn live_link_ids(conn: &Connection, ids: &[String]) -> Result<Vec<String>> {
    let mut live = Vec::with_capacity(ids.len());
    for id in ids {
        let tombstoned = conn
            .query_row(
                "SELECT 1 FROM memory_tombstones WHERE id = ?1",
                [id],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if !tombstoned {
            live.push(id.clone());
        }
    }
    Ok(live)
}

fn validate_scope(scope: &MemoryScope) -> Result<()> {
    if scope.project_key.trim().is_empty() {
        return Err(AxelError::Memory(
            "store_scoped: empty project_key — refusing unscoped write (fail closed)".into(),
        ));
    }
    if scope.retention == Retention::NeverPersist {
        return Err(AxelError::Memory(
            "store_scoped: retention=never_persist must not be written to the .r8".into(),
        ));
    }
    Ok(())
}

fn insert_scoped_row(
    conn: &Connection,
    memory: &Memory,
    scope: &MemoryScope,
    category_override: Option<&str>,
    links: &RecordLinks,
) -> Result<()> {
    let live_links = RecordLinks {
        supersedes: live_link_ids(conn, &links.supersedes)?,
        contradicts: live_link_ids(conn, &links.contradicts)?,
        confirmed_by: live_link_ids(conn, &links.confirmed_by)?,
        invalidated_at: links.invalidated_at,
        superseded_by: links.superseded_by.clone(),
    };
    // Tombstone check — a forgotten id must stay forgotten.
    let tombstoned: Option<String> = conn
        .query_row(
            "SELECT id FROM memory_tombstones WHERE id = ?1",
            [&memory.id],
            |r| r.get(0),
        )
        .optional()?;
    if tombstoned.is_some() {
        return Err(AxelError::Memory(format!(
            "store_scoped: id {} is tombstoned; refusing re-insert",
            memory.id
        )));
    }

    conn.execute(
        "INSERT INTO memories
            (id, category, topic, title, abstract_text, content, confidence,
             importance, source_sessions, tags, related_topics, created, updated,
             trust_level, signature, project_key, sensitivity, retention,
             provenance, expires_at, superseded_by, supersedes, contradicts,
             confirmed_by, invalidated_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20,?21,?22,?23,?24,?25)",
        params![
            memory.id,
            category_override.unwrap_or_else(|| memory.category.as_str()),
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
            scope.project_key,
            scope.sensitivity.as_str(),
            scope.retention.as_str(),
            scope.provenance,
            memory.expires_at.map(|t| t.to_rfc3339()),
            memory.superseded_by,
            serde_json::to_string(&live_links.supersedes)?,
            serde_json::to_string(&live_links.contradicts)?,
            serde_json::to_string(&live_links.confirmed_by)?,
            live_links.invalidated_at.map(|t| t.to_rfc3339()),
        ],
    )?;

    // Lexical index. Secret bodies are NEVER indexed — title only, so no
    // body-derived snippet or body-text match is possible.
    match scope.sensitivity {
        Sensitivity::Normal => {
            conn.execute(
                "INSERT INTO memories_fts (mem_id, title, abstract, content)
                 VALUES (?1, ?2, ?3, ?4)",
                params![
                    memory.id,
                    memory.title,
                    memory.abstract_text,
                    memory.content
                ],
            )?;
        }
        Sensitivity::Secret => {
            conn.execute(
                "INSERT INTO memories_fts (mem_id, title, abstract, content)
                 VALUES (?1, ?2, '', '')",
                params![memory.id, memory.title],
            )?;
        }
    }
    Ok(())
}

/// Store one signed+validated memory under `scope`, transactionally.
///
/// The caller (AxelBrain) is responsible for signing and pipeline
/// validation before this is invoked.
pub fn store_scoped(conn: &Connection, memory: &Memory, scope: &MemoryScope) -> Result<()> {
    validate_scope(scope)?;
    conn.execute_batch("BEGIN IMMEDIATE")?;
    match insert_scoped_row(conn, memory, scope, None, &RecordLinks::default()) {
        Ok(()) => {
            conn.execute_batch("COMMIT")?;
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

/// Store one memory using a typed episodic class in the existing category
/// column. `synchronous=FULL` makes the commit durable before `Ok` is
/// returned, including for connections opened from legacy `.r8` files.
pub fn store_scoped_typed(
    conn: &Connection,
    memory: &Memory,
    scope: &MemoryScope,
    record_class: &str,
) -> Result<()> {
    store_scoped_typed_with_links(conn, memory, scope, record_class, &RecordLinks::default())
}

/// Store one typed memory together with its contradiction-graph links.
pub fn store_scoped_typed_with_links(
    conn: &Connection,
    memory: &Memory,
    scope: &MemoryScope,
    record_class: &str,
    links: &RecordLinks,
) -> Result<()> {
    validate_scope(scope)?;
    if record_class.trim().is_empty() {
        return Err(AxelError::Memory(
            "store_scoped_typed: empty record class".into(),
        ));
    }
    // Enable the strongest SQLite durability before migration too: legacy
    // files may require schema writes, and those are part of the acknowledged
    // capture operation.
    conn.execute_batch("PRAGMA synchronous=FULL")?;
    migrate(conn)?;
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let result = (|| -> Result<()> {
        insert_scoped_row(conn, memory, scope, Some(record_class), links)?;
        for target in &links.supersedes {
            conn.execute(
                "UPDATE memories SET superseded_by = ?1, invalidated_at = ?2
                  WHERE id = ?3 AND project_key = ?4
                    AND id NOT IN (SELECT id FROM memory_tombstones)",
                params![
                    memory.id,
                    Utc::now().to_rfc3339(),
                    target,
                    scope.project_key
                ],
            )?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => {
            conn.execute_batch("COMMIT")?;
            Ok(())
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

/// Store many memories in one transaction (crash-safe: all or nothing).
pub fn store_scoped_batch(conn: &Connection, items: &[(Memory, MemoryScope)]) -> Result<()> {
    for (_, scope) in items {
        validate_scope(scope)?;
    }
    conn.execute_batch("BEGIN IMMEDIATE")?;
    for (memory, scope) in items {
        if let Err(e) = insert_scoped_row(conn, memory, scope, None, &RecordLinks::default()) {
            let _ = conn.execute_batch("ROLLBACK");
            return Err(e);
        }
    }
    conn.execute_batch("COMMIT")?;
    Ok(())
}

// ── Search ──────────────────────────────────────────────────────────────

/// Escape a free-text query into a safe FTS5 MATCH expression:
/// each whitespace token becomes a quoted phrase, joined with AND.
fn fts_query(text: &str) -> Option<String> {
    let tokens: Vec<String> = text
        .split_whitespace()
        .map(|t| format!("\"{}\"", t.replace('"', "")))
        .filter(|t| t.len() > 2)
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
    }
}

fn clamp_limit(limit: usize) -> usize {
    limit.clamp(1, SEARCH_LIMIT_CAP)
}

fn bounded_chars(s: &str, max: usize) -> (String, bool) {
    let mut out = String::new();
    let mut truncated = false;
    for (i, c) in s.chars().enumerate() {
        if i >= max {
            truncated = true;
            break;
        }
        out.push(c);
    }
    (out, truncated)
}

struct RawRow {
    id: String,
    project_key: String,
    title: String,
    content: String,
    category: String,
    tags: String,
    created: String,
    sensitivity: String,
    retention: String,
    provenance: String,
    supersedes: String,
    contradicts: String,
    confirmed_by: String,
    invalidated_at: Option<String>,
    superseded_by: Option<String>,
}

fn parse_links(r: &RawRow) -> RecordLinks {
    RecordLinks {
        supersedes: serde_json::from_str(&r.supersedes).unwrap_or_default(),
        contradicts: serde_json::from_str(&r.contradicts).unwrap_or_default(),
        confirmed_by: serde_json::from_str(&r.confirmed_by).unwrap_or_default(),
        invalidated_at: r
            .invalidated_at
            .as_deref()
            .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
            .map(|value| value.with_timezone(&Utc)),
        superseded_by: r.superseded_by.clone(),
    }
}

fn row_to_descriptor(r: RawRow) -> ScopedDescriptor {
    let links = parse_links(&r);
    let sensitivity = Sensitivity::parse(&r.sensitivity).unwrap_or(Sensitivity::Secret);
    let retention = Retention::parse(&r.retention).unwrap_or(Retention::LocalOnly);
    // Secret => no body-derived snippet, ever.
    let snippet = match sensitivity {
        Sensitivity::Normal => Some(bounded_chars(&r.content, SNIPPET_MAX_CHARS).0),
        Sensitivity::Secret => None,
    };
    ScopedDescriptor {
        id: r.id,
        project_key: r.project_key,
        title: r.title,
        snippet,
        category: r.category,
        tags: serde_json::from_str(&r.tags).unwrap_or_default(),
        created: r.created,
        sensitivity,
        retention,
        provenance: r.provenance,
        links,
    }
}

/// Project-scoped, bounded lexical search. Fully offline (FTS5 only).
pub fn search_scoped(conn: &Connection, query: &ScopedQuery) -> Result<Vec<ScopedDescriptor>> {
    if query.project_key.trim().is_empty() {
        return Err(AxelError::Memory(
            "search_scoped: empty project_key — refusing unscoped read (fail closed)".into(),
        ));
    }
    let limit = clamp_limit(query.limit);
    let now = Utc::now().to_rfc3339();

    let mut sql = String::from(
        "SELECT m.id, m.project_key, m.title, m.content, m.category, m.tags,
                m.created, m.sensitivity, m.retention, m.provenance,
                m.supersedes, m.contradicts, m.confirmed_by, m.invalidated_at,
                m.superseded_by
           FROM memories m ",
    );
    let mut binds: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();

    let fts = query.text.as_deref().and_then(fts_query);
    if let Some(match_expr) = &fts {
        sql.push_str("JOIN memories_fts f ON f.mem_id = m.id AND memories_fts MATCH ? ");
        binds.push(Box::new(match_expr.clone()));
    }
    sql.push_str(
        "WHERE m.project_key = ?
           AND m.id NOT IN (SELECT id FROM memory_tombstones)
           AND (m.expires_at IS NULL OR m.expires_at > ?) ",
    );
    binds.push(Box::new(query.project_key.clone()));
    binds.push(Box::new(now));
    for tag in &query.tags {
        sql.push_str("AND m.tags LIKE ? ");
        binds.push(Box::new(format!("%\"{}\"%", tag.replace(['%', '_'], ""))));
    }
    if let Some(since) = query.since {
        sql.push_str("AND m.created >= ? ");
        binds.push(Box::new(since.to_rfc3339()));
    }
    if let Some(until) = query.until {
        sql.push_str("AND m.created <= ? ");
        binds.push(Box::new(until.to_rfc3339()));
    }
    if fts.is_some() {
        sql.push_str("ORDER BY rank LIMIT ?");
    } else {
        sql.push_str("ORDER BY m.created DESC LIMIT ?");
    }
    binds.push(Box::new(limit as i64));

    let mut stmt = conn.prepare(&sql)?;
    let params_ref: Vec<&dyn rusqlite::ToSql> = binds.iter().map(|b| b.as_ref()).collect();
    let mut rows = stmt.query(params_ref.as_slice())?;

    // Allocation proportional to the clamped limit, not total matches.
    let mut out = Vec::with_capacity(limit);
    while let Some(row) = rows.next()? {
        out.push(row_to_descriptor(RawRow {
            id: row.get(0)?,
            project_key: row.get(1)?,
            title: row.get(2)?,
            content: row.get(3)?,
            category: row.get(4)?,
            tags: row.get(5)?,
            created: row.get(6)?,
            sensitivity: row.get(7)?,
            retention: row.get(8)?,
            provenance: row.get(9)?,
            supersedes: row.get(10)?,
            contradicts: row.get(11)?,
            confirmed_by: row.get(12)?,
            invalidated_at: row.get(13)?,
            superseded_by: row.get(14)?,
        }));
        if out.len() >= limit {
            break;
        }
    }
    Ok(out)
}

// ── Fetch ───────────────────────────────────────────────────────────────

/// Exact-ID, project-checked fetch. Secret/withheld classes never expose
/// the body through this (model-visible) surface.
pub fn fetch_scoped(
    conn: &Connection,
    id: &str,
    project_key: &str,
) -> Result<Option<ScopedMemoryView>> {
    if project_key.trim().is_empty() {
        return Err(AxelError::Memory(
            "fetch_scoped: empty project_key — refusing unscoped read (fail closed)".into(),
        ));
    }
    let now = Utc::now().to_rfc3339();
    let mut stmt = conn.prepare(
        "SELECT m.id, m.project_key, m.title, m.content, m.category, m.tags,
                m.created, m.sensitivity, m.retention, m.provenance,
                m.supersedes, m.contradicts, m.confirmed_by, m.invalidated_at,
                m.superseded_by
           FROM memories m
          WHERE m.id = ?1 AND m.project_key = ?2
            AND m.id NOT IN (SELECT id FROM memory_tombstones)
            AND (m.expires_at IS NULL OR m.expires_at > ?3)",
    )?;
    let raw = stmt
        .query_row(params![id, project_key, now], |row| {
            Ok(RawRow {
                id: row.get(0)?,
                project_key: row.get(1)?,
                title: row.get(2)?,
                content: row.get(3)?,
                category: row.get(4)?,
                tags: row.get(5)?,
                created: row.get(6)?,
                sensitivity: row.get(7)?,
                retention: row.get(8)?,
                provenance: row.get(9)?,
                supersedes: row.get(10)?,
                contradicts: row.get(11)?,
                confirmed_by: row.get(12)?,
                invalidated_at: row.get(13)?,
                superseded_by: row.get(14)?,
            })
        })
        .optional()?;
    let raw = match raw {
        Some(r) => r,
        None => return Ok(None),
    };
    let content = raw.content.clone();
    let descriptor = row_to_descriptor(raw);

    let (body, body_truncated, reason) = if descriptor.sensitivity == Sensitivity::Secret {
        (
            None,
            false,
            Some("sensitivity=secret: body withheld from model-visible output".to_string()),
        )
    } else if !descriptor.retention.model_visible() {
        (
            None,
            false,
            Some(format!(
                "retention={}: body withheld at model-visibility boundary",
                descriptor.retention.as_str()
            )),
        )
    } else {
        let (b, t) = bounded_chars(&content, BODY_MAX_CHARS);
        (Some(b), t, None)
    };

    Ok(Some(ScopedMemoryView {
        descriptor,
        body,
        body_truncated,
        body_withheld_reason: reason,
        // Filled in by AxelBrain::fetch_scoped (needs the brain's signer).
        verified: None,
    }))
}

// ── Forget ──────────────────────────────────────────────────────────────

/// Exact, project-scoped tombstone + delete. Returns `false` when the id
/// does not exist in that project scope (fail closed for cross-project).
pub fn forget_scoped(conn: &Connection, id: &str, project_key: &str) -> Result<bool> {
    if project_key.trim().is_empty() {
        return Err(AxelError::Memory(
            "forget_scoped: empty project_key — refusing unscoped delete (fail closed)".into(),
        ));
    }
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let result = (|| -> Result<bool> {
        let owned: Option<String> = conn
            .query_row(
                "SELECT id FROM memories WHERE id = ?1 AND project_key = ?2",
                params![id, project_key],
                |r| r.get(0),
            )
            .optional()?;
        if owned.is_none() {
            return Ok(false);
        }
        conn.execute(
            "INSERT OR REPLACE INTO memory_tombstones (id, project_key, deleted_at)
             VALUES (?1, ?2, ?3)",
            params![id, project_key, Utc::now().to_rfc3339()],
        )?;
        conn.execute(
            "DELETE FROM memories WHERE id = ?1 AND project_key = ?2",
            params![id, project_key],
        )?;
        // Scrub graph edges to the forgotten id; rebuilding the search index
        // must never make a tombstoned record reachable through a survivor.
        let mut stmt = conn.prepare(
            "SELECT id, supersedes, contradicts, confirmed_by
               FROM memories WHERE project_key = ?1",
        )?;
        let rows = stmt.query_map([project_key], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        let mut updates = Vec::new();
        for row in rows {
            let (survivor, supersedes, contradicts, confirmed_by) = row?;
            let scrub = |json: &str| -> Vec<String> {
                serde_json::from_str::<Vec<String>>(json)
                    .unwrap_or_default()
                    .into_iter()
                    .filter(|linked| linked != id)
                    .collect()
            };
            updates.push((
                survivor,
                serde_json::to_string(&scrub(&supersedes))?,
                serde_json::to_string(&scrub(&contradicts))?,
                serde_json::to_string(&scrub(&confirmed_by))?,
            ));
        }
        drop(stmt);
        for (survivor, supersedes, contradicts, confirmed_by) in updates {
            conn.execute(
                "UPDATE memories
                    SET supersedes = ?1, contradicts = ?2, confirmed_by = ?3,
                        superseded_by = CASE WHEN superseded_by = ?4 THEN NULL ELSE superseded_by END
                  WHERE id = ?5",
                params![supersedes, contradicts, confirmed_by, id, survivor],
            )?;
        }
        conn.execute("DELETE FROM memories_fts WHERE mem_id = ?1", params![id])?;
        Ok(true)
    })();
    match result {
        Ok(v) => {
            conn.execute_batch("COMMIT")?;
            Ok(v)
        }
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(e)
        }
    }
}

/// Rebuild the lexical index from live rows only. Graph links remain attached
/// to records and tombstones are excluded defensively.
pub fn rebuild_search_index(conn: &Connection) -> Result<()> {
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let result = (|| -> Result<()> {
        conn.execute("DELETE FROM memories_fts", [])?;
        conn.execute(
            "INSERT INTO memories_fts (mem_id, title, abstract, content)
             SELECT m.id, m.title,
                    CASE WHEN m.sensitivity = 'secret' THEN '' ELSE m.abstract_text END,
                    CASE WHEN m.sensitivity = 'secret' THEN '' ELSE m.content END
               FROM memories m
              WHERE m.id NOT IN (SELECT id FROM memory_tombstones)",
            [],
        )?;
        Ok(())
    })();
    match result {
        Ok(()) => {
            conn.execute_batch("COMMIT")?;
            Ok(())
        }
        Err(error) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(error)
        }
    }
}

// ── Retention inspection ────────────────────────────────────────────────

/// Per-retention-class counts for one project scope.
pub fn retention_stats(conn: &Connection, project_key: &str) -> Result<Vec<(String, u64)>> {
    let mut stmt = conn.prepare(
        "SELECT retention, COUNT(*) FROM memories
          WHERE project_key = ?1
            AND id NOT IN (SELECT id FROM memory_tombstones)
          GROUP BY retention ORDER BY retention",
    )?;
    let rows = stmt.query_map([project_key], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as u64))
    })?;
    let mut out = Vec::new();
    for r in rows {
        out.push(r?);
    }
    Ok(out)
}
