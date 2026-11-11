//! The one row writer, `purge_many`, hide and restore (with twins), supersede, the daily
//! cap, pauses.
//!
//! Every function here that writes runs inside the caller's transaction
//! (`brain_v2::begin_write` or `begin_write_front` … `COMMIT`) and refuses a connection
//! that has none open. Each works inside a savepoint, so a failed call leaves none of its
//! own writes behind. **After an error from `store` the caller rolls back; it never
//! commits.** The error may mean that SQLite has already rolled the whole transaction
//! back (a full disk, an I/O error): whatever the caller wrote before the call is then
//! gone too, and statements it ran afterwards would commit one by one.
//!
//! Hide, restore and supersede never change saved text. `purge_many` is the one way a
//! row leaves.

use std::collections::{HashMap, HashSet};

use axel_memkoshi::memory::{Memory, MemoryCategory};
use axel_memkoshi::security::MemorySigner;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};

use crate::error::{AxelError, Result};
use crate::project_memory::{self, MemoryScope, RecordLinks, Retention, Sensitivity};

/// The project key of global rows, and of the tombstone of an old row that has none.
pub const GLOBAL_KEY: &str = "p0000000000000000";

/// A chat's model can hide or supersede at most this many saved memories in 24 hours.
pub const MODEL_DAILY_CAP: usize = 10;
pub const CAP_WINDOW_MS: i64 = 24 * 60 * 60 * 1000;
/// The refusal of a `forget` with `id`, or of a `remember` with `supersedes`, past the cap.
pub const CAP_REFUSAL: &str = "limit reached: this chat has hidden or superseded 10 memories in 24 hours. Ask the user to run \"axel-v2 forget <id>\".";

const ORIGIN_KINDS: [&str; 6] = ["archive", "note_file", "capture", "tool", "cli", "legacy"];
const TIME_SOURCES: [&str; 4] = ["exact", "archive", "observed", "mtime"];
const ABSTRACT_WORDS: usize = 40;

pub struct NewRow<'a> {
    pub id: String,
    pub class: &'a str,
    pub project_key: &'a str,
    pub title: String,
    pub abstract_text: String,
    pub content: String,
    pub tags: Vec<String>,
    pub created_ms: i64,
    pub time_source: &'a str,
    pub sensitivity: Sensitivity,
    pub provenance: String,
    pub expires_ms: Option<i64>,
    pub importance: f64,
    pub origin_kind: &'a str,
    pub origin_ref: &'a str,
    pub unit_pos: Option<String>,
    pub session_id: Option<&'a str>,
    pub logical: Option<&'a str>,
    pub links: RecordLinks,
    pub turn_key: Option<String>,
    /// `(hidden_ms, hidden_by)`: a twin is born hidden.
    pub hidden: Option<(i64, String)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Inserted {
    New { vid: i64 },
    Exists,
    Tombstoned,
}

fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

/// Runs `work` inside a savepoint of the caller's transaction: all of it or none. On an
/// error the caller gets the first error, and rolls back.
fn atomic<T>(conn: &Connection, work: impl FnOnce() -> Result<T>) -> Result<T> {
    if conn.is_autocommit() {
        return Err(AxelError::Other(
            "store: no transaction is open; a write starts with brain_v2::begin_write".into(),
        ));
    }
    conn.execute_batch("SAVEPOINT axel_store")?;
    match work() {
        Ok(v) => {
            conn.execute_batch("RELEASE axel_store")?;
            Ok(v)
        }
        Err(e) => {
            // RULING: wave3-c11 ruling 2: when the savepoint cannot be taken back and a
            // transaction is still open, a half-done call must never be committed (in
            // `purge_many` that would be the `axel_purge` row): the whole transaction goes.
            // When SQLite rolled it back by itself there is nothing left to take back.
            let back = conn.execute_batch("ROLLBACK TO axel_store; RELEASE axel_store");
            if back.is_err() && !conn.is_autocommit() {
                let _ = conn.execute_batch("ROLLBACK");
            }
            Err(e)
        }
    }
}

fn exists(conn: &Connection, sql: &str, id: &str) -> Result<bool> {
    Ok(conn.query_row(sql, [id], |_| Ok(())).optional()?.is_some())
}

fn time_of(ms: i64, what: &str) -> Result<DateTime<Utc>> {
    DateTime::<Utc>::from_timestamp_millis(ms)
        .ok_or_else(|| AxelError::Memory(format!("insert_row: {what} {ms} is not a time")))
}

// ── The row writer ──────────────────────────────────────────────────────────

/// The one row writer: the `memories` row, its word row and its `memory_ext` row, or none
/// of them. Title, abstract and content are redacted here, before the hash, the word index
/// and the signature. It opens no transaction: the caller holds one.
pub fn insert_row(conn: &Connection, signer: Option<&MemorySigner>, row: &NewRow) -> Result<Inserted> {
    if !ORIGIN_KINDS.contains(&row.origin_kind) {
        return Err(AxelError::Memory(format!("insert_row: origin_kind {:?}", row.origin_kind)));
    }
    if !TIME_SOURCES.contains(&row.time_source) {
        return Err(AxelError::Memory(format!("insert_row: time_source {:?}", row.time_source)));
    }
    atomic(conn, || {
        if exists(conn, "SELECT 1 FROM memory_tombstones WHERE id = ?1", &row.id)? {
            return Ok(Inserted::Tombstoned);
        }
        if exists(conn, "SELECT 1 FROM memories WHERE id = ?1", &row.id)? {
            return Ok(Inserted::Exists);
        }

        let title = crate::screen::redact(&row.title).text.into_owned();
        let content = crate::screen::redact(&row.content).text.into_owned();
        // SPEC: 1747 says "`abstract_text` = the first 40 words" and `NewRow` has the field:
        // a caller's abstract is kept (redacted); an empty one is the first 40 words of the
        // content, except for a secret row, whose body is copied nowhere.
        let abstract_text = if !row.abstract_text.trim().is_empty() {
            crate::screen::redact(&row.abstract_text).text.into_owned()
        } else if row.sensitivity == Sensitivity::Secret {
            String::new()
        } else {
            content.split_whitespace().take(ABSTRACT_WORDS).collect::<Vec<_>>().join(" ")
        };

        // The signing rule (section 1): the enum for one of the five old categories,
        // `Events` otherwise; the class goes into the column.
        let category = MemoryCategory::parse(row.class).unwrap_or(MemoryCategory::Events);
        // SPEC: 1729-1736 gives `NewRow` no topic; `memories.topic` is the class.
        let mut memory = Memory::new(category, row.class, title, content);
        memory.id = row.id.clone();
        memory.abstract_text = abstract_text;
        memory.importance = row.importance;
        memory.tags = row.tags.clone();
        memory.source_sessions = row.session_id.map(|s| vec![s.to_string()]).unwrap_or_default();
        memory.created = time_of(row.created_ms, "created_ms")?;
        memory.expires_at = match row.expires_ms {
            Some(ms) => Some(time_of(ms, "expires_ms")?),
            None => None,
        };
        memory.signature = signer.map(|s| s.sign(&memory));

        let scope = MemoryScope {
            project_key: row.project_key.to_string(),
            sensitivity: row.sensitivity,
            retention: Retention::Standard,
            provenance: row.provenance.clone(),
        };
        project_memory::insert_scoped_typed(conn, &memory, &scope, row.class, &row.links)?;

        let content_hash = hex::encode(Sha256::digest(crate::archive::normalise(&memory.content).as_bytes()));
        let (hidden_ms, hidden_by) = match &row.hidden {
            Some((ms, by)) => (Some(*ms), Some(by.as_str())),
            None => (None, None),
        };
        conn.execute(
            "INSERT INTO memory_ext(mem_id, origin_kind, origin_ref, unit_pos, session_id, logical,
                                    content_hash, turn_key, created_ms, time_source, hidden_ms, hidden_by,
                                    embed_model, embedding)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, NULL, NULL)",
            params![
                row.id,
                row.origin_kind,
                row.origin_ref,
                row.unit_pos,
                row.session_id,
                row.logical,
                content_hash,
                row.turn_key,
                row.created_ms,
                row.time_source,
                hidden_ms,
                hidden_by,
            ],
        )?;
        Ok(Inserted::New { vid: conn.last_insert_rowid() })
    })
}

// ── Purge ───────────────────────────────────────────────────────────────────

/// The ids named as `"m:<id>"` in a recall-log `detail`.
fn names_one_of(detail: &str, set: &HashSet<&str>) -> bool {
    const OPEN: &str = "\"m:";
    let mut rest = detail;
    while let Some(at) = rest.find(OPEN) {
        rest = &rest[at + OPEN.len()..];
        if let Some(end) = rest.find('"') {
            if set.contains(&rest[..end]) {
                return true;
            }
        }
    }
    false
}

/// What replaces a row once its replacement `purged` is gone: the walk follows
/// `superseded_by` through the rows of the purge set (`next` holds theirs) to the first
/// id outside it. It stops after as many steps as the set has ids, so two rows that name
/// each other end it. `None`: nothing replaces the row any more.
fn replacement_after(
    purged: &str,
    set: &HashSet<&str>,
    next: &HashMap<String, Option<String>>,
) -> Option<String> {
    let mut at = purged;
    for _ in 0..set.len() {
        let after = next.get(at)?.as_deref()?;
        if !set.contains(after) {
            return Some(after.to_string());
        }
        at = after;
    }
    None
}

fn without(json: &str, set: &HashSet<&str>) -> (Vec<String>, bool) {
    let all: Vec<String> = serde_json::from_str(json).unwrap_or_default();
    let before = all.len();
    let kept: Vec<String> = all.into_iter().filter(|id| !set.contains(id.as_str())).collect();
    let changed = kept.len() != before;
    (kept, changed)
}

/// Inside the caller's transaction; the one way rows leave. Returns the number of
/// `memories` rows deleted. `forget_log`: the rows are forgotten (true: their recall-log
/// rows go) or only replaced (false). The caller sends `VecDelta::Remove` after the commit.
pub fn purge_many(conn: &Connection, ids: &[String], tombstone: bool, forget_log: bool) -> Result<usize> {
    if ids.is_empty() {
        return Ok(0);
    }
    // RULING: wave3-c11 ruling 3: the pragma is set per connection, and without it the
    // freed pages keep the text. Read once, before anything is written. The ruling refuses
    // 0; `FAST` (2) is refused too, because it leaves the text in pages on the free list.
    let secure_delete: i64 = conn.query_row("PRAGMA secure_delete", [], |r| r.get(0))?;
    if secure_delete != 1 {
        return Err(AxelError::Other(format!(
            "purge: this connection has PRAGMA secure_delete = {secure_delete}; a purge needs \
             secure_delete = ON (a connection of brain_v2::connect has it)"
        )));
    }
    atomic(conn, || {
        // 0. The delete guard lets `DELETE FROM memories` through only while this row exists.
        conn.execute("INSERT INTO axel_purge(one) VALUES (1)", [])?;
        conn.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS purge_ids(id TEXT PRIMARY KEY);
             DELETE FROM temp.purge_ids;",
        )?;
        {
            let mut insert = conn.prepare("INSERT OR IGNORE INTO temp.purge_ids(id) VALUES (?1)")?;
            for id in ids {
                insert.execute([id])?;
            }
        }
        let set: HashSet<&str> = ids.iter().map(String::as_str).collect();

        // 1. The recall log, while `memory_ext` still has the rows.
        if forget_log {
            conn.execute(
                "DELETE FROM recall_log
                  WHERE (logical, query_hash) IN (SELECT logical, turn_key FROM memory_ext
                                                   WHERE mem_id IN (SELECT id FROM temp.purge_ids)
                                                     AND turn_key IS NOT NULL)",
                [],
            )?;
            let named: Vec<i64> = {
                let mut stmt = conn.prepare("SELECT id, detail FROM recall_log")?;
                let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
                let mut out = Vec::new();
                for row in rows {
                    let (log_id, detail) = row?;
                    if names_one_of(&detail, &set) {
                        out.push(log_id);
                    }
                }
                out
            };
            let mut delete = conn.prepare("DELETE FROM recall_log WHERE id = ?1")?;
            for log_id in named {
                delete.execute([log_id])?;
            }
        }

        // 2. The link scrub: one read, and only the rows that name an id of the set are
        // rewritten.
        // RULING: wave3-c11 ruling 5: the purge of a replacement restores the row it
        // replaced. `after`: for a row that stays and whose `superseded_by` is purged, the
        // row that replaces it afterwards, if any.
        let next: HashMap<String, Option<String>> = {
            let mut stmt =
                conn.prepare("SELECT id, superseded_by FROM memories WHERE id IN (SELECT id FROM temp.purge_ids)")?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        let mut after: Vec<(String, Option<String>)> = Vec::new();
        let linked: Vec<(String, String, String, String)> = {
            let mut stmt = conn.prepare(
                "SELECT id, superseded_by, supersedes, contradicts, confirmed_by FROM memories
                  WHERE superseded_by IN (SELECT id FROM temp.purge_ids) OR supersedes <> '[]'
                     OR contradicts <> '[]' OR confirmed_by <> '[]'",
            )?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?,
                    r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    r.get::<_, Option<String>>(4)?.unwrap_or_default(),
                ))
            })?;
            let mut out = Vec::new();
            for row in rows {
                let (id, superseded_by, supersedes, contradicts, confirmed_by) = row?;
                let (supersedes, a) = without(&supersedes, &set);
                let (contradicts, b) = without(&contradicts, &set);
                let (confirmed_by, c) = without(&confirmed_by, &set);
                let purged = superseded_by.filter(|s| set.contains(s.as_str()));
                if purged.is_some() || a || b || c {
                    out.push((
                        id.clone(),
                        serde_json::to_string(&supersedes)?,
                        serde_json::to_string(&contradicts)?,
                        serde_json::to_string(&confirmed_by)?,
                    ));
                }
                if let Some(purged) = purged {
                    if !set.contains(id.as_str()) {
                        let replacement = replacement_after(&purged, &set, &next).filter(|r| *r != id);
                        after.push((id, replacement));
                    }
                }
            }
            out
        };
        {
            let mut update =
                conn.prepare("UPDATE memories SET supersedes = ?1, contradicts = ?2, confirmed_by = ?3 WHERE id = ?4")?;
            for (id, supersedes, contradicts, confirmed_by) in &linked {
                update.execute(params![supersedes, contradicts, confirmed_by, id])?;
            }
        }
        for (id, replacement) in &after {
            // A replacement that is not a row replaces nothing.
            let replacement = match replacement {
                Some(r) if exists(conn, "SELECT 1 FROM memories WHERE id = ?1", r)? => Some(r),
                _ => None,
            };
            match replacement {
                // A -> B -> C, B purged: A is replaced by C and stays out.
                Some(r) => {
                    conn.execute("UPDATE memories SET superseded_by = ?1 WHERE id = ?2", params![r, id])?;
                }
                // The link is gone and A is as it was before it: the two columns are
                // cleared together, and the shadow mark unless A is also hidden.
                None => {
                    conn.execute("UPDATE memories SET superseded_by = NULL, invalidated_at = NULL WHERE id = ?1", [id])?;
                    clear_shadow(conn, id)?;
                }
            }
        }

        // 3 and 4. The word rows, the tombstones when asked for, the rows; the trigger
        // takes the `memory_ext` rows and with them the vectors on disk.
        let gone = project_memory::delete_rows(conn, ids, tombstone.then_some(GLOBAL_KEY))?;

        // 5. The shadow documents, only for ids that have one.
        conn.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS purge_docs(doc_id TEXT PRIMARY KEY);
             DELETE FROM temp.purge_docs;",
        )?;
        let shadows = conn.execute(
            "INSERT OR IGNORE INTO temp.purge_docs(doc_id)
             SELECT doc_id FROM documents
              WHERE doc_id IN (SELECT id FROM temp.purge_ids)
                AND CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.type') END = 'memory'",
            [],
        )?;
        if shadows > 0 {
            conn.execute("DELETE FROM chunks_fts WHERE doc_id IN (SELECT doc_id FROM temp.purge_docs)", [])?;
            conn.execute("DELETE FROM documents WHERE doc_id IN (SELECT doc_id FROM temp.purge_docs)", [])?;
        }

        // 6. Graph nodes: `nodes.content` can hold text.
        conn.execute(
            "DELETE FROM edges WHERE source_id IN (SELECT id FROM temp.purge_ids)
                                  OR target_id IN (SELECT id FROM temp.purge_ids)",
            [],
        )?;
        conn.execute(
            "DELETE FROM nodes WHERE id IN (SELECT id FROM temp.purge_ids)
                                  OR source_file IN (SELECT id FROM temp.purge_ids)",
            [],
        )?;

        // 7. The guard closes again: the row is never committed.
        conn.execute("DELETE FROM axel_purge", [])?;
        conn.execute_batch("DELETE FROM temp.purge_ids; DELETE FROM temp.purge_docs;")?;
        Ok(gone)
    })
}

/// `purge_many` of one id.
pub fn purge(conn: &Connection, id: &str, tombstone: bool, forget_log: bool) -> Result<usize> {
    purge_many(conn, &[id.to_string()], tombstone, forget_log)
}

// ── Hide, supersede, restore ────────────────────────────────────────────────

const SHADOW_IS_MEMORY: &str =
    "CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.type') END = 'memory'";

/// The mark the command line's `search` reads on the shadow document of a hidden or
/// superseded memory (R-8).
fn mark_shadow(conn: &Connection, id: &str) -> Result<()> {
    conn.execute(
        &format!("UPDATE documents SET hidden = 'memory_hidden' WHERE doc_id = ?1 AND {SHADOW_IS_MEMORY}"),
        [id],
    )?;
    Ok(())
}

/// Cleared only when the memory is neither hidden nor superseded any more.
fn clear_shadow(conn: &Connection, id: &str) -> Result<()> {
    conn.execute(
        &format!(
            "UPDATE documents SET hidden = NULL
              WHERE doc_id = ?1 AND hidden = 'memory_hidden' AND {SHADOW_IS_MEMORY}
                AND NOT EXISTS (SELECT 1 FROM memory_ext WHERE mem_id = ?1 AND hidden_ms IS NOT NULL)
                AND NOT EXISTS (SELECT 1 FROM memories WHERE id = ?1 AND superseded_by IS NOT NULL)"
        ),
        [id],
    )?;
    Ok(())
}

/// Hides one row by id: nothing is deleted and no text changes. For a capture row, the
/// rows of the chat's newest archive with the same `turn_key` are hidden with it as
/// `twin:<id>`. Returns the number of rows hidden (0: no such row, or hidden already).
// SPEC: 3202 gives the statement without `AND hidden_ms IS NULL`. A second hide of a hidden
// row would overwrite who hid it and when (a `twin:` mark among them), and restore could
// no longer bring back what the first hide hid: it changes nothing here.
// RULING: wave3-c11 ruling 1: the one exception is a `pause` mark. A hide by anyone but a
// pause takes it over (`hidden_by` and `hidden_ms` become the new hide's), so `resume`,
// which restores only rows still marked `pause`, leaves the row hidden.
pub fn hide(conn: &Connection, id: &str, hidden_by: &str) -> Result<usize> {
    if hidden_by.trim().is_empty() {
        return Err(AxelError::Memory("hide: empty hidden_by".into()));
    }
    atomic(conn, || {
        let now = now_ms();
        let mut hidden = conn.execute(
            "UPDATE memory_ext SET hidden_ms = ?1, hidden_by = ?2
              WHERE mem_id = ?3 AND (hidden_ms IS NULL OR (hidden_by = 'pause' AND ?2 <> 'pause'))",
            params![now, hidden_by, id],
        )?;
        if hidden == 0 {
            return Ok(0);
        }
        mark_shadow(conn, id)?;

        let (origin, turn_key, logical, session_id, scope_dir): (
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
        ) = conn.query_row(
            "SELECT e.origin_kind, e.turn_key, e.logical, e.session_id, m.project_key
               FROM memory_ext e JOIN memories m ON m.id = e.mem_id WHERE e.mem_id = ?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )?;
        let turn_key = turn_key.filter(|k| !k.is_empty());
        if let (true, Some(turn_key), Some(scope_dir)) = (origin == "capture", turn_key, scope_dir) {
            // The chat's newest archive in this directory, in the order of `chat_files`
            // (section 2.3); older archives are not searched.
            let newest: Option<String> = conn
                .query_row(
                    "SELECT archive_id FROM archive_index
                      WHERE scope_dir = ?1 AND (logical = ?2 OR session_id = ?3)
                        AND COALESCE(time_source, '') <> 'mtime'
                      ORDER BY cover_ms DESC, first_seen_ms DESC, rel_name DESC LIMIT 1",
                    params![scope_dir, logical, session_id],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(archive_id) = newest {
                hidden += conn.execute(
                    "UPDATE memory_ext SET hidden_ms = ?1, hidden_by = ?2
                      WHERE origin_kind = 'archive' AND origin_ref = ?3 AND turn_key = ?4
                        AND hidden_ms IS NULL
                        AND mem_id IN (SELECT id FROM memories WHERE project_key = ?5)",
                    params![now, format!("twin:{id}"), archive_id, turn_key, scope_dir],
                )?;
            }
        }
        Ok(hidden)
    })
}

/// Marks `id` as replaced by `by`: the old row is kept, text unchanged, and no longer
/// recalled. By id, so an old memory with a NULL project can be superseded (R1). Returns
/// the number of rows marked.
// SPEC: 538 and 4508 give `store.rs` "supersede" and 3202-3205 its statement; no signature
// is written anywhere. This is the one commit 18a calls for each id of `supersedes`.
pub fn supersede(conn: &Connection, id: &str, by: &str) -> Result<usize> {
    if id == by || by.trim().is_empty() {
        return Err(AxelError::Memory("supersede: a row is replaced by another row".into()));
    }
    atomic(conn, || {
        let n = conn.execute(
            "UPDATE memories SET superseded_by = ?1, invalidated_at = ?2 WHERE id = ?3",
            params![by, Utc::now().to_rfc3339(), id],
        )?;
        if n > 0 {
            mark_shadow(conn, id)?;
        }
        Ok(n)
    })
}

/// Undoes a hide and a supersede of one id; no text changes. A hidden capture: its
/// `twin:` rows become visible and, when there was one, the capture is purged (the
/// archive's rows now stand for the turn); with none the capture itself is visible again.
/// Returns the number of rows that are back (0: nothing to undo).
pub fn restore(conn: &Connection, id: &str) -> Result<usize> {
    atomic(conn, || restore_one(conn, id))
}

fn restore_one(conn: &Connection, id: &str) -> Result<usize> {
    let row: Option<(String, bool, Option<String>)> = conn
        .query_row(
            "SELECT origin_kind, hidden_ms IS NOT NULL, hidden_by FROM memory_ext WHERE mem_id = ?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .optional()?;
    let Some((origin, hidden, hidden_by)) = row else {
        return Ok(0);
    };

    // SPEC: 3415 says what restore does for a capture and nothing for one of its twins.
    // A twin made visible alone would leave the capture hidden beside it, and a later
    // restore of the capture would show the turn twice: the turn is restored as one,
    // through its capture, while that capture is still there.
    if let (true, Some(capture)) = (hidden, hidden_by.as_deref().and_then(|by| by.strip_prefix("twin:"))) {
        let waiting = capture != id
            && exists(
                conn,
                "SELECT 1 FROM memory_ext WHERE mem_id = ?1 AND origin_kind = 'capture' AND hidden_ms IS NOT NULL",
                capture,
            )?;
        if waiting {
            return restore_one(conn, capture);
        }
    }

    let mut back = conn.execute(
        "UPDATE memories SET superseded_by = NULL, invalidated_at = NULL
          WHERE id = ?1 AND (superseded_by IS NOT NULL OR invalidated_at IS NOT NULL)",
        [id],
    )?;
    if hidden {
        let twins = if origin == "capture" {
            conn.execute(
                "UPDATE memory_ext SET hidden_ms = NULL, hidden_by = NULL WHERE hidden_by = ?1",
                [format!("twin:{id}")],
            )?
        } else {
            0
        };
        if twins > 0 {
            // Replaced, not forgotten: no tombstone, and the log rows stay.
            purge_many(conn, &[id.to_string()], false, false)?;
            back = twins;
        } else {
            conn.execute("UPDATE memory_ext SET hidden_ms = NULL, hidden_by = NULL WHERE mem_id = ?1", [id])?;
            back = back.max(1);
        }
    }
    clear_shadow(conn, id)?;
    Ok(back)
}

// ── The daily cap ───────────────────────────────────────────────────────────

/// `(hidden, superseded)`: rows this chat's model hid on or after `since_ms`, and rows
/// superseded by a row this chat saved on or after it. A pause's hides do not count.
pub fn model_changes_since(conn: &Connection, session_id: &str, since_ms: i64) -> Result<(usize, usize)> {
    let hidden: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memory_ext WHERE hidden_by = ?1 AND hidden_ms >= ?2",
        params![format!("model:{session_id}"), since_ms],
        |r| r.get(0),
    )?;
    let superseded: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memories m JOIN memory_ext e ON e.mem_id = m.superseded_by
          WHERE e.origin_kind = 'tool' AND e.origin_ref = ?1 AND e.created_ms >= ?2",
        params![session_id, since_ms],
        |r| r.get(0),
    )?;
    Ok((hidden as usize, superseded as usize))
}

/// The daily cap (R-1): may this chat's model hide or supersede `more` rows now? False
/// when that would take the sum of the last 24 hours past `MODEL_DAILY_CAP`; the call is
/// then refused whole with `CAP_REFUSAL`.
pub fn cap_allows(conn: &Connection, session_id: &str, now_ms: i64, more: usize) -> Result<bool> {
    let (hidden, superseded) = model_changes_since(conn, session_id, now_ms - CAP_WINDOW_MS)?;
    Ok(hidden + superseded + more <= MODEL_DAILY_CAP)
}

// ── Pauses ──────────────────────────────────────────────────────────────────

fn pause_kind(what: &str) -> Result<()> {
    if what == "learning" || what == "recall" {
        Ok(())
    } else {
        Err(AxelError::Memory(format!("pause: what {what:?} (learning | recall)")))
    }
}

/// The pause row of one chat; `learning` also hides the chat's live captures
/// (`hidden_by = 'pause'`). Returns the number of captures hidden.
pub fn set_pause(
    conn: &Connection,
    logical: &str,
    session_id: Option<&str>,
    what: &str,
    set_by: &str,
) -> Result<usize> {
    pause_kind(what)?;
    atomic(conn, || {
        let now = now_ms();
        // An existing pause keeps its `since_ms` and who set it.
        conn.execute(
            "INSERT INTO chat_pauses(logical, what, since_ms, set_by, session_id) VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(logical, what) DO UPDATE
                SET session_id = COALESCE(chat_pauses.session_id, excluded.session_id)",
            params![logical, what, now, set_by, session_id],
        )?;
        if what != "learning" {
            return Ok(0);
        }
        Ok(conn.execute(
            "UPDATE memory_ext SET hidden_ms = ?1, hidden_by = 'pause'
              WHERE origin_kind = 'capture' AND hidden_ms IS NULL AND (logical = ?2 OR session_id = ?3)",
            params![now, logical, session_id],
        )?)
    })
}

/// Removes the pause row; for `learning`, the rows the pause hid are visible again.
/// Returns the number of rows that are back.
// SPEC: 2851 says "clears `hidden_ms` on the rows with `hidden_by = 'pause'` of that chat".
// Each goes through the restore of one id, so a `twin:` row an archive got meanwhile comes
// back with its capture (3415); with no twin that is the one statement.
pub fn clear_pause(conn: &Connection, logical: &str, session_id: Option<&str>, what: &str) -> Result<usize> {
    pause_kind(what)?;
    atomic(conn, || {
        conn.execute(
            "DELETE FROM chat_pauses WHERE what = ?1 AND (logical = ?2 OR session_id = ?3)",
            params![what, logical, session_id],
        )?;
        if what != "learning" {
            return Ok(0);
        }
        let ids: Vec<String> = {
            let mut stmt = conn.prepare(
                "SELECT mem_id FROM memory_ext
                  WHERE hidden_by = 'pause' AND (logical = ?1 OR session_id = ?2) ORDER BY vid",
            )?;
            let rows = stmt.query_map(params![logical, session_id], |r| r.get(0))?;
            rows.collect::<rusqlite::Result<_>>()?
        };
        let mut back = 0;
        for id in &ids {
            back += restore_one(conn, id)?;
        }
        Ok(back)
    })
}
