//! The one row writer, `purge_many`, hide and restore (with twins), supersede, the daily
//! cap, pauses. Commit 11 writes this file.

use axel_memkoshi::security::MemorySigner;
use rusqlite::Connection;

use crate::error::Result;
use crate::project_memory::{RecordLinks, Sensitivity};

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

// SPEC: the stub rule (4749) says "Ok with zero work"; of the three answers at 1694 only
// `Exists` says that nothing was written.
pub fn insert_row(conn: &Connection, signer: Option<&MemorySigner>, row: &NewRow) -> Result<Inserted> {
    let _ = (conn, signer, row);
    Ok(Inserted::Exists)
}

/// Inside the caller's transaction; the one way rows leave.
pub fn purge_many(conn: &Connection, ids: &[String], tombstone: bool, forget_log: bool) -> Result<usize> {
    let _ = (conn, ids, tombstone, forget_log);
    Ok(0)
}

/// `purge_many` of one id.
pub fn purge(conn: &Connection, id: &str, tombstone: bool, forget_log: bool) -> Result<usize> {
    purge_many(conn, &[id.to_string()], tombstone, forget_log)
}

// SPEC: the five functions below are named in sections 3 to 5 with their arguments at most,
// never with types or a result (`store::hide` 3250; `store::restore(conn, id)` 3293;
// `store::set_pause(conn, logical, session_id, what, set_by)` 2776; `store::clear_pause`
// 2780; `store::model_changes_since(conn, session_id, since_ms) -> (hidden, superseded)`
// 3273). The types and the `Result<usize>` (rows changed) are assumed. Commit 11 settles
// them before wave 4 calls them.

pub fn hide(conn: &Connection, id: &str, hidden_by: &str) -> Result<usize> {
    let _ = (conn, id, hidden_by);
    Ok(0)
}

pub fn restore(conn: &Connection, id: &str) -> Result<usize> {
    let _ = (conn, id);
    Ok(0)
}

pub fn set_pause(
    conn: &Connection,
    logical: &str,
    session_id: Option<&str>,
    what: &str,
    set_by: &str,
) -> Result<usize> {
    let _ = (conn, logical, session_id, what, set_by);
    Ok(0)
}

pub fn clear_pause(conn: &Connection, logical: &str, session_id: Option<&str>, what: &str) -> Result<usize> {
    let _ = (conn, logical, session_id, what);
    Ok(0)
}

/// `(hidden, superseded)`.
pub fn model_changes_since(conn: &Connection, session_id: &str, since_ms: i64) -> Result<(usize, usize)> {
    let _ = (conn, session_id, since_ms);
    Ok((0, 0))
}
