//! Post, read, forget on `forum_posts` (spec section 6, "Storage").
//!
//! The behaviour is core's sidecar (`axel-memory-service/src/forum.rs`) on two tables of the
//! brain. The functions that write open no transaction: the caller did, and it rolls back
//! after any error. Forum rows never enter `memories`, `documents` or a vector.

use std::time::{Duration, Instant};

use rusqlite::types::ValueRef;
use rusqlite::{params, Connection, OptionalExtension};

use crate::error::{AxelError, Result};
use crate::ext::types::Caller;
use crate::forum::{self, Author, Cursor, Entry, Envelope, Page, Post, Read, Receipt, Status};
use crate::{index_archives, screen};

/// A read examines at most this many rows.
pub const READ_MAX_ROWS: usize = 5_000;
/// A read examines rows for at most this long, in milliseconds.
pub const READ_MAX_MS: u64 = 250;
/// Rows are fetched this many at a time.
const FETCH: usize = 64;
const DAY_MS: i64 = 86_400_000;

// SPEC: 4523-4529 give the codes (`[not_found]`, `[conflict]`, invalid) and "core's
// guidance" without the sentences. They are core's (`memory_backend/process.rs`,
// `service_error`) after the code, with the wire name of the read tool.
pub const NOT_FOUND: &str = "forum: [not_found] referenced record or thread is unavailable in this scope. For a new forum thread omit/null thread_id and reply_to and provide a title. For a reply, first axel_forum_read and copy an exact live thread ID; never invent IDs.";
pub const INVALID_REFERENCE: &str = "forum: [invalid] check declared argument types and bounds. For forum tools, omit unused optional fields or use null; never invent thread/reply IDs.";
pub const CONFLICT: &str = "forum: [conflict] record ID is unavailable; do not bypass a tombstone or change scope to retry.";
pub const TOO_LARGE: &str = "forum: [size_limit] the next post does not fit a page";
pub const UNAVAILABLE: &str = "forum: forum post unavailable in this project";
pub const NO_TRANSACTION: &str =
    "forum: no transaction is open; a write starts with brain_v2::begin_write";

fn refuse(text: impl Into<String>) -> AxelError {
    AxelError::Other(text.into())
}

/// The text of an error this file made to refuse a request (it starts `forum: `); `None`
/// for a failure of the brain itself.
pub fn refusal(err: &AxelError) -> Option<&str> {
    match err {
        AxelError::Other(text) if text.starts_with("forum: ") => Some(text),
        _ => None,
    }
}

fn need_transaction(conn: &Connection) -> Result<()> {
    if conn.is_autocommit() {
        return Err(refuse(NO_TRANSACTION));
    }
    Ok(())
}

const COLUMNS: &str = "id, timestamp_ms, project_key, origin_project, thread_id, reply_to, title, \
    body, author_actor, author_group, author_parent, request_key, retention_days, digest, \
    expires_ms, digest_ok";

/// One row, read without trusting its column types.
struct Stored {
    id: String,
    timestamp_ms: i64,
    expires_ms: i64,
    digest_ok: bool,
    envelope: Envelope,
    body: String,
}

fn text(r: &rusqlite::Row<'_>, i: usize) -> rusqlite::Result<Option<String>> {
    Ok(match r.get_ref(i)? {
        ValueRef::Text(bytes) => std::str::from_utf8(bytes).ok().map(str::to_owned),
        _ => None,
    })
}

fn integer(r: &rusqlite::Row<'_>, i: usize) -> rusqlite::Result<Option<i64>> {
    Ok(match r.get_ref(i)? {
        ValueRef::Integer(n) => Some(n),
        _ => None,
    })
}

/// `(id, timestamp_ms, the row)`. The row is `None` when a column is not what the table
/// says it holds; its place in the order is still known.
fn stored(r: &rusqlite::Row<'_>) -> rusqlite::Result<(String, i64, Option<Stored>)> {
    let id: String = r.get(0)?;
    let timestamp_ms: i64 = r.get(1)?;
    let reply_to = match r.get_ref(5)? {
        ValueRef::Null => Some(None),
        _ => text(r, 5)?.map(Some),
    };
    let parent = match r.get_ref(10)? {
        ValueRef::Null => Some(None),
        _ => text(r, 10)?.map(Some),
    };
    let row = columns(r, &id, timestamp_ms, reply_to, parent);
    Ok((id, timestamp_ms, row))
}

fn columns(
    r: &rusqlite::Row<'_>,
    id: &str,
    timestamp_ms: i64,
    reply_to: Option<Option<String>>,
    parent: Option<Option<String>>,
) -> Option<Stored> {
    Some(Stored {
        id: id.to_owned(),
        timestamp_ms,
        expires_ms: integer(r, 14).ok()??,
        digest_ok: integer(r, 15).ok()?? == 1,
        envelope: Envelope {
            version: 1,
            project: text(r, 3).ok()??,
            author: Author { actor: text(r, 8).ok()??, group: text(r, 9).ok()??, parent: parent? },
            request_key: text(r, 11).ok()??,
            thread_id: text(r, 4).ok()??,
            reply_to: reply_to?,
            title: text(r, 6).ok()??,
            retention_days: u32::try_from(integer(r, 12).ok()??).ok()?,
            digest: text(r, 13).ok()??,
        },
        body: text(r, 7).ok()??,
    })
}

impl Stored {
    /// The whole post, when the row is marked sound and still passes its digest.
    fn into_entry(self) -> Option<Entry> {
        if !self.digest_ok {
            return None;
        }
        let entry = Entry {
            id: self.id,
            timestamp_ms: u64::try_from(self.timestamp_ms).ok()?,
            envelope: self.envelope,
            body: self.body,
            truncated: false,
        };
        entry.validate().ok()?;
        Some(entry)
    }
}

fn by_id(conn: &Connection, id: &str) -> Result<Option<(String, Option<Stored>)>> {
    let sql = format!("SELECT {COLUMNS} FROM forum_posts WHERE id = ?1");
    Ok(conn
        .query_row(&sql, [id], |r| {
            let project_key = text(r, 2)?.unwrap_or_default();
            Ok((project_key, stored(r)?.2))
        })
        .optional()?)
}

/// The envelope of a post that is listed in `project`, has not expired at `now` and passes
/// its digest.
fn live(conn: &Connection, project: &str, id: &str, now: i64) -> Result<Option<Envelope>> {
    let Some((project_key, Some(row))) = by_id(conn, id)? else { return Ok(None) };
    if project_key != project || row.expires_ms <= now {
        return Ok(None);
    }
    Ok(row.into_entry().map(|entry| entry.envelope))
}

fn credential(hit: &screen::Hit) -> AxelError {
    let what = match (hit.rule, &hit.name) {
        ("prefix", Some(prefix)) => format!("a token starting \"{prefix}\""),
        ("jwt", _) => "a JWT".to_string(),
        ("key_block", _) => "a private key block".to_string(),
        ("url_password", _) => "a password inside a URL".to_string(),
        _ => "a token of a known shape".to_string(),
    };
    refuse(format!(
        "forum: [invalid] the post contains what looks like a credential ({what}); remove it and post again"
    ))
}

/// True for a character `screen::show` drops without a trace (bidirectional controls,
/// zero-width and other invisible characters). `show` of the one character is empty then;
/// whitespace, which `show` turns into a gap, is not such a character.
// SPEC: ruling R-17.5 names "what `screen::show` removes"; `screen` keeps that test private
// (`removed`), so it is asked through `show`, once per distinct non-ASCII character.
// RULING: R-17.5, second ruling: the variation selectors and the zero-width joiner pass,
// because emoji are made of them. The credential check does not depend on this one: `screen`
// drops these characters before it matches, so a token split by one is still refused.
fn invisible(c: char) -> bool {
    if emoji_part(c) {
        return false;
    }
    !c.is_ascii() && !c.is_whitespace() && screen::show(c.encode_utf8(&mut [0; 4]), 1).is_empty()
}

/// U+FE00 to U+FE0F (variation selectors) and U+200D (the joiner).
fn emoji_part(c: char) -> bool {
    matches!(c, '\u{FE00}'..='\u{FE0F}' | '\u{200D}')
}

/// The first of `body`, `title`, `request_key` that holds an invisible character.
pub fn invisible_field(post: &Post) -> Option<&'static str> {
    let mut seen: Vec<(char, bool)> = Vec::new();
    for (field, text) in [("body", &post.body), ("title", &post.title), ("request_key", &post.request_key)] {
        for c in text.chars().filter(|c| !c.is_ascii()) {
            let bad = match seen.iter().find(|(known, _)| *known == c) {
                Some((_, bad)) => *bad,
                None => {
                    let bad = invisible(c);
                    seen.push((c, bad));
                    bad
                }
            };
            if bad {
                return Some(field);
            }
        }
    }
    None
}

/// What a post with an invisible character in `field` is told. The character is not repeated.
pub fn invisible_text(field: &str) -> String {
    format!(
        "forum: [invalid] {field} holds an invisible character (a zero-width or text-direction \
         control); remove it and post again"
    )
}

/// One post, inside the caller's transaction. `clock_ms` is the caller's clock read inside
/// that transaction.
// SPEC: 4508 writes `post(conn, project, author, post)`, and step 6 needs `clock_ms`, which
// the tests replace with a stub: it is the fifth argument.
pub fn post(conn: &Connection, project: &str, author: &Author, post: &Post, clock_ms: i64) -> Result<Receipt> {
    need_transaction(conn)?;
    // 1. A certain credential shape refuses the post; the value is never repeated.
    for part in [&post.body, &post.title, &post.request_key] {
        if let Some(hit) = screen::certain_hit(part) {
            return Err(credential(&hit));
        }
    }
    if let Some(field) = invisible_field(post) {
        return Err(refuse(invisible_text(field)));
    }
    // 2.
    let envelope = Envelope::new(project, author.clone(), post).map_err(|e| refuse(format!("forum: {e}")))?;
    let id = envelope.id();
    let receipt = |status, timestamp_ms| Receipt {
        id: id.clone(),
        thread_id: envelope.thread_id.clone(),
        digest: envelope.digest.clone(),
        status,
        timestamp_ms,
    };
    // 3.
    let tombstoned: bool =
        conn.query_row("SELECT EXISTS(SELECT 1 FROM forum_tombstones WHERE id = ?1)", [&id], |r| r.get(0))?;
    if tombstoned {
        return Ok(receipt(Status::Tombstoned, None));
    }
    // 4. Before the references, as core does: a reply already committed stays reconcilable
    // after its root is deleted.
    if let Some((_, prior)) = by_id(conn, &id)? {
        let Some(prior) = prior.filter(|p| p.envelope == envelope && p.body == post.body) else {
            return Err(refuse(CONFLICT));
        };
        if prior.expires_ms <= clock_ms {
            return Ok(receipt(Status::Tombstoned, None));
        }
        let first = u64::try_from(prior.timestamp_ms).map_err(|_| refuse(CONFLICT))?;
        return Ok(receipt(Status::Duplicate, Some(first)));
    }
    // 5.
    if !envelope.is_root() {
        let root = live(conn, project, &envelope.thread_id, clock_ms)?.ok_or_else(|| refuse(NOT_FOUND))?;
        if !root.is_root() {
            return Err(refuse(NOT_FOUND));
        }
        if let Some(parent) = &envelope.reply_to {
            let parent = live(conn, project, parent, clock_ms)?.ok_or_else(|| refuse(NOT_FOUND))?;
            if parent.thread_id != envelope.thread_id {
                return Err(refuse(INVALID_REFERENCE));
            }
        }
    }
    // 6. Never at or before a time a poller of this project has already passed.
    let last: Option<i64> =
        conn.query_row("SELECT MAX(timestamp_ms) FROM forum_posts WHERE project_key = ?1", [project], |r| r.get(0))?;
    let now = last.map_or(clock_ms, |last| clock_ms.max(last.saturating_add(1)));
    let timestamp = u64::try_from(now).map_err(|_| refuse("forum: [invalid] the clock is before 1970"))?;
    // 7.
    conn.execute(
        "INSERT INTO forum_posts(id, project_key, origin_project, thread_id, reply_to, title, body, \
         author_actor, author_group, author_parent, request_key, retention_days, digest, \
         timestamp_ms, expires_ms, origin, digest_ok) \
         VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, 'axel', 1)",
        params![
            id,
            project,
            envelope.thread_id,
            envelope.reply_to,
            envelope.title,
            post.body,
            envelope.author.actor,
            envelope.author.group,
            envelope.author.parent,
            envelope.request_key,
            envelope.retention_days,
            envelope.digest,
            now,
            // From the clock, not from `now`: one row dated in the future moves every later
            // timestamp past it, and must not move every later expiry with it.
            clock_ms.saturating_add(i64::from(envelope.retention_days) * DAY_MS),
        ],
    )?;
    Ok(receipt(Status::Created, Some(timestamp)))
}

/// Why a read stopped before it ran out of rows or filled its page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    Rows,
    Time,
}

/// What one read did. Counts and a duration only.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReadStats {
    /// Rows the scan looked at.
    pub examined: usize,
    /// Rows that were not listed because they failed `Entry::validate` or could not be read.
    pub skipped: usize,
    pub stopped: Option<Stop>,
    pub elapsed: Duration,
}

/// The live posts of `project`: its roots, or one thread. No transaction is needed.
pub fn read(conn: &Connection, project: &str, read: &Read, now: i64) -> Result<Page> {
    Ok(read_counted(conn, project, read, now)?.0)
}

/// [`read`], and what it did.
pub fn read_counted(conn: &Connection, project: &str, read: &Read, now: i64) -> Result<(Page, ReadStats)> {
    read_bounded(conn, project, read, now, READ_MAX_ROWS, Duration::from_millis(READ_MAX_MS))
}

/// [`read`] with its two bounds as arguments. A bound is looked at between two rows and
/// never before the first, so every call moves on by at least one row.
// SPEC: 4536-4541: the rows SQLite itself passes over (expired, `digest_ok = 0`, replies in
// a root listing) are filtered in the statement, as the spec writes it, and are therefore
// not among the rows counted or timed here.
pub fn read_bounded(
    conn: &Connection,
    project: &str,
    read: &Read,
    now: i64,
    max_rows: usize,
    max_time: Duration,
) -> Result<(Page, ReadStats)> {
    read.validate().map_err(|e| refuse(format!("forum: {e}")))?;
    let started = Instant::now();
    let query = read.query.as_ref().map(|q| q.to_lowercase());
    let listing = read.thread_id.is_none();
    let sql = format!(
        "SELECT {COLUMNS} FROM forum_posts \
         WHERE project_key = ?1 AND digest_ok = 1 AND expires_ms > ?2 AND {} \
           AND (timestamp_ms, id) > (?3, ?4) \
         ORDER BY timestamp_ms, id LIMIT {FETCH}",
        if listing { "thread_id = id" } else { "thread_id = ?5" }
    );
    let mut stmt = conn.prepare(&sql)?;
    let (mut at_ms, mut at_id) = match &read.after {
        Some(cursor) => (cursor.timestamp_ms as i64, cursor.id.clone()),
        None => (i64::MIN, String::new()),
    };
    let mut page = Page { entries: Vec::new(), next: None };
    let mut stats = ReadStats::default();
    let mut examined_last: Option<Cursor> = None;
    let mut body_bytes = 0;
    let mut more = false;
    'scan: loop {
        let rows: Vec<(String, i64, Option<Stored>)> = match &read.thread_id {
            None => stmt.query_map(params![project, now, at_ms, at_id], stored)?.collect::<rusqlite::Result<_>>()?,
            Some(thread) => {
                stmt.query_map(params![project, now, at_ms, at_id, thread], stored)?.collect::<rusqlite::Result<_>>()?
            }
        };
        let fetched = rows.len();
        for (id, timestamp_ms, row) in rows {
            if stats.examined > 0 && (stats.examined >= max_rows || started.elapsed() >= max_time) {
                stats.stopped = Some(if stats.examined >= max_rows { Stop::Rows } else { Stop::Time });
                break 'scan;
            }
            stats.examined += 1;
            let here = Cursor { timestamp_ms: u64::try_from(timestamp_ms).unwrap_or(0), id: id.clone() };
            at_ms = timestamp_ms;
            at_id = id;
            let Some(row) = row else {
                stats.skipped += 1;
                examined_last = Some(here);
                continue;
            };
            if let Some(q) = &query {
                if !row.body.to_lowercase().contains(q) && !row.envelope.title.to_lowercase().contains(q) {
                    examined_last = Some(here);
                    continue;
                }
            }
            // The digest is checked on the whole body; a row that fails is passed over.
            let Some(entry) = row.into_entry() else {
                stats.skipped += 1;
                examined_last = Some(here);
                continue;
            };
            // The sidecar's budgeting loop.
            if page.entries.len() == read.limit {
                more = true;
                break 'scan;
            }
            let Entry { id, timestamp_ms, envelope, mut body, .. } = entry;
            let truncated = listing && body.len() > forum::SNIPPET_BYTES;
            if truncated {
                let mut end = forum::SNIPPET_BYTES;
                while !body.is_char_boundary(end) {
                    end -= 1;
                }
                body.truncate(end);
            }
            let length = body.len();
            if body_bytes + length > forum::PAGE_BODY_BYTES {
                more = true;
                break 'scan;
            }
            let entry = Entry { id, timestamp_ms, envelope, body, truncated };
            page.next = Some(entry.cursor()); // Reserve cursor bytes while budgeting.
            page.entries.push(entry);
            if serde_json::to_vec(&page)?.len() > forum::PAGE_BYTES {
                page.entries.pop();
                more = true;
                break 'scan;
            }
            body_bytes += length;
            examined_last = Some(here);
        }
        if fetched < FETCH {
            break;
        }
    }
    if more && page.entries.is_empty() {
        return Err(refuse(TOO_LARGE));
    }
    page.next = if more {
        // A has-more cursor: the last entry given.
        page.entries.last().map(Entry::cursor)
    } else if stats.stopped.is_some() {
        // A bound ended the scan: the last row looked at, listed or not.
        examined_last
    } else {
        None
    };
    stats.elapsed = started.elapsed();
    Ok((page, stats))
}

/// Deletes one post of `project`, live or expired, and leaves a tombstone that names who
/// asked. Inside the caller's transaction; nobody's authorship is compared.
// SPEC: 4550 writes `forget(conn, project, id, caller)`; `deleted_ms` needs the caller's
// clock, which is the fifth argument.
// SPEC: 4550-4557 do not say what `source` a tombstone made here gets: `tool`, the
// column's default.
// SPEC: 4553: the body must be gone from the file, so a connection without
// `secure_delete` is refused, as `store::purge_many` refuses it.
pub fn forget(conn: &Connection, project: &str, id: &str, caller: &Caller, now: i64) -> Result<()> {
    need_transaction(conn)?;
    if !forum::valid_id(id) {
        return Err(refuse("forum: id must be an exact msg-<64 lowercase hex> forum post ID"));
    }
    let secure_delete: i64 = conn.query_row("PRAGMA secure_delete", [], |r| r.get(0))?;
    if secure_delete != 1 {
        return Err(refuse(format!(
            "forum: this connection has PRAGMA secure_delete = {secure_delete}; a forget needs \
             secure_delete = ON (a connection of brain_v2::connect has it)"
        )));
    }
    let there: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM forum_posts WHERE id = ?1 AND project_key = ?2)",
        params![id, project],
        |r| r.get(0),
    )?;
    if !there {
        return Err(refuse(UNAVAILABLE));
    }
    let author = caller.author.as_ref();
    conn.execute(
        // The first record of a deletion wins: a tombstone already there (an imported one) is kept.
        "INSERT OR IGNORE INTO forum_tombstones(id, project_key, deleted_ms, source, by_actor, by_group, \
         by_parent, by_session) VALUES (?1, ?2, ?3, 'tool', ?4, ?5, ?6, ?7)",
        params![
            id,
            project,
            now,
            author.map(|a| a.actor.as_str()),
            author.map(|a| a.group.as_str()),
            author.and_then(|a| a.parent.as_deref()),
            caller.session_id.as_deref(),
        ],
    )?;
    conn.execute("DELETE FROM forum_posts WHERE id = ?1 AND project_key = ?2", params![id, project])?;
    index_archives::mark_purged(conn)
}
