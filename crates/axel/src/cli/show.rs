//! `show <id>`: one memory in full, or one archive (`ctx-<id>`). Commit 18b writes this file.
//!
//! It also holds what the other commands of commit 18b share (`forget`, `restore`, `pause`,
//! `resume`, `recall`, `search`, `memories`): where they print, how an id given on the
//! command line finds its row, the write lock, dates. Everything printed that came from a
//! row passes `screen::show` (one line) or `screen::show_block` (a body).

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::process::ExitCode;

use axel_memkoshi::memory::{Memory, MemoryCategory};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};

use crate::brain_v2::{self, Role};
use crate::error::{AxelError, Result};
use crate::ext::types::BrainArg;
use crate::screen;

#[derive(clap::Args)]
pub struct Args {
    /// A row id, or `ctx-<id>` for an archive
    pub id: String,
}

// ── Shared by the commands of commit 18b ────────────────────────────────────

/// Exit codes: done; refused or not found (nothing changed); a usage error.
pub const OK: u8 = 0;
pub const FAILED: u8 = 1;
pub const USAGE: u8 = 2;

/// What a write that could not get the lock says. A command waits for the lock as long as
/// `begin_write` does on a command-line connection (its busy timeout).
pub const BUSY: &str =
    "the brain is busy (the extension or another command is writing to it); nothing was changed; try again in a moment";

/// Where a command prints, and where `forget --now` reads its answer.
pub struct Term<'a> {
    pub out: &'a mut dyn Write,
    pub err: &'a mut dyn Write,
    pub input: &'a mut dyn BufRead,
}

/// Runs `work` on the process's own stdout, stderr and stdin.
pub fn on_std(work: impl FnOnce(&mut Term) -> Result<u8>) -> Result<ExitCode> {
    let (stdout, stderr, stdin) = (io::stdout(), io::stderr(), io::stdin());
    let (mut out, mut err, mut input) = (stdout.lock(), stderr.lock(), stdin.lock());
    let mut term = Term { out: &mut out, err: &mut err, input: &mut input };
    let code = work(&mut term)?;
    term.out.flush()?;
    Ok(ExitCode::from(code))
}

/// The brain, opened as every command line write opens it (secure delete on).
pub fn open(path: &Path) -> Result<Connection> {
    brain_v2::connect(path, Role::Cli)
}

/// [`open`], then [`forget_pass`]: for a command that prints text of rows.
pub fn open_to_read(path: &Path) -> Result<Connection> {
    let conn = open(path)?;
    forget_pass(&conn)?;
    Ok(conn)
}

/// What the extension does before it serves the brain (`indexer_start::forget_pass`), in
/// the base the brain records: archives that are tombstones now and notes files that
/// changed. Nothing is written unless a file changed; a lock that did not come in time is
/// `BUSY`, and the command prints nothing of the brain.
// RULING: v2 real-binaries round 2, item 3: forgotten text is not printed by a command
// run while no extension has applied the forget.
pub fn forget_pass(conn: &Connection) -> Result<()> {
    let Some(base) = crate::cli::forget::recorded_base(conn)? else { return Ok(()) };
    let signer = brain_v2::signer(conn)?;
    let clock: crate::ext::types::Clock = std::sync::Arc::new(now_ms);
    let mut gone = Vec::new();
    crate::index_archives::forget_tombstoned(conn, &base, &mut gone)
        .and_then(|_| crate::index_notes::forget_changed(conn, signer.as_ref(), &base, &clock, &mut gone))
        .map(|_| ())
        .map_err(|e| if is_busy(&e) { AxelError::Other(BUSY.into()) } else { e })
}

fn is_busy(e: &AxelError) -> bool {
    matches!(
        e,
        AxelError::Db(rusqlite::Error::SqliteFailure(f, _))
            if f.code == rusqlite::ErrorCode::DatabaseBusy || f.code == rusqlite::ErrorCode::DatabaseLocked
    )
}

/// One write: `begin_write`, the work, `COMMIT`. After any error nothing is committed; a
/// lock that did not come in time is `BUSY`.
pub fn write<T>(conn: &Connection, work: impl FnOnce() -> Result<T>) -> Result<T> {
    if let Err(e) = brain_v2::begin_write(conn) {
        return Err(if is_busy(&e) { AxelError::Other(BUSY.into()) } else { e });
    }
    let out = work().and_then(|value| {
        conn.execute_batch("COMMIT")?;
        Ok(value)
    });
    if out.is_err() && !conn.is_autocommit() {
        let _ = conn.execute_batch("ROLLBACK");
    }
    out.map_err(|e| if is_busy(&e) { AxelError::Other(BUSY.into()) } else { e })
}

pub fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

/// `2026-11-10 14:05 UTC`.
pub fn date(ms: i64) -> String {
    DateTime::<Utc>::from_timestamp_millis(ms)
        .map(|t| t.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_else(|| format!("{ms} ms"))
}

/// `2026-11-10`.
pub fn day(ms: i64) -> String {
    DateTime::<Utc>::from_timestamp_millis(ms)
        .map(|t| t.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| format!("{ms} ms"))
}

/// A row's text field on one line, harmless on a terminal.
pub fn one(text: &str) -> String {
    screen::show(text, 300)
}

/// What an id given on the command line found.
pub enum Found {
    One(String),
    Missing,
    /// Two or more rows start with it: their ids, at most `MANY_LISTED + 1`.
    Many(Vec<String>),
}

pub const MANY_LISTED: usize = 10;

/// The id without the `m:` that `search` and `recall` print before it.
pub fn bare(given: &str) -> &str {
    given.trim().strip_prefix("m:").unwrap_or(given.trim())
}

/// A memory by its exact id, or by the start of one that only one row has.
// SPEC: 4193-4214 do not say whether an id may be shortened. The task allows a prefix and
// refuses one that two rows share, listing them.
pub fn find_memory(conn: &Connection, given: &str) -> Result<Found> {
    let id = bare(given);
    if id.is_empty() {
        return Ok(Found::Missing);
    }
    let exact: Option<String> =
        conn.query_row("SELECT id FROM memories WHERE id = ?1", [id], |r| r.get(0)).optional()?;
    if let Some(id) = exact {
        return Ok(Found::One(id));
    }
    let mut stmt = conn.prepare(
        "SELECT id FROM memories WHERE typeof(id) = 'text' AND substr(id, 1, length(?1)) = ?1 ORDER BY id LIMIT ?2",
    )?;
    let ids: Vec<String> = stmt
        .query_map(params![id, (MANY_LISTED + 1) as i64], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(match ids.len() {
        0 => Found::Missing,
        1 => Found::One(ids.into_iter().next().unwrap_or_default()),
        _ => Found::Many(ids),
    })
}

/// Every id, or none: an id that is a file, an archive, unknown or shared by two rows is
/// reported on `err` and the answer is `None`, so the caller changes nothing.
pub fn find_all(conn: &Connection, given: &[String], err: &mut dyn Write) -> Result<Option<Vec<String>>> {
    let mut found = Vec::new();
    let mut ok = true;
    for g in given {
        let g = g.trim();
        if g.starts_with("doc:") {
            writeln!(err, "{}: that is a file of yours; it is changed by editing the file", one(g))?;
            ok = false;
            continue;
        }
        if g.starts_with("ctx-") {
            writeln!(
                err,
                "{}: that is a Synaps archive; forget it in Synaps with memory_forget, or bring back the turns it left out with \"axel-v2 restore --forget-turns {}\"",
                one(g),
                one(g)
            )?;
            ok = false;
            continue;
        }
        match find_memory(conn, g)? {
            Found::One(id) => {
                if !found.contains(&id) {
                    found.push(id);
                }
            }
            Found::Missing => {
                writeln!(err, "{}: not found", one(g))?;
                ok = false;
            }
            Found::Many(ids) => {
                let more = if ids.len() > MANY_LISTED { ", and more" } else { "" };
                let listed: Vec<String> = ids.iter().take(MANY_LISTED).map(|id| format!("m:{}", one(id))).collect();
                writeln!(err, "{}: more than one row starts with this: {}{more}", one(g), listed.join(", "))?;
                ok = false;
            }
        }
    }
    Ok(ok.then_some(found))
}

/// One row of `memories` with its `memory_ext` row, as the commands print it.
pub struct RowInfo {
    pub id: String,
    pub kind: String,
    pub topic: String,
    pub title: String,
    pub abstract_text: String,
    pub content: String,
    pub signature: Option<String>,
    pub project_key: Option<String>,
    pub secret: bool,
    pub provenance: String,
    pub created: String,
    pub expires_at: Option<String>,
    pub superseded_by: Option<String>,
    pub invalidated_at: Option<String>,
    pub links: [String; 3],
    pub tags: String,
    pub origin: String,
    pub origin_ref: String,
    pub unit_pos: Option<String>,
    pub session_id: Option<String>,
    pub logical: Option<String>,
    pub turn_key: Option<String>,
    pub created_ms: Option<i64>,
    pub time_source: Option<String>,
    pub hidden_ms: Option<i64>,
    pub hidden_by: Option<String>,
}

const ROW_SQL: &str = "SELECT m.id, m.category, m.topic, m.title, m.abstract_text, m.content, m.signature,
        m.project_key, m.sensitivity, m.provenance, m.created, m.expires_at, m.superseded_by,
        m.invalidated_at, m.supersedes, m.contradicts, m.confirmed_by, m.tags,
        e.origin_kind, e.origin_ref, e.unit_pos, e.session_id, e.logical, e.turn_key, e.created_ms,
        e.time_source, e.hidden_ms, e.hidden_by
   FROM memories m LEFT JOIN memory_ext e ON e.mem_id = m.id";

fn row_info_of(r: &rusqlite::Row) -> rusqlite::Result<RowInfo> {
    let text = |i: usize| -> rusqlite::Result<String> { Ok(brain_v2::text_at(r, i)?.unwrap_or_default()) };
    let opt = |i: usize| -> rusqlite::Result<Option<String>> { brain_v2::text_at(r, i) };
    Ok(RowInfo {
        id: text(0)?,
        kind: text(1)?,
        topic: text(2)?,
        title: text(3)?,
        abstract_text: text(4)?,
        content: text(5)?,
        signature: opt(6)?,
        project_key: opt(7)?,
        secret: opt(8)?.as_deref() == Some("secret"),
        provenance: text(9)?,
        created: text(10)?,
        expires_at: opt(11)?,
        superseded_by: opt(12)?,
        invalidated_at: opt(13)?,
        links: [text(14)?, text(15)?, text(16)?],
        tags: text(17)?,
        origin: opt(18)?.unwrap_or_else(|| "unknown".to_string()),
        origin_ref: text(19)?,
        unit_pos: opt(20)?,
        session_id: opt(21)?,
        logical: opt(22)?,
        turn_key: opt(23)?.filter(|k| !k.is_empty()),
        created_ms: r.get(24)?,
        time_source: opt(25)?,
        hidden_ms: r.get(26)?,
        hidden_by: opt(27)?,
    })
}

pub fn row_info(conn: &Connection, id: &str) -> Result<Option<RowInfo>> {
    Ok(conn.query_row(&format!("{ROW_SQL} WHERE m.id = ?1"), [id], row_info_of).optional()?)
}

/// Rows chosen by a `WHERE` on `m` and `e`, in `ORDER BY` order.
pub fn rows_where(conn: &Connection, filter: &str, args: impl rusqlite::Params) -> Result<Vec<RowInfo>> {
    let mut stmt = conn.prepare(&format!("{ROW_SQL} {filter}"))?;
    let rows = stmt.query_map(args, row_info_of)?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// "unchanged since saved": the signature checked with the brain's key. It says nothing
/// about who wrote the row: the key is in the same file.
pub fn unchanged(conn: &Connection, row: &RowInfo) -> Result<&'static str> {
    let Some(signature) = row.signature.clone() else { return Ok("not signed") };
    let Some(signer) = brain_v2::signer(conn)? else { return Ok("not signed") };
    let category = MemoryCategory::parse(&row.kind).unwrap_or(MemoryCategory::Events);
    let mut memory = Memory::new(category, row.topic.clone(), row.title.clone(), row.content.clone());
    memory.id = row.id.clone();
    memory.signature = Some(signature);
    Ok(if signer.verify(&memory) { "yes" } else { "no" })
}

/// `hidden by cli on 2026-11-10 14:05 UTC` / `superseded by m:… on …` / `visible`.
pub fn state(row: &RowInfo) -> String {
    let mut parts = Vec::new();
    if let Some(ms) = row.hidden_ms {
        parts.push(format!("hidden by {} on {}", one(row.hidden_by.as_deref().unwrap_or("?")), date(ms)));
    }
    if let Some(by) = &row.superseded_by {
        let when = row.invalidated_at.as_deref().map(|t| format!(" on {}", one(t))).unwrap_or_default();
        parts.push(format!("superseded by m:{}{when}", one(by)));
    } else if let Some(t) = &row.invalidated_at {
        parts.push(format!("invalidated on {}", one(t)));
    }
    if parts.is_empty() {
        "visible".to_string()
    } else {
        parts.join("; ")
    }
}

// ── show ────────────────────────────────────────────────────────────────────

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    on_std(|term| run_with(args, &brain.path, term))
}

pub fn run_with(args: &Args, path: &Path, term: &mut Term) -> Result<u8> {
    let conn = open_to_read(path)?;
    let given = args.id.trim();
    if let Some(archive) = given.strip_prefix("ctx-") {
        return show_archive(&conn, archive, term);
    }
    let Some(ids) = find_all(&conn, &[given.to_string()], term.err)? else {
        return Ok(FAILED);
    };
    let Some(row) = row_info(&conn, &ids[0])? else {
        writeln!(term.err, "{}: not found", one(given))?;
        return Ok(FAILED);
    };
    let out = &mut *term.out;
    writeln!(out, "m:{}", one(&row.id))?;
    writeln!(out, "  kind: {} · topic: {}", one(&row.kind), one(&row.topic))?;
    let from = if row.origin_ref.is_empty() { String::new() } else { format!(" ({})", one(&row.origin_ref)) };
    writeln!(out, "  origin: {}{from} · provenance: {}", one(&row.origin), one(&row.provenance))?;
    if let Some(pos) = &row.unit_pos {
        writeln!(out, "  position in its archive: {}", one(pos))?;
    }
    writeln!(out, "  project: {}", one(row.project_key.as_deref().unwrap_or("none (an old memory)")))?;
    let when = match (row.created_ms, &row.time_source) {
        (Some(ms), Some(source)) => format!("{} ({})", date(ms), one(source)),
        _ => one(&row.created),
    };
    writeln!(out, "  saved: {when}")?;
    if let Some(t) = &row.expires_at {
        writeln!(out, "  expires: {}", one(t))?;
    }
    writeln!(out, "  state: {}", state(&row))?;
    for (name, links) in ["supersedes", "contradicts", "confirmed by"].iter().zip(&row.links) {
        if !links.is_empty() && links != "[]" {
            writeln!(out, "  {name}: {}", one(links))?;
        }
    }
    if !row.tags.is_empty() && row.tags != "[]" {
        writeln!(out, "  tags: {}", one(&row.tags))?;
    }
    writeln!(out, "  unchanged since saved: {}", unchanged(&conn, &row)?)?;
    let twins = rows_where(
        &conn,
        "WHERE e.hidden_by = ?1 ORDER BY e.vid",
        [format!("twin:{}", row.id)],
    )?;
    if !twins.is_empty() {
        writeln!(out, "  hidden with it ({}):", twins.len())?;
        for twin in &twins {
            let pos = twin.unit_pos.as_deref().map(|p| format!(" {}", one(p))).unwrap_or_default();
            writeln!(out, "    m:{} (archive ctx-{}{pos})", one(&twin.id), one(&twin.origin_ref))?;
        }
    }
    writeln!(out, "  title: {}", one(&row.title))?;
    if row.secret {
        // RULING: a secret row's text is not printed; the spec gives no flag for it.
        writeln!(out, "  text: this memory is secret; its text is not shown")?;
    } else {
        writeln!(out, "  text:")?;
        for line in screen::show_block(&row.content, 20_000).lines() {
            writeln!(out, "    {line}")?;
        }
    }
    Ok(OK)
}

/// The rows of one archive hidden as `cli:unproven` now, oldest first:
/// `m:<id> · <date> · <first line of its reply>`.
fn unproven_rows(conn: &Connection, scope_dir: &str, archive_id: &str) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT e.mem_id, e.created_ms, m.content, e.turn_key, m.sensitivity
           FROM memory_ext e JOIN memories m ON m.id = e.mem_id
          WHERE e.origin_kind = 'archive' AND e.origin_ref = ?1 AND m.project_key = ?2
            AND e.hidden_ms IS NOT NULL AND e.hidden_by = ?3
          ORDER BY e.created_ms, e.vid",
    )?;
    let rows = stmt.query_map(params![archive_id, scope_dir, crate::index_archives::UNPROVEN], |r| {
        Ok((
            crate::brain_v2::text_at(r, 0)?.unwrap_or_default(),
            r.get::<_, Option<i64>>(1)?,
            crate::brain_v2::text_at(r, 2)?.unwrap_or_default(),
            crate::brain_v2::text_at(r, 3)?,
            crate::brain_v2::text_at(r, 4)?,
        ))
    })?;
    let mut lines = Vec::new();
    for row in rows {
        let (id, created_ms, content, turn_key, sensitivity) = row?;
        let when = created_ms.map(date).unwrap_or_else(|| "no date".to_string());
        let text = if sensitivity.as_deref() == Some("secret") {
            "(a secret memory; its text is not shown)".to_string()
        } else {
            crate::screen::show(crate::cli::forget::reply_line(&content, turn_key.as_deref()), 120)
        };
        lines.push(format!("m:{} · {when} · {text}", one(&id)));
    }
    Ok(lines)
}

/// The `archive_index` rows of one archive, and the turns it leaves out.
// SPEC: 3729 says `show ctx-<id>` "gives their positions". Version 1 records how many
// units an archive left out (`forget_units`) and the marks it gave, not the positions of
// the units that are not rows; reading them would mean reading core's file. Shown: the
// count, the marks, the rows that are there.
fn show_archive(conn: &Connection, given: &str, term: &mut Term) -> Result<u8> {
    let Some(archive_id) = find_archive(conn, given, term.err)? else {
        return Ok(FAILED);
    };
    let out = &mut *term.out;
    let mut stmt = conn.prepare(
        "SELECT scope_dir, rel_name, state, skip_reason, cover_ms, rows_indexed, forget_units,
                keep_forget_turns, unmatched_hidden, session_id, cli_units, cli_unproven
           FROM archive_index WHERE archive_id = ?1 ORDER BY scope_dir, rel_name",
    )?;
    type ArchiveRow = ((String, String, String, Option<String>, Option<i64>), (i64, i64, i64, i64, Option<String>), (i64, i64));
    let rows: Vec<ArchiveRow> = stmt
        .query_map([&archive_id], |r| {
            Ok((
                (r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?),
                (r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?),
                (r.get(10)?, r.get(11)?),
            ))
        })?
        .collect::<rusqlite::Result<_>>()?;
    writeln!(out, "ctx-{}", one(&archive_id))?;
    for ((scope_dir, rel_name, file_state, reason, cover_ms), (rows_indexed, forget_units, keep, unmatched, session), (cli_units, cli_unproven)) in &rows {
        writeln!(out, "  file: {} in {}", one(rel_name), one(scope_dir))?;
        let reason = reason.as_deref().map(|r| format!(" ({})", one(r))).unwrap_or_default();
        writeln!(out, "    state: {}{reason}", one(file_state))?;
        if let Some(ms) = cover_ms {
            writeln!(out, "    sealed: {}", date(*ms))?;
        }
        if let Some(s) = session {
            writeln!(out, "    session: {}", one(s))?;
        }
        writeln!(out, "    rows indexed: {rows_indexed}")?;
        writeln!(out, "    turns left out because a forget tool was called: {forget_units}")?;
        // RULING: v2 c18b review round 3, item 1: what turns deleted with `forget --now` hold
        // back, taken and hidden without proof.
        writeln!(out, "    turns left out because they were deleted with forget --now: {cli_units}")?;
        if *cli_unproven > 0 {
            writeln!(
                out,
                "    turns hidden because they have the prompt of a deleted turn and their text does not prove them that turn: {cli_unproven} (\"axel-v2 restore <id>\" brings one back)"
            )?;
        }
        // RULING: v2 c18b review round 4, item 2: the ids, so that one can be brought back.
        let unproven = unproven_rows(conn, scope_dir, &archive_id)?;
        if !unproven.is_empty() {
            writeln!(out, "    hidden now as not proven ({}):", unproven.len())?;
            for line in &unproven {
                writeln!(out, "      {line}")?;
            }
        }
        if *keep == 1 {
            writeln!(out, "    the forget rules are off for this file (restore --forget-turns)")?;
        }
        if *unmatched > 0 {
            writeln!(out, "    a hidden turn could not be matched in it; its archived form is visible ({unmatched})")?;
        }
    }
    let mut marks = conn.prepare(
        "SELECT turn_key, head, tail FROM forgotten_turns WHERE archive_id = ?1 ORDER BY id",
    )?;
    let marks: Vec<(String, i64, i64)> = marks
        .query_map([&archive_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    if !marks.is_empty() {
        writeln!(out, "  forget marks it gave ({}):", marks.len())?;
        for (key, head, tail) in &marks {
            let key = if key.is_empty() { "no prompt key".to_string() } else { format!("prompt key {}", one(key)) };
            let mut at = Vec::new();
            if *head == 1 {
                at.push("its leading span");
            }
            if *tail == 1 {
                at.push("its last span");
            }
            let at = if at.is_empty() { String::new() } else { format!(", {}", at.join(" and ")) };
            writeln!(out, "    {key}{at}")?;
        }
    }
    let positions = rows_where(
        conn,
        "WHERE e.origin_kind = 'archive' AND e.origin_ref = ?1 ORDER BY e.unit_pos",
        [&archive_id],
    )?;
    if !positions.is_empty() {
        let listed: Vec<String> = positions
            .iter()
            .map(|r| {
                let hidden = if r.hidden_ms.is_some() { " (hidden)" } else { "" };
                format!("{}{hidden}", one(r.unit_pos.as_deref().unwrap_or("?")))
            })
            .collect();
        writeln!(out, "  positions that are rows: {}", listed.join(", "))?;
    }
    Ok(OK)
}

/// An archive id (32 hex) by itself or by the start of one; `None` after a message.
pub fn find_archive(conn: &Connection, given: &str, err: &mut dyn Write) -> Result<Option<String>> {
    let id = given.trim().trim_start_matches("ctx-");
    if id.is_empty() {
        writeln!(err, "ctx-: no archive id given")?;
        return Ok(None);
    }
    let mut stmt = conn.prepare(
        "SELECT DISTINCT archive_id FROM archive_index WHERE substr(archive_id, 1, length(?1)) = ?1
          ORDER BY archive_id LIMIT ?2",
    )?;
    let ids: Vec<String> = stmt
        .query_map(params![id, (MANY_LISTED + 1) as i64], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    if let Some(exact) = ids.iter().find(|a| a.as_str() == id) {
        return Ok(Some(exact.clone()));
    }
    match ids.len() {
        0 => {
            writeln!(err, "ctx-{}: no such archive in the brain", one(id))?;
            Ok(None)
        }
        1 => Ok(ids.into_iter().next()),
        _ => {
            let listed: Vec<String> = ids.iter().take(MANY_LISTED).map(|a| format!("ctx-{}", one(a))).collect();
            writeln!(err, "ctx-{}: more than one archive starts with this: {}; nothing was changed", one(id), listed.join(", "))?;
            Ok(None)
        }
    }
}
