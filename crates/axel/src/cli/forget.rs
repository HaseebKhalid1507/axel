//! `forget [<id>...] [--now] [--hidden]`. Commit 18b writes this file.
//!
//! `forget <id>...` hides, as the tool does, with `hidden_by = 'cli'` and no cap: nothing
//! is deleted. `forget --now <id>...` and `forget --now --hidden` delete for good, and only
//! rows that have been hidden for `forget_now_min_hidden_hours` (24, or more). This is the
//! one place text leaves the brain on a person's word: the rows, their word rows, their
//! shadow documents and graph nodes (`store::purge_many`), then the write-ahead log
//! (a `TRUNCATE` checkpoint) and the old search's `<brain>.r8.hnsw`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rusqlite::{params, Connection, OptionalExtension};

use crate::cli::show::{self, find_all, now_ms, one, open, row_info, rows_where, write, RowInfo, Term, FAILED, OK, USAGE};
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::{index_archives, screen, store};

#[derive(clap::Args)]
pub struct Args {
    /// Memory ID (e.g. mem_abc12345)
    pub ids: Vec<String>,
    /// Purge rows that have been hidden for at least 24 hours
    #[arg(long)]
    pub now: bool,
    /// With --now: the rows hidden by a model or from the command line
    #[arg(long)]
    pub hidden: bool,
}

/// The line of `<base>/config` that sets how long a row must have been hidden.
pub const MIN_HIDDEN_KEY: &str = "extension.axel.forget_now_min_hidden_hours";
/// A missing, unreadable or malformed value is this (R3-6d).
pub const DEFAULT_MIN_HIDDEN_HOURS: u64 = 24;

const HOUR_MS: i64 = 60 * 60 * 1000;

/// The Synaps bases whose `config` may set the wait, beside the one the brain records.
#[derive(Clone, Debug, Default)]
pub struct Bases {
    /// `$SYNAPS_BASE_DIR`, when it is set.
    pub env: Option<PathBuf>,
    /// `~/.synaps-cli`.
    pub home: Option<PathBuf>,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let bases = Bases {
        env: std::env::var_os("SYNAPS_BASE_DIR").filter(|v| !v.is_empty()).map(PathBuf::from),
        home: dirs::home_dir().map(|home| home.join(".synaps-cli")),
    };
    show::on_std(|term| run_with(args, &brain.path, &bases, term))
}

pub fn run_with(args: &Args, path: &Path, bases: &Bases, term: &mut Term) -> Result<u8> {
    if args.hidden && !args.now {
        writeln!(term.err, "--hidden goes with --now: \"axel-v2 forget --now --hidden\"")?;
        return Ok(USAGE);
    }
    if args.ids.is_empty() && !args.hidden {
        writeln!(term.err, "name the rows to forget: \"axel-v2 forget <id>...\" (or \"axel-v2 forget --now --hidden\")")?;
        return Ok(USAGE);
    }
    let conn = open(path)?;
    let Some(ids) = find_all(&conn, &args.ids, term.err)? else {
        writeln!(term.err, "Nothing was changed.")?;
        return Ok(FAILED);
    };
    let mut rows = Vec::new();
    for id in &ids {
        match row_info(&conn, id)? {
            Some(row) => rows.push(row),
            None => {
                writeln!(term.err, "m:{}: not found\nNothing was changed.", one(id))?;
                return Ok(FAILED);
            }
        }
    }
    // As the tool does: a row that mirrors a Synaps file is forgotten in Synaps. A hidden
    // one (the archived form of a hidden turn) is taken by `--now`.
    let mirrors: Vec<&RowInfo> =
        rows.iter().filter(|r| r.hidden_ms.is_none() && (r.origin == "archive" || r.origin == "note_file")).collect();
    if !mirrors.is_empty() {
        for row in mirrors {
            writeln!(
                term.err,
                "m:{}: this mirrors a Synaps archive or note; forget it in Synaps with memory_forget (ctx-… or mem-…)",
                one(&row.id)
            )?;
        }
        writeln!(term.err, "Nothing was changed.")?;
        return Ok(FAILED);
    }
    if !args.now {
        return hide_all(&conn, &rows, term);
    }
    forget_now(&conn, path, bases, args.hidden, rows, term)
}

/// `forget <id>...`: one transaction, every id or none.
fn hide_all(conn: &Connection, rows: &[RowInfo], term: &mut Term) -> Result<u8> {
    let counts = write(conn, || {
        let mut counts = Vec::new();
        for row in rows {
            counts.push(store::hide(conn, &row.id, "cli")?);
        }
        Ok(counts)
    })?;
    for (row, n) in rows.iter().zip(counts) {
        let id = one(&row.id);
        if n == 0 {
            writeln!(term.out, "m:{id} was hidden already ({}); nothing changed.", show::state(row))?;
            continue;
        }
        writeln!(
            term.out,
            "Hidden m:{id}. It is no longer recalled. Nothing was deleted: restore it with \"axel-v2 restore {id}\"."
        )?;
        // RULING: v2 c18b review, item 1. What was hidden with it is shown, with its date
        // and first line: rows with the same prompt are not always the same turn.
        let twins = twins_of(conn, row)?;
        if !twins.is_empty() {
            writeln!(
                term.out,
                "  Hidden with it, archived rows of the chat with the same prompt (\"axel-v2 restore {id}\" brings them back too):"
            )?;
            for twin in &twins {
                writeln!(term.out, "    {}", twin_line(&twin.row))?;
            }
        }
    }
    Ok(OK)
}

/// A row hidden as the twin of a capture, and whether its text proves it the same turn.
struct Twin {
    row: RowInfo,
    proven: bool,
}

/// The rows hidden as `twin:<row>`, oldest first.
fn twins_of(conn: &Connection, row: &RowInfo) -> Result<Vec<Twin>> {
    let rows = rows_where(conn, "WHERE e.hidden_by = ?1 ORDER BY e.vid", [format!("twin:{}", row.id)])?;
    Ok(rows
        .into_iter()
        .map(|twin| Twin { proven: row.origin == "capture" && store::same_turn(&row.content, &twin.content), row: twin })
        .collect())
}

/// `m:<id> · 2026-11-10 14:05 UTC · ctx-<archive> · <first line of its text>`.
fn twin_line(row: &RowInfo) -> String {
    let when = row.created_ms.map(show::date).unwrap_or_else(|| "no date".to_string());
    let text = if row.secret {
        "(a secret memory; its text is not shown)".to_string()
    } else {
        screen::show(row.content.lines().find(|l| !l.trim().is_empty()).unwrap_or(""), 120)
    };
    format!("m:{} · {when} · ctx-{} · {text}", one(&row.id), one(&row.origin_ref))
}

// ── --now ───────────────────────────────────────────────────────────────────

/// The base the brain records (the extension writes it), if any. The command line gets no
/// `initialize` message.
pub fn recorded_base(conn: &Connection) -> Result<Option<PathBuf>> {
    let recorded: Option<String> = conn
        .query_row("SELECT value FROM context_data WHERE layer = 'axel' AND key = 'synaps_base'", [], |r| {
            r.get::<_, Option<String>>(0)
        })
        .optional()?
        .flatten()
        .filter(|v| !v.trim().is_empty());
    Ok(recorded.map(PathBuf::from))
}

/// The wait in hours: the largest of the default and of every whole number of hours set in
/// the `config` of `$SYNAPS_BASE_DIR`, of the base the brain records and of `~/.synaps-cli`.
// RULING: v2 c18b review, item 3. The environment may only lengthen the wait: a shell that
// points `SYNAPS_BASE_DIR` at a folder of its own cannot shorten it, and a config that is
// missing, unreadable or malformed contributes nothing. No value goes below the default.
pub fn min_hidden_hours(conn: &Connection, bases: &Bases) -> Result<u64> {
    let mut all: Vec<PathBuf> = Vec::new();
    all.extend(bases.env.clone());
    all.extend(recorded_base(conn)?);
    all.extend(bases.home.clone());
    Ok(all
        .iter()
        .filter_map(|base| crate::ext::config::base_config_value(base, MIN_HIDDEN_KEY))
        .filter_map(|v| v.trim().parse::<u64>().ok())
        .fold(DEFAULT_MIN_HIDDEN_HOURS, u64::max))
}

/// What the purge did, for the lines printed after it.
struct Purged {
    deleted: usize,
    shadows: i64,
    back: usize,
    left_out: Vec<String>,
    changed: Vec<String>,
}

fn forget_now(
    conn: &Connection,
    path: &Path,
    bases: &Bases,
    hidden: bool,
    rows: Vec<RowInfo>,
    term: &mut Term,
) -> Result<u8> {
    let hours = min_hidden_hours(conn, bases)?;
    let wait_ms = i64::try_from(hours.saturating_mul(HOUR_MS as u64)).unwrap_or(i64::MAX);
    let now = now_ms();
    let cutoff = now.saturating_sub(wait_ms);

    // 1. Named rows that are not hidden yet are hidden now and deleted on a later day.
    let (visible, mut chosen): (Vec<RowInfo>, Vec<RowInfo>) = rows.into_iter().partition(|r| r.hidden_ms.is_none());
    if !visible.is_empty() {
        write(conn, || {
            for row in &visible {
                store::hide(conn, &row.id, "cli")?;
            }
            Ok(())
        })?;
        for row in &visible {
            writeln!(
                term.out,
                "Hidden m:{}; nothing of it is deleted today. It can be deleted for good after {}.",
                one(&row.id),
                show::date(now.saturating_add(wait_ms))
            )?;
        }
    }

    // 2. What is hidden: the named rows (a pause row is deleted by id only), and with
    // `--hidden` every row hidden by a model or from the command line.
    if hidden {
        let more = rows_where(
            conn,
            "WHERE e.hidden_ms IS NOT NULL AND (e.hidden_by = 'cli' OR e.hidden_by LIKE 'model:%') ORDER BY e.hidden_ms, m.id",
            [],
        )?;
        for row in more {
            if !chosen.iter().any(|c| c.id == row.id) {
                chosen.push(row);
            }
        }
    }
    let (ripe, young): (Vec<RowInfo>, Vec<RowInfo>) =
        chosen.into_iter().partition(|r| r.hidden_ms.is_some_and(|ms| ms <= cutoff));
    if !young.is_empty() {
        writeln!(term.out, "Hidden too recently to be deleted (the wait is {hours} h):")?;
        for row in &young {
            let at = row.hidden_ms.unwrap_or(now);
            writeln!(
                term.out,
                "  m:{}: hidden {} h ago; can be deleted after {}",
                one(&row.id),
                (now - at).max(0) / HOUR_MS,
                show::date(at.saturating_add(wait_ms))
            )?;
        }
    }
    if ripe.is_empty() {
        writeln!(term.out, "Nothing can be deleted now.")?;
        return Ok(OK);
    }

    // 3. The rows hidden with each hidden capture: those proven the same turn go with it,
    // every other one stays hidden (v2 c18b review, item 1).
    let mut twins: Vec<Vec<Twin>> = Vec::new();
    for row in &ripe {
        let of = twins_of(conn, row)?;
        twins.push(of.into_iter().filter(|t| !ripe.iter().any(|r| r.id == t.row.id)).collect());
    }
    let twin_count = twins.iter().flatten().filter(|t| t.proven).count();
    let total = ripe.len() + twin_count;

    writeln!(term.out, "These will be deleted for good:")?;
    for (row, its) in ripe.iter().zip(&twins) {
        print_row(term.out, row, now)?;
        for twin in its {
            if twin.proven {
                writeln!(term.out, "    with it, its archived row (the same text): {}", twin_line(&twin.row))?;
            } else {
                writeln!(
                    term.out,
                    "    not proven the same turn; left hidden (\"axel-v2 restore {}\" brings it back): {}",
                    one(&twin.row.id),
                    twin_line(&twin.row)
                )?;
            }
        }
    }
    if twin_count > 0 {
        writeln!(term.out, "{twin_count} of the {total} are archived rows with the same text as a hidden turn.")?;
    }
    write!(term.out, "Delete {total} memories for good? [y/N] ")?;
    term.out.flush()?;
    let mut answer = String::new();
    term.input.read_line(&mut answer)?;
    if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        writeln!(term.out, "\nNothing was deleted.")?;
        return Ok(OK);
    }

    // 4. One transaction: the purge, the forget marks, the purge mark.
    let purged = write(conn, || purge(conn, &ripe, &twins))?;
    if purged.shadows > 0 {
        // The old search's vector cache is keyed by `documents` rows; the next `search`
        // builds it again.
        let hnsw = path.with_extension("r8.hnsw");
        if let Err(e) = std::fs::remove_file(&hnsw) {
            if e.kind() != std::io::ErrorKind::NotFound {
                writeln!(term.err, "could not remove {}: {e}; delete it by hand", hnsw.display())?;
            }
        }
    }
    let log_clear = checkpoint(conn, path)?;

    writeln!(term.out, "Deleted {} memories for good.", purged.deleted)?;
    for id in &purged.changed {
        writeln!(term.out, "  m:{}: changed since it was listed; left as it is.", one(id))?;
    }
    if !purged.left_out.is_empty() {
        writeln!(
            term.out,
            "The same turns were also taken out of {} more archived rows of the chat.",
            purged.left_out.len()
        )?;
    }
    if purged.back > 0 {
        // RULING: wave 4, rules for wave 5: nothing tells the vector index; say what to expect.
        writeln!(
            term.out,
            "{} memories that a deleted row had replaced are recalled again; a running extension finds them by meaning after its next resync (about a minute).",
            purged.back
        )?;
    }
    if !log_clear {
        writeln!(
            term.out,
            "The brain's log file still holds the deleted text: it was in use. The running extension clears it when it is next idle."
        )?;
    }
    Ok(OK)
}

fn print_row(out: &mut dyn Write, row: &RowInfo, now: i64) -> Result<()> {
    let ago = row.hidden_ms.map(|ms| (now - ms).max(0) / HOUR_MS).unwrap_or(0);
    writeln!(
        out,
        "  m:{} · {} · {} · hidden by {} {ago} h ago",
        one(&row.id),
        one(&row.origin),
        one(&row.kind),
        one(row.hidden_by.as_deref().unwrap_or("?"))
    )?;
    if row.secret {
        writeln!(out, "    (a secret memory; its text is not shown)")?;
        return Ok(());
    }
    for line in screen::show_block(&format!("{}\n{}", row.title, row.content), 400).lines() {
        writeln!(out, "    {line}")?;
    }
    Ok(())
}

fn hidden_ms_of(conn: &Connection, id: &str) -> Result<Option<Option<i64>>> {
    Ok(conn.query_row("SELECT hidden_ms FROM memory_ext WHERE mem_id = ?1", [id], |r| r.get(0)).optional()?)
}

/// Inside the caller's transaction. A row that changed since it was listed (restored,
/// gone) is left as it is.
fn purge(conn: &Connection, ripe: &[RowInfo], twins: &[Vec<Twin>]) -> Result<Purged> {
    let mut ids: Vec<String> = Vec::new();
    let mut changed = Vec::new();
    // `(logical, session_id, turn_key, created_ms)` of each turn deleted by its own id.
    let mut marks: Vec<(String, Option<String>, String, i64)> = Vec::new();
    // The twins left hidden, with their key: the forget mark below must not take them.
    let mut kept: Vec<(String, Option<String>)> = Vec::new();
    for (row, its) in ripe.iter().zip(twins) {
        if hidden_ms_of(conn, &row.id)? != Some(row.hidden_ms) {
            changed.push(row.id.clone());
            continue;
        }
        ids.push(row.id.clone());
        for twin in its {
            let still: Option<Option<String>> = conn
                .query_row("SELECT hidden_by FROM memory_ext WHERE mem_id = ?1", [&twin.row.id], |r| r.get(0))
                .optional()?;
            if still.flatten().as_deref() != Some(format!("twin:{}", row.id).as_str()) || ids.contains(&twin.row.id) {
                continue;
            }
            // Proven again inside the transaction, on the rows as they are now.
            if row.origin == "capture" && store::proven_twin(conn, &row.id, &twin.row.id)? {
                ids.push(twin.row.id.clone());
            } else {
                kept.push((twin.row.id.clone(), twin.row.turn_key.clone()));
            }
        }
        // A hidden turn, deleted: the archive of its window must not bring it back (B3-1.7).
        if row.origin == "capture" || row.origin == "archive" {
            if let (Some(logical), Some(key), Some(at)) = (&row.logical, &row.turn_key, row.created_ms) {
                marks.push((logical.clone(), row.session_id.clone(), key.clone(), at));
            }
        }
    }
    if ids.is_empty() {
        return Ok(Purged { deleted: 0, shadows: 0, back: 0, left_out: Vec::new(), changed });
    }
    let mut shadows = 0i64;
    let mut replaced: Vec<String> = Vec::new();
    for id in &ids {
        shadows += conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE doc_id = ?1
                AND CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.type') END = 'memory'",
            [id],
            |r| r.get::<_, i64>(0),
        )?;
        let mut stmt = conn.prepare("SELECT id FROM memories WHERE superseded_by = ?1")?;
        for old in stmt.query_map([id], |r| r.get::<_, String>(0))? {
            let old = old?;
            if !ids.contains(&old) && !replaced.contains(&old) {
                replaced.push(old);
            }
        }
    }

    let deleted = store::purge_many(conn, &ids, true, true)?;
    // RULING: v2 c18b review, item 1: a twin not proven the same turn is never purged. The
    // forget mark takes the chat's archived rows by key, hidden ones too, so for the length
    // of the calls below the kept twins have no key; it is put back before the commit.
    for (id, _) in &kept {
        conn.execute("UPDATE memory_ext SET turn_key = NULL WHERE mem_id = ?1", [id])?;
    }
    let mut left_out = Vec::new();
    for (logical, session_id, key, at) in &marks {
        left_out.extend(index_archives::mark_forgotten_turn(
            conn,
            logical,
            session_id.as_deref(),
            "cli",
            *at,
            *at,
            Some(key.as_str()),
        )?);
    }
    for (id, key) in &kept {
        conn.execute("UPDATE memory_ext SET turn_key = ?2 WHERE mem_id = ?1", params![id, key])?;
    }
    // RULING: wave 4, rules for wave 5: every caller that forgets text writes the purge
    // mark in its own transaction. `checkpoint` below takes it away again when it could
    // clear the log itself.
    index_archives::mark_purged(conn)?;

    let mut back = 0;
    for old in &replaced {
        let superseded: Option<Option<String>> = conn
            .query_row("SELECT superseded_by FROM memories WHERE id = ?1", [old], |r| r.get(0))
            .optional()?;
        if superseded == Some(None) {
            back += 1;
        }
    }
    Ok(Purged { deleted, shadows, back, left_out, changed })
}

fn wal_len(path: &Path) -> u64 {
    let mut wal = path.as_os_str().to_owned();
    wal.push("-wal");
    std::fs::metadata(PathBuf::from(wal)).map(|m| m.len()).unwrap_or(0)
}

/// The write-ahead log still holds the frames written when the deleted rows were
/// inserted: `wal_checkpoint(TRUNCATE)`, as the indexer's idle turn does (2.7). When it was
/// not blocked and nobody has written since, the purge mark goes, and a second checkpoint
/// takes that small frame out too. Blocked: the mark stays and the running extension's
/// indexer finishes. True when the log was cleared here. Never deletes a `-wal` file.
fn checkpoint(conn: &Connection, path: &Path) -> Result<bool> {
    let busy: i64 = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0))?;
    if busy != 0 {
        return Ok(false);
    }
    // Under the write lock, an empty log means no writer came between the checkpoint and
    // now: whatever the mark stands for has been checkpointed.
    let cleared = write(conn, || {
        if wal_len(path) != 0 {
            return Ok(false);
        }
        conn.execute("DELETE FROM context_data WHERE key = 'purge_pending' AND layer = 'axel'", [])?;
        Ok(true)
    })?;
    if cleared {
        let _ = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get::<_, i64>(0));
    }
    Ok(true)
}
