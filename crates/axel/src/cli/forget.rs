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

use crate::cli::show::{self, bare, find_all, now_ms, one, open, row_info, rows_where, write, RowInfo, Term, FAILED, OK, USAGE};
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

/// What `forget --now` says when the rows it would delete are not the rows it listed.
pub const CHANGED: &str = "the brain changed while you read; nothing was deleted; run it again";

/// The exit code of a brain that is busy or changed under the command: try again.
pub const AGAIN: u8 = 75;

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
    // As `show::on_std`, except that a stdout that cannot be flushed at the end (a closed
    // pipe) does not turn a purge that is done into an error (v2 c18b review, item 5).
    let (stdout, stderr, stdin) = (std::io::stdout(), std::io::stderr(), std::io::stdin());
    let (mut out, mut err, mut input) = (stdout.lock(), stderr.lock(), stdin.lock());
    let mut term = Term { out: &mut out, err: &mut err, input: &mut input };
    let code = run_with(args, &brain.path, &bases, &mut term)?;
    let _ = term.out.flush();
    Ok(ExitCode::from(code))
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
    // RULING: v2 c18b review, item 9. A row with no `memory_ext` row cannot be hidden (the
    // hide is written there): it is refused, not reported as hidden. And `--now` deletes
    // for good only rows named by their whole id.
    let mut refused = false;
    for row in rows.iter().filter(|r| r.origin == "unknown") {
        writeln!(term.err, "m:{}: not adopted (the brain has no memory_ext row for it); it cannot be hidden or deleted here", one(&row.id))?;
        refused = true;
    }
    if args.now {
        for given in &args.ids {
            if !ids.iter().any(|id| id == bare(given)) {
                writeln!(term.err, "{}: --now needs the whole id of the row (\"axel-v2 show {}\" gives it)", one(given), one(given))?;
                refused = true;
            }
        }
    }
    if refused {
        writeln!(term.err, "Nothing was changed.")?;
        return Ok(FAILED);
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

/// `m:<id> · 2026-11-10 14:05 UTC · ctx-<archive> · <first line of its reply>`.
// RULING: v2 c18b review round 2, item 4. Every row listed here has the same prompt and,
// in one archive, the same date: the first line of the reply is what tells them apart.
fn twin_line(row: &RowInfo) -> String {
    let when = row.created_ms.map(show::date).unwrap_or_else(|| "no date".to_string());
    let text = if row.secret {
        "(a secret memory; its text is not shown)".to_string()
    } else {
        screen::show(reply_line(&row.content), 120)
    };
    format!("m:{} · {when} · ctx-{} · {text}", one(&row.id), one(&row.origin_ref))
}

/// The first line of a turn's reply: the line that starts with `Assistant`; else the first
/// non-empty line after the prompt's first; else the first line.
fn reply_line(content: &str) -> &str {
    let mut lines = content.lines().filter(|l| !l.trim().is_empty());
    let first = lines.next().unwrap_or("");
    content
        .lines()
        .find(|l| l.starts_with("Assistant"))
        .or_else(|| lines.next())
        .unwrap_or(first)
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
    /// The visible archived rows hidden as `cli:unproven` (v2 c18b review round 3, item 2).
    unproven: Vec<String>,
    /// Each deleted turn (a capture or an archived row) and the Synaps archive that still
    /// holds it, when one is known.
    in_synaps: Vec<(String, Option<String>)>,
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

    // 1. Named rows that are not hidden yet are hidden now and deleted on a later day.
    let (visible, chosen): (Vec<RowInfo>, Vec<RowInfo>) = rows.into_iter().partition(|r| r.hidden_ms.is_none());
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

    // RULING: v2 c18b review round 2, item 7. Step 1 is committed: an error after it (a
    // busy brain) says what it did, not "nothing was changed".
    match delete_hidden(conn, path, hidden, chosen, hours, wait_ms, now, term) {
        Err(e) if !visible.is_empty() && e.to_string().contains(show::BUSY) => {
            let ids: Vec<String> = visible.iter().map(|r| format!("m:{}", one(&r.id))).collect();
            writeln!(
                term.err,
                "the brain is busy (the extension or another command is writing to it); nothing was deleted, but {} {} hidden first and {} hidden (\"axel-v2 restore <id>\" undoes that); try again in a moment",
                ids.join(", "),
                if ids.len() == 1 { "was" } else { "were" },
                if ids.len() == 1 { "stays" } else { "stay" }
            )?;
            Ok(AGAIN)
        }
        other => other,
    }
}

/// Steps 2 to 4 of `forget --now`: what is hidden long enough goes, once asked.
#[allow(clippy::too_many_arguments)]
fn delete_hidden(
    conn: &Connection,
    path: &Path,
    hidden: bool,
    mut chosen: Vec<RowInfo>,
    hours: u64,
    wait_ms: i64,
    now: i64,
    term: &mut Term,
) -> Result<u8> {
    let cutoff = now.saturating_sub(wait_ms);
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
                    "    not proven the same turn; left hidden (\"axel-v2 restore {id}\" brings it back; \"axel-v2 forget --now {id}\" deletes it too): {}",
                    twin_line(&twin.row),
                    id = one(&twin.row.id)
                )?;
            }
        }
    }
    if twin_count > 0 {
        writeln!(term.out, "{twin_count} of the {total} are archived rows with the same text as a hidden turn.")?;
    }
    // RULING: v2 c18b review, item 7. What the forget mark of the deleted turns takes is
    // found by doing it, in a transaction that is rolled back, and listed before the question.
    let (taken, doubt) = mark_takes(conn, &ripe, &twins)?;
    if !taken.is_empty() {
        writeln!(
            term.out,
            "The forget mark of the deleted turns also takes these {} archived copies of them out of the brain (the same prompt and the same text; each stays in its Synaps archive):",
            taken.len()
        )?;
        let mut archives: Vec<&str> = Vec::new();
        for row in &taken {
            if !archives.contains(&row.origin_ref.as_str()) {
                archives.push(&row.origin_ref);
            }
        }
        for archive in archives {
            writeln!(term.out, "  ctx-{}:", one(archive))?;
            for row in taken.iter().filter(|r| r.origin_ref == archive) {
                writeln!(term.out, "    {}", twin_line(row))?;
            }
        }
        // RULING: v2 c18b review, item 7, read with item 4: a kept archive obeys a `cli` mark
        // (item 4), so "restore --forget-turns ctx-…" does not bring these rows back while
        // the mark stands. Said as it is, not as a way back. Since round 2, item 1, the mark
        // takes only copies of the turns named, so this is true of those turns only.
        writeln!(
            term.out,
            "  No command brings these copies back into the brain while the mark stands: \"axel-v2 restore --forget-turns ctx-…\" obeys it too. Other turns with the same prompt stay."
        )?;
    }
    // RULING: v2 c18b review round 3, item 2: what has the prompt of a deleted turn, near
    // it, and is not proven that turn is listed and hidden with the purge.
    if !doubt.is_empty() {
        writeln!(
            term.out,
            "These {} archived rows have the prompt of a deleted turn but their text does not prove them that turn; they will be hidden, not deleted:",
            doubt.len()
        )?;
        for row in &doubt {
            writeln!(
                term.out,
                "    {} (\"axel-v2 restore {id}\" brings it back; \"axel-v2 forget --now {id}\" deletes it once it has been hidden {hours} h)",
                twin_line(row),
                id = one(&row.id)
            )?;
        }
    }
    // RULING: v2 c18b review round 2, item 4: the question counts all that goes.
    let hide = if doubt.is_empty() { String::new() } else { format!(" and hide {} more", doubt.len()) };
    if taken.is_empty() {
        write!(term.out, "Delete {total} memories for good{hide}? [y/N] ")?;
    } else {
        write!(term.out, "Delete {total} memories for good, take {} archived rows out of the brain{hide}? [y/N] ", taken.len())?;
    }
    term.out.flush()?;
    let mut answer = String::new();
    term.input.read_line(&mut answer)?;
    if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        writeln!(term.out, "\nNothing was deleted.")?;
        return Ok(OK);
    }

    // 4. One transaction: the purge, the forget marks, the purge mark.
    // RULING: v2 c18b review round 2, item 5. Inside it, the rows, their twins and what the
    // mark takes are read again; anything that is not what was listed rolls it back.
    let listed = Listed {
        taken: taken.iter().map(|r| r.id.clone()).collect(),
        unproven: doubt.iter().map(|r| r.id.clone()).collect(),
    };
    let purged = match write(conn, || purge(conn, &ripe, &twins, Some(&listed))) {
        Err(e) if e.to_string().contains(CHANGED) => {
            writeln!(term.err, "{CHANGED}")?;
            return Ok(AGAIN);
        }
        other => other?,
    };
    // RULING: v2 c18b review, item 5. The rows are gone once that COMMIT holds: nothing
    // after it can turn the result into an error (a retry would only find "not found").
    // The "Deleted" line comes first; every later write, the cache file and the checkpoint
    // can fail without changing the answer, which is exit 0.
    after_purge(conn, path, &purged, term);
    Ok(OK)
}

/// What is said and done after the purge's COMMIT. Nothing here can fail the command.
// RULING: v2 c18b review round 2, item 6. No output after the COMMIT uses `?`: a closed
// stdout must not skip the cleanup (the `.hnsw` file, the checkpoint), which always runs.
fn after_purge(conn: &Connection, path: &Path, purged: &Purged, term: &mut Term) {
    let _ = writeln!(term.out, "Deleted {} memories for good.", purged.deleted);
    let _ = term.out.flush();
    let hnsw_failed = if purged.shadows > 0 {
        // The old search's vector cache is keyed by `documents` rows; the next `search`
        // builds it again.
        let hnsw = path.with_extension("r8.hnsw");
        match std::fs::remove_file(&hnsw) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Some(format!("{}: {e}", hnsw.display())),
            _ => None,
        }
    } else {
        None
    };
    let log_clear = checkpoint(conn, path).unwrap_or(false);
    if let Some(failed) = hnsw_failed {
        let _ = writeln!(term.err, "could not remove {failed}; delete it by hand");
    }
    said_after(purged, log_clear, term);
}

fn said_after(purged: &Purged, log_clear: bool, term: &mut Term) {
    // RULING: v2 c18b review, item 4: a deleted turn is still in Synaps's own archive.
    for (id, archive) in &purged.in_synaps {
        let _ = match archive {
            Some(archive) => writeln!(
                term.out,
                "  m:{}: the turn is still in Synaps's archive ctx-{a}. To forget it there, ask the assistant in that project to call memory_forget with the id \"ctx-{a}\" (that forgets the whole archive, every turn in it).",
                one(id),
                a = one(archive)
            ),
            None => writeln!(
                term.out,
                "  m:{}: if Synaps archived this chat, the turn is still in that archive (none is indexed yet). To forget it there, ask the assistant to call memory_forget with its id (ctx-…).",
                one(id)
            ),
        };
    }
    if !purged.left_out.is_empty() {
        let _ = writeln!(
            term.out,
            "The forget mark took {} archived copies of the deleted turns out of the brain, as listed above.",
            purged.left_out.len()
        );
    }
    if !purged.unproven.is_empty() {
        let _ = writeln!(
            term.out,
            "{} archived rows with the prompt of a deleted turn are hidden, as listed above.",
            purged.unproven.len()
        );
    }
    if purged.back > 0 {
        // RULING: wave 4, rules for wave 5: nothing tells the vector index; say what to expect.
        let _ = writeln!(
            term.out,
            "{} memories that a deleted row had replaced are recalled again; a running extension finds them by meaning after its next resync (about a minute).",
            purged.back
        );
    }
    if !log_clear {
        let _ = writeln!(
            term.out,
            "The brain's log file still holds the deleted text: it was in use. The running extension clears it when it is next idle."
        );
    }
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

/// The Synaps archive that holds a deleted turn: an archived row's own; for a capture, the
/// archive of its proven twin, else the first archive of its chat sealed at or after it.
fn synaps_archive(conn: &Connection, row: &RowInfo, twins: &[Twin], ids: &[String]) -> Result<Option<String>> {
    if row.origin == "archive" {
        return Ok(Some(row.origin_ref.clone()));
    }
    if let Some(twin) = twins.iter().find(|t| t.proven && ids.contains(&t.row.id)) {
        return Ok(Some(twin.row.origin_ref.clone()));
    }
    let Some(at) = row.created_ms else { return Ok(None) };
    Ok(conn
        .query_row(
            "SELECT archive_id FROM archive_index
              WHERE scope_dir = ?1 AND ((?2 IS NOT NULL AND logical = ?2) OR (?3 IS NOT NULL AND session_id = ?3))
                AND cover_ms >= ?4
              ORDER BY cover_ms, rel_name LIMIT 1",
            params![row.project_key, row.logical, row.session_id, at],
            |r| crate::brain_v2::text_at(r, 0),
        )
        .optional()?
        .flatten())
}

/// The ids of what the forget marks of `purge` did when the list was made.
#[derive(Default)]
struct Listed {
    taken: Vec<String>,
    unproven: Vec<String>,
}

/// What the forget marks of `purge` do (beside the rows named and their proven twins): the
/// archived rows they take, and the visible ones they hide as `cli:unproven`. Found by
/// running `purge` itself in a transaction that is always rolled back; the rows are read
/// inside it, before the rollback.
fn mark_takes(conn: &Connection, ripe: &[RowInfo], twins: &[Vec<Twin>]) -> Result<(Vec<RowInfo>, Vec<RowInfo>)> {
    if !ripe.iter().any(|r| (r.origin == "capture" || r.origin == "archive") && r.turn_key.is_some()) {
        return Ok((Vec::new(), Vec::new()));
    }
    if let Err(e) = crate::brain_v2::begin_write(conn) {
        return Err(busy_or(e));
    }
    let done = (|| -> Result<(Vec<RowInfo>, Vec<RowInfo>)> {
        let mut before: Vec<RowInfo> = Vec::new();
        for row in ripe {
            if let Some(key) = row.turn_key.as_deref() {
                before.extend(rows_where(conn, "WHERE e.origin_kind = 'archive' AND e.turn_key = ?1 ORDER BY e.created_ms, e.vid", [key])?);
            }
        }
        let p = purge(conn, ripe, twins, None)?;
        let taken = before.into_iter().filter(|r| p.left_out.contains(&r.id)).fold(Vec::new(), |mut all: Vec<RowInfo>, r| {
            if !all.iter().any(|a| a.id == r.id) {
                all.push(r);
            }
            all
        });
        let mut hidden = Vec::new();
        for id in &p.unproven {
            hidden.extend(row_info(conn, id)?);
        }
        Ok((taken, hidden))
    })();
    if !conn.is_autocommit() {
        conn.execute_batch("ROLLBACK")?;
    }
    done.map_err(busy_or)
}

/// `show::BUSY` for a lock that did not come in time; any other error as it is.
fn busy_or(e: crate::error::AxelError) -> crate::error::AxelError {
    use rusqlite::ErrorCode::{DatabaseBusy, DatabaseLocked};
    match &e {
        crate::error::AxelError::Db(rusqlite::Error::SqliteFailure(f, _)) if matches!(f.code, DatabaseBusy | DatabaseLocked) => {
            crate::error::AxelError::Other(show::BUSY.into())
        }
        _ => e,
    }
}

/// Inside the caller's transaction. With `listed`, the ids of the rows the mark took when
/// the list was made, anything that is not as listed (a named row, its twins and whether
/// each is proven, what the mark takes) is the error `CHANGED`, and the caller rolls back.
/// Without it (the dry run), a row that changed is left as it is.
fn purge(conn: &Connection, ripe: &[RowInfo], twins: &[Vec<Twin>], listed: Option<&Listed>) -> Result<Purged> {
    let changed = || Err(crate::error::AxelError::Other(CHANGED.into()));
    let mut ids: Vec<String> = Vec::new();
    // `(logical, session_id, turn_key, created_ms, text)` of each turn deleted by its own id.
    let mut marks: Vec<(String, Option<String>, String, i64, index_archives::TurnText)> = Vec::new();
    // The twins left hidden: the forget mark below must not take them.
    let mut kept: Vec<String> = Vec::new();
    for (row, its) in ripe.iter().zip(twins) {
        if hidden_ms_of(conn, &row.id)? != Some(row.hidden_ms) {
            if listed.is_some() {
                return changed();
            }
            continue;
        }
        if listed.is_some() {
            let fresh = row_info(conn, &row.id)?;
            let now = match &fresh {
                Some(fresh) => twins_of(conn, fresh)?,
                None => return changed(),
            };
            let shape = |twins: &mut dyn Iterator<Item = &Twin>| {
                let mut shape: Vec<(String, bool)> = twins.map(|t| (t.row.id.clone(), t.proven)).collect();
                shape.sort();
                shape
            };
            let now = shape(&mut now.iter().filter(|t| !ripe.iter().any(|r| r.id == t.row.id)));
            if now != shape(&mut its.iter()) {
                return changed();
            }
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
                kept.push(twin.row.id.clone());
            }
        }
        // A hidden turn, deleted: the archive of its window must not bring it back (B3-1.7).
        if row.origin == "capture" || row.origin == "archive" {
            if let (Some(logical), Some(key), Some(at)) = (&row.logical, &row.turn_key, row.created_ms) {
                // RULING: v2 c18b review round 2, item 1: the mark keeps hashes of the turn's
                // text, read inside this transaction, and takes only that turn.
                let content: String = conn
                    .query_row("SELECT content FROM memories WHERE id = ?1", [&row.id], |r| crate::brain_v2::text_at(r, 0))?
                    .unwrap_or_default();
                let text = if row.origin == "archive" {
                    index_archives::TurnText::of_archived(&content)
                } else {
                    index_archives::TurnText::of(&content)
                };
                marks.push((logical.clone(), row.session_id.clone(), key.clone(), at, text));
            }
        }
    }
    if ids.is_empty() {
        return Ok(Purged { deleted: 0, shadows: 0, back: 0, left_out: Vec::new(), unproven: Vec::new(), in_synaps: Vec::new() });
    }
    let mut in_synaps = Vec::new();
    for (row, its) in ripe.iter().zip(twins) {
        if ids.contains(&row.id) && (row.origin == "capture" || row.origin == "archive") {
            in_synaps.push((row.id.clone(), synaps_archive(conn, row, its, &ids)?));
        }
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

    // RULING: v2 c18b review round 3, item 2: the rest of a split turn of an archived row
    // deleted here, read before it goes.
    let mut rests = Vec::new();
    for id in &ids {
        rests.extend(index_archives::rest_of_turn(conn, id)?);
    }
    let deleted = store::purge_many(conn, &ids, true, true)?;
    // RULING: v2 c18b review, item 1: a twin not proven the same turn is never purged.
    // RULING: v2 c18b review round 3, item 2. The forget mark sees the kept twins with their
    // keys (round 2 took the keys away for its length): a mark takes a row only when its
    // hash proves it, which `same_turn` then proves too, so it takes no kept twin; and it
    // must see them to hide the rest of a split turn whose first half is one.
    let mut left_out = Vec::new();
    let mut unproven = Vec::new();
    for (logical, session_id, key, at, text) in &marks {
        let did = index_archives::mark_deleted_turn(conn, logical, session_id.as_deref(), *at, Some(key.as_str()), text)?;
        left_out.extend(did.taken);
        unproven.extend(did.unproven);
    }
    index_archives::hide_unproven(conn, &rests, &left_out, &mut unproven)?;
    if left_out.iter().any(|id| kept.contains(id)) {
        return changed();
    }
    if let Some(listed) = listed {
        let sorted = |ids: &[String]| {
            let mut ids = ids.to_vec();
            ids.sort();
            ids.dedup();
            ids
        };
        if sorted(&left_out) != sorted(&listed.taken) || sorted(&unproven) != sorted(&listed.unproven) {
            return changed();
        }
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
    Ok(Purged { deleted, shadows, back, left_out, unproven, in_synaps })
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
