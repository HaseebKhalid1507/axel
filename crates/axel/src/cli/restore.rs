//! `restore [<id>...] [--by-model --since <date> [--yes]] [--forget-turns <ctx-id>]`.
//! Commit 18b writes this file.
//!
//! A row goes back through `store::restore_hide` (the shadow mark, the twins of a hidden
//! turn) or `store::restore_supersede`, one change at a time, in one transaction.
//! `--forget-turns` asks the next scan to read an archive again with the forget rules off
//! for it.

use std::path::Path;
use std::process::ExitCode;

use chrono::{DateTime, NaiveDate};
use rusqlite::{params, Connection};

use crate::cli::show::{self, find_all, find_archive, one, open, row_info, rows_where, write, RowInfo, Term, FAILED, OK, USAGE};
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::{index_archives, screen, store};

#[derive(clap::Args)]
pub struct Args {
    /// Ids of hidden rows to bring back
    pub ids: Vec<String>,
    /// Every row a model hid or superseded (with --since), in every project
    #[arg(long)]
    pub by_model: bool,
    /// With --by-model: since this date, a day (2026-11-01, read as midnight UTC) or an
    /// RFC 3339 time
    #[arg(long, value_name = "DATE")]
    pub since: Option<String>,
    /// With --by-model: do not ask before restoring
    #[arg(long)]
    pub yes: bool,
    /// Index again the turns of this archive that a forget tool left out
    #[arg(long, value_name = "CTX-ID")]
    pub forget_turns: Option<String>,
}

/// The words that `restore --forget-turns` prints before it writes (4.1, risk 3).
pub const FORGET_TURNS_WARNING: &str = "the forgotten one among them";

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    show::on_std(|term| run_with(args, &brain.path, term))
}

/// `2026-11-01` (midnight UTC) or an RFC 3339 time.
// SPEC: 4135 does not give the form of `<date>`. A day is read as midnight UTC, the zone
// every date this command line prints is in.
pub fn parse_since(given: &str) -> Option<i64> {
    let given = given.trim();
    if let Ok(day) = NaiveDate::parse_from_str(given, "%Y-%m-%d") {
        return day.and_hms_opt(0, 0, 0).map(|t| t.and_utc().timestamp_millis());
    }
    DateTime::parse_from_rfc3339(given).ok().map(|t| t.timestamp_millis())
}

pub fn run_with(args: &Args, path: &Path, term: &mut Term) -> Result<u8> {
    let forms = usize::from(!args.ids.is_empty()) + usize::from(args.by_model) + usize::from(args.forget_turns.is_some());
    if forms != 1 || (args.since.is_some() && !args.by_model) || (args.yes && !args.by_model) {
        writeln!(
            term.err,
            "give one of: ids (\"axel-v2 restore <id>...\"), \"--by-model --since <date> [--yes]\", \"--forget-turns ctx-<id>\""
        )?;
        return Ok(USAGE);
    }
    let conn = open(path)?;
    if let Some(archive) = &args.forget_turns {
        return forget_turns(&conn, archive, term);
    }
    if args.by_model {
        let Some(since) = args.since.as_deref().and_then(parse_since) else {
            writeln!(term.err, "--by-model needs --since <date>, a day such as 2026-11-01 (UTC)")?;
            return Ok(USAGE);
        };
        return by_model(&conn, since, args.yes, term);
    }

    let Some(ids) = find_all(&conn, &args.ids, term.err)? else {
        writeln!(term.err, "Nothing was changed.")?;
        return Ok(FAILED);
    };
    let mut before = Vec::new();
    for id in &ids {
        let twins = conn.query_row(
            "SELECT COUNT(*) FROM memory_ext WHERE hidden_by = ?1",
            [format!("twin:{id}")],
            |r| r.get::<_, i64>(0),
        )?;
        let hidden = row_info(&conn, id)?.is_some_and(|row| row.hidden_ms.is_some());
        before.push((twins, hidden));
    }
    // RULING: v2 c18b review, item 6. One change per call: a hidden row gets its hide back
    // and keeps a supersede (said below); a row that is only superseded gets that undone.
    let backs = write(&conn, || {
        let mut backs = Vec::new();
        for (id, (_, hidden)) in ids.iter().zip(&before) {
            backs.push(if *hidden { store::restore_hide(&conn, id)? } else { store::restore_supersede(&conn, id)? });
        }
        Ok(backs)
    })?;
    for ((full, n), (twins, _)) in ids.iter().zip(backs).zip(before) {
        let id = one(full);
        let now = row_info(&conn, full)?;
        if n == 0 {
            writeln!(term.out, "m:{id} was neither hidden nor superseded; nothing to undo.")?;
        } else if twins > 0 && now.is_none() {
            writeln!(
                term.out,
                "Restored m:{id}: the turn is back as its archived row; the stand-in row is gone ({n} archived rows with its prompt are visible again)."
            )?;
        } else if twins > 0 {
            // RULING: v2 c18b review, item 1: no archived row proved the same turn.
            writeln!(
                term.out,
                "Restored m:{id}, and the {twins} archived rows with its prompt hidden with it (none has its text, so it stays)."
            )?;
        } else {
            writeln!(term.out, "Restored m:{id}.")?;
        }
        if let Some(by) = now.as_ref().and_then(|row| row.superseded_by.as_deref()) {
            writeln!(
                term.out,
                "  m:{id} is still superseded by m:{}; \"axel-v2 restore {id}\" again undoes that too.",
                one(by)
            )?;
        }
    }
    Ok(OK)
}

/// `m:<id> · <first line of its text>`.
fn brief(row: &RowInfo) -> String {
    let text = if row.secret {
        "(a secret memory; its text is not shown)".to_string()
    } else {
        let first = format!("{}\n{}", row.title, row.content);
        screen::show(first.lines().find(|l| !l.trim().is_empty()).unwrap_or(""), 120)
    };
    format!("m:{} · {text}", one(&row.id))
}

/// Every row hidden by a model since `since`, and every row superseded since then by a
/// row the assistant saved (R3-6b), in every project: listed, asked, then undone.
// RULING: v2 c18b review, item 6. A superseded row gets only its supersede undone; its hide
// too only when a model hid it (`model:…`). A hide by the person or a pause stays.
fn by_model(conn: &Connection, since: i64, yes: bool, term: &mut Term) -> Result<u8> {
    let hidden = rows_where(conn, "WHERE e.hidden_by LIKE 'model:%' AND e.hidden_ms >= ?1 ORDER BY e.hidden_ms, m.id", [since])?;
    let superseded = rows_where(
        conn,
        "WHERE m.superseded_by IN (SELECT mem_id FROM memory_ext WHERE origin_kind = 'tool' AND created_ms >= ?1)
          ORDER BY m.id",
        [since],
    )?;
    let since_text = show::date(since);
    if hidden.is_empty() && superseded.is_empty() {
        writeln!(term.out, "0 hidden and 0 superseded memories restored (no change by the assistant since {since_text}).")?;
        return Ok(OK);
    }
    writeln!(term.out, "Changes by the assistant since {since_text}, in every project:")?;
    for row in &hidden {
        let when = row.hidden_ms.map(show::date).unwrap_or_default();
        writeln!(term.out, "  hidden by {} on {when}: {}", one(row.hidden_by.as_deref().unwrap_or("?")), brief(row))?;
    }
    for row in &superseded {
        writeln!(term.out, "  superseded by m:{}: {}", one(row.superseded_by.as_deref().unwrap_or("?")), brief(row))?;
    }
    if !yes {
        write!(term.out, "Undo these {} changes? [y/N] ", hidden.len() + superseded.len())?;
        term.out.flush()?;
        let mut answer = String::new();
        term.input.read_line(&mut answer)?;
        if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            writeln!(term.out, "\nNothing was changed.")?;
            return Ok(OK);
        }
    }

    // `(id, what was undone)`, in the order listed.
    let (lines, back_hidden, back_superseded) = write(conn, || {
        let mut lines: Vec<String> = Vec::new();
        let (mut back_hidden, mut back_superseded) = (0usize, 0usize);
        for row in &hidden {
            if store::restore_hide(conn, &row.id)? == 0 {
                continue;
            }
            back_hidden += 1;
            let mut line = format!("m:{}: no longer hidden", one(&row.id));
            if let Some(by) = superseded.iter().find(|s| s.id == row.id).and_then(|s| s.superseded_by.as_deref()) {
                if store::restore_supersede(conn, &row.id)? > 0 {
                    back_superseded += 1;
                    line.push_str(&format!("; also no longer superseded by m:{}", one(by)));
                }
            }
            lines.push(line);
        }
        for row in superseded.iter().filter(|s| !hidden.iter().any(|h| h.id == s.id)) {
            if store::restore_supersede(conn, &row.id)? == 0 {
                continue;
            }
            back_superseded += 1;
            let by = one(row.superseded_by.as_deref().unwrap_or("?"));
            let mut line = format!("m:{}: no longer superseded by m:{by}", one(&row.id));
            match row.hidden_by.as_deref() {
                Some(model) if model.starts_with("model:") => {
                    if store::restore_hide(conn, &row.id)? > 0 {
                        back_hidden += 1;
                        line.push_str(&format!("; also no longer hidden (by {})", one(model)));
                    }
                }
                Some(other) if row.hidden_ms.is_some() => {
                    line.push_str(&format!("; still hidden (by {}: \"axel-v2 restore {}\" undoes that)", one(other), one(&row.id)));
                }
                _ => {}
            }
            lines.push(line);
        }
        Ok((lines, back_hidden, back_superseded))
    })?;
    for line in &lines {
        writeln!(term.out, "  {line}")?;
    }
    writeln!(
        term.out,
        "{back_hidden} hidden and {back_superseded} superseded memories restored (changes by the assistant since {since_text})."
    )?;
    Ok(OK)
}

/// `--forget-turns ctx-<id>` (R3-5): per archive, all or nothing.
fn forget_turns(conn: &Connection, given: &str, term: &mut Term) -> Result<u8> {
    let Some(archive_id) = find_archive(conn, given, term.err)? else {
        writeln!(term.err, "Nothing was changed.")?;
        return Ok(FAILED);
    };
    let id = one(&archive_id);
    let units: i64 = conn.query_row(
        "SELECT COALESCE(SUM(forget_units), 0) FROM archive_index WHERE archive_id = ?1",
        [&archive_id],
        |r| r.get(0),
    )?;
    if units == 0 {
        writeln!(term.out, "ctx-{id} leaves no turn out; nothing changed.")?;
        return Ok(OK);
    }
    writeln!(term.out, "this brings back every turn of ctx-{id}, {FORGET_TURNS_WARNING}")?;
    write(conn, || {
        conn.execute("UPDATE archive_index SET keep_forget_turns = 1 WHERE archive_id = ?1", [&archive_id])?;
        conn.execute("DELETE FROM forgotten_turns WHERE archive_id = ?1", [&archive_id])?;
        Ok(())
    })?;
    writeln!(
        term.out,
        "ctx-{id}: {units} turns left out will be indexed by the next scan of the running extension. \
         What comes back is not left out again by a later forget."
    )?;

    // A mark reaches one window each way: the archives of the chat near this one that
    // still leave turns out.
    let mut near: Vec<(String, i64)> = Vec::new();
    let files: Vec<(String, Option<String>, Option<String>, i64)> = {
        let mut stmt = conn.prepare(
            "SELECT scope_dir, logical, session_id, cover_ms FROM archive_index
              WHERE archive_id = ?1 AND cover_ms IS NOT NULL",
        )?;
        let rows = stmt.query_map([&archive_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    for (scope_dir, logical, session_id, cover_ms) in &files {
        let logical = logical.as_deref().unwrap_or("");
        let (start, end) = index_archives::near_stretch(conn, scope_dir, logical, session_id.as_deref(), *cover_ms);
        let mut stmt = conn.prepare(
            "SELECT archive_id, forget_units FROM archive_index
              WHERE scope_dir = ?1 AND ((?2 <> '' AND logical = ?2) OR (?3 IS NOT NULL AND session_id = ?3))
                AND archive_id <> ?4 AND forget_units > 0 AND keep_forget_turns = 0
                AND cover_ms > ?5 AND cover_ms <= ?6
              ORDER BY cover_ms",
        )?;
        let rows = stmt.query_map(params![scope_dir, logical, session_id, archive_id, start, end], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?;
        for row in rows {
            let row = row?;
            if !near.iter().any(|(a, _)| *a == row.0) {
                near.push(row);
            }
        }
    }
    if near.is_empty() {
        writeln!(term.out, "No neighbouring archive of this chat leaves turns out.")?;
    } else {
        writeln!(term.out, "Neighbouring archives of this chat that still leave turns out:")?;
        for (other, n) in &near {
            let other = one(other);
            writeln!(term.out, "  ctx-{other}: {n} turns (\"axel-v2 restore --forget-turns ctx-{other}\")")?;
        }
    }
    Ok(OK)
}
