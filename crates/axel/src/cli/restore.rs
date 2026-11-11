//! `restore [<id>...] [--by-model --since <date>] [--forget-turns <ctx-id>]`. Commit 18b
//! writes this file.
//!
//! Every row goes back through `store::restore` (the shadow mark, the twins of a hidden
//! turn), one id at a time, in one transaction. `--forget-turns` asks the next scan to read
//! an archive again with the forget rules off for it.

use std::path::Path;
use std::process::ExitCode;

use chrono::{DateTime, NaiveDate};
use rusqlite::{params, Connection};

use crate::cli::show::{self, find_all, find_archive, one, open, write, Term, FAILED, OK, USAGE};
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::{index_archives, store};

#[derive(clap::Args)]
pub struct Args {
    /// Ids of hidden rows to bring back
    pub ids: Vec<String>,
    /// Every row a model hid or superseded (with --since)
    #[arg(long)]
    pub by_model: bool,
    /// With --by-model: since this date
    #[arg(long, value_name = "DATE")]
    pub since: Option<String>,
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
    if forms != 1 || (args.since.is_some() && !args.by_model) {
        writeln!(
            term.err,
            "give one of: ids (\"axel-v2 restore <id>...\"), \"--by-model --since <date>\", \"--forget-turns ctx-<id>\""
        )?;
        return Ok(USAGE);
    }
    let conn = open(path)?;
    if let Some(archive) = &args.forget_turns {
        return forget_turns(&conn, archive, term);
    }
    if args.by_model {
        let Some(since) = args.since.as_deref().and_then(parse_since) else {
            writeln!(term.err, "--by-model needs --since <date>, a day such as 2026-11-01")?;
            return Ok(USAGE);
        };
        return by_model(&conn, since, term);
    }

    let Some(ids) = find_all(&conn, &args.ids, term.err)? else {
        writeln!(term.err, "Nothing was changed.")?;
        return Ok(FAILED);
    };
    let mut twins = Vec::new();
    for id in &ids {
        twins.push(conn.query_row(
            "SELECT COUNT(*) FROM memory_ext WHERE hidden_by = ?1",
            [format!("twin:{id}")],
            |r| r.get::<_, i64>(0),
        )?);
    }
    let backs = write(&conn, || ids.iter().map(|id| store::restore(&conn, id)).collect::<Result<Vec<usize>>>())?;
    for ((id, n), twins) in ids.iter().zip(backs).zip(twins) {
        let id = one(id);
        if n == 0 {
            writeln!(term.out, "m:{id} was neither hidden nor superseded; nothing to undo.")?;
        } else if twins > 0 {
            writeln!(
                term.out,
                "Restored m:{id}: the turn is back as its archived form ({n} rows); the stand-in row is gone."
            )?;
        } else {
            writeln!(term.out, "Restored m:{id}.")?;
        }
    }
    Ok(OK)
}

fn ids_of(conn: &Connection, sql: &str, since: i64) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([since], |r| r.get(0))?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}

/// Every row hidden by a model since `since`, and every row superseded since then by a
/// row the assistant saved, each through `store::restore` (R3-6b).
fn by_model(conn: &Connection, since: i64, term: &mut Term) -> Result<u8> {
    let hidden = ids_of(
        conn,
        "SELECT mem_id FROM memory_ext WHERE hidden_by LIKE 'model:%' AND hidden_ms >= ?1 ORDER BY hidden_ms, mem_id",
        since,
    )?;
    let superseded = ids_of(
        conn,
        "SELECT m.id FROM memories m JOIN memory_ext e ON e.mem_id = m.superseded_by
          WHERE e.origin_kind = 'tool' AND e.created_ms >= ?1 ORDER BY e.created_ms, m.id",
        since,
    )?;
    let (back_hidden, back_superseded) = write(conn, || {
        let mut done: Vec<&String> = Vec::new();
        let mut counts = (Vec::new(), Vec::new());
        for (list, out) in [(&hidden, &mut counts.0), (&superseded, &mut counts.1)] {
            for id in list {
                // A row both hidden and superseded is restored once and counted in both.
                if done.contains(&id) || store::restore(conn, id)? > 0 {
                    out.push(id.clone());
                }
                done.push(id);
            }
        }
        Ok(counts)
    })?;
    let since_text = show::date(since);
    if !back_hidden.is_empty() {
        let listed: Vec<String> = back_hidden.iter().map(|id| format!("m:{}", one(id))).collect();
        writeln!(term.out, "Hidden by the assistant, restored: {}", listed.join(", "))?;
    }
    if !back_superseded.is_empty() {
        let listed: Vec<String> = back_superseded.iter().map(|id| format!("m:{}", one(id))).collect();
        writeln!(term.out, "Superseded by the assistant, restored: {}", listed.join(", "))?;
    }
    writeln!(
        term.out,
        "{} hidden and {} superseded memories restored (changes by the assistant since {since_text}).",
        back_hidden.len(),
        back_superseded.len()
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
