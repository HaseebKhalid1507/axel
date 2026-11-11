//! `recall [<query>...]`. Commit 18b writes this file.
//!
//! With no query: the handoff, then the newest saved memories (origin `tool`, `cli` or
//! `legacy`; not hidden, not superseded, not expired). With a query: the old search, as
//! before, without hidden hits (as `search`). An agent reads this output (`jawz-axel boot`
//! feeds it into a context), so every body passes `screen::show` and is ONE line: a stored
//! line break followed by text that looks like the next item's label cannot start a line
//! of its own (4.3, W2-8.3).

use std::path::Path;
use std::process::ExitCode;

use chrono::DateTime;
use rusqlite::OptionalExtension;

use crate::cli::show::{self, one, open, rows_where, Term, FAILED, OK};
use crate::cli::strip_frontmatter;
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::screen;
use crate::search::BrainSearch;

#[derive(clap::Args)]
pub struct Args {
    /// Optional search query (omit for boot context)
    pub query: Vec<String>,
}

/// How many saved memories the boot context lists.
const SAVED_LISTED: usize = 5;

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    show::on_std(|term| {
        if args.query.is_empty() {
            return boot(&brain.path, term);
        }
        let mut search = BrainSearch::open(&brain.path)?;
        query(args, &mut search, term)
    })
}

/// Who saved a row, in the words recall uses.
fn saved_by(origin: &str) -> &'static str {
    match origin {
        "tool" => "saved by the assistant",
        "cli" => "saved with the axel command",
        _ => "an old memory",
    }
}

pub fn boot(path: &Path, term: &mut Term) -> Result<u8> {
    let conn = open(path)?;
    let mut has_content = false;

    // The handoff's one home, read as `MemoryStorage::get_context("handoff")` reads it.
    let handoff: Option<String> = conn
        .query_row(
            "SELECT value FROM context_data WHERE key = 'handoff'
              ORDER BY CASE layer WHEN 'session' THEN 0 WHEN 'boot' THEN 1 WHEN 'archive' THEN 2 ELSE 3 END
              LIMIT 1",
            [],
            |r| crate::brain_v2::text_at(r, 0),
        )
        .optional()?
        .flatten();
    if let Some(handoff) = handoff.filter(|h| !h.trim().is_empty()) {
        writeln!(term.out, "[Handoff]\n{}\n", screen::show(&handoff, 2_000))?;
        has_content = true;
    }

    let now = show::now_ms();
    let candidates = rows_where(
        &conn,
        "WHERE e.origin_kind IN ('tool', 'cli', 'legacy') AND e.hidden_ms IS NULL
           AND m.superseded_by IS NULL AND m.invalidated_at IS NULL
         ORDER BY e.created_ms DESC, m.id LIMIT 200",
        [],
    )?;
    let saved: Vec<_> = candidates
        .into_iter()
        .filter(|r| {
            r.expires_at
                .as_deref()
                .and_then(|t| DateTime::parse_from_rfc3339(t).ok())
                .is_none_or(|t| t.timestamp_millis() > now)
        })
        .take(SAVED_LISTED)
        .collect();
    if !saved.is_empty() {
        writeln!(term.out, "[Saved Memories]")?;
        for row in &saved {
            let day = row.created_ms.map(show::day).unwrap_or_default();
            let body = if row.secret {
                format!("{} (a secret memory; its text is not shown)", row.title)
            } else {
                let text = if row.abstract_text.trim().is_empty() { &row.content } else { &row.abstract_text };
                format!("{}: {}", row.title, text)
            };
            writeln!(
                term.out,
                "  • [{} · {} · m:{} · {day}] {}",
                one(&row.kind),
                saved_by(&row.origin),
                one(&row.id),
                screen::show(&body, 400)
            )?;
        }
        has_content = true;
    }

    if !has_content {
        writeln!(term.err, "Brain is empty. Use `axel remember` or `axel index` to add content.")?;
    }
    let (total, saved_total): (i64, i64) = conn.query_row(
        "SELECT COUNT(*), COALESCE(SUM(origin_kind IN ('tool', 'cli', 'legacy') AND hidden_ms IS NULL), 0) FROM memory_ext",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    writeln!(term.out, "\n[Stats] {total} rows, {saved_total} saved memories")?;
    Ok(OK)
}

pub fn query(args: &Args, search: &mut BrainSearch, term: &mut Term) -> Result<u8> {
    let query = args.query.join(" ");
    let results = crate::cli::search::visible_hits(search, &query, 5)?;
    if results.is_empty() {
        writeln!(term.err, "No results for \"{}\"", one(&query))?;
        return Ok(FAILED);
    }
    writeln!(term.out, "🔍 \"{}\" — {} results\n", one(&query), results.len())?;
    for result in &results {
        let preview = screen::show(&strip_frontmatter(&result.content), 200);
        writeln!(term.out, "[{}] {preview}…\n", one(&result.doc_id))?;
    }
    Ok(OK)
}
