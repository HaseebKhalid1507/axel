//! `memories [--limit <n>]`. Commit 18b writes this file.
//!
//! The newest rows of `memories`, as before (chat rows among them, decision 16), with
//! their state and "unchanged since saved"; every field passes the screen.

use std::path::Path;
use std::process::ExitCode;

use crate::cli::show::{self, one, open_to_read, rows_where, Term, FAILED, OK};
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::screen;

#[derive(clap::Args)]
pub struct Args {
    /// Maximum memories to list
    #[arg(long, default_value = "10")]
    pub limit: usize,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    show::on_std(|term| run_with(args, &brain.path, term))
}

pub fn run_with(args: &Args, path: &Path, term: &mut Term) -> Result<u8> {
    let conn = open_to_read(path)?;
    let limit = i64::try_from(args.limit).unwrap_or(i64::MAX);
    let rows = rows_where(&conn, "ORDER BY m.created DESC, m.id LIMIT ?1", [limit])?;
    if rows.is_empty() {
        writeln!(term.err, "No memories stored yet. Use `{} remember` to add some.", crate::cmd_name())?;
        return Ok(FAILED);
    }
    for row in &rows {
        let signature = match show::unchanged(&conn, row)? {
            "yes" => "✓",
            "no" => "⚠ CHANGED SINCE SAVED",
            _ => "⚠ UNSIGNED",
        };
        writeln!(term.out, "[{}] {} {signature}", one(&row.id), one(&row.title))?;
        writeln!(
            term.out,
            "  kind: {} | origin: {} | {} | tags: {}",
            one(&row.kind),
            one(&row.origin),
            show::state(row),
            one(&row.tags)
        )?;
        if row.secret {
            writeln!(term.out, "  (a secret memory; its text is not shown)")?;
        } else if !row.abstract_text.trim().is_empty() {
            for line in screen::show_block(&row.abstract_text, 600).lines() {
                writeln!(term.out, "  {line}")?;
            }
        }
        writeln!(term.out)?;
    }
    Ok(OK)
}
