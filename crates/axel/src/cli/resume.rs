//! `resume <session id> <learning|recall|both> [--catch-up]`. Commit 18b writes this file.
//!
//! Removes the chat's pause rows and brings back the captures a `learning` pause hid, each
//! through the restore of one id (`store::clear_pause`). `--catch-up` deletes the chat's
//! `archive_index` rows skipped as `paused`, so that the next scan reads those files as new
//! (B-6); nothing of the scan's is called here.

use std::path::Path;
use std::process::ExitCode;

use rusqlite::params;

use crate::cli::pause::kinds;
use crate::cli::show::{self, open, write, Term, OK, USAGE};
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::{project, store};

#[derive(clap::Args)]
pub struct Args {
    /// The chat's session id
    pub session_id: String,
    /// What to resume
    // SPEC: 3523 (4.3): what to resume is required, as for `pause`; the skeleton's `Option`
    // is now required.
    #[arg(value_parser = ["learning", "recall", "both"])]
    pub what: String,
    /// Index the archives of this chat that were sealed during the pause
    #[arg(long)]
    pub catch_up: bool,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    show::on_std(|term| run_with(args, &brain.path, term))
}

pub fn run_with(args: &Args, path: &Path, term: &mut Term) -> Result<u8> {
    let session_id = args.session_id.trim();
    let Some(kinds) = kinds(&args.what) else {
        writeln!(term.err, "say what to resume: learning, recall or both")?;
        return Ok(USAGE);
    };
    if session_id.is_empty() {
        writeln!(term.err, "give the chat's session id")?;
        return Ok(USAGE);
    }
    if args.catch_up && !kinds.contains(&"learning") {
        writeln!(term.err, "--catch-up goes with learning: the archives were skipped for a learning pause")?;
        return Ok(USAGE);
    }
    let conn = open(path)?;
    let logical = project::logical(session_id);
    let (lifted, back, caught_up) = write(&conn, || {
        let mut lifted = Vec::new();
        let mut back = 0;
        for what in kinds {
            let paused: i64 = conn.query_row(
                "SELECT COUNT(*) FROM chat_pauses WHERE what = ?1 AND (logical = ?2 OR session_id = ?3)",
                params![what, logical, session_id],
                |r| r.get(0),
            )?;
            if paused > 0 {
                lifted.push(*what);
            }
            back += store::clear_pause(&conn, &logical, Some(session_id), what)?;
        }
        let caught_up = if args.catch_up {
            conn.execute(
                "DELETE FROM archive_index
                  WHERE state = 'skipped' AND skip_reason = 'paused' AND (logical = ?1 OR session_id = ?2)",
                params![logical, session_id],
            )?
        } else {
            0
        };
        Ok((lifted, back, caught_up))
    })?;
    let shown = show::one(session_id);
    for what in kinds {
        if !lifted.contains(what) {
            writeln!(term.out, "Chat {shown} had no {what} pause.")?;
        } else if *what == "learning" {
            writeln!(term.out, "Learning is back on for chat {shown}; {back} hidden turns are back.")?;
        } else {
            writeln!(term.out, "Recall is back on for chat {shown}.")?;
        }
    }
    if args.catch_up {
        writeln!(
            term.out,
            "{caught_up} archives skipped during the pause will be indexed by the next scan of the running extension."
        )?;
    } else if kinds.contains(&"learning") {
        writeln!(
            term.out,
            "Archives sealed during the pause stay unindexed; \"axel-v2 resume {shown} learning --catch-up\" indexes them."
        )?;
    }
    Ok(OK)
}
