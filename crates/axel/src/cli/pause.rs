//! `pause <session id> <learning|recall|both>`. Commit 18b writes this file.
//!
//! The same rows as the tool's `forget {"stop": …}` (section 4): a `chat_pauses` row for the
//! chat, and for `learning` its live captures hidden as `pause`. Only `resume` lifts it.

use std::path::Path;
use std::process::ExitCode;

use crate::cli::show::{self, open, write, Term, FAILED, OK, USAGE};
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::{project, store};

#[derive(clap::Args)]
pub struct Args {
    /// The chat's session id
    pub session_id: String,
    /// What to pause
    // SPEC: 3523 (4.3): what to pause is required and there is no default; the skeleton's
    // `Option` is now required.
    #[arg(value_parser = ["learning", "recall", "both"])]
    pub what: String,
    /// Pause a chat the brain knows nothing of yet (no capture, archive or recall)
    #[arg(long)]
    pub force: bool,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    show::on_std(|term| run_with(args, &brain.path, term))
}

/// `learning`, `recall`, or both of them.
pub fn kinds(what: &str) -> Option<&'static [&'static str]> {
    match what {
        "learning" => Some(&["learning"]),
        "recall" => Some(&["recall"]),
        "both" => Some(&["learning", "recall"]),
        _ => None,
    }
}

pub fn run_with(args: &Args, path: &Path, term: &mut Term) -> Result<u8> {
    let session_id = args.session_id.trim();
    let Some(kinds) = kinds(&args.what) else {
        writeln!(term.err, "say what to pause: learning, recall or both")?;
        return Ok(USAGE);
    };
    if session_id.is_empty() {
        writeln!(term.err, "give the chat's session id")?;
        return Ok(USAGE);
    }
    let conn = open(path)?;
    let logical = project::logical(session_id);
    // RULING: v2 c18b review, item 8. A chat id no capture, archive or recall-log row knows
    // is most likely mistyped: it is not paused unless `--force` says so.
    let known: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM memory_ext WHERE session_id = ?2 OR logical = ?1)
             OR EXISTS(SELECT 1 FROM archive_index WHERE session_id = ?2 OR logical = ?1)
             OR EXISTS(SELECT 1 FROM recall_log WHERE session_id = ?2 OR logical = ?1)",
        rusqlite::params![logical, session_id],
        |r| r.get(0),
    )?;
    if !known && !args.force {
        writeln!(
            term.err,
            "no capture, archive or recall of chat {} is in the brain: check the session id (\"{} pause --force …\" pauses it anyway). Nothing was changed.",
            show::one(session_id),
            crate::cmd_name()
        )?;
        return Ok(FAILED);
    }
    let hidden = write(&conn, || {
        let mut hidden = 0;
        for what in kinds {
            hidden += store::set_pause(&conn, &logical, Some(session_id), what, "cli")?;
        }
        Ok(hidden)
    })?;
    let shown = show::one(session_id);
    for what in kinds {
        match *what {
            "learning" => writeln!(
                term.out,
                "Learning is paused for chat {shown}: none of its turns is learned and none of its archives is indexed. \
                 {hidden} recent turns were hidden with it. What was archived and indexed before stays."
            )?,
            _ => writeln!(term.out, "Recall is paused for chat {shown}: nothing is recalled into it.")?,
        }
    }
    writeln!(term.out, "Lift it with \"{} resume {shown} {}\".", crate::cmd_name(), args.what)?;
    Ok(OK)
}
