//! `resume <session id> [learning|recall|both] [--catch-up]`. Commit 18b writes this file.

use std::process::ExitCode;

use crate::error::Result;
use crate::ext::types::BrainArg;

#[derive(clap::Args)]
pub struct Args {
    /// The chat's session id
    pub session_id: String,
    // SPEC: 4819 writes `resume <session id> [--catch-up]`; 2763 adds
    // `[learning|recall|both]`. The longer form is declared, as for `pause`.
    /// What to resume
    #[arg(value_parser = ["learning", "recall", "both"])]
    pub what: Option<String>,
    /// Index the archives of this chat that were sealed during the pause
    #[arg(long)]
    pub catch_up: bool,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let _ = (args, brain);
    eprintln!("not built yet");
    Ok(ExitCode::from(2))
}
