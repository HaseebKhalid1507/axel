//! `pause <session id> [learning|recall|both]`. Commit 18b writes this file.

use std::process::ExitCode;

use crate::error::Result;
use crate::ext::types::BrainArg;

#[derive(clap::Args)]
pub struct Args {
    /// The chat's session id
    pub session_id: String,
    // SPEC: 4818-4819 write `pause <session id>` alone; 2762 writes
    // `pause <session id> [learning|recall|both]`. The longer form is declared ("the
    // sections are the authority", 4817). No default is written, so there is none here.
    /// What to pause
    #[arg(value_parser = ["learning", "recall", "both"])]
    pub what: Option<String>,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let _ = (args, brain);
    eprintln!("not built yet");
    Ok(ExitCode::from(2))
}
