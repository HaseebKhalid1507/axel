//! `status [--json] [--check]`. Commit 21 writes this file.

use std::process::ExitCode;

use crate::error::Result;
use crate::ext::types::BrainArg;

#[derive(clap::Args)]
pub struct Args {
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
    /// Exit with a status that says whether anything is wrong
    #[arg(long)]
    pub check: bool,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let _ = (args, brain);
    eprintln!("not built yet");
    Ok(ExitCode::from(2))
}
