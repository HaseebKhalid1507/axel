//! `show <id>`. Commit 18b writes this file.

use std::process::ExitCode;

use crate::error::Result;
use crate::ext::types::BrainArg;

#[derive(clap::Args)]
pub struct Args {
    /// A row id, or `ctx-<id>` for an archive
    pub id: String,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let _ = (args, brain);
    eprintln!("not built yet");
    Ok(ExitCode::from(2))
}
