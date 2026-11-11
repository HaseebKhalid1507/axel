//! `model fetch`. Commit 21 writes this file.

use std::process::ExitCode;

use crate::error::Result;
use crate::ext::types::BrainArg;

#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(clap::Subcommand)]
pub enum Cmd {
    /// Download the embedding model
    Fetch,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let _ = (args, brain);
    eprintln!("not built yet");
    Ok(ExitCode::from(2))
}
