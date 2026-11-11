//! `why [--last <n>] [--session <id>] [--json]`. Commit 21 writes this file.

use std::process::ExitCode;

use crate::error::Result;
use crate::ext::types::BrainArg;

#[derive(clap::Args)]
pub struct Args {
    /// The last N recalls
    #[arg(long, value_name = "N")]
    pub last: Option<usize>,
    // SPEC: 4819-4820 write `why [--last <n>] [--json]`; 2557 writes
    // `why [--last N] [--session <id>]`. All three flags are declared.
    /// Only the recalls of this chat
    #[arg(long, value_name = "ID")]
    pub session: Option<String>,
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let _ = (args, brain);
    eprintln!("not built yet");
    Ok(ExitCode::from(2))
}
