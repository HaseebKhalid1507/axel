//! `restore [<id>...] [--by-model --since <date>] [--forget-turns <ctx-id>]`. Commit 18b
//! writes this file.

use std::process::ExitCode;

use crate::error::Result;
use crate::ext::types::BrainArg;

#[derive(clap::Args)]
pub struct Args {
    /// Ids of hidden rows to bring back
    pub ids: Vec<String>,
    /// Every row a model hid or superseded (with --since)
    #[arg(long)]
    pub by_model: bool,
    /// With --by-model: since this date
    #[arg(long, value_name = "DATE")]
    pub since: Option<String>,
    /// Index again the turns of this archive that a forget tool left out
    #[arg(long, value_name = "CTX-ID")]
    pub forget_turns: Option<String>,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let _ = (args, brain);
    eprintln!("not built yet");
    Ok(ExitCode::from(2))
}
