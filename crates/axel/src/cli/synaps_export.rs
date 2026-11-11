//! `synaps-export`, with the arguments cut-spec A2 gives. AX-1 writes this file, its `Args`
//! too; commit 3 writes that `Args` empty.

use std::process::ExitCode;

use crate::error::Result;
use crate::ext::types::BrainArg;

#[derive(clap::Args)]
pub struct Args {}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let _ = (args, brain);
    eprintln!("not built yet");
    Ok(ExitCode::from(2))
}
