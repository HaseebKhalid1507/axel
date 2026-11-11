//! `forget [<id>...] [--now] [--hidden]`. Commit 18b writes this file.

use std::process::ExitCode;

use axel_memkoshi::storage::MemoryStorage;

use crate::cli::ensure_brain;
use crate::error::Result;
use crate::ext::types::BrainArg;

#[derive(clap::Args)]
pub struct Args {
    /// Memory ID (e.g. mem_abc12345)
    pub ids: Vec<String>,
    /// Purge rows that have been hidden for at least 24 hours
    #[arg(long)]
    pub now: bool,
    /// With --now: the rows hidden by a model or from the command line
    #[arg(long)]
    pub hidden: bool,
}

// SPEC: 4805-4807 say the stub body is "today's code moved as it is", and today's `forget`
// takes exactly one id and no flag (4817 gives the new form). The stub runs today's code
// for each id; with a flag, or with no id, it is the part that is not built yet.
pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    if args.now || args.hidden || args.ids.is_empty() {
        eprintln!("not built yet");
        return Ok(ExitCode::from(2));
    }
    let path = &brain.path;
    ensure_brain(path)?;

    let storage = MemoryStorage::open_existing(path)?;
    let mut code = ExitCode::SUCCESS;
    for id in &args.ids {
        if storage.delete_memory(id)? {
            println!("✓ Forgotten: {id}");
        } else {
            eprintln!("Memory not found: {id}");
            code = ExitCode::FAILURE;
        }
    }
    Ok(code)
}
