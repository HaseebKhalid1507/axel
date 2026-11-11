//! `plugin install [--plugins-dir <dir>] [--force] [--print]` (988). Commit 20 writes this
//! file.

use std::path::PathBuf;
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
    /// Write the plugin manifest into the Synaps plugins folder
    Install {
        #[arg(long, value_name = "DIR")]
        plugins_dir: Option<PathBuf>,
        /// Replace an existing manifest, keeping a copy
        #[arg(long)]
        force: bool,
        /// Print the manifest and write nothing
        #[arg(long)]
        print: bool,
    },
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let _ = (args, brain);
    eprintln!("not built yet");
    Ok(ExitCode::from(2))
}
