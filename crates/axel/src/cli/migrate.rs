//! `migrate init | synaps | verify | catch-up` (section 1, 1037-1040). Commit 19 writes this
//! file. The destination is the global `--brain`; `run` refuses one that came from the
//! environment (`brain.from_env`).

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
    /// Make a v2 brain from a copy of the old axel.r8
    Init {
        #[arg(long, value_name = "FILE")]
        from_axel: PathBuf,
        /// Move an existing new brain aside and start again
        #[arg(long)]
        replace: bool,
        /// With --replace: even when an extension is running on the destination
        #[arg(long)]
        even_if_running: bool,
    },
    /// Bring over the forum posts and forum tombstones of the Synaps brain.r8
    Synaps {
        #[arg(long, value_name = "FILE")]
        from: PathBuf,
        /// May be repeated
        #[arg(long, value_name = "OLD=NEW")]
        map: Vec<String>,
        // SPEC: 1037-1040 do not list `--accept-foreign`; 1154 gives it to `migrate synaps`
        // ("Exit status 1 when N is not 0, unless `--accept-foreign`") and 1182 to
        // `migrate verify`. It is declared on both.
        #[arg(long)]
        accept_foreign: bool,
    },
    /// Compare the new brain with both sources
    Verify {
        #[arg(long, value_name = "FILE")]
        from_axel: PathBuf,
        #[arg(long, value_name = "FILE")]
        from_synaps: PathBuf,
        /// May be repeated
        #[arg(long, value_name = "OLD=NEW")]
        map: Vec<String>,
        #[arg(long)]
        accept_foreign: bool,
    },
    /// Bring over what was written to the old brain after the import
    CatchUp {
        #[arg(long, value_name = "FILE")]
        from_axel: PathBuf,
    },
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let _ = (args, brain);
    eprintln!("not built yet");
    Ok(ExitCode::from(2))
}
