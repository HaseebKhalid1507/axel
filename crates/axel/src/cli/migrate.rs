//! `migrate init | synaps | verify | catch-up` (section 1). The work is in `import.rs`; this
//! file is the flags, the one JSON report on standard output, and the exit code. The
//! destination is the global `--brain`; `import` refuses one that came from the environment
//! (`brain.from_env`).

use std::path::PathBuf;
use std::process::ExitCode;

use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::import::{self, CatchUpOpts, InitOpts, Outcome, SynapsOpts, VerifyOpts};

#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(clap::Subcommand)]
pub enum Cmd {
    /// Make the brain (brain.r8) from a copy of the old Axel brain (axel.r8)
    Init {
        #[arg(long, value_name = "FILE")]
        from_axel: PathBuf,
        /// Move an existing brain.r8 aside and start again
        #[arg(long)]
        replace: bool,
        /// With --replace: even when an extension is running on the destination
        #[arg(long, requires = "replace")]
        even_if_running: bool,
        /// The Synaps base this brain is for (default: <home>/.synaps-cli)
        #[arg(long, value_name = "DIR")]
        synaps_base: Option<PathBuf>,
        /// The file `status` watches for writes after the import (default: --from-axel)
        #[arg(long, value_name = "PATH")]
        watch: Option<PathBuf>,
    },
    /// Bring over the forum posts and forum tombstones of the Synaps brain.r8
    Synaps {
        #[arg(long, value_name = "FILE")]
        from: PathBuf,
        /// May be repeated
        #[arg(long, value_name = "OLD=NEW")]
        map: Vec<String>,
        /// Exit 0 although rows that are neither forum posts nor Synaps notes stay behind
        #[arg(long)]
        accept_foreign: bool,
    },
    /// Compare the brain (brain.r8) with both sources
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
        /// The file `status` watches for writes after this run (default: --from-axel)
        #[arg(long, value_name = "PATH")]
        watch: Option<PathBuf>,
    },
}

/// Runs the command and returns its report and exit code. Prints nothing on standard output.
pub fn execute(args: &Args, brain: &BrainArg) -> Outcome {
    match &args.cmd {
        Cmd::Init { from_axel, replace, even_if_running, synaps_base, watch } => import::init(
            brain,
            &InitOpts {
                from_axel: from_axel.clone(),
                replace: *replace,
                even_if_running: *even_if_running,
                synaps_base: synaps_base.clone(),
                watch: watch.clone(),
            },
        ),
        Cmd::Synaps { from, map, accept_foreign } => import::synaps(
            brain,
            &SynapsOpts { from: from.clone(), map: map.clone(), accept_foreign: *accept_foreign },
        ),
        Cmd::Verify { from_axel, from_synaps, map, accept_foreign } => import::verify(
            brain,
            &VerifyOpts {
                from_axel: from_axel.clone(),
                from_synaps: from_synaps.clone(),
                map: map.clone(),
                accept_foreign: *accept_foreign,
            },
        ),
        Cmd::CatchUp { from_axel, watch } => {
            import::catch_up(brain, &CatchUpOpts { from_axel: from_axel.clone(), watch: watch.clone() })
        }
    }
}

/// One JSON report on standard output; what a person reads goes to standard error.
pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let outcome = execute(args, brain);
    println!("{}", serde_json::to_string_pretty(&outcome.report)?);
    Ok(ExitCode::from(outcome.code))
}
