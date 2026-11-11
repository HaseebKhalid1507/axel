//! `model fetch`: the one command that uses the network (section 11, "The model"). It
//! downloads the model into `ext::config::default_model_dir(home)`, the folder the
//! extension looks in when config `model_dir` is not set. Every other entry point stays
//! offline (`velocirag::download::is_offline()`).

use std::path::Path;
use std::process::ExitCode;

use velocirag::download::{self, ModelSpec, EMBEDDER_MODEL};

use crate::error::{AxelError, Result};
use crate::ext::config::default_model_dir;
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

pub fn run(args: &Args, _brain: &BrainArg) -> Result<ExitCode> {
    match args.cmd {
        Cmd::Fetch => {
            let home = dirs::home_dir().ok_or_else(|| AxelError::Other("no home folder: nowhere to put the model".into()))?;
            fetch(&default_model_dir(&home))
        }
    }
}

fn fetch(dir: &Path) -> Result<ExitCode> {
    if let Some(graph) = download::model_files_in(dir) {
        println!("the model files are already in {} ({})", dir.display(), graph.display());
        return Ok(ExitCode::SUCCESS);
    }
    let Some(dir_text) = dir.to_str() else {
        return Err(AxelError::Other(format!("the model folder is not UTF-8: {}", dir.display())));
    };
    // `ensure_model` puts the files in `<models_cache_dir>/<local_dir>`, and the function
    // that takes the folder (`ensure_model_in`) is `pub(crate)` in velocirag. An absolute
    // `local_dir` replaces the cache folder when joined, so the files land in `dir`. Leaked
    // once, by a process that ends right after.
    let local_dir: &'static str = Box::leak(dir_text.to_string().into_boxed_str());
    let spec = ModelSpec { repo: EMBEDDER_MODEL.repo, local_dir, files: EMBEDDER_MODEL.files };
    println!("downloading {} into {}", spec.repo, dir.display());
    download::set_offline(false);
    let got = download::ensure_model(&spec);
    download::set_offline(true);
    let got = got?;
    match download::model_files_in(&got) {
        Some(graph) => {
            println!("the model is in {} ({})", got.display(), graph.display());
            Ok(ExitCode::SUCCESS)
        }
        None => Err(AxelError::Other(format!("the download ended but the model files are not in {}", got.display()))),
    }
}
