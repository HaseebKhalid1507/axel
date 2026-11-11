//! `memories [--limit <n>]`. Commit 18b writes this file; the body is today's, moved as it
//! is.

use std::process::ExitCode;

use axel_memkoshi::storage::MemoryStorage;

use crate::cli::ensure_brain;
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::r8::Brain;

#[derive(clap::Args)]
pub struct Args {
    /// Maximum memories to list
    #[arg(long, default_value = "10")]
    pub limit: usize,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let path = &brain.path;
    ensure_brain(path)?;

    let brain = Brain::open(path)?;
    let signer = brain.signer();
    drop(brain);

    let storage = MemoryStorage::open_existing(path)?;
    let memories = storage.list_memories(args.limit)?;

    if memories.is_empty() {
        eprintln!("No memories stored yet. Use `axel remember` to add some.");
        return Ok(ExitCode::FAILURE);
    }

    for mem in &memories {
        let sig_status = match (&signer, &mem.signature) {
            (Some(s), Some(_)) => if s.verify(mem) { "✓" } else { "⚠ TAMPERED" },
            (Some(_), None) => "⚠ UNSIGNED",
            (None, _) => "",
        };

        println!("[{}] {} (importance: {:.1}) {}", mem.id, mem.title, mem.importance, sig_status);
        println!("  category: {:?} | topic: {} | tags: {:?}", mem.category, mem.topic, mem.tags);
        if !mem.abstract_text.is_empty() {
            println!("  {}", mem.abstract_text);
        }
        println!();
    }
    Ok(ExitCode::SUCCESS)
}
