//! `recall [<query>...]`. Commit 18b writes this file; the body is today's, moved as it is.

use std::process::ExitCode;

use axel_memkoshi::storage::MemoryStorage;

use crate::cli::{ensure_brain, strip_frontmatter};
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::search::BrainSearch;

#[derive(clap::Args)]
pub struct Args {
    /// Optional search query (omit for boot context)
    pub query: Vec<String>,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let path = &brain.path;
    let query_parts = &args.query;
    ensure_brain(path)?;

    if query_parts.is_empty() {
        // Boot context
        let storage = MemoryStorage::open_existing(path)?;
        let stats = storage.stats()?;
        let mut has_content = false;

        if let Some(handoff) = storage.get_context("handoff")? {
            if !handoff.is_empty() {
                println!("[Handoff]\n{handoff}\n");
                has_content = true;
            }
        }

        let memories = storage.list_memories(5)?;
        if !memories.is_empty() {
            println!("[Recent Memories]");
            for mem in &memories {
                println!("  • [{:?}] {}: {}", mem.category, mem.title, mem.abstract_text);
            }
            has_content = true;
        }

        if !has_content {
            eprintln!("Brain is empty. Use `axel remember` or `axel index` to add content.");
        }

        println!("\n[Stats] {} memories, {} staged, {} events",
            stats.total_memories, stats.staged_count, stats.event_count);
        Ok(ExitCode::SUCCESS)
    } else {
        // Query-based recall
        let query = query_parts.join(" ");
        let mut search = BrainSearch::open(path)?;
        let response = search.search(&query, 5)?;

        if response.results.is_empty() {
            eprintln!("No results for \"{query}\"");
            return Ok(ExitCode::FAILURE);
        }

        println!("🔍 \"{}\" — {} results\n", query, response.results.len());
        for result in &response.results {
            let clean = strip_frontmatter(&result.content);
            let preview: String = clean.chars().take(200).collect();
            println!("[{}] {preview}…\n", result.doc_id);
        }
        Ok(ExitCode::SUCCESS)
    }
}
