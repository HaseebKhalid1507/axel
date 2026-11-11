//! `search <query>... [--limit <n>] [--json]`. Commit 18b writes this file; the body is
//! today's, moved as it is.

use std::process::ExitCode;
use std::time::Instant;

use crate::cli::{ensure_brain, strip_frontmatter};
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::search::BrainSearch;

#[derive(clap::Args)]
pub struct Args {
    /// Search query
    pub query: Vec<String>,
    /// Maximum results to return
    #[arg(long, default_value = "5")]
    pub limit: usize,
    /// Output as JSON for scripting
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let path = &brain.path;
    let json_mode = args.json;
    let query = args.query.join(" ");
    if query.is_empty() {
        eprintln!("Please provide a search query.");
        return Ok(ExitCode::FAILURE);
    }

    ensure_brain(path)?;

    let mut search = BrainSearch::open(path)?;
    let start = Instant::now();
    let limit = args.limit.min(100); // cap to prevent OOM on large brains
    let response = search.search(&query, limit)?;
    let ms = start.elapsed().as_millis();

    // Record search feedback for consolidation
    search.record_search_feedback(&query, &response.results);

    if json_mode {
        let results: Vec<serde_json::Value> = response.results.iter().map(|r| {
            serde_json::json!({
                "doc_id": r.doc_id,
                "score": r.score,
                "content": strip_frontmatter(&r.content),
            })
        }).collect();
        println!("{}", serde_json::json!({
            "query": query,
            "count": results.len(),
            "ms": ms,
            "results": results,
        }));
        return if results.is_empty() {
            Ok(ExitCode::FAILURE)
        } else {
            Ok(ExitCode::SUCCESS)
        };
    }

    if response.results.is_empty() {
        eprintln!("No results for \"{query}\"");
        return Ok(ExitCode::FAILURE);
    }

    println!("🔍 \"{}\" — {} results ({ms}ms)\n", query, response.results.len());
    for (i, result) in response.results.iter().enumerate() {
        let clean = strip_frontmatter(&result.content);
        let preview: String = clean
            .lines()
            .filter(|l| !l.trim().is_empty())
            .take(3)
            .collect::<Vec<_>>()
            .join(" ")
            .chars()
            .take(120)
            .collect();
        println!("  {}. [{}] (score: {:.3})", i + 1, result.doc_id, result.score);
        println!("     {preview}…\n");
    }
    Ok(ExitCode::SUCCESS)
}
