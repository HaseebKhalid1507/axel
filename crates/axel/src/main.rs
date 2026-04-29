//! Axel CLI — portable agent intelligence from the command line.
//!
//! Usage:
//!   axel init [--name NAME]           Create a new .r8 brain
//!   axel index <path>                 Index a file or directory into the brain
//!   axel search <query> [--limit N]   Search the brain
//!   axel remember <content>           Store a memory
//!   axel recall [query]               Get boot context / search memories
//!   axel stats                        Show brain statistics
//!   axel memories [--limit N]         List stored memories

use std::path::{Path, PathBuf};
use std::time::Instant;

use axel::config::AxelConfig;
use axel::r8::Brain;
use axel::search::BrainSearch;
use axel_memkoshi::memory::{Memory, MemoryCategory};
use axel_memkoshi::storage::MemoryStorage;

fn brain_path() -> PathBuf {
    std::env::var("AXEL_BRAIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| AxelConfig::default().brain_path)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "init" => cmd_init(&args[2..]),
        "index" => cmd_index(&args[2..]),
        "search" => cmd_search(&args[2..]),
        "remember" => cmd_remember(&args[2..]),
        "recall" => cmd_recall(&args[2..]),
        "stats" => cmd_stats(),
        "memories" => cmd_memories(&args[2..]),
        "help" | "--help" | "-h" => { print_usage(); Ok(()) }
        other => {
            eprintln!("Unknown command: {other}");
            print_usage();
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn print_usage() {
    eprintln!("axel — Portable Agent Intelligence");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  axel init [--name NAME]           Create a new .r8 brain");
    eprintln!("  axel index <path>                 Index a file or directory");
    eprintln!("  axel search <query> [--limit N]   Search the brain");
    eprintln!("  axel remember <content>           Store a memory");
    eprintln!("  axel recall [query]               Get relevant context");
    eprintln!("  axel stats                        Show brain stats");
    eprintln!("  axel memories [--limit N]         List stored memories");
    eprintln!();
    eprintln!("Environment:");
    eprintln!("  AXEL_BRAIN    Path to .r8 file (default: ~/.config/axel/axel.r8)");
}

fn cmd_init(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = brain_path();
    let name = args.iter()
        .position(|a| a == "--name")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    if path.exists() {
        eprintln!("Brain already exists at {}", path.display());
        eprintln!("To start fresh, delete it first.");
        std::process::exit(1);
    }

    let brain = Brain::create(&path, name)?;
    println!("✓ Brain created: {}", path.display());
    if let Some(n) = brain.meta().agent_name.as_deref() {
        println!("  Agent: {n}");
    }
    Ok(())
}

fn cmd_index(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: axel index <path>");
        std::process::exit(1);
    }

    let target = Path::new(&args[0]);
    let path = brain_path();
    ensure_brain(&path)?;

    let mut search = BrainSearch::open(&path)?;
    let start = Instant::now();
    let mut count = 0;

    if target.is_dir() {
        for entry in walkdir::WalkDir::new(target)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "md" || ext == "txt"))
        {
            let content = std::fs::read_to_string(entry.path())?;
            if content.len() < 50 { continue; }

            let doc_id = entry.path()
                .strip_prefix(target).unwrap_or(entry.path())
                .to_string_lossy()
                .replace('/', "::")
                .replace(".md", "")
                .replace(".txt", "");

            search.index_document(&doc_id, &content, None, Some(&entry.path().to_string_lossy()))?;
            count += 1;
        }
    } else {
        let content = std::fs::read_to_string(target)?;
        let doc_id = target.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "doc".to_string());
        search.index_document(&doc_id, &content, None, Some(&target.to_string_lossy()))?;
        count = 1;
    }

    println!("✓ Indexed {count} documents ({:.1}s)", start.elapsed().as_secs_f64());
    Ok(())
}

fn cmd_search(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: axel search <query> [--limit N] [--json]");
        std::process::exit(1);
    }

    let json_mode = args.iter().any(|a| a == "--json");
    let limit = args.iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let query: String = args.iter()
        .filter(|a| a.as_str() != "--json" && a.as_str() != "--limit")
        .take_while(|a| a.parse::<usize>().is_err() || args.iter().position(|x| x == "--limit").map_or(true, |li| args.iter().position(|x| x.as_str() == a.as_str()).unwrap() != li + 1))
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");

    let path = brain_path();
    ensure_brain(&path)?;

    let mut search = BrainSearch::open(&path)?;
    let start = Instant::now();
    let response = search.search(&query, limit)?;
    let ms = start.elapsed().as_millis();

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
        return Ok(());
    }

    if response.results.is_empty() {
        println!("No results for \"{query}\"");
        return Ok(());
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
    Ok(())
}

fn cmd_remember(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    if args.is_empty() {
        eprintln!("Usage: axel remember [--category CAT] [--topic TOPIC] <content>");
        eprintln!("  Categories: events, preferences, entities, cases, patterns");
        std::process::exit(1);
    }

    // Parse flags
    let category = args.iter()
        .position(|a| a == "--category")
        .and_then(|i| args.get(i + 1))
        .map(|s| match s.to_lowercase().as_str() {
            "preferences" | "pref" => MemoryCategory::Preferences,
            "entities" | "entity" => MemoryCategory::Entities,
            "cases" | "case" => MemoryCategory::Cases,
            "patterns" | "pattern" => MemoryCategory::Patterns,
            _ => MemoryCategory::Events,
        })
        .unwrap_or(MemoryCategory::Events);

    let topic = args.iter()
        .position(|a| a == "--topic")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "general".to_string());

    // Collect content (everything that's not a flag or flag value)
    let skip_indices: std::collections::HashSet<usize> = {
        let mut s = std::collections::HashSet::new();
        for (i, a) in args.iter().enumerate() {
            if a == "--category" || a == "--topic" {
                s.insert(i);
                s.insert(i + 1);
            }
        }
        s
    };
    let content: String = args.iter().enumerate()
        .filter(|(i, _)| !skip_indices.contains(i))
        .map(|(_, a)| a.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    if content.is_empty() {
        eprintln!("No content provided.");
        std::process::exit(1);
    }

    let path = brain_path();
    ensure_brain(&path)?;

    // Validate through pipeline
    let title: String = content.chars().take(60).collect();
    let mem = Memory::new(category, &topic, &title, &content);

    use axel_memkoshi::pipeline::MemoryPipeline;
    let pipeline = MemoryPipeline::new();
    if let Err(errors) = pipeline.validate(&mem) {
        eprintln!("Validation failed:");
        for e in &errors {
            eprintln!("  • {e}");
        }
        std::process::exit(1);
    }

    // Store
    let mut storage = MemoryStorage::open(&path)?;
    let staged = storage.stage_memory(&mem)?;
    storage.approve(&staged.memory.id)?;

    // Index for search
    let mut search = BrainSearch::open(&path)?;
    search.index_memory(&mem)?;

    println!("✓ Remembered: {} ({})", mem.title, mem.id);
    println!("  category: {:?} | topic: {}", category, topic);
    Ok(())
}

fn cmd_recall(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let path = brain_path();
    ensure_brain(&path)?;

    if args.is_empty() {
        // Boot context — handoff + recent memories
        let mut storage = MemoryStorage::open(&path)?;
        let stats = storage.stats()?;

        if let Some(handoff) = storage.get_context("handoff")? {
            println!("[Handoff]\n{handoff}\n");
        }

        let memories = storage.list_memories(5)?;
        if !memories.is_empty() {
            println!("[Recent Memories]");
            for mem in &memories {
                println!("  • [{}] {}: {}", format!("{:?}", mem.category), mem.title, mem.abstract_text);
            }
        }

        println!("\n[Stats] {} memories, {} staged, {} events",
            stats.total_memories, stats.staged_count, stats.event_count);
    } else {
        // Search-based recall
        let query = args.join(" ");
        let mut search = BrainSearch::open(&path)?;
        let response = search.search(&query, 5)?;

        for result in &response.results {
            let preview: String = result.content.chars().take(200).collect();
            println!("[{}] {preview}…\n", result.doc_id);
        }
    }
    Ok(())
}

fn cmd_stats() -> Result<(), Box<dyn std::error::Error>> {
    let path = brain_path();
    ensure_brain(&path)?;

    let brain = Brain::open(&path)?;
    let meta = brain.meta();

    let doc_count: i64 = brain.conn().query_row(
        "SELECT COUNT(*) FROM documents", [], |r| r.get(0)
    ).unwrap_or(0);

    let mut storage = MemoryStorage::open(&path)?;
    let stats = storage.stats()?;

    let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    println!("═══ Brain: {} ═══", path.display());
    println!("  Agent:      {}", meta.agent_name.as_deref().unwrap_or("(unnamed)"));
    println!("  Model:      {} ({}d)", meta.embedder_model, meta.embedding_dim);
    println!("  Schema:     v{}", meta.schema_version);
    println!("  Created:    {}", meta.created);
    println!("  Modified:   {}", meta.last_modified);
    println!("  Documents:  {doc_count}");
    println!("  Memories:   {}", stats.total_memories);
    println!("  Staged:     {}", stats.staged_count);
    println!("  Events:     {}", stats.event_count);
    println!("  File size:  {:.1} MB", file_size as f64 / 1024.0 / 1024.0);
    Ok(())
}

fn cmd_memories(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let limit = args.iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let path = brain_path();
    ensure_brain(&path)?;

    let mut storage = MemoryStorage::open(&path)?;
    let memories = storage.list_memories(limit)?;

    if memories.is_empty() {
        println!("No memories stored yet.");
        return Ok(());
    }

    for mem in &memories {
        println!("[{}] {} (importance: {:.1})", mem.id, mem.title, mem.importance);
        println!("  category: {} | topic: {} | tags: {:?}", format!("{:?}", mem.category), mem.topic, mem.tags);
        if !mem.abstract_text.is_empty() {
            println!("  {}", mem.abstract_text);
        }
        println!();
    }
    Ok(())
}

fn ensure_brain(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        eprintln!("No brain at {}. Run `axel init` first.", path.display());
        std::process::exit(1);
    }
    Ok(())
}

/// Strip YAML frontmatter (--- ... ---) from markdown content.
fn strip_frontmatter(content: &str) -> String {
    let trimmed = content.trim_start();
    if trimmed.starts_with("---") {
        if let Some(end) = trimmed[3..].find("---") {
            return trimmed[end + 6..].trim_start().to_string();
        }
    }
    content.to_string()
}
