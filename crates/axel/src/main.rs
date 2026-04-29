//! Axel CLI — portable agent intelligence from the command line.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};

use axel::config::AxelConfig;
use axel::r8::Brain;
use axel::search::BrainSearch;
use axel_memkoshi::memory::{Memory, MemoryCategory};
use axel_memkoshi::pipeline::MemoryPipeline;
use axel_memkoshi::storage::MemoryStorage;

/// Axel — Portable Agent Intelligence
///
/// Search, memory, and session awareness in one .r8 file.
#[derive(Parser)]
#[command(name = "axel", version, about, long_about = None)]
struct Cli {
    /// Path to .r8 brain file (overrides AXEL_BRAIN env var)
    #[arg(long, global = true, env = "AXEL_BRAIN")]
    brain: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new .r8 brain
    Init {
        /// Agent name for the brain
        #[arg(long)]
        name: Option<String>,
    },
    /// Index a file or directory into the brain
    Index {
        /// Path to file or directory to index
        path: PathBuf,
    },
    /// Search the brain
    Search {
        /// Search query
        query: Vec<String>,
        /// Maximum results to return
        #[arg(long, default_value = "5")]
        limit: usize,
        /// Output as JSON for scripting
        #[arg(long)]
        json: bool,
    },
    /// Store a memory permanently
    Remember {
        /// Memory content
        content: Vec<String>,
        /// Memory category
        #[arg(long, default_value = "events")]
        category: String,
        /// Memory topic
        #[arg(long, default_value = "general")]
        topic: String,
    },
    /// Get relevant context (boot context or query-based recall)
    Recall {
        /// Optional search query (omit for boot context)
        query: Vec<String>,
    },
    /// Extract memories from a transcript or text file
    Extract {
        /// Path to file, or inline text
        input: Vec<String>,
    },
    /// Manage session handoff
    Handoff {
        /// Subcommand: set, get, or clear
        #[arg(default_value = "get")]
        action: String,
        /// Content for 'set' action
        content: Vec<String>,
    },
    /// Delete a memory by ID
    Forget {
        /// Memory ID (e.g. mem_abc12345)
        id: String,
    },
    /// Show brain statistics
    Stats,
    /// List stored memories
    Memories {
        /// Maximum memories to list
        #[arg(long, default_value = "10")]
        limit: usize,
    },
}

fn brain_path(cli: &Cli) -> PathBuf {
    cli.brain.clone().unwrap_or_else(|| AxelConfig::default().brain_path)
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match &cli.command {
        Commands::Init { name } => cmd_init(&cli, name.as_deref()),
        Commands::Index { path } => cmd_index(&cli, path),
        Commands::Search { query, limit, json } => cmd_search(&cli, query, *limit, *json),
        Commands::Remember { content, category, topic } => cmd_remember(&cli, content, category, topic),
        Commands::Recall { query } => cmd_recall(&cli, query),
        Commands::Extract { input } => cmd_extract(&cli, input),
        Commands::Handoff { action, content } => cmd_handoff(&cli, action, content),
        Commands::Forget { id } => cmd_forget(&cli, id),
        Commands::Stats => cmd_stats(&cli),
        Commands::Memories { limit } => cmd_memories(&cli, *limit),
    };

    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

// ── Commands ────────────────────────────────────────────────────────────────

fn cmd_init(cli: &Cli, name: Option<&str>) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let path = brain_path(cli);
    if path.exists() {
        eprintln!("Brain already exists at {}", path.display());
        eprintln!("To start fresh, delete it first.");
        return Ok(ExitCode::FAILURE);
    }

    let brain = Brain::create(&path, name)?;
    println!("✓ Brain created: {}", path.display());
    if let Some(n) = brain.meta().agent_name.as_deref() {
        println!("  Agent: {n}");
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_index(cli: &Cli, target: &Path) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let path = brain_path(cli);
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
    Ok(ExitCode::SUCCESS)
}

fn cmd_search(
    cli: &Cli,
    query_parts: &[String],
    limit: usize,
    json_mode: bool,
) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let query = query_parts.join(" ");
    if query.is_empty() {
        eprintln!("Please provide a search query.");
        return Ok(ExitCode::FAILURE);
    }

    let path = brain_path(cli);
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

fn cmd_remember(
    cli: &Cli,
    content_parts: &[String],
    category_str: &str,
    topic: &str,
) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let content = content_parts.join(" ");
    if content.is_empty() {
        eprintln!("Please provide memory content.");
        return Ok(ExitCode::FAILURE);
    }

    let category = parse_category(category_str)?;

    let path = brain_path(cli);
    ensure_brain(&path)?;

    let title: String = content.chars().take(60).collect();
    let mut mem = Memory::new(category, topic, &title, &content);

    // Validate
    let pipeline = MemoryPipeline::new();
    if let Err(errors) = pipeline.validate(&mem) {
        eprintln!("Validation failed:");
        for e in &errors {
            eprintln!("  • {e}");
        }
        return Ok(ExitCode::FAILURE);
    }

    // Sign if brain has a signing key
    let brain = Brain::open(&path)?;
    if let Some(signer) = brain.signer() {
        mem.signature = Some(signer.sign(&mem));
    }
    drop(brain);

    // Store
    let mut storage = MemoryStorage::open_existing(&path)?;
    let staged = storage.stage_memory(&mem)?;
    storage.approve(&staged.memory.id)?;

    // Index for search
    let mut search = BrainSearch::open(&path)?;
    search.index_memory(&mem)?;

    println!("✓ Remembered: {} ({})", mem.title, mem.id);
    println!("  category: {:?} | topic: {}", category, topic);
    Ok(ExitCode::SUCCESS)
}

fn cmd_recall(cli: &Cli, query_parts: &[String]) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let path = brain_path(cli);
    ensure_brain(&path)?;

    if query_parts.is_empty() {
        // Boot context
        let storage = MemoryStorage::open_existing(&path)?;
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
        let mut search = BrainSearch::open(&path)?;
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

fn cmd_extract(cli: &Cli, input_parts: &[String]) -> Result<ExitCode, Box<dyn std::error::Error>> {
    if input_parts.is_empty() {
        eprintln!("Please provide a file path or text to extract from.");
        return Ok(ExitCode::FAILURE);
    }

    let path = brain_path(cli);
    ensure_brain(&path)?;

    // Read from file or treat as inline text
    let text = if input_parts.len() == 1 && Path::new(&input_parts[0]).exists() {
        std::fs::read_to_string(&input_parts[0])?
    } else {
        input_parts.join(" ")
    };

    // Extract via regex
    let raw_memories = axel_stelline::extractor::extract_regex(&text);
    if raw_memories.is_empty() {
        eprintln!("No memories extracted from input.");
        return Ok(ExitCode::FAILURE);
    }

    // Quality gate
    let result = axel_stelline::quality::quality_gate(raw_memories);
    println!("Extracted: {} accepted, {} rejected\n", result.accepted.len(), result.rejected.len());

    for (mem, reason) in &result.rejected {
        eprintln!("  ✗ Rejected: {} ({})", mem.title, reason);
    }

    if result.accepted.is_empty() {
        eprintln!("No memories passed quality gate.");
        return Ok(ExitCode::FAILURE);
    }

    // Store accepted memories
    let brain = Brain::open(&path)?;
    let signer = brain.signer();
    drop(brain);

    let mut storage = MemoryStorage::open_existing(&path)?;
    let mut search = BrainSearch::open(&path)?;
    let mut stored = 0;
    let pipeline = MemoryPipeline::new();

    for mem in &result.accepted {
        if let Err(errors) = pipeline.validate(mem) {
            eprintln!("  ✗ Validation failed for '{}': {}", mem.title, errors.join(", "));
            continue;
        }

        // Sign if brain has a signing key
        let mut signed_mem = mem.clone();
        if let Some(ref s) = signer {
            signed_mem.signature = Some(s.sign(&signed_mem));
        }

        match storage.stage_memory(&signed_mem) {
            Ok(staged) => {
                let _ = storage.approve(&staged.memory.id);
                let _ = search.index_memory(&signed_mem);
                stored += 1;
                println!("  ✓ [{:?}] {}", mem.category, mem.title);
            }
            Err(e) => eprintln!("  ✗ Failed to store: {e}"),
        }
    }

    println!("\n✓ Stored {stored} memories");
    Ok(ExitCode::SUCCESS)
}

fn cmd_handoff(cli: &Cli, action: &str, content_parts: &[String]) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let path = brain_path(cli);
    ensure_brain(&path)?;
    let storage = MemoryStorage::open_existing(&path)?;

    const MAX_HANDOFF_CHARS: usize = 4096;

    match action {
        "get" => {
            match storage.get_context("handoff")? {
                Some(handoff) if !handoff.is_empty() => println!("{handoff}"),
                _ => {
                    eprintln!("No handoff set.");
                    return Ok(ExitCode::FAILURE);
                }
            }
        }
        "set" => {
            let content = content_parts.join(" ");
            if content.is_empty() {
                eprintln!("Please provide handoff content.");
                return Ok(ExitCode::FAILURE);
            }
            if content.len() > MAX_HANDOFF_CHARS {
                eprintln!("Handoff too long ({} chars, max {MAX_HANDOFF_CHARS})", content.len());
                return Ok(ExitCode::FAILURE);
            }
            storage.set_context("handoff", &content, "boot")?;
            println!("✓ Handoff set ({} chars)", content.len());
        }
        "clear" => {
            storage.set_context("handoff", "", "boot")?;
            println!("✓ Handoff cleared");
        }
        other => {
            // Treat unknown action as "set" with action word included
            let content = std::iter::once(other.to_string())
                .chain(content_parts.iter().cloned())
                .collect::<Vec<_>>()
                .join(" ");
            if content.len() > MAX_HANDOFF_CHARS {
                eprintln!("Handoff too long ({} chars, max {MAX_HANDOFF_CHARS})", content.len());
                return Ok(ExitCode::FAILURE);
            }
            storage.set_context("handoff", &content, "boot")?;
            println!("✓ Handoff set ({} chars)", content.len());
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn cmd_forget(cli: &Cli, id: &str) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let path = brain_path(cli);
    ensure_brain(&path)?;

    let storage = MemoryStorage::open_existing(&path)?;
    if storage.delete_memory(id)? {
        println!("✓ Forgotten: {id}");
        Ok(ExitCode::SUCCESS)
    } else {
        eprintln!("Memory not found: {id}");
        Ok(ExitCode::FAILURE)
    }
}

fn cmd_stats(cli: &Cli) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let path = brain_path(cli);
    ensure_brain(&path)?;

    let brain = Brain::open(&path)?;
    let meta = brain.meta();

    let doc_count: i64 = brain.conn().query_row(
        "SELECT COUNT(*) FROM documents", [], |r| r.get(0)
    ).unwrap_or(0);

    let storage = MemoryStorage::open_existing(&path)?;
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
    Ok(ExitCode::SUCCESS)
}

fn cmd_memories(cli: &Cli, limit: usize) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let path = brain_path(cli);
    ensure_brain(&path)?;

    let brain = Brain::open(&path)?;
    let signer = brain.signer();
    drop(brain);

    let storage = MemoryStorage::open_existing(&path)?;
    let memories = storage.list_memories(limit)?;

    if memories.is_empty() {
        eprintln!("No memories stored yet. Use `axel remember` or `axel extract` to add some.");
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

// ── Helpers ─────────────────────────────────────────────────────────────────

fn ensure_brain(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        eprintln!("No brain at {}. Run `axel init` first.", path.display());
        std::process::exit(1);
    }
    Ok(())
}

fn parse_category(s: &str) -> Result<MemoryCategory, Box<dyn std::error::Error>> {
    match s.to_lowercase().as_str() {
        "events" | "event" => Ok(MemoryCategory::Events),
        "preferences" | "pref" => Ok(MemoryCategory::Preferences),
        "entities" | "entity" => Ok(MemoryCategory::Entities),
        "cases" | "case" => Ok(MemoryCategory::Cases),
        "patterns" | "pattern" => Ok(MemoryCategory::Patterns),
        other => {
            Err(format!(
                "Unknown category: '{}'. Valid: events, preferences, entities, cases, patterns",
                other
            ).into())
        }
    }
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
