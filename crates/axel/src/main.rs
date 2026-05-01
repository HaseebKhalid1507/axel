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
        /// Optional source name to prefix doc_ids (e.g. "mikoshi")
        #[arg(long)]
        source: Option<String>,
    },
    /// Incrementally sync a directory into the brain
    ///
    /// Walks `path`, re-indexing only files whose mtime is newer than
    /// the stored `indexed_at`, and prunes DB entries whose `file_path`
    /// no longer exists on disk (scoped to `path`).
    IndexSync {
        /// Path to directory to sync
        path: PathBuf,
        /// Optional source name to prefix doc_ids (e.g. "mikoshi")
        #[arg(long)]
        source: Option<String>,
        /// Skip pruning of deleted files
        #[arg(long)]
        no_prune: bool,
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
    /// Run as a SynapsCLI extension (JSON-RPC over stdio)
    /// Run a consolidation pass on the brain
    ///
    /// Reindexes changed files, strengthens accessed documents,
    /// reorganizes graph edges, and prunes stale content.
    Consolidate {
        /// Run specific phase only
        #[arg(long, value_parser = ["reindex", "strengthen", "reorganize", "prune"])]
        phase: Option<String>,
        /// Preview changes without applying them
        #[arg(long)]
        dry_run: bool,
        /// Verbose output (per-document details)
        #[arg(short, long)]
        verbose: bool,
        /// Path to sources config TOML
        #[arg(long)]
        sources: Option<PathBuf>,
        /// Show consolidation history
        #[arg(long)]
        history: bool,
    },
    Extension,
    /// Run as an MCP server (exposes search/remember/recall as tools)
    Mcp,
}

fn brain_path(cli: &Cli) -> PathBuf {
    cli.brain.clone().unwrap_or_else(|| AxelConfig::default().brain_path)
}

fn load_sources(override_path: Option<&Path>) -> Result<Vec<axel::consolidate::SourceDir>, Box<dyn std::error::Error>> {
    // Single source of truth lives in `axel::consolidate` so the MCP handler
    // and CLI agree. Errors degrade to defaults rather than aborting.
    Ok(axel::consolidate::load_sources(override_path))
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match &cli.command {
        Commands::Init { name } => cmd_init(&cli, name.as_deref()),
        Commands::Index { path, source } => cmd_index(&cli, path, source.as_deref()),
        Commands::IndexSync { path, source, no_prune } => cmd_index_sync(&cli, path, source.as_deref(), *no_prune),
        Commands::Search { query, limit, json } => cmd_search(&cli, query, *limit, *json),
        Commands::Remember { content, category, topic } => cmd_remember(&cli, content, category, topic),
        Commands::Recall { query } => cmd_recall(&cli, query),
        Commands::Handoff { action, content } => cmd_handoff(&cli, action, content),
        Commands::Forget { id } => cmd_forget(&cli, id),
        Commands::Stats => cmd_stats(&cli),
        Commands::Memories { limit } => cmd_memories(&cli, *limit),
        Commands::Consolidate { phase, dry_run, verbose, sources, history } =>
            if *history {
                cmd_consolidate_history(&cli)
            } else {
                cmd_consolidate(&cli, phase.as_deref(), *dry_run, *verbose, sources.as_deref())
            },
        Commands::Extension => {
            let path = brain_path(&cli);
            axel::extension::run(&path).map(|_| ExitCode::SUCCESS)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        }
        Commands::Mcp => {
            let path = brain_path(&cli);
            axel::mcp::run(&path).map(|_| ExitCode::SUCCESS)
                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
        }
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

fn cmd_index(cli: &Cli, target: &Path, source: Option<&str>) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let path = brain_path(cli);
    ensure_brain(&path)?;

    let mut search = BrainSearch::open(&path)?;
    let start = Instant::now();
    let mut count = 0;

    if target.is_dir() {
        for entry in walkdir::WalkDir::new(target)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "md" || ext == "txt"))
        {
            let content = std::fs::read_to_string(entry.path())?;
            if content.len() < 50 { continue; }

            let relative_id = entry.path()
                .strip_prefix(target).unwrap_or(entry.path())
                .to_string_lossy()
                .replace('/', "::")
                .replace(".md", "")
                .replace(".txt", "");

            let doc_id = if let Some(src) = source {
                format!("{src}::{relative_id}")
            } else {
                relative_id
            };

            search.index_document(&doc_id, &content, None, Some(&entry.path().to_string_lossy()))?;
            count += 1;
        }
    } else {
        let content = std::fs::read_to_string(target)?;
        let stem = target.file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "doc".to_string());
        let doc_id = if let Some(src) = source {
            format!("{src}::{stem}")
        } else {
            stem
        };
        search.index_document(&doc_id, &content, None, Some(&target.to_string_lossy()))?;
        count = 1;
    }

    println!("✓ Indexed {count} documents ({:.1}s)", start.elapsed().as_secs_f64());
    Ok(ExitCode::SUCCESS)
}

fn cmd_index_sync(
    cli: &Cli,
    target: &Path,
    source: Option<&str>,
    no_prune: bool,
) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let path = brain_path(cli);
    ensure_brain(&path)?;

    if !target.is_dir() {
        eprintln!("index-sync requires a directory, got: {}", target.display());
        return Ok(ExitCode::FAILURE);
    }

    // Canonicalize so the LIKE-prefix match in the DB lines up with what
    // `index_document(file_path=…)` originally stored (absolute paths).
    let target_abs = std::fs::canonicalize(target).unwrap_or_else(|_| target.to_path_buf());
    let prefix = {
        let mut s = target_abs.to_string_lossy().to_string();
        if !s.ends_with('/') { s.push('/'); }
        s
    };

    let start = Instant::now();
    let mut search = BrainSearch::open(&path)?;

    // Map of absolute file_path → indexed_at unix seconds for everything
    // currently in the brain under this source dir.
    let indexed: std::collections::HashMap<String, f64> = search
        .db()
        .indexed_files_under(&prefix)
        .map_err(|e| format!("DB read failed: {e}"))?
        .into_iter()
        .collect();

    let mut checked = 0usize;
    let mut reindexed = 0usize;
    let mut new_files = 0usize;
    let mut seen_on_disk: std::collections::HashSet<String> = std::collections::HashSet::new();

    for entry in walkdir::WalkDir::new(&target_abs)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "md" || ext == "txt"))
    {
        checked += 1;
        let file_path = entry.path();
        let abs = match std::fs::canonicalize(file_path) {
            Ok(p) => p,
            Err(_) => file_path.to_path_buf(),
        };
        let abs_str = abs.to_string_lossy().to_string();
        seen_on_disk.insert(abs_str.clone());

        // mtime as unix seconds
        let mtime = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        let is_new = !indexed.contains_key(&abs_str);
        let needs = match indexed.get(&abs_str) {
            None => true,
            Some(&indexed_at) => mtime > indexed_at + 0.5, // 0.5s slack for second-precision TIMESTAMP
        };

        if !needs { continue; }

        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if content.len() < 50 { continue; }

        let relative_id = file_path
            .strip_prefix(&target_abs).unwrap_or(file_path)
            .to_string_lossy()
            .replace('/', "::")
            .replace(".md", "")
            .replace(".txt", "");
        let doc_id = if let Some(src) = source {
            format!("{src}::{relative_id}")
        } else {
            relative_id
        };

        search.index_document(&doc_id, &content, None, Some(&abs_str))?;
        reindexed += 1;
        if is_new { new_files += 1; }
    }

    // Prune: anything in DB under this prefix that wasn't seen on disk.
    let mut pruned = 0usize;
    if !no_prune {
        for (file_path, _) in &indexed {
            if !seen_on_disk.contains(file_path) && !std::path::Path::new(file_path).exists() {
                let n = search.db().delete_documents_by_file(file_path)
                    .map_err(|e| format!("Delete failed: {e}"))?;
                if n > 0 { pruned += 1; }
            }
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "✓ sync [{}] checked={checked} reindexed={reindexed} (new={new_files}) pruned={pruned} ({elapsed:.1}s)",
        source.unwrap_or("(no-source)")
    );
    Ok(ExitCode::SUCCESS)
}

fn cmd_consolidate(
    cli: &Cli,
    phase: Option<&str>,
    dry_run: bool,
    verbose: bool,
    sources_path: Option<&Path>,
) -> Result<ExitCode, Box<dyn std::error::Error>> {
    use axel::consolidate::{self, Phase, ConsolidateOptions};

    let path = brain_path(cli);
    ensure_brain(&path)?;

    let mut search = BrainSearch::open(&path)?;

    // Load sources from TOML config or use defaults
    let sources = load_sources(sources_path)?;

    let phases = match phase {
        Some("reindex") => [Phase::Reindex].into_iter().collect(),
        Some("strengthen") => [Phase::Strengthen].into_iter().collect(),
        Some("reorganize") => [Phase::Reorganize].into_iter().collect(),
        Some("prune") => [Phase::Prune].into_iter().collect(),
        _ => std::collections::HashSet::new(), // empty = all
    };

    let opts = ConsolidateOptions {
        sources,
        phases,
        dry_run,
        verbose,
    };

    if dry_run {
        println!("🔍 Dry run — no changes will be written\n");
    }

    let stats = consolidate::consolidate(&mut search, &opts)?;

    println!("\n═══ Consolidation {} ═══", if dry_run { "(dry run)" } else { "complete" });
    let skip_info = if stats.reindex.skipped > 0 {
        format!(", {} skipped", stats.reindex.skipped)
    } else {
        String::new()
    };
    println!("  Phase 1 — Reindex:    {} checked, {} reindexed ({} new), {} pruned{}",
        stats.reindex.checked, stats.reindex.reindexed, stats.reindex.new_files, stats.reindex.pruned, skip_info);
    println!("  Phase 2 — Strengthen: {} boosted, {} decayed, {} extinction",
        stats.strengthen.boosted, stats.strengthen.decayed, stats.strengthen.extinction_signals);
    println!("  Phase 3 — Reorganize: {} pairs, +{} edges, ~{} updated, -{} removed",
        stats.reorganize.co_retrieval_pairs, stats.reorganize.edges_added,
        stats.reorganize.edges_updated, stats.reorganize.edges_removed);
    println!("  Phase 4 — Prune:      {} removed, {} flagged, {} misaligned",
        stats.prune.removed, stats.prune.flagged, stats.prune.misaligned);
    println!("  Duration: {:.1}s", stats.duration_secs);

    // Surface flagged candidates so the "human review" mechanism is actionable.
    // Always print when there are any; verbose adds the misaligned breakdown.
    if !stats.prune_candidates.is_empty() {
        println!("\n── Prune candidates (review) ──");
        for c in &stats.prune_candidates {
            if !verbose && c.reason == "misaligned_embedding" { continue; }
            println!("  [{}] {}  exc={:.3} access={} age={}d",
                c.reason, c.doc_id, c.excitability, c.access_count, c.age_days);
        }
    }

    Ok(ExitCode::SUCCESS)
}

fn cmd_consolidate_history(cli: &Cli) -> Result<ExitCode, Box<dyn std::error::Error>> {
    let path = brain_path(cli);
    ensure_brain(&path)?;

    let search = BrainSearch::open(&path)?;
    let db = search.db();

    let mut stmt = db.conn().prepare(
        "SELECT id, started_at, finished_at, phase1_reindexed, phase1_pruned,
                phase2_boosted, phase2_decayed, phase3_edges_added, phase3_edges_updated,
                phase4_flagged, phase4_removed, duration_secs
         FROM consolidation_log ORDER BY id DESC LIMIT 20"
    )?;

    let rows: Vec<(i64, String, Option<String>, i64, i64, i64, i64, i64, i64, i64, i64, f64)> = stmt
        .query_map([], |row| {
            Ok((
                row.get(0)?, row.get(1)?, row.get(2)?,
                row.get(3)?, row.get(4)?, row.get(5)?,
                row.get(6)?, row.get(7)?, row.get(8)?,
                row.get(9)?, row.get(10)?, row.get(11)?,
            ))
        })?
        .filter_map(|r| r.ok())
        .collect();

    if rows.is_empty() {
        println!("No consolidation runs recorded.");
        return Ok(ExitCode::SUCCESS);
    }

    println!("═══ Consolidation History (last {}) ═══\n", rows.len());
    for (id, started, _finished, reindexed, pruned, boosted, decayed, edges_add, edges_upd, flagged, removed, dur) in &rows {
        let ts = started.split('T').next().unwrap_or(started);
        let time = started.split('T').nth(1).and_then(|t| t.split('.').next()).unwrap_or("");
        println!("  #{id}  {ts} {time}  ({dur:.1}s)");
        println!("    reindex: {reindexed} indexed, {pruned} pruned | strengthen: {boosted} ↑ {decayed} ↓");
        println!("    graph: +{edges_add} ~{edges_upd} | prune: {removed} removed, {flagged} flagged\n");
    }

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

    // Log search hits as document access events (same as MCP path)
    let db = search.db();
    for r in &response.results {
        let _ = db.log_document_access(&r.doc_id, "search_hit", Some(&query), Some(r.score), None);
        let _ = db.increment_document_access(&r.doc_id);
    }
    let top_ids: Vec<&str> = response.results.iter().take(5).map(|r| r.doc_id.as_str()).collect();
    for i in 0..top_ids.len() {
        for j in (i+1)..top_ids.len() {
            let _ = db.log_co_retrieval(top_ids[i], top_ids[j], &query);
        }
    }

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

    // Consolidation metrics — batched queries
    let conn = brain.conn();
    let (accessed_docs, avg_excitability, min_excitability, max_excitability): (i64, f64, f64, f64) = conn.query_row(
        "SELECT COUNT(CASE WHEN access_count > 0 THEN 1 END),
                COALESCE(AVG(excitability), 0.5),
                COALESCE(MIN(excitability), 0.5),
                COALESCE(MAX(excitability), 0.5)
         FROM documents",
        [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
    ).unwrap_or((0, 0.5, 0.5, 0.5));
    let access_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM document_access", [], |r| r.get(0)
    ).unwrap_or(0);
    let co_ret_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM co_retrieval", [], |r| r.get(0)
    ).unwrap_or(0);
    let consolidation_runs: i64 = conn.query_row(
        "SELECT COUNT(*) FROM consolidation_log", [], |r| r.get(0)
    ).unwrap_or(0);

    println!("\n  ── Consolidation ──");
    println!("  Runs:           {consolidation_runs}");
    println!("  Access events:  {access_count} ({accessed_docs} unique docs)");
    println!("  Co-retrievals:  {co_ret_count}");
    println!("  Excitability:   μ={avg_excitability:.3}  min={min_excitability:.3}  max={max_excitability:.3}");

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
    if let Some(after) = trimmed.strip_prefix("---") {
        if let Some(end) = after.find("---") {
            return after[end + 3..].trim_start().to_string();
        }
    }
    content.to_string()
}
