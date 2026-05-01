//! Consolidation engine — biologically-inspired memory lifecycle.
//!
//! Four phases: reindex → strengthen → reorganize → prune.
//! See `docs/CONSOLIDATION.md` for the full spec.

pub mod reindex;
pub mod strengthen;
pub mod reorganize;
pub mod prune;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::Utc;
use velocirag::db::ConsolidationLogEntry;

use crate::error::Result;
use crate::search::BrainSearch;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Phase {
    Reindex,
    Strengthen,
    Reorganize,
    Prune,
}

#[derive(Debug, Clone)]
pub struct SourceDir {
    pub path: PathBuf,
    pub name: String,
    pub priority: Priority,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Priority {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone)]
pub struct ConsolidateOptions {
    pub sources: Vec<SourceDir>,
    /// Empty set means "run all phases".
    pub phases: HashSet<Phase>,
    pub dry_run: bool,
    pub verbose: bool,
}

#[derive(Debug, Default)]
pub struct ConsolidateStats {
    pub reindex: reindex::ReindexStats,
    pub strengthen: strengthen::StrengthenStats,
    pub reorganize: reorganize::ReorganizeStats,
    pub prune: prune::PruneStats,
    pub prune_candidates: Vec<prune::PruneCandidate>,
    pub duration_secs: f64,
}

fn wants(phases: &HashSet<Phase>, p: Phase) -> bool {
    phases.is_empty() || phases.contains(&p)
}

/// Hardcoded fallback source list. Single source of truth — both the CLI's
/// `load_sources()` and the MCP `axel_consolidate` handler call this when
/// `sources.toml` is absent or unparseable.
pub fn default_sources() -> Vec<SourceDir> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/haseeb".to_string());
    vec![
        SourceDir { path: PathBuf::from(format!("{home}/Jawz/mikoshi/Notes/")), name: "mikoshi".into(), priority: Priority::High },
        SourceDir { path: PathBuf::from(format!("{home}/Jawz/data/context/")), name: "context".into(), priority: Priority::High },
        SourceDir { path: PathBuf::from(format!("{home}/Jawz/notes/")), name: "notes".into(), priority: Priority::Medium },
        SourceDir { path: PathBuf::from(format!("{home}/Jawz/slack/diary/")), name: "slack-diary".into(), priority: Priority::Low },
        SourceDir { path: PathBuf::from(format!("{home}/Jawz/data/context/memories/permanent/")), name: "memories-legacy".into(), priority: Priority::Medium },
        SourceDir { path: PathBuf::from(format!("{home}/.stelline/memkoshi/exports/")), name: "memories".into(), priority: Priority::Medium },
    ]
}

/// Resolve sources from (in priority order):
///   1. `override_path` (e.g. `--sources` CLI flag)
///   2. `$AXEL_SOURCES`
///   3. `~/.config/axel/sources.toml`
///   4. `default_sources()` fallback
pub fn load_sources(override_path: Option<&Path>) -> Vec<SourceDir> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/haseeb".to_string());

    let config_path = override_path
        .map(PathBuf::from)
        .or_else(|| std::env::var("AXEL_SOURCES").ok().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(format!("{home}/.config/axel/sources.toml")));

    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(parsed) = content.parse::<toml::Value>() {
                if let Some(sources) = parsed.get("source").and_then(|v| v.as_array()) {
                    let mut result = Vec::new();
                    for src in sources {
                        let name = src.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let path_str = src.get("path").and_then(|v| v.as_str()).unwrap_or("");
                        let priority_str = src.get("priority").and_then(|v| v.as_str()).unwrap_or("medium");
                        let expanded = path_str.replace("~", &home);
                        let priority = match priority_str {
                            "high" => Priority::High,
                            "low" => Priority::Low,
                            _ => Priority::Medium,
                        };
                        result.push(SourceDir {
                            path: PathBuf::from(expanded),
                            name: name.to_string(),
                            priority,
                        });
                    }
                    if !result.is_empty() {
                        return result;
                    }
                }
            }
        }
    }

    default_sources()
}

/// Run a consolidation pass.
pub fn consolidate(search: &mut BrainSearch, opts: &ConsolidateOptions) -> Result<ConsolidateStats> {
    let started_at = Utc::now().to_rfc3339();
    let start = Instant::now();
    let mut stats = ConsolidateStats::default();
    let mut partial_failure = false;

    // In-progress marker (skip in dry-run). Crash leaves finished_at = NULL.
    let log_id: Option<i64> = if !opts.dry_run {
        match search.db().start_consolidation_log(&started_at) {
            Ok(id) => Some(id),
            Err(e) => {
                eprintln!("⚠ failed to insert consolidation_log start row: {e}");
                None
            }
        }
    } else {
        None
    };

    // Phase 1: Reindex — walk source dirs, re-embed deltas, prune deleted.
    if wants(&opts.phases, Phase::Reindex) {
        let mut sources: Vec<&SourceDir> = opts.sources.iter().collect();
        sources.sort_by_key(|s| match s.priority {
            Priority::High => 0,
            Priority::Medium => 1,
            Priority::Low => 2,
        });

        for src in sources {
            if opts.verbose {
                eprintln!("⟳ reindex [{}] ({})", src.name, src.path.display());
            }
            match reindex::reindex_source(search, src, opts.dry_run) {
                Ok(s) => {
                    stats.reindex.checked += s.checked;
                    stats.reindex.reindexed += s.reindexed;
                    stats.reindex.new_files += s.new_files;
                    stats.reindex.pruned += s.pruned;
                    stats.reindex.skipped += s.skipped;
                }
                Err(e) => {
                    partial_failure = true;
                    eprintln!("⚠ reindex source [{}] failed: {e}", src.name);
                }
            }
        }
    }

    // Phase 2: Strengthen — reconsolidation from access log.
    if wants(&opts.phases, Phase::Strengthen) {
        if opts.verbose { eprintln!("⟳ strengthen"); }
        match strengthen::strengthen(search, opts.dry_run, opts.verbose) {
            Ok(s) => stats.strengthen = s,
            Err(e) => {
                partial_failure = true;
                eprintln!("⚠ strengthen phase failed: {e}");
            }
        }
    }

    // Phase 3: Reorganize — co-retrieval graph maintenance.
    if wants(&opts.phases, Phase::Reorganize) {
        if opts.verbose { eprintln!("⟳ reorganize"); }
        match reorganize::reorganize(search, opts.dry_run) {
            Ok(s) => stats.reorganize = s,
            Err(e) => {
                partial_failure = true;
                eprintln!("⚠ reorganize phase failed: {e}");
            }
        }
    }

    // Phase 4: Prune — decay + flag stale docs.
    if wants(&opts.phases, Phase::Prune) {
        if opts.verbose { eprintln!("⟳ prune"); }
        let priorities: HashMap<String, Priority> = opts
            .sources
            .iter()
            .map(|s| (s.name.clone(), s.priority))
            .collect();
        match prune::prune_with_priorities(search, &priorities, opts.dry_run) {
            Ok((pstats, candidates)) => {
                stats.prune = pstats;
                stats.prune_candidates = candidates;
            }
            Err(e) => {
                partial_failure = true;
                eprintln!("⚠ prune phase failed: {e}");
            }
        }
    }

    // Bounded retention — co_retrieval / document_access grow without limit
    // otherwise. 90 days matches the longest decay window upstream.
    if !opts.dry_run {
        let conn = search.db().conn();
        let _ = conn.execute(
            "DELETE FROM co_retrieval WHERE timestamp < datetime('now', '-90 days')",
            [],
        );
        let _ = conn.execute(
            "DELETE FROM document_access WHERE timestamp < datetime('now', '-90 days')",
            [],
        );
    }

    let duration = start.elapsed().as_secs_f64();
    stats.duration_secs = duration;

    // Always write the audit log (skip in dry-run).
    if !opts.dry_run {
        let entry = ConsolidationLogEntry {
            started_at: started_at.clone(),
            finished_at: Some(Utc::now().to_rfc3339()),
            phase1_reindexed: stats.reindex.reindexed as i64,
            phase1_pruned: stats.reindex.pruned as i64,
            phase2_boosted: stats.strengthen.boosted as i64,
            phase2_decayed: stats.strengthen.decayed as i64,
            phase3_edges_added: stats.reorganize.edges_added as i64,
            phase3_edges_updated: stats.reorganize.edges_updated as i64,
            phase4_flagged: stats.prune.flagged as i64,
            phase4_removed: stats.prune.removed as i64,
            duration_secs: Some(duration),
        };
        let result = match log_id {
            Some(id) => search.db().update_consolidation_log(id, &entry),
            None => search.db().insert_consolidation_log(&entry),
        };
        if let Err(e) = result {
            eprintln!("⚠ failed to write consolidation_log: {e}");
        }
    }

    if partial_failure {
        eprintln!("⚠ consolidation completed with partial failures");
    }

    Ok(stats)
}
