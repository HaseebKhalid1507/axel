//! Consolidation engine — biologically-inspired memory lifecycle.
//!
//! Four phases: reindex → strengthen → reorganize → prune.
//! See `docs/CONSOLIDATION.md` for the full spec.

pub mod reindex;
pub mod strengthen;
pub mod reorganize;
pub mod prune;

use std::collections::HashSet;
use std::path::PathBuf;
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
    pub duration_secs: f64,
}

fn wants(phases: &HashSet<Phase>, p: Phase) -> bool {
    phases.is_empty() || phases.contains(&p)
}

/// Run a consolidation pass.
pub fn consolidate(search: &mut BrainSearch, opts: &ConsolidateOptions) -> Result<ConsolidateStats> {
    let started_at = Utc::now().to_rfc3339();
    let start = Instant::now();
    let mut stats = ConsolidateStats::default();

    // Phase 1: Reindex — walk source dirs, re-embed deltas, prune deleted.
    if wants(&opts.phases, Phase::Reindex) {
        // Sort sources by priority (High first).
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
            let s = reindex::reindex_source(search, src, opts.dry_run)?;
            stats.reindex.checked += s.checked;
            stats.reindex.reindexed += s.reindexed;
            stats.reindex.new_files += s.new_files;
            stats.reindex.pruned += s.pruned;
        }
    }

    // Phase 2: Strengthen — reconsolidation from access log.
    if wants(&opts.phases, Phase::Strengthen) {
        if opts.verbose { eprintln!("⟳ strengthen"); }
        stats.strengthen = strengthen::strengthen(search, opts.dry_run)?;
    }

    // Phase 3: Reorganize — co-retrieval graph maintenance.
    if wants(&opts.phases, Phase::Reorganize) {
        if opts.verbose { eprintln!("⟳ reorganize"); }
        stats.reorganize = reorganize::reorganize(search, opts.dry_run)?;
    }

    // Phase 4: Prune — decay + flag stale docs.
    if wants(&opts.phases, Phase::Prune) {
        if opts.verbose { eprintln!("⟳ prune"); }
        stats.prune = prune::prune(search, opts.dry_run)?;
    }

    let duration = start.elapsed().as_secs_f64();
    stats.duration_secs = duration;

    // Audit log entry — skip in dry-run so repeated dry runs stay clean.
    if !opts.dry_run {
        let entry = ConsolidationLogEntry {
            started_at,
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
        search.db().insert_consolidation_log(&entry)
            .map_err(|e| crate::error::AxelError::Search(format!("log insert failed: {e}")))?;
    }

    Ok(stats)
}
