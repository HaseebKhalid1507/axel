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

use crate::error::{AxelError, Result};
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

/// The user's home folder: `$HOME`, when it is set and not empty. Nothing
/// else is tried; a caller with no home gets an error, not a guessed path.
fn home_from_env() -> Option<String> {
    std::env::var("HOME").ok().filter(|home| !home.is_empty())
}

fn no_home(what: &str) -> AxelError {
    AxelError::Other(format!("HOME is not set: cannot resolve {what}"))
}

/// Hardcoded fallback source list. Single source of truth — both the CLI's
/// `load_sources()` and the MCP `axel_consolidate` handler call this when
/// `sources.toml` is absent or unparseable.
///
/// The folders are under `$HOME`; without it this is an error.
pub fn default_sources() -> Result<Vec<SourceDir>> {
    default_sources_in(home_from_env().as_deref())
}

fn default_sources_in(home: Option<&str>) -> Result<Vec<SourceDir>> {
    let home = home.ok_or_else(|| no_home("the default source folders"))?;
    Ok(vec![
        SourceDir { path: PathBuf::from(format!("{home}/Jawz/mikoshi/Notes/")), name: "mikoshi".into(), priority: Priority::High },
        SourceDir { path: PathBuf::from(format!("{home}/Jawz/data/context/")), name: "context".into(), priority: Priority::High },
        SourceDir { path: PathBuf::from(format!("{home}/Jawz/notes/")), name: "notes".into(), priority: Priority::Medium },
        SourceDir { path: PathBuf::from(format!("{home}/Jawz/slack/diary/")), name: "slack-diary".into(), priority: Priority::Low },
        SourceDir { path: PathBuf::from(format!("{home}/Jawz/data/context/memories/permanent/")), name: "memories-legacy".into(), priority: Priority::Medium },
        SourceDir { path: PathBuf::from(format!("{home}/.stelline/memkoshi/exports/")), name: "memories".into(), priority: Priority::Medium },
    ])
}

/// Resolve sources from (in priority order):
///   1. `override_path` (e.g. `--sources` CLI flag)
///   2. `$AXEL_SOURCES`
///   3. `~/.config/axel/sources.toml`
///   4. `default_sources()` fallback
///
/// `$HOME` is needed for 3, for 4 and for a `~` in a configured path. When
/// it is needed and not set, this is an error: no path is guessed.
pub fn load_sources(override_path: Option<&Path>) -> Result<Vec<SourceDir>> {
    let config_path = override_path
        .map(PathBuf::from)
        .or_else(|| std::env::var("AXEL_SOURCES").ok().map(PathBuf::from));
    load_sources_from(config_path, home_from_env().as_deref())
}

/// [`load_sources`] with the explicit config path (1 or 2) and the home
/// folder given by the caller.
fn load_sources_from(config_path: Option<PathBuf>, home: Option<&str>) -> Result<Vec<SourceDir>> {
    let config_path =
        config_path.or_else(|| home.map(|home| PathBuf::from(format!("{home}/.config/axel/sources.toml"))));

    if let Some(config_path) = config_path.filter(|path| path.exists()) {
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(parsed) = content.parse::<toml::Value>() {
                if let Some(sources) = parsed.get("source").and_then(|v| v.as_array()) {
                    let mut result = Vec::new();
                    for src in sources {
                        let name = src.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");
                        let path_str = src.get("path").and_then(|v| v.as_str()).unwrap_or("");
                        let priority_str = src.get("priority").and_then(|v| v.as_str()).unwrap_or("medium");
                        let expanded = if path_str.contains('~') {
                            let home = home.ok_or_else(|| {
                                no_home(&format!("`{path_str}` in {}", config_path.display()))
                            })?;
                            path_str.replace('~', home)
                        } else {
                            path_str.to_string()
                        };
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
                        return Ok(result);
                    }
                }
            }
        }
    }

    default_sources_in(home)
}

#[cfg(test)]
mod source_tests {
    use super::*;

    // The home folder is passed in: no test changes the process environment.

    #[test]
    fn without_a_home_the_default_sources_are_an_error_not_a_guessed_path() {
        let error = default_sources_in(None).unwrap_err().to_string();
        assert!(error.contains("HOME is not set"), "{error}");
        let error = load_sources_from(None, None).unwrap_err().to_string();
        assert!(error.contains("HOME is not set"), "{error}");
    }

    #[test]
    fn the_default_sources_are_under_the_given_home() {
        let sources = default_sources_in(Some("/somewhere/else")).unwrap();
        assert_eq!(sources.len(), 6);
        assert!(sources.iter().all(|source| source.path.starts_with("/somewhere/else")));
    }

    #[test]
    fn a_config_file_needs_a_home_only_for_a_tilde() {
        let dir = tempfile::tempdir().unwrap();
        let absolute = dir.path().join("absolute.toml");
        std::fs::write(&absolute, "[[source]]\nname = \"notes\"\npath = \"/srv/notes\"\npriority = \"high\"\n").unwrap();
        let sources = load_sources_from(Some(absolute), None).unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].path, PathBuf::from("/srv/notes"));
        assert_eq!(sources[0].priority, Priority::High);

        let tilde = dir.path().join("tilde.toml");
        std::fs::write(&tilde, "[[source]]\nname = \"notes\"\npath = \"~/notes\"\n").unwrap();
        let error = load_sources_from(Some(tilde.clone()), None).unwrap_err().to_string();
        assert!(error.contains("HOME is not set") && error.contains("~/notes"), "{error}");
        let sources = load_sources_from(Some(tilde), Some("/somewhere/else")).unwrap();
        assert_eq!(sources[0].path, PathBuf::from("/somewhere/else/notes"));
    }
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
use std::time::Duration;

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

use crate::project_memory::{MemoryScope, Retention, Sensitivity};

/// Closed opt-in switch for `memory.auto_consolidate`. Deserializing an
/// absent value is therefore fail-closed (`off`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AutoConsolidate {
    #[default]
    Off,
    On,
}

/// Hard limits for one project-memory maintenance transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectConsolidationConfig {
    pub auto_consolidate: AutoConsolidate,
    /// Maximum rows examined and maximum live rows materialized for reindexing.
    pub max_records: usize,
    pub max_runtime: Duration,
}

impl Default for ProjectConsolidationConfig {
    fn default() -> Self {
        Self {
            auto_consolidate: AutoConsolidate::Off,
            max_records: 256,
            max_runtime: Duration::from_millis(100),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectConsolidationOutcome {
    Off,
    Completed,
    Cancelled,
    TimeBudgetExhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectConsolidationReport {
    pub outcome: ProjectConsolidationOutcome,
    pub examined: usize,
    pub pruned: usize,
    pub indexed: usize,
}

impl ProjectConsolidationReport {
    fn off() -> Self {
        Self {
            outcome: ProjectConsolidationOutcome::Off,
            examined: 0,
            pruned: 0,
            indexed: 0,
        }
    }
}

/// Cooperative cancellation boundary. Implementations must be fast and must
/// not mutate consolidation state.
pub trait ConsolidationCancellation {
    fn is_cancelled(&self) -> bool;
}

pub struct NeverCancel;

impl ConsolidationCancellation for NeverCancel {
    fn is_cancelled(&self) -> bool {
        false
    }
}

fn sensitivity_subset(output: Sensitivity, input: Sensitivity) -> bool {
    output == input || (output == Sensitivity::Secret && input == Sensitivity::Normal)
}

fn retention_disclosure_rank(retention: Retention) -> u8 {
    match retention {
        Retention::Standard => 4,
        Retention::VisibleAfterConsent => 3,
        Retention::LocalOnly => 2,
        Retention::PersistNeverTransmit => 1,
        Retention::NeverPersist => 0,
    }
}

/// Whether an output scope is no more project- or disclosure-eligible than an
/// input scope. Provenance does not grant scope and is intentionally ignored.
pub fn scope_is_subset(output: &MemoryScope, input: &MemoryScope) -> bool {
    output.project_key == input.project_key
        && sensitivity_subset(output.sensitivity, input.sensitivity)
        && retention_disclosure_rank(output.retention) <= retention_disclosure_rank(input.retention)
}

/// Compute a conservative scope for a derived record. Mixing project scopes is
/// rejected; disclosure is the intersection (most restrictive input policy).
pub fn consolidation_output_scope(inputs: &[MemoryScope]) -> Option<MemoryScope> {
    let first = inputs.first()?;
    if inputs
        .iter()
        .any(|scope| scope.project_key != first.project_key)
    {
        return None;
    }
    let sensitivity = if inputs
        .iter()
        .any(|scope| scope.sensitivity == Sensitivity::Secret)
    {
        Sensitivity::Secret
    } else {
        Sensitivity::Normal
    };
    let retention = inputs
        .iter()
        .map(|scope| scope.retention)
        .min_by_key(|retention| retention_disclosure_rank(*retention))?;
    let output = MemoryScope {
        project_key: first.project_key.clone(),
        sensitivity,
        retention,
        provenance: "axel:consolidation".to_owned(),
    };
    inputs
        .iter()
        .all(|input| scope_is_subset(&output, input))
        .then_some(output)
}

fn bounded_outcome(
    cancel: &dyn ConsolidationCancellation,
    started: Instant,
    max_runtime: Duration,
) -> Option<ProjectConsolidationOutcome> {
    if cancel.is_cancelled() {
        Some(ProjectConsolidationOutcome::Cancelled)
    } else if started.elapsed() >= max_runtime {
        Some(ProjectConsolidationOutcome::TimeBudgetExhausted)
    } else {
        None
    }
}

/// Run bounded project-memory maintenance. Every mutation is one transaction;
/// cancellation or budget exhaustion rolls it all back. Tombstones are never
/// deleted and both prune and index SQL exclude them defensively.
pub fn consolidate_project_memory(
    conn: &Connection,
    project_key: &str,
    config: &ProjectConsolidationConfig,
    cancel: &dyn ConsolidationCancellation,
) -> Result<ProjectConsolidationReport> {
    if config.auto_consolidate == AutoConsolidate::Off {
        return Ok(ProjectConsolidationReport::off());
    }
    if project_key.trim().is_empty() {
        return Err(AxelError::Memory(
            "consolidate: empty project_key — refusing unscoped maintenance".into(),
        ));
    }
    let started = Instant::now();
    if let Some(outcome) = bounded_outcome(cancel, started, config.max_runtime) {
        return Ok(ProjectConsolidationReport {
            outcome,
            examined: 0,
            pruned: 0,
            indexed: 0,
        });
    }
    let limit = config.max_records.min(i64::MAX as usize);
    if limit == 0 {
        return Ok(ProjectConsolidationReport {
            outcome: ProjectConsolidationOutcome::Completed,
            examined: 0,
            pruned: 0,
            indexed: 0,
        });
    }

    conn.execute_batch("BEGIN IMMEDIATE")?;
    let run = (|| -> Result<ProjectConsolidationReport> {
        // IDs only: memory remains O(max_records), independent of table size.
        let mut statement = conn.prepare(
            "SELECT id, expires_at FROM memories
              WHERE project_key = ?1
                AND id NOT IN (SELECT id FROM memory_tombstones)
              ORDER BY CASE WHEN expires_at IS NOT NULL AND expires_at <= ?2 THEN 0 ELSE 1 END,
                       created, id
              LIMIT ?3",
        )?;
        let now = Utc::now().to_rfc3339();
        let rows = statement.query_map(params![project_key, now, limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
        })?;
        let mut ids = Vec::with_capacity(limit);
        for row in rows {
            if bounded_outcome(cancel, started, config.max_runtime).is_some() {
                return Err(AxelError::Memory("consolidation interrupted".into()));
            }
            ids.push(row?);
        }
        drop(statement);

        let mut pruned = 0;
        for (id, expires_at) in &ids {
            if bounded_outcome(cancel, started, config.max_runtime).is_some() {
                return Err(AxelError::Memory("consolidation interrupted".into()));
            }
            if expires_at
                .as_deref()
                .is_some_and(|expiry| expiry <= now.as_str())
            {
                pruned += conn.execute(
                    "DELETE FROM memories
                      WHERE id = ?1 AND project_key = ?2
                        AND id NOT IN (SELECT id FROM memory_tombstones)",
                    params![id, project_key],
                )?;
            }
        }

        if bounded_outcome(cancel, started, config.max_runtime).is_some() {
            return Err(AxelError::Memory("consolidation interrupted".into()));
        }
        conn.execute(
            "DELETE FROM memories_fts
              WHERE mem_id IN (
                SELECT id FROM memories WHERE project_key = ?1
                UNION
                SELECT id FROM memory_tombstones WHERE project_key = ?1
              )",
            [project_key],
        )?;
        let indexed = conn.execute(
            "INSERT INTO memories_fts (mem_id, title, abstract, content)
             SELECT m.id, m.title,
                    CASE WHEN m.sensitivity = 'secret' THEN '' ELSE m.abstract_text END,
                    CASE WHEN m.sensitivity = 'secret' THEN '' ELSE m.content END
               FROM memories m
              WHERE m.project_key = ?1
                AND m.id NOT IN (SELECT id FROM memory_tombstones)
              ORDER BY m.created DESC, m.id
              LIMIT ?2",
            params![project_key, limit as i64],
        )?;
        if bounded_outcome(cancel, started, config.max_runtime).is_some() {
            return Err(AxelError::Memory("consolidation interrupted".into()));
        }

        Ok(ProjectConsolidationReport {
            outcome: ProjectConsolidationOutcome::Completed,
            examined: ids.len(),
            pruned,
            indexed,
        })
    })();

    match run {
        Ok(report) => {
            conn.execute_batch("COMMIT")?;
            Ok(report)
        }
        Err(AxelError::Memory(message)) if message == "consolidation interrupted" => {
            let _ = conn.execute_batch("ROLLBACK");
            let outcome = bounded_outcome(cancel, started, config.max_runtime)
                .unwrap_or(ProjectConsolidationOutcome::Cancelled);
            Ok(ProjectConsolidationReport {
                outcome,
                examined: 0,
                pruned: 0,
                indexed: 0,
            })
        }
        Err(error) => {
            let _ = conn.execute_batch("ROLLBACK");
            Err(error)
        }
    }
}

#[cfg(test)]
mod project_memory_tests {
    use std::cell::Cell;
    use std::time::Duration;

    use chrono::{Duration as ChronoDuration, Utc};
    use rusqlite::{params, Connection};

    use super::*;
    use crate::project_memory::{self, MemoryScope, Retention, Sensitivity};

    fn database() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                abstract_text TEXT NOT NULL,
                content TEXT NOT NULL,
                created TEXT NOT NULL,
                updated TEXT,
                project_key TEXT,
                sensitivity TEXT NOT NULL DEFAULT 'normal',
                retention TEXT NOT NULL DEFAULT 'standard',
                provenance TEXT NOT NULL DEFAULT '',
                expires_at TEXT,
                superseded_by TEXT,
                supersedes TEXT NOT NULL DEFAULT '[]',
                contradicts TEXT NOT NULL DEFAULT '[]',
                confirmed_by TEXT NOT NULL DEFAULT '[]',
                invalidated_at TEXT
            );",
        )
        .unwrap();
        project_memory::migrate(&conn).unwrap();
        conn
    }

    fn insert_memory(
        conn: &Connection,
        id: &str,
        project: &str,
        sensitivity: &str,
        retention: &str,
        expires_at: Option<&str>,
    ) {
        conn.execute(
            "INSERT INTO memories
                 (id, title, abstract_text, content, created, project_key,
                  sensitivity, retention, expires_at)
             VALUES (?1, ?2, ?2, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                id,
                format!("memory {id}"),
                Utc::now().to_rfc3339(),
                project,
                sensitivity,
                retention,
                expires_at,
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memories_fts (mem_id, title, abstract, content)
             VALUES (?1, ?2, ?2, ?2)",
            params![id, format!("memory {id}")],
        )
        .unwrap();
    }

    fn enabled(max_records: usize) -> ProjectConsolidationConfig {
        ProjectConsolidationConfig {
            auto_consolidate: AutoConsolidate::On,
            max_records,
            max_runtime: Duration::from_secs(1),
        }
    }

    #[test]
    fn auto_consolidation_is_off_by_default() {
        let conn = database();
        let expired = (Utc::now() - ChronoDuration::days(1)).to_rfc3339();
        insert_memory(
            &conn,
            "expired",
            "project-a",
            "normal",
            "standard",
            Some(&expired),
        );

        let report = consolidate_project_memory(
            &conn,
            "project-a",
            &ProjectConsolidationConfig::default(),
            &NeverCancel,
        )
        .unwrap();

        assert_eq!(report.outcome, ProjectConsolidationOutcome::Off);
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
    }

    #[test]
    fn property_consolidation_output_scope_is_subset_of_every_input_scope() {
        let sensitivities = [Sensitivity::Normal, Sensitivity::Secret];
        let retentions = [
            Retention::Standard,
            Retention::LocalOnly,
            Retention::VisibleAfterConsent,
            Retention::PersistNeverTransmit,
        ];

        for project in ["project-a", "project-b"] {
            for sensitivity in sensitivities {
                for retention in retentions {
                    let input = MemoryScope {
                        project_key: project.to_owned(),
                        sensitivity,
                        retention,
                        provenance: "capture".to_owned(),
                    };
                    let output = consolidation_output_scope(&[input.clone(), input.clone()])
                        .expect("equal input scopes can be consolidated");
                    assert!(scope_is_subset(&output, &input));
                }
            }
        }

        let project_a = MemoryScope {
            project_key: "project-a".to_owned(),
            ..MemoryScope::default()
        };
        let project_b = MemoryScope {
            project_key: "project-b".to_owned(),
            ..MemoryScope::default()
        };
        assert!(consolidation_output_scope(&[project_a, project_b]).is_none());
    }

    #[test]
    fn tombstones_survive_consolidation_and_index_rebuild() {
        let conn = database();
        insert_memory(&conn, "forgotten", "project-a", "normal", "standard", None);
        conn.execute(
            "INSERT INTO memory_tombstones (id, project_key, deleted_at)
             VALUES ('forgotten', 'project-a', ?1)",
            [Utc::now().to_rfc3339()],
        )
        .unwrap();

        let report =
            consolidate_project_memory(&conn, "project-a", &enabled(16), &NeverCancel).unwrap();
        assert_eq!(report.outcome, ProjectConsolidationOutcome::Completed);
        project_memory::rebuild_search_index(&conn).unwrap();

        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM memory_tombstones WHERE id = 'forgotten'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
            1
        );
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM memories_fts WHERE mem_id = 'forgotten'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
            0,
            "a forgotten id must never resurface in the derived index"
        );
    }

    struct CancelAfter {
        remaining_checks: Cell<usize>,
    }

    impl ConsolidationCancellation for CancelAfter {
        fn is_cancelled(&self) -> bool {
            let remaining = self.remaining_checks.get();
            self.remaining_checks.set(remaining.saturating_sub(1));
            remaining == 0
        }
    }

    #[test]
    fn run_is_record_bounded_and_mid_run_cancellation_rolls_back() {
        let conn = database();
        let expired = (Utc::now() - ChronoDuration::days(1)).to_rfc3339();
        for index in 0..10 {
            insert_memory(
                &conn,
                &format!("expired-{index}"),
                "project-a",
                "normal",
                "standard",
                Some(&expired),
            );
        }

        let bounded =
            consolidate_project_memory(&conn, "project-a", &enabled(3), &NeverCancel).unwrap();
        assert!(bounded.examined <= 3);
        assert_eq!(bounded.pruned, 3);
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            7
        );

        let before: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .unwrap();
        let cancelled = consolidate_project_memory(
            &conn,
            "project-a",
            &enabled(7),
            &CancelAfter {
                remaining_checks: Cell::new(4),
            },
        )
        .unwrap();
        assert_eq!(cancelled.outcome, ProjectConsolidationOutcome::Cancelled);
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            before,
            "a cancelled run must roll back every partial mutation"
        );

        let timed_out = consolidate_project_memory(
            &conn,
            "project-a",
            &ProjectConsolidationConfig {
                max_runtime: Duration::ZERO,
                ..enabled(7)
            },
            &NeverCancel,
        )
        .unwrap();
        assert_eq!(
            timed_out.outcome,
            ProjectConsolidationOutcome::TimeBudgetExhausted
        );
    }
}
