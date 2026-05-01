//! Phase 4 — Prune. Decay-driven cleanup. Arc-style contrast enhancement.
//!
//! Auto-removes only documents that meet strict criteria from low-priority
//! sources. Everything else is flagged for human review.
//!
//! Spec: `docs/CONSOLIDATION.md` Phase 4.

use std::collections::HashMap;

use rusqlite::params;

use super::Priority;
use crate::error::Result;
use crate::search::BrainSearch;

/// Stats reported by a Phase 4 pass.
///
/// Field names track the consolidation_log columns (`removed`, `flagged`).
#[derive(Debug, Default, Clone)]
pub struct PruneStats {
    /// Met all auto-remove criteria → deleted from the index.
    pub removed: usize,
    /// Met staleness criteria but came from non-Low source → human review.
    pub flagged: usize,
    /// Misaligned-embedding candidates (>=5 hits, avg score < 0.015).
    pub misaligned: usize,
}

#[derive(Debug, Clone)]
pub struct PruneCandidate {
    pub doc_id: String,
    pub reason: String,
    pub excitability: f64,
    pub access_count: i64,
    pub age_days: i64,
}

/// Extract the source name from a doc_id of the form `source::path::...`.
fn source_of(doc_id: &str) -> &str {
    match doc_id.find("::") {
        Some(i) => &doc_id[..i],
        None => doc_id,
    }
}

/// Default prune entry point used by `consolidate::consolidate()`. Without
/// source priorities, nothing is auto-removed — every stale document is
/// flagged for review (the safe default).
pub fn prune(search: &BrainSearch, dry_run: bool) -> Result<PruneStats> {
    let priorities: HashMap<String, Priority> = HashMap::new();
    let (stats, _candidates) = prune_with_priorities(search, &priorities, dry_run)?;
    Ok(stats)
}

/// Spec'd prune. Use this when source priorities are available (e.g. when
/// the CLI threads `opts.sources` through). Returns the candidate list so
/// the caller can render a `--report`.
pub fn prune_with_priorities(
    search: &BrainSearch,
    source_priorities: &HashMap<String, Priority>,
    dry_run: bool,
) -> Result<(PruneStats, Vec<PruneCandidate>)> {
    let mut stats = PruneStats::default();
    let mut candidates: Vec<PruneCandidate> = Vec::new();

    let db = search.db();
    let conn = db.conn();

    // 1. Stale documents: low excitability, never accessed, past grace period.
    //    `documents` columns may differ on older brains; degrade gracefully.
    let dead: Vec<(String, Option<String>, f64, i64, i64)> = match conn.prepare(
        "SELECT doc_id, file_path, excitability, access_count,
                CAST(julianday('now') - julianday(created) AS INTEGER) as age_days
         FROM documents
         WHERE excitability < 0.15
           AND access_count = 0
           AND CAST(julianday('now') - julianday(created) AS INTEGER) > 60",
    ) {
        Ok(mut stmt) => {
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, f64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        }
        Err(_) => Vec::new(),
    };

    for (doc_id, file_path, excitability, access_count, age_days) in dead {
        let source = source_of(&doc_id).to_string();
        let is_low = matches!(source_priorities.get(&source), Some(Priority::Low));

        if is_low {
            // Auto-remove. Prefer file-path delete (cascades chunks); fall
            // back to a direct row delete when file_path is missing.
            if !dry_run {
                if let Some(fp) = file_path.as_deref() {
                    db.delete_documents_by_file(fp)?;
                } else {
                    conn.execute(
                        "DELETE FROM documents WHERE doc_id = ?1",
                        params![doc_id],
                    )?;
                }
            }
            stats.removed += 1;
        } else {
            candidates.push(PruneCandidate {
                doc_id,
                reason: "stale_low_excitability".to_string(),
                excitability,
                access_count,
                age_days,
            });
            stats.flagged += 1;
        }
    }

    // 2. Misaligned embeddings — chronically poor search scores. Arc analog.
    let misaligned: Vec<(String, i64, f64)> = match conn.prepare(
        "SELECT da.doc_id, COUNT(*) as hits, AVG(da.score) as avg_score
         FROM document_access da
         WHERE da.access_type = 'search_hit'
         GROUP BY da.doc_id
         HAVING hits >= 5 AND avg_score < 0.015",
    ) {
        Ok(mut stmt) => {
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, f64>(2).unwrap_or(0.0),
                ))
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        }
        Err(_) => Vec::new(),
    };

    for (doc_id, hits, _avg_score) in misaligned {
        // Best-effort enrichment for the report.
        let meta: Option<(f64, i64, i64)> = conn
            .query_row(
                "SELECT excitability, access_count,
                        CAST(julianday('now') - julianday(created) AS INTEGER)
                 FROM documents WHERE doc_id = ?1",
                params![doc_id],
                |row| {
                    Ok((
                        row.get::<_, f64>(0).unwrap_or(0.5),
                        row.get::<_, i64>(1).unwrap_or(hits),
                        row.get::<_, i64>(2).unwrap_or(0),
                    ))
                },
            )
            .ok();

        let (excitability, access_count, age_days) = meta.unwrap_or((0.5, hits, 0));
        candidates.push(PruneCandidate {
            doc_id,
            reason: "misaligned_embedding".to_string(),
            excitability,
            access_count,
            age_days,
        });
        stats.misaligned += 1;
    }

    Ok((stats, candidates))
}
