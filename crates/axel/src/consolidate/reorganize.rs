//! Phase 3 — Reorganize. Co-retrieval graph maintenance.
//!
//! Documents that repeatedly co-appear in search results get edges. Edges
//! that aren't reinforced decay. Hebbian wiring for the brain.
//!
//! Spec: `docs/CONSOLIDATION.md` Phase 3.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use rusqlite::params;
use serde_json::json;

use crate::error::Result;
use crate::search::BrainSearch;

/// Stats reported by a Phase 3 pass.
///
/// Field names track the consolidation_log columns (`edges_added`,
/// `edges_updated`) so mod.rs can wire them straight through.
#[derive(Debug, Default, Clone)]
pub struct ReorganizeStats {
    /// Pairs from `co_retrieval` that crossed the threshold this pass.
    pub co_retrieval_pairs: usize,
    /// New `co_retrieved` edges created.
    pub edges_added: usize,
    /// Existing `co_retrieved` edges whose weight was bumped.
    pub edges_updated: usize,
    /// Stale edges whose weight was reduced.
    pub edges_decayed: usize,
    /// Stale edges invalidated (weight fell below removal threshold).
    pub edges_removed: usize,
}

const CO_RETRIEVAL_THRESHOLD: i64 = 3;
const EDGE_REMOVAL_THRESHOLD: f64 = 0.2;
const EDGE_DECAY_FACTOR: f64 = 0.8;
const STALE_DAYS: i64 = 30;

/// Deterministic id for the co_retrieved edge between two doc_ids. Order-
/// independent so swapping a/b yields the same id.
pub fn coret_edge_id(a: &str, b: &str) -> String {
    let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
    let mut h = DefaultHasher::new();
    lo.hash(&mut h);
    "::".hash(&mut h);
    hi.hash(&mut h);
    format!("coret_{:x}", h.finish())
}

pub fn reorganize(search: &BrainSearch, dry_run: bool) -> Result<ReorganizeStats> {
    let mut stats = ReorganizeStats::default();
    let db = search.db();
    let conn = db.conn();

    // 1. Window start = last consolidation finished_at, else epoch.
    let since: String = db
        .last_consolidation_time()?
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string());

    // 2. Aggregate co_retrieval pairs since `since`.
    //    `co_retrieval` may not exist on older brains — degrade gracefully.
    let pairs: Vec<(String, String, i64)> = match conn.prepare(
        "SELECT doc_id_a, doc_id_b, COUNT(*) as co_count
         FROM co_retrieval
         WHERE timestamp > ?1
         GROUP BY doc_id_a, doc_id_b
         HAVING COUNT(*) >= ?2",
    ) {
        Ok(mut stmt) => {
            let rows = stmt.query_map(params![since, CO_RETRIEVAL_THRESHOLD], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?
        }
        Err(_) => return Ok(stats),
    };

    stats.co_retrieval_pairs = pairs.len();

    // 3. Upsert co_retrieved edges for each surviving pair.
    for (a, b, count) in &pairs {
        let edge_id = coret_edge_id(a, b);
        let bump = (*count as f64) / 10.0;

        // Existing live edge (either direction)?
        let existing: Option<(String, f64)> = conn
            .query_row(
                "SELECT id, weight FROM edges
                 WHERE type = 'co_retrieved'
                   AND ((source_id = ?1 AND target_id = ?2)
                        OR (source_id = ?2 AND target_id = ?1))
                   AND (valid_to IS NULL OR valid_to > datetime('now'))
                 LIMIT 1",
                params![a, b],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?)),
            )
            .ok();

        if dry_run {
            if existing.is_some() {
                stats.edges_updated += 1;
            } else {
                stats.edges_added += 1;
            }
            continue;
        }

        match existing {
            Some((id, w)) => {
                let new_w = (w + bump).min(1.0);
                conn.execute(
                    "UPDATE edges SET weight = ?1 WHERE id = ?2",
                    params![new_w, id],
                )?;
                stats.edges_updated += 1;
            }
            None => {
                let weight = bump.min(1.0);
                db.insert_edge(
                    &edge_id,
                    a,
                    b,
                    "co_retrieved",
                    weight,
                    0.7, // co-retrieval is a soft signal
                    &json!({"source": "consolidation_reorganize"}),
                    None,
                    None,
                    None,
                )?;
                stats.edges_added += 1;
            }
        }
    }

    // 4. Decay stale co_retrieved edges — those with no recent reinforcement.
    // Pre-load all recently reinforced pairs to avoid N+1 queries.
    let recent_pairs: std::collections::HashSet<(String, String)> = {
        let mut set = std::collections::HashSet::new();
        if let Ok(mut stmt) = conn.prepare(
            "SELECT DISTINCT doc_id_a, doc_id_b FROM co_retrieval
             WHERE timestamp > datetime('now', ?1)",
        ) {
            if let Ok(rows) = stmt.query_map(params![format!("-{} days", STALE_DAYS)], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            }) {
                for row in rows.flatten() {
                    // Store both orderings for fast lookup
                    set.insert((row.0.clone(), row.1.clone()));
                    set.insert((row.1, row.0));
                }
            }
        }
        set
    };

    let live_edges: Vec<(String, String, String, f64)> = {
        let mut stmt = conn.prepare(
            "SELECT id, source_id, target_id, weight FROM edges
             WHERE type = 'co_retrieved'
               AND (valid_to IS NULL OR valid_to > datetime('now'))",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, f64>(3)?,
            ))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };

    for (id, src, tgt, weight) in live_edges {
        // Check pre-loaded set instead of per-edge query
        if recent_pairs.contains(&(src.clone(), tgt.clone())) {
            continue; // freshly reinforced — leave alone
        }

        if weight < EDGE_REMOVAL_THRESHOLD {
            if !dry_run {
                db.invalidate_edge(&id)?;
            }
            stats.edges_removed += 1;
        } else {
            let new_w = weight * EDGE_DECAY_FACTOR;
            if !dry_run {
                conn.execute(
                    "UPDATE edges SET weight = ?1 WHERE id = ?2",
                    params![new_w, id],
                )?;
            }
            stats.edges_decayed += 1;
        }
    }

    Ok(stats)
}
