//! Phase 2: Strengthen — retrieval-based reconsolidation.
//!
//! Boost excitability for accessed docs; decay untouched ones. Mirrors the
//! biological model where each retrieval briefly destabilizes a memory and
//! either strengthens it (action) or signals extinction (no follow-through).

use std::collections::HashMap;

use rusqlite::params;

use crate::error::{AxelError, Result};
use crate::search::BrainSearch;

#[derive(Debug, Default, Clone)]
pub struct StrengthenStats {
    pub boosted: usize,
    pub decayed: usize,
    pub extinction_signals: usize,
}

const BOOST_SCALE: f64 = 0.05;
const BOOST_CAP: f64 = 0.2;
const DECAY_RATE_PER_WEEK: f64 = 0.02;
const DECAY_CAP: f64 = 0.3;
const GRACE_DAYS: f64 = 14.0;
const SCORE_EXTINCTION_THRESHOLD: f64 = 0.02;
const EXTINCTION_PENALTY: f64 = 0.05;
const EXCITABILITY_FLOOR: f64 = 0.1;
const EXCITABILITY_CEILING: f64 = 1.0;

pub fn strengthen(search: &BrainSearch, dry_run: bool) -> Result<StrengthenStats> {
    let mut stats = StrengthenStats::default();
    let db = search.db();

    let since = db
        .last_consolidation_time()
        .map_err(|e| AxelError::Search(format!("last_consolidation_time: {e}")))?
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string());

    let accesses = db
        .get_document_accesses_since(&since)
        .map_err(|e| AxelError::Search(format!("get_document_accesses_since: {e}")))?;

    // Group by doc_id — collect (count, score_sum, score_n).
    let mut grouped: HashMap<String, (usize, f64, usize)> = HashMap::new();
    for a in &accesses {
        let entry = grouped.entry(a.doc_id.clone()).or_insert((0, 0.0, 0));
        entry.0 += 1;
        if let Some(s) = a.score {
            entry.1 += s;
            entry.2 += 1;
        }
    }

    let conn = db.conn();

    // Boost / extinction for accessed docs.
    for (doc_id, (count, score_sum, score_n)) in &grouped {
        let current: f64 = match conn.query_row(
            "SELECT excitability FROM documents WHERE doc_id = ?1",
            params![doc_id],
            |r| r.get::<_, f64>(0),
        ) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let avg_score = if *score_n > 0 { score_sum / (*score_n as f64) } else { 1.0 };

        let new_excitability = if avg_score < SCORE_EXTINCTION_THRESHOLD {
            stats.extinction_signals += 1;
            (current - EXTINCTION_PENALTY).max(EXCITABILITY_FLOOR)
        } else {
            stats.boosted += 1;
            let boost = (BOOST_SCALE * ((*count as f64) + 1.0).ln()).min(BOOST_CAP);
            (current + boost).min(EXCITABILITY_CEILING)
        };

        if !dry_run {
            conn.execute(
                "UPDATE documents SET excitability = ?1 WHERE doc_id = ?2",
                params![new_excitability, doc_id],
            ).map_err(|e| AxelError::Search(format!("update excitability: {e}")))?;
        }
    }

    // Decay untouched docs older than the grace period.
    let mut stmt = conn.prepare(
        "SELECT doc_id, excitability,
                COALESCE(
                    (julianday('now') - julianday(last_accessed)),
                    (julianday('now') - julianday(indexed_at)),
                    (julianday('now') - julianday(created)),
                    0
                ) AS days_inactive
         FROM documents
         WHERE (last_accessed IS NULL OR last_accessed < datetime('now', ?1))
           AND (indexed_at  IS NULL OR indexed_at  < datetime('now', ?1))",
    ).map_err(|e| AxelError::Search(format!("prepare decay query: {e}")))?;

    let grace_clause = format!("-{} days", GRACE_DAYS as i64);
    let rows = stmt.query_map(params![grace_clause], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, f64>(1)?,
            row.get::<_, f64>(2)?,
        ))
    }).map_err(|e| AxelError::Search(format!("query decay: {e}")))?;

    let mut updates: Vec<(String, f64)> = Vec::new();
    for r in rows {
        let (doc_id, excitability, days_inactive) = r
            .map_err(|e| AxelError::Search(format!("decay row: {e}")))?;
        if grouped.contains_key(&doc_id) { continue; }
        if days_inactive < GRACE_DAYS { continue; }

        let weeks_inactive = days_inactive / 7.0;
        let penalty = (DECAY_RATE_PER_WEEK * weeks_inactive).min(DECAY_CAP);
        let new_excitability = (excitability - penalty).max(EXCITABILITY_FLOOR);
        if (new_excitability - excitability).abs() < f64::EPSILON { continue; }
        updates.push((doc_id, new_excitability));
    }
    drop(stmt);

    for (doc_id, new_excitability) in updates {
        stats.decayed += 1;
        if !dry_run {
            conn.execute(
                "UPDATE documents SET excitability = ?1 WHERE doc_id = ?2",
                params![new_excitability, doc_id],
            ).map_err(|e| AxelError::Search(format!("decay update: {e}")))?;
        }
    }

    Ok(stats)
}
