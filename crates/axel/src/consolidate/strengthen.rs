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
#[allow(dead_code)]
const DECAY_RATE_PER_WEEK: f64 = 0.02;
#[allow(dead_code)]
const DECAY_CAP: f64 = 0.3;
const GRACE_DAYS: f64 = 14.0;
const SCORE_EXTINCTION_THRESHOLD: f64 = 0.02;
const EXTINCTION_PENALTY: f64 = 0.05;
const EXCITABILITY_FLOOR: f64 = 0.1;
const EXCITABILITY_CEILING: f64 = 1.0;

pub fn strengthen(search: &BrainSearch, dry_run: bool, verbose: bool) -> Result<StrengthenStats> {
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

    // Pre-load excitabilities for all accessed docs in one query — eliminates
    // the N+1 SELECT (one round-trip per doc) that dominated this phase.
    let doc_ids: Vec<String> = grouped.keys().cloned().collect();
    let mut current_excit: HashMap<String, f64> = HashMap::with_capacity(doc_ids.len());
    if !doc_ids.is_empty() {
        // Chunk to stay under SQLite's parameter limit (default 32766).
        for chunk in doc_ids.chunks(500) {
            let placeholders = std::iter::repeat("?").take(chunk.len()).collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT doc_id, excitability FROM documents WHERE doc_id IN ({placeholders})"
            );
            let mut stmt = conn.prepare(&sql)
                .map_err(|e| AxelError::Search(format!("prepare excitability bulk: {e}")))?;
            let params_vec: Vec<&dyn rusqlite::ToSql> =
                chunk.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
            let rows = stmt.query_map(params_vec.as_slice(), |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
            }).map_err(|e| AxelError::Search(format!("query excitability bulk: {e}")))?;
            for r in rows {
                let (id, v) = r.map_err(|e| AxelError::Search(format!("excit row: {e}")))?;
                current_excit.insert(id, v);
            }
        }
    }

    // Compute new values in memory, then flush in one transaction.
    let mut boost_updates: Vec<(String, f64)> = Vec::with_capacity(grouped.len());

    for (doc_id, (count, score_sum, score_n)) in &grouped {
        let current = match current_excit.get(doc_id) {
            Some(v) => *v,
            None => continue, // doc was deleted between access-log write and now
        };

        let avg_score = if *score_n > 0 { score_sum / (*score_n as f64) } else { 1.0 };

        let new_excitability = if avg_score < SCORE_EXTINCTION_THRESHOLD {
            stats.extinction_signals += 1;
            let new = (current - EXTINCTION_PENALTY).max(EXCITABILITY_FLOOR);
            if verbose {
                eprintln!("  ↘ {doc_id}  {current:.3} → {new:.3}  (extinction, avg_score={avg_score:.4}, {count} hits)");
            }
            new
        } else {
            stats.boosted += 1;
            // log2 to match memkoshi/decay.rs and the spec; ln grew ~1.44× slower.
            let boost = (BOOST_SCALE * ((*count as f64) + 1.0).log2()).min(BOOST_CAP);
            let new = (current + boost).min(EXCITABILITY_CEILING);
            if verbose {
                eprintln!("  ↗ {doc_id}  {current:.3} → {new:.3}  (boost +{boost:.4}, {count} hits)");
            }
            new
        };

        boost_updates.push((doc_id.clone(), new_excitability));
    }

    if !dry_run && !boost_updates.is_empty() {
        let tx = conn.unchecked_transaction()
            .map_err(|e| AxelError::Search(format!("begin tx: {e}")))?;
        {
            let mut stmt = tx.prepare(
                "UPDATE documents SET excitability = ?1 WHERE doc_id = ?2"
            ).map_err(|e| AxelError::Search(format!("prepare boost update: {e}")))?;
            for (doc_id, new_excit) in &boost_updates {
                stmt.execute(params![new_excit, doc_id])
                    .map_err(|e| AxelError::Search(format!("update excitability: {e}")))?;
            }
        }
        tx.commit().map_err(|e| AxelError::Search(format!("commit boost tx: {e}")))?;
    }

    // Decay untouched docs older than the grace period.
    let mut stmt = conn.prepare(
        "SELECT doc_id, excitability, COALESCE(access_count, 0),
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
            row.get::<_, i64>(2)?,
            row.get::<_, f64>(3)?,
        ))
    }).map_err(|e| AxelError::Search(format!("query decay: {e}")))?;

    let mut updates: Vec<(String, f64)> = Vec::new();
    for r in rows {
        let (doc_id, excitability, access_count, days_inactive) = r
            .map_err(|e| AxelError::Search(format!("decay row: {e}")))?;
        if grouped.contains_key(&doc_id) { continue; }
        if days_inactive < GRACE_DAYS { continue; }

        // Ebbinghaus exponential forgetting curve (Murre & Dros, 2015)
        // S = stability, increases with access count
        const S_BASE: f64 = 60.0; // base stability in days (raised from 30 per review — 30 was too aggressive for knowledge notes)
        let stability = S_BASE * (1.0 + (1.0 + access_count as f64).ln());
        let retention = (-days_inactive / stability).exp();
        let new_excitability = (excitability * retention).max(EXCITABILITY_FLOOR);
        if (new_excitability - excitability).abs() < f64::EPSILON { continue; }
        updates.push((doc_id, new_excitability));
    }
    drop(stmt);

    if !dry_run && !updates.is_empty() {
        let tx = conn.unchecked_transaction()
            .map_err(|e| AxelError::Search(format!("begin decay tx: {e}")))?;
        {
            let mut stmt = tx.prepare(
                "UPDATE documents SET excitability = ?1 WHERE doc_id = ?2"
            ).map_err(|e| AxelError::Search(format!("prepare decay update: {e}")))?;
            for (doc_id, new_excitability) in &updates {
                stats.decayed += 1;
                stmt.execute(params![new_excitability, doc_id])
                    .map_err(|e| AxelError::Search(format!("decay update: {e}")))?;
            }
        }
        tx.commit().map_err(|e| AxelError::Search(format!("commit decay tx: {e}")))?;
    } else {
        // dry_run: still count what would have changed
        stats.decayed += updates.len();
    }

    Ok(stats)
}
