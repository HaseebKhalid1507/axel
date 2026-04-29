//! Importance decay and access-driven boosting.
//!
//! A simple recency/frequency heuristic:
//!
//! * Memories accessed within the last 7 days have their `importance`
//!   nudged upward by `min(0.2, 0.05 * log2(access_count + 1))`.
//! * Memories untouched for more than 14 days are penalised by
//!   `0.02 * weeks_inactive`, capped at `-0.3` and floored at `0.1`.

use chrono::{DateTime, Utc};
use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::storage::MemoryStorage;
use crate::error::Result;

/// Counts returned by [`decay_and_boost`].
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct DecayStats {
    /// Number of memories whose importance was raised.
    pub boosted: usize,
    /// Number of memories whose importance was lowered.
    pub decayed: usize,
}

/// Apply the decay/boost pass. Returns counts of affected memories.
pub fn decay_and_boost(storage: &MemoryStorage) -> Result<DecayStats> {
    let now = Utc::now();
    let conn = storage.conn();
    let mut stats = DecayStats::default();

    // Boost: memories accessed in the past 7 days.
    let mut stmt = conn.prepare(
        r#"SELECT memory_id, COUNT(*) AS n, MAX(timestamp) AS last
             FROM memory_access
            GROUP BY memory_id"#,
    )?;
    let rows: Vec<(String, i64, String)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);

    for (memory_id, count, last) in rows {
        let Some(last_ts) = parse(&last) else { continue };
        let days = (now - last_ts).num_days();

        let importance: Option<f64> = conn
            .query_row(
                "SELECT importance FROM memories WHERE id = ?1",
                params![memory_id],
                |r| r.get(0),
            )
            .ok();
        let Some(current) = importance else { continue };

        if days <= 7 {
            let bump = (0.05 * ((count as f64) + 1.0).log2()).min(0.2);
            let new = (current + bump).clamp(0.0, 1.0);
            if (new - current).abs() > f64::EPSILON {
                conn.execute(
                    "UPDATE memories SET importance = ?1, updated = ?2 WHERE id = ?3",
                    params![new, now.to_rfc3339(), memory_id],
                )?;
                stats.boosted += 1;
            }
        } else if days > 14 {
            let weeks = days as f64 / 7.0;
            let penalty = (0.02 * weeks).min(0.3);
            let new = (current - penalty).max(0.1);
            if (new - current).abs() > f64::EPSILON {
                conn.execute(
                    "UPDATE memories SET importance = ?1, updated = ?2 WHERE id = ?3",
                    params![new, now.to_rfc3339(), memory_id],
                )?;
                stats.decayed += 1;
            }
        }
    }

    Ok(stats)
}

fn parse(s: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|t| t.with_timezone(&Utc))
}
