//! Pattern detection over the event log.
//!
//! Three families of patterns are surfaced:
//!
//! * **Frequency** — targets repeatedly searched for.
//! * **Knowledge gaps** — searches that consistently return zero results.
//! * **Temporal** — queries clustered on a particular weekday.

use serde::{Deserialize, Serialize};

use crate::storage::MemoryStorage;
use crate::error::Result;

/// Kind of pattern detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PatternType {
    /// Frequently accessed target.
    Frequency,
    /// Searches that consistently return zero results.
    Gap,
    /// Time-of-week clustered behaviour.
    Temporal,
}

/// A detected pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pattern {
    /// Pattern family.
    pub pattern_type: PatternType,
    /// Short identifier (target id, query string, weekday+query, …).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Confidence in `[0.0, 1.0]`, derived from sample size.
    pub confidence: f64,
    /// How many events backed the pattern.
    pub sample_size: usize,
}

/// Runs the SQL queries that surface patterns.
pub struct PatternDetector<'a> {
    storage: &'a MemoryStorage,
}

impl<'a> PatternDetector<'a> {
    /// Construct a detector over `storage`.
    pub fn new(storage: &'a MemoryStorage) -> Self {
        Self { storage }
    }

    /// Run all three detectors and return their union.
    pub fn detect(&self) -> Result<Vec<Pattern>> {
        let mut out = Vec::new();
        out.extend(self.frequency()?);
        out.extend(self.gaps()?);
        out.extend(self.temporal()?);
        Ok(out)
    }

    fn frequency(&self) -> Result<Vec<Pattern>> {
        let conn = self.storage.conn();
        let mut stmt = conn.prepare(
            r#"SELECT target_id, COUNT(*) AS n
                 FROM events
                WHERE event_type = 'search'
                  AND target_id IS NOT NULL
                GROUP BY target_id
               HAVING n >= 3
                ORDER BY n DESC"#,
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (target, n) = r?;
            let n = n as usize;
            out.push(Pattern {
                pattern_type: PatternType::Frequency,
                name: target.clone(),
                description: format!("`{target}` was searched {n} times"),
                confidence: confidence_from_n(n),
                sample_size: n,
            });
        }
        Ok(out)
    }

    fn gaps(&self) -> Result<Vec<Pattern>> {
        let conn = self.storage.conn();
        // `results_count` lives inside the JSON `metadata` column.
        let mut stmt = conn.prepare(
            r#"SELECT query, COUNT(*) AS n
                 FROM events
                WHERE event_type = 'search_complete'
                  AND query IS NOT NULL
                  AND json_extract(metadata, '$.results_count') = 0
                GROUP BY query
               HAVING n >= 3
                ORDER BY n DESC"#,
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (query, n) = r?;
            let n = n as usize;
            out.push(Pattern {
                pattern_type: PatternType::Gap,
                name: query.clone(),
                description: format!(
                    "`{query}` returned zero results {n} times — knowledge gap"
                ),
                confidence: confidence_from_n(n),
                sample_size: n,
            });
        }
        Ok(out)
    }

    fn temporal(&self) -> Result<Vec<Pattern>> {
        let conn = self.storage.conn();
        // strftime('%w', ...) → 0=Sunday … 6=Saturday in SQLite.
        let mut stmt = conn.prepare(
            r#"SELECT strftime('%w', timestamp) AS weekday,
                       query,
                       COUNT(*) AS n
                  FROM events
                 WHERE query IS NOT NULL
                 GROUP BY weekday, query
                HAVING n >= 2
                 ORDER BY n DESC"#,
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for r in rows {
            let (weekday, query, n) = r?;
            let n = n as usize;
            let day = weekday_name(&weekday);
            out.push(Pattern {
                pattern_type: PatternType::Temporal,
                name: format!("{day}:{query}"),
                description: format!("`{query}` recurs on {day} ({n} times)"),
                confidence: confidence_from_n(n),
                sample_size: n,
            });
        }
        Ok(out)
    }
}

fn confidence_from_n(n: usize) -> f64 {
    // Saturating logistic-ish curve: 3 ≈ 0.5, 10 ≈ 0.83, 30 ≈ 0.95.
    let n = n as f64;
    n / (n + 3.0)
}

fn weekday_name(w: &str) -> &'static str {
    match w {
        "0" => "Sunday",
        "1" => "Monday",
        "2" => "Tuesday",
        "3" => "Wednesday",
        "4" => "Thursday",
        "5" => "Friday",
        "6" => "Saturday",
        _ => "Unknown",
    }
}
