//! `search <query>... [--limit <n>] [--json]`. Commit 18b writes this file.
//!
//! The old four-layer search over `documents`, as before, with two changes (R-8, R3-6c): it
//! asks for three times the limit and drops every hit whose `documents` row is hidden (a
//! file that is gone; the shadow of a hidden or superseded memory) and every hit whose id is
//! that of a hidden or superseded memory, whichever layer returned it (the graph layer reads
//! `nodes`, which have no `documents` row to test); and what it prints passes the screen.
//! The shadow documents of visible memories stay: this is the search the owner's agents use
//! to find his old memories.

use std::collections::HashSet;
use std::process::ExitCode;
use std::time::Instant;

use rusqlite::Connection;
use velocirag::search::SearchResult;

use crate::cli::show::{self, one, Term, FAILED, OK};
use crate::cli::strip_frontmatter;
use crate::error::Result;
use crate::ext::types::BrainArg;
use crate::screen;
use crate::search::BrainSearch;

#[derive(clap::Args)]
pub struct Args {
    /// Search query
    pub query: Vec<String>,
    /// Maximum results to return
    #[arg(long, default_value = "5")]
    pub limit: usize,
    /// Output as JSON for scripting
    #[arg(long)]
    pub json: bool,
}

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    show::on_std(|term| {
        if args.query.join(" ").trim().is_empty() {
            writeln!(term.err, "Please provide a search query.")?;
            return Ok(FAILED);
        }
        let mut search = BrainSearch::open(&brain.path)?;
        run_with(args, &mut search, term)
    })
}

/// At most this many results: `--limit` is capped (to prevent OOM on large brains).
pub fn limit_of(args: &Args) -> usize {
    args.limit.min(100)
}

/// The ids of every hidden or superseded memory, read once.
pub fn hidden_ids(conn: &Connection) -> Result<HashSet<String>> {
    let mut stmt = conn.prepare(
        "SELECT mem_id FROM memory_ext WHERE hidden_ms IS NOT NULL
         UNION SELECT id FROM memories WHERE superseded_by IS NOT NULL OR invalidated_at IS NOT NULL",
    )?;
    let rows = stmt.query_map([], |r| crate::brain_v2::text_at(r, 0))?;
    let mut ids = HashSet::new();
    for id in rows {
        if let Some(id) = id? {
            ids.insert(id);
        }
    }
    Ok(ids)
}

/// False for a hit that must not be shown: its id (without the graph layer's `graph:`) is
/// a hidden or superseded memory, or its `documents` row is hidden.
pub fn shown(conn: &Connection, hidden: &HashSet<String>, doc_id: &str) -> Result<bool> {
    let bare = doc_id.strip_prefix("graph:").unwrap_or(doc_id);
    if hidden.contains(doc_id) || hidden.contains(bare) {
        return Ok(false);
    }
    let gone: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM documents WHERE doc_id IN (?1, ?2) AND hidden IS NOT NULL)",
        [doc_id, bare],
        |r| r.get(0),
    )?;
    Ok(!gone)
}

/// `PRAGMA data_version`: it changes when another connection commits to the brain.
// RULING: v2 c18b review round 2, item 7: exact where a count and a sum of ids are not (a
// document deleted and another written with its id). Any commit of another writer counts,
// so the cache file goes more often than needed; the next search builds it again.
fn data_version(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("PRAGMA data_version", [], |r| r.get(0))?)
}

/// The search for three times `limit`, the hidden hits dropped, at most `limit` kept.
pub fn visible_hits(search: &mut BrainSearch, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    let before = data_version(search.db().conn())?;
    let response = search.search(query, limit.saturating_mul(3))?;
    let conn = search.db().conn();
    // RULING: v2 c18b review, item 9. The first search of a process reads every vector and
    // then writes `<brain>.r8.hnsw`; a `forget --now` that commits and removes that file in
    // between would find it written back with the deleted shadow's vector. When another
    // writer committed while this search ran, the file it may have written goes, and the
    // next search builds it again.
    if data_version(conn)? != before {
        let _ = std::fs::remove_file(search.db().path().with_extension("r8.hnsw"));
    }
    let hidden = hidden_ids(conn)?;
    let mut kept = Vec::new();
    for hit in response.results {
        if kept.len() == limit {
            break;
        }
        if shown(conn, &hidden, &hit.doc_id)? {
            kept.push(hit);
        }
    }
    Ok(kept)
}

pub fn run_with(args: &Args, search: &mut BrainSearch, term: &mut Term) -> Result<u8> {
    let query = args.query.join(" ");
    if query.trim().is_empty() {
        writeln!(term.err, "Please provide a search query.")?;
        return Ok(FAILED);
    }
    let start = Instant::now();
    let limit = limit_of(args);
    let results = visible_hits(search, &query, limit)?;
    let ms = start.elapsed().as_millis();

    // Record search feedback for consolidation
    search.record_search_feedback(&query, &results);

    if args.json {
        let results: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "doc_id": one(&r.doc_id),
                    "score": r.score,
                    "content": screen::show_block(&strip_frontmatter(&r.content), 20_000),
                })
            })
            .collect();
        writeln!(
            term.out,
            "{}",
            serde_json::json!({
                "query": query,
                "count": results.len(),
                "ms": ms,
                "results": results,
            })
        )?;
        return Ok(if results.is_empty() { FAILED } else { OK });
    }

    if results.is_empty() {
        writeln!(term.err, "No results for \"{}\"", one(&query))?;
        return Ok(FAILED);
    }

    writeln!(term.out, "🔍 \"{}\" — {} results ({ms}ms)\n", one(&query), results.len())?;
    for (i, result) in results.iter().enumerate() {
        let clean = strip_frontmatter(&result.content);
        let first: Vec<&str> = clean.lines().filter(|l| !l.trim().is_empty()).take(3).collect();
        let preview = screen::show_block(&first.join(" "), 120);
        writeln!(term.out, "  {}. [{}] (score: {:.3})", i + 1, one(&result.doc_id), result.score)?;
        writeln!(term.out, "     {preview}…\n")?;
    }
    Ok(OK)
}
