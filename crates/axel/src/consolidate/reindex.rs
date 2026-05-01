//! Phase 1: Reindex — walk source dirs, re-embed deltas, prune deleted files.
//!
//! Mirrors the logic of `cmd_index_sync` in main.rs but as a reusable function.
//! Adds competitive allocation: new files get `similar_to` edges to high-
//! excitability neighbors (CREB analog).

use std::collections::HashSet;

use rusqlite::params;
use serde_json::json;

use crate::error::{AxelError, Result};
use crate::search::BrainSearch;

use super::SourceDir;

#[derive(Debug, Default, Clone)]
pub struct ReindexStats {
    pub checked: usize,
    pub reindexed: usize,
    pub new_files: usize,
    pub pruned: usize,
}

const ALLOCATION_K: usize = 5;
const ALLOCATION_THRESHOLD: f64 = 0.6;

/// Reindex a single source directory.
pub fn reindex_source(
    search: &mut BrainSearch,
    source: &SourceDir,
    dry_run: bool,
) -> Result<ReindexStats> {
    let mut stats = ReindexStats::default();

    if !source.path.is_dir() {
        return Err(AxelError::Other(format!(
            "reindex source not a directory: {}",
            source.path.display()
        )));
    }

    let target_abs = std::fs::canonicalize(&source.path)
        .unwrap_or_else(|_| source.path.clone());
    let prefix = {
        let mut s = target_abs.to_string_lossy().to_string();
        if !s.ends_with('/') { s.push('/'); }
        s
    };

    // Map of absolute file_path → indexed_at unix seconds.
    let indexed: std::collections::HashMap<String, f64> = search
        .db()
        .indexed_files_under(&prefix)
        .map_err(|e| AxelError::Search(format!("DB read failed: {e}")))?
        .into_iter()
        .collect();

    let mut seen_on_disk: HashSet<String> = HashSet::new();
    let mut newly_indexed: Vec<String> = Vec::new();

    for entry in walkdir::WalkDir::new(&target_abs)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "md" || ext == "txt"))
    {
        stats.checked += 1;
        let file_path = entry.path();
        let abs = std::fs::canonicalize(file_path).unwrap_or_else(|_| file_path.to_path_buf());
        let abs_str = abs.to_string_lossy().to_string();
        seen_on_disk.insert(abs_str.clone());

        let mtime = entry
            .metadata().ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        let is_new = !indexed.contains_key(&abs_str);
        let needs = match indexed.get(&abs_str) {
            None => true,
            Some(&indexed_at) => mtime > indexed_at + 0.5,
        };
        if !needs { continue; }

        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if content.len() < 50 { continue; }

        let relative_id = file_path
            .strip_prefix(&target_abs).unwrap_or(file_path)
            .to_string_lossy()
            .replace('/', "::")
            .replace(".md", "")
            .replace(".txt", "");
        let doc_id = format!("{}::{}", source.name, relative_id);

        if !dry_run {
            if let Err(e) = search.index_document(&doc_id, &content, None, Some(&abs_str)) {
                eprintln!("⚠ index_document failed for {abs_str}: {e}");
                continue;
            }
            if is_new { newly_indexed.push(doc_id.clone()); }
        }
        stats.reindexed += 1;
        if is_new { stats.new_files += 1; }
    }

    // Prune: anything in DB under this prefix that no longer exists on disk.
    for (file_path, _) in &indexed {
        if !seen_on_disk.contains(file_path) && !std::path::Path::new(file_path).exists() {
            if !dry_run {
                let n = search.db().delete_documents_by_file(file_path)
                    .map_err(|e| AxelError::Search(format!("delete failed: {e}")))?;
                if n > 0 { stats.pruned += 1; }
            } else {
                stats.pruned += 1;
            }
        }
    }

    // Competitive allocation: link new docs to high-excitability neighbors.
    if !dry_run {
        for new_id in &newly_indexed {
            if let Err(e) = allocate_new_doc(search, new_id) {
                tracing::warn!("allocation failed for {new_id}: {e}");
            }
        }
    }

    Ok(stats)
}

/// For a freshly indexed doc, search nearest neighbors and create
/// `similar_to` edges to those whose excitability exceeds the threshold.
fn allocate_new_doc(search: &mut BrainSearch, new_doc_id: &str) -> Result<()> {
    // Pull the content back so we can use it as the query.
    let content: Option<String> = search
        .db()
        .conn()
        .query_row(
            "SELECT content FROM documents WHERE doc_id = ?1",
            params![new_doc_id],
            |r| r.get::<_, String>(0),
        )
        .ok();

    let Some(content) = content else { return Ok(()); };

    // Search the brain — the new doc itself will appear at rank 1; skip it.
    let response = search.search(&content, ALLOCATION_K + 1)?;

    for hit in response.results.iter().filter(|r| r.doc_id != new_doc_id).take(ALLOCATION_K) {
        let excitability: f64 = match search.db().conn().query_row(
            "SELECT excitability FROM documents WHERE doc_id = ?1",
            params![&hit.doc_id],
            |r| r.get::<_, f64>(0),
        ) {
            Ok(v) => v,
            Err(_) => continue,
        };

        if excitability <= ALLOCATION_THRESHOLD { continue; }

        let edge_id = format!(
            "alloc::{}::{}::{}",
            new_doc_id,
            hit.doc_id,
            chrono::Utc::now().timestamp_millis()
        );
        if let Err(e) = search.db().insert_edge(
            &edge_id,
            new_doc_id,
            &hit.doc_id,
            "similar_to",
            hit.score,
            0.8,
            &json!({"source": "consolidation_allocation"}),
            None,
            None,
            None,
        ) {
            tracing::warn!("insert_edge failed: {e}");
        }
    }
    Ok(())
}
