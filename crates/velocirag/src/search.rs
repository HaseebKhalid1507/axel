//! Four-layer fusion search engine.
//!
//! Combines:
//!   1. Vector similarity (HNSW via usearch)
//!   2. BM25 keyword search (SQLite FTS5)
//!   3. Knowledge graph traversal
//!   4. Metadata filtering
//!
//! Results are fused via Reciprocal Rank Fusion (RRF) with optional
//! cross-encoder reranking.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::db::Database;
use crate::embedder::Embedder;
use crate::index::VectorIndex;
use crate::reranker::{Reranker, RerankInput};
use crate::rrf::{self, FusedResult, RankedResult};
use crate::error::Result;

// ── Config ──────────────────────────────────────────────────────────────────

const DEFAULT_LIMIT: usize = 10;
const DEFAULT_RRF_K: usize = 60;
const VECTOR_CANDIDATES_MULTIPLIER: usize = 3;
const KEYWORD_CANDIDATES_MULTIPLIER: usize = 3;
const GRAPH_DEPTH: usize = 2;
const GRAPH_MAX_RESULTS: usize = 20;
const DEFAULT_BM25_SCORE: f64 = 0.1;

// ── Types ───────────────────────────────────────────────────────────────────

/// A search result with all metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub doc_id: String,
    pub content: String,
    pub score: f64,
    pub source: String, // "vector", "keyword", "graph", "fused"
    pub metadata: serde_json::Value,
}

/// Stats about the search execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStats {
    pub total_ms: f64,
    pub vector_ms: Option<f64>,
    pub keyword_ms: Option<f64>,
    pub graph_ms: Option<f64>,
    pub metadata_ms: Option<f64>,
    pub rerank_ms: Option<f64>,
    pub vector_candidates: usize,
    pub keyword_candidates: usize,
    pub graph_candidates: usize,
    pub metadata_candidates: usize,
    pub fused_count: usize,
    pub final_count: usize,
}

/// Search response containing results and stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub stats: SearchStats,
}

/// Options for controlling search behavior.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub limit: usize,
    pub rrf_k: usize,
    pub use_reranker: bool,
    pub layers: SearchLayers,
}

#[derive(Debug, Clone)]
pub struct SearchLayers {
    pub vector: bool,
    pub keyword: bool,
    pub graph: bool,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: DEFAULT_LIMIT,
            rrf_k: DEFAULT_RRF_K,
            use_reranker: true,
            layers: SearchLayers {
                vector: true,
                keyword: true,
                graph: true,
            },
        }
    }
}

// ── Search Engine ───────────────────────────────────────────────────────────

pub struct SearchEngine<'a> {
    db: &'a Database,
    index: &'a VectorIndex,
    embedder: &'a mut Embedder,
    reranker: Option<&'a mut Reranker>,
}

impl<'a> SearchEngine<'a> {
    pub fn new(
        db: &'a Database,
        index: &'a VectorIndex,
        embedder: &'a mut Embedder,
        reranker: Option<&'a mut Reranker>,
    ) -> Self {
        Self {
            db,
            index,
            embedder,
            reranker,
        }
    }

    /// Run a four-layer fusion search.
    pub fn search(&mut self, query: &str, opts: &SearchOptions) -> Result<SearchResponse> {
        let start = Instant::now();
        let mut stats = SearchStats {
            total_ms: 0.0,
            vector_ms: None,
            keyword_ms: None,
            graph_ms: None,
            metadata_ms: None,
            rerank_ms: None,
            vector_candidates: 0,
            keyword_candidates: 0,
            graph_candidates: 0,
            metadata_candidates: 0,
            fused_count: 0,
            final_count: 0,
        };
        let mut results_lists: Vec<Vec<RankedResult>> = Vec::new();

        if opts.layers.vector {
            self.retrieve_vector(query, opts, &mut results_lists, &mut stats)?;
        }

        let expanded_query = self.expand_query(query, &results_lists, opts);

        if opts.layers.keyword {
            self.retrieve_keyword(&expanded_query, opts, &mut results_lists, &mut stats)?;
        }

        if opts.layers.graph {
            self.retrieve_graph(query, &mut results_lists, &mut stats)?;
        }

        self.retrieve_metadata(query, opts, &mut results_lists, &mut stats)?;

        let mut fused = rrf::reciprocal_rank_fusion(&results_lists, opts.rrf_k);
        stats.fused_count = fused.len();

        self.apply_excitability_boost(&mut fused);
        self.apply_graph_boost(&mut fused, opts);
        self.apply_mmr_diversity(&mut fused, opts);

        let final_results = self.finalize(query, fused, opts, &mut stats)?;

        stats.final_count = final_results.len();
        stats.total_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(SearchResponse {
            results: final_results,
            stats,
        })
    }

    fn retrieve_vector(
        &mut self,
        query: &str,
        opts: &SearchOptions,
        results_lists: &mut Vec<Vec<RankedResult>>,
        stats: &mut SearchStats,
    ) -> Result<()> {
        let t = Instant::now();
        let query_embedding = self.embedder.embed_one(query)?;
        let k = opts.limit * VECTOR_CANDIDATES_MULTIPLIER;
        let vector_results = self.index.search(&query_embedding, k)?;

        let ranked: Vec<RankedResult> = vector_results
            .into_iter()
            .filter_map(|vr| {
                let doc = self.db.conn().query_row(
                    "SELECT doc_id, content, metadata FROM documents WHERE id = ?1",
                    [vr.id as i64],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, String>(2)?,
                        ))
                    },
                ).ok()?;
                Some(RankedResult {
                    doc_id: doc.0,
                    content: doc.1,
                    score: vr.score as f64,
                    metadata: serde_json::from_str(&doc.2).unwrap_or_default(),
                })
            })
            .collect();

        stats.vector_candidates = ranked.len();
        stats.vector_ms = Some(t.elapsed().as_secs_f64() * 1000.0);
        results_lists.push(ranked);
        Ok(())
    }

    fn expand_query(&self, query: &str, results_lists: &[Vec<RankedResult>], opts: &SearchOptions) -> String {
        if opts.layers.vector && !results_lists.is_empty() && !results_lists[0].is_empty()
            && results_lists[0][0].score > 0.02
        {
            let top_content = &results_lists[0][0].content;
            let query_lower = query.to_lowercase();
            let expansion_terms: Vec<String> = extract_top_terms(top_content, 3)
                .into_iter()
                .filter(|t| !query_lower.contains(t.as_str()))
                .collect();
            if expansion_terms.is_empty() {
                query.to_string()
            } else {
                format!("{} {}", query, expansion_terms.join(" "))
            }
        } else {
            query.to_string()
        }
    }

    fn retrieve_keyword(
        &self,
        expanded_query: &str,
        opts: &SearchOptions,
        results_lists: &mut Vec<Vec<RankedResult>>,
        stats: &mut SearchStats,
    ) -> Result<()> {
        let t = Instant::now();
        let k = opts.limit * KEYWORD_CANDIDATES_MULTIPLIER;
        let fts_results = self.db.keyword_search(expanded_query, k)?;

        let ranked: Vec<RankedResult> = fts_results
            .into_iter()
            .map(|fts| {
                let score = if fts.bm25_rank == 0.0 {
                    DEFAULT_BM25_SCORE
                } else {
                    (1.0 + (fts.bm25_rank / 50.0)).clamp(0.0, 1.0)
                };
                RankedResult {
                    doc_id: fts.doc_id,
                    content: fts.content,
                    score,
                    metadata: serde_json::json!({
                        "bm25_rank": fts.bm25_rank,
                        "snippet": fts.snippet,
                    }),
                }
            })
            .collect();

        stats.keyword_candidates = ranked.len();
        stats.keyword_ms = Some(t.elapsed().as_secs_f64() * 1000.0);
        results_lists.push(ranked);
        Ok(())
    }

    fn retrieve_graph(
        &self,
        query: &str,
        results_lists: &mut Vec<Vec<RankedResult>>,
        stats: &mut SearchStats,
    ) -> Result<()> {
        let t = Instant::now();
        let graph_results = self.graph_search(query)?;
        stats.graph_candidates = graph_results.len();
        stats.graph_ms = Some(t.elapsed().as_secs_f64() * 1000.0);
        if !graph_results.is_empty() {
            results_lists.push(graph_results);
        }
        Ok(())
    }

    fn retrieve_metadata(
        &self,
        query: &str,
        _opts: &SearchOptions,
        results_lists: &mut Vec<Vec<RankedResult>>,
        stats: &mut SearchStats,
    ) -> Result<()> {
        let t = Instant::now();
        let mut metadata_results = Vec::new();

        let words: Vec<&str> = query.split_whitespace()
            .filter(|w| w.len() >= 3)
            .collect();

        for word in &words {
            if let Ok(tag_docs) = self.db.search_by_tag(word, 10) {
                for (doc_id, content) in tag_docs {
                    metadata_results.push(RankedResult {
                        doc_id,
                        content,
                        score: 0.7,
                        metadata: serde_json::json!({"source": "tag_match", "tag": word}),
                    });
                }
            }
            if let Ok(ref_docs) = self.db.search_by_cross_ref(word, 10) {
                for (doc_id, content) in ref_docs {
                    metadata_results.push(RankedResult {
                        doc_id,
                        content,
                        score: 0.6,
                        metadata: serde_json::json!({"source": "cross_ref", "ref": word}),
                    });
                }
            }
        }

        let mut seen = std::collections::HashSet::new();
        metadata_results.retain(|r| seen.insert(r.doc_id.clone()));

        let metadata_ms = t.elapsed().as_secs_f64() * 1000.0;
        if !metadata_results.is_empty() {
            stats.metadata_candidates = metadata_results.len();
            stats.metadata_ms = Some(metadata_ms);
            results_lists.push(metadata_results);
        }
        Ok(())
    }

    fn apply_excitability_boost(&self, fused: &mut Vec<FusedResult>) {
        if !fused.is_empty() {
            let placeholders: Vec<String> = (0..fused.len()).map(|i| format!("?{}", i + 1)).collect();
            let sql = format!(
                "SELECT doc_id, COALESCE(excitability, 0.5),
                        COALESCE((julianday('now') - julianday(last_accessed)), 0),
                        COALESCE((julianday('now') - julianday(indexed_at)), 30)
                 FROM documents WHERE doc_id IN ({})",
                placeholders.join(",")
            );
            let mut stmt = self.db.conn().prepare(&sql).ok();
            let mut excitability_map: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
            if let Some(ref mut s) = stmt {
                let params: Vec<&dyn rusqlite::ToSql> = fused.iter().map(|r| &r.doc_id as &dyn rusqlite::ToSql).collect();
                if let Ok(rows) = s.query_map(params.as_slice(), |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, f64>(3)?,
                    ))
                }) {
                    for row in rows.flatten() {
                        let (doc_id, stored_excitability, days_since_access, days_since_indexed) = row;
                        // Ebbinghaus exponential decay (Murre & Dros, 2015).
                        let decay_factor = (-days_since_access / 60.0).exp().max(0.7);
                        // Recency boost (YouTube DNN paper, Covington et al. 2016):
                        // Recently modified docs get a mild boost.
                        // log(1 + 1/days) gives ~0.7 for today, ~0.3 for 7 days, ~0.1 for 30 days
                        let recency = (1.0 + 1.0 / (days_since_indexed + 1.0)).ln().min(0.7);
                        let effective = stored_excitability * decay_factor + recency * 0.05;
                        excitability_map.insert(doc_id, effective.min(1.0));
                    }
                }
            }
            for result in fused.iter_mut() {
                let excitability = excitability_map.get(&result.doc_id).copied().unwrap_or(0.5);
                // Boost range: 0.9x (excitability=0.0) to 1.1x (excitability=1.0)
                let boost = 0.9 + (excitability * 0.2);
                result.rrf_score *= boost;
            }
            // Re-sort after boosting
            fused.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    fn apply_graph_boost(&self, fused: &mut Vec<FusedResult>, opts: &SearchOptions) {
        const GRAPH_BOOST_FACTOR: f64 = 0.3;
        const GRAPH_SOURCES_TOPK: usize = 5;

        if !fused.is_empty() {
            // Cheap gate: skip entirely if the graph has no live co_retrieved edges.
            let has_coret: bool = self.db.conn()
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM edges WHERE type = 'co_retrieved'
                                   AND (valid_to IS NULL OR valid_to > datetime('now')) LIMIT 1)",
                    [],
                    |row| row.get::<_, i64>(0).map(|n| n != 0),
                )
                .unwrap_or(false);

            if has_coret {
                // Snapshot top-K parents (doc_id, rrf_score) — `fused` will mutate below.
                let parents: Vec<(String, f64)> = fused.iter()
                    .take(GRAPH_SOURCES_TOPK)
                    .map(|f| (f.doc_id.clone(), f.rrf_score))
                    .collect();

                // doc_id -> position in `fused`, for O(1) bumps.
                let mut index_map: std::collections::HashMap<String, usize> =
                    fused.iter().enumerate().map(|(i, f)| (f.doc_id.clone(), i)).collect();

                // Aggregate boosts before mutating, so multiple parents stack.
                let mut boost_accum: std::collections::HashMap<String, f64> =
                    std::collections::HashMap::new();

                let conn = self.db.conn();
                {
                    let mut fwd = conn.prepare_cached(
                        "SELECT target_id, weight FROM edges
                         WHERE source_id = ?1 AND type = 'co_retrieved'
                           AND (valid_to IS NULL OR valid_to > datetime('now'))",
                    ).ok();
                    let mut rev = conn.prepare_cached(
                        "SELECT source_id, weight FROM edges
                         WHERE target_id = ?1 AND type = 'co_retrieved'
                           AND (valid_to IS NULL OR valid_to > datetime('now'))",
                    ).ok();

                    for (parent_id, parent_score) in &parents {
                        let mut neighbors: Vec<(String, f64)> = Vec::new();
                        if let Some(ref mut s) = fwd {
                            if let Ok(rows) = s.query_map([parent_id], |row| {
                                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                            }) {
                                neighbors.extend(rows.flatten());
                            }
                        }
                        if let Some(ref mut s) = rev {
                            if let Ok(rows) = s.query_map([parent_id], |row| {
                                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                            }) {
                                neighbors.extend(rows.flatten());
                            }
                        }
                        for (neighbor_id, weight) in neighbors {
                            if neighbor_id == *parent_id { continue; } // self-loop guard
                            let graph_score = parent_score * weight.clamp(0.0, 1.0) * GRAPH_BOOST_FACTOR;
                            *boost_accum.entry(neighbor_id).or_insert(0.0) += graph_score;
                        }
                    }
                } // drop cached stmts so we can re-borrow conn

                // Apply boosts: bump existing or add new candidates (capped).
                let max_total = opts.limit * 2;
                for (neighbor_id, graph_score) in boost_accum {
                    if let Some(&idx) = index_map.get(&neighbor_id) {
                        fused[idx].rrf_score += graph_score;
                    } else if fused.len() < max_total {
                        let row = conn.query_row(
                            "SELECT content, metadata FROM documents WHERE doc_id = ?1",
                            [&neighbor_id],
                            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                        ).ok();
                        if let Some((content, metadata_str)) = row {
                            let doc_metadata: serde_json::Value =
                                serde_json::from_str(&metadata_str).unwrap_or_else(|_| serde_json::json!({}));
                            index_map.insert(neighbor_id.clone(), fused.len());
                            fused.push(FusedResult {
                                doc_id: neighbor_id,
                                content,
                                rrf_score: graph_score,
                                original_score: 0.0,
                                metadata: serde_json::json!({
                                    "source": "graph_boost",
                                    "graph_score": graph_score,
                                    "doc_metadata": doc_metadata,
                                }),
                            });
                        }
                    }
                }

                // Re-sort after spreading activation.
                fused.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
    }

    fn apply_mmr_diversity(&self, fused: &mut Vec<FusedResult>, opts: &SearchOptions) {
        if fused.len() > opts.limit {
            const LAMBDA: f32 = 0.7;

            // Batch-load embeddings for candidates by doc_id.
            let placeholders: Vec<String> = (0..fused.len()).map(|i| format!("?{}", i + 1)).collect();
            let sql = format!(
                "SELECT doc_id, embedding FROM documents WHERE doc_id IN ({})",
                placeholders.join(",")
            );
            let mut emb_map: std::collections::HashMap<String, Vec<f32>> = std::collections::HashMap::new();
            if let Ok(mut stmt) = self.db.conn().prepare(&sql) {
                let params: Vec<&dyn rusqlite::ToSql> = fused.iter().map(|r| &r.doc_id as &dyn rusqlite::ToSql).collect();
                if let Ok(rows) = stmt.query_map(params.as_slice(), |row| {
                    let doc_id: String = row.get(0)?;
                    let blob: Vec<u8> = row.get(1)?;
                    let v: Vec<f32> = blob
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Ok((doc_id, v))
                }) {
                    for r in rows.flatten() {
                        emb_map.insert(r.0, r.1);
                    }
                }
            }

            fn cosine(a: &[f32], b: &[f32]) -> f32 {
                if a.len() != b.len() || a.is_empty() {
                    return 0.0;
                }
                let mut dot = 0.0f32;
                let mut na = 0.0f32;
                let mut nb = 0.0f32;
                for i in 0..a.len() {
                    dot += a[i] * b[i];
                    na += a[i] * a[i];
                    nb += b[i] * b[i];
                }
                let denom = na.sqrt() * nb.sqrt();
                if denom == 0.0 { 0.0 } else { dot / denom }
            }

            // Normalize relevance to [0,1] for stable MMR scaling.
            let max_rel = fused.iter().map(|r| r.rrf_score).fold(f64::MIN, f64::max);
            let min_rel = fused.iter().map(|r| r.rrf_score).fold(f64::MAX, f64::min);
            let span = (max_rel - min_rel).max(1e-9);

            let mut remaining: Vec<FusedResult> = std::mem::take(fused);
            let mut selected: Vec<FusedResult> = Vec::with_capacity(opts.limit);
            let mut selected_embs: Vec<Vec<f32>> = Vec::with_capacity(opts.limit);

            // Seed with highest-scoring doc.
            let first = remaining.swap_remove(0);
            if let Some(e) = emb_map.get(&first.doc_id).cloned() {
                selected_embs.push(e);
            } else {
                selected_embs.push(Vec::new());
            }
            selected.push(first);

            while selected.len() < opts.limit && !remaining.is_empty() {
                let mut best_idx = 0usize;
                let mut best_score = f32::MIN;
                for (i, cand) in remaining.iter().enumerate() {
                    let rel = ((cand.rrf_score - min_rel) / span) as f32;
                    let max_sim = if let Some(ce) = emb_map.get(&cand.doc_id) {
                        selected_embs
                            .iter()
                            .map(|se| if se.is_empty() { 0.0 } else { cosine(ce, se) })
                            .fold(f32::MIN, f32::max)
                    } else {
                        0.0
                    };
                    let max_sim = if max_sim == f32::MIN { 0.0 } else { max_sim };
                    let mmr = LAMBDA * rel - (1.0 - LAMBDA) * max_sim;
                    if mmr > best_score {
                        best_score = mmr;
                        best_idx = i;
                    }
                }
                let chosen = remaining.swap_remove(best_idx);
                let emb = emb_map.get(&chosen.doc_id).cloned().unwrap_or_default();
                selected_embs.push(emb);
                selected.push(chosen);
            }

            *fused = selected;
        }
    }

    fn finalize(
        &mut self,
        query: &str,
        fused: Vec<FusedResult>,
        opts: &SearchOptions,
        stats: &mut SearchStats,
    ) -> Result<Vec<SearchResult>> {
        let final_results = if opts.use_reranker && self.reranker.is_some() {
            let t = Instant::now();
            let reranked = self.rerank(query, &fused, opts.limit)?;
            stats.rerank_ms = Some(t.elapsed().as_secs_f64() * 1000.0);
            reranked
        } else {
            fused
                .into_iter()
                .take(opts.limit)
                .map(|f| SearchResult {
                    doc_id: f.doc_id,
                    content: f.content,
                    score: f.rrf_score,
                    source: "fused".to_string(),
                    metadata: f.metadata,
                })
                .collect()
        };
        Ok(final_results)
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn graph_search(&self, query: &str) -> Result<Vec<RankedResult>> {
        // Find nodes whose titles match the query terms
        let words: Vec<&str> = query.split_whitespace()
            .filter(|w| w.len() >= 3)  // skip short words
            .collect();
        if words.is_empty() {
            return Ok(Vec::new());
        }

        let mut matched_nodes = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();

        for word in &words {
            let pattern = format!("%{}%", word.to_lowercase());
            let mut stmt = self.db.conn().prepare(
                "SELECT id, title, content FROM nodes WHERE LOWER(title) LIKE ?1 LIMIT 10",
            )?;
            let rows = stmt.query_map([&pattern], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            })?;
            for row in rows {
                let r = row?;
                if seen_ids.insert(r.0.clone()) {
                    matched_nodes.push(r);
                }
            }
        }

        if matched_nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Traverse neighbors of matched nodes
        let mut results = Vec::new();
        let _query_lower = query.to_lowercase();

        for (node_id, title, _content) in &matched_nodes {
            let neighbors = self.db.get_neighbors(node_id, GRAPH_DEPTH, GRAPH_MAX_RESULTS)?;
            for (neighbor_node, edge) in &neighbors {
                let doc_content = neighbor_node
                    .content
                    .as_deref()
                    .unwrap_or(&neighbor_node.title);

                // Filter out noise: skip nodes with very short content (empty journal entries etc.)
                let content_len = doc_content.len();
                if content_len < 20 {
                    continue;
                }

                // Score: base weight × confidence, boosted by content relevance
                let base_score = edge.weight * edge.confidence;

                // Boost if the neighbor's title or content matches query terms
                let title_lower = neighbor_node.title.to_lowercase();
                let content_lower = doc_content.to_lowercase();
                let mut relevance_boost = 1.0;
                for word in &words {
                    let w = word.to_lowercase();
                    if title_lower.contains(&w) {
                        relevance_boost += 0.5;
                    }
                    if content_lower.contains(&w) {
                        relevance_boost += 0.3;
                    }
                }

                // Penalize very generic connections (temporal edges to unrelated notes)
                let type_weight = match edge.edge_type.as_str() {
                    "references" => 1.5,
                    "tagged_as" => 1.3,
                    "mentions" => 1.2,
                    "similar_to" => 1.1,
                    "discusses" => 1.0,
                    "temporal" => 0.4,  // temporal edges are weak signals
                    _ => 0.8,
                };

                let final_score = base_score * relevance_boost * type_weight;

                results.push(RankedResult {
                    doc_id: format!("graph:{}", neighbor_node.id),
                    content: doc_content.to_string(),
                    score: final_score,
                    metadata: serde_json::json!({
                        "source_node": title,
                        "edge_type": edge.edge_type,
                        "node_type": neighbor_node.node_type,
                        "base_score": base_score,
                        "relevance_boost": relevance_boost,
                    }),
                });
            }
        }

        // Also add the matched nodes themselves (they're direct title matches!)
        for (node_id, title, content) in &matched_nodes {
            let doc_content = content.as_deref().unwrap_or(title);
            if doc_content.len() < 20 { continue; }

            // Direct title match gets a high score
            let mut relevance = 1.0;
            for word in &words {
                if title.to_lowercase().contains(&word.to_lowercase()) {
                    relevance += 0.5;
                }
            }

            results.push(RankedResult {
                doc_id: format!("graph:{}", node_id),
                content: doc_content.to_string(),
                score: relevance,
                metadata: serde_json::json!({
                    "source_node": title,
                    "edge_type": "direct_match",
                    "node_type": "matched",
                }),
            });
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Deduplicate by doc_id
        let mut seen = std::collections::HashSet::new();
        results.retain(|r| seen.insert(r.doc_id.clone()));

        // Cap results
        results.truncate(GRAPH_MAX_RESULTS);

        Ok(results)
    }

    fn rerank(&mut self, query: &str, fused: &[FusedResult], limit: usize) -> Result<Vec<SearchResult>> {
        let reranker = self.reranker.as_mut().unwrap();
        let inputs: Vec<RerankInput> = fused.iter().map(|f| RerankInput {
            content: f.content.clone(),
            original_score: f.rrf_score,
            doc_id: f.doc_id.clone(),
            metadata: f.metadata.clone(),
        }).collect();

        let reranked = reranker.rerank(query, inputs, limit)?;
        Ok(reranked.into_iter().map(|r| SearchResult {
            doc_id: r.doc_id,
            content: r.content,
            score: r.rerank_score,
            source: "reranked".to_string(),
            metadata: r.metadata,
        }).collect())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pseudo-Relevance Feedback helper
// Extracts top-N most frequent meaningful terms from a passage of text.
// Used to expand BM25 queries with terms drawn from the top vector hit.
// ─────────────────────────────────────────────────────────────────────────────
fn extract_top_terms(text: &str, n: usize) -> Vec<String> {
    use std::collections::HashMap;

    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
        "into", "through", "during", "before", "after", "above", "below",
        "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
        "this", "that", "these", "those", "it", "its", "they", "them",
        "he", "she", "we", "you", "i", "me", "my", "our", "your", "his", "her",
        "which", "who", "whom", "what", "where", "when", "how", "why",
        "all", "each", "every", "any", "some", "no", "more", "most", "other",
    ];

    let mut counts: HashMap<String, usize> = HashMap::new();
    for word in text.split_whitespace() {
        let clean = word
            .to_lowercase()
            .trim_matches(|c: char| !c.is_alphanumeric())
            .to_string();
        if clean.len() > 3 && !STOPWORDS.contains(&clean.as_str()) {
            *counts.entry(clean).or_insert(0) += 1;
        }
    }

    let mut sorted: Vec<_> = counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.into_iter().take(n).map(|(w, _)| w).collect()
}
