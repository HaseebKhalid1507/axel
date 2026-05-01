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
        let mut results_lists: Vec<Vec<RankedResult>> = Vec::new();

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

        // ═══ LAYER 1: VECTOR SIMILARITY ═══
        if opts.layers.vector {
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
        }

        // ═══ LAYER 2: KEYWORD SEARCH (BM25 via FTS5) ═══
        if opts.layers.keyword {
            let t = Instant::now();
            let k = opts.limit * KEYWORD_CANDIDATES_MULTIPLIER;
            let fts_results = self.db.keyword_search(query, k)?;

            let ranked: Vec<RankedResult> = fts_results
                .into_iter()
                .map(|fts| {
                    // Normalize BM25 rank to 0-1 score
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
        }

        // ═══ LAYER 3: KNOWLEDGE GRAPH ═══
        if opts.layers.graph {
            let t = Instant::now();
            // Find graph nodes matching the query, then traverse
            let graph_results = self.graph_search(query)?;
            stats.graph_candidates = graph_results.len();
            stats.graph_ms = Some(t.elapsed().as_secs_f64() * 1000.0);
            if !graph_results.is_empty() {
                results_lists.push(graph_results);
            }
        }

        // ═══ LAYER 4: METADATA (tags + cross-refs) ═══
        {
            let t = Instant::now();
            let mut metadata_results = Vec::new();

            // Search by tags matching query words
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

            // Dedup
            let mut seen = std::collections::HashSet::new();
            metadata_results.retain(|r| seen.insert(r.doc_id.clone()));

            let metadata_ms = t.elapsed().as_secs_f64() * 1000.0;
            if !metadata_results.is_empty() {
                stats.metadata_candidates = metadata_results.len();
                stats.metadata_ms = Some(metadata_ms);
                results_lists.push(metadata_results);
            }
        }

        // ═══ FUSE via RRF ═══
        let mut fused = rrf::reciprocal_rank_fusion(&results_lists, opts.rrf_k);
        stats.fused_count = fused.len();

        // ═══ EXCITABILITY BOOST ═══
        // Documents with higher excitability get a mild ranking boost.
        // This closes the consolidation feedback loop: accessed docs rise,
        // neglected docs sink — matching biological reconsolidation.
        for result in &mut fused {
            let excitability: f64 = self.db.conn().query_row(
                "SELECT COALESCE(excitability, 0.5) FROM documents WHERE doc_id = ?1",
                [&result.doc_id],
                |row| row.get(0),
            ).unwrap_or(0.5);
            // Boost range: 0.9x (excitability=0.0) to 1.1x (excitability=1.0)
            let boost = 0.9 + (excitability * 0.2);
            result.rrf_score *= boost;
        }
        // Re-sort after boosting
        fused.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap_or(std::cmp::Ordering::Equal));

        // ═══ RERANK (optional) ═══
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

        stats.final_count = final_results.len();
        stats.total_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(SearchResponse {
            results: final_results,
            stats,
        })
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
