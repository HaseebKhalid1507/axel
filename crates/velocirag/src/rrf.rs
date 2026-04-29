//! Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists.
//!
//! Direct port from the Python VelociRAG implementation.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

// ── Constants ───────────────────────────────────────────────────────────────

const DEFAULT_RRF_K: usize = 60;
const MAX_FUSION_RESULTS: usize = 1000;

// ── Types ───────────────────────────────────────────────────────────────────

/// A search result that can be fused across multiple ranked lists.
#[derive(Debug, Clone)]
pub struct RankedResult {
    /// Unique document identifier for deduplication.
    pub doc_id: String,
    /// Document content.
    pub content: String,
    /// Similarity or relevance score from the source ranker.
    pub score: f64,
    /// Arbitrary metadata.
    pub metadata: serde_json::Value,
}

/// A result after RRF fusion with the combined score.
#[derive(Debug, Clone)]
pub struct FusedResult {
    pub doc_id: String,
    pub content: String,
    pub rrf_score: f64,
    pub original_score: f64,
    pub metadata: serde_json::Value,
}

// ── RRF ─────────────────────────────────────────────────────────────────────

/// Combine multiple result lists using Reciprocal Rank Fusion.
///
/// RRF Score = Σ 1/(k + rank) across all lists where the document appears.
///
/// Results are sorted by RRF score (highest first).
pub fn reciprocal_rank_fusion(
    results_lists: &[Vec<RankedResult>],
    k: usize,
) -> Vec<FusedResult> {
    let k = if k == 0 { DEFAULT_RRF_K } else { k };

    if results_lists.is_empty() {
        return Vec::new();
    }

    // Memory protection: truncate if too many total results
    let total: usize = results_lists.iter().map(|r| r.len()).sum();
    let max_per_set = if total > MAX_FUSION_RESULTS {
        let valid_count = results_lists.iter().filter(|r| !r.is_empty()).count();
        if valid_count > 0 {
            MAX_FUSION_RESULTS / valid_count
        } else {
            0
        }
    } else {
        usize::MAX
    };

    let mut doc_scores: HashMap<String, f64> = HashMap::new();
    let mut doc_map: HashMap<String, &RankedResult> = HashMap::new();

    for result_list in results_lists {
        for (rank, result) in result_list.iter().take(max_per_set).enumerate() {
            let doc_id = if result.doc_id.is_empty() {
                content_hash(&result.content)
            } else {
                result.doc_id.clone()
            };

            // Keep version with highest original score
            let existing = doc_map.get(&doc_id);
            if existing.is_none() || result.score > existing.unwrap().score {
                doc_map.insert(doc_id.clone(), result);
            }

            // Accumulate RRF score
            let rrf_score = 1.0 / (k as f64 + (rank + 1) as f64);
            *doc_scores.entry(doc_id).or_insert(0.0) += rrf_score;
        }
    }

    // Sort by RRF score descending
    let mut scored: Vec<_> = doc_scores.into_iter().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    scored
        .into_iter()
        .map(|(doc_id, rrf_score)| {
            let result = doc_map[&doc_id];
            FusedResult {
                doc_id,
                content: result.content.clone(),
                rrf_score: (rrf_score * 10000.0).round() / 10000.0,
                original_score: result.score,
                metadata: result.metadata.clone(),
            }
        })
        .collect()
}

fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("hash_{}", hex::encode(hasher.finalize()))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(doc_id: &str, content: &str, score: f64) -> RankedResult {
        RankedResult {
            doc_id: doc_id.to_string(),
            content: content.to_string(),
            score,
            metadata: serde_json::json!({}),
        }
    }

    #[test]
    fn test_single_list() {
        let lists = vec![vec![
            make_result("a", "doc a", 0.9),
            make_result("b", "doc b", 0.8),
        ]];
        let fused = reciprocal_rank_fusion(&lists, 60);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].doc_id, "a");
        assert!(fused[0].rrf_score > fused[1].rrf_score);
    }

    #[test]
    fn test_overlapping_lists() {
        let lists = vec![
            vec![
                make_result("a", "doc a", 0.9),
                make_result("b", "doc b", 0.8),
            ],
            vec![
                make_result("b", "doc b", 0.95),
                make_result("c", "doc c", 0.7),
            ],
        ];
        let fused = reciprocal_rank_fusion(&lists, 60);
        assert_eq!(fused.len(), 3);
        // "b" appears in both lists, should have highest RRF score
        assert_eq!(fused[0].doc_id, "b");
    }

    #[test]
    fn test_empty_input() {
        let fused = reciprocal_rank_fusion(&[], 60);
        assert!(fused.is_empty());
    }
}
