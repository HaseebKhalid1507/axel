//! Cross-encoder reranker (ONNX Runtime).
//!
//! Uses TinyBERT cross-encoder for query-document relevance scoring.
//! Same lazy-loading pattern as the embedder — model loaded on first rerank() call.
//! Port of velocirag/reranker.py.

use std::path::PathBuf;
use std::sync::Mutex;

use ndarray::Array2;
use ort::session::Session;
use tokenizers::Tokenizer;

use crate::error::{Result, VelociError};

// ── Constants ───────────────────────────────────────────────────────────────

const MODEL_NAME: &str = "cross-encoder/ms-marco-TinyBERT-L-2-v2";
const MAX_SEQ_LENGTH: usize = 512;
const MAX_EXCERPT_LENGTH: usize = 2000;
const EXCERPT_HEAD: usize = 1000;
const EXCERPT_TAIL: usize = 1000;

// ── Reranker ───────────────────────────────────────────────────────────────

pub struct Reranker {
    session: Mutex<Option<Session>>,
    tokenizer: Option<Tokenizer>,
    loaded: bool,
    load_error: Option<String>,
}

/// A scored result for reranking.
#[derive(Debug, Clone)]
pub struct RerankInput {
    pub content: String,
    pub original_score: f64,
    pub doc_id: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct RerankOutput {
    pub content: String,
    pub rerank_score: f64,
    pub original_score: f64,
    pub doc_id: String,
    pub metadata: serde_json::Value,
}

impl Reranker {
    /// Create a new reranker with lazy model loading.
    pub fn new() -> Self {
        Self {
            session: Mutex::new(None),
            tokenizer: None,
            loaded: false,
            load_error: None,
        }
    }

    /// Rerank results by query-document relevance.
    ///
    /// Scores query-document pairs using cross-encoder and returns
    /// results sorted by relevance, limited to `limit` entries.
    /// Gracefully degrades if model unavailable — returns original order.
    pub fn rerank(&mut self, query: &str, results: Vec<RerankInput>, limit: usize) -> Result<Vec<RerankOutput>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        // Ensure model is loaded
        if !self.loaded && self.load_error.is_none() {
            self.load_model();
        }

        // Graceful degradation
        if self.load_error.is_some() {
            tracing::warn!("Reranker unavailable, returning unranked results");
            return Ok(results.into_iter().take(limit).map(|r| RerankOutput {
                rerank_score: r.original_score,
                content: r.content,
                original_score: r.original_score,
                doc_id: r.doc_id,
                metadata: r.metadata,
            }).collect());
        }

        // Build query-document pairs
        let pairs: Vec<(String, String)> = results.iter()
            .map(|r| (query.to_string(), excerpt_content(&r.content)))
            .collect();

        match self.predict(&pairs) {
            Ok(scores) => {
                let mut scored: Vec<(f64, RerankOutput)> = results.into_iter()
                    .zip(scores.iter())
                    .map(|(r, &score)| {
                        (score, RerankOutput {
                            rerank_score: score,
                            content: r.content,
                            original_score: r.original_score,
                            doc_id: r.doc_id,
                            metadata: r.metadata,
                        })
                    })
                    .collect();

                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                Ok(scored.into_iter().take(limit).map(|(_, r)| r).collect())
            }
            Err(e) => {
                tracing::warn!("Reranking failed: {}, returning unranked results", e);
                Ok(results.into_iter().take(limit).map(|r| RerankOutput {
                    rerank_score: r.original_score,
                    content: r.content,
                    original_score: r.original_score,
                    doc_id: r.doc_id,
                    metadata: r.metadata,
                }).collect())
            }
        }
    }

    /// Whether the model is loaded.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Get load error message, if any.
    pub fn load_error(&self) -> Option<&str> {
        self.load_error.as_deref()
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn load_model(&mut self) {
        let model_dir = resolve_model_dir();

        let onnx_path = model_dir.join("onnx").join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !onnx_path.exists() || !tokenizer_path.exists() {
            self.load_error = Some(format!(
                "Cross-encoder model not found at {}. Download from huggingface: {}",
                model_dir.display(), MODEL_NAME
            ));
            tracing::error!("{}", self.load_error.as_ref().unwrap());
            return;
        }

        tracing::info!("Loading cross-encoder model from {}", onnx_path.display());

        // Load ONNX session
        let session = match Session::builder()
            .map_err(|e| format!("Session builder failed: {}", e))
        {
            Ok(builder) => {
                match builder.with_intra_threads(num_cpus().min(4)) {
                    Ok(mut b) => match b.commit_from_file(&onnx_path) {
                        Ok(s) => s,
                        Err(e) => {
                            self.load_error = Some(format!("Load failed: {}", e));
                            tracing::error!("Failed to load cross-encoder ONNX model: {}", e);
                            return;
                        }
                    },
                    Err(e) => {
                        self.load_error = Some(format!("Thread config failed: {}", e));
                        return;
                    }
                }
            }
            Err(e) => {
                self.load_error = Some(e.to_string());
                tracing::error!("Failed to create session builder: {}", e);
                return;
            }
        };

        // Load tokenizer
        let mut tokenizer = match Tokenizer::from_file(&tokenizer_path) {
            Ok(t) => t,
            Err(e) => {
                self.load_error = Some(format!("Failed to load tokenizer: {}", e));
                tracing::error!("{}", self.load_error.as_ref().unwrap());
                return;
            }
        };

        // Enable truncation and padding
        tokenizer.with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAX_SEQ_LENGTH,
            ..Default::default()
        })).ok();
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            ..Default::default()
        }));

        *self.session.lock().unwrap() = Some(session);
        self.tokenizer = Some(tokenizer);
        self.loaded = true;

        tracing::info!("Cross-encoder model loaded: {}", MODEL_NAME);
    }

    /// Run cross-encoder inference on query-document pairs.
    /// Returns sigmoid-normalized relevance scores.
    fn predict(&self, pairs: &[(String, String)]) -> Result<Vec<f64>> {
        let tokenizer = self.tokenizer.as_ref()
            .ok_or_else(|| VelociError::Embedding("Reranker tokenizer not loaded".into()))?;
        let mut session_guard = self.session.lock().unwrap();
        let session = session_guard.as_mut()
            .ok_or_else(|| VelociError::Embedding("Reranker session not loaded".into()))?;

        // Tokenize pairs — cross-encoder takes (query, document) as sentence pairs
        let pair_refs: Vec<(&str, &str)> = pairs.iter()
            .map(|(q, d)| (q.as_str(), d.as_str()))
            .collect();

        let encodings = tokenizer
            .encode_batch(pair_refs, true)
            .map_err(|e| VelociError::Embedding(format!("Tokenization failed: {}", e)))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Build input tensors
        let mut input_ids = Array2::<i64>::zeros((batch_size, max_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, max_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch_size, max_len));

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();
            let len = ids.len().min(max_len);
            for j in 0..len {
                input_ids[[i, j]] = ids[j] as i64;
                attention_mask[[i, j]] = mask[j] as i64;
                token_type_ids[[i, j]] = type_ids[j] as i64;
            }
        }

        // Create tensors
        let input_ids_tensor = ort::value::TensorRef::from_array_view(&input_ids)
            .map_err(|e| VelociError::Embedding(format!("Input tensor failed: {}", e)))?;
        let attention_mask_tensor = ort::value::TensorRef::from_array_view(&attention_mask)
            .map_err(|e| VelociError::Embedding(format!("Input tensor failed: {}", e)))?;
        let token_type_ids_tensor = ort::value::TensorRef::from_array_view(&token_type_ids)
            .map_err(|e| VelociError::Embedding(format!("Input tensor failed: {}", e)))?;

        let inputs = ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ];

        let outputs = session.run(inputs)
            .map_err(|e| VelociError::Embedding(format!("Cross-encoder inference failed: {}", e)))?;

        // Extract logits
        let (shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| VelociError::Embedding(format!("Output extraction failed: {}", e)))?;

        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

        let logits: Vec<f32> = if dims.len() == 2 && dims[1] == 1 {
            // shape [batch, 1] — regression logit
            (0..batch_size).map(|i| data[i]).collect()
        } else if dims.len() == 2 && dims[1] == 2 {
            // shape [batch, 2] — binary classification, take positive class
            (0..batch_size).map(|i| data[i * 2 + 1]).collect()
        } else {
            // shape [batch] — already flat
            data[..batch_size].to_vec()
        };

        // Sigmoid normalization → [0, 1]
        let scores = logits.iter()
            .map(|&x| 1.0 / (1.0 + (-x as f64).exp()))
            .collect();

        Ok(scores)
    }
}

impl Default for Reranker {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

/// Smart excerpting for long documents.
/// Takes head + tail to capture intro and conclusion.
fn excerpt_content(content: &str) -> String {
    if content.len() <= MAX_EXCERPT_LENGTH {
        return content.to_string();
    }
    // Find safe char boundaries to avoid panicking on multi-byte UTF-8
    let head_end = content.floor_char_boundary(EXCERPT_HEAD);
    let tail_start = content.ceil_char_boundary(content.len().saturating_sub(EXCERPT_TAIL));
    let head = &content[..head_end];
    let tail = &content[tail_start..];
    format!("{}\n...\n{}", head.trim_end(), tail.trim_start())
}

fn resolve_model_dir() -> PathBuf {
    // Try huggingface hub cache first (no download needed)
    let hf_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("huggingface")
        .join("hub")
        .join("models--cross-encoder--ms-marco-TinyBERT-L-2-v2");

    if hf_dir.exists() {
        let snapshots = hf_dir.join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.join("onnx/model.onnx").exists() {
                    return path;
                }
            }
        }
    }

    // Auto-download from HuggingFace
    match crate::download::ensure_model(&crate::download::RERANKER_MODEL) {
        Ok(path) => path,
        Err(e) => {
            tracing::error!("Failed to download reranker model: {}", e);
            // Return the default dir — load_model() will handle the missing files gracefully
            crate::download::models_cache_dir().join("cross-encoder-tinybert")
        }
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_excerpt_short() {
        let content = "Short content here.";
        assert_eq!(excerpt_content(content), content);
    }

    #[test]
    fn test_excerpt_long() {
        let content = "A".repeat(3000);
        let result = excerpt_content(&content);
        assert!(result.len() < content.len());
        assert!(result.contains("..."));
    }

    #[test]
    fn test_reranker_no_model() {
        let mut reranker = Reranker::new();
        // Should gracefully degrade when model not present
        let results = vec![
            RerankInput {
                content: "Hello world".to_string(),
                original_score: 0.9,
                doc_id: "doc1".to_string(),
                metadata: serde_json::json!({}),
            },
            RerankInput {
                content: "Goodbye world".to_string(),
                original_score: 0.5,
                doc_id: "doc2".to_string(),
                metadata: serde_json::json!({}),
            },
        ];
        let reranked = reranker.rerank("hello", results, 10).unwrap();
        assert_eq!(reranked.len(), 2);
        // Without model, should pass through with original scores
        assert_eq!(reranked[0].doc_id, "doc1");
    }
}
