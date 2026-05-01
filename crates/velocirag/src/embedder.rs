//! ONNX Runtime embedding engine with LRU caching.
//!
//! Uses all-MiniLM-L6-v2 via ONNX Runtime for fast CPU inference.
//! MD5 content hashing with LRU eviction. Lazy model loading.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use md5::{Digest, Md5};
use ndarray::Array2;
use ort::session::Session;
use tokenizers::Tokenizer;

use crate::error::{Result, VelociError};

// ── Constants ───────────────────────────────────────────────────────────────

const MODEL_NAME: &str = "all-MiniLM-L6-v2";
const _MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
use crate::EMBEDDING_DIM;
const DEFAULT_CACHE_SIZE: usize = 10_000;
const MAX_BATCH_SIZE: usize = 256;
const MAX_SEQ_LENGTH: usize = 256;

// ── Embedder ────────────────────────────────────────────────────────────────

pub struct Embedder {
    session: Option<Session>,
    tokenizer: Option<Tokenizer>,
    cache: Mutex<LruCache>,
    cache_dir: Option<PathBuf>,
    normalize: bool,
}

/// Simple LRU cache backed by a Vec (order = recency) and HashMap for O(1) lookup.
/// Supports disk persistence via JSON save/load.
struct LruCache {
    entries: Vec<(String, Vec<f32>)>,
    index: HashMap<String, usize>,
    max_size: usize,
    new_entries: usize, // track unsaved entries
}

const CACHE_FILENAME: &str = "embedding_cache.json";
const CACHE_VERSION: u32 = 3;
const CACHE_SAVE_INTERVAL: usize = 200;

impl LruCache {
    fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_size.min(1024)),
            index: HashMap::with_capacity(max_size.min(1024)),
            max_size,
            new_entries: 0,
        }
    }

    fn get(&mut self, key: &str) -> Option<Vec<f32>> {
        if let Some(&idx) = self.index.get(key) {
            // Move to end (most recent)
            let entry = self.entries.remove(idx);
            let vec = entry.1.clone();
            self.entries.push(entry);
            // Rebuild index (expensive but rare relative to embed calls)
            self.rebuild_index();
            Some(vec)
        } else {
            None
        }
    }

    fn insert(&mut self, key: String, value: Vec<f32>) {
        if self.entries.len() >= self.max_size {
            // Evict oldest
            let evicted = self.entries.remove(0);
            self.index.remove(&evicted.0);
            self.rebuild_index();
        }
        let idx = self.entries.len();
        self.entries.push((key.clone(), value));
        self.index.insert(key, idx);
        self.new_entries += 1;
    }

    fn rebuild_index(&mut self) {
        self.index.clear();
        for (i, (key, _)) in self.entries.iter().enumerate() {
            self.index.insert(key.clone(), i);
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Save cache to disk as JSON.
    fn save_to_disk(&mut self, dir: &std::path::Path) -> std::result::Result<(), String> {
        let cache_path = dir.join(CACHE_FILENAME);
        let data = serde_json::json!({
            "version": CACHE_VERSION,
            "model": MODEL_NAME,
            "dim": EMBEDDING_DIM,
            "embeddings": self.entries.iter().map(|(k, v)| {
                serde_json::json!({"hash": k, "embedding": v})
            }).collect::<Vec<_>>(),
        });

        let temp_path = cache_path.with_extension("json.tmp");
        std::fs::write(&temp_path, serde_json::to_string(&data).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        std::fs::rename(&temp_path, &cache_path).map_err(|e| e.to_string())?;

        tracing::info!("Cache saved: {} entries to {}", self.entries.len(), cache_path.display());
        self.new_entries = 0;
        Ok(())
    }

    /// Load cache from disk.
    fn load_from_disk(&mut self, dir: &std::path::Path) -> std::result::Result<(), String> {
        let cache_path = dir.join(CACHE_FILENAME);
        if !cache_path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&cache_path).map_err(|e| e.to_string())?;
        let data: serde_json::Value = serde_json::from_str(&content).map_err(|e| e.to_string())?;

        // Validate version
        let version = data.get("version").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        if version != CACHE_VERSION {
            tracing::warn!("Cache version mismatch (got {}, expected {}). Starting fresh.", version, CACHE_VERSION);
            return Ok(());
        }

        // Validate dimension
        let dim = data.get("dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if dim != EMBEDDING_DIM {
            tracing::warn!("Cache dim mismatch (got {}, expected {}). Starting fresh.", dim, EMBEDDING_DIM);
            return Ok(());
        }

        let embeddings = data.get("embeddings").and_then(|v| v.as_array());
        let Some(embeddings) = embeddings else {
            return Ok(());
        };

        let mut loaded = 0;
        for entry in embeddings {
            let hash = entry.get("hash").and_then(|v| v.as_str());
            let embedding = entry.get("embedding").and_then(|v| v.as_array());

            if let (Some(hash), Some(emb_arr)) = (hash, embedding) {
                let emb: Vec<f32> = emb_arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                if emb.len() == EMBEDDING_DIM {
                    self.insert(hash.to_string(), emb);
                    loaded += 1;
                }
            }
        }

        self.new_entries = 0; // don't count loaded entries as "new"
        tracing::info!("Cache loaded: {} entries from {}", loaded, cache_path.display());
        Ok(())
    }
}

impl Embedder {
    /// Create a new embedder. Model is loaded lazily on first embed() call.
    pub fn new(cache_dir: Option<PathBuf>, cache_size: Option<usize>, normalize: bool) -> Self {
        let mut cache = LruCache::new(cache_size.unwrap_or(DEFAULT_CACHE_SIZE));

        // Load disk cache if cache_dir provided
        if let Some(ref dir) = cache_dir {
            if dir.exists() {
                if let Err(e) = cache.load_from_disk(dir) {
                    tracing::warn!("Failed to load embedding cache: {}", e);
                }
            }
        }

        Self {
            session: None,
            tokenizer: None,
            cache: Mutex::new(cache),
            cache_dir,
            normalize,
        }
    }

    /// Save the embedding cache to disk.
    pub fn save_cache(&self) {
        if let Some(ref dir) = self.cache_dir {
            std::fs::create_dir_all(dir).ok();
            let mut cache = self.cache.lock().unwrap();
            if let Err(e) = cache.save_to_disk(dir) {
                tracing::warn!("Failed to save embedding cache: {}", e);
            }
        }
    }

    /// Save cache if enough new entries have accumulated.
    fn maybe_save_cache(&self) {
        let should_save = {
            let cache = self.cache.lock().unwrap();
            cache.new_entries >= CACHE_SAVE_INTERVAL
        };
        if should_save {
            self.save_cache();
        }
    }

    /// Embed one or more texts. Returns an (N, 384) matrix of embeddings.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        self.ensure_model_loaded()?;

        // Check cache for all texts
        let mut results: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        let mut uncached_indices = Vec::new();
        let mut uncached_texts = Vec::new();

        {
            let mut cache = self.cache.lock().unwrap();
            for (i, text) in texts.iter().enumerate() {
                let hash = md5_hash(text);
                if let Some(cached) = cache.get(&hash) {
                    results.push(Some(cached));
                } else {
                    results.push(None);
                    uncached_indices.push(i);
                    uncached_texts.push(*text);
                }
            }
        }

        // Embed uncached texts in batches
        if !uncached_texts.is_empty() {
            let new_embeddings = self.embed_batch(&uncached_texts)?;

            let mut cache = self.cache.lock().unwrap();
            for (batch_idx, &original_idx) in uncached_indices.iter().enumerate() {
                let embedding = new_embeddings[batch_idx].clone();
                let hash = md5_hash(texts[original_idx]);
                cache.insert(hash, embedding.clone());
                results[original_idx] = Some(embedding);
            }
        }

        Ok(results.into_iter().map(|r| r.unwrap()).collect())
    }

    /// Embed a single text.
    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed(&[text])?;
        self.maybe_save_cache();
        Ok(results.into_iter().next().unwrap())
    }

    /// Number of cached embeddings.
    pub fn cache_len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Embedding dimensionality.
    pub fn dim(&self) -> usize {
        EMBEDDING_DIM
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn ensure_model_loaded(&mut self) -> Result<()> {
        if self.session.is_some() {
            return Ok(());
        }

        let model_dir = self.resolve_model_dir()?;
        // Support both layouts: onnx/model.onnx (HF style) and model.onnx (flat/Python style)
        let onnx_path = if model_dir.join("onnx").join("model.onnx").exists() {
            model_dir.join("onnx").join("model.onnx")
        } else {
            model_dir.join("model.onnx")
        };
        let tokenizer_path = model_dir.join("tokenizer.json");

        if !onnx_path.exists() || !tokenizer_path.exists() {
            return Err(VelociError::Embedding(format!(
                "Model files not found at {}. Run `velocirag download-model` first, \
                 or ensure huggingface-hub has cached the model.",
                model_dir.display()
            )));
        }

        tracing::info!("Loading ONNX model from {}", onnx_path.display());

        let session = Session::builder()
            .map_err(|e| VelociError::Embedding(format!("Failed to create session builder: {}", e)))?
            .with_intra_threads(num_cpus())
            .map_err(|e| VelociError::Embedding(format!("Failed to set thread count: {}", e)))?
            .commit_from_file(&onnx_path)
            .map_err(|e| VelociError::Embedding(format!("Failed to load ONNX model: {}", e)))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| VelociError::Embedding(format!("Failed to load tokenizer: {}", e)))?;

        self.session = Some(session);
        self.tokenizer = Some(tokenizer);

        tracing::info!("Model loaded (dim={})", EMBEDDING_DIM);
        Ok(())
    }

    fn resolve_model_dir(&self) -> Result<PathBuf> {
        // Check custom cache dir first
        if let Some(ref dir) = self.cache_dir {
            if dir.join("onnx/model.onnx").exists() {
                return Ok(dir.clone());
            }
            // Also check flat layout (Python velocirag stores model.onnx directly)
            if dir.join("model.onnx").exists() && dir.join("tokenizer.json").exists() {
                return Ok(dir.clone());
            }
        }

        // Try huggingface hub cache first (no download needed)
        let hf_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("huggingface")
            .join("hub")
            .join(format!("models--sentence-transformers--{}", MODEL_NAME));

        if hf_dir.exists() {
            let snapshots = hf_dir.join("snapshots");
            if let Ok(entries) = std::fs::read_dir(&snapshots) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.join("onnx/model.onnx").exists() {
                        return Ok(path);
                    }
                }
            }
        }

        // Auto-download from HuggingFace
        crate::download::ensure_model(&crate::download::EMBEDDER_MODEL)
    }

    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let session = self.session.as_mut().unwrap();
        let tokenizer = self.tokenizer.as_ref().unwrap();

        let mut all_embeddings = Vec::with_capacity(texts.len());

        // Process in sub-batches
        for chunk in texts.chunks(MAX_BATCH_SIZE) {
            let encodings = tokenizer
                .encode_batch(chunk.to_vec(), true)
                .map_err(|e| VelociError::Embedding(format!("Tokenization failed: {}", e)))?;

            let batch_size = encodings.len();
            let max_len = encodings
                .iter()
                .map(|e| e.get_ids().len().min(MAX_SEQ_LENGTH))
                .max()
                .unwrap_or(0);

            // Build padded input tensors
            let mut input_ids = Array2::<i64>::zeros((batch_size, max_len));
            let mut attention_mask = Array2::<i64>::zeros((batch_size, max_len));
            let token_type_ids = Array2::<i64>::zeros((batch_size, max_len));

            for (i, encoding) in encodings.iter().enumerate() {
                let ids = encoding.get_ids();
                let mask = encoding.get_attention_mask();
                let len = ids.len().min(max_len);
                for j in 0..len {
                    input_ids[[i, j]] = ids[j] as i64;
                    attention_mask[[i, j]] = mask[j] as i64;
                }
            }

            // Run inference — ort v2 needs TensorRef for array views
            let input_ids_tensor = ort::value::TensorRef::from_array_view(&input_ids)
                .map_err(|e| VelociError::Embedding(format!("Input creation failed: {}", e)))?;
            let attention_mask_tensor = ort::value::TensorRef::from_array_view(&attention_mask)
                .map_err(|e| VelociError::Embedding(format!("Input creation failed: {}", e)))?;
            let token_type_ids_tensor = ort::value::TensorRef::from_array_view(&token_type_ids)
                .map_err(|e| VelociError::Embedding(format!("Input creation failed: {}", e)))?;

            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ];

            let outputs = session
                .run(inputs)
                .map_err(|e| VelociError::Embedding(format!("ONNX inference failed: {}", e)))?;

            // Extract embeddings — mean pooling over token dimension
            let output_value = &outputs[0];
            let (shape, data) = output_value
                .try_extract_tensor::<f32>()
                .map_err(|e| VelociError::Embedding(format!("Output extraction failed: {}", e)))?;

            // shape = [batch, seq_len, hidden_dim]
            let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let seq_dim = dims.get(1).copied().unwrap_or(0);
            let hidden_dim = dims.get(2).copied().unwrap_or(EMBEDDING_DIM);

            // Mean pool: average over seq_len dimension, masked by attention
            for i in 0..batch_size {
                let mut embedding = vec![0.0f32; EMBEDDING_DIM];
                let seq_len = encodings[i].get_ids().len().min(max_len).min(seq_dim);

                for j in 0..seq_len {
                    if attention_mask[[i, j]] == 1 {
                        let offset = i * seq_dim * hidden_dim + j * hidden_dim;
                        for k in 0..EMBEDDING_DIM.min(hidden_dim) {
                            embedding[k] += data[offset + k];
                        }
                    }
                }

                // Divide by number of non-padding tokens
                let count = seq_len as f32;
                if count > 0.0 {
                    for val in &mut embedding {
                        *val /= count;
                    }
                }

                // L2 normalize if requested
                if self.normalize {
                    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for val in &mut embedding {
                            *val /= norm;
                        }
                    }
                }

                all_embeddings.push(embedding);
            }
        }

        Ok(all_embeddings)
    }
}

// ── Utility ─────────────────────────────────────────────────────────────────

fn md5_hash(text: &str) -> String {
    let mut hasher = Md5::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

impl Drop for Embedder {
    fn drop(&mut self) {
        self.save_cache();
    }
}
