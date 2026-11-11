//! ONNX Runtime embedding engine with LRU caching.
//!
//! Uses all-MiniLM-L6-v2 via ONNX Runtime for fast CPU inference.
//! MD5 content hashing with LRU eviction. Lazy model loading.
//!
//! Three ways to build one:
//! * [`Embedder::new`]: the model is found (cache folder, HuggingFace hub
//!   cache, models folder) and loaded on the first `embed()` call. It is
//!   never downloaded unless downloads were switched on
//!   ([`crate::download::set_offline`]).
//! * [`Embedder::from_model_dir`]: the model is loaded now, from exactly the
//!   folder given.
//! * [`Embedder::hashing`]: no model at all, a hashed bag of words. For
//!   tests, and for brains recorded as [`HASH_MODEL_ID`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
/// In-memory cache size of an embedder that has no disk cache.
const MEMORY_ONLY_CACHE_SIZE: usize = 256;

/// [`Embedder::model_id`] of the model-free hashing backend.
pub const HASH_MODEL_ID: &str = "hash-384";

/// What turns text into vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    /// all-MiniLM-L6-v2 through ONNX Runtime.
    Onnx,
    /// No model: see [`hash_embed`].
    Hash,
}

// ── Embedder ────────────────────────────────────────────────────────────────

pub struct Embedder {
    session: Option<Session>,
    tokenizer: Option<Tokenizer>,
    cache: Mutex<LruCache>,
    cache_dir: Option<PathBuf>,
    normalize: bool,
    backend: Backend,
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
            backend: Backend::Onnx,
        }
    }

    /// Load the model from exactly `dir`, now: `onnx/model.onnx`, else
    /// `model.onnx`, and `tokenizer.json`. No other folder is looked at and
    /// nothing is downloaded or created; missing files are an error. The
    /// embedder has no disk cache and an in-memory cache of 256 entries.
    pub fn from_model_dir(dir: &Path, threads: usize, normalize: bool) -> Result<Self> {
        let (session, tokenizer) = load_model(dir, threads)?;
        Ok(Self {
            session: Some(session),
            tokenizer: Some(tokenizer),
            cache: Mutex::new(LruCache::new(MEMORY_ONLY_CACHE_SIZE)),
            cache_dir: None,
            normalize,
            backend: Backend::Onnx,
        })
    }

    /// An embedder with no model: each text becomes a hashed bag of its
    /// words (see [`hash_embed`]). Deterministic, touches no file and no
    /// network. Its vectors only say how many words two texts share, so it
    /// is for tests and for brains recorded as [`HASH_MODEL_ID`]; they must
    /// never be mixed with vectors of the ONNX model.
    pub fn hashing() -> Self {
        Self {
            session: None,
            tokenizer: None,
            cache: Mutex::new(LruCache::new(MEMORY_ONLY_CACHE_SIZE)),
            cache_dir: None,
            normalize: true,
            backend: Backend::Hash,
        }
    }

    /// The name of the model that makes this embedder's vectors:
    /// `"all-MiniLM-L6-v2"` or [`HASH_MODEL_ID`]. A brain records it, and
    /// vectors of different models must not share an index.
    pub fn model_id(&self) -> &'static str {
        match self.backend {
            Backend::Onnx => MODEL_NAME,
            Backend::Hash => HASH_MODEL_ID,
        }
    }

    /// Save the embedding cache to disk.
    ///
    /// No-op when the cache is empty — a lexical-only session must leave
    /// zero filesystem footprint under the cache dir (T35: no implicit
    /// model/cache activity by default).
    pub fn save_cache(&self) {
        if let Some(ref dir) = self.cache_dir {
            let mut cache = self.cache.lock().unwrap();
            if cache.entries.is_empty() {
                return;
            }
            std::fs::create_dir_all(dir).ok();
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

        if self.backend == Backend::Hash {
            return Ok(texts.iter().map(|text| hash_embed(text)).collect());
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
        let (session, tokenizer) = load_model(&model_dir, num_cpus())?;

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

        // The models folder. Offline (the default) this is an error when the
        // files are missing; it downloads only after `set_offline(false)`.
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

// ── Model loading ───────────────────────────────────────────────────────────

/// Load the ONNX session and the tokenizer from `model_dir`. Reads only;
/// missing files are an error.
fn load_model(model_dir: &Path, threads: usize) -> Result<(Session, Tokenizer)> {
    // Support both layouts: onnx/model.onnx (HF style) and model.onnx (flat/Python style)
    let onnx_path = if model_dir.join("onnx").join("model.onnx").exists() {
        model_dir.join("onnx").join("model.onnx")
    } else {
        model_dir.join("model.onnx")
    };
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !onnx_path.exists() || !tokenizer_path.exists() {
        return Err(VelociError::Embedding(format!(
            "Model files not found at {} (onnx/model.onnx or model.onnx, and tokenizer.json). \
             Run `axel model fetch`.",
            model_dir.display()
        )));
    }

    tracing::info!("Loading ONNX model from {}", onnx_path.display());

    let session = Session::builder()
        .map_err(|e| VelociError::Embedding(format!("Failed to create session builder: {}", e)))?
        .with_intra_threads(threads.max(1))
        .map_err(|e| VelociError::Embedding(format!("Failed to set thread count: {}", e)))?
        .commit_from_file(&onnx_path)
        .map_err(|e| VelociError::Embedding(format!("Failed to load ONNX model: {}", e)))?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| VelociError::Embedding(format!("Failed to load tokenizer: {}", e)))?;

    Ok((session, tokenizer))
}

// ── Hashing backend ─────────────────────────────────────────────────────────

/// The model-free vector of a text. Its words are the runs of alphanumeric
/// characters, lowercased, of 2 or more characters. Each word is hashed with
/// FNV-1a 64 and adds +1 or -1 (bit 63 of the hash set: -1) at index
/// `hash % 384`. The sum is L2-normalised. A text with no word gives the
/// zero vector.
fn hash_embed(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0f32; EMBEDDING_DIM];
    let mut word = String::new();
    let mut add = |word: &mut String| {
        if word.chars().count() >= 2 {
            let hash = fnv1a64(word.as_bytes());
            let sign = if hash >> 63 == 1 { -1.0 } else { 1.0 };
            vector[(hash % EMBEDDING_DIM as u64) as usize] += sign;
        }
        word.clear();
    };
    for c in text.chars() {
        if c.is_alphanumeric() {
            word.extend(c.to_lowercase());
        } else {
            add(&mut word);
        }
    }
    add(&mut word);

    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

/// FNV-1a, 64 bit.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        // Both are unit length.
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    #[test]
    fn fnv1a64_matches_the_reference_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn hashing_is_deterministic_and_unit_length() {
        let text = "Rust is a systems programming language focused on safety";
        let mut one = Embedder::hashing();
        let mut two = Embedder::hashing();
        let first = one.embed_one(text).unwrap();

        assert_eq!(first.len(), EMBEDDING_DIM);
        assert_eq!(one.dim(), EMBEDDING_DIM);
        assert!((norm(&first) - 1.0).abs() < 1e-5, "norm {}", norm(&first));
        // The same instance again, another instance, and the batch call.
        assert_eq!(one.embed_one(text).unwrap(), first);
        assert_eq!(two.embed_one(text).unwrap(), first);
        assert_eq!(two.embed(&["other words", text]).unwrap()[1], first);
    }

    #[test]
    fn hashing_puts_texts_that_share_words_closer() {
        let mut embedder = Embedder::hashing();
        let query = embedder.embed_one("memory safety in rust").unwrap();
        let shares = embedder
            .embed_one("rust programming language memory safety borrow checker")
            .unwrap();
        let shares_none = embedder
            .embed_one("the quick brown fox jumps over the lazy dog")
            .unwrap();

        assert!(cosine(&query, &shares) > 0.4, "{}", cosine(&query, &shares));
        assert!(cosine(&query, &shares) > cosine(&query, &shares_none) + 0.2);
    }

    #[test]
    fn hashing_words_are_lowercase_alphanumeric_runs_of_two_or_more() {
        let mut embedder = Embedder::hashing();
        // Case and punctuation do not matter.
        assert_eq!(
            embedder.embed_one("Borrow-Checker: RUST!").unwrap(),
            embedder.embed_one("borrow checker rust").unwrap()
        );
        // One word, whatever its count, is one direction.
        assert_eq!(
            embedder.embed_one("rust rust rust").unwrap(),
            embedder.embed_one("rust").unwrap()
        );
        // Single characters are not words; a text without words is the zero vector.
        assert_eq!(
            embedder.embed_one("a rust b").unwrap(),
            embedder.embed_one("rust").unwrap()
        );
        assert!(embedder.embed_one("a b c ! ?").unwrap().iter().all(|x| *x == 0.0));
        assert!(embedder.embed_one("").unwrap().iter().all(|x| *x == 0.0));
    }

    #[test]
    fn model_id_names_the_backend_without_loading_anything() {
        assert_eq!(Embedder::hashing().model_id(), "hash-384");
        assert_eq!(Embedder::hashing().model_id(), HASH_MODEL_ID);
        // Lazy: constructing the ONNX embedder loads no model.
        assert_eq!(Embedder::new(None, None, true).model_id(), "all-MiniLM-L6-v2");
    }

    #[test]
    fn from_model_dir_on_an_empty_folder_is_an_error_and_creates_nothing() {
        let dir = tempfile::TempDir::new().unwrap();

        let error = match Embedder::from_model_dir(dir.path(), 2, true) {
            Ok(_) => panic!("an empty folder must not give an embedder"),
            Err(error) => error.to_string(),
        };

        assert!(error.contains(&dir.path().display().to_string()), "{error}");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0, "the folder must stay empty");

        // A folder that does not exist is an error too, and is not created.
        let missing = dir.path().join("missing");
        assert!(Embedder::from_model_dir(&missing, 2, true).is_err());
        assert!(!missing.exists());
    }

    #[test]
    #[ignore = "needs the ONNX model: set AXEL_TEST_MODEL_DIR and run with --ignored"]
    fn from_model_dir_loads_the_model_and_embeds() {
        let dir = std::env::var_os("AXEL_TEST_MODEL_DIR")
            .map(PathBuf::from)
            .expect("set AXEL_TEST_MODEL_DIR");
        let mut embedder = Embedder::from_model_dir(&dir, 2, true).unwrap();

        assert_eq!(embedder.model_id(), "all-MiniLM-L6-v2");
        let vectors = embedder
            .embed(&["a cat sat on the mat", "a kitten rested on the rug", "quarterly tax filing"])
            .unwrap();
        assert!((norm(&vectors[0]) - 1.0).abs() < 1e-4);
        // Real semantics: no shared content words, still the closer pair.
        assert!(cosine(&vectors[0], &vectors[1]) > cosine(&vectors[0], &vectors[2]));
    }
}
