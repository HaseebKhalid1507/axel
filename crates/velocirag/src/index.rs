//! HNSW vector index for approximate nearest neighbor search.
//!
//! Wraps usearch for fast ANN queries. Replaces FAISS from the Python version.
//! Supports incremental adds and persistence to disk.

use std::path::{Path, PathBuf};

use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::error::{Result, VelociError};

// ── Constants ───────────────────────────────────────────────────────────────

const EMBEDDING_DIM: usize = 384;
const DEFAULT_CONNECTIVITY: usize = 16;     // M parameter — edges per node
const DEFAULT_EF_CONSTRUCTION: usize = 128; // Build-time search depth
const DEFAULT_EF_SEARCH: usize = 64;        // Query-time search depth

// ── Vector Index ────────────────────────────────────────────────────────────

pub struct VectorIndex {
    index: Index,
    count: usize,
    path: Option<PathBuf>,
}

/// A search result from the vector index.
#[derive(Debug, Clone)]
pub struct VectorResult {
    /// Row ID in the database (used as the usearch key).
    pub id: u64,
    /// Cosine similarity (higher = more similar).
    pub score: f32,
}

fn make_options() -> IndexOptions {
    IndexOptions {
        dimensions: EMBEDDING_DIM,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        connectivity: DEFAULT_CONNECTIVITY,
        expansion_add: DEFAULT_EF_CONSTRUCTION,
        expansion_search: DEFAULT_EF_SEARCH,
        multi: false,
    }
}

fn create_index() -> Result<Index> {
    let opts = make_options();
    let index = Index::new(&opts)
        .map_err(|e| VelociError::Index(format!("Failed to create index: {}", e)))?;
    Ok(index)
}

impl VectorIndex {
    /// Create a new empty index.
    pub fn new() -> Result<Self> {
        let index = create_index()?;

        // Reserve initial capacity
        index
            .reserve(10_000)
            .map_err(|e| VelociError::Index(format!("Failed to reserve: {}", e)))?;

        Ok(Self {
            index,
            count: 0,
            path: None,
        })
    }

    /// Load an index from disk.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(VelociError::NotFound(format!(
                "Index file not found: {}",
                path.display()
            )));
        }

        let index = create_index()?;

        index
            .load(path.to_str().unwrap())
            .map_err(|e| VelociError::Index(format!("Failed to load index: {}", e)))?;

        let count = index.size();

        Ok(Self {
            index,
            count,
            path: Some(path.to_path_buf()),
        })
    }

    /// Save the index to disk.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        self.index
            .save(path.to_str().unwrap())
            .map_err(|e| VelociError::Index(format!("Failed to save index: {}", e)))?;
        Ok(())
    }

    /// Add a vector with the given key (database row ID).
    pub fn add(&mut self, key: u64, embedding: &[f32]) -> Result<()> {
        if embedding.len() != EMBEDDING_DIM {
            return Err(VelociError::DimensionMismatch {
                expected: EMBEDDING_DIM,
                got: embedding.len(),
            });
        }

        // Auto-expand capacity
        if self.count >= self.index.capacity() {
            let new_cap = (self.index.capacity() * 2).max(10_000);
            self.index
                .reserve(new_cap)
                .map_err(|e| VelociError::Index(format!("Failed to reserve: {}", e)))?;
        }

        self.index
            .add(key, embedding)
            .map_err(|e| VelociError::Index(format!("Failed to add vector: {}", e)))?;
        self.count += 1;
        Ok(())
    }

    /// Add a batch of vectors.
    pub fn add_batch(&mut self, keys: &[u64], embeddings: &[Vec<f32>]) -> Result<()> {
        for (key, emb) in keys.iter().zip(embeddings.iter()) {
            self.add(*key, emb)?;
        }
        Ok(())
    }

    /// Search for the k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<VectorResult>> {
        if query.len() != EMBEDDING_DIM {
            return Err(VelociError::DimensionMismatch {
                expected: EMBEDDING_DIM,
                got: query.len(),
            });
        }

        if self.count == 0 {
            return Ok(Vec::new());
        }

        let k = k.min(self.count);
        let results = self
            .index
            .search(query, k)
            .map_err(|e| VelociError::Index(format!("Search failed: {}", e)))?;

        Ok(results
            .keys
            .iter()
            .zip(results.distances.iter())
            .map(|(&id, &dist)| VectorResult {
                id,
                // usearch cosine distance = 1 - similarity, so invert
                score: 1.0 - dist,
            })
            .collect())
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Rebuild the index from a set of (key, embedding) pairs.
    /// Used after bulk inserts or when loading from database.
    pub fn rebuild(&mut self, data: &[(u64, Vec<f32>)]) -> Result<()> {
        let new_index = create_index()?;

        new_index
            .reserve(data.len().max(1_000))
            .map_err(|e| VelociError::Index(format!("Failed to reserve: {}", e)))?;

        for (key, emb) in data {
            new_index
                .add(*key, emb)
                .map_err(|e| VelociError::Index(format!("Failed to add: {}", e)))?;
        }

        self.index = new_index;
        self.count = data.len();
        Ok(())
    }
}

impl Default for VectorIndex {
    fn default() -> Self {
        Self::new().expect("Failed to create default vector index")
    }
}
