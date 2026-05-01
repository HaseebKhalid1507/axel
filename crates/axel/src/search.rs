//! Search bridge — connects a `.r8` Brain to VelociRAG's 4-layer search.
//!
//! The Brain owns the SQLite file. VelociRAG opens a second read connection
//! to the same file (SQLite WAL supports unlimited concurrent readers).
//! The HNSW vector index is built in memory from stored embeddings on first
//! search, then persisted to disk for fast warm starts on subsequent opens.

use std::path::{Path, PathBuf};

use velocirag::db::Database;
use velocirag::embedder::Embedder;
use velocirag::index::VectorIndex;
use velocirag::search::{SearchEngine, SearchOptions, SearchResponse};

use crate::error::{AxelError, Result};

/// Search handle for a `.r8` brain.
///
/// Owns a VelociRAG Database (second connection to the .r8 file),
/// an Embedder (ONNX model), and a VectorIndex (HNSW, built from
/// stored embeddings on first use and persisted to disk).
pub struct BrainSearch {
    db: Database,
    embedder: Embedder,
    index: VectorIndex,
    index_built: bool,
    /// Path where the HNSW index is persisted: `<brain>.r8.hnsw`
    cache_path: PathBuf,
    /// Set to true when the live index has been modified since the last save.
    cache_stale: bool,
}

impl BrainSearch {
    /// Initialize search for a brain at the given path.
    ///
    /// Opens a second SQLite connection (read-only for search).
    /// Downloads the embedding model on first use if not cached.
    /// If a matching on-disk HNSW cache exists, loads it instead of
    /// rebuilding from raw embeddings.
    pub fn open(brain_path: &Path) -> Result<Self> {
        // VelociRAG's Database::open expects a directory and creates velocirag.db inside it.
        // For .r8 files, we need from_connection with the actual file path.
        let conn = rusqlite::Connection::open_with_flags(
            brain_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(AxelError::Db)?;

        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(AxelError::Db)?;

        let db = Database::from_connection(conn, brain_path.to_path_buf())
            .map_err(|e| AxelError::Search(e.to_string()))?;

        // Wire embedding cache: persist to ~/.cache/axel/embeddings so
        // repeated embed() calls for the same content are free across sessions.
        let embedding_cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("axel")
            .join("embeddings");

        let embedder = Embedder::new(Some(embedding_cache_dir), None, true);

        // Derive the HNSW index cache path from the brain file path.
        let cache_path = brain_path.with_extension("r8.hnsw");

        // Try to load a warm index from disk if it exists and the vector
        // count matches what the database currently holds.
        let (index, index_built) = if cache_path.exists() {
            let db_count = db
                .document_count()
                .unwrap_or(0);

            match VectorIndex::load(&cache_path) {
                Ok(loaded) if loaded.len() == db_count && db_count > 0 => {
                    eprintln!("Loading cached index ({} vectors)...", loaded.len());
                    (loaded, true)
                }
                Ok(_) => {
                    // Count mismatch — stale cache. Fall back to empty; build_index() rebuilds.
                    tracing::debug!("HNSW cache count mismatch, will rebuild on next search.");
                    let idx = VectorIndex::new()
                        .map_err(|e| AxelError::Search(format!("Failed to create index: {e}")))?;
                    (idx, false)
                }
                Err(e) => {
                    tracing::warn!("Failed to load HNSW cache, will rebuild: {e}");
                    let idx = VectorIndex::new()
                        .map_err(|e| AxelError::Search(format!("Failed to create index: {e}")))?;
                    (idx, false)
                }
            }
        } else {
            let idx = VectorIndex::new()
                .map_err(|e| AxelError::Search(format!("Failed to create index: {e}")))?;
            (idx, false)
        };

        Ok(Self {
            db,
            embedder,
            index,
            index_built,
            cache_path,
            cache_stale: false,
        })
    }

    /// Build the HNSW vector index from embeddings stored in SQLite.
    ///
    /// This is the "cold start" path — loads all embeddings from the database
    /// and builds an in-memory HNSW index. For 7K docs this takes ~70ms.
    /// Called automatically on first search if not already built.
    /// After building, persists the index to `<brain>.r8.hnsw`.
    pub fn build_index(&mut self) -> Result<()> {
        if self.index_built {
            return Ok(());
        }

        let embeddings = self.db.load_all_embeddings()
            .map_err(|e| AxelError::Search(format!("Failed to load embeddings: {e}")))?;

        if !embeddings.is_empty() {
            eprintln!("Building search index ({} vectors)...", embeddings.len());
        }

        for (key, embedding) in &embeddings {
            self.index.add(*key as u64, embedding)
                .map_err(|e| AxelError::Search(format!("Failed to add to index: {e}")))?;
        }

        self.index_built = true;
        tracing::info!(count = embeddings.len(), "HNSW index built from stored embeddings");

        // Persist the freshly built index so the next open() is a warm load.
        if let Err(e) = self.index.save(&self.cache_path) {
            tracing::warn!("Failed to persist HNSW index cache: {e}");
        } else {
            tracing::debug!("HNSW index saved to {}", self.cache_path.display());
        }
        self.cache_stale = false;

        Ok(())
    }

    /// Flush any pending index changes to disk.
    ///
    /// Called automatically on drop if the cache is stale, but can also be
    /// called explicitly after bulk `index_document()` operations.
    pub fn flush(&mut self) -> Result<()> {
        if self.cache_stale && self.index_built {
            self.index.save(&self.cache_path)
                .map_err(|e| AxelError::Search(format!("Failed to flush HNSW cache: {e}")))?;
            self.cache_stale = false;
            tracing::debug!("HNSW index flushed to {}", self.cache_path.display());
        }
        Ok(())
    }

    /// Search the brain for documents matching a query.
    ///
    /// Builds the HNSW index on first call (cold start).
    /// Uses all 4 layers: vector similarity, BM25 keywords, graph, metadata.
    pub fn search(&mut self, query: &str, limit: usize) -> Result<SearchResponse> {
        self.build_index()?;

        let opts = SearchOptions {
            limit,
            ..Default::default()
        };

        let mut engine = SearchEngine::new(
            &self.db,
            &self.index,
            &mut self.embedder,
            None, // no reranker by default
        );

        engine.search(query, &opts)
            .map_err(|e| AxelError::Search(e.to_string()))
    }

    /// Index a document into the brain (embed + store + FTS).
    ///
    /// After adding to the live index the on-disk cache is marked stale.
    /// Call `flush()` to persist immediately, or rely on the `Drop` impl.
    pub fn index_document(
        &mut self,
        doc_id: &str,
        content: &str,
        metadata: Option<serde_json::Value>,
        file_path: Option<&str>,
    ) -> Result<()> {
        let embedding = self.embedder.embed_one(content)
            .map_err(|e| AxelError::Search(format!("Embedding failed: {e}")))?;

        let rowid = self.db.insert_document(
            doc_id,
            content,
            &metadata.unwrap_or(serde_json::json!({})),
            &embedding,
            file_path,
        ).map_err(|e| AxelError::Search(format!("Insert failed: {e}")))?;

        // Add to live index so it's searchable immediately.
        self.index.add(rowid as u64, &embedding)
            .map_err(|e| AxelError::Search(format!("Index add failed: {e}")))?;

        self.index_built = true; // mark as built since we have at least one vector
        self.cache_stale = true; // on-disk cache is now behind the live index
        Ok(())
    }

    /// Index a memory as a searchable document.
    pub fn index_memory(&mut self, memory: &axel_memkoshi::memory::Memory) -> Result<()> {
        let content = format!(
            "{}\n\n{}\n\n{}",
            memory.title, memory.abstract_text, memory.content
        );
        let metadata = serde_json::json!({
            "type": "memory",
            "category": format!("{:?}", memory.category),
            "topic": memory.topic,
            "importance": memory.importance,
            "tags": memory.tags,
        });

        self.index_document(&memory.id, &content, Some(metadata), None)
    }

    /// Get a reference to the underlying database.
    pub fn db(&self) -> &Database {
        &self.db
    }

    /// Record search feedback: access events + co-retrieval pairs.
    /// Called after every search (CLI and MCP) to feed consolidation data.
    /// All logging is best-effort — failures are silently ignored.
    pub fn record_search_feedback(&self, query: &str, results: &[velocirag::search::SearchResult]) {
        let db = &self.db;
        for r in results {
            let _ = db.log_document_access(&r.doc_id, "search_hit", Some(query), Some(r.score), None);
            let _ = db.increment_document_access(&r.doc_id);
        }
        let top_ids: Vec<&str> = results.iter().take(5).map(|r| r.doc_id.as_str()).collect();
        for i in 0..top_ids.len() {
            for j in (i+1)..top_ids.len() {
                let _ = db.log_co_retrieval(top_ids[i], top_ids[j], query);
            }
        }
    }
}

impl Drop for BrainSearch {
    /// Flush a stale HNSW cache to disk on drop so we don't lose incremental
    /// index_document() additions across sessions.
    fn drop(&mut self) {
        if self.cache_stale && self.index_built {
            if let Err(e) = self.index.save(&self.cache_path) {
                tracing::warn!("Failed to persist HNSW index on drop: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r8::Brain;

    fn tmp_brain_with_search() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.r8");
        Brain::create(&path, Some("test")).unwrap();
        (dir, path)
    }

    #[test]
    fn open_search_on_brain() {
        let (_dir, path) = tmp_brain_with_search();
        let search = BrainSearch::open(&path);
        assert!(search.is_ok(), "Failed to open search: {:?}", search.err());
    }

    #[test]
    fn index_and_search_document() {
        let (_dir, path) = tmp_brain_with_search();
        let mut search = BrainSearch::open(&path).unwrap();

        // Index a document
        search.index_document(
            "doc_1",
            "Rust is a systems programming language focused on safety and performance",
            None,
            None,
        ).unwrap();

        // Search for it
        let results = search.search("systems programming safety", 5).unwrap();
        assert!(!results.results.is_empty(), "Expected search results");
        assert_eq!(results.results[0].doc_id, "doc_1");
    }

    #[test]
    fn index_multiple_and_rank() {
        let (_dir, path) = tmp_brain_with_search();
        let mut search = BrainSearch::open(&path).unwrap();

        search.index_document("d1", "The quick brown fox jumps over the lazy dog", None, None).unwrap();
        search.index_document("d2", "Rust programming language memory safety borrow checker", None, None).unwrap();
        search.index_document("d3", "Python is great for data science and machine learning", None, None).unwrap();

        let results = search.search("memory safety in Rust", 3).unwrap();
        assert!(!results.results.is_empty());
        // d2 should rank highest for this query
        assert_eq!(results.results[0].doc_id, "d2");
    }

    #[test]
    fn index_memory_and_search() {
        use axel_memkoshi::memory::{Memory, MemoryCategory};

        let (_dir, path) = tmp_brain_with_search();
        let mut search = BrainSearch::open(&path).unwrap();

        let mut mem = Memory::new(
            MemoryCategory::Events,
            "project-orchid",
            "Chose FastAPI for Orchid backend",
            "Decided to use FastAPI with SQLAlchemy for the Orchid project backend. PostgreSQL for production database.",
        );
        mem.abstract_text = "Architecture decision for Orchid project".to_string();

        search.index_memory(&mem).unwrap();

        let results = search.search("Orchid backend architecture", 5).unwrap();
        assert!(!results.results.is_empty());
        assert_eq!(results.results[0].doc_id, mem.id);
    }

    #[test]
    fn empty_brain_search_returns_empty() {
        let (_dir, path) = tmp_brain_with_search();
        let mut search = BrainSearch::open(&path).unwrap();

        let results = search.search("anything", 5).unwrap();
        assert!(results.results.is_empty());
    }

    #[test]
    fn cache_stale_flag_set_after_index_document() {
        let (_dir, path) = tmp_brain_with_search();
        let mut search = BrainSearch::open(&path).unwrap();

        assert!(!search.cache_stale, "Cache should start clean");
        search.index_document("d1", "hello world", None, None).unwrap();
        assert!(search.cache_stale, "Cache should be stale after indexing");
    }

    #[test]
    fn flush_clears_stale_flag() {
        let (_dir, path) = tmp_brain_with_search();
        let mut search = BrainSearch::open(&path).unwrap();

        search.index_document("d1", "hello world", None, None).unwrap();
        assert!(search.cache_stale);

        search.flush().unwrap();
        assert!(!search.cache_stale, "Cache should be clean after flush");
    }

    #[test]
    fn warm_load_matches_cold_build() {
        let (dir, path) = tmp_brain_with_search();

        // Cold build: index two docs, flush cache.
        {
            let mut search = BrainSearch::open(&path).unwrap();
            search.index_document("a", "alpha document", None, None).unwrap();
            search.index_document("b", "beta document", None, None).unwrap();
            search.flush().unwrap();
        } // drop also saves, but flush already did it cleanly

        // Warm load: cache file should exist and count should match.
        let cache_path = path.with_extension("r8.hnsw");
        assert!(cache_path.exists(), "HNSW cache file should exist after flush");

        let search = BrainSearch::open(&path).unwrap();
        assert!(search.index_built, "Index should be pre-built from warm cache");
        assert_eq!(search.index.len(), 2, "Warm load should have 2 vectors");

        // Suppress unused variable warning — dir must stay alive for path to be valid.
        drop(dir);
    }
}
