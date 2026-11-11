//! Search bridge — connects a `.r8` Brain to VelociRAG's 4-layer search.
//!
//! The Brain owns the SQLite file. VelociRAG opens a second read connection
//! to the same file (SQLite WAL supports unlimited concurrent readers).
//! The HNSW vector index is built in memory from stored embeddings on first
//! search, then persisted to disk for fast warm starts on subsequent opens.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use velocirag::db::Database;
use velocirag::embedder::Embedder;
use velocirag::index::VectorIndex;
use velocirag::search::{SearchEngine, SearchOptions, SearchResponse};

use crate::error::{AxelError, Result};
use crate::r8::{BrainMeta, DEFAULT_EMBEDDER, EMBEDDING_DIM};

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
    /// The `documents` ids whose vectors are in `index`.
    ids: HashSet<i64>,
}

/// The ids of every row of `documents`: the rows that have a vector.
fn document_ids(conn: &rusqlite::Connection) -> rusqlite::Result<HashSet<i64>> {
    let mut stmt = conn.prepare("SELECT id FROM documents")?;
    let ids = stmt.query_map([], |r| r.get::<_, i64>(0))?.collect();
    ids
}

impl BrainSearch {
    /// Initialize search for a brain at the given path, with the default
    /// embedder (see [`BrainSearch::default_embedder`]).
    ///
    /// The embedding model is loaded on the first embed and is never
    /// downloaded here: without its files, embedding is an error.
    ///
    /// The file is checked first: the default embedder, which reads the
    /// user's embedding cache, is built only for a brain that is accepted.
    pub fn open(brain_path: &Path) -> Result<Self> {
        Self::open_with(brain_path, DEFAULT_EMBEDDER, EMBEDDING_DIM, Self::default_embedder)
    }

    /// The embedder `open` uses: all-MiniLM-L6-v2, loaded on first use, with
    /// the embedding cache persisted to ~/.cache/axel/embeddings so repeated
    /// embed() calls for the same content are free across sessions.
    pub(crate) fn default_embedder() -> Embedder {
        let embedding_cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from(".cache"))
            .join("axel")
            .join("embeddings");

        Embedder::new(Some(embedding_cache_dir), None, true)
    }

    /// Refuse an embedder whose model, or whose vector dimension, is not
    /// the one the brain was built with: vectors of two models must not
    /// share an index. The model name is one per backend, so the dimension
    /// is the only other thing the brain records that can tell two apart.
    pub(crate) fn check_embedder(meta: &BrainMeta, embedder: &Embedder) -> Result<()> {
        Self::check_model(
            &meta.embedder_model,
            i64::from(meta.embedding_dim),
            embedder.model_id(),
            embedder.dim(),
        )
    }

    /// [`BrainSearch::check_embedder`] for the embedder `open` would build,
    /// without building it.
    pub(crate) fn check_default_embedder(meta: &BrainMeta) -> Result<()> {
        Self::check_model(
            &meta.embedder_model,
            i64::from(meta.embedding_dim),
            DEFAULT_EMBEDDER,
            EMBEDDING_DIM,
        )
    }

    fn check_model(recorded_model: &str, recorded_dim: i64, model_id: &str, dim: usize) -> Result<()> {
        if recorded_model != model_id {
            Err(AxelError::ModelMismatch {
                expected: recorded_model.to_string(),
                actual: model_id.to_string(),
            })
        } else if i64::try_from(dim).ok() != Some(recorded_dim) {
            Err(AxelError::ModelMismatch {
                expected: format!("{recorded_model} ({recorded_dim} dimensions)"),
                actual: format!("{model_id} ({dim} dimensions)"),
            })
        } else {
            Ok(())
        }
    }

    /// Initialize search for a brain at the given path, embedding with
    /// `embedder`.
    ///
    /// Opens a second SQLite connection to the brain file. Refuses, before
    /// writing anything, an embedder whose `model_id` or `dim` differs from
    /// the brain's `brain_meta` (`embedder_model`, `embedding_dim`).
    /// If a matching on-disk HNSW cache exists, loads it instead of
    /// rebuilding from raw embeddings.
    pub fn open_with_embedder(brain_path: &Path, embedder: Embedder) -> Result<Self> {
        Self::open_with(brain_path, embedder.model_id(), embedder.dim(), move || embedder)
    }

    /// Open the file, refuse it unless it records `model_id` and `dim`, and
    /// only then call `make_embedder`, which must give an embedder of that
    /// model.
    fn open_with(
        brain_path: &Path,
        model_id: &'static str,
        dim: usize,
        make_embedder: impl FnOnce() -> Embedder,
    ) -> Result<Self> {
        // VelociRAG's Database::open expects a directory and creates velocirag.db inside it.
        // For .r8 files, we need from_connection with the actual file path.
        let conn = rusqlite::Connection::open_with_flags(
            brain_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
                | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(AxelError::Db)?;

        let (recorded_model, recorded_dim): (Option<String>, Option<i64>) = conn
            .query_row(
                "SELECT json_extract(value, '$.embedder_model'), json_extract(value, '$.embedding_dim') \
                 FROM brain_meta WHERE key = 'meta'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(|e| {
                AxelError::Search(format!(
                    "{} has no readable brain_meta: {e}",
                    brain_path.display()
                ))
            })?;
        let recorded_model = recorded_model.ok_or_else(|| {
            AxelError::Search(format!(
                "{} does not record its embedder model",
                brain_path.display()
            ))
        })?;
        let recorded_dim = recorded_dim.ok_or_else(|| {
            AxelError::Search(format!(
                "{} does not record its embedding dimension",
                brain_path.display()
            ))
        })?;
        Self::check_model(&recorded_model, recorded_dim, model_id, dim)?;
        let embedder = make_embedder();
        debug_assert_eq!(embedder.model_id(), model_id);
        debug_assert_eq!(embedder.dim(), dim);

        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(AxelError::Db)?;

        let db = Database::from_connection(conn, brain_path.to_path_buf())
            .map_err(|e| AxelError::Search(e.to_string()))?;

        // Derive the HNSW index cache path from the brain file path.
        let cache_path = brain_path.with_extension("r8.hnsw");

        // Try to load a warm index from disk if it exists and the vector
        // count matches what the database currently holds.
        let (index, index_built) = if cache_path.exists() {
            let db_count = db
                .document_count()
                .unwrap_or(0);

            // RULING: v2 c18b review round 5, item 1: the count, then every row's id. Another
            // writer may have deleted one row and added another; the file then lacks the new
            // row's vector, and is not loaded.
            let holds_every_row = |loaded: &VectorIndex| {
                document_ids(db.conn()).is_ok_and(|ids| ids.iter().all(|id| loaded.contains(*id as u64)))
            };
            match VectorIndex::load(&cache_path) {
                Ok(loaded) if loaded.len() == db_count && db_count > 0 && holds_every_row(&loaded) => {
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

        // A warm file was written when its vectors were those of `documents` (`save_cache`).
        let ids = if index_built { document_ids(db.conn()).map_err(AxelError::Db)? } else { HashSet::new() };
        Ok(Self {
            db,
            embedder,
            index,
            index_built,
            cache_path,
            cache_stale: false,
            ids,
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
            self.ids.insert(*key);
        }

        self.index_built = true;
        tracing::info!(count = embeddings.len(), "HNSW index built from stored embeddings");

        // Persist the freshly built index so the next open() is a warm load.
        if let Err(e) = self.save_cache() {
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
            self.save_cache()
                .map_err(|e| AxelError::Search(format!("Failed to flush HNSW cache: {e}")))?;
            self.cache_stale = false;
            tracing::debug!("HNSW index flushed to {}", self.cache_path.display());
        }
        Ok(())
    }

    /// Writes `<brain>.r8.hnsw`, the one way every save goes (`build_index`, `flush`, the
    /// flush on drop): only when the index holds the vectors of `documents`, no more and no
    /// fewer (every id it holds is a row there, and it holds as many vectors as there are
    /// rows); else the file is deleted, and the next open builds the index again.
    // RULING: v2 c18b review round 3, item 6. A `forget --now` that commits and removes the
    // file between this handle's read and its save would find the file written back with
    // a deleted vector.
    // RULING: v2 c18b review round 4, item 3: the test is what the file holds, not whether
    // another connection committed: `AxelBrain` writes through three connections, and a
    // `remember` must not cost the next search a rebuild.
    fn save_cache(&self) -> std::result::Result<(), String> {
        let docs = document_ids(self.db.conn()).map_err(|e| e.to_string())?;
        let same = self.index.len() == docs.len() && self.ids.len() == docs.len() && self.ids.is_subset(&docs);
        if same {
            return self.index.save(&self.cache_path).map_err(|e| e.to_string());
        }
        match std::fs::remove_file(&self.cache_path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e.to_string()),
            _ => Ok(()),
        }
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
        self.ids.insert(rowid);

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
            if let Err(e) = self.save_cache() {
                tracing::warn!("Failed to persist HNSW index on drop: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r8::Brain;

    use velocirag::embedder::HASH_MODEL_ID;

    /// A brain recorded as built with the hashing embedder: these tests
    /// embed, and no test loads the ONNX model.
    fn tmp_brain_with_search() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.r8");
        Brain::create_with_model(&path, Some("test"), HASH_MODEL_ID, velocirag::EMBEDDING_DIM)
            .unwrap();
        (dir, path)
    }

    fn open_hashing(path: &Path) -> Result<BrainSearch> {
        BrainSearch::open_with_embedder(path, Embedder::hashing())
    }

    fn refusal(result: Result<BrainSearch>) -> AxelError {
        match result {
            Ok(_) => panic!("an embedder of another model must be refused"),
            Err(error) => error,
        }
    }

    #[test]
    fn open_search_on_brain() {
        let (_dir, path) = tmp_brain_with_search();
        let search = open_hashing(&path);
        assert!(search.is_ok(), "Failed to open search: {:?}", search.err());
    }

    #[test]
    fn hash_embedder_on_a_minilm_brain_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("minilm.r8");
        Brain::create(&path, Some("test")).unwrap();
        let tables_before = table_count(&path);

        let error = refusal(open_hashing(&path));
        assert!(
            matches!(
                &error,
                AxelError::ModelMismatch { expected, actual }
                    if expected == "all-MiniLM-L6-v2" && actual == "hash-384"
            ),
            "{error}"
        );
        // Refused before anything was written: no search tables, no index file.
        assert_eq!(table_count(&path), tables_before);
        assert!(!path.with_extension("r8.hnsw").exists());

        // The brain's own model is still accepted (nothing is loaded by opening).
        assert!(BrainSearch::open_with_embedder(&path, Embedder::new(None, None, true)).is_ok());
    }

    #[test]
    fn minilm_embedder_on_a_hash_brain_is_refused() {
        let (_dir, path) = tmp_brain_with_search();

        let error = refusal(BrainSearch::open_with_embedder(
            &path,
            Embedder::new(None, None, true),
        ));
        assert!(
            matches!(
                &error,
                AxelError::ModelMismatch { expected, actual }
                    if expected == "hash-384" && actual == "all-MiniLM-L6-v2"
            ),
            "{error}"
        );
        assert!(open_hashing(&path).is_ok());
    }

    #[test]
    fn a_brain_of_another_dimension_is_refused_unchanged() {
        // The model's name matches; only the recorded dimension differs.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide.r8");
        drop(Brain::create_with_model(&path, Some("test"), HASH_MODEL_ID, 768).unwrap());
        let tables_before = table_count(&path);
        let bytes_before = std::fs::read(&path).unwrap();

        let error = refusal(open_hashing(&path));
        assert!(
            matches!(
                &error,
                AxelError::ModelMismatch { expected, actual }
                    if expected == "hash-384 (768 dimensions)" && actual == "hash-384 (384 dimensions)"
            ),
            "{error}"
        );
        assert!(!path.with_extension("r8.hnsw").exists());
        assert_eq!(std::fs::read(&path).unwrap(), bytes_before, "the file must be byte-identical");
        assert_eq!(table_count(&path), tables_before);

        // The same for the default model's name and the default opener.
        let path = dir.path().join("wide-minilm.r8");
        drop(Brain::create_with_model(&path, Some("test"), DEFAULT_EMBEDDER, 768).unwrap());
        let bytes_before = std::fs::read(&path).unwrap();
        let error = refusal(BrainSearch::open(&path));
        assert!(matches!(&error, AxelError::ModelMismatch { .. }), "{error}");
        assert!(error.to_string().contains("768 dimensions"), "{error}");
        assert_eq!(std::fs::read(&path).unwrap(), bytes_before, "the file must be byte-identical");
    }

    #[test]
    fn a_file_without_brain_meta_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plain.db");
        rusqlite::Connection::open(&path)
            .unwrap()
            .execute_batch("CREATE TABLE t(v)")
            .unwrap();

        let error = refusal(open_hashing(&path));
        assert!(error.to_string().contains("brain_meta"), "{error}");
        assert_eq!(table_count(&path), 1);
    }

    fn table_count(path: &Path) -> i64 {
        rusqlite::Connection::open(path)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM sqlite_master", [], |row| row.get(0))
            .unwrap()
    }

    #[test]
    fn index_and_search_document() {
        let (_dir, path) = tmp_brain_with_search();
        let mut search = open_hashing(&path).unwrap();

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
        let mut search = open_hashing(&path).unwrap();

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
        let mut search = open_hashing(&path).unwrap();

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
        let mut search = open_hashing(&path).unwrap();

        let results = search.search("anything", 5).unwrap();
        assert!(results.results.is_empty());
    }

    #[test]
    fn cache_stale_flag_set_after_index_document() {
        let (_dir, path) = tmp_brain_with_search();
        let mut search = open_hashing(&path).unwrap();

        assert!(!search.cache_stale, "Cache should start clean");
        search.index_document("d1", "hello world", None, None).unwrap();
        assert!(search.cache_stale, "Cache should be stale after indexing");
    }

    #[test]
    fn flush_clears_stale_flag() {
        let (_dir, path) = tmp_brain_with_search();
        let mut search = open_hashing(&path).unwrap();

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
            let mut search = open_hashing(&path).unwrap();
            search.index_document("a", "alpha document", None, None).unwrap();
            search.index_document("b", "beta document", None, None).unwrap();
            search.flush().unwrap();
        } // drop also saves, but flush already did it cleanly

        // Warm load: cache file should exist and count should match.
        let cache_path = path.with_extension("r8.hnsw");
        assert!(cache_path.exists(), "HNSW cache file should exist after flush");

        let search = open_hashing(&path).unwrap();
        assert!(search.index_built, "Index should be pre-built from warm cache");
        assert_eq!(search.index.len(), 2, "Warm load should have 2 vectors");

        // Suppress unused variable warning — dir must stay alive for path to be valid.
        drop(dir);
    }

    /// A brain with two documents, its cache file removed, and a handle that has read
    /// every vector (`build_index`) and saved.
    fn read_then_cache_removed() -> (tempfile::TempDir, PathBuf, BrainSearch) {
        let (dir, path) = tmp_brain_with_search();
        {
            let mut search = open_hashing(&path).unwrap();
            search.index_document("a", "alpha document", None, None).unwrap();
            search.index_document("b", "beta document", None, None).unwrap();
        }
        let cache = path.with_extension("r8.hnsw");
        std::fs::remove_file(&cache).unwrap();
        let mut reader = open_hashing(&path).unwrap();
        reader.build_index().unwrap();
        assert!(cache.exists(), "build_index did not save");
        // What `forget --now` does after its commit.
        std::fs::remove_file(&cache).unwrap();
        (dir, path, reader)
    }

    /// v2 c18b review round 3, item 6, round 4, item 3: a handle that holds the vector of a
    /// row another connection purged does not write the cache file back, on `flush` or on
    /// drop.
    #[test]
    fn a_save_after_another_connection_purged_a_row_deletes_the_cache_file() {
        let (_dir, path, mut reader) = read_then_cache_removed();
        let cache = path.with_extension("r8.hnsw");
        let other = rusqlite::Connection::open(&path).unwrap();
        other.execute("DELETE FROM documents WHERE doc_id = 'a'", []).unwrap();
        reader.index_document("c", "gamma document", None, None).unwrap();
        reader.flush().unwrap();
        assert!(!cache.exists(), "flush wrote the cache back after another writer's commit");
        reader.index_document("d", "delta document", None, None).unwrap();
        std::fs::write(&cache, b"old").unwrap();
        drop(reader);
        assert!(!cache.exists(), "the drop wrote the cache back, or left an old one");
    }

    /// The same count is not enough: another connection purges a row and writes another,
    /// and the handle that holds the purged row's vector does not write the file.
    #[test]
    fn a_save_with_the_same_count_but_a_purged_row_deletes_the_cache_file() {
        let (_dir, path, mut reader) = read_then_cache_removed();
        let cache = path.with_extension("r8.hnsw");
        let other = rusqlite::Connection::open(&path).unwrap();
        other
            .execute_batch(
                "INSERT INTO documents(doc_id, content, metadata, embedding, file_path, indexed_at)
                   SELECT 'z', content, metadata, embedding, file_path, indexed_at FROM documents WHERE doc_id = 'b';
                 DELETE FROM documents WHERE doc_id = 'a';",
            )
            .unwrap();
        reader.index_document("c", "gamma document", None, None).unwrap();
        std::fs::write(&cache, b"old").unwrap();
        drop(reader);
        assert!(!cache.exists(), "the drop wrote the cache back, or left an old one");
    }

    /// An index that lacks a row another connection wrote is not written either.
    #[test]
    fn a_save_that_lacks_a_row_another_connection_wrote_deletes_the_cache_file() {
        let (_dir, path, mut reader) = read_then_cache_removed();
        let cache = path.with_extension("r8.hnsw");
        let other = rusqlite::Connection::open(&path).unwrap();
        other
            .execute(
                "INSERT INTO documents(doc_id, content, metadata, embedding, file_path, indexed_at)
                   SELECT 'z', content, metadata, embedding, file_path, indexed_at FROM documents WHERE doc_id = 'b'",
                [],
            )
            .unwrap();
        reader.index_document("c", "gamma document", None, None).unwrap();
        reader.flush().unwrap();
        assert!(!cache.exists(), "flush wrote a cache that lacks a row");
    }

    /// Another connection's commit that leaves `documents` as the index holds it does not
    /// stop the save (round 4, item 3).
    #[test]
    fn a_save_after_another_connection_committed_elsewhere_writes_the_cache_file() {
        let (_dir, path, mut reader) = read_then_cache_removed();
        let other = rusqlite::Connection::open(&path).unwrap();
        other.execute("INSERT INTO context_data(key, value, layer) VALUES ('k', 'v', 'test')", []).unwrap();
        reader.index_document("c", "gamma document", None, None).unwrap();
        reader.flush().unwrap();
        assert!(path.with_extension("r8.hnsw").exists());
    }

    /// v2 c18b review round 4, item 3: a `remember` (written through `AxelBrain`'s other
    /// connections) after a search keeps the file, and the next search loads it warm.
    #[test]
    fn remember_after_a_search_keeps_the_cache_and_the_next_search_does_not_rebuild() {
        use axel_memkoshi::memory::{Memory, MemoryCategory};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.r8");
        let remember = |text: &str| {
            let mut brain = crate::brain::AxelBrain::open_or_create_with(&path, None, Embedder::hashing()).unwrap();
            brain.search_mut().search("anything at all", 3).unwrap();
            brain.remember_full(Memory::new(MemoryCategory::Events, "harbour", text, text)).unwrap();
        };
        remember("the harbour ferry runs on the winter schedule from November");
        remember("the harbour ferry stops at the island twice a day in winter");
        let cache = path.with_extension("r8.hnsw");
        assert!(cache.exists(), "the remember deleted the cache file");
        let search = open_hashing(&path).unwrap();
        assert!(search.index_built, "the next search would rebuild the index");
        assert_eq!(search.index.len(), 2);
    }

    /// v2 c18b review round 5, item 1: between two opens another writer deletes one row
    /// and adds one (the count is the same); the second open rebuilds, and the new row is
    /// found by its vector.
    #[test]
    fn a_warm_file_that_lacks_a_row_is_not_loaded_and_the_row_is_found() {
        let (_dir, path) = tmp_brain_with_search();
        {
            let mut search = open_hashing(&path).unwrap();
            search.index_document("a", "alpha document about lighthouses", None, None).unwrap();
            search.index_document("b", "beta document about orchards", None, None).unwrap();
        }
        let cache = path.with_extension("r8.hnsw");
        assert!(cache.exists());
        {
            // As the extension writes: its own connection, its own vector.
            let mut other = open_hashing(&path).unwrap();
            let embedding = other.embedder.embed_one("zeta document about glaciers").unwrap();
            other.db.insert_document("z", "zeta document about glaciers", &serde_json::json!({}), &embedding, None).unwrap();
            other.db.conn().execute("DELETE FROM documents WHERE doc_id = 'a'", []).unwrap();
            // Nothing of its own index changed: its drop leaves the file as it is.
        }
        assert!(cache.exists(), "the file is gone: the test proves nothing");
        let mut search = open_hashing(&path).unwrap();
        assert!(!search.index_built, "the stale file was loaded");
        let hits = search.search("glaciers", 5).unwrap();
        assert!(hits.results.iter().any(|r| r.doc_id == "z"));
        assert!(search.index.contains(search.db.get_document("z").unwrap().unwrap().id as u64));
    }

    /// The handle's own commits do not stop its save (`index`, `index-sync`).
    #[test]
    fn a_save_after_only_its_own_commits_writes_the_cache_file() {
        let (_dir, path, mut reader) = read_then_cache_removed();
        reader.index_document("c", "gamma document", None, None).unwrap();
        reader.flush().unwrap();
        assert!(path.with_extension("r8.hnsw").exists());
    }
}
