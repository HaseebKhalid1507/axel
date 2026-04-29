//! Search bridge — connects a `.r8` Brain to VelociRAG's 4-layer search.
//!
//! The Brain owns the SQLite file. VelociRAG opens a second read connection
//! to the same file (SQLite WAL supports unlimited concurrent readers).
//! The HNSW vector index is built in memory from stored embeddings on first
//! search, then cached for the session.

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
/// stored embeddings on first use).
pub struct BrainSearch {
    db: Database,
    embedder: Embedder,
    index: VectorIndex,
    index_built: bool,
}

impl BrainSearch {
    /// Initialize search for a brain at the given path.
    ///
    /// Opens a second SQLite connection (read-only for search).
    /// Downloads the embedding model on first use if not cached.
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

        let embedder = Embedder::new(None, None, true);

        let index = VectorIndex::new()
            .map_err(|e| AxelError::Search(format!("Failed to create index: {e}")))?;

        Ok(Self {
            db,
            embedder,
            index,
            index_built: false,
        })
    }

    /// Build the HNSW vector index from embeddings stored in SQLite.
    ///
    /// This is the "cold start" path — loads all embeddings from the database
    /// and builds an in-memory HNSW index. For 7K docs this takes ~70ms.
    /// Called automatically on first search if not already built.
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

        // Add to live index so it's searchable immediately
        self.index.add(rowid as u64, &embedding)
            .map_err(|e| AxelError::Search(format!("Index add failed: {e}")))?;

        self.index_built = true; // mark as built since we have at least one vector
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
}
