pub mod db;
pub mod embedder;
pub mod index;
pub mod search;
pub mod graph;
pub mod chunker;
pub mod rrf;
pub mod reranker;
pub mod pipeline;
pub mod error;
pub mod frontmatter;
pub mod variants;
pub mod analyzers;
pub mod download;

/// Embedding dimension for the default model (all-MiniLM-L6-v2).
/// Single source of truth — used by db.rs, embedder.rs, and tests.
pub const EMBEDDING_DIM: usize = 384;
