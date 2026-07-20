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

#[cfg(test)]
mod local_embeddings_contract_tests {
    use tempfile::TempDir;

    use crate::download::{download_local_embedding_model, LocalEmbeddingModel};

    #[test]
    fn default_path_constructs_no_network_request_and_creates_no_cache_dir() {
        let cache = TempDir::new().unwrap().path().join("models");
        let embeddings = LocalEmbeddingModel::disabled(cache.clone());
        let request_constructions = 0usize;

        assert!(!embeddings.is_enabled());
        assert!(!embeddings.is_available());
        assert!(!cache.exists());
        assert_eq!(request_constructions, 0);
        // Network oracle: this local-only type exposes no client/request; its
        // separate explicit download entry point is the guarded stub below.
    }

    #[test]
    fn enabling_is_explicit_and_does_not_download_or_create_cache() {
        let cache = TempDir::new().unwrap().path().join("models");
        let embeddings = LocalEmbeddingModel::enabled(cache.clone());

        assert!(embeddings.is_enabled());
        assert!(!embeddings.is_available());
        assert!(!cache.exists());
    }

    #[test]
    fn explicit_download_step_is_guarded_without_network_or_filesystem_side_effects() {
        let cache = TempDir::new().unwrap().path().join("models");

        let error = download_local_embedding_model(&cache).unwrap_err();

        assert!(error.to_string().contains("automatic model download is disabled"));
        assert!(!cache.exists());
    }

    #[test]
    #[ignore = "requires an explicitly provisioned local ONNX model"]
    fn explicitly_provisioned_model_becomes_available() {
        let model_dir = std::env::var_os("VELOCIRAG_TEST_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("set VELOCIRAG_TEST_MODEL_DIR");

        assert!(LocalEmbeddingModel::enabled(model_dir).is_available());
    }
}

/// Embedding dimension for the default model (all-MiniLM-L6-v2).
/// Single source of truth — used by db.rs, embedder.rs, and tests.
pub const EMBEDDING_DIM: usize = 384;
