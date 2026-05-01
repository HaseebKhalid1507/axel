//! Error types for the Axel crate.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AxelError {
    #[error("Database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Search error: {0}")]
    Search(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Memkoshi error: {0}")]
    Memkoshi(#[from] axel_memkoshi::error::MemkoshiError),

    #[error("Velocirag error: {0}")]
    Velocirag(#[from] velocirag::error::VelociError),

    #[error("Schema version mismatch: file has v{file}, runtime expects v{expected}")]
    SchemaMismatch { file: i32, expected: i32 },

    #[error("Model mismatch: .r8 built with {expected}, system has {actual}")]
    ModelMismatch { expected: String, actual: String },

    #[error("Brain not found: {0}")]
    NotFound(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, AxelError>;
