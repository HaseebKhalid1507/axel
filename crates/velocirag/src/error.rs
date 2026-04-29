//! VelociRAG error types.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum VelociError {
    #[error("Database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("ONNX Runtime error: {0}")]
    Ort(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid argument: {0}")]
    InvalidArg(String),

    #[error("{0}")]
    Other(String),
}

impl<T: std::fmt::Debug> From<ort::Error<T>> for VelociError {
    fn from(e: ort::Error<T>) -> Self {
        VelociError::Ort(format!("{:?}", e))
    }
}

pub type Result<T> = std::result::Result<T, VelociError>;
