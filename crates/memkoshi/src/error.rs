//! Error types for the Memkoshi memory system.

use thiserror::Error;

/// Errors produced by the Memkoshi memory system.
#[derive(Debug, Error)]
pub enum MemkoshiError {
    /// Underlying SQLite failure.
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// JSON (de)serialisation failure.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// A row referenced an unknown enum tag.
    #[error("invalid enum value for {field}: {value}")]
    InvalidEnum {
        /// Column name.
        field: &'static str,
        /// Offending value.
        value: String,
    },

    /// A lookup by id returned nothing.
    #[error("memory not found: {0}")]
    NotFound(String),

    /// Schema version on disk is newer than this binary understands.
    #[error("schema version {found} is newer than supported {supported}")]
    SchemaTooNew {
        /// Version read from disk.
        found: i64,
        /// Highest version this binary supports.
        supported: i64,
    },

    /// Validation failure.
    #[error("validation error: {0}")]
    Validation(String),

    /// Duplicate memory detected.
    #[error("duplicate memory: {0}")]
    Duplicate(String),

    /// Security error.
    #[error("security error: {0}")]
    Security(String),

    /// Catch-all.
    #[error("{0}")]
    Other(String),
}

/// Result alias.
pub type Result<T> = std::result::Result<T, MemkoshiError>;
