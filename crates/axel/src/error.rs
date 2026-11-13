//! Error types for the Axel crate.

use std::path::PathBuf;

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

    // The brain kinds of v2 (B3-3b). Nobody edits this file after commit 3.
    // SPEC: the document gives the text of NoBrain (893) and of V1Brain (879-881) only; the
    // other five messages are commit 3's wording.
    #[error("no brain at {}. \"{cmd} init\" makes an empty one; \"{cmd} migrate init --from-axel <axel.r8>\" makes one from the old Axel brain.", .0.display(), cmd = crate::cmd_name())]
    NoBrain(PathBuf),

    #[error("this is the old Axel brain (axel.r8). It is the rollback copy and this Axel does not change it. \"{cmd} migrate init --from-axel <it>\" makes a brain from it.", cmd = crate::cmd_name())]
    V1Brain(PathBuf),

    #[error("the brain is at version {file} and this binary knows version {binary}; it is not written")]
    BrainTooNew { file: i64, binary: i64 },

    #[error("the brain is not ready")]
    BrainNotReady,

    #[error("the brain was changed by another binary; restart Synaps")]
    BrainChanged,

    #[error("the brain is damaged: {0}")]
    BrainDamaged(String),

    #[error("the brain belongs to another Synaps base: {}", .0.display())]
    OtherBase(PathBuf),
}

pub type Result<T> = std::result::Result<T, AxelError>;
