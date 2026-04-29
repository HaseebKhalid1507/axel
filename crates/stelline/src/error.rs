use thiserror::Error;

#[derive(Debug, Error)]
pub enum StellineError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, StellineError>;
