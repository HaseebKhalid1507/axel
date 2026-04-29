//! Axel plugin configuration.

use std::path::PathBuf;

/// Plugin configuration, loaded from `~/.synaps-cli/axel.toml`.
#[derive(Debug, Clone)]
pub struct AxelConfig {
    /// Path to the default .r8 brain file.
    pub brain_path: PathBuf,

    /// Maximum tokens to inject per turn (Tier 0 + Tier 1).
    pub injection_budget: usize,

    /// Number of memories to retrieve for context injection.
    pub injection_top_k: usize,

    /// Enable automatic regex extraction on session end.
    pub auto_extract: bool,

    /// Enable the reranker (requires additional model download).
    pub reranker: bool,

    /// Model cache directory for ONNX models.
    pub model_cache: PathBuf,
}

impl Default for AxelConfig {
    fn default() -> Self {
        let base = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from(".config"))
            .join("axel");

        Self {
            brain_path: base.join("axel.r8"),
            injection_budget: 700,
            injection_top_k: 5,
            auto_extract: true,
            reranker: false,
            model_cache: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from(".cache"))
                .join("axel")
                .join("models"),
        }
    }
}
