//! Axel plugin configuration.

use std::path::{Path, PathBuf};

/// The data folder for `home`.
///
/// When `home` is this process's own home it is `dirs::data_dir()`, so `XDG_DATA_HOME` and
/// the platform's rule are honoured and the command line (`AxelConfig::default`) and the
/// extension (`ext::config::default_brain`) name the same default brain. For any other
/// `home` (a test's) it is `<home>/.local/share`.
pub fn data_dir_for(home: &Path) -> PathBuf {
    match (dirs::data_dir(), dirs::home_dir()) {
        (Some(data), Some(real)) if real == home => data,
        _ => home.join(".local/share"),
    }
}

/// Plugin configuration, loaded from `~/.synaps-cli/axel.toml`.
#[derive(Debug, Clone)]
pub struct AxelConfig {
    /// Path to the default brain file: `dirs::data_dir()/axel/brain.r8`.
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
        // The v2 brain. `axel.r8` under the config folder is the old brain and is never
        // the default again.
        let base = dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from(".local/share"))
            .join("axel");

        Self {
            brain_path: base.join("brain.r8"),
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
