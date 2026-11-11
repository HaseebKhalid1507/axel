//! Axel plugin configuration.

use std::path::{Path, PathBuf};

/// The data folder for `home`: `<home>/.local/share`, always.
///
/// RULING: wave2-c12a review finding 2. `XDG_DATA_HOME` is never read: the host starts the
/// extension with a cleared environment, so the extension never sees it, while the owner's
/// shell may set it for the command line. Reading it on one side only would make the
/// command line (`AxelConfig::default`) and the extension (`ext::config::default_brain`)
/// name different default brains.
pub fn data_dir_for(home: &Path) -> PathBuf {
    home.join(".local/share")
}

/// Plugin configuration, loaded from `~/.synaps-cli/axel.toml`.
#[derive(Debug, Clone)]
pub struct AxelConfig {
    /// Path to the default brain file: `<home>/.local/share/axel/brain.r8`, whatever
    /// `XDG_DATA_HOME` says (`data_dir_for`).
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
        // the default again. RULING: wave2-c12a review finding 2: the same folder as the
        // extension's default (`data_dir_for`); with no home, `.local/share` relative.
        let base = dirs::home_dir()
            .map(|home| data_dir_for(&home))
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
