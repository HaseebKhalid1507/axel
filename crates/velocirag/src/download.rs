//! Download models from HuggingFace, and only when asked.
//!
//! The process is **offline by default** ([`is_offline`]): `ensure_model`
//! returns the model folder when the files are already there and an error
//! otherwise, without building a URL. Only an explicit fetch command calls
//! [`set_offline`]`(false)` first.
//!
//! [`LocalEmbeddingModel`] is the guarded E3 path: it is off by
//! default, enablement is separate from the explicit download stub, and none
//! of its operations can construct a network request.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Result, VelociError};

const HF_BASE: &str = "https://huggingface.co";

/// The command that puts the model files in place. Every message that tells
/// a person how to get a model takes the name from here.
pub const FETCH_COMMAND: &str = "axel-v2 model fetch";

/// While true, nothing in this crate downloads a model.
static OFFLINE: AtomicBool = AtomicBool::new(true);

/// Forbid (`true`) or allow (`false`) model downloads in this process.
/// Offline is the default for every entry point; only [`FETCH_COMMAND`]
/// switches it off.
pub fn set_offline(offline: bool) {
    OFFLINE.store(offline, Ordering::SeqCst);
}

/// Whether model downloads are forbidden in this process (the default).
pub fn is_offline() -> bool {
    OFFLINE.load(Ordering::SeqCst)
}

/// Model registry — what to download and where.
pub struct ModelSpec {
    pub repo: &'static str,
    pub local_dir: &'static str,
    pub files: &'static [(&'static str, &'static str)], // (remote_path, local_path)
}

pub const EMBEDDER_MODEL: ModelSpec = ModelSpec {
    repo: "sentence-transformers/all-MiniLM-L6-v2",
    local_dir: "all-MiniLM-L6-v2",
    files: &[
        ("onnx/model.onnx", "onnx/model.onnx"),
        ("tokenizer.json", "tokenizer.json"),
    ],
};

pub const RERANKER_MODEL: ModelSpec = ModelSpec {
    repo: "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    local_dir: "cross-encoder-tinybert",
    files: &[
        ("onnx/model.onnx", "onnx/model.onnx"),
        ("tokenizer.json", "tokenizer.json"),
    ],
};

/// Explicit, local-only embedding model state for continuous-memory ranking.
/// Default is disabled. Enabling merely permits already-provisioned files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalEmbeddingModel {
    enabled: bool,
    model_dir: PathBuf,
}

impl LocalEmbeddingModel {
    pub fn disabled(model_dir: PathBuf) -> Self {
        Self { enabled: false, model_dir }
    }

    pub fn enabled(model_dir: PathBuf) -> Self {
        Self { enabled: true, model_dir }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn is_available(&self) -> bool {
        self.enabled
            && EMBEDDER_MODEL.files.iter()
                .all(|(_, local)| self.model_dir.join(local).is_file())
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }
}

impl Default for LocalEmbeddingModel {
    fn default() -> Self {
        Self::disabled(models_cache_dir().join(EMBEDDER_MODEL.local_dir))
    }
}

/// Separate explicit download step for the E3 local-only path. This is a
/// guarded stub by design: it performs no I/O and constructs no request.
pub fn download_local_embedding_model(_model_dir: &Path) -> Result<()> {
    Err(VelociError::Embedding(
        "automatic model download is disabled; provision the model explicitly".into(),
    ))
}

/// Resolve the cache directory for velocirag models.
pub fn models_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("velocirag")
        .join("models")
}

/// Return the local folder of a model whose files are all present.
///
/// When files are missing: offline (the default), an error that names the
/// folder and [`FETCH_COMMAND`], with nothing created and no URL built;
/// otherwise the files are downloaded from HuggingFace.
pub fn ensure_model(spec: &ModelSpec) -> Result<PathBuf> {
    ensure_model_in(&models_cache_dir(), spec)
}

/// What "the model files are in `dir`" means, for the loader, for the
/// offline check and for the embedder's lookup alike: `tokenizer.json` and
/// one of `onnx/model.onnx` (HuggingFace layout) and `model.onnx` (flat
/// layout, what Python velocirag leaves), in that order of preference.
/// Returns the graph it found. Only existence is checked.
pub fn model_files_in(dir: &Path) -> Option<PathBuf> {
    if !dir.join("tokenizer.json").exists() {
        return None;
    }
    [dir.join("onnx").join("model.onnx"), dir.join("model.onnx")]
        .into_iter()
        .find(|graph| graph.exists())
}

/// The one message for a folder in which [`model_files_in`] finds no model.
pub(crate) fn missing_model_message(dir: &Path) -> String {
    format!(
        "the model files are not in {}. Copy model.onnx (or onnx/model.onnx) and tokenizer.json \
         into it, or run \"{FETCH_COMMAND}\".",
        dir.display()
    )
}

/// [`ensure_model`] with the models folder given by the caller instead of
/// taken from the user's cache directory.
pub(crate) fn ensure_model_in(models_dir: &Path, spec: &ModelSpec) -> Result<PathBuf> {
    let model_dir = models_dir.join(spec.local_dir);

    // The files are there in either layout: nothing to do.
    if model_files_in(&model_dir).is_some() {
        return Ok(model_dir);
    }

    if is_offline() {
        return Err(VelociError::Embedding(missing_model_message(&model_dir)));
    }

    tracing::info!("Downloading model: {} ...", spec.repo);

    for (remote, local) in spec.files {
        let local_path = model_dir.join(local);

        if local_path.exists() {
            continue;
        }

        // Create parent dirs
        if let Some(parent) = local_path.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                VelociError::Embedding(format!("Failed to create dir {}: {}", parent.display(), e))
            })?;
        }

        let url = format!("{}/{}/resolve/main/{}", HF_BASE, spec.repo, remote);
        download_file(&url, &local_path)?;
    }

    tracing::info!("Model ready: {}", model_dir.display());
    Ok(model_dir)
}

/// Download a single file from a URL to a local path.
/// Uses a temp file + rename for atomic writes.
fn download_file(url: &str, dest: &Path) -> Result<()> {
    let file_name = dest.file_name().unwrap_or_default().to_string_lossy();
    eprintln!("  📥 Downloading {} ...", file_name);

    let response = ureq::get(url)
        .call()
        .map_err(|e| VelociError::Embedding(format!("Download failed for {}: {}", url, e)))?;

    let status = response.status();
    if status != 200 {
        return Err(VelociError::Embedding(format!(
            "HTTP {} downloading {}",
            status, url
        )));
    }

    // Get content length for progress
    let content_len = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok());

    // Stream to temp file
    let tmp_path = dest.with_extension("tmp");
    let mut file = fs::File::create(&tmp_path).map_err(|e| {
        VelociError::Embedding(format!("Failed to create {}: {}", tmp_path.display(), e))
    })?;

    let mut reader = response.into_body().into_reader();
    let mut buf = [0u8; 65536];
    let mut downloaded: u64 = 0;
    let mut last_pct: u64 = 0;

    loop {
        let n = reader.read(&mut buf).map_err(|e| {
            VelociError::Embedding(format!("Read error downloading {}: {}", file_name, e))
        })?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n]).map_err(|e| {
            VelociError::Embedding(format!("Write error for {}: {}", file_name, e))
        })?;
        downloaded += n as u64;

        // Simple progress indicator
        if let Some(total) = content_len {
            let pct = (downloaded * 100) / total;
            if pct >= last_pct + 10 {
                eprintln!("  📥 {} — {}%", file_name, pct);
                last_pct = pct;
            }
        }
    }

    // Atomic rename
    fs::rename(&tmp_path, dest).map_err(|e| {
        VelociError::Embedding(format!("Failed to rename {} → {}: {}", tmp_path.display(), dest.display(), e))
    })?;

    let size_mb = downloaded as f64 / 1_048_576.0;
    eprintln!("  ✅ {} ({:.1} MB)", file_name, size_mb);

    Ok(())
}
