//! Auto-download models from HuggingFace on first use.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{Result, VelociError};

const HF_BASE: &str = "https://huggingface.co";

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

/// Resolve the cache directory for velocirag models.
pub fn models_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from(".cache"))
        .join("velocirag")
        .join("models")
}

/// Ensure a model is downloaded. Returns the local model directory.
/// Downloads from HuggingFace if files are missing.
pub fn ensure_model(spec: &ModelSpec) -> Result<PathBuf> {
    let model_dir = models_cache_dir().join(spec.local_dir);

    // Check if all files already exist
    let all_present = spec.files.iter().all(|(_, local)| model_dir.join(local).exists());
    if all_present {
        return Ok(model_dir);
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
