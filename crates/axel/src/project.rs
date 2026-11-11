//! Project key and directory from `cwd` (pure: no database, no state).
//!
//! `derive_key` is core's plain-file rule (`ProjectScope::discover_with_override(start, None)`
//! and `for_canonical_root` in `agent-core/src/memory/store.rs`); `project_dir` and `logical`
//! are core's `digest_parts` of one part (`ArchiveStore::keys` in `core/context_archive.rs`).

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectRef {
    pub key: String,
    pub scope_dir: String,
}

const MARKERS: [&str; 2] = [".git", ".synaps-project"];
const MAX_ASCENT: usize = 64;

/// sha256(be64(len) || bytes), hex: core's `digest_parts` of one part.
fn digest_one(bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update((bytes.len() as u64).to_be_bytes());
    hash.update(bytes);
    hex::encode(hash.finalize())
}

/// sha256(be64(len) || key), hex.
pub fn project_dir(key: &str) -> String {
    digest_one(key.as_bytes())
}

/// `p` + the first 16 hex of SHA-256 of a canonical path.
fn key_of_canonical(canonical: &Path) -> String {
    let digest = Sha256::digest(canonical.as_os_str().as_encoded_bytes());
    format!("p{}", hex::encode(&digest[..8]))
}

/// The plain-file rule, without the env override.
pub fn derive_key(cwd: &Path) -> Option<String> {
    let canonical = cwd.canonicalize().ok()?;
    let mut dir = canonical.as_path();
    for _ in 0..MAX_ASCENT {
        if MARKERS
            .iter()
            .any(|marker| dir.join(marker).symlink_metadata().is_ok())
        {
            return Some(key_of_canonical(dir));
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => break,
        }
    }
    Some(key_of_canonical(&canonical))
}

/// `derive_key`, then `project_dir`. `None` for a `cwd` that is not absolute.
pub fn resolve(cwd: Option<&str>) -> Option<ProjectRef> {
    // RULING: wave2-rulings, commit 5, item 1: a relative `cwd` would be canonicalised
    // against the extension's own folder, which is not the session's.
    let cwd = Path::new(cwd?);
    if !cwd.is_absolute() {
        return None;
    }
    let key = derive_key(cwd)?;
    let scope_dir = project_dir(&key);
    Some(ProjectRef { key, scope_dir })
}

/// sha256(be64(len) || session_id), hex.
pub fn logical(session_id: &str) -> String {
    digest_one(session_id.as_bytes())
}

/// B16b: the config value when set; else `X` when `plugin_root` is `<X>/plugins/<name>` and
/// `X` holds a file `config` or a directory `context-archives`; else `<home>/.synaps-cli`.
pub fn synaps_base(cfg: Option<&Path>, plugin_root: &Path, home: &Path) -> PathBuf {
    if let Some(cfg) = cfg {
        return cfg.to_path_buf();
    }
    let x = plugin_root
        .parent()
        .filter(|plugins| plugins.file_name().is_some_and(|n| n == "plugins"))
        .and_then(Path::parent);
    if let Some(x) = x {
        if x.join("config").is_file() || x.join("context-archives").is_dir() {
            return x.to_path_buf();
        }
    }
    home.join(".synaps-cli")
}
