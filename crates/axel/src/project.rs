//! Project key and directory from `cwd` (pure: no database, no state). Commit 5 writes this
//! file.

use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectRef {
    pub key: String,
    pub scope_dir: String,
}

/// sha256(be64(len) || key), hex.
pub fn project_dir(key: &str) -> String {
    let _ = key;
    String::new()
}

/// The plain-file rule, without the env override.
pub fn derive_key(cwd: &Path) -> Option<String> {
    let _ = cwd;
    None
}

/// `derive_key`, then `project_dir`.
pub fn resolve(cwd: Option<&str>) -> Option<ProjectRef> {
    let _ = cwd;
    None
}

/// sha256(be64(len) || session_id), hex.
pub fn logical(session_id: &str) -> String {
    let _ = session_id;
    String::new()
}

/// B16b. The stub gives the last of the three answers: `<home>/.synaps-cli`.
pub fn synaps_base(cfg: Option<&Path>, plugin_root: &Path, home: &Path) -> PathBuf {
    let _ = (cfg, plugin_root);
    home.join(".synaps-cli")
}
