//! One file per subcommand that v2 adds or rewrites. Each declares `Args` (its flags) and
//! `run`; `main.rs` holds only the variant and the call. The commit that owns a file may add
//! to its `Args`.

use std::path::Path;

use crate::error::Result;

pub mod forget;
pub mod memories;
pub mod migrate;
pub mod model;
pub mod pause;
pub mod plugin;
pub mod recall;
pub mod restore;
pub mod resume;
pub mod search;
pub mod show;
pub mod status;
pub mod synaps_export;
pub mod why;

// The two helpers below are `main.rs`'s, moved as they are with the four subcommands that
// use them; `main.rs` calls `ensure_brain` for the subcommands it keeps.

pub fn ensure_brain(path: &Path) -> Result<()> {
    if !path.exists() {
        eprintln!("No brain at {}. Run `{} init` first.", path.display(), crate::cmd_name());
        std::process::exit(1);
    }
    Ok(())
}

/// Strip YAML frontmatter (--- ... ---) from markdown content.
pub fn strip_frontmatter(content: &str) -> String {
    let trimmed = content.trim_start();
    if let Some(after) = trimmed.strip_prefix("---") {
        if let Some(end) = after.find("---") {
            return after[end + 3..].trim_start().to_string();
        }
    }
    content.to_string()
}
