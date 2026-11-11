//! `axel extension`: the process Synaps starts. The loop itself is `ext::server`.

use crate::error::Result;

/// Serves stdin and stdout until `shutdown` or end of input; it ends the process itself.
pub fn run(brain_path: &std::path::Path) -> Result<()> {
    crate::ext::server::run(brain_path)
}
