#[cfg(not(unix))] compile_error!("axel v2 builds on Linux and macOS only");
pub mod r8;
pub mod error;
pub mod config;
pub mod inject;
pub mod search;
pub mod brain;
pub mod extension;
pub mod consolidate;
pub mod project_memory;
pub mod chat_history;
pub mod retrieval;

// v2 (spec, "Shape of the process"): one owning commit per file.
pub mod ext;
pub mod brain_v2;
pub mod store;
pub mod project;
pub mod archive;
mod fs_open;
pub mod notes;
pub mod indexer;
pub mod indexer_start;
pub mod index_archives;
pub mod index_notes;
pub mod docsync;
pub mod embed;
pub mod vindex;
pub mod recall;
pub mod capture;
pub mod forum;
pub mod forum_store;
pub mod screen;
pub mod import;
pub mod cli;
pub mod synaps_export;

/// The name the binary was run as (the basename of `argv[0]`), set once by `main()`.
static CMD_NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Records the command name from `argv[0]`; an empty or non-UTF-8 name leaves `axel`.
pub fn set_cmd_name(argv0: Option<&std::ffi::OsStr>) {
    let name = argv0
        .and_then(|a| std::path::Path::new(a).file_name())
        .and_then(|n| n.to_str())
        .filter(|n| !n.is_empty());
    if let Some(name) = name {
        let _ = CMD_NAME.set(name.to_string());
        velocirag::download::set_program_name(name);
    }
}

/// The command name for hints such as `axel restore <id>`: `axel` unless `main()` set it.
pub fn cmd_name() -> &'static str {
    CMD_NAME.get().map_or("axel", String::as_str)
}

// Convenience re-export — the main entry point for library consumers
pub use brain::{AxelBrain, MemoryPatch};
