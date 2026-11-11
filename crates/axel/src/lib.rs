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

// Convenience re-export — the main entry point for library consumers
pub use brain::{AxelBrain, MemoryPatch};
