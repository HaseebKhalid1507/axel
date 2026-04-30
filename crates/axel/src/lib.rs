pub mod r8;
pub mod error;
pub mod config;
pub mod inject;
pub mod search;
pub mod brain;
pub mod extension;

// Convenience re-export — the main entry point for library consumers
pub use brain::AxelBrain;
