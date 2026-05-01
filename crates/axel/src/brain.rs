//! Unified brain handle for library consumers (e.g., SynapsCLI).
//!
//! `AxelBrain` wraps the `.r8` file, search engine, and memory storage
//! into a single struct with a clean API. Open once, use for the whole session.
//!
//! ```rust,ignore
//! use axel::brain::AxelBrain;
//!
//! let mut brain = AxelBrain::open_or_create("~/.config/axel/axel.r8", Some("jawz"))?;
//!
//! // Boot: get context for system prompt injection
//! let context = brain.boot_context()?;
//!
//! // Search during session
//! let results = brain.search("CPT eligibility", 5)?;
//!
//! // Store a memory
//! brain.remember("JR runs Praxis AI", "Entities", 0.7)?;
//!
//! // Handoff between sessions
//! brain.set_handoff("CS646 final May 12. Study modules 7-11.")?;
//! ```

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use axel_memkoshi::memory::{Memory, MemoryCategory};
use axel_memkoshi::pipeline::MemoryPipeline;
use axel_memkoshi::storage::MemoryStorage;
use velocirag::search::SearchResponse;

use crate::error::{AxelError, Result};
use crate::inject::{self, InjectionContext, InjectionEntry};
use crate::r8::Brain;
use crate::search::BrainSearch;

/// Unified brain handle — one struct for all Axel operations.
///
/// Owns the `.r8` file (Brain), the search engine (BrainSearch),
/// and memory storage (MemoryStorage). Thread-safe reads, exclusive writes.
pub struct AxelBrain {
    brain: Brain,
    search: BrainSearch,
    storage: MemoryStorage,
    pipeline: MemoryPipeline,
    path: PathBuf,
    /// Memory IDs already injected this session (dedup tracking).
    seen_memory_ids: HashSet<String>,
}

impl AxelBrain {
    /// Open an existing brain or create a new one.
    pub fn open_or_create(path: impl AsRef<Path>, agent_name: Option<&str>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let brain = Brain::open_or_create(&path, agent_name)?;
        let search = BrainSearch::open(&path)?;
        let storage = MemoryStorage::open_existing(&path)?;
        let pipeline = MemoryPipeline::new();

        Ok(Self {
            brain,
            search,
            storage,
            pipeline,
            path,
            seen_memory_ids: HashSet::new(),
        })
    }

    /// Open an existing brain. Fails if it doesn't exist.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let brain = Brain::open(&path)?;
        let search = BrainSearch::open(&path)?;
        let storage = MemoryStorage::open_existing(&path)?;
        let pipeline = MemoryPipeline::new();

        Ok(Self {
            brain,
            search,
            storage,
            pipeline,
            path,
            seen_memory_ids: HashSet::new(),
        })
    }

    // ── Boot ────────────────────────────────────────────────────────────

    /// Get boot context for system prompt injection.
    ///
    /// Returns formatted text containing:
    /// - Tier 0: Last session handoff (if any)
    /// - Tier 1: Recent/important memories
    ///
    /// Budget: ~700 tokens total.
    pub fn boot_context(&mut self) -> Result<InjectionContext> {
        let handoff = self.get_handoff()?;
        let memories = self.storage.list_memories(20)?;

        let entries: Vec<InjectionEntry> = memories.iter().map(|m| {
            InjectionEntry {
                memory_id: m.id.clone(),
                title: m.title.clone(),
                abstract_text: m.content.chars().take(200).collect(),
                category: format!("{:?}", m.category),
                importance: m.importance,
                relevance_score: m.importance, // boot context uses importance as proxy
            }
        }).collect();

        let ctx = inject::build_injection(
            handoff.as_deref(),
            &entries,
            &self.seen_memory_ids,
        );

        // Track injected memory IDs
        for id in &ctx.included_ids {
            self.seen_memory_ids.insert(id.clone());
        }

        Ok(ctx)
    }

    /// Get context relevant to a specific query (for mid-session injection).
    ///
    /// Searches the brain and formats top results for system prompt injection.
    pub fn contextual_recall(&mut self, query: &str, limit: usize) -> Result<InjectionContext> {
        let search_results = self.search.search(query, limit)?;

        let entries: Vec<InjectionEntry> = search_results.results.iter().map(|r| {
            InjectionEntry {
                memory_id: r.doc_id.clone(),
                title: r.doc_id.clone(),
                abstract_text: r.content.chars().take(200).collect(),
                category: r.source.clone(),
                importance: 0.5,
                relevance_score: r.score as f64,
            }
        }).collect();

        let ctx = inject::build_injection(
            None,
            &entries,
            &self.seen_memory_ids,
        );

        for id in &ctx.included_ids {
            self.seen_memory_ids.insert(id.clone());
        }

        Ok(ctx)
    }

    // ── Search ──────────────────────────────────────────────────────────

    /// Search the brain (documents + memories + graph).
    pub fn search(&mut self, query: &str, limit: usize) -> Result<SearchResponse> {
        self.search.search(query, limit)
    }

    /// Access the underlying VelociRAG database (for access logging,
    /// consolidation, etc.).
    pub fn search_db(&self) -> &velocirag::db::Database {
        self.search.db()
    }

    // ── Memory ──────────────────────────────────────────────────────────

    /// Store a memory. Returns the memory ID.
    ///
    /// The memory is validated, signed (if signing key exists), and indexed
    /// for search immediately.
    pub fn remember(
        &mut self,
        content: &str,
        category: &str,
        importance: f64,
    ) -> Result<String> {
        self.remember_with_ttl(content, category, importance, None)
    }

    /// Store a memory with optional TTL. Returns the memory ID.
    ///
    /// The memory is validated, signed (if signing key exists), and indexed
    /// for search immediately. If `ttl_hours` is provided, the memory will
    /// expire after that many hours.
    pub fn remember_with_ttl(
        &mut self,
        content: &str,
        category: &str,
        importance: f64,
        ttl_hours: Option<u64>,
    ) -> Result<String> {
        let cat = match category.to_lowercase().as_str() {
            "events" | "event" => MemoryCategory::Events,
            "preferences" | "preference" | "pref" => MemoryCategory::Preferences,
            "entities" | "entity" => MemoryCategory::Entities,
            "cases" | "case" => MemoryCategory::Cases,
            _ => MemoryCategory::Events,
        };

        // Build title from first line or first N chars
        let title = content.lines().next().unwrap_or(content)
            .chars().take(80).collect::<String>();

        let mut memory = Memory::new(
            cat,
            category.to_string(),
            title,
            content.to_string(),
        );
        memory.importance = importance.clamp(0.0, 1.0);

        // Set TTL if provided
        if let Some(hours) = ttl_hours {
            memory.set_ttl(hours);
        }

        // Sign if brain has a signing key
        if let Some(ref signer) = self.brain.signer() {
            memory.signature = Some(signer.sign(&memory));
        }

        // Validate
        if let Err(errors) = self.pipeline.validate(&memory) {
            return Err(AxelError::Other(format!("Validation failed: {}", errors.join(", "))));
        }

        // Store
        let staged = self.storage.stage_memory(&memory)?;
        self.storage.approve(&staged.memory.id)?;

        // Index for search
        let _ = self.search.index_memory(&memory);

        let id = memory.id.clone();
        Ok(id)
    }

    /// List recent memories.
    pub fn memories(&self, limit: usize) -> Result<Vec<Memory>> {
        Ok(self.storage.list_memories(limit)?)
    }

    /// Delete a memory by ID.
    pub fn forget(&mut self, id: &str) -> Result<bool> {
        Ok(self.storage.delete_memory(id)?)
    }

    /// Remove all expired memories and return the count of deleted memories.
    pub fn prune_expired(&mut self) -> Result<u64> {
        Ok(self.storage.prune_expired()?)
    }

    /// Update a memory's content and/or importance.
    /// Re-signs the memory if a signing key exists and re-indexes for search.
    /// Returns true if the memory was found and updated.
    pub fn update_memory(
        &mut self,
        id: &str,
        new_content: Option<&str>,
        new_importance: Option<f64>,
    ) -> Result<bool> {
        // Update in storage
        if !self.storage.update_memory(id, new_content, new_importance)? {
            return Ok(false); // Memory not found
        }

        // Get the updated memory for re-signing and re-indexing
        if let Some(mut memory) = self.storage.get_memory(id)? {
            // Re-sign if brain has a signing key
            if let Some(ref signer) = self.brain.signer() {
                memory.signature = Some(signer.sign(&memory));
                // Update the signature in the database
                self.storage.store_memory(&memory)?;
            }

            // Re-index for search
            let _ = self.search.index_memory(&memory);
        }

        Ok(true)
    }

    /// Get a memory by ID with verification status.
    pub fn get_memory_with_verification(&self, id: &str) -> Result<Option<(Memory, bool)>> {
        match self.storage.get_memory(id)? {
            Some(memory) => {
                let verified = match &self.brain.signer() {
                    Some(signer) => signer.verify(&memory),
                    None => false, // No signer available
                };
                Ok(Some((memory, verified)))
            },
            None => Ok(None),
        }
    }

    // ── Handoff ─────────────────────────────────────────────────────────

    /// Set the session handoff note.
    pub fn set_handoff(&self, content: &str) -> Result<()> {
        self.brain.conn().execute(
            "INSERT OR REPLACE INTO context_data (layer, key, value, updated)
             VALUES ('session', 'handoff', ?1, datetime('now'))",
            [content],
        )?;
        Ok(())
    }

    /// Get the current handoff note.
    pub fn get_handoff(&self) -> Result<Option<String>> {
        let result = self.brain.conn().query_row(
            "SELECT value FROM context_data WHERE layer = 'session' AND key = 'handoff'",
            [],
            |row| row.get::<_, String>(0),
        );
        match result {
            Ok(v) => Ok(Some(v)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Clear the handoff note.
    pub fn clear_handoff(&self) -> Result<()> {
        self.brain.conn().execute(
            "DELETE FROM context_data WHERE layer = 'session' AND key = 'handoff'",
            [],
        )?;
        Ok(())
    }

    // ── Index ───────────────────────────────────────────────────────────

    /// Index a file or directory into the brain.
    pub fn index_path(&mut self, path: impl AsRef<Path>) -> Result<usize> {
        let path = path.as_ref();
        if path.is_dir() {
            let mut count = 0;
            for entry in walkdir::WalkDir::new(path)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_file())
            {
                let ext = entry.path().extension().and_then(|e| e.to_str()).unwrap_or("");
                if matches!(ext, "md" | "txt" | "rs" | "py" | "js" | "ts" | "toml" | "yaml" | "yml" | "json") {
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        let title = entry.path().file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("untitled");
                        let source = entry.path().to_string_lossy().to_string();
                        self.search.index_document(title, &content, None, Some(&source))?;
                        count += 1;
                    }
                }
            }
            Ok(count)
        } else {
            let content = std::fs::read_to_string(path)
                .map_err(|e| AxelError::Other(format!("Failed to read {}: {}", path.display(), e)))?;
            let title = path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("untitled");
            let source = path.to_string_lossy().to_string();
            self.search.index_document(title, &content, None, Some(&source))?;
            Ok(1)
        }
    }

    // ── Stats ───────────────────────────────────────────────────────────

    /// Get brain metadata.
    pub fn meta(&self) -> &crate::r8::BrainMeta {
        self.brain.meta()
    }

    /// Get the brain file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flush the search index to disk.
    pub fn flush(&mut self) -> Result<()> {
        self.search.flush()
    }
}

impl Drop for AxelBrain {
    fn drop(&mut self) {
        // Best-effort flush on drop
        let _ = self.search.flush();
    }
}
