//! Validation and deduplication pipeline for incoming memories.

use crate::memory::Memory;

const INJECTION_PATTERNS: &[&str] = &[
    "remember that you must always",
    "for all future queries",
    "system instruction:",
    "override settings",
    "ignore previous instructions",
];

/// Pre-storage validation and deduplication.
///
/// Stateless — instances exist only to provide a tidy namespace; all
/// methods could equally be free functions.
#[derive(Debug, Default, Clone, Copy)]
pub struct MemoryPipeline;

impl MemoryPipeline {
    /// Construct a new pipeline.
    pub fn new() -> Self {
        Self
    }

    /// Validate a memory. Returns `Err(errors)` with a list of human-
    /// readable problems if any rule fails.
    pub fn validate(&self, memory: &Memory) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if memory.content.chars().count() < 50 {
            errors.push("content must be at least 50 characters".to_string());
        }
        if memory.title.chars().count() < 10 {
            errors.push("title must be at least 10 characters".to_string());
        }
        if memory.topic.chars().count() < 5 {
            errors.push("topic must be at least 5 characters".to_string());
        }
        if !(0.0..=1.0).contains(&memory.importance) {
            errors.push("importance must be in [0.0, 1.0]".to_string());
        }
        // Category is enforced by the type system; the check exists for
        // parity with the Python source where category was a string.

        let lowered = memory.content.to_lowercase();
        for pat in INJECTION_PATTERNS {
            if lowered.contains(pat) {
                errors.push(format!("content contains injection pattern: {pat:?}"));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Return `true` if `memory` looks like a duplicate of any record in
    /// `existing` based on a normalised Levenshtein similarity over titles
    /// (threshold 0.85).
    pub fn deduplicate(&self, memory: &Memory, existing: &[Memory]) -> bool {
        let a = memory.title.to_lowercase();
        for other in existing {
            let b = other.title.to_lowercase();
            let max_len = a.chars().count().max(b.chars().count());
            if max_len == 0 {
                continue;
            }
            let dist = levenshtein(&a, &b);
            let sim = 1.0 - (dist as f64 / max_len as f64);
            if sim >= 0.85 {
                return true;
            }
        }
        false
    }
}

/// Classic dynamic-programming Levenshtein edit distance.
fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1)
                .min(prev[j] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}
