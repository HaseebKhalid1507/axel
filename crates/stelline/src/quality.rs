//! Quality gate for extracted memories.
//!
//! Before a [`Memory`] enters the permanent store it must pass a set of
//! structural and semantic thresholds enforced here. Rejection is explicit:
//! every discarded memory carries a human-readable reason so callers can
//! log, debug, or surface it in a review UI.
//!
//! ## Rejection criteria
//!
//! | Field | Threshold | Reason string |
//! |---|---|---|
//! | `content` length | < 50 chars | `"Content too short"` |
//! | `title` length | < 10 chars | `"Title too short"` |
//! | `topic` length | < 5 chars | `"Topic too short"` |
//! | `importance` | < 0.3 | `"Importance below threshold"` |
//! | `category` | not in valid set | `"Invalid category"` |

use axel_memkoshi::memory::{Memory, MemoryCategory};

// ── Thresholds (single source of truth) ──────────────────────────────────────

const MIN_CONTENT_LEN: usize = 50;
const MIN_TITLE_LEN: usize = 10;
const MIN_TOPIC_LEN: usize = 5;
const MIN_IMPORTANCE: f64 = 0.3;

// ── Public types ──────────────────────────────────────────────────────────────

/// The result of running a batch of memories through the quality gate.
///
/// All memories that satisfied every criterion are in [`accepted`]; every
/// memory that failed at least one criterion is in [`rejected`] together with
/// a human-readable explanation.
///
/// [`accepted`]: QualityResult::accepted
/// [`rejected`]: QualityResult::rejected
#[derive(Debug)]
pub struct QualityResult {
    /// Memories that passed all quality checks.
    pub accepted: Vec<Memory>,
    /// Memories that failed at least one check, paired with the failure reason.
    pub rejected: Vec<(Memory, String)>,
}

impl QualityResult {
    /// Total number of memories evaluated.
    pub fn total(&self) -> usize {
        self.accepted.len() + self.rejected.len()
    }

    /// Fraction of memories that passed (0.0 – 1.0).
    ///
    /// Returns `0.0` when the batch was empty.
    pub fn acceptance_rate(&self) -> f64 {
        if self.total() == 0 {
            return 0.0;
        }
        self.accepted.len() as f64 / self.total() as f64
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Apply the quality gate to a batch of memories.
///
/// Each memory is tested against every criterion in sequence; the **first**
/// failing criterion produces the rejection reason (short-circuit). This
/// keeps rejection reasons unambiguous — one reason per memory.
///
/// Accepted memories are returned in the same order they were passed in.
///
/// # Example
///
/// ```rust,ignore
/// let result = quality_gate(candidate_memories);
/// println!("Accepted {}/{}", result.accepted.len(), result.total());
/// for (mem, reason) in &result.rejected {
///     eprintln!("Rejected '{}': {}", mem.title, reason);
/// }
/// ```
pub fn quality_gate(memories: Vec<Memory>) -> QualityResult {
    let mut accepted = Vec::with_capacity(memories.len());
    let mut rejected = Vec::new();

    for memory in memories {
        match check_quality(&memory) {
            None => accepted.push(memory),
            Some(reason) => rejected.push((memory, reason)),
        }
    }

    QualityResult { accepted, rejected }
}

// ── Internals ─────────────────────────────────────────────────────────────────

/// Run all quality checks on a single memory.
///
/// Returns `Some(reason)` on the first failure, `None` if everything passes.
fn check_quality(memory: &Memory) -> Option<String> {
    if memory.content.len() < MIN_CONTENT_LEN {
        return Some(format!(
            "Content too short ({} < {} chars)",
            memory.content.len(),
            MIN_CONTENT_LEN
        ));
    }

    if memory.title.len() < MIN_TITLE_LEN {
        return Some(format!(
            "Title too short ({} < {} chars)",
            memory.title.len(),
            MIN_TITLE_LEN
        ));
    }

    if memory.topic.len() < MIN_TOPIC_LEN {
        return Some(format!(
            "Topic too short ({} < {} chars)",
            memory.topic.len(),
            MIN_TOPIC_LEN
        ));
    }

    if memory.importance < MIN_IMPORTANCE {
        return Some(format!(
            "Importance below threshold ({:.2} < {:.2})",
            memory.importance, MIN_IMPORTANCE
        ));
    }

    if !is_valid_category(&memory.category) {
        return Some(format!(
            "Invalid category: {:?}",
            memory.category
        ));
    }

    None
}

/// All [`MemoryCategory`] variants are currently valid; this function is a
/// forward-compatibility hook that lets callers extend the valid set without
/// touching downstream match arms.
fn is_valid_category(category: &MemoryCategory) -> bool {
    matches!(
        category,
        MemoryCategory::Preferences
            | MemoryCategory::Entities
            | MemoryCategory::Events
            | MemoryCategory::Cases
            | MemoryCategory::Patterns
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axel_memkoshi::memory::Memory;

    fn good_memory() -> Memory {
        let mut m = Memory::new(
            MemoryCategory::Events,
            "project-axel",
            "Shipped the Stelline crate",
            "Today we decided to ship the Stelline session intelligence crate as part of Axel.",
        );
        m.importance = 0.8;
        m
    }

    #[test]
    fn accepts_valid_memory() {
        let result = quality_gate(vec![good_memory()]);
        assert_eq!(result.accepted.len(), 1);
        assert_eq!(result.rejected.len(), 0);
    }

    #[test]
    fn rejects_short_content() {
        let mut m = good_memory();
        m.content = "Too short.".into();
        let result = quality_gate(vec![m]);
        assert_eq!(result.rejected.len(), 1);
        assert!(result.rejected[0].1.contains("Content too short"));
    }

    #[test]
    fn rejects_short_title() {
        let mut m = good_memory();
        m.title = "Short".into(); // 5 chars < 10
        let result = quality_gate(vec![m]);
        assert_eq!(result.rejected.len(), 1);
        assert!(result.rejected[0].1.contains("Title too short"));
    }

    #[test]
    fn rejects_short_topic() {
        let mut m = good_memory();
        m.topic = "ax".into(); // 2 chars < 5
        let result = quality_gate(vec![m]);
        assert_eq!(result.rejected.len(), 1);
        assert!(result.rejected[0].1.contains("Topic too short"));
    }

    #[test]
    fn rejects_low_importance() {
        let mut m = good_memory();
        m.importance = 0.1;
        let result = quality_gate(vec![m]);
        assert_eq!(result.rejected.len(), 1);
        assert!(result.rejected[0].1.contains("Importance below threshold"));
    }

    #[test]
    fn acceptance_rate_empty_batch() {
        let result = quality_gate(vec![]);
        assert_eq!(result.acceptance_rate(), 0.0);
        assert_eq!(result.total(), 0);
    }

    #[test]
    fn acceptance_rate_all_pass() {
        let result = quality_gate(vec![good_memory(), good_memory()]);
        assert!((result.acceptance_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn acceptance_rate_half_pass() {
        let mut bad = good_memory();
        bad.content = "short".into();
        let result = quality_gate(vec![good_memory(), bad]);
        assert!((result.acceptance_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn content_exact_boundary_passes() {
        let mut m = good_memory();
        // Exactly MIN_CONTENT_LEN characters — must pass.
        m.content = "a".repeat(MIN_CONTENT_LEN);
        let result = quality_gate(vec![m]);
        assert_eq!(result.accepted.len(), 1);
    }

    #[test]
    fn content_one_below_boundary_rejects() {
        let mut m = good_memory();
        m.content = "a".repeat(MIN_CONTENT_LEN - 1);
        let result = quality_gate(vec![m]);
        assert_eq!(result.rejected.len(), 1);
    }
}
