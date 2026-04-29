//! Context injection — search the brain and format results for LLM consumption.
//!
//! Before each user message, Axel searches the `.r8` for relevant context
//! and formats it for injection into the system prompt.
//!
//! # Token Budget
//!
//! - Tier 0 (≤200 tokens): Handoff from last session (always injected)
//! - Tier 1 (≤500 tokens): Top-K memories by search relevance
//! - Total automatic budget: ~700 tokens per turn
//! - Tier 2 (uncapped): Agent calls `axel_search` explicitly

use std::collections::HashSet;

/// Rough token estimate: ~4 chars per token for English text.
const CHARS_PER_TOKEN: usize = 4;

/// Maximum chars for Tier 0 (handoff).
const TIER0_BUDGET: usize = 200 * CHARS_PER_TOKEN;

/// Maximum chars for Tier 1 (relevant memories).
const TIER1_BUDGET: usize = 500 * CHARS_PER_TOKEN;

/// A memory formatted for injection.
#[derive(Debug, Clone)]
pub struct InjectionEntry {
    pub memory_id: String,
    pub title: String,
    pub abstract_text: String,
    pub category: String,
    pub importance: f64,
    pub relevance_score: f64,
}

/// Result of context injection preparation.
#[derive(Debug)]
pub struct InjectionContext {
    /// Formatted string ready to prepend to system prompt.
    pub formatted: String,
    /// IDs of memories included (for dedup tracking).
    pub included_ids: Vec<String>,
    /// Estimated token count.
    pub estimated_tokens: usize,
}

/// Format a handoff note for Tier 0 injection.
pub fn format_handoff(handoff: &str) -> String {
    let truncated = truncate_to_budget(handoff, TIER0_BUDGET);
    if truncated.is_empty() {
        return String::new();
    }
    format!("[Last session handoff]\n{}\n", truncated)
}

/// Format memory entries for Tier 1 injection.
///
/// Excludes any memory IDs already present in the conversation
/// (tracked via `seen_ids`).
pub fn format_memories(
    entries: &[InjectionEntry],
    seen_ids: &HashSet<String>,
) -> InjectionContext {
    let mut formatted = String::new();
    let mut included = Vec::new();
    let mut chars_used = 0;

    let header = "[Relevant context from memory]\n";
    formatted.push_str(header);
    chars_used += header.len();

    for entry in entries {
        if seen_ids.contains(&entry.memory_id) {
            continue;
        }

        let line = format!(
            "- [{}] {}: {}\n",
            entry.category, entry.title, entry.abstract_text
        );

        if chars_used + line.len() > TIER1_BUDGET {
            break;
        }

        formatted.push_str(&line);
        chars_used += line.len();
        included.push(entry.memory_id.clone());
    }

    if included.is_empty() {
        return InjectionContext {
            formatted: String::new(),
            included_ids: Vec::new(),
            estimated_tokens: 0,
        };
    }

    InjectionContext {
        estimated_tokens: chars_used / CHARS_PER_TOKEN,
        formatted,
        included_ids: included,
    }
}

/// Build the full injection string (Tier 0 + Tier 1).
pub fn build_injection(
    handoff: Option<&str>,
    memories: &[InjectionEntry],
    seen_ids: &HashSet<String>,
) -> InjectionContext {
    let mut full = String::new();
    let mut all_ids = Vec::new();

    if let Some(h) = handoff {
        let t0 = format_handoff(h);
        full.push_str(&t0);
    }

    let t1 = format_memories(memories, seen_ids);
    full.push_str(&t1.formatted);
    all_ids.extend(t1.included_ids);

    let tokens = full.len() / CHARS_PER_TOKEN;
    InjectionContext {
        formatted: full,
        included_ids: all_ids,
        estimated_tokens: tokens,
    }
}

/// Truncate text to fit within a character budget, breaking at word boundaries.
fn truncate_to_budget(text: &str, max_chars: usize) -> &str {
    if text.len() <= max_chars {
        return text;
    }
    // Find last space before budget
    match text[..max_chars].rfind(' ') {
        Some(pos) => &text[..pos],
        None => &text[..max_chars],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(id: &str, title: &str, abs: &str, cat: &str) -> InjectionEntry {
        InjectionEntry {
            memory_id: id.to_string(),
            title: title.to_string(),
            abstract_text: abs.to_string(),
            category: cat.to_string(),
            importance: 0.5,
            relevance_score: 0.9,
        }
    }

    #[test]
    fn handoff_formats_correctly() {
        let result = format_handoff("Continue working on Orchid auth middleware");
        assert!(result.contains("[Last session handoff]"));
        assert!(result.contains("Orchid"));
    }

    #[test]
    fn empty_handoff_returns_empty() {
        assert!(format_handoff("").is_empty());
    }

    #[test]
    fn memories_skip_seen_ids() {
        let entries = vec![
            make_entry("mem_1", "Title 1", "Abstract 1", "events"),
            make_entry("mem_2", "Title 2", "Abstract 2", "events"),
        ];
        let mut seen = HashSet::new();
        seen.insert("mem_1".to_string());

        let result = format_memories(&entries, &seen);
        assert_eq!(result.included_ids, vec!["mem_2"]);
        assert!(!result.formatted.contains("Title 1"));
        assert!(result.formatted.contains("Title 2"));
    }

    #[test]
    fn memories_respect_budget() {
        // Create entries that would exceed budget
        let entries: Vec<_> = (0..100)
            .map(|i| make_entry(
                &format!("mem_{i}"),
                &format!("Very Long Title Number {i} That Takes Space"),
                &format!("This is a fairly long abstract for memory number {i} with lots of words"),
                "events",
            ))
            .collect();

        let result = format_memories(&entries, &HashSet::new());
        // Should not include all 100
        assert!(result.included_ids.len() < 100);
        assert!(result.estimated_tokens <= 500);
    }

    #[test]
    fn full_injection_combines_tiers() {
        let entries = vec![
            make_entry("mem_1", "Project Orchid", "FastAPI backend", "events"),
        ];
        let result = build_injection(
            Some("Finish auth middleware"),
            &entries,
            &HashSet::new(),
        );
        assert!(result.formatted.contains("handoff"));
        assert!(result.formatted.contains("Orchid"));
        assert!(result.estimated_tokens > 0);
    }

    #[test]
    fn truncate_at_word_boundary() {
        let text = "hello world this is a test";
        let result = truncate_to_budget(text, 15);
        assert_eq!(result, "hello world");
    }
}
