//! Regex-based memory extraction — the free, zero-API-cost extraction path.
//!
//! [`extract_regex`] scans a formatted transcript string for lines that match
//! one of four semantic pattern groups and promotes each match to a candidate
//! [`Memory`]. No model calls, no tokens spent, no network — just pattern
//! matching and a bit of heuristic scoring.
//!
//! The extracted memories are **candidates**; they must still pass the quality
//! gate ([`crate::quality`]) and deduplication ([`crate::dedup`]) before being
//! committed to the `.r8` store.
//!
//! ## Pattern groups
//!
//! | Group | Keywords | Category | Base importance |
//! |---|---|---|---|
//! | Events | decided, chose, agreed, shipped, deployed, merged, fixed, built, created, completed, finished, launched | `Events` | 0.6 – 0.8 |
//! | Preferences | prefer, always use, don't like, switched to, configured, set up | `Preferences` | 0.6 |
//! | Entities | names after "with/from/by/@", URLs, project refs | `Entities` | 0.5 |
//! | Cases | error, bug, crash, issue, problem, workaround, solution, fixed by | `Cases` | 0.65 |
//!
//! ## Noise filter
//!
//! Lines are discarded before pattern matching when they:
//! - Are shorter than 20 characters.
//! - Start with `$`, `#`, or ` ``` ` (shell prompt / comment / code fence).
//! - Contain more than 50 % non-alphanumeric characters (dense code/binary).

use std::sync::OnceLock;

use axel_memkoshi::memory::{Memory, MemoryCategory};
use regex::Regex;
use tracing::trace;

// ── Compiled regex patterns (initialised once) ────────────────────────────────

struct Patterns {
    events_high: Regex,   // importance 0.8
    events_mid: Regex,    // importance 0.7
    preferences: Regex,   // importance 0.6
    entities: Regex,      // importance 0.5
    cases: Regex,         // importance 0.65
    url: Regex,
    noise_code_fence: Regex,
}

impl Patterns {
    fn new() -> Self {
        Self {
            // High-signal action verbs — decisions and shipping moments.
            events_high: Regex::new(
                r"(?i)\b(decided|shipped|deployed|merged|launched|completed|finished)\b",
            )
            .expect("static regex"),

            // Mid-signal action verbs — construction and fixing.
            events_mid: Regex::new(
                r"(?i)\b(chose|agreed|fixed|built|created)\b",
            )
            .expect("static regex"),

            // Preference and configuration signals.
            preferences: Regex::new(
                r"(?i)\b(prefer|always use|don't like|switched to|configured|set up)\b",
            )
            .expect("static regex"),

            // Entity detection — capitalised words after relational tokens or @-mentions.
            entities: Regex::new(
                r"(?:(?:with|from|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)|@\w+)",
            )
            .expect("static regex"),

            // Problem / resolution signals.
            cases: Regex::new(
                r"(?i)\b(error|bug|crash|issue|problem|workaround|solution|fixed by)\b",
            )
            .expect("static regex"),

            // URL pattern — presence of a URL is its own entity signal.
            url: Regex::new(r"https?://[^\s]+").expect("static regex"),

            // Code fence opener — lines starting with backtick runs.
            noise_code_fence: Regex::new(r"^```").expect("static regex"),
        }
    }
}

static PATTERNS: OnceLock<Patterns> = OnceLock::new();

fn patterns() -> &'static Patterns {
    PATTERNS.get_or_init(Patterns::new)
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Extract candidate memories from a formatted transcript using regex patterns.
///
/// `transcript` should be the output of [`crate::parser::format_transcript`],
/// though any plain-text document works. Each matching line can produce at most
/// **one** memory (the highest-priority pattern wins for that line).
///
/// The returned memories are unsorted, undeduped candidates. Pass them through
/// [`crate::quality::quality_gate`] and [`crate::dedup`] before storage.
///
/// # Performance
///
/// The regex patterns are compiled once on first call (via [`OnceLock`]) and
/// reused for all subsequent calls. For large transcripts the function is
/// linear in the number of lines.
pub fn extract_regex(transcript: &str) -> Vec<Memory> {
    let p = patterns();
    let mut memories = Vec::new();

    for line in transcript.lines() {
        let trimmed = line.trim();

        if is_noise(trimmed, p) {
            trace!(line = %truncate(trimmed, 60), "Skipping noisy line");
            continue;
        }

        // Pattern priority order: cases (most actionable) → events high →
        // events mid → preferences → entities.
        if let Some(mem) = try_extract_case(trimmed, p) {
            memories.push(mem);
        } else if let Some(mem) = try_extract_event_high(trimmed, p) {
            memories.push(mem);
        } else if let Some(mem) = try_extract_event_mid(trimmed, p) {
            memories.push(mem);
        } else if let Some(mem) = try_extract_preference(trimmed, p) {
            memories.push(mem);
        } else if let Some(mem) = try_extract_entity(trimmed, p) {
            memories.push(mem);
        }
    }

    memories
}

// ── Noise filtering ───────────────────────────────────────────────────────────

/// Return `true` when a line is too noisy to extract a meaningful memory from.
fn is_noise(line: &str, p: &Patterns) -> bool {
    // 1. Too short to carry meaning.
    if line.len() < 20 {
        return true;
    }

    // 2. Shell prompt, markdown heading, or code fence.
    if line.starts_with('$') || line.starts_with('#') || p.noise_code_fence.is_match(line) {
        return true;
    }

    // 3. Dense non-alphanumeric content → probably code or binary.
    let non_alnum = line
        .chars()
        .filter(|c| !c.is_alphanumeric() && *c != ' ')
        .count();
    let ratio = non_alnum as f64 / line.len() as f64;
    if ratio > 0.50 {
        return true;
    }

    false
}

// ── Per-group extractors ───────────────────────────────────────────────────────

fn try_extract_case(line: &str, p: &Patterns) -> Option<Memory> {
    if !p.cases.is_match(line) {
        return None;
    }
    let title = build_title(line);
    let mut m = Memory::new(
        MemoryCategory::Cases,
        derive_topic(line),
        &title,
        line,
    );
    m.importance = 0.65;
    Some(m)
}

fn try_extract_event_high(line: &str, p: &Patterns) -> Option<Memory> {
    if !p.events_high.is_match(line) {
        return None;
    }
    let title = build_title(line);
    let mut m = Memory::new(
        MemoryCategory::Events,
        derive_topic(line),
        &title,
        line,
    );
    m.importance = 0.8;
    Some(m)
}

fn try_extract_event_mid(line: &str, p: &Patterns) -> Option<Memory> {
    if !p.events_mid.is_match(line) {
        return None;
    }
    let title = build_title(line);
    let mut m = Memory::new(
        MemoryCategory::Events,
        derive_topic(line),
        &title,
        line,
    );
    m.importance = 0.7;
    Some(m)
}

fn try_extract_preference(line: &str, p: &Patterns) -> Option<Memory> {
    if !p.preferences.is_match(line) {
        return None;
    }
    let title = build_title(line);
    let mut m = Memory::new(
        MemoryCategory::Preferences,
        derive_topic(line),
        &title,
        line,
    );
    m.importance = 0.6;
    Some(m)
}

fn try_extract_entity(line: &str, p: &Patterns) -> Option<Memory> {
    // Accept lines with an @-mention, a relational name, or a URL.
    if !p.entities.is_match(line) && !p.url.is_match(line) {
        return None;
    }
    let title = build_title(line);
    let mut m = Memory::new(
        MemoryCategory::Entities,
        derive_topic(line),
        &title,
        line,
    );
    m.importance = 0.5;
    Some(m)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Produce a title from the first ≤60 characters of a line.
fn build_title(line: &str) -> String {
    // Strip the `ROLE: ` prefix that format_transcript inserts, if present.
    let stripped = strip_role_prefix(line);
    let chars: String = stripped.chars().take(60).collect();
    chars.trim_end_matches(|c: char| !c.is_alphanumeric()).to_owned()
}

/// Strip a `"ROLE: "` prefix produced by [`crate::parser::format_transcript`].
fn strip_role_prefix(line: &str) -> &str {
    // Lines look like "USER: ..." or "ASSISTANT: ..."
    if let Some(colon_pos) = line.find(": ") {
        let prefix = &line[..colon_pos];
        if prefix.chars().all(|c| c.is_uppercase() || c == '_') {
            return &line[colon_pos + 2..];
        }
    }
    line
}

/// Derive a short topic label from a line.
///
/// Strategy:
/// 1. If the line has a `ROLE: ` prefix, use the role as a topic seed.
/// 2. Otherwise extract the first two meaningful words.
fn derive_topic(line: &str) -> &str {
    let stripped = strip_role_prefix(line);

    // Take the first "word" that is longer than 3 characters as the topic.
    for word in stripped.split_whitespace() {
        let clean: String = word.chars().filter(|c| c.is_alphanumeric()).collect();
        if clean.len() > 3 {
            // We need to return a `&str` slice of the original input to avoid
            // allocation, but since we can't return a reference to `clean` we
            // fall back to a static topic when the first meaningful word is
            // found only after stripping punctuation.  For the common case the
            // word boundary in the original string matches.
            if let Some(start) = line.find(clean.as_str()) {
                let end = (start + clean.len()).min(line.len());
                return &line[start..end];
            }
        }
    }

    "general"
}

/// Truncate a string for trace logging (does not allocate).
fn truncate(s: &str, max_chars: usize) -> &str {
    let end = s
        .char_indices()
        .nth(max_chars)
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    &s[..end]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_event_from_shipped_line() {
        let transcript =
            "USER: We shipped the new auth module to production yesterday.";
        let mems = extract_regex(transcript);
        assert!(!mems.is_empty(), "Should extract at least one memory");
        let event = mems.iter().find(|m| m.category == MemoryCategory::Events);
        assert!(event.is_some(), "Should produce an Events memory");
        assert!((event.unwrap().importance - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn extracts_preference_line() {
        let transcript =
            "ASSISTANT: I always use cargo fmt before committing Rust code.";
        let mems = extract_regex(transcript);
        let pref = mems.iter().find(|m| m.category == MemoryCategory::Preferences);
        assert!(pref.is_some());
        assert!((pref.unwrap().importance - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn extracts_case_from_error_line() {
        let transcript =
            "USER: We hit a crash in the embedder when the model file is missing on cold start.";
        let mems = extract_regex(transcript);
        let case = mems.iter().find(|m| m.category == MemoryCategory::Cases);
        assert!(case.is_some());
        assert!((case.unwrap().importance - 0.65).abs() < f64::EPSILON);
    }

    #[test]
    fn extracts_entity_from_url() {
        let transcript =
            "ASSISTANT: See the docs at https://docs.rs/axel for more information about usage.";
        let mems = extract_regex(transcript);
        let entity = mems.iter().find(|m| m.category == MemoryCategory::Entities);
        assert!(entity.is_some());
    }

    #[test]
    fn skips_short_lines() {
        let transcript = "ls\ncd /tmp\nok";
        let mems = extract_regex(transcript);
        assert!(mems.is_empty(), "Short lines should produce no memories");
    }

    #[test]
    fn skips_shell_prompt_lines() {
        let transcript = "$ cargo build --release\n$ ls -la";
        let mems = extract_regex(transcript);
        assert!(mems.is_empty());
    }

    #[test]
    fn skips_code_fence_lines() {
        let transcript = "```rust\nlet x = 1;\n```";
        let mems = extract_regex(transcript);
        assert!(mems.is_empty());
    }

    #[test]
    fn skips_dense_code_lines() {
        // >50% non-alphanumeric (lots of symbols)
        let transcript = "!!@#$%^&*(){}[]|<>?/\\:;,~`_+-=";
        let mems = extract_regex(transcript);
        assert!(mems.is_empty());
    }

    #[test]
    fn cases_take_priority_over_events() {
        // "fixed" is in both events_mid and cases — cases should win.
        let line =
            "ASSISTANT: We fixed by adding retry logic after the crash in the auth service module.";
        let mems = extract_regex(line);
        assert!(!mems.is_empty());
        assert_eq!(mems[0].category, MemoryCategory::Cases);
    }

    #[test]
    fn title_truncated_to_sixty_chars() {
        let long =
            "USER: We decided that the entire configuration system should be rewritten from scratch in Rust.";
        let mems = extract_regex(long);
        assert!(!mems.is_empty());
        assert!(mems[0].title.len() <= 60);
    }

    #[test]
    fn mid_event_importance() {
        let transcript =
            "USER: We built the first working prototype of the new search pipeline last week.";
        let mems = extract_regex(transcript);
        let event = mems.iter().find(|m| m.category == MemoryCategory::Events);
        assert!(event.is_some());
        assert!((event.unwrap().importance - 0.7).abs() < f64::EPSILON);
    }
}
