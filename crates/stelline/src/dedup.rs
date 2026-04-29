use axel_memkoshi::memory::Memory;

/// Check if a new memory is a duplicate of any existing memory.
/// Uses normalized Levenshtein distance on titles, threshold 0.85.
pub fn is_duplicate(new_memory: &Memory, existing: &[Memory]) -> bool {
    existing.iter().any(|m| {
        let dist = normalized_levenshtein(&new_memory.title, &m.title);
        dist >= 0.85
    })
}

/// Normalized Levenshtein similarity (1.0 = identical, 0.0 = completely different).
pub fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    let a = a.to_lowercase();
    let b = b.to_lowercase();

    if a == b {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let len_a = a.chars().count();
    let len_b = b.chars().count();
    let max_len = len_a.max(len_b);

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let mut prev: Vec<usize> = (0..=len_b).collect();
    let mut curr = vec![0usize; len_b + 1];

    for i in 1..=len_a {
        curr[0] = i;
        for j in 1..=len_b {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    let distance = prev[len_b];
    1.0 - (distance as f64 / max_len as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── normalized_levenshtein ────────────────────────────────────────────────

    #[test]
    fn identical_strings_score_one() {
        assert_eq!(normalized_levenshtein("hello world", "hello world"), 1.0);
    }

    #[test]
    fn identical_case_insensitive() {
        assert_eq!(normalized_levenshtein("Rust", "rust"), 1.0);
    }

    #[test]
    fn completely_different_strings_score_low() {
        let score = normalized_levenshtein("abcdefgh", "xyzwvuts");
        assert!(score < 0.3, "expected low score, got {score}");
    }

    #[test]
    fn similar_titles_score_above_threshold() {
        // One character typo — should stay above 0.85 for short edits on long strings.
        let score = normalized_levenshtein(
            "Levenshtein distance algorithm",
            "Levenshtein distance algorithme",
        );
        assert!(score >= 0.85, "expected ≥0.85, got {score}");
    }

    #[test]
    fn empty_a_scores_zero() {
        assert_eq!(normalized_levenshtein("", "something"), 0.0);
    }

    #[test]
    fn empty_b_scores_zero() {
        assert_eq!(normalized_levenshtein("something", ""), 0.0);
    }

    #[test]
    fn both_empty_scores_one() {
        // Both empty → equal strings, fast-path returns 1.0.
        assert_eq!(normalized_levenshtein("", ""), 1.0);
    }

    // ── is_duplicate ─────────────────────────────────────────────────────────

    fn make_memory(title: &str) -> Memory {
        Memory {
            title: title.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn duplicate_detected_for_similar_title() {
        let existing = vec![make_memory("Levenshtein distance algorithm")];
        let candidate = make_memory("Levenshtein distance algorithme"); // one char diff
        assert!(is_duplicate(&candidate, &existing));
    }

    #[test]
    fn no_duplicate_for_distinct_title() {
        let existing = vec![make_memory("Quantum computing basics")];
        let candidate = make_memory("How to brew espresso");
        assert!(!is_duplicate(&candidate, &existing));
    }

    #[test]
    fn no_duplicate_against_empty_list() {
        let candidate = make_memory("anything");
        assert!(!is_duplicate(&candidate, &[]));
    }

    #[test]
    fn exact_title_match_is_duplicate() {
        let existing = vec![make_memory("Rust ownership model")];
        let candidate = make_memory("Rust ownership model");
        assert!(is_duplicate(&candidate, &existing));
    }
}
