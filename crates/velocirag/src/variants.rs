//! Query variant generation for improved search recall.
//!
//! Generates normalized query variants to handle common text variations
//! (casing, spacing, punctuation, acronyms) that users might search for differently.
//! Port of velocirag/variants.py.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use regex::Regex;

// ── Constants ───────────────────────────────────────────────────────────────

const MAX_VARIANTS: usize = 12;

// ── Acronym maps ────────────────────────────────────────────────────────────

static ACRONYM_MAP: Lazy<HashMap<&str, &str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("ml", "machine learning");
    m.insert("ai", "artificial intelligence");
    m.insert("nlp", "natural language processing");
    m.insert("llm", "large language model");
    m.insert("rag", "retrieval augmented generation");
    m.insert("nn", "neural network");
    m.insert("cnn", "convolutional neural network");
    m.insert("rnn", "recurrent neural network");
    m.insert("gpu", "graphics processing unit");
    m.insert("cpu", "central processing unit");
    m.insert("api", "application programming interface");
    m.insert("cli", "command line interface");
    m.insert("db", "database");
    m.insert("sql", "structured query language");
    m.insert("ssh", "secure shell");
    m.insert("tls", "transport layer security");
    m.insert("ssl", "secure sockets layer");
    m.insert("dns", "domain name system");
    m.insert("dhcp", "dynamic host configuration protocol");
    m.insert("vpn", "virtual private network");
    m.insert("vm", "virtual machine");
    m.insert("os", "operating system");
    m.insert("ci", "continuous integration");
    m.insert("cd", "continuous deployment");
    m.insert("k8s", "kubernetes");
    m.insert("tf", "tensorflow");
    m.insert("ner", "named entity recognition");
    m.insert("rrf", "reciprocal rank fusion");
    m.insert("bm25", "best match 25");
    m.insert("fts", "full text search");
    m.insert("orm", "object relational mapping");
    m.insert("jwt", "json web token");
    m.insert("oauth", "open authorization");
    m.insert("cors", "cross origin resource sharing");
    m.insert("xss", "cross site scripting");
    m.insert("csrf", "cross site request forgery");
    m.insert("sqli", "sql injection");
    m.insert("mitm", "man in the middle");
    m.insert("ddos", "distributed denial of service");
    m.insert("ids", "intrusion detection system");
    m.insert("ips", "intrusion prevention system");
    m.insert("siem", "security information event management");
    m.insert("osint", "open source intelligence");
    m.insert("ctf", "capture the flag");
    m
});

static REVERSE_ACRONYM_MAP: Lazy<HashMap<&str, &str>> = Lazy::new(|| {
    ACRONYM_MAP.iter().map(|(&k, &v)| (v, k)).collect()
});

// ── Variant generation ──────────────────────────────────────────────────────

/// Generate normalized query variants for improved recall.
/// Original query always appears first in the returned list.
///
/// Patterns handled:
/// - Case variants: "CS656" → "cs656"
/// - Spacing: "CS656" ↔ "CS 656"
/// - Hyphens: "CS-656" → "CS656", "CS 656"
/// - Underscores: "file_name" → "file name", "filename"
/// - Dots: "script.py" → "script py", "scriptpy"
/// - Acronym expansion (bidirectional)
/// - Question → statement rewriting
pub fn generate_variants(query: &str) -> Vec<String> {
    if query.trim().is_empty() {
        return Vec::new();
    }

    let mut variants = vec![query.to_string()];

    // Case variants
    let lower = query.to_lowercase();
    if lower != query {
        push_unique(&mut variants, lower.clone());
    }

    // Pattern 1: Letters followed by numbers → add space
    // e.g., CS656 → CS 656
    let re_letter_num = Regex::new(r"\b([A-Za-z]+)(\d+)\b").unwrap();
    let spaced = re_letter_num.replace_all(query, "$1 $2").to_string();
    if spaced != query {
        push_unique(&mut variants, spaced.clone());
        let spaced_lower = spaced.to_lowercase();
        if spaced_lower != spaced {
            push_unique(&mut variants, spaced_lower);
        }
    }

    // Pattern 2: Letters space numbers → remove space
    // e.g., CS 656 → CS656
    let re_letter_space_num = Regex::new(r"\b([A-Za-z]+)\s+(\d+)\b").unwrap();
    let compressed = re_letter_space_num.replace_all(query, "$1$2").to_string();
    if compressed != query {
        push_unique(&mut variants, compressed.clone());
        let compressed_lower = compressed.to_lowercase();
        if compressed_lower != compressed {
            push_unique(&mut variants, compressed_lower);
        }
    }

    // Pattern 3: Hyphens
    if query.contains('-') {
        let no_hyphen = query.replace('-', "");
        push_unique(&mut variants, no_hyphen.clone());
        push_unique(&mut variants, no_hyphen.to_lowercase());

        let space_hyphen = query.replace('-', " ");
        push_unique(&mut variants, space_hyphen.clone());
        push_unique(&mut variants, space_hyphen.to_lowercase());
    }

    // Pattern 4: Underscores
    if query.contains('_') {
        let no_underscore = query.replace('_', "");
        push_unique(&mut variants, no_underscore);

        let space_underscore = query.replace('_', " ");
        push_unique(&mut variants, space_underscore);
    }

    // Pattern 5: Dots
    if query.contains('.') {
        let no_dot = query.replace('.', "");
        push_unique(&mut variants, no_dot);

        let space_dot = query.replace('.', " ");
        push_unique(&mut variants, space_dot);
    }

    // Pattern 6: Multi-word query — add word pairs
    let words: Vec<&str> = query.split_whitespace().collect();
    if words.len() >= 3 {
        let first_two = format!("{} {}", words[0], words[1]);
        push_unique(&mut variants, first_two.clone());
        push_unique(&mut variants, first_two.to_lowercase());

        let last_two = format!("{} {}", words[words.len() - 2], words[words.len() - 1]);
        push_unique(&mut variants, last_two.clone());
        push_unique(&mut variants, last_two.to_lowercase());
    }

    // Pattern 7: Acronym expansion (bidirectional)
    let query_lower_trimmed = query.to_lowercase();
    let query_lower_trimmed = query_lower_trimmed.trim();
    if let Some(&expansion) = ACRONYM_MAP.get(query_lower_trimmed) {
        push_unique(&mut variants, expansion.to_string());
    } else if let Some(&acronym) = REVERSE_ACRONYM_MAP.get(query_lower_trimmed) {
        push_unique(&mut variants, acronym.to_string());
    }
    // Also check individual words
    for word in query.to_lowercase().split_whitespace() {
        if let Some(&expansion) = ACRONYM_MAP.get(word) {
            let expanded = query.to_lowercase().replace(word, expansion);
            push_unique(&mut variants, expanded);
        }
    }

    // Pattern 8: Question → statement rewrite
    let q_lower = query.to_lowercase();
    let question_prefixes = [
        "what is ", "what are ", "how to ", "how do ", "how does ",
        "why is ", "why are ", "why does ", "when did ", "when does ",
        "where is ", "where are ", "who is ", "who are ",
    ];
    for prefix in &question_prefixes {
        if q_lower.starts_with(prefix) {
            let statement = query[prefix.len()..].trim().trim_end_matches('?').to_string();
            if !statement.is_empty() {
                push_unique(&mut variants, statement);
            }
            break;
        }
    }

    variants.truncate(MAX_VARIANTS);
    variants
}

fn push_unique(variants: &mut Vec<String>, value: String) {
    if !value.is_empty() && !variants.contains(&value) {
        variants.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_case_variants() {
        let v = generate_variants("CS656");
        assert!(v.contains(&"CS656".to_string()));
        assert!(v.contains(&"cs656".to_string()));
    }

    #[test]
    fn test_spacing_variants() {
        let v = generate_variants("CS656");
        assert!(v.contains(&"CS 656".to_string()));

        let v2 = generate_variants("CS 656");
        assert!(v2.contains(&"CS656".to_string()));
    }

    #[test]
    fn test_hyphen_variants() {
        let v = generate_variants("hello-world");
        assert!(v.contains(&"helloworld".to_string()));
        assert!(v.contains(&"hello world".to_string()));
    }

    #[test]
    fn test_acronym_expansion() {
        let v = generate_variants("ml");
        assert!(v.contains(&"machine learning".to_string()));
    }

    #[test]
    fn test_question_rewrite() {
        let v = generate_variants("what is machine learning?");
        assert!(v.contains(&"machine learning".to_string()));
    }

    #[test]
    fn test_empty() {
        assert!(generate_variants("").is_empty());
        assert!(generate_variants("   ").is_empty());
    }

    #[test]
    fn test_max_variants() {
        let v = generate_variants("CS-656_test.file some more words here");
        assert!(v.len() <= MAX_VARIANTS);
    }
}
