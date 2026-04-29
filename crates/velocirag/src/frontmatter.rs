//! YAML frontmatter parsing for markdown documents.
//!
//! Extracts YAML frontmatter, #hashtags, and [[wiki links]] from markdown content.
//! Port of velocirag/frontmatter.py.

use std::collections::HashSet;

use regex::Regex;
use serde_json::Value;

/// Parse YAML frontmatter from markdown content.
///
/// Returns (frontmatter_dict, body_without_frontmatter).
/// If no frontmatter found, returns (empty object, original content).
pub fn parse_frontmatter(content: &str) -> (Value, String) {
    if content.trim().is_empty() {
        return (Value::Object(Default::default()), content.to_string());
    }

    // Match ---\n...\n---\n at the start of the document
    let re = Regex::new(r"(?sm)\A---\s*\n(.*?)^---\s*\n").unwrap();

    let Some(caps) = re.captures(content) else {
        return (Value::Object(Default::default()), content.to_string());
    };

    let yaml_content = &caps[1];
    let body = &content[caps.get(0).unwrap().end()..];

    // Parse YAML — we use serde_yaml
    match serde_yaml::from_str::<Value>(yaml_content) {
        Ok(Value::Object(map)) => {
            let normalized = normalize_frontmatter(Value::Object(map));
            (normalized, body.to_string())
        }
        Ok(_) => {
            // Non-dict YAML (string, list, etc.) — treat as empty
            tracing::warn!("Frontmatter parsed to non-object type");
            (Value::Object(Default::default()), body.to_string())
        }
        Err(e) => {
            tracing::warn!("Failed to parse YAML frontmatter: {}", e);
            (Value::Object(Default::default()), body.to_string())
        }
    }
}

/// Extract #hashtags from markdown content.
///
/// Matches: #tag, #tag-name, #tag_name
/// Does not match: ##header, # header, #123 (numbers only)
pub fn extract_tags_from_content(content: &str) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }

    let re = Regex::new(r"(?:^|\s)#([a-zA-Z][a-zA-Z0-9_-]*)").unwrap();
    let mut tags = Vec::new();
    let mut seen = HashSet::new();

    for cap in re.captures_iter(content) {
        let tag = cap[1].to_lowercase();
        if !tag.is_empty() && seen.insert(tag.clone()) {
            tags.push(tag);
        }
    }

    tags
}

/// Extract [[wiki links]] from markdown content.
///
/// Handles both `[[target]]` and `[[display|target]]` formats.
pub fn extract_wiki_links(content: &str) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }

    let re = Regex::new(r"\[\[([^\[\]]+)\]\]").unwrap();
    let mut links = Vec::new();
    let mut seen = HashSet::new();

    for cap in re.captures_iter(content) {
        let raw = &cap[1];
        // Handle display|target format — take the target part
        let target = if let Some(pos) = raw.find('|') {
            raw[pos + 1..].trim()
        } else {
            raw.trim()
        };

        if !target.is_empty() && seen.insert(target.to_string()) {
            links.push(target.to_string());
        }
    }

    links
}

/// Normalize frontmatter values for JSON compatibility.
/// Converts any non-JSON-native types to strings.
fn normalize_frontmatter(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let normalized = map
                .into_iter()
                .map(|(k, v)| (k, normalize_frontmatter(v)))
                .collect();
            Value::Object(normalized)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(normalize_frontmatter).collect())
        }
        // serde_yaml already handles most types; just pass through
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_frontmatter() {
        let content = "---\ntitle: Hello World\ntags:\n  - rust\n  - rag\n---\n# Body\n\nContent here.";
        let (fm, body) = parse_frontmatter(content);
        assert_eq!(fm["title"], "Hello World");
        assert!(body.contains("# Body"));
        assert!(body.contains("Content here."));
    }

    #[test]
    fn test_no_frontmatter() {
        let content = "# Just a heading\n\nNo frontmatter here.";
        let (fm, body) = parse_frontmatter(content);
        assert!(fm.as_object().unwrap().is_empty());
        assert_eq!(body, content);
    }

    #[test]
    fn test_extract_tags() {
        let content = "Some text #rust and #machine-learning here\n#python too\nNot ##heading";
        let tags = extract_tags_from_content(content);
        assert!(tags.contains(&"rust".to_string()));
        assert!(tags.contains(&"machine-learning".to_string()));
        assert!(tags.contains(&"python".to_string()));
    }

    #[test]
    fn test_extract_wiki_links() {
        let content = "See [[Some Note]] and [[display text|Actual Target]] for details.";
        let links = extract_wiki_links(content);
        assert!(links.contains(&"Some Note".to_string()));
        assert!(links.contains(&"Actual Target".to_string()));
    }
}
