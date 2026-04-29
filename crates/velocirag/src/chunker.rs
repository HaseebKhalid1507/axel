//! Markdown chunking for VelociRAG.
//!
//! Splits markdown documents into semantic chunks based on ## and ### headings,
//! preserving parent context. Port of velocirag/chunker.py.

use std::path::Path;

use regex::Regex;
use sha2::{Digest, Sha256};

// ── Constants ───────────────────────────────────────────────────────────────

const MAX_CHUNK_SIZE: usize = 4000;
const PARAGRAPH_CHUNK_TARGET: usize = 3000; // Target size for splitting headerless content
const MIN_SECTION_SIZE: usize = 10;
const MIN_FILE_SIZE_FOR_CHUNKING: usize = 200;

// ── Types ───────────────────────────────────────────────────────────────────

/// A chunk of a document.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub doc_id: String,
    pub content: String,
    pub file_path: String,
    pub chunk_index: usize,
    pub metadata: serde_json::Value,
}

// ── Chunking ────────────────────────────────────────────────────────────────

/// Split a markdown document into chunks based on ## and ### headings.
/// Preserves parent heading context (# → ## → ###).
pub fn chunk_markdown(content: &str, file_path: &str) -> Vec<Chunk> {
    if content.trim().is_empty() {
        return Vec::new();
    }

    let file_stem = Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Strip frontmatter
    let body = strip_frontmatter(content);

    // Extract h1 title if present
    let h1_re = Regex::new(r"(?m)^#\s+(.+)$").unwrap();
    let h1_title = h1_re.captures(&body[..body.len().min(500)])
        .map(|c| c[1].trim().to_string());

    // Small file — return as single chunk
    if body.trim().len() < MIN_FILE_SIZE_FOR_CHUNKING {
        if body.trim().len() < MIN_SECTION_SIZE {
            return Vec::new();
        }
        return vec![Chunk {
            doc_id: make_doc_id(file_path, 0),
            content: body.trim().to_string(),
            file_path: file_path.to_string(),
            chunk_index: 0,
            metadata: serde_json::json!({
                "heading": h1_title.as_deref().unwrap_or(""),
                "file": file_stem,
                "section": "full_document",
            }),
        }];
    }

    // Find ## and ### headers
    let header_re = Regex::new(r"(?m)^(#{2,3})\s+(.+)$").unwrap();
    let headers: Vec<_> = header_re.captures_iter(&body).collect();

    if headers.is_empty() {
        // No sub-headers — split on paragraph/sentence boundaries
        let trimmed = body.trim().to_string();
        if trimmed.len() <= PARAGRAPH_CHUNK_TARGET {
            if trimmed.len() < MIN_SECTION_SIZE {
                return Vec::new();
            }
            return vec![Chunk {
                doc_id: make_doc_id(file_path, 0),
                content: trimmed,
                file_path: file_path.to_string(),
                chunk_index: 0,
                metadata: serde_json::json!({
                    "heading": h1_title.as_deref().unwrap_or(""),
                    "file": file_stem,
                    "section": "full_document",
                }),
            }];
        }

        return split_by_paragraphs(&trimmed, file_path, file_stem, h1_title.as_deref());
    }

    let mut chunks = Vec::new();
    let mut chunk_index = 0;
    let mut current_h2: Option<String> = None;

    // Process content before first header (intro section)
    let first_header_start = headers[0].get(0).unwrap().start();
    if first_header_start > 0 {
        let intro = body[..first_header_start].trim();
        if intro.len() >= MIN_SECTION_SIZE {
            let mut intro_content = intro.to_string();
            if intro_content.len() > MAX_CHUNK_SIZE {
                intro_content = safe_truncate(&intro_content, MAX_CHUNK_SIZE);
            }
            chunks.push(Chunk {
                doc_id: make_doc_id(file_path, chunk_index),
                content: intro_content,
                file_path: file_path.to_string(),
                chunk_index,
                metadata: serde_json::json!({
                    "heading": h1_title.as_deref().unwrap_or(""),
                    "file": file_stem,
                    "section": "intro",
                }),
            });
            chunk_index += 1;
        }
    }

    // Process each header section
    for (i, cap) in headers.iter().enumerate() {
        let header_start = cap.get(0).unwrap().start();
        let header_end = if i + 1 < headers.len() {
            headers[i + 1].get(0).unwrap().start()
        } else {
            body.len()
        };

        let header_level = cap[1].len(); // 2 = ##, 3 = ###
        let header_text = cap[2].trim().to_string();
        let section_content = body[header_start..header_end].trim();

        // Update parent tracking
        if header_level == 2 {
            current_h2 = Some(header_text.clone());
        }

        // Build content with parent context
        let mut full_content = String::new();
        if let Some(ref h1) = h1_title {
            full_content.push_str(&format!("# {}\n\n", h1));
        }
        if header_level == 3 {
            if let Some(ref h2) = current_h2 {
                full_content.push_str(&format!("## {}\n\n", h2));
            }
        }
        full_content.push_str(section_content);

        // Skip empty sections
        if full_content.trim().len() < MIN_SECTION_SIZE {
            continue;
        }

        // Truncate if too long
        if full_content.len() > MAX_CHUNK_SIZE {
            full_content = safe_truncate(&full_content, MAX_CHUNK_SIZE);
        }

        // Determine parent header
        let parent_header = if header_level == 2 {
            h1_title.clone()
        } else if header_level == 3 {
            current_h2.clone()
        } else {
            None
        };

        chunks.push(Chunk {
            doc_id: make_doc_id(file_path, chunk_index),
            content: full_content.trim().to_string(),
            file_path: file_path.to_string(),
            chunk_index,
            metadata: serde_json::json!({
                "heading": header_text,
                "file": file_stem,
                "section": header_text,
                "parent_header": parent_header,
            }),
        });
        chunk_index += 1;
    }

    chunks
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn make_doc_id(file_path: &str, chunk_index: usize) -> String {
    let mut hasher = Sha256::new();
    hasher.update(file_path.as_bytes());
    hasher.update(chunk_index.to_le_bytes());
    let hash = hex::encode(hasher.finalize());
    format!("{}_{}", &hash[..12], chunk_index)
}

/// Strip YAML frontmatter from markdown content.
fn strip_frontmatter(content: &str) -> String {
    let re = Regex::new(r"(?sm)\A---\s*\n.*?^---\s*\n").unwrap();
    re.replace(content, "").to_string()
}

/// Truncate string at a char boundary, appending "..." 
fn safe_truncate(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }
    let mut end = max_len.saturating_sub(3);
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &text[..end])
}

/// Split content without headers into chunks at paragraph/sentence boundaries.
/// Targets ~PARAGRAPH_CHUNK_TARGET chars per chunk. Tries double-newline splits first,
/// then single newlines, then sentence boundaries (. ! ?).
fn split_by_paragraphs(text: &str, file_path: &str, file_stem: &str, h1_title: Option<&str>) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut chunk_index = 0;
    let mut current = String::new();

    // Split on double newlines first
    let paragraphs: Vec<&str> = text.split("\n\n").collect();

    for para in &paragraphs {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }

        // If this single paragraph is too big, split it further
        if para.len() > MAX_CHUNK_SIZE {
            // Flush current buffer first
            if current.trim().len() >= MIN_SECTION_SIZE {
                chunks.push(make_chunk(file_path, file_stem, h1_title, chunk_index, &current));
                chunk_index += 1;
                current.clear();
            }

            // Split the big paragraph on sentence boundaries
            let sentences = split_sentences(para);
            let mut sent_buf = String::new();
            for sent in &sentences {
                if sent_buf.len() + sent.len() > PARAGRAPH_CHUNK_TARGET && !sent_buf.is_empty() {
                    chunks.push(make_chunk(file_path, file_stem, h1_title, chunk_index, &sent_buf));
                    chunk_index += 1;
                    sent_buf.clear();
                }
                if !sent_buf.is_empty() {
                    sent_buf.push(' ');
                }
                sent_buf.push_str(sent);
            }
            if sent_buf.trim().len() >= MIN_SECTION_SIZE {
                current = sent_buf;
            }
            continue;
        }

        // Would adding this paragraph exceed target?
        if current.len() + para.len() + 2 > PARAGRAPH_CHUNK_TARGET && !current.is_empty() {
            if current.trim().len() >= MIN_SECTION_SIZE {
                chunks.push(make_chunk(file_path, file_stem, h1_title, chunk_index, &current));
                chunk_index += 1;
            }
            current.clear();
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(para);
    }

    // Flush remainder
    if current.trim().len() >= MIN_SECTION_SIZE {
        chunks.push(make_chunk(file_path, file_stem, h1_title, chunk_index, &current));
    }

    chunks
}

fn make_chunk(file_path: &str, file_stem: &str, h1_title: Option<&str>, index: usize, content: &str) -> Chunk {
    Chunk {
        doc_id: make_doc_id(file_path, index),
        content: content.trim().to_string(),
        file_path: file_path.to_string(),
        chunk_index: index,
        metadata: serde_json::json!({
            "heading": h1_title.unwrap_or(""),
            "file": file_stem,
            "section": format!("part_{}", index + 1),
        }),
    }
}

/// Split text into sentences on ". ", "! ", "? " boundaries.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        current.push(chars[i]);

        // Check for sentence boundary: punctuation followed by space
        if (chars[i] == '.' || chars[i] == '!' || chars[i] == '?')
            && i + 1 < len
            && chars[i + 1].is_whitespace()
        {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }

        i += 1;
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_chunking() {
        let content = "# Introduction\n\nThis is the intro with enough content to be valid.\n\n## Details\n\nThis is the details section with enough content to be a valid chunk on its own.";
        let chunks = chunk_markdown(content, "test.md");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_empty_content() {
        let chunks = chunk_markdown("", "test.md");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_parent_context() {
        let content = "# Main Title\n\n## Section A\n\nSome content here.\n\n### Subsection A1\n\nMore detailed content in subsection.";
        let chunks = chunk_markdown(content, "test.md");
        // The ### chunk should contain parent ## context
        if let Some(sub) = chunks.iter().find(|c| c.content.contains("Subsection A1")) {
            assert!(sub.content.contains("# Main Title"));
            assert!(sub.content.contains("## Section A"));
        }
    }

    #[test]
    fn test_frontmatter_stripped() {
        let content = "---\ntitle: Test\ntags:\n  - rust\n---\n# Body\n\nContent here that is long enough.";
        let chunks = chunk_markdown(content, "test.md");
        assert!(!chunks.is_empty());
        assert!(!chunks[0].content.contains("---"));
    }
}
