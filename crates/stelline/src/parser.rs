//! Session transcript parser.
//!
//! Parses SynapsCLI JSONL session files into a clean sequence of
//! [`TranscriptMessage`] values. Each line of the session file holds one JSON
//! object; this module normalises the various message shapes into a single,
//! uniform type that downstream extractors can work with.
//!
//! ## JSONL shapes understood
//!
//! ```jsonc
//! // Human turn
//! {"type":"user","message":{"role":"user","content":"..."}}
//!
//! // Assistant turn — content is an array of typed blocks
//! {"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"..."}]}}
//!
//! // Tool invocation (skipped — too noisy for memory extraction)
//! {"type":"tool_use","name":"bash","input":{"command":"ls"}}
//!
//! // Tool result (skipped)
//! {"type":"tool_result","content":"..."}
//! ```

use std::path::Path;

use serde::Deserialize;
use tracing::warn;

use crate::error::Result;
use crate::error::StellineError;

// ── Public types ─────────────────────────────────────────────────────────────

/// A single, normalised message extracted from a session transcript.
#[derive(Debug, Clone)]
pub struct TranscriptMessage {
    /// Speaker role: `"user"` or `"assistant"`.
    pub role: String,
    /// Plain-text content of the message. For assistant turns that contain
    /// multiple text blocks the blocks are joined with `"\n"`.
    pub content: String,
    /// ISO-8601 timestamp, if the session file recorded one.
    pub timestamp: Option<String>,
}

// ── JSONL serde shapes ───────────────────────────────────────────────────────
//
// These types mirror exactly what SynapsCLI writes; they are private
// implementation detail — callers only ever see `TranscriptMessage`.

/// Top-level entry in the session JSONL file.
#[derive(Debug, Deserialize)]
struct SessionEntry {
    #[serde(rename = "type")]
    entry_type: String,
    /// Present on `user` and `assistant` entries.
    message: Option<RawMessage>,
    /// Optional wall-clock timestamp (not always emitted by SynapsCLI).
    timestamp: Option<String>,
}

/// The `message` object nested inside user/assistant entries.
#[derive(Debug, Deserialize)]
struct RawMessage {
    role: String,
    /// Content can arrive as a plain string **or** as a JSON array of typed
    /// content blocks. [`RawContent`] handles both shapes via `untagged`.
    content: RawContent,
}

/// Represents the polymorphic `content` field of a [`RawMessage`].
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawContent {
    /// Simple string form: `"content": "hello"`.
    Text(String),
    /// Array form: `"content": [{"type":"text","text":"..."}]`.
    Blocks(Vec<ContentBlock>),
}

/// A single content block in the array form.
#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    /// Present when `block_type == "text"`.
    text: Option<String>,
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Parse a SynapsCLI JSONL session file into a transcript.
///
/// Lines that represent tool invocations (`tool_use`) or tool results
/// (`tool_result`) are intentionally skipped — they are too noisy and
/// structured for meaningful memory extraction. Malformed lines emit a
/// warning and are skipped rather than aborting the parse.
///
/// # Errors
///
/// Returns [`StellineError::Io`] if the file cannot be opened or read.
/// Individual malformed lines produce warnings via `tracing` but do **not**
/// cause an error return.
pub fn parse_session(path: &Path) -> Result<Vec<TranscriptMessage>> {
    let raw = std::fs::read_to_string(path).map_err(StellineError::Io)?;
    let mut messages = Vec::new();

    for (idx, line) in raw.lines().enumerate() {
        let line: &str = line.trim();
        if line.is_empty() {
            continue;
        }

        let entry: SessionEntry = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(source) => {
                warn!(
                    line = idx + 1,
                    content = %truncate(line, 80),
                    "Skipping malformed JSONL line: {source}"
                );
                continue;
            }
        };

        // Skip tool noise — not useful for memory extraction.
        if matches!(entry.entry_type.as_str(), "tool_use" | "tool_result") {
            continue;
        }

        let Some(msg) = entry.message else {
            // Some synthetic entries (e.g. session metadata) have no `message`.
            continue;
        };

        let role = msg.role.clone();
        if !matches!(role.as_str(), "user" | "assistant") {
            continue;
        }

        let content = extract_text_content(&msg.content);
        if content.is_empty() {
            continue;
        }

        messages.push(TranscriptMessage {
            role,
            content,
            timestamp: entry.timestamp,
        });
    }

    Ok(messages)
}

/// Format a slice of transcript messages into a human-readable string.
///
/// Each message is rendered as `"ROLE: content"` on its own line, where
/// `ROLE` is the uppercased role string. Messages are separated by a single
/// blank line for readability.
///
/// # Example
///
/// ```
/// # use axel_stelline::parser::{TranscriptMessage, format_transcript};
/// let msgs = vec![
///     TranscriptMessage { role: "user".into(), content: "Hello".into(), timestamp: None },
///     TranscriptMessage { role: "assistant".into(), content: "Hi there".into(), timestamp: None },
/// ];
/// let text = format_transcript(&msgs);
/// assert!(text.contains("USER: Hello"));
/// assert!(text.contains("ASSISTANT: Hi there"));
/// ```
pub fn format_transcript(messages: &[TranscriptMessage]) -> String {
    messages
        .iter()
        .map(|m| format!("{}: {}", m.role.to_uppercase(), m.content))
        .collect::<Vec<_>>()
        .join("\n\n")
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Collapse a [`RawContent`] to plain text.
///
/// For block arrays, only `"text"` blocks are collected; other block types
/// (e.g. `"image"`, `"tool_use"`) are ignored. Blocks are joined with `"\n"`.
fn extract_text_content(content: &RawContent) -> String {
    match content {
        RawContent::Text(s) => s.trim().to_owned(),
        RawContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| {
                if b.block_type == "text" {
                    b.text.as_deref().map(str::trim).filter(|t| !t.is_empty())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

/// Truncate a string to `max_chars` with an ellipsis, for log messages.
fn truncate(s: &str, max_chars: usize) -> &str {
    // Return a prefix slice; we tolerate splitting at a byte boundary that
    // isn't a char boundary only in the log label, so a simple byte slice is
    // acceptable here.
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
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_session(lines: &[&str]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        f
    }

    #[test]
    fn parses_user_and_assistant_string_content() {
        let session = write_session(&[
            r#"{"type":"user","message":{"role":"user","content":"What is Rust?"}}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":"Rust is a systems language."}}"#,
        ]);
        let msgs = parse_session(session.path()).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "What is Rust?");
        assert_eq!(msgs[1].role, "assistant");
        assert_eq!(msgs[1].content, "Rust is a systems language.");
    }

    #[test]
    fn parses_assistant_block_content() {
        let session = write_session(&[
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Block one"},{"type":"text","text":"Block two"}]}}"#,
        ]);
        let msgs = parse_session(session.path()).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "Block one\nBlock two");
    }

    #[test]
    fn skips_tool_use_and_tool_result() {
        let session = write_session(&[
            r#"{"type":"user","message":{"role":"user","content":"Run ls"}}"#,
            r#"{"type":"tool_use","name":"bash","input":{"command":"ls"}}"#,
            r#"{"type":"tool_result","content":"file.txt"}"#,
        ]);
        let msgs = parse_session(session.path()).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
    }

    #[test]
    fn skips_non_text_blocks_in_array() {
        let session = write_session(&[
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"image","source":"..."},{"type":"text","text":"Here is the result"}]}}"#,
        ]);
        let msgs = parse_session(session.path()).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "Here is the result");
    }

    #[test]
    fn skips_malformed_lines_without_error() {
        let session = write_session(&[
            r#"{"type":"user","message":{"role":"user","content":"Good line"}}"#,
            r#"not json at all {{{{"#,
            r#"{"type":"user","message":{"role":"user","content":"Another good line"}}"#,
        ]);
        let msgs = parse_session(session.path()).unwrap();
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn skips_empty_lines() {
        let session = write_session(&[
            "",
            r#"{"type":"user","message":{"role":"user","content":"Hello"}}"#,
            "   ",
        ]);
        let msgs = parse_session(session.path()).unwrap();
        assert_eq!(msgs.len(), 1);
    }

    #[test]
    fn format_transcript_uppercases_role() {
        let msgs = vec![TranscriptMessage {
            role: "user".into(),
            content: "Hello".into(),
            timestamp: None,
        }];
        assert!(format_transcript(&msgs).starts_with("USER: Hello"));
    }

    #[test]
    fn format_transcript_joins_with_blank_line() {
        let msgs = vec![
            TranscriptMessage {
                role: "user".into(),
                content: "Hi".into(),
                timestamp: None,
            },
            TranscriptMessage {
                role: "assistant".into(),
                content: "Hello".into(),
                timestamp: None,
            },
        ];
        let text = format_transcript(&msgs);
        assert!(text.contains("\n\n"));
    }

    #[test]
    fn format_transcript_empty_slice() {
        assert_eq!(format_transcript(&[]), "");
    }
}
