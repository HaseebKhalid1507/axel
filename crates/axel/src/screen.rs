//! `redact`, `show`, `core_withholds`. Commit 8 writes this file.

use std::borrow::Cow;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    /// "prefix" | "jwt" | "key_block" | "url_password" | "named_value" | "long_token".
    pub rule: &'static str,
    /// The prefix, or the name before the value: never the value.
    pub name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Redacted<'a> {
    pub text: Cow<'a, str>,
    pub blanked: usize,
    pub hits: Vec<Hit>,
}

/// Blanks credential-shaped tokens; the rest is untouched. The stub returns its input
/// unchanged.
pub fn redact(text: &str) -> Redacted<'_> {
    Redacted { text: Cow::Borrowed(text), blanked: 0, hits: Vec::new() }
}

/// The first hit of the four certain rules (the forum's test).
pub fn certain_hit(text: &str) -> Option<Hit> {
    let _ = text;
    None
}

/// Redact, then make it one harmless line. The stub returns its input unchanged.
pub fn show(text: &str, max_chars: usize) -> String {
    let _ = max_chars;
    text.to_string()
}

/// The same, keeping line breaks: command-line output. The stub returns its input unchanged.
pub fn show_block(text: &str, max_chars: usize) -> String {
    let _ = max_chars;
    text.to_string()
}

/// The copy of core's withholding rule. The stub returns `false`.
pub fn core_withholds(text: &str) -> bool {
    let _ = text;
    false
}
