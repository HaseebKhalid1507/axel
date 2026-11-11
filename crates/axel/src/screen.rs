//! What text may be stored or shown: `redact` blanks credential-shaped tokens, `show` and
//! `show_block` make stored text harmless for a model or a terminal, and `core_withholds`
//! is the copy of the rule Synaps core uses to withhold a string from its archives.
//!
//! Pure. A shape test, not a secret detector, in both directions (spec section 8).

use std::borrow::Cow;
use std::sync::OnceLock;

const BLANK: &str = "(redacted)";

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

/// Blanks credential-shaped tokens with `(redacted)`; the rest is untouched.
pub fn redact(text: &str) -> Redacted<'_> {
    let view = View::new(text);
    let spans = scan(view.as_str(), false);
    if spans.is_empty() {
        return Redacted { text: Cow::Borrowed(text), blanked: 0, hits: Vec::new() };
    }
    let mut out = String::with_capacity(text.len());
    let mut at = 0;
    let mut hits = Vec::with_capacity(spans.len());
    for span in spans {
        let (start, end) = view.original(span.start, span.end);
        out.push_str(&text[at..start]);
        out.push_str(BLANK);
        at = end;
        hits.push(span.hit);
    }
    out.push_str(&text[at..]);
    Redacted { text: Cow::Owned(out), blanked: hits.len(), hits }
}

/// Where `redact` blanks: the byte range of each blanked token in `text`, in order, with its
/// hit. For measuring what the rules do to real text (`examples/screen_measure.rs`); a
/// caller that stores or shows text uses `redact`.
// RULING: wave2-c8 review A1: measure first. `Hit` already names the rule; this adds where.
pub fn redact_spans(text: &str) -> Vec<(std::ops::Range<usize>, Hit)> {
    let view = View::new(text);
    scan(view.as_str(), false)
        .into_iter()
        .map(|span| {
            let (start, end) = view.original(span.start, span.end);
            (start..end, span.hit)
        })
        .collect()
}

/// The first hit of the four certain rules (the forum's test).
pub fn certain_hit(text: &str) -> Option<Hit> {
    let view = View::new(text);
    scan(view.as_str(), true).into_iter().next().map(|span| span.hit)
}

/// Redact, then make it one harmless line.
pub fn show(text: &str, max_chars: usize) -> String {
    cut(clean_line(&redact(text).text), max_chars)
}

/// The same, keeping line breaks: command-line output.
// SPEC: 4202: "the same five steps line by line" does not say whether `max_chars` bounds a
// line or the block. It bounds the whole block here, so the output is never longer than asked.
pub fn show_block(text: &str, max_chars: usize) -> String {
    let redacted = redact(text);
    let lines: Vec<String> = redacted.text.split('\n').map(clean_line).collect();
    cut(lines.join("\n"), max_chars)
}

/// The copy of core's withholding rule: true when the ASCII-lowercase form of `text`
/// contains one of the words of `core_withheld.txt`. Core stores such a string as
/// `[archive: withheld]`, whole. A copy that cannot be read withholds everything.
// RULING: wave2-c8 review A4: this is core's FUNCTION, not core's decision. Core applies
// `credential_like` to the output of its redactor, so a word the redactor removed is stored
// after all, and it withholds some classes of content whatever their words. Called on the
// text Axel was given, this is close to what core does with a string, not equal to it:
// Axel may skip a turn core stores, and core may withhold one this function passes.
// `scripts/check-core-rule.sh` proves the words and the text of the function (the
// `# sha256` line of the copy), and nothing of what core does around it.
pub fn core_withholds(text: &str) -> bool {
    let rule = core_rule();
    if !rule.ok {
        return true;
    }
    let lower = text.to_ascii_lowercase();
    rule.words.iter().any(|needle| lower.contains(needle.as_str()))
}

/// The words of the copy, in core's order.
pub fn core_rule_words() -> &'static [String] {
    &core_rule().words
}

/// The core commit the copy was taken at (the file's first line).
pub fn core_rule_commit() -> &'static str {
    core_rule().commit
}

struct CoreRule {
    commit: &'static str,
    words: Vec<String>,
    ok: bool,
}

fn core_rule() -> &'static CoreRule {
    static RULE: OnceLock<CoreRule> = OnceLock::new();
    RULE.get_or_init(|| {
        let mut lines = include_str!("core_withheld.txt").lines();
        let commit = lines.next().and_then(|l| l.strip_prefix("# core ")).map(str::trim).unwrap_or("");
        let mut words = Vec::new();
        let mut ok = !commit.is_empty();
        // A `#` line after the first is a comment: the `# sha256` of core's function, which
        // the script reads. No word starts with `#`: a word is a JSON string.
        for line in lines.map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
            match serde_json::from_str::<String>(line) {
                Ok(word) if !word.is_empty() => words.push(word),
                _ => ok = false,
            }
        }
        ok &= !words.is_empty();
        CoreRule { commit, words, ok }
    })
}

// ---------------------------------------------------------------------------------------
// Characters
// ---------------------------------------------------------------------------------------

/// Removed by `show` and skipped by the scanner: format characters (general category Cf),
/// variation selectors, private-use and unassigned-plane characters.
// RULING: wave2-c8 review A2: and the characters that print as nothing without being Cf:
// U+034F, the Hangul fillers U+115F, U+1160, U+3164 and U+FFA0, the Khmer inherent vowels
// U+17B4 and U+17B5, the Mongolian selectors U+180B to U+180D, and the unassigned U+2065.
const INVISIBLE: &[(u32, u32)] = &[
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x0600, 0x0605),
    (0x061C, 0x061C),
    (0x06DD, 0x06DD),
    (0x070F, 0x070F),
    (0x0890, 0x0891),
    (0x08E2, 0x08E2),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180E),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xE000, 0xF8FF),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF9, 0xFFFB),
    (0x110BD, 0x110BD),
    (0x110CD, 0x110CD),
    (0x13430, 0x1343F),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0001, 0xE0001),
    (0xE0020, 0xE007F),
    (0xE0100, 0xE01EF),
    (0xF0000, 0x10FFFF),
];

/// Line break, carriage return, tab and the line and paragraph separators become a space.
fn becomes_space(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\t' | '\u{2028}' | '\u{2029}')
}

/// Characters `show` drops without a trace. The scanner drops the same ones before it
/// matches, so `redact` sees a token exactly as `show` would otherwise reveal it (R-9a).
fn removed(c: char) -> bool {
    if becomes_space(c) {
        return false;
    }
    if c.is_ascii() {
        return c.is_ascii_control();
    }
    let n = c as u32;
    c.is_control() || INVISIBLE.iter().any(|&(lo, hi)| lo <= n && n <= hi)
}

// RULING: wave2-c8 review A5: the look-alikes the review lists are mapped as the others are
// (the first four arms). U+2329 and U+232A are added with them: U+3008 and U+3009 are
// their canonical forms, and they print the same.
fn bracket(c: char) -> char {
    match c {
        '\u{2045}' | '\u{FE5D}' | '\u{2772}' => '(',
        '\u{2046}' | '\u{FE5E}' | '\u{2773}' => ')',
        '\u{2329}' | '\u{FE64}' | '\u{27EA}' => '\u{2039}',
        '\u{232A}' | '\u{FE65}' | '\u{27EB}' => '\u{203A}',
        '[' | '\u{FF3B}' | '\u{3010}' | '\u{3014}' | '\u{3016}' | '\u{3018}' | '\u{301A}' | '\u{27E6}' => '(',
        ']' | '\u{FF3D}' | '\u{3011}' | '\u{3015}' | '\u{3017}' | '\u{3019}' | '\u{301B}' | '\u{27E7}' => ')',
        '<' | '\u{FF1C}' | '\u{3008}' | '\u{300A}' | '\u{27E8}' => '\u{2039}',
        '>' | '\u{FF1E}' | '\u{3009}' | '\u{300B}' | '\u{27E9}' => '\u{203A}',
        other => other,
    }
}

/// Steps 2 to 4 of `show` on text that holds no line break the caller wants kept.
fn clean_line(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut gap = false;
    for c in text.chars() {
        if removed(c) {
            continue;
        }
        if becomes_space(c) || c.is_whitespace() {
            gap = true;
            continue;
        }
        if gap && !out.is_empty() {
            out.push(' ');
        }
        gap = false;
        out.push(bracket(c));
    }
    out
}

/// Step 5: at most `max_chars` characters, the last one `…` when something was cut.
fn cut(text: String, max_chars: usize) -> String {
    let Some((at, _)) = text.char_indices().nth(max_chars) else {
        return text;
    };
    if max_chars == 0 {
        return String::new();
    }
    let keep = text[..at].char_indices().next_back().map_or(0, |(i, _)| i);
    let mut out = text[..keep].to_string();
    out.push('…');
    out
}

// ---------------------------------------------------------------------------------------
// The scanner's view: the text without the characters `removed` names
// ---------------------------------------------------------------------------------------

struct View<'a> {
    text: Cow<'a, str>,
    /// For each byte of an owned view, its offset in the original. Empty: the view is the text.
    map: Vec<usize>,
}

impl<'a> View<'a> {
    fn new(text: &'a str) -> Self {
        if !text.chars().any(removed) {
            return View { text: Cow::Borrowed(text), map: Vec::new() };
        }
        let mut kept = String::with_capacity(text.len());
        let mut map = Vec::with_capacity(text.len());
        for (at, c) in text.char_indices() {
            if !removed(c) {
                kept.push(c);
                map.extend(at..at + c.len_utf8());
            }
        }
        View { text: Cow::Owned(kept), map }
    }

    fn as_str(&self) -> &str {
        &self.text
    }

    /// A span of the view (never empty) as a span of the original. It includes the removed
    /// characters inside it, and none before or after.
    fn original(&self, start: usize, end: usize) -> (usize, usize) {
        if self.map.is_empty() {
            (start, end)
        } else {
            (self.map[start], self.map[end - 1] + 1)
        }
    }
}

// ---------------------------------------------------------------------------------------
// The scanner: one pass over the bytes, no regular expressions
// ---------------------------------------------------------------------------------------

struct Span {
    start: usize,
    end: usize,
    hit: Hit,
}

// RULING: screen measurement, ruling 5: `GOCSPX-` (a Google OAuth client secret) is a
// certain prefix.
const PREFIXES: [&str; 18] = [
    "sk-", "sk_live_", "sk_test_", "rk_live_", "pk_live_", "ghp_", "gho_", "ghu_", "ghs_", "ghr_",
    "github_pat_", "glpat-", "xoxb-", "xoxp-", "xoxa-", "xoxs-", "ya29.", "GOCSPX-",
];

// RULING: screen measurement, ruling 8: `secret_key`, `private_key`, `signing_key` and
// `encryption_key` are names. Not `pass`, not `pwd`, not `_key` in general.
// The first that matches at a place is the one tried: `secret_key` stands before `secret`.
const NAMES: [&str; 13] = [
    "password", "passwd", "private_key", "secret_key", "secret", "signing_key", "token", "api_key", "apikey",
    "api-key", "access_key", "authorization", "encryption_key",
];

/// `[A-Za-z0-9_-]`
fn is_word(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

/// `[A-Za-z0-9+_-]`
// RULING: screen measurement, ruling 2: `=` is not in the class. A name in front of `=` is
// not part of a token (`NAME=value` keeps `NAME=`), and its capitals do not make a one-case
// value mixed.
fn is_long(b: u8) -> bool {
    is_word(b) || b == b'+'
}

/// `[A-Za-z0-9._~+/=-]`
fn is_bearer(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"._~+/=-".contains(&b)
}

fn is_upper_digit(b: u8) -> bool {
    b.is_ascii_uppercase() || b.is_ascii_digit()
}

fn run_end(b: &[u8], from: usize, class: fn(u8) -> bool) -> usize {
    let mut i = from;
    while i < b.len() && class(b[i]) {
        i += 1;
    }
    i
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

fn skip_blank(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && (b[i] == b' ' || b[i] == b'\t') {
        i += 1;
    }
    i
}

/// What the scanner remembers so that a hostile megabyte costs one pass, not one pass per
/// candidate: the ends of the `is_word` runs it has measured, and the last place where a
/// value turned out to read as code.
#[derive(Default)]
struct Memo {
    runs: [(usize, usize); 4],
    code: Option<(usize, usize)>,
}

impl Memo {
    /// The end of the `is_word` run that holds `from` (`from` itself when there is none).
    fn word_end(&mut self, slot: usize, b: &[u8], from: usize) -> usize {
        let (start, end) = self.runs[slot];
        if start <= from && from < end {
            return end;
        }
        let end = run_end(b, from, is_word);
        self.runs[slot] = (from, end);
        end
    }
}

fn hit(rule: &'static str, name: Option<&str>) -> Hit {
    Hit { rule, name: name.map(str::to_string) }
}

/// All spans to blank, in order and not overlapping. With `certain`, only the four certain
/// rules are tried and the scan stops at the first hit.
fn scan(s: &str, certain: bool) -> Vec<Span> {
    let b = s.as_bytes();
    let mut spans = Vec::new();
    let mut memo = Memo::default();
    let mut url: Option<(usize, usize)> = None;
    let mut long_done = 0;
    let mut i = 0;
    while i < b.len() {
        if let Some((start, end)) = url {
            if i >= start {
                url = None;
            }
            if i == start {
                spans.push(Span { start, end, hit: hit("url_password", None) });
                if certain {
                    return spans;
                }
                i = end;
                continue;
            }
        }
        // The character before a match is not a letter or a digit.
        // SPEC: 4137: "a letter or a digit" is read as ASCII. A key glued to a letter of a
        // script written without spaces is blanked, not kept.
        let anchored = i == 0 || !b[i - 1].is_ascii_alphanumeric();
        // SPEC: 4144: a private key block is matched wherever its dashes start: it is never
        // "inside a word", and one glued to a letter is still a key.
        let mut found = key_block(b, i);
        if found.is_none() && anchored {
            found = prefix(b, i, &mut memo).or_else(|| jwt(b, i, &mut memo));
        }
        if found.is_none() && b[i] == b':' {
            if let Some(password) = url_password(b, i) {
                url = Some(password);
            }
        }
        if found.is_none() && !certain {
            found = named_value(s, i, &mut memo);
            if found.is_none() && anchored {
                found = bearer(b, i);
            }
            if found.is_none() && i >= long_done && is_long(b[i]) {
                long_done = run_end(b, i, is_long);
                found = long_token(b, i, long_done);
            }
        }
        match found {
            Some(span) => {
                i = span.end;
                spans.push(span);
                if certain {
                    return spans;
                }
            }
            None => i += 1,
        }
    }
    spans
}

/// From `-----BEGIN ` + anything + `PRIVATE KEY-----` to the matching `-----END … KEY-----`,
/// or to the end of the text.
// SPEC: 4144: "anything" is at most 80 bytes of the same line, and the tail of the END line
// likewise; an END line that does not close ends the block at the end of the text.
fn key_block(b: &[u8], i: usize) -> Option<Span> {
    const LABEL: usize = 80;
    if !b[i..].starts_with(b"-----BEGIN ") {
        return None;
    }
    let label = i + 11;
    let same_line = |from: usize| {
        let limit = (from + LABEL + 16).min(b.len());
        from + b[from..limit].iter().position(|&c| c == b'\n').unwrap_or(limit - from)
    };
    let head = label + find(&b[label..same_line(label)], b"PRIVATE KEY-----")? + 16;
    let end = find(&b[head..], b"-----END ")
        .map(|at| head + at + 9)
        .and_then(|tail| find(&b[tail..same_line(tail)], b"KEY-----").map(|at| tail + at + 8))
        .unwrap_or(b.len());
    Some(Span { start: i, end, hit: hit("key_block", None) })
}

fn prefix(b: &[u8], i: usize, memo: &mut Memo) -> Option<Span> {
    if !matches!(b[i], b's' | b'r' | b'p' | b'g' | b'x' | b'y' | b'A' | b'G') {
        return None;
    }
    let rest = &b[i..];
    for p in PREFIXES {
        if rest.starts_with(p.as_bytes()) {
            let from = i + p.len();
            let end = memo.word_end(0, b, from);
            if end - from >= 16 {
                return Some(Span { start: i, end, hit: hit("prefix", Some(p)) });
            }
        }
    }
    for p in ["AKIA", "ASIA"] {
        // Exactly 16: a seventeenth character of the class is another shape.
        if rest.starts_with(p.as_bytes())
            && rest.len() >= 20
            && rest[4..20].iter().all(|&c| is_upper_digit(c))
            && !rest.get(20).is_some_and(|&c| is_upper_digit(c))
        {
            return Some(Span { start: i, end: i + 20, hit: hit("prefix", Some(p)) });
        }
    }
    if rest.starts_with(b"AIza") {
        // SPEC: 4142: "followed by 35": with more than 35 the whole run is blanked, so that
        // no tail of a key is left standing.
        let end = memo.word_end(0, b, i + 4);
        if end - (i + 4) >= 35 {
            return Some(Span { start: i, end, hit: hit("prefix", Some("AIza")) });
        }
    }
    None
}

/// `eyJ`, then three runs of at least 8 of `[A-Za-z0-9_-]` separated by two dots.
fn jwt(b: &[u8], i: usize, memo: &mut Memo) -> Option<Span> {
    if !b[i..].starts_with(b"eyJ") {
        return None;
    }
    let mut from = i + 3;
    for slot in 1..=3 {
        let end = memo.word_end(slot, b, from);
        if end - from < 8 {
            return None;
        }
        if slot == 3 {
            return Some(Span { start: i, end, hit: hit("jwt", Some("eyJ")) });
        }
        if b.get(end) != Some(&b'.') {
            return None;
        }
        from = end + 1;
    }
    None
}

/// At the `:` of `<scheme>://<user>:<password>@`: the span of the password.
fn url_password(b: &[u8], i: usize) -> Option<(usize, usize)> {
    if !b[i..].starts_with(b"://") || i == 0 {
        return None;
    }
    let scheme = b[i - 1];
    if !(scheme.is_ascii_alphanumeric() || b"+.-".contains(&scheme)) {
        return None;
    }
    let from = i + 3;
    let stop = |c: &u8| c.is_ascii_whitespace() || b"/?#\"'`<>".contains(c);
    let authority = &b[from..from + b[from..].iter().position(stop).unwrap_or(b.len() - from)];
    let at = authority.iter().rposition(|&c| c == b'@')?;
    let colon = authority[..at].iter().position(|&c| c == b':')?;
    let (start, end) = (from + colon + 1, from + at);
    (end > start && &b[start..end] != BLANK.as_bytes()).then_some((start, end))
}

/// A name that ends with one of `NAMES`, then `=` or `:`, optional spaces, an optional quote
/// and an optional `Bearer `, then a value of at least 8 characters with no space. Not when
/// the value reads as code.
// SPEC: 4146: three things the row does not say. Spaces may stand before the `=` or `:`
// (its own example `next_token = self.lexer…` has them). One quote may close the name
// (`"api_key": "…"`), so that JSON is covered. A value that opened with a quote ends at
// that quote. The Hit's name is the matched ending, in lowercase, not the whole name: the
// text before it is free text and could itself be a token.
// RULING: wave2-c8 review A2: an escaped quote (`\"`, as in JSON inside a JSON string or a
// log line) is accepted wherever one plain quote is: after the name and before the value. A
// value that opened with an escaped quote ends at the next escaped quote. `Basic ` is
// skipped the way `Bearer ` is.
fn named_value(s: &str, i: usize, memo: &mut Memo) -> Option<Span> {
    let b = s.as_bytes();
    if !matches!(b[i] | 0x20, b'p' | b's' | b't' | b'a' | b'e') {
        return None;
    }
    let name = NAMES
        .into_iter()
        .find(|n| b.len() - i >= n.len() && b[i..i + n.len()].eq_ignore_ascii_case(n.as_bytes()))?;
    // A quote at `at`, plain or escaped: its length in bytes and the quote.
    let quote_at = |at: usize| match (b.get(at), b.get(at + 1)) {
        (Some(&q @ (b'"' | b'\'')), _) => Some((1, q)),
        (Some(b'\\'), Some(&q @ (b'"' | b'\''))) => Some((2, q)),
        _ => None,
    };
    let name_end = i + name.len();
    let mut j = name_end;
    if let Some((len, _)) = quote_at(j) {
        j += len;
    }
    let unquoted = j == name_end;
    j = skip_blank(b, j);
    if matches!(b.get(j), Some(b'=' | b':')) {
        j = skip_blank(b, j + 1);
    } else if !(unquoted && j > name_end && long_flag(b, i)) {
        // RULING: screen measurement, ruling 8: a long flag takes a space. When the name is
        // part of a word that starts with `--`, one or more spaces stand where `=` or `:`
        // does. Not `-p`, not `-u`.
        return None;
    }
    let (mut quote, mut escaped) = (None, false);
    if let Some((len, q)) = quote_at(j) {
        quote = Some(q as char);
        escaped = len == 2;
        j += len;
    }
    for scheme in ["bearer ", "basic "] {
        if b.len() - j >= scheme.len() && b[j..j + scheme.len()].eq_ignore_ascii_case(scheme.as_bytes()) {
            j += scheme.len();
            break;
        }
    }
    let start = j;
    if quote.is_none() && memo.code.is_some_and(|(from, at)| from <= start && start < at) {
        return None;
    }
    let (mut end, mut chars, mut digit, mut colon) = (b.len(), 0usize, false, false);
    for (off, c) in s[start..].char_indices() {
        let closes = quote == Some(c)
            || (escaped && c == '\\' && b.get(start + off + 1).is_some_and(|&next| quote == Some(next as char)));
        if c.is_whitespace() || closes {
            end = start + off;
            break;
        }
        if matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>') || (c == ':' && colon) {
            memo.code = Some((start, start + off));
            return None;
        }
        colon = c == ':';
        digit |= c.is_ascii_digit();
        chars += 1;
    }
    if chars < 8 || (!digit && chars < 16) {
        return None;
    }
    Some(Span { start, end, hit: hit("named_value", Some(name)) })
}

/// Is the name at `i` part of a word of `[A-Za-z0-9_-]` that starts with `--`?
fn long_flag(b: &[u8], i: usize) -> bool {
    let mut start = i;
    while start > 0 && is_word(b[start - 1]) {
        start -= 1;
    }
    start < i && b[start..].starts_with(b"--")
}

/// `Bearer ` followed by at least 20 of `[A-Za-z0-9._~+/=-]`: the value.
fn bearer(b: &[u8], i: usize) -> Option<Span> {
    if b.len() - i < 7 || !b[i..i + 7].eq_ignore_ascii_case(b"bearer ") {
        return None;
    }
    let start = i + 7;
    let end = run_end(b, start, is_bearer);
    (end - start >= 20).then(|| Span { start, end, hit: hit("named_value", Some("bearer")) })
}

/// A run of at least 32 of `[A-Za-z0-9+_-]` with at least 3 lowercase letters, 3 uppercase
/// letters and 2 digits, in which at least one piece mixes cases. Up to two `=` directly
/// after the run are blanked with it.
fn long_token(b: &[u8], start: usize, end: usize) -> Option<Span> {
    if end - start < 32 {
        return None;
    }
    let run = &b[start..end];
    let count = |class: fn(&u8) -> bool| run.iter().filter(|c| class(c)).count();
    // RULING: screen measurement, ruling 4: 2 digits are enough (it was 3).
    if count(u8::is_ascii_lowercase) < 3 || count(u8::is_ascii_uppercase) < 3 || count(u8::is_ascii_digit) < 2 {
        return None;
    }
    // RULING: screen measurement, ruling 3: a run in which no piece mixes cases is a slug (a
    // dated Title-Case name, a model file name), not a token. Pieces lie between `-` and
    // `_`; a piece of fewer than 4 characters is not looked at; a piece mixes cases when it
    // has a lowercase letter, and an uppercase letter after its first character.
    let mixes = |piece: &[u8]| {
        piece.len() >= 4 && piece.iter().any(u8::is_ascii_lowercase) && piece[1..].iter().any(u8::is_ascii_uppercase)
    };
    if !run.split(|&c| c == b'-' || c == b'_').any(mixes) {
        return None;
    }
    // RULING: screen measurement, ruling 2: base64 padding belongs to the token.
    let padding = b[end..].iter().take(2).take_while(|&&c| c == b'=').count();
    Some(Span { start, end: end + padding, hit: hit("long_token", None) })
}

// ---------------------------------------------------------------------------------------
// Tests (spec section 8: 1 to 3, 11, 13, 14)
// ---------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokens are joined from parts, so no key-shaped literal is in the repository.
    fn mk(parts: &[&str]) -> String {
        parts.concat()
    }

    const LOW_A: &str = "abcdefghij0123456789";
    const LOW_B: &str = "klmnopqrst9876543210";

    // Test 1.
    #[test]
    fn redact_keeps_code_prose_hashes_and_paths_whole() {
        let cases: [(&str, String); 12] = [
            ("lexer call", "next_token = self.lexer.next_token()?;".into()),
            ("string new", "password: String::new()".into()),
            ("environ", "api_key=os.environ[\"X\"]".into()),
            ("already redacted", "secret: [REDACTED]".into()),
            ("required", "password: required".into()),
            ("task-list", "the task-list for disk-usage".into()),
            ("secretary", "my secretary Nakia asked".into()),
            ("policy", "the password policy needs authorization from ops".into()),
            ("git hash", format!("commit {} landed", "9f2c1a7e04".repeat(4))),
            ("project dir", format!("dir {} here", "0a1b2c3d4e5f6789".repeat(4))),
            ("path", "/home/u/Projects/SynapsCLI/crates/agent-engine/src/tools/memory.rs".into()),
            ("archive id", "ctx-9f2c1a7e04b35d6688c1f0a2b7d94e13".into()),
        ];
        for (case, text) in &cases {
            let r = redact(text);
            assert!(r.text == *text, "case {case}: the text was changed");
            assert!(matches!(r.text, Cow::Borrowed(_)), "case {case}: the text was copied");
            assert!(r.blanked == 0 && r.hits.is_empty(), "case {case}: {} hits", r.hits.len());
        }
    }

    // Test 2.
    #[test]
    fn redact_blanks_each_shape_and_keeps_the_rest() {
        let b64 = mk(&["aGVsbG8gd29ybGQgdGhpcyBp", "cyBhIHRlc3QgMTIzNDU="]);
        assert_eq!(b64.chars().count(), 44, "the base64 case is 44 characters");
        // (case, the text between "before " and " after", what is left of it, rule, name)
        let cases: Vec<(&str, String, &str, &str, Option<&str>)> = vec![
            ("sk-", mk(&["sk", "-", LOW_A, LOW_B]), BLANK, "prefix", Some("sk-")),
            ("sk_live_", mk(&["sk_", "live_", LOW_A, "abcd"]), BLANK, "prefix", Some("sk_live_")),
            ("gho_", mk(&["gho", "_", LOW_A, LOW_B]), BLANK, "prefix", Some("gho_")),
            ("glpat-", mk(&["glpat", "-", LOW_A]), BLANK, "prefix", Some("glpat-")),
            ("xoxp-", mk(&["xoxp", "-", "1234-5678-", LOW_A]), BLANK, "prefix", Some("xoxp-")),
            ("AIza", mk(&["AI", "za", "SyA-", LOW_A, "_", "ABCDEFGHIJ"]), BLANK, "prefix", Some("AIza")),
            ("AKIA", mk(&["AKIA", "ABCDEFGH", "12345678"]), BLANK, "prefix", Some("AKIA")),
            ("GOCSPX-", mk(&["GOC", "SPX-", LOW_A, "klmnopqr"]), BLANK, "prefix", Some("GOCSPX-")),
            (
                "jwt",
                mk(&["ey", "J0eXAiOiJKV1Qi", ".", "eyJzdWIiOiIxMjM0", ".", "c2lnbmF0dXJlX3g"]),
                BLANK,
                "jwt",
                Some("eyJ"),
            ),
            ("token=", mk(&["token=", "abcdef", "123456"]), "token=(redacted)", "named_value", Some("token")),
            (
                "url",
                mk(&["postgres://user:", "hunter", "2xyz", "@host/db"]),
                "postgres://user:(redacted)@host/db",
                "url_password",
                None,
            ),
            (
                "authorization bearer",
                mk(&["Authorization: Bearer ", "abcdefghijklmno", "pqrstuvwxyz0123"]),
                "Authorization: Bearer (redacted)",
                "named_value",
                Some("authorization"),
            ),
            (
                "pem",
                mk(&[
                    "-----BEGIN RSA PRIVATE",
                    " KEY-----\nMIIEowIBAAKCAQEA\nq1w2e3r4t5y6\n-----END RSA PRIVATE",
                    " KEY-----",
                ]),
                BLANK,
                "key_block",
                None,
            ),
            ("base64", b64, BLANK, "long_token", None),
        ];
        for (case, token, left, rule, name) in &cases {
            let text = format!("words before {token} words after");
            let r = redact(&text);
            assert!(r.text == format!("words before {left} words after"), "case {case}: wrong text");
            assert!(r.blanked == 1 && r.hits.len() == 1, "case {case}: blanked {}", r.blanked);
            assert!(r.hits[0].rule == *rule, "case {case}: rule {}", r.hits[0].rule);
            assert!(r.hits[0].name.as_deref() == *name, "case {case}: wrong name");
            let again = redact(&r.text);
            assert!(again.blanked == 0, "case {case}: a second pass blanked {}", again.blanked);
        }
    }

    #[test]
    fn redact_edges_of_the_shapes() {
        let bearer = mk(&["Bearer ", LOW_A, ".", LOW_B]);
        let open_pem = mk(&["-----BEGIN OPENSSH PRIVATE", " KEY-----\nb3BlbnNzaC1rZXk\nmore"]);
        let json = mk(&["{\"api_key\": \"", "abcd1234", "efgh", "\"}"]);
        let escaped = mk(&["{\\\"password\\\": \\\"", "abcd1234", "efgh", "\\\"}"]);
        assert!(escaped.starts_with(r#"{\"password\": \"abcd"#) && escaped.ends_with(r#"efgh\"}"#));
        let escaped_name = mk(&["\\'api_key\\' = '", "abcd1234", "efgh", "'"]);
        let basic = mk(&["Authorization: Basic ", "dXNlcjpwYXNz", "d29yZA=="]);
        let basic_quoted = mk(&["\"authorization\": \"basic ", "dXNlcjpwYXNz", "d29yZA==", "\""]);
        let in_word = mk(&["task", "-", LOW_A, LOW_B]);
        let akia17 = mk(&["AKIA", "ABCDEFGH", "123456789"]);
        let two = format!("a {} b {} c", mk(&["ghp", "_", LOW_A]), mk(&["xoxb", "-", LOW_B]));
        // (case, text, expected text, blanked)
        let cases: Vec<(&str, String, String, usize)> = vec![
            ("bare bearer", format!("sent {bearer} ok"), "sent Bearer (redacted) ok".into(), 1),
            ("pem with no end", format!("key: {open_pem}"), "key: (redacted)".into(), 1),
            ("json", json, "{\"api_key\": \"(redacted)\"}".into(), 1),
            ("escaped json", escaped, r#"{\"password\": \"(redacted)\"}"#.into(), 1),
            ("escaped quote after the name", escaped_name, r"\'api_key\' = '(redacted)'".into(), 1),
            ("authorization basic", basic, "Authorization: Basic (redacted)".into(), 1),
            ("basic in quotes", basic_quoted, "\"authorization\": \"basic (redacted)\"".into(), 1),
            ("basic and a short value", "authorization: basic abc123".into(), "authorization: basic abc123".into(), 0),
            ("prefix inside a word", in_word.clone(), in_word, 0),
            ("AKIA and 17", akia17.clone(), akia17, 0),
            ("url without password", "https://example.com:8080/a@b".into(), "https://example.com:8080/a@b".into(), 0),
            ("ssh url", "ssh://git@host:22/repo".into(), "ssh://git@host:22/repo".into(), 0),
            ("two tokens", two, "a (redacted) b (redacted) c".into(), 2),
            ("short value", "token=abc123".into(), "token=abc123".into(), 0),
            ("empty", String::new(), String::new(), 0),
        ];
        for (case, text, expected, blanked) in &cases {
            let r = redact(text);
            assert!(r.text == *expected, "case {case}: wrong text");
            assert!(r.blanked == *blanked && r.hits.len() == *blanked, "case {case}: blanked {}", r.blanked);
        }
    }

    // RULING: screen measurement, rulings 2, 3 and 4. Every token is joined from halves.
    #[test]
    fn long_token_takes_keys_and_leaves_names_slugs_and_ids() {
        let hex = "0a1b2c3d4e5f6789";
        let slug = mk(&["2025-11-03-Axel-V2", "-Design-Review-Notes"]);
        let model = mk(&["Gemma-12B-A1B-", "Instructions-Q4_K_M"]);
        assert_eq!(model.len(), 33, "the model file name is 33 characters");
        let tool_id = mk(&["call=toolu_01", "Xk7Pq2Wm9Zr", "4Tn6Bv8Hs3L"]);
        assert_eq!(tool_id.len() - 5, 30, "the tool-use id is 30 characters after the =");
        let one_digit = mk(&["AbCdEfGhIjKlMnOp", "QrStUvWxYzAbCdE7"]);
        assert_eq!(one_digit.len(), 32);
        let kept: Vec<(&str, String)> = vec![
            ("dated title-case name", slug),
            ("model file name", model),
            ("name and 24 hex", format!("SOME_NAME={}", &hex.repeat(2)[..24])),
            ("sha and 40 hex", format!("SHA={}", &hex.repeat(3)[..40])),
            ("sha and 64 hex", format!("SHA={}", hex.repeat(4))),
            ("tool-use id", tool_id),
            ("32 with one digit", one_digit),
        ];
        for (case, text) in &kept {
            let r = redact(text);
            assert!(r.blanked == 0 && r.text == *text, "case {case}: blanked {}", r.blanked);
        }

        let mixed = mk(&["Xk7Pq2Wm9Zr4-Tn6Bv8", "Hs3Lc5_Jd1Fg0Qy8Ue2Io6Ab"]);
        assert_eq!(mixed.len(), 43);
        assert!(mixed.contains('-') && mixed.contains('_'));
        let padded = mk(&["aGVsbG8gd29ybGQgdGhpcyBp", "cyBhIHRlc3QgMTIzNDU="]);
        assert!(padded.len() == 44 && padded.ends_with('='));
        let two_digits = mk(&["AbCdEfGhIjKlMnOp", "QrStUvWxYzAb4dE7"]);
        assert_eq!(two_digits.len(), 32);
        assert_eq!(two_digits.bytes().filter(u8::is_ascii_digit).count(), 2);
        // (case, text, what is left)
        let blanked: Vec<(&str, String, String)> = vec![
            ("43 mixed", mixed.clone(), BLANK.into()),
            ("name and 43 mixed", format!("state={mixed}"), format!("state={BLANK}")),
            ("44 base64 with padding", format!("a {padded} b"), format!("a {BLANK} b")),
            ("two = of padding, not three", format!("{mixed}==="), format!("{BLANK}=")),
            ("32 with two digits", two_digits, BLANK.into()),
        ];
        for (case, text, left) in &blanked {
            let r = redact(text);
            assert!(r.text == *left, "case {case}: wrong text");
            assert!(r.blanked == 1 && r.hits[0].rule == "long_token", "case {case}: blanked {}", r.blanked);
            assert!(r.hits[0].name.is_none(), "case {case}: a name");
        }
    }

    // RULING: screen measurement, rulings 2 to 4: the rule in numbers, so that a later change
    // cannot quietly open a hole. Random base64url tokens; the foreman's own run of the rule
    // gave 99.4% at 43 characters and 97.0% at 32.
    #[test]
    fn long_token_takes_nearly_every_random_base64url_token() {
        const LETTERS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        const TOKENS: usize = 20_000;
        // SplitMix64: fixed seed, the same tokens on every run.
        let mut state = 0x5EED_0008_u64;
        let mut next = move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for (len, at_least) in [(43usize, 98.5f64), (32, 95.0)] {
            let mut taken = 0usize;
            for _ in 0..TOKENS {
                let token: String = (0..len).map(|_| LETTERS[(next() >> 58) as usize] as char).collect();
                taken += usize::from(redact(&token).text == BLANK);
            }
            let percent = taken as f64 * 100.0 / TOKENS as f64;
            println!("long_token numbers: {len} characters: {taken} of {TOKENS} blanked, {percent:.2}%");
            assert!(percent >= at_least, "{len} characters: {percent:.2}% blanked, under {at_least}%");
        }
    }

    // RULING: screen measurement, rulings 5 and 8.
    #[test]
    fn a_new_prefix_four_more_names_and_a_long_flag() {
        let google = mk(&["GOC", "SPX-", LOW_A, "klmnopqr"]);
        let certain = certain_hit(&format!("client secret {google} here"));
        assert!(certain == Some(hit("prefix", Some("GOCSPX-"))), "case GOCSPX-: certain_hit");
        let short = mk(&["GOC", "SPX-", "abcdefghij01234"]);
        assert!(certain_hit(&short).is_none() && redact(&short).blanked == 0, "case GOCSPX- and 15: a hit");
        let glued = format!("x{google}");
        assert!(certain_hit(&glued).is_none(), "case GOCSPX- inside a word: a certain hit");

        let value = mk(&["abcd1234", "efgh"]);
        let hex16 = "0a1b2c3d4e5f6789";
        // (case, text, what is left, the hit's name or None for no hit)
        let cases: Vec<(&str, String, String, Option<&str>)> = vec![
            ("secret_key", format!("secret_key={value}"), format!("secret_key={BLANK}"), Some("secret_key")),
            ("AWS_SECRET_KEY", format!("AWS_SECRET_KEY: {value}"), format!("AWS_SECRET_KEY: {BLANK}"), Some("secret_key")),
            ("private_key", format!("private_key = \"{value}\""), format!("private_key = \"{BLANK}\""), Some("private_key")),
            ("signing_key", format!("jwt_signing_key:{value}"), format!("jwt_signing_key:{BLANK}"), Some("signing_key")),
            ("ENCRYPTION_KEY", format!("ENCRYPTION_KEY={value}"), format!("ENCRYPTION_KEY={BLANK}"), Some("encryption_key")),
            ("secret still", format!("secret={value}"), format!("secret={BLANK}"), Some("secret")),
            ("project_key", "project_key = p429dc9cabb570899".into(), "project_key = p429dc9cabb570899".into(), None),
            ("turn_key", format!("turn_key: {hex16}"), format!("turn_key: {hex16}"), None),
            ("--password", "run --password hunter2abc now".into(), format!("run --password {BLANK} now"), Some("password")),
            ("--db-password", "--db-password   hunter2abc".into(), format!("--db-password   {BLANK}"), Some("password")),
            ("--api-key", format!("--api-key {value}"), format!("--api-key {BLANK}"), Some("api-key")),
            ("--password=", "--password=hunter2abc".into(), format!("--password={BLANK}"), Some("password")),
            ("--token and a variable", "--token $GH_TOKEN".into(), "--token $GH_TOKEN".into(), None),
            ("--api-key and code", "--api-key \"$(cat f)\"".into(), "--api-key \"$(cat f)\"".into(), None),
            ("cargo -p", "cargo test -p some-crate-name-2".into(), "cargo test -p some-crate-name-2".into(), None),
            ("mkdir -p", "mkdir -p /a/b/c12345678".into(), "mkdir -p /a/b/c12345678".into(), None),
            ("no dashes", "password hunter2abc".into(), "password hunter2abc".into(), None),
            ("one dash", "-password hunter2abc".into(), "-password hunter2abc".into(), None),
            ("dashes not at the start of the word", "a--password hunter2abc".into(), "a--password hunter2abc".into(), None),
            ("--token-budget", "--token-budget 100000000".into(), "--token-budget 100000000".into(), None),
            ("flag and a line break", "--password\nhunter2abc".into(), "--password\nhunter2abc".into(), None),
        ];
        for (case, text, left, name) in &cases {
            let r = redact(text);
            assert!(r.text == *left, "case {case}: wrong text");
            match name {
                Some(name) => {
                    assert!(r.blanked == 1 && r.hits[0].rule == "named_value", "case {case}: blanked {}", r.blanked);
                    assert!(r.hits[0].name.as_deref() == Some(*name), "case {case}: wrong name");
                }
                None => assert!(r.blanked == 0, "case {case}: blanked {}", r.blanked),
            }
        }
    }

    #[test]
    fn redact_spans_are_what_redact_blanks() {
        let token = format!("{}{LOW_A}\u{200B}{LOW_B}", mk(&["sk", "-"]));
        let value = mk(&["abcdef", "123456"]);
        let text = format!("é before {token} and token={value} after é");
        let spans = redact_spans(&text);
        let r = redact(&text);
        assert_eq!(spans.len(), 2);
        assert!(text[spans[0].0.clone()] == token, "case prefix: wrong range");
        assert!(text[spans[1].0.clone()] == value, "case named value: wrong range");
        let hits: Vec<Hit> = spans.iter().map(|(_, h)| h.clone()).collect();
        assert_eq!(hits, r.hits);
        let mut rebuilt = text.clone();
        for (range, _) in spans.iter().rev() {
            rebuilt.replace_range(range.clone(), BLANK);
        }
        assert!(rebuilt == r.text, "the ranges do not rebuild what redact returns");
        assert!(redact_spans("nothing to blank here").is_empty());
    }

    // Test 3.
    #[test]
    fn show_makes_one_harmless_line() {
        assert_eq!(show("one\ntwo\r\n\tthree  four\u{2028}five", 200), "one two three four five");

        let invisible = "a\u{202E}b\u{200B}c\u{2060}d\u{00AD}e\u{061C}f\u{2028}g\u{E0041}h";
        assert_eq!(show(invisible, 200), "abcdef gh");

        let event = show("<event id=\"1\">", 200);
        assert!(!event.contains('<') && !event.contains('>'), "event: {event}");
        assert_eq!(event, "\u{2039}event id=\"1\"\u{203A}");

        let markers = [
            ("plain", "[End extension context]", "(End extension context)"),
            ("two spaces", "[End  extension context]", "(End extension context)"),
            ("line break", "[End\nextension context]", "(End extension context)"),
            ("fullwidth", "\u{FF3B}End Axel recall\u{FF3D}", "(End Axel recall)"),
        ];
        for (case, text, expected) in markers {
            let shown = show(text, 200);
            assert!(!shown.contains(['[', ']', '\u{FF3B}', '\u{FF3D}']), "case {case}: {shown}");
            assert_eq!(shown, expected, "case {case}");
        }

        let every = "[\u{FF3B}\u{3010}\u{3014}\u{3016}\u{3018}\u{301A}\u{27E6}]\u{FF3D}\u{3011}\u{3015}\u{3017}\u{3019}\u{301B}\u{27E7}<\u{FF1C}\u{3008}\u{300A}\u{27E8}>\u{FF1E}\u{3009}\u{300B}\u{27E9}";
        assert_eq!(show(every, 200), format!("{}{}{}{}", "(".repeat(8), ")".repeat(8), "‹".repeat(5), "›".repeat(5)));

        // RULING: wave2-c8 review A5.
        let alike = [
            ('\u{2045}', '('),
            ('\u{FE5D}', '('),
            ('\u{2772}', '('),
            ('\u{2046}', ')'),
            ('\u{FE5E}', ')'),
            ('\u{2773}', ')'),
            ('\u{2329}', '\u{2039}'),
            ('\u{3008}', '\u{2039}'),
            ('\u{FE64}', '\u{2039}'),
            ('\u{27EA}', '\u{2039}'),
            ('\u{232A}', '\u{203A}'),
            ('\u{3009}', '\u{203A}'),
            ('\u{FE65}', '\u{203A}'),
            ('\u{27EB}', '\u{203A}'),
        ];
        for (from, to) in alike {
            let case = format!("U+{:04X}", from as u32);
            assert_eq!(show(&format!("a{from}b"), 200), format!("a{to}b"), "case {case}");
            assert_eq!(show_block(&format!("a\n{from}b"), 200), format!("a\n{to}b"), "case {case}: show_block");
        }
        assert_eq!(
            show("\u{2045}End extension context\u{2046} \u{27EA}event\u{27EB}", 200),
            "(End extension context) \u{2039}event\u{203A}"
        );

        let long = "é√ words ".repeat(111) + "é";
        assert_eq!(long.chars().count(), 1000);
        let shown = show(&long, 100);
        assert_eq!(shown.chars().count(), 100);
        assert!(shown.ends_with('…'));
        assert!(long.starts_with(shown.trim_end_matches('…')));
        assert_eq!(show(&long, 1000), long);
        assert_eq!(show("abc", 0), "");
        assert_eq!(show("abcd", 3), "ab…");

        let token = mk(&["sk", "-", LOW_A, LOW_B]);
        assert_eq!(show(&format!("key\n{token}\nend"), 200), "key (redacted) end");
    }

    // Test 11.
    #[test]
    fn invisible_characters_do_not_hide_a_key() {
        for (case, invisible) in [("zero-width space", '\u{200B}'), ("soft hyphen", '\u{00AD}')] {
            let token = format!("{}{LOW_A}{invisible}{LOW_B}", mk(&["sk", "-"]));
            let text = format!("words before {token} words after");
            let r = redact(&text);
            assert!(r.text == "words before (redacted) words after", "case {case}: wrong text");
            assert!(r.blanked == 1 && r.hits.len() == 1, "case {case}: blanked {}", r.blanked);
            assert!(r.hits[0].rule == "prefix", "case {case}: rule {}", r.hits[0].rule);
            assert!(r.hits[0].name.as_deref() == Some("sk-"), "case {case}: wrong name");
            for h in &r.hits {
                let name = h.name.as_deref().unwrap_or("");
                assert!(!name.contains(LOW_A) && !name.contains(LOW_B), "case {case}: a Hit holds the value");
                assert!(name.len() <= 16, "case {case}: a Hit's name is {} bytes", name.len());
            }
            assert!(certain_hit(&text) == Some(hit("prefix", Some("sk-"))), "case {case}: certain_hit");
        }
        // RULING: wave2-c8 review A2: the invisible characters the table lacked. A key split
        // by one is blanked, and `show` does not put a key back together.
        let added = [
            '\u{034F}', '\u{115F}', '\u{1160}', '\u{3164}', '\u{FFA0}', '\u{17B4}', '\u{17B5}', '\u{180B}',
            '\u{180C}', '\u{180D}', '\u{2065}',
        ];
        for invisible in added {
            let case = format!("U+{:04X}", invisible as u32);
            let token = format!("{}{LOW_A}{invisible}{LOW_B}", mk(&["sk", "-"]));
            let text = format!("words before {token} words after");
            let r = redact(&text);
            assert!(r.text == "words before (redacted) words after", "case {case}: redact left the key");
            assert!(r.blanked == 1 && r.hits[0].rule == "prefix", "case {case}: blanked {}", r.blanked);
            assert!(certain_hit(&text) == Some(hit("prefix", Some("sk-"))), "case {case}: certain_hit");
            for shown in [show(&text, 200), show_block(&text, 200)] {
                assert!(shown == "words before (redacted) words after", "case {case}: show printed the key");
            }
            // Where there is no key, `show` still removes the character and joins nothing new.
            assert!(show(&format!("ab{invisible}cd"), 200) == "abcd", "case {case}: show kept it");
            let halves = format!("{}{LOW_A}{invisible}\n{LOW_B}", mk(&["sk", "-"]));
            let shown = show(&halves, 200);
            assert!(!shown.contains(&format!("{LOW_A}{LOW_B}")), "case {case}: show joined two lines into a key");
        }
        // Outside a match nothing is removed by `redact`.
        let kept = "soft\u{00AD}hyphen and zero\u{200B}width";
        assert!(matches!(redact(kept).text, Cow::Borrowed(t) if t == kept));
    }

    // Test 13.
    #[test]
    fn show_block_keeps_lines_and_nothing_else() {
        let token = mk(&["sk", "-", LOW_A, LOW_B]);
        let text = format!("line one\r\nline \u{1b}[2Ktwo {token}\n\u{202E}line three");
        let shown = show_block(&text, 500);
        let lines: Vec<&str> = shown.split('\n').collect();
        assert_eq!(lines, ["line one", "line (2Ktwo (redacted)", "line three"]);
        assert!(!shown.contains(['\r', '\u{1b}', '\u{202E}', '[']));
        assert!(!shown.contains(LOW_A) && !shown.contains(LOW_B), "the token is still there");

        assert_eq!(show_block("a\n\nb", 500), "a\n\nb");
        assert_eq!(show_block("abc\ndef", 5), "abc\n…");

        assert!(certain_hit("password: String::new()").is_none(), "case string new");
        assert!(certain_hit("token=abcdefgh1234").is_none(), "case token=");
        let jwt = mk(&["ey", "J0eXAiOiJKV1Qi", ".", "eyJzdWIiOiIxMjM0", ".", "c2lnbmF0dXJlX3g"]);
        let found = certain_hit(&format!("token={jwt}")).expect("case jwt: no hit");
        assert_eq!(found.rule, "jwt");
        let url = mk(&["see https://bob:", "pw", "@example.com/x"]);
        assert_eq!(certain_hit(&url).map(|h| h.rule), Some("url_password"));
    }

    // Test 14, and the strings core's own tests withhold with this rule
    // (context_archive.rs at a373ff06: "password=bad", "password=synthetic-credential",
    // "the password is example-only"). Core has no direct test of `credential_like`.
    #[test]
    fn core_withholds_is_cores_rule() {
        for text in ["my Password is hunter2", "the task-list", "my secretary", "Bearer x"] {
            assert!(core_withholds(text), "case {text:?} is withheld by core");
        }
        for text in ["hello there", "bearer", "", "pass word", "s k-", "api key"] {
            assert!(!core_withholds(text), "case {text:?} is not withheld by core");
        }
        for text in ["password=bad", "password=synthetic-credential", "the password is example-only"] {
            assert!(core_withholds(text), "core's case {text:?}");
        }
        // The case rule is ASCII lowercase, the match is a substring, one case per word.
        let one_each = [
            "PASSWORD", "Passwd", "API_KEY", "ApiKey", "api-KEY", "ACCESS_TOKEN", "Refresh_Token", "SECRET",
            "Authorization", "BEARER x", "PRIVATE KEY", "disk-usage", "GHP_", "GitHub_PAT_", "XOXB-", "Nakia",
            "EYJH",
        ];
        for (text, word) in one_each.iter().zip(core_rule_words()) {
            assert!(core_withholds(&format!("x{text}y")), "case {text:?} for word {word:?}");
            assert!(text.to_ascii_lowercase().contains(word.as_str()), "case {text:?} is not word {word:?}");
        }
        // ASCII lowercase only: the Kelvin sign is not a "k".
        assert!(!core_withholds("S\u{212A}-"), "case kelvin");

        let words = core_rule_words();
        assert_eq!(words.len(), 17);
        for word in words {
            assert!(!word.is_empty(), "an empty word");
            assert!(*word == word.to_lowercase(), "word {word:?} is not lowercase");
        }
        assert_eq!(words[9], "bearer ");

        let first = include_str!("core_withheld.txt").lines().next().unwrap();
        let commit = first.strip_prefix("# core ").expect("the first line names a core commit");
        assert!(commit.len() >= 7 && commit.bytes().all(|b| b.is_ascii_hexdigit()), "commit {commit:?}");
        assert_eq!(core_rule_commit(), commit);
        // RULING: wave2-c8 review A4: the copy carries the hash of core's whole function.
        let hashes: Vec<&str> =
            include_str!("core_withheld.txt").lines().filter_map(|l| l.strip_prefix("# sha256 ")).collect();
        assert_eq!(hashes.len(), 1, "one sha256 line");
        let hash = hashes[0];
        assert!(hash.len() == 64 && hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')), "hash {hash:?}");
        assert!(hash.bytes().any(|b| b != b'0'), "the hash is a placeholder");
    }

    // The spec gives no size or time bound; this one only catches a scan that is quadratic.
    #[test]
    fn redact_is_one_pass_on_a_megabyte() {
        let fill = |unit: &str, tail: &str| unit.repeat((1 << 20) / unit.len()) + tail;
        let cases = [
            ("prose", fill("the quick brown fox jumps over the lazy dog. ", ""), 0),
            ("names before code", fill("token=", "("), 0),
            ("jwt heads", fill("eyJaaaaaaaa-", ""), 0),
            ("jwt heads, two runs", fill("eyJaaaaaaaa-", ".bbbbbbbb"), 0),
            ("key block heads", fill("-----BEGIN ", ""), 0),
            ("url heads", fill("a://b:", ""), 0),
            ("prefix heads", fill("sk-a ", ""), 0),
            ("one long name chain", fill("token=", ""), 1),
            ("long flags", fill("--token ", ""), 0),
            ("one long flag word", fill("-token", " hunter2abc"), 0),
        ];
        for (case, text, blanked) in &cases {
            let started = std::time::Instant::now();
            let r = redact(text);
            let took = started.elapsed();
            assert!(r.blanked == *blanked, "case {case}: blanked {}", r.blanked);
            assert!(took.as_secs() < 5, "case {case}: {took:?} for {} bytes", text.len());
        }
    }
}
