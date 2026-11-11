//! Archive file reader, the carried head, classification, unit extraction,
//! `forgotten_keys` (pure functions, no database). Commit 6 writes this file.
//!
//! The file format is Synaps core's (`docs/extensions/context-archives.md`); what the
//! extension does with a record is section 2.2 of the spec.

use std::fs;
use std::io::Read;
use std::path::Path;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::screen;

/// A version 1 file is at most 16 MiB.
const MAX_BYTES: u64 = 16 * 1024 * 1024;
/// A version 1 archive holds at most 4,096 messages.
const MAX_MESSAGES: u64 = 4096;
/// A version 1 note is at most 8,192 bytes.
const MAX_NOTE_BYTES: usize = 8192;
const BODY_BYTES: usize = 4096;
const AUX_BYTES: usize = 2000;
const INPUT_HEAD_CHARS: usize = 160;
const MIN_CHARS: usize = 60;
const TITLE_WORDS: usize = 12;
const WITHHELD: &str = "[archive: withheld]";
const CONTINUATION: &str = "[Context continuation";

#[derive(Clone, Debug)]
pub enum ArchiveFile {
    Live(LiveArchive),
    Tombstone { id: String, logical: String, digest: String },
}

#[derive(Clone, Debug)]
pub struct LiveArchive {
    pub id: String,
    pub scope: String,
    pub logical: String,
    pub digest: String,
    pub created_ms: Option<i64>,
    pub source_message_count: u32,
    pub note: String,
    pub messages: Vec<ArchivedMessage>,
    pub file_size: u64,
    /// 4.1: the record's own top-level `carried`, when core wrote it.
    pub carried: Option<u32>,
    /// 4.1: what an event said about this file; `read_archive` sets `Unknown`, the indexer
    /// fills it.
    pub start: Start,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Start {
    #[default]
    Unknown,
    Whole,
    Carried,
}

impl Start {
    /// `part_no > 1` → `Whole`; `part_no = 1` and `win_no > 2` → `Carried`; anything else →
    /// `Unknown`.
    // SPEC: 1442 says `part_no > 1` → `Whole` and 2275 says `(None, Some(2))` is `Unknown`:
    // an event that does not give both numbers decides nothing.
    pub fn from_event(win_no: Option<i64>, part_no: Option<i64>) -> Start {
        match (win_no, part_no) {
            (Some(_), Some(p)) if p > 1 => Start::Whole,
            (Some(w), Some(1)) if w > 2 => Start::Carried,
            _ => Start::Unknown,
        }
    }
}

/// The role of an archived message; not `brain_v2::Role`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Clone, Debug)]
pub struct ArchivedMessage {
    pub source_index: u32,
    pub role: Role,
    pub blocks: Vec<Block>,
}

#[derive(Clone, Debug)]
pub enum Block {
    Text(String),
    ToolUse { name: String, input_head: String },
    ToolResult { is_error: bool },
    Other,
}

#[derive(Clone, Debug)]
pub struct ArchiveEntry {
    pub rel_name: String,
    pub id_from_name: Option<String>,
    pub size: u64,
}

/// Id letters x, c, r, n.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnitKind {
    Exchange,
    Continued,
    Report,
    Note,
}

#[derive(Clone, Debug)]
pub struct Unit {
    pub kind: UnitKind,
    pub source_index: u32,
    pub block_index: u32,
    pub title: String,
    pub body: String,
    pub aux: String,
    pub turn_key: Option<String>,
}

/// The text stored in `archive_index.skip_reason`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkipReason(pub String);

#[derive(Clone, Debug)]
pub struct ArchivePolicy {
    pub event_sources: Vec<String>,
    /// `["axel_forget", "memory_forget"]`.
    pub forget_tools: Vec<String>,
    /// `restore --forget-turns`: rule 9 and the marks are off for this file.
    pub keep_forget_turns: bool,
}

impl ArchivePolicy {
    /// An entry, or an entry + '_' + digits.
    pub fn is_forget_tool(&self, name: &str) -> bool {
        self.forget_tools.iter().any(|tool| match name.strip_prefix(tool.as_str()) {
            Some("") => true,
            Some(rest) => match rest.strip_prefix('_') {
                Some(digits) => !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()),
                None => false,
            },
            None => false,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForgetMark {
    pub turn_key: Option<String>,
    pub head: bool,
    pub tail: bool,
}

// ---------------------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------------------

/// `000.json` to `127.json`.
fn is_slot_name(name: &str) -> bool {
    match name.strip_suffix(".json") {
        Some(n) if n.len() == 3 && n.bytes().all(|b| b.is_ascii_digit()) => {
            n.parse::<u32>().is_ok_and(|v| v <= 127)
        }
        _ => false,
    }
}

/// `s` plus exactly six digits with a value from 1 to 999,999.
fn is_shelf_name(name: &str) -> bool {
    match name.strip_prefix('s') {
        Some(n) if n.len() == 6 && n.bytes().all(|b| b.is_ascii_digit()) => {
            n.parse::<u32>().is_ok_and(|v| v >= 1)
        }
        _ => false,
    }
}

fn is_id(s: &str) -> bool {
    s.len() == 32 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The id of a file named `<32 hex>.json`.
fn id_of_name(name: &str) -> Option<&str> {
    name.strip_suffix(".json").filter(|id| is_id(id))
}

/// The names in a directory, sorted. A link, a file or nothing has none.
fn listing(dir: &Path) -> Vec<String> {
    match fs::symlink_metadata(dir) {
        Ok(meta) if meta.file_type().is_dir() => {}
        _ => return Vec::new(),
    }
    let mut names: Vec<String> = match fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.file_name().into_string().ok())
            .collect(),
        Err(_) => Vec::new(),
    };
    names.sort();
    names
}

/// `lstat` only; opens nothing. Slot files, then per shelf in numeric order the archive
/// files, each with its `lstat` size. An entry with an archive's name that is not a regular
/// file is listed too: `read_archive` refuses it with its reason.
pub fn list_project(dir: &Path) -> Vec<ArchiveEntry> {
    let names = listing(dir);
    let mut out = Vec::new();
    for name in names.iter().filter(|n| is_slot_name(n)) {
        if let Ok(meta) = fs::symlink_metadata(dir.join(name)) {
            out.push(ArchiveEntry { rel_name: name.clone(), id_from_name: None, size: meta.len() });
        }
    }
    // Six digits each, so the sorted order is the numeric order.
    for shelf in names.iter().filter(|n| is_shelf_name(n)) {
        let shelf_dir = dir.join(shelf);
        for name in listing(&shelf_dir) {
            let Some(id) = id_of_name(&name) else { continue };
            if let Ok(meta) = fs::symlink_metadata(shelf_dir.join(&name)) {
                out.push(ArchiveEntry {
                    rel_name: format!("{shelf}/{name}"),
                    id_from_name: Some(id.to_string()),
                    size: meta.len(),
                });
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// Reading one file
// ---------------------------------------------------------------------------------------

fn skip<T>(reason: impl Into<String>) -> Result<T, SkipReason> {
    Err(SkipReason(reason.into()))
}

fn dir_name(dir: Option<&Path>) -> Option<&str> {
    dir.and_then(|d| d.file_name()).and_then(|n| n.to_str())
}

#[cfg(unix)]
fn same_file(a: &fs::Metadata, b: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    a.dev() == b.dev() && a.ino() == b.ino()
}

#[cfg(not(unix))]
fn same_file(_a: &fs::Metadata, _b: &fs::Metadata) -> bool {
    true
}

/// Reads one archive file whole, or refuses it with a reason; never partly. `path` is
/// `<project dir>/NNN.json` or `<project dir>/sNNNNNN/<32 hex>.json`, and the project
/// directory's name is `scope_dir`. It never follows a link and never writes.
pub fn read_archive(
    path: &Path,
    scope_dir: &str,
    expect_id: Option<&str>,
) -> Result<ArchiveFile, SkipReason> {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return skip("not an archive name");
    };
    let parent = path.parent();
    let (name_id, project) = if is_slot_name(name) {
        (None, parent)
    } else if let Some(id) = id_of_name(name) {
        if !dir_name(parent).is_some_and(is_shelf_name) {
            return skip("not an archive name: not in a shelf");
        }
        match parent.map(fs::symlink_metadata) {
            Some(Ok(meta)) if meta.file_type().is_dir() => {}
            _ => return skip("the shelf is not a directory"),
        }
        (Some(id), parent.and_then(|p| p.parent()))
    } else {
        return skip("not an archive name");
    };
    if dir_name(project) != Some(scope_dir) {
        return skip("not in the project directory");
    }

    let listed = match fs::symlink_metadata(path) {
        Ok(meta) => meta,
        Err(e) => return skip(format!("cannot stat the file: {:?}", e.kind())),
    };
    if !listed.file_type().is_file() {
        return skip("not a regular file");
    }
    if listed.len() > MAX_BYTES {
        return skip("larger than 16 MiB");
    }
    let mut bytes = Vec::with_capacity(listed.len() as usize + 1);
    {
        let file = match fs::File::open(path) {
            Ok(file) => file,
            Err(e) => return skip(format!("cannot open the file: {:?}", e.kind())),
        };
        // The name can have been replaced between the lstat and the open.
        match file.metadata() {
            Ok(meta) if meta.file_type().is_file() && same_file(&listed, &meta) => {}
            _ => return skip("not a regular file"),
        }
        if let Err(e) = file.take(MAX_BYTES + 1).read_to_end(&mut bytes) {
            return skip(format!("cannot read the file: {:?}", e.kind()));
        }
    }
    if bytes.len() as u64 > MAX_BYTES {
        return skip("larger than 16 MiB");
    }
    let record: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(_) => return skip("not valid JSON"),
    };
    parse_record(&record, scope_dir, name_id, expect_id, bytes.len() as u64)
}

fn str_field<'a>(obj: &'a Map<String, Value>, key: &str) -> Result<&'a str, SkipReason> {
    match obj.get(key).and_then(Value::as_str) {
        Some(s) => Ok(s),
        None => skip(format!("{key} is not a string")),
    }
}

fn parse_record(
    record: &Value,
    scope_dir: &str,
    name_id: Option<&str>,
    expect_id: Option<&str>,
    file_size: u64,
) -> Result<ArchiveFile, SkipReason> {
    let Some(obj) = record.as_object() else { return skip("not a record") };
    // The version first: a float, a string or nothing is not a record.
    match obj.get("version") {
        Some(Value::Number(n)) if n.is_i64() || n.is_u64() => {
            if n.as_u64() != Some(1) {
                return skip(format!("version {n} is not 1"));
            }
        }
        _ => return skip("not a record: no integer version"),
    }
    if str_field(obj, "scope")? != scope_dir {
        return skip("a record of another project");
    }
    let id = str_field(obj, "id")?;
    if !is_id(id) {
        return skip("the id is not 32 hex characters");
    }
    if name_id.is_some_and(|n| n != id) {
        return skip("the id is not the file's name");
    }
    if expect_id.is_some_and(|e| e != id) {
        return skip("the id is not the expected id");
    }
    let Some(body) = obj.get("body").and_then(Value::as_object) else {
        return skip("body is not an object");
    };
    match body.get("state").and_then(Value::as_str) {
        Some("tombstone") => {
            // RULING: wave2-c6-review finding 1: a file that passed the version, scope and id
            // checks and says `tombstone` is a forget, whatever else it holds (`created_ms`,
            // other top-level fields, other `body` fields): core treats exactly such a file
            // as forgotten. Nothing else is read from it; a `logical` or `digest` that is
            // not a string is the empty string, which names no archive.
            let text = |key: &str| obj.get(key).and_then(Value::as_str).unwrap_or("").to_string();
            Ok(ArchiveFile::Tombstone {
                id: id.to_string(),
                logical: text("logical"),
                digest: text("digest"),
            })
        }
        Some("live") => {
            let logical = str_field(obj, "logical")?;
            let digest = str_field(obj, "digest")?;
            // SPEC: 1482 reads a `created_ms` that is not an integer as `None`; core's page
            // (the authority for the format) says such a file is not a record. Refused.
            let created_ms = match obj.get("created_ms") {
                None => None,
                Some(v) => match v.as_u64().and_then(|ms| i64::try_from(ms).ok()) {
                    Some(ms) => Some(ms),
                    None => return skip("created_ms is not a whole number of milliseconds"),
                },
            };
            let count = match body.get("source_message_count").and_then(Value::as_u64) {
                Some(n) if n <= MAX_MESSAGES => n,
                _ => return skip("source_message_count is not an integer from 0 to 4096"),
            };
            let note = str_field(body, "note")?;
            if note.len() > MAX_NOTE_BYTES {
                return skip("the note is larger than 8192 bytes");
            }
            let Some(entries) = body.get("messages").and_then(Value::as_array) else {
                return skip("messages is not an array");
            };
            let mut messages = Vec::with_capacity(entries.len());
            let mut next_index = 0u64;
            for entry in entries {
                let message = parse_entry(entry, next_index, count)?;
                next_index = u64::from(message.source_index) + 1;
                messages.push(message);
            }
            // A `carried` that is not a non-negative integer, or is larger than the number
            // of stored messages, is read as having none.
            let carried = obj
                .get("carried")
                .and_then(Value::as_u64)
                .filter(|n| *n <= messages.len() as u64)
                .map(|n| n as u32);
            Ok(ArchiveFile::Live(LiveArchive {
                id: id.to_string(),
                scope: scope_dir.to_string(),
                logical: logical.to_string(),
                digest: digest.to_string(),
                created_ms,
                source_message_count: count as u32,
                note: note.to_string(),
                messages,
                file_size,
                carried,
                start: Start::Unknown,
            }))
        }
        _ => skip("unknown state"),
    }
}

/// One entry of `messages`. `source_index` is strictly increasing and below the count.
fn parse_entry(entry: &Value, at_least: u64, count: u64) -> Result<ArchivedMessage, SkipReason> {
    let Some(entry) = entry.as_object() else { return skip("a message entry is not an object") };
    let source_index = match entry.get("source_index").and_then(Value::as_u64) {
        Some(i) if i >= at_least && i < count => i as u32,
        _ => return skip("a source_index is out of order or not below source_message_count"),
    };
    match entry.get("block_indices").and_then(Value::as_array) {
        Some(indices) if indices.iter().all(Value::is_u64) => {}
        _ => return skip("block_indices is not an array of integers"),
    }
    let Some(message) = entry.get("message").and_then(Value::as_object) else {
        return skip("a message is not an object");
    };
    let role = match message.get("role").and_then(Value::as_str) {
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        _ => return skip("a message is neither user nor assistant"),
    };
    let blocks = match message.get("content") {
        Some(Value::String(text)) => vec![Block::Text(text.clone())],
        // One block per stored element, so a block's index is its place in the file.
        Some(Value::Array(items)) => items.iter().map(parse_block).collect(),
        _ => return skip("a message has no content"),
    };
    Ok(ArchivedMessage { source_index, role, blocks })
}

/// Unknown block types and keys are ignored: the block is `Other`.
fn parse_block(block: &Value) -> Block {
    let Some(block) = block.as_object() else { return Block::Other };
    match block.get("type").and_then(Value::as_str) {
        Some("text") => match block.get("text").and_then(Value::as_str) {
            Some(text) => Block::Text(text.to_string()),
            None => Block::Other,
        },
        Some("tool_use") => match block.get("name").and_then(Value::as_str) {
            Some(name) => Block::ToolUse {
                name: name.to_string(),
                input_head: input_head(block.get("input")),
            },
            None => Block::Other,
        },
        Some("tool_result") => Block::ToolResult {
            is_error: block.get("is_error").and_then(Value::as_bool).unwrap_or(false),
        },
        _ => Block::Other,
    }
}

/// The first 160 chars of a tool input as compact JSON. An object with exactly one key
/// starting `_archive_` is a placeholder, whatever the key. The whole text is redacted
/// before it is cut, so that the cut cannot leave half a token that no rule knows.
fn input_head(input: Option<&Value>) -> String {
    let text = match input {
        None => String::new(),
        Some(Value::Object(obj)) if obj.len() == 1 => match obj.iter().next() {
            Some((key, _)) if key == "_archive_withheld" => "(withheld)".to_string(),
            Some((key, _)) if key == "_archive_omitted" => "(omitted)".to_string(),
            Some((key, Value::String(text))) if key.starts_with("_archive_") => text.clone(),
            _ => Value::Object(obj.clone()).to_string(),
        },
        Some(value) => value.to_string(),
    };
    screen::redact(&text).text.chars().take(INPUT_HEAD_CHARS).collect()
}

// ---------------------------------------------------------------------------------------
// The carried head
// ---------------------------------------------------------------------------------------

/// The position in `a.messages` of the last message that yields nothing (the carried tail);
/// `None` when the file does not start with copies. Decided from `a.carried`, then
/// `a.start`, then the first `source_index` (4.1).
pub fn carried_head(a: &LiveArchive) -> Option<usize> {
    let len = a.messages.len();
    let first = a.messages.first()?;
    let first_assistant =
        |upto: usize| a.messages[..upto].iter().position(|m| m.role == Role::Assistant);
    // (a) the record says so. A mark larger than the message count is no mark.
    if let Some(n) = a.carried.map(|n| n as usize).filter(|n| *n <= len) {
        if n == 0 {
            return None;
        }
        // No assistant message among the copies: the n-th message stands in for the tail.
        return Some(first_assistant(n).unwrap_or(n - 1));
    }
    let successor = match a.start {
        // (b) the event said so.
        Start::Whole => false,
        Start::Carried => true,
        // (c) neither is there: by the first stored message.
        Start::Unknown => {
            first.source_index > 0
                || (first.role == Role::User
                    && matches!(first.blocks.first(), Some(Block::Text(t)) if t.starts_with(CONTINUATION)))
        }
    };
    if !successor {
        return None;
    }
    // A successor with no assistant message has no tail: nothing in it yields a unit.
    Some(first_assistant(len).unwrap_or(len - 1))
}

// ---------------------------------------------------------------------------------------
// Spans
// ---------------------------------------------------------------------------------------

enum UserText<'a> {
    Event { source: &'a str, kind: &'a str, text: &'a str },
    Withheld,
    Skip,
    Human,
}

/// Rule 1: what a `user` text block is.
fn classify(text: &str) -> UserText<'_> {
    if text.starts_with("<event ") {
        let tag_end = text.find('>').unwrap_or(text.len());
        let tag = &text[..tag_end];
        let inner = text.get(tag_end + 1..).unwrap_or("");
        return UserText::Event {
            source: attribute(tag, "source"),
            kind: attribute(tag, "type"),
            text: inner.strip_suffix("</event>").unwrap_or(inner).trim(),
        };
    }
    if text == WITHHELD {
        return UserText::Withheld;
    }
    // SPEC: 1574: a text block of white space alone is no prompt (it would end the span of
    // the prompt before it and take its reply); it is passed over.
    if text.starts_with("[Request interrupted")
        || text.starts_with(CONTINUATION)
        || text.starts_with("[Host context advisory")
        || text.trim().is_empty()
    {
        return UserText::Skip;
    }
    UserText::Human
}

/// The value of ` name="…"` in an event's opening tag, or the empty string.
fn attribute<'a>(tag: &'a str, name: &str) -> &'a str {
    let needle = format!(" {name}=\"");
    match tag.find(&needle) {
        Some(at) => {
            let value = &tag[at + needle.len()..];
            &value[..value.find('"').unwrap_or(value.len())]
        }
        None => "",
    }
}

enum Opener<'a> {
    /// The messages before the file's first prompt (after the carried tail in a successor).
    Leading,
    Human { text: &'a str, source_index: u32, block_index: u32 },
    Withheld,
}

struct Event<'a> {
    source: &'a str,
    kind: &'a str,
    text: &'a str,
    source_index: u32,
    block_index: u32,
}

struct Span<'a> {
    opener: Opener<'a>,
    /// The assistant text blocks, with their positions.
    texts: Vec<(u32, u32, &'a str)>,
    /// `(name, input_head)` of every tool call.
    tools: Vec<(&'a str, &'a str)>,
    events: Vec<Event<'a>>,
    /// Rule 9: a `tool_use` block of the span names a forget tool.
    forget: bool,
}

impl<'a> Span<'a> {
    fn new(opener: Opener<'a>) -> Self {
        Span { opener, texts: Vec::new(), tools: Vec::new(), events: Vec::new(), forget: false }
    }
}

/// The file as spans, in file order; the first is always the leading span. In a successor
/// the walk starts after the carried tail, and the tail's tool calls count as the leading
/// span's for rule 9.
fn spans<'a>(a: &'a LiveArchive, policy: &ArchivePolicy) -> Vec<Span<'a>> {
    let head = carried_head(a);
    let mut current = Span::new(Opener::Leading);
    if let Some(h) = head {
        current.forget = a.messages[..=h].iter().flat_map(|m| m.blocks.iter()).any(
            |b| matches!(b, Block::ToolUse { name, .. } if policy.is_forget_tool(name)),
        );
    }
    let mut out = Vec::new();
    for message in &a.messages[head.map_or(0, |h| h + 1)..] {
        for (at, block) in message.blocks.iter().enumerate() {
            let (source_index, block_index) = (message.source_index, at as u32);
            match (message.role, block) {
                (_, Block::ToolUse { name, input_head }) => {
                    current.tools.push((name, input_head));
                    current.forget |= policy.is_forget_tool(name);
                }
                (Role::Assistant, Block::Text(text)) if !text.trim().is_empty() => {
                    current.texts.push((source_index, block_index, text.trim()));
                }
                (Role::Assistant, Block::Text(_)) => {}
                (Role::User, Block::Text(text)) => match classify(text) {
                    UserText::Event { source, kind, text } => {
                        current.events.push(Event { source, kind, text, source_index, block_index })
                    }
                    UserText::Withheld => {
                        out.push(std::mem::replace(&mut current, Span::new(Opener::Withheld)))
                    }
                    UserText::Human => {
                        let opener = Opener::Human { text, source_index, block_index };
                        out.push(std::mem::replace(&mut current, Span::new(opener)))
                    }
                    UserText::Skip => {}
                },
                _ => {}
            }
        }
    }
    out.push(current);
    out
}

/// Rule 9: which spans yield no unit, and for each span that holds a forget call the pair
/// `(its place, the place of the span its turn began in)`.
///
/// RULING: wave2-c6-review finding 2: a withheld text proves nothing about a new turn
/// having begun (it may be a worker's report in the middle of a turn, stored as the same
/// placeholder as a withheld prompt). So when the span that holds the call was opened by a
/// withheld text, the spans before it go too, back to and including the nearest span a
/// human prompt opened, or the leading span. The cost, one earlier exchange left out when
/// the placeholder really was a new prompt, is accepted: `restore --forget-turns` brings
/// it back, and nothing brings a forgotten prompt back out of the index.
///
/// Known limit: a HUMAN text between the prompt and the call (a steer, or "yes, that one"
/// after the model asked which) opens a span of its own, and the span before it, which
/// holds the prompt that names the thing, is kept. No rule tells such a text from a new
/// request without dropping real exchanges.
fn forget_spans(spans: &[Span<'_>]) -> (Vec<bool>, Vec<(usize, usize)>) {
    let mut dropped = vec![false; spans.len()];
    let mut turns = Vec::new();
    for (at, span) in spans.iter().enumerate() {
        if !span.forget {
            continue;
        }
        // The first span is always the leading one, so the walk ends there at the latest.
        let mut root = at;
        while root > 0 && matches!(spans[root].opener, Opener::Withheld) {
            root -= 1;
        }
        dropped[root..=at].iter_mut().for_each(|d| *d = true);
        turns.push((at, root));
    }
    (dropped, turns)
}

// ---------------------------------------------------------------------------------------
// Units
// ---------------------------------------------------------------------------------------

/// Cut at `max` bytes on a char boundary.
fn cut(mut s: String, max: usize) -> String {
    if s.len() > max {
        let mut at = max;
        while !s.is_char_boundary(at) {
            at -= 1;
        }
        s.truncate(at);
    }
    s
}

fn redact(s: &str) -> String {
    screen::redact(s).text.into_owned()
}

/// The first 12 words.
fn title_of(text: &str) -> String {
    text.split_whitespace().take(TITLE_WORDS).collect::<Vec<_>>().join(" ")
}

pub fn extract_units(a: &LiveArchive, policy: &ArchivePolicy) -> Vec<Unit> {
    let mut units = Vec::new();
    let spans = spans(a, policy);
    let (dropped, _) = forget_spans(&spans);
    for (at, span) in spans.iter().enumerate() {
        // Rule 9: a span of a forget turn yields no unit, its reports included.
        if dropped[at] && !policy.keep_forget_turns {
            continue;
        }
        // SPEC: 1582 does not say what joins the text blocks of a reply: a line break.
        let reply = span.texts.iter().map(|t| t.2).collect::<Vec<_>>().join("\n");
        let reply_chars = reply.chars().count();
        let aux = || {
            let lines: Vec<String> =
                span.tools.iter().map(|(name, input)| format!("{name}: {input}")).collect();
            cut(redact(&lines.join("\n")), AUX_BYTES)
        };
        let mut of_span = Vec::new();
        match span.opener {
            Opener::Human { text, source_index, block_index } => {
                // The whole prompt, redacted, before any cut (B-2f).
                let prompt = redact(text);
                let shown = prompt.trim();
                if reply_chars > 0 && text.trim().chars().count() + reply_chars >= MIN_CHARS {
                    of_span.push(Unit {
                        kind: UnitKind::Exchange,
                        source_index,
                        block_index,
                        title: title_of(shown),
                        body: cut(
                            format!("User: {shown}\nAssistant: {}", redact(&reply)),
                            BODY_BYTES,
                        ),
                        aux: aux(),
                        turn_key: Some(turn_key(&prompt)),
                    });
                }
            }
            Opener::Leading | Opener::Withheld => {
                if let (Some(first), true) = (span.texts.first(), reply_chars >= MIN_CHARS) {
                    let text = redact(&reply);
                    of_span.push(Unit {
                        kind: UnitKind::Continued,
                        source_index: first.0,
                        block_index: first.1,
                        title: title_of(&text),
                        body: cut(format!("Assistant (continued): {text}"), BODY_BYTES),
                        aux: aux(),
                        turn_key: None,
                    });
                }
            }
        }
        for event in &span.events {
            // Rules 4 and 5.
            // SPEC: 1599: `UnitKind` has no kind for an event stored because its source is
            // in `event_sources`; it is a `Report` unit, like a worker's.
            let stored = event.kind == "subagent_completion"
                || policy.event_sources.iter().any(|s| s == event.source);
            if !stored || event.text.is_empty() {
                continue;
            }
            let text = redact(event.text);
            of_span.push(Unit {
                kind: UnitKind::Report,
                source_index: event.source_index,
                block_index: event.block_index,
                title: title_of(&text),
                body: cut(text, BODY_BYTES),
                aux: String::new(),
                turn_key: None,
            });
        }
        of_span.sort_by_key(|u| (u.source_index, u.block_index));
        units.append(&mut of_span);
    }
    // Rule 6: the working note.
    // RULING: wave2-c6-review finding 3: no note unit from a file in which rule 9 dropped a
    // span. The note is text the model wrote at the rollover, and it can restate what was
    // forgotten in this window.
    let forgot = !policy.keep_forget_turns && dropped.iter().any(|d| *d);
    if !forgot && !a.note.trim().is_empty() && a.note.trim() != WITHHELD {
        units.push(Unit {
            kind: UnitKind::Note,
            source_index: 0,
            block_index: 0,
            title: "Working note at rollover".to_string(),
            body: redact(a.note.trim()),
            aux: String::new(),
            turn_key: None,
        });
    }
    units
}

/// Rule 10 of section 2.2: one mark for each span that holds a forget call, in file order.
pub fn forgotten_keys(a: &LiveArchive, policy: &ArchivePolicy) -> Vec<ForgetMark> {
    if policy.keep_forget_turns {
        return Vec::new();
    }
    let spans = spans(a, policy);
    let (_, turns) = forget_spans(&spans);
    turns
        .into_iter()
        .map(|(at, root)| ForgetMark {
            // RULING: wave2-c6-review finding 2: the key is that of the prompt the turn
            // began with, past any withheld text between it and the call.
            turn_key: match spans[root].opener {
                Opener::Human { text, .. } => Some(turn_key(&redact(text))),
                // `forget_spans` never stops at a withheld opener.
                Opener::Withheld => None,
                Opener::Leading => carried_prompt_key(a),
            },
            // The turn began before this file's first prompt.
            head: root == 0,
            // Only a human prompt after the call shows that the turn ended in this file.
            tail: !spans[at + 1..].iter().any(|s| matches!(s.opener, Opener::Human { .. })),
        })
        .collect()
}

/// The key of the nearest human prompt before the leading span: the last one of the
/// carried head, as this file stores it. `None` when the file has no carried head, the head
/// has no prompt, or a withheld prompt is nearer.
fn carried_prompt_key(a: &LiveArchive) -> Option<String> {
    let head = carried_head(a)?;
    let mut key = None;
    for message in a.messages[..=head].iter().filter(|m| m.role == Role::User) {
        for block in &message.blocks {
            let Block::Text(text) = block else { continue };
            match classify(text) {
                UserText::Human => key = Some(turn_key(&redact(text))),
                UserText::Withheld => key = None,
                UserText::Event { .. } | UserText::Skip => {}
            }
        }
    }
    key
}

// ---------------------------------------------------------------------------------------
// Ids and keys
// ---------------------------------------------------------------------------------------

/// `"mem_" + hex(sha256("axel/unit/1" || 0 || scope_dir || 0 || archive_id || 0 ||
/// kind letter || dec(source_index) || "." || dec(block_index)))[..16]`.
pub fn unit_id(scope_dir: &str, archive_id: &str, u: &Unit) -> String {
    let letter = match u.kind {
        UnitKind::Exchange => 'x',
        UnitKind::Continued => 'c',
        UnitKind::Report => 'r',
        UnitKind::Note => 'n',
    };
    let mut hasher = Sha256::new();
    hasher.update(b"axel/unit/1\0");
    hasher.update(scope_dir.as_bytes());
    hasher.update([0u8]);
    hasher.update(archive_id.as_bytes());
    hasher.update([0u8]);
    hasher.update(format!("{letter}{}.{}", u.source_index, u.block_index).as_bytes());
    format!("mem_{}", &hex::encode(hasher.finalize())[..16])
}

/// hex(sha256("axel/turn/3" || 0 || normalise(prompt)))[..16]. Callers pass the WHOLE prompt
/// after `screen::redact`, before any cut (B-2f).
pub fn turn_key(prompt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"axel/turn/3\0");
    hasher.update(normalise(prompt).as_bytes());
    hex::encode(hasher.finalize())[..16].to_string()
}

/// Trim, then every run of whitespace becomes one space.
pub fn normalise(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
