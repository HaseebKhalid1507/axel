//! Core notes JSONL reader (pure): steps 1 and 2 of section 2.6, no database.
//!
//! A reading of core's `store.rs` (`MemoryRecord`, `TombstoneLine`, the line scan of
//! `load_live_project_records`), with the newer core's `"tags": null` as no tags. Its
//! contract test is `tests/notes_reader_tests.rs` with `tests/fixtures/notes/`.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use serde::{Deserialize, Deserializer};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoteRecord {
    pub id: String,
    pub content: String,
    pub tags: Vec<String>,
    pub timestamp_ms: i64,
    pub sensitivity: NoteSensitivity,
    pub max_age_days: Option<u32>,
}

/// Absent in the line: `Normal`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoteSensitivity {
    Normal,
    Sensitive,
    Secret,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotesFile {
    /// What `lstat` gave before the read.
    pub size: u64,
    pub mtime_ns: i64,
    /// Section 2.6, step 2: already filtered; an id on two live lines is the later line.
    pub live: Vec<NoteRecord>,
    // RULING: wave2-c7 review B2: the reader returns the tombstoned ids and the bad lines.
    /// Every id a tombstone line names, in file order, each once, whatever its project.
    pub tombstoned: Vec<String>,
    /// Lines that are not UTF-8, not JSON or not a JSON object, skipped.
    pub bad: Vec<BadLine>,
    /// The file does not end with a line break and its last line is not a JSON object.
    pub torn_tail: bool,
    /// JSON objects that are neither a record nor a tombstone (core skips them too).
    pub skipped_records: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BadLine {
    /// Counted from 1.
    pub line: usize,
    /// The first 16 hex characters of the SHA-256 of the line's bytes, without its line break.
    pub hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NotesError {
    /// The size moved under the read: try next tick.
    #[error("notes file changed during the read")]
    Changed,
    /// Above [`MAX_FILE_BYTES`]: refused before anything is read.
    #[error("notes file of {size} bytes is above the limit")]
    TooLarge { size: u64 },
    #[error("notes file: {0}")]
    Io(String),
}

// Core's `MemoryRecord` (store.rs:9-40 at a373ff06), read side only. No
// `deny_unknown_fields`, as in core.
#[derive(Deserialize)]
#[allow(dead_code)]
struct CoreRecord {
    namespace: String,
    timestamp_ms: u64,
    content: String,
    #[serde(default, deserialize_with = "null_as_no_tags")]
    tags: Vec<String>,
    #[serde(default)]
    meta: Option<serde_json::Value>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    project: Option<String>,
    #[serde(default)]
    provenance: Option<CoreProvenance>,
    #[serde(default)]
    sensitivity: Option<CoreSensitivity>,
    #[serde(default)]
    retention: Option<CoreRetention>,
}

/// The newer core: `"tags": null` says what an absent `tags` says.
fn null_as_no_tags<'de, D: Deserializer<'de>>(reader: D) -> Result<Vec<String>, D::Error> {
    Ok(Option::<Vec<String>>::deserialize(reader)?.unwrap_or_default())
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct CoreProvenance {
    source: String,
    #[serde(default)]
    session: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum CoreSensitivity {
    Normal,
    Sensitive,
    Secret,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum CoreRetention {
    Standard,
    MaxAgeDays(u32),
}

// Core's `TombstoneLine` (store.rs:799-803 at a373ff06).
#[derive(Deserialize)]
#[allow(dead_code)]
struct TombstoneLine {
    tombstone: String,
    timestamp_ms: u64,
}

fn mtime_ns(meta: &std::fs::Metadata) -> i64 {
    meta.mtime()
        .saturating_mul(1_000_000_000)
        .saturating_add(meta.mtime_nsec())
}

// RULING: wave2-c7 review B3: over 4,000 notes at core's 16 KiB limit (`store.rs:96`).
pub const MAX_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// The same file, unchanged: device, inode, size and time.
fn same(a: &std::fs::Metadata, b: &std::fs::Metadata) -> bool {
    a.dev() == b.dev() && a.ino() == b.ino() && a.len() == b.len() && mtime_ns(a) == mtime_ns(b)
}

/// Steps 1 and 2 of section 2.6, and nothing of step 3: no database.
pub fn read_file(path: &Path, project_key: &str) -> Result<NotesFile, NotesError> {
    let io = |e: std::io::Error| NotesError::Io(e.to_string());
    let before = std::fs::symlink_metadata(path).map_err(io)?;
    if !before.file_type().is_file() {
        return Err(NotesError::Io("not a regular file".into()));
    }
    if before.len() > MAX_FILE_BYTES {
        return Err(NotesError::TooLarge { size: before.len() });
    }
    // RULING: wave2-c7 review B4: a link or a FIFO swapped in after the lstat neither is
    // followed nor blocks, and the handle is checked against the lstat BEFORE the read.
    // (`fs_open::open_no_wait`: O_NOFOLLOW | O_NONBLOCK from `libc`, shared with the archive
    // reader.)
    let file = crate::fs_open::open_no_wait(path).map_err(io)?;
    let opened = file.metadata().map_err(io)?;
    // SPEC: 1899-1900 names the size only; a different device or inode (the path replaced
    // between the lstat and the open) or time is treated as a change too.
    if !same(&opened, &before) {
        return Err(NotesError::Changed);
    }
    if !opened.file_type().is_file() {
        return Err(NotesError::Io("not a regular file".into()));
    }
    let mut bytes = Vec::with_capacity(before.len() as usize);
    (&file)
        .take(before.len() + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    // RULING: wave2-c7 review B6: the time is compared after the read too.
    let after = file.metadata().map_err(io)?;
    if bytes.len() as u64 != before.len() || !same(&after, &before) {
        return Err(NotesError::Changed);
    }
    Ok(parse(&bytes, project_key, before.len(), mtime_ns(&before)))
}

/// The first 16 hex characters of the SHA-256 of `raw`.
fn line_hash(raw: &[u8]) -> String {
    let digest = Sha256::digest(raw);
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// What one line is, once it is known to be a JSON object.
enum Parsed {
    Tombstone(String),
    Record(CoreRecord),
    /// An object that is neither (core skips it).
    Neither,
}

/// Core's order (`store.rs:866-873` at a373ff06): a tombstone first, then a record, each
/// parsed from the text, so a repeated key is refused as serde's derive refuses it, and an
/// unknown key is accepted as core accepts it (neither struct denies unknown fields).
fn classify(line: &str) -> Parsed {
    if let Ok(tomb) = serde_json::from_str::<TombstoneLine>(line) {
        return Parsed::Tombstone(tomb.tombstone);
    }
    match serde_json::from_str::<CoreRecord>(line) {
        Ok(rec) => Parsed::Record(rec),
        Err(_) => Parsed::Neither,
    }
}

// RULING: wave2-c7 review B1: a line that is not UTF-8, not JSON or not a JSON object is
// skipped and returned in `bad`, never fatal: core heals a torn tail by writing a line break
// in front of the next record, so an old fragment stays a bad middle line for good, and
// core's readers skip it. A last line with no line break that is not an object is the torn
// tail: neither a record nor in `bad`. One with no break that is an object is read as core
// reads it (core's next append writes the break in front of its own line).
fn parse(bytes: &[u8], project_key: &str, size: u64, mtime_ns: i64) -> NotesFile {
    let mut out = NotesFile {
        size,
        mtime_ns,
        live: Vec::new(),
        tombstoned: Vec::new(),
        bad: Vec::new(),
        torn_tail: false,
        skipped_records: 0,
    };
    let mut tombstoned: HashSet<String> = HashSet::new();
    let mut slot: HashMap<String, usize> = HashMap::new();
    // RULING: wave2-c7 review B3: the lines are iterated, not collected.
    let mut pieces = bytes.split(|b| *b == b'\n').enumerate().peekable();
    while let Some((index, raw)) = pieces.next() {
        let last = pieces.peek().is_none();
        if last && raw.is_empty() {
            break; // the empty piece after the final line break, or an empty file
        }
        let line = std::str::from_utf8(raw).ok();
        if line.is_some_and(|line| line.trim().is_empty()) {
            continue; // blank, as core skips it
        }
        // RULING: wave2-c7 review B5: `Value` only answers "is this line an object"; the
        // text itself is then parsed as core parses it.
        let object = line.filter(|line| {
            serde_json::from_str::<serde_json::Value>(line).is_ok_and(|value| value.is_object())
        });
        let Some(line) = object else {
            if last {
                out.torn_tail = true;
            } else {
                out.bad.push(BadLine {
                    line: index + 1,
                    hash: line_hash(raw),
                });
            }
            continue;
        };
        let rec = match classify(line) {
            Parsed::Tombstone(id) => {
                if tombstoned.insert(id.clone()) {
                    out.tombstoned.push(id);
                }
                continue;
            }
            Parsed::Neither => {
                out.skipped_records += 1;
                continue;
            }
            Parsed::Record(rec) => rec,
        };
        if rec.project.as_deref() != Some(project_key) {
            continue;
        }
        let Some(id) = rec.id else {
            continue;
        };
        // RULING: wave2-c7 review B5 (the timestamp): core's field is u64, the declared one
        // i64; a value past i64::MAX saturates, and the note is kept as core keeps it.
        let timestamp_ms = i64::try_from(rec.timestamp_ms).unwrap_or(i64::MAX);
        let note = NoteRecord {
            id: id.clone(),
            content: rec.content,
            tags: rec.tags,
            timestamp_ms,
            sensitivity: match rec.sensitivity {
                None | Some(CoreSensitivity::Normal) => NoteSensitivity::Normal,
                Some(CoreSensitivity::Sensitive) => NoteSensitivity::Sensitive,
                Some(CoreSensitivity::Secret) => NoteSensitivity::Secret,
            },
            max_age_days: match rec.retention {
                Some(CoreRetention::MaxAgeDays(days)) => Some(days),
                None | Some(CoreRetention::Standard) => None,
            },
        };
        match slot.get(&id) {
            Some(&at) => out.live[at] = note,
            None => {
                slot.insert(id, out.live.len());
                out.live.push(note);
            }
        }
    }
    out.live.retain(|note| !tombstoned.contains(&note.id));
    out
}
