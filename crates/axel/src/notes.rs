//! Core notes JSONL reader (pure): steps 1 and 2 of section 2.6, no database.
//!
//! A reading of core's `store.rs` (`MemoryRecord`, `TombstoneLine`, the line scan of
//! `load_live_project_records`), with the newer core's `"tags": null` as no tags. Its
//! contract test is `tests/notes_tests.rs` with `tests/fixtures/notes/`.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

use serde::{Deserialize, Deserializer};

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
    /// Section 2.6, step 2: already filtered.
    pub live: Vec<NoteRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NotesError {
    /// The size moved under the read: try next tick.
    #[error("notes file changed during the read")]
    Changed,
    /// `note_files.ok = 0` with this reason.
    #[error("line {line}: {reason}")]
    BadLine { line: usize, reason: String },
    #[error("notes file: {0}")]
    Io(String),
}

// Core's `MemoryRecord` (store.rs:9-39), read side only.
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

// Core's `TombstoneLine` (store.rs:521-525).
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

/// Steps 1 and 2 of section 2.6, and nothing of step 3: no database.
pub fn read_file(path: &Path, project_key: &str) -> Result<NotesFile, NotesError> {
    let io = |e: std::io::Error| NotesError::Io(e.to_string());
    let before = std::fs::symlink_metadata(path).map_err(io)?;
    if !before.file_type().is_file() {
        return Err(NotesError::Io("not a regular file".into()));
    }
    let mut file = std::fs::File::open(path).map_err(io)?;
    let mut bytes = Vec::with_capacity(before.len() as usize);
    file.read_to_end(&mut bytes).map_err(io)?;
    let after = file.metadata().map_err(io)?;
    // SPEC: 1899-1900 names the size only; a different inode (the path replaced between the
    // lstat and the open) is treated as a change too.
    if bytes.len() as u64 != before.len()
        || after.len() != before.len()
        || after.ino() != before.ino()
        || after.dev() != before.dev()
    {
        return Err(NotesError::Changed);
    }
    let live = parse(&bytes, project_key)?;
    Ok(NotesFile {
        size: before.len(),
        mtime_ns: mtime_ns(&before),
        live,
    })
}

/// One line that is not a JSON object (not UTF-8, not JSON, or JSON that is not an object)
/// abandons the file, except a last line with no line break: a torn append, ignored.
// SPEC: 1901-1905 against the task brief ("a line that is not UTF-8 or not JSON is skipped
// and counted, never fatal", as the newer core does): the spec's rule is kept, because
// skipping would remove a note's row on a damaged read (test 21) and `NotesFile` has no
// count. A JSON object that is not a valid record is skipped, as core skips it.
fn parse(bytes: &[u8], project_key: &str) -> Result<Vec<NoteRecord>, NotesError> {
    let mut tombstoned: HashSet<String> = HashSet::new();
    let mut live: Vec<NoteRecord> = Vec::new();
    let mut slot: HashMap<String, usize> = HashMap::new();
    let ends_with_break = bytes.last() == Some(&b'\n');
    let mut lines: Vec<&[u8]> = bytes.split(|b| *b == b'\n').collect();
    if ends_with_break || bytes.is_empty() {
        lines.pop(); // the empty piece after the last break
    }
    let last = lines.len();
    for (index, raw) in lines.into_iter().enumerate() {
        let number = index + 1;
        let torn = number == last && !ends_with_break;
        let bad = |reason: &str| NotesError::BadLine {
            line: number,
            reason: reason.to_string(),
        };
        let line = match std::str::from_utf8(raw) {
            Ok(line) => line,
            Err(_) if torn => continue,
            Err(_) => return Err(bad("not UTF-8")),
        };
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(value) => value,
            Err(_) if torn => continue,
            Err(e) => return Err(bad(&format!("not JSON: {e}"))),
        };
        if !value.is_object() {
            if torn {
                continue;
            }
            return Err(bad("not a JSON object"));
        }
        if let Ok(tomb) = TombstoneLine::deserialize(&value) {
            tombstoned.insert(tomb.tombstone);
            continue;
        }
        let Ok(rec) = CoreRecord::deserialize(&value) else {
            continue;
        };
        if rec.project.as_deref() != Some(project_key) {
            continue;
        }
        let Some(id) = rec.id else {
            continue;
        };
        // SPEC: core's timestamp is u64, the declared field i64: a value past i64::MAX is
        // not a record this reader can keep, and is skipped.
        let Ok(timestamp_ms) = i64::try_from(rec.timestamp_ms) else {
            continue;
        };
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
            Some(&at) => live[at] = note,
            None => {
                slot.insert(id, live.len());
                live.push(note);
            }
        }
    }
    live.retain(|note| !tombstoned.contains(&note.id));
    Ok(live)
}
