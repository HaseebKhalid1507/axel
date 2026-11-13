//! `axel synaps-export`: read a Synaps `brain.r8` read-only and write its
//! archives and notes as the files Synaps core's importer reads.
//!
//! This is the only place that knows the Synaps tables of `brain.r8`
//! (`synaps_history`, `synaps_fingerprints`, `synaps_scope_members`,
//! `memories`, `memory_tombstones`). The source is opened immutable and is
//! never written. Nothing here prints; the manifest holds ids and counts,
//! never content.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use rusqlite::{Connection, OpenFlags};
use serde::de::{DeserializeOwned, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::value::RawValue;
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Name of the creation-time field in core's archive record. It follows
/// Synaps core's second store round (S-2), where the name is not final yet:
/// change it here and nowhere else.
pub const CREATED_MS_FIELD: &str = "created_ms";

pub const MANIFEST_FILE: &str = "manifest.json";
pub const ARCHIVES_FILE: &str = "archives.jsonl";
pub const NOTES_FILE: &str = "notes.jsonl";

const RECORD_SCHEMA: &str = "synaps-axel/1";
const FORUM_NAMESPACE: &str = "forum";
const FORUM_META_KEY: &str = "_synaps_forum";
const CAPTURES_NAMESPACE: &str = "captures";

#[derive(Debug, thiserror::Error)]
pub enum ExportError {
    /// A non-empty `-wal` or `-journal` beside the source. Exit code 3.
    #[error("{}", sidecar_message(.sidecar, .suffix))]
    Sidecar { sidecar: PathBuf, suffix: &'static str },
    /// `--out` exists and is not an empty directory. Exit code 2.
    #[error("{0}")]
    Out(String),
    #[error("{0}")]
    Source(String),
    #[error("the source file changed while it was read (size, mtime or SHA-256 differ); nothing of this export may be used")]
    SourceChanged,
    #[error("project key {0:?} cannot be a directory name")]
    ProjectKey(String),
    #[error("i/o: {0}")]
    Io(#[from] io::Error),
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

impl ExportError {
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Sidecar { .. } => 3,
            Self::Out(_) => 2,
            _ => 1,
        }
    }
}

fn sidecar_message(sidecar: &Path, suffix: &str) -> String {
    let fix = if suffix == "-wal" {
        "Stop Synaps, copy brain.r8, brain.r8-wal and brain.r8-shm to another directory, \
         run \"sqlite3 <copy> 'PRAGMA wal_checkpoint(TRUNCATE)'\" there, and export from the copy."
    } else {
        "Stop Synaps, copy brain.r8 and brain.r8-journal to another directory, \
         run \"sqlite3 <copy> 'PRAGMA integrity_check'\" there so that the journal is rolled back, \
         and export from the copy."
    };
    format!(
        "{} is not empty: the last writer did not finish cleanly, or is still running. {fix}",
        sidecar.display()
    )
}

type Result<T> = std::result::Result<T, ExportError>;

// ── The manifest ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Manifest {
    pub source: Source,
    /// One entry for every project key in `memories` or `synaps_history`
    /// (also a key that has nothing but forum posts), and for a key that has
    /// only a note tombstone.
    pub projects: BTreeMap<String, Project>,
    /// Left out: forum posts, all keys together.
    pub forum_posts: u64,
    /// Left out: `msg-` rows of `memory_tombstones`, forgotten forum posts.
    pub forum_tombstones: u64,
    /// Left out: rows of `synaps_captures`. A capture's note that is still in
    /// `memories` is named in `check_failed` as `capture_note`.
    pub captures: u64,
    /// Left out: rows of `memories` that are not Synaps records.
    pub foreign_memories: ForeignMemories,
    /// Left out: `history` fingerprints that match no row of `synaps_history`.
    pub orphan_fingerprints: Vec<OrphanFingerprint>,
    pub fingerprints_by_kind: FingerprintsByKind,
    /// Ids of live archives whose digest is not the one recomputed. The lines
    /// are written all the same.
    pub digest_mismatch: Vec<String>,
    /// Live note rows the helper hides and that are NOT written: a row whose
    /// id has a tombstone (exported as the tombstone only), and a row whose
    /// `note` fingerprint was forgotten in its scope group.
    pub notes_suppressed: u64,
    /// Rows that failed another check and were NOT written.
    pub check_failed: Vec<CheckFailed>,
}

impl Manifest {
    /// False when the command has to exit 1 although everything was written.
    pub fn clean(&self) -> bool {
        self.digest_mismatch.is_empty() && self.check_failed.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Source {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Project {
    /// Lines in `archives.jsonl` (`live + tombstones`).
    pub archives: u64,
    pub live: u64,
    pub tombstones: u64,
    /// Message entries in the live lines.
    pub messages: u64,
    /// Note lines in `notes.jsonl`.
    pub notes: u64,
    /// Tombstone lines in `notes.jsonl`.
    pub note_tombstones: u64,
    pub forum_posts: u64,
    pub forum_tombstones: u64,
    pub captures: u64,
    pub foreign_memories: u64,
    /// File name to SHA-256. Both files always exist, possibly empty.
    pub sha256: BTreeMap<String, String>,
    /// Distinct `logical` values of the written archives, sorted.
    pub logical: Vec<String>,
    /// Set when the key shares its scope group with another key.
    pub scope_group: Option<ScopeGroup>,
    /// Distinct `meta._synaps_repository.source_worktree_root` of its notes.
    pub source_worktree_roots: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ScopeGroup {
    pub canonical: String,
    pub members: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ForeignMemories {
    pub total: u64,
    pub list: Vec<ForeignMemory>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForeignMemory {
    /// `null` for a row without a project key.
    pub project_key: Option<String>,
    pub id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OrphanFingerprint {
    pub project_key: String,
    pub value: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct FingerprintsByKind {
    pub history: u64,
    pub note: u64,
    pub capture: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CheckFailed {
    pub project_key: Option<String>,
    pub id: String,
    pub check: &'static str,
}

// ── Entry ───────────────────────────────────────────────────────────────────

/// Export `source` (a Synaps `brain.r8`) into `out`, which must not exist or
/// must be empty. Writes `<out>/<project_key>/{archives,notes}.jsonl` and
/// `<out>/manifest.json`, and returns the manifest. `Ok` does not mean the
/// export is clean: see [`Manifest::clean`].
pub fn export(source: &Path, out: &Path) -> Result<Manifest> {
    let meta = fs::symlink_metadata(source)
        .map_err(|e| ExportError::Source(format!("{}: {e}", source.display())))?;
    if !meta.file_type().is_file() {
        return Err(ExportError::Source(format!(
            "{} is not a regular file",
            source.display()
        )));
    }
    // The file itself may not be a link (refused above); the folders above it may be
    // (`/var` on macOS is one), so they are resolved before SQLite's NOFOLLOW, which
    // refuses a link in any component, sees the path. As `brain_v2::open_source`.
    let given = source;
    let source = &resolve_parent(source)?;
    refuse_sidecars(source)?;
    check_out(out)?;

    // The source is opened and checked before `--out` is created, so that a
    // refusal leaves no directory behind.
    let before = SourceStat::read(source)?;
    let conn = open_immutable(source)?;
    for table in ["synaps_history", "memories"] {
        let exists = table_exists(&conn, table).map_err(|e| {
            ExportError::Source(format!("{}: not a readable brain ({e})", source.display()))
        })?;
        if !exists {
            return Err(ExportError::Source(format!(
                "{} has no table {table}: not a Synaps brain",
                source.display()
            )));
        }
    }
    prepare_out(out)?;
    let mut manifest = Manifest {
        source: Source {
            path: given.to_string_lossy().into_owned(),
            bytes: before.bytes,
            sha256: before.sha256.clone(),
        },
        projects: BTreeMap::new(),
        forum_posts: 0,
        forum_tombstones: 0,
        captures: 0,
        foreign_memories: ForeignMemories::default(),
        orphan_fingerprints: Vec::new(),
        fingerprints_by_kind: FingerprintsByKind::default(),
        digest_mismatch: Vec::new(),
        notes_suppressed: 0,
        check_failed: Vec::new(),
    };
    let groups = Groups::read(&conn)?;
    let fingerprints = read_fingerprints(&conn, &groups, &mut manifest)?;
    let history_values = export_archives(&conn, out, &groups, &fingerprints, &mut manifest)?;
    manifest.orphan_fingerprints = fingerprints
        .history
        .iter()
        // Within the fingerprint's own scope group: the same value under an
        // unrelated key is another archive, which this row does not hide.
        .filter(|(project_key, value)| {
            !history_values
                .get(groups.canonical(project_key))
                .is_some_and(|values| values.contains(value))
        })
        .map(|(project_key, value)| OrphanFingerprint {
            project_key: project_key.clone(),
            value: value.clone(),
        })
        .collect();
    let capture_notes = read_captures(&conn, &mut manifest)?;
    export_notes(&conn, out, &groups, &fingerprints, &capture_notes, &mut manifest)?;
    drop(conn);

    // Every key gets both files, and its scope group if it has one.
    for (key, project) in manifest.projects.iter_mut() {
        for name in [ARCHIVES_FILE, NOTES_FILE] {
            if !project.sha256.contains_key(name) {
                let sha = HashedFile::create(&project_file(out, key, name)?)?.finish()?;
                project.sha256.insert(name.to_owned(), sha);
            }
        }
        project.scope_group = groups.group(key);
    }

    if SourceStat::read(source)? != before {
        return Err(ExportError::SourceChanged);
    }
    refuse_sidecars(source)?;

    let mut text = serde_json::to_string_pretty(&manifest)?;
    text.push('\n');
    let mut file = create_private(&out.join(MANIFEST_FILE))?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    Ok(manifest)
}

// ── The source ──────────────────────────────────────────────────────────────

#[derive(PartialEq, Eq)]
struct SourceStat {
    bytes: u64,
    mtime: SystemTime,
    sha256: String,
}

impl SourceStat {
    fn read(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 16];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        let meta = file.metadata()?;
        Ok(Self {
            bytes: meta.len(),
            mtime: meta.modified()?,
            sha256: hex::encode(hasher.finalize()),
        })
    }
}

/// `source` with its folder resolved (`canonicalize`) and its last component kept.
fn resolve_parent(source: &Path) -> Result<PathBuf> {
    let name = source
        .file_name()
        .ok_or_else(|| ExportError::Source(format!("{} is not a regular file", source.display())))?;
    let parent = match source.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let parent = parent
        .canonicalize()
        .map_err(|e| ExportError::Source(format!("{}: {e}", source.display())))?;
    Ok(parent.join(name))
}

/// `immutable=1` would silently ignore what a `-wal` or `-journal` holds.
fn refuse_sidecars(source: &Path) -> Result<()> {
    for suffix in ["-wal", "-journal"] {
        let mut name = source.as_os_str().to_owned();
        name.push(suffix);
        let sidecar = PathBuf::from(name);
        match fs::symlink_metadata(&sidecar) {
            Ok(meta) if meta.len() > 0 => return Err(ExportError::Sidecar { sidecar, suffix }),
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

/// `file:<brain>?mode=ro&immutable=1`: no lock, no `-shm`, no write.
fn open_immutable(source: &Path) -> Result<Connection> {
    let text = source
        .to_str()
        .ok_or_else(|| ExportError::Source("the source path is not UTF-8".into()))?;
    let mut uri = String::from("file:");
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || b"/-_.".contains(&b) {
            uri.push(b as char);
        } else {
            let _ = write!(uri, "%{b:02X}");
        }
    }
    uri.push_str("?mode=ro&immutable=1");
    // NOFOLLOW refuses a symbolic link in ANY component of the path; `export` hands it
    // a path whose folders are resolved, so only a file swapped for a link since the
    // check in `export` is refused here.
    Connection::open_with_flags(
        uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )
    .map_err(|e| {
        // The link is named only when the file is one now (swapped in since the check).
        let link = fs::symlink_metadata(source).is_ok_and(|m| m.file_type().is_symlink());
        let why = if link { "; it is a symbolic link, which is refused: give the file's real path" } else { "" };
        ExportError::Source(format!("{} cannot be opened ({e}){why}.", source.display()))
    })
}

fn table_exists(conn: &Connection, name: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
        [name],
        |r| r.get(0),
    )?)
}

// ── The output ──────────────────────────────────────────────────────────────

/// `--out` must not exist or must be an empty directory. Creates nothing;
/// returns whether it exists.
fn check_out(out: &Path) -> Result<bool> {
    match fs::read_dir(out) {
        Ok(mut entries) => {
            if entries.next().is_some() {
                return Err(ExportError::Out(format!(
                    "{} is not empty: --out must not exist or must be an empty directory",
                    out.display()
                )));
            }
            Ok(true)
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(ExportError::Out(format!("{}: {e}", out.display()))),
    }
}

fn prepare_out(out: &Path) -> Result<()> {
    if !check_out(out)? {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(out)
            .map_err(|e| ExportError::Out(format!("{}: {e}", out.display())))?;
    }
    fs::set_permissions(out, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

fn create_private(path: &Path) -> io::Result<File> {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

/// `<out>/<key>/<name>`, making the 0700 directory. The key is used exactly
/// as the brain has it, so it must be usable as one path component.
fn project_file(out: &Path, key: &str, name: &str) -> Result<PathBuf> {
    if key.is_empty()
        || key == "."
        || key == ".."
        || key == MANIFEST_FILE
        || key.contains(['/', '\\', '\0'])
    {
        return Err(ExportError::ProjectKey(key.to_owned()));
    }
    let dir = out.join(key);
    match fs::DirBuilder::new().mode(0o700).create(&dir) {
        Ok(()) => fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?,
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e.into()),
    }
    Ok(dir.join(name))
}

struct HashedFile {
    writer: BufWriter<File>,
    hasher: Sha256,
}

impl HashedFile {
    fn create(path: &Path) -> Result<Self> {
        Ok(Self {
            writer: BufWriter::new(create_private(path)?),
            hasher: Sha256::new(),
        })
    }
    fn line(&mut self, line: &str) -> Result<()> {
        for part in [line.as_bytes(), b"\n"] {
            self.writer.write_all(part)?;
            self.hasher.update(part);
        }
        Ok(())
    }
    fn finish(self) -> Result<String> {
        let file = self.writer.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        Ok(hex::encode(self.hasher.finalize()))
    }
}

// ── Hashes ──────────────────────────────────────────────────────────────────

/// SHA-256 over `be64(len) ‖ part` for each part, lowercase hex.
fn hash_parts(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part);
    }
    hex::encode(hasher.finalize())
}

/// "From a project key to the directory name": core's scope of a key.
pub fn scope_of(project_key: &str) -> String {
    hash_parts(&[project_key.as_bytes()])
}

/// Core's projection digest: `be64(64)‖logical ‖ be64(8)‖be64(source_count)
/// ‖ be64(len)‖messages`, with `messages` taken byte for byte.
pub fn projection_digest(logical: &str, source_count: u64, messages: &str) -> String {
    hash_parts(&[
        logical.as_bytes(),
        &source_count.to_be_bytes(),
        messages.as_bytes(),
    ])
}

fn js(text: &str) -> Result<String> {
    Ok(serde_json::to_string(text)?)
}

// ── Scope groups and fingerprints ───────────────────────────────────────────

/// `synaps_scope_members`: an alias shares one canonical key with others. A
/// brain without the table, or a key without a row, is a group of one.
struct Groups {
    canonical: HashMap<String, String>,
    members: HashMap<String, Vec<String>>,
}

impl Groups {
    fn read(conn: &Connection) -> Result<Self> {
        let mut groups = Self {
            canonical: HashMap::new(),
            members: HashMap::new(),
        };
        if !table_exists(conn, "synaps_scope_members")? {
            return Ok(groups);
        }
        let mut stmt =
            conn.prepare("SELECT member,canonical FROM synaps_scope_members ORDER BY member")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let (member, canonical): (String, String) = (row.get(0)?, row.get(1)?);
            groups
                .members
                .entry(canonical.clone())
                .or_default()
                .push(member.clone());
            groups.canonical.insert(member, canonical);
        }
        Ok(groups)
    }
    fn canonical<'a>(&'a self, key: &'a str) -> &'a str {
        self.canonical.get(key).map_or(key, String::as_str)
    }
    fn group(&self, key: &str) -> Option<ScopeGroup> {
        let canonical = self.canonical(key);
        let members = self.members.get(canonical)?;
        (members.len() > 1).then(|| ScopeGroup {
            canonical: canonical.to_owned(),
            members: members.clone(),
        })
    }
}

struct Fingerprints {
    /// Every `history` row as `(project_key, value)`, sorted.
    history: Vec<(String, String)>,
    /// Canonical key of a scope group to its `history` values.
    suppressed: HashMap<String, HashSet<String>>,
    /// Canonical key of a scope group to its `note` values.
    notes: HashMap<String, HashSet<String>>,
}

fn read_fingerprints(
    conn: &Connection,
    groups: &Groups,
    manifest: &mut Manifest,
) -> Result<Fingerprints> {
    let mut found = Fingerprints {
        history: Vec::new(),
        suppressed: HashMap::new(),
        notes: HashMap::new(),
    };
    if !table_exists(conn, "synaps_fingerprints")? {
        return Ok(found);
    }
    let mut stmt = conn.prepare(
        "SELECT project_key,kind,digest FROM synaps_fingerprints ORDER BY project_key,kind,digest",
    )?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let (key, kind, value): (String, String, String) = (row.get(0)?, row.get(1)?, row.get(2)?);
        let counts = &mut manifest.fingerprints_by_kind;
        match kind.as_str() {
            "history" => {
                counts.history += 1;
                found
                    .suppressed
                    .entry(groups.canonical(&key).to_owned())
                    .or_default()
                    .insert(value.clone());
                found.history.push((key, value));
            }
            // A forget of a note or of a forum post. Never an orphan, never
            // a reason to stop; it hides every live note of its scope group
            // that has the same fingerprint (see `note_fingerprint`).
            "note" => {
                counts.note += 1;
                found
                    .notes
                    .entry(groups.canonical(&key).to_owned())
                    .or_default()
                    .insert(value);
            }
            "capture" => counts.capture += 1,
            _ => manifest.check_failed.push(CheckFailed {
                project_key: Some(key),
                id: value,
                check: "fingerprint_kind",
            }),
        }
    }
    Ok(found)
}

// ── Archives ────────────────────────────────────────────────────────────────

/// Counts the rows of `synaps_captures` and returns their note ids. Captures
/// are not exported: core has no place for them.
fn read_captures(conn: &Connection, manifest: &mut Manifest) -> Result<HashSet<String>> {
    let mut notes = HashSet::new();
    if !table_exists(conn, "synaps_captures")? {
        return Ok(notes);
    }
    let mut stmt = conn.prepare("SELECT project_key,note_id FROM synaps_captures")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let (key, note): (String, String) = (row.get(0)?, row.get(1)?);
        manifest.captures += 1;
        manifest.projects.entry(key).or_default().captures += 1;
        notes.insert(note);
    }
    Ok(notes)
}

/// Writes one line per row of `synaps_history` and returns, for the canonical
/// key of every scope group, each `digest` and `fingerprint` its rows hold.
fn export_archives(
    conn: &Connection,
    out: &Path,
    groups: &Groups,
    fingerprints: &Fingerprints,
    manifest: &mut Manifest,
) -> Result<HashMap<String, HashSet<String>>> {
    let mut values: HashMap<String, HashSet<String>> = HashMap::new();
    let mut seen = HashSet::new();
    let mut logical_by_key: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut open: Option<(String, HashedFile)> = None;
    let mut stmt = conn.prepare(
        "SELECT project_key,id,logical,digest,fingerprint,source_count,message_count,messages,note,created_ms,tombstoned
         FROM synaps_history ORDER BY project_key,created_ms,id",
    )?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let key: String = row.get(0)?;
        let id: String = row.get(1)?;
        let logical: String = row.get(2)?;
        let digest: String = row.get(3)?;
        let fingerprint: Option<String> = row.get(4)?;
        let source_count: i64 = row.get(5)?;
        let message_count: i64 = row.get(6)?;
        let messages: String = row.get(7)?;
        let note: String = row.get(8)?;
        let created_ms: i64 = row.get(9)?;
        let tombstoned: i64 = row.get(10)?;

        manifest.projects.entry(key.clone()).or_default();
        let group_values = values.entry(groups.canonical(&key).to_owned()).or_default();
        group_values.insert(digest.clone());
        if let Some(fingerprint) = &fingerprint {
            group_values.insert(fingerprint.clone());
        }
        let fail = |check: &'static str, manifest: &mut Manifest| {
            manifest.check_failed.push(CheckFailed {
                project_key: Some(key.clone()),
                id: id.clone(),
                check,
            });
        };
        if !seen.insert(id.clone()) {
            fail("duplicate_id", manifest);
            continue;
        }
        // The helper's own reads hide a row whose digest or fingerprint was
        // forgotten anywhere in its scope group, tombstoned or not.
        let suppressed = fingerprints
            .suppressed
            .get(groups.canonical(&key))
            .is_some_and(|set| {
                set.contains(&digest) || fingerprint.as_ref().is_some_and(|f| set.contains(f))
            });
        let tombstone = tombstoned != 0 || suppressed;
        let mut entries = 0u64;
        let line = if tombstone {
            // No created_ms and no body fields: nothing of a forgotten
            // archive reaches a plain file.
            archive_line(&key, &logical, &id, &digest, None, "{\"state\":\"tombstone\"}")?
        } else {
            let Ok(source_count) = u64::try_from(source_count) else {
                fail("source_count", manifest);
                continue;
            };
            // Counted, never turned into a Value: the bytes are the digest's.
            let Ok(parsed) = serde_json::from_str::<Vec<IgnoredAny>>(&messages) else {
                fail("messages_not_array", manifest);
                continue;
            };
            if messages.contains('\n') {
                fail("messages_newline", manifest);
                continue;
            }
            entries = parsed.len() as u64;
            if u64::try_from(message_count).ok() != Some(entries) {
                fail("message_count", manifest);
                continue;
            }
            if projection_digest(&logical, source_count, &messages) != digest {
                manifest.digest_mismatch.push(id.clone());
            }
            let body = format!(
                "{{\"state\":\"live\",\"source_message_count\":{source_count},\"messages\":{messages},\"note\":{}}}",
                js(&note)?
            );
            archive_line(&key, &logical, &id, &digest, Some(created_ms), &body)?
        };

        if open.as_ref().map(|(k, _)| k.as_str()) != Some(key.as_str()) {
            if let Some((done, file)) = open.take() {
                finish_file(manifest, &done, ARCHIVES_FILE, file)?;
            }
            let file = HashedFile::create(&project_file(out, &key, ARCHIVES_FILE)?)?;
            open = Some((key.clone(), file));
        }
        if let Some((_, file)) = open.as_mut() {
            file.line(&line)?;
        }
        logical_by_key.entry(key.clone()).or_default().insert(logical);
        let project = manifest.projects.entry(key).or_default();
        project.archives += 1;
        if tombstone {
            project.tombstones += 1;
        } else {
            project.live += 1;
            project.messages += entries;
        }
    }
    if let Some((done, file)) = open.take() {
        finish_file(manifest, &done, ARCHIVES_FILE, file)?;
    }
    for (key, logical) in logical_by_key {
        manifest.projects.entry(key).or_default().logical = logical.into_iter().collect();
    }
    Ok(values)
}

fn finish_file(manifest: &mut Manifest, key: &str, name: &str, file: HashedFile) -> Result<()> {
    let sha = file.finish()?;
    manifest
        .projects
        .entry(key.to_owned())
        .or_default()
        .sha256
        .insert(name.to_owned(), sha);
    Ok(())
}

fn archive_line(
    key: &str,
    logical: &str,
    id: &str,
    digest: &str,
    created_ms: Option<i64>,
    body: &str,
) -> Result<String> {
    let mut line = String::with_capacity(body.len() + 384);
    let _ = write!(
        line,
        "{{\"version\":1,\"scope\":{},\"logical\":{},\"id\":{},\"digest\":{}",
        js(&scope_of(key))?,
        js(logical)?,
        js(id)?,
        js(digest)?
    );
    if let Some(created_ms) = created_ms {
        let _ = write!(line, ",{}:{created_ms}", js(CREATED_MS_FIELD)?);
    }
    let _ = write!(line, ",\"body\":{body}}}");
    Ok(line)
}

// ── Notes ───────────────────────────────────────────────────────────────────

/// The `provenance` column of a Synaps record:
/// `{"schema":"synaps-axel/1","record":{…, "content":""},"content_bytes":N}`.
#[derive(Deserialize)]
struct Envelope {
    schema: Option<Value>,
    record: Option<Box<RawValue>>,
    content_bytes: Option<Value>,
}

/// The record's fields in the order and with the bytes the helper wrote, so
/// that `meta` and everything else is carried as it is.
struct Fields(Vec<(String, Box<RawValue>)>);

impl<'de> Deserialize<'de> for Fields {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Fields;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Fields, A::Error> {
                let mut fields = Vec::new();
                while let Some(entry) = map.next_entry::<String, Box<RawValue>>()? {
                    fields.push(entry);
                }
                Ok(Fields(fields))
            }
        }
        deserializer.deserialize_map(V)
    }
}

impl Fields {
    fn get<T: DeserializeOwned>(&self, name: &str) -> Option<T> {
        let (_, raw) = self.0.iter().find(|(key, _)| key == name)?;
        serde_json::from_str(raw.get()).ok()
    }
    /// The record with `content` set: exactly a core `MemoryRecord` line.
    fn line(&self, content: &str) -> Result<String> {
        let content = js(content)?;
        let mut line = String::from("{");
        let mut set = false;
        for (key, raw) in &self.0 {
            if line.len() > 1 {
                line.push(',');
            }
            line.push_str(&js(key)?);
            line.push(':');
            if key == "content" {
                line.push_str(&content);
                set = true;
            } else {
                line.push_str(raw.get());
            }
        }
        if !set {
            if line.len() > 1 {
                line.push(',');
            }
            line.push_str("\"content\":");
            line.push_str(&content);
        }
        line.push('}');
        Ok(line)
    }
}

/// The helper's `note_fingerprint` (`sidecar database.rs:71-82`, used by its
/// read filter at `service.rs:138`): `hash_parts` of the record's namespace,
/// `provenance.source`, `provenance.session` and the `content` column. A
/// record without a session has no fingerprint.
fn note_fingerprint(fields: &Fields, content: &str) -> Option<String> {
    let namespace: String = fields.get("namespace")?;
    let provenance: Value = fields.get("provenance")?;
    let source = provenance.get("source")?.as_str()?;
    let session = provenance.get("session")?.as_str()?;
    Some(hash_parts(&[
        namespace.as_bytes(),
        source.as_bytes(),
        session.as_bytes(),
        content.as_bytes(),
    ]))
}

struct NoteLine {
    key: String,
    timestamp_ms: u64,
    id: String,
    line: String,
    tombstone: bool,
}

fn export_notes(
    conn: &Connection,
    out: &Path,
    groups: &Groups,
    fingerprints: &Fingerprints,
    capture_notes: &HashSet<String>,
    manifest: &mut Manifest,
) -> Result<()> {
    let mut lines: Vec<NoteLine> = Vec::new();

    // Tombstones first: a note whose id has one is exported as the tombstone
    // only, and its text is never written.
    let mut forgotten: HashMap<String, HashSet<String>> = HashMap::new();
    if table_exists(conn, "memory_tombstones")? {
        let mut stmt =
            conn.prepare("SELECT id,project_key,deleted_at FROM memory_tombstones ORDER BY id")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let (id, key, deleted_at): (String, String, String) =
                (row.get(0)?, row.get(1)?, row.get(2)?);
            forgotten
                .entry(groups.canonical(&key).to_owned())
                .or_default()
                .insert(id.clone());
            if id.starts_with("msg-") {
                // A forgotten forum post: carried by `migrate synaps`.
                manifest.forum_tombstones += 1;
                manifest.projects.entry(key).or_default().forum_tombstones += 1;
                continue;
            }
            let check = if !id.starts_with("mem-") {
                Some("tombstone_id")
            } else {
                None
            };
            let deleted_ms = chrono::DateTime::parse_from_rfc3339(&deleted_at)
                .ok()
                .and_then(|t| u64::try_from(t.timestamp_millis()).ok());
            let (Some(timestamp_ms), None) = (deleted_ms, check) else {
                manifest.check_failed.push(CheckFailed {
                    project_key: Some(key),
                    id,
                    check: check.unwrap_or("tombstone_deleted_at"),
                });
                continue;
            };
            manifest.projects.entry(key.clone()).or_default();
            lines.push(NoteLine {
                line: format!("{{\"tombstone\":{},\"timestamp_ms\":{timestamp_ms}}}", js(&id)?),
                key,
                timestamp_ms,
                id,
                tombstone: true,
            });
        }
    }

    let mut roots: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut seen = HashSet::new();
    let mut stmt = conn.prepare("SELECT id,project_key,provenance,content FROM memories ORDER BY id")?;
    let mut rows = stmt.query([])?;
    while let Some(row) = rows.next()? {
        let id: String = row.get(0)?;
        let key: Option<String> = row.get::<_, Option<String>>(1)?.filter(|k| !k.is_empty());
        let provenance: Option<String> = row.get(2)?;
        let content: String = row.get(3)?;

        let envelope = provenance
            .as_deref()
            .and_then(|text| serde_json::from_str::<Envelope>(text).ok())
            .filter(|e| e.schema.as_ref().and_then(Value::as_str) == Some(RECORD_SCHEMA));
        let Some(envelope) = envelope else {
            // Made with `axel remember`: not a Synaps note. Counted and left.
            if let Some(key) = &key {
                manifest.projects.entry(key.clone()).or_default().foreign_memories += 1;
            }
            manifest.foreign_memories.total += 1;
            manifest.foreign_memories.list.push(ForeignMemory { project_key: key, id });
            continue;
        };
        let fail = |check: &'static str, manifest: &mut Manifest| {
            manifest.check_failed.push(CheckFailed {
                project_key: key.clone(),
                id: id.clone(),
                check,
            });
        };
        let Some(key) = key.as_deref() else {
            fail("project_key_missing", manifest);
            continue;
        };
        manifest.projects.entry(key.to_owned()).or_default();
        let fields = envelope
            .record
            .as_ref()
            .and_then(|raw| serde_json::from_str::<Fields>(raw.get()).ok());
        let Some(fields) = fields else {
            fail("record", manifest);
            continue;
        };
        let namespace: Option<String> = fields.get("namespace");
        let record_id: Option<String> = fields.get("id");
        let meta: Option<Value> = fields.get("meta");
        // The helper's own test (sidecar/forum.rs): any one marker reserves it.
        if namespace.as_deref() == Some(FORUM_NAMESPACE)
            || id.starts_with("msg-")
            || record_id.as_deref().is_some_and(|i| i.starts_with("msg-"))
            || meta.as_ref().is_some_and(|m| m.get(FORUM_META_KEY).is_some())
        {
            manifest.forum_posts += 1;
            manifest.projects.entry(key.to_owned()).or_default().forum_posts += 1;
            continue;
        }
        if forgotten
            .get(groups.canonical(key))
            .is_some_and(|ids| ids.contains(&id))
        {
            manifest.notes_suppressed += 1;
            continue;
        }
        // One of two identical notes of one session was forgotten: the forget
        // left a `note` fingerprint, and the helper's reads hide the twin.
        if let (Some(values), Some(fingerprint)) = (
            fingerprints.notes.get(groups.canonical(key)),
            note_fingerprint(&fields, &content),
        ) {
            if values.contains(&fingerprint) {
                manifest.notes_suppressed += 1;
                continue;
            }
        }
        // A capture's note (sidecar capture.rs:218): not a project note, and
        // named for what it is rather than as a namespace failure.
        if capture_notes.contains(&id) || namespace.as_deref() == Some(CAPTURES_NAMESPACE) {
            fail("capture_note", manifest);
            continue;
        }
        let project: Option<String> = fields.get("project");
        let timestamp_ms: Option<u64> = fields.get("timestamp_ms");
        let check = if record_id.as_deref() != Some(id.as_str()) {
            Some("record_id")
        } else if project.as_deref() != Some(key) {
            Some("project")
        } else if namespace.as_deref() != Some(format!("project-{key}").as_str()) {
            Some("namespace")
        } else if envelope.content_bytes.as_ref().and_then(Value::as_u64)
            != Some(content.len() as u64)
        {
            Some("content_bytes")
        } else if timestamp_ms.is_none() {
            Some("timestamp_ms")
        } else if !seen.insert(id.clone()) {
            Some("duplicate_id")
        } else {
            None
        };
        let line = fields.line(&content)?;
        let (Some(timestamp_ms), None) = (timestamp_ms, check.or(line.contains('\n').then_some("record_newline"))) else {
            fail(check.unwrap_or("record_newline"), manifest);
            continue;
        };
        if let Some(root) = meta
            .as_ref()
            .and_then(|m| m.get("_synaps_repository"))
            .and_then(|r| r.get("source_worktree_root"))
            .and_then(Value::as_str)
        {
            roots.entry(key.to_owned()).or_default().insert(root.to_owned());
        }
        lines.push(NoteLine {
            key: key.to_owned(),
            timestamp_ms,
            id,
            line,
            tombstone: false,
        });
    }

    // Oldest first within each key: ORDER BY timestamp_ms, id.
    lines.sort_by(|a, b| {
        (&a.key, a.timestamp_ms, &a.id).cmp(&(&b.key, b.timestamp_ms, &b.id))
    });
    let mut open: Option<(String, HashedFile)> = None;
    for entry in lines {
        if open.as_ref().map(|(k, _)| k.as_str()) != Some(entry.key.as_str()) {
            if let Some((done, file)) = open.take() {
                finish_file(manifest, &done, NOTES_FILE, file)?;
            }
            let file = HashedFile::create(&project_file(out, &entry.key, NOTES_FILE)?)?;
            open = Some((entry.key.clone(), file));
        }
        if let Some((_, file)) = open.as_mut() {
            file.line(&entry.line)?;
        }
        let project = manifest.projects.entry(entry.key).or_default();
        if entry.tombstone {
            project.note_tombstones += 1;
        } else {
            project.notes += 1;
        }
    }
    if let Some((done, file)) = open.take() {
        finish_file(manifest, &done, NOTES_FILE, file)?;
    }
    for (key, roots) in roots {
        manifest.projects.entry(key).or_default().source_worktree_roots =
            roots.into_iter().collect();
    }
    Ok(())
}
