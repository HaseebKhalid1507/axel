//! `migrate init | synaps | verify | catch-up` (spec section 1, "Getting the existing content
//! in without loss").
//!
//! Every command returns an [`Outcome`]: one JSON report and an exit code. A source is read
//! by a file copy or through an immutable read-only connection and is never opened for
//! writing. Nothing here reads the environment for a destination, a source or a base, except
//! the home folder for the default of `--synaps-base`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::Duration;

use chrono::{DateTime, NaiveDateTime, Utc};
use rusqlite::types::{ToSqlOutput, ValueRef};
use rusqlite::{params, params_from_iter, Connection, ErrorCode, OpenFlags, OptionalExtension, ToSql};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::brain_v2::{self, OpenOpts, Role};
use crate::error::AxelError;
use crate::ext::types::BrainArg;
use crate::forum::{self, Envelope};
use crate::index_archives;
use crate::r8::{DEFAULT_EMBEDDER, EMBEDDING_DIM};

// ── Exit codes ──────────────────────────────────────────────────────────────

/// The command did what it says.
pub const EXIT_OK: u8 = 0;
/// It ran, and something is short: `verify` found a shortfall, `synaps` found foreign rows
/// (without `--accept-foreign`) or a forum row it cannot read, `catch-up` found a table
/// whose columns differ.
pub const EXIT_SHORTFALL: u8 = 1;
/// Refused before anything was touched: the destination, a flag, or a path.
pub const EXIT_REFUSED: u8 = 2;
/// A source cannot be trusted: it changed while it was read, failed its integrity check,
/// has a log that cannot be ruled out, or is not the kind of file the command reads.
pub const EXIT_SOURCE: u8 = 3;
/// `init --replace`: an extension is running on the destination.
pub const EXIT_RUNNING: u8 = 4;
/// Any other failure (the disk, SQLite). The report says what state was left.
pub const EXIT_FAILED: u8 = 5;

/// A status row younger than this means an extension is running (R-9j).
pub const RUNNING_WINDOW_MS: i64 = 120_000;
const DAY_MS: i64 = 86_400_000;
/// A list of ids in a report holds at most this many; `count` is always the whole number.
const LIST_MAX: usize = 200;
const SIDES: [&str; 3] = ["-wal", "-shm", "-journal"];
const TEMP_PREFIXES: [&str; 3] = [".init-", ".verify-", ".catchup-"];
const RECORD_SCHEMA: &str = "synaps-axel/1";
/// The columns `memories` has had since version 1 (`r8.rs`).
const MEMORY_COLUMNS: [&str; 15] = [
    "id", "category", "topic", "title", "abstract_text", "content", "confidence", "importance",
    "source_sessions", "tags", "related_topics", "created", "updated", "trust_level", "signature",
];
/// The tables whose rows somebody saved (X1), and the two with a rule of their own.
const SAVED_TABLES: [&str; 3] = ["context_data", "staged_memories", "patterns"];
const NOT_USAGE: [&str; 8] = [
    "memories", "documents", "context_data", "staged_memories", "patterns", "kv", "brain_meta",
    "sqlite_sequence",
];

/// What a command answers: the report it prints, and its exit code.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub code: u8,
    pub report: Value,
}

#[derive(Debug, Clone, Default)]
pub struct InitOpts {
    pub from_axel: PathBuf,
    pub replace: bool,
    pub even_if_running: bool,
    pub synaps_base: Option<PathBuf>,
    pub watch: Option<PathBuf>,
}

#[derive(Debug, Clone, Default)]
pub struct SynapsOpts {
    pub from: PathBuf,
    pub map: Vec<String>,
    pub accept_foreign: bool,
}

#[derive(Debug, Clone, Default)]
pub struct VerifyOpts {
    pub from_axel: PathBuf,
    pub from_synaps: PathBuf,
    pub map: Vec<String>,
    pub accept_foreign: bool,
}

#[derive(Debug, Clone, Default)]
pub struct CatchUpOpts {
    pub from_axel: PathBuf,
    pub watch: Option<PathBuf>,
}

// ── Stopping ────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Stop {
    code: u8,
    message: String,
}

fn stop(code: u8, message: impl Into<String>) -> Stop {
    Stop { code, message: message.into() }
}

impl From<rusqlite::Error> for Stop {
    fn from(e: rusqlite::Error) -> Self {
        stop(EXIT_FAILED, format!("SQLite: {e}"))
    }
}
impl From<std::io::Error> for Stop {
    fn from(e: std::io::Error) -> Self {
        stop(EXIT_FAILED, format!("i/o: {e}"))
    }
}
impl From<AxelError> for Stop {
    fn from(e: AxelError) -> Self {
        stop(EXIT_FAILED, e.to_string())
    }
}
impl From<serde_json::Error> for Stop {
    fn from(e: serde_json::Error) -> Self {
        stop(EXIT_FAILED, format!("JSON: {e}"))
    }
}

type Res<T> = Result<T, Stop>;
type Report = Map<String, Value>;

/// What a test hangs at the named points of a command: called with the point's name and a
/// path (the destination, the copy or the source, as the point says); `true` stops the
/// command there with exit 5.
pub type Seam = Box<dyn Fn(&str, &Path) -> bool>;

thread_local! {
    static SEAM: std::cell::RefCell<Option<Seam>> = const { std::cell::RefCell::new(None) };
}

/// For tests only: the seam of the commands run on this thread. Nothing else sets it.
#[doc(hidden)]
pub fn set_test_seam(seam: Option<Seam>) {
    SEAM.with(|s| *s.borrow_mut() = seam);
}

fn seam(point: &str, path: &Path) -> Res<()> {
    if SEAM.with(|s| s.borrow().as_ref().is_some_and(|f| f(point, path))) {
        return Err(stop(EXIT_FAILED, format!("stopped at {point} (test seam)")));
    }
    Ok(())
}

/// A temporary directory beside the destination. Deleted by `finish` on every path, and by
/// `Drop` if a panic passes it.
struct Temp {
    dir: PathBuf,
    gone: bool,
}

impl Temp {
    fn delete(&mut self) -> std::io::Result<()> {
        if self.gone {
            return Ok(());
        }
        match fs::remove_dir_all(&self.dir) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        self.gone = true;
        Ok(())
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        if !self.gone && self.delete().is_ok() {
            eprintln!("temporary copy deleted: {}", self.dir.display());
        }
    }
}

/// Deletes the temporary copy and says so in the report and on standard error. False when
/// it could not be deleted.
fn delete_temp(temp: &mut Option<Temp>, rep: &mut Report) -> bool {
    let Some(mut t) = temp.take() else { return true };
    let name = t.dir.to_string_lossy().into_owned();
    match t.delete() {
        Ok(()) => {
            eprintln!("temporary copy deleted: {name}");
            rep.insert("temporary_copy_deleted".into(), json!(name));
            true
        }
        Err(e) => {
            t.gone = true; // Said here; `Drop` does not try again.
            eprintln!("temporary copy NOT deleted: {name}: {e}");
            rep.insert("temporary_copy_not_deleted".into(), json!(format!("{name}: {e}")));
            false
        }
    }
}

fn finish(command: &str, mut rep: Report, mut temp: Option<Temp>, result: Res<u8>) -> Outcome {
    let (mut code, mut error) = match result {
        Ok(code) => (code, None),
        Err(s) => (s.code, Some(s.message)),
    };
    if !delete_temp(&mut temp, &mut rep) && code == EXIT_OK {
        code = EXIT_FAILED;
        error = Some("the temporary copy could not be deleted".into());
    }
    let mut out = Map::new();
    out.insert("command".into(), json!(command));
    out.insert("ok".into(), json!(code == EXIT_OK));
    out.insert("exit".into(), json!(code));
    if let Some(e) = &error {
        eprintln!("axel-v2 {command}: {e}");
        out.insert("error".into(), json!(e));
    }
    for (k, v) in rep {
        out.insert(k, v);
    }
    Outcome { code, report: Value::Object(out) }
}

// ── Small helpers ───────────────────────────────────────────────────────────

fn now_ms() -> i64 {
    Utc::now().timestamp_millis()
}

fn shown(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn side(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn id_list(ids: &[String]) -> Value {
    json!({
        "count": ids.len(),
        "ids": ids.iter().take(LIST_MAX).collect::<Vec<_>>(),
        "truncated": ids.len() > LIST_MAX,
    })
}

/// `"name"`, for a name that came out of `sqlite_master`.
fn q(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// A time as the old and the new code write one, in milliseconds.
fn parse_time(s: &str) -> Option<i64> {
    if let Ok(t) = DateTime::parse_from_rfc3339(s) {
        return Some(t.timestamp_millis());
    }
    for format in ["%Y-%m-%d %H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S%.f"] {
        if let Ok(t) = NaiveDateTime::parse_from_str(s, format) {
            return Some(t.and_utc().timestamp_millis());
        }
    }
    None
}

/// `file:<path>?<query>`, with the characters a URI gives a meaning to escaped.
fn file_uri(path: &Path, query: &str) -> String {
    let mut out = String::from("file:");
    for c in path.to_string_lossy().chars() {
        match c {
            '%' => out.push_str("%25"),
            '?' => out.push_str("%3f"),
            '#' => out.push_str("%23"),
            c => out.push(c),
        }
    }
    out.push('?');
    out.push_str(query);
    out
}

/// One value of one column, with its storage class and its bytes: two texts are the same
/// only when their bytes are.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Cell {
    Null,
    Int(i64),
    Real(u64),
    Text(Vec<u8>),
    Blob(Vec<u8>),
}

impl Cell {
    fn of(v: ValueRef<'_>) -> Self {
        match v {
            ValueRef::Null => Cell::Null,
            ValueRef::Integer(n) => Cell::Int(n),
            ValueRef::Real(f) => Cell::Real(f.to_bits()),
            ValueRef::Text(b) => Cell::Text(b.to_vec()),
            ValueRef::Blob(b) => Cell::Blob(b.to_vec()),
        }
    }
    fn text(&self) -> Option<&str> {
        match self {
            Cell::Text(b) => std::str::from_utf8(b).ok(),
            _ => None,
        }
    }
    /// For a report: an id or a key, never a body.
    fn show(&self) -> String {
        match self {
            Cell::Null => "NULL".into(),
            Cell::Int(n) => n.to_string(),
            Cell::Real(bits) => f64::from_bits(*bits).to_string(),
            Cell::Text(b) => String::from_utf8_lossy(b).into_owned(),
            Cell::Blob(b) => format!("x'{}'", hex::encode(b)),
        }
    }
    fn json(&self) -> Value {
        match self {
            Cell::Null => Value::Null,
            Cell::Int(n) => json!(n),
            Cell::Real(bits) => json!(f64::from_bits(*bits)),
            Cell::Text(b) => json!(String::from_utf8_lossy(b)),
            Cell::Blob(b) => json!({ "blob_hex": hex::encode(b) }),
        }
    }
}

impl ToSql for Cell {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Borrowed(match self {
            Cell::Null => ValueRef::Null,
            Cell::Int(n) => ValueRef::Integer(*n),
            Cell::Real(bits) => ValueRef::Real(f64::from_bits(*bits)),
            Cell::Text(b) => ValueRef::Text(b),
            Cell::Blob(b) => ValueRef::Blob(b),
        }))
    }
}

fn has_table(conn: &Connection, schema: &str, name: &str) -> rusqlite::Result<bool> {
    conn.query_row(
        &format!("SELECT EXISTS(SELECT 1 FROM {schema}.sqlite_master WHERE type = 'table' AND name = ?1)"),
        [name],
        |r| r.get(0),
    )
}

fn count(conn: &Connection, schema: &str, table: &str) -> rusqlite::Result<i64> {
    conn.query_row(&format!("SELECT COUNT(*) FROM {schema}.{}", q(table)), [], |r| r.get(0))
}

/// The ordinary tables of a file: no virtual table, none of an FTS index's shadow tables,
/// nothing of SQLite's own.
fn plain_tables(conn: &Connection, schema: &str) -> rusqlite::Result<Vec<String>> {
    let mut stmt = conn.prepare(&format!("SELECT name, sql FROM {schema}.sqlite_master WHERE type = 'table'"))?;
    let all: Vec<(String, Option<String>)> =
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<rusqlite::Result<_>>()?;
    let virtual_names: Vec<&str> = all
        .iter()
        .filter(|(_, sql)| sql.as_deref().is_some_and(|s| s.trim_start().to_uppercase().starts_with("CREATE VIRTUAL")))
        .map(|(name, _)| name.as_str())
        .collect();
    let mut out = Vec::new();
    for (name, _) in &all {
        let shadow = virtual_names.iter().any(|v| {
            ["_data", "_idx", "_content", "_docsize", "_config"].iter().any(|s| *name == format!("{v}{s}"))
        });
        if name.starts_with("sqlite_") || virtual_names.contains(&name.as_str()) || shadow {
            continue;
        }
        out.push(name.clone());
    }
    out.sort();
    Ok(out)
}

/// `(name, position in the primary key or 0)` of every column.
fn columns(conn: &Connection, schema: &str, table: &str) -> rusqlite::Result<Vec<(String, i64)>> {
    let mut stmt = conn.prepare(&format!("PRAGMA {schema}.table_info({})", q(table)))?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(1)?, r.get::<_, i64>(5)?)))?;
    rows.collect()
}

fn column_names(conn: &Connection, schema: &str, table: &str) -> rusqlite::Result<Vec<String>> {
    Ok(columns(conn, schema, table)?.into_iter().map(|(name, _)| name).collect())
}

fn key_columns(conn: &Connection, schema: &str, table: &str) -> rusqlite::Result<Vec<String>> {
    let mut keys: Vec<(i64, String)> =
        columns(conn, schema, table)?.into_iter().filter(|(_, pk)| *pk > 0).map(|(n, pk)| (pk, n)).collect();
    keys.sort();
    if keys.is_empty() {
        return Ok(vec!["rowid".into()]);
    }
    Ok(keys.into_iter().map(|(_, n)| n).collect())
}

// ── Files ───────────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, Debug)]
struct FileStat {
    size: u64,
    mtime_ns: i64,
    sha256: String,
}

fn mtime_ns(m: &fs::Metadata) -> i64 {
    m.mtime().saturating_mul(1_000_000_000).saturating_add(m.mtime_nsec())
}

/// Size, time and SHA-256 of a file opened for reading; `None` when there is none.
fn file_stat(path: &Path) -> std::io::Result<Option<FileStat>> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
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
    Ok(Some(FileStat { size: meta.len(), mtime_ns: mtime_ns(&meta), sha256: hex::encode(hasher.finalize()) }))
}

/// The source and the files beside it, as they are at one moment.
#[derive(Clone, PartialEq, Eq, Debug)]
struct SourceStat {
    main: FileStat,
    sides: Vec<Option<FileStat>>,
}

impl SourceStat {
    fn read(path: &Path) -> Res<Self> {
        let main = file_stat(path)
            .map_err(|e| stop(EXIT_SOURCE, format!("{}: {e}", path.display())))?
            .ok_or_else(|| stop(EXIT_SOURCE, format!("{} is gone", path.display())))?;
        let mut sides = Vec::new();
        for suffix in SIDES {
            let name = side(path, suffix);
            sides.push(file_stat(&name).map_err(|e| stop(EXIT_SOURCE, format!("{}: {e}", name.display())))?);
        }
        Ok(Self { main, sides })
    }
    /// A `-wal` or a `-journal` with bytes in it: the main file alone is not the database.
    fn has_log(&self) -> bool {
        [0, 2].iter().any(|i| self.sides[*i].as_ref().is_some_and(|s| s.size > 0))
    }
    fn json(&self, path: &Path) -> Value {
        let one = |s: &Option<FileStat>| match s {
            Some(s) => json!({ "size": s.size, "mtime_ns": s.mtime_ns, "sha256": s.sha256 }),
            None => Value::Null,
        };
        json!({
            "path": shown(path),
            "size": self.main.size,
            "mtime_ns": self.main.mtime_ns,
            "sha256": self.main.sha256,
            "wal": one(&self.sides[0]),
            "shm": one(&self.sides[1]),
            "journal": one(&self.sides[2]),
        })
    }
}

/// Copies with mode 0600 and returns the size and SHA-256 of the bytes written.
fn copy_hashed(from: &Path, to: &Path) -> std::io::Result<(u64, String)> {
    let mut input = File::open(from)?;
    let mut out = fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(to)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 16];
    let mut size = 0u64;
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        out.write_all(&buffer[..n])?;
        size += n as u64;
    }
    out.sync_all()?;
    Ok((size, hex::encode(hasher.finalize())))
}

fn sync_dir(dir: &Path) -> std::io::Result<()> {
    File::open(dir)?.sync_all()
}

/// The path with its folder resolved; the file itself need not exist.
fn resolve(path: &Path) -> Res<PathBuf> {
    if let Ok(p) = path.canonicalize() {
        return Ok(p);
    }
    let name = path
        .file_name()
        .ok_or_else(|| stop(EXIT_REFUSED, format!("{} is not a file name", path.display())))?;
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let parent = match parent.canonicalize() {
        Ok(p) => p,
        Err(_) if parent.is_absolute() => parent,
        Err(_) => std::env::current_dir()?.join(parent),
    };
    Ok(parent.join(name))
}

/// A source named by a flag: it must be a file. Returns its canonical path.
fn source_file(path: &Path, flag: &str) -> Res<PathBuf> {
    let meta = fs::metadata(path).map_err(|e| stop(EXIT_REFUSED, format!("{flag} {}: {e}", path.display())))?;
    if !meta.is_file() {
        return Err(stop(EXIT_REFUSED, format!("{flag} {} is not a regular file", path.display())));
    }
    path.canonicalize().map_err(|e| stop(EXIT_REFUSED, format!("{flag} {}: {e}", path.display())))
}

/// The destination: never from the environment, never a source. Returns `D` and the
/// canonical sources, in the order given.
fn destination(brain: &BrainArg, sources: &[(&Path, &str)]) -> Res<(PathBuf, Vec<PathBuf>)> {
    if brain.from_env {
        return Err(stop(
            EXIT_REFUSED,
            "the destination came from the AXEL_BRAIN environment variable, which a migration never \
             uses (scripts export it pointing at the old brain). Pass --brain <the new brain>.",
        ));
    }
    let dest = resolve(&brain.path)?;
    let mut resolved = Vec::new();
    for (path, flag) in sources {
        let source = source_file(path, flag)?;
        if source == dest {
            return Err(stop(
                EXIT_REFUSED,
                format!("the destination {} is the file given as {flag}; a source is never written", dest.display()),
            ));
        }
        resolved.push(source);
    }
    Ok((dest, resolved))
}

/// The destination of `synaps`, `verify` and `catch-up`: a version 2 brain that exists.
fn existing_destination(brain: &BrainArg, sources: &[(&Path, &str)]) -> Res<(PathBuf, Vec<PathBuf>)> {
    let (dest, sources) = destination(brain, sources)?;
    brain_v2::require_v2(&dest).map_err(|e| stop(EXIT_REFUSED, format!("{}: {e}", dest.display())))?;
    Ok((dest, sources))
}

fn folder_of(dest: &Path) -> PathBuf {
    match dest.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// True when a process with this pid exists. Where that cannot be told, true: a directory
/// that may be in use is left.
fn pid_alive(pid: u32) -> bool {
    if !Path::new("/proc/self").exists() {
        return true;
    }
    Path::new("/proc").join(pid.to_string()).exists()
}

/// Deletes the temporary directories a killed run left beside the destination, and says so.
fn clean_stale(folder: &Path, rep: &mut Report) {
    let Ok(entries) = fs::read_dir(folder) else { return };
    let mut deleted = Vec::new();
    let mut left = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = TEMP_PREFIXES.iter().find_map(|p| name.strip_prefix(p)) else { continue };
        let path = entry.path();
        if !fs::symlink_metadata(&path).is_ok_and(|m| m.is_dir()) {
            continue;
        }
        let pid = rest.split('-').next().and_then(|p| p.parse::<u32>().ok());
        match pid {
            Some(pid) if pid != std::process::id() && !pid_alive(pid) => match fs::remove_dir_all(&path) {
                Ok(()) => {
                    eprintln!("temporary copy of an earlier run deleted: {}", path.display());
                    deleted.push(shown(&path));
                }
                Err(e) => left.push(format!("{}: {e}", path.display())),
            },
            Some(pid) => left.push(format!("{}: process {pid} is alive", path.display())),
            None => left.push(format!("{}: no process id in the name", path.display())),
        }
    }
    if !deleted.is_empty() {
        rep.insert("stale_temporaries_deleted".into(), json!(deleted));
    }
    if !left.is_empty() {
        rep.insert("temporaries_left".into(), json!(left));
    }
}

/// What a checked copy is: where it is, what the source looked like, and the row count of
/// each of its tables.
struct Checked {
    db: PathBuf,
    stat: SourceStat,
    counts: BTreeMap<String, i64>,
}

/// The lines `PRAGMA integrity_check` answers; `["ok"]` for a sound file.
fn integrity(conn: &Connection) -> rusqlite::Result<Vec<String>> {
    let mut check = conn.prepare("PRAGMA integrity_check")?;
    let rows = check.query_map([], |r| r.get::<_, String>(0))?;
    rows.collect()
}

fn source_trouble(what: &str, e: rusqlite::Error) -> Stop {
    stop(EXIT_SOURCE, format!("{what}: {e}"))
}

/// Steps 2 to 4 of `init`: copy the source and its side files into a new directory in
/// `folder`, prove the source did not change meanwhile, apply the log to the copy, and
/// check the copy. The source is read by a file copy and by nothing else.
fn checked_copy(source: &Path, folder: &Path, prefix: &str, temp: &mut Option<Temp>) -> Res<Checked> {
    // 2.
    let before = SourceStat::read(source)?;
    // 3.
    let dir = folder.join(format!("{prefix}{}-{}", std::process::id(), now_ms()));
    fs::DirBuilder::new().mode(0o700).create(&dir)?;
    *temp = Some(Temp { dir: dir.clone(), gone: false });
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))?;
    let db = dir.join("axel.r8");
    let changed = || stop(EXIT_SOURCE, "the source changed while it was being copied");
    let (size, sha) = copy_hashed(source, &db)?;
    if size != before.main.size || sha != before.main.sha256 {
        return Err(changed());
    }
    for (i, suffix) in SIDES.iter().enumerate() {
        let from = side(source, suffix);
        match (&before.sides[i], copy_hashed(&from, &side(&db, suffix))) {
            (Some(was), Ok((size, sha))) if size == was.size && sha == was.sha256 => {}
            (None, Err(e)) if e.kind() == std::io::ErrorKind::NotFound => {}
            (Some(_), Err(e)) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
            _ => return Err(changed()),
        }
    }
    if SourceStat::read(source)? != before {
        return Err(changed());
    }
    // 4.
    let counts = {
        let what = "the copy of the source cannot be read as a database";
        let conn = Connection::open_with_flags(&db, OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX)
            .map_err(|e| source_trouble(what, e))?;
        let busy: i64 = conn
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0))
            .map_err(|e| source_trouble(what, e))?;
        if busy != 0 {
            return Err(stop(EXIT_SOURCE, "the log of the copy could not be applied (checkpoint busy)"));
        }
        let verdict = integrity(&conn)
            .map_err(|e| source_trouble("the integrity check of the copy failed", e))?;
        if verdict != ["ok"] {
            return Err(stop(
                EXIT_SOURCE,
                format!(
                    "the integrity check of the copy did not answer ok: {}",
                    verdict.first().map(String::as_str).unwrap_or("no answer")
                ),
            ));
        }
        let mut counts = BTreeMap::new();
        for table in plain_tables(&conn, "main").map_err(|e| source_trouble(what, e))? {
            counts.insert(table.clone(), count(&conn, "main", &table).map_err(|e| source_trouble(what, e))?);
        }
        counts
    };
    for suffix in SIDES {
        match fs::remove_file(side(&db, suffix)) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    File::open(&db)?.sync_all()?;
    sync_dir(&dir)?;
    Ok(Checked { db, stat: before, counts })
}

/// What `status` will compare the old brain with (section 7, "The old brain written after
/// the import"): the file is `lstat`ed, never opened.
fn watch_detail(path: &Path) -> Res<Value> {
    let path = path
        .canonicalize()
        .map_err(|e| stop(EXIT_REFUSED, format!("--watch {}: {e}", path.display())))?;
    let main = fs::symlink_metadata(&path)?;
    let wal = match fs::symlink_metadata(side(&path, "-wal")) {
        Ok(m) => Some(m),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e.into()),
    };
    Ok(json!({
        "path": shown(&path),
        "size": main.len(),
        "mtime_ns": mtime_ns(&main),
        "wal_size": wal.as_ref().map(|m| m.len()),
        "wal_mtime_ns": wal.as_ref().map(mtime_ns),
    }))
}

#[allow(clippy::too_many_arguments)]
fn log_row(
    conn: &Connection,
    source: &str,
    sha: &str,
    item: &str,
    rows_in: i64,
    imported: i64,
    present: i64,
    flagged: i64,
    skipped: i64,
    detail: &str,
) -> rusqlite::Result<()> {
    conn.execute(
        "INSERT INTO import_log(at, source, source_sha256, item, rows_in, rows_imported, rows_present, \
         rows_flagged, rows_skipped, detail) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![Utc::now().to_rfc3339(), source, sha, item, rows_in, imported, present, flagged, skipped, detail],
    )?;
    Ok(())
}

/// Runs `body` inside `begin_write`; any error rolls back, and so does a failed `COMMIT`.
fn in_write<T>(conn: &Connection, body: impl FnOnce() -> Res<T>) -> Res<T> {
    brain_v2::begin_write(conn)?;
    let done = body().and_then(|value| {
        conn.execute_batch("COMMIT")?;
        Ok(value)
    });
    if done.is_err() && !conn.is_autocommit() {
        let _ = conn.execute_batch("ROLLBACK");
    }
    done
}

// ── migrate init ────────────────────────────────────────────────────────────

/// `axel.r8` → a new version 2 brain at the destination.
pub fn init(brain: &BrainArg, opts: &InitOpts) -> Outcome {
    let mut rep = Report::new();
    let mut temp = None;
    let result = init_steps(brain, opts, &mut rep, &mut temp);
    rep.entry("destination_made").or_insert(json!(false));
    finish("migrate init", rep, temp, result)
}

/// A few words about a file that is in the way, read without writing to it or beside it.
fn describe_existing(dest: &Path) -> Option<String> {
    // `immutable=1` makes no `-wal` and no `-shm`; with a log beside the file it would
    // read past it, so then the open is an ordinary read-only one (the side files exist).
    let wal = fs::metadata(side(dest, "-wal")).map(|m| m.len()).unwrap_or(0);
    let query = if wal == 0 { "mode=ro&immutable=1" } else { "mode=ro" };
    let conn = Connection::open_with_flags(
        file_uri(dest, query),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .ok()?;
    conn.busy_timeout(Duration::from_millis(1_000)).ok()?;
    let version: i64 = conn
        .query_row("SELECT json_extract(value, '$.schema_version') FROM brain_meta WHERE key = 'meta'", [], |r| {
            r.get(0)
        })
        .ok()?;
    let rows = |table: &str| -> String {
        match has_table(&conn, "main", table).and_then(|there| if there { count(&conn, "main", table) } else { Ok(0) }) {
            Ok(n) => n.to_string(),
            Err(_) => "?".into(),
        }
    };
    Some(format!("version {version}, {} memories, {} forum posts", rows("memories"), rows("forum_posts")))
}

/// How long ago the destination's status row was written; `None` when it has none.
fn status_age_ms(dest: &Path) -> Res<Option<i64>> {
    let cannot = |e: String| {
        stop(
            EXIT_RUNNING,
            format!(
                "cannot tell whether an Axel extension is running on {} ({e}); stop Synaps first, or pass \
                 --even-if-running",
                dest.display()
            ),
        )
    };
    let conn = match Connection::open_with_flags(
        file_uri(dest, "mode=ro"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    ) {
        Ok(c) => c,
        Err(e) => return Err(cannot(e.to_string())),
    };
    let _ = conn.busy_timeout(Duration::from_millis(1_000));
    let there = match has_table(&conn, "main", "context_data") {
        Ok(there) => there,
        // Not a database: no extension can be running on it.
        Err(e) if e.sqlite_error_code() == Some(ErrorCode::NotADatabase) => return Ok(None),
        Err(e) => return Err(cannot(e.to_string())),
    };
    if !there {
        return Ok(None);
    }
    let value: Option<String> = conn
        .query_row("SELECT value FROM context_data WHERE layer = 'axel' AND key = 'status'", [], |r| r.get(0))
        .optional()
        .map_err(|e| cannot(e.to_string()))?;
    let Some(value) = value else { return Ok(None) };
    let written = serde_json::from_str::<Value>(&value).ok().and_then(|v| v.get("written_ms").and_then(Value::as_i64));
    match written {
        Some(written) => Ok(Some(now_ms() - written)),
        None => Err(cannot("its status row has no written_ms".into())),
    }
}

/// Step 5's refusal (R-9j): exit 4 when the destination's status row is younger than two
/// minutes.
fn refuse_if_running(dest: &Path) -> Res<()> {
    match status_age_ms(dest)? {
        Some(age) if age < RUNNING_WINDOW_MS => Err(stop(
            EXIT_RUNNING,
            format!(
                "an Axel extension is running on this brain (status written {} s ago); stop Synaps first, or \
                 pass --even-if-running",
                age / 1000
            ),
        )),
        _ => Ok(()),
    }
}

fn init_steps(brain: &BrainArg, opts: &InitOpts, rep: &mut Report, temp: &mut Option<Temp>) -> Res<u8> {
    let (dest, sources) = destination(brain, &[(opts.from_axel.as_path(), "--from-axel")])?;
    let source = &sources[0];
    rep.insert("destination".into(), json!(shown(&dest)));

    // 1. A destination that exists is kept.
    let existing = fs::symlink_metadata(&dest).ok();
    if let Some(meta) = &existing {
        if !meta.is_file() {
            return Err(stop(
                EXIT_REFUSED,
                format!("{} exists and is not a regular file; it is kept", dest.display()),
            ));
        }
        if !opts.replace {
            let what = describe_existing(&dest).map(|d| format!(" ({d})")).unwrap_or_default();
            return Err(stop(
                EXIT_REFUSED,
                format!(
                    "a new brain already exists at {}{what}; it is kept. Pass --replace to move it aside and \
                     start again from axel.r8.",
                    dest.display()
                ),
            ));
        }
    }

    // A log or an index of a brain that is no longer there would be taken for the new
    // brain's own when it is put at `D`: refused, naming them.
    if existing.is_none() {
        let mut beside: Vec<String> = SIDES.iter().map(|s| side(&dest, s)).chain([hnsw_of(&dest)])
            .filter(|p| fs::symlink_metadata(p).is_ok())
            .map(|p| shown(&p))
            .collect();
        if !beside.is_empty() {
            beside.sort();
            return Err(stop(
                EXIT_REFUSED,
                format!(
                    "{} is not there, but {} beside it would be taken for the new brain's own; move them \
                     away and run this again. Nothing was done.",
                    dest.display(),
                    beside.join(", ")
                ),
            ));
        }
    }

    // What 7b and 8 will record is settled before anything is copied.
    let (base, base_from) = match &opts.synaps_base {
        Some(dir) => (dir.clone(), "--synaps-base"),
        None => {
            let home = dirs::home_dir().ok_or_else(|| {
                stop(EXIT_REFUSED, "no home folder is known; pass --synaps-base <the Synaps base directory>")
            })?;
            (home.join(".synaps-cli"), "<home>/.synaps-cli")
        }
    };
    // The same text `indexer_start::open` writes and compares.
    let base = base.canonicalize().unwrap_or(base).to_string_lossy().into_owned();
    if let Some(watch) = &opts.watch {
        source_file(watch, "--watch")?;
    }

    // SPEC: 1344: step 5's first refusal is made here too, before anything is copied or
    // cleaned, so that a refusal leaves no trace at all; step 5 asks again before it moves
    // anything, for an extension that started during the copy.
    if existing.is_some() && !opts.even_if_running {
        refuse_if_running(&dest)?;
    }

    let folder = folder_of(&dest);
    if !folder.exists() {
        fs::DirBuilder::new().recursive(true).mode(0o700).create(&folder)?;
    }
    clean_stale(&folder, rep);

    // 2 to 5. Nothing is put at `D` before the new brain is finished.
    let (db, aside_note) = prepare(&dest, source, opts, existing.is_some(), &base, base_from, rep, temp)
        .map_err(|s| {
            let sep = if s.message.ends_with('.') { " " } else { ". " };
            stop(s.code, format!("{}{sep}Nothing was put at {}.", s.message, dest.display()))
        })?;

    // 6.
    // SPEC: 1351 renames `T/axel.r8` to `D` and 1352-1357 then open, migrate, record and
    // `VACUUM` the file at `D`. Here all of that ran on `T/axel.r8` (ruling, wave5-c19-review
    // finding 1), and putting the finished file at `D` is the very last step: until it, `D`
    // is absent or the old brain, never a brain that is not finished. It is put there by a
    // hard link, which never writes over a file that appeared at `D` meanwhile (an
    // extension that started makes an empty brain); on a file system without hard links by
    // `renameat2(RENAME_NOREPLACE)`, which never does either; where neither exists, the
    // command refuses.
    match place(&db, &dest) {
        Ok(how) => {
            rep.insert("placed_by".into(), json!(how));
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            return Err(stop(
                EXIT_REFUSED,
                format!(
                    "a file appeared at {} while the new brain was being made; it is kept and nothing was put \
                     there. Stop Synaps and run this again with --replace.{aside_note}",
                    dest.display()
                ),
            ))
        }
        Err(e) => {
            return Err(stop(
                EXIT_FAILED,
                format!("the new brain could not be put at {}: {e}. Nothing was put there.{aside_note}", dest.display()),
            ))
        }
    }
    rep.insert("destination_made".into(), json!(true));
    if let Err(e) = sync_dir(&folder) {
        return Err(stop(
            EXIT_FAILED,
            format!("the new brain is at {} and is finished, but its folder could not be synced: {e}", dest.display()),
        ));
    }
    Ok(EXIT_OK)
}

/// Puts the finished brain at `to` without ever writing over a file there; says how.
fn place(from: &Path, to: &Path) -> std::io::Result<&'static str> {
    match fs::hard_link(from, to) {
        Ok(()) => return Ok("hard link"),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => return Err(e),
        // The answers of a file system that has no hard links.
        Err(e) if matches!(e.raw_os_error(), Some(libc::EPERM | libc::EOPNOTSUPP | libc::EMLINK | libc::ENOSYS)) => {}
        Err(e) => return Err(e),
    }
    rename_noreplace(from, to).map(|()| "renameat2(RENAME_NOREPLACE)")
}

#[cfg(target_os = "linux")]
fn rename_noreplace(from: &Path, to: &Path) -> std::io::Result<()> {
    use std::os::unix::ffi::OsStrExt;
    let from = std::ffi::CString::new(from.as_os_str().as_bytes())?;
    let to = std::ffi::CString::new(to.as_os_str().as_bytes())?;
    // SAFETY: two NUL-terminated paths that live across the call.
    let done = unsafe {
        libc::renameat2(libc::AT_FDCWD, from.as_ptr(), libc::AT_FDCWD, to.as_ptr(), libc::RENAME_NOREPLACE)
    };
    if done == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "linux"))]
fn rename_noreplace(_from: &Path, _to: &Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "this file system has no hard links and this system no renameat2: refused rather than risk a file at the destination",
    ))
}

/// Steps 2 to 5: the checked copy, made into a finished, closed and synced version 2 brain
/// at its temporary name, and then, with `--replace`, the old `D` moved aside. Returns the
/// finished file and the note of where the old brain went.
#[allow(clippy::too_many_arguments)]
fn prepare(
    dest: &Path,
    source: &Path,
    opts: &InitOpts,
    existing: bool,
    base: &str,
    base_from: &str,
    rep: &mut Report,
    temp: &mut Option<Temp>,
) -> Res<(PathBuf, String)> {
    let folder = folder_of(dest);
    // 2 to 4.
    let copy = checked_copy(source, &folder, ".init-", temp)?;
    rep.insert("source".into(), copy.stat.json(source));
    rep.insert("source_log_carried".into(), json!(copy.stat.sides[0].as_ref().is_some_and(|s| s.size > 0)));
    // The numbers `status` compares with: of `--watch` when given, else of the source as
    // step 2 found it (taken again by `lstat`, which is how `status` reads them).
    let watch = watch_detail(opts.watch.as_deref().unwrap_or(source))?;

    // What `open_v2` would refuse is refused first, as a source this command cannot import.
    match brain_v2::require_v2(&copy.db) {
        Ok(()) | Err(AxelError::V1Brain(_)) => {}
        Err(e) => {
            return Err(stop(EXIT_SOURCE, format!("{} is not a brain this command can import: {e}", source.display())))
        }
    }
    for suffix in SIDES {
        let _ = fs::remove_file(side(&copy.db, suffix));
    }
    seam("checked", &copy.db)?;

    // 7, 7b, 8 and 7a, on the copy.
    upgrade(&copy.db, source, &copy, base, base_from, &watch, rep)?;
    // Closed: nothing that SQLite or `open_v2` made may be left beside it.
    let dir = folder_of(&copy.db);
    let mut left: Vec<String> = fs::read_dir(&dir)?
        .map(|e| e.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<_>>()?;
    left.retain(|n| n != "axel.r8");
    if !left.is_empty() {
        left.sort();
        return Err(stop(
            EXIT_FAILED,
            format!("the new brain was closed and files are still beside it: {}", left.join(", ")),
        ));
    }
    seam("closed", &copy.db)?;
    fs::set_permissions(&copy.db, fs::Permissions::from_mode(0o600))?;
    File::open(&copy.db)?.sync_all()?;
    sync_dir(&dir)?;
    seam("synced", &copy.db)?;

    // 5. `--replace`: the old destination is moved aside, never deleted.
    let mut aside_note = String::new();
    if existing {
        if !opts.even_if_running {
            refuse_if_running(dest)?;
        }
        let checkpoint = Connection::open_with_flags(
            dest,
            OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .and_then(|c| c.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get::<_, i64>(0)));
        let checkpoint = match checkpoint {
            Ok(0) => "done".to_string(),
            Ok(_) => "busy: the log was moved with the file".to_string(),
            Err(e) => format!("not done ({e}): the file was moved as it was"),
        };
        let stamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        let mut aside = side(dest, &format!(".replaced-{stamp}"));
        let mut n = 1;
        // SPEC: 1349: `<brain>.r8.hnsw`, the vector index of the old search, belongs to the
        // brain that is moved: it goes with it.
        let hnsw = hnsw_of(dest);
        let taken = |aside: &Path| {
            [String::new(), "-wal".into(), "-shm".into(), ".hnsw".into()]
                .iter()
                .any(|s| fs::symlink_metadata(side(aside, s)).is_ok())
        };
        while taken(&aside) {
            n += 1;
            aside = side(dest, &format!(".replaced-{stamp}.{n}"));
        }
        let mut moved = Vec::new();
        fs::rename(dest, &aside)?;
        aside_note = format!(" The brain that was there before is at {}.", aside.display());
        for (from, to) in [
            (side(dest, "-wal"), side(&aside, "-wal")),
            (side(dest, "-shm"), side(&aside, "-shm")),
            (hnsw, side(&aside, ".hnsw")),
        ] {
            match fs::rename(&from, &to) {
                Ok(()) => moved.push(shown(&to)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    return Err(stop(
                        EXIT_FAILED,
                        format!("{} could not be moved aside: {e}.{aside_note}", from.display()),
                    ))
                }
            }
        }
        rep.insert(
            "replaced".into(),
            json!({ "moved_from": shown(dest), "moved_to": shown(&aside), "with": moved, "checkpoint": checkpoint }),
        );
        seam("replaced", dest).map_err(|s| stop(s.code, format!("{}.{aside_note}", s.message)))?;
    }
    Ok((copy.db, aside_note))
}

fn hnsw_of(brain: &Path) -> PathBuf {
    brain.with_extension("r8.hnsw")
}

/// Steps 7, 7b, 8 and 7a on `db`, the copy at its temporary name, which nobody else has
/// open. The connection is closed when this returns.
fn upgrade(
    db: &Path,
    source: &Path,
    copy: &Checked,
    base: &str,
    base_from: &str,
    watch: &Value,
    rep: &mut Report,
) -> Res<()> {
    // 7.
    let opts = OpenOpts { agent: None, model: DEFAULT_EMBEDDER, dim: EMBEDDING_DIM, upgrade_v1: true };
    let brain = brain_v2::open_v2(db, &opts)?;
    let conn = &brain.conn;
    conn.execute_batch("PRAGMA synchronous=FULL")?;
    seam("open_v2", db)?;

    // 7b and 8, in one transaction.
    let previous_base = in_write(conn, || {
        let previous: Option<String> = conn
            .query_row("SELECT value FROM context_data WHERE layer = 'axel' AND key = 'synaps_base'", [], |r| r.get(0))
            .optional()?;
        conn.execute(
            "INSERT INTO context_data(key, layer, value) VALUES('synaps_base', 'axel', ?1)
             ON CONFLICT(key, layer) DO UPDATE SET value = excluded.value",
            [base],
        )?;
        let source_text = shown(source);
        let sha = &copy.stat.main.sha256;
        let total: i64 = copy.counts.values().sum();
        log_row(conn, &source_text, sha, "init", total, total, 0, 0, 0, &watch.to_string())?;
        for (table, n) in &copy.counts {
            log_row(conn, &source_text, sha, &format!("count:{table}"), *n, *n, 0, 0, 0, "")?;
        }
        Ok(previous)
    })?;
    seam("7b_8", db)?;
    eprintln!("base recorded: {base}");
    rep.insert("base_recorded".into(), json!(base));
    rep.insert("base_from".into(), json!(base_from));
    if let Some(previous) = previous_base.filter(|p| p != base) {
        rep.insert("base_replaced".into(), json!(previous));
    }
    rep.insert("watch".into(), watch.clone());
    rep.insert("tables".into(), json!(copy.counts));

    // SPEC: 1518-1522 with ruling 2 of the task: `unreadable_memories` counts a row whose
    // `id`, `content` OR `title` is not text. Adoption skips on another test (`id` or
    // `content` not text, or not UTF-8) and its number is printed beside it.
    let unreadable: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memories
          WHERE typeof(id) <> 'text' OR typeof(content) <> 'text' OR typeof(title) <> 'text'",
        [],
        |r| r.get(0),
    )?;
    rep.insert("unreadable_memories".into(), json!(unreadable));
    rep.insert("adoption_skipped_memories".into(), json!(brain.unreadable_memories));
    let fts = column_names(conn, "main", "memories_fts")?;
    rep.insert("memories_fts_columns".into(), json!(fts));
    rep.insert("memories_fts_columns_ok".into(), json!(fts == ["mem_id", "title", "abstract", "content"]));
    let graph: i64 = conn.query_row(
        "SELECT COUNT(*) FROM nodes
          WHERE id IN (SELECT id FROM memories) OR source_file IN (SELECT id FROM memories)",
        [],
        |r| r.get(0),
    )?;
    eprintln!("memory_graph_nodes: {graph}");
    rep.insert("memory_graph_nodes".into(), json!(graph));
    for (key, sql) in [
        ("memories", "SELECT COUNT(*) FROM memories"),
        ("memories_adopted", "SELECT COUNT(*) FROM memory_ext"),
        ("memories_fts_rows", "SELECT COUNT(*) FROM memories_fts"),
        ("documents", "SELECT COUNT(*) FROM documents"),
    ] {
        let n: i64 = conn.query_row(sql, [], |r| r.get(0))?;
        rep.insert(key.into(), json!(n));
    }

    // 7a.
    // SPEC: 1356 puts the `VACUUM` before steps 7b and 8, and 1001 says it is "the last
    // statement `migrate init` runs on the new file". It is last here, so that the
    // integrity check that follows it is of the file as the command leaves it.
    conn.execute_batch("VACUUM")?;
    seam("vacuum", db)?;
    let busy: i64 = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0))?;
    if busy != 0 {
        return Err(stop(EXIT_FAILED, "the checkpoint after VACUUM was blocked: another process has the new brain open"));
    }
    seam("checkpoint", db)?;
    let verdict = integrity(conn)?;
    if verdict != ["ok"] {
        return Err(stop(
            EXIT_SOURCE,
            format!(
                "the integrity check of the new brain did not answer ok: {}",
                verdict.first().map(String::as_str).unwrap_or("no answer")
            ),
        ));
    }
    rep.insert("vacuum".into(), json!("done"));
    rep.insert("integrity_check".into(), json!("ok"));
    seam("integrity_check", db)?;
    brain.conn.close().map_err(|(_, e)| stop(EXIT_FAILED, format!("the new brain could not be closed: {e}")))?;
    Ok(())
}

// ── The Synaps brain.r8 ─────────────────────────────────────────────────────

/// `--map <old key>=<new key>`, repeated.
fn parse_map(pairs: &[String]) -> Res<HashMap<String, String>> {
    let key = |s: &str| {
        s.len() == 17
            && s.starts_with('p')
            && s[1..].bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    let mut map = HashMap::new();
    for pair in pairs {
        let (old, new) = pair
            .split_once('=')
            .ok_or_else(|| stop(EXIT_REFUSED, format!("--map {pair:?}: expected <old key>=<new key>")))?;
        if !key(old) || !key(new) {
            return Err(stop(EXIT_REFUSED, format!("--map {pair:?}: both keys must match ^p[0-9a-f]{{16}}$")));
        }
        if map.insert(old.to_string(), new.to_string()).is_some() {
            return Err(stop(EXIT_REFUSED, format!("--map names the old key {old} twice")));
        }
    }
    Ok(map)
}

fn mapped<'a>(map: &'a HashMap<String, String>, key: &'a str) -> &'a str {
    map.get(key).map(String::as_str).unwrap_or(key)
}

/// One forum row of `brain.r8`, as the columns of `forum_posts`.
struct ForumRow {
    id: String,
    body: String,
    origin_project: String,
    thread_id: String,
    reply_to: Option<String>,
    title: String,
    actor: String,
    group: String,
    parent: Option<String>,
    request_key: String,
    retention_days: i64,
    digest: String,
    /// The envelope reads, its digest is the one recomputed, and the id is the digest's.
    sound: bool,
    /// `record.timestamp_ms`, or the row's `created` when the record has none.
    timestamp_ms: Option<i64>,
    raw: String,
}

struct SynapsSource {
    forum: Vec<ForumRow>,
    /// Forum rows whose `id` or `content` is not text: they cannot be a `forum_posts` row.
    forum_unreadable: Vec<String>,
    notes: i64,
    foreign: Vec<String>,
    /// `(id, project_key, deleted_ms)` of the `msg-` tombstones.
    forum_tombstones: Vec<(String, String, Option<i64>)>,
    note_tombstones: i64,
    archives: i64,
    tables_missing: Vec<&'static str>,
}

fn read_synaps(conn: &Connection, path: &Path) -> Res<SynapsSource> {
    let not_brain = |e: String| stop(EXIT_SOURCE, format!("{} is not a readable Synaps brain: {e}", path.display()));
    if !has_table(conn, "main", "memories").map_err(|e| not_brain(e.to_string()))? {
        return Err(not_brain("it has no table memories".into()));
    }
    let mut out = SynapsSource {
        forum: Vec::new(),
        forum_unreadable: Vec::new(),
        notes: 0,
        foreign: Vec::new(),
        forum_tombstones: Vec::new(),
        note_tombstones: 0,
        archives: 0,
        tables_missing: Vec::new(),
    };

    if has_table(conn, "main", "memory_tombstones")? {
        let mut stmt = conn.prepare("SELECT id, project_key, deleted_at FROM memory_tombstones ORDER BY id")?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let id = Cell::of(r.get_ref(0)?);
            match id.text() {
                Some(id) if id.starts_with("msg-") => {
                    let project = Cell::of(r.get_ref(1)?).text().unwrap_or("").to_string();
                    let deleted = Cell::of(r.get_ref(2)?).text().and_then(parse_time);
                    out.forum_tombstones.push((id.to_string(), project, deleted));
                }
                _ => out.note_tombstones += 1,
            }
        }
    } else {
        out.tables_missing.push("memory_tombstones");
    }
    if has_table(conn, "main", "synaps_history")? {
        out.archives = count(conn, "main", "synaps_history")?;
    } else {
        out.tables_missing.push("synaps_history");
    }

    let mut stmt = conn.prepare("SELECT * FROM memories ORDER BY id")?;
    let names: Vec<String> = stmt.column_names().iter().map(|s| s.to_string()).collect();
    let at = |name: &str| names.iter().position(|n| n == name);
    let (Some(id_at), Some(content_at)) = (at("id"), at("content")) else {
        return Err(not_brain("memories has no id or no content column".into()));
    };
    let (provenance_at, project_at, created_at) = (at("provenance"), at("project_key"), at("created"));
    let mut rows = stmt.query([])?;
    while let Some(r) = rows.next()? {
        let cells: Vec<Cell> = (0..names.len()).map(|i| r.get_ref(i).map(Cell::of)).collect::<rusqlite::Result<_>>()?;
        let id = &cells[id_at];
        let provenance: Option<Value> =
            provenance_at.and_then(|i| cells[i].text()).and_then(|t| serde_json::from_str(t).ok());
        let record = provenance.as_ref().and_then(|p| p.get("record"));
        let envelope = record.and_then(|r| r.get("meta")).and_then(|m| m.get(forum::META_KEY));
        // The helper's own test (`sidecars/axel-memory-service/src/forum.rs:13-20`): any one
        // marker reserves the row for the forum.
        let msg = |v: Option<&str>| v.is_some_and(|s| s.starts_with("msg-"));
        let is_forum = record.and_then(|r| r.get("namespace")).and_then(Value::as_str) == Some(forum::NAMESPACE)
            || msg(id.text())
            || msg(record.and_then(|r| r.get("id")).and_then(Value::as_str))
            || envelope.is_some();
        if !is_forum {
            let synaps = provenance.as_ref().and_then(|p| p.get("schema")).and_then(Value::as_str) == Some(RECORD_SCHEMA);
            if synaps {
                out.notes += 1;
            } else {
                out.foreign.push(id.show());
            }
            continue;
        }
        let (Some(id), Some(body)) = (id.text(), cells[content_at].text()) else {
            out.forum_unreadable.push(id.show());
            continue;
        };
        let raw: Map<String, Value> = names.iter().cloned().zip(cells.iter().map(Cell::json)).collect();
        let raw = Value::Object(raw).to_string();
        let record_ms = record
            .and_then(|r| r.get("timestamp_ms"))
            .and_then(Value::as_u64)
            .and_then(|n| i64::try_from(n).ok());
        let created_ms = created_at.and_then(|i| cells[i].text()).and_then(parse_time).filter(|n| *n >= 0);
        let column_project = project_at.and_then(|i| cells[i].text()).unwrap_or("").to_string();
        let parsed: Option<Envelope> = envelope.and_then(|e| serde_json::from_value(e.clone()).ok());
        let row = match parsed {
            Some(e) => ForumRow {
                // The digest is recomputed with `Envelope::new(origin_project, author, post)`.
                sound: record_ms.is_some() && e.id() == id && e.validate(body).is_ok(),
                id: id.to_string(),
                body: body.to_string(),
                origin_project: e.project,
                thread_id: e.thread_id,
                reply_to: e.reply_to,
                title: e.title,
                actor: e.author.actor,
                group: e.author.group,
                parent: e.author.parent,
                request_key: e.request_key,
                retention_days: i64::from(e.retention_days),
                digest: e.digest,
                timestamp_ms: record_ms.or(created_ms),
                raw,
            },
            // SPEC: 1441 keeps a row whose digest does not match, flagged. A row whose
            // envelope does not read at all is kept the same way: what its envelope holds
            // as strings, '' for the rest, `digest_ok = 0`, and the whole source row in
            // `raw`. No read lists it.
            None => {
                let s = |name: &str| envelope.and_then(|e| e.get(name)).and_then(Value::as_str).map(str::to_string);
                let author =
                    |name: &str| envelope.and_then(|e| e.get("author")).and_then(|a| a.get(name)).and_then(Value::as_str);
                ForumRow {
                    sound: false,
                    id: id.to_string(),
                    body: body.to_string(),
                    origin_project: s("project").unwrap_or(column_project),
                    thread_id: s("thread_id").unwrap_or_else(|| id.to_string()),
                    reply_to: s("reply_to"),
                    title: s("title").unwrap_or_default(),
                    actor: author("actor").unwrap_or("").to_string(),
                    group: author("group").unwrap_or("").to_string(),
                    parent: author("parent").map(str::to_string),
                    request_key: s("request_key").unwrap_or_default(),
                    retention_days: envelope
                        .and_then(|e| e.get("retention_days"))
                        .and_then(Value::as_i64)
                        .filter(|n| (0..=365).contains(n))
                        .unwrap_or(0),
                    digest: s("digest").unwrap_or_default(),
                    timestamp_ms: record_ms.or(created_ms),
                    raw,
                }
            }
        };
        out.forum.push(row);
    }
    Ok(out)
}

fn open_synaps(path: &Path) -> Res<Connection> {
    brain_v2::open_source(path).map_err(|e| stop(EXIT_SOURCE, e.to_string()))
}

/// Forum posts and forum tombstones of `brain.r8` → the new brain.
pub fn synaps(brain: &BrainArg, opts: &SynapsOpts) -> Outcome {
    let mut rep = Report::new();
    let result = synaps_steps(brain, opts, &mut rep);
    finish("migrate synaps", rep, None, result)
}

fn synaps_steps(brain: &BrainArg, opts: &SynapsOpts, rep: &mut Report) -> Res<u8> {
    let map = parse_map(&opts.map)?;
    let (dest, sources) = existing_destination(brain, &[(opts.from.as_path(), "--from")])?;
    let source = &sources[0];
    rep.insert("destination".into(), json!(shown(&dest)));

    let before = SourceStat::read(source)?;
    let read = {
        let conn = open_synaps(source)?;
        read_synaps(&conn, source)?
    };
    rep.insert("source".into(), json!({ "path": shown(source), "size": before.main.size, "sha256": before.main.sha256 }));

    let conn = brain_v2::connect(&dest, Role::Cli)?;
    let source_text = shown(source);
    let sha = before.main.sha256.clone();
    let now = now_ms();
    let done = in_write(&conn, || {
        // Tombstones first: a forum forget made in the old helper is carried (C-7).
        let mut tombstones_new = 0i64;
        let mut removed = Vec::new();
        for (id, project, deleted) in &read.forum_tombstones {
            tombstones_new += conn.execute(
                "INSERT OR IGNORE INTO forum_tombstones(id, project_key, deleted_ms, source)
                 VALUES (?1, ?2, ?3, 'import')",
                params![id, mapped(&map, project), deleted.unwrap_or(now)],
            )? as i64;
            // SPEC: 1440 does not say what happens to a post that an earlier run imported
            // and the source has forgotten since. He asked for it to be gone: it goes, and
            // the report names it.
            if conn.execute("DELETE FROM forum_posts WHERE id = ?1", [id])? > 0 {
                removed.push(id.clone());
            }
        }
        if !removed.is_empty() {
            index_archives::mark_purged(&conn)?;
        }

        let mut imported = 0i64;
        let mut present = 0i64;
        let mut rehomed = Vec::new();
        let mut flagged = Vec::new();
        let mut conflicts = Vec::new();
        let mut skipped = Vec::new();
        let mut clamped = Vec::new();
        for row in &read.forum {
            let tombstoned: bool =
                conn.query_row("SELECT EXISTS(SELECT 1 FROM forum_tombstones WHERE id = ?1)", [&row.id], |r| r.get(0))?;
            if tombstoned {
                skipped.push(row.id.clone());
                continue;
            }
            let project_key = mapped(&map, &row.origin_project);
            let existing: Option<(Cell, Cell)> = conn
                .query_row("SELECT body, project_key FROM forum_posts WHERE id = ?1", [&row.id], |r| {
                    Ok((Cell::of(r.get_ref(0)?), Cell::of(r.get_ref(1)?)))
                })
                .optional()?;
            if let Some((body, key)) = existing {
                if body != Cell::Text(row.body.as_bytes().to_vec()) {
                    // Left alone.
                    conflicts.push(row.id.clone());
                    continue;
                }
                present += 1;
                if key.text() != Some(project_key) {
                    conn.execute("UPDATE forum_posts SET project_key = ?2 WHERE id = ?1", params![row.id, project_key])?;
                    rehomed.push(row.id.clone());
                }
                continue;
            }
            // RULING 1 of the task: a date later than now is stored as now, so that it does
            // not date every later post of the project after it; `raw` keeps the original.
            let given = row.timestamp_ms.unwrap_or(now);
            let timestamp = given.min(now);
            if given > now {
                clamped.push(row.id.clone());
            }
            let sound = row.sound && row.timestamp_ms.is_some();
            conn.execute(
                "INSERT OR IGNORE INTO forum_posts(id, project_key, origin_project, thread_id, reply_to, title, body,
                    author_actor, author_group, author_parent, request_key, retention_days, digest,
                    timestamp_ms, expires_ms, origin, digest_ok, raw)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, 'brain.r8', ?16, ?17)",
                params![
                    row.id,
                    project_key,
                    row.origin_project,
                    row.thread_id,
                    row.reply_to,
                    row.title,
                    row.body,
                    row.actor,
                    row.group,
                    row.parent,
                    row.request_key,
                    row.retention_days,
                    row.digest,
                    timestamp,
                    timestamp.saturating_add(row.retention_days.saturating_mul(DAY_MS)),
                    sound,
                    row.raw,
                ],
            )?;
            imported += 1;
            if !sound {
                flagged.push(row.id.clone());
            }
        }
        let bad: i64 = conn.query_row("SELECT COUNT(*) FROM forum_posts WHERE digest_ok = 0", [], |r| r.get(0))?;

        let tombstones_in = read.forum_tombstones.len() as i64;
        log_row(
            &conn,
            &source_text,
            &sha,
            "forum_tombstones",
            tombstones_in,
            tombstones_new,
            tombstones_in - tombstones_new,
            0,
            0,
            &json!({ "posts_removed": removed }).to_string(),
        )?;
        log_row(
            &conn,
            &source_text,
            &sha,
            "forum_posts",
            (read.forum.len() + read.forum_unreadable.len()) as i64,
            imported,
            present,
            (flagged.len() + conflicts.len()) as i64,
            (skipped.len() + read.forum_unreadable.len()) as i64,
            &json!({
                "skipped": skipped, "flagged": flagged, "conflicts": conflicts, "rehomed": rehomed,
                "dates_clamped": clamped, "unreadable": read.forum_unreadable, "map": opts.map,
            })
            .to_string(),
        )?;
        for (item, n) in [
            ("notes", read.notes),
            ("note_tombstones", read.note_tombstones),
            ("archives", read.archives),
            ("foreign", read.foreign.len() as i64),
        ] {
            log_row(&conn, &source_text, &sha, item, n, 0, 0, 0, n, r#"{"left_in":"brain.r8"}"#)?;
        }

        // The source is as it was when the first byte was read, or nothing is kept.
        if SourceStat::read(source)? != before {
            return Err(stop(EXIT_SOURCE, "the source changed while it was read; nothing was imported"));
        }

        rep.insert(
            "forum_tombstones".into(),
            json!({ "rows_in": tombstones_in, "rows_imported": tombstones_new, "rows_present": tombstones_in - tombstones_new }),
        );
        rep.insert("posts_removed_by_source_tombstone".into(), id_list(&removed));
        rep.insert(
            "forum_posts".into(),
            json!({
                "rows_in": read.forum.len() + read.forum_unreadable.len(),
                "rows_imported": imported,
                "rows_present": present,
                "rows_flagged": flagged.len() + conflicts.len(),
                "rows_skipped": skipped.len(),
            }),
        );
        rep.insert("rows_skipped".into(), id_list(&skipped));
        rep.insert("rows_flagged".into(), id_list(&flagged));
        rep.insert("rows_conflicting".into(), id_list(&conflicts));
        rep.insert("rows_rehomed".into(), id_list(&rehomed));
        rep.insert("forum_dates_clamped".into(), id_list(&clamped));
        rep.insert("forum_unreadable".into(), id_list(&read.forum_unreadable));
        rep.insert("digest_not_ok_in_brain".into(), json!(bad));
        Ok(())
    });
    done?;

    rep.insert(
        "left_for_move_sh".into(),
        json!({ "notes": read.notes, "note_tombstones": read.note_tombstones, "archives": read.archives }),
    );
    eprintln!(
        "left for move.sh: notes {}, note tombstones {}, archives {}",
        read.notes, read.note_tombstones, read.archives
    );
    rep.insert("foreign_memories".into(), id_list(&read.foreign));
    eprintln!("foreign_memories: {}, left in brain.r8", read.foreign.len());
    rep.insert("accept_foreign".into(), json!(opts.accept_foreign));
    if !read.tables_missing.is_empty() {
        rep.insert("source_tables_missing".into(), json!(read.tables_missing));
    }
    if !read.forum_unreadable.is_empty() {
        return Err(stop(
            EXIT_SHORTFALL,
            format!(
                "{} forum row(s) of the source have an id or a body that is not text and were not imported",
                read.forum_unreadable.len()
            ),
        ));
    }
    if !read.foreign.is_empty() && !opts.accept_foreign {
        return Err(stop(
            EXIT_SHORTFALL,
            format!(
                "foreign_memories: {}, left in brain.r8 (rows that are neither forum posts nor Synaps notes). \
                 The forum rows were imported. Pass --accept-foreign to accept that they stay behind.",
                read.foreign.len()
            ),
        ));
    }
    Ok(EXIT_OK)
}

// ── Reading axel.r8 for verify and catch-up ─────────────────────────────────

/// The old brain, readable without touching it: the file itself when it has no log, else
/// a checked copy beside the destination (steps 3 and 4 of `init`).
struct AxelSource {
    /// The file to open: the source, or the copy.
    db: PathBuf,
    stat: SourceStat,
    copied: bool,
}

fn axel_source(source: &Path, dest: &Path, prefix: &str, rep: &mut Report, temp: &mut Option<Temp>) -> Res<AxelSource> {
    let stat = SourceStat::read(source)?;
    let found = if stat.has_log() {
        let folder = folder_of(dest);
        clean_stale(&folder, rep);
        let copy = checked_copy(source, &folder, prefix, temp)?;
        AxelSource { db: copy.db, stat: copy.stat, copied: true }
    } else {
        AxelSource { db: source.to_path_buf(), stat, copied: false }
    };
    rep.insert("source".into(), found.stat.json(source));
    rep.insert("source_read_from".into(), json!(if found.copied { "a checked temporary copy" } else { "the file itself, immutable" }));
    Ok(found)
}

fn must_be_axel(conn: &Connection, schema: &str, source: &Path) -> Res<()> {
    for table in ["memories", "brain_meta"] {
        let there = has_table(conn, schema, table)
            .map_err(|e| stop(EXIT_SOURCE, format!("{} cannot be read as a database: {e}", source.display())))?;
        if !there {
            return Err(stop(EXIT_SOURCE, format!("{} has no table {table}: not an Axel brain", source.display())));
        }
    }
    Ok(())
}

/// The count `migrate init` recorded for a table, when there is one.
fn recorded_count(conn: &Connection, table: &str) -> rusqlite::Result<Option<i64>> {
    conn.query_row(
        "SELECT rows_in FROM import_log WHERE item = ?1 ORDER BY id DESC LIMIT 1",
        [format!("count:{table}")],
        |r| r.get(0),
    )
    .optional()
}

// ── migrate verify ──────────────────────────────────────────────────────────

struct Items {
    list: Vec<Value>,
    failed: bool,
}

impl Items {
    fn add(&mut self, item: &str, ok: Option<bool>, line: String, mut more: Value) {
        let mark = match ok {
            Some(true) => "ok  ",
            Some(false) => "FAIL",
            None => "note",
        };
        eprintln!("{mark} {item}: {line}");
        if ok == Some(false) {
            self.failed = true;
        }
        if let Some(map) = more.as_object_mut() {
            map.insert("item".into(), json!(item));
            map.insert("ok".into(), json!(ok));
            map.insert("line".into(), json!(line));
        }
        self.list.push(more);
    }
}

/// Compares the new brain with both sources. Writes nothing to the brain.
pub fn verify(brain: &BrainArg, opts: &VerifyOpts) -> Outcome {
    let mut rep = Report::new();
    let mut temp = None;
    let result = verify_steps(brain, opts, &mut rep, &mut temp);
    finish("migrate verify", rep, temp, result)
}

/// The rows of `table` in `src` whose key is not in `dst`.
fn missing_keys(src: &Connection, dst: &Connection, table: &str, keys: &[String]) -> Res<Vec<String>> {
    let list = keys.iter().map(|k| q(k)).collect::<Vec<_>>().join(", ");
    let mut read = src.prepare(&format!("SELECT {list} FROM {}", q(table)))?;
    let dst_has = has_table(dst, "main", table)?;
    let test = keys.iter().enumerate().map(|(i, k)| format!("{} IS ?{}", q(k), i + 1)).collect::<Vec<_>>().join(" AND ");
    let mut find = if dst_has { Some(dst.prepare(&format!("SELECT 1 FROM {} WHERE {test}", q(table)))?) } else { None };
    let mut missing = Vec::new();
    let mut rows = read.query([])?;
    while let Some(r) = rows.next()? {
        let key: Vec<Cell> = (0..keys.len()).map(|i| r.get_ref(i).map(Cell::of)).collect::<rusqlite::Result<_>>()?;
        let there = match find.as_mut() {
            Some(find) => find.exists(params_from_iter(key.iter()))?,
            None => false,
        };
        if !there {
            missing.push(key.iter().map(Cell::show).collect::<Vec<_>>().join("/"));
        }
    }
    Ok(missing)
}

fn verify_steps(brain: &BrainArg, opts: &VerifyOpts, rep: &mut Report, temp: &mut Option<Temp>) -> Res<u8> {
    let map = parse_map(&opts.map)?;
    let (dest, sources) =
        existing_destination(brain, &[(opts.from_axel.as_path(), "--from-axel"), (opts.from_synaps.as_path(), "--from-synaps")])?;
    let (axel_path, synaps_path) = (&sources[0], &sources[1]);
    rep.insert("destination".into(), json!(shown(&dest)));

    let found = axel_source(axel_path, &dest, ".verify-", rep, temp)?;
    let src = open_synaps(&found.db)?;
    must_be_axel(&src, "main", axel_path)?;
    let synaps_stat = SourceStat::read(synaps_path)?;
    let synaps_read = {
        let conn = open_synaps(synaps_path)?;
        read_synaps(&conn, synaps_path)?
    };
    rep.insert(
        "source_synaps".into(),
        json!({ "path": shown(synaps_path), "size": synaps_stat.main.size, "sha256": synaps_stat.main.sha256 }),
    );

    // Read-only: nothing is written to the brain.
    let dst = Connection::open_with_flags(
        file_uri(&dest, "mode=ro"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    dst.busy_timeout(Duration::from_secs(5))?;
    dst.execute_batch("PRAGMA query_only=ON")?;

    let mut items = Items { list: Vec::new(), failed: false };

    // Saved: memories, by id and by the bytes of the text.
    {
        let mut read = src.prepare("SELECT id, content FROM memories")?;
        let mut find = dst.prepare("SELECT content FROM memories WHERE id IS ?1")?;
        let mut tomb = dst.prepare("SELECT 1 FROM memory_tombstones WHERE id IS ?1")?;
        let (mut total, mut same, mut tombstoned) = (0i64, 0i64, 0i64);
        let (mut missing, mut changed) = (Vec::new(), Vec::new());
        let mut rows = read.query([])?;
        while let Some(r) = rows.next()? {
            total += 1;
            let id = Cell::of(r.get_ref(0)?);
            let content = Cell::of(r.get_ref(1)?);
            let there: Option<Cell> = find.query_row([&id], |d| Ok(Cell::of(d.get_ref(0)?))).optional()?;
            match there {
                Some(c) if c == content => same += 1,
                _ if tomb.exists([&id])? => tombstoned += 1,
                Some(_) => changed.push(id.show()),
                None => missing.push(id.show()),
            }
        }
        let ok = missing.is_empty() && changed.is_empty();
        items.add(
            "memories",
            Some(ok),
            format!(
                "{total} in the source: {same} present with the same text, {tombstoned} forgotten in the new brain, \
                 {} missing{}, {} with different text{}",
                missing.len(),
                names(&missing),
                changed.len(),
                names(&changed)
            ),
            json!({ "source": total, "same": same, "tombstoned": tombstoned,
                    "missing": id_list(&missing), "changed": id_list(&changed) }),
        );
    }

    // Saved: every primary key of the three tables.
    for table in SAVED_TABLES {
        if !has_table(&src, "main", table)? {
            items.add(table, None, "not in the source".into(), json!({}));
            continue;
        }
        let keys = key_columns(&src, "main", table)?;
        let missing = missing_keys(&src, &dst, table, &keys)?;
        items.add(
            table,
            Some(missing.is_empty()),
            format!("{} row(s) of the source missing{}", missing.len(), names(&missing)),
            json!({ "key": keys, "missing": id_list(&missing) }),
        );
    }
    // The handoff: the same text, or a later one in the new brain.
    if has_table(&src, "main", "context_data")? {
        let sql = "SELECT value, updated FROM context_data WHERE layer = 'session' AND key = 'handoff'";
        let read = |c: &Connection| -> rusqlite::Result<Option<(Cell, Cell)>> {
            c.query_row(sql, [], |r| Ok((Cell::of(r.get_ref(0)?), Cell::of(r.get_ref(1)?)))).optional()
        };
        let old = read(&src)?;
        let new = if has_table(&dst, "main", "context_data")? { read(&dst)? } else { None };
        let (ok, line) = match (&old, &new) {
            (None, _) => (true, "the source has none".to_string()),
            (Some(_), None) => (false, "the handoff of the source is missing".to_string()),
            (Some((a, _)), Some((b, _))) if a == b => (true, "the same text".to_string()),
            (Some((_, at_old)), Some((_, at_new))) => {
                match (at_old.text().and_then(parse_time), at_new.text().and_then(parse_time)) {
                    (Some(o), Some(n)) if n > o => (true, "the new brain has a later handoff".to_string()),
                    (Some(_), Some(_)) => (false, "the handoff of the source differs and is not older".to_string()),
                    _ => (false, "the handoff differs and its times cannot be compared".to_string()),
                }
            }
        };
        items.add("handoff", Some(ok), line, json!({}));
    }

    // Copies of files: every doc_id.
    if has_table(&src, "main", "documents")? {
        let missing = missing_keys(&src, &dst, "documents", &["doc_id".to_string()])?;
        // SPEC: 1460 says "every doc_id is present", and test 16 that a memory purged in the
        // new brain does not fail `verify`: the purge deleted its shadow document too. A
        // doc_id that is a tombstoned memory's is therefore not a shortfall.
        let mut tomb = dst.prepare("SELECT 1 FROM memory_tombstones WHERE id = ?1")?;
        let mut really = Vec::new();
        let mut forgotten = 0;
        for id in missing {
            if tomb.exists([&id])? {
                forgotten += 1;
            } else {
                really.push(id);
            }
        }
        items.add(
            "documents",
            Some(really.is_empty()),
            format!(
                "{} document(s) of the source missing{}; {forgotten} are the shadow of a forgotten memory",
                really.len(),
                names(&really)
            ),
            json!({ "missing": id_list(&really), "shadows_of_forgotten": forgotten }),
        );
    }

    // Usage and derived: at least the count `init` recorded.
    let new_tables: BTreeSet<String> = plain_tables(&dst, "main")?.into_iter().collect();
    for table in plain_tables(&src, "main")? {
        if NOT_USAGE.contains(&table.as_str()) {
            continue;
        }
        let in_source = count(&src, "main", &table)?;
        if !new_tables.contains(&table) {
            items.add(
                &table,
                Some(in_source == 0),
                format!("the new brain has no such table; the source has {in_source} row(s)"),
                json!({ "source": in_source }),
            );
            continue;
        }
        let in_new = count(&dst, "main", &table)?;
        let recorded = recorded_count(&dst, &table)?;
        let bar = recorded.unwrap_or(in_source);
        let gained = (in_source - bar).max(0);
        let mut line = format!(
            "{in_new} row(s), {bar} needed ({})",
            if recorded.is_some() { "recorded by migrate init" } else { "the source's count: no init record" }
        );
        if gained > 0 {
            line.push_str(&format!(
                "; source has {gained} more rows since the import (searches, sessions and graph links written \
                 by the old plugin); not carried"
            ));
        }
        items.add(
            &table,
            Some(in_new >= bar),
            line,
            json!({ "class": "usage", "new_brain": in_new, "recorded": recorded, "source": in_source, "source_gained": gained }),
        );
    }

    // The new brain itself.
    {
        let mut per: HashMap<Cell, i64> = HashMap::new();
        let mut stmt = dst.prepare("SELECT mem_id, COUNT(*) FROM memories_fts GROUP BY mem_id")?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            per.insert(Cell::of(r.get_ref(0)?), r.get(1)?);
        }
        let mut ids: HashSet<Cell> = HashSet::new();
        let mut wrong = Vec::new();
        let mut stmt = dst.prepare("SELECT id FROM memories")?;
        let mut rows = stmt.query([])?;
        while let Some(r) = rows.next()? {
            let id = Cell::of(r.get_ref(0)?);
            if per.get(&id).copied().unwrap_or(0) != 1 {
                wrong.push(id.show());
            }
            ids.insert(id);
        }
        let stray: Vec<String> = per.keys().filter(|k| !ids.contains(*k)).map(Cell::show).collect();
        items.add(
            "memories_fts",
            Some(wrong.is_empty() && stray.is_empty()),
            format!(
                "{} memories without exactly one word row{}; {} word row(s) of no memory",
                wrong.len(),
                names(&wrong),
                stray.len()
            ),
            json!({ "not_one": id_list(&wrong), "stray": id_list(&stray) }),
        );
        let mut stmt = dst.prepare(
            "SELECT m.id FROM memories m WHERE NOT EXISTS (SELECT 1 FROM memory_ext e WHERE e.mem_id = m.id)",
        )?;
        let bare: Vec<String> =
            stmt.query_map([], |r| Ok(Cell::of(r.get_ref(0)?).show()))?.collect::<rusqlite::Result<_>>()?;
        items.add(
            "memory_ext",
            Some(bare.is_empty()),
            format!("{} memories without a memory_ext row{}", bare.len(), names(&bare)),
            json!({ "missing": id_list(&bare) }),
        );
    }

    // The forum.
    {
        let mut find = dst.prepare("SELECT body, project_key FROM forum_posts WHERE id = ?1")?;
        let mut tomb = dst.prepare("SELECT 1 FROM forum_tombstones WHERE id = ?1")?;
        let (mut same, mut tombstoned) = (0i64, 0i64);
        let (mut missing, mut changed, mut elsewhere) = (Vec::new(), Vec::new(), Vec::new());
        for row in &synaps_read.forum {
            let there: Option<(Cell, Cell)> = find
                .query_row([&row.id], |r| Ok((Cell::of(r.get_ref(0)?), Cell::of(r.get_ref(1)?))))
                .optional()?;
            let key = mapped(&map, &row.origin_project);
            match there {
                Some((body, at)) if body == Cell::Text(row.body.as_bytes().to_vec()) && at.text() == Some(key) => {
                    same += 1
                }
                _ if tomb.exists([&row.id])? => tombstoned += 1,
                Some((body, _)) if body != Cell::Text(row.body.as_bytes().to_vec()) => changed.push(row.id.clone()),
                Some(_) => elsewhere.push(row.id.clone()),
                None => missing.push(row.id.clone()),
            }
        }
        let unreadable = &synaps_read.forum_unreadable;
        let ok = missing.is_empty() && changed.is_empty() && elsewhere.is_empty() && unreadable.is_empty();
        items.add(
            "forum_posts",
            Some(ok),
            format!(
                "{} in brain.r8: {same} present with the same body under the mapped key, {tombstoned} forgotten, \
                 {} missing{}, {} with a different body{}, {} under another project key{}, {} unreadable in the source",
                synaps_read.forum.len() + unreadable.len(),
                missing.len(),
                names(&missing),
                changed.len(),
                names(&changed),
                elsewhere.len(),
                names(&elsewhere),
                unreadable.len()
            ),
            json!({ "same": same, "tombstoned": tombstoned, "missing": id_list(&missing), "changed": id_list(&changed),
                    "wrong_project": id_list(&elsewhere), "unreadable": id_list(unreadable) }),
        );
        let bad: i64 = dst.query_row("SELECT COUNT(*) FROM forum_posts WHERE digest_ok = 0", [], |r| r.get(0))?;
        items.add("forum_digest", None, format!("{bad} forum post(s) with digest_ok = 0 (kept, never listed)"), json!({ "digest_not_ok": bad }));
        let foreign = synaps_read.foreign.len();
        items.add(
            "foreign_memories",
            Some(foreign == 0 || opts.accept_foreign),
            format!(
                "{foreign}, left in brain.r8{}",
                if foreign > 0 && opts.accept_foreign { " (accepted: --accept-foreign)" } else { "" }
            ),
            json!({ "foreign": id_list(&synaps_read.foreign), "accepted": opts.accept_foreign }),
        );
        items.add(
            "notes_and_archives",
            None,
            format!(
                "notes {}, note tombstones {}, archives {}: checked by verify-move.py",
                synaps_read.notes, synaps_read.note_tombstones, synaps_read.archives
            ),
            json!({ "notes": synaps_read.notes, "note_tombstones": synaps_read.note_tombstones, "archives": synaps_read.archives }),
        );
    }

    // Neither source moved while it was read.
    if !found.copied && SourceStat::read(axel_path)? != found.stat {
        return Err(stop(EXIT_SOURCE, "--from-axel changed while it was read; nothing of this verify may be used"));
    }
    if SourceStat::read(synaps_path)? != synaps_stat {
        return Err(stop(EXIT_SOURCE, "--from-synaps changed while it was read; nothing of this verify may be used"));
    }

    let failed = items.failed;
    let shortfalls: Vec<&str> = items
        .list
        .iter()
        .filter(|i| i.get("ok") == Some(&json!(false)))
        .filter_map(|i| i.get("item").and_then(Value::as_str))
        .collect();
    rep.insert("shortfalls".into(), json!(shortfalls));
    rep.insert("items".into(), json!(items.list));
    if failed {
        return Err(stop(EXIT_SHORTFALL, "something that was saved is not in the new brain; see the items marked FAIL"));
    }
    Ok(EXIT_OK)
}

/// The first few names of a list, for a line.
fn names(ids: &[String]) -> String {
    if ids.is_empty() {
        return String::new();
    }
    let first: Vec<&str> = ids.iter().take(5).map(String::as_str).collect();
    format!(" ({}{})", first.join(", "), if ids.len() > 5 { ", …" } else { "" })
}

// ── migrate catch-up ────────────────────────────────────────────────────────

/// Brings over what was saved in the old brain after the import.
pub fn catch_up(brain: &BrainArg, opts: &CatchUpOpts) -> Outcome {
    let mut rep = Report::new();
    let mut temp = None;
    let result = catch_up_steps(brain, opts, &mut rep, &mut temp);
    finish("migrate catch-up", rep, temp, result)
}

fn joined(names: &[String]) -> String {
    names.iter().map(|n| q(n)).collect::<Vec<_>>().join(", ")
}

fn id_column(conn: &Connection, sql: &str) -> rusqlite::Result<Vec<String>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map([], |r| Ok(Cell::of(r.get_ref(0)?).show()))?;
    rows.collect()
}

fn catch_up_steps(brain: &BrainArg, opts: &CatchUpOpts, rep: &mut Report, temp: &mut Option<Temp>) -> Res<u8> {
    let (dest, sources) = existing_destination(brain, &[(opts.from_axel.as_path(), "--from-axel")])?;
    let source = &sources[0];
    rep.insert("destination".into(), json!(shown(&dest)));
    if let Some(watch) = &opts.watch {
        source_file(watch, "--watch")?;
    }
    let found = axel_source(source, &dest, ".catchup-", rep, temp)?;
    let watch = watch_detail(opts.watch.as_deref().unwrap_or(source.as_path()))?;

    let conn = brain_v2::connect(&dest, Role::Cli)?;
    // Attached read-only and immutable: no lock is taken on it and no file is made beside it.
    conn.execute("ATTACH DATABASE ?1 AS src", [file_uri(&found.db, "mode=ro&immutable=1")])
        .map_err(|e| stop(EXIT_SOURCE, format!("{} cannot be attached: {e}", source.display())))?;
    must_be_axel(&conn, "src", source)?;

    // The column lists, before anything is changed.
    let mut differ = Vec::new();
    let mut lists: HashMap<&str, Vec<String>> = HashMap::new();
    for table in ["memories", "documents", "context_data", "staged_memories", "patterns"] {
        if !has_table(&conn, "src", table)? {
            continue;
        }
        let old = column_names(&conn, "src", table)?;
        let new = if has_table(&conn, "main", table)? { column_names(&conn, "main", table)? } else { Vec::new() };
        // SPEC: 1485 compares the lists of the three small tables. `memories` and
        // `documents` got columns in the new brain that the old file has not: for them
        // every column of the source must exist in the new brain, and `memories` must have
        // the 15 original ones.
        let same = match table {
            "memories" => {
                old.iter().all(|c| new.contains(c)) && MEMORY_COLUMNS.iter().all(|c| old.iter().any(|o| o == c))
            }
            "documents" => old.iter().all(|c| new.contains(c)) && old.iter().any(|c| c == "doc_id"),
            _ => {
                let (mut a, mut b) = (old.clone(), new.clone());
                a.sort();
                b.sort();
                a == b
            }
        };
        if !same {
            differ.push(json!({ "table": table, "source": old, "new_brain": new }));
        }
        lists.insert(table, old);
    }
    if !differ.is_empty() {
        rep.insert("columns_differ".into(), json!(differ));
        return Err(stop(EXIT_SHORTFALL, "a table's columns differ between the source and the new brain; nothing was changed"));
    }

    let source_text = shown(source);
    let sha = found.stat.main.sha256.clone();
    let has_docs = lists.contains_key("documents");
    let done = in_write(&conn, || {
        conn.execute_batch(
            "CREATE TEMP TABLE IF NOT EXISTS cu_mem(id PRIMARY KEY);
             CREATE TEMP TABLE IF NOT EXISTS cu_doc(doc_id PRIMARY KEY);
             DELETE FROM temp.cu_mem; DELETE FROM temp.cu_doc;",
        )?;
        // memories: in neither `memories` nor `memory_tombstones` of the new brain. One pass
        // over the source sorts every row: the same text, changed in the source, forgotten
        // here, or new.
        let (mut new_ids, mut not_back, mut changed, mut null_ids) = (Vec::new(), Vec::new(), Vec::new(), 0usize);
        {
            let mut read = conn.prepare("SELECT id, content FROM src.memories")?;
            let mut here = conn.prepare("SELECT content FROM main.memories WHERE id = ?1")?;
            let mut tomb = conn.prepare("SELECT 1 FROM main.memory_tombstones WHERE id = ?1")?;
            let mut take = conn.prepare("INSERT OR IGNORE INTO temp.cu_mem(id) VALUES (?1)")?;
            let mut rows = read.query([])?;
            while let Some(r) = rows.next()? {
                let (id, content) = (Cell::of(r.get_ref(0)?), Cell::of(r.get_ref(1)?));
                if id == Cell::Null {
                    null_ids += 1;
                    continue;
                }
                match here.query_row([&id], |d| Ok(Cell::of(d.get_ref(0)?))).optional()? {
                    Some(c) if c == content => {}
                    Some(_) => changed.push(id.show()),
                    None if tomb.exists([&id])? => not_back.push(id.show()),
                    None => {
                        take.execute([&id])?;
                        new_ids.push(id.show());
                    }
                }
            }
        }
        let cols = joined(&lists["memories"]);
        let memories = conn.execute(
            &format!(
                "INSERT INTO main.memories({cols}) SELECT {cols} FROM src.memories
                  WHERE id IN (SELECT id FROM temp.cu_mem)"
            ),
            [],
        )?;

        // documents: every one the new brain lacks, the shadows of the new memories among
        // them; never the shadow of a memory that is forgotten here.
        let (mut shadows, mut documents, mut chunks) = (0usize, 0usize, 0usize);
        if has_docs {
            let mut read = conn.prepare("SELECT doc_id FROM src.documents WHERE doc_id IS NOT NULL")?;
            let mut here = conn.prepare("SELECT 1 FROM main.documents WHERE doc_id = ?1")?;
            let mut tomb = conn.prepare("SELECT 1 FROM main.memory_tombstones WHERE id = ?1")?;
            let mut take = conn.prepare("INSERT OR IGNORE INTO temp.cu_doc(doc_id) VALUES (?1)")?;
            let mut rows = read.query([])?;
            while let Some(r) = rows.next()? {
                let doc_id = Cell::of(r.get_ref(0)?);
                if !here.exists([&doc_id])? && !tomb.exists([&doc_id])? {
                    take.execute([&doc_id])?;
                }
            }
        }
        if has_docs {
            shadows = conn.query_row(
                "SELECT COUNT(*) FROM temp.cu_doc WHERE doc_id IN (SELECT id FROM src.memories)",
                [],
                |r| r.get::<_, i64>(0),
            )? as usize;
            // The integer key is the new brain's to give.
            let doc_cols: Vec<String> = lists["documents"].iter().filter(|c| *c != "id").cloned().collect();
            let cols = joined(&doc_cols);
            let all = conn.execute(
                &format!(
                    "INSERT INTO main.documents({cols}) SELECT {cols} FROM src.documents
                      WHERE doc_id IN (SELECT doc_id FROM temp.cu_doc)"
                ),
                [],
            )?;
            documents = all - shadows;
            let fts_ok = |schema: &str| -> rusqlite::Result<bool> {
                Ok(conn
                    .query_row(
                        &format!("SELECT EXISTS(SELECT 1 FROM {schema}.sqlite_master WHERE name = 'chunks_fts')"),
                        [],
                        |r| r.get::<_, bool>(0),
                    )?
                    && {
                        let have = column_names(&conn, schema, "chunks_fts")?;
                        ["doc_id", "content", "file_path"].iter().all(|c| have.iter().any(|h| h == c))
                    })
            };
            if fts_ok("src")? && fts_ok("main")? {
                chunks = conn.execute(
                    "INSERT INTO main.chunks_fts(doc_id, content, file_path)
                     SELECT c.doc_id, c.content, c.file_path FROM src.chunks_fts c
                      WHERE c.doc_id IN (SELECT doc_id FROM temp.cu_doc)
                        AND NOT EXISTS (SELECT 1 FROM main.chunks_fts m WHERE m.doc_id = c.doc_id)",
                    [],
                )?;
            }
        }

        // Adoption of the new memories as `legacy`, and their word rows: what
        // `brain_v2::adopt_orphans` does, inside this transaction.
        let model: Option<String> = conn
            .query_row("SELECT json_extract(value, '$.embedder_model') FROM main.brain_meta WHERE key = 'meta'", [], |r| {
                r.get(0)
            })
            .optional()?
            .flatten();
        let mut unreadable = 0usize;
        {
            let mut read = conn.prepare(
                "SELECT m.id, m.content, m.created, d.doc_id IS NOT NULL, d.embedding
                   FROM main.memories m LEFT JOIN main.documents d ON d.doc_id = m.id
                  WHERE m.id IN (SELECT id FROM temp.cu_mem)",
            )?;
            let mut insert = conn.prepare(
                "INSERT OR IGNORE INTO main.memory_ext(mem_id, origin_kind, origin_ref, content_hash, created_ms,
                                                       time_source, embed_model, embedding)
                 VALUES (?1, 'legacy', '', ?2, ?3, ?4, ?5, ?6)",
            )?;
            let mut rows = read.query([])?;
            while let Some(r) = rows.next()? {
                let (id, content) = (Cell::of(r.get_ref(0)?), Cell::of(r.get_ref(1)?));
                let (Some(id), Some(content)) = (id.text(), content.text()) else {
                    unreadable += 1;
                    continue;
                };
                let hash = hex::encode(Sha256::digest(crate::archive::normalise(content).as_bytes()));
                let created = Cell::of(r.get_ref(2)?);
                let (ms, time_source) = match created.text().and_then(parse_time) {
                    Some(ms) => (ms, "exact"),
                    None => (now_ms(), "observed"),
                };
                let shadow: bool = r.get(3)?;
                let embedding = match r.get_ref(4)? {
                    ValueRef::Blob(b) => Some(b.to_vec()),
                    _ => None,
                };
                insert.execute(params![id, hash, ms, time_source, if shadow { model.as_deref() } else { None }, embedding])?;
            }
        }
        conn.execute(
            "INSERT INTO main.memories_fts(mem_id, title, abstract, content)
             SELECT m.id, m.title,
                    CASE WHEN m.sensitivity = 'secret' THEN '' ELSE m.abstract_text END,
                    CASE WHEN m.sensitivity = 'secret' THEN '' ELSE m.content END
               FROM main.memories m
              WHERE m.id IN (SELECT id FROM temp.cu_mem)
                AND NOT EXISTS (SELECT 1 FROM main.memories_fts f WHERE f.mem_id = m.id)
                AND typeof(m.id) = 'text' AND typeof(m.content) = 'text'",
            [],
        )?;

        // The handoff: taken when the source's is the later one.
        let mut handoff = "the source has none";
        if lists.contains_key("context_data") {
            let sql = |schema: &str| {
                format!("SELECT value, updated FROM {schema}.context_data WHERE layer = 'session' AND key = 'handoff'")
            };
            let read = |schema: &str| -> rusqlite::Result<Option<(Cell, Cell)>> {
                conn.query_row(&sql(schema), [], |r| Ok((Cell::of(r.get_ref(0)?), Cell::of(r.get_ref(1)?)))).optional()
            };
            handoff = match (read("src")?, read("main")?) {
                (None, _) => "the source has none",
                (Some(_), None) => "new",
                (Some((a, _)), Some((b, _))) if a == b => "the same",
                (Some((value, at_old)), Some((_, at_new))) => {
                    match (at_old.text().and_then(parse_time), at_new.text().and_then(parse_time)) {
                        (Some(o), Some(n)) if o > n => {
                            conn.execute(
                                "UPDATE main.context_data SET value = ?1, updated = ?2
                                  WHERE layer = 'session' AND key = 'handoff'",
                                params![value, at_old],
                            )?;
                            "taken: the source's is later"
                        }
                        (Some(_), Some(_)) => "kept: the new brain's is not older",
                        _ => "kept: the times cannot be compared",
                    }
                }
            };
        }

        // The rest of what somebody saved, by primary key.
        let mut saved = Map::new();
        for table in SAVED_TABLES {
            let Some(columns) = lists.get(table) else { continue };
            let cols = joined(columns);
            let n = conn.execute(
                &format!("INSERT OR IGNORE INTO main.{t}({cols}) SELECT {cols} FROM src.{t}", t = q(table)),
                [],
            )?;
            saved.insert(table.to_string(), json!(n));
        }

        // Usage and derived: not copied; how many rows each gained.
        let mut not_carried = Map::new();
        let new_tables: BTreeSet<String> = plain_tables(&conn, "main")?.into_iter().collect();
        for table in plain_tables(&conn, "src")? {
            if NOT_USAGE.contains(&table.as_str()) || !new_tables.contains(&table) {
                continue;
            }
            let in_source = count(&conn, "src", &table)?;
            let bar = match recorded_count(&conn, &table)? {
                Some(n) => n,
                None => count(&conn, "main", &table)?,
            };
            let gained = (in_source - bar).max(0);
            if gained > 0 {
                eprintln!("{table}: {gained} rows not carried");
                not_carried.insert(table, json!(gained));
            }
        }

        // A memory forgotten in the old brain since the import is still in the new one.
        // Nothing is deleted here; a person decides.
        let forgotten_there = if has_table(&conn, "src", "memory_tombstones")? {
            id_column(&conn, "SELECT id FROM src.memory_tombstones WHERE id IN (SELECT id FROM main.memories)")?
        } else {
            Vec::new()
        };

        let total = (memories + shadows + documents) as i64;
        log_row(&conn, &source_text, &sha, "catch-up", total, total, 0, changed.len() as i64, not_back.len() as i64, &watch.to_string())?;

        if !found.copied && SourceStat::read(source)? != found.stat {
            return Err(stop(EXIT_SOURCE, "the source changed while it was read; nothing was changed"));
        }

        rep.insert(
            "memories".into(),
            json!({ "brought": memories, "ids": id_list(&new_ids), "unreadable_not_adopted": unreadable,
                    "null_id_not_brought": null_ids }),
        );
        rep.insert("memory_shadow_documents".into(), json!(shadows));
        rep.insert("documents".into(), json!(documents));
        rep.insert("chunks_fts_rows".into(), json!(chunks));
        rep.insert("changed_in_source".into(), id_list(&changed));
        rep.insert("tombstoned_not_brought_back".into(), id_list(&not_back));
        rep.insert("forgotten_in_source_still_in_new_brain".into(), id_list(&forgotten_there));
        rep.insert("handoff".into(), json!(handoff));
        rep.insert("saved_rows".into(), Value::Object(saved));
        rep.insert("not_carried".into(), Value::Object(not_carried));
        rep.insert("watch".into(), watch.clone());
        Ok(())
    });
    done?;
    conn.execute_batch("DETACH DATABASE src")?;
    Ok(EXIT_OK)
}
