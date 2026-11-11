//! `status [--json] [--check]`: the one place a person looks (section 7).
//!
//! It never writes: not to the brain, not to a status file, not to the old brain it
//! `lstat`s. The brain is opened read-only, and with `immutable=1` when there is no
//! `-journal` and no non-empty `-wal` beside it, so no `-wal` or `-shm` appears beside a file
//! that had none (wave 4 rulings, the line for 21); if the file changed during that read it
//! is read again with `mode=ro` (`brain.read_as` says which). Everything it prints that came from a row or a file passes
//! `screen::show`.

use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use chrono::{TimeZone, Utc};
use rusqlite::{Connection, ErrorCode, OpenFlags, OptionalExtension};
use serde_json::{json, Map, Value};

use crate::brain_v2::{text_at, LATEST};
use crate::error::{AxelError, Result};
use crate::ext::config::{default_model_dir, ExtConfig};
use crate::ext::types::BrainArg;
use crate::r8::SCHEMA_VERSION;
use crate::screen;

#[derive(clap::Args)]
pub struct Args {
    /// Output as JSON
    #[arg(long)]
    pub json: bool,
    /// Also check the file's integrity, the word index and the stored vectors
    #[arg(long)]
    pub check: bool,
}

/// A status file or row written longer ago than this says "not running or stuck".
const FRESH_MS: i64 = 120_000;
const DAY_MS: i64 = 86_400_000;
/// Longest text taken from a row or a file.
const FIELD: usize = 400;
/// A status file is a few hundred bytes; anything longer is not read past this.
const STATUS_FILE_MAX: u64 = 64 * 1024;

pub fn run(args: &Args, brain: &BrainArg) -> Result<ExitCode> {
    let now = Utc::now().timestamp_millis();
    let home = dirs::home_dir();
    let report = collect(&brain.path, args.check, now, home.as_deref());
    if args.json {
        println!("{}", report.json);
    } else {
        for line in &report.lines {
            println!("{line}");
        }
    }
    Ok(ExitCode::from(if report.wrong { 1 } else { 0 }))
}

/// What `status` prints, and whether anything is wrong (exit 1).
pub struct Report {
    pub json: Value,
    pub lines: Vec<String>,
    pub wrong: bool,
}

pub fn collect(path: &Path, check: bool, now: i64, home: Option<&Path>) -> Report {
    collect_with(path, check, now, home, &|_: &Path| {})
}

/// `collect`, with `after_reads` called on the file read once every read of the first try
/// is done and before the file is looked at again: a test seam, to change the file there.
pub fn collect_with(
    path: &Path,
    check: bool,
    now: i64,
    home: Option<&Path>,
    after_reads: &dyn Fn(&Path),
) -> Report {
    let data_dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let files = status_files(&data_dir, now);
    // RULING: wave 5 fix round of 21, finding 2: SQLite names the side files after the file
    // a link points to, so they are looked for there.
    let real = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());

    let before = FileState::of(&real);
    if before.journal || before.wal_size > 0 {
        return read_once(path, &real, &files, check, now, home, ReadAs::ReadOnly);
    }
    let first = read_once(path, &real, &files, check, now, home, ReadAs::Immutable);
    after_reads(&real);
    // RULING: wave 5 fix round of 21, finding 2: `immutable=1` takes no lock. A file that
    // changed, or a side file that appeared, while it was read may have been read torn: it
    // is read once more, the ordinary way.
    if FileState::of(&real).changed_since(&before) {
        return read_once(path, &real, &files, check, now, home, ReadAs::AfterAChange);
    }
    first
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadAs {
    Immutable,
    ReadOnly,
    AfterAChange,
}

impl ReadAs {
    fn as_str(self) -> &'static str {
        match self {
            ReadAs::Immutable => "immutable",
            ReadAs::ReadOnly => "read-only",
            ReadAs::AfterAChange => "read-only after a change",
        }
    }
}

/// What tells that the file was written: the main file's (dev, inode, size, mtime_ns), and
/// which side files there are.
#[derive(PartialEq, Eq)]
struct FileState {
    main: Option<(u64, u64, u64, i64)>,
    wal: bool,
    shm: bool,
    journal: bool,
    wal_size: u64,
}

impl FileState {
    fn of(real: &Path) -> FileState {
        let main = real.metadata().ok().map(|m| (m.dev(), m.ino(), m.size(), mtime_ns(&m)));
        let wal = side(real, "-wal").symlink_metadata().ok();
        FileState {
            main,
            wal: wal.is_some(),
            shm: side(real, "-shm").symlink_metadata().is_ok(),
            journal: side(real, "-journal").symlink_metadata().is_ok(),
            wal_size: wal.map_or(0, |m| m.len()),
        }
    }

    fn changed_since(&self, before: &FileState) -> bool {
        self.main != before.main
            || (self.wal && !before.wal)
            || (self.shm && !before.shm)
            || (self.journal && !before.journal)
    }
}

/// One try: open `real` as `how` says, read everything, say how it was read.
#[allow(clippy::too_many_arguments)]
fn read_once(
    path: &Path,
    real: &Path,
    files: &Files,
    check: bool,
    now: i64,
    home: Option<&Path>,
    how: ReadAs,
) -> Report {
    let mut report = read_report(path, real, files, check, now, home, how != ReadAs::Immutable);
    report.json["brain"]["read_as"] = json!(how.as_str());
    report
}

fn read_report(
    path: &Path,
    real: &Path,
    files: &Files,
    check: bool,
    now: i64,
    home: Option<&Path>,
    read_only: bool,
) -> Report {
    let opened = open_read_only(path, real, read_only).and_then(|conn| {
        let version = version_of(&conn, path)?;
        Ok((conn, version))
    });
    let (conn, version) = match opened {
        Ok(v) => v,
        Err(e) => {
            // `status` is not behind the gate (B3-5.3): its first line is what `require_v2`
            // would have said, in that function's words.
            let extension = shown(extension_json(files, None, now));
            let json = json!({
                "brain": { "path": path.to_string_lossy(), "error": e.to_string(), "kind": kind_of(&e) },
                "extension": extension,
            });
            let mut lines = vec![e.to_string()];
            extension_lines(&json["extension"], &mut lines);
            return Report { json, lines, wrong: true };
        }
    };

    match brain_report(&conn, path, version, files, check, now, home) {
        Ok(report) => report,
        Err(e) => {
            let msg = format!("could not read the brain {}: {e}", path.display());
            let json = json!({
                "brain": { "path": path.to_string_lossy(), "error": msg, "kind": kind_of(&e) },
                "extension": shown(extension_json(files, None, now)),
            });
            let mut lines = vec![msg];
            extension_lines(&json["extension"], &mut lines);
            Report { json, lines, wrong: true }
        }
    }
}

// ── Opening without writing ─────────────────────────────────────────────────

fn side(path: &Path, suffix: &str) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(suffix);
    PathBuf::from(s)
}

/// `file:<path>?<query>`, the three characters a URI gives a meaning to escaped.
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

fn damaged(e: rusqlite::Error) -> AxelError {
    match e.sqlite_error_code() {
        Some(ErrorCode::NotADatabase) | Some(ErrorCode::DatabaseCorrupt) => AxelError::BrainDamaged(e.to_string()),
        _ => AxelError::Db(e),
    }
}

// RULING: wave 4 rulings, the line for 21: `status` must not create a `-wal` or `-shm`
// beside a file that had none. `brain_v2::require_v2` and `connect` (spec 1113, 4815) both
// can: a `mode=ro` look at a WAL file creates the two. With no `-journal` and no frame in a
// `-wal` there is nothing but the main file to read, so `immutable=1` reads it alone and
// creates nothing; `collect_with` reads again the ordinary way if it changed meanwhile.
// Otherwise `mode=ro`, which may create a missing `-shm` beside a file that has a side file.
fn open_read_only(path: &Path, real: &Path, read_only: bool) -> Result<Connection> {
    if !path.exists() {
        return Err(AxelError::NoBrain(path.to_path_buf()));
    }
    let query = if read_only { "mode=ro" } else { "mode=ro&immutable=1" };
    let conn = Connection::open_with_flags(
        file_uri(real, query),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(damaged)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    Ok(conn)
}

/// `require_v2`'s answers, read on this connection.
fn version_of(conn: &Connection, path: &Path) -> Result<i32> {
    let version: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).map_err(damaged)?;
    if version == LATEST {
        return Ok(version);
    }
    if version > LATEST {
        return Err(AxelError::BrainTooNew { file: i64::from(version), binary: i64::from(LATEST) });
    }
    let has_meta: bool = conn
        .query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'brain_meta')", [], |r| {
            r.get(0)
        })
        .map_err(damaged)?;
    let meta: Option<i64> = if has_meta {
        conn.query_row(
            "SELECT json_extract(value, '$.schema_version') FROM brain_meta WHERE key = 'meta'",
            [],
            |r| r.get::<_, Option<i64>>(0),
        )
        .optional()
        .map_err(damaged)?
        .flatten()
    } else {
        None
    };
    if meta == Some(i64::from(SCHEMA_VERSION)) {
        Err(AxelError::V1Brain(path.to_path_buf()))
    } else {
        Err(AxelError::BrainNotReady)
    }
}

fn kind_of(e: &AxelError) -> &'static str {
    match e {
        AxelError::NoBrain(_) => "NoBrain",
        AxelError::V1Brain(_) => "V1Brain",
        AxelError::BrainTooNew { .. } => "BrainTooNew",
        AxelError::BrainDamaged(_) => "BrainDamaged",
        AxelError::BrainNotReady => "BrainNotReady",
        _ => "Error",
    }
}

// ── The status files ────────────────────────────────────────────────────────

struct Files {
    /// Alive, written in the last two minutes; worst first.
    live: Vec<Value>,
    /// `{"pid", "why"}`: a dead process, an old file, a file that cannot be read.
    stale: Vec<Value>,
}

fn pid_alive(pid: u32) -> bool {
    let Ok(pid) = i32::try_from(pid) else { return false };
    if pid <= 0 {
        return false;
    }
    // SAFETY: signal 0 sends nothing; it only asks whether the process exists.
    let rc = unsafe { libc::kill(pid, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Field 22 of `/proc/<pid>/stat`: the clock ticks after boot at which the process started.
/// It never changes while the process lives, and no wall clock goes into it. `None` where
/// there is no `/proc` (macOS) or it cannot be read.
fn start_ticks_of(path: &str) -> Option<u64> {
    let stat = std::fs::read_to_string(path).ok()?;
    // Field 2 is `(comm)`, which may hold spaces and `)`: count from the last `)`.
    stat.rsplit_once(')').and_then(|(_, rest)| rest.split_whitespace().nth(19)).and_then(|t| t.parse().ok())
}

/// This process's start ticks, for its status file (`ext/server.rs`). Read once.
// RULING: v2 c18a review, item 5.
pub fn own_start_ticks() -> Option<u64> {
    static TICKS: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *TICKS.get_or_init(|| start_ticks_of("/proc/self/stat"))
}

/// The process now holding `pid` is not the one that wrote the file. With the file's
/// `start_ticks` and a readable `/proc/<pid>/stat`: they differ. Otherwise (an older
/// extension wrote the file, or there is no `/proc`): `started_after`.
// RULING: v2 c18a review, item 5. Ticks, not the wall clock: a clock stepped forward after
// the extension started moves the start `btime` gives, not the ticks.
fn pid_reused(pid: u32, body: &Value) -> bool {
    match (body["start_ticks"].as_u64(), start_ticks_of(&format!("/proc/{pid}/stat"))) {
        (Some(written), Some(now)) => written != now,
        _ => body["started_ms"].as_i64().is_some_and(|started| started_after(pid, started)),
    }
}

/// RULING: wave 5 fix round of 21, finding 7: the process now holding `pid` started after
/// the file's `started_ms`, so it is another process (Linux: `/proc/<pid>/stat` field 22,
/// clock ticks after `btime`). Elsewhere, or when it cannot be read, false.
fn started_after(pid: u32, started_ms: i64) -> bool {
    // `btime` is whole seconds, and the file is written after the process starts.
    const SLACK_MS: i64 = 2_000;
    let Some(ticks) = start_ticks_of(&format!("/proc/{pid}/stat")).and_then(|t| i64::try_from(t).ok()) else {
        return false;
    };
    let Some(btime) = std::fs::read_to_string("/proc/stat").ok().and_then(|s| {
        s.lines().find_map(|l| l.strip_prefix("btime ").and_then(|v| v.trim().parse::<i64>().ok()))
    }) else {
        return false;
    };
    // SAFETY: sysconf only reads a constant of the system.
    // `c_long` is `i64` on 64-bit Linux and `i32` on 32-bit.
    #[allow(clippy::useless_conversion)]
    let hz = i64::from(unsafe { libc::sysconf(libc::_SC_CLK_TCK) });
    if hz <= 0 {
        return false;
    }
    btime * 1000 + ticks * 1000 / hz > started_ms + SLACK_MS
}

fn mode_rank(mode: &str) -> u8 {
    match mode {
        "degraded" => 2,
        "ready" => 0,
        _ => 1,
    }
}

fn read_small(path: &Path) -> Option<String> {
    let meta = path.symlink_metadata().ok()?;
    if !meta.file_type().is_file() {
        return None;
    }
    let mut text = String::new();
    std::fs::File::open(path).ok()?.take(STATUS_FILE_MAX).read_to_string(&mut text).ok()?;
    Some(text)
}

fn mtime_ms(meta: &std::fs::Metadata) -> i64 {
    meta.mtime() * 1000 + meta.mtime_nsec() / 1_000_000
}

fn status_files(dir: &Path, now: i64) -> Files {
    let mut live = Vec::new();
    let mut stale = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return Files { live, stale } };
    let mut found: Vec<(u32, PathBuf)> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let pid = name.strip_prefix("extension.")?.strip_suffix(".status")?;
            if pid.is_empty() || !pid.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            Some((pid.parse().ok()?, entry.path()))
        })
        .collect();
    found.sort();
    for (pid, path) in found {
        let body: Option<Value> =
            read_small(&path).and_then(|t| serde_json::from_str::<Value>(&t).ok()).filter(Value::is_object);
        let Some(body) = body else {
            stale.push(json!({ "pid": pid, "why": "unreadable" }));
            continue;
        };
        if !pid_alive(pid) {
            stale.push(json!({ "pid": pid, "why": "not alive" }));
            continue;
        }
        if pid_reused(pid, &body) {
            stale.push(json!({ "pid": pid, "why": "pid reused by a later process" }));
            continue;
        }
        let written = body["written_ms"]
            .as_i64()
            .or_else(|| path.symlink_metadata().ok().map(|m| mtime_ms(&m)))
            .unwrap_or(0);
        if now - written > FRESH_MS {
            stale.push(json!({ "pid": pid, "why": format!("not written for {} s", (now - written) / 1000) }));
            continue;
        }
        let mode = body["mode"].as_str().unwrap_or("starting").to_string();
        let degraded = mode == "degraded";
        live.push(json!({
            "pid": pid,
            "mode": mode,
            "reason": if degraded { body["reason"].as_str().unwrap_or("None") } else { "None" },
            "reason_at": if degraded { body["reason_at"].clone() } else { Value::Null },
            "started_ms": body["started_ms"].clone(),
            "written_ms": written,
        }));
    }
    live.sort_by(|a, b| {
        let rank = |v: &Value| mode_rank(v["mode"].as_str().unwrap_or(""));
        rank(b).cmp(&rank(a)).then(a["pid"].as_u64().cmp(&b["pid"].as_u64()))
    });
    Files { live, stale }
}

/// What a person does about a reason.
fn hint(reason: &str) -> Option<&'static str> {
    Some(match reason {
        "FrontStopped" | "IndexerStopped" | "BrainChanged" => "restart Synaps",
        "BrainDamaged" => "the file is kept as it is; restore it from a backup of the brain",
        "BrainTooNew" => "this brain was written by a newer axel-v2; install that one",
        "OtherBase" => "this brain records another Synaps base",
        "BrainOldShape" => crate::brain_v2::OLD_SHAPE,
        "V1Brain" => "this is a version 1 brain; \"axel-v2 migrate init --from-axel <it>\" makes a v2 copy",
        "NoBrain" | "BrainNotReady" => "retried every 60 s",
        _ => return None,
    })
}

/// The `extension` object: mode and reason from the status files, the rest from the row.
fn extension_json(files: &Files, row: Option<&Map<String, Value>>, now: i64) -> Value {
    let worst = files.live.first();
    let get = |k: &str| row.and_then(|r| r.get(k)).cloned().unwrap_or(Value::Null);
    let row_written = row.and_then(|r| r.get("written_ms")).and_then(Value::as_i64);
    let row_fresh = row_written.is_some_and(|w| now - w <= FRESH_MS);
    let reason = worst.and_then(|w| w["reason"].as_str()).unwrap_or("None");
    let embed = if !row_fresh {
        Value::Null
    } else if get("embed_stopped") == json!(true) {
        json!("stopped")
    } else if get("embed_stuck") == json!(true) {
        json!("stuck")
    } else if get("model_loaded") == json!(true) {
        json!("running")
    } else {
        json!("unavailable")
    };
    let mut ext = json!({
        "running": worst.is_some(),
        // SPEC: 4860 names three modes; with no process alive there is none: null.
        "mode": worst.map(|w| w["mode"].clone()).unwrap_or(Value::Null),
        "reason": reason,
        "reason_at": worst.map(|w| w["reason_at"].clone()).unwrap_or(Value::Null),
        "pid": worst.map(|w| w["pid"].clone()).unwrap_or(Value::Null),
        // SPEC: 4754 takes `started_ms` from the status files, 6278 (C-5) asks that it equal
        // the row's. The file's when a process is alive (the same process writes both),
        // else the row's.
        "started_ms": worst.map(|w| w["started_ms"].clone()).filter(|v| !v.is_null()).unwrap_or_else(|| get("started_ms")),
        "status_written_ms": get("written_ms"),
        "status_row_age_ms": row_written.map(|w| now - w),
        "scan": row.and_then(|r| r.get("scan")).cloned().unwrap_or(json!("pending")),
        "indexer_running": row_fresh && get("indexer_running") == json!(true),
        "embed": embed,
        "hint": hint(reason),
        "stale_status_files": files.stale,
    });
    if files.live.len() > 1 {
        ext["processes"] = json!(files.live);
    }
    ext
}

fn extension_lines(ext: &Value, lines: &mut Vec<String>) {
    for stale in ext["stale_status_files"].as_array().into_iter().flatten() {
        lines.push(format!(
            "stale status file found: pid {} ({}); it is not counted",
            stale["pid"],
            stale["why"].as_str().unwrap_or("")
        ));
    }
    if ext["running"] != json!(true) {
        lines.push("extension: not running".to_string());
    } else {
        let mut line = format!("extension: {} (pid {})", ext["mode"].as_str().unwrap_or("starting"), ext["pid"]);
        if ext["mode"] == json!("degraded") {
            line.push_str(&format!(": {}", ext["reason"].as_str().unwrap_or("None")));
            if let Some(at) = ext["reason_at"].as_str() {
                line.push_str(&format!(" at {at}"));
            }
            if let Some(h) = ext["hint"].as_str() {
                line.push_str(&format!("; {h}"));
            }
        }
        if let Some(started) = ext["started_ms"].as_i64() {
            line.push_str(&format!(", started {}", date(started)));
        }
        lines.push(line);
        for p in ext["processes"].as_array().into_iter().flatten() {
            lines.push(format!(
                "  pid {}: {} {}",
                p["pid"],
                p["mode"].as_str().unwrap_or(""),
                if p["mode"] == json!("degraded") { p["reason"].as_str().unwrap_or("") } else { "" }
            ));
        }
    }
}

// ── The brain ───────────────────────────────────────────────────────────────

fn date(ms: i64) -> String {
    Utc.timestamp_millis_opt(ms)
        .single()
        .map(|d| d.format("%Y-%m-%d %H:%M UTC").to_string())
        .unwrap_or_else(|| ms.to_string())
}

fn count(conn: &Connection, sql: &str) -> Result<i64> {
    Ok(conn.query_row(sql, [], |r| r.get::<_, Option<i64>>(0))?.unwrap_or(0))
}

fn grouped(conn: &Connection, sql: &str, p: &[&dyn rusqlite::ToSql]) -> Result<Map<String, Value>> {
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map(p, |r| Ok((r.get::<_, Option<String>>(0)?.unwrap_or_default(), r.get::<_, i64>(1)?)))?;
    let mut out = Map::new();
    for row in rows {
        let (k, n) = row?;
        out.insert(k, json!(n));
    }
    Ok(out)
}

/// The memories adoption skips: no `memory_ext` row, and an `id` or a `content` that
/// `brain_v2::text_at` does not read (not text, or not UTF-8). RULING: wave 5 fix round of
/// 21, finding 3: the same test as adoption, never `typeof`.
fn unreadable(conn: &Connection) -> Result<i64> {
    let mut stmt = conn.prepare(
        "SELECT m.id, m.content FROM memories m
          WHERE NOT EXISTS (SELECT 1 FROM memory_ext e WHERE e.mem_id = m.id)",
    )?;
    let mut rows = stmt.query([])?;
    let mut n = 0;
    while let Some(r) = rows.next()? {
        if text_at(r, 0)?.is_none() || text_at(r, 1)?.is_none() {
            n += 1;
        }
    }
    Ok(n)
}

fn status_row(conn: &Connection) -> Result<Option<Map<String, Value>>> {
    let text: Option<String> = conn
        .query_row("SELECT value FROM context_data WHERE key = 'status' AND layer = 'axel'", [], |r| r.get(0))
        .optional()?;
    Ok(text.and_then(|t| serde_json::from_str::<Value>(&t).ok()).and_then(|v| match v {
        Value::Object(m) => Some(m),
        _ => None,
    }))
}

/// Every string in `v` through `screen::show`: what came from a row or a file.
fn shown(v: Value) -> Value {
    match v {
        Value::String(s) => Value::String(screen::show(&s, FIELD)),
        Value::Array(a) => Value::Array(a.into_iter().map(shown).collect()),
        Value::Object(m) => Value::Object(m.into_iter().map(|(k, v)| (screen::show(&k, FIELD), shown(v))).collect()),
        other => other,
    }
}

#[allow(clippy::too_many_arguments)]
fn brain_report(
    conn: &Connection,
    path: &Path,
    version: i32,
    files: &Files,
    check: bool,
    now: i64,
    home: Option<&Path>,
) -> Result<Report> {
    let row = status_row(conn)?;
    let r = |k: &str| row.as_ref().and_then(|m| m.get(k)).cloned().unwrap_or(Value::Null);
    let since = now - 30 * DAY_MS;
    let mut wrong = false;

    // What was degraded: a live process says so in its file. A row is written only when
    // `Ready`, so a row that says `degraded` is not expected; one written in the last two
    // minutes is still believed.
    let degraded_file = files.live.iter().any(|p| p["mode"] == json!("degraded"));
    let degraded_row = r("mode") == json!("degraded")
        && r("written_ms").as_i64().is_some_and(|w| now - w <= FRESH_MS);
    if degraded_file || degraded_row {
        wrong = true;
    }

    let mut extension = extension_json(files, row.as_ref(), now);
    if !degraded_file && degraded_row {
        extension["reason"] = r("reason");
        extension["hint"] = json!(r("reason").as_str().and_then(hint));
    }

    // Counts by origin, archives, notes, documents, vectors.
    let by_origin = grouped(conn, "SELECT origin_kind, COUNT(*) FROM memory_ext GROUP BY 1 ORDER BY 1", &[])?;
    let memories = count(conn, "SELECT COUNT(*) FROM memories")?;
    let states = grouped(conn, "SELECT state, COUNT(*) FROM archive_index GROUP BY 1", &[])?;
    let skipped_by_reason = grouped(
        conn,
        "SELECT COALESCE(skip_reason, ''), COUNT(*) FROM archive_index WHERE state = 'skipped' GROUP BY 1 ORDER BY 1",
        &[],
    )?;
    let base: Option<PathBuf> = r("base").as_str().filter(|s| !s.is_empty()).map(PathBuf::from);
    let files_gone = match &base {
        Some(base) => {
            let root = base.join("context-archives");
            let mut stmt = conn.prepare("SELECT scope_dir, rel_name FROM archive_index")?;
            let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
            let mut gone = 0;
            for row in rows {
                let (dir, name) = row?;
                if root.join(dir).join(name).symlink_metadata().is_err() {
                    gone += 1;
                }
            }
            json!(gone)
        }
        None => Value::Null,
    };
    let st = |k: &str| states.get(k).cloned().unwrap_or(json!(0));

    let mut notes = Vec::new();
    let mut notes_not_ok = 0;
    {
        let mut stmt =
            conn.prepare("SELECT path, ok, reason, bad_lines, bad_lines_changed_ms, live_count FROM note_files ORDER BY path")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, Option<i64>>(4)?,
                r.get::<_, i64>(5)?,
            ))
        })?;
        for row in rows {
            let (path, ok, reason, bad, changed, live) = row?;
            if ok == 0 {
                notes_not_ok += 1;
            }
            let bad_lines = bad.as_deref().map_or(0, |b| b.split(',').filter(|h| !h.is_empty()).count());
            let ids_skipped = reason
                .as_deref()
                .and_then(|r| r.split("ids_skipped:").nth(1))
                .and_then(|n| n.split(|c: char| !c.is_ascii_digit() && c != ' ').next())
                .and_then(|n| n.trim().parse::<i64>().ok())
                .unwrap_or(0);
            notes.push(json!({
                "path": path, "ok": ok != 0, "reason": reason, "live": live,
                "bad_lines": bad_lines, "bad_lines_changed_ms": changed, "ids_skipped": ids_skipped,
            }));
        }
    }
    if notes_not_ok > 0 {
        wrong = true;
    }

    let documents = count(conn, "SELECT COUNT(*) FROM documents")?;
    let documents_hidden = count(conn, "SELECT COUNT(*) FROM documents WHERE hidden IS NOT NULL")?;
    let vectors_present = count(conn, "SELECT COUNT(*) FROM memory_ext WHERE embedding IS NOT NULL")?;
    let vectors_waiting = count(
        conn,
        "SELECT COUNT(*) FROM memory_ext e JOIN memories m ON m.id = e.mem_id
          WHERE e.embedding IS NULL AND m.sensitivity <> 'secret'",
    )?;
    let unreadable_memories = match r("unreadable_memories") {
        Value::Null => json!(unreadable(conn)?),
        v => v,
    };
    // RULING: wave 4 rulings, the line for 21: the age of the purge mark, read now (the
    // row's `purge_pending_ms` is its age when the row was written).
    let purge_pending_ms: Option<i64> = conn
        .query_row(
            "SELECT CAST((julianday('now') - julianday(updated)) * 86400000 AS INTEGER)
               FROM context_data WHERE key = 'purge_pending' AND layer = 'axel'",
            [],
            |r| r.get::<_, Option<i64>>(0),
        )
        .optional()?
        .map(|age| age.unwrap_or(0).max(0));

    // The model.
    let brain_model: Option<String> = conn
        .query_row("SELECT json_extract(value, '$.embedder_model') FROM brain_meta WHERE key = 'meta'", [], |r| r.get(0))
        .optional()?
        .flatten();
    let model_dir: Option<PathBuf> = match r("model_dir").as_str() {
        Some(d) if !d.is_empty() => Some(PathBuf::from(d)),
        _ => home.map(default_model_dir),
    };
    let model_files = model_dir.as_deref().and_then(velocirag::download::model_files_in);

    // Hidden and superseded (R-1, R3-6a): over all time, and of them the last 30 days.
    let hidden_by = {
        let mut m = Map::new();
        for k in ["model", "cli", "pause", "twin", "unproven"] {
            m.insert(k.to_string(), json!(0));
        }
        for (k, v) in grouped(
            conn,
            "SELECT CASE WHEN substr(hidden_by, 1, 6) = 'model:' THEN 'model'
                         WHEN substr(hidden_by, 1, 5) = 'twin:' THEN 'twin'
                         WHEN hidden_by = 'cli:unproven' THEN 'unproven'
                         WHEN hidden_by IN ('cli', 'pause') THEN hidden_by
                         ELSE 'other' END, COUNT(*)
               FROM memory_ext WHERE hidden_ms IS NOT NULL GROUP BY 1",
            &[],
        )? {
            m.insert(k, v);
        }
        m
    };
    // RULING: v2 c18b review round 4, item 2: where the rows hidden as not proven are, so
    // that `show ctx-<id>` lists their ids.
    let unproven_archives: Vec<String> = {
        let mut stmt = conn.prepare(
            "SELECT origin_ref FROM memory_ext
              WHERE hidden_ms IS NOT NULL AND hidden_by = 'cli:unproven' AND origin_kind = 'archive'
              GROUP BY origin_ref ORDER BY MAX(hidden_ms) DESC, origin_ref LIMIT 5",
        )?;
        let rows = stmt.query_map([], |r| text_at(r, 0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?.into_iter().flatten().collect()
    };
    // RULING: v2 c18b review round 4, item 5: the files no forget mark reaches.
    let mtime_files = count(conn, "SELECT COUNT(*) FROM archive_index WHERE state = 'indexed' AND time_source = 'mtime'")?;
    let hidden_by_model = count(
        conn,
        "SELECT COUNT(*) FROM memory_ext WHERE hidden_ms IS NOT NULL AND substr(hidden_by, 1, 6) = 'model:'",
    )?;
    let hidden_by_model_30d: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memory_ext WHERE hidden_ms >= ?1 AND substr(hidden_by, 1, 6) = 'model:'",
        [since],
        |r| r.get(0),
    )?;
    let superseded_by_model = count(
        conn,
        "SELECT COUNT(*) FROM memories m JOIN memory_ext e ON e.mem_id = m.superseded_by WHERE e.origin_kind = 'tool'",
    )?;
    let superseded_by_model_30d: i64 = conn.query_row(
        "SELECT COUNT(*) FROM memories m JOIN memory_ext e ON e.mem_id = m.superseded_by
          WHERE e.origin_kind = 'tool' AND e.created_ms >= ?1",
        [since],
        |r| r.get(0),
    )?;

    // Turns left out because a forget tool was called in them (R3-5, 4.1 risk 2).
    let (units, archives): (i64, i64) = conn.query_row(
        "SELECT COALESCE(SUM(forget_units), 0), COUNT(*) FROM archive_index WHERE forget_units > 0",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    // SPEC: 4831 does not say which time puts an archive in the last 30 days: the
    // archive's own time, else when it was first seen.
    let units_30d: i64 = conn.query_row(
        "SELECT COALESCE(SUM(forget_units), 0) FROM archive_index
          WHERE forget_units > 0 AND COALESCE(created_ms, first_seen_ms) >= ?1",
        [since],
        |r| r.get(0),
    )?;
    let mut top_chats = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT logical, SUM(forget_units), COUNT(*) FROM archive_index WHERE forget_units > 0
              GROUP BY logical ORDER BY 2 DESC, 1 LIMIT 3",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(json!({ "logical": r.get::<_, Option<String>>(0)?, "units": r.get::<_, i64>(1)?, "archives": r.get::<_, i64>(2)? }))
        })?;
        for row in rows {
            top_chats.push(row?);
        }
    }
    let mut unmatched = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT archive_id, SUM(unmatched_hidden) FROM archive_index WHERE unmatched_hidden > 0
              GROUP BY archive_id ORDER BY archive_id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(json!({ "archive": format!("ctx-{}", r.get::<_, String>(0)?), "turns": r.get::<_, i64>(1)? }))
        })?;
        for row in rows {
            unmatched.push(row?);
        }
    }

    let pauses = {
        let mut stmt = conn.prepare("SELECT logical, what, since_ms, set_by FROM chat_pauses ORDER BY since_ms, logical")?;
        let rows = stmt.query_map([], |r| {
            Ok(json!({ "logical": r.get::<_, String>(0)?, "what": r.get::<_, String>(1)?,
                       "since_ms": r.get::<_, i64>(2)?, "set_by": r.get::<_, String>(3)? }))
        })?;
        rows.collect::<rusqlite::Result<Vec<Value>>>()?
    };

    // The forum: by project only, never by author or group (section 6).
    let posts_by_project = grouped(
        conn,
        "SELECT project_key, COUNT(*) FROM forum_posts GROUP BY 1 ORDER BY 1",
        &[],
    )?;
    let digest_bad = count(conn, "SELECT COUNT(*) FROM forum_posts WHERE digest_ok = 0")?;
    let mut forum = json!({
        "posts_by_project": posts_by_project,
        "digest_bad": digest_bad,
        "decided_by": decided_by(base.as_deref(), home),
    });
    // From the row and from nowhere else; absent when no extension has written one (C-5).
    if let Some(row) = &row {
        if let Some(f) = row.get("forum").and_then(Value::as_object) {
            for k in ["declared", "core_owns"] {
                if let Some(v) = f.get(k) {
                    forum[k] = v.clone();
                }
            }
        }
    }

    let old_brain = old_brain(conn)?;

    let mut json = json!({
        "extension": extension,
        "counts": { "memories": memories, "by_origin": by_origin },
        "archives": {
            "indexed": st("indexed"), "tombstone": st("tombstone"), "skipped": st("skipped"),
            "skipped_by_reason": skipped_by_reason, "files_gone": files_gone,
        },
        "notes": { "files": notes.len(), "not_ok": notes_not_ok, "list": notes },
        "documents": { "total": documents, "hidden": documents_hidden, "waiting": r("docs_waiting") },
        "vectors": { "present": vectors_present, "waiting": vectors_waiting },
        "model": r("model"),
        "brain_model": brain_model,
        "model_matches": r("model_matches"),
        "embed_stuck": r("embed_stuck"),
        "model_loaded": r("model_loaded"),
        "model_dir": model_dir.as_ref().map(|d| d.to_string_lossy().into_owned()),
        "model_files": model_files.as_ref().map(|f| f.to_string_lossy().into_owned()),
        "embedder": r("embedder"),
        "breaker": { "open": r("breaker_open"), "trips": r("breaker_trips"), "hook_misses": r("hook_misses") },
        "jobs_dropped": r("jobs_dropped"),
        "sessions_dropped": r("sessions_dropped"),
        "tokens_blanked": r("tokens_blanked"),
        "turns_withheld": r("turns_withheld"),
        "turns_dropped": r("turns_dropped"),
        "docs_waiting": r("docs_waiting"),
        "vectors_waiting": r("vectors_waiting"),
        "write_errors": r("write_errors"),
        "panics": r("panics"),
        "unreadable_memories": unreadable_memories,
        "purge_pending_ms": purge_pending_ms,
        "core_rule": { "words": screen::core_rule_words().len(), "commit": screen::core_rule_commit() },
        "pauses": pauses,
        "hidden_by": hidden_by,
        "unproven_archives": unproven_archives,
        "mtime_files": mtime_files,
        "hidden_by_model": hidden_by_model,
        "superseded_by_model": superseded_by_model,
        "hidden_by_model_30d": hidden_by_model_30d,
        "superseded_by_model_30d": superseded_by_model_30d,
        "forget_turns": { "units": units, "archives": archives, "units_30d": units_30d, "top_chats": top_chats },
        "hidden_turns_unmatched": unmatched,
        "old_brain": old_brain,
        "forum": forum,
        "base": r("base"),
        "warnings": r("warnings"),
    });

    if check {
        let (checked, bad) = run_check(conn, row.as_ref(), model_files.as_deref())?;
        json["check"] = checked;
        wrong |= bad;
    }

    let mut json = shown(json);
    let size = path.metadata().map(|m| m.len()).unwrap_or(0);
    json["brain"] = json!({ "path": path.to_string_lossy(), "size": size, "version": version });
    let lines = text(&json, path, check);
    Ok(Report { json, lines, wrong })
}

/// The config file that says core owns the forum, for the base the row records.
// SPEC: 4837 asks for `forum.decided_by`; the status row (as built) does not carry it. It is
// asked again here, as `forum = auto` would ask it, of the base the row names; null with no
// row.
fn decided_by(base: Option<&Path>, home: Option<&Path>) -> Value {
    let Some(base) = base.filter(|b| b.is_absolute()) else { return Value::Null };
    let home = home.map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("/"));
    let params = json!({ "config": { "synaps_base": base.to_string_lossy(), "forum": "auto" } });
    let cfg = ExtConfig::from_initialize(&params, &home);
    cfg.forum_decided_by().map(|p| json!(p.to_string_lossy())).unwrap_or(Value::Null)
}

// ── The old brain written after the import (R-9h; 4.3, Q-b) ───────────────

fn mtime_ns(meta: &std::fs::Metadata) -> i64 {
    meta.mtime().saturating_mul(1_000_000_000).saturating_add(meta.mtime_nsec())
}

// SPEC: 4906 names the `init` row and says a `catch-up` writes a new record that ends the
// line; the item names are commit 19's. The newest `import_log` row whose `detail` holds a
// `path` is the record.
fn old_brain(conn: &Connection) -> Result<Value> {
    let detail: Option<String> = conn
        .query_row(
            "SELECT detail FROM import_log
              WHERE json_valid(detail) AND json_type(detail, '$.path') = 'text'
              ORDER BY id DESC LIMIT 1",
            [],
            |r| r.get(0),
        )
        .optional()?;
    let Some(d) = detail.and_then(|t| serde_json::from_str::<Value>(&t).ok()) else { return Ok(Value::Null) };
    let Some(path) = d["path"].as_str().map(PathBuf::from) else { return Ok(Value::Null) };
    // `lstat` only: neither file is opened.
    let main = path.symlink_metadata().ok();
    let wal = side(&path, "-wal").symlink_metadata().ok();
    let main_moved = main.as_ref().is_some_and(|m| {
        Some(m.len() as i64) != d["size"].as_i64() || Some(mtime_ns(m)) != d["mtime_ns"].as_i64()
    });
    // An empty `-wal` is never evidence; `-shm` is never looked at.
    let wal_moved = wal.as_ref().is_some_and(|w| {
        w.len() > 0 && (Some(w.len() as i64) != d["wal_size"].as_i64() || Some(mtime_ns(w)) != d["wal_mtime_ns"].as_i64())
    });
    let latest = [main.as_ref(), wal.as_ref()].into_iter().flatten().map(mtime_ms).max();
    Ok(json!({
        "path": path.to_string_lossy(),
        "present": main.is_some(),
        "written_after_import": main_moved || wal_moved,
        "date": latest.map(date),
    }))
}

// ── --check ─────────────────────────────────────────────────────────────────

fn run_check(conn: &Connection, row: Option<&Map<String, Value>>, model: Option<&Path>) -> Result<(Value, bool)> {
    let mut bad = false;
    let integrity: Vec<String> = {
        let mut stmt = conn.prepare("PRAGMA integrity_check")?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<_>>()?
    };
    let integrity_ok = integrity.len() == 1 && integrity[0] == "ok";
    bad |= !integrity_ok;
    let memories = count(conn, "SELECT COUNT(*) FROM memories")?;
    let fts = count(conn, "SELECT COUNT(*) FROM memories_fts")?;
    let ext = count(conn, "SELECT COUNT(*) FROM memory_ext")?;
    // SPEC: 4839 compares the two counts with `memories`; a memory whose id or content is
    // not text is never adopted and gets no word row (W2-4.5, R-12b item 3), so both are
    // compared with the readable ones.
    let unreadable = unreadable(conn)?;
    let counts_ok = fts == memories - unreadable && ext == memories - unreadable;
    bad |= !counts_ok;
    let vectors = match model {
        Some(graph) => cosine_check(conn, graph, row),
        None => json!({ "ran": false, "why": "no model files" }),
    };
    if vectors["ok"] == json!(false) {
        bad = true;
    }
    Ok((
        json!({
            "integrity": integrity, "integrity_ok": integrity_ok,
            "memories": memories, "memories_fts": fts, "memory_ext": ext, "unreadable": unreadable,
            "counts_ok": counts_ok, "vectors": vectors,
        }),
        bad,
    ))
}

/// The cosine between the stored vector of three sampled documents and a fresh embedding
/// of the same text (section 0, point 22): below 0.99 it says which file to try.
fn cosine_check(conn: &Connection, graph: &Path, row: Option<&Map<String, Value>>) -> Value {
    let Some(dir) = graph.parent().map(|p| if p.ends_with("onnx") { p.parent().unwrap_or(p) } else { p }) else {
        return json!({ "ran": false, "why": "no model folder" });
    };
    if row.and_then(|r| r.get("embedder")).and_then(Value::as_str) == Some("hash") {
        return json!({ "ran": false, "why": "embedder = hash" });
    }
    let samples: Vec<(String, Vec<u8>)> = match conn.prepare(
        "SELECT content, embedding FROM documents WHERE hidden IS NULL AND length(embedding) = 1536
          ORDER BY id LIMIT 3",
    ) {
        Ok(mut stmt) => match stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?))) {
            Ok(rows) => rows.flatten().collect(),
            Err(e) => return json!({ "ran": false, "why": e.to_string() }),
        },
        Err(e) => return json!({ "ran": false, "why": e.to_string() }),
    };
    if samples.is_empty() {
        return json!({ "ran": false, "why": "no document with a stored vector" });
    }
    let mut embedder = match velocirag::embedder::Embedder::from_model_dir(dir, 2, true) {
        Ok(e) => e,
        Err(e) => return json!({ "ran": false, "ok": false, "why": e.to_string() }),
    };
    let mut lowest = f64::MAX;
    for (text, blob) in &samples {
        let stored: Vec<f32> = blob.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let Ok(fresh) = embedder.embed_one(text) else { continue };
        let dot: f64 = stored.iter().zip(&fresh).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum();
        let na: f64 = stored.iter().map(|a| f64::from(*a).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = fresh.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>().sqrt();
        if na > 0.0 && nb > 0.0 {
            lowest = lowest.min(dot / (na * nb));
        }
    }
    if lowest == f64::MAX {
        return json!({ "ran": false, "why": "no sample could be embedded" });
    }
    let ok = lowest >= 0.99;
    let other = if graph.ends_with("onnx/model.onnx") { dir.join("model.onnx") } else { dir.join("onnx/model.onnx") };
    json!({
        "ran": true, "ok": ok, "lowest_cosine": lowest, "graph": graph.to_string_lossy(),
        "try": if ok { Value::Null } else { json!(other.to_string_lossy()) },
    })
}

// ── The text ────────────────────────────────────────────────────────────────

/// The lines, built from the JSON after `shown`.
fn text(j: &Value, path: &Path, check: bool) -> Vec<String> {
    let s = |v: &Value| match v {
        Value::Null => "-".to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let mut l = Vec::new();
    l.push(format!("brain: {}, {} bytes, version {}", path.display(), j["brain"]["size"], j["brain"]["version"]));
    extension_lines(&j["extension"], &mut l);
    let ext = &j["extension"];
    match ext["status_row_age_ms"].as_i64() {
        None => l.push("status row: none (no extension has written one)".to_string()),
        Some(age) => l.push(format!(
            "status row: written {} s ago{}; scan {}",
            age / 1000,
            if age > FRESH_MS { " (older than two minutes: not running or stuck)" } else { "" },
            s(&ext["scan"])
        )),
    }
    l.push(format!("indexer: {}; embedding: {}", if ext["indexer_running"] == json!(true) { "running" } else { "not running" }, s(&ext["embed"])));

    let by_origin: Vec<String> = j["counts"]["by_origin"]
        .as_object()
        .into_iter()
        .flatten()
        .map(|(k, v)| format!("{k} {v}"))
        .collect();
    l.push(format!("memories: {} ({})", j["counts"]["memories"], if by_origin.is_empty() { "none adopted".to_string() } else { by_origin.join(", ") }));
    let a = &j["archives"];
    let reasons: Vec<String> =
        a["skipped_by_reason"].as_object().into_iter().flatten().map(|(k, v)| format!("{k} {v}")).collect();
    l.push(format!(
        "archives: {} indexed, {} skipped{}, {} tombstoned, files gone: {}",
        a["indexed"],
        a["skipped"],
        if reasons.is_empty() { String::new() } else { format!(" ({})", reasons.join(", ")) },
        a["tombstone"],
        s(&a["files_gone"])
    ));
    l.push(format!("notes files: {} ({} with ok = 0)", j["notes"]["files"], j["notes"]["not_ok"]));
    for n in j["notes"]["list"].as_array().into_iter().flatten() {
        let mut line = format!("  {}: ", s(&n["path"]));
        if n["ok"] == json!(false) {
            line.push_str(&format!("ok = 0 ({}); ", s(&n["reason"])));
        }
        line.push_str(&format!("{} bad lines", n["bad_lines"]));
        if let Some(at) = n["bad_lines_changed_ms"].as_i64() {
            line.push_str(&format!(", set changed {}", date(at)));
        }
        line.push_str(&format!(", ids_skipped {}", n["ids_skipped"]));
        l.push(line);
    }
    l.push(format!(
        "documents: {}, hidden {}, waiting {}",
        j["documents"]["total"],
        j["documents"]["hidden"],
        s(&j["documents"]["waiting"])
    ));
    l.push(format!("vectors: {} present, {} waiting", j["vectors"]["present"], j["vectors"]["waiting"]));
    l.push(format!(
        "model: {} in {}; loaded: {}; brain's model: {}; matches: {}; stuck: {}",
        s(&j["model"]),
        s(&j["model_dir"]),
        s(&j["model_loaded"]),
        s(&j["brain_model"]),
        s(&j["model_matches"]),
        s(&j["embed_stuck"])
    ));
    if j["model_files"].is_null() {
        l.push(format!("model files: not found in {}", s(&j["model_dir"])));
    }
    let b = &j["breaker"];
    l.push(format!("breaker: open {}, trips {}, hook misses {}", s(&b["open"]), s(&b["trips"]), s(&b["hook_misses"])));
    l.push(format!(
        "jobs dropped {}, sessions dropped {}, write errors {}, panics {}",
        s(&j["jobs_dropped"]),
        s(&j["sessions_dropped"]),
        s(&j["write_errors"]),
        s(&j["panics"])
    ));
    l.push(format!(
        "tokens blanked by the screen {}; turns not captured because core would withhold them {}; turns dropped {}",
        s(&j["tokens_blanked"]),
        s(&j["turns_withheld"]),
        s(&j["turns_dropped"])
    ));
    l.push(format!("core rule: {} words, as of core {}", j["core_rule"]["words"], s(&j["core_rule"]["commit"])));
    l.push(format!("docs waiting {}; unreadable memories {}", s(&j["docs_waiting"]), s(&j["unreadable_memories"])));
    if let Some(age) = j["purge_pending_ms"].as_i64() {
        l.push(format!("purge mark: {} s old (the log is not yet truncated)", age / 1000));
    }
    let pauses = j["pauses"].as_array().map_or(0, Vec::len);
    l.push(format!("pauses in force: {pauses}"));
    for p in j["pauses"].as_array().into_iter().flatten() {
        l.push(format!("  {} {} since {} by {}", s(&p["logical"]), s(&p["what"]), p["since_ms"].as_i64().map_or("-".into(), date), s(&p["set_by"])));
    }
    let h = &j["hidden_by"];
    l.push(format!(
        "hidden rows: model {}, cli {}, pause {}, twin {}",
        h["model"], h["cli"], h["pause"], h["twin"]
    ));
    // RULING: v2 c18b review round 3, item 1: one count.
    if h["unproven"].as_i64().unwrap_or(0) > 0 {
        l.push(format!(
            "hidden with the prompt of a turn deleted with forget --now, not proven that turn: {} (axel-v2 restore <id> brings one back)",
            h["unproven"]
        ));
        let at: Vec<String> = j["unproven_archives"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|id| format!("ctx-{}", screen::show(id, 64)))
            .collect();
        if !at.is_empty() {
            l.push(format!("  in {} (axel-v2 show ctx-<id> lists them)", at.join(", ")));
        }
    }
    l.push(format!(
        "archive files timed by their modification time (no forget mark reaches them; forget --now hides their rows with a deleted turn's prompt): {}",
        j["mtime_files"]
    ));
    let (hm, sm) = (j["hidden_by_model"].as_i64().unwrap_or(0), j["superseded_by_model"].as_i64().unwrap_or(0));
    if hm > 0 || sm > 0 {
        let recent = j["hidden_by_model_30d"].as_i64().unwrap_or(0) + j["superseded_by_model_30d"].as_i64().unwrap_or(0);
        l.push(format!("{hm} memories hidden and {sm} superseded by the assistant, {recent} of them in the last 30 days"));
    }
    let f = &j["forget_turns"];
    if f["units"].as_i64().unwrap_or(0) > 0 {
        l.push(format!(
            "{} turns in {} archives are left out because a forget tool was called in them",
            f["units"], f["archives"]
        ));
        for c in f["top_chats"].as_array().into_iter().flatten() {
            l.push(format!("  {}: {} turns in {} archives", s(&c["logical"]), c["units"], c["archives"]));
        }
    }
    for u in j["hidden_turns_unmatched"].as_array().into_iter().flatten() {
        l.push(format!("a hidden turn could not be matched in {}; its archived form is visible", s(&u["archive"])));
    }
    let o = &j["old_brain"];
    if o["written_after_import"] == json!(true) {
        let p = s(&o["path"]);
        l.push(format!(
            "the old brain {p} was written after the import ({}); \"axel-v2 migrate catch-up --from-axel {p}\" brings over what was saved there",
            s(&o["date"])
        ));
    }
    let fo = &j["forum"];
    let yes = |v: &Value| if v == &json!(true) { "yes" } else { "no" };
    let mut line = format!("forum: declared {}, core owns it {}", yes(&fo["declared"]), yes(&fo["core_owns"]));
    if let Some(by) = fo["decided_by"].as_str() {
        line.push_str(&format!(" (decided by {by})"));
    }
    l.push(line);
    let posts: Vec<String> =
        fo["posts_by_project"].as_object().into_iter().flatten().map(|(k, v)| format!("{k} {v}")).collect();
    l.push(format!("forum posts by project: {}", if posts.is_empty() { "none".into() } else { posts.join(", ") }));
    if fo["digest_bad"].as_i64().unwrap_or(0) > 0 {
        l.push(format!("forum rows that fail their digest: {}", fo["digest_bad"]));
    }
    l.push(format!("base: {}", s(&j["base"])));
    for w in j["warnings"].as_array().into_iter().flatten() {
        l.push(format!("warning: {}", s(w)));
    }
    if check {
        let c = &j["check"];
        l.push(format!(
            "check: integrity {}; memories {}, memories_fts {}, memory_ext {} (unreadable {}): {}",
            if c["integrity_ok"] == json!(true) { "ok".to_string() } else { s(&c["integrity"]) },
            c["memories"],
            c["memories_fts"],
            c["memory_ext"],
            c["unreadable"],
            if c["counts_ok"] == json!(true) { "ok" } else { "MISMATCH" }
        ));
        let v = &c["vectors"];
        if v["ran"] == json!(true) {
            l.push(format!(
                "check: lowest cosine {:.4} with {}{}",
                v["lowest_cosine"].as_f64().unwrap_or(0.0),
                s(&v["graph"]),
                v["try"].as_str().map(|t| format!("; below 0.99: try {t}")).unwrap_or_default()
            ));
        } else {
            l.push(format!("check: vectors not compared ({})", s(&v["why"])));
        }
    }
    l
}

