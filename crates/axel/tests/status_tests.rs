//! Commit 21: `status` (its exit codes and keys), `why`, and that only `model fetch` goes
//! online. Section 10, "Tests per commit", the row of 21; the wave 4 rulings for 21.
//!
//! Every brain is a temp file; the binary is run with `--brain` inside the temp folder.
//! `model fetch` is never run.

use std::collections::BTreeMap;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use axel::brain_v2::{self, OpenOpts, V2Brain};
use axel::project_memory::{RecordLinks, Sensitivity};
use axel::store::{self, NewRow};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use tempfile::TempDir;

const DAY_MS: i64 = 24 * 60 * 60 * 1000;
const PROJECT: &str = "1111111111111111111111111111111111111111111111111111111111111111";

// ── Helpers ─────────────────────────────────────────────────────────────────

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn new_brain(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("brain.r8");
    drop(open(&path));
    path
}

fn open(path: &Path) -> V2Brain {
    let opts = OpenOpts { agent: Some("test"), model: "hash-384", dim: 384, upgrade_v1: false };
    brain_v2::open_v2(path, &opts).unwrap()
}

/// Plain SQL on the brain, closed again before the binary runs.
fn sql(path: &Path, statements: &str) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(statements).unwrap();
}

fn axel(path: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_axel"))
        .arg("--brain")
        .arg(path)
        .args(args)
        .env_remove("AXEL_BRAIN")
        .output()
        .unwrap()
}

fn status_json(path: &Path) -> (i32, Value) {
    let out = axel(path, &["status", "--json"]);
    let text = String::from_utf8(out.stdout).unwrap();
    let v: Value = serde_json::from_str(text.trim()).unwrap_or_else(|e| panic!("{e}: {text}"));
    (out.status.code().unwrap(), v)
}

fn status_text(path: &Path) -> (i32, String) {
    let out = axel(path, &["status"]);
    (out.status.code().unwrap(), String::from_utf8(out.stdout).unwrap())
}

/// The status row as `indexer_start::write_status` writes it, with these values.
fn status_row(written_ms: i64, started_ms: i64, scan: &str, mode: &str, base: &Path) -> Value {
    json!({
        "pid": std::process::id(), "version": "test", "mode": mode, "reason": "None", "reason_at": null,
        "written_ms": written_ms, "started_ms": started_ms, "scan": scan,
        "indexer_running": true, "embed_stopped": false, "embed_stuck": false, "model_loaded": false,
        "model": null, "brain_model": "hash-384", "model_matches": null, "model_dir": "/nonexistent/model",
        "embedder": "hash", "vectors_waiting": 0, "docs_waiting": 4, "unreadable_memories": 2,
        "purge_pending_ms": null, "breaker_open": false, "breaker_trips": 0, "hook_misses": 0,
        "sessions_dropped": 0, "jobs_dropped": 1, "tokens_blanked": 3, "turns_withheld": 5,
        "turns_dropped": 6, "write_errors": 0, "panics": 0,
        "forum": { "declared": true, "core_owns": false },
        "base": base.to_string_lossy(), "warnings": [],
    })
}

fn write_row(path: &Path, row: &Value) {
    let conn = Connection::open(path).unwrap();
    conn.execute(
        "INSERT INTO context_data(key, layer, value, updated) VALUES('status', 'axel', ?1, datetime('now'))
         ON CONFLICT(key, layer) DO UPDATE SET value = excluded.value",
        [row.to_string()],
    )
    .unwrap();
}

/// `<data_dir>/extension.<pid>.status`, as `ext/server.rs` writes it.
fn write_status_file(dir: &Path, pid: u32, mode: &str, reason: &str, reason_at: Option<&str>) {
    let body = json!({
        "pid": pid, "mode": mode, "reason": reason, "reason_at": reason_at,
        "written_ms": now_ms(), "started_ms": now_ms() - 1000,
    });
    std::fs::write(dir.join(format!("extension.{pid}.status")), body.to_string()).unwrap();
}

fn archive(conn: &Connection, rel: &str, logical: &str, forget_units: i64, created_ms: i64) {
    conn.execute(
        "INSERT INTO archive_index(scope_dir, rel_name, archive_id, logical, state, forget_units, created_ms, first_seen_ms)
         VALUES ('aa', ?1, ?1, ?2, 'indexed', ?3, ?4, ?4)",
        params![rel, logical, forget_units, created_ms],
    )
    .unwrap();
}

fn row<'a>(id: &str, origin_kind: &'a str, origin_ref: &'a str, created_ms: i64) -> NewRow<'a> {
    NewRow {
        id: id.to_string(),
        class: "note",
        project_key: PROJECT,
        title: format!("title of {id}"),
        abstract_text: String::new(),
        content: format!("the content of {id}, a row of the status tests"),
        tags: Vec::new(),
        created_ms,
        time_source: "exact",
        sensitivity: Sensitivity::Normal,
        provenance: "test".to_string(),
        expires_ms: None,
        importance: 0.5,
        origin_kind,
        origin_ref,
        unit_pos: None,
        session_id: None,
        logical: None,
        links: RecordLinks::default(),
        turn_key: None,
        hidden: None,
    }
}

fn tx<T>(conn: &Connection, work: impl FnOnce() -> T) -> T {
    brain_v2::begin_write(conn).unwrap();
    let out = work();
    conn.execute_batch("COMMIT").unwrap();
    out
}

/// Every file of the folder: name, size, modification time.
fn folder(dir: &Path) -> BTreeMap<String, (u64, i64, i64)> {
    std::fs::read_dir(dir)
        .unwrap()
        .flatten()
        .map(|e| {
            let m = e.metadata().unwrap();
            (e.file_name().into_string().unwrap(), (m.len(), m.mtime(), m.mtime_nsec()))
        })
        .collect()
}

fn at<'a>(v: &'a Value, dotted: &str) -> &'a Value {
    dotted.split('.').fold(v, |v, k| &v[k])
}

fn has(v: &Value, dotted: &str) -> bool {
    let mut cur = v;
    for k in dotted.split('.') {
        match cur.get(k) {
            Some(next) => cur = next,
            None => return false,
        }
    }
    true
}

// ── The keys of section 7 ───────────────────────────────────────────────────

#[test]
fn status_json_has_every_key_of_section_7() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let now = now_ms();
    write_row(&path, &status_row(now, now - 5000, "done", "ready", dir.path()));
    let (code, v) = status_json(&path);
    assert_eq!(code, 0, "{v:#}");
    for key in [
        "brain.path", "brain.size", "brain.version",
        "extension.running", "extension.mode", "extension.reason", "extension.reason_at", "extension.started_ms",
        "extension.status_written_ms", "extension.scan", "extension.indexer_running", "extension.embed",
        "counts.memories", "counts.by_origin",
        "archives.indexed", "archives.skipped", "archives.skipped_by_reason", "archives.tombstone", "archives.files_gone",
        "notes.files", "notes.not_ok", "notes.list",
        "documents.total", "documents.hidden", "documents.waiting",
        "vectors.present", "vectors.waiting",
        "model_dir", "model_loaded", "model", "brain_model", "model_matches", "embed_stuck",
        "breaker.open", "breaker.trips", "jobs_dropped", "tokens_blanked", "turns_withheld", "core_rule",
        "pauses", "hidden_by.model", "hidden_by.cli", "hidden_by.pause", "hidden_by.twin",
        "hidden_by_model", "superseded_by_model", "hidden_by_model_30d", "superseded_by_model_30d",
        "forget_turns.units", "forget_turns.archives", "forget_turns.units_30d", "forget_turns.top_chats",
        "hidden_turns_unmatched", "old_brain", "forum.posts_by_project", "forum.digest_bad",
        "forum.declared", "forum.core_owns", "forum.decided_by", "unreadable_memories", "base",
        // RULING: wave 4 rulings, the line for 21.
        "docs_waiting", "turns_dropped", "purge_pending_ms",
    ] {
        assert!(has(&v, key), "{key} is missing:\n{v:#}");
    }
    assert_eq!(at(&v, "brain.version"), &json!(1));
    assert_eq!(v["core_rule"]["words"], json!(axel::screen::core_rule_words().len()));
    assert_eq!(v["docs_waiting"], json!(4));
    assert_eq!(v["turns_dropped"], json!(6));
    assert_eq!(v["unreadable_memories"], json!(2));
    assert_eq!(v["forum"]["declared"], json!(true));
    assert_eq!(v["forum"]["core_owns"], json!(false));

    let (_, text) = status_text(&path);
    let rule = format!("core rule: {} words, as of core {}", axel::screen::core_rule_words().len(), axel::screen::core_rule_commit());
    assert!(text.contains(&rule), "{text}");
}

#[test]
fn forum_keys_are_absent_without_a_row() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let (code, v) = status_json(&path);
    assert_eq!(code, 0);
    assert!(!has(&v, "forum.declared") && !has(&v, "forum.core_owns"), "{v:#}");
}

#[test]
fn status_written_ms_and_started_ms_equal_what_the_row_holds() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let (written, started) = (now_ms() - 3000, now_ms() - 90_000);
    write_row(&path, &status_row(written, started, "running", "ready", dir.path()));
    let (_, v) = status_json(&path);
    assert_eq!(v["extension"]["status_written_ms"], json!(written));
    assert_eq!(v["extension"]["started_ms"], json!(started));
}

// RULING: wave 4 rulings, the line for 21: the purge mark's age.
#[test]
fn the_purge_mark_is_shown_with_its_age() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let (_, v) = status_json(&path);
    assert_eq!(v["purge_pending_ms"], Value::Null);
    sql(&path, "INSERT INTO context_data(key, layer, value, updated) VALUES('purge_pending', 'axel', '1', datetime('now', '-300 seconds'))");
    let (_, v) = status_json(&path);
    let age = v["purge_pending_ms"].as_i64().unwrap();
    assert!((299_000..330_000).contains(&age), "{age}");
    let (_, text) = status_text(&path);
    assert!(text.contains("purge mark: "), "{text}");
}

// ── Exit codes (C-1) ────────────────────────────────────────────────────────

#[test]
fn exit_0_on_a_sound_brain_with_no_status_row() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let (code, v) = status_json(&path);
    assert_eq!(code, 0, "{v:#}");
    assert_eq!(v["extension"]["running"], json!(false));
    assert_eq!(v["extension"]["scan"], json!("pending"));
    assert_eq!(status_text(&path).0, 0);
}

#[test]
fn exit_0_with_a_row_that_says_scan_running() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let now = now_ms();
    write_row(&path, &status_row(now, now, "running", "ready", dir.path()));
    write_status_file(dir.path(), std::process::id(), "ready", "None", None);
    let (code, v) = status_json(&path);
    assert_eq!(code, 0, "{v:#}");
    assert_eq!(v["extension"]["scan"], json!("running"));
    assert_eq!(v["extension"]["running"], json!(true));
    assert_eq!(v["extension"]["mode"], json!("ready"));
}

#[test]
fn exit_1_with_a_status_that_says_degraded() {
    // The status file of a live process (this test's own pid) says degraded.
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let now = now_ms();
    write_row(&path, &status_row(now, now, "done", "ready", dir.path()));
    write_status_file(dir.path(), std::process::id(), "degraded", "FrontStopped", Some("src/ext/front.rs:12"));
    let (code, v) = status_json(&path);
    assert_eq!(code, 1, "{v:#}");
    assert_eq!(v["extension"]["mode"], json!("degraded"));
    assert_eq!(v["extension"]["reason"], json!("FrontStopped"));
    assert_eq!(v["extension"]["reason_at"], json!("src/ext/front.rs:12"));
    // RULING: wave 4 rulings, the line for 21: FrontStopped says "restart Synaps".
    let (code, text) = status_text(&path);
    assert_eq!(code, 1);
    assert!(text.contains("FrontStopped") && text.contains("restart Synaps"), "{text}");

    // A fresh row that says degraded, and no status file at all.
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let mut row = status_row(now_ms(), now_ms(), "done", "degraded", dir.path());
    row["reason"] = json!("IndexerStopped");
    write_row(&path, &row);
    let (code, v) = status_json(&path);
    assert_eq!(code, 1, "{v:#}");
}

#[test]
fn exit_1_and_a_first_line_naming_the_kind_on_a_missing_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("none.r8");
    let (code, text) = status_text(&path);
    assert_eq!(code, 1);
    let first = text.lines().next().unwrap();
    assert!(first.starts_with(&format!("no brain at {}", path.display())), "{text}");
    let (code, v) = status_json(&path);
    assert_eq!(code, 1);
    assert_eq!(v["brain"]["kind"], json!("NoBrain"));
    assert!(v["brain"]["error"].as_str().unwrap().starts_with("no brain at"));
    assert!(!path.exists(), "status created the brain");
    assert!(folder(dir.path()).is_empty(), "{:?}", folder(dir.path()));
}

#[test]
fn exit_1_and_a_first_line_naming_the_kind_on_a_version_1_file() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("axel.r8");
    drop(axel::r8::Brain::create(&path, Some("old")).unwrap());
    let before = folder(dir.path());
    let (code, text) = status_text(&path);
    assert_eq!(code, 1);
    assert!(text.lines().next().unwrap().starts_with("this is the old Axel brain"), "{text}");
    let (code, v) = status_json(&path);
    assert_eq!(code, 1);
    assert_eq!(v["brain"]["kind"], json!("V1Brain"));
    assert_eq!(folder(dir.path()), before, "status changed the folder of a version 1 brain");
}

#[test]
fn exit_1_when_a_notes_file_has_ok_0() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    sql(
        &path,
        "INSERT INTO note_files(path, project_key, size, mtime_ns, ok, reason, live_count, scanned_ms, bad_lines, bad_lines_changed_ms)
         VALUES ('memory/a.json', 'p', 10, 1, 1, 'ids_skipped: 2', 3, 1, 'h1,h2', 1700000000000)",
    );
    let (code, v) = status_json(&path);
    assert_eq!(code, 0, "{v:#}");
    assert_eq!(v["notes"]["list"][0]["bad_lines"], json!(2));
    assert_eq!(v["notes"]["list"][0]["ids_skipped"], json!(2));
    assert_eq!(v["notes"]["list"][0]["bad_lines_changed_ms"], json!(1_700_000_000_000i64));

    sql(&path, "UPDATE note_files SET ok = 0, reason = 'unreadable'");
    let (code, v) = status_json(&path);
    assert_eq!(code, 1, "{v:#}");
    assert_eq!(v["notes"]["not_ok"], json!(1));
    let (code, text) = status_text(&path);
    assert_eq!(code, 1);
    assert!(text.contains("memory/a.json: ok = 0"), "{text}");
}

// ── forget_turns (R3-5; 4.1, risk 2) ────────────────────────────────────────

#[test]
fn forget_turns_top_chats_are_the_three_with_most_in_order_ties_by_logical() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let now = now_ms();
    {
        let conn = Connection::open(&path).unwrap();
        // Four chats with 5, 3, 3 and 1; the 5 is two archives; the two 3s tie and are
        // written in the order that sorts last first.
        archive(&conn, "c5-1.json", "chat-c", 3, now);
        archive(&conn, "c5-2.json", "chat-c", 2, now);
        archive(&conn, "b3.json", "chat-b", 3, now);
        archive(&conn, "a3.json", "chat-a", 3, now);
        archive(&conn, "d1.json", "chat-d", 1, now - 365 * DAY_MS);
        archive(&conn, "zero.json", "chat-z", 0, now);
    }
    let (code, v) = status_json(&path);
    assert_eq!(code, 0);
    let f = &v["forget_turns"];
    assert_eq!(f["units"], json!(12));
    assert_eq!(f["archives"], json!(5));
    assert_eq!(f["units_30d"], json!(11));
    assert_eq!(
        f["top_chats"],
        json!([
            { "logical": "chat-c", "units": 5, "archives": 2 },
            { "logical": "chat-a", "units": 3, "archives": 1 },
            { "logical": "chat-b", "units": 3, "archives": 1 },
        ])
    );
    let (_, text) = status_text(&path);
    assert!(text.contains("12 turns in 5 archives are left out because a forget tool was called in them"), "{text}");
    let c = text.find("chat-c: 5 turns in 2 archives").expect(&text);
    let a = text.find("chat-a: 3 turns in 1 archives").expect(&text);
    let b = text.find("chat-b: 3 turns in 1 archives").expect(&text);
    assert!(c < a && a < b, "{text}");
    assert!(!text.contains("chat-d:"), "{text}");
}

#[test]
fn a_hidden_turn_that_could_not_be_matched_is_named() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    sql(
        &path,
        "INSERT INTO archive_index(scope_dir, rel_name, archive_id, state, unmatched_hidden, first_seen_ms)
         VALUES ('aa', 'x.json', '0123456789abcdef0123456789abcdef', 'indexed', 1, 1)",
    );
    let (_, v) = status_json(&path);
    assert_eq!(v["hidden_turns_unmatched"][0]["archive"], json!("ctx-0123456789abcdef0123456789abcdef"));
    let (_, text) = status_text(&path);
    assert!(text.contains(
        "a hidden turn could not be matched in ctx-0123456789abcdef0123456789abcdef; its archived form is visible"
    ), "{text}");
}

// ── Hidden and superseded (section 5, test 14, the status half; R3-6a) ──────

#[test]
fn hidden_and_superseded_by_the_model_before_and_after_a_restore() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.r8");
    let now = now_ms();
    {
        let brain = open(&path);
        let conn = &brain.conn;
        let signer = brain.signer.as_ref();
        for (id, origin, by, created) in [
            ("mem_t1", "cli", "", now),
            ("mem_t2", "cli", "", now),
            ("mem_t3", "cli", "", now),
            ("mem_old", "cli", "", now - 3 * DAY_MS),
            ("mem_new", "tool", "S", now - DAY_MS),
            ("mem_c", "cli", "", now),
            ("mem_y", "cli", "", now - 400 * DAY_MS),
        ] {
            tx(conn, || store::insert_row(conn, signer, &row(id, origin, by, created)).unwrap());
        }
        tx(conn, || {
            for id in ["mem_t1", "mem_t2", "mem_t3", "mem_y"] {
                assert_eq!(store::hide(conn, id, "model:S").unwrap(), 1);
            }
            assert_eq!(store::hide(conn, "mem_c", "cli").unwrap(), 1);
            assert_eq!(store::supersede(conn, "mem_old", "mem_new").unwrap(), 1);
        });
        conn.execute("UPDATE memory_ext SET hidden_ms = ?1 WHERE mem_id = 'mem_y'", [now - 365 * DAY_MS]).unwrap();
    }
    let (_, v) = status_json(&path);
    assert_eq!(
        (v["hidden_by_model"].clone(), v["superseded_by_model"].clone(), v["hidden_by_model_30d"].clone(), v["superseded_by_model_30d"].clone()),
        (json!(4), json!(1), json!(3), json!(1)),
        "{v:#}"
    );
    assert_eq!(v["hidden_by"]["cli"], json!(1));
    let (_, text) = status_text(&path);
    assert!(text.contains("4 memories hidden and 1 superseded by the assistant, 4 of them in the last 30 days"), "{text}");

    {
        let brain = open(&path);
        let conn = &brain.conn;
        tx(conn, || {
            for id in ["mem_t1", "mem_t2", "mem_t3", "mem_old"] {
                assert!(store::restore(conn, id).unwrap() >= 1, "{id}");
            }
        });
    }
    let (_, v) = status_json(&path);
    assert_eq!(
        (v["hidden_by_model"].clone(), v["superseded_by_model"].clone(), v["hidden_by_model_30d"].clone(), v["superseded_by_model_30d"].clone()),
        (json!(1), json!(0), json!(0), json!(0)),
        "{v:#}"
    );
    let (_, text) = status_text(&path);
    assert!(text.contains("1 memories hidden and 0 superseded by the assistant, 0 of them in the last 30 days"), "{text}");
}

// ── The old brain written after the import (C-4; 4.3, Q-b) ──────────────────

fn record_import(brain: &Path, old: &Path, wal: Option<&Path>) {
    let m = std::fs::symlink_metadata(old).unwrap();
    let (wal_size, wal_mtime) = match wal.map(|w| std::fs::symlink_metadata(w).unwrap()) {
        Some(w) => (json!(w.len()), json!(w.mtime() * 1_000_000_000 + w.mtime_nsec())),
        None => (Value::Null, Value::Null),
    };
    let detail = json!({
        "path": old.to_string_lossy(), "size": m.len(), "mtime_ns": m.mtime() * 1_000_000_000 + m.mtime_nsec(),
        "wal_size": wal_size, "wal_mtime_ns": wal_mtime,
    });
    let conn = Connection::open(brain).unwrap();
    conn.execute(
        "INSERT INTO import_log(at, source, source_sha256, item, rows_in, rows_imported, rows_present, rows_flagged, detail)
         VALUES (datetime('now'), ?1, 'x', 'init', 0, 0, 0, 0, ?2)",
        params![old.to_string_lossy(), detail.to_string()],
    )
    .unwrap();
}

fn written_after(path: &Path) -> Value {
    status_json(path).1["old_brain"]["written_after_import"].clone()
}

#[test]
fn old_brain_written_after_import_by_its_wal_alone() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let old_dir = TempDir::new().unwrap();
    let old = old_dir.path().join("axel.r8");
    std::fs::write(&old, b"the old brain, as it was copied").unwrap();
    record_import(&path, &old, None);
    assert_eq!(written_after(&path), json!(false));

    // A reader's look leaves an empty `-wal` (and a `-shm`), later than the import: not
    // evidence.
    let wal = old_dir.path().join("axel.r8-wal");
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&wal, b"").unwrap();
    std::fs::write(old_dir.path().join("axel.r8-shm"), b"shared memory").unwrap();
    assert_eq!(written_after(&path), json!(false));

    // Only the `-wal` is newer, and holds a frame: written.
    let main_before = std::fs::symlink_metadata(&old).unwrap();
    std::fs::write(&wal, b"a frame a writer left").unwrap();
    let main_after = std::fs::symlink_metadata(&old).unwrap();
    assert_eq!((main_before.len(), main_before.mtime_nsec(), main_before.mtime()), (main_after.len(), main_after.mtime_nsec(), main_after.mtime()));
    let (code, v) = status_json(&path);
    assert_eq!(code, 0, "not an error");
    assert_eq!(v["old_brain"]["written_after_import"], json!(true), "{v:#}");
    let (code, text) = status_text(&path);
    assert_eq!(code, 0);
    let p = old.display();
    assert!(
        text.contains(&format!("the old brain {p} was written after the import (")),
        "{text}"
    );
    assert!(text.contains(&format!("\"axel migrate catch-up --from-axel {p}\" brings over what was saved there")), "{text}");

    // A new record (a catch-up) taken now ends the line.
    record_import(&path, &old, Some(&wal));
    assert_eq!(written_after(&path), json!(false));
}

#[test]
fn old_brain_written_after_import_by_its_main_file() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let old_dir = TempDir::new().unwrap();
    let old = old_dir.path().join("axel.r8");
    std::fs::write(&old, b"the old brain").unwrap();
    record_import(&path, &old, None);
    assert_eq!(written_after(&path), json!(false));
    std::fs::write(&old, b"the old brain, written again").unwrap();
    assert_eq!(written_after(&path), json!(true));
}

// ── The status files (RULING: wave 4 rulings, the line for 21) ──────────────

/// A pid that was alive and is not any more.
fn dead_pid() -> u32 {
    let mut child = Command::new("true").spawn().unwrap();
    let pid = child.id();
    child.wait().unwrap();
    pid
}

#[test]
fn a_status_file_whose_pid_is_not_alive_is_not_ready() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let pid = dead_pid();
    write_status_file(dir.path(), pid, "ready", "None", None);
    let (code, v) = status_json(&path);
    assert_eq!(code, 0);
    assert_eq!(v["extension"]["running"], json!(false), "{v:#}");
    assert_eq!(v["extension"]["mode"], Value::Null);
    assert_eq!(v["extension"]["stale_status_files"][0]["pid"], json!(pid));
    let (_, text) = status_text(&path);
    assert!(text.contains(&format!("stale status file found: pid {pid}")), "{text}");
    assert!(text.contains("extension: not running"), "{text}");

    // A dead process that was degraded does not make the exit 1 either.
    write_status_file(dir.path(), pid, "degraded", "FrontStopped", None);
    assert_eq!(status_json(&path).0, 0);
}

// RULING: wave 5 fix round of 21, finding 7: an alive pid whose process started after the
// file's `started_ms` is another process that was given the same number.
#[test]
fn a_status_file_whose_pid_was_reused_by_a_later_process_is_not_ready() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let pid = std::process::id();
    let old = json!({
        "pid": pid, "version": "test", "mode": "degraded", "reason": "FrontStopped", "reason_at": null,
        "written_ms": now_ms(), "started_ms": now_ms() - DAY_MS,
    });
    std::fs::write(dir.path().join(format!("extension.{pid}.status")), old.to_string()).unwrap();
    let (code, v) = status_json(&path);
    assert_eq!(v["extension"]["running"], json!(false), "{v:#}");
    assert_eq!(v["extension"]["stale_status_files"][0]["pid"], json!(pid), "{v:#}");
    assert_eq!(code, 0, "{v:#}");
}

#[test]
fn two_live_processes_are_listed_and_the_worst_is_on_top() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let mut child = Command::new("sleep").arg("30").spawn().unwrap();
    write_status_file(dir.path(), std::process::id(), "ready", "None", None);
    write_status_file(dir.path(), child.id(), "starting", "None", None);
    let (code, v) = status_json(&path);
    let _ = child.kill();
    let _ = child.wait();
    assert_eq!(code, 0);
    assert_eq!(v["extension"]["mode"], json!("starting"));
    assert_eq!(v["extension"]["processes"].as_array().map(Vec::len), Some(2), "{v:#}");
}

// ── Nothing is written by a read ─────────────────────────────────────────────

#[test]
fn status_creates_no_side_file_and_changes_no_file() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let now = now_ms();
    write_row(&path, &status_row(now, now, "done", "ready", dir.path()));
    write_status_file(dir.path(), std::process::id(), "ready", "None", None);
    let before = folder(dir.path());
    assert!(!before.contains_key("brain.r8-wal") && !before.contains_key("brain.r8-shm"), "{before:?}");
    for args in [&["status"][..], &["status", "--json"], &["status", "--check"], &["status", "--json", "--check"]] {
        let out = axel(&path, args);
        assert!(out.status.code().is_some(), "{args:?}");
        assert_eq!(folder(dir.path()), before, "{args:?} changed the folder");
    }
}

// RULING: wave 5 fix round of 21, finding 2: a symlinked brain's side files are named
// after the target, and a `-wal` with frames in it means an ordinary read-only open.
#[test]
fn a_brain_reached_through_a_link_whose_target_has_a_wal_is_not_read_immutable() {
    let dir = TempDir::new().unwrap();
    let real = new_brain(&dir);
    let link = dir.path().join("link.r8");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    assert_eq!(status_json(&link).1["brain"]["read_as"], json!("immutable"));

    // An open writer whose frames are still in the `-wal`.
    let writer = Connection::open(&real).unwrap();
    let mode: String = writer.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0)).unwrap();
    assert_eq!(mode, "wal");
    writer.execute_batch("PRAGMA wal_autocheckpoint = 0; CREATE TABLE held(x); INSERT INTO held VALUES (1);").unwrap();
    assert!(std::fs::metadata(dir.path().join("brain.r8-wal")).unwrap().len() > 0);
    assert!(!dir.path().join("link.r8-wal").exists());
    let (_, v) = status_json(&link);
    drop(writer);
    assert_eq!(v["brain"]["read_as"], json!("read-only"), "{v:#}");
    assert_eq!(v["brain"]["version"], json!(brain_v2::LATEST), "{v:#}");
}

// RULING: wave 5 fix round of 21, finding 2: a file written between the check and the end
// of an immutable read is read again with `mode=ro`.
#[test]
fn a_change_during_an_immutable_read_leads_to_a_second_read() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let home = dir.path().join("home");
    let quiet = axel::cli::status::collect(&path, false, now_ms(), Some(&home));
    assert_eq!(quiet.json["brain"]["read_as"], json!("immutable"));

    let changed = axel::cli::status::collect_with(&path, false, now_ms(), Some(&home), &|real: &Path| {
        sql(real, "CREATE TABLE written_meanwhile(x); DROP TABLE written_meanwhile;")
    });
    assert_eq!(changed.json["brain"]["read_as"], json!("read-only after a change"), "{:#}", changed.json);
    assert!(!changed.wrong, "{:#}", changed.json);
}

#[test]
fn check_on_a_sound_brain_exits_0() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let out = axel(&path, &["status", "--json", "--check"]);
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(out.status.code(), Some(0), "{v:#}");
    assert_eq!(v["check"]["integrity_ok"], json!(true));
    assert_eq!(v["check"]["counts_ok"], json!(true));
}

// RULING: wave 5 fix round of 21, finding 3: a memory whose `content` is not UTF-8 is
// text to `typeof` but is skipped by adoption; `--check` counts it the way adoption does.
#[test]
fn a_memory_whose_content_is_not_utf8_does_not_make_a_sound_brain_mismatch() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    sql(
        &path,
        "INSERT INTO memories (id, category, topic, title, abstract_text, content, confidence, importance,
           source_sessions, tags, related_topics, created, trust_level)
         VALUES ('mem_bad', 'events', 't', 'x', '', CAST(X'66FF6F' AS TEXT), 'high', 0.5, '[]', '[]', '[]',
           '2025-01-01T00:00:00Z', 1.0)",
    );
    let typeof_: String = Connection::open(&path)
        .unwrap()
        .query_row("SELECT typeof(content) FROM memories WHERE id = 'mem_bad'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(typeof_, "text");
    drop(open(&path)); // adoption runs, and skips it
    let out = axel(&path, &["status", "--json", "--check"]);
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["check"]["unreadable"], json!(1), "{v:#}");
    assert_eq!(v["check"]["counts_ok"], json!(true), "{v:#}");
    assert_eq!(v["unreadable_memories"], json!(1), "{v:#}");
    assert_eq!(out.status.code(), Some(0), "{v:#}");
}

// RULING: wave 4 rulings, the line for 21: what came from a row passes `screen::show`.
#[test]
fn what_status_prints_from_a_row_is_shown_on_one_line() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let conn = Connection::open(&path).unwrap();
    conn.execute(
        "INSERT INTO note_files(path, project_key, size, mtime_ns, ok, reason, live_count, scanned_ms)
         VALUES (?1, 'p', 1, 1, 0, 'bad', 0, 1)",
        ["memory/a.json\nextension: ready [forged]"],
    )
    .unwrap();
    drop(conn);
    let (_, text) = status_text(&path);
    assert!(!text.lines().any(|l| l.starts_with("extension: ready")), "{text}");
    assert!(!text.contains("[forged]"), "{text}");
}

// ── why ─────────────────────────────────────────────────────────────────────

#[test]
fn why_last_1_prints_the_pick_of_a_logged_recall() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("brain.r8");
    {
        let brain = open(&path);
        let conn = &brain.conn;
        tx(conn, || store::insert_row(conn, brain.signer.as_ref(), &row("mem_pick", "cli", "", now_ms())).unwrap());
        let detail = |r: &str| {
            json!({
                "v": 1, "query_tokens": [], "embed": { "used": false, "ms": null, "fallback": "model_unavailable" },
                "thresholds": { "min_sem": 0.3, "min_lex": 0.5 },
                "generators": { "mem_words": 1, "doc_words": 0, "by_meaning": 0, "past_floor": 1, "after_dedupe": 1 },
                "picks": [{ "n": 1, "ref": r, "kind": "note", "origin": "cli", "scope": "project",
                            "created": "2026-11-10T21:04:11Z",
                            "parts": { "lexical": 0.67, "semantic": 0.0, "session": 0.0, "class": 0.9, "recency": 0.82, "total": 2.43 },
                            "reasons": ["lexical_match"] }],
                "dropped": [{ "ref": "doc:notes::x", "why": "below_floor" }],
            })
            .to_string()
        };
        for (ts, r) in [(1_000i64, "m:mem_older"), (2_000, "m:mem_pick")] {
            conn.execute(
                "INSERT INTO recall_log(ts_ms, session_id, logical, query_hash, outcome, total_ms, candidates, selected, detail)
                 VALUES (?1, 'S', 'L', 'h', 'injected', 12.5, 3, 1, ?2)",
                params![ts, detail(r)],
            )
            .unwrap();
        }
    }
    let out = axel(&path, &["why", "--last", "1"]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("1. [m:mem_pick] saved with the axel command"), "{text}");
    assert!(text.contains("total 2.43 = words 0.67×2.0 + meaning 0×2.5 + same chat 0 + kind 0.9×0.75 + recent 0.82×0.5 · lexical_match"), "{text}");
    assert!(text.contains("title of mem_pick"), "{text}");
    assert!(text.contains("dropped [doc:notes::x] below_floor"), "{text}");
    assert!(!text.contains("mem_older"), "{text}");

    let out = axel(&path, &["why", "--last", "2", "--json"]);
    let lines: Vec<Value> =
        String::from_utf8(out.stdout).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1]["outcome"], json!("injected"));
    assert_eq!(lines[1]["total_ms"], json!(12.5));
    assert_eq!(lines[1]["picks"][0]["title"], json!("title of mem_pick"));
    // A row that is gone leaves no title.
    assert_eq!(lines[0]["picks"][0]["title"], Value::Null);
    assert_eq!(lines[0]["picks"][0]["row"], json!("gone"));

    let out = axel(&path, &["why", "--session", "other"]);
    assert_eq!(String::from_utf8(out.stdout).unwrap().trim(), "no recall is logged");
}

// ── Only `model fetch` goes online ──────────────────────────────────────────

#[test]
fn every_entry_point_but_model_fetch_leaves_the_process_offline() {
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    let brain = axel::ext::types::BrainArg { path: path.clone(), from_env: false, on_command_line: true };
    assert!(velocirag::download::is_offline());
    axel::cli::status::run(&axel::cli::status::Args { json: true, check: true }, &brain).unwrap();
    assert!(velocirag::download::is_offline());
    axel::cli::why::run(&axel::cli::why::Args { last: Some(3), session: None, json: false }, &brain).unwrap();
    assert!(velocirag::download::is_offline());

    // And no other place in the binary's source switches it off.
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found = Vec::new();
    let mut stack = vec![src.clone()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let text = std::fs::read_to_string(&p).unwrap();
                if text.contains("set_offline(false)") {
                    found.push(p.strip_prefix(&src).unwrap().to_path_buf());
                }
            }
        }
    }
    assert_eq!(found, vec![PathBuf::from("cli/model.rs")]);
}

// ── The process by its start ticks (RULING: v2 c18a review, item 5) ─────────

/// Field 22 of `/proc/<pid>/stat`; `None` where there is no `/proc`.
fn start_ticks(pid: u32) -> Option<u64> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    stat.rsplit_once(')')?.1.split_whitespace().nth(19)?.parse().ok()
}

/// A status file of `pid` that says `ready`, with these start ticks and `started_ms`.
fn ticked_status_file(dir: &Path, pid: u32, ticks: Option<u64>, started_ms: i64) {
    let body = json!({
        "pid": pid, "mode": "ready", "reason": "None", "reason_at": null,
        "written_ms": now_ms(), "started_ms": started_ms, "start_ticks": ticks,
    });
    std::fs::write(dir.join(format!("extension.{pid}.status")), body.to_string()).unwrap();
}

// A wall clock stepped forward after the extension started makes its `started_ms` look a
// day older than the start `btime` gives; the ticks are the same, so it is the same process.
#[test]
fn a_live_extension_is_known_by_its_start_ticks_after_the_wall_clock_stepped() {
    let mut child = Command::new("sleep").arg("30").spawn().unwrap();
    let Some(ticks) = start_ticks(child.id()) else {
        // No `/proc` (macOS): the wall-clock test of finding 7 is all there is.
        let _ = child.kill();
        let _ = child.wait();
        return;
    };
    let dir = TempDir::new().unwrap();
    let path = new_brain(&dir);
    ticked_status_file(dir.path(), child.id(), Some(ticks), now_ms() - DAY_MS);
    let (code, v) = status_json(&path);

    // Other ticks, a `started_ms` of now: another process holds the pid.
    ticked_status_file(dir.path(), child.id(), Some(ticks + 1), now_ms());
    let (_, reused) = status_json(&path);
    // No ticks in the file (an older extension): the wall-clock test, as before.
    ticked_status_file(dir.path(), child.id(), None, now_ms() - DAY_MS);
    let (_, older) = status_json(&path);
    let _ = child.kill();
    let _ = child.wait();

    assert_eq!(code, 0, "{v:#}");
    assert_eq!(v["extension"]["running"], json!(true), "{v:#}");
    assert_eq!(v["extension"]["pid"], json!(child.id()), "{v:#}");
    assert_eq!(v["extension"]["stale_status_files"], json!([]), "{v:#}");

    assert_eq!(reused["extension"]["running"], json!(false), "{reused:#}");
    assert_eq!(reused["extension"]["stale_status_files"][0]["why"], json!("pid reused by a later process"), "{reused:#}");

    assert_eq!(older["extension"]["running"], json!(false), "{older:#}");
    assert_eq!(older["extension"]["stale_status_files"][0]["why"], json!("pid reused by a later process"), "{older:#}");
}
