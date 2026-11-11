//! Commit 19, `migrate init | synaps | verify | catch-up`: spec section 1, tests 9 to 16, 21
//! and 22, and the rulings made after the spec was frozen (a forum date later than now,
//! rows that are not text, an empty brain in the way, the base from the flag).
//!
//! The two sources are built from the `CREATE TABLE` text in `fixtures/` and synthetic rows.
//! Everything is in temp folders; no test resolves a default path.

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;

use axel::brain_v2::{self, OpenOpts, Role};
use axel::ext::types::{BrainArg, Caller};
use axel::forum::{Author, Envelope, Post, Read, Status as PostStatus};
use axel::import::{self, CatchUpOpts, InitOpts, Outcome, SynapsOpts, VerifyOpts};
use axel::{forum_store, index_archives, store};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const AXEL_SQL: &str = include_str!("fixtures/schema_axel_r8.sql");
const BRAIN_SQL: &str = include_str!("fixtures/schema_brain_r8.sql");
const X: &str = "p1111111111111111";
const Y: &str = "p2222222222222222";
const Z: &str = "p3333333333333333";
const DAY_MS: i64 = 86_400_000;
const M1: &str = "mem-aaaa0001";
const M2: &str = "mem-bbbb0002";
const M3: &str = "mem-cccc0003";

// ── Fixture ─────────────────────────────────────────────────────────────────

struct Fx {
    dir: TempDir,
    /// The old Axel brain.
    old: PathBuf,
    /// The Synaps brain.
    syn: PathBuf,
    /// The destination.
    new: PathBuf,
    base: PathBuf,
}

fn fx() -> Fx {
    let dir = tempfile::tempdir().unwrap();
    // The temp folder may be reached through a link; every path here is the real one.
    let root = dir.path().canonicalize().unwrap();
    for sub in ["old", "syn", "new", "base"] {
        fs::create_dir(root.join(sub)).unwrap();
    }
    Fx {
        old: root.join("old/axel.r8"),
        syn: root.join("syn/brain.r8"),
        new: root.join("new/brain.r8"),
        base: root.join("base"),
        dir,
    }
}

impl Fx {
    fn root(&self) -> PathBuf {
        self.dir.path().canonicalize().unwrap()
    }
    fn dest(&self) -> BrainArg {
        arg(&self.new)
    }
    fn init_opts(&self, from: &Path) -> InitOpts {
        InitOpts { from_axel: from.to_path_buf(), synaps_base: Some(self.base.clone()), ..InitOpts::default() }
    }
    fn init(&self, from: &Path) -> Outcome {
        import::init(&self.dest(), &self.init_opts(from))
    }
    fn synaps(&self, map: &[&str], accept_foreign: bool) -> Outcome {
        import::synaps(
            &self.dest(),
            &SynapsOpts { from: self.syn.clone(), map: map.iter().map(|s| s.to_string()).collect(), accept_foreign },
        )
    }
    fn verify_at(&self, brain: &Path, from_axel: &Path, map: &[&str]) -> Outcome {
        import::verify(
            &arg(brain),
            &VerifyOpts {
                from_axel: from_axel.to_path_buf(),
                from_synaps: self.syn.clone(),
                map: map.iter().map(|s| s.to_string()).collect(),
                accept_foreign: false,
            },
        )
    }
    fn verify(&self, map: &[&str]) -> Outcome {
        self.verify_at(&self.new, &self.old, map)
    }
    fn catch_up(&self) -> Outcome {
        import::catch_up(&self.dest(), &CatchUpOpts { from_axel: self.old.clone(), watch: None })
    }
}

fn arg(path: &Path) -> BrainArg {
    BrainArg { path: path.to_path_buf(), from_env: false, on_command_line: true }
}

fn show(name: &str, out: &Outcome) {
    println!("REPORT {name} (exit {}): {}", out.code, out.report);
}

fn sha(path: &Path) -> String {
    hex::encode(Sha256::digest(fs::read(path).unwrap()))
}

fn side(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

fn listing(folder: &Path) -> Vec<String> {
    let mut names: Vec<String> =
        fs::read_dir(folder).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    names
}

/// Hash, size, mode, modification time and change time of a source and of the files beside
/// it, and the names and the times of its folder. A write, a chmod, a new file, or a file
/// made and removed again beside it (what SQLite does when it opens a database to write)
/// changes it; a read does not.
#[derive(PartialEq, Eq, Debug)]
struct State {
    files: Vec<Option<FileState>>,
    names: Vec<String>,
    folder: ((i64, i64), (i64, i64)),
}

#[derive(PartialEq, Eq, Debug)]
struct FileState {
    sha: String,
    size: u64,
    mode: u32,
    mtime: (i64, i64),
    ctime: (i64, i64),
}

fn state(path: &Path) -> State {
    let mut files = Vec::new();
    for suffix in ["", "-wal", "-shm", "-journal"] {
        let file = side(path, suffix);
        files.push(fs::symlink_metadata(&file).ok().map(|m| FileState {
            sha: sha(&file),
            size: m.len(),
            mode: m.mode(),
            mtime: (m.mtime(), m.mtime_nsec()),
            ctime: (m.ctime(), m.ctime_nsec()),
        }));
    }
    let folder = fs::metadata(path.parent().unwrap()).unwrap();
    State {
        files,
        names: listing(path.parent().unwrap()),
        folder: ((folder.mtime(), folder.mtime_nsec()), (folder.ctime(), folder.ctime_nsec())),
    }
}

fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |r| r.get(0)).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn text(conn: &Connection, sql: &str) -> String {
    conn.query_row(sql, [], |r| r.get(0)).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn ids(v: &Value) -> Vec<String> {
    v["ids"].as_array().unwrap_or_else(|| panic!("no ids in {v}")).iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

fn item<'a>(report: &'a Value, name: &str) -> &'a Value {
    report["items"]
        .as_array()
        .unwrap_or_else(|| panic!("no items in {report}"))
        .iter()
        .find(|i| i["item"] == name)
        .unwrap_or_else(|| panic!("no item {name} in {report}"))
}

fn shortfalls(report: &Value) -> Vec<String> {
    report["shortfalls"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
}

/// The temporary directories of this crate's commands in a folder.
fn temporaries(folder: &Path) -> Vec<String> {
    listing(folder).into_iter().filter(|n| n.starts_with(".init-") || n.starts_with(".verify-") || n.starts_with(".catchup-")).collect()
}

fn vector(seed: f32) -> Vec<u8> {
    (0..384).flat_map(|i| (seed + i as f32 / 1000.0).to_le_bytes()).collect()
}

// ── The old Axel brain ──────────────────────────────────────────────────────

fn add_memory(conn: &Connection, id: &str, title: &str, content: &str, seed: f32) {
    conn.execute(
        "INSERT INTO memories(id, category, topic, title, abstract_text, content, created, signature)
         VALUES (?1, 'events', 'sea', ?2, ?3, ?4, '2026-01-02T03:04:05+00:00', 'sig')",
        params![id, title, format!("about {title}"), content],
    )
    .unwrap();
    // The shadow document `index_memory` wrote for each memory.
    conn.execute(
        "INSERT INTO documents(doc_id, content, metadata, embedding, indexed_at) VALUES (?1, ?2, ?3, ?4, CURRENT_TIMESTAMP)",
        params![id, content, json!({ "type": "memory" }).to_string(), vector(seed)],
    )
    .unwrap();
    conn.execute("INSERT INTO chunks_fts(doc_id, content, file_path) VALUES (?1, ?2, NULL)", params![id, content]).unwrap();
}

fn add_document(conn: &Connection, doc_id: &str, content: &str, file: &str, seed: f32) {
    conn.execute(
        "INSERT INTO documents(doc_id, content, metadata, embedding, file_path, indexed_at)
         VALUES (?1, ?2, '{}', ?3, ?4, CURRENT_TIMESTAMP)",
        params![doc_id, content, vector(seed), file],
    )
    .unwrap();
    conn.execute("INSERT INTO chunks_fts(doc_id, content, file_path) VALUES (?1, ?2, ?3)", params![doc_id, content, file])
        .unwrap();
}

fn add_access(conn: &Connection, doc_id: &str) {
    conn.execute("INSERT INTO document_access(doc_id, access_type, query) VALUES (?1, 'search', 'tides')", [doc_id]).unwrap();
}

/// A version 1 brain: 3 memories with their shadow documents, a note-file document, a
/// handoff, a staged memory, a pattern, and usage rows. In WAL mode, as the real file is.
fn make_axel(path: &Path) {
    let conn = Connection::open(path).unwrap();
    let mode: String = conn.query_row("PRAGMA journal_mode=WAL", [], |r| r.get(0)).unwrap();
    assert_eq!(mode, "wal");
    conn.execute_batch(AXEL_SQL).unwrap();
    let meta = json!({
        "schema_version": 1, "embedder_model": "hash-384", "embedding_dim": 384, "agent_name": "old",
        "created": "2026-01-01T00:00:00+00:00", "last_modified": "2026-01-01T00:00:00+00:00",
        "document_count": 0, "memory_count": 0, "signing_key": "ab".repeat(32),
    });
    conn.execute("INSERT INTO brain_meta(key, value) VALUES ('meta', ?1)", [meta.to_string()]).unwrap();
    conn.execute("INSERT INTO kv(key, value) VALUES ('schema_version', '3')", []).unwrap();
    add_memory(&conn, M1, "harbor note", "the otter keeps  the harbor ledger", 0.25);
    add_memory(&conn, M2, "lantern note", "a narwhal carried the lantern home", -0.5);
    add_memory(&conn, M3, "kiln note", "the potter fired the kiln at dawn", 0.75);
    add_document(&conn, "note-1", "a note file about tides", "/notes/tides.md", 0.0);
    conn.execute_batch(
        "INSERT INTO context_data(key, value, layer, updated) VALUES ('handoff', 'first handoff', 'session', '2026-01-01 10:00:00');
         INSERT INTO context_data(key, value, layer, updated) VALUES ('theme', 'dark', 'prefs', '2026-01-01 10:00:00');
         INSERT INTO staged_memories(id, category, topic, title, content, created, staged_at)
              VALUES ('stg-1', 'events', 'sea', 'staged one', 'a staged thought', '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00');
         INSERT INTO patterns(pattern_type, name, description) VALUES ('habit', 'morning', 'writes at dawn');
         INSERT INTO memory_access(memory_id, access_type) VALUES ('mem-bbbb0002', 'recall');
         INSERT INTO sessions(id, source) VALUES ('s-1', 'cli');
         INSERT INTO events(event_type, target_id) VALUES ('search', 'note-1');
         INSERT INTO co_retrieval(doc_id_a, doc_id_b, query) VALUES ('note-1', 'mem-bbbb0002', 'tides');",
    )
    .unwrap();
    add_access(&conn, "note-1");
    add_access(&conn, M2);
    conn.close().unwrap();
    assert!(!side(path, "-wal").exists());
}

/// Runs `change` on the source and closes it, so that no log is left beside it.
fn change_axel(path: &Path, change: impl FnOnce(&Connection)) {
    let conn = Connection::open(path).unwrap();
    change(&conn);
    conn.close().unwrap();
    assert!(!side(path, "-wal").exists());
}

/// A copy of the old brain taken while a writer has a row in the log that is not in the
/// main file: `<folder>/axel.r8`, `-wal` (not empty) and `-shm`, all read-only.
fn snapshot_with_log(old: &Path, folder: &Path) -> PathBuf {
    fs::create_dir(folder).unwrap();
    let conn = Connection::open(old).unwrap();
    conn.execute_batch("PRAGMA wal_autocheckpoint=0").unwrap();
    add_memory(&conn, "mem-dddd0004", "kelp note", "the heron counted kelp at dusk", 0.5);
    assert!(fs::metadata(side(old, "-wal")).unwrap().len() > 0);
    let copy = folder.join("axel.r8");
    for suffix in ["", "-wal", "-shm"] {
        fs::copy(side(old, suffix), side(&copy, suffix)).unwrap();
        fs::set_permissions(side(&copy, suffix), fs::Permissions::from_mode(0o400)).unwrap();
    }
    drop(conn);
    // The main file of the copy does not have the row: only its log does.
    let bytes = fs::read(&copy).unwrap();
    assert!(!bytes.windows(b"heron counted kelp".len()).any(|w| w == b"heron counted kelp"));
    copy
}

fn empty_v2(path: &Path) {
    let opts = OpenOpts { agent: None, model: "hash-384", dim: 384, upgrade_v1: false };
    drop(brain_v2::open_v2(path, &opts).unwrap());
}

/// Applies the log and closes, so that the brain is one file.
fn settle(path: &Path) {
    let conn = Connection::open(path).unwrap();
    let busy: i64 = conn.query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |r| r.get(0)).unwrap();
    assert_eq!(busy, 0);
    conn.close().unwrap();
}

/// A copy of a brain to break, in a folder of its own.
fn clone_brain(fx: &Fx, from: &Path, name: &str) -> PathBuf {
    settle(from);
    let folder = fx.root().join(name);
    fs::create_dir(&folder).unwrap();
    let to = folder.join("brain.r8");
    fs::copy(from, &to).unwrap();
    to
}

fn word_hits(conn: &Connection, word: &str) -> Vec<String> {
    let mut stmt = conn.prepare("SELECT mem_id FROM memories_fts WHERE memories_fts MATCH ?1 ORDER BY mem_id").unwrap();
    let rows = stmt.query_map([word], |r| r.get(0)).unwrap();
    rows.collect::<rusqlite::Result<_>>().unwrap()
}

// ── The Synaps brain ────────────────────────────────────────────────────────

fn hex32(tag: &str) -> String {
    hex::encode(&Sha256::digest(tag.as_bytes())[..16])
}

fn who(tag: &str) -> Author {
    let author = Author { actor: format!("actor-{}", hex32(tag)), group: format!("group-{}", hex32("group")), parent: None };
    author.validate().unwrap();
    author
}

fn root_post(key: &str, title: &str, body: &str, days: u32) -> Post {
    Post { request_key: key.into(), thread_id: None, reply_to: None, title: title.into(), body: body.into(), retention_days: days }
}

fn synaps_conn(path: &Path) -> Connection {
    let fresh = !path.exists();
    let conn = Connection::open(path).unwrap();
    if fresh {
        conn.execute_batch(BRAIN_SQL).unwrap();
    }
    conn
}

fn memory_row(conn: &Connection, id: &str, key: Option<&str>, provenance: &str, content: &str) {
    conn.execute(
        "INSERT INTO memories(id, category, topic, title, abstract_text, content, created, project_key, provenance)
         VALUES (?1, 'events', 'synaps', '', '', ?2, '1970-01-01T00:00:00+00:00', ?3, ?4)",
        params![id, content, key, provenance],
    )
    .unwrap();
}

/// A forum post as the helper stores it: the envelope in `record.meta._synaps_forum`, the
/// body in the `content` column. `stored_body` is what the column holds.
fn forum_row(conn: &Connection, project: &str, author: &Author, post: &Post, ts: i64, stored_body: &str) -> String {
    let envelope = Envelope::new(project, author.clone(), post).unwrap();
    let id = envelope.id();
    let record = json!({
        "namespace": "forum", "timestamp_ms": ts, "content": "", "tags": [],
        "meta": { "_synaps_forum": envelope }, "id": id, "project": project,
        "provenance": { "source": envelope.source(), "session": author.group },
        "sensitivity": "normal", "retention": { "max_age_days": post.retention_days },
    });
    let provenance = json!({ "schema": "synaps-axel/1", "record": record, "content_bytes": stored_body.len() });
    memory_row(conn, &id, Some(project), &provenance.to_string(), stored_body);
    id
}

fn note_row(conn: &Connection, key: &str, id: &str, content: &str) {
    let record = json!({
        "namespace": format!("project-{key}"), "timestamp_ms": 5, "content": "", "tags": ["t"], "id": id,
        "project": key, "provenance": { "source": "test" }, "sensitivity": "normal", "retention": "standard",
    });
    let provenance = json!({ "schema": "synaps-axel/1", "record": record, "content_bytes": content.len() });
    memory_row(conn, id, Some(key), &provenance.to_string(), content);
}

fn history_row(conn: &Connection, key: &str, id: &str) {
    conn.execute(
        "INSERT INTO synaps_history(project_key, id, logical, digest, source_count, message_count, messages, note, created_ms)
         VALUES (?1, ?2, 'l', 'd', 1, 1, '[]', '', 7)",
        params![key, id],
    )
    .unwrap();
}

fn tombstone_row(conn: &Connection, id: &str, key: &str) {
    conn.execute(
        "INSERT INTO memory_tombstones(id, project_key, deleted_at) VALUES (?1, ?2, '2026-02-03T04:05:06+00:00')",
        params![id, key],
    )
    .unwrap();
}

struct Posts {
    root: String,
    reply: String,
    expired: String,
    author: Author,
    root_post: Post,
}

/// 3 forum posts under `X`: a live root, a reply to it, an expired root.
fn three_posts(conn: &Connection, now: i64) -> Posts {
    let author = who("ada");
    let root_post = root_post("finding-1", "Finding", "Synthetic peer note", 30);
    let root = forum_row(conn, X, &author, &root_post, now - 5_000, &root_post.body);
    let reply_post = Post {
        request_key: "reply-1".into(),
        thread_id: Some(root.clone()),
        reply_to: Some(root.clone()),
        title: String::new(),
        body: "A synthetic answer".into(),
        retention_days: 30,
    };
    let reply = forum_row(conn, X, &who("bo"), &reply_post, now - 4_000, &reply_post.body);
    let old = root_post_of("old-1", "Old", "Long gone");
    let expired = forum_row(conn, X, &author, &old, 1_600_000_000_000, &old.body);
    Posts { root, reply, expired, author, root_post }
}

fn root_post_of(key: &str, title: &str, body: &str) -> Post {
    root_post(key, title, body, 1)
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

fn forum_ids(conn: &Connection) -> Vec<String> {
    let mut stmt = conn.prepare("SELECT id FROM forum_posts ORDER BY id").unwrap();
    let rows = stmt.query_map([], |r| r.get(0)).unwrap();
    rows.collect::<rusqlite::Result<_>>().unwrap()
}

fn forget_in_new_brain(brain: &Path, project: &str, id: &str) {
    let conn = brain_v2::connect(brain, Role::Cli).unwrap();
    brain_v2::begin_write(&conn).unwrap();
    let caller = Caller { session_id: Some("s-test".into()), cwd: None, author: Some(who("ada")) };
    forum_store::forget(&conn, project, id, &caller, now_ms()).unwrap();
    conn.execute_batch("COMMIT").unwrap();
}

fn purge_in_new_brain(brain: &Path, id: &str) {
    let conn = brain_v2::connect(brain, Role::Cli).unwrap();
    brain_v2::begin_write(&conn).unwrap();
    assert_eq!(store::purge_many(&conn, &[id.to_string()], true, true).unwrap(), 1);
    index_archives::mark_purged(&conn).unwrap();
    conn.execute_batch("COMMIT").unwrap();
}

// ── 9 ───────────────────────────────────────────────────────────────────────

#[test]
fn t09_init_carries_the_log_and_leaves_read_only_sources_as_they_were() {
    let fx = fx();
    make_axel(&fx.old);
    let source = snapshot_with_log(&fx.old, &fx.root().join("snap"));
    // The folder is read-only too: nothing can appear beside the source.
    let folder = source.parent().unwrap().to_path_buf();
    fs::set_permissions(&folder, fs::Permissions::from_mode(0o500)).unwrap();
    let before = state(&source);
    assert!(before.files[1].as_ref().is_some_and(|f| f.size > 0), "the fixture has no log");
    // A directory a killed run left.
    let stale = fx.new.parent().unwrap().join(".init-999999999-1");
    fs::create_dir(&stale).unwrap();
    fs::write(stale.join("axel.r8"), b"left behind").unwrap();

    let out = fx.init(&source);
    show("init", &out);
    let after = state(&source);
    fs::set_permissions(&folder, fs::Permissions::from_mode(0o700)).unwrap();
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(out.report["ok"], true);
    assert_eq!(before, after, "a source file changed, or a file appeared beside it");

    // The log was carried: the row that was only in `-wal` is in the new brain.
    assert_eq!(out.report["source_log_carried"], true);
    let conn = Connection::open(&fx.new).unwrap();
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM memories"), 4);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM memories WHERE id = 'mem-dddd0004'"), 1);
    assert_eq!(text(&conn, "SELECT content FROM memories WHERE id = 'mem-aaaa0001'"), "the otter keeps  the harbor ledger");
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM documents"), 5);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM document_access"), 2);
    assert_eq!(text(&conn, "SELECT value FROM context_data WHERE key = 'handoff'"), "first handoff");
    assert_eq!(text(&conn, "SELECT json_extract(value, '$.signing_key') FROM brain_meta"), "ab".repeat(32));
    // Version 2, every memory adopted as legacy with its vector, and found by a word.
    assert_eq!(count(&conn, "PRAGMA user_version"), 1);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM memory_ext WHERE origin_kind = 'legacy' AND embedding IS NOT NULL"), 4);
    assert_eq!(word_hits(&conn, "heron"), ["mem-dddd0004"]);
    assert_eq!(word_hits(&conn, "narwhal"), [M2]);
    let vec: Vec<u8> = conn.query_row("SELECT embedding FROM memory_ext WHERE mem_id = ?1", [M1], |r| r.get(0)).unwrap();
    assert_eq!(vec, vector(0.25));
    // The base, from the flag.
    assert_eq!(text(&conn, "SELECT value FROM context_data WHERE layer = 'axel' AND key = 'synaps_base'"), fx.base.to_string_lossy());
    assert_eq!(out.report["base_recorded"], json!(fx.base.to_string_lossy()));
    // What `status` will watch: the source's path and what it looked like.
    let detail: Value = serde_json::from_str(&text(&conn, "SELECT detail FROM import_log WHERE item = 'init'")).unwrap();
    assert_eq!(detail["path"], json!(source.to_string_lossy()));
    assert_eq!(detail["size"], json!(fs::metadata(&source).unwrap().len()));
    assert_eq!(detail["wal_size"], json!(fs::metadata(side(&source, "-wal")).unwrap().len()));
    assert!(detail["mtime_ns"].is_i64() && detail["wal_mtime_ns"].is_i64());
    assert_eq!(text(&conn, "SELECT source_sha256 FROM import_log WHERE item = 'init'"), sha(&source));
    drop(conn);

    assert_eq!(out.report["unreadable_memories"], 0);
    assert_eq!(out.report["memories_not_text"], 0);
    assert_eq!(out.report["memories_fts_columns_ok"], true);
    assert_eq!(out.report["integrity_check"], "ok");
    assert_eq!(fs::metadata(&fx.new).unwrap().mode() & 0o777, 0o600);
    // No temporary directory is left: this run's, and the killed run's.
    assert!(temporaries(fx.new.parent().unwrap()).is_empty(), "{:?}", listing(fx.new.parent().unwrap()));
    assert!(out.report["temporary_copy_deleted"].as_str().unwrap().contains(".init-"));
    assert_eq!(out.report["stale_temporaries_deleted"], json!([stale.to_string_lossy()]));
}

#[test]
fn t09_every_failure_of_init_deletes_the_temporary_copy_and_says_so() {
    let fx = fx();
    make_axel(&fx.old);

    // A copy that is not a database past its first page: every page after it is
    // overwritten. (`f2_…` is the copy that only its integrity check refuses.)
    let broken = fx.root().join("broken.r8");
    let mut bytes = fs::read(&fx.old).unwrap();
    assert!(bytes.len() > 3 * 4096);
    for b in &mut bytes[4096..] {
        *b = 0xAB;
    }
    fs::write(&broken, &bytes).unwrap();
    let before = state(&broken);
    let out = fx.init(&broken);
    show("init, integrity check fails", &out);
    assert_eq!(out.code, 3, "{}", out.report);
    assert_eq!(out.report["ok"], false);
    assert!(out.report["temporary_copy_deleted"].as_str().unwrap().contains(".init-"), "{}", out.report);
    assert!(listing(fx.new.parent().unwrap()).is_empty(), "{:?}", listing(fx.new.parent().unwrap()));
    assert_eq!(state(&broken), before);

    // A database that is not an Axel brain is refused before anything is put at the destination.
    drop(synaps_conn(&fx.syn));
    let out = fx.init(&fx.syn);
    assert_eq!(out.code, 3, "{}", out.report);
    assert!(out.report["temporary_copy_deleted"].is_string(), "{}", out.report);
    assert!(listing(fx.new.parent().unwrap()).is_empty());

    // A source that is not there, and a --watch that is not there: refused, nothing made.
    let out = fx.init(&fx.root().join("missing.r8"));
    assert_eq!(out.code, 2, "{}", out.report);
    let mut opts = fx.init_opts(&fx.old);
    opts.watch = Some(fx.root().join("no-such-file"));
    let out = import::init(&fx.dest(), &opts);
    assert_eq!(out.code, 2, "{}", out.report);
    assert!(listing(fx.new.parent().unwrap()).is_empty());
}

#[test]
fn t09_watch_records_the_file_that_will_be_watched() {
    let fx = fx();
    make_axel(&fx.old);
    let watched = fx.root().join("watched.r8");
    fs::write(&watched, b"twelve bytes").unwrap();
    let mut opts = fx.init_opts(&fx.old);
    opts.watch = Some(watched.clone());
    let out = import::init(&fx.dest(), &opts);
    assert_eq!(out.code, 0, "{}", out.report);
    let conn = Connection::open(&fx.new).unwrap();
    let detail: Value = serde_json::from_str(&text(&conn, "SELECT detail FROM import_log WHERE item = 'init'")).unwrap();
    assert_eq!(detail["path"], json!(watched.to_string_lossy()));
    assert_eq!(detail["size"], 12);
    assert_eq!(detail["mtime_ns"], json!(fs::metadata(&watched).unwrap().mtime() * 1_000_000_000 + fs::metadata(&watched).unwrap().mtime_nsec()));
    assert_eq!(detail["wal_size"], Value::Null);
    assert_eq!(detail["wal_mtime_ns"], Value::Null);
    let mut keys: Vec<&str> = detail.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort();
    assert_eq!(keys, ["mtime_ns", "path", "size", "wal_mtime_ns", "wal_size"]);
    // The row's source is still the file that was read.
    assert_eq!(text(&conn, "SELECT source FROM import_log WHERE item = 'init'"), fx.old.to_string_lossy());
}

// ── 10 ──────────────────────────────────────────────────────────────────────

#[test]
fn t10_an_existing_destination_is_kept_and_replace_moves_it_aside() {
    let fx = fx();
    make_axel(&fx.old);
    // The empty brain the extension makes when it starts and finds none (ruling 3).
    empty_v2(&fx.new);
    settle(&fx.new);
    let folder = fx.new.parent().unwrap().to_path_buf();
    fs::write(folder.join("brain.r8.hnsw"), b"old index").unwrap();
    let old_bytes = sha(&fx.new);
    let names = listing(&folder);
    let source_before = state(&fx.old);

    let out = fx.init(&fx.old);
    show("init, destination exists", &out);
    assert_eq!(out.code, 2, "{}", out.report);
    let error = out.report["error"].as_str().unwrap();
    assert!(error.contains(&format!("a new brain already exists at {}", fx.new.display())), "{error}");
    assert!(error.contains("(version 2, 0 memories, 0 forum posts)"), "{error}");
    assert!(error.contains("it is kept. Pass --replace to move it aside and start again from axel.r8."), "{error}");
    assert_eq!(sha(&fx.new), old_bytes);
    assert_eq!(listing(&folder), names, "the refusal made a file");
    assert_eq!(state(&fx.old), source_before);

    let mut opts = fx.init_opts(&fx.old);
    opts.replace = true;
    let out = import::init(&fx.dest(), &opts);
    show("init --replace", &out);
    assert_eq!(out.code, 0, "{}", out.report);
    let aside: Vec<String> = listing(&folder).into_iter().filter(|n| n.starts_with("brain.r8.replaced-")).collect();
    let main: Vec<&String> = aside.iter().filter(|n| !n.ends_with(".hnsw") && !n.ends_with("-wal") && !n.ends_with("-shm")).collect();
    assert_eq!(main.len(), 1, "{aside:?}");
    // Moved, never deleted: the old bytes are there.
    assert_eq!(sha(&folder.join(main[0])), old_bytes);
    assert_eq!(out.report["replaced"]["moved_to"], json!(folder.join(main[0]).to_string_lossy()));
    assert_eq!(fs::read(folder.join(format!("{}.hnsw", main[0]))).unwrap(), b"old index");
    assert!(!folder.join("brain.r8.hnsw").exists());
    // And the destination is new.
    let conn = Connection::open(&fx.new).unwrap();
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM memories"), 3);
    assert!(temporaries(&folder).is_empty());
    assert_eq!(state(&fx.old), source_before);
}

// ── 11 ──────────────────────────────────────────────────────────────────────

#[test]
fn t11_a_destination_from_the_environment_or_equal_to_a_source_is_refused() {
    let fx = fx();
    make_axel(&fx.old);
    drop(synaps_conn(&fx.syn));
    let before = state(&fx.old);

    // Through the binary: `--brain` is absent and AXEL_BRAIN names the destination.
    let done = Command::new(env!("CARGO_BIN_EXE_axel"))
        .env("AXEL_BRAIN", &fx.new)
        .args(["migrate", "init", "--from-axel"])
        .arg(&fx.old)
        .arg("--synaps-base")
        .arg(&fx.base)
        .output()
        .unwrap();
    assert_eq!(done.status.code(), Some(2), "{}", String::from_utf8_lossy(&done.stderr));
    let report: Value = serde_json::from_slice(&done.stdout).unwrap();
    assert!(report["error"].as_str().unwrap().contains("AXEL_BRAIN"), "{report}");
    assert!(!fx.new.exists());
    assert!(listing(fx.new.parent().unwrap()).is_empty());

    // The other three refuse it too, before they look at anything.
    let env = BrainArg { path: fx.new.clone(), from_env: true, on_command_line: false };
    let synaps = SynapsOpts { from: fx.syn.clone(), ..SynapsOpts::default() };
    let verify = VerifyOpts { from_axel: fx.old.clone(), from_synaps: fx.syn.clone(), ..VerifyOpts::default() };
    let catch_up = CatchUpOpts { from_axel: fx.old.clone(), watch: None };
    for out in [import::synaps(&env, &synaps), import::verify(&env, &verify), import::catch_up(&env, &catch_up)] {
        assert_eq!(out.code, 2, "{}", out.report);
        assert!(out.report["error"].as_str().unwrap().contains("AXEL_BRAIN"), "{}", out.report);
    }

    // A destination that is the source, also when it is named another way.
    let link = fx.root().join("link-to-old");
    std::os::unix::fs::symlink(fx.old.parent().unwrap(), &link).unwrap();
    for same in [fx.old.clone(), link.join("axel.r8")] {
        let mut opts = fx.init_opts(&fx.old);
        opts.replace = true;
        let out = import::init(&arg(&same), &opts);
        assert_eq!(out.code, 2, "{}", out.report);
        assert!(out.report["error"].as_str().unwrap().contains("a source is never written"), "{}", out.report);
    }
    fs::remove_file(&link).unwrap();
    assert_eq!(state(&fx.old), before);
}

// ── 12 ──────────────────────────────────────────────────────────────────────

#[test]
fn t12_synaps_brings_the_forum_and_nothing_else() {
    let fx = fx();
    empty_v2(&fx.new);
    let now = now_ms();
    let conn = synaps_conn(&fx.syn);
    let posts = three_posts(&conn, now);
    // A post whose stored body is not the one its digest was made of.
    let bent_post = root_post("bent-1", "Bent", "the original body", 30);
    let bent = forum_row(&conn, X, &who("cy"), &bent_post, now - 3_000, "a body somebody changed");
    // A forum forget made in the old helper: the tombstone, and no row.
    let gone = Envelope::new(X, who("ada"), &root_post("gone-1", "Gone", "forgotten there", 30)).unwrap().id();
    tombstone_row(&conn, &gone, X);
    tombstone_row(&conn, "mem-note-forgotten", X);
    note_row(&conn, X, "mem-note-1", "a synthetic note about kelp");
    note_row(&conn, X, "mem-note-2", "another synthetic note");
    memory_row(&conn, "mem-foreign-1", None, "", "made with axel remember, not by Synaps");
    history_row(&conn, X, "h-1");
    history_row(&conn, X, "h-2");
    conn.close().unwrap();
    let before = state(&fx.syn);

    let out = fx.synaps(&[], false);
    show("synaps", &out);
    // Foreign rows without --accept-foreign: exit 1, and the forum is imported all the same.
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(state(&fx.syn), before, "the Synaps source changed, or a file appeared beside it");
    let new = Connection::open(&fx.new).unwrap();
    let mut expected = vec![posts.root.clone(), posts.reply.clone(), posts.expired.clone(), bent.clone()];
    expected.sort();
    assert_eq!(forum_ids(&new), expected);
    for id in [&posts.root, &posts.reply, &posts.expired] {
        let (ok, origin, key, origin_project): (i64, String, String, String) = new
            .query_row("SELECT digest_ok, origin, project_key, origin_project FROM forum_posts WHERE id = ?1", [id], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
            })
            .unwrap();
        assert_eq!((ok, origin.as_str(), key.as_str(), origin_project.as_str()), (1, "brain.r8", X, X));
    }
    let (ts, expires, raw): (i64, i64, String) = new
        .query_row("SELECT timestamp_ms, expires_ms, raw FROM forum_posts WHERE id = ?1", [&posts.expired], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!((ts, expires), (1_600_000_000_000, 1_600_000_000_000 + DAY_MS));
    let raw: Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(raw["id"], json!(posts.expired));
    assert_eq!(raw["content"], "Long gone");
    assert_eq!(count(&new, &format!("SELECT digest_ok FROM forum_posts WHERE id = '{bent}'")), 0);
    assert_eq!(text(&new, &format!("SELECT body FROM forum_posts WHERE id = '{bent}'")), "a body somebody changed");
    assert_eq!(text(&new, &format!("SELECT source FROM forum_tombstones WHERE id = '{gone}'")), "import");
    assert_eq!(count(&new, "SELECT COUNT(*) FROM forum_tombstones"), 1);
    // No note, archive or foreign row is written anywhere.
    for table in ["memories", "memory_ext", "memories_fts", "documents", "archive_index", "memory_tombstones", "note_files"] {
        assert_eq!(count(&new, &format!("SELECT COUNT(*) FROM {table}")), 0, "{table}");
    }
    drop(new);
    assert_eq!(out.report["forum_posts"], json!({ "rows_in": 4, "rows_imported": 4, "rows_present": 0, "rows_flagged": 1, "rows_skipped": 0 }));
    assert_eq!(ids(&out.report["rows_flagged"]), [bent.as_str()]);
    assert_eq!(out.report["forum_tombstones"]["rows_imported"], 1);
    assert_eq!(out.report["left_for_move_sh"], json!({ "notes": 2, "note_tombstones": 1, "archives": 2 }));
    assert_eq!(ids(&out.report["foreign_memories"]), ["mem-foreign-1"]);
    assert_eq!(out.report["forum_dates_clamped"]["count"], 0);

    // The forum reads what was imported: the live root, not the expired one, not the bent one;
    // and the same post made again is a duplicate.
    let conn = brain_v2::connect(&fx.new, Role::Cli).unwrap();
    let page = forum_store::read(&conn, X, &Read::default(), now_ms()).unwrap();
    assert_eq!(page.entries.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), [posts.root.as_str()]);
    brain_v2::begin_write(&conn).unwrap();
    let receipt = forum_store::post(&conn, X, &posts.author, &posts.root_post, now_ms()).unwrap();
    assert_eq!(receipt.status, PostStatus::Duplicate);
    conn.execute_batch("ROLLBACK").unwrap();
    drop(conn);

    // A second run imports nothing new, and with --accept-foreign it exits 0.
    let again = fx.synaps(&[], true);
    show("synaps, second run", &again);
    assert_eq!(again.code, 0, "{}", again.report);
    assert_eq!(again.report["forum_posts"], json!({ "rows_in": 4, "rows_imported": 0, "rows_present": 4, "rows_flagged": 0, "rows_skipped": 0 }));
    assert_eq!(again.report["forum_tombstones"], json!({ "rows_in": 1, "rows_imported": 0, "rows_present": 1 }));
    let new = Connection::open(&fx.new).unwrap();
    assert_eq!(forum_ids(&new), expected);
    assert_eq!(count(&new, "SELECT COUNT(*) FROM forum_tombstones"), 1);
    assert_eq!(count(&new, "SELECT COUNT(*) FROM import_log WHERE item = 'forum_posts'"), 2);
    drop(new);
    assert_eq!(state(&fx.syn), before);

    // A source with a non-empty log is refused, and nothing is imported from it.
    fs::write(side(&fx.syn, "-wal"), b"frames somebody did not checkpoint").unwrap();
    settle(&fx.new);
    let hash = sha(&fx.new);
    let refused = fx.synaps(&[], true);
    assert_eq!(refused.code, 3, "{}", refused.report);
    assert!(refused.report["error"].as_str().unwrap().contains("-wal"), "{}", refused.report);
    settle(&fx.new);
    assert_eq!(sha(&fx.new), hash);
}

// ── 13 ──────────────────────────────────────────────────────────────────────

#[test]
fn t13_map_homes_the_posts_and_a_second_map_moves_them() {
    let fx = fx();
    empty_v2(&fx.new);
    let conn = synaps_conn(&fx.syn);
    let posts = three_posts(&conn, now_ms());
    let gone = Envelope::new(X, who("ada"), &root_post("gone-1", "Gone", "forgotten there", 30)).unwrap().id();
    tombstone_row(&conn, &gone, X);
    conn.close().unwrap();

    let xy = format!("{X}={Y}");
    let out = fx.synaps(&[&xy], false);
    assert_eq!(out.code, 0, "{}", out.report);
    let new = Connection::open(&fx.new).unwrap();
    assert_eq!(count(&new, &format!("SELECT COUNT(*) FROM forum_posts WHERE project_key = '{Y}' AND origin_project = '{X}' AND digest_ok = 1")), 3);
    assert_eq!(text(&new, "SELECT project_key FROM forum_tombstones"), Y);
    drop(new);
    // Read where they now live.
    let conn = brain_v2::connect(&fx.new, Role::Cli).unwrap();
    let page = forum_store::read(&conn, Y, &Read::default(), now_ms()).unwrap();
    assert_eq!(page.entries.len(), 1);
    assert_eq!(page.entries[0].envelope.project, X);
    assert!(forum_store::read(&conn, X, &Read::default(), now_ms()).unwrap().entries.is_empty());
    drop(conn);

    let xz = format!("{X}={Z}");
    let out = fx.synaps(&[&xz], false);
    show("synaps --map, second map", &out);
    assert_eq!(out.code, 0, "{}", out.report);
    let mut moved = ids(&out.report["rows_rehomed"]);
    moved.sort();
    let mut all = vec![posts.root, posts.reply, posts.expired];
    all.sort();
    assert_eq!(moved, all);
    let new = Connection::open(&fx.new).unwrap();
    assert_eq!(count(&new, &format!("SELECT COUNT(*) FROM forum_posts WHERE project_key = '{Z}' AND origin_project = '{X}' AND digest_ok = 1")), 3);
    assert_eq!(count(&new, "SELECT COUNT(*) FROM forum_posts"), 3);
    drop(new);

    // A pair that is not two keys, and an old key named twice: refused, nothing read.
    for bad in [vec!["p1111111111111111=home".to_string()], vec![xy.clone(), xz.clone()], vec!["nokeys".to_string()]] {
        let out = import::synaps(&fx.dest(), &SynapsOpts { from: fx.syn.clone(), map: bad.clone(), accept_foreign: false });
        assert_eq!(out.code, 2, "{bad:?}: {}", out.report);
    }
    let new = Connection::open(&fx.new).unwrap();
    assert_eq!(count(&new, &format!("SELECT COUNT(*) FROM forum_posts WHERE project_key = '{Z}'")), 3);
}

// ── 14 ──────────────────────────────────────────────────────────────────────

#[test]
fn t14_a_post_forgotten_in_the_new_brain_stays_forgotten() {
    let fx = fx();
    empty_v2(&fx.new);
    let conn = synaps_conn(&fx.syn);
    let posts = three_posts(&conn, now_ms());
    conn.close().unwrap();
    assert_eq!(fx.synaps(&[], false).code, 0);

    forget_in_new_brain(&fx.new, X, &posts.reply);
    for run in 0..3 {
        let out = fx.synaps(&[], false);
        if run == 0 {
            show("synaps after a forget", &out);
        }
        assert_eq!(out.code, 0, "{}", out.report);
        assert_eq!(ids(&out.report["rows_skipped"]), [posts.reply.as_str()], "run {run}");
        assert_eq!(out.report["forum_posts"]["rows_imported"], 0);
        let new = Connection::open(&fx.new).unwrap();
        assert_eq!(count(&new, &format!("SELECT COUNT(*) FROM forum_posts WHERE id = '{}'", posts.reply)), 0, "run {run}: the forgotten post came back");
        assert_eq!(count(&new, "SELECT COUNT(*) FROM forum_posts"), 2);
        assert_eq!(text(&new, &format!("SELECT source FROM forum_tombstones WHERE id = '{}'", posts.reply)), "tool");
    }
}

#[test]
fn t14_a_post_forgotten_in_the_source_does_not_arrive_and_does_not_stay() {
    let fx = fx();
    empty_v2(&fx.new);
    let conn = synaps_conn(&fx.syn);
    let posts = three_posts(&conn, now_ms());
    // The source holds the row AND its tombstone: the tombstone wins.
    tombstone_row(&conn, &posts.reply, X);
    conn.close().unwrap();
    let out = fx.synaps(&[], false);
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(ids(&out.report["rows_skipped"]), [posts.reply.as_str()]);
    let new = Connection::open(&fx.new).unwrap();
    assert_eq!(count(&new, &format!("SELECT COUNT(*) FROM forum_posts WHERE id = '{}'", posts.reply)), 0);
    assert_eq!(count(&new, "SELECT COUNT(*) FROM forum_posts"), 2);
    drop(new);

    // Later the source forgets a post an earlier run imported: the next run takes it out.
    let conn = synaps_conn(&fx.syn);
    conn.execute("DELETE FROM memories WHERE id = ?1", [&posts.root]).unwrap();
    tombstone_row(&conn, &posts.root, X);
    conn.close().unwrap();
    for run in 0..2 {
        let out = fx.synaps(&[], false);
        assert_eq!(out.code, 0, "{}", out.report);
        if run == 0 {
            assert_eq!(ids(&out.report["posts_removed_by_source_tombstone"]), [posts.root.as_str()]);
        }
        let new = Connection::open(&fx.new).unwrap();
        assert_eq!(forum_ids(&new), [posts.expired.as_str()], "run {run}");
        assert_eq!(count(&new, "SELECT COUNT(*) FROM forum_tombstones"), 2);
        assert_eq!(count(&new, "SELECT COUNT(*) FROM context_data WHERE layer = 'axel' AND key = 'purge_pending'"), 1);
    }
}

// ── Ruling 1: a forum date later than now ───────────────────────────────────

#[test]
fn r1_a_forum_date_later_than_now_is_stored_as_now_and_reported() {
    let fx = fx();
    empty_v2(&fx.new);
    let year_2100 = 4_102_444_800_000i64;
    let conn = synaps_conn(&fx.syn);
    let author = who("ada");
    let post = root_post("future-1", "From the future", "dated in 2100", 30);
    let id = forum_row(&conn, X, &author, &post, year_2100, &post.body);
    conn.close().unwrap();

    let before = now_ms();
    let out = fx.synaps(&[], false);
    show("synaps, a date in 2100", &out);
    let after = now_ms();
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(out.report["forum_dates_clamped"]["count"], 1);
    assert_eq!(ids(&out.report["forum_dates_clamped"]), [id.as_str()]);
    let new = Connection::open(&fx.new).unwrap();
    let (ts, expires, ok, raw): (i64, i64, i64, String) = new
        .query_row("SELECT timestamp_ms, expires_ms, digest_ok, raw FROM forum_posts WHERE id = ?1", [&id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))
        })
        .unwrap();
    assert!((before..=after).contains(&ts), "{ts}");
    assert_eq!(expires, ts + 30 * DAY_MS);
    assert_eq!(ok, 1);
    // The original date is kept in the source row.
    assert!(raw.contains("4102444800000"), "{raw}");
    drop(new);

    // The next post of the project is dated now, not after 2100; and the clamped post reads.
    let conn = brain_v2::connect(&fx.new, Role::Cli).unwrap();
    brain_v2::begin_write(&conn).unwrap();
    let next = root_post("next-1", "Next", "posted after the import", 30);
    let receipt = forum_store::post(&conn, X, &who("bo"), &next, now_ms()).unwrap();
    conn.execute_batch("COMMIT").unwrap();
    assert_eq!(receipt.status, PostStatus::Created);
    assert!((receipt.timestamp_ms.unwrap() as i64) < now_ms() + 60_000, "{:?}", receipt.timestamp_ms);
    let page = forum_store::read(&conn, X, &Read::default(), now_ms()).unwrap();
    assert_eq!(page.entries.len(), 2);
    assert_eq!(page.entries[0].id, id);

    // A second run leaves the stored date alone.
    drop(conn);
    let again = fx.synaps(&[], false);
    assert_eq!(again.code, 0);
    assert_eq!(again.report["forum_dates_clamped"]["count"], 0);
    let new = Connection::open(&fx.new).unwrap();
    assert_eq!(count(&new, &format!("SELECT timestamp_ms FROM forum_posts WHERE id = '{id}'")), ts);
}

// ── Ruling 2: rows that are not text ────────────────────────────────────────

#[test]
fn r2_init_counts_memories_whose_id_content_or_title_is_not_text() {
    let fx = fx();
    make_axel(&fx.old);
    change_axel(&fx.old, |conn| {
        // A title that is a blob: adoption reads the row, the rehearsal's count names it.
        conn.execute(
            "INSERT INTO memories(id, category, topic, title, content, created) VALUES ('mem-blob-title', 'events', 'sea', x'00ff', 'readable text', '2026-01-02T03:04:05+00:00')",
            [],
        )
        .unwrap();
        // A content that is a blob: adoption skips it.
        conn.execute(
            "INSERT INTO memories(id, category, topic, title, content, created) VALUES ('mem-blob-content', 'events', 'sea', 'a title', x'00ff', '2026-01-02T03:04:05+00:00')",
            [],
        )
        .unwrap();
    });
    let out = fx.init(&fx.old);
    show("init, rows that are not text", &out);
    assert_eq!(out.code, 0, "{}", out.report);
    // Adoption's count, the number `status` shows under this name (finding 6)…
    assert_eq!(out.report["unreadable_memories"], 1);
    // …and the rows whose id, content or title is not text.
    assert_eq!(out.report["memories_not_text"], 2);
    assert!(out.report.get("adoption_skipped_memories").is_none());
    let conn = Connection::open(&fx.new).unwrap();
    // Both rows are in the new brain as they were.
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM memories"), 5);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM memories WHERE typeof(title) = 'blob' OR typeof(content) = 'blob'"), 2);
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM memory_ext"), 4);
}

// ── 15 ──────────────────────────────────────────────────────────────────────

/// `init` and `synaps` with `X=Y`, on the fixtures of tests 9 and 12 without a foreign row.
fn migrated(fx: &Fx) -> (Posts, String) {
    make_axel(&fx.old);
    let conn = synaps_conn(&fx.syn);
    let posts = three_posts(&conn, now_ms());
    note_row(&conn, X, "mem-note-1", "a synthetic note about kelp");
    history_row(&conn, X, "h-1");
    conn.close().unwrap();
    let map = format!("{X}={Y}");
    let out = fx.init(&fx.old);
    assert_eq!(out.code, 0, "{}", out.report);
    let out = fx.synaps(&[&map], false);
    assert_eq!(out.code, 0, "{}", out.report);
    (posts, map)
}

#[test]
fn t15_verify_passes_after_the_import_and_fails_on_each_loss() {
    let fx = fx();
    let (posts, map) = migrated(&fx);
    let (old_before, syn_before) = (state(&fx.old), state(&fx.syn));

    // It passes, and writes nothing to the brain.
    settle(&fx.new);
    let hash = sha(&fx.new);
    let out = fx.verify(&[&map]);
    show("verify", &out);
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(out.report["shortfalls"], json!([]));
    assert_eq!(item(&out.report, "memories")["same"], 3);
    assert_eq!(item(&out.report, "forum_posts")["same"], 3);
    assert_eq!(item(&out.report, "notes_and_archives")["notes"], 1);
    assert_eq!(sha(&fx.new), hash, "verify wrote to the brain");
    settle(&fx.new);
    assert_eq!(sha(&fx.new), hash, "verify left something in the brain's log");
    assert_eq!((state(&fx.old), state(&fx.syn)), (old_before, syn_before));

    let run = |brain: &Path, map: &str| fx.verify_at(brain, &fx.old, &[map]);

    // One imported forum row deleted without a tombstone.
    let broken = clone_brain(&fx, &fx.new, "case-forum");
    Connection::open(&broken).unwrap().execute("DELETE FROM forum_posts WHERE id = ?1", [&posts.reply]).unwrap();
    let out = run(&broken, &map);
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(shortfalls(&out.report), ["forum_posts"]);
    assert_eq!(ids(&item(&out.report, "forum_posts")["missing"]), [posts.reply.as_str()]);

    // The text of one memory differs by one byte.
    let broken = clone_brain(&fx, &fx.new, "case-text");
    Connection::open(&broken).unwrap().execute("UPDATE memories SET content = content || '.' WHERE id = ?1", [M2]).unwrap();
    let out = run(&broken, &map);
    show("verify, one byte of one memory differs", &out);
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(shortfalls(&out.report), ["memories"]);
    assert_eq!(ids(&item(&out.report, "memories")["changed"]), [M2]);

    // One saved row is missing.
    let broken = clone_brain(&fx, &fx.new, "case-missing");
    Connection::open(&broken)
        .unwrap()
        .execute_batch(&format!(
            "BEGIN; INSERT INTO axel_purge(one) VALUES (1); DELETE FROM memories WHERE id = '{M3}';
             DELETE FROM memories_fts WHERE mem_id = '{M3}'; DELETE FROM axel_purge; COMMIT;"
        ))
        .unwrap();
    let out = run(&broken, &map);
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(shortfalls(&out.report), ["memories"]);
    assert_eq!(ids(&item(&out.report, "memories")["missing"]), [M3]);

    // One word row deleted.
    let broken = clone_brain(&fx, &fx.new, "case-fts");
    Connection::open(&broken).unwrap().execute("DELETE FROM memories_fts WHERE mem_id = ?1", [M1]).unwrap();
    let out = run(&broken, &map);
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(shortfalls(&out.report), ["memories_fts"]);

    // One memory_ext row deleted.
    let broken = clone_brain(&fx, &fx.new, "case-ext");
    Connection::open(&broken).unwrap().execute("DELETE FROM memory_ext WHERE mem_id = ?1", [M1]).unwrap();
    let out = run(&broken, &map);
    assert_eq!(shortfalls(&out.report), ["memory_ext"]);

    // A staged row and a document deleted.
    let broken = clone_brain(&fx, &fx.new, "case-saved");
    Connection::open(&broken)
        .unwrap()
        .execute_batch("DELETE FROM staged_memories WHERE id = 'stg-1'; DELETE FROM documents WHERE doc_id = 'note-1';")
        .unwrap();
    let out = run(&broken, &map);
    assert_eq!(out.code, 1);
    assert_eq!(shortfalls(&out.report), ["staged_memories", "documents"]);

    // Another --map than the import used.
    let out = fx.verify(&[&format!("{X}={Z}")]);
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(shortfalls(&out.report), ["forum_posts"]);
    assert_eq!(item(&out.report, "forum_posts")["wrong_project"]["count"], 3);

    // A forgotten post is not a shortfall.
    forget_in_new_brain(&fx.new, Y, &posts.reply);
    let out = fx.verify(&[&map]);
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(item(&out.report, "forum_posts")["tombstoned"], 1);

    // A foreign row in brain.r8 fails it, unless accepted.
    let conn = synaps_conn(&fx.syn);
    memory_row(&conn, "mem-foreign-1", None, "", "not a Synaps record");
    conn.close().unwrap();
    let out = fx.verify(&[&map]);
    assert_eq!(shortfalls(&out.report), ["foreign_memories"]);
    let mut opts = VerifyOpts { from_axel: fx.old.clone(), from_synaps: fx.syn.clone(), map: vec![map.clone()], accept_foreign: true };
    assert_eq!(import::verify(&fx.dest(), &opts).code, 0);
    // A --map that is not a pair of keys is refused here too.
    opts.map = vec!["bad".into()];
    assert_eq!(import::verify(&fx.dest(), &opts).code, 2);
}

#[test]
fn t15_verify_holds_usage_tables_to_the_count_init_recorded() {
    let fx = fx();
    let (_, map) = migrated(&fx);
    // The old plugin goes on writing usage rows after the import.
    change_axel(&fx.old, |conn| add_access(conn, "note-1"));
    let out = fx.verify(&[&map]);
    show("verify, the source gained a usage row", &out);
    assert_eq!(out.code, 0, "{}", out.report);
    let access = item(&out.report, "document_access");
    assert_eq!((access["recorded"].as_i64(), access["source"].as_i64(), access["new_brain"].as_i64()), (Some(2), Some(3), Some(2)));
    assert_eq!(access["source_gained"], 1);
    assert!(access["line"].as_str().unwrap().contains(
        "source has 1 more rows since the import (searches, sessions and graph links written by the old plugin); not carried"
    ));

    // One usage row lost from the new brain fails it.
    let broken = clone_brain(&fx, &fx.new, "case-usage");
    Connection::open(&broken).unwrap().execute("DELETE FROM document_access WHERE id = 1", []).unwrap();
    let out = fx.verify_at(&broken, &fx.old, &[&map]);
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(shortfalls(&out.report), ["document_access"]);

    // With no init record the source's own count is the bar.
    let bare = clone_brain(&fx, &fx.new, "case-no-record");
    Connection::open(&bare).unwrap().execute("DELETE FROM import_log WHERE item LIKE 'count:%'", []).unwrap();
    let out = fx.verify_at(&bare, &fx.old, &[&map]);
    assert_eq!(shortfalls(&out.report), ["document_access"]);
}

#[test]
fn t15_verify_reads_a_source_with_a_log_from_a_copy_it_deletes() {
    let fx = fx();
    let (_, map) = migrated(&fx);
    // The copy has one memory more, in its log only: verify must see it.
    let source = snapshot_with_log(&fx.old, &fx.root().join("snap"));
    let before = state(&source);
    let out = fx.verify_at(&fx.new, &source, &[&map]);
    show("verify, a source with a log", &out);
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(out.report["source_read_from"], "a checked temporary copy");
    assert_eq!(ids(&item(&out.report, "memories")["missing"]), ["mem-dddd0004"]);
    assert!(out.report["temporary_copy_deleted"].as_str().unwrap().contains(".verify-"), "{}", out.report);
    assert!(temporaries(fx.new.parent().unwrap()).is_empty(), "{:?}", listing(fx.new.parent().unwrap()));
    assert_eq!(state(&source), before);
}

// ── 16 ──────────────────────────────────────────────────────────────────────

#[test]
fn t16_catch_up_and_verify_agree() {
    let fx = fx();
    let (_, map) = migrated(&fx);
    let new_memory = "mem-eeee0005";
    change_axel(&fx.old, |conn| {
        add_memory(conn, new_memory, "quay note", "the gull mended the quay rope", 0.125);
        add_document(conn, "note-2", "a second note file about barnacles", "/notes/barnacles.md", 0.375);
        conn.execute("UPDATE context_data SET value = 'second handoff', updated = '2031-05-06 07:08:09' WHERE key = 'handoff'", [])
            .unwrap();
        conn.execute(
            "INSERT INTO staged_memories(id, category, topic, title, content, created, staged_at)
             VALUES ('stg-2', 'events', 'sea', 'staged two', 'another staged thought', '2026-03-01T00:00:00+00:00', '2026-03-01T00:00:00+00:00')",
            [],
        )
        .unwrap();
        for _ in 0..3 {
            add_access(conn, "note-1");
        }
    });
    let source_before = state(&fx.old);

    // Without a catch-up verify names what was saved since, and not the access rows.
    let out = fx.verify(&[&map]);
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(shortfalls(&out.report), ["memories", "staged_memories", "handoff", "documents"]);
    assert_eq!(ids(&item(&out.report, "memories")["missing"]), [new_memory]);
    assert_eq!(ids(&item(&out.report, "staged_memories")["missing"]), ["stg-2"]);
    let mut docs = ids(&item(&out.report, "documents")["missing"]);
    docs.sort();
    assert_eq!(docs, [new_memory, "note-2"]);
    assert_eq!(item(&out.report, "document_access")["ok"], true);

    let out = fx.catch_up();
    show("catch-up", &out);
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(out.report["memories"]["brought"], 1);
    assert_eq!(ids(&out.report["memories"]["ids"]), [new_memory]);
    assert_eq!(out.report["memory_shadow_documents"], 1);
    assert_eq!(out.report["documents"], 1);
    assert_eq!(out.report["not_carried"], json!({ "document_access": 3 }));
    assert_eq!(out.report["handoff"], "taken: the source's is later");
    assert_eq!(out.report["saved_rows"]["staged_memories"], 1);
    assert_eq!(state(&fx.old), source_before, "catch-up changed the source, or made a file beside it");

    let verified = fx.verify(&[&map]);
    assert_eq!(verified.code, 0, "{}", verified.report);
    {
        let new = Connection::open(&fx.new).unwrap();
        // The memory: its row, its shadow document with the vector, a word, origin legacy.
        assert_eq!(text(&new, &format!("SELECT content FROM memories WHERE id = '{new_memory}'")), "the gull mended the quay rope");
        assert_eq!(text(&new, &format!("SELECT signature FROM memories WHERE id = '{new_memory}'")), "sig");
        let vec: Vec<u8> = new.query_row("SELECT embedding FROM documents WHERE doc_id = ?1", [new_memory], |r| r.get(0)).unwrap();
        assert_eq!(vec, vector(0.125));
        assert_eq!(word_hits(&new, "gull"), [new_memory]);
        let (origin, model, vec): (String, String, Vec<u8>) = new
            .query_row("SELECT origin_kind, embed_model, embedding FROM memory_ext WHERE mem_id = ?1", [new_memory], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
        assert_eq!((origin.as_str(), model.as_str()), ("legacy", "hash-384"));
        assert_eq!(vec, vector(0.125));
        // The document: its row, its vector, its word row.
        let vec: Vec<u8> = new.query_row("SELECT embedding FROM documents WHERE doc_id = 'note-2'", [], |r| r.get(0)).unwrap();
        assert_eq!(vec, vector(0.375));
        assert_eq!(count(&new, "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'barnacles'"), 1);
        assert_eq!(count(&new, "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'gull'"), 1);
        assert_eq!(text(&new, "SELECT value FROM context_data WHERE key = 'handoff'"), "second handoff");
        // The usage rows were not copied.
        assert_eq!(count(&new, "SELECT COUNT(*) FROM document_access"), 2);
        let detail: Value = serde_json::from_str(&text(&new, "SELECT detail FROM import_log WHERE item = 'catch-up'")).unwrap();
        assert_eq!(detail["path"], json!(fx.old.to_string_lossy()));
        assert_eq!(detail["size"], json!(fs::metadata(&fx.old).unwrap().len()));
    }

    // A second catch-up copies nothing.
    let again = fx.catch_up();
    assert_eq!(again.code, 0, "{}", again.report);
    assert_eq!((again.report["memories"]["brought"].as_i64(), again.report["documents"].as_i64(), again.report["memory_shadow_documents"].as_i64()), (Some(0), Some(0), Some(0)));
    assert_eq!(again.report["saved_rows"], json!({ "context_data": 0, "staged_memories": 0, "patterns": 0 }));
    assert_eq!(again.report["handoff"], "the same");

    // A memory purged in the new brain does not come back and does not fail verify.
    purge_in_new_brain(&fx.new, M1);
    let out = fx.catch_up();
    show("catch-up, a memory forgotten in the new brain", &out);
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(out.report["memories"]["brought"], 0);
    assert_eq!(ids(&out.report["tombstoned_not_brought_back"]), [M1]);
    {
        let new = Connection::open(&fx.new).unwrap();
        assert_eq!(count(&new, &format!("SELECT COUNT(*) FROM memories WHERE id = '{M1}'")), 0, "the forgotten memory came back");
        assert_eq!(count(&new, &format!("SELECT COUNT(*) FROM documents WHERE doc_id = '{M1}'")), 0, "its shadow document came back");
        assert_eq!(count(&new, "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'otter'"), 0);
        assert!(word_hits(&new, "otter").is_empty());
    }
    let verified = fx.verify(&[&map]);
    assert_eq!(verified.code, 0, "{}", verified.report);
    assert_eq!(item(&verified.report, "memories")["tombstoned"], 1);

    // A memory whose text was changed in the source is listed, not overwritten, and verify fails on it.
    change_axel(&fx.old, |conn| {
        conn.execute("UPDATE memories SET content = 'a narwhal carried the lantern away' WHERE id = ?1", [M2]).unwrap();
    });
    let out = fx.catch_up();
    show("catch-up, a text changed in the source", &out);
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(ids(&out.report["changed_in_source"]), [M2]);
    let new = Connection::open(&fx.new).unwrap();
    assert_eq!(text(&new, &format!("SELECT content FROM memories WHERE id = '{M2}'")), "a narwhal carried the lantern home");
    drop(new);
    let verified = fx.verify(&[&map]);
    assert_eq!(verified.code, 1, "{}", verified.report);
    assert_eq!(shortfalls(&verified.report), ["memories"]);
    assert_eq!(ids(&item(&verified.report, "memories")["changed"]), [M2]);

    // `status` takes the newest row whose detail holds a `path` as the record of the import:
    // only `init` and `catch-up` write one, and the newest is the last catch-up's.
    let new = Connection::open(&fx.new).unwrap();
    let with_path = "SELECT item FROM import_log WHERE json_valid(detail) AND json_type(detail, '$.path') = 'text'";
    let mut stmt = new.prepare(&format!("{with_path} ORDER BY id")).unwrap();
    let items: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().collect::<rusqlite::Result<_>>().unwrap();
    assert_eq!(items, ["init", "catch-up", "catch-up", "catch-up", "catch-up"]);
    assert!(count(&new, "SELECT COUNT(*) FROM import_log WHERE item = 'forum_posts'") > 0);
}

#[test]
fn t16_catch_up_changes_nothing_when_a_table_has_other_columns() {
    let fx = fx();
    migrated(&fx);
    change_axel(&fx.old, |conn| {
        conn.execute_batch("ALTER TABLE patterns ADD COLUMN weight REAL;").unwrap();
        add_memory(conn, "mem-eeee0005", "quay note", "the gull mended the quay rope", 0.125);
    });
    settle(&fx.new);
    let hash = sha(&fx.new);
    let out = fx.catch_up();
    show("catch-up, columns differ", &out);
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(out.report["columns_differ"][0]["table"], "patterns");
    settle(&fx.new);
    assert_eq!(sha(&fx.new), hash, "catch-up changed the brain");
}

// ── 21 ──────────────────────────────────────────────────────────────────────

#[test]
fn t21_init_records_the_count_of_every_source_table_and_prints_the_graph_nodes() {
    let fx = fx();
    make_axel(&fx.old);
    change_axel(&fx.old, |conn| {
        conn.execute_batch(&format!(
            "INSERT INTO nodes(id, type, title, content, created_at) VALUES ('{M2}', 'memory', 'lantern', 'a narwhal', '2026-01-01');
             INSERT INTO nodes(id, type, title, content, created_at) VALUES ('n-other', 'note', 'tides', 'tides', '2026-01-01');
             INSERT INTO edges(id, source_id, target_id, type, weight, confidence, created_at)
                  VALUES ('e-1', '{M2}', 'n-other', 'mentions', 1.0, 1.0, '2026-01-01');
             INSERT INTO edges(id, source_id, target_id, type, weight, confidence, created_at)
                  VALUES ('e-2', 'n-other', '{M2}', 'mentions', 1.0, 1.0, '2026-01-01');"
        ))
        .unwrap();
    });
    // The tables of the source and their counts, read here and not by the code under test.
    let mut expected: BTreeMap<String, i64> = BTreeMap::new();
    {
        let old = Connection::open(&fx.old).unwrap();
        let mut stmt = old
            .prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'chunks_fts%'")
            .unwrap();
        let names: Vec<String> = stmt.query_map([], |r| r.get(0)).unwrap().collect::<rusqlite::Result<_>>().unwrap();
        for name in names {
            expected.insert(name.clone(), count(&old, &format!("SELECT COUNT(*) FROM {name}")));
        }
    }
    assert_eq!(expected.len(), 20, "{expected:?}");
    assert_eq!(expected["document_access"], 2);

    let out = fx.init(&fx.old);
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(out.report["memory_graph_nodes"], 1);
    let new = Connection::open(&fx.new).unwrap();
    let mut recorded: BTreeMap<String, i64> = BTreeMap::new();
    {
        let mut stmt = new.prepare("SELECT item, rows_in FROM import_log WHERE item LIKE 'count:%'").unwrap();
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))).unwrap();
        for row in rows {
            let (item, n) = row.unwrap();
            assert!(recorded.insert(item["count:".len()..].to_string(), n).is_none(), "{item} twice");
        }
    }
    assert_eq!(recorded, expected);
    assert_eq!(count(&new, "SELECT COUNT(*) FROM import_log WHERE item = 'init'"), 1);
    drop(new);

    // Forgetting that memory takes its node and its edges.
    purge_in_new_brain(&fx.new, M2);
    let new = Connection::open(&fx.new).unwrap();
    assert_eq!(text(&new, "SELECT group_concat(id) FROM nodes"), "n-other");
    assert_eq!(count(&new, "SELECT COUNT(*) FROM edges"), 0);
}

// ── 22 ──────────────────────────────────────────────────────────────────────

fn with_status_row(path: &Path, age_ms: i64) {
    empty_v2(path);
    let conn = Connection::open(path).unwrap();
    conn.execute(
        "INSERT INTO context_data(key, layer, value, updated) VALUES ('status', 'axel', ?1, datetime('now'))",
        [json!({ "pid": 1, "mode": "ready", "written_ms": now_ms() - age_ms }).to_string()],
    )
    .unwrap();
    conn.close().unwrap();
}

#[test]
fn t22_replace_refuses_while_an_extension_is_running_on_the_destination() {
    let fx = fx();
    make_axel(&fx.old);
    with_status_row(&fx.new, 30_000);
    settle(&fx.new);
    let hash = sha(&fx.new);
    let source_before = state(&fx.old);
    let folder = fx.new.parent().unwrap().to_path_buf();

    let mut opts = fx.init_opts(&fx.old);
    opts.replace = true;
    let out = import::init(&fx.dest(), &opts);
    show("init --replace, an extension is running", &out);
    assert_eq!(out.code, 4, "{}", out.report);
    let error = out.report["error"].as_str().unwrap();
    assert!(error.starts_with("an Axel extension is running on this brain (status written 3"), "{error}");
    assert!(error.ends_with("s ago); stop Synaps first, or pass --even-if-running"), "{error}");
    // Neither file changed, nothing was moved, and no copy was made: the refusal comes first.
    assert_eq!(sha(&fx.new), hash);
    assert_eq!(state(&fx.old), source_before);
    assert!(!listing(&folder).iter().any(|n| n.contains(".replaced-") || n.starts_with(".init-")), "{:?}", listing(&folder));
    assert!(out.report.get("temporary_copy_deleted").is_none(), "{}", out.report);
    assert!(out.report.get("source").is_none(), "the source was read before the refusal: {}", out.report);

    opts.even_if_running = true;
    let out = import::init(&fx.dest(), &opts);
    assert_eq!(out.code, 0, "{}", out.report);
    let aside = |n: &&String| n.contains(".replaced-") && !n.ends_with("-wal") && !n.ends_with("-shm");
    assert_eq!(listing(&folder).iter().filter(aside).count(), 1);
    assert_eq!(count(&Connection::open(&fx.new).unwrap(), "SELECT COUNT(*) FROM memories"), 3);

    // A status row ten minutes old is nobody running.
    let other = self::fx();
    make_axel(&other.old);
    with_status_row(&other.new, 600_000);
    let mut opts = other.init_opts(&other.old);
    opts.replace = true;
    let out = import::init(&other.dest(), &opts);
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(count(&Connection::open(&other.new).unwrap(), "SELECT COUNT(*) FROM memories"), 3);
}

// ── The command lines ───────────────────────────────────────────────────────

/// The four commands through the binary, as `cutover.sh` runs them: one JSON report on
/// standard output each, and the exit code it names.
#[test]
fn the_four_commands_through_the_binary() {
    let fx = fx();
    make_axel(&fx.old);
    let conn = synaps_conn(&fx.syn);
    three_posts(&conn, now_ms());
    conn.close().unwrap();
    let map = format!("{X}={Y}");
    let run = |args: &[&str]| -> (i32, Value) {
        let done = Command::new(env!("CARGO_BIN_EXE_axel"))
            // Neither variable may decide anything (ruling 4).
            .env("AXEL_BRAIN", &fx.old)
            .env("SYNAPS_BASE_DIR", fx.root().join("not-the-base"))
            .arg("--brain")
            .arg(&fx.new)
            .arg("migrate")
            .args(args)
            .output()
            .unwrap();
        let report: Value = serde_json::from_slice(&done.stdout)
            .unwrap_or_else(|e| panic!("{args:?}: standard output is not one JSON value ({e}): {}", String::from_utf8_lossy(&done.stdout)));
        let code = done.status.code().unwrap();
        assert_eq!(report["exit"], code, "{report}");
        assert_eq!(report["ok"], code == 0, "{report}");
        println!("REPORT (binary) migrate {}: exit {code}\n{}", args[0], String::from_utf8_lossy(&done.stdout));
        (code, report)
    };
    let (old, syn, base) = (fx.old.to_str().unwrap(), fx.syn.to_str().unwrap(), fx.base.to_str().unwrap());

    let (code, report) = run(&["init", "--from-axel", old, "--synaps-base", base, "--watch", old]);
    assert_eq!(code, 0, "{report}");
    assert_eq!(report["command"], "migrate init");
    assert_eq!(report["base_recorded"], base);
    assert_eq!(report["base_from"], "--synaps-base");
    let (code, report) = run(&["init", "--from-axel", old, "--synaps-base", base]);
    assert_eq!(code, 2, "{report}");
    let (code, report) = run(&["synaps", "--from", syn, "--map", &map, "--accept-foreign"]);
    assert_eq!(code, 0, "{report}");
    assert_eq!(report["forum_posts"]["rows_imported"], 3);
    let (code, report) = run(&["verify", "--from-axel", old, "--from-synaps", syn, "--map", &map, "--accept-foreign"]);
    assert_eq!(code, 0, "{report}");
    let (code, report) = run(&["catch-up", "--from-axel", old, "--watch", old]);
    assert_eq!(code, 0, "{report}");
    assert_eq!(report["memories"]["brought"], 0);
    let (code, report) = run(&["init", "--from-axel", old, "--synaps-base", base, "--replace", "--even-if-running"]);
    assert_eq!(code, 0, "{report}");

    // --even-if-running without --replace is not a command line.
    let done = Command::new(env!("CARGO_BIN_EXE_axel"))
        .env_remove("AXEL_BRAIN")
        .arg("--brain")
        .arg(&fx.new)
        .args(["migrate", "init", "--from-axel", old, "--even-if-running"])
        .output()
        .unwrap();
    assert_eq!(done.status.code(), Some(2));
    assert!(done.stdout.is_empty());
}

// ── The fix round (wave5-c19-review) ────────────────────────────────────────

/// Runs `body` with a seam on this thread and takes it away again.
fn with_seam<T>(seam: impl Fn(&str, &Path) -> bool + 'static, body: impl FnOnce() -> T) -> T {
    import::set_test_seam(Some(Box::new(seam)));
    let out = body();
    import::set_test_seam(None);
    out
}

/// Finding 1: the new brain is finished at its temporary name, and the link to the
/// destination is the last step. A stop after any step before it leaves nothing at the
/// destination, deletes the temporary copy, and the report says both.
#[test]
fn f1_a_stop_before_the_link_leaves_nothing_at_the_destination() {
    let fx = fx();
    make_axel(&fx.old);
    let folder = fx.new.parent().unwrap().to_path_buf();
    for step in ["checked", "open_v2", "7b_8", "vacuum", "checkpoint", "integrity_check", "closed", "synced"] {
        let out = with_seam(move |point, _| point == step, || fx.init(&fx.old));
        show(&format!("init, stopped at {step}"), &out);
        assert_eq!(out.code, 5, "{step}: {}", out.report);
        let error = out.report["error"].as_str().unwrap();
        assert!(error.starts_with(&format!("stopped at {step}")), "{error}");
        assert!(error.ends_with(&format!("Nothing was put at {}.", fx.new.display())), "{error}");
        assert_eq!(out.report["destination_made"], false, "{step}");
        assert!(out.report["temporary_copy_deleted"].as_str().unwrap().contains(".init-"), "{step}: {}", out.report);
        assert!(listing(&folder).is_empty(), "{step}: {:?}", listing(&folder));
    }

    // A file left beside the closed new brain stops it too.
    let out = with_seam(
        |point, db| {
            if point == "integrity_check" {
                fs::write(db.with_extension("r8.hnsw"), b"stray").unwrap();
            }
            false
        },
        || fx.init(&fx.old),
    );
    assert_eq!(out.code, 5, "{}", out.report);
    assert!(out.report["error"].as_str().unwrap().contains("files are still beside it: axel.r8.hnsw"), "{}", out.report);
    assert!(listing(&folder).is_empty(), "{:?}", listing(&folder));

    // With --replace, a stop after the old brain was moved aside: nothing at the
    // destination, and the report says where the old brain is.
    empty_v2(&fx.new);
    settle(&fx.new);
    let old_bytes = sha(&fx.new);
    let mut opts = fx.init_opts(&fx.old);
    opts.replace = true;
    let out = with_seam(|point, _| point == "replaced", || import::init(&fx.dest(), &opts));
    show("init --replace, stopped after the move", &out);
    assert_eq!(out.code, 5, "{}", out.report);
    assert!(!fx.new.exists());
    let moved_to = PathBuf::from(out.report["replaced"]["moved_to"].as_str().unwrap());
    assert_eq!(sha(&moved_to), old_bytes);
    assert!(out.report["error"].as_str().unwrap().contains(&format!("The brain that was there before is at {}.", moved_to.display())));
    assert!(temporaries(&folder).is_empty(), "{:?}", listing(&folder));

    // And a run that is not stopped makes it, in one step, by a hard link.
    let out = fx.init(&fx.old);
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(out.report["destination_made"], true);
    assert_eq!(out.report["placed_by"], "hard link");
    assert_eq!(listing(&folder).iter().filter(|n| n.starts_with("brain.r8") && !n.contains(".replaced-")).collect::<Vec<_>>(), ["brain.r8"]);
}

/// Points `idx_memories_category` at another column: every row is still read, `COUNT(*)`
/// works, and only `integrity_check` sees that the index does not match its table.
fn damage_an_index(path: &Path) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch("PRAGMA writable_schema=ON").unwrap();
    let n = conn
        .execute(
            "UPDATE sqlite_master SET sql = 'CREATE INDEX idx_memories_category ON memories(topic)'
              WHERE name = 'idx_memories_category'",
            [],
        )
        .unwrap();
    assert_eq!(n, 1);
    let version: i64 = conn.query_row("PRAGMA schema_version", [], |r| r.get(0)).unwrap();
    conn.execute_batch(&format!("PRAGMA schema_version = {}; PRAGMA writable_schema=OFF;", version + 1)).unwrap();
    conn.close().unwrap();
    let conn = Connection::open(path).unwrap();
    assert_eq!(count(&conn, "SELECT COUNT(*) FROM memories"), 3);
    assert_ne!(text(&conn, "PRAGMA integrity_check"), "ok");
    conn.close().unwrap();
}

/// Finding 2: a copy whose only fault is one an integrity check finds stops the import.
#[test]
fn f2_a_copy_that_fails_only_its_integrity_check_is_refused() {
    let fx = fx();
    make_axel(&fx.old);
    let broken = fx.root().join("index.r8");
    fs::copy(&fx.old, &broken).unwrap();
    damage_an_index(&broken);
    let before = state(&broken);
    let out = fx.init(&broken);
    show("init, an index does not match its table", &out);
    assert_eq!(out.code, 3, "{}", out.report);
    let error = out.report["error"].as_str().unwrap();
    assert!(error.starts_with("the integrity check of the copy did not answer ok: "), "{error}");
    assert_eq!(out.report["destination_made"], false);
    assert!(out.report["temporary_copy_deleted"].as_str().unwrap().contains(".init-"), "{}", out.report);
    assert!(listing(fx.new.parent().unwrap()).is_empty(), "{:?}", listing(fx.new.parent().unwrap()));
    assert_eq!(state(&broken), before);
}

/// Finding 3: the destination is absent but a log or an index is beside it: refused,
/// naming it, and nothing is done.
#[test]
fn f3_files_beside_an_absent_destination_are_refused() {
    let fx = fx();
    make_axel(&fx.old);
    let folder = fx.new.parent().unwrap().to_path_buf();
    let source_before = state(&fx.old);
    for name in ["brain.r8-wal", "brain.r8-shm", "brain.r8-journal", "brain.r8.hnsw"] {
        let stray = folder.join(name);
        fs::write(&stray, b"from a brain that is gone").unwrap();
        let out = fx.init(&fx.old);
        show(&format!("init, {name} beside an absent destination"), &out);
        assert_eq!(out.code, 2, "{name}: {}", out.report);
        let error = out.report["error"].as_str().unwrap();
        assert!(error.contains(&format!("but {} beside it", stray.display())), "{error}");
        assert!(error.ends_with("Nothing was done."), "{error}");
        assert_eq!(listing(&folder), [name], "{name}");
        assert!(out.report.get("source").is_none(), "{}", out.report);
        fs::remove_file(&stray).unwrap();
    }
    assert_eq!(state(&fx.old), source_before);
}

/// Finding 4: a file that appears at the destination after `--replace` moved the old brain
/// aside is kept: exit 4, and the report says where the old brain went.
#[test]
fn f4_a_file_that_appears_after_the_move_is_kept_with_exit_4() {
    let fx = fx();
    make_axel(&fx.old);
    empty_v2(&fx.new);
    settle(&fx.new);
    let old_bytes = sha(&fx.new);
    let mut opts = fx.init_opts(&fx.old);
    opts.replace = true;
    let out = with_seam(
        |point, dest| {
            if point == "replaced" {
                fs::write(dest, b"made by an extension that started").unwrap();
            }
            false
        },
        || import::init(&fx.dest(), &opts),
    );
    show("init --replace, a file appeared after the move", &out);
    assert_eq!(out.code, 4, "{}", out.report);
    let moved_to = PathBuf::from(out.report["replaced"]["moved_to"].as_str().unwrap());
    let error = out.report["error"].as_str().unwrap();
    assert!(error.starts_with(&format!("a file appeared at {} while the new brain was being made; it is kept", fx.new.display())), "{error}");
    assert!(error.ends_with(&format!("The brain that was there before is at {}.", moved_to.display())), "{error}");
    assert_eq!(fs::read(&fx.new).unwrap(), b"made by an extension that started");
    assert_eq!(sha(&moved_to), old_bytes);
    assert_eq!(out.report["destination_made"], false);
    assert!(temporaries(fx.new.parent().unwrap()).is_empty());
}

/// Finding 7: a saved row whose key is in both brains and whose values the old plugin
/// changed since the import keeps the new brain's values and is listed, table by table.
#[test]
fn f7_catch_up_lists_saved_rows_changed_in_the_source() {
    let fx = fx();
    make_axel(&fx.old);
    assert_eq!(fx.init(&fx.old).code, 0);
    change_axel(&fx.old, |conn| {
        conn.execute_batch(
            "UPDATE context_data SET value = 'light' WHERE key = 'theme';
             UPDATE staged_memories SET title = 'staged one, renamed' WHERE id = 'stg-1';
             UPDATE patterns SET confidence = 0.9 WHERE name = 'morning';",
        )
        .unwrap();
    });
    let out = fx.catch_up();
    show("catch-up, saved rows changed in the source", &out);
    assert_eq!(out.code, 0, "{}", out.report);
    let changed = &out.report["saved_changed_in_source"];
    assert_eq!(changed["context_data"]["key"], json!(["key", "layer"]));
    assert_eq!(changed["context_data"]["keys"], json!(["theme/prefs"]));
    assert_eq!(changed["staged_memories"]["keys"], json!(["stg-1"]));
    assert_eq!(changed["patterns"]["count"], 1);
    assert_eq!(out.report["saved_rows"], json!({ "context_data": 0, "staged_memories": 0, "patterns": 0 }));
    // Never overwritten.
    let new = Connection::open(&fx.new).unwrap();
    assert_eq!(text(&new, "SELECT value FROM context_data WHERE key = 'theme'"), "dark");
    assert_eq!(text(&new, "SELECT title FROM staged_memories WHERE id = 'stg-1'"), "staged one");
    // A catch-up with nothing changed lists nothing.
    let other = self::fx();
    make_axel(&other.old);
    assert_eq!(other.init(&other.old).code, 0);
    let out = other.catch_up();
    assert_eq!(out.code, 0, "{}", out.report);
    for table in ["context_data", "staged_memories", "patterns"] {
        assert_eq!(out.report["saved_changed_in_source"][table]["count"], 0, "{}", out.report);
    }
}

/// Finding 8: a `--watch` that is not the same bytes as the source is said, with both
/// hashes, by `init` and by `catch-up`; the exit is the same.
#[test]
fn f8_a_watched_file_that_differs_from_the_source_is_said() {
    let fx = fx();
    make_axel(&fx.old);
    let backup = fx.root().join("backup/axel.r8");
    fs::create_dir(backup.parent().unwrap()).unwrap();
    fs::copy(&fx.old, &backup).unwrap();
    let watched_before = state(&fx.old);

    // The same bytes: nothing to say.
    let mut opts = fx.init_opts(&backup);
    opts.watch = Some(fx.old.clone());
    let out = import::init(&fx.dest(), &opts);
    assert_eq!(out.code, 0, "{}", out.report);
    assert!(out.report.get("watch_differs_from_source").is_none(), "{}", out.report);

    // The live file was written to after the backup was taken.
    change_axel(&fx.old, |conn| add_memory(conn, "mem-ffff0006", "late note", "written after the copy", 0.5));
    let watched_before_catch_up = state(&fx.old);
    let out = import::catch_up(&fx.dest(), &CatchUpOpts { from_axel: backup.clone(), watch: Some(fx.old.clone()) });
    show("catch-up, the watched file differs", &out);
    assert_eq!(out.code, 0, "{}", out.report);
    let differs = &out.report["watch_differs_from_source"];
    assert_eq!(differs["watch"]["sha256"], json!(sha(&fx.old)));
    assert_eq!(differs["source"]["sha256"], json!(sha(&backup)));
    assert_eq!(differs["watch"]["wal_sha256"], Value::Null);

    fs::remove_file(&fx.new).unwrap();
    let out = import::init(&fx.dest(), &opts);
    assert_eq!(out.code, 0, "{}", out.report);
    assert_eq!(out.report["watch_differs_from_source"]["watch"]["path"], json!(fx.old.to_string_lossy()));
    // The watched file is read, never written.
    assert_eq!(state(&fx.old), watched_before_catch_up);
    assert_ne!(watched_before, watched_before_catch_up);
}

/// Finding 9: a usage table the destination lacks fails `verify` when `init` recorded rows
/// in it, even if the source has none left.
#[test]
fn f9_verify_fails_a_missing_table_that_init_counted() {
    let fx = fx();
    let (_, map) = migrated(&fx);
    change_axel(&fx.old, |conn| conn.execute_batch("DELETE FROM events").unwrap());
    Connection::open(&fx.new).unwrap().execute_batch("DROP TABLE events").unwrap();
    let out = fx.verify(&[&map]);
    show("verify, a counted table is missing", &out);
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(shortfalls(&out.report), ["events"]);
    assert_eq!(item(&out.report, "events")["recorded"], 1);
}

/// Finding 9: a table `init` counted that the source no longer has is still checked, and
/// said.
#[test]
fn f9_verify_reports_a_counted_table_gone_from_the_source() {
    let fx = fx();
    let (_, map) = migrated(&fx);
    change_axel(&fx.old, |conn| conn.execute_batch("DROP TABLE sessions").unwrap());
    let out = fx.verify(&[&map]);
    assert_eq!(out.code, 0, "{}", out.report);
    let sessions = item(&out.report, "sessions");
    assert_eq!(sessions["gone_from_source"], true, "{sessions}");
    assert_eq!(sessions["ok"], true);
    Connection::open(&fx.new).unwrap().execute_batch("DELETE FROM sessions").unwrap();
    let out = fx.verify(&[&map]);
    assert_eq!(out.code, 1, "{}", out.report);
    assert_eq!(shortfalls(&out.report), ["sessions"]);
}

/// Finding 10: the source under another name (a hard link) is the source.
#[test]
fn f10_a_destination_that_is_a_hard_link_of_the_source_is_refused() {
    let fx = fx();
    make_axel(&fx.old);
    fs::hard_link(&fx.old, &fx.new).unwrap();
    let before = state(&fx.old);
    let mut opts = fx.init_opts(&fx.old);
    opts.replace = true;
    opts.even_if_running = true;
    let out = import::init(&fx.dest(), &opts);
    assert_eq!(out.code, 2, "{}", out.report);
    assert!(out.report["error"].as_str().unwrap().contains("a source is never written"), "{}", out.report);
    assert_eq!(state(&fx.old), before);
    assert_eq!(listing(fx.new.parent().unwrap()), ["brain.r8"]);
}

/// Finding 11: the temporary directory of a process that is alive is left: pid 1's, and
/// this process's own.
#[test]
fn f11_temporaries_of_live_processes_are_left() {
    let fx = fx();
    make_axel(&fx.old);
    let folder = fx.new.parent().unwrap().to_path_buf();
    let own = format!(".init-{}-1", std::process::id());
    for name in [".init-1-1", own.as_str()] {
        fs::create_dir(folder.join(name)).unwrap();
        fs::write(folder.join(name).join("axel.r8"), b"in use").unwrap();
    }
    let out = fx.init(&fx.old);
    assert_eq!(out.code, 0, "{}", out.report);
    assert!(folder.join(".init-1-1/axel.r8").exists());
    assert!(folder.join(&own).join("axel.r8").exists());
    assert!(out.report.get("stale_temporaries_deleted").is_none(), "{}", out.report);
    assert_eq!(out.report["temporaries_left"].as_array().unwrap().len(), 2, "{}", out.report);
}

/// Overwrites every page of a database after the first, keeping its size.
fn tear(path: &Path) {
    let mut bytes = fs::read(path).unwrap();
    for b in &mut bytes[4096..] {
        *b = 0xAB;
    }
    fs::write(path, &bytes).unwrap();
}

/// Finding 12: a source that is written while it is read in place, so that SQLite fails
/// on it, is exit 3, not 5.
#[test]
fn f12_a_source_torn_while_read_is_exit_3() {
    let fx = fx();
    let (_, map) = migrated(&fx);
    let out = with_seam(|point, source| { if point == "opened" { tear(source) } false }, || fx.verify(&[&map]));
    show("verify, the source torn while read", &out);
    assert_eq!(out.code, 3, "{}", out.report);
    let error = out.report["error"].as_str().unwrap();
    assert!(error.starts_with(&format!("the source {} changed while it was read; nothing was changed", fx.old.display())), "{error}");

    let other = self::fx();
    make_axel(&other.old);
    assert_eq!(other.init(&other.old).code, 0);
    let hash = sha(&other.new);
    let out = with_seam(|point, source| { if point == "opened" { tear(source) } false }, || other.catch_up());
    assert_eq!(out.code, 3, "{}", out.report);
    settle(&other.new);
    assert_eq!(sha(&other.new), hash, "catch-up changed the new brain");
}

/// Finding 14: the new brain's own integrity failure is not the source's: exit 5, and the
/// message says the source is fine and nothing was put at the destination.
#[test]
fn f14_the_new_brains_own_integrity_failure_is_exit_5() {
    let fx = fx();
    make_axel(&fx.old);
    let out = with_seam(|point, db| { if point == "checkpoint" { damage_an_index(db) } false }, || fx.init(&fx.old));
    show("init, the new brain fails its integrity check", &out);
    assert_eq!(out.code, 5, "{}", out.report);
    let error = out.report["error"].as_str().unwrap();
    assert!(error.starts_with("the integrity check of the new brain did not answer ok: "), "{error}");
    assert!(error.contains("the source is fine (its copy passed its own check) and the new brain is not made"), "{error}");
    assert!(error.ends_with(&format!("Nothing was put at {}.", fx.new.display())), "{error}");
    assert!(listing(fx.new.parent().unwrap()).is_empty());
}
