//! Commit 14, the notes pass: section 2, tests 20 and 21 (both as rewritten in 4.3), and
//! `on_forgotten` for a note.
//!
//! Test 20 is the functions named `t20_*`, test 21 the functions named `t21_*`; the others
//! guard single rules of section 2.6 on their own.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver};
use std::sync::Arc;
use std::time::Duration;

use axel::brain_v2::{self, OpenOpts};
use axel::embed::EmbedService;
use axel::ext::config::{EmbedderKind, ExtConfig, ForumMode};
use axel::ext::types::{IndexerCtx, Job, Mode, Status, TestHooks};
use axel::index_notes::{self, PassReport};
use axel::project::project_dir;
use axel::project_memory::{RecordLinks, Sensitivity};
use axel::store::{self, Inserted, NewRow};
use axel::vindex::VecDelta;
use rusqlite::Connection;
use serde_json::json;
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const KEY: &str = "p1111111111111111";
const KEY_B: &str = "p2222222222222222";

// ── Helpers ─────────────────────────────────────────────────────────────────

struct World {
    _tmp: TempDir,
    base: PathBuf,
    ctx: IndexerCtx,
    _jobs_rx: Receiver<Job>,
    _vec_rx: Receiver<VecDelta>,
}

fn cfg(root: &Path) -> ExtConfig {
    ExtConfig {
        brain: root.join("brain.r8"),
        base: root.join("base"),
        data_dir: root.join("data"),
        recall: true,
        recall_max_items: 5,
        recall_max_chars: 1600,
        recall_budget_ms: 150,
        recall_min_sem: 0.3,
        recall_min_lex: 0.5,
        recall_on_events: false,
        recall_exclude_paths: Vec::new(),
        capture: true,
        archives: true,
        forum: ForumMode::Auto,
        worker_scope_all: false,
        forget_now_min_hidden_hours: 24,
        handoff_on_start: false,
        sources: None,
        archive_events: Vec::new(),
        embedder: EmbedderKind::Off,
        model_dir: root.join("model"),
        log_queries: false,
        warnings: Vec::new(),
        home: root.join("home"),
    }
}

/// A temp base with an empty `memory/`, a fresh version 2 brain, and an indexer context in
/// mode `Ready` whose connection came through `open_v2`.
fn world() -> World {
    let tmp = TempDir::new().unwrap();
    let cfg = cfg(tmp.path());
    let base = cfg.base.clone();
    fs::create_dir_all(base.join("memory")).unwrap();
    let opts = OpenOpts { agent: Some("test"), model: "hash-384", dim: 384, upgrade_v1: false };
    let brain = brain_v2::open_v2(&cfg.brain, &opts).unwrap();
    let (jobs, jobs_rx) = sync_channel::<Job>(1024);
    let (vec_tx, vec_rx) = channel::<VecDelta>();
    let test = Arc::new(TestHooks::default());
    let ctx = IndexerCtx {
        cfg,
        conn: Some(brain.conn),
        signer: brain.signer,
        embed: EmbedService::start(None, jobs.clone(), test.clone()),
        vec_tx,
        jobs,
        status: Arc::new(Status::default()),
        mode: Arc::new(AtomicU8::new(Mode::Ready as u8)),
        last_run: HashMap::new(),
        open_tried: None,
        vectors_sent: None,
        stop: Arc::new(AtomicBool::new(false)),
        brain_file: None,
        clock: Arc::new(|| chrono::Utc::now().timestamp_millis()),
        test,
        writing: Arc::new(AtomicBool::new(false)),
    };
    World { _tmp: tmp, base, ctx, _jobs_rx: jobs_rx, _vec_rx: vec_rx }
}

impl World {
    fn conn(&self) -> &Connection {
        self.ctx.conn.as_ref().unwrap()
    }

    fn file(&self, key: &str) -> PathBuf {
        self.base.join("memory").join(file_name(key))
    }

    /// Writes the notes file of `key`. The pause makes sure its time moves, so that a
    /// rewrite of the same length is seen too.
    fn write(&self, key: &str, body: &str) {
        std::thread::sleep(Duration::from_millis(25));
        fs::write(self.file(key), body).unwrap();
    }

    fn pass(&mut self) -> PassReport {
        let report = index_notes::pass_counted(&mut self.ctx).unwrap();
        assert!(self.conn().is_autocommit(), "a pass leaves no transaction open");
        report
    }

    /// The ids of every `note_file` row, sorted.
    fn notes(&self) -> Vec<String> {
        let mut stmt = self
            .conn()
            .prepare("SELECT mem_id FROM memory_ext WHERE origin_kind = 'note_file' ORDER BY mem_id")
            .unwrap();
        let rows = stmt.query_map([], |r| r.get(0)).unwrap();
        rows.map(|r| r.unwrap()).collect()
    }

    fn count(&self, sql: &str) -> i64 {
        self.conn().query_row(sql, [], |r| r.get(0)).unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    fn text(&self, sql: &str, arg: &str) -> Option<String> {
        self.conn().query_row(sql, [arg], |r| r.get(0)).unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    /// One column of the `note_files` row of `key`.
    fn nf_text(&self, key: &str, column: &str) -> Option<String> {
        let path = self.file(key).to_string_lossy().into_owned();
        self.text(&format!("SELECT {column} FROM note_files WHERE path = ?1"), &path)
    }

    fn nf_int(&self, key: &str, column: &str) -> Option<i64> {
        let path = self.file(key).to_string_lossy().into_owned();
        self.conn()
            .query_row(&format!("SELECT {column} FROM note_files WHERE path = ?1"), [path], |r| r.get(0))
            .unwrap()
    }

    fn nf_rows(&self) -> i64 {
        self.count("SELECT count(*) FROM note_files")
    }

    fn tombstones(&self) -> i64 {
        self.count("SELECT count(*) FROM memory_tombstones")
    }

    fn purge_pending(&self) -> i64 {
        self.count("SELECT count(*) FROM context_data WHERE key = 'purge_pending' AND layer = 'axel'")
    }

    fn content(&self, id: &str) -> Option<String> {
        self.text("SELECT (SELECT content FROM memories WHERE id = ?1)", id)
    }

    /// A row Axel made itself (`axel_remember`), in the project of `KEY`.
    fn put_tool_row(&self, id: &str, content: &str) {
        let scope = project_dir(KEY);
        let row = NewRow {
            id: id.to_string(),
            class: "preferences",
            project_key: &scope,
            title: format!("title of {id}"),
            abstract_text: String::new(),
            content: content.to_string(),
            tags: Vec::new(),
            created_ms: 1_700_000_000_000,
            time_source: "exact",
            sensitivity: Sensitivity::Normal,
            provenance: "test".to_string(),
            expires_ms: None,
            importance: 0.5,
            origin_kind: "tool",
            origin_ref: "",
            unit_pos: None,
            session_id: None,
            logical: None,
            links: RecordLinks::default(),
            turn_key: None,
            hidden: None,
        };
        let conn = self.conn();
        brain_v2::begin_write(conn).unwrap();
        let out = store::insert_row(conn, self.ctx.signer.as_ref(), &row).unwrap();
        conn.execute_batch("COMMIT").unwrap();
        assert!(matches!(out, Inserted::New { .. }), "{id}: {out:?}");
    }

    /// Everything saved of one row, to show that it was not touched.
    fn saved(&self, id: &str) -> Option<String> {
        self.text(
            "SELECT (SELECT quote(m.title) || quote(m.content) || quote(m.tags) || quote(m.project_key)
                         || quote(m.provenance) || quote(e.origin_kind) || quote(e.origin_ref)
                         || quote(e.hidden_ms) || quote(e.created_ms)
                       FROM memories m JOIN memory_ext e ON e.mem_id = m.id WHERE m.id = ?1)",
            id,
        )
    }
}

fn file_name(key: &str) -> String {
    format!("project-{key}.jsonl")
}

/// One record line of core's `MemoryRecord`, with its line break.
fn rec_of(key: &str, id: &str, ts: i64, content: &str) -> String {
    let mut line = json!({
        "namespace": format!("project-{key}"), "timestamp_ms": ts, "content": content,
        "tags": [], "id": id, "project": key
    })
    .to_string();
    line.push('\n');
    line
}

fn rec(id: &str, ts: i64, content: &str) -> String {
    rec_of(KEY, id, ts, content)
}

/// One tombstone line of core's `TombstoneLine`, with its line break.
fn tomb(id: &str) -> String {
    let mut line = json!({ "tombstone": id, "timestamp_ms": 1_700_000_009_000i64 }).to_string();
    line.push('\n');
    line
}

fn hash16(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().take(8).map(|b| format!("{b:02x}")).collect()
}

fn ids(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

const N1: &str = "mem-one";
const N2: &str = "mem-two";
const N3: &str = "mem-three";

fn n1() -> String {
    rec(N1, 1_700_000_000_000, "the first note is about alpha")
}
fn n2() -> String {
    rec(N2, 1_700_000_000_100, "the second note is about beta")
}
fn n3() -> String {
    rec(N3, 1_700_000_000_200, "the third note is about gamma")
}

/// Three notes in the file of `KEY`, mirrored by one pass.
fn three_mirrored() -> World {
    let mut w = world();
    w.write(KEY, &format!("{}{}{}", n1(), n2(), n3()));
    let report = w.pass();
    assert_eq!(report.inserted, 3);
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    w
}

// ── Test 20 ─────────────────────────────────────────────────────────────────

/// 3 records, 1 tombstone line, a torn last line, 1 record of another project, 1 secret
/// record → 3 normal rows and 1 secret row whose body is in no word index.
#[test]
fn t20_three_records_a_tombstone_a_torn_tail_another_project_and_a_secret() {
    let mut w = world();
    let long = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen";
    let sensitive = json!({
        "namespace": format!("project-{KEY}"), "timestamp_ms": 1_700_000_000_100i64,
        "content": "two is sensitive", "tags": ["x"], "id": N2, "project": KEY, "sensitivity": "sensitive"
    });
    let expiring = json!({
        "namespace": format!("project-{KEY}"), "timestamp_ms": 1_700_000_000_200i64,
        "content": "three expires", "tags": null, "id": N3, "project": KEY,
        "retention": { "max_age_days": 30 }
    });
    let secret = json!({
        "namespace": format!("project-{KEY}"), "timestamp_ms": 1_700_000_000_600i64,
        "content": "zebrasecretbody is the launch code", "tags": ["creds"], "id": "mem-secret",
        "project": KEY, "sensitivity": "secret"
    });
    let body = format!(
        "{}{sensitive}\n{expiring}\n{}{}{secret}\n{}",
        rec(N1, 1_700_000_000_000, long),
        tomb("mem-dead"),
        rec_of(KEY_B, "mem-other", 1_700_000_000_300, "a note of another project"),
        r#"{"namespace":"project-p1111111111111111","timestamp_ms":17000"#,
    );
    w.write(KEY, &body);
    let report = w.pass();
    assert_eq!(report.inserted, 4);
    assert_eq!((report.files_read, report.purged), (1, 0));

    assert_eq!(w.notes(), ids(&[N1, "mem-secret", N3, N2]));
    assert_eq!(w.count("SELECT count(*) FROM memories"), 4);
    assert_eq!(w.count("SELECT count(*) FROM memories WHERE sensitivity = 'normal'"), 3);
    assert_eq!(w.count("SELECT count(*) FROM memories WHERE sensitivity = 'secret'"), 1);

    // Every value of the row, as 2.6 names it.
    let scope = project_dir(KEY);
    let name = file_name(KEY);
    assert_eq!(
        w.count(&format!(
            "SELECT count(*) FROM memories m JOIN memory_ext e ON e.mem_id = m.id
              WHERE m.category = 'note' AND m.project_key = '{scope}'
                AND m.provenance = 'synaps:note:{name}' AND e.origin_kind = 'note_file'
                AND e.origin_ref = '{name}' AND e.time_source = 'exact' AND e.hidden_ms IS NULL"
        )),
        4
    );
    assert_eq!(
        w.text("SELECT title FROM memories WHERE id = ?1", N1).unwrap(),
        "one two three four five six seven eight nine ten eleven twelve"
    );
    assert_eq!(w.content(N1).unwrap(), long);
    assert_eq!(
        w.count(&format!("SELECT created_ms FROM memory_ext WHERE mem_id = '{N1}'")),
        1_700_000_000_000
    );
    // `Sensitive` is a normal row with the tag.
    assert_eq!(w.text("SELECT sensitivity FROM memories WHERE id = ?1", N2).unwrap(), "normal");
    assert_eq!(w.text("SELECT tags FROM memories WHERE id = ?1", N2).unwrap(), r#"["x","sensitive"]"#);
    // The expiry: the note's time and 30 days.
    let expires = w.text("SELECT expires_at FROM memories WHERE id = ?1", N3).unwrap();
    let expires_ms = chrono::DateTime::parse_from_rfc3339(&expires).unwrap().timestamp_millis();
    assert_eq!(expires_ms, 1_700_000_000_200 + 30 * 86_400_000);
    assert_eq!(w.text("SELECT expires_at FROM memories WHERE id = ?1", N1), None);

    // The secret row: its body is in no word index, not even its first words as a title.
    assert_eq!(w.content("mem-secret").unwrap(), "zebrasecretbody is the launch code");
    for word in ["zebrasecretbody", "launch"] {
        assert_eq!(
            w.count(&format!("SELECT count(*) FROM memories_fts WHERE memories_fts MATCH '{word}'")),
            0,
            "{word} is in the memory word index"
        );
        assert_eq!(
            w.count(&format!("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH '{word}'")),
            0,
            "{word} is in the chunk word index"
        );
    }
    let fts: String = w
        .conn()
        .query_row(
            "SELECT title || '|' || abstract || '|' || content FROM memories_fts WHERE mem_id = 'mem-secret'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(fts, "Secret note||");
    assert_eq!(w.text("SELECT abstract_text FROM memories WHERE id = ?1", "mem-secret").unwrap(), "");
    // A normal row is in it.
    assert_eq!(w.count("SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'thirteen'"), 1);

    // The record of another project and the torn tail are no rows; no tombstone is written.
    assert_eq!(w.content("mem-other"), None);
    assert_eq!(w.tombstones(), 0);
    assert_eq!(w.purge_pending(), 0, "nothing was forgotten");

    assert_eq!(w.nf_rows(), 1);
    assert_eq!(w.nf_int(KEY, "ok"), Some(1));
    assert_eq!(w.nf_int(KEY, "live_count"), Some(4));
    assert_eq!(w.nf_text(KEY, "project_key").unwrap(), KEY);
    assert_eq!(w.nf_text(KEY, "reason"), None);
    assert_eq!(w.nf_text(KEY, "bad_lines"), None);
    assert_eq!(w.nf_int(KEY, "bad_lines_changed_ms"), None);
    assert_eq!(
        w.text("SELECT project_key FROM projects WHERE scope_dir = ?1", &scope).unwrap(),
        KEY
    );

    // The same file again: skipped by its size and time.
    let report = w.pass();
    assert_eq!((report.files_unchanged, report.files_read, report.inserted), (1, 0, 0));
    assert_eq!(w.notes().len(), 4);
}

#[test]
fn t20_an_id_on_two_live_lines_is_one_row_with_the_later_text() {
    let mut w = world();
    w.write(
        KEY,
        &format!("{}{}{}", rec(N1, 1, "the earlier text"), n2(), rec(N1, 2, "the later text")),
    );
    w.pass();
    assert_eq!(w.notes(), ids(&[N1, N2]));
    assert_eq!(w.content(N1).unwrap(), "the later text");
    assert_eq!(w.count("SELECT count(*) FROM memories"), 2);
}

/// Guards what the owner would lose: a note he forgot through core must leave the brain.
#[test]
fn t20_appending_a_tombstone_removes_the_row_and_writes_no_tombstone_row() {
    let mut w = three_mirrored();
    assert_eq!(w.count("SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'beta'"), 1);
    w.write(KEY, &format!("{}{}{}{}", n1(), n2(), n3(), tomb(N2)));
    let report = w.pass();
    assert_eq!(report.purged, 1, "the tombstone line purges its row in one pass");
    assert_eq!(w.notes(), ids(&[N1, N3]));
    assert_eq!(w.content(N2), None);
    assert_eq!(w.count("SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'beta'"), 0);
    assert_eq!(w.tombstones(), 0, "no memory_tombstones row is written for a note");
    assert_eq!(w.purge_pending(), 1, "a purge that forgets calls mark_purged before its COMMIT");
    assert_eq!(w.nf_int(KEY, "live_count"), Some(2));

    // The file's own tombstone line keeps it out: a later record line under a tombstone is
    // not live for core, and is not mirrored.
    w.write(KEY, &format!("{}{}{}{}{}", n1(), n2(), n3(), tomb(N2), n2()));
    w.pass();
    assert_eq!(w.notes(), ids(&[N1, N3]));
}

#[test]
fn t20_a_rewrite_without_a_record_removes_it_and_a_tool_row_is_never_removed() {
    let mut w = three_mirrored();
    w.put_tool_row("mem-tool", "what axel was told to remember");
    let tool = w.saved("mem-tool");
    assert!(tool.is_some());

    // The sweep: core rewrites the file without the record.
    w.write(KEY, &format!("{}{}", n1(), n2()));
    let report = w.pass();
    assert_eq!(report.purged, 1);
    assert_eq!(w.notes(), ids(&[N1, N2]));
    assert_eq!(w.content(N3), None);
    assert_eq!(w.tombstones(), 0);
    assert_eq!(w.purge_pending(), 1);
    assert_eq!(w.saved("mem-tool"), tool, "a tool row is never touched by this pass");

    // Nor by an empty file, nor by the file going away.
    w.write(KEY, "");
    w.pass();
    assert_eq!(w.notes(), ids(&[]));
    assert_eq!(w.saved("mem-tool"), tool);
    fs::remove_file(w.file(KEY)).unwrap();
    w.pass();
    assert_eq!(w.saved("mem-tool"), tool);
}

#[test]
fn t20_an_id_that_is_already_a_tool_row_is_not_mirrored_and_the_tool_row_is_unchanged() {
    let mut w = world();
    w.put_tool_row("mem-shared", "the text axel holds under this id");
    let tool = w.saved("mem-shared");

    w.write(KEY, &format!("{}{}", n1(), rec("mem-shared", 5, "the text of core's note")));
    let report = w.pass();
    assert_eq!(report.inserted, 1);
    assert_eq!(w.notes(), ids(&[N1]));
    assert_eq!(w.saved("mem-shared"), tool);
    assert_eq!(w.content("mem-shared").unwrap(), "the text axel holds under this id");
    assert_eq!(w.nf_text(KEY, "reason").unwrap(), "ids_skipped: 1");
    assert_eq!(w.nf_int(KEY, "ok"), Some(1));

    // A tombstone line for that id, and then its absence, delete nothing that is not this
    // file's own mirror.
    w.write(KEY, &format!("{}{}{}", n1(), rec("mem-shared", 5, "the text of core's note"), tomb("mem-shared")));
    let report = w.pass();
    assert_eq!(report.purged, 0);
    assert_eq!(w.saved("mem-shared"), tool);
    assert_eq!(w.nf_text(KEY, "reason"), None, "the id is no longer live, so nothing is skipped");
    w.write(KEY, &n1());
    w.pass();
    assert_eq!(w.saved("mem-shared"), tool);
    assert_eq!(w.notes(), ids(&[N1]));
}

#[test]
fn t20_the_same_id_in_two_project_files_is_mirrored_for_the_first_file_only() {
    let mut w = world();
    w.write(KEY, &format!("{}{}", n1(), rec("mem-both", 7, "the note as the first project has it")));
    w.write(KEY_B, &rec_of(KEY_B, "mem-both", 8, "the note as the second project has it"));
    let report = w.pass();
    assert_eq!((report.files_read, report.inserted), (2, 2));
    assert_eq!(w.notes(), ids(&["mem-both", N1]));
    assert_eq!(w.content("mem-both").unwrap(), "the note as the first project has it");
    assert_eq!(
        w.text("SELECT origin_ref FROM memory_ext WHERE mem_id = ?1", "mem-both").unwrap(),
        file_name(KEY)
    );
    assert_eq!(
        w.text("SELECT project_key FROM memories WHERE id = ?1", "mem-both").unwrap(),
        project_dir(KEY)
    );
    assert_eq!(w.nf_text(KEY, "reason"), None);
    assert_eq!(w.nf_text(KEY_B, "reason").unwrap(), "ids_skipped: 1");
    assert_eq!(w.nf_int(KEY_B, "ok"), Some(1));
    let first = w.saved("mem-both");

    // The second file forgets its own note of that id, and then goes away: the first
    // file's row is another file's row and stays.
    w.write(
        KEY_B,
        &format!("{}{}", rec_of(KEY_B, "mem-both", 8, "the note as the second project has it"), tomb("mem-both")),
    );
    let report = w.pass();
    assert_eq!(report.purged, 0);
    assert_eq!(w.saved("mem-both"), first);
    fs::remove_file(w.file(KEY_B)).unwrap();
    let report = w.pass();
    assert_eq!((report.files_gone, report.purged), (1, 0));
    assert_eq!(w.saved("mem-both"), first);
    assert_eq!(w.notes(), ids(&["mem-both", N1]));
}

#[test]
fn t20_an_id_with_a_space_in_it_is_skipped() {
    let mut w = world();
    let longest = "a".repeat(128);
    let too_long = "a".repeat(129);
    w.write(
        KEY,
        &format!(
            "{}{}{}{}{}{}",
            n1(),
            rec("mem with space", 2, "an id with a space"),
            rec("mem\u{7}bell", 3, "an id with a control character"),
            rec(&too_long, 4, "an id of 129 bytes"),
            rec(&longest, 5, "an id of 128 bytes"),
            rec("", 6, "an empty id"),
        ),
    );
    let report = w.pass();
    assert_eq!(report.inserted, 2);
    assert_eq!(w.notes(), ids(&[longest.as_str(), N1]));
    assert_eq!(w.count("SELECT count(*) FROM memories"), 2);
    assert_eq!(w.nf_text(KEY, "reason").unwrap(), "ids_skipped: 4");
    assert_eq!(w.nf_int(KEY, "ok"), Some(1));

    assert!(index_notes::valid_note_id("mem-0123_abc.def:9"));
    for bad in ["", " ", "a b", "a\tb", "a\nb", "a\u{0}b", "a\u{a0}b", "a\u{85}b"] {
        assert!(!index_notes::valid_note_id(bad), "{bad:?}");
    }
}

#[test]
fn t20_deleting_the_notes_file_removes_its_rows_and_its_note_files_row() {
    let mut w = three_mirrored();
    w.write(KEY_B, &rec_of(KEY_B, "mem-kept", 9, "a note of the project that stays"));
    w.put_tool_row("mem-tool", "what axel was told to remember");
    w.pass();
    assert_eq!(w.nf_rows(), 2);
    assert_eq!(w.notes().len(), 4);

    fs::remove_file(w.file(KEY)).unwrap();
    let report = w.pass();
    assert!(report.listed);
    assert_eq!((report.files_gone, report.purged), (1, 3));
    assert_eq!(w.notes(), ids(&["mem-kept"]));
    assert_eq!(w.nf_rows(), 1);
    assert_eq!(w.nf_int(KEY_B, "ok"), Some(1));
    assert_eq!(w.tombstones(), 0);
    assert_eq!(w.purge_pending(), 1);
    assert!(w.saved("mem-tool").is_some());
    assert_eq!(w.count("SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'gamma'"), 0);

    // No tombstones: the file back in place is mirrored again.
    w.write(KEY, &format!("{}{}{}", n1(), n2(), n3()));
    w.pass();
    assert_eq!(w.notes(), ids(&["mem-kept", N1, N3, N2]));
}

/// Guards what the owner would lose: every note, because a folder was away for a moment.
#[test]
fn t20_a_pass_with_the_memory_folder_unreadable_removes_nothing() {
    let mut w = three_mirrored();
    let memory = w.base.join("memory");
    let away = w.base.join("memory.away");
    let row_before = w.nf_int(KEY, "mtime_ns");

    // The folder is not there (a mount that is away).
    fs::rename(&memory, &away).unwrap();
    let report = w.pass();
    assert_eq!(w.notes(), ids(&[N1, N3, N2]), "a folder that cannot be listed removes nothing");
    assert_eq!(w.nf_rows(), 1);
    assert_eq!((report.files_gone, report.purged), (0, 0));
    assert!(!report.listed);

    // Something that is not a folder is in its place.
    fs::write(&memory, b"not a folder").unwrap();
    let report = w.pass();
    assert!(!report.listed);
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    assert_eq!(w.nf_rows(), 1);
    fs::remove_file(&memory).unwrap();

    // The base itself is gone.
    let base_away = w.base.with_extension("away");
    fs::rename(&w.base, &base_away).unwrap();
    let report = w.pass();
    assert!(!report.listed);
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    fs::rename(&base_away, &w.base).unwrap();

    // Back again: the same files, nothing to do.
    fs::rename(&away, &memory).unwrap();
    let report = w.pass();
    assert!(report.listed);
    assert_eq!((report.files_unchanged, report.files_gone, report.purged), (1, 0, 0));
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    assert_eq!(w.nf_int(KEY, "mtime_ns"), row_before);
    assert_eq!(w.purge_pending(), 0);
    assert_eq!(w.tombstones(), 0);
}

// ── Test 21 ─────────────────────────────────────────────────────────────────

const GARBAGE_1: &str = "{\"namespace\":\"project-p1111111111111111\",\"timesta";
const GARBAGE_2: &str = "this line is not json at all";

#[test]
fn t21_a_to_d_bad_lines() {
    let mut w = three_mirrored();

    // (a) a bad line appears in the middle and all 3 records are still there.
    w.write(KEY, &format!("{}{GARBAGE_1}\n{}{}", n1(), n2(), n3()));
    let before = chrono::Utc::now().timestamp_millis();
    let report = w.pass();
    assert_eq!((report.inserted, report.purged), (0, 0));
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    assert_eq!(w.nf_int(KEY, "ok"), Some(1));
    let h1 = hash16(GARBAGE_1.as_bytes());
    assert_eq!(w.nf_text(KEY, "bad_lines").unwrap(), h1);
    let changed_a = w.nf_int(KEY, "bad_lines_changed_ms").expect("bad_lines_changed_ms is set");
    assert!(changed_a >= before && changed_a <= chrono::Utc::now().timestamp_millis());
    assert_eq!(w.nf_int(KEY, "mtime_ns"), Some(0), "the next pass reads the file again");
    // That next pass finds the set unchanged and nothing to do.
    let report = w.pass();
    assert_eq!((report.files_read, report.purged), (1, 0));
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    assert_ne!(w.nf_int(KEY, "mtime_ns"), Some(0));
    assert_eq!(w.nf_int(KEY, "bad_lines_changed_ms"), Some(changed_a), "the set did not change");
    assert_eq!(w.pass().files_unchanged, 1);

    // (b) a forget still lands: with that bad line in place, a tombstone for note 2.
    w.write(KEY, &format!("{}{GARBAGE_1}\n{}{}{}", n1(), n2(), n3(), tomb(N2)));
    let report = w.pass();
    assert_eq!(report.purged, 1);
    assert_eq!(w.notes(), ids(&[N1, N3]), "the row is gone after one pass");
    assert_eq!(w.tombstones(), 0);
    assert_eq!(w.nf_text(KEY, "bad_lines").unwrap(), h1);

    // (c) absence waits one pass when the bad set changed: note 3's line becomes garbage.
    w.write(KEY, &format!("{}{GARBAGE_1}\n{}{GARBAGE_2}\n{}", n1(), n2(), tomb(N2)));
    let report = w.pass();
    assert_eq!(report.purged, 0);
    assert_eq!(w.notes(), ids(&[N1, N3]), "note 3's row is still there after the first pass");
    let mut two = [h1.clone(), hash16(GARBAGE_2.as_bytes())];
    two.sort();
    assert_eq!(w.nf_text(KEY, "bad_lines").unwrap(), two.join(","));
    assert!(w.nf_int(KEY, "bad_lines_changed_ms").unwrap() >= changed_a);
    assert_eq!(w.nf_int(KEY, "ok"), Some(1));
    // The second pass, with the file untouched.
    let report = w.pass();
    assert_eq!(report.purged, 1);
    assert_eq!(w.notes(), ids(&[N1]), "after the second pass the row is gone");
    assert_eq!(w.content(N3), None);
    assert_eq!(w.tombstones(), 0);

    // (d) absence with the bad set unchanged is immediate: the file without note 1, both
    // bad lines kept.
    w.write(KEY, &format!("{GARBAGE_1}\n{}{GARBAGE_2}\n{}", n2(), tomb(N2)));
    let report = w.pass();
    assert_eq!(report.purged, 1);
    assert_eq!(w.notes(), ids(&[]), "gone after one pass");
    assert_eq!(w.nf_text(KEY, "bad_lines").unwrap(), two.join(","));
    assert_eq!(w.nf_int(KEY, "live_count"), Some(0));
    assert_eq!(w.tombstones(), 0);
}

#[test]
fn t21_e_a_file_above_64_mib_is_refused_and_the_mirror_is_unchanged() {
    let mut w = three_mirrored();
    let rows_before = w.saved(N1);
    std::thread::sleep(Duration::from_millis(25));
    // A sparse file: no data is written.
    let sparse = fs::File::create(w.file(KEY)).unwrap();
    sparse.set_len(64 * 1024 * 1024 + 1).unwrap();
    drop(sparse);

    let report = w.pass();
    assert_eq!((report.files_refused, report.purged, report.inserted), (1, 0, 0));
    assert_eq!(w.nf_int(KEY, "ok"), Some(0));
    assert_eq!(w.nf_int(KEY, "size"), Some(64 * 1024 * 1024 + 1));
    let reason = w.nf_text(KEY, "reason").unwrap();
    assert!(reason.contains("above the limit"), "{reason}");
    assert_eq!(w.notes(), ids(&[N1, N3, N2]), "the mirror is unchanged");
    assert_eq!(w.saved(N1), rows_before);
    assert_eq!(w.nf_int(KEY, "live_count"), Some(3));
    assert_eq!(w.purge_pending(), 0);

    // A refused file is tried again at every pass (wave 4, R-14 item 1), and a file that
    // stays refused for the same reason writes nothing.
    let scanned = w.nf_int(KEY, "scanned_ms");
    std::thread::sleep(Duration::from_millis(5));
    let report = w.pass();
    assert_eq!((report.files_unchanged, report.files_refused), (0, 1));
    assert_eq!(w.nf_int(KEY, "scanned_ms"), scanned, "the refusal row was written again");
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));

    // A readable file again: read, and `ok` is 1.
    w.write(KEY, &format!("{}{}", n1(), n2()));
    let report = w.pass();
    assert_eq!(report.purged, 1);
    assert_eq!(w.nf_int(KEY, "ok"), Some(1));
    assert_eq!(w.nf_text(KEY, "reason"), None);
    assert_eq!(w.notes(), ids(&[N1, N2]));
}

#[test]
fn t21_f_an_empty_file_removes_the_rows_and_the_full_file_mirrors_them_again() {
    let mut w = three_mirrored();
    w.write(KEY, "");
    let report = w.pass();
    assert_eq!(report.purged, 3);
    assert_eq!(w.notes(), ids(&[]));
    assert_eq!(w.count("SELECT count(*) FROM memories"), 0);
    assert_eq!(w.tombstones(), 0, "the rows are removed with no tombstone");
    assert_eq!(w.nf_int(KEY, "live_count"), Some(0));

    w.write(KEY, &format!("{}{}{}", n1(), n2(), n3()));
    let report = w.pass();
    assert_eq!(report.inserted, 3);
    assert_eq!(w.notes(), ids(&[N1, N3, N2]), "they are mirrored again");
    assert_eq!(w.content(N2).unwrap(), "the second note is about beta");
}

// ── Single rules of 2.6 ─────────────────────────────────────────────────────

/// Guards what the owner would lose: a note whose line was damaged by a torn append must
/// not go in the pass that first sees the damage.
#[test]
fn a_note_whose_line_was_just_damaged_survives_exactly_one_pass() {
    let mut w = three_mirrored();
    w.write(KEY, &format!("{}{GARBAGE_2}\n{}", n1(), n3()));
    let report = w.pass();
    assert_eq!(report.purged, 0, "absence is held back in the pass that sees a new bad line");
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    assert_eq!(w.content(N2).unwrap(), "the second note is about beta");

    // Core heals it before the next pass: the record is back, the bad line stays.
    w.write(KEY, &format!("{}{GARBAGE_2}\n{}{}", n1(), n3(), n2()));
    let report = w.pass();
    assert_eq!(report.purged, 0);
    assert_eq!(w.notes(), ids(&[N1, N3, N2]), "a healed file loses nothing");

    // Damaged again and left so: one pass longer, then gone.
    w.write(KEY, &format!("{}{GARBAGE_2}\n{GARBAGE_1}\n{}", n1(), n3()));
    assert_eq!(w.pass().purged, 0);
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    assert_eq!(w.pass().purged, 1);
    assert_eq!(w.notes(), ids(&[N1, N3]));
}

#[test]
fn a_tombstone_lands_in_the_very_pass_that_holds_absence_back() {
    let mut w = three_mirrored();
    // One write: a new bad line, note 3 absent, and a tombstone for note 2.
    w.write(KEY, &format!("{}{GARBAGE_1}\n{}{}", n1(), n2(), tomb(N2)));
    let report = w.pass();
    assert_eq!(report.purged, 1);
    assert_eq!(w.notes(), ids(&[N1, N3]), "the tombstone deletes, absence waits");
    assert_eq!(w.pass().purged, 1);
    assert_eq!(w.notes(), ids(&[N1]));
    assert_eq!(w.tombstones(), 0);
}

#[test]
fn a_bad_line_that_goes_away_is_a_change_of_the_set_too() {
    let mut w = world();
    w.write(KEY, &format!("{}{GARBAGE_1}\n{}{}", n1(), n2(), n3()));
    w.pass();
    w.pass();
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    // A rewrite drops the bad line and note 3 together.
    w.write(KEY, &format!("{}{}", n1(), n2()));
    assert_eq!(w.pass().purged, 0);
    assert_eq!(w.nf_text(KEY, "bad_lines"), None);
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    assert_eq!(w.pass().purged, 1);
    assert_eq!(w.notes(), ids(&[N1, N2]));
}

/// An I/O failure never drops rows, and never hides the file from the next pass (wave 4,
/// R-14 item 1): a tombstone appended just before the failed read lands as soon as the file
/// can be read, with its size and time unchanged. Where the test's user reads every file
/// whatever its mode (root), the row a failed read leaves is written by hand.
#[test]
fn a_file_that_could_not_be_opened_is_read_again_and_its_tombstone_lands() {
    use std::os::unix::fs::PermissionsExt;
    let mut w = three_mirrored();
    std::thread::sleep(Duration::from_millis(25));
    fs::write(w.file(KEY), format!("{}{}{}{}", n1(), n2(), n3(), tomb(N2))).unwrap();
    fs::set_permissions(w.file(KEY), fs::Permissions::from_mode(0o000)).unwrap();
    if fs::File::open(w.file(KEY)).is_err() {
        let report = w.pass();
        assert_eq!((report.files_refused, report.files_unchanged, report.purged), (1, 0, 0));
        assert_eq!(w.nf_int(KEY, "ok"), Some(0));
        assert!(w.nf_text(KEY, "reason").is_some());
        assert_eq!(w.nf_int(KEY, "live_count"), Some(3), "the mirror's count is as it was");
        // Still refused: tried again, nothing written.
        let scanned = w.nf_int(KEY, "scanned_ms");
        std::thread::sleep(Duration::from_millis(5));
        let report = w.pass();
        assert_eq!((report.files_refused, report.files_unchanged), (1, 0));
        assert_eq!(w.nf_int(KEY, "scanned_ms"), scanned, "a file that stays refused writes every pass");
    } else {
        use std::os::unix::fs::MetadataExt;
        let meta = fs::metadata(w.file(KEY)).unwrap();
        let path = w.file(KEY).to_string_lossy().into_owned();
        let mtime_ns = meta.mtime() * 1_000_000_000 + meta.mtime_nsec();
        w.conn()
            .execute(
                "UPDATE note_files SET ok = 0, reason = 'cannot open', size = ?1, mtime_ns = ?2 WHERE path = ?3",
                rusqlite::params![meta.len() as i64, mtime_ns, path],
            )
            .unwrap();
    }
    assert_eq!(w.notes(), ids(&[N1, N3, N2]), "an I/O failure never drops rows");

    // Readable again. A change of mode does not change the file's size or time.
    fs::set_permissions(w.file(KEY), fs::Permissions::from_mode(0o644)).unwrap();
    let report = w.pass();
    assert_eq!((report.files_read, report.files_unchanged, report.purged), (1, 0, 1));
    assert_eq!(w.notes(), ids(&[N1, N3]), "the note the owner forgot is gone");
    assert_eq!(w.nf_int(KEY, "ok"), Some(1));
    assert_eq!(w.nf_text(KEY, "reason"), None);
    assert_eq!(w.pass().files_unchanged, 1);
}

#[test]
fn a_link_or_a_folder_under_a_notes_name_is_not_read_and_removes_nothing() {
    let mut w = three_mirrored();
    let elsewhere = w.base.join("elsewhere.jsonl");
    fs::write(&elsewhere, n1()).unwrap();
    fs::remove_file(w.file(KEY)).unwrap();
    std::os::unix::fs::symlink(&elsewhere, w.file(KEY)).unwrap();
    let report = w.pass();
    assert_eq!((report.files_read, report.files_gone, report.purged), (0, 0, 0));
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));

    // Names that are not a notes file are not looked at.
    fs::write(w.base.join("memory").join("project-p111111111111111g.jsonl"), n1()).unwrap();
    fs::write(w.base.join("memory").join("project-P1111111111111111.jsonl"), n1()).unwrap();
    fs::write(w.base.join("memory").join("project-p1111111111111111.jsonl.bak"), n1()).unwrap();
    fs::write(w.base.join("memory").join("global.jsonl"), n1()).unwrap();
    let report = w.pass();
    assert_eq!(report.files_read, 0);
    assert_eq!(w.nf_rows(), 1);
}

#[test]
fn a_pass_does_nothing_without_a_connection_or_when_the_mode_is_not_ready() {
    let mut w = world();
    w.write(KEY, &format!("{}{}", n1(), n2()));
    for mode in [Mode::Starting, Mode::Degraded] {
        w.ctx.mode.store(mode as u8, Ordering::SeqCst);
        assert_eq!(w.pass(), PassReport::default());
        assert_eq!(w.notes(), ids(&[]));
        assert_eq!(w.nf_rows(), 0);
    }
    w.ctx.mode.store(Mode::Ready as u8, Ordering::SeqCst);
    let conn = w.ctx.conn.take();
    assert!(index_notes::pass(&mut w.ctx).is_ok());
    assert!(index_notes::on_forgotten(&mut w.ctx, N1, KEY).is_ok());
    w.ctx.conn = conn;
    assert_eq!(w.notes(), ids(&[]));
    assert!(index_notes::pass(&mut w.ctx).is_ok());
    assert_eq!(w.notes(), ids(&[N1, N2]));
}

#[test]
fn on_forgotten_purges_the_note_of_that_project_and_nothing_else() {
    let mut w = three_mirrored();
    w.put_tool_row("mem-tool", "what axel was told to remember");
    let tool = w.saved("mem-tool");

    // Another project's key, an id that is no row, and a row of another origin: nothing.
    index_notes::on_forgotten(&mut w.ctx, N2, KEY_B).unwrap();
    index_notes::on_forgotten(&mut w.ctx, "mem-nobody", KEY).unwrap();
    index_notes::on_forgotten(&mut w.ctx, "mem-tool", KEY).unwrap();
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    assert_eq!(w.saved("mem-tool"), tool);
    assert_eq!(w.purge_pending(), 0);
    assert!(w.conn().is_autocommit());

    // The event's key, as it is.
    index_notes::on_forgotten(&mut w.ctx, N2, KEY).unwrap();
    assert!(w.conn().is_autocommit());
    assert_eq!(w.notes(), ids(&[N1, N3]));
    assert_eq!(w.content(N2), None);
    assert_eq!(w.count("SELECT count(*) FROM memories_fts WHERE memories_fts MATCH 'beta'"), 0);
    assert_eq!(w.tombstones(), 0);
    assert_eq!(w.purge_pending(), 1);
    assert_eq!(w.saved("mem-tool"), tool);

    // Core has written its tombstone line by the time the pass reads the file again.
    w.write(KEY, &format!("{}{}{}{}", n1(), n2(), n3(), tomb(N2)));
    w.pass();
    assert_eq!(w.notes(), ids(&[N1, N3]));
}

#[test]
fn a_record_with_a_time_no_row_can_hold_is_skipped_and_the_rest_of_the_file_is_mirrored() {
    let mut w = world();
    let line = format!(
        "{{\"namespace\":\"project-{KEY}\",\"timestamp_ms\":18446744073709551615,\"content\":\"far future\",\"tags\":[],\"id\":\"mem-far\",\"project\":\"{KEY}\"}}\n"
    );
    w.write(KEY, &format!("{}{line}{}", n1(), n2()));
    let report = w.pass();
    assert_eq!(report.inserted, 2);
    assert_eq!(w.notes(), ids(&[N1, N2]));
    assert_eq!(w.nf_text(KEY, "reason").unwrap(), "ids_skipped: 1");
    assert_eq!(w.nf_int(KEY, "ok"), Some(1));
}

// ── Wave 4, R-14 item 2: a note rewritten under the same id ─────────────────

/// One record line with every field a note can change.
fn rec_full(id: &str, ts: i64, content: &str, tags: &[&str], sensitivity: Option<&str>, days: Option<u32>) -> String {
    let mut value = json!({
        "namespace": format!("project-{KEY}"), "timestamp_ms": ts, "content": content,
        "tags": tags, "id": id, "project": KEY
    });
    if let Some(s) = sensitivity {
        value["sensitivity"] = json!(s);
    }
    if let Some(days) = days {
        value["retention"] = json!({ "max_age_days": days });
    }
    format!("{value}\n")
}

fn fts(w: &World, word: &str) -> i64 {
    w.count(&format!("SELECT count(*) FROM memories_fts WHERE memories_fts MATCH '{word}'"))
}

/// Guards what the owner would see: he edits a note to take something out, and the model
/// goes on being shown the old text.
#[test]
fn a_note_rewritten_under_the_same_id_loses_its_old_text_and_keeps_its_hide() {
    let mut w = three_mirrored();
    w.conn()
        .execute("UPDATE memory_ext SET hidden_ms = 1234, hidden_by = 'cli' WHERE mem_id = ?1", [N2])
        .unwrap();
    let (one, three) = (w.saved(N1), w.saved(N3));
    let vid_one = w.count(&format!("SELECT vid FROM memory_ext WHERE mem_id = '{N1}'"));
    assert_eq!(fts(&w, "beta"), 1);

    w.write(KEY, &format!("{}{}{}", n1(), rec(N2, 1_700_000_000_100, "the second note is about delta"), n3()));
    let report = w.pass();
    assert_eq!((report.rewritten, report.purged, report.inserted), (1, 0, 0));
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
    assert_eq!(w.content(N2).unwrap(), "the second note is about delta");
    assert_eq!(w.text("SELECT title FROM memories WHERE id = ?1", N2).unwrap(), "the second note is about delta");
    assert_eq!((fts(&w, "beta"), fts(&w, "delta")), (0, 1), "the word index holds the old text");
    assert_eq!(w.count("SELECT count(*) FROM memories WHERE content LIKE '%beta%'"), 0);
    assert_eq!(w.purge_pending(), 1, "old text was forgotten: the log is to be truncated");
    assert_eq!(w.tombstones(), 0);
    // An edit never unhides.
    assert_eq!(w.count(&format!("SELECT hidden_ms FROM memory_ext WHERE mem_id = '{N2}'")), 1234);
    assert_eq!(w.text("SELECT hidden_by FROM memory_ext WHERE mem_id = ?1", N2).unwrap(), "cli");
    // The notes that did not change are the rows they were.
    assert_eq!((w.saved(N1), w.saved(N3)), (one, three));
    assert_eq!(w.count(&format!("SELECT vid FROM memory_ext WHERE mem_id = '{N1}'")), vid_one);
    assert_eq!(w.nf_int(KEY, "live_count"), Some(3));
    assert_eq!(w.nf_text(KEY, "reason"), None);

    // The same file with a new time: read, and nothing is rewritten.
    w.write(KEY, &format!("{}{}{}", n1(), rec(N2, 1_700_000_000_100, "the second note is about delta"), n3()));
    let report = w.pass();
    assert_eq!((report.files_read, report.rewritten, report.purged, report.inserted), (1, 0, 0, 0));
    assert_eq!(w.count(&format!("SELECT hidden_ms FROM memory_ext WHERE mem_id = '{N2}'")), 1234);
}

/// Guards what the owner would lose: he makes a note secret, and its body stays in the
/// word index.
#[test]
fn a_note_made_secret_under_the_same_id_leaves_the_word_index() {
    let mut w = three_mirrored();
    assert_eq!(fts(&w, "alpha"), 1);
    let secret = rec_full(N1, 1_700_000_000_000, "the first note is about alpha", &[], Some("secret"), None);
    w.write(KEY, &format!("{secret}{}{}", n2(), n3()));
    let report = w.pass();
    assert_eq!((report.rewritten, report.purged, report.inserted), (1, 0, 0));
    assert_eq!(w.text("SELECT sensitivity FROM memories WHERE id = ?1", N1).unwrap(), "secret");
    assert_eq!(w.content(N1).unwrap(), "the first note is about alpha");
    assert_eq!(fts(&w, "alpha"), 0, "a secret body is in the word index");
    assert_eq!(w.text("SELECT title FROM memories WHERE id = ?1", N1).unwrap(), "Secret note");
    assert_eq!(w.count(&format!("SELECT count(*) FROM memory_ext WHERE mem_id = '{N1}' AND embedding IS NULL")), 1);
    assert_eq!(w.purge_pending(), 1);
}

#[test]
fn a_change_of_tags_or_of_expiry_rewrites_the_row_too() {
    let mut w = three_mirrored();
    let text = "the first note is about alpha";
    let expiry = |w: &World| w.text("SELECT expires_at FROM memories WHERE id = ?1", N1);

    w.write(KEY, &format!("{}{}{}", rec_full(N1, 1_700_000_000_000, text, &["kept"], None, None), n2(), n3()));
    assert_eq!(w.pass().rewritten, 1);
    assert_eq!(w.text("SELECT tags FROM memories WHERE id = ?1", N1).unwrap(), r#"["kept"]"#);
    assert_eq!(expiry(&w), None);

    w.write(KEY, &format!("{}{}{}", rec_full(N1, 1_700_000_000_000, text, &["kept"], None, Some(7)), n2(), n3()));
    assert_eq!(w.pass().rewritten, 1);
    let expires_ms = chrono::DateTime::parse_from_rfc3339(&expiry(&w).unwrap()).unwrap().timestamp_millis();
    assert_eq!(expires_ms, 1_700_000_000_000 + 7 * 86_400_000);

    // `sensitive` is a tag: a change too. And back to the same record: nothing.
    w.write(KEY, &format!("{}{}{}", rec_full(N1, 1_700_000_000_000, text, &["kept"], Some("sensitive"), Some(7)), n2(), n3()));
    assert_eq!(w.pass().rewritten, 1);
    assert_eq!(w.text("SELECT tags FROM memories WHERE id = ?1", N1).unwrap(), r#"["kept","sensitive"]"#);
    w.write(KEY, &format!("{}{}{}", rec_full(N1, 1_700_000_000_000, text, &["kept"], Some("sensitive"), Some(7)), n2(), n3()));
    let report = w.pass();
    assert_eq!((report.files_read, report.rewritten), (1, 0));
    assert_eq!(w.notes(), ids(&[N1, N3, N2]));
}

// ── Wave 4, R-14 items 3 and 5: one file that fails ─────────────────────────

/// A rollback half way, and the pass goes on: the first file's transaction purges a row
/// and then fails on an insert. None of it is committed, the file's `note_files` row is as
/// it was, the next file is mirrored in the same pass, and the pass returns no error.
#[test]
fn a_file_whose_transaction_fails_half_way_is_rolled_back_and_the_next_file_is_still_read() {
    let mut w = three_mirrored();
    let (size, mtime) = (w.nf_int(KEY, "size"), w.nf_int(KEY, "mtime_ns"));
    w.conn()
        .execute_batch(
            "CREATE TRIGGER test_abort BEFORE INSERT ON memory_ext WHEN NEW.mem_id = 'mem-boom'
             BEGIN SELECT RAISE(ABORT, 'test: this note is refused'); END;",
        )
        .unwrap();
    // The purge of note 2 comes first in the transaction, then the insert that fails.
    let failing = format!("{}{}{}{}{}", n1(), n2(), n3(), tomb(N2), rec("mem-boom", 9, "the note that fails"));
    w.write(KEY, &failing);
    w.write(KEY_B, &rec_of(KEY_B, "mem-next", 9, "a note of the next file"));

    let report = w.pass();
    assert_eq!((report.files_failed, report.files_read, report.purged, report.inserted), (1, 1, 0, 1));
    assert!(report.listed, "the step for files that are gone still ran");
    assert_eq!(w.notes(), ids(&["mem-next", N1, N3, N2]), "half a transaction was committed");
    assert_eq!(w.content(N2).unwrap(), "the second note is about beta");
    assert_eq!(fts(&w, "beta"), 1);
    assert_eq!(w.purge_pending(), 0);
    assert_eq!((w.nf_int(KEY, "size"), w.nf_int(KEY, "mtime_ns")), (size, mtime), "the file is read again");
    assert_eq!(w.nf_int(KEY_B, "ok"), Some(1));
    assert_eq!(w.ctx.status.write_errors.load(Ordering::Relaxed), 1);
    assert_eq!(w.ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8);

    // It fails again at the next pass and is counted again; nothing else is touched.
    let report = w.pass();
    assert_eq!((report.files_failed, report.files_unchanged), (1, 1));
    assert_eq!(w.notes(), ids(&["mem-next", N1, N3, N2]));

    // Once it can be written, the forget lands.
    w.conn().execute_batch("DROP TRIGGER test_abort").unwrap();
    let report = w.pass();
    assert_eq!((report.files_failed, report.purged, report.inserted), (0, 1, 1));
    assert_eq!(w.notes(), ids(&["mem-boom", "mem-next", N1, N3]));
    assert_eq!(w.purge_pending(), 1);
}

/// A panic inside one file's transaction (here: the clock, which is read inside it) is
/// caught, rolled back and counted, and the next file is read.
#[test]
fn a_panic_in_one_file_is_rolled_back_and_counted_and_the_pass_goes_on() {
    let mut w = three_mirrored();
    let armed = Arc::new(AtomicBool::new(true));
    let flag = armed.clone();
    w.ctx.clock = Arc::new(move || {
        if flag.swap(false, Ordering::SeqCst) {
            panic!("test: the clock panics once");
        }
        chrono::Utc::now().timestamp_millis()
    });
    w.write(KEY, &format!("{}{}{}{}", n1(), n2(), n3(), tomb(N2)));
    w.write(KEY_B, &rec_of(KEY_B, "mem-next", 9, "a note of the next file"));

    let report = w.pass();
    assert_eq!((report.files_failed, report.files_read, report.purged, report.inserted), (1, 1, 0, 1));
    assert_eq!(w.ctx.status.panics.load(Ordering::Relaxed), 1);
    assert_eq!(w.ctx.status.write_errors.load(Ordering::Relaxed), 0);
    assert_eq!(w.notes(), ids(&["mem-next", N1, N3, N2]));
    assert!(!armed.load(Ordering::SeqCst));

    let report = w.pass();
    assert_eq!((report.files_failed, report.purged), (0, 1));
    assert_eq!(w.notes(), ids(&["mem-next", N1, N3]));
}

/// The step for files that are gone is isolated file by file too.
#[test]
fn a_gone_file_whose_purge_fails_does_not_stop_the_next_gone_file() {
    let mut w = three_mirrored();
    w.write(KEY_B, &rec_of(KEY_B, "mem-next", 9, "a note of the next file"));
    w.pass();
    w.conn()
        .execute_batch(&format!(
            "CREATE TRIGGER test_abort BEFORE DELETE ON note_files WHEN OLD.project_key = '{KEY}'
             BEGIN SELECT RAISE(ABORT, 'test: this row stays'); END;"
        ))
        .unwrap();
    fs::remove_file(w.file(KEY)).unwrap();
    fs::remove_file(w.file(KEY_B)).unwrap();

    let report = w.pass();
    assert_eq!((report.files_failed, report.files_gone, report.purged), (1, 1, 1));
    assert_eq!(w.notes(), ids(&[N1, N3, N2]), "the failed file's purge was rolled back whole");
    assert_eq!(w.nf_rows(), 1);
    assert_eq!(w.ctx.status.write_errors.load(Ordering::Relaxed), 1);

    w.conn().execute_batch("DROP TRIGGER test_abort").unwrap();
    let report = w.pass();
    assert_eq!((report.files_failed, report.files_gone, report.purged), (0, 1, 3));
    assert_eq!(w.notes(), ids(&[]));
    assert_eq!(w.nf_rows(), 0);
}
