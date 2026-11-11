//! Commit 13a: archives indexed, first half (spec section 2: tests 2, 7 to 9, 11 to 18,
//! 33 (c), 34 (a), (b), (c), (e), and the credential test).
//!
//! Every test has a temp base, a temp brain and a fixed clock; none touches the network or
//! a model file. Where a test of the spec names a forget mark or a twin, that clause is
//! commit 13b's.

use std::io::{BufReader, Read, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axel::brain_v2::{self, Role};
use axel::ext::config::ExtConfig;
use axel::ext::rpc::{write_frame, FrameReader};
use axel::ext::server::serve;
use axel::ext::types::{Boot, IndexerCtx, Job, Mode, ServeEnd, Status, TestHooks};
use axel::project_memory::{RecordLinks, Sensitivity};
use axel::store::{self, NewRow};
use axel::vindex::VecDelta;
use axel::{index_archives, indexer, indexer_start, project};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use tempfile::TempDir;
use velocirag::embedder::Embedder;

const KEY: &str = "p0123456789abcdef";
const KEY_2: &str = "pfedcba9876543210";
/// The time of the first archive of a test.
const T0: i64 = 1_700_000_000_000;
/// What the test clock says: after every archive of a test, before the real time.
const NOW: i64 = 1_750_000_000_000;
const SHELF: &str = "s000001";

// ── A base, a brain, an indexer ─────────────────────────────────────────────

struct Env {
    tmp: TempDir,
}

struct Run {
    ctx: IndexerCtx,
    jobs_rx: Receiver<Job>,
    _vec_rx: Receiver<VecDelta>,
}

impl Env {
    fn new() -> Env {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("base").join("context-archives")).unwrap();
        std::fs::create_dir_all(tmp.path().join("model")).unwrap();
        Env { tmp }
    }
    fn brain(&self) -> PathBuf {
        self.tmp.path().join("brain").join("brain.r8")
    }
    fn wal(&self) -> PathBuf {
        self.tmp.path().join("brain").join("brain.r8-wal")
    }
    fn archives(&self) -> PathBuf {
        self.tmp.path().join("base").join("context-archives")
    }
    fn config(&self, embedder: &str) -> Value {
        json!({
            "brain": self.brain().to_string_lossy(),
            "synaps_base": self.tmp.path().join("base").to_string_lossy(),
            "model_dir": self.tmp.path().join("model").to_string_lossy(),
            "embedder": embedder,
        })
    }
    fn cfg(&self, embedder: &str) -> ExtConfig {
        ExtConfig::from_initialize(&json!({"config": self.config(embedder)}), self.tmp.path())
    }
    fn run(&self) -> Run {
        self.run_with(TestHooks::default(), None)
    }
    fn run_with(&self, test: TestHooks, embedder: Option<Embedder>) -> Run {
        let kind = if embedder.is_some() { "hash" } else { "off" };
        let boot = Boot {
            embedder,
            clock: Arc::new(|| NOW),
            test: Arc::new(test),
            status: Arc::new(Status::default()),
        };
        let (ctx, jobs_rx, vec_rx) = indexer_start::open_ctx(self.cfg(kind), boot);
        assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8, "the brain did not open");
        Run { ctx, jobs_rx, _vec_rx: vec_rx }
    }
    /// Writes one archive file; the directories are made on the way.
    fn put(&self, scope: &str, shelf: &str, id: &str, text: &str) -> PathBuf {
        let dir = self.archives().join(scope).join(shelf);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("{id}.json"));
        std::fs::write(&path, text).unwrap();
        path
    }
    fn put_live(&self, scope: &str, a: &Arch) -> PathBuf {
        self.put(scope, SHELF, &a.id, &a.json(scope))
    }
    fn put_tombstone(&self, scope: &str, a: &Arch) -> PathBuf {
        self.put(scope, SHELF, &a.id, &a.tombstone(scope))
    }
}

impl Run {
    fn db(&self) -> &Connection {
        self.ctx.conn.as_ref().unwrap()
    }
    fn scan(&mut self) {
        index_archives::scan(&mut self.ctx).unwrap();
    }
    fn job(&mut self, job: Job) {
        indexer::dispatch(&mut self.ctx, job);
    }
    /// What `indexer::run` does when no job arrived for a second.
    fn idle_turn(&mut self) {
        indexer::guarded(&mut self.ctx, index_archives::checkpoint_after_purge);
    }
    fn errors(&self) -> u64 {
        self.ctx.status.write_errors.load(Ordering::Relaxed)
    }
    fn panics(&self) -> u64 {
        self.ctx.status.panics.load(Ordering::Relaxed)
    }
}

fn scope(key: &str) -> String {
    project::project_dir(key)
}

fn rel(id: &str) -> String {
    format!("{SHELF}/{id}.json")
}

// ── An archive file ─────────────────────────────────────────────────────────

#[derive(Clone)]
struct Arch {
    id: String,
    logical: String,
    digest: String,
    created_ms: Option<i64>,
    carried: Option<u32>,
    first_index: u32,
    turns: Vec<(String, String)>,
    note: String,
}

impl Arch {
    /// Archive number `n` of the chat `session`, with no message yet.
    fn new(n: u32, session: &str) -> Arch {
        Arch {
            id: format!("{n:032x}"),
            logical: project::logical(session),
            digest: format!("{n:064x}"),
            created_ms: Some(T0),
            carried: None,
            first_index: 0,
            turns: Vec::new(),
            note: String::new(),
        }
    }
    fn at(mut self, created_ms: i64) -> Arch {
        self.created_ms = Some(created_ms);
        self
    }
    fn turn(mut self, prompt: &str, reply: &str) -> Arch {
        self.turns.push((prompt.to_string(), reply.to_string()));
        self
    }
    /// One exchange about `topic`, long enough to be a unit.
    fn about(self, topic: &str) -> Arch {
        self.turn(
            &format!("what did we decide about the {topic} when we talked it through?"),
            &format!("we decided that the {topic} stays as it is until the next release is out"),
        )
    }
    fn json(&self, scope: &str) -> String {
        let mut messages = Vec::new();
        for (i, (prompt, reply)) in self.turns.iter().enumerate() {
            let at = self.first_index + 2 * i as u32;
            messages.push(json!({"source_index": at, "block_indices": [],
                                 "message": {"role": "user", "content": prompt}}));
            messages.push(json!({"source_index": at + 1, "block_indices": [],
                                 "message": {"role": "assistant", "content": reply}}));
        }
        let mut record = json!({
            "version": 1, "scope": scope, "logical": self.logical, "id": self.id, "digest": self.digest,
            "body": {"state": "live",
                     "source_message_count": self.first_index + 2 * self.turns.len() as u32,
                     "messages": messages, "note": self.note},
        });
        if let Some(ms) = self.created_ms {
            record["created_ms"] = json!(ms);
        }
        if let Some(n) = self.carried {
            record["carried"] = json!(n);
        }
        record.to_string()
    }
    /// What core leaves of a forgotten archive: the same header, no body.
    fn tombstone(&self, scope: &str) -> String {
        json!({"version": 1, "scope": scope, "logical": self.logical, "id": self.id,
               "digest": self.digest, "body": {"state": "tombstone"}})
        .to_string()
    }
}

fn archived(a: &Arch, key: &str, path: Option<&Path>, session: Option<&str>, window: Option<i64>, part: Option<i64>) -> Job {
    Job::Archive {
        archive_id: a.id.clone(),
        path: path.map(|p| p.to_string_lossy().into_owned()),
        project_key: key.to_string(),
        cwd: None,
        session_id: session.map(str::to_string),
        received_ms: NOW - 1_000,
        window,
        part,
    }
}

fn forgotten(a: &Arch, key: &str) -> Job {
    Job::Forgotten { id: a.id.clone(), kind: "archive".to_string(), project_key: key.to_string() }
}

// ── What the brain holds ────────────────────────────────────────────────────

fn tx<T>(conn: &Connection, work: impl FnOnce() -> T) -> T {
    brain_v2::begin_write(conn).unwrap();
    let out = work();
    conn.execute_batch("COMMIT").unwrap();
    out
}

fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |r| r.get(0)).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// The rows of one archive of one directory, in the order they were written:
/// `(id, unit_pos, content)`.
fn rows(conn: &Connection, scope: &str, a: &Arch) -> Vec<(String, String, String)> {
    let mut stmt = conn
        .prepare(
            "SELECT e.mem_id, e.unit_pos, m.content FROM memory_ext e JOIN memories m ON m.id = e.mem_id
              WHERE e.origin_kind = 'archive' AND e.origin_ref = ?1 AND m.project_key = ?2 ORDER BY e.vid",
        )
        .unwrap();
    let got = stmt.query_map(params![a.id, scope], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).unwrap();
    got.collect::<rusqlite::Result<_>>().unwrap()
}

fn positions(conn: &Connection, scope: &str, a: &Arch) -> Vec<String> {
    rows(conn, scope, a).into_iter().map(|r| r.1).collect()
}

/// `(state, skip_reason)` of a file's `archive_index` row.
fn state(conn: &Connection, scope: &str, rel_name: &str) -> Option<(String, Option<String>)> {
    conn.query_row(
        "SELECT state, skip_reason FROM archive_index WHERE scope_dir = ?1 AND rel_name = ?2",
        params![scope, rel_name],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )
    .ok()
}

fn state_of(conn: &Connection, scope: &str, a: &Arch) -> String {
    state(conn, scope, &rel(&a.id)).map(|s| s.0).unwrap_or_else(|| "no row".to_string())
}

fn index_rows(conn: &Connection) -> i64 {
    count(conn, "SELECT COUNT(*) FROM archive_index")
}

fn archive_memories(conn: &Connection) -> i64 {
    count(conn, "SELECT COUNT(*) FROM memory_ext WHERE origin_kind = 'archive'")
}

fn has(conn: &Connection, id: &str) -> bool {
    conn.query_row("SELECT COUNT(*) FROM memories WHERE id = ?1", [id], |r| r.get::<_, i64>(0)).unwrap() == 1
}

/// A captured turn of `session` in `scope`, written the way capture writes one.
fn capture(run: &Run, id: &str, scope: &str, session: &str, created_ms: i64, hidden_by: Option<&str>) {
    let logical = project::logical(session);
    let new = NewRow {
        id: id.to_string(),
        class: "episodic_turn",
        project_key: scope,
        title: format!("capture {id}"),
        abstract_text: String::new(),
        content: format!("User: a captured prompt of {id}\nAssistant: and the reply that was captured with it"),
        tags: Vec::new(),
        created_ms,
        time_source: "exact",
        sensitivity: Sensitivity::Normal,
        provenance: "test".to_string(),
        expires_ms: None,
        importance: 0.5,
        origin_kind: "capture",
        origin_ref: session,
        unit_pos: None,
        session_id: Some(session),
        logical: Some(&logical),
        links: RecordLinks::default(),
        turn_key: Some(format!("{:016x}", created_ms)),
        hidden: hidden_by.map(|by| (created_ms, by.to_string())),
    };
    tx(run.db(), || store::insert_row(run.db(), run.ctx.signer.as_ref(), &new).unwrap());
    assert!(has(run.db(), id));
}

fn purge_pending(conn: &Connection) -> bool {
    count(conn, "SELECT COUNT(*) FROM context_data WHERE key = 'purge_pending' AND layer = 'axel'") == 1
}

/// Rows of any table that keeps text which hold the marker.
fn in_tables(conn: &Connection, marker: &str) -> i64 {
    conn.query_row(
        "SELECT (SELECT COUNT(*) FROM memories WHERE title LIKE ?1 OR abstract_text LIKE ?1 OR content LIKE ?1)
              + (SELECT COUNT(*) FROM memories_fts WHERE title LIKE ?1 OR abstract LIKE ?1 OR content LIKE ?1)
              + (SELECT COUNT(*) FROM recall_log WHERE query_preview LIKE ?1 OR detail LIKE ?1)",
        [format!("%{marker}%")],
        |r| r.get(0),
    )
    .unwrap()
}

/// Hits of the marker in a file's bytes; a missing file has none. The tail of the marker
/// is counted too: the word index stores a term without the prefix it shares with its
/// neighbour.
fn in_file(path: &Path, marker: &str) -> usize {
    let bytes = std::fs::read(path).unwrap_or_default();
    let hits = |needle: &str| bytes.windows(needle.len()).filter(|w| *w == needle.as_bytes()).count();
    hits(marker) + hits(&marker[6..])
}

fn wait(what: &str, mut done: impl FnMut() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < Duration::from_secs(30), "waited 30 s for: {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// `(path, size, mode, mtime)` of every entry under a folder.
fn tree(root: &Path) -> Vec<(PathBuf, u64, u32, i64, i64)> {
    let mut out = Vec::new();
    let mut todo = vec![root.to_path_buf()];
    while let Some(dir) = todo.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let meta = std::fs::symlink_metadata(&path).unwrap();
            out.push((path.clone(), meta.len(), meta.mode(), meta.mtime(), meta.mtime_nsec()));
            if meta.file_type().is_dir() {
                todo.push(path);
            }
        }
    }
    out.sort();
    out
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

// ── Test 2: the must-skip directory ─────────────────────────────────────────

#[test]
fn t02_a_directory_of_things_to_skip_is_scanned_to_the_end_and_the_good_files_are_indexed() {
    let env = Env::new();
    let dir = scope(KEY);
    let good_1 = Arch::new(1, "s-two").about("retry limit");
    let good_2 = Arch::new(2, "s-two").at(T0 + 60_000).about("upload queue");
    let good_1_path = env.put_live(&dir, &good_1);
    env.put(&dir, "s000002", &good_2.id, &good_2.json(&dir));

    // Listed, and refused by the reader with a reason.
    let link = Arch::new(10, "s-two");
    std::os::unix::fs::symlink(&good_1_path, env.archives().join(&dir).join(SHELF).join(format!("{}.json", link.id)))
        .unwrap();
    let big = Arch::new(11, "s-two");
    let big_path = env.put(&dir, SHELF, &big.id, "");
    std::fs::File::options().write(true).open(&big_path).unwrap().set_len(16 * 1024 * 1024 + 1).unwrap();
    let version_2 = Arch::new(12, "s-two").about("version two");
    env.put(&dir, SHELF, &version_2.id, &version_2.json(&dir).replacen("\"version\":1", "\"version\":2", 1));
    let other_scope = Arch::new(13, "s-two").about("another project");
    env.put(&dir, SHELF, &other_scope.id, &other_scope.json(&scope(KEY_2)));
    let other_id = Arch::new(14, "s-two").about("another id");
    env.put(&dir, SHELF, &Arch::new(15, "s-two").id, &other_id.json(&dir));
    let truncated = Arch::new(16, "s-two").about("a torn file");
    let whole = truncated.json(&dir);
    env.put(&dir, SHELF, &truncated.id, &whole[..whole.len() / 2]);
    let unknown_state = Arch::new(17, "s-two").about("an unknown state");
    env.put(&dir, SHELF, &unknown_state.id, &unknown_state.json(&dir).replacen("\"live\"", "\"frozen\"", 1));
    assert!(version_2.json(&dir).contains("\"version\":1") && unknown_state.json(&dir).contains("\"live\""));

    // Not archive names: never opened, and no row.
    let stray = Arch::new(20, "s-two").about("a stray file");
    std::fs::write(env.archives().join(&dir).join("128.json"), stray.json(&dir)).unwrap();
    for shelf in ["s1", "s000000", "s1000000"] {
        env.put(&dir, shelf, &stray.id, &stray.json(&dir));
    }
    std::fs::write(env.archives().join(&dir).join(SHELF).join("pending.0123456789abcdef.tmp"), stray.json(&dir))
        .unwrap();
    std::fs::write(env.archives().join(&dir).join("lock"), "").unwrap();

    let mut run = env.run();
    run.scan();
    assert_eq!(run.errors(), 0);
    assert_eq!(run.ctx.status.scan.load(Ordering::Relaxed), 2, "the scan did not finish");

    let conn = run.db();
    assert_eq!(state_of(conn, &dir, &good_1), "indexed");
    assert_eq!(state(conn, &dir, &format!("s000002/{}.json", good_2.id)).unwrap().0, "indexed");
    assert_eq!(positions(conn, &dir, &good_1), ["x0.0"]);
    assert_eq!(positions(conn, &dir, &good_2), ["x0.0"]);
    for (bad, why) in [
        (&link.id, "not a regular file"),
        (&big.id, "larger than 16 MiB"),
        (&version_2.id, "version 2 is not 1"),
        (&other_scope.id, "a record of another project"),
        (&Arch::new(15, "s-two").id, "the id is not the file's name"),
        (&truncated.id, "not valid JSON"),
        (&unknown_state.id, "unknown state"),
    ] {
        let got = state(conn, &dir, &rel(bad)).unwrap_or_else(|| panic!("no row for {why}"));
        assert_eq!(got, ("skipped".to_string(), Some(why.to_string())));
    }
    assert_eq!(index_rows(conn), 9, "a row for a name that is not an archive");
    assert_eq!(archive_memories(conn), 2);

    // The second scan reads none of them again.
    run.scan();
    assert_eq!(index_rows(run.db()), 9);
    assert_eq!(archive_memories(run.db()), 2);
    assert_eq!(run.errors(), 0);
}

// ── Test 7: twice, and a transaction that did not finish ────────────────────

#[test]
fn t07_indexing_the_fixture_twice_inserts_nothing_the_second_time() {
    let env = Env::new();
    let fixture = "01a8f6fb3ec12e1d59d07bd6e3943c84c18415e3111dd5cba218b7ce9218b0c7";
    copy_tree(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/context-archives").join(fixture),
        &env.archives().join(fixture),
    );
    let mut run = env.run();
    run.scan();
    let conn = run.db();
    let first = archive_memories(conn);
    assert!(first >= 3, "the fixture gave {first} rows");
    assert_eq!(count(conn, "SELECT SUM(rows_indexed) FROM archive_index"), first);
    assert_eq!(index_rows(conn), 5);
    assert_eq!(
        state(conn, fixture, "s000001/693c14d38641fb504a642f0e7cbfaac9.json").unwrap().0,
        "tombstone"
    );
    // The slot file has no `created_ms`: its rows say so.
    assert_eq!(
        count(conn, "SELECT COUNT(*) FROM archive_index WHERE rel_name = '000.json' AND time_source = 'mtime'"),
        1
    );
    let newest = count(conn, "SELECT MAX(vid) FROM memory_ext");

    // A second scan opens nothing; with the rows of the files gone it reads them all
    // again, and every unit is already there.
    run.scan();
    assert_eq!(archive_memories(run.db()), first);
    tx(run.db(), || run.db().execute("DELETE FROM archive_index WHERE state = 'indexed'", []).unwrap());
    run.scan();
    let conn = run.db();
    assert_eq!(archive_memories(conn), first);
    assert_eq!(count(conn, "SELECT MAX(vid) FROM memory_ext"), newest, "a row was written twice");
    assert_eq!(count(conn, "SELECT SUM(rows_indexed) FROM archive_index"), first);
    assert_eq!(run.errors(), 0);
}

/// Aborts the insert of the unit at `pos` of one archive.
fn abort_unit(conn: &Connection, a: &Arch, pos: &str) {
    conn.execute_batch(&format!(
        "CREATE TRIGGER test_abort BEFORE INSERT ON memory_ext
          WHEN NEW.origin_ref = '{}' AND NEW.unit_pos = '{pos}'
          BEGIN SELECT RAISE(ABORT, 'test: this unit is refused'); END;",
        a.id
    ))
    .unwrap();
}

#[test]
fn t07_t34e_an_error_rolls_the_archive_back_and_the_next_scan_completes_it() {
    let env = Env::new();
    let dir = scope(KEY);
    let torn = Arch::new(1, "s-torn").about("first thing").about("second thing");
    let next = Arch::new(2, "s-torn").at(T0 + 60_000).about("third thing");
    env.put_live(&dir, &torn);
    env.put_live(&dir, &next);
    let mut run = env.run();
    abort_unit(run.db(), &torn, "x2.0");

    run.scan();
    let conn = run.db();
    assert!(conn.is_autocommit(), "the failed transaction is still open");
    assert!(rows(conn, &dir, &torn).is_empty(), "half an archive was committed");
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memories"), 1, "only the next archive's row");
    assert_eq!(state(conn, &dir, &rel(&torn.id)), None);
    assert_eq!(positions(conn, &dir, &next), ["x0.0"], "the scan stopped at the error");
    assert_eq!(state_of(conn, &dir, &next), "indexed");
    assert_eq!(run.errors(), 1);
    assert_eq!(run.ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8);

    // Test 7: the half-done one has no rows, and the next scan completes it.
    run.db().execute_batch("DROP TRIGGER test_abort").unwrap();
    run.scan();
    assert_eq!(positions(run.db(), &dir, &torn), ["x0.0", "x2.0"]);
    assert_eq!(state_of(run.db(), &dir, &torn), "indexed");
    assert_eq!(run.errors(), 1);
}

// ── Test 8: the reader never writes ─────────────────────────────────────────

#[test]
fn t08_a_scan_and_a_forget_change_nothing_under_context_archives() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = Arch::new(1, "s-eight").about("retry limit");
    let gone = Arch::new(2, "s-eight").at(T0 + 60_000).about("upload queue");
    env.put_live(&dir, &a);
    env.put_live(&dir, &gone);
    env.put_tombstone(&dir, &Arch::new(3, "s-eight"));
    std::fs::write(env.archives().join(&dir).join("lock"), "").unwrap();
    let before = tree(&env.archives());

    let mut run = env.run();
    run.scan();
    assert_eq!(positions(run.db(), &dir, &gone), ["x0.0"]);
    run.job(forgotten(&gone, KEY));
    assert!(rows(run.db(), &dir, &gone).is_empty());
    run.idle_turn();
    run.scan();
    assert_eq!(tree(&env.archives()), before);
    assert_eq!(run.errors(), 0);
}

#[test]
fn t08_the_scan_succeeds_with_the_tree_read_only() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = Arch::new(1, "s-eight").about("retry limit");
    env.put_live(&dir, &a);
    let entries = tree(&env.archives());
    let set = |write: bool| {
        for (path, _, mode, _, _) in &entries {
            let mode = if write { *mode } else { *mode & !0o222 };
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
        }
    };
    set(false);
    let mut run = env.run();
    run.scan();
    let indexed = positions(run.db(), &dir, &a);
    let errors = run.errors();
    set(true);
    assert_eq!(indexed, ["x0.0"]);
    assert_eq!(errors, 0);
}

// ── Test 9: forget, by the scan ─────────────────────────────────────────────

const REPLY_MARKER: &str = "quokkazephyrmarkerone";
const PROMPT_MARKER: &str = "wombatglaciermarkertwo";

fn log_row(conn: &Connection, logical: &str, scope: &str, ts_ms: i64, preview: &str) -> i64 {
    tx(conn, || {
        conn.execute(
            "INSERT INTO recall_log(ts_ms, session_id, logical, scope_dir, query_preview, query_hash, outcome,
                                    total_ms, detail)
             VALUES (?1, NULL, ?2, ?3, ?4, 'feedfacefeedface', 'ok', 1.0, '{\"picks\":[\"m:mem_another_row\"]}')",
            params![ts_ms, logical, scope, preview],
        )
        .unwrap();
        conn.last_insert_rowid()
    })
}

#[test]
fn t09_an_archive_tombstoned_on_disk_is_forgotten_by_the_scan_and_its_text_is_nowhere() {
    let env = Env::new();
    let dir = scope(KEY);
    let shared = "please continue with the plan as we wrote it down this morning";
    let a = Arch::new(1, "s-nine")
        .turn(shared, &format!("the first step is {REPLY_MARKER} and then the build runs again"))
        .turn(
            &format!("the {PROMPT_MARKER} is what I meant, what do you make of it?"),
            "I make of it that the second step has to wait for the first one",
        );
    let b = Arch::new(2, "s-nine")
        .at(T0 + 600_000)
        .turn(shared, "the plan is done and nothing of it is left to do today, all green");
    env.put_live(&dir, &a);
    env.put_live(&dir, &b);
    let mut run = env.run();
    run.scan();
    assert_eq!(positions(run.db(), &dir, &a), ["x0.0", "x2.0"]);
    assert_eq!(positions(run.db(), &dir, &b), ["x0.0"]);
    assert!(in_tables(run.db(), REPLY_MARKER) > 0 && in_tables(run.db(), PROMPT_MARKER) > 0);

    // A recall for A's prompt was logged; no key and no pick leads to that row.
    let old_log = log_row(run.db(), &a.logical, &dir, T0 - 5_000, &format!("the {PROMPT_MARKER} is what I meant"));
    let later_log = log_row(run.db(), &a.logical, &dir, T0 + 300_000, "a later question of the chat");

    env.put_tombstone(&dir, &a);
    run.scan();
    let conn = run.db();
    assert!(rows(conn, &dir, &a).is_empty());
    assert_eq!(positions(conn, &dir, &b), ["x0.0"]);
    assert_eq!(state_of(conn, &dir, &a), "tombstone");
    assert_eq!(count(conn, &format!("SELECT COUNT(*) FROM recall_log WHERE id = {old_log}")), 0);
    assert_eq!(count(conn, &format!("SELECT COUNT(*) FROM recall_log WHERE id = {later_log}")), 1);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memory_tombstones"), 0);
    assert!(purge_pending(conn), "the scan's forget left no mark for the log");

    let before = (archive_memories(conn), count(conn, "SELECT MAX(vid) FROM memory_ext"));
    run.scan();
    assert_eq!((archive_memories(run.db()), count(run.db(), "SELECT MAX(vid) FROM memory_ext")), before);

    run.idle_turn();
    assert!(!purge_pending(run.db()));
    for marker in [REPLY_MARKER, PROMPT_MARKER] {
        assert_eq!(in_tables(run.db(), marker), 0, "{marker} is in a table");
        assert_eq!(in_file(&env.brain(), marker), 0, "{marker} is in brain.r8");
        assert_eq!(in_file(&env.wal(), marker), 0, "{marker} is in brain.r8-wal");
    }
    assert_eq!(run.errors(), 0);
}

// ── Test 11: captures ───────────────────────────────────────────────────────

#[test]
fn t11_a_covered_capture_goes_and_a_later_one_stays() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    capture(&run, "mem_cap_before", &dir, "s-cap", T0 - 60_000, None);
    capture(&run, "mem_cap_after", &dir, "s-cap", T0 + 60_000, None);
    capture(&run, "mem_cap_other_chat", &dir, "s-other", T0 - 60_000, None);
    capture(&run, "mem_cap_other_dir", &scope(KEY_2), "s-cap", T0 - 60_000, None);

    let a = Arch::new(1, "s-cap").about("retry limit");
    env.put_live(&dir, &a);
    run.scan();
    let conn = run.db();
    assert!(!has(conn, "mem_cap_before"), "the stand-in outlived its archive");
    assert_eq!(positions(conn, &dir, &a), ["x0.0"]);
    assert!(has(conn, "mem_cap_after"));
    assert!(has(conn, "mem_cap_other_chat") && has(conn, "mem_cap_other_dir"));
    // Replaced, not forgotten: no tombstone, and no mark for the log.
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memory_tombstones"), 0);
    assert!(!purge_pending(conn));
}

#[test]
fn t11_a_capture_of_a_chat_whose_file_is_first_seen_as_a_tombstone_goes() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    capture(&run, "mem_cap_forgotten", &dir, "s-tomb", T0 - 60_000, None);
    capture(&run, "mem_cap_other_chat", &dir, "s-other", T0 - 60_000, None);
    let a = Arch::new(1, "s-tomb");
    env.put_tombstone(&dir, &a);
    run.scan();
    let conn = run.db();
    assert_eq!(state_of(conn, &dir, &a), "tombstone");
    assert!(!has(conn, "mem_cap_forgotten"));
    assert!(has(conn, "mem_cap_other_chat"));
    assert!(purge_pending(conn));
    assert_eq!(run.errors(), 0);
}

#[test]
fn t11_a_hidden_capture_stays_until_the_forget_and_one_older_than_the_window_stays() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let a = Arch::new(1, "s-bound").at(T0).about("retry limit");
    // A skipped archive of the chat stands between the old capture and this archive.
    tx(run.db(), || {
        run.db()
            .execute(
                "INSERT INTO archive_index(scope_dir, rel_name, archive_id, logical, state, skip_reason, file_size,
                                           created_ms, time_source, cover_ms, first_seen_ms)
                 VALUES (?1, 's000001/000000000000000000000000000000ff.json', '000000000000000000000000000000ff',
                         ?2, 'skipped', 'paused', 10, ?3, 'archive', ?3, ?3)",
                params![dir, a.logical, T0 - 600_000],
            )
            .unwrap()
    });
    capture(&run, "mem_cap_old", &dir, "s-bound", T0 - 900_000, None);
    capture(&run, "mem_cap_in_window", &dir, "s-bound", T0 - 300_000, None);
    capture(&run, "mem_cap_hidden", &dir, "s-bound", T0 - 200_000, Some("cli"));

    env.put_live(&dir, &a);
    run.scan();
    let conn = run.db();
    assert_eq!(positions(conn, &dir, &a), ["x0.0"]);
    assert!(!has(conn, "mem_cap_in_window"));
    assert!(has(conn, "mem_cap_hidden"), "a hidden capture was deleted by an archive");
    assert!(has(conn, "mem_cap_old"), "a capture older than the window was deleted");

    env.put_tombstone(&dir, &a);
    run.job(forgotten(&a, KEY));
    let conn = run.db();
    assert!(rows(conn, &dir, &a).is_empty());
    assert!(!has(conn, "mem_cap_hidden"), "forgotten text is kept by a hide");
    assert!(has(conn, "mem_cap_old"));
    assert_eq!(run.errors(), 0);
}

// ── Test 12: a forget before anyone knows the key ───────────────────────────

#[test]
fn t12_a_forget_for_a_directory_only_the_scan_has_met_takes_effect() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = Arch::new(1, "s-twelve").about("retry limit");
    let b = Arch::new(2, "s-twelve").at(T0 + 60_000).about("upload queue");
    env.put_live(&dir, &a);
    env.put_live(&dir, &b);
    let mut run = env.run();
    run.scan();
    assert_eq!(count(run.db(), "SELECT COUNT(*) FROM projects"), 0, "nothing told the brain the key");
    assert_eq!(positions(run.db(), &dir, &a), ["x0.0"]);

    run.job(forgotten(&a, KEY));
    let conn = run.db();
    assert!(rows(conn, &dir, &a).is_empty());
    assert_eq!(state_of(conn, &dir, &a), "tombstone");
    assert!(purge_pending(conn));
    assert_eq!(positions(conn, &dir, &b), ["x0.0"]);

    // The same with the file tombstoned and no event at all.
    env.put_tombstone(&dir, &b);
    run.scan();
    assert!(rows(run.db(), &dir, &b).is_empty());
    assert_eq!(state_of(run.db(), &dir, &b), "tombstone");
    // Neither comes back.
    run.scan();
    assert_eq!(archive_memories(run.db()), 0);
    assert_eq!(run.errors(), 0);
}

#[test]
fn t12_a_forget_for_an_archive_nobody_has_seen_keeps_it_out_and_one_for_nothing_does_nothing() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = Arch::new(1, "s-twelve").about("retry limit");
    env.put_live(&dir, &a);
    let mut run = env.run();

    // No file and no row for the id: nothing, not even a mark.
    run.job(forgotten(&Arch::new(9, "s-twelve"), KEY));
    assert_eq!(index_rows(run.db()), 0);
    assert!(!purge_pending(run.db()));

    // The event arrives before any scan: the file is still live on disk and is never
    // indexed.
    run.job(forgotten(&a, KEY));
    assert_eq!(state_of(run.db(), &dir, &a), "tombstone");
    run.scan();
    run.job(archived(&a, KEY, None, None, Some(2), Some(1)));
    assert_eq!(archive_memories(run.db()), 0);
    assert_eq!(state_of(run.db(), &dir, &a), "tombstone");
    assert_eq!(run.errors(), 0);
}

// ── Test 13: the same id or digest twice ────────────────────────────────────

#[test]
fn t13_a_forget_reaches_every_copy_of_a_digest() {
    let env = Env::new();
    let dir = scope(KEY);
    let one = Arch::new(1, "s-thirteen").about("retry limit");
    let mut two = Arch::new(2, "s-thirteen").about("retry limit");
    two.digest = one.digest.clone();
    let other = Arch::new(3, "s-thirteen").at(T0 + 60_000).about("upload queue");
    env.put_live(&dir, &one);
    env.put_live(&dir, &two);
    env.put_live(&dir, &other);
    let mut run = env.run();
    run.scan();
    let (rows_one, rows_two) = (rows(run.db(), &dir, &one), rows(run.db(), &dir, &two));
    assert_eq!((rows_one.len(), rows_two.len()), (1, 1));
    assert_ne!(rows_one[0].0, rows_two[0].0, "two archive ids, two sets of rows");

    env.put_tombstone(&dir, &one);
    run.job(forgotten(&one, KEY));
    let conn = run.db();
    assert!(rows(conn, &dir, &one).is_empty() && rows(conn, &dir, &two).is_empty());
    assert_eq!(state_of(conn, &dir, &one), "tombstone");
    assert_eq!(state_of(conn, &dir, &two), "tombstone");
    assert_eq!(positions(conn, &dir, &other), ["x0.0"]);
    // The second copy is still a live file, and stays out.
    run.scan();
    assert!(rows(run.db(), &dir, &two).is_empty());
    assert_eq!(run.errors(), 0);
}

#[test]
fn t13_a_live_file_whose_id_has_a_tombstone_row_in_the_directory_is_not_indexed() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = Arch::new(1, "s-thirteen").about("retry limit");
    env.put_tombstone(&dir, &a);
    env.put(&dir, "s000002", &a.id, &a.json(&dir));
    let mut run = env.run();
    run.scan();
    let conn = run.db();
    assert_eq!(state_of(conn, &dir, &a), "tombstone");
    assert_eq!(
        state(conn, &dir, &format!("s000002/{}.json", a.id)),
        Some(("skipped".to_string(), Some("forgotten_elsewhere".to_string())))
    );
    assert_eq!(archive_memories(conn), 0);

    // The other order: the live copy is indexed first, the tombstone is met after it.
    let b = Arch::new(2, "s-thirteen").about("upload queue");
    env.put_live(&dir, &b);
    env.put(&dir, "s000002", &b.id, &b.tombstone(&dir));
    run.scan();
    let conn = run.db();
    assert!(rows(conn, &dir, &b).is_empty());
    assert_eq!(state_of(conn, &dir, &b), "tombstone");
    assert_eq!(run.errors(), 0);
}

// ── Test 14: the import shape ───────────────────────────────────────────────

#[test]
fn t14_one_archive_in_two_projects_is_two_sets_of_rows_and_a_forget_takes_one() {
    let env = Env::new();
    let (dir, dir_2) = (scope(KEY), scope(KEY_2));
    let a = Arch::new(1, "s-fourteen").about("retry limit");
    env.put_live(&dir, &a);
    env.put_live(&dir_2, &a);
    // A tombstone under an id never seen live.
    let never = Arch::new(2, "s-fourteen");
    env.put_tombstone(&dir, &never);
    let mut run = env.run();
    run.scan();
    let conn = run.db();
    let (here, there) = (rows(conn, &dir, &a), rows(conn, &dir_2, &a));
    assert_eq!((here.len(), there.len()), (1, 1));
    assert_ne!(here[0].0, there[0].0);
    assert_eq!(count(conn, &format!("SELECT COUNT(*) FROM archive_index WHERE archive_id = '{}'", never.id)), 1);
    assert_eq!(state_of(conn, &dir, &never), "tombstone");
    assert_eq!(run.errors(), 0);

    run.job(forgotten(&a, KEY));
    let conn = run.db();
    assert!(rows(conn, &dir, &a).is_empty());
    assert_eq!(rows(conn, &dir_2, &a), there, "a forget in one project reached the other");
    assert_eq!(state_of(conn, &dir_2, &a), "indexed");
    assert_eq!(run.errors(), 0);
}

// ── The scan's "gone" rule (2.5; the whole of test 26 is commit 13b's) ───────

#[test]
fn a_gone_file_keeps_its_rows_unless_its_archive_is_live_in_another_project() {
    let env = Env::new();
    let (wrong, right) = (scope(KEY), scope(KEY_2));
    let moved = Arch::new(1, "s-moved").about("retry limit");
    let deleted = Arch::new(2, "s-moved").at(T0 + 60_000).about("upload queue");
    let wrong_path = env.put_live(&wrong, &moved);
    let deleted_path = env.put_live(&wrong, &deleted);
    let mut run = env.run();
    run.scan();

    // Gone before the copy exists: one scan changes nothing.
    std::fs::remove_file(&wrong_path).unwrap();
    std::fs::remove_file(&deleted_path).unwrap();
    run.scan();
    assert_eq!(positions(run.db(), &wrong, &moved), ["x0.0"]);
    assert_eq!(positions(run.db(), &wrong, &deleted), ["x0.0"]);

    // The copy appears in the right directory: the scan after it settles the move.
    env.put_live(&right, &moved);
    run.scan();
    let conn = run.db();
    assert!(rows(conn, &wrong, &moved).is_empty(), "the gone copy kept its rows");
    assert_eq!(positions(conn, &right, &moved), ["x0.0"]);
    assert_eq!(
        conn.query_row("SELECT scope_dir, state FROM archive_index WHERE archive_id = ?1", [&moved.id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })
        .unwrap(),
        (right.clone(), "indexed".to_string())
    );
    assert_eq!(count(conn, &format!("SELECT COUNT(*) FROM archive_index WHERE archive_id = '{}'", moved.id)), 1);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memory_tombstones"), 0);
    assert!(!purge_pending(conn), "a move is not a forget");
    // A gone file with no copy elsewhere keeps its rows.
    assert_eq!(positions(conn, &wrong, &deleted), ["x0.0"]);
    assert_eq!(state_of(conn, &wrong, &deleted), "indexed");
    assert_eq!(run.errors(), 0);
}

// ── Test 15: a replaced live file ───────────────────────────────────────────

#[test]
fn t15_a_live_file_replaced_by_another_live_file_is_mirrored_again() {
    let env = Env::new();
    let dir = scope(KEY);
    let old = Arch::new(1, "s-fifteen")
        .turn("what colour is the build lamp on the wall today?", "the lamp is oldcrimsonlamp and has been since noon")
        .about("upload queue");
    env.put_live(&dir, &old);
    let mut run = env.run();
    run.scan();
    assert_eq!(positions(run.db(), &dir, &old), ["x0.0", "x2.0"]);

    let mut new = Arch::new(1, "s-fifteen").turn(
        "what colour is the build lamp on the wall today?",
        "the lamp is newemeraldlamp, somebody fixed the build an hour ago",
    );
    new.digest = format!("{:064x}", 77);
    assert_ne!(old.json(&dir).len(), new.json(&dir).len());
    env.put_live(&dir, &new);
    run.scan();
    let conn = run.db();
    let now = rows(conn, &dir, &new);
    assert_eq!(now.len(), 1, "the rows of the old file are still there");
    assert!(now[0].2.contains("newemeraldlamp"));
    assert_eq!(in_tables(conn, "oldcrimsonlamp"), 0);
    assert_eq!(state_of(conn, &dir, &new), "indexed");
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memory_tombstones"), 0);
    assert_eq!(run.errors(), 0);
}

// ── Test 16: events ─────────────────────────────────────────────────────────

#[test]
fn t16_a_job_finds_its_file_by_id_or_by_a_path_in_the_directory_and_indexes_once() {
    let env = Env::new();
    let dir = scope(KEY);
    let by_id = Arch::new(1, "s-sixteen").about("retry limit");
    let by_path = Arch::new(2, "s-sixteen").at(T0 + 60_000).about("upload queue");
    let outside = Arch::new(3, "s-sixteen").at(T0 + 120_000).about("deploy window");
    env.put_live(&dir, &by_id);
    let path = env.put_live(&dir, &by_path);
    env.put_live(&dir, &outside);
    // The same id under another project: a path that leads there is not used.
    let elsewhere = env.put_live(&scope(KEY_2), &outside);
    let mut run = env.run();

    run.job(archived(&by_id, KEY, None, Some("s-sixteen"), Some(2), Some(1)));
    assert_eq!(positions(run.db(), &dir, &by_id), ["x0.0"]);
    run.job(archived(&by_path, KEY, Some(&path), Some("s-sixteen"), Some(2), Some(1)));
    assert_eq!(positions(run.db(), &dir, &by_path), ["x0.0"]);
    run.job(archived(&outside, KEY, Some(&elsewhere), Some("s-sixteen"), Some(2), Some(1)));
    let conn = run.db();
    assert_eq!(positions(conn, &dir, &outside), ["x0.0"], "the directory was not searched");
    assert!(rows(conn, &scope(KEY_2), &outside).is_empty(), "a path outside the project was read");
    assert_eq!(index_rows(conn), 3);
    assert_eq!(
        conn.query_row("SELECT project_key, COUNT(*) FROM projects WHERE scope_dir = ?1", [&dir], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?))
        })
        .unwrap(),
        (KEY.to_string(), 1)
    );
    // The row keeps what the event said, and the rows carry the session.
    assert_eq!(
        count(conn, "SELECT COUNT(*) FROM archive_index WHERE session_id = 's-sixteen' AND win_no = 2 AND part_no = 1"),
        3
    );
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memory_ext WHERE session_id = 's-sixteen'"), 3);

    // The same event twice indexes once.
    let before = (archive_memories(conn), count(conn, "SELECT MAX(vid) FROM memory_ext"));
    run.job(archived(&by_id, KEY, None, Some("s-sixteen"), Some(2), Some(1)));
    assert_eq!((archive_memories(run.db()), count(run.db(), "SELECT MAX(vid) FROM memory_ext")), before);
    assert_eq!(index_rows(run.db()), 3);
    // An id with no file is nothing.
    run.job(archived(&Arch::new(99, "s-sixteen"), KEY, None, None, Some(2), Some(1)));
    assert_eq!(index_rows(run.db()), 3);
    assert_eq!(run.errors(), 0);
}

#[test]
fn t16_three_parts_of_one_window_delivered_out_of_order_are_three_indexed_archives() {
    let env = Env::new();
    let dir = scope(KEY);
    let parts: Vec<Arch> =
        (1..=3u32).map(|n| Arch::new(n, "s-parts").at(T0 + i64::from(n)).about(&format!("part {n}"))).collect();
    for part in &parts {
        env.put_live(&dir, part);
    }
    let mut run = env.run();
    for n in [3usize, 1, 2] {
        run.job(archived(&parts[n - 1], KEY, None, Some("s-parts"), Some(2), Some(n as i64)));
    }
    let conn = run.db();
    for (i, part) in parts.iter().enumerate() {
        assert_eq!(state_of(conn, &dir, part), "indexed");
        assert_eq!(positions(conn, &dir, part), ["x0.0"]);
        assert_eq!(
            count(conn, &format!("SELECT part_no FROM archive_index WHERE archive_id = '{}'", part.id)),
            i as i64 + 1
        );
    }
    assert_eq!(run.errors(), 0);
}

// A host for the two halves of test 16 that are the rpc thread's.

struct PipeReader {
    rx: Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
    patience: Option<Duration>,
}

impl Read for PipeReader {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        while self.pos >= self.buf.len() {
            let next = match self.patience {
                None => self.rx.recv().ok(),
                Some(d) => match self.rx.recv_timeout(d) {
                    Ok(b) => Some(b),
                    Err(mpsc::RecvTimeoutError::Disconnected) => None,
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        return Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "no answer"))
                    }
                },
            };
            match next {
                Some(b) => {
                    self.buf = b;
                    self.pos = 0;
                }
                None => return Ok(0),
            }
        }
        let n = out.len().min(self.buf.len() - self.pos);
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}

struct PipeWriter(Sender<Vec<u8>>);

impl Write for PipeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .send(buf.to_vec())
            .map(|_| buf.len())
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "closed"))
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// `serve` on its own thread, with the test as the host.
struct Host {
    to: Option<PipeWriter>,
    from: FrameReader<BufReader<PipeReader>>,
    end: Receiver<ServeEnd>,
    status: Arc<Status>,
    next_id: u64,
}

impl Host {
    /// Started, initialised, and with the indexer's first scan behind it.
    fn ready(env: &Env, test: TestHooks) -> Host {
        let status = Arc::new(Status::default());
        let boot = Boot {
            embedder: None,
            clock: Arc::new(|| chrono::Utc::now().timestamp_millis()),
            test: Arc::new(test),
            status: status.clone(),
        };
        let (to_tx, to_rx) = mpsc::channel();
        let (from_tx, from_rx) = mpsc::channel();
        let stdin = PipeReader { rx: to_rx, buf: Vec::new(), pos: 0, patience: None };
        let stdout = PipeReader { rx: from_rx, buf: Vec::new(), pos: 0, patience: Some(Duration::from_secs(20)) };
        let (end_tx, end) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = end_tx.send(serve(stdin, PipeWriter(from_tx), boot));
        });
        let mut host = Host {
            to: Some(PipeWriter(to_tx)),
            from: FrameReader::new(BufReader::new(stdout)),
            end,
            status,
            next_id: 0,
        };
        let answer = host.call("initialize", json!({"config": env.config("off")}));
        assert_eq!(answer["result"]["protocol_version"], 1);
        let status = host.status.clone();
        wait("the first scan", || status.scan.load(Ordering::Relaxed) == 2);
        host
    }

    fn call(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let body = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let mut frame = Vec::new();
        write_frame(&mut frame, body.to_string().as_bytes()).unwrap();
        self.to.as_mut().unwrap().write_all(&frame).unwrap();
        let answer = self.from.next_frame().expect("the process closed its output").expect("no answer in time");
        let answer: Value = serde_json::from_slice(&answer).unwrap();
        assert_eq!(answer["id"], id, "answer to another request: {answer}");
        answer
    }

    /// One `on_context_archived` event; returns the hook's answer.
    fn archived(&mut self, archive_id: &str, path: Option<&Path>) -> Value {
        let params = json!({
            "kind": "on_context_archived",
            "session_id": "s-host",
            "cwd": null,
            "message": null,
            "data": {"archive_id": archive_id, "project_key": KEY, "window": 2, "part": 1,
                     "path": path.map(|p| p.to_string_lossy().into_owned())},
        });
        self.call("hook.handle", params)["result"].clone()
    }

    fn close(mut self) -> ServeEnd {
        self.to = None;
        self.end.recv_timeout(Duration::from_secs(20)).expect("serve did not return at end of input")
    }
}

#[test]
fn t16_an_archive_id_that_is_not_32_hex_is_dropped_on_the_rpc_thread() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut host = Host::ready(&env, TestHooks::default());
    // After the first scan, so only an event can index them.
    let named = Arch::new(1, "s-host").about("retry limit");
    let good = Arch::new(2, "s-host").at(T0 + 60_000).about("upload queue");
    let named_path = env.put_live(&dir, &named);
    env.put_live(&dir, &good);

    assert_eq!(host.archived("not-thirty-two-hex", Some(&named_path)), json!({"action": "continue"}));
    assert_eq!(host.archived(&good.id, None), json!({"action": "continue"}));
    let peek = brain_v2::connect(&env.brain(), Role::Cli).unwrap();
    wait("the good event's archive", || state_of(&peek, &dir, &good) == "indexed");
    assert_eq!(index_rows(&peek), 1, "the event with a bad id became a job");
    assert_eq!(state(&peek, &dir, &rel(&named.id)), None);
    wait("the log line of the dropped event", || {
        std::fs::read_to_string(env.brain().with_file_name("extension.log"))
            .unwrap_or_default()
            .contains("on_context_archived: not valid, no job")
    });
    drop(peek);
    host.close();
}

#[test]
fn t16_two_hundred_events_are_each_answered_before_any_is_processed() {
    let env = Env::new();
    let dir = scope(KEY);
    // The indexer waits three seconds before each job.
    let mut host = Host::ready(&env, TestHooks { indexer_sleep_ms: 3_000, ..TestHooks::default() });
    let files: Vec<Arch> = (1..=200u32)
        .map(|n| Arch::new(n, "s-host").at(T0 + i64::from(n) * 60_000).about(&format!("matter {n}")))
        .collect();
    for a in &files {
        env.put_live(&dir, a);
    }
    let start = Instant::now();
    for a in &files {
        assert_eq!(host.archived(&a.id, None), json!({"action": "continue"}));
    }
    let answered = start.elapsed();
    let peek = brain_v2::connect(&env.brain(), Role::Cli).unwrap();
    assert_eq!(index_rows(&peek), 0, "a job was processed before the last event was answered");
    assert!(answered < Duration::from_millis(2_500), "200 answers took {answered:?}");
    assert_eq!(host.status.jobs_dropped.load(Ordering::Relaxed), 0);
    // And they are real jobs: the first is indexed when its turn comes.
    wait("the first job", || index_rows(&peek) >= 1);
    drop(peek);
    host.close();
}

// ── Test 17: pause, and archives = off ──────────────────────────────────────

#[test]
fn t17_an_archive_of_a_paused_chat_is_skipped_and_is_indexed_after_its_row_is_deleted() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let a = Arch::new(1, "s-pause").about("retry limit");
    let free = Arch::new(2, "s-free").about("upload queue");
    capture(&run, "mem_cap_paused", &dir, "s-pause", T0 - 60_000, None);
    tx(run.db(), || store::set_pause(run.db(), &a.logical, Some("s-pause"), "learning", "cli").unwrap());
    env.put_live(&dir, &a);
    env.put_live(&dir, &free);

    run.scan();
    let conn = run.db();
    assert_eq!(state(conn, &dir, &rel(&a.id)), Some(("skipped".to_string(), Some("paused".to_string()))));
    assert!(rows(conn, &dir, &a).is_empty());
    assert!(has(conn, "mem_cap_paused"), "a paused archive purged a capture");
    assert_eq!(positions(conn, &dir, &free), ["x0.0"]);
    // The chat and the times are recorded, and nothing else.
    assert_eq!(
        conn.query_row(
            "SELECT logical, created_ms, time_source, cover_ms, digest FROM archive_index WHERE archive_id = ?1",
            [&a.id],
            |r| Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, i64>(3)?,
                r.get::<_, Option<String>>(4)?
            )),
        )
        .unwrap(),
        (a.logical.clone(), T0, "archive".to_string(), T0, None)
    );
    // A paused row is not read again.
    run.scan();
    assert!(rows(run.db(), &dir, &a).is_empty());

    // What `resume --catch-up` does: the pause is lifted and the row deleted.
    tx(run.db(), || {
        store::clear_pause(run.db(), &a.logical, Some("s-pause"), "learning").unwrap();
        run.db().execute("DELETE FROM archive_index WHERE skip_reason = 'paused'", []).unwrap();
    });
    run.scan();
    let conn = run.db();
    assert_eq!(state_of(conn, &dir, &a), "indexed");
    assert_eq!(positions(conn, &dir, &a), ["x0.0"]);
    assert!(!has(conn, "mem_cap_paused"), "the cover step left the capture");
    assert_eq!(run.errors(), 0);
}

#[test]
fn t17_with_archives_off_nothing_is_indexed_no_row_is_written_and_a_forget_still_lands() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = Arch::new(1, "s-off").about("retry limit");
    let b = Arch::new(2, "s-off").at(T0 + 60_000).about("upload queue");
    env.put_live(&dir, &a);
    let mut run = env.run();
    run.ctx.cfg.archives = false;
    run.scan();
    run.job(archived(&a, KEY, None, Some("s-off"), Some(2), Some(1)));
    assert_eq!((index_rows(run.db()), archive_memories(run.db())), (0, 0));

    // A file indexed while the switch was on is still forgotten while it is off.
    run.ctx.cfg.archives = true;
    run.scan();
    assert_eq!(positions(run.db(), &dir, &a), ["x0.0"]);
    run.ctx.cfg.archives = false;
    env.put_live(&dir, &b);
    env.put_tombstone(&dir, &a);
    run.scan();
    let conn = run.db();
    assert!(rows(conn, &dir, &a).is_empty());
    assert_eq!(state_of(conn, &dir, &a), "tombstone");
    assert_eq!(state(conn, &dir, &rel(&b.id)), None);
    // And so is one forgotten by an event.
    let c = Arch::new(3, "s-off").at(T0 + 120_000).about("deploy window");
    env.put_tombstone(&dir, &c);
    run.job(forgotten(&c, KEY));
    assert_eq!(state_of(run.db(), &dir, &c), "tombstone");

    // Switching it on indexes what accumulated.
    run.ctx.cfg.archives = true;
    run.scan();
    assert_eq!(positions(run.db(), &dir, &b), ["x0.0"]);
    assert_eq!(archive_memories(run.db()), 1);
    assert_eq!(run.errors(), 0);
}

// ── Test 18: a file that panics ─────────────────────────────────────────────

#[test]
fn t18_a_file_that_panics_is_skipped_the_next_is_indexed_and_it_is_not_read_again() {
    let env = Env::new();
    let dir = scope(KEY);
    let bad = Arch::new(1, "s-panic").about("retry limit");
    let good = Arch::new(2, "s-panic").at(T0 + 60_000).about("upload queue");
    env.put_live(&dir, &bad);
    env.put_live(&dir, &good);
    let mut run = env.run_with(TestHooks { panic_archive_id: Some(bad.id.clone()), ..TestHooks::default() }, None);
    run.scan();
    let conn = run.db();
    assert_eq!(state(conn, &dir, &rel(&bad.id)), Some(("skipped".to_string(), Some("panic".to_string()))));
    assert_eq!(
        count(conn, &format!("SELECT indexer_version FROM archive_index WHERE archive_id = '{}'", bad.id)),
        index_archives::INDEXER_VERSION
    );
    assert!(rows(conn, &dir, &bad).is_empty());
    assert_eq!(positions(conn, &dir, &good), ["x0.0"]);
    assert!(conn.is_autocommit());
    assert_eq!((run.panics(), run.errors()), (1, 0));
    assert_eq!(run.ctx.status.scan.load(Ordering::Relaxed), 2);

    run.scan();
    assert_eq!(run.panics(), 1, "the file that panicked was read again");
}

// ── Test 33 (c): the event's part fields ────────────────────────────────────

/// A file whose first `source_index` is 1: read as the scan finds it, it is a successor
/// and its first exchange is its carried head.
fn later_part(n: u32) -> Arch {
    let mut a = Arch::new(n, "s-part").about("retry limit").about("upload queue");
    a.first_index = 1;
    a
}

#[test]
fn t33c_a_later_part_named_by_an_event_is_read_whole() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = later_part(1);
    env.put_live(&dir, &a);
    let mut run = env.run();
    run.job(archived(&a, KEY, None, Some("s-part"), Some(5), Some(2)));
    let conn = run.db();
    assert_eq!(positions(conn, &dir, &a), ["x1.0", "x3.0"]);
    assert_eq!(
        count(conn, "SELECT COUNT(*) FROM archive_index WHERE state = 'indexed' AND win_no = 5 AND part_no = 2"),
        1
    );
    assert_eq!(run.errors(), 0);
}

#[test]
fn t33c_the_scan_got_there_first_and_the_job_adds_what_it_left_out() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = later_part(1);
    env.put_live(&dir, &a);
    let mut run = env.run();
    run.scan();
    let first = rows(run.db(), &dir, &a);
    assert_eq!(first.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(), ["x3.0"]);
    let vid = count(run.db(), "SELECT MAX(vid) FROM memory_ext");

    run.job(archived(&a, KEY, None, Some("s-part"), Some(5), Some(2)));
    let conn = run.db();
    let now = rows(conn, &dir, &a);
    assert_eq!(now.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(), ["x3.0", "x1.0"]);
    assert_eq!(now[0], first[0], "the row that was there changed");
    assert_eq!(
        count(conn, &format!("SELECT vid FROM memory_ext WHERE mem_id = '{}'", first[0].0)),
        vid,
        "the row that was there was written again"
    );
    assert_eq!(
        count(
            conn,
            "SELECT COUNT(*) FROM archive_index
              WHERE state = 'indexed' AND win_no = 5 AND part_no = 2 AND rows_indexed = 2 AND session_id = 's-part'"
        ),
        1
    );

    // An event that says nothing new about how the file is read opens nothing: a first
    // part stays as the scan read it.
    let b = later_part(2).at(T0 + 60_000);
    env.put_live(&dir, &b);
    run.scan();
    run.job(archived(&b, KEY, None, Some("s-part"), Some(2), Some(1)));
    assert_eq!(positions(run.db(), &dir, &b), ["x3.0"]);
    assert_eq!(count(run.db(), "SELECT COUNT(*) FROM archive_index WHERE win_no = 2 AND part_no = 1"), 1);
    assert_eq!(run.errors(), 0);
}

#[test]
fn t33c_a_record_with_carried_zero_is_read_whole_by_the_scan() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut a = later_part(1);
    a.carried = Some(0);
    env.put_live(&dir, &a);
    let mut run = env.run();
    run.scan();
    assert_eq!(positions(run.db(), &dir, &a), ["x1.0", "x3.0"]);
    assert_eq!(count(run.db(), "SELECT COUNT(*) FROM archive_index WHERE win_no IS NULL AND part_no IS NULL"), 1);
}

// ── Test 34 (a): the log is truncated after a forget ────────────────────────

const WAL_MARKER: &str = "axolotlharbourmarker";

fn marked(n: u32, session: &str) -> Arch {
    Arch::new(n, session).turn(
        "what is the name of the thing we must never write down anywhere?",
        &format!("its name is {WAL_MARKER} and I will not say it a second time"),
    )
}

/// An archive whose reply holds the marker is indexed, tombstoned on disk as core does it,
/// and forgotten through `on_forgotten`. No idle turn yet.
fn indexed_then_forgotten(env: &Env) -> (Run, Arch) {
    let dir = scope(KEY);
    let a = marked(1, "s-wal");
    env.put_live(&dir, &a);
    let mut run = env.run();
    run.job(archived(&a, KEY, None, Some("s-wal"), Some(2), Some(1)));
    assert_eq!(positions(run.db(), &dir, &a), ["x0.0"]);
    assert!(in_tables(run.db(), WAL_MARKER) > 0);
    env.put_tombstone(&dir, &a);
    run.job(forgotten(&a, KEY));
    assert!(rows(run.db(), &dir, &a).is_empty());
    assert_eq!(in_tables(run.db(), WAL_MARKER), 0);
    assert_eq!(run.errors(), 0);
    (run, a)
}

#[test]
fn t34a_without_the_idle_turn_the_mark_row_is_still_there_and_so_is_the_text_in_the_log() {
    let env = Env::new();
    let (run, _a) = indexed_then_forgotten(&env);
    assert!(purge_pending(run.db()), "the forget left no mark");
    // What the mark is for: the row is gone from every table, and the frames written when
    // it was inserted are still in the log.
    assert!(in_file(&env.wal(), WAL_MARKER) > 0, "the log does not hold the text: the test proves nothing");
}

#[test]
fn t34a_one_idle_turn_takes_the_text_out_of_the_log_and_removes_the_mark() {
    let env = Env::new();
    let (mut run, _a) = indexed_then_forgotten(&env);
    assert!(in_file(&env.wal(), WAL_MARKER) > 0);
    run.idle_turn();
    // The indexer's connection is still open: nothing here is the work of a last close.
    assert_eq!(in_file(&env.wal(), WAL_MARKER), 0, "the marker is in brain.r8-wal");
    assert_eq!(in_file(&env.brain(), WAL_MARKER), 0, "the marker is in brain.r8");
    assert!(!purge_pending(run.db()));
    assert_eq!(run.errors(), 0);
    // With no mark an idle turn does nothing: the log is not touched.
    let log = std::fs::metadata(env.wal()).map(|m| m.len()).unwrap_or(0);
    run.idle_turn();
    assert_eq!(std::fs::metadata(env.wal()).map(|m| m.len()).unwrap_or(0), log);
}

#[test]
fn t34a_the_loop_itself_truncates_the_log_on_its_idle_turn() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = marked(1, "s-loop");
    env.put_live(&dir, &a);
    let Run { ctx, jobs_rx, _vec_rx } = env.run_with(TestHooks::default(), Some(Embedder::hashing()));
    let jobs = ctx.jobs.clone();
    let stop = ctx.stop.clone();
    let status = ctx.status.clone();
    let indexer = std::thread::spawn(move || indexer::run(ctx, jobs_rx));

    let peek = brain_v2::connect(&env.brain(), Role::Cli).unwrap();
    jobs.try_send(archived(&a, KEY, None, Some("s-loop"), Some(2), Some(1))).unwrap();
    wait("the archive", || state_of(&peek, &dir, &a) == "indexed");
    assert!(in_tables(&peek, WAL_MARKER) > 0);
    env.put_tombstone(&dir, &a);
    jobs.try_send(forgotten(&a, KEY)).unwrap();
    wait("the forget", || state_of(&peek, &dir, &a) == "tombstone");
    assert_eq!(in_tables(&peek, WAL_MARKER), 0);
    // This test's own connection is closed: only the indexer's is open from here on.
    drop(peek);

    // The loop is alive and its connection open; only bytes are read.
    wait("the idle turn", || in_file(&env.wal(), WAL_MARKER) == 0);
    assert_eq!(in_file(&env.brain(), WAL_MARKER), 0, "the marker is in brain.r8");
    assert!(!indexer.is_finished(), "the loop ended: the log may have been emptied by the last close");

    let peek = brain_v2::connect(&env.brain(), Role::Cli).unwrap();
    wait("the mark row", || !purge_pending(&peek));
    assert_eq!(in_file(&env.wal(), WAL_MARKER), 0);
    assert_eq!(status.write_errors.load(Ordering::Relaxed), 0);
    assert_eq!(status.panics.load(Ordering::Relaxed), 0);
    stop.store(true, Ordering::SeqCst);
    let _ = jobs.try_send(Job::Shutdown);
    indexer.join().unwrap();
}

#[test]
fn t34a_a_reader_blocks_the_checkpoint_the_mark_stays_and_the_next_idle_turn_finishes() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = marked(1, "s-wal");
    env.put_live(&dir, &a);
    let mut run = env.run();
    run.job(archived(&a, KEY, None, Some("s-wal"), Some(2), Some(1)));
    assert_eq!(positions(run.db(), &dir, &a), ["x0.0"]);

    // Another connection reads, and keeps its view of the brain from before the forget.
    let reader = brain_v2::connect(&env.brain(), Role::Cli).unwrap();
    reader.execute_batch("BEGIN").unwrap();
    assert!(in_tables(&reader, WAL_MARKER) > 0);

    env.put_tombstone(&dir, &a);
    run.job(forgotten(&a, KEY));
    assert!(purge_pending(run.db()));

    let start = Instant::now();
    run.idle_turn();
    let waited = start.elapsed();
    assert!(purge_pending(run.db()), "the mark was removed though the log could not be truncated");
    assert!(in_file(&env.wal(), WAL_MARKER) > 0, "the reader did not hold the log: the test proves nothing");
    assert!(waited < Duration::from_secs(2), "a blocked checkpoint held the indexer for {waited:?}");
    assert_eq!(count(run.db(), "PRAGMA busy_timeout"), 5_000, "the indexer's busy timeout was not put back");
    assert_eq!(run.errors(), 0);

    // The reader is done: the next idle turn finishes the work.
    reader.execute_batch("ROLLBACK").unwrap();
    run.idle_turn();
    assert!(!purge_pending(run.db()));
    assert_eq!(in_file(&env.wal(), WAL_MARKER), 0, "the marker is in brain.r8-wal");
    assert_eq!(in_file(&env.brain(), WAL_MARKER), 0, "the marker is in brain.r8");
    assert_eq!(run.errors(), 0);
}

// ── Test 34 (b): a file that stops being a live record ──────────────────────

#[test]
fn t34b_an_indexed_file_that_is_no_longer_a_record_loses_its_rows_and_an_io_failure_does_not() {
    let env = Env::new();
    let dir = scope(KEY);
    let refused = Arch::new(1, "s-b").about("retry limit");
    let unreadable = Arch::new(2, "s-b").at(T0 + 60_000).about("upload queue");
    let refused_path = env.put_live(&dir, &refused);
    let unreadable_path = env.put_live(&dir, &unreadable);
    let mut run = env.run();
    run.scan();
    assert_eq!(positions(run.db(), &dir, &refused), ["x0.0"]);
    assert_eq!(positions(run.db(), &dir, &unreadable), ["x0.0"]);

    // Valid JSON of another length that says `version: 2`.
    std::fs::write(&refused_path, json!({"version": 2, "scope": dir, "id": refused.id}).to_string()).unwrap();
    // A directory of the same name: nothing can be read from it.
    std::fs::remove_file(&unreadable_path).unwrap();
    std::fs::create_dir(&unreadable_path).unwrap();
    std::fs::write(unreadable_path.join("filler"), "x".repeat(64)).unwrap();

    run.scan();
    let conn = run.db();
    assert!(rows(conn, &dir, &refused).is_empty(), "a file that is no longer a live record kept its rows");
    assert_eq!(state_of(conn, &dir, &refused), "tombstone");
    assert!(purge_pending(conn));
    assert_eq!(positions(conn, &dir, &unreadable), ["x0.0"], "an unreadable path lost its rows");
    assert_eq!(state_of(conn, &dir, &unreadable), "indexed");
    assert_eq!(run.errors(), 0);

    // The file is back as it was: nothing was lost, and nothing is written twice.
    std::fs::remove_file(unreadable_path.join("filler")).unwrap();
    std::fs::remove_dir(&unreadable_path).unwrap();
    env.put_live(&dir, &unreadable);
    let before = count(run.db(), "SELECT MAX(vid) FROM memory_ext");
    run.scan();
    assert_eq!(positions(run.db(), &dir, &unreadable), ["x0.0"]);
    assert_eq!(count(run.db(), "SELECT MAX(vid) FROM memory_ext"), before);
    // The refused file stays forgotten.
    assert!(rows(run.db(), &dir, &refused).is_empty());
}

// ── Test 34 (c): a scan stops when the mode is not Ready ────────────────────

#[test]
fn t34c_a_scan_stops_between_two_files_when_the_mode_is_not_ready() {
    let env = Env::new();
    let dir = scope(KEY);
    for n in 1..=3u32 {
        env.put_live(&dir, &Arch::new(n, "s-c").at(T0 + i64::from(n) * 60_000).about(&format!("matter {n}")));
    }
    let mut run = env.run();
    // The clock stores `Degraded` once the first file's row is in the brain.
    let peek = Mutex::new(brain_v2::connect(&env.brain(), Role::Cli).unwrap());
    let mode = run.ctx.mode.clone();
    run.ctx.clock = Arc::new(move || {
        if index_rows(&peek.lock().unwrap()) >= 1 {
            mode.store(Mode::Degraded as u8, Ordering::SeqCst);
        }
        NOW
    });
    assert!(index_archives::scan(&mut run.ctx).is_ok());
    assert_eq!(run.ctx.mode.load(Ordering::SeqCst), Mode::Degraded as u8);
    assert_eq!(index_rows(run.db()), 1, "a degraded brain was written again");
    assert_eq!(archive_memories(run.db()), 1);
    assert_ne!(run.ctx.status.scan.load(Ordering::Relaxed), 2, "a scan that stopped says it is done");
    assert_eq!(run.errors(), 0);
}

// ── One more: a credential-shaped token in a reply ──────────────────────────

#[test]
fn an_archive_whose_reply_holds_a_credential_shaped_token_is_indexed_with_the_token_blanked() {
    let env = Env::new();
    let dir = scope(KEY);
    // Joined from parts, so no key-shaped literal is in the repository.
    let token = ["sk", "-", &"a1B2c3D4e5".repeat(4)].concat();
    let a = Arch::new(1, "s-token").turn(
        "which key does the staging uploader use at the moment, do you know?",
        &format!("the staging uploader uses {token} and it expires at the end of the month"),
    );
    env.put_live(&dir, &a);
    let mut run = env.run();
    run.scan();
    let conn = run.db();
    let got = rows(conn, &dir, &a);
    assert_eq!(got.len(), 1);
    assert!(got[0].2.contains("the staging uploader uses"), "the reply is not in the row: {}", got[0].2);
    assert!(got[0].2.contains("expires at the end of the month"));
    assert!(!got[0].2.contains(&token), "the token is in the row");
    assert_eq!(in_tables(conn, &token), 0);
    assert_eq!(in_tables(conn, &token[3..]), 0);
    assert_eq!(run.errors(), 0);
}
