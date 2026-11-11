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
use axel::vindex::{VecDelta, VecKey};
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
    vec_rx: Receiver<VecDelta>,
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
        Run { ctx, jobs_rx, vec_rx }
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
    /// The `vid`s the front thread was told to remove from its vector array, so far.
    fn removed(&self) -> Vec<i64> {
        let mut out = Vec::new();
        while let Ok(delta) = self.vec_rx.try_recv() {
            if let VecDelta::Remove(keys) = delta {
                out.extend(keys.into_iter().filter_map(|key| match key {
                    VecKey::Mem(vid) => Some(vid),
                    VecKey::Doc(_) => None,
                }));
            }
        }
        out.sort_unstable();
        out
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
    /// The stored messages, in order: `{"role", "content"}`.
    items: Vec<Value>,
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
            items: Vec::new(),
            note: String::new(),
        }
    }
    fn at(mut self, created_ms: i64) -> Arch {
        self.created_ms = Some(created_ms);
        self
    }
    fn turn(self, prompt: &str, reply: &str) -> Arch {
        self.user(prompt).says(reply)
    }
    /// A user message with one text.
    fn user(mut self, text: &str) -> Arch {
        self.items.push(json!({"role": "user", "content": text}));
        self
    }
    /// A prompt core withheld.
    fn withheld(self) -> Arch {
        self.user("[archive: withheld]")
    }
    /// An assistant message with one text.
    fn says(mut self, text: &str) -> Arch {
        self.items.push(json!({"role": "assistant", "content": text}));
        self
    }
    /// An assistant message with a text and a call of `tool`.
    fn calls(mut self, text: &str, tool: &str) -> Arch {
        self.items.push(json!({"role": "assistant", "content": [
            {"type": "text", "text": text},
            {"type": "tool_use", "id": "toolu_01", "name": tool, "input": {"id": "m:mem_0000000000000000"}},
        ]}));
        self
    }
    /// A file that starts at `source_index` 1: read with no event, it is a successor.
    fn successor(mut self) -> Arch {
        self.first_index = 1;
        self
    }
    fn noted(mut self, note: &str) -> Arch {
        self.note = note.to_string();
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
        let messages: Vec<Value> = self
            .items
            .iter()
            .enumerate()
            .map(|(i, message)| {
                json!({"source_index": self.first_index + i as u32, "block_indices": [], "message": message})
            })
            .collect();
        let mut record = json!({
            "version": 1, "scope": scope, "logical": self.logical, "id": self.id, "digest": self.digest,
            "body": {"state": "live",
                     "source_message_count": self.first_index + self.items.len() as u32,
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
    capture_keyed(run, id, scope, session, created_ms, hidden_by, &format!("{:016x}", created_ms));
}

/// The same, for a turn whose prompt has the key `turn_key`.
fn capture_keyed(
    run: &Run,
    id: &str,
    scope: &str,
    session: &str,
    created_ms: i64,
    hidden_by: Option<&str>,
    turn_key: &str,
) {
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
        turn_key: Some(turn_key.to_string()),
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
    // RULING: wave4 R-13 item 4. A tombstone says nothing of when the archive was sealed,
    // so the chat's captures go up to now, with no lower bound: also one made after the
    // time the archive had (`T0`). The choice errs towards forgetting, and is pinned here.
    capture(&run, "mem_cap_after_the_archive", &dir, "s-tomb", T0 + 3_600_000, None);
    capture(&run, "mem_cap_other_chat", &dir, "s-other", T0 - 60_000, None);
    let a = Arch::new(1, "s-tomb");
    env.put_tombstone(&dir, &a);
    run.scan();
    let conn = run.db();
    assert_eq!(state_of(conn, &dir, &a), "tombstone");
    assert!(!has(conn, "mem_cap_forgotten"));
    assert!(!has(conn, "mem_cap_after_the_archive"), "a capture made after the archive outlived the forget");
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
    // Commit 13b's clause: no unit of the archive has the hidden capture's key, so nothing
    // is hidden with it and the archive counts it.
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memory_ext WHERE origin_kind = 'archive' AND hidden_ms IS NOT NULL"), 0);
    assert_eq!(unmatched_hidden(conn, &dir, &a), 1);
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

    // The other order: the live copy is indexed first, the tombstone is met after it. One
    // scan takes a tombstone before the new live files (ruling 1a), so two scans here.
    let b = Arch::new(2, "s-thirteen").about("upload queue");
    env.put_live(&dir, &b);
    run.scan();
    assert_eq!(state_of(run.db(), &dir, &b), "indexed");
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
    let Run { ctx, jobs_rx, vec_rx: _vec_rx } = env.run_with(TestHooks::default(), Some(Embedder::hashing()));
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

/// Guards what the owner would lose: the front thread forgets a turn just after the
/// checkpoint, and the mark of that purge is deleted with the one the checkpoint finished.
#[test]
fn a_purge_committed_between_the_checkpoint_and_the_marks_delete_keeps_its_mark() {
    const LATE: &str = "mem_gapmarkerlatepurge";
    let env = Env::new();
    let (mut run, _a) = indexed_then_forgotten(&env);
    capture(&run, LATE, &scope(KEY), "s-gap", T0, None);
    let front = brain_v2::connect(&env.brain(), Role::Cli).unwrap();
    let before = count(run.db(), "SELECT CAST(value AS INTEGER) FROM context_data WHERE key = 'purge_pending'");
    index_archives::checkpoint_after_purge_then(&mut run.ctx, || {
        // Another writer forgets a row, with its own mark, before the indexer's delete.
        brain_v2::begin_write(&front).unwrap();
        assert_eq!(store::purge_many(&front, &[LATE.to_string()], false, true).unwrap(), 1);
        index_archives::mark_purged(&front).unwrap();
        front.execute_batch("COMMIT").unwrap();
    })
    .unwrap();
    let conn = run.db();
    assert!(!has(conn, LATE));
    assert!(purge_pending(conn), "the mark of a purge the checkpoint did not see was deleted");
    assert!(count(conn, "SELECT CAST(value AS INTEGER) FROM context_data WHERE key = 'purge_pending'") > before);
    assert!(in_file(&env.brain(), "gapmarkerlatepurge") > 0, "the brain does not hold the text: the test proves nothing");
    // The next idle turn finishes it.
    run.idle_turn();
    assert!(!purge_pending(run.db()));
    assert_eq!(in_file(&env.wal(), "gapmarkerlatepurge"), 0, "the marker is in brain.r8-wal");
    assert_eq!(in_file(&env.brain(), "gapmarkerlatepurge"), 0, "the marker is in brain.r8");
    assert_eq!(run.errors(), 0);
}

// ── Test 34 (b): a file that stops being a live record ──────────────────────

#[test]
fn t34b_an_indexed_file_that_is_refused_loses_its_own_rows_and_is_skipped_and_an_io_failure_does_nothing() {
    let env = Env::new();
    let dir = scope(KEY);
    let refused = Arch::new(1, "s-b").about("retry limit");
    let unreadable = Arch::new(2, "s-b").at(T0 + 60_000).about("upload queue");
    // Another file with the digest of the first: a forget would reach it, a refusal does not.
    let mut copy = Arch::new(3, "s-b").at(T0 + 120_000).about("deploy window");
    copy.digest = refused.digest.clone();
    let refused_path = env.put_live(&dir, &refused);
    let unreadable_path = env.put_live(&dir, &unreadable);
    env.put_live(&dir, &copy);
    let mut run = env.run();
    // A hidden capture of the archive's window and a log row of the chat: a forget takes both.
    capture(&run, "mem_cap_hidden", &dir, "s-b", T0 - 1_000, Some("cli"));
    let log = log_row(run.db(), &refused.logical, &dir, T0 - 500, "a question of that window");
    run.scan();
    assert_eq!(positions(run.db(), &dir, &refused), ["x0.0"]);
    assert_eq!(positions(run.db(), &dir, &unreadable), ["x0.0"]);
    assert_eq!(positions(run.db(), &dir, &copy), ["x0.0"]);
    let old_vid = count(run.db(), &format!("SELECT vid FROM memory_ext WHERE origin_ref = '{}'", refused.id));
    run.removed();

    // Not JSON, of another length.
    let live = refused.json(&dir);
    std::fs::write(&refused_path, "x".repeat(live.len() + 40)).unwrap();
    // A directory of the same name: nothing can be read from it.
    std::fs::remove_file(&unreadable_path).unwrap();
    std::fs::create_dir(&unreadable_path).unwrap();
    std::fs::write(unreadable_path.join("filler"), "x".repeat(64)).unwrap();

    run.scan();
    let conn = run.db();
    assert!(rows(conn, &dir, &refused).is_empty(), "a file that is no longer a live record kept its rows");
    let (state_now, reason) = state(conn, &dir, &rel(&refused.id)).unwrap();
    assert_eq!(state_now, "skipped", "a refused file is not a forgotten one");
    assert!(reason.is_some_and(|r| !r.is_empty()));
    assert!(purge_pending(conn));
    assert_eq!(run.removed(), [old_vid], "the front thread was not told to drop the vector");
    // Nothing by digest, no capture, no log row.
    assert_eq!(positions(conn, &dir, &copy), ["x0.0"], "the refusal reached another file by its digest");
    assert!(has(conn, "mem_cap_hidden"), "the refusal purged a capture");
    assert_eq!(count(conn, &format!("SELECT COUNT(*) FROM recall_log WHERE id = {log}")), 1);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM archive_index WHERE state = 'tombstone'"), 0);
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
    assert!(rows(run.db(), &dir, &refused).is_empty());

    // The refused file is read again when its TIME changes, its size being the same: the
    // live record again, padded to the length of the refused bytes.
    let seen_ms = count(
        run.db(),
        &format!("SELECT created_ms FROM archive_index WHERE rel_name = '{}'", rel(&refused.id)),
    );
    std::fs::write(&refused_path, format!("{live}{}", " ".repeat(40))).unwrap();
    set_mtime(&refused_path, seen_ms);
    run.scan();
    assert_eq!(state_of(run.db(), &dir, &refused), "skipped", "the same size and the same time, and it was read");
    set_mtime(&refused_path, seen_ms + 7_000);
    run.scan();
    assert_eq!(state_of(run.db(), &dir, &refused), "indexed", "another time, and it was not read again");
    assert_eq!(positions(run.db(), &dir, &refused), ["x0.0"]);
    assert_eq!(run.errors(), 0);
}

fn set_mtime(path: &Path, ms: i64) {
    let file = std::fs::File::options().write(true).open(path).unwrap();
    file.set_modified(std::time::UNIX_EPOCH + Duration::from_millis(ms as u64)).unwrap();
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

// ═════════════════════════════════════════════════════════════════════════════
// Commit 13b: the forget marks, the twins, the live call
// (section 2: tests 24 to 27, 29 to 31, 33 (a) and (b), 34 (d))
// ═════════════════════════════════════════════════════════════════════════════

/// One window apart.
const W: i64 = 600_000;
const FORGET_PROMPT: &str = "forget that I told you about the offer from Acme, I have not accepted it";

fn key_of(prompt: &str) -> String {
    axel::archive::turn_key(&axel::screen::redact(prompt).text)
}

fn forget_units(conn: &Connection, scope: &str, a: &Arch) -> i64 {
    conn.query_row(
        "SELECT forget_units FROM archive_index WHERE scope_dir = ?1 AND archive_id = ?2",
        params![scope, a.id],
        |r| r.get(0),
    )
    .unwrap()
}

fn unmatched_hidden(conn: &Connection, scope: &str, a: &Arch) -> i64 {
    conn.query_row(
        "SELECT unmatched_hidden FROM archive_index WHERE scope_dir = ?1 AND archive_id = ?2",
        params![scope, a.id],
        |r| r.get(0),
    )
    .unwrap()
}

/// `(source, turn_key, head, tail, archive_id)` of every `forgotten_turns` row.
fn forgot_rows(conn: &Connection) -> Vec<(String, String, i64, i64, String)> {
    let mut stmt = conn
        .prepare("SELECT source, turn_key, head, tail, archive_id FROM forgotten_turns ORDER BY id")
        .unwrap();
    let got = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?))).unwrap();
    got.collect::<rusqlite::Result<_>>().unwrap()
}

/// `hidden_by` of an archive's rows, by position.
fn hidden_by(conn: &Connection, scope: &str, a: &Arch) -> Vec<(String, Option<String>)> {
    let mut stmt = conn
        .prepare(
            "SELECT e.unit_pos, e.hidden_by FROM memory_ext e JOIN memories m ON m.id = e.mem_id
              WHERE e.origin_kind = 'archive' AND e.origin_ref = ?1 AND m.project_key = ?2 ORDER BY e.vid",
        )
        .unwrap();
    let got = stmt.query_map(params![a.id, scope], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    got.collect::<rusqlite::Result<_>>().unwrap()
}

/// The live call, as the front thread makes it: inside a transaction of the caller.
fn live_call(run: &Run, session: &str, from_ms: i64, to_ms: i64, key: Option<&str>) -> Vec<String> {
    let logical = project::logical(session);
    tx(run.db(), || {
        index_archives::mark_forgotten_turn(run.db(), &logical, Some(session), "call", from_ms, to_ms, key).unwrap()
    })
}

/// Every archive row of the brain: `(id, archive id, unit_pos, hidden_by)`, sorted.
fn all_rows(conn: &Connection) -> Vec<(String, String, String, Option<String>)> {
    let mut stmt = conn
        .prepare(
            "SELECT mem_id, origin_ref, unit_pos, hidden_by FROM memory_ext WHERE origin_kind = 'archive'
              ORDER BY mem_id",
        )
        .unwrap();
    let got = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?))).unwrap();
    got.collect::<rusqlite::Result<_>>().unwrap()
}

/// Puts the files one at a time, in this order, with a scan after each.
fn index_in_order(env: &Env, run: &mut Run, scope: &str, files: &[Arch], order: &[usize]) {
    for at in order {
        env.put_live(scope, &files[*at]);
        run.scan();
    }
    assert_eq!(run.errors(), 0);
}

fn reply(mark: &str) -> String {
    format!("{mark} is what I have to say about it, and it is long enough to be a unit of its own")
}

// ── Test 24: the forget turn across a seal, by time ─────────────────────────

/// A first window with two exchanges, and its successor: a carried prompt, the tail, new
/// assistant text, a new exchange.
fn sealed_pair(n: u32, session: &str, first_ms: i64, second_ms: i64) -> (Arch, Arch) {
    let a = Arch::new(n, session)
        .at(first_ms)
        .turn("what is the first thing on the list for today?", &reply("FIRSTANSWER"))
        .turn("and what did the vendor say about the contract?", &reply("VENDORANSWER"));
    let b = Arch::new(n + 1, session)
        .at(second_ms)
        .successor()
        .user("and what did the vendor say about the contract?")
        .says(&reply("CARRIEDTAIL"))
        .says(&reply("RUNNINGON"))
        .turn("what is next on the list after the vendor?", &reply("NEXTANSWER"));
    (a, b)
}

#[test]
fn t24_a_call_row_leaves_out_the_turn_on_both_sides_of_a_seal_and_only_in_its_chat() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    assert!(live_call(&run, "s-time", 1_000, 2_000, None).is_empty());
    assert_eq!(forgot_rows(run.db()), [("call".to_string(), String::new(), 0, 0, String::new())]);

    let (a, b) = sealed_pair(1, "s-time", 1_500, 900_000);
    let (other_a, other_b) = sealed_pair(3, "t-time", 1_500, 900_000);
    index_in_order(&env, &mut run, &dir, &[a.clone(), b.clone(), other_a.clone(), other_b.clone()], &[0, 1, 2, 3]);
    let conn = run.db();
    // A was sealed inside the turn: its last exchange is the turn's beginning.
    assert_eq!(positions(conn, &dir, &a), ["x0.0"]);
    assert_eq!(in_tables(conn, "VENDORANSWER"), 2, "only the other chat's row and word row hold it");
    // B's window starts inside the turn: its leading span is the turn running on.
    assert_eq!(positions(conn, &dir, &b), ["x4.0"]);
    assert_eq!((forget_units(conn, &dir, &a), forget_units(conn, &dir, &b)), (1, 1));
    // The other chat, with the same times, is indexed whole.
    assert_eq!(positions(conn, &dir, &other_a), ["x0.0", "x2.0"]);
    assert_eq!(positions(conn, &dir, &other_b), ["c3.0", "x4.0"]);
    assert_eq!((forget_units(conn, &dir, &other_a), forget_units(conn, &dir, &other_b)), (0, 0));
}

#[test]
fn t24_the_live_call_removes_exactly_the_last_exchange_of_an_archive_sealed_inside_the_turn() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let (a, b) = sealed_pair(1, "s-time", 1_500, 900_000);
    index_in_order(&env, &mut run, &dir, &[a.clone(), b.clone()], &[0, 1]);
    let before = rows(run.db(), &dir, &a);
    assert_eq!(positions(run.db(), &dir, &a), ["x0.0", "x2.0"]);
    assert_eq!(positions(run.db(), &dir, &b), ["c3.0", "x4.0"]);
    // A recall of the turn was logged before the call, and one long before it.
    let own_log = log_row(run.db(), &a.logical, &dir, 1_200, "and what did the vendor say");
    let old_log = log_row(run.db(), &a.logical, &dir, 500, "an older question");

    let purged = live_call(&run, "s-time", 1_000, 2_000, None);
    let conn = run.db();
    let b_rows = rows(conn, &dir, &b);
    // A's last exchange, and B's leading `Continued` unit (its window starts at 1,500).
    assert_eq!(purged.len(), 2);
    assert!(purged.contains(&before[1].0), "the id of A's last exchange was not returned");
    assert_eq!(rows(conn, &dir, &a), before[..1]);
    assert_eq!(b_rows.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(), ["x4.0"]);
    assert_eq!((forget_units(conn, &dir, &a), forget_units(conn, &dir, &b)), (1, 1));
    assert!(purge_pending(conn));
    assert_eq!(count(conn, &format!("SELECT COUNT(*) FROM recall_log WHERE id = {own_log}")), 0);
    assert_eq!(count(conn, &format!("SELECT COUNT(*) FROM recall_log WHERE id = {old_log}")), 1);

    // A second forget in the same turn is the same row with a later end, and takes
    // nothing more: the archives the row already covered are not visited again.
    assert!(live_call(&run, "s-time", 1_000, 2_500, None).is_empty());
    assert_eq!(count(run.db(), "SELECT COUNT(*) FROM forgotten_turns"), 1);
    assert_eq!(count(run.db(), "SELECT to_ms FROM forgotten_turns"), 2_500);
    assert_eq!(positions(run.db(), &dir, &a), ["x0.0"]);
    assert_eq!(positions(run.db(), &dir, &b), ["x4.0"]);
}

#[test]
fn t24_an_archive_whose_time_is_from_mtime_takes_no_part() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    live_call(&run, "s-mtime", NOW - 10, NOW + 10, None);
    live_call(&run, "s-timed", NOW - 10, NOW + 10, None);
    // No `created_ms`: found by the scan, its time is the file's, and its cover time is
    // when it was first seen, which is inside the interval.
    let mut untimed = Arch::new(1, "s-mtime").about("retry limit").about("upload queue");
    untimed.created_ms = None;
    let timed = Arch::new(2, "s-timed").at(NOW).about("retry limit").about("upload queue");
    index_in_order(&env, &mut run, &dir, &[untimed.clone(), timed.clone()], &[0, 1]);
    let conn = run.db();
    assert_eq!(count(conn, "SELECT COUNT(*) FROM archive_index WHERE time_source = 'mtime' AND cover_ms BETWEEN 1749999999990 AND 1750000000010"), 1);
    assert_eq!(positions(conn, &dir, &untimed), ["x0.0", "x2.0"], "a file timed by mtime was touched");
    assert_eq!(positions(conn, &dir, &timed), ["x0.0"]);
}

#[test]
fn the_live_call_reaches_one_window_each_way_by_key_and_refuses_what_it_is_not_for() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let files: Vec<Arch> = (1..=5u32)
        .map(|n| Arch::new(n, "s-key").at(T0 + i64::from(n) * W).turn("continue", &reply(&format!("CONTINUED{n}"))))
        .collect();
    let other = Arch::new(9, "t-key").at(T0 + 4 * W).turn("continue", &reply("OTHERCHAT"));
    let mut all = files.clone();
    all.push(other.clone());
    index_in_order(&env, &mut run, &dir, &all, &[0, 1, 2, 3, 4, 5]);

    // The call is made in the fifth window, with the key of the hook's text.
    let key = key_of("continue");
    let purged = live_call(&run, "s-key", T0 + 4 * W + W / 2, T0 + 4 * W + W / 2 + 5_000, Some(&key));
    let conn = run.db();
    assert_eq!(purged.len(), 2);
    for (n, a) in files.iter().enumerate() {
        // Its own window and the one before; never further.
        let expect: &[&str] = if n >= 3 { &[] } else { &["x0.0"] };
        assert_eq!(positions(conn, &dir, a), expect, "archive {}", n + 1);
    }
    assert_eq!(positions(conn, &dir, &other), ["x0.0"]);
    assert_eq!(forgot_rows(conn), [("call".to_string(), key.clone(), 0, 0, String::new())]);

    // No transaction of the caller: refused, and nothing is written.
    let logical = project::logical("s-key");
    assert!(index_archives::mark_forgotten_turn(conn, &logical, Some("s-key"), "call", 1, 2, None).is_err());
    assert_eq!(count(conn, "SELECT COUNT(*) FROM forgotten_turns"), 1);
    // `archive` rows are the indexer's.
    brain_v2::begin_write(conn).unwrap();
    assert!(index_archives::mark_forgotten_turn(conn, &logical, Some("s-key"), "archive", 1, 2, None).is_err());
    conn.execute_batch("ROLLBACK").unwrap();
}

// ── Test 25: twins ──────────────────────────────────────────────────────────

const HIDDEN_PROMPT: &str = "what did the doctor say about the results of last week?";
const SEEN_PROMPT: &str = "what is the name of the hotel we booked for the trip?";

#[test]
fn t25_a_unit_with_the_key_of_a_hidden_capture_is_born_hidden_as_its_twin() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    capture_keyed(&run, "mem_cap_h", &dir, "s-twin", T0 - 100_000, Some("cli"), &key_of(HIDDEN_PROMPT));
    capture_keyed(&run, "mem_cap_v", &dir, "s-twin", T0 - 50_000, None, &key_of(SEEN_PROMPT));
    let a = Arch::new(1, "s-twin")
        .turn(HIDDEN_PROMPT, &reply("DOCTORANSWER"))
        .turn(SEEN_PROMPT, &reply("HOTELANSWER"))
        .about("upload queue");
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&a), &[0]);
    let conn = run.db();
    assert_eq!(
        hidden_by(conn, &dir, &a),
        [
            ("x0.0".to_string(), Some("twin:mem_cap_h".to_string())),
            ("x2.0".to_string(), None),
            ("x4.0".to_string(), None)
        ]
    );
    // The twin is hidden since the capture was.
    assert_eq!(
        count(conn, "SELECT COUNT(*) FROM memory_ext WHERE hidden_by = 'twin:mem_cap_h' AND hidden_ms = (SELECT hidden_ms FROM memory_ext WHERE mem_id = 'mem_cap_h')"),
        1
    );
    assert!(!has(conn, "mem_cap_v"), "the visible stand-in outlived its archive");
    assert!(has(conn, "mem_cap_h"), "the record of the hide was deleted");
    assert_eq!(unmatched_hidden(conn, &dir, &a), 0);

    assert_eq!(tx(conn, || store::restore(conn, "mem_cap_h").unwrap()), 1);
    assert!(hidden_by(conn, &dir, &a).iter().all(|(_, by)| by.is_none()));
    assert!(!has(conn, "mem_cap_h"));
}

#[test]
fn t25_when_core_changed_the_text_no_key_matches_nothing_is_hidden_and_it_is_counted() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    capture_keyed(&run, "mem_cap_h", &dir, "s-twin", T0 - 100_000, Some("cli"), &key_of(HIDDEN_PROMPT));
    // The prompt of the hidden turn is stored as the placeholder.
    let a = Arch::new(1, "s-twin")
        .withheld()
        .says(&reply("DOCTORANSWER"))
        .turn(SEEN_PROMPT, &reply("HOTELANSWER"))
        .about("upload queue");
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&a), &[0]);
    let conn = run.db();
    let got = hidden_by(conn, &dir, &a);
    assert_eq!(got.iter().map(|r| r.0.as_str()).collect::<Vec<_>>(), ["c1.0", "x2.0", "x4.0"]);
    assert!(got.iter().all(|(_, by)| by.is_none()), "a unit was hidden with no key to go by: {got:?}");
    assert_eq!(unmatched_hidden(conn, &dir, &a), 1);

    assert_eq!(tx(conn, || store::restore(conn, "mem_cap_h").unwrap()), 1);
    assert!(has(conn, "mem_cap_h"));
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memory_ext WHERE mem_id = 'mem_cap_h' AND hidden_ms IS NULL"), 1);
}

// ── Test 26, the rest: what a move takes with it ────────────────────────────

#[test]
fn t26_a_moved_archive_takes_the_visible_capture_it_covered_and_writes_no_tombstone() {
    let env = Env::new();
    let (wrong, right) = (scope(KEY), scope(KEY_2));
    let moved = Arch::new(1, "s-moved").about("retry limit");
    let wrong_path = env.put_live(&wrong, &moved);
    let mut run = env.run();
    run.scan();
    let old_ids: Vec<String> = rows(run.db(), &wrong, &moved).into_iter().map(|r| r.0).collect();
    // Captures of the chat filed under the wrong directory: one the archive covers, one
    // it covers that is hidden, one from after it.
    capture(&run, "mem_cap_covered", &wrong, "s-moved", T0 - 1_000, None);
    capture(&run, "mem_cap_hidden", &wrong, "s-moved", T0 - 2_000, Some("cli"));
    capture(&run, "mem_cap_later", &wrong, "s-moved", T0 + 60_000, None);

    // Both copies exist: both directories hold its rows.
    env.put_live(&right, &moved);
    run.scan();
    assert_eq!(positions(run.db(), &wrong, &moved), ["x0.0"]);
    assert_eq!(positions(run.db(), &right, &moved), ["x0.0"]);
    assert!(has(run.db(), "mem_cap_covered"));

    std::fs::remove_file(&wrong_path).unwrap();
    run.scan();
    let conn = run.db();
    assert!(rows(conn, &wrong, &moved).is_empty());
    assert_eq!(positions(conn, &right, &moved), ["x0.0"]);
    assert!(!has(conn, "mem_cap_covered"), "the stand-in under the old directory outlived the move");
    assert!(has(conn, "mem_cap_hidden") && has(conn, "mem_cap_later"));
    assert_eq!(count(conn, "SELECT COUNT(*) FROM archive_index WHERE state = 'tombstone'"), 0);
    assert_eq!(count(conn, &format!("SELECT COUNT(*) FROM archive_index WHERE archive_id = '{}'", moved.id)), 1);
    for id in &old_ids {
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM memory_tombstones WHERE id = ?1", [id], |r| r.get::<_, i64>(0)).unwrap(),
            0
        );
    }
    // What a search from each folder can find: the word rows are under R's key only.
    assert_eq!(
        conn.query_row(
            "SELECT group_concat(DISTINCT m.project_key) FROM memories_fts f JOIN memories m ON m.id = f.mem_id
              WHERE memories_fts MATCH 'retry' AND m.id LIKE 'mem_%' AND m.topic = 'episodic_turn'
                AND m.id IN (SELECT mem_id FROM memory_ext WHERE origin_kind = 'archive')",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        right
    );
    assert_eq!(run.errors(), 0);
}

// ── Test 27: pause by session ───────────────────────────────────────────────

#[test]
fn t27_a_pause_set_for_the_session_holds_for_an_archive_with_another_logical() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    tx(run.db(), || store::set_pause(run.db(), "some-other-logical", Some("ephemeral-1234"), "learning", "cli").unwrap());
    let a = Arch::new(1, "the-records-own-chat-id").about("retry limit");
    env.put_live(&dir, &a);
    run.job(archived(&a, KEY, None, Some("ephemeral-1234"), Some(2), Some(1)));
    let conn = run.db();
    assert_eq!(state(conn, &dir, &rel(&a.id)), Some(("skipped".to_string(), Some("paused".to_string()))));
    assert!(rows(conn, &dir, &a).is_empty());
    assert_eq!(count(conn, "SELECT COUNT(*) FROM archive_index WHERE session_id = 'ephemeral-1234'"), 1);
    assert_eq!(run.errors(), 0);
}

// ── Test 29: marks between files ────────────────────────────────────────────

/// Chat `session`, five archives one window apart. A: its last exchange is `p` with a
/// first reply. B, a successor: `p` carried, the tail, assistant text with a forget call,
/// a new exchange. C, a successor: `p` carried, and a new exchange whose prompt is `p`
/// again. D and E: an exchange whose prompt is `p`.
fn five_windows(first: u32, session: &str, p: &str) -> Vec<Arch> {
    let at = |n: u32| T0 + i64::from(n) * W;
    vec![
        Arch::new(first, session)
            .at(at(1))
            .turn("what is on the calendar for the rest of this week?", &reply("CALENDAR"))
            .turn(p, &reply("FIRSTREPLYTOP")),
        Arch::new(first + 1, session)
            .at(at(2))
            .successor()
            .user(p)
            .says(&reply("TAILOFA"))
            .calls(&reply("FORGETTINGNOW"), "axel_forget")
            .turn("thank you, and what is the weather doing tomorrow?", &reply("WEATHER")),
        Arch::new(first + 2, session)
            .at(at(3))
            .successor()
            .user(p)
            .says(&reply("TAILOFB"))
            .turn(p, &reply("ASKEDAGAININC")),
        Arch::new(first + 3, session).at(at(4)).turn(p, &reply("ASKEDAGAININD")),
        Arch::new(first + 4, session).at(at(5)).turn(p, &reply("ASKEDAGAININE")),
    ]
}

#[test]
fn t29_a_mark_reaches_the_file_before_and_the_window_after_and_no_further() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let files = five_windows(1, "s-marks", FORGET_PROMPT);
    // Chat T, with the same prompt text in the same windows and no forget call.
    let other = [
        Arch::new(21, "t-marks").at(T0 + W).turn(FORGET_PROMPT, &reply("OTHERCHATONE")),
        Arch::new(22, "t-marks").at(T0 + 3 * W).turn(FORGET_PROMPT, &reply("OTHERCHATTHREE")),
    ];
    index_in_order(&env, &mut run, &dir, &other, &[0, 1]);
    index_in_order(&env, &mut run, &dir, &files, &[0, 1, 2, 3, 4]);
    let conn = run.db();
    // No row of S holds A's exchange of P (the `head` mark, and the key).
    assert_eq!(positions(conn, &dir, &files[0]), ["x0.0"]);
    assert_eq!(in_tables(conn, "FIRSTREPLYTOP"), 0);
    // B's leading span yields nothing and its new exchange is a row.
    assert_eq!(positions(conn, &dir, &files[1]), ["x4.0"]);
    assert_eq!(in_tables(conn, "FORGETTINGNOW"), 0);
    // C's exchange with P's key is not a row: the window after, by key.
    assert!(rows(conn, &dir, &files[2]).is_empty());
    // D's and E's are rows: out of reach.
    assert_eq!(positions(conn, &dir, &files[3]), ["x0.0"]);
    assert_eq!(positions(conn, &dir, &files[4]), ["x0.0"]);
    assert_eq!(
        forgot_rows(conn),
        [("archive".to_string(), key_of(FORGET_PROMPT), 1, 0, files[1].id.clone())]
    );
    let units: Vec<i64> = files.iter().map(|a| forget_units(conn, &dir, a)).collect();
    assert_eq!(units, [1, 1, 1, 0, 0]);
    // Chat T is untouched.
    assert_eq!(positions(conn, &dir, &other[0]), ["x0.0"]);
    assert_eq!(positions(conn, &dir, &other[1]), ["x0.0"]);
    assert!(purge_pending(conn), "A's exchange was purged and left no mark for the log");
}

#[test]
fn t29_the_order_in_which_the_files_arrive_does_not_matter() {
    let dir = scope(KEY);
    let files = five_windows(1, "s-marks", FORGET_PROMPT);
    let end_state = |order: &[usize]| {
        let env = Env::new();
        let mut run = env.run();
        index_in_order(&env, &mut run, &dir, &files, order);
        let units: Vec<i64> = files.iter().map(|a| forget_units(run.db(), &dir, a)).collect();
        (all_rows(run.db()), units, forgot_rows(run.db()))
    };
    let in_order = end_state(&[0, 1, 2, 3, 4]);
    assert_eq!(in_order.0.len(), 4);
    for order in [[4, 3, 2, 1, 0], [1, 0, 2, 3, 4], [2, 1, 0, 4, 3]] {
        assert_eq!(end_state(&order), in_order, "indexed in the order {order:?}");
    }
}

#[test]
fn t29_the_same_with_a_token_in_the_prompt_and_with_the_prompt_withheld() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    // A token that `redact` blanks: the key is of the text as stored, in both files.
    let token = ["sk", "-", &"a1B2c3D4e5".repeat(4)].concat();
    let with_token = format!("{FORGET_PROMPT}, and the key was {token}");
    let files = five_windows(1, "s-token", &with_token);
    index_in_order(&env, &mut run, &dir, &files, &[0, 1, 2, 3, 4]);
    let conn = run.db();
    assert_eq!(positions(conn, &dir, &files[0]), ["x0.0"]);
    assert_eq!(positions(conn, &dir, &files[1]), ["x4.0"]);
    assert!(rows(conn, &dir, &files[2]).is_empty());
    assert_eq!(positions(conn, &dir, &files[3]), ["x0.0"]);

    // Stored as the placeholder in A and in B: B's mark has no key, and A's last unit, a
    // `Continued` after the withheld prompt, goes by `head` alone.
    let a = Arch::new(11, "s-withheld")
        .at(T0 + W)
        .turn("what is on the calendar for the rest of this week?", &reply("CALENDAR"))
        .withheld()
        .says(&reply("REPLYTOWITHHELD"));
    let b = Arch::new(12, "s-withheld")
        .at(T0 + 2 * W)
        .successor()
        .withheld()
        .says(&reply("TAILOFA"))
        .calls(&reply("FORGETTINGNOW"), "axel_forget")
        .turn("thank you, and what is the weather doing tomorrow?", &reply("WEATHER"));
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&a), &[0]);
    assert_eq!(positions(run.db(), &dir, &a), ["x0.0", "c3.0"]);
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&b), &[0]);
    let conn = run.db();
    assert_eq!(positions(conn, &dir, &a), ["x0.0"]);
    assert_eq!(in_tables(conn, "REPLYTOWITHHELD"), 0);
    assert_eq!(positions(conn, &dir, &b), ["x4.0"]);
    assert!(forgot_rows(conn).contains(&("archive".to_string(), String::new(), 1, 0, b.id.clone())));
}

#[test]
fn t29_a_tail_mark_takes_the_leading_continued_unit_of_the_file_after_and_parts_are_neighbours() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let asked = "please forget what I said about the salary of my colleague";
    // A' ends with a span that holds the call and more assistant text.
    let ends_in_call = |n: u32, session: &str| {
        Arch::new(n, session)
            .at(T0 + W)
            .turn("what is on the calendar for the rest of this week?", &reply("CALENDAR"))
            .user(asked)
            .calls(&reply("FORGETTINGNOW"), "axel_forget")
            .says(&reply("MORETEXTAFTER"))
    };
    // B', its successor: new assistant text in its leading span, and no call.
    let runs_on = |n: u32, session: &str| {
        Arch::new(n, session)
            .at(T0 + 2 * W)
            .successor()
            .user(asked)
            .says(&reply("TAILOFA"))
            .says(&reply("STILLRUNNING"))
            .turn("thank you, and what is the weather doing tomorrow?", &reply("WEATHER"))
    };
    let (a, b) = (ends_in_call(1, "s-tail"), runs_on(2, "s-tail"));
    index_in_order(&env, &mut run, &dir, &[a.clone(), b.clone()], &[0, 1]);
    let conn = run.db();
    assert_eq!(positions(conn, &dir, &a), ["x0.0"]);
    assert_eq!(positions(conn, &dir, &b), ["x4.0"], "the turn running on past the seal is a row");
    assert_eq!(in_tables(conn, "STILLRUNNING"), 0);
    assert!(forgot_rows(conn).contains(&("archive".to_string(), key_of(asked), 0, 1, a.id.clone())));

    // With a human prompt after the call's span in A' (`tail = 0`), it is a row.
    let a = ends_in_call(3, "s-no-tail").turn("and now something else entirely, what time is it?", &reply("TIME"));
    let b = runs_on(4, "s-no-tail");
    index_in_order(&env, &mut run, &dir, &[a.clone(), b.clone()], &[0, 1]);
    let conn = run.db();
    assert_eq!(positions(conn, &dir, &b), ["c3.0", "x4.0"]);
    assert!(forgot_rows(conn).contains(&("archive".to_string(), key_of(asked), 0, 0, a.id.clone())));

    // Parts of one seal: part 1 ends with a prompt and a reply; part 2, one millisecond
    // later and counted from 0, begins with assistant text that holds the call.
    let part_1 = Arch::new(5, "s-parts")
        .at(T0 + W)
        .turn("what is on the calendar for the rest of this week?", &reply("CALENDAR"))
        .turn(asked, &reply("FIRSTREPLYTOP"));
    let part_2 = Arch::new(6, "s-parts")
        .at(T0 + W + 1)
        .calls(&reply("FORGETTINGNOW"), "axel_forget")
        .turn("thank you, and what is the weather doing tomorrow?", &reply("WEATHER"));
    index_in_order(&env, &mut run, &dir, &[part_1.clone(), part_2.clone()], &[0, 1]);
    let conn = run.db();
    assert_eq!(positions(conn, &dir, &part_1), ["x0.0"], "part 1's last exchange is still a row");
    assert_eq!(in_tables(conn, "FIRSTREPLYTOP"), 0);
    assert_eq!(positions(conn, &dir, &part_2), ["x1.0"]);
    assert!(forgot_rows(conn).contains(&("archive".to_string(), String::new(), 1, 0, part_2.id.clone())));
}

// ── Test 30: left-out turns are counted and can be brought back ─────────────

#[test]
fn t30_left_out_turns_are_counted_and_come_back_when_the_row_says_keep() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let a = Arch::new(1, "s-keep")
        .about("retry limit")
        .user("forget the thing about the first matter, the one from Monday")
        .calls(&reply("FIRSTFORGET"), "axel_forget")
        .about("upload queue")
        .user("and forget the note about the second matter as well, please")
        .calls(&reply("SECONDFORGET"), "memory_forget_2")
        .about("deploy window");
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&a), &[0]);
    let before = rows(run.db(), &dir, &a);
    assert_eq!(before.iter().map(|r| r.1.as_str()).collect::<Vec<_>>(), ["x0.0", "x4.0", "x8.0"]);
    assert_eq!(forget_units(run.db(), &dir, &a), 2);
    assert_eq!(forgot_rows(run.db()).len(), 2);
    let vids = |conn: &Connection| -> Vec<i64> {
        before.iter().map(|r| count(conn, &format!("SELECT vid FROM memory_ext WHERE mem_id = '{}'", r.0))).collect()
    };
    let old_vids = vids(run.db());

    // What `restore --forget-turns` does.
    tx(run.db(), || {
        run.db().execute("UPDATE archive_index SET keep_forget_turns = 1 WHERE archive_id = ?1", [&a.id]).unwrap();
        run.db().execute("DELETE FROM forgotten_turns WHERE archive_id = ?1", [&a.id]).unwrap();
    });
    run.scan();
    let conn = run.db();
    assert_eq!(positions(conn, &dir, &a), ["x0.0", "x4.0", "x8.0", "x2.0", "x6.0"]);
    assert_eq!(rows(conn, &dir, &a)[..3], before[..]);
    assert_eq!(vids(conn), old_vids, "a row that was there was written again");
    assert_eq!(forget_units(conn, &dir, &a), 0);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM forgotten_turns"), 0, "a restored file gave marks again");
    assert_eq!(count(conn, "SELECT keep_forget_turns FROM archive_index"), 1);

    // The scan after that has nothing to do.
    let newest = count(conn, "SELECT MAX(vid) FROM memory_ext");
    run.scan();
    assert_eq!(count(run.db(), "SELECT MAX(vid) FROM memory_ext"), newest);
    assert_eq!(run.errors(), 0);
}

// ── Test 31: twins are bounded ──────────────────────────────────────────────

#[test]
fn t31_a_hidden_capture_hides_its_twins_in_three_windows_and_no_further() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let windows = [2u32, 4, 5, 6, 8];
    let files: Vec<Arch> = windows
        .iter()
        .map(|n| Arch::new(*n, "s-bounded").at(T0 + i64::from(*n) * W).turn("continue", &reply(&format!("WINDOW{n}"))))
        .collect();
    // A hidden capture with the prompt "continue", made in window 5.
    capture_keyed(&run, "mem_cap_h", &dir, "s-bounded", T0 + 4 * W + W / 2, Some("cli"), &key_of("continue"));
    // As they were sealed, one scan each. Window 2, indexed while it is the newest known,
    // takes the later hidden capture for one of the window that is still open; when
    // window 4 arrives its twin is judged again and is no twin (ruling 1c).
    env.put_live(&dir, &files[0]);
    run.scan();
    assert_eq!(hidden_by(run.db(), &dir, &files[0])[0].1.as_deref(), Some("twin:mem_cap_h"));
    index_in_order(&env, &mut run, &dir, &files, &[1, 2, 3, 4]);
    let conn = run.db();
    for (n, a) in windows.iter().zip(&files) {
        let expect = if matches!(n, 4..=6) { Some("twin:mem_cap_h".to_string()) } else { None };
        assert_eq!(hidden_by(conn, &dir, a), [("x0.0".to_string(), expect)], "window {n}");
        assert_eq!(unmatched_hidden(conn, &dir, a), 0);
    }
    assert!(has(conn, "mem_cap_h"));

    // `store::hide` of a capture marks rows of the chat's newest archive only.
    capture_keyed(&run, "mem_cap_late", &dir, "s-bounded", T0 + 8 * W + W / 2, None, &key_of("continue"));
    assert_eq!(tx(conn, || store::hide(conn, "mem_cap_late", "cli").unwrap()), 2);
    assert_eq!(hidden_by(conn, &dir, &files[4])[0].1.as_deref(), Some("twin:mem_cap_late"));
    assert_eq!(hidden_by(conn, &dir, &files[0])[0].1, None);
}

#[test]
fn t31_a_hidden_capture_whose_key_no_unit_has_hides_nothing_else_and_is_counted() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    capture_keyed(&run, "mem_cap_h", &dir, "s-bounded", T0 - 100_000, Some("cli"), &key_of("a prompt the archive does not hold"));
    let a = Arch::new(1, "s-bounded").about("retry limit").about("upload queue");
    let later = Arch::new(2, "s-bounded").at(T0 + W).about("deploy window");
    index_in_order(&env, &mut run, &dir, &[a.clone(), later.clone()], &[0, 1]);
    let conn = run.db();
    assert!(hidden_by(conn, &dir, &a).iter().all(|(_, by)| by.is_none()));
    assert!(hidden_by(conn, &dir, &later).iter().all(|(_, by)| by.is_none()));
    // Counted by the archive of its own window, and by no other.
    assert_eq!((unmatched_hidden(conn, &dir, &a), unmatched_hidden(conn, &dir, &later)), (1, 0));
}

// ── Test 33 (a): two marks without a key ────────────────────────────────────

#[test]
fn t33a_two_marks_of_one_file_without_a_key_are_one_row_with_head_and_tail() {
    let dir = scope(KEY);
    let before = Arch::new(1, "s-two")
        .at(T0 + W)
        .turn("what is on the calendar for the rest of this week?", &reply("CALENDAR"))
        .turn("there is a thing I should not have told you, about my brother", &reply("LASTBEFORE"));
    // Its leading span holds a forget call, and its last span, opened by a withheld
    // prompt, holds another.
    let marks = Arch::new(2, "s-two")
        .at(T0 + 2 * W)
        .calls(&reply("FIRSTCALL"), "axel_forget")
        .withheld()
        .calls(&reply("SECONDCALL"), "axel_forget");
    let after = Arch::new(3, "s-two")
        .at(T0 + 3 * W)
        .says(&reply("LEADINGAFTER"))
        .turn("thank you, and what is the weather doing tomorrow?", &reply("WEATHER"));
    let files = [before, marks, after];
    for order in [[0usize, 2, 1], [1, 0, 2], [0, 1, 2]] {
        let env = Env::new();
        let mut run = env.run();
        index_in_order(&env, &mut run, &dir, &files, &order);
        let conn = run.db();
        assert_eq!(
            forgot_rows(conn),
            [("archive".to_string(), String::new(), 1, 1, files[1].id.clone())],
            "order {order:?}"
        );
        assert_eq!(positions(conn, &dir, &files[0]), ["x0.0"], "order {order:?}");
        assert!(rows(conn, &dir, &files[1]).is_empty());
        assert_eq!(positions(conn, &dir, &files[2]), ["x1.0"], "order {order:?}");
        assert_eq!(in_tables(conn, "LASTBEFORE") + in_tables(conn, "LEADINGAFTER"), 0);
        assert_eq!((forget_units(conn, &dir, &files[0]), forget_units(conn, &dir, &files[2])), (1, 1));
    }
}

// ── Test 33 (b): a restored archive is spared ───────────────────────────────

#[test]
fn t33b_no_forget_mark_of_any_source_takes_a_row_of_a_restored_archive() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let yes = |n: u32, at: i64, mark: &str| Arch::new(n, "s-yes").at(T0 + at * W).turn("yes", &reply(mark));
    let forgets = |n: u32, at: i64| {
        Arch::new(n, "s-yes")
            .at(T0 + at * W)
            .user("yes")
            .calls(&reply("FORGETTINGNOW"), "axel_forget")
            .turn("thank you, and what is the weather doing tomorrow?", &reply("WEATHER"))
    };
    let (a, b, c) = (yes(1, 1, "YESINA"), forgets(2, 3), yes(3, 4, "YESINC"));
    index_in_order(&env, &mut run, &dir, &[a.clone(), b.clone(), c.clone()], &[0, 1, 2]);
    assert!(rows(run.db(), &dir, &a).is_empty() && rows(run.db(), &dir, &c).is_empty());
    assert_eq!((forget_units(run.db(), &dir, &a), forget_units(run.db(), &dir, &c)), (1, 1));

    // `restore --forget-turns` on A, and a scan.
    tx(run.db(), || {
        run.db().execute("UPDATE archive_index SET keep_forget_turns = 1 WHERE archive_id = ?1", [&a.id]).unwrap();
        run.db().execute("DELETE FROM forgotten_turns WHERE archive_id = ?1", [&a.id]).unwrap();
    });
    run.scan();
    assert_eq!(positions(run.db(), &dir, &a), ["x0.0"], "the restored turn is not back");
    assert_eq!(forget_units(run.db(), &dir, &a), 0);

    // A fourth archive between A and B holds another forget turn with the prompt "yes".
    let between = forgets(4, 2);
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&between), &[0]);
    assert_eq!(positions(run.db(), &dir, &a), ["x0.0"], "an archive's mark took a row of a restored archive");
    // And the live call with that key.
    let purged = live_call(&run, "s-yes", T0 + 2 * W - 5_000, T0 + 2 * W - 1_000, Some(&key_of("yes")));
    let conn = run.db();
    assert!(purged.is_empty(), "{purged:?}");
    assert_eq!(positions(conn, &dir, &a), ["x0.0"], "the live call took a row of a restored archive");
    assert!(rows(conn, &dir, &c).is_empty());
    assert_eq!(in_tables(conn, "YESINC"), 0);
    assert_eq!(forget_units(conn, &dir, &a), 0);
}

// ── Test 34 (d): the next file's note ───────────────────────────────────────

#[test]
fn t34d_a_tail_mark_takes_the_leading_continued_unit_and_the_note_of_the_next_file() {
    let dir = scope(KEY);
    // A ends in a forget span with no human prompt after it.
    let a = Arch::new(1, "s-note")
        .at(T0 + W)
        .about("retry limit")
        .user("forget what I told you about the diagnosis, all of it")
        .calls(&reply("FORGETTINGNOW"), "axel_forget");
    // B follows, with a leading `Continued` unit and a working note.
    let b = Arch::new(2, "s-note")
        .at(T0 + 2 * W)
        .says(&reply("RUNNINGONINB"))
        .about("upload queue")
        .noted("Working on: the user asked to forget NOTERESTATESIT, done.");
    // The same B in a chat with no forget.
    let plain = {
        let mut plain = b.clone();
        plain.id = format!("{:032x}", 9);
        plain.logical = project::logical("t-note");
        plain
    };
    for order in [[0usize, 1], [1, 0]] {
        let env = Env::new();
        let mut run = env.run();
        index_in_order(&env, &mut run, &dir, std::slice::from_ref(&plain), &[0]);
        assert_eq!(positions(run.db(), &dir, &plain), ["c0.0", "x1.0", "n0.0"]);
        index_in_order(&env, &mut run, &dir, &[a.clone(), b.clone()], &order);
        let conn = run.db();
        assert_eq!(positions(conn, &dir, &a), ["x0.0"]);
        assert_eq!(positions(conn, &dir, &b), ["x1.0"], "order {order:?}");
        assert_eq!(forget_units(conn, &dir, &b), 2, "order {order:?}");
        assert_eq!(positions(conn, &dir, &plain), ["c0.0", "x1.0", "n0.0"]);
        // Only the other chat's rows hold the two texts.
        assert_eq!(in_tables(conn, "RUNNINGONINB"), 2);
        assert_eq!(in_tables(conn, "NOTERESTATESIT"), 2);
    }
}

// ── Three more of commit 13a's: paths that had no test ──────────────────────

#[test]
fn a_record_whose_created_ms_is_no_time_is_skipped_with_a_reason() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let a = Arch::new(1, "s-time").at(i64::MAX).about("retry limit");
    let good = Arch::new(2, "s-time").about("upload queue");
    index_in_order(&env, &mut run, &dir, &[a.clone(), good.clone()], &[0, 1]);
    let conn = run.db();
    assert_eq!(
        state(conn, &dir, &rel(&a.id)),
        Some(("skipped".to_string(), Some("created_ms is not a time".to_string())))
    );
    assert!(rows(conn, &dir, &a).is_empty());
    assert_eq!(positions(conn, &dir, &good), ["x0.0"]);
    run.scan();
    assert_eq!(run.errors(), 0);
}

#[test]
fn a_job_that_names_a_slot_file_indexes_it_and_one_with_another_id_writes_nothing() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = Arch::new(1, "s-slot").about("retry limit");
    std::fs::create_dir_all(env.archives().join(&dir)).unwrap();
    let slot = env.archives().join(&dir).join("007.json");
    std::fs::write(&slot, a.json(&dir)).unwrap();
    let mut run = env.run();

    // The slot holds another archive than the event names: the job leaves it to the scan.
    run.job(archived(&Arch::new(2, "s-slot"), KEY, Some(&slot), None, Some(2), Some(1)));
    assert_eq!((index_rows(run.db()), archive_memories(run.db())), (0, 0));

    run.job(archived(&a, KEY, Some(&slot), Some("s-slot"), Some(2), Some(1)));
    let conn = run.db();
    assert_eq!(state(conn, &dir, "007.json").map(|s| s.0).as_deref(), Some("indexed"));
    assert_eq!(positions(conn, &dir, &a), ["x0.0"]);
    assert_eq!(run.errors(), 0);
}

#[test]
fn a_file_that_panicked_is_read_again_by_a_binary_with_a_higher_indexer_version() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = Arch::new(1, "s-panic").about("retry limit");
    env.put_live(&dir, &a);
    {
        let mut run = env.run_with(TestHooks { panic_archive_id: Some(a.id.clone()), ..TestHooks::default() }, None);
        run.scan();
        assert_eq!(state(run.db(), &dir, &rel(&a.id)).unwrap().1.as_deref(), Some("panic"));
        // The row was written by an older indexer than this binary's.
        tx(run.db(), || {
            run.db()
                .execute("UPDATE archive_index SET indexer_version = ?1", [index_archives::INDEXER_VERSION - 1])
                .unwrap()
        });
    }
    let mut run = env.run();
    run.scan();
    let conn = run.db();
    assert_eq!(state_of(conn, &dir, &a), "indexed");
    assert_eq!(positions(conn, &dir, &a), ["x0.0"]);
    assert_eq!(
        count(conn, "SELECT indexer_version FROM archive_index"),
        index_archives::INDEXER_VERSION
    );
    assert_eq!((run.panics(), run.errors()), (0, 0));
}

#[test]
fn t33a_two_marks_with_one_key_keep_the_head_of_the_first_and_the_tail_of_the_second() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let before = Arch::new(1, "s-max")
        .at(T0 + W)
        .about("retry limit")
        .turn("tell me again what the plan for Friday was, in one line", &reply("LASTBEFORE"));
    // The leading span holds a call (key from the carried prompt, `head`), and the last
    // exchange, with the same prompt, holds another (`tail`): one row, both set.
    let marks = Arch::new(2, "s-max")
        .at(T0 + 2 * W)
        .successor()
        .user(FORGET_PROMPT)
        .says(&reply("TAILOFBEFORE"))
        .calls(&reply("FIRSTCALL"), "axel_forget")
        .user(FORGET_PROMPT)
        .calls(&reply("SECONDCALL"), "axel_forget");
    let after = Arch::new(3, "s-max").at(T0 + 3 * W).says(&reply("LEADINGAFTER")).about("upload queue");
    // The file with the marks first: its neighbours are judged from the row alone.
    index_in_order(&env, &mut run, &dir, &[before.clone(), marks.clone(), after.clone()], &[1, 0, 2]);
    let conn = run.db();
    assert_eq!(forgot_rows(conn), [("archive".to_string(), key_of(FORGET_PROMPT), 1, 1, marks.id.clone())]);
    assert_eq!(positions(conn, &dir, &before), ["x0.0"], "the head of the first mark was lost");
    assert_eq!(positions(conn, &dir, &after), ["x1.0"], "the tail of the second mark was lost");
}

// ── The turn's own recall-log row (review of commit 15) ─────────────────────

#[test]
fn the_forget_mark_takes_the_recall_log_rows_of_its_turn_and_of_no_other_chat_or_time() {
    let env = Env::new();
    let dir = scope(KEY);
    let run = env.run();
    let conn = run.db();
    let plant = |ts_ms: i64, logical: Option<&str>, session: Option<&str>, preview: &str| -> i64 {
        tx(conn, || {
            conn.execute(
                "INSERT INTO recall_log(ts_ms, session_id, logical, scope_dir, query_preview, query_hash, outcome,
                                        total_ms, detail)
                 VALUES (?1, ?2, ?3, ?4, ?5, 'feedfacefeedface', 'ok', 1.0, '{}')",
                params![ts_ms, session, logical, dir, preview],
            )
            .unwrap();
            conn.last_insert_rowid()
        })
    };
    let chat = project::logical("s-log");
    let other = project::logical("t-log");
    // Written at the start of the turn, before the model called anything.
    let of_turn = plant(T0, Some(&chat), Some("s-log"), "forget what I said about LOGGEDPROMPT");
    let by_session = plant(T0 + 10, None, Some("s-log"), "the same turn, known by its session only");
    let older = plant(T0 - 1, Some(&chat), Some("s-log"), "an older question of the same chat");
    let elsewhere = plant(T0, Some(&other), Some("t-log"), "another chat at the same moment");
    let there = |id: i64| count(conn, &format!("SELECT COUNT(*) FROM recall_log WHERE id = {id}")) == 1;

    assert!(live_call(&run, "s-log", T0, T0 + 30_000, None).is_empty());
    assert!(!there(of_turn), "the turn's prompt is still in the recall log");
    assert!(!there(by_session));
    assert!(there(older) && there(elsewhere));
    assert_eq!(in_tables(conn, "LOGGEDPROMPT"), 0);
    assert!(purge_pending(conn), "the log row went and nothing asks for the checkpoint");

    // A `cli` mark has one moment, and takes the log rows of that moment only.
    let at_it = plant(T0 + 50_000, Some(&chat), None, "the hidden turn's own recall");
    let after_it = plant(T0 + 50_001, Some(&chat), None, "a later question");
    tx(conn, || {
        index_archives::mark_forgotten_turn(conn, &chat, None, "cli", T0 + 50_000, T0 + 50_000, None).unwrap()
    });
    assert!(!there(at_it));
    assert!(there(after_it));
}

// ═════════════════════════════════════════════════════════════════════════════
// The fix round of wave 4 (rulings R-13, and R-13 round 2 on the review of 13b)
// ═════════════════════════════════════════════════════════════════════════════

// ── R-13 item 3: the checkpoint from the tick ───────────────────────────────

#[test]
fn the_tick_truncates_the_log_when_the_loop_is_never_idle() {
    let env = Env::new();
    let (mut run, _a) = indexed_then_forgotten(&env);
    assert!(purge_pending(run.db()));
    assert!(in_file(&env.wal(), WAL_MARKER) > 0, "the marker never was in the log: the test proves nothing");
    // No idle turn: only the tick.
    indexer::tick(&mut run.ctx);
    assert!(!purge_pending(run.db()), "the tick did not try the checkpoint");
    assert_eq!(in_file(&env.wal(), WAL_MARKER), 0, "the marker is in brain.r8-wal");
    assert_eq!(run.errors(), 0);
}

#[test]
fn a_writer_blocks_the_checkpoint_and_that_is_not_a_write_error() {
    let env = Env::new();
    let (mut run, _a) = indexed_then_forgotten(&env);
    let writer = brain_v2::connect(&env.brain(), Role::Cli).unwrap();
    brain_v2::begin_write(&writer).unwrap();
    run.idle_turn();
    assert!(purge_pending(run.db()), "the mark went though another connection held the write lock");
    assert_eq!(run.errors(), 0, "a blocked checkpoint was counted as a write error");
    assert_eq!(count(run.db(), "PRAGMA busy_timeout"), 5_000);
    writer.execute_batch("ROLLBACK").unwrap();
    run.idle_turn();
    assert!(!purge_pending(run.db()));
    assert_eq!(run.errors(), 0);
}

// ── R-13 item 4: hides carried over a replaced file; vectors removed ────────

#[test]
fn a_replaced_file_keeps_the_hide_of_the_row_at_the_same_position() {
    let env = Env::new();
    let dir = scope(KEY);
    let old = Arch::new(1, "s-carry")
        .turn("what colour is the build lamp on the wall today?", "the lamp is oldcrimsonlamp and has been since noon")
        .about("upload queue");
    env.put_live(&dir, &old);
    let mut run = env.run();
    run.scan();
    let first = rows(run.db(), &dir, &old)[0].0.clone();
    assert_eq!(tx(run.db(), || store::hide(run.db(), &first, "cli").unwrap()), 1);

    let mut new = Arch::new(1, "s-carry")
        .turn("what colour is the build lamp on the wall today?", "the lamp is newemeraldlamp, somebody fixed the build")
        .about("deploy window");
    new.digest = format!("{:064x}", 78);
    assert_ne!(old.json(&dir).len(), new.json(&dir).len());
    env.put_live(&dir, &new);
    run.scan();
    let conn = run.db();
    assert!(rows(conn, &dir, &new)[0].2.contains("newemeraldlamp"));
    assert_eq!(
        hidden_by(conn, &dir, &new),
        [("x0.0".to_string(), Some("cli".to_string())), ("x2.0".to_string(), None)],
        "replacing the file brought back a row a person hid"
    );
    assert_eq!(run.errors(), 0);
}

/// Guards what the owner would lose: a row he hid, in a file that was replaced, came back
/// as the twin of a hidden capture, and the twin mark is what a re-judging or a restore
/// clears.
#[test]
fn a_hide_carried_over_a_replace_wins_over_a_twin_and_outlives_rejudging_and_restore() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let last = |text: &str| Arch::new(8, "s-keephide").at(T0 + 8 * W).turn(HIDDEN_PROMPT, &reply(text)).about("upload queue");
    let old = last("OLDHIDDENTEXT");
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&old), &[0]);
    let first = rows(run.db(), &dir, &old)[0].0.clone();
    assert_eq!(tx(run.db(), || store::hide(run.db(), &first, "cli").unwrap()), 1);
    // A hidden capture of the same prompt, far back: inside the stretch of the newest file.
    capture_keyed(&run, "mem_cap_far", &dir, "s-keephide", T0 + W / 2, Some("cli"), &key_of(HIDDEN_PROMPT));

    // The file is replaced, with the same turn at the same position.
    let mut new = last("NEWHIDDENTEXTLONGER");
    new.digest = format!("{:064x}", 88);
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&new), &[0]);
    let held = |run: &Run| hidden_by(run.db(), &dir, &new);
    let kept = [("x0.0".to_string(), Some("cli".to_string())), ("x2.0".to_string(), None)];
    assert_eq!(held(&run), kept, "the owner's hide became a twin mark");

    // Files between: the capture falls out of the stretch and twins are judged again.
    let between: Vec<Arch> = [2u32, 4, 6]
        .iter()
        .map(|n| Arch::new(*n, "s-keephide").at(T0 + i64::from(*n) * W).about(&format!("matter {n}")))
        .collect();
    index_in_order(&env, &mut run, &dir, &between, &[0, 1, 2]);
    assert_eq!(held(&run), kept, "the re-judging cleared the owner's hide");
    // The capture is restored.
    assert_eq!(tx(run.db(), || store::restore(run.db(), "mem_cap_far").unwrap()), 1);
    assert_eq!(held(&run), kept, "the restore of the capture cleared the owner's hide");
    assert_eq!(run.errors(), 0);
}

#[test]
fn the_front_thread_is_told_to_remove_the_vectors_of_purged_rows() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = Arch::new(1, "s-vec").about("retry limit").about("upload queue");
    env.put_live(&dir, &a);
    let mut run = env.run();
    capture(&run, "mem_cap_covered", &dir, "s-vec", T0 - 60_000, None);
    let vid = |conn: &Connection, id: &str| count(conn, &format!("SELECT vid FROM memory_ext WHERE mem_id = '{id}'"));
    let stand_in = vid(run.db(), "mem_cap_covered");
    run.scan();
    assert_eq!(run.removed(), [stand_in], "the covered capture's vector was left in the array");

    let mut ids: Vec<i64> = rows(run.db(), &dir, &a).iter().map(|r| vid(run.db(), &r.0)).collect();
    ids.sort_unstable();
    assert_eq!(ids.len(), 2);
    // The forget of the archive takes both.
    env.put_tombstone(&dir, &a);
    run.job(forgotten(&a, KEY));
    assert_eq!(run.removed(), ids);
    // Nothing is sent for a transaction that purged nothing.
    run.scan();
    assert!(run.removed().is_empty());
}

// ── R-13 item 5: the job leaves the tree as it is ───────────────────────────

#[test]
fn the_archived_job_changes_nothing_under_context_archives() {
    let env = Env::new();
    let dir = scope(KEY);
    let a = Arch::new(1, "s-jobtree").about("retry limit");
    let b = Arch::new(2, "s-jobtree")
        .at(T0 + W)
        .user(FORGET_PROMPT)
        .calls(&reply("FORGETTINGNOW"), "axel_forget")
        .about("upload queue");
    let path = env.put_live(&dir, &a);
    env.put_live(&dir, &b);
    std::fs::write(env.archives().join(&dir).join("lock"), "").unwrap();
    let before = tree(&env.archives());

    let mut run = env.run();
    run.job(archived(&a, KEY, Some(&path), Some("s-jobtree"), Some(1), Some(1)));
    run.job(archived(&b, KEY, None, Some("s-jobtree"), Some(2), Some(1)));
    // The same event twice, and one for a file that is not there.
    run.job(archived(&b, KEY, None, Some("s-jobtree"), Some(2), Some(1)));
    run.job(archived(&Arch::new(9, "s-jobtree"), KEY, None, None, None, None));
    assert_eq!(positions(run.db(), &dir, &a), ["x0.0"]);
    assert_eq!(positions(run.db(), &dir, &b), ["x2.0"]);
    run.idle_turn();
    assert_eq!(tree(&env.archives()), before);
    assert_eq!(run.errors(), 0);
}

// ── Round 2, ruling 1: the order of indexing ────────────────────────────────

/// What the brain holds of these files, with every archive named by its place in seal
/// order: the rows `(file, unit_pos, hidden_by)`, `forget_units` of each file, and the
/// marks `(source, key, head, tail, file)`.
type Shape = (Vec<(usize, String, Option<String>)>, Vec<i64>, Vec<(String, String, i64, i64, usize)>);

fn shape(conn: &Connection, dir: &str, files: &[Arch]) -> Shape {
    let mut held = Vec::new();
    for (at, a) in files.iter().enumerate() {
        let mut of_file = hidden_by(conn, dir, a);
        of_file.sort();
        held.extend(of_file.into_iter().map(|(pos, by)| (at, pos, by)));
    }
    let units = files.iter().map(|a| forget_units(conn, dir, a)).collect();
    let mut marks: Vec<_> = forgot_rows(conn)
        .into_iter()
        .map(|(source, key, head, tail, id)| (source, key, head, tail, files.iter().position(|a| a.id == id).unwrap()))
        .collect();
    marks.sort();
    (held, units, marks)
}

fn four_windows() -> Vec<Arch> {
    let mut files = five_windows(1, "s-order", FORGET_PROMPT);
    files.truncate(4);
    files
}

/// The four files indexed one at a time as they were sealed.
fn sealed_in_order() -> Shape {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let files = four_windows();
    index_in_order(&env, &mut run, &dir, &files, &[0, 1, 2, 3]);
    let got = shape(run.db(), &dir, &files);
    // A keeps its first exchange, B its new one, C nothing, and D, out of reach, its own.
    assert_eq!(got.0.iter().map(|r| (r.0, r.1.as_str())).collect::<Vec<_>>(), [(0, "x0.0"), (1, "x4.0"), (3, "x0.0")]);
    assert_eq!(got.1, [1, 1, 1, 0]);
    got
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for rest in permutations(n - 1) {
        for at in 0..=rest.len() {
            let mut next = rest.clone();
            next.insert(at, n - 1);
            out.push(next);
        }
    }
    out
}

#[test]
fn one_scan_gives_the_same_rows_whatever_the_names_and_the_times_of_the_files() {
    let dir = scope(KEY);
    let expect = sealed_in_order();
    let orders = permutations(4);
    assert_eq!(orders.len(), 24);
    for names in &orders {
        for times in ["as sealed", "equal", "reversed"] {
            let env = Env::new();
            let mut run = env.run();
            // The scan lists by name, and a name is an id: file `i` in seal order gets
            // the id whose place among the names is `names[i]`.
            let mut files = four_windows();
            for (a, name) in files.iter_mut().zip(names) {
                a.id = format!("{:032x}", name + 1);
            }
            for (at, a) in files.iter().enumerate() {
                let path = env.put_live(&dir, a);
                let step = match times {
                    "as sealed" => at as i64,
                    "equal" => 0,
                    _ => -(at as i64),
                };
                set_mtime(&path, 1_600_000_000_000 + step * 5_000);
            }
            run.scan();
            assert_eq!(shape(run.db(), &dir, &files), expect, "names {names:?}, file times {times}");
            assert_eq!(run.errors(), 0);
        }
    }
}

#[test]
fn a_far_file_scanned_alone_before_the_files_between_gives_the_same_rows_as_seal_order() {
    // D is scanned alone; the next scan finds A, B and C. Each is indexed knowing the new
    // files of its scan sealed after it, so when B is indexed C stands between B and D,
    // and D is not in "the window after" B.
    let dir = scope(KEY);
    let expect = sealed_in_order();
    let env = Env::new();
    let mut run = env.run();
    let files = four_windows();
    env.put_live(&dir, &files[3]);
    run.scan();
    assert_eq!(positions(run.db(), &dir, &files[3]), ["x0.0"]);
    for a in &files[..3] {
        env.put_live(&dir, a);
    }
    run.scan();
    assert_eq!(shape(run.db(), &dir, &files), expect, "D was indexed before the files between");
    assert_eq!(run.errors(), 0);
}

// ── Round 3, item 4: what stands as a neighbour in the sorted pass ──────────

#[test]
fn a_record_whose_created_ms_is_no_time_stands_as_no_files_neighbour() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let key = key_of(FORGET_PROMPT);
    // A mark made long after the newest real file's seal (as in the 1b test).
    assert!(live_call(&run, "s-notime", T0 + 5 * W, T0 + 5 * W + 1_000, Some(&key)).is_empty());
    let real = Arch::new(1, "s-notime").at(T0).turn(FORGET_PROMPT, &reply("NEWESTREALFILE"));
    // Far ahead, and no time a row can hold: the indexer refuses it.
    let bogus = Arch::new(2, "s-notime").at(9_000_000_000_000_000_000).about("retry limit");
    env.put_live(&dir, &real);
    env.put_live(&dir, &bogus);
    run.scan();
    let conn = run.db();
    assert_eq!(state(conn, &dir, &rel(&bogus.id)).and_then(|s| s.1).as_deref(), Some("created_ms is not a time"));
    assert_eq!(
        positions(conn, &dir, &real),
        ["x0.0"],
        "a record the indexer refuses ended the stretch of the newest real file"
    );
    assert_eq!(run.errors(), 0);
}

#[test]
fn a_file_that_panics_in_the_sorted_pass_is_no_later_file_for_the_files_after_it() {
    // C panics. D, after it, is judged with C absent, as the brain will hold it: B's mark
    // reaches D's same-key turn, which is left out and counted. (B, before C, was judged
    // with C between; its own step 3b did not reach D, and D took the mark itself.)
    let env = Env::new();
    let dir = scope(KEY);
    let files = four_windows();
    let mut run = env.run_with(TestHooks { panic_archive_id: Some(files[2].id.clone()), ..TestHooks::default() }, None);
    for a in &files {
        env.put_live(&dir, a);
    }
    run.scan();
    let conn = run.db();
    assert_eq!(state(conn, &dir, &rel(&files[2].id)).and_then(|s| s.1).as_deref(), Some("panic"));
    assert_eq!(run.panics(), 1);
    assert_eq!(positions(conn, &dir, &files[0]), ["x0.0"]);
    assert_eq!(positions(conn, &dir, &files[1]), ["x4.0"]);
    assert!(rows(conn, &dir, &files[3]).is_empty(), "D was judged with the file that panicked between");
    assert_eq!(forget_units(conn, &dir, &files[3]), 1);
    assert_eq!(run.errors(), 0);
}

#[test]
fn a_skipped_file_read_again_is_indexed_in_seal_order_with_the_new_files() {
    // D is indexed while B cannot be read as a record. The next scan finds B whole and A
    // and C new: B is judged knowing C, which lies between it and D.
    let dir = scope(KEY);
    let expect = sealed_in_order();
    let env = Env::new();
    let mut run = env.run();
    let files = four_windows();
    let b = env.put_live(&dir, &files[1]);
    std::fs::write(&b, "x".repeat(files[1].json(&dir).len() + 40)).unwrap();
    env.put_live(&dir, &files[3]);
    run.scan();
    assert_eq!(state_of(run.db(), &dir, &files[1]), "skipped");
    assert_eq!(positions(run.db(), &dir, &files[3]), ["x0.0"]);
    for a in &files[..3] {
        env.put_live(&dir, a);
    }
    run.scan();
    assert_eq!(shape(run.db(), &dir, &files), expect, "B was read again before the files between");
    assert_eq!(run.errors(), 0);
}

#[test]
fn a_mark_whose_window_after_is_not_known_yet_takes_the_far_files_turn_and_counts_it() {
    // FINDING (reported, not hidden): "near" is judged from the files known at that
    // moment. D is scanned alone, then B alone, then A and C. When B is indexed nothing
    // says a file lies between B and D, so D stands in "the window after" B and step 3b of
    // B's mark takes D's turn with the same prompt. C's arrival repairs nothing: a unit
    // that was purged is read again only by `restore --forget-turns`. It is counted in D's
    // `forget_units`, and everything else is as in seal order.
    let dir = scope(KEY);
    let expect = sealed_in_order();
    let env = Env::new();
    let mut run = env.run();
    let files = four_windows();
    index_in_order(&env, &mut run, &dir, &files, &[3, 1]);
    assert!(rows(run.db(), &dir, &files[3]).is_empty());
    env.put_live(&dir, &files[0]);
    env.put_live(&dir, &files[2]);
    run.scan();
    let got = shape(run.db(), &dir, &files);
    assert_eq!(got.0, expect.0[..2], "only D's row is missing");
    assert_eq!(got.2, expect.2);
    assert!(rows(run.db(), &dir, &files[3]).is_empty());
    assert_eq!(got.1, [1, 1, 1, 1], "the unit D lost is not in forget_units");
    assert_eq!(run.errors(), 0);
}

#[test]
fn a_file_that_is_the_newest_known_takes_no_mark_made_long_after_its_seal() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let key = key_of(FORGET_PROMPT);
    // Marks made before any file of the chat is known: one five windows after the file's
    // seal, one inside the seal's own span.
    assert!(live_call(&run, "s-late", T0 + 5 * W, T0 + 5 * W + 1_000, Some(&key)).is_empty());
    assert!(live_call(&run, "s-soon", T0 + 5_000, T0 + 6_000, Some(&key)).is_empty());
    let late = Arch::new(1, "s-late").at(T0).turn(FORGET_PROMPT, &reply("FARFROMTHEMARK"));
    let soon = Arch::new(2, "s-soon").at(T0).turn(FORGET_PROMPT, &reply("RIGHTATTHEMARK"));
    index_in_order(&env, &mut run, &dir, &[late.clone(), soon.clone()], &[0, 1]);
    assert_eq!(positions(run.db(), &dir, &late), ["x0.0"], "files between it and the mark may simply not be known");
    assert!(rows(run.db(), &dir, &soon).is_empty());
    // A mark made now, with the file known, reaches it: the window after it is open.
    assert_eq!(live_call(&run, "s-late", T0 + 6 * W, T0 + 6 * W, Some(&key)).len(), 1);
    assert!(rows(run.db(), &dir, &late).is_empty());
}

#[test]
fn a_twin_whose_capture_fell_out_of_the_stretch_becomes_the_twin_of_one_inside_it() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let key = key_of("continue");
    capture_keyed(&run, "mem_cap_old", &dir, "s-repoint", T0 + W / 2, Some("pause"), &key);
    capture_keyed(&run, "mem_cap_near", &dir, "s-repoint", T0 + 7 * W + W / 2, Some("pause"), &key);
    let last = Arch::new(8, "s-repoint").at(T0 + 8 * W).turn("continue", &reply("WINDOWEIGHT"));
    let between: Vec<Arch> = [2u32, 4, 6]
        .iter()
        .map(|n| Arch::new(*n, "s-repoint").at(T0 + i64::from(*n) * W).about(&format!("matter {n}")))
        .collect();
    // Alone, window 8 reaches back to the beginning and takes the oldest hidden capture.
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&last), &[0]);
    assert_eq!(hidden_by(run.db(), &dir, &last)[0].1.as_deref(), Some("twin:mem_cap_old"));
    index_in_order(&env, &mut run, &dir, &between, &[0, 1, 2]);
    assert_eq!(
        hidden_by(run.db(), &dir, &last),
        [("x0.0".to_string(), Some("twin:mem_cap_near".to_string()))],
        "the old capture is three windows away; the near one hides the turn"
    );
    assert!(has(run.db(), "mem_cap_old") && has(run.db(), "mem_cap_near"));
}

// ── Round 2, rulings 2 and 3: the recall log ────────────────────────────────

fn plant_log(conn: &Connection, dir: &str, logical: &str, ts_ms: i64, hash: &str, preview: &str) -> i64 {
    tx(conn, || {
        conn.execute(
            "INSERT INTO recall_log(ts_ms, session_id, logical, scope_dir, query_preview, query_hash, outcome,
                                    total_ms, detail)
             VALUES (?1, NULL, ?2, ?3, ?4, ?5, 'ok', 1.0, '{}')",
            params![ts_ms, logical, dir, preview, hash],
        )
        .unwrap();
        conn.last_insert_rowid()
    })
}

fn logged(conn: &Connection, id: i64) -> bool {
    count(conn, &format!("SELECT COUNT(*) FROM recall_log WHERE id = {id}")) == 1
}

#[test]
fn a_mark_read_from_an_archive_takes_the_log_row_of_its_prompt() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let files = four_windows();
    let chat = files[0].logical.clone();
    let other = project::logical("t-order");
    let key = key_of(FORGET_PROMPT);
    let conn = run.db();
    // The forgotten turn began in A's window and its forget call is in B (sealed at 2 W).
    let of_turn = plant_log(conn, &dir, &chat, T0 + W - 5_000, &key, "forget that I told you about LOGGEDOFFER");
    let other_prompt = plant_log(conn, &dir, &chat, T0 + W - 4_000, "0123456789abcdef", "another question");
    let other_chat = plant_log(conn, &dir, &other, T0 + W - 5_000, &key, "the same words in another chat");
    let far_later = plant_log(conn, &dir, &chat, T0 + 9 * W, &key, "the same words, weeks later");
    // B alone: no row with the key exists, so no purge takes the log row on its way.
    index_in_order(&env, &mut run, &dir, &files, &[1]);
    let conn = run.db();
    assert!(!logged(conn, of_turn), "the prompt of a forgotten turn is still in the recall log");
    assert_eq!(in_tables(conn, "LOGGEDOFFER"), 0);
    assert!(logged(conn, other_prompt) && logged(conn, other_chat) && logged(conn, far_later));
    assert!(purge_pending(conn));
}

#[test]
fn the_live_call_takes_the_log_row_by_key_and_with_nothing_pending_the_newest_one() {
    let env = Env::new();
    let dir = scope(KEY);
    let run = env.run();
    let conn = run.db();
    let chat = project::logical("s-nopending");
    let other = project::logical("t-nopending");
    let key = key_of(FORGET_PROMPT);

    // Nothing pending: the front gives `from_ms` = the time of the call, later than the
    // turn's log row.
    let older = plant_log(conn, &dir, &chat, T0 - 60_000, "aaaaaaaaaaaaaaaa", "the turn before");
    let of_turn = plant_log(conn, &dir, &chat, T0 - 2_000, "bbbbbbbbbbbbbbbb", "forget what I said about NOPENDING");
    let elsewhere = plant_log(conn, &dir, &other, T0 - 1_000, "bbbbbbbbbbbbbbbb", "another chat, later");
    assert!(live_call(&run, "s-nopending", T0, T0, None).is_empty());
    assert!(!logged(conn, of_turn), "with no pending turn the turn's log row stays");
    assert!(logged(conn, older) && logged(conn, elsewhere));
    // A second forget in the same turn, and a retry of the first mark: the turn's row is
    // gone already, and the row of the turn before is not taken for it.
    assert!(live_call(&run, "s-nopending", T0 + 1_000, T0 + 1_000, None).is_empty());
    assert!(live_call(&run, "s-nopending", T0, T0 + 5_000, None).is_empty());
    assert!(logged(conn, older), "a second mark took the log row of an older turn");

    // By key: a row with the prompt's hash from before `from_ms` goes, and no other.
    let same_key = plant_log(conn, &dir, &chat, T0 + 20_000, &key, "forget that I told you about BYKEYONLY");
    let other_key = plant_log(conn, &dir, &chat, T0 + 20_001, "cccccccccccccccc", "a question in between");
    let after = plant_log(conn, &dir, &chat, T0 + 40_001, &key, "the same words after the mark's end");
    assert!(live_call(&run, "s-nopending", T0 + 30_000, T0 + 40_000, Some(&key)).is_empty());
    assert!(!logged(conn, same_key), "the log row with the key of the forgotten prompt stays");
    assert!(logged(conn, other_key) && logged(conn, after) && logged(conn, older));
}

#[test]
fn a_keyless_mark_after_a_keyed_one_in_the_same_turn_takes_no_older_log_row() {
    let env = Env::new();
    let dir = scope(KEY);
    let run = env.run();
    let conn = run.db();
    let chat = project::logical("s-keyedfirst");
    let key = key_of(FORGET_PROMPT);
    let older = plant_log(conn, &dir, &chat, T0 - 60_000, "aaaaaaaaaaaaaaaa", "the turn before");
    let of_turn = plant_log(conn, &dir, &chat, T0 - 2_000, &key, "forget that I told you about KEYEDFIRST");
    // The turn's mark with its key, from the hook's arrival: it takes the turn's row.
    assert!(live_call(&run, "s-keyedfirst", T0 - 3_000, T0 - 1_000, Some(&key)).is_empty());
    assert!(!logged(conn, of_turn));
    // A second forget in the same turn, with nothing pending: no key, after the first.
    assert!(live_call(&run, "s-keyedfirst", T0, T0, None).is_empty());
    assert!(logged(conn, older), "a keyless mark after a keyed one took the log row of the turn before");
}

#[test]
fn a_keyless_mark_that_found_its_turns_log_row_takes_no_other() {
    let env = Env::new();
    let dir = scope(KEY);
    let run = env.run();
    let conn = run.db();
    let chat = project::logical("s-foundone");
    let older = plant_log(conn, &dir, &chat, T0 - 60_000, "aaaaaaaaaaaaaaaa", "the turn before");
    let of_turn = plant_log(conn, &dir, &chat, T0 + 500, "bbbbbbbbbbbbbbbb", "forget what I said about FOUNDONE");
    // The first mark of the chat, whose interval holds the turn's log row.
    assert!(live_call(&run, "s-foundone", T0, T0 + 1_000, None).is_empty());
    assert!(!logged(conn, of_turn));
    assert!(logged(conn, older), "a mark that found its turn's row also took the newest one before it");
}

#[test]
fn a_mark_written_late_takes_no_log_row_from_after_its_turn() {
    let env = Env::new();
    let dir = scope(KEY);
    let run = env.run();
    let conn = run.db();
    let chat = project::logical("s-latemark");
    let of_turn = plant_log(conn, &dir, &chat, T0, "aaaaaaaaaaaaaaaa", "the forgotten turn");
    let at_end = plant_log(conn, &dir, &chat, T0 + 30_000, "bbbbbbbbbbbbbbbb", "at the mark's end");
    let next_turn = plant_log(conn, &dir, &chat, T0 + 30_001, "cccccccccccccccc", "the next turn");
    let much_later = plant_log(conn, &dir, &chat, T0 + 600_000, "dddddddddddddddd", "ten minutes later");
    // The front could not write the mark and writes it minutes later, with the turn's times.
    assert!(live_call(&run, "s-latemark", T0, T0 + 30_000, None).is_empty());
    assert!(!logged(conn, of_turn) && !logged(conn, at_end));
    assert!(logged(conn, next_turn) && logged(conn, much_later), "a late mark took the log rows of later turns");
    // And once more, as a retry does: nothing else goes.
    assert!(live_call(&run, "s-latemark", T0, T0 + 30_000, None).is_empty());
    assert!(logged(conn, next_turn) && logged(conn, much_later));
}

// ── Round 2, rulings 6 and 7 ────────────────────────────────────────────────

#[test]
fn two_chats_with_an_empty_logical_do_not_share_their_marks() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let mut marks = Arch::new(1, "s-empty-one")
        .at(T0)
        .user(FORGET_PROMPT)
        .calls(&reply("FORGETTINGNOW"), "axel_forget")
        .about("upload queue");
    let mut plain = Arch::new(2, "s-empty-two").at(T0 + 1_000).turn(FORGET_PROMPT, &reply("ANOTHERCHATSTURN"));
    marks.logical = String::new();
    plain.logical = String::new();
    index_in_order(&env, &mut run, &dir, &[marks.clone(), plain.clone()], &[0, 1]);
    assert_eq!(forgot_rows(run.db()).len(), 1);
    assert_eq!(positions(run.db(), &dir, &plain), ["x0.0"], "a chat with no `logical` took another chat's mark");
}

#[test]
fn a_replaced_file_gives_up_the_marks_and_the_keep_flag_of_the_file_before_it() {
    let env = Env::new();
    let dir = scope(KEY);
    let mut run = env.run();
    let old = Arch::new(1, "s-swap")
        .about("retry limit")
        .user(FORGET_PROMPT)
        .calls(&reply("OLDFORGET"), "axel_forget")
        .about("upload queue");
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&old), &[0]);
    assert_eq!(forgot_rows(run.db()).len(), 1);

    // Another file under the name, with no forget turn: the old file's mark is not its.
    let mut plain = Arch::new(1, "s-swap").about("deploy window");
    plain.digest = format!("{:064x}", 91);
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&plain), &[0]);
    assert_eq!(positions(run.db(), &dir, &plain), ["x0.0"]);
    assert!(forgot_rows(run.db()).is_empty(), "the mark of a file that is gone still leaves units out");

    // `restore --forget-turns` was asked for this file; then a third file takes the name,
    // with a forget turn of its own. The flag was not asked for that one.
    tx(run.db(), || run.db().execute("UPDATE archive_index SET keep_forget_turns = 1", []).unwrap());
    let mut third = Arch::new(1, "s-swap")
        .about("release notes")
        .user("forget the thing about the first matter, the one from Monday")
        .calls(&reply("THIRDFORGET"), "axel_forget")
        .about("rollback plan");
    third.digest = format!("{:064x}", 92);
    index_in_order(&env, &mut run, &dir, std::slice::from_ref(&third), &[0]);
    let conn = run.db();
    assert_eq!(positions(conn, &dir, &third), ["x0.0", "x4.0"], "the forget turn of a new file was indexed");
    assert_eq!(in_tables(conn, "THIRDFORGET"), 0);
    assert_eq!(count(conn, "SELECT keep_forget_turns FROM archive_index"), 0);
    assert_eq!(forgot_rows(conn).len(), 1);
    assert_eq!(run.errors(), 0);
}
