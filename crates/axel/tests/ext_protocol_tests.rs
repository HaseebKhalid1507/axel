//! Commit 12b, the runtime: section 7's tests 1, 2 (the half that needs a loop), 3, 4, 7
//! (first half), 8 to 11 and 13 to 16; section 3's test 15; and the rules of row 12b of
//! "Tests per commit". Everything lives in temp folders; the embedder is the hash one or
//! none; recall and the tools are stubs, so no test needs them to answer normally.

use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axel::brain_v2::{self, OpenOpts};
use axel::ext::config::ExtConfig;
use axel::ext::front::{self, FrontAnswer, FrontCall, FrontReply, FrontWork, MarkEnd};
use axel::ext::rpc::{write_frame, FrameReader};
use axel::ext::server::{self, degrade, recover, serve, BUSY_TEXT};
use axel::ext::tools_forum::NO_CALLER;
use axel::ext::types::{
    Boot, Caller, Carry, DegradedReason, ForgetMark, FrontState, HookEvent, HookOut, Job, Mode, ServeEnd, Status,
    TestHooks,
};
use axel::project_memory::{RecordLinks, Retention, ScopedDescriptor, Sensitivity};
use axel::recall::{Origin, Pick, RecallConfig};
use axel::retrieval::ScoreParts;
use axel::store::{self, Inserted, NewRow};
use axel::error::AxelError;
use axel::ext::types::IndexerCtx;
use axel::indexer;
use axel::indexer_start;
use axel::vindex::{VecDelta, VecItem, VecKey, VecScope};
use rusqlite::Connection;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use velocirag::embedder::Embedder;

// ── In-memory pipes ─────────────────────────────────────────────────────────

/// The reading end of a pipe. End of input when the writer is gone; a read that waits
/// longer than `patience` fails, so a test that expects an answer never hangs.
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

fn pipe(patience: Option<Duration>) -> (PipeWriter, PipeReader) {
    let (tx, rx) = mpsc::channel();
    (PipeWriter(tx), PipeReader { rx, buf: Vec::new(), pos: 0, patience })
}

// ── A host ──────────────────────────────────────────────────────────────────

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
}

/// Folders of one test: `<tmp>/brain/brain.r8`, `<tmp>/base`, `<tmp>/model` (empty).
struct Dirs {
    tmp: TempDir,
}

impl Dirs {
    fn new() -> Dirs {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("base")).unwrap();
        std::fs::create_dir_all(tmp.path().join("model")).unwrap();
        Dirs { tmp }
    }
    fn path(&self) -> &Path {
        self.tmp.path()
    }
    fn brain(&self) -> PathBuf {
        self.path().join("brain").join("brain.r8")
    }
    fn log(&self) -> PathBuf {
        self.path().join("brain").join("extension.log")
    }
    /// The `config` object of `initialize.params`.
    fn config(&self, embedder: &str) -> Value {
        self.config_for(&self.brain(), &self.path().join("base"), embedder)
    }
    fn config_for(&self, brain: &Path, base: &Path, embedder: &str) -> Value {
        json!({
            "brain": brain.to_string_lossy(),
            "synaps_base": base.to_string_lossy(),
            "model_dir": self.path().join("model").to_string_lossy(),
            "embedder": embedder,
        })
    }
    fn ext_config(&self, embedder: &str) -> ExtConfig {
        ExtConfig::from_initialize(&json!({"config": self.config(embedder)}), self.path())
    }
}

fn boot(embedder: Option<Embedder>, test: TestHooks) -> (Boot, Arc<Status>) {
    let status = Arc::new(Status::default());
    let boot = Boot { embedder, clock: Arc::new(now_ms), test: Arc::new(test), status: status.clone() };
    (boot, status)
}

/// `serve` on its own thread, with this test as the host.
struct Host {
    to: Option<PipeWriter>,
    from: FrameReader<BufReader<PipeReader>>,
    end: Receiver<ServeEnd>,
    status: Arc<Status>,
    next_id: u64,
}

impl Host {
    fn start(embedder: Option<Embedder>, test: TestHooks) -> Host {
        let (boot, status) = boot(embedder, test);
        let (to, stdin) = pipe(None);
        let (stdout, from) = pipe(Some(Duration::from_secs(20)));
        let (end_tx, end) = mpsc::channel();
        std::thread::spawn(move || {
            let _ = end_tx.send(serve(stdin, stdout, boot));
        });
        Host { to: Some(to), from: FrameReader::new(BufReader::new(from)), end, status, next_id: 0 }
    }

    /// Started, initialised, and with the indexer's first attempt at the brain behind it.
    fn ready(dirs: &Dirs, embedder: Option<Embedder>, test: TestHooks) -> Host {
        let kind = if embedder.is_some() { "hash" } else { "off" };
        let mut host = Host::start(embedder, test);
        let answer = host.call("initialize", json!({"config": dirs.config(kind)}));
        assert_eq!(answer["result"]["protocol_version"], 1);
        host.wait_until("the status row is written", |h| h.status.status_written_ms.load(Ordering::Relaxed) > 0);
        host
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.to.as_mut().unwrap().write_all(bytes).unwrap();
    }

    fn send(&mut self, body: &Value) {
        let mut frame = Vec::new();
        write_frame(&mut frame, body.to_string().as_bytes()).unwrap();
        self.raw(&frame);
    }

    fn read(&mut self) -> Value {
        let body = self.from.next_frame().expect("the process closed its output").expect("no answer in time");
        serde_json::from_slice(&body).unwrap()
    }

    /// One request and its answer, which must carry the request's id.
    fn call(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let answer = self.read();
        assert_eq!(answer["id"], id, "answer to another request: {answer}");
        answer
    }

    fn timed(&mut self, method: &str, params: Value) -> (Value, Duration) {
        let start = Instant::now();
        let answer = self.call(method, params);
        (answer, start.elapsed())
    }

    fn hook(&mut self, kind: &str, session: &str) -> Value {
        self.call("hook.handle", hook_params(kind, session))["result"].clone()
    }

    fn tool(&mut self, name: &str) -> Value {
        self.call(
            "tool.call",
            json!({"name": name, "input": {"query": "MARKER-7731 retry limit"}, "session_id": "s-tool", "cwd": null}),
        )
    }

    fn wait_until(&mut self, what: &str, mut done: impl FnMut(&mut Host) -> bool) {
        let start = Instant::now();
        while !done(self) {
            assert!(start.elapsed() < Duration::from_secs(15), "waited 15 s for: {what}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Asks a tool until the answer names a reason other than `starting`; returns the text.
    /// NO_CALLER is the real tool's answer while the mode is still Ready (the test host sends
    /// no `caller`), so it means "not degraded yet".
    fn wait_degraded(&mut self) -> String {
        let start = Instant::now();
        loop {
            let answer = self.tool("search");
            let text = answer["result"]["content"].as_str().unwrap_or("").to_string();
            if answer["result"]["is_error"] == true && !text.contains("starting") && text != NO_CALLER {
                return text;
            }
            assert!(start.elapsed() < Duration::from_secs(15), "never degraded: {answer}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Closes the input and returns what `serve` returned.
    fn close(mut self) -> ServeEnd {
        self.to = None;
        self.end.recv_timeout(Duration::from_secs(10)).expect("serve did not return at end of input")
    }
}

fn hook_params(kind: &str, session: &str) -> Value {
    json!({
        "kind": kind,
        "session_id": session,
        "cwd": null,
        "message": "MARKER-7731 what did we decide about the retry limit",
        "data": {
            "archive_id": "0123456789abcdef0123456789abcdef",
            "project_key": "p0123456789abcdef",
            "id": "0123456789abcdef0123456789abcdef",
            "kind": "archive",
            "window": 2,
            "part": 1
        }
    })
}

fn continues() -> Value {
    json!({"action": "continue"})
}

const HOOKS: [&str; 6] = [
    "before_message",
    "on_message_complete",
    "on_session_start",
    "on_session_end",
    "on_context_archived",
    "on_memory_forgotten",
];

fn sha256(path: &Path) -> String {
    hex::encode(Sha256::digest(std::fs::read(path).unwrap()))
}

fn read_log(dirs: &Dirs) -> String {
    std::fs::read_to_string(dirs.log()).unwrap_or_default()
}

fn wait_for_log(dirs: &Dirs, needle: &str) -> String {
    let start = Instant::now();
    loop {
        let log = read_log(dirs);
        if log.contains(needle) {
            return log;
        }
        assert!(start.elapsed() < Duration::from_secs(15), "the log never held {needle:?}:\n{log}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A version 2 brain made the way `init` makes one, recorded for the hash model.
fn make_brain(path: &Path) -> Connection {
    let opts = OpenOpts { agent: None, model: "hash-384", dim: 384, upgrade_v1: false };
    brain_v2::open_v2(path, &opts).unwrap().conn
}

fn blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn unit(i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 384];
    v[i % 384] = 1.0;
    v
}

/// A memory row and its `memory_ext` row, written with plain SQL (`store` is in this wave).
fn insert_memory(conn: &Connection, id: &str, project: Option<&str>, vector: Option<&[f32]>, hidden: bool) -> i64 {
    conn.execute(
        "INSERT INTO memories(id, category, topic, title, abstract_text, content, created, project_key)
         VALUES(?1, 'events', 'topic', ?2, '', ?3, '2025-01-01T00:00:00Z', ?4)",
        rusqlite::params![id, format!("title of {id}"), format!("the body of {id} about retry limits"), project],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO memory_ext(mem_id, origin_kind, content_hash, created_ms, hidden_ms, hidden_by, embed_model, embedding)
         VALUES(?1, 'legacy', ?1, 1000, ?2, ?3, ?4, ?5)",
        rusqlite::params![
            id,
            hidden.then_some(2000i64),
            hidden.then_some("cli"),
            vector.map(|_| "hash-384"),
            vector.map(blob)
        ],
    )
    .unwrap();
    conn.query_row("SELECT vid FROM memory_ext WHERE mem_id = ?1", [id], |r| r.get(0)).unwrap()
}

fn insert_document(conn: &Connection, doc_id: &str, metadata: &str, hidden: Option<&str>, i: usize) -> i64 {
    conn.execute(
        "INSERT INTO documents(doc_id, content, metadata, embedding, file_path, hidden)
         VALUES(?1, 'a note file', ?2, ?3, ?1, ?4)",
        rusqlite::params![doc_id, metadata, blob(&unit(i)), hidden],
    )
    .unwrap();
    conn.query_row("SELECT id FROM documents WHERE doc_id = ?1", [doc_id], |r| r.get(0)).unwrap()
}

fn status_row(path: &Path) -> Option<Value> {
    let conn = Connection::open(path).ok()?;
    let text: String = conn
        .query_row("SELECT value FROM context_data WHERE layer = 'axel' AND key = 'status'", [], |r| r.get(0))
        .ok()?;
    serde_json::from_str(&text).ok()
}

// ── Section 7, test 1 ───────────────────────────────────────────────────────

/// What a brain that cannot be opened must still give: `initialize`, every hook
/// `continue`, a tool the unavailable sentence. Returns the tool's text.
fn serves_without_a_brain(dirs: &Dirs, brain: &Path) -> String {
    let mut host = Host::start(None, TestHooks::default());
    let config = dirs.config_for(brain, &dirs.path().join("base"), "off");
    let answer = host.call("initialize", json!({"config": config, "plugin_root": "/nonexistent"}));
    assert_eq!(answer["result"]["protocol_version"], 1);
    assert!(answer["result"]["capabilities"]["tools"].is_array(), "the tool list: {answer}");

    let text = host.wait_degraded();
    assert!(text.starts_with("Axel is unavailable: "), "{text}");
    assert!(text.ends_with(". Synaps itself is not affected."), "{text}");
    assert_eq!(Mode::from_u8(2), Some(Mode::Degraded));
    let reason = DegradedReason::from_u8(host.status.reason.load(Ordering::SeqCst));
    assert_ne!(reason, DegradedReason::None);
    assert!(text.contains(reason.as_str()), "{text} does not name {reason:?}");

    for kind in HOOKS {
        assert_eq!(host.hook(kind, "s1"), continues(), "{kind}");
    }
    assert_eq!(host.close(), ServeEnd::InputClosed);
    text
}

#[test]
fn t01_initialize_is_answered_when_the_brain_path_is_a_directory() {
    let dirs = Dirs::new();
    let brain = dirs.path().join("a-directory");
    std::fs::create_dir_all(&brain).unwrap();
    serves_without_a_brain(&dirs, &brain);
    assert!(brain.is_dir());
    assert_eq!(std::fs::read_dir(&brain).unwrap().count(), 0, "nothing was put in the folder");
}

#[test]
fn t01_initialize_is_answered_when_the_brain_is_random_bytes_and_the_file_is_unchanged() {
    let dirs = Dirs::new();
    let brain = dirs.path().join("noise.r8");
    let noise: Vec<u8> = (0..8192u32).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    std::fs::write(&brain, noise).unwrap();
    let before = sha256(&brain);
    let text = serves_without_a_brain(&dirs, &brain);
    assert!(text.contains("BrainDamaged"), "{text}");
    assert_eq!(sha256(&brain), before, "the file of random bytes was changed");
}

#[test]
fn t01_initialize_is_answered_when_the_brain_cannot_be_written() {
    // The brain's folder is a file: nothing can be made there, whoever runs the test.
    let dirs = Dirs::new();
    let blocker = dirs.path().join("not-a-folder");
    std::fs::write(&blocker, b"x").unwrap();
    serves_without_a_brain(&dirs, &blocker.join("brain.r8"));
    assert_eq!(std::fs::read(&blocker).unwrap(), b"x");
}

// ── Section 7, tests 2 (the loop's half) and 3 ──────────────────────────────

#[test]
fn t02_an_unknown_method_with_an_id_is_minus_32601_and_bad_frames_do_not_end_the_loop() {
    let mut host = Host::start(None, TestHooks::default());
    // Before `initialize`, and between garbage.
    host.raw(b"garbage\r\n");
    host.raw(b"Content-Length: 9\r\n\r\nnot json!");
    host.raw(b"Content-Length: 0\r\n\r\n");
    host.send(&json!({"jsonrpc": "2.0", "method": "no.id.no.answer"}));
    let answer = host.call("no.such.method", json!({}));
    assert_eq!(answer["error"]["code"], -32601, "{answer}");
    assert!(answer.get("result").is_none());

    // A hook and a tool before `initialize` are answered too.
    assert_eq!(host.hook("before_message", "s1"), continues());
    let tool = host.tool("search");
    assert_eq!(tool["result"]["is_error"], true, "{tool}");
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

#[test]
fn t03_every_hook_kind_and_an_unknown_one_answer_continue() {
    let dirs = Dirs::new();
    let mut host = Host::ready(&dirs, None, TestHooks::default());
    for kind in HOOKS.iter().chain(["on_something_new"].iter()) {
        assert_eq!(host.hook(kind, "s1"), continues(), "{kind}");
    }
    // Events that do not validate are answered all the same.
    let bad = host.call("hook.handle", json!({"kind": "on_context_archived", "data": {"archive_id": "../x"}}));
    assert_eq!(bad["result"], continues());
    let bad = host.call("hook.handle", json!({"kind": "on_memory_forgotten", "data": null}));
    assert_eq!(bad["result"], continues());
    let none = host.call("hook.handle", Value::Null);
    assert_eq!(none["result"], continues());

    // The mode is Ready: a tool reaches the dispatch. The test host sends no `caller`, so
    // the real tool answers NO_CALLER (commit 18a), which only the dispatch can say.
    let tool = host.tool("search");
    assert_eq!(tool["result"]["content"], NO_CALLER, "{tool}");
    assert_eq!(host.status.hook_misses.load(Ordering::Relaxed), 0);
    assert_eq!(host.status.panics.load(Ordering::Relaxed), 0);

    let row = status_row(&dirs.brain()).expect("the status row");
    assert_eq!(row["mode"], "ready");
    assert_eq!(row["reason"], "None");
    assert!(row["written_ms"].as_i64().unwrap() >= row["started_ms"].as_i64().unwrap());
    assert!(row["started_ms"].as_i64().unwrap() > 0);
    // The scan is real since commit 13a: the row says how far it is when the row is written,
    // which may be before the first tick (`pending`) or after it (`running`, `done`).
    let scan = row["scan"].as_str().expect("scan is a string");
    assert!(matches!(scan, "pending" | "running" | "done"), "scan = {scan}");
    assert_eq!(row["forum"]["declared"], true);
    assert_eq!(row["forum"]["core_owns"], false);
    assert_eq!(row["embed_stopped"], false);
    assert_eq!(row["model_loaded"], false);
    assert_eq!(row["model_matches"], Value::Null, "no model is running: {row}");

    host.send(&json!({"jsonrpc": "2.0", "id": 900, "method": "shutdown"}));
    assert_eq!(host.read()["id"], 900);
    assert_eq!(host.end.recv_timeout(Duration::from_secs(5)).unwrap(), ServeEnd::Shutdown);
}

// ── Section 7, test 4; section 3, test 14 ───────────────────────────────────

// SPEC: 4014-4017 give 250 ms and 20 ms. The second is asserted as 100 ms and the first as
// 400 ms: this file runs beside other builds, and the thing tested (answered at the budget,
// not after the 3 s the front thread sleeps) is an order of magnitude away from either.
#[test]
fn t04_a_stuck_front_thread_costs_one_budget_then_nothing_and_opens_the_breaker() {
    let dirs = Dirs::new();
    let mut host = Host::ready(&dirs, None, TestHooks { front_sleep_ms: 3000, ..TestHooks::default() });

    let (first, took) = host.timed("hook.handle", hook_params("before_message", "s1"));
    assert_eq!(first["result"], continues());
    assert!(took >= Duration::from_millis(140), "answered before the budget: {took:?}");
    assert!(took < Duration::from_millis(400), "the first call took {took:?}");
    assert!(!host.status.breaker_open.load(Ordering::Relaxed));

    let (second, took) = host.timed("hook.handle", hook_params("before_message", "s1"));
    assert_eq!(second["result"], continues());
    assert!(took < Duration::from_millis(100), "the second call was forwarded: {took:?}");
    assert!(!host.status.breaker_open.load(Ordering::Relaxed), "two misses do not open the breaker");

    let (third, _) = host.timed("hook.handle", hook_params("before_message", "s1"));
    assert_eq!(third["result"], continues());
    assert!(host.status.breaker_open.load(Ordering::Relaxed), "the third miss opens the breaker");
    assert_eq!(host.status.breaker_trips.load(Ordering::Relaxed), 1);

    let (next, took) = host.timed("hook.handle", hook_params("before_message", "s1"));
    assert_eq!(next["result"], continues());
    assert!(took < Duration::from_millis(100), "{took:?}");
    assert_eq!(host.status.hook_misses.load(Ordering::Relaxed), 4);
    assert_eq!(host.status.breaker_trips.load(Ordering::Relaxed), 1);

    let (tool, took) = {
        let start = Instant::now();
        let answer = host.tool("search");
        (answer, start.elapsed())
    };
    assert_eq!(tool["result"]["is_error"], true, "{tool}");
    assert_eq!(tool["result"]["content"], BUSY_TEXT);
    assert!(took < Duration::from_millis(100), "the tool was forwarded: {took:?}");

    // Notify-only events never wait for the front thread.
    let (event, took) = host.timed("hook.handle", hook_params("on_context_archived", "s1"));
    assert_eq!(event["result"], continues());
    assert!(took < Duration::from_millis(100), "{took:?}");

    host.send(&json!({"jsonrpc": "2.0", "id": 901, "method": "shutdown"}));
    assert_eq!(host.read()["id"], 901);
    assert_eq!(host.end.recv_timeout(Duration::from_secs(5)).unwrap(), ServeEnd::Shutdown);
}

// ── Section 7, test 7 (first half) and test 8 ───────────────────────────────

#[test]
fn t07_a_brain_from_the_future_degrades_and_the_file_is_unchanged() {
    let dirs = Dirs::new();
    {
        let conn = make_brain(&dirs.brain());
        conn.execute_batch("PRAGMA user_version = 99; PRAGMA wal_checkpoint(TRUNCATE);").unwrap();
    }
    let before = sha256(&dirs.brain());

    let mut host = Host::start(Some(Embedder::hashing()), TestHooks::default());
    host.call("initialize", json!({"config": dirs.config("hash")}));
    let text = host.wait_degraded();
    assert_eq!(text, "Axel is unavailable: BrainTooNew. Synaps itself is not affected.");
    assert_eq!(DegradedReason::from_u8(host.status.reason.load(Ordering::SeqCst)), DegradedReason::BrainTooNew);
    assert_eq!(host.hook("before_message", "s1"), continues());
    assert_eq!(host.hook("on_context_archived", "s1"), continues());
    assert_eq!(host.close(), ServeEnd::InputClosed);

    assert_eq!(sha256(&dirs.brain()), before, "a brain at version 99 was written");
    // Wave 3 ruling 12: nor was anything written beside it, to wait there for a checkpoint.
    let mut wal = dirs.brain().into_os_string();
    wal.push("-wal");
    assert_eq!(std::fs::metadata(&wal).map(|m| m.len()).unwrap_or(0), 0, "the log beside the brain has content");
    assert_eq!(host_status_written(&dirs), None);
}

fn host_status_written(dirs: &Dirs) -> Option<Value> {
    let conn = Connection::open_with_flags(dirs.brain(), rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).ok()?;
    conn.query_row("SELECT value FROM context_data WHERE layer = 'axel' AND key = 'status'", [], |r| {
        r.get::<_, String>(0)
    })
    .ok()
    .and_then(|t| serde_json::from_str(&t).ok())
}

#[test]
fn t08_a_brain_recorded_for_another_base_degrades_and_no_row_is_written() {
    let dirs = Dirs::new();
    let base_a = dirs.path().join("base");
    let base_b = dirs.path().join("base-b");
    std::fs::create_dir_all(&base_b).unwrap();

    // Served once with base A: the brain records it.
    {
        let mut host = Host::ready(&dirs, None, TestHooks::default());
        assert_eq!(host.hook("before_message", "s1"), continues());
        assert_eq!(host.close(), ServeEnd::InputClosed);
    }
    let recorded: String = {
        let conn = Connection::open(dirs.brain()).unwrap();
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);").unwrap();
        conn.query_row("SELECT value FROM context_data WHERE layer = 'axel' AND key = 'synaps_base'", [], |r| r.get(0))
            .unwrap()
    };
    assert_eq!(PathBuf::from(&recorded), base_a.canonicalize().unwrap());
    let before = sha256(&dirs.brain());
    let row_before = status_row(&dirs.brain()).unwrap();

    // Served with base B.
    let mut host = Host::start(None, TestHooks::default());
    host.call("initialize", json!({"config": dirs.config_for(&dirs.brain(), &base_b, "off")}));
    let text = host.wait_degraded();
    assert_eq!(text, "Axel is unavailable: OtherBase. Synaps itself is not affected.");
    assert_eq!(host.hook("before_message", "s1"), continues());
    assert_eq!(host.hook("on_context_archived", "s1"), continues());
    assert_eq!(host.hook("on_memory_forgotten", "s1"), continues());
    assert_eq!(host.close(), ServeEnd::InputClosed);

    assert_eq!(sha256(&dirs.brain()), before, "a brain of another base was written");
    assert_eq!(status_row(&dirs.brain()).unwrap(), row_before, "the status row is the first process's");
}

// ── Section 7, tests 9 and 10 ───────────────────────────────────────────────

#[test]
fn t09_a_handler_that_panics_is_an_error_answer_and_the_next_call_works() {
    let dirs = Dirs::new();
    let test = TestHooks { panic_tool: Some("search".to_string()), ..TestHooks::default() };
    let mut host = Host::ready(&dirs, None, test);

    let answer = host.tool("search");
    assert_eq!(answer["error"]["code"], -32603, "{answer}");
    assert!(answer.get("result").is_none());
    assert_eq!(host.status.panics.load(Ordering::Relaxed), 1);

    // The next calls work: another tool reaches the dispatch, a hook is answered.
    let other = host.tool("remember");
    assert_eq!(other["result"]["content"], NO_CALLER, "{other}");
    assert_eq!(host.hook("before_message", "s1"), continues());
    assert_eq!(host.status.hook_misses.load(Ordering::Relaxed), 0);

    let log = wait_for_log(&dirs, "panic at ");
    let line = log.lines().find(|l| l.contains("panic at ")).unwrap();
    let at = line.split("panic at ").nth(1).unwrap();
    let (file, number) = at.rsplit_once(':').expect("file:line");
    assert!(file.ends_with("tools.rs"), "{line}");
    assert!(number.parse::<u32>().is_ok(), "{line}");
    assert!(!log.contains("MARKER-7731"), "the log holds text of the call");
    assert!(!log.contains("test switch"), "the log holds the panic's message");
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

#[test]
fn t10_the_log_is_renamed_at_its_limit() {
    let dirs = Dirs::new();
    let test = TestHooks { log_limit_bytes: Some(4096), ..TestHooks::default() };
    let mut host = Host::ready(&dirs, None, test);
    // Each skipped frame is one short line: an error kind, never the frame.
    for _ in 0..200 {
        host.raw(b"Content-Length: 16\r\n\r\nMARKER-7731 !!!!");
    }
    assert_eq!(host.hook("before_message", "s1"), continues());

    let old = dirs.path().join("brain").join("extension.log.1");
    assert!(old.is_file(), "no extension.log.1");
    let old_len = std::fs::metadata(&old).unwrap().len();
    assert!(old_len > 4096 && old_len < 4096 + 256, "extension.log.1 is {old_len} bytes");
    let len = std::fs::metadata(dirs.log()).map(|m| m.len()).unwrap_or(0);
    assert!(len < 4096 + 256, "extension.log is {len} bytes");
    let both = format!("{}{}", std::fs::read_to_string(&old).unwrap(), read_log(&dirs));
    assert!(both.contains("frame skipped: NotJson"));
    assert!(!both.contains("MARKER-7731"));
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

// ── Section 7, test 11 ──────────────────────────────────────────────────────

#[test]
fn t11_shutdown_returns_within_500_ms_with_a_bulk_embedding_job_queued() {
    let dirs = Dirs::new();
    {
        let conn = make_brain(&dirs.brain());
        for i in 0..100 {
            insert_memory(&conn, &format!("mem_{i:03}"), None, None, false);
        }
    }
    // The indexer sleeps 1.5 s before each job, so the embedded batch that the embed thread
    // hands back is still in the indexer's hands when `shutdown` arrives.
    let test = TestHooks { indexer_sleep_ms: 1500, ..TestHooks::default() };
    let mut host = Host::ready(&dirs, Some(Embedder::hashing()), test);
    host.wait_until("the pump has given rows to the embed thread", |h| {
        h.status.vectors_waiting.load(Ordering::Relaxed) == 100
    });
    std::thread::sleep(Duration::from_millis(100));

    let start = Instant::now();
    host.send(&json!({"jsonrpc": "2.0", "id": 902, "method": "shutdown"}));
    assert_eq!(host.read()["id"], 902);
    let end = host.end.recv_timeout(Duration::from_secs(10)).expect("serve did not return");
    let took = start.elapsed();
    assert_eq!(end, ServeEnd::Shutdown);
    assert!(took < Duration::from_millis(500), "shutdown took {took:?}");

    // Nothing of the batch is written, not even once the indexer has woken with the job in
    // its hands (in a test the process goes on; a real one has exited by then).
    std::thread::sleep(Duration::from_millis(1800));
    let conn = Connection::open(dirs.brain()).unwrap();
    let waiting: i64 =
        conn.query_row("SELECT COUNT(*) FROM memory_ext WHERE embedding IS NULL", [], |r| r.get(0)).unwrap();
    assert_eq!(waiting, 100);
}

// ── Section 7, tests 13 and 14 ──────────────────────────────────────────────

#[test]
fn t13_an_indexer_that_ends_degrades_the_process_and_every_frame_is_still_answered() {
    let dirs = Dirs::new();
    let test = TestHooks { indexer_stop_after_jobs: Some(1), ..TestHooks::default() };
    let mut host = Host::ready(&dirs, None, test);
    assert_eq!(host.tool("search")["result"]["content"], NO_CALLER, "the mode is Ready before the job");

    assert_eq!(host.hook("on_context_archived", "s1"), continues());
    let text = host.wait_degraded();
    assert_eq!(text, "Axel is unavailable: IndexerStopped. Synaps itself is not affected.");
    assert_eq!(
        DegradedReason::from_u8(host.status.reason.load(Ordering::SeqCst)),
        DegradedReason::IndexerStopped
    );
    let at = host.status.reason_place().expect("a file and line").to_string();
    let (file, line) = at.rsplit_once(':').unwrap();
    assert!(file.ends_with(".rs") && line.parse::<u32>().is_ok(), "{at}");

    assert_eq!(host.hook("before_message", "s1"), continues());
    assert_eq!(host.hook("on_context_archived", "s1"), continues());
    assert_eq!(host.hook("on_memory_forgotten", "s1"), continues());
    assert_eq!(host.status.hook_misses.load(Ordering::Relaxed), 0);
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

#[test]
fn t13_a_job_that_panics_does_not_end_the_indexer() {
    let dirs = Dirs::new();
    let test = TestHooks { panic_on_forgotten_job: true, ..TestHooks::default() };
    let mut host = Host::ready(&dirs, None, test);

    assert_eq!(host.hook("on_memory_forgotten", "s1"), continues());
    host.wait_until("the job has panicked", |h| h.status.panics.load(Ordering::Relaxed) == 1);
    let log = wait_for_log(&dirs, "panic at ");
    assert!(log.contains("indexer.rs:"), "{log}");
    assert!(!log.contains("test switch"), "{log}");

    // The next job runs, and the indexer is still there for the one after.
    for _ in 0..3 {
        assert_eq!(host.hook("on_context_archived", "s1"), continues());
        std::thread::sleep(Duration::from_millis(50));
    }
    assert_eq!(host.hook("on_memory_forgotten", "s1"), continues());
    host.wait_until("the second job has panicked", |h| h.status.panics.load(Ordering::Relaxed) == 2);
    assert_eq!(host.tool("search")["result"]["content"], NO_CALLER, "the mode is still Ready");
    assert_eq!(DegradedReason::from_u8(host.status.reason.load(Ordering::SeqCst)), DegradedReason::None);
    assert!(!read_log(&dirs).contains("indexer stopped"));
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

#[test]
fn t14_a_front_thread_that_hangs_ends_serve_with_front_stuck() {
    let dirs = Dirs::new();
    let test = TestHooks { front_sleep_ms: 2000, front_hang_limit_ms: Some(300), ..TestHooks::default() };
    let mut host = Host::ready(&dirs, None, test);

    let start = Instant::now();
    let (answer, took) = host.timed("hook.handle", hook_params("before_message", "s1"));
    assert_eq!(answer["result"], continues());
    assert!(took >= Duration::from_millis(140) && took < Duration::from_millis(400), "{took:?}");

    let end = host.end.recv_timeout(Duration::from_secs(5)).expect("serve did not return");
    assert_eq!(end, ServeEnd::FrontStuck);
    assert!(start.elapsed() < Duration::from_secs(1), "{:?}", start.elapsed());

    let log = read_log(&dirs);
    let line = log.lines().find(|l| l.contains("front stuck in ")).expect("the log line");
    assert!(line.ends_with("front stuck in hook.handle before_message"), "{line}");
    assert!(!log.contains("MARKER-7731"));
}

// ── Section 7, test 15 ──────────────────────────────────────────────────────

#[test]
fn t15_a_dead_embed_thread_answers_none_at_once_and_the_status_says_so() {
    let dirs = Dirs::new();
    drop(make_brain(&dirs.brain()));
    let test = TestHooks { panic_embed: true, ..TestHooks::default() };

    // On the state from `open_state`: no hook sends a query at this commit.
    let (boot_front, _) = boot(Some(Embedder::hashing()), test.clone());
    let (st, _jobs_rx, _vec_tx) = front::open_state(dirs.ext_config("hash"), boot_front);
    assert_eq!(st.mode.load(Ordering::SeqCst), Mode::Ready as u8);
    assert!(!st.embed.stopped());
    assert_eq!(st.embed.embed_query("the first query", Instant::now() + Duration::from_secs(5)), None);
    let start = Instant::now();
    while !st.embed.stopped() {
        assert!(start.elapsed() < Duration::from_secs(5), "the embed thread did not end");
        std::thread::sleep(Duration::from_millis(5));
    }
    let start = Instant::now();
    assert_eq!(st.embed.embed_query("the second query", Instant::now() + Duration::from_secs(5)), None);
    assert!(start.elapsed() < Duration::from_millis(100), "{:?}", start.elapsed());

    // The status row of an indexer that shares the dead service.
    let (boot_indexer, status) = boot(None, test.clone());
    let (mut ctx, _jobs_rx_2, _vec_rx) = indexer_start::open_ctx(dirs.ext_config("hash"), boot_indexer);
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8);
    ctx.embed = st.embed.clone();
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    indexer_start::write_status(&mut ctx).unwrap();
    assert!(!status.model_loaded.load(Ordering::Relaxed));
    let row = status_row(&dirs.brain()).unwrap();
    assert_eq!(row["embed_stopped"], true);
    assert_eq!(row["model_loaded"], false);
    drop((ctx, st));

    // In a running process: the pump's first batch kills the embed thread; hooks go on.
    {
        let conn = Connection::open(dirs.brain()).unwrap();
        insert_memory(&conn, "mem_waiting", None, None, false);
    }
    let mut host = Host::ready(&dirs, Some(Embedder::hashing()), test);
    assert!(host.status.model_loaded.load(Ordering::Relaxed), "the model matches the brain at start");
    host.wait_until("the status says the model is gone", |h| !h.status.model_loaded.load(Ordering::Relaxed));
    for kind in HOOKS {
        assert_eq!(host.hook(kind, "s1"), continues(), "{kind}");
    }
    assert_eq!(host.tool("search")["result"]["content"], NO_CALLER, "the mode is still Ready");
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

// ── Section 7, test 16 ──────────────────────────────────────────────────────

#[test]
fn t16_a_hook_answered_without_forwarding_is_carried_to_the_next_forwarded_call() {
    let dirs = Dirs::new();
    let mut host = Host::ready(&dirs, None, TestHooks { front_sleep_ms: 700, ..TestHooks::default() });

    // Forwarded; the front thread sleeps past the budget.
    assert_eq!(host.hook("before_message", "session-S"), continues());
    assert_eq!(host.status.sessions_dropped.load(Ordering::Relaxed), 0);
    // Answered by the pump while it sleeps.
    let (a, took_a) = host.timed("hook.handle", hook_params("on_message_complete", "session-S"));
    let (b, took_b) = host.timed("hook.handle", hook_params("before_message", "session-S"));
    assert_eq!((a["result"].clone(), b["result"].clone()), (continues(), continues()));
    assert!(took_a < Duration::from_millis(100) && took_b < Duration::from_millis(100), "{took_a:?} {took_b:?}");
    assert_eq!(host.status.sessions_dropped.load(Ordering::Relaxed), 1, "one chat, two hooks");
    assert_eq!(host.status.hook_misses.load(Ordering::Relaxed), 3);
    // Core reports a forget in chat F: that chat goes with the next call too.
    assert_eq!(host.hook("on_memory_forgotten", "session-F"), continues());
    assert!(!read_log(&dirs).contains("front carry"), "nothing was forwarded yet");

    // The front thread returns; the next call is forwarded and carries the list.
    wait_for_log(&dirs, "front late in hook.handle before_message");
    assert_eq!(host.hook("on_session_start", "session-T"), continues());
    let log = wait_for_log(&dirs, "front carry ");
    assert!(log.contains("front carry skipped=1 forgot=1"), "{log}");
    assert_eq!(log.matches("front carry ").count(), 1);
    assert_eq!(host.status.sessions_dropped.load(Ordering::Relaxed), 1);
    assert!(!log.contains("MARKER-7731"));
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

// ── Row 12b: the reason before the mode ─────────────────────────────────────

#[test]
fn the_reason_is_stored_before_the_mode() {
    // A watcher reads the mode and then the reason, as `status` and the tools do. Each round
    // is fenced by a generation counter, so a pair read across a reset is not judged.
    let status = Arc::new(Status::default());
    let mode = Arc::new(AtomicU8::new(Mode::Starting as u8));
    let generation = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let seen_without_reason = Arc::new(AtomicU64::new(0));

    let watcher = {
        let (status, mode, generation, stop, bad) =
            (status.clone(), mode.clone(), generation.clone(), stop.clone(), seen_without_reason.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let before = generation.load(Ordering::SeqCst);
                if before % 2 == 1 {
                    continue;
                }
                let m = mode.load(Ordering::SeqCst);
                let r = status.reason.load(Ordering::SeqCst);
                if generation.load(Ordering::SeqCst) == before
                    && m == Mode::Degraded as u8
                    && DegradedReason::from_u8(r) == DegradedReason::None
                {
                    bad.fetch_add(1, Ordering::Relaxed);
                }
            }
        })
    };
    for _ in 0..400_000 {
        generation.fetch_add(1, Ordering::SeqCst);
        mode.store(Mode::Starting as u8, Ordering::SeqCst);
        status.reason.store(DegradedReason::None as u8, Ordering::SeqCst);
        generation.fetch_add(1, Ordering::SeqCst);
        degrade(&status, &mode, DegradedReason::IndexerStopped, None);
        std::hint::spin_loop();
    }
    stop.store(true, Ordering::Relaxed);
    watcher.join().unwrap();
    assert_eq!(seen_without_reason.load(Ordering::Relaxed), 0, "Degraded was visible before its reason");

    // And the reason a reader finds is the one `status` prints.
    assert_eq!(mode.load(Ordering::SeqCst), Mode::Degraded as u8);
    assert_eq!(DegradedReason::from_u8(status.reason.load(Ordering::SeqCst)).as_str(), "IndexerStopped");
    assert_eq!(server::unavailable_reason(&status, &mode), "IndexerStopped");
    assert_eq!(server::unavailable_reason(&status, &AtomicU8::new(Mode::Starting as u8)), "starting");
    degrade(&status, &mode, DegradedReason::IndexerStopped, Some("a.rs:1".to_string()));
    degrade(&status, &mode, DegradedReason::IndexerStopped, Some("b.rs:2".to_string()));
    assert_eq!(status.reason_place(), Some("a.rs:1"), "the first writer wins");
}

// ── Row 12b: open_state and open_ctx hand back the other ends (B3-3a) ───────

#[test]
fn open_state_and_open_ctx_hand_back_the_other_ends_of_their_channels() {
    let dirs = Dirs::new();

    // No brain yet: the front state has no connection and never makes a brain.
    let (b, _) = boot(None, TestHooks::default());
    let (st, jobs_rx, vec_tx) = front::open_state(dirs.ext_config("off"), b);
    assert!(st.conn.is_none());
    assert_eq!(st.mode.load(Ordering::SeqCst), Mode::Starting as u8);
    assert!(!dirs.brain().exists());
    assert!(st.jobs.try_send(Job::Tick).is_ok());
    assert!(front::send_job(&st, Job::Tick));
    assert!(matches!(jobs_rx.try_recv(), Ok(Job::Tick)));
    assert!(vec_tx.send(VecDelta::Remove(vec![VecKey::Mem(1)])).is_ok());
    assert!(matches!(st.vec_rx.try_recv(), Ok(VecDelta::Remove(_))));
    assert_eq!(st.mode.load(Ordering::SeqCst), Mode::Starting as u8, "a held receiver degrades nothing");
    // Dropped, the first send says the indexer has stopped: reason, then mode.
    drop(jobs_rx);
    assert!(!front::send_job(&st, Job::Tick));
    assert_eq!(st.mode.load(Ordering::SeqCst), Mode::Degraded as u8);
    assert_eq!(DegradedReason::from_u8(st.status.reason.load(Ordering::SeqCst)), DegradedReason::IndexerStopped);
    assert!(st.status.reason_place().is_some());
    drop(st);

    // The indexer's side makes the brain, records the base and writes the status row.
    let (b, status) = boot(Some(Embedder::hashing()), TestHooks::default());
    let (ctx, jobs_rx, vec_rx) = indexer_start::open_ctx(dirs.ext_config("hash"), b);
    assert!(ctx.conn.is_some());
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8);
    assert!(ctx.jobs.try_send(Job::Tick).is_ok());
    assert!(matches!(jobs_rx.try_recv(), Ok(Job::Tick)));
    assert!(matches!(vec_rx.try_recv(), Ok(VecDelta::Replace(items)) if items.is_empty()));
    assert!(status.model_loaded.load(Ordering::Relaxed));
    assert!(status.status_written_ms.load(Ordering::Relaxed) > 0);
    let row = status_row(&dirs.brain()).unwrap();
    for key in [
        "pid", "mode", "reason", "written_ms", "started_ms", "scan", "indexer_running", "embed_stopped",
        "model_loaded", "model", "model_dir", "embedder", "vectors_waiting", "docs_waiting", "breaker_open",
        "breaker_trips",
        "hook_misses", "sessions_dropped", "jobs_dropped", "tokens_blanked", "turns_withheld", "turns_dropped",
        "write_errors",
        "panics", "forum", "base", "warnings",
    ] {
        assert!(row.get(key).is_some(), "the status row has no {key}");
    }
    assert_eq!(row["model"], "hash-384");
    assert_eq!(row["model_loaded"], true);
    assert_eq!((&row["brain_model"], &row["model_matches"]), (&json!("hash-384"), &json!(true)), "{row}");

    // Now there is a brain: a front state connects to it and is Ready.
    let (b, _) = boot(None, TestHooks::default());
    let (st, _jobs_rx, _vec_tx) = front::open_state(dirs.ext_config("off"), b);
    assert!(st.conn.is_some());
    assert_eq!(st.mode.load(Ordering::SeqCst), Mode::Ready as u8);

    // Who may see what.
    let cfg = dirs.ext_config("off");
    let cwd = dirs.path().to_string_lossy().to_string();
    let chat = front::view_for(&Caller { session_id: Some("s1".into()), cwd: Some(cwd.clone()), author: None }, &cfg);
    assert!(chat.project.is_some() && chat.global && chat.documents);
    assert_eq!(chat.own_session.as_deref(), Some("s1"));
    let worker = front::view_for(&Caller { session_id: None, cwd: Some(cwd.clone()), author: None }, &cfg);
    assert!(worker.project.is_some() && !worker.global && !worker.documents && worker.own_session.is_none());
    let mut all = cfg.clone();
    all.worker_scope_all = true;
    let worker = front::view_for(&Caller { session_id: None, cwd: Some(cwd), author: None }, &all);
    assert!(worker.global && worker.documents);
    let nowhere = front::view_for(&Caller { session_id: Some("s1".into()), cwd: None, author: None }, &cfg);
    assert!(nowhere.project.is_none());
}

// ── Section 3, test 15: the first vector load ───────────────────────────────

#[test]
fn s3_t15_the_first_vector_load_holds_the_visible_file_and_the_visible_memory() {
    let dirs = Dirs::new();
    let (doc_visible, mem_visible) = {
        let conn = make_brain(&dirs.brain());
        let doc_visible = insert_document(&conn, "notes/a.md", "{}", None, 1);
        insert_document(&conn, "notes/hidden.md", "{}", Some("cli"), 2);
        // A visible old memory with its shadow document, and a hidden one with its shadow.
        let mem_visible = insert_memory(&conn, "mem_old_visible", None, Some(&unit(3)), false);
        insert_document(&conn, "memory/mem_old_visible", r#"{"type":"memory"}"#, None, 3);
        insert_memory(&conn, "mem_old_hidden", None, Some(&unit(4)), true);
        insert_document(&conn, "memory/mem_old_hidden", r#"{"type":"memory"}"#, Some("memory_hidden"), 4);
        (doc_visible, mem_visible)
    };

    let (b, _) = boot(Some(Embedder::hashing()), TestHooks::default());
    let (mut ctx, _jobs_rx, vec_rx) = indexer_start::open_ctx(dirs.ext_config("hash"), b);
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8);
    let Ok(VecDelta::Replace(items)) = vec_rx.try_recv() else { panic!("no first Replace") };
    let mut keys: Vec<VecKey> = items.iter().map(|i| i.key).collect();
    keys.sort();
    assert_eq!(keys, vec![VecKey::Doc(doc_visible), VecKey::Mem(mem_visible)]);
    for item in &items {
        match item.key {
            VecKey::Doc(_) => assert_eq!((&item.scope, &item.v), (&VecScope::Documents, &unit(1))),
            VecKey::Mem(_) => assert_eq!((&item.scope, &item.v), (&VecScope::Global, &unit(3))),
        }
    }
    assert!(vec_rx.try_recv().is_err(), "one delta");

    // Nothing changed: the next resync sends nothing.
    indexer_start::resync_vectors(&mut ctx).unwrap();
    assert!(vec_rx.try_recv().is_err());
    // Another process hides the file: the whole set is sent again, without it.
    Connection::open(dirs.brain())
        .unwrap()
        .execute("UPDATE documents SET hidden = 'cli' WHERE id = ?1", [doc_visible])
        .unwrap();
    indexer_start::resync_vectors(&mut ctx).unwrap();
    let Ok(VecDelta::Replace(items)) = vec_rx.try_recv() else { panic!("no second Replace") };
    assert_eq!(items.iter().map(|i| i.key).collect::<Vec<_>>(), vec![VecKey::Mem(mem_visible)]);
}

// ── The embedding backlog ───────────────────────────────────────────────────

#[test]
fn the_backlog_is_embedded_once_written_and_sent_to_the_front_thread() {
    let dirs = Dirs::new();
    let project = "a".repeat(64);
    let (vid_a, vid_b) = {
        let conn = make_brain(&dirs.brain());
        let a = insert_memory(&conn, "mem_a", None, None, false);
        let b = insert_memory(&conn, "mem_b", Some(&project), None, false);
        conn.execute("UPDATE memories SET sensitivity = 'secret' WHERE id = 'mem_a'", []).unwrap();
        conn.execute("UPDATE memories SET sensitivity = 'normal' WHERE id = 'mem_b'", []).unwrap();
        insert_memory(&conn, "mem_c", None, None, false);
        (a, b)
    };
    // The batch takes 300 ms, so it is still with the embed thread at the second pump.
    let (b, status) = boot(Some(Embedder::hashing()), TestHooks { embed_sleep_ms: 300, ..TestHooks::default() });
    let (mut ctx, jobs_rx, vec_rx) = indexer_start::open_ctx(dirs.ext_config("hash"), b);
    assert!(matches!(vec_rx.try_recv(), Ok(VecDelta::Replace(items)) if items.is_empty()));

    indexer_start::pump_embeddings(&mut ctx).unwrap();
    assert_eq!(status.vectors_waiting.load(Ordering::Relaxed), 2, "a secret row is never embedded");
    assert_eq!(ctx.embed.backlog(), 2);
    // While the batch is out, nothing is offered a second time.
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    let Ok(Job::Embedded { model, vectors, out }) = jobs_rx.recv_timeout(Duration::from_secs(10)) else {
        panic!("no Job::Embedded")
    };
    assert!(jobs_rx.recv_timeout(Duration::from_millis(200)).is_err(), "the rows were queued twice");
    assert_eq!(model, "hash-384");
    assert_eq!(vectors.len(), 2);
    assert!(vectors.iter().all(|(k, _)| *k != VecKey::Mem(vid_a)));

    indexer_start::on_embedded(&mut ctx, &model, vectors.clone(), out.as_deref()).unwrap();
    // The job is finished with when it is dropped, as `indexer::dispatch` drops it.
    assert_eq!(ctx.embed.backlog(), 2);
    let out_kept = out.clone();
    drop(out);
    assert_eq!(ctx.embed.backlog(), 2, "a clone of the job still holds the count");
    let out = out_kept;
    let Ok(VecDelta::Upsert(items)) = vec_rx.try_recv() else { panic!("no Upsert") };
    assert_eq!(items.len(), 2);
    let item_b = items.iter().find(|i| i.key == VecKey::Mem(vid_b)).unwrap();
    assert_eq!(item_b.scope, VecScope::Project(project.clone()));
    let conn = Connection::open(dirs.brain()).unwrap();
    let (stored_model, stored): (String, Vec<u8>) = conn
        .query_row("SELECT embed_model, embedding FROM memory_ext WHERE vid = ?1", [vid_b], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(stored_model, "hash-384");
    assert_eq!(stored, blob(&item_b.v));
    let secret: Option<Vec<u8>> =
        conn.query_row("SELECT embedding FROM memory_ext WHERE vid = ?1", [vid_a], |r| r.get(0)).unwrap();
    assert!(secret.is_none());

    // What was sent is counted as sent: no resync follows; and a second copy of the job
    // writes nothing.
    indexer_start::resync_vectors(&mut ctx).unwrap();
    assert!(vec_rx.try_recv().is_err());
    indexer_start::on_embedded(&mut ctx, &model, vectors, out.as_deref()).unwrap();
    assert!(vec_rx.try_recv().is_err());
    drop(out);
    assert_eq!(ctx.embed.backlog(), 0);
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    assert_eq!(status.vectors_waiting.load(Ordering::Relaxed), 0);
    assert_eq!(ctx.embed.backlog(), 0);

    // A degraded process writes nothing: not a vector, not its status row.
    insert_memory(&conn, "mem_d", None, None, false);
    let written = status.status_written_ms.load(Ordering::Relaxed);
    degrade(&ctx.status, &ctx.mode, DegradedReason::BrainChanged, None);
    indexer_start::on_embedded(&mut ctx, "hash-384", vec![(VecKey::Mem(vid_a), unit(9))], None).unwrap();
    indexer_start::write_status(&mut ctx).unwrap();
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    assert_eq!(ctx.embed.backlog(), 0);
    assert_eq!(status.status_written_ms.load(Ordering::Relaxed), written);
    let secret: Option<Vec<u8>> =
        conn.query_row("SELECT embedding FROM memory_ext WHERE vid = ?1", [vid_a], |r| r.get(0)).unwrap();
    assert!(secret.is_none());
    indexer_start::ensure_open(&mut ctx);
    assert!(ctx.conn.is_none(), "a degraded indexer gives up its connection");
    indexer_start::ensure_open(&mut ctx);
    assert!(ctx.conn.is_none(), "and a changed brain is not opened again");
}

#[test]
fn a_model_that_is_not_the_brains_runs_on_words_alone() {
    let dirs = Dirs::new();
    {
        // Recorded for the ONNX model's name; the running embedder is the hash one.
        let opts = OpenOpts { agent: None, model: "all-MiniLM-L6-v2", dim: 384, upgrade_v1: false };
        let conn = brain_v2::open_v2(&dirs.brain(), &opts).unwrap().conn;
        insert_document(&conn, "notes/a.md", "{}", None, 1);
        insert_memory(&conn, "mem_a", None, None, false);
    }
    let (b, status) = boot(Some(Embedder::hashing()), TestHooks::default());
    let (mut ctx, jobs_rx, vec_rx) = indexer_start::open_ctx(dirs.ext_config("hash"), b);
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8, "it is not a reason to degrade");
    assert!(!status.model_loaded.load(Ordering::Relaxed));
    assert!(vec_rx.try_recv().is_err(), "no vector of another model is loaded");
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    assert_eq!(ctx.embed.backlog(), 0);
    assert!(jobs_rx.recv_timeout(Duration::from_millis(100)).is_err(), "nothing is embedded");
    let row = status_row(&dirs.brain()).unwrap();
    assert_eq!(row["model_loaded"], false);
    // Wave 3 ruling 9: the row says which two models, and that they differ.
    assert_eq!((&row["model"], &row["brain_model"]), (&json!("hash-384"), &json!("all-MiniLM-L6-v2")), "{row}");
    assert_eq!(row["model_matches"], false, "{row}");
    assert_eq!(row["embed_stuck"], false, "{row}");
}

// ── Wave 3 ruling 1: the status file, one per process ───────────────────────

fn status_file(dirs: &Dirs) -> PathBuf {
    server::status_file_path(dirs.brain().parent().unwrap(), std::process::id())
}

fn read_status_file(dirs: &Dirs) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(status_file(dirs)).ok()?).ok()
}

/// Waits until this process's status file names `mode`; returns the object.
fn wait_status_file(dirs: &Dirs, mode: &str, within: Duration) -> Value {
    let start = Instant::now();
    loop {
        let seen = read_status_file(dirs);
        if let Some(v) = seen.as_ref().filter(|v| v["mode"] == mode) {
            return v.clone();
        }
        assert!(start.elapsed() < within, "the status file never said {mode:?}: {seen:?}");
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn r1_a_process_degraded_at_start_says_so_in_its_status_file_and_the_brain_is_unchanged() {
    let dirs = Dirs::new();
    {
        let conn = make_brain(&dirs.brain());
        conn.execute_batch("PRAGMA user_version = 99; PRAGMA wal_checkpoint(TRUNCATE);").unwrap();
    }
    let before = sha256(&dirs.brain());

    let mut host = Host::start(Some(Embedder::hashing()), TestHooks::default());
    host.call("initialize", json!({"config": dirs.config("hash")}));
    let said = wait_status_file(&dirs, "degraded", Duration::from_secs(15));
    assert_eq!(said["reason"], "BrainTooNew", "{said}");
    assert_eq!(said["pid"], std::process::id());
    assert!(said["started_ms"].as_i64().unwrap() > 0 && said["written_ms"].as_i64().unwrap() > 0, "{said}");
    assert_eq!(host.hook("before_message", "s1"), continues());

    assert_eq!(host.close(), ServeEnd::InputClosed);
    assert!(!status_file(&dirs).exists(), "a clean end leaves its status file");
    assert_eq!(sha256(&dirs.brain()), before, "a brain at version 99 was written");
    assert_eq!(host_status_written(&dirs), None);
}

#[test]
fn r1_a_healthy_process_says_ready_removes_old_files_at_start_and_its_own_at_a_clean_end() {
    use std::os::unix::fs::PermissionsExt;
    let dirs = Dirs::new();
    let folder = dirs.brain().parent().unwrap().to_path_buf();
    std::fs::create_dir_all(&folder).unwrap();
    // Two files of other processes (pids 1 and 2 are never a test's): one 25 hours old.
    let (old, fresh) = (server::status_file_path(&folder, 1), server::status_file_path(&folder, 2));
    for path in [&old, &fresh] {
        std::fs::write(path, br#"{"pid":1,"mode":"ready"}"#).unwrap();
    }
    let long_ago = std::time::SystemTime::now() - Duration::from_secs(25 * 3600);
    std::fs::File::options().write(true).open(&old).unwrap().set_modified(long_ago).unwrap();

    let host = Host::ready(&dirs, None, TestHooks::default());
    let said = wait_status_file(&dirs, "ready", Duration::from_secs(15));
    let mut keys: Vec<&str> = said.as_object().unwrap().keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["mode", "pid", "reason", "reason_at", "started_ms", "written_ms"]);
    assert_eq!((&said["reason"], &said["reason_at"]), (&json!("None"), &Value::Null), "{said}");
    assert_eq!(said["pid"], std::process::id());
    assert_eq!(std::fs::metadata(status_file(&dirs)).unwrap().permissions().mode() & 0o777, 0o600);
    assert!(!old.exists(), "a status file 25 hours old was left");
    assert!(fresh.exists(), "another process's fresh status file was removed");
    assert!(!folder.join(format!("extension.{}.status.tmp", std::process::id())).exists());

    assert_eq!(host.close(), ServeEnd::InputClosed);
    assert!(!status_file(&dirs).exists(), "a clean end leaves its status file");
    assert!(fresh.exists(), "the end removed another process's status file");
}

// ── Wave 3 ruling 2: the front thread is watched outside a call ─────────────

const FRONT_STOPPED_TEXT: &str = "Axel is unavailable: FrontStopped. Synaps itself is not affected.";

/// After `initialize`: the process is alive, says why in its status file and to a tool, and
/// answers every hook.
fn assert_serves_without_a_front_thread(host: &mut Host, dirs: &Dirs) {
    assert_eq!(DegradedReason::from_u8(host.status.reason.load(Ordering::SeqCst)), DegradedReason::FrontStopped);
    for kind in HOOKS {
        assert_eq!(host.hook(kind, "s1"), continues(), "{kind}");
    }
    let tool = host.tool("search");
    assert_eq!(tool["result"]["is_error"], true, "{tool}");
    assert_eq!(tool["result"]["content"], FRONT_STOPPED_TEXT);
    // The reason stands: no thread brings the mode back.
    std::thread::sleep(Duration::from_millis(600));
    assert_eq!(wait_status_file(dirs, "degraded", Duration::from_secs(1))["reason"], "FrontStopped");
    assert_eq!(host.tool("search")["result"]["content"], FRONT_STOPPED_TEXT);
    assert!(host.end.try_recv().is_err(), "serve returned");
    assert!(!dirs.brain().exists(), "a brain was made by a process with no front thread");
}

#[test]
fn r2_a_front_thread_that_panics_before_its_loop_degrades_the_process_which_stays() {
    let dirs = Dirs::new();
    let test = TestHooks { panic_front_start: true, ..TestHooks::default() };
    let mut host = Host::start(None, test);
    host.call("initialize", json!({"config": dirs.config("off")}));
    let said = wait_status_file(&dirs, "degraded", Duration::from_secs(1));
    assert_eq!(said["reason"], "FrontStopped", "{said}");
    let at = said["reason_at"].as_str().expect("the panic's file and line");
    assert!(at.contains("server.rs:"), "{at}");
    assert_eq!(host.status.reason_place(), Some(at));

    assert_serves_without_a_front_thread(&mut host, &dirs);
    let log = read_log(&dirs);
    assert!(log.contains("front thread ended") && !log.contains("test switch"), "{log}");
    assert_eq!(host.close(), ServeEnd::InputClosed);
    assert!(!status_file(&dirs).exists());
}

#[test]
fn r2_a_front_thread_that_never_hands_over_the_queue_degrades_the_process_which_stays() {
    let dirs = Dirs::new();
    let test = TestHooks { hang_front_start: true, front_start_limit_ms: Some(300), ..TestHooks::default() };
    let mut host = Host::start(None, test);
    let start = Instant::now();
    host.call("initialize", json!({"config": dirs.config("off")}));
    // Before the limit it is a process that is starting.
    assert_eq!(read_status_file(&dirs).expect("the file is there from initialize")["mode"], "starting");
    let said = wait_status_file(&dirs, "degraded", Duration::from_secs(2));
    assert!(start.elapsed() >= Duration::from_millis(290), "{:?}", start.elapsed());
    assert_eq!((&said["reason"], &said["reason_at"]), (&json!("FrontStopped"), &Value::Null), "{said}");

    assert_serves_without_a_front_thread(&mut host, &dirs);
    assert!(read_log(&dirs).contains("front thread did not start in time"));
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

// ── Wave 3 ruling 3: a late tool does not open the breaker ──────────────────

#[test]
fn r3_hooks_answered_behind_a_late_tool_are_counted_and_do_not_open_the_breaker() {
    let dirs = Dirs::new();
    let mut host = Host::ready(&dirs, None, TestHooks { front_sleep_ms: 2400, ..TestHooks::default() });

    // The tool is forwarded and is late at its 2 s.
    let start = Instant::now();
    let tool = host.tool("search");
    assert_eq!(tool["result"]["content"], BUSY_TEXT, "{tool}");
    assert!(start.elapsed() >= Duration::from_millis(1900), "{:?}", start.elapsed());

    // Three messages while the front thread is still inside it.
    for n in 1..=3 {
        let (answer, took) = host.timed("hook.handle", hook_params("before_message", "s1"));
        assert_eq!(answer["result"], continues());
        assert!(took < Duration::from_millis(100), "message {n} was forwarded: {took:?}");
    }
    assert!(start.elapsed() < Duration::from_millis(2400), "the tool had returned: {:?}", start.elapsed());
    assert_eq!(host.status.hook_misses.load(Ordering::Relaxed), 3);
    assert!(!host.status.breaker_open.load(Ordering::Relaxed), "a late tool opened the breaker");
    assert_eq!(host.status.breaker_trips.load(Ordering::Relaxed), 0);

    // The tool returns; the next message is forwarded (it waits its budget for a front
    // thread that sleeps again), and the breaker, which is what switches recall off, is
    // still closed when it goes.
    wait_for_log(&dirs, "front late in tool.call");
    let (answer, took) = host.timed("hook.handle", hook_params("before_message", "s1"));
    assert_eq!(answer["result"], continues());
    assert!(took >= Duration::from_millis(140), "not forwarded: {took:?}");
    assert!(!host.status.breaker_open.load(Ordering::Relaxed));
    assert_eq!(host.status.breaker_trips.load(Ordering::Relaxed), 0);
    assert_eq!(host.status.hook_misses.load(Ordering::Relaxed), 4);

    host.send(&json!({"jsonrpc": "2.0", "id": 902, "method": "shutdown"}));
    assert_eq!(host.read()["id"], 902);
    assert_eq!(host.end.recv_timeout(Duration::from_secs(5)).unwrap(), ServeEnd::Shutdown);
}

// ── Wave 3 ruling 8 and commit 11's ruling 2: `indexer::guarded` ────────────

const MARKER_ROW: &str = "INSERT INTO context_data(key, value, layer) VALUES('r8-marker', 'x', 'test')";

fn marker_rows(conn: &Connection) -> i64 {
    conn.query_row("SELECT COUNT(*) FROM context_data WHERE key = 'r8-marker'", [], |r| r.get(0)).unwrap()
}

/// A handler that begins a write, writes a row and never ends its transaction.
fn begin_and_write(ctx: &mut IndexerCtx) {
    let conn = ctx.conn.as_ref().unwrap();
    brain_v2::begin_write(conn).unwrap();
    conn.execute(MARKER_ROW, []).unwrap();
    assert!(!conn.is_autocommit());
}

#[test]
fn r8_a_handler_that_fails_inside_a_write_leaves_no_transaction_and_no_row() {
    let dirs = Dirs::new();
    let (b, status) = boot(None, TestHooks::default());
    let (mut ctx, _jobs_rx, _vec_rx) = indexer_start::open_ctx(dirs.ext_config("off"), b);
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8);

    // An error.
    indexer::guarded(&mut ctx, |ctx| {
        begin_and_write(ctx);
        Err(AxelError::Other("the handler failed".to_string()))
    });
    assert_eq!(status.write_errors.load(Ordering::Relaxed), 1);
    // A panic.
    for round in 0..2 {
        if round == 1 {
            indexer::guarded(&mut ctx, |ctx| {
                begin_and_write(ctx);
                panic!("the handler panicked")
            });
            assert_eq!(status.panics.load(Ordering::Relaxed), 1);
        }
        let conn = ctx.conn.as_ref().unwrap();
        assert!(conn.is_autocommit(), "round {round}: the failed handler's transaction is still open");
        brain_v2::begin_write(conn).expect("the next write begins");
        assert_eq!(marker_rows(conn), 0, "round {round}");
        conn.execute_batch("COMMIT").unwrap();
        let other = Connection::open(dirs.brain()).unwrap();
        assert_eq!(marker_rows(&other), 0, "round {round}: the failed handler's row is in the file");
    }
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8);

    // A handler that ends well keeps what it wrote.
    indexer::guarded(&mut ctx, |ctx| {
        begin_and_write(ctx);
        ctx.conn.as_ref().unwrap().execute_batch("COMMIT").unwrap();
        Ok(())
    });
    assert_eq!(marker_rows(&Connection::open(dirs.brain()).unwrap()), 1);
}

#[test]
fn r8_a_degraded_indexer_runs_no_handler() {
    let dirs = Dirs::new();
    // Control: on a connection without the tables the tick's handlers fail, and say so.
    let (b, status) = boot(None, TestHooks::default());
    let (mut ctx, _jobs_rx, _vec_rx) = indexer_start::open_ctx(dirs.ext_config("off"), b);
    ctx.conn = Some(Connection::open_in_memory().unwrap());
    ctx.last_run.clear();
    indexer::tick(&mut ctx);
    assert!(status.write_errors.load(Ordering::Relaxed) >= 1, "the control ran no handler");

    // The same, degraded: nothing runs, in the tick, for a job, or handed in.
    let (b, status) = boot(None, TestHooks { panic_on_forgotten_job: true, ..TestHooks::default() });
    let (mut ctx, _jobs_rx, _vec_rx) = indexer_start::open_ctx(dirs.ext_config("off"), b);
    ctx.conn = Some(Connection::open_in_memory().unwrap());
    ctx.last_run.clear();
    degrade(&ctx.status, &ctx.mode, DegradedReason::BrainDamaged, None);
    indexer::tick(&mut ctx);
    indexer::dispatch(
        &mut ctx,
        Job::Forgotten { id: "id".to_string(), kind: "note".to_string(), project_key: "p0123456789abcdef".to_string() },
    );
    let mut ran = false;
    indexer::guarded(&mut ctx, |_| {
        ran = true;
        Ok(())
    });
    assert!(!ran, "a handler ran in a degraded process");
    assert_eq!(status.write_errors.load(Ordering::Relaxed), 0);
    assert_eq!(status.panics.load(Ordering::Relaxed), 0);
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Degraded as u8);
}

// ── Wave 3 rulings 5 and 6: the file under the process ──────────────────────

#[test]
fn r5_a_brain_file_replaced_under_the_process_degrades_it_and_the_new_file_is_not_written() {
    let dirs = Dirs::new();
    let mut host = Host::ready(&dirs, None, TestHooks { status_tick_ms: Some(50), ..TestHooks::default() });
    assert_eq!(wait_status_file(&dirs, "ready", Duration::from_secs(15))["reason"], "None");

    // The same bytes under a new inode, as a restored backup would be.
    let copy = dirs.brain().with_extension("copy");
    std::fs::copy(dirs.brain(), &copy).unwrap();
    let before = sha256(&copy);
    std::fs::rename(&copy, dirs.brain()).unwrap();

    let said = wait_status_file(&dirs, "degraded", Duration::from_secs(10));
    assert_eq!(said["reason"], "BrainChanged", "{said}");
    let written = host.status.status_written_ms.load(Ordering::Relaxed);
    assert_eq!(
        host.tool("search")["result"]["content"],
        "Axel is unavailable: BrainChanged. Synaps itself is not affected."
    );
    assert_eq!(host.hook("before_message", "s1"), continues());
    // More than one turn of the indexer's loop: no status row follows.
    std::thread::sleep(Duration::from_millis(1300));
    assert_eq!(host.status.status_written_ms.load(Ordering::Relaxed), written);
    assert!(wait_for_log(&dirs, "brain file replaced or gone").matches("brain file replaced").count() == 1);
    assert_eq!(host.close(), ServeEnd::InputClosed);
    assert_eq!(sha256(&dirs.brain()), before, "the new file was written");

    // A path that is gone is the same.
    let other = Dirs::new();
    let (b, status) = boot(None, TestHooks::default());
    let (mut ctx, _jobs_rx, _vec_rx) = indexer_start::open_ctx(other.ext_config("off"), b);
    assert!(ctx.brain_file.is_some());
    indexer_start::write_status(&mut ctx).unwrap();
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8);
    std::fs::remove_file(other.brain()).unwrap();
    indexer_start::write_status(&mut ctx).unwrap();
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Degraded as u8);
    assert_eq!(DegradedReason::from_u8(status.reason.load(Ordering::SeqCst)), DegradedReason::BrainChanged);
    assert!(!other.brain().exists(), "a brain was made where one had gone");
}

#[test]
fn r6_a_look_that_cannot_read_the_brain_is_not_ready_and_open_v2_does_not_run() {
    let dirs = Dirs::new();
    drop(make_brain(&dirs.brain()));
    // Another connection holds the file: every read by anyone else is `SQLITE_BUSY`.
    let holder = Connection::open(dirs.brain()).unwrap();
    holder
        .execute_batch(
            "PRAGMA locking_mode = EXCLUSIVE; BEGIN IMMEDIATE;
             INSERT INTO context_data(key, value, layer) VALUES('held', 'x', 'test'); COMMIT;",
        )
        .unwrap();

    let mut host = Host::start(None, TestHooks::default());
    let start = Instant::now();
    host.call("initialize", json!({"config": dirs.config("off")}));
    // The look gives up after its 1 s. `open_v2`, had it run, would wait 5 s more.
    wait_for_log(&dirs, "brain not open: BrainNotReady");
    assert!(start.elapsed() < Duration::from_secs(3), "open_v2 ran after the look failed: {:?}", start.elapsed());
    assert_eq!(DegradedReason::from_u8(host.status.reason.load(Ordering::SeqCst)), DegradedReason::BrainNotReady);

    // Released: nothing opens the brain before the next retry, a minute away.
    drop(holder);
    std::thread::sleep(Duration::from_millis(400));
    assert_eq!(
        host.tool("search")["result"]["content"],
        "Axel is unavailable: BrainNotReady. Synaps itself is not affected."
    );
    assert_eq!(host.close(), ServeEnd::InputClosed);
    let conn = Connection::open(dirs.brain()).unwrap();
    let rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM context_data WHERE layer = 'axel'", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rows, 0, "the brain was written by a process that could not read its base");
}

// ── Wave 3 ruling 7: a forget mark that could not get the write lock ────────

// `mark_forgotten_turn` is a stub until commit 18a, so what is seen here is the front
// thread's log: the mark is kept, and written once the lock is free.
#[test]
fn r7_a_forget_mark_that_could_not_get_the_write_lock_is_written_when_it_is_free() {
    let dirs = Dirs::new();
    let mut host = Host::ready(&dirs, None, TestHooks::default());
    let holder = Connection::open(dirs.brain()).unwrap();
    holder.execute_batch("BEGIN IMMEDIATE").unwrap();

    // Core's forget ran in chat S; the next forwarded call carries it to the front thread.
    assert_eq!(host.hook("on_memory_forgotten", "session-S"), continues());
    assert_eq!(host.hook("before_message", "session-S"), continues());
    wait_for_log(&dirs, "front carry: forget mark not written, kept");
    // Still locked: tried between calls and after this one, and still kept.
    std::thread::sleep(Duration::from_millis(600));
    host.wait_until("the front thread is free again", |h| h.hook("on_session_start", "other") == continues());
    assert!(!read_log(&dirs).contains("forget marks written late"));

    holder.execute_batch("ROLLBACK").unwrap();
    assert_eq!(host.hook("before_message", "session-S"), continues());
    let log = wait_for_log(&dirs, "front carry: forget marks written late: 1");
    assert_eq!(log.matches("forget mark not written, kept").count(), 1, "{log}");
    assert!(!log.contains("session-S"), "{log}");
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

// ── Wave 3 ruling 9: an embed thread held inside a batch ────────────────────

#[test]
fn r9_an_embed_thread_held_inside_a_batch_is_stuck_and_a_query_gets_none_at_once() {
    let dirs = Dirs::new();
    {
        let conn = make_brain(&dirs.brain());
        insert_memory(&conn, "mem_waiting", None, None, false);
    }
    // Every batch takes 1.5 s; one that has run 200 ms counts as hung.
    let test = TestHooks { embed_sleep_ms: 1500, embed_stuck_limit_ms: Some(200), ..TestHooks::default() };
    let (b, status) = boot(Some(Embedder::hashing()), test);
    let (mut ctx, _jobs_rx, _vec_rx) = indexer_start::open_ctx(dirs.ext_config("hash"), b);
    assert!(!ctx.embed.stuck());
    assert_eq!(status_row(&dirs.brain()).unwrap()["embed_stuck"], false);

    indexer_start::pump_embeddings(&mut ctx).unwrap();
    assert!(status.model_loaded.load(Ordering::Relaxed));
    std::thread::sleep(Duration::from_millis(400));
    assert!(ctx.embed.stuck() && !ctx.embed.stopped());

    // The query path: no model, at once, with five seconds of deadline left.
    let start = Instant::now();
    assert_eq!(ctx.embed.embed_query("retry limit", Instant::now() + Duration::from_secs(5)), None);
    assert!(start.elapsed() < Duration::from_millis(100), "the query waited: {:?}", start.elapsed());

    // The status tick says so, and the pump offers nothing.
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    assert!(!status.model_loaded.load(Ordering::Relaxed));
    indexer_start::write_status(&mut ctx).unwrap();
    let row = status_row(&dirs.brain()).unwrap();
    assert_eq!((&row["embed_stuck"], &row["embed_stopped"], &row["model_loaded"]), (&json!(true), &json!(false), &json!(false)), "{row}");
    assert_eq!(row["model_matches"], true, "{row}");

    // The batch returns after all: the model is back.
    let start = Instant::now();
    while ctx.embed.stuck() {
        assert!(start.elapsed() < Duration::from_secs(10), "the batch never returned");
        std::thread::sleep(Duration::from_millis(20));
    }
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    assert!(status.model_loaded.load(Ordering::Relaxed));
}

// ── Wave 3 rulings 10 and 11 ────────────────────────────────────────────────

#[test]
fn r10_two_threads_at_the_log_limit_lose_no_generation() {
    const LIMIT: u64 = 2048;
    let dirs = Dirs::new();
    let folder = dirs.path().join("logs");
    let log = server::Log::new(&folder, LIMIT);
    let old = folder.join("extension.log.1");
    let writers: Vec<_> = (0..4)
        .map(|n| {
            let (log, old) = (log.clone(), old.clone());
            std::thread::spawn(move || {
                let mut short = 0u32;
                for i in 0..1500 {
                    log.line(&format!("writer {n} line {i:05} of the log at its limit"));
                    // Whenever there is a kept generation, it is a whole one.
                    if std::fs::metadata(&old).is_ok_and(|m| m.len() <= LIMIT) {
                        short += 1;
                    }
                }
                short
            })
        })
        .collect();
    let short: u32 = writers.into_iter().map(|w| w.join().unwrap()).sum();
    assert_eq!(short, 0, "extension.log.1 was seen {short} times holding less than a full file");
    assert!(std::fs::metadata(&old).unwrap().len() > LIMIT);
    assert!(std::fs::metadata(folder.join("extension.log")).map(|m| m.len()).unwrap_or(0) < LIMIT + 4096);
}

#[test]
fn r10_the_stop_flag_ends_an_indexer_loop_whose_queue_is_full() {
    let dirs = Dirs::new();
    // Every job takes 50 ms: the 1,024 queued would take 51 s.
    let (b, _status) = boot(None, TestHooks { indexer_sleep_ms: 50, ..TestHooks::default() });
    let (ctx, jobs_rx, _vec_rx) = indexer_start::open_ctx(dirs.ext_config("off"), b);
    let job = || Job::Forgotten { id: "id".into(), kind: "other".into(), project_key: "p0123456789abcdef".into() };
    while ctx.jobs.try_send(job()).is_ok() {}
    assert!(ctx.jobs.try_send(Job::Shutdown).is_err(), "the queue has room for Shutdown");

    let (stop, jobs) = (ctx.stop.clone(), ctx.jobs.clone());
    let (done_tx, done) = mpsc::channel();
    std::thread::spawn(move || {
        indexer::run(ctx, jobs_rx);
        let _ = done_tx.send(());
    });
    std::thread::sleep(Duration::from_millis(200));
    stop.store(true, Ordering::SeqCst);
    let start = Instant::now();
    done.recv_timeout(Duration::from_secs(3)).expect("the loop went on after the stop flag");
    assert!(start.elapsed() < Duration::from_secs(2), "{:?}", start.elapsed());
    drop(jobs);
}

// ── Wave 3 ruling 12: the breaker closes; a retry recovers ──────────────────

#[test]
fn r12_the_breaker_closes_after_its_time_and_the_next_message_is_forwarded() {
    let dirs = Dirs::new();
    let test = TestHooks { front_sleep_ms: 400, breaker_open_ms: Some(700), ..TestHooks::default() };
    let mut host = Host::ready(&dirs, None, test);
    // One message late at its budget, two answered behind it: three misses.
    for _ in 0..3 {
        assert_eq!(host.hook("before_message", "s1"), continues());
    }
    assert!(host.status.breaker_open.load(Ordering::Relaxed));
    assert_eq!(host.status.breaker_trips.load(Ordering::Relaxed), 1);

    let start = Instant::now();
    while host.status.breaker_open.load(Ordering::Relaxed) {
        assert!(start.elapsed() < Duration::from_secs(4), "the breaker never closed");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(start.elapsed() >= Duration::from_millis(400), "it closed before its time: {:?}", start.elapsed());
    assert!(read_log(&dirs).contains("breaker closed"));

    // Forwarded: it waits its budget for a front thread that sleeps again. One miss does
    // not open the breaker a second time.
    let (answer, took) = host.timed("hook.handle", hook_params("before_message", "s1"));
    assert_eq!(answer["result"], continues());
    assert!(took >= Duration::from_millis(140), "not forwarded: {took:?}");
    assert!(!host.status.breaker_open.load(Ordering::Relaxed));
    assert_eq!(host.status.breaker_trips.load(Ordering::Relaxed), 1);
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

#[test]
fn r12_a_retry_opens_a_brain_that_has_become_ready() {
    let dirs = Dirs::new();
    {
        let conn = make_brain(&dirs.brain());
        conn.execute_batch("PRAGMA user_version = 99; PRAGMA wal_checkpoint(TRUNCATE);").unwrap();
    }
    let mut host = Host::start(None, TestHooks { open_retry_ms: Some(200), ..TestHooks::default() });
    host.call("initialize", json!({"config": dirs.config("off")}));
    assert_eq!(host.wait_degraded(), "Axel is unavailable: BrainTooNew. Synaps itself is not affected.");
    assert_eq!(wait_status_file(&dirs, "degraded", Duration::from_secs(5))["reason"], "BrainTooNew");

    // The brain becomes one this binary reads; nothing tells the process.
    Connection::open(dirs.brain())
        .unwrap()
        .execute_batch(&format!("PRAGMA user_version = {}", brain_v2::LATEST))
        .unwrap();
    let said = wait_status_file(&dirs, "ready", Duration::from_secs(10));
    assert_eq!((&said["reason"], &said["reason_at"]), (&json!("None"), &Value::Null), "{said}");
    assert_eq!(DegradedReason::from_u8(host.status.reason.load(Ordering::SeqCst)), DegradedReason::None);
    host.wait_until("the status row is written", |h| h.status.status_written_ms.load(Ordering::Relaxed) > 0);
    host.wait_until("a tool is forwarded", |h| h.tool("search")["result"]["content"] == NO_CALLER);
    assert_eq!(read_log(&dirs).matches("brain open").count(), 1);
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

#[test]
fn r12_on_recovery_the_mode_is_never_degraded_with_no_reason() {
    // A watcher reads the mode, the reason, and the mode again. Each round is fenced by a
    // generation counter, so three reads that span two rounds are not judged.
    let status = Arc::new(Status::default());
    let mode = Arc::new(AtomicU8::new(Mode::Ready as u8));
    let generation = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let seen_without_reason = Arc::new(AtomicU64::new(0));

    let watcher = {
        let (status, mode, generation, stop, bad) =
            (status.clone(), mode.clone(), generation.clone(), stop.clone(), seen_without_reason.clone());
        std::thread::spawn(move || {
            let degraded = Mode::Degraded as u8;
            while !stop.load(Ordering::Relaxed) {
                let before = generation.load(Ordering::SeqCst);
                let first = mode.load(Ordering::SeqCst);
                let reason = status.reason.load(Ordering::SeqCst);
                let second = mode.load(Ordering::SeqCst);
                if generation.load(Ordering::SeqCst) == before
                    && first == degraded
                    && second == degraded
                    && DegradedReason::from_u8(reason) == DegradedReason::None
                {
                    bad.fetch_add(1, Ordering::Relaxed);
                }
            }
        })
    };
    for _ in 0..400_000 {
        generation.fetch_add(1, Ordering::SeqCst);
        degrade(&status, &mode, DegradedReason::BrainNotReady, None);
        std::hint::spin_loop();
        recover(&status, &mode);
    }
    stop.store(true, Ordering::Relaxed);
    watcher.join().unwrap();
    assert_eq!(seen_without_reason.load(Ordering::Relaxed), 0, "Degraded was visible with its reason cleared");
    assert_eq!(mode.load(Ordering::SeqCst), Mode::Ready as u8);
}

// ── Wave 4, R-12b item 1: a forget mark is kept whatever kept it out ────────

fn forget_mark(session: &str) -> ForgetMark {
    ForgetMark { session_id: session.to_string(), from_ms: 1_000, to_ms: 2_000, turn_key: None }
}

/// `front::run` on its own thread, logging to the test's folder.
fn front_on_a_thread(st: FrontState) -> (Sender<FrontCall>, Receiver<FrontReply>, std::thread::JoinHandle<()>) {
    let (calls, calls_rx) = mpsc::channel();
    let (replies_tx, replies) = mpsc::channel();
    let log = server::Log::new(&st.cfg.data_dir, 1 << 20);
    let thread = std::thread::spawn(move || {
        server::set_thread_log(Some(log));
        front::run(st, calls_rx, replies_tx);
    });
    (calls, replies, thread)
}

#[test]
fn w4_1_a_forget_mark_that_arrives_before_the_brain_is_open_is_written_once_it_is() {
    let dirs = Dirs::new();
    // No brain yet: the mode is `Starting` and the front thread has no connection.
    let (b, _) = boot(None, TestHooks::default());
    let (st, _jobs_rx, _vec_tx) = front::open_state(dirs.ext_config("off"), b);
    assert!(st.conn.is_none());
    let mode = st.mode.clone();
    let (calls, replies, thread) = front_on_a_thread(st);

    let carry = Carry { skipped: Vec::new(), forgot: vec!["session-S".to_string()] };
    let work = FrontWork::Tool { name: "search".to_string(), input: json!({}), caller: Caller::default() };
    calls.send(FrontCall { seq: 1, carry, work }).unwrap();
    let reply = replies.recv_timeout(Duration::from_secs(10)).expect("no reply");
    assert!(matches!(reply.answer, FrontAnswer::Tool(Err(_))));
    let log = wait_for_log(&dirs, "front carry: forget mark not written, kept");
    assert!(!log.contains("written late"), "{log}");

    // The brain appears and the process is ready: the idle turn writes the mark.
    drop(make_brain(&dirs.brain()));
    mode.store(Mode::Ready as u8, Ordering::SeqCst);
    let log = wait_for_log(&dirs, "front carry: forget marks written late: 1");
    assert!(!log.contains("session-S"), "{log}");
    drop(calls);
    thread.join().unwrap();
    assert!(!read_log(&dirs).contains("lost at the end"));
}

#[test]
fn w4_1_a_forget_mark_that_fails_is_dropped_and_does_not_block_the_ones_behind_it() {
    let dirs = Dirs::new();
    drop(make_brain(&dirs.brain()));
    let (b, _) = boot(None, TestHooks::default());
    let (mut st, _jobs_rx, _vec_tx) = front::open_state(dirs.ext_config("off"), b);
    let tried = std::cell::RefCell::new(Vec::new());
    let write = |_: &FrontState, mark: &ForgetMark| {
        tried.borrow_mut().push(mark.session_id.clone());
        match mark.session_id.as_str() {
            "b" => MarkEnd::Failed("Db"),
            "d" => MarkEnd::NotNow,
            _ => MarkEnd::Written,
        }
    };
    st.forget_marks = ["a", "b", "c", "d", "e"].map(forget_mark).to_vec();
    assert_eq!(front::retry_marks(&mut st, write), 2);
    assert_eq!(*tried.borrow(), ["a", "b", "c", "d"], "each mark is tried once, up to the one that must wait");
    assert_eq!(st.forget_marks, ["d", "e"].map(forget_mark).to_vec());
}

#[test]
fn w4_1_the_kept_forget_marks_are_tried_once_more_before_the_front_thread_ends() {
    let dirs = Dirs::new();
    drop(make_brain(&dirs.brain()));
    let (b, _) = boot(None, TestHooks::default());
    let (mut st, _jobs_rx, _vec_tx) = front::open_state(dirs.ext_config("off"), b);
    assert_eq!(st.mode.load(Ordering::SeqCst), Mode::Ready as u8);
    st.forget_marks = vec![forget_mark("session-S"), forget_mark("session-T")];
    // The rpc thread is gone before the loop's first turn: no idle turn, no call.
    let (calls, _replies, thread) = front_on_a_thread(st);
    drop(calls);
    thread.join().unwrap();
    let log = read_log(&dirs);
    assert!(log.contains("front carry: forget marks written late: 2"), "{log}");
    assert!(!log.contains("lost at the end"), "{log}");
}

// ── Wave 4, R-12b item 2: the memories that cannot be read are counted ──────

fn insert_unreadable_memory(conn: &Connection, id: &str) {
    conn.execute(
        "INSERT INTO memories(id, category, topic, title, abstract_text, content, created)
         VALUES(?1, 'events', 'topic', 'a blob', '', x'FFFE0001', '2025-01-01T00:00:00Z')",
        [id],
    )
    .unwrap();
}

#[test]
fn w4_2_the_unreadable_memories_are_counted_at_open_and_by_the_tick_and_the_row_says_so() {
    let dirs = Dirs::new();
    {
        let conn = make_brain(&dirs.brain());
        insert_memory(&conn, "mem_fine", None, None, false);
        insert_unreadable_memory(&conn, "mem_blob_1");
    }
    let (b, status) = boot(None, TestHooks::default());
    let (mut ctx, _jobs_rx, _vec_rx) = indexer_start::open_ctx(dirs.ext_config("off"), b);
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8);
    assert_eq!(status.unreadable_memories.load(Ordering::Relaxed), 1, "open_v2's count was dropped");
    assert_eq!(status_row(&dirs.brain()).unwrap()["unreadable_memories"], 1);

    // Another one arrives under the running process: the tick's adoption counts both.
    insert_unreadable_memory(ctx.conn.as_ref().unwrap(), "mem_blob_2");
    indexer::tick(&mut ctx);
    assert_eq!(status.unreadable_memories.load(Ordering::Relaxed), 2, "the tick's count was dropped");
    indexer_start::write_status(&mut ctx).unwrap();
    assert_eq!(status_row(&dirs.brain()).unwrap()["unreadable_memories"], 2);
    assert_eq!(status.panics.load(Ordering::Relaxed), 0);
}

// ── Wave 4, R-12b item 3: a title that is not text does not stop the pump ───

#[test]
fn w4_3_a_title_that_is_not_text_does_not_stop_the_embedding_of_the_other_rows() {
    let dirs = Dirs::new();
    {
        let conn = make_brain(&dirs.brain());
        insert_memory(&conn, "mem_fine", None, None, false);
        insert_memory(&conn, "mem_badtitle", None, None, false);
        insert_memory(&conn, "mem_badbytes", None, None, false);
        conn.execute("UPDATE memories SET title = x'FFFE0001' WHERE id = 'mem_badtitle'", []).unwrap();
        conn.execute("UPDATE memories SET title = CAST(x'C328' AS TEXT) WHERE id = 'mem_badbytes'", []).unwrap();
    }
    let (b, _status) = boot(Some(Embedder::hashing()), TestHooks::default());
    let (mut ctx, jobs_rx, _vec_rx) = indexer_start::open_ctx(dirs.ext_config("hash"), b);
    indexer_start::pump_embeddings(&mut ctx).expect("one unreadable title failed the pump");
    let Ok(Job::Embedded { vectors, .. }) = jobs_rx.recv_timeout(Duration::from_secs(10)) else {
        panic!("no Job::Embedded")
    };
    assert_eq!(vectors.len(), 3);
}

// ── Wave 4, R-12b item 4: a vector goes only to the row its text was read from ──

fn stored_vector(conn: &Connection, mem_id: &str) -> Option<Vec<u8>> {
    conn.query_row("SELECT embedding FROM memory_ext WHERE mem_id = ?1", [mem_id], |r| r.get(0)).unwrap()
}

fn hashed_blob(text: &str) -> Vec<u8> {
    blob(&Embedder::hashing().embed(&[text]).unwrap()[0])
}

#[test]
fn w4_4_a_row_purged_while_its_text_is_embedded_does_not_give_its_vector_to_the_next_row() {
    let dirs = Dirs::new();
    let conn = make_brain(&dirs.brain());
    let old = insert_memory(&conn, "mem_old", None, None, false);
    // Every batch stays 400 ms with the embed thread.
    let test = TestHooks { embed_sleep_ms: 400, ..TestHooks::default() };
    let (b, _status) = boot(Some(Embedder::hashing()), test);
    let (mut ctx, jobs_rx, vec_rx) = indexer_start::open_ctx(dirs.ext_config("hash"), b);
    assert!(matches!(vec_rx.try_recv(), Ok(VecDelta::Replace(_))));
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    assert_eq!(ctx.embed.backlog(), 1);

    // While the text is out: the row is purged and another is stored. It gets the number.
    // (The delete guard lets a memory go only while `axel_purge` holds its row, as in
    // `store::purge_many`; the trigger on `memories` removes the `memory_ext` row.)
    conn.execute_batch(
        "BEGIN IMMEDIATE;
         INSERT INTO axel_purge(one) VALUES (1);
         DELETE FROM memories WHERE id = 'mem_old';
         DELETE FROM axel_purge;
         COMMIT;",
    )
    .unwrap();
    assert_eq!(conn.query_row("SELECT COUNT(*) FROM memory_ext", [], |r| r.get::<_, i64>(0)).unwrap(), 0);
    let new = insert_memory(&conn, "mem_new", None, None, false);
    assert_eq!(new, old, "SQLite gives the number of the purged newest row to the next row");

    let Ok(Job::Embedded { model, vectors, out }) = jobs_rx.recv_timeout(Duration::from_secs(10)) else {
        panic!("no Job::Embedded")
    };
    assert_eq!(vectors.iter().map(|(k, _)| *k).collect::<Vec<_>>(), [VecKey::Mem(old)]);
    indexer_start::on_embedded(&mut ctx, &model, vectors, out.as_deref()).unwrap();
    assert_eq!(stored_vector(&conn, "mem_new"), None, "the new row got the vector of the purged text");
    assert!(vec_rx.try_recv().is_err(), "the front thread was sent the vector of the purged text");
    drop(out);

    // The new row is embedded in its own turn, from its own text.
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    let Ok(Job::Embedded { model, vectors, out }) = jobs_rx.recv_timeout(Duration::from_secs(10)) else {
        panic!("no Job::Embedded")
    };
    indexer_start::on_embedded(&mut ctx, &model, vectors, out.as_deref()).unwrap();
    assert_eq!(
        stored_vector(&conn, "mem_new"),
        Some(hashed_blob("title of mem_new\nthe body of mem_new about retry limits"))
    );
    assert!(matches!(vec_rx.try_recv(), Ok(VecDelta::Upsert(items)) if items.len() == 1));
}

#[test]
fn w4_4_a_row_made_secret_or_rewritten_while_its_text_is_embedded_gets_no_vector() {
    let dirs = Dirs::new();
    let conn = make_brain(&dirs.brain());
    insert_memory(&conn, "mem_kept", None, None, false);
    insert_memory(&conn, "mem_secret", None, None, false);
    insert_memory(&conn, "mem_rewritten", None, None, false);
    let test = TestHooks { embed_sleep_ms: 400, ..TestHooks::default() };
    let (b, _status) = boot(Some(Embedder::hashing()), test);
    let (mut ctx, jobs_rx, vec_rx) = indexer_start::open_ctx(dirs.ext_config("hash"), b);
    assert!(matches!(vec_rx.try_recv(), Ok(VecDelta::Replace(_))));
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    assert_eq!(ctx.embed.backlog(), 3);

    conn.execute("UPDATE memories SET sensitivity = 'secret' WHERE id = 'mem_secret'", []).unwrap();
    conn.execute("UPDATE memory_ext SET content_hash = 'another body' WHERE mem_id = 'mem_rewritten'", []).unwrap();

    let Ok(Job::Embedded { model, vectors, out }) = jobs_rx.recv_timeout(Duration::from_secs(10)) else {
        panic!("no Job::Embedded")
    };
    assert_eq!(vectors.len(), 3);
    // A job that does not say which rows it was read from writes nothing.
    indexer_start::on_embedded(&mut ctx, &model, vectors.clone(), None).unwrap();
    assert_eq!(stored_vector(&conn, "mem_kept"), None);
    indexer_start::on_embedded(&mut ctx, &model, vectors, out.as_deref()).unwrap();
    assert!(stored_vector(&conn, "mem_kept").is_some());
    assert_eq!(stored_vector(&conn, "mem_secret"), None, "a secret row got a vector");
    assert_eq!(stored_vector(&conn, "mem_rewritten"), None, "a rewritten row got the vector of its old text");
    assert!(matches!(vec_rx.try_recv(), Ok(VecDelta::Upsert(items)) if items.len() == 1));
}

// Wave 5 review, finding 4: `content_hash` covers the content only; the text sent is
// `title\ncontent`.
#[test]
fn w4_4_a_title_edited_while_its_text_is_embedded_gets_no_vector_of_the_old_title() {
    let dirs = Dirs::new();
    let conn = make_brain(&dirs.brain());
    let vid = insert_memory(&conn, "mem_titled", None, None, false);
    let test = TestHooks { embed_sleep_ms: 400, ..TestHooks::default() };
    let (b, _status) = boot(Some(Embedder::hashing()), test);
    let (mut ctx, jobs_rx, vec_rx) = indexer_start::open_ctx(dirs.ext_config("hash"), b);
    assert!(matches!(vec_rx.try_recv(), Ok(VecDelta::Replace(_))));
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    assert_eq!(ctx.embed.backlog(), 1);

    // Same `mem_id`, same `vid`, same `content_hash`: only the title moves.
    conn.execute("UPDATE memories SET title = 'a new title' WHERE id = 'mem_titled'", []).unwrap();
    assert_eq!(conn.query_row("SELECT vid FROM memory_ext WHERE mem_id = 'mem_titled'", [], |r| r.get::<_, i64>(0)).unwrap(), vid);

    let Ok(Job::Embedded { model, vectors, out }) = jobs_rx.recv_timeout(Duration::from_secs(10)) else {
        panic!("no Job::Embedded")
    };
    indexer_start::on_embedded(&mut ctx, &model, vectors, out.as_deref()).unwrap();
    assert_eq!(stored_vector(&conn, "mem_titled"), None, "the row got the vector of its old title");
    assert!(vec_rx.try_recv().is_err());
    drop(out);

    // Its own text is embedded in the next turn.
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    let Ok(Job::Embedded { model, vectors, out }) = jobs_rx.recv_timeout(Duration::from_secs(10)) else {
        panic!("no Job::Embedded")
    };
    indexer_start::on_embedded(&mut ctx, &model, vectors, out.as_deref()).unwrap();
    assert_eq!(
        stored_vector(&conn, "mem_titled"),
        Some(hashed_blob("a new title\nthe body of mem_titled about retry limits"))
    );
}

// ── Wave 4, R-12b item 5: the backlog goes down when the job is done ────────

#[test]
fn w4_5_rows_whose_vectors_wait_in_the_queue_are_not_offered_a_second_time() {
    let dirs = Dirs::new();
    let conn = make_brain(&dirs.brain());
    for id in ["mem_a", "mem_b", "mem_c"] {
        insert_memory(&conn, id, None, None, false);
    }
    let (b, _status) = boot(Some(Embedder::hashing()), TestHooks::default());
    let (mut ctx, jobs_rx, _vec_rx) = indexer_start::open_ctx(dirs.ext_config("hash"), b);
    indexer_start::pump_embeddings(&mut ctx).unwrap();

    // The job is on the indexer's queue, behind whatever else is there: taken off the
    // queue and not yet handled, as it is while another job runs.
    let Ok(job) = jobs_rx.recv_timeout(Duration::from_secs(10)) else { panic!("no Job::Embedded") };
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(ctx.embed.backlog(), 3, "the count came off when the job was queued");
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    assert!(jobs_rx.recv_timeout(Duration::from_millis(500)).is_err(), "the same rows were embedded twice");

    // Handled the way the loop handles it: the count comes off, and nothing waits.
    indexer::dispatch(&mut ctx, job);
    assert_eq!(ctx.embed.backlog(), 0);
    assert_eq!(conn.query_row("SELECT COUNT(*) FROM memory_ext WHERE embedding IS NULL", [], |r| r.get::<_, i64>(0)).unwrap(), 0);

    // A job thrown away unhandled (a degraded process runs no handler) gives its count back too.
    insert_memory(&conn, "mem_d", None, None, false);
    indexer_start::pump_embeddings(&mut ctx).unwrap();
    let Ok(job) = jobs_rx.recv_timeout(Duration::from_secs(10)) else { panic!("no Job::Embedded") };
    assert_eq!(ctx.embed.backlog(), 1);
    degrade(&ctx.status, &ctx.mode, DegradedReason::BrainNotReady, None);
    indexer::dispatch(&mut ctx, job);
    assert_eq!(ctx.embed.backlog(), 0);
    assert_eq!(stored_vector(&conn, "mem_d"), None);
}

// ── Wave 4, R-12b item 6: the place is kept with its reason ─────────────────

#[test]
fn w4_6_after_a_recovery_a_second_failure_does_not_show_the_place_of_the_first() {
    // On the counters: a place is read only beside the reason it was given with.
    let mode = AtomicU8::new(Mode::Ready as u8);
    let status = Status::default();
    degrade(&status, &mode, DegradedReason::IndexerStopped, Some("first.rs:1".to_string()));
    assert_eq!(status.reason_place(), Some("first.rs:1"));
    recover(&status, &mode);
    assert_eq!(status.reason_place(), None);
    degrade(&status, &mode, DegradedReason::BrainChanged, None);
    assert_eq!(status.reason_place(), None, "the place of the first failure beside the second");
    degrade(&status, &mode, DegradedReason::FrontStopped, Some("second.rs:2".to_string()));
    assert_eq!(status.reason_place(), Some("second.rs:2"));

    // In the status file of a running process: an earlier failure left its place, then the
    // brain file goes.
    let dirs = Dirs::new();
    let host = Host::ready(&dirs, None, TestHooks { status_tick_ms: Some(50), ..TestHooks::default() });
    host.status.set_reason_place(DegradedReason::IndexerStopped, "first.rs:1".to_string());
    std::fs::remove_file(dirs.brain()).unwrap();
    let said = wait_status_file(&dirs, "degraded", Duration::from_secs(10));
    assert_eq!((&said["reason"], &said["reason_at"]), (&json!("BrainChanged"), &Value::Null), "{said}");
    assert_eq!(host.close(), ServeEnd::InputClosed);
}

// ── Wave 4, R-12b item 7: the config keeps the home it was read with ────────

#[test]
fn w4_7_the_config_keeps_the_home_it_was_read_with() {
    let home = TempDir::new().unwrap();
    let cfg = ExtConfig::from_initialize(&json!({"config": {"sources": "~/notes/sources.toml"}}), home.path());
    assert_eq!(cfg.home, home.path());
    assert_eq!(cfg.sources.as_deref(), Some(home.path().join("notes/sources.toml").as_path()));
    assert_ne!(Some(cfg.home.as_os_str()), std::env::var_os("HOME").as_deref(), "the test's home is not the process's");
}

// ── Wave 4, R-12b item 8: the status row says how long a purge has waited ───

#[test]
fn w4_8_the_status_row_gives_the_age_of_the_purge_mark() {
    let dirs = Dirs::new();
    let (b, _status) = boot(None, TestHooks::default());
    let (mut ctx, _jobs_rx, _vec_rx) = indexer_start::open_ctx(dirs.ext_config("off"), b);
    let row = status_row(&dirs.brain()).unwrap();
    assert!(row.as_object().unwrap().contains_key("purge_pending_ms"), "{row}");
    assert_eq!(row["purge_pending_ms"], Value::Null, "{row}");

    // A purge leaves its mark, as `index_archives::mark_purged` writes it.
    let conn = Connection::open(dirs.brain()).unwrap();
    axel::index_archives::mark_purged(&conn).unwrap();
    indexer_start::write_status(&mut ctx).unwrap();
    let age = status_row(&dirs.brain()).unwrap()["purge_pending_ms"].as_i64().expect("an age in milliseconds");
    assert!((0..60_000).contains(&age), "{age}");

    // The mark has waited ten minutes.
    conn.execute(
        "UPDATE context_data SET updated = datetime('now', '-10 minutes') WHERE key = 'purge_pending' AND layer = 'axel'",
        [],
    )
    .unwrap();
    indexer_start::write_status(&mut ctx).unwrap();
    let age = status_row(&dirs.brain()).unwrap()["purge_pending_ms"].as_i64().unwrap();
    assert!((599_000..660_000).contains(&age), "{age}");

    // The log was truncated and the mark is gone.
    conn.execute("DELETE FROM context_data WHERE key = 'purge_pending' AND layer = 'axel'", []).unwrap();
    indexer_start::write_status(&mut ctx).unwrap();
    assert_eq!(status_row(&dirs.brain()).unwrap()["purge_pending_ms"], Value::Null);
}

// ── v2 front follow-up ──────────────────────────────────────────────────────

const FRONT_PROJECT: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const FRONT_T: i64 = 1_760_000_000_000;

/// One archive row of chat `session-S`, the last turn of an archive sealed at `FRONT_T`:
/// what a forget call at that time purges. Returns its `vid`.
fn forgettable_turn(dirs: &Dirs) -> i64 {
    let conn = make_brain(&dirs.brain());
    let logical = axel::project::logical("session-S");
    conn.execute_batch("BEGIN IMMEDIATE").unwrap();
    let row = NewRow {
        id: "mem_turn".to_string(),
        class: "episodic_turn",
        project_key: FRONT_PROJECT,
        title: "the forgotten turn".to_string(),
        abstract_text: String::new(),
        content: "User: the deploy key rotation | Assistant: rotated on the build host".to_string(),
        tags: Vec::new(),
        created_ms: FRONT_T - 1_000,
        time_source: "archive",
        sensitivity: Sensitivity::Normal,
        provenance: "test".to_string(),
        expires_ms: None,
        importance: 0.5,
        origin_kind: "archive",
        origin_ref: "arch-1",
        unit_pos: Some("x3.0".to_string()),
        session_id: Some("session-S"),
        logical: Some(&logical),
        links: RecordLinks::default(),
        turn_key: Some("9f2c1a7e04b35d66".to_string()),
        hidden: None,
    };
    let Inserted::New { vid } = store::insert_row(&conn, None, &row).unwrap() else { panic!("not inserted") };
    conn.execute(
        "INSERT INTO archive_index(scope_dir, rel_name, archive_id, logical, state, time_source, cover_ms,
                                   session_id, first_seen_ms)
         VALUES (?1, 's000001/arch-1.json', 'arch-1', ?2, 'indexed', 'archive', ?3, 'session-S', ?3)",
        rusqlite::params![FRONT_PROJECT, logical, FRONT_T],
    )
    .unwrap();
    conn.execute_batch("COMMIT").unwrap();
    vid
}

/// A ready front state whose clock stands at `FRONT_T`, with the row's vector in its array,
/// and the other ends of its channels.
fn front_with_vector(dirs: &Dirs, vid: i64) -> (FrontState, Receiver<Job>, Sender<VecDelta>) {
    let status = Arc::new(Status::default());
    let b = Boot { embedder: None, clock: Arc::new(|| FRONT_T), test: Arc::new(TestHooks::default()), status };
    let (mut st, jobs_rx, vec_tx) = front::open_state(dirs.ext_config("off"), b);
    assert_eq!(st.mode.load(Ordering::SeqCst), Mode::Ready as u8);
    let item = VecItem { key: VecKey::Mem(vid), scope: VecScope::Project(FRONT_PROJECT.to_string()), v: unit(3) };
    assert_eq!(st.index.apply(VecDelta::Upsert(vec![item])), 0);
    assert!(st.index.get(&VecKey::Mem(vid)).is_some());
    (st, jobs_rx, vec_tx)
}

fn forget_in_s() -> FrontCall {
    FrontCall {
        seq: 1,
        carry: Carry { skipped: Vec::new(), forgot: vec!["session-S".to_string()] },
        work: FrontWork::Tool { name: "search".to_string(), input: json!({}), caller: Caller::default() },
    }
}

fn rows_named(dirs: &Dirs, id: &str) -> i64 {
    Connection::open(dirs.brain())
        .unwrap()
        .query_row("SELECT COUNT(*) FROM memories WHERE id = ?1", [id], |r| r.get(0))
        .unwrap()
}

#[test]
fn f1_a_forget_call_takes_the_purged_rows_vector_out_of_the_array() {
    let dirs = Dirs::new();
    let vid = forgettable_turn(&dirs);
    let (mut st, _jobs_rx, _vec_tx) = front_with_vector(&dirs, vid);
    front::handle(&mut st, forget_in_s());
    assert_eq!(rows_named(&dirs, "mem_turn"), 0, "the mark purged nothing");
    assert!(st.index.get(&VecKey::Mem(vid)).is_none(), "the vector of a purged row is still in the array");
}

#[test]
fn f1_a_kept_mark_written_late_takes_the_vector_out_too() {
    let dirs = Dirs::new();
    let vid = forgettable_turn(&dirs);
    let (mut st, _jobs_rx, _vec_tx) = front_with_vector(&dirs, vid);
    st.forget_marks =
        vec![ForgetMark { session_id: "session-S".to_string(), from_ms: FRONT_T, to_ms: FRONT_T, turn_key: None }];
    assert_eq!(front::retry_written(&mut st), 1);
    assert_eq!(rows_named(&dirs, "mem_turn"), 0);
    assert!(st.index.get(&VecKey::Mem(vid)).is_none(), "the vector of a purged row is still in the array");
}

#[test]
fn f1_a_mark_whose_commit_fails_removes_no_vector() {
    let dirs = Dirs::new();
    let vid = forgettable_turn(&dirs);
    // The purge runs, then the `COMMIT` fails: a deferred foreign key is left broken.
    Connection::open(dirs.brain())
        .unwrap()
        .execute_batch(
            "CREATE TABLE t_parent(id TEXT PRIMARY KEY);
             CREATE TABLE t_child(p TEXT REFERENCES t_parent(id) DEFERRABLE INITIALLY DEFERRED);
             CREATE TRIGGER t_break AFTER UPDATE OF forget_units ON archive_index
             BEGIN INSERT INTO t_child VALUES ('nobody'); END;",
        )
        .unwrap();
    let (mut st, _jobs_rx, _vec_tx) = front_with_vector(&dirs, vid);
    front::handle(&mut st, forget_in_s());
    assert_eq!(rows_named(&dirs, "mem_turn"), 1, "the transaction was not rolled back");
    let marks: i64 = Connection::open(dirs.brain())
        .unwrap()
        .query_row("SELECT COUNT(*) FROM forgotten_turns", [], |r| r.get(0))
        .unwrap();
    assert_eq!(marks, 0);
    assert!(st.index.get(&VecKey::Mem(vid)).is_some(), "a rolled-back mark removed a vector");
}

#[test]
fn f2_a_carry_that_takes_the_budget_leaves_recall_none_of_it() {
    let dirs = Dirs::new();
    drop(make_brain(&dirs.brain()));
    let (b, _status) = boot(None, TestHooks::default());
    let (mut st, jobs_rx, _vec_tx) = front::open_state(dirs.ext_config("off"), b);
    assert!(st.cfg.recall);
    // The forget mark of the carry waits the front thread's 1.5 s for a locked brain.
    let holder = Connection::open(dirs.brain()).unwrap();
    holder.execute_batch("BEGIN IMMEDIATE").unwrap();
    let ev = HookEvent {
        kind: "before_message".to_string(),
        session_id: Some("session-S".to_string()),
        cwd: None,
        message: Some("where is the retry limit of the deploy set".to_string()),
        data: json!({}),
        received_ms: now_ms(),
    };
    let deadline = Instant::now() + Duration::from_millis(st.cfg.recall_budget_ms);
    let call = FrontCall {
        seq: 1,
        carry: Carry { skipped: Vec::new(), forgot: vec!["session-S".to_string()] },
        work: FrontWork::Hook { ev, recall_allowed: true, deadline },
    };
    let start = Instant::now();
    let answer = front::handle(&mut st, call);
    assert!(start.elapsed() >= Duration::from_millis(1_000), "the carry did not wait: {:?}", start.elapsed());
    assert!(matches!(answer, FrontAnswer::Hook(HookOut::Continue)));
    assert_eq!(st.forget_marks.len(), 1);
    let outcomes: Vec<String> = jobs_rx
        .try_iter()
        .filter_map(|job| match job {
            Job::LogRecall(trace) => Some(trace.outcome),
            _ => None,
        })
        .collect();
    assert_eq!(outcomes, ["timeout"], "recall worked a budget of its own after the rpc thread's had passed");
    holder.execute_batch("ROLLBACK").unwrap();
}

#[test]
fn f3_a_label_takes_its_home_from_the_config_not_from_home() {
    let dirs = Dirs::new();
    let ext = dirs.ext_config("off");
    assert_eq!(ext.home, dirs.path());
    assert_ne!(std::env::var_os("HOME").as_deref(), Some(dirs.path().as_os_str()));
    let cfg = RecallConfig::from_ext(&ext, now_ms());
    assert_eq!(cfg.home.as_deref(), Some(dirs.path()));
    let file = dirs.path().join("notes").join("deploy.md");
    let pick = Pick {
        n: 1,
        r#ref: "doc:notes::deploy".to_string(),
        descriptor: ScopedDescriptor {
            id: "doc:notes::deploy".to_string(),
            project_key: String::new(),
            title: "deploy".to_string(),
            snippet: Some("the deploy notes".to_string()),
            category: "note".to_string(),
            tags: Vec::new(),
            created: String::new(),
            sensitivity: Sensitivity::Normal,
            retention: Retention::Standard,
            provenance: file.to_string_lossy().into_owned(),
            links: RecordLinks::default(),
        },
        reasons: Vec::new(),
        parts: ScoreParts::default(),
        origin: Origin::Document,
        date: None,
    };
    assert_eq!(axel::inject::who(&pick, &cfg, None), "file in ~/notes: deploy.md");
}

#[test]
fn f4_a_turn_the_carry_drops_is_counted_at_once() {
    let dirs = Dirs::new();
    let (b, status) = boot(None, TestHooks::default());
    let (mut st, _jobs_rx, _vec_tx) = front::open_state(dirs.ext_config("off"), b);
    assert!(st.cfg.capture);
    st.capture.on_user(Some("session-D"), Some("/tmp/project"), "how is the deploy key rotated on the host", now_ms());
    assert!(st.capture.is_pending("session-D"));
    let call = FrontCall {
        seq: 1,
        carry: Carry { skipped: vec!["session-D".to_string()], forgot: Vec::new() },
        work: FrontWork::Tool { name: "search".to_string(), input: json!({}), caller: Caller::default() },
    };
    front::handle(&mut st, call);
    assert!(!st.capture.is_pending("session-D"));
    assert_eq!(status.turns_dropped.load(Ordering::Relaxed), 1, "the dropped turn waits for another hook");
}

#[test]
fn f5_kept_marks_on_a_locked_brain_are_reported_lost_inside_serves_wait() {
    let dirs = Dirs::new();
    let mut host = Host::ready(&dirs, None, TestHooks::default());
    let holder = Connection::open(dirs.brain()).unwrap();
    holder.execute_batch("BEGIN IMMEDIATE").unwrap();
    assert_eq!(host.hook("on_memory_forgotten", "session-S"), continues());
    assert_eq!(host.hook("before_message", "session-S"), continues());
    wait_for_log(&dirs, "front carry: forget mark not written, kept");
    // The front thread has answered the late hook and is idle again.
    std::thread::sleep(Duration::from_millis(300));

    let start = Instant::now();
    assert_eq!(host.close(), ServeEnd::InputClosed);
    let took = start.elapsed();
    let log = read_log(&dirs);
    assert!(log.contains("front carry: forget marks lost at the end: 1"), "{log}");
    assert!(took < Duration::from_millis(1_000), "{took:?}");
    holder.execute_batch("ROLLBACK").unwrap();
}

// ── DegradedReason::COUNT ───────────────────────────────────────────────────

#[test]
fn every_degraded_reason_has_its_own_place_in_status() {
    assert_eq!(DegradedReason::ALL.len(), DegradedReason::COUNT);
    let status = Status::default();
    assert_eq!(status.reason_at.len(), DegradedReason::COUNT);
    let mut names = std::collections::HashSet::new();
    for (i, reason) in DegradedReason::ALL.into_iter().enumerate() {
        assert_eq!(reason as usize, i);
        assert_eq!(DegradedReason::from_u8(i as u8), reason, "from_u8 does not read {reason:?} back");
        assert!(names.insert(reason.as_str()), "{} twice", reason.as_str());
        status.set_reason_place(reason, format!("place-{i}"));
        status.reason.store(reason as u8, std::sync::atomic::Ordering::SeqCst);
        assert_eq!(status.reason_place(), Some(format!("place-{i}").as_str()), "{reason:?} has no place");
    }
    // A value past the last reason names none: the enum has exactly `COUNT` variants that
    // `from_u8` reads.
    for v in DegradedReason::COUNT..=usize::from(u8::MAX) {
        assert_eq!(DegradedReason::from_u8(v as u8), DegradedReason::None, "{v} names a reason past COUNT");
    }
}
