//! Commit 23, end to end through the real binary: section 7 test 12, section 8 tests 6, 9
//! and 10, and the binary half of section 1 test 17 (every subcommand clap lists).
//!
//! Every process runs `env!("CARGO_BIN_EXE_axel")` with `HOME` pointing into the test's own
//! temp folder, so the default brain, the default note folders and the model folder are all
//! under it. The extension is driven as Synaps drives it: JSON-RPC frames on its stdin and
//! stdout. Nothing waits a fixed time: every wait polls with a deadline.

use std::fs::File;
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Output, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use axel::ext::rpc::{write_frame, FrameError, FrameReader};
use axel::ext::tools_core::WORKER_REFUSAL;
use axel::ext::tools_forum::NO_CALLER;
use axel::project;
use axel_memkoshi::memory::{Memory, MemoryCategory};
use axel_memkoshi::storage::MemoryStorage;
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tempfile::TempDir;

const BIN: &str = env!("CARGO_BIN_EXE_axel");
/// No test of this file may take longer on its own (wave 6 brief).
const TEST_LIMIT: Duration = Duration::from_secs(60);
/// The longest any one poll waits.
const POLL_LIMIT: Duration = Duration::from_secs(20);
const ANSWER_LIMIT: Duration = Duration::from_secs(10);

const CORE_FIXTURE: &str = "01a8f6fb3ec12e1d59d07bd6e3943c84c18415e3111dd5cba218b7ce9218b0c7";
const CARRIED_FIXTURE: &str = "c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6c6";
const RETRY_ARCHIVE: &str = "s000001/b436b6196e6ca0758c70d5faed628f29.json";
const NEW_ID: &str = "e2e00000000000000000000000000023";
const DOWN_ID: &str = "e2e00000000000000000000000000024";
const DECOY_ID: &str = "dec00000000000000000000000000010";

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn sha256(path: &Path) -> String {
    hex::encode(Sha256::digest(std::fs::read(path).unwrap()))
}

/// Every file under `dir`, as paths relative to it, sorted.
fn files_under(dir: &Path) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if entry.file_type().is_ok_and(|t| t.is_dir()) {
                walk(root, &path, out);
            } else {
                out.push(path.strip_prefix(root).unwrap().to_string_lossy().into_owned());
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

fn wait_for<T>(what: &str, mut probe: impl FnMut() -> Option<T>) -> T {
    let start = Instant::now();
    loop {
        if let Some(v) = probe() {
            return v;
        }
        assert!(start.elapsed() < POLL_LIMIT, "waited {POLL_LIMIT:?} for: {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

// ── A machine: a temp home, a Synaps base, two projects, a note folder ─────

struct World {
    tmp: TempDir,
    /// The project key and directory of `proj`, as core derives them.
    key: String,
    scope: String,
}

impl World {
    fn new() -> World {
        let tmp = TempDir::new().unwrap();
        for dir in ["home", "base/context-archives", "base/memory", "proj/.git", "other/.git", "notes", "model"] {
            std::fs::create_dir_all(tmp.path().join(dir)).unwrap();
        }
        let sources = format!(
            "[[source]]\nname = \"e2e\"\npath = \"{}\"\npriority = \"medium\"\n",
            tmp.path().join("notes").display()
        );
        std::fs::write(tmp.path().join("sources.toml"), sources).unwrap();
        let p = project::resolve(Some(tmp.path().join("proj").to_str().unwrap())).unwrap();
        World { tmp, key: p.key, scope: p.scope_dir }
    }
    fn root(&self) -> &Path {
        self.tmp.path()
    }
    fn home(&self) -> PathBuf {
        self.root().join("home")
    }
    fn base(&self) -> PathBuf {
        self.root().join("base")
    }
    fn proj(&self) -> PathBuf {
        self.root().join("proj")
    }
    fn other(&self) -> PathBuf {
        self.root().join("other")
    }
    fn notes(&self) -> PathBuf {
        self.root().join("notes")
    }
    /// The default brain of the temp home: no config key and no `AXEL_BRAIN` name it.
    fn brain(&self) -> PathBuf {
        self.home().join(".local/share/axel/brain.r8")
    }
    fn project_archives(&self) -> PathBuf {
        self.base().join("context-archives").join(&self.scope)
    }
    /// The `config` of `initialize.params`. The brain is not named: it is the default one.
    fn config(&self) -> Value {
        json!({
            "synaps_base": self.base().to_string_lossy(),
            "embedder": "hash",
            "model_dir": self.root().join("model").to_string_lossy(),
            "sources": self.root().join("sources.toml").to_string_lossy(),
            "recall_budget_ms": "2000",
        })
    }

    /// One archive record with this project's directory as its `scope`, at `rel` under it.
    fn put_archive(&self, record: &Path, rel: &str) -> PathBuf {
        let mut value: Value = serde_json::from_slice(&std::fs::read(record).unwrap()).unwrap();
        value["scope"] = json!(self.scope);
        let path = self.project_archives().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, value.to_string()).unwrap();
        path
    }

    /// Core's fixture archives, and the copy of the retry archive whose exchange is read whole.
    fn put_fixture_archives(&self) -> usize {
        let core = fixtures().join("context-archives").join(CORE_FIXTURE);
        let mut n = 0;
        for rel in files_under(&core) {
            let from = if rel == RETRY_ARCHIVE {
                fixtures().join("context-archives").join(CARRIED_FIXTURE).join(RETRY_ARCHIVE)
            } else {
                core.join(&rel)
            };
            self.put_archive(&from, &rel);
            n += 1;
        }
        n
    }

    fn put_e2e_archive(&self, fixture: &str, id: &str) -> PathBuf {
        self.put_archive(&fixtures().join("e2e").join(fixture), &format!("s000001/{id}.json"))
    }

    /// What core leaves of a forgotten archive: the same header, no body.
    fn tombstone(&self, path: &Path) {
        let live: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let dead = json!({"version": 1, "scope": live["scope"], "logical": live["logical"], "id": live["id"],
                          "digest": live["digest"], "body": {"state": "tombstone"}});
        std::fs::write(path, dead.to_string()).unwrap();
    }

    /// The project's notes file, as core writes it.
    fn put_notes(&self, notes: &[(&str, &str)]) {
        let mut text = String::new();
        for (i, (id, content)) in notes.iter().enumerate() {
            let record = json!({"namespace": format!("project-{}", self.key), "timestamp_ms": 1_777_000_000_000i64 + i as i64,
                                "content": content, "tags": [], "id": id, "project": self.key,
                                "sensitivity": "normal", "retention": "standard"});
            text.push_str(&record.to_string());
            text.push('\n');
        }
        std::fs::write(self.base().join("memory").join(format!("project-{}.jsonl", self.key)), text).unwrap();
    }

    /// `axel status --json`, as a person runs it.
    fn status(&self) -> Value {
        let out = Command::new(BIN)
            .arg("--brain")
            .arg(self.brain())
            .args(["status", "--json"])
            .current_dir(self.root())
            .env("HOME", self.home())
            .env_remove("AXEL_BRAIN")
            .output()
            .unwrap();
        serde_json::from_slice(&out.stdout).unwrap_or(Value::Null)
    }

    /// Polls `status --json` until `done` holds; returns that status.
    fn wait_status(&self, what: &str, done: impl Fn(&Value) -> bool) -> Value {
        let start = Instant::now();
        loop {
            let status = self.status();
            if done(&status) {
                return status;
            }
            assert!(start.elapsed() < POLL_LIMIT, "waited {POLL_LIMIT:?} for: {what}; status --json said {status}");
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    fn wait_ready(&self, what: &str) -> Value {
        self.wait_status(what, |s| s["extension"]["mode"] == "ready")
    }

    /// A read-only look at the brain, closed when dropped.
    fn look(&self) -> Connection {
        let conn = Connection::open_with_flags(self.brain(), OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)
            .unwrap();
        conn.busy_timeout(Duration::from_secs(5)).unwrap();
        conn
    }

    fn count(&self, sql: &str, param: &str) -> i64 {
        self.look().query_row(sql, [param], |r| r.get(0)).unwrap_or_else(|e| panic!("{sql}: {e}"))
    }

    /// Memories whose title or content holds `text`.
    fn rows_holding(&self, text: &str) -> i64 {
        self.count("SELECT COUNT(*) FROM memories WHERE instr(content, ?1) > 0 OR instr(title, ?1) > 0", text)
    }

    fn rows_of_archive(&self, id: &str) -> i64 {
        self.count("SELECT COUNT(*) FROM memory_ext WHERE origin_kind = 'archive' AND origin_ref = ?1", id)
    }

    fn memory_ids(&self) -> Vec<String> {
        let conn = self.look();
        let mut stmt = conn.prepare("SELECT id FROM memories ORDER BY id").unwrap();
        let ids = stmt.query_map([], |r| r.get(0)).unwrap();
        ids.collect::<rusqlite::Result<_>>().unwrap()
    }
}

fn archives_known(s: &Value) -> i64 {
    ["indexed", "tombstone", "skipped"].iter().map(|k| s["archives"][k].as_i64().unwrap_or(0)).sum()
}

// ── The host: the real binary on pipes ──────────────────────────────────────

/// A forum author as a chat's run sends it in `caller`.
fn author() -> Value {
    json!({"actor": format!("actor-{}", "a1".repeat(16)), "group": format!("group-{}", "b2".repeat(16))})
}

struct Ext {
    child: Child,
    stdin: Option<ChildStdin>,
    frames: Receiver<Vec<u8>>,
    next_id: u64,
}

impl Ext {
    /// `axel extension`, then `initialize`.
    fn start(w: &World, life: &str) -> Ext {
        let stderr = File::create(w.root().join(format!("stderr-{life}"))).unwrap();
        let mut child = Command::new(BIN)
            .arg("extension")
            .current_dir(w.root())
            .env("HOME", w.home())
            .env_remove("AXEL_BRAIN")
            .env_remove("AXEL_SOURCES")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr))
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, frames) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = FrameReader::new(BufReader::new(stdout));
            while let Some(frame) = reader.next_frame() {
                match frame {
                    Ok(body) => {
                        if tx.send(body).is_err() {
                            return;
                        }
                    }
                    Err(FrameError::Io(_) | FrameError::Truncated) => return,
                    Err(_) => {}
                }
            }
        });
        let stdin = child.stdin.take();
        let mut ext = Ext { child, stdin, frames, next_id: 0 };
        let answer = ext.call(
            "initialize",
            json!({"synaps_version": "e2e", "extension_protocol_version": 1, "plugin_id": "axel",
                   "plugin_root": w.root().join("plugins/axel").to_string_lossy(), "config": w.config()}),
        );
        assert_eq!(answer["result"]["protocol_version"], 1, "initialize: {answer}");
        ext
    }

    fn call(&mut self, method: &str, params: Value) -> Value {
        self.next_id += 1;
        let id = self.next_id;
        let body = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
        let mut frame = Vec::new();
        write_frame(&mut frame, body.to_string().as_bytes()).unwrap();
        let stdin = self.stdin.as_mut().expect("input closed");
        stdin.write_all(&frame).unwrap();
        stdin.flush().unwrap();
        let answer: Value = serde_json::from_slice(
            &self.frames.recv_timeout(ANSWER_LIMIT).unwrap_or_else(|e| panic!("no answer to {method}: {e}")),
        )
        .unwrap();
        assert_eq!(answer["id"], id, "answer to another request: {answer}");
        answer
    }

    /// `hook.handle`; returns `result`.
    fn hook(&mut self, kind: &str, session: Option<&str>, cwd: Option<&Path>, message: Option<&str>, data: Value) -> Value {
        let params = json!({"kind": kind, "session_id": session, "cwd": cwd.map(|c| c.to_string_lossy()),
                            "message": message, "data": data});
        self.call("hook.handle", params)["result"].clone()
    }

    fn prompt(&mut self, session: &str, cwd: &Path, message: &str) -> Value {
        self.hook("before_message", Some(session), Some(cwd), Some(message), Value::Null)
    }

    /// `tool.call`; returns `result` (`content`, `is_error`).
    fn tool(&mut self, name: &str, input: Value, session: Option<&str>, cwd: &Path, caller: Option<Value>) -> Value {
        let mut params = json!({"name": name, "input": input, "session_id": session, "cwd": cwd.to_string_lossy()});
        if let Some(caller) = caller {
            params["caller"] = caller;
        }
        let answer = self.call("tool.call", params);
        assert!(answer["result"].is_object(), "tool.call {name}: {answer}");
        answer["result"].clone()
    }

    /// A chat's `search` in the folder `cwd`; the text of the answer.
    fn search(&mut self, query: &str, cwd: &Path) -> String {
        let out = self.tool("search", json!({"query": query, "limit": 10}), Some("s-searcher"), cwd, Some(author()));
        assert_eq!(out["is_error"], false, "search {query:?}: {out}");
        out["content"].as_str().unwrap().to_string()
    }

    /// SIGKILL: no shutdown, no flush, no cleanup.
    fn kill(mut self) {
        self.child.kill().unwrap();
        self.child.wait().unwrap();
    }

    /// `shutdown`, then the process must exit by itself.
    fn shutdown(mut self) {
        let answer = self.call("shutdown", Value::Null);
        assert!(answer.get("error").is_none(), "shutdown: {answer}");
        let code = wait_for("the process to exit after shutdown", || self.child.try_wait().unwrap());
        assert!(code.success(), "exit after shutdown: {code:?}");
    }
}

impl Drop for Ext {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The recall block's item lines that hold `text`.
fn items_with(block: &Value, text: &str) -> Vec<String> {
    let content = block["content"].as_str().unwrap_or("");
    content.lines().filter(|l| l.contains(text)).map(str::to_string).collect()
}

/// The JSON a forum tool answers, after its authority banner.
fn forum_json(out: &Value) -> Value {
    assert_eq!(out["is_error"], false, "forum tool: {out}");
    let text = out["content"].as_str().unwrap();
    let (_, json) = text.split_once('\n').unwrap_or_else(|| panic!("no banner: {text}"));
    serde_json::from_str(json).unwrap()
}

// ── Section 7, test 12: the whole story ─────────────────────────────────────

#[test]
fn s7_t12_end_to_end_through_the_binary() {
    let started = Instant::now();
    let w = World::new();
    let (proj, other) = (w.proj(), w.other());

    // 1. The fixture archives and a notes file in place, then the process.
    let placed = w.put_fixture_archives() as i64;
    w.put_notes(&[("mem-e2e-deploys", "Deploys to staging happen on Thursdays after the standup meeting.")]);
    let mut ext = Ext::start(&w, "1");
    let s = w.wait_status("step 1: ready, every archive and the notes file indexed", |s| {
        s["extension"]["mode"] == "ready"
            && archives_known(s) == placed
            && s["notes"]["list"].as_array().is_some_and(|l| l.iter().any(|n| n["live"] == 1))
    });
    assert_eq!(s["archives"]["tombstone"], 1, "step 1: core's tombstone fixture is a tombstone: {s}");
    assert_eq!(s["archives"]["skipped"], 0, "step 1: no fixture archive skipped: {s}");
    assert_eq!(s["notes"]["not_ok"], 0, "step 1: the notes file read: {s}");
    assert!(w.rows_of_archive("b436b6196e6ca0758c70d5faed628f29") >= 1, "step 1: the retry archive has rows");
    assert_eq!(w.rows_holding("Thursdays after the standup"), 1, "step 1: the note is one row");

    // 2. A matching prompt gets the archive's answer, labelled as an earlier chat.
    let block = ext.prompt("s-e2e-1", &proj, "Where is the retry limit set?");
    assert_eq!(block["action"], "inject", "step 2: before_message injects: {block}");
    let lines = items_with(&block, "MAX_RETRIES is 5");
    assert!(
        lines.iter().any(|l| l.contains("[earlier chat in this project, ")),
        "step 2: \"MAX_RETRIES is 5\" with its earlier chat label: {block}"
    );

    // 3. The turn completes; a second chat recalls it.
    let reply = "The retry limit lives in src/net/retry.rs; we raised it to seven after the outage on Tuesday.";
    let done = ext.hook("on_message_complete", Some("s-e2e-1"), Some(&proj), Some(reply), json!({"has_tool_use": false}));
    assert_eq!(done, json!({"action": "continue"}), "step 3: on_message_complete answers continue");
    w.wait_status("step 3: the turn is captured", |s| s["counts"]["by_origin"]["capture"] == 1);
    let block = ext.prompt("s-e2e-2", &proj, "what was the retry limit raised to after the outage on Tuesday");
    assert_eq!(block["action"], "inject", "step 3: the second chat gets a recall: {block}");
    assert!(
        items_with(&block, "raised it to seven after the outage").iter().any(|l| l.contains("[earlier chat in this project, ")),
        "step 3: the second chat recalls the first chat's turn: {block}"
    );

    // 4. A new archive arrives with its event and is indexed.
    let new_path = w.put_e2e_archive("new-archive.json", NEW_ID);
    let before = w.status();
    let archived = ext.hook(
        "on_context_archived",
        Some("s-e2e-3"),
        Some(&proj),
        None,
        json!({"archive_id": NEW_ID, "project_key": w.key, "path": new_path.to_string_lossy()}),
    );
    assert_eq!(archived, json!({"action": "continue"}), "step 4: on_context_archived answers continue");
    w.wait_status("step 4: the archived file is indexed", |s| {
        s["archives"]["indexed"].as_i64() == before["archives"]["indexed"].as_i64().map(|n| n + 1)
    });
    assert!(w.rows_of_archive(NEW_ID) >= 1, "step 4: the new archive has rows");
    assert!(ext.search("how do we roll back the canary deploy", &proj).contains("quokkaledger"), "step 4: search finds it");

    // 5. The file becomes a tombstone and core says it was forgotten: it is gone.
    w.tombstone(&new_path);
    let forgotten = ext.hook(
        "on_memory_forgotten",
        Some("s-e2e-3"),
        Some(&proj),
        None,
        json!({"id": NEW_ID, "kind": "archive", "project_key": w.key}),
    );
    assert_eq!(forgotten, json!({"action": "continue"}), "step 5: on_memory_forgotten answers continue");
    w.wait_status("step 5: the forgotten archive is a tombstone", |s| s["archives"]["tombstone"] == 2);
    assert_eq!(w.rows_of_archive(NEW_ID), 0, "step 5: the forgotten archive has no row");
    assert_eq!(w.rows_holding("quokkaledger"), 0, "step 5: no row holds the forgotten text");
    assert!(!ext.search("how do we roll back the canary deploy", &proj).contains("quokkaledger"), "step 5: search no longer finds it");

    // 6. Another folder does not see this project's rows.
    assert!(ext.search("where is the retry limit set", &proj).contains("MAX_RETRIES"), "step 6: the project's own folder sees them");
    let elsewhere = ext.tool("search", json!({"query": "where is the retry limit set", "limit": 10}), Some("s-elsewhere"), &other, Some(author()));
    let text = elsewhere["content"].as_str().unwrap_or("");
    assert!(
        !text.contains("MAX_RETRIES") && !text.contains("seven after the outage"),
        "step 6: a search from another folder saw the project's rows: {elsewhere}"
    );

    // 7. A call without a session id (a worker) cannot remember; one without a caller neither.
    let fact = "The canary deploy needs two approvals before it starts on Fridays.";
    let worker = ext.tool("remember", json!({"content": fact}), None, &proj, Some(author()));
    assert_eq!(
        (&worker["is_error"], worker["content"].as_str()),
        (&json!(true), Some(WORKER_REFUSAL)),
        "step 7: remember without session_id is refused: {worker}"
    );
    let no_caller = ext.tool("remember", json!({"content": fact}), Some("s-e2e-1"), &proj, None);
    assert_eq!(
        (&no_caller["is_error"], no_caller["content"].as_str()),
        (&json!(true), Some(NO_CALLER)),
        "step 7: remember without caller is refused: {no_caller}"
    );
    assert_eq!(w.rows_holding("two approvals"), 0, "step 7: nothing was remembered");

    // 8. The forum round-trips.
    let body = "The retry limit is seven since the Tuesday outage.";
    let post = json!({"request_key": "e2e-retry-1", "title": "Retry limit", "body": body});
    let receipt = forum_json(&ext.tool("forum_post", post.clone(), Some("s-e2e-1"), &proj, Some(author())));
    assert_eq!(receipt["result"]["status"], "created", "step 8: forum_post creates: {receipt}");
    let post_id = receipt["result"]["id"].as_str().unwrap().to_string();
    let thread = forum_json(&ext.tool("forum_read", json!({"thread_id": post_id}), Some("s-e2e-2"), &proj, Some(author())));
    assert_eq!(thread["result"]["entries"][0]["id"], post_id.as_str(), "step 8: forum_read finds the post: {thread}");
    assert_eq!(thread["result"]["entries"][0]["body"], body, "step 8: forum_read returns its body: {thread}");

    // 9. Killed, a file written while it is down, started again.
    let ids_before = w.memory_ids();
    let posts_before = w.count("SELECT COUNT(*) FROM forum_posts WHERE project_key = ?1", &w.key);
    assert_eq!(posts_before, 1, "step 8: the post is stored under the project's key");
    let known_before = archives_known(&w.status());
    ext.kill();
    w.put_e2e_archive("while-down.json", DOWN_ID);
    let mut ext = Ext::start(&w, "2");
    w.wait_status("step 9: the second process is ready and its scan indexed the file written while down", |s| {
        s["extension"]["mode"] == "ready" && archives_known(s) == known_before + 1
    });
    let ids_after = w.memory_ids();
    let lost: Vec<&String> = ids_before.iter().filter(|id| !ids_after.contains(id)).collect();
    assert!(lost.is_empty(), "step 9: rows lost across the kill: {lost:?}");
    let added: Vec<&String> = ids_after.iter().filter(|id| !ids_before.contains(id)).collect();
    assert_eq!(added.len() as i64, w.rows_of_archive(DOWN_ID), "step 9: the only new rows are the new file's: {added:?}");
    assert!(!added.is_empty(), "step 9: the file written while down was indexed");
    let twice: i64 = w
        .look()
        .query_row(
            "SELECT COUNT(*) FROM (SELECT 1 FROM memories GROUP BY content, project_key HAVING COUNT(*) > 1)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(twice, 0, "step 9: a text is stored twice");
    assert_eq!(
        w.count("SELECT COUNT(*) FROM forum_posts WHERE project_key = ?1", &w.key),
        posts_before,
        "step 9: the forum has the same posts"
    );
    let block = ext.prompt("s-e2e-4", &proj, "Where is the retry limit set?");
    assert!(
        !items_with(&block, "MAX_RETRIES is 5").is_empty(),
        "step 9: the archive is still recalled after the restart: {block}"
    );
    assert!(
        ext.search("what was the retry limit raised to after the outage", &proj).contains("seven after the outage"),
        "step 9: the captured turn is still there"
    );
    let again = forum_json(&ext.tool("forum_post", post, Some("s-e2e-4"), &proj, Some(author())));
    assert_eq!(again["result"]["status"], "duplicate", "step 9: the same post again is a duplicate: {again}");
    assert_eq!(again["result"]["id"], post_id.as_str(), "step 9: the post keeps its id");
    let thread = forum_json(&ext.tool("forum_read", json!({"thread_id": post_id}), Some("s-e2e-4"), &proj, Some(author())));
    assert_eq!(thread["result"]["entries"][0]["body"], body, "step 9: forum_read still returns the post: {thread}");
    assert_eq!(w.rows_holding("quokkaledger"), 0, "step 9: the forgotten archive did not come back");
    assert_eq!(w.rows_holding("two approvals"), 0, "step 9: the refused memory did not appear");
    ext.shutdown();

    // 10. The temp home holds the brain's folder and nothing else.
    let unexpected: Vec<String> = files_under(&w.home())
        .into_iter()
        .filter(|rel| {
            let Some(name) = rel.strip_prefix(".local/share/axel/") else { return true };
            let status_file = name
                .strip_prefix("extension.")
                .and_then(|n| n.strip_suffix(".status"))
                .is_some_and(|pid| !pid.is_empty() && pid.bytes().all(|b| b.is_ascii_digit()));
            !(status_file
                || ["brain.r8", "brain.r8-wal", "brain.r8-shm", "extension.log", "extension.log.1"].contains(&name))
        })
        .collect();
    assert!(unexpected.is_empty(), "step 10: files in the temp HOME the test did not expect: {unexpected:?}");
    assert!(w.brain().is_file() && w.brain().with_file_name("extension.log").is_file(), "step 10: the brain and the log");
    assert!(started.elapsed() < TEST_LIMIT, "the test took {:?}", started.elapsed());
}

// ── Section 8, test 6: a secret in a note file ──────────────────────────────

#[test]
fn s8_t06_a_note_file_with_an_api_key_is_indexed_whole_and_shown_with_the_value_blanked() {
    let started = Instant::now();
    let w = World::new();
    // Built from two halves: no key-shaped literal is in the repository.
    let value = ["q7Rk2mWp", "9xLt4Zbv"].concat();
    let text = format!(
        "The staging deploy uses api_key = {value} and the key is rotated on the first of every month by the ops team.\n"
    );
    let file = w.notes().join("staging-deploy.md");
    std::fs::write(&file, &text).unwrap();
    let mut ext = Ext::start(&w, "1");
    w.wait_ready("the extension is ready");
    let stored: String = wait_for("the note file is indexed", || {
        w.look()
            .query_row("SELECT content FROM documents WHERE file_path = ?1", [file.to_string_lossy()], |r| r.get(0))
            .ok()
    });
    assert_eq!(stored, text, "the note file is indexed whole");

    let answer = ext.search("staging deploy key rotated every month by the ops team", &w.proj());
    let line = answer
        .lines()
        .find(|l| l.contains("rotated on the first"))
        .unwrap_or_else(|| panic!("the note file's passage is not in the answer: {answer}"));
    assert!(line.contains("The staging deploy uses api_key"), "the words before the value: {line}");
    assert!(line.contains("(redacted)"), "the value is blanked: {line}");
    assert!(line.contains("by the ops team"), "the words after the value: {line}");
    assert!(!answer.contains(&value) && !answer.contains("q7Rk2mWp") && !answer.contains("9xLt4Zbv"), "{answer}");
    ext.shutdown();
    assert!(started.elapsed() < TEST_LIMIT, "the test took {:?}", started.elapsed());
}

// ── Section 8, test 9: the source tripwire ──────────────────────────────────

fn rs_files(dir: &Path) -> Vec<PathBuf> {
    files_under(dir).into_iter().filter(|rel| rel.ends_with(".rs")).map(|rel| dir.join(rel)).collect()
}

fn holds(path: &Path, words: &[&str]) -> bool {
    let text = std::fs::read_to_string(path).unwrap();
    words.iter().any(|w| text.contains(w))
}

#[test]
fn s8_t09_no_network_code_outside_the_download_and_no_current_dir_where_a_folder_decides() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let net = ["ureq", "TcpStream"];

    let axel_src = crates.join("axel/src");
    let files = rs_files(&axel_src);
    assert!(files.len() > 30, "the axel sources were not found: {files:?}");
    let networked: Vec<&PathBuf> = files.iter().filter(|f| holds(f, &net)).collect();
    assert!(networked.is_empty(), "network code under axel/src: {networked:?}");

    let velocirag = rs_files(&crates.join("velocirag/src"));
    let networked: Vec<String> = velocirag
        .iter()
        .filter(|f| holds(f, &net))
        .map(|f| f.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(networked, ["download.rs"], "network code under velocirag/src");

    let mut deciders = rs_files(&axel_src.join("ext"));
    for name in ["project.rs", "recall.rs", "capture.rs", "forum_store.rs", "indexer.rs", "index_archives.rs", "index_notes.rs"] {
        let path = axel_src.join(name);
        assert!(path.is_file(), "{name} is not there");
        deciders.push(path);
    }
    assert!(deciders.len() > 7, "the ext sources were not found");
    let with_cwd: Vec<&PathBuf> = deciders.iter().filter(|f| holds(f, &["current_dir"])).collect();
    assert!(with_cwd.is_empty(), "current_dir where a folder decides: {with_cwd:?}");
}

// ── Section 8, test 10: an event's path outside the project ─────────────────

#[test]
fn s8_t10_an_archive_event_whose_path_is_a_decoy_outside_the_project_is_not_opened() {
    let started = Instant::now();
    let w = World::new();
    std::fs::create_dir_all(w.project_archives()).unwrap();
    // The decoy: a live archive of this project's scope and of the event's id, in no
    // project directory.
    let mut decoy: Value = serde_json::from_slice(&std::fs::read(fixtures().join("e2e/decoy.json")).unwrap()).unwrap();
    decoy["scope"] = json!(w.scope);
    let decoy_path = w.root().join("elsewhere").join("s000001").join(format!("{DECOY_ID}.json"));
    std::fs::create_dir_all(decoy_path.parent().unwrap()).unwrap();
    std::fs::write(&decoy_path, decoy.to_string()).unwrap();

    let mut ext = Ext::start(&w, "1");
    w.wait_ready("the extension is ready");
    let real = w.put_e2e_archive("new-archive.json", NEW_ID);
    let event = |id: &str, path: &Path| json!({"archive_id": id, "project_key": w.key, "path": path.to_string_lossy()});
    let answered = ext.hook("on_context_archived", Some("s-decoy"), Some(&w.proj()), None, event(DECOY_ID, &decoy_path));
    assert_eq!(answered, json!({"action": "continue"}));
    // The jobs run in order on one thread: once the next one is done, the decoy's is too.
    ext.hook("on_context_archived", Some("s-decoy"), Some(&w.proj()), None, event(NEW_ID, &real));
    wait_for("the real archive after the decoy is indexed", || (w.rows_of_archive(NEW_ID) > 0).then_some(()));

    assert_eq!(w.rows_holding("DECOYMARKER8812"), 0, "the decoy's marker is in a row");
    assert_eq!(w.count("SELECT COUNT(*) FROM archive_index WHERE archive_id = ?1", DECOY_ID), 0, "the decoy has an index row");
    ext.shutdown();
    for file in ["brain.r8", "brain.r8-wal"] {
        let path = w.brain().with_file_name(file);
        let bytes = std::fs::read(&path).unwrap_or_default();
        assert!(!bytes.windows(15).any(|b| b == b"DECOYMARKER8812"), "the decoy's marker is in {file}");
    }
    assert!(started.elapsed() < TEST_LIMIT, "the test took {:?}", started.elapsed());
}

// ── Section 1, test 17, the binary half: every subcommand on a version 1 brain ─

/// The subcommands that do not pass the gate (R-2).
const EXEMPT: [&str; 7] = ["extension", "migrate", "plugin", "model", "init", "synaps-export", "status"];

fn axel(dir: &Path, args: &[&str], brain_env: Option<&Path>) -> Output {
    let mut command = Command::new(BIN);
    command.args(args).current_dir(dir);
    command.env("HOME", dir.join("home"));
    command.env_remove("AXEL_BRAIN");
    if let Some(brain) = brain_env {
        command.env("AXEL_BRAIN", brain);
    }
    command.output().unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Every subcommand `axel --help` lists, which is every one clap knows, but `help`.
fn subcommands(dir: &Path) -> Vec<String> {
    let help = text(&axel(dir, &["--help"], None).stdout);
    let mut names = Vec::new();
    let mut inside = false;
    for line in help.lines() {
        if line.trim_end() == "Commands:" {
            inside = true;
            continue;
        }
        if inside {
            if line.trim().is_empty() {
                break;
            }
            // A command's line is indented by two; a wrapped description by more.
            if line.starts_with("  ") && !line.starts_with("   ") {
                names.push(line.split_whitespace().next().unwrap().to_string());
            }
        }
    }
    names.retain(|n| n != "help");
    assert!(names.len() > EXEMPT.len(), "no subcommands in `axel --help`:\n{help}");
    names
}

/// The smallest arguments clap accepts for `sub`: every required argument of its usage
/// line, each the first of its possible values or `x`.
fn smallest_args(dir: &Path, sub: &str) -> Vec<String> {
    let help = text(&axel(dir, &[sub, "--help"], None).stdout);
    let usage = help.lines().find_map(|l| l.strip_prefix("Usage:")).unwrap_or_else(|| panic!("{sub}: no usage:\n{help}"));
    let mut args = vec![sub.to_string()];
    let mut depth = 0usize;
    for token in usage.split_whitespace().skip(2) {
        let optional = depth > 0 || token.starts_with('[');
        depth += token.matches('[').count();
        depth = depth.saturating_sub(token.matches(']').count());
        if optional {
            continue;
        }
        if token.starts_with('<') {
            let name = token.trim_end_matches("...");
            args.push(first_value(&help, name));
        } else {
            args.push(token.to_string());
        }
    }
    args
}

/// The first possible value the help lists for `<NAME>`, or `x`.
fn first_value(help: &str, name: &str) -> String {
    let mut lines = help.lines().skip_while(|l| !l.trim_start().starts_with(name));
    let mut described = lines.next().unwrap_or("").to_string();
    for more in lines.take_while(|l| l.starts_with("   ") && !l.trim_start().starts_with(['<', '-'])) {
        described.push(' ');
        described.push_str(more.trim());
    }
    described
        .split_once("[possible values: ")
        .and_then(|(_, values)| values.split([',', ']']).next())
        .map(|v| v.trim().to_string())
        .unwrap_or_else(|| "x".to_string())
}

#[test]
fn s1_t17_every_gated_subcommand_refuses_a_version_1_brain_and_leaves_it_unchanged() {
    let tmp = TempDir::new().unwrap();
    let dir = tmp.path();
    std::fs::create_dir_all(dir.join("home")).unwrap();
    std::fs::create_dir_all(dir.join("old")).unwrap();
    let old = dir.join("old").join("axel.r8");
    drop(axel::r8::Brain::create(&old, Some("old")).unwrap());
    let memory = Memory::new(MemoryCategory::Events, "test", "kept", "the one memory of the old brain");
    MemoryStorage::open_existing(&old).unwrap().store_memory(&memory).unwrap();
    let hash = sha256(&old);
    let folder = files_under(&dir.join("old"));
    let missing = dir.join("none").join("brain.r8");

    let all = subcommands(dir);
    for name in EXEMPT {
        assert!(all.iter().any(|n| n == name), "the exempt subcommand {name} is not in clap's list: {all:?}");
    }
    let gated: Vec<&String> = all.iter().filter(|n| !EXEMPT.contains(&n.as_str())).collect();
    for known in ["index", "search", "remember", "forget", "show", "pause", "resume", "restore", "why"] {
        assert!(gated.iter().any(|n| *n == known), "{known} is not among the gated subcommands: {gated:?}");
    }

    // `status` is not behind the gate: exit 1, and its first line names the kind.
    let out = axel(dir, &["status"], Some(&old));
    assert_eq!(out.status.code(), Some(1), "status on a version 1 brain");
    let first = text(&out.stdout).lines().next().unwrap_or("").to_string();
    assert!(first.starts_with("this is a version 1 brain"), "status's first line: {first}");
    assert_eq!(sha256(&old), hash, "status changed the version 1 brain");
    assert_eq!(files_under(&dir.join("old")), folder, "status left a file beside the version 1 brain");

    // `synaps-export` reads a brain it is given; the gate does not refuse it.
    let out = axel(dir, &["--brain", old.to_str().unwrap(), "synaps-export", "--out", dir.join("export").to_str().unwrap()], None);
    assert!(!text(&out.stderr).contains("this is a version 1 brain"), "synaps-export was refused: {}", text(&out.stderr));
    assert_eq!(files_under(&dir.join("old")), folder, "synaps-export left a file beside its source");
    assert_eq!(sha256(&old), hash, "synaps-export changed its source");

    for sub in &gated {
        let args = smallest_args(dir, sub);
        let args: Vec<&str> = args.iter().map(String::as_str).collect();

        let out = axel(dir, &args, Some(&old));
        let said = text(&out.stderr);
        assert!(!out.status.success(), "{args:?} on a version 1 brain exited 0");
        assert!(said.contains("this is a version 1 brain"), "{args:?} did not give the gate's message: {said}");
        assert_eq!(sha256(&old), hash, "{args:?} changed the version 1 brain");
        // SQLite's own side files may appear (the gate reads a file in WAL mode); nothing else
        // does, and the log stays empty.
        let beside: Vec<String> =
            files_under(&dir.join("old")).into_iter().filter(|f| f != "axel.r8-wal" && f != "axel.r8-shm").collect();
        assert_eq!(beside, folder, "{args:?} left a file beside the version 1 brain");
        let wal = std::fs::metadata(dir.join("old").join("axel.r8-wal")).map(|m| m.len()).unwrap_or(0);
        assert_eq!(wal, 0, "{args:?} wrote into the version 1 brain's log");

        let out = axel(dir, &args, Some(&missing));
        let said = text(&out.stderr);
        assert!(!out.status.success(), "{args:?} on a missing brain exited 0");
        assert!(
            said.contains("axel-v2 init") && said.contains("axel-v2 migrate init"),
            "{args:?} on a missing brain does not name both ways to make one: {said}"
        );
        assert!(!missing.parent().unwrap().exists(), "{args:?} created the missing brain's folder");
    }

    // `mcp` is gone.
    let out = axel(dir, &["mcp"], Some(&old));
    assert_eq!(out.status.code(), Some(2), "axel mcp: {}", text(&out.stderr));
    assert!(text(&out.stderr).contains("unrecognized subcommand"), "axel mcp: {}", text(&out.stderr));
    assert!(!all.iter().any(|n| n == "mcp"));
    assert_eq!(sha256(&old), hash);
    assert!(files_under(&dir.join("home")).is_empty(), "a command wrote into the temp home");
}

/// Real-binaries round, item 2: an archive core forgot while no extension ran is not
/// served by the next extension, not even by its first call, before its first scan.
// RULING: v2 real-binaries round, item 2
#[test]
fn rb2_an_archive_forgotten_while_no_extension_ran_is_not_served_at_the_next_start() {
    let w = World::new();
    let proj = w.proj();
    let path = w.put_e2e_archive("new-archive.json", NEW_ID);
    let ext = Ext::start(&w, "1");
    w.wait_ready("the first process ready");
    wait_for("the archive indexed", || (w.rows_of_archive(NEW_ID) >= 1).then_some(()));
    ext.shutdown();

    // Core forgets it while nothing runs; the next process is asked at once.
    w.tombstone(&path);
    let mut ext = Ext::start(&w, "2");
    let found = ext.search("how do we roll back the canary deploy", &proj);
    assert!(!found.contains("quokkaledger"), "the forgotten archive was served: {found}");
    let block = ext.prompt("s-rb2", &proj, "how do we roll back the canary deploy");
    assert!(!block.to_string().contains("quokkaledger"), "recall served the forgotten archive: {block}");
    assert_eq!(w.rows_of_archive(NEW_ID), 0);
    ext.shutdown();
}
