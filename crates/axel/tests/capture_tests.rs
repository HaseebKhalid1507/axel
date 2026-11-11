//! Commit 16, capture: section 4 tests 1 to 5, 6 (the capture half) and 7 to 11.
//!
//! `CaptureState` is tested as it is (no brain); the writes go to a temp brain. Nothing
//! here reads the model or the network.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use axel::brain_v2::{self, OpenOpts, Role, V2Brain};
use axel::capture::{self, CaptureState, CompletedTurn, Written};
use axel::ext::config::ExtConfig;
use axel::ext::front;
use axel::ext::server::serve;
use axel::ext::types::{Boot, FrontState, HookEvent, IndexerCtx, Job, Mode, ServeEnd, Status, TestHooks};
use axel::vindex::VecDelta;
use axel::{archive, indexer, indexer_start, project, recall, screen, store};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use tempfile::TempDir;
use velocirag::embedder::Embedder;

// None of core's 17 words is in these texts.
const PROMPT: &str = "how does the indexer decide which archive files it reads first";
const REPLY: &str = "It reads the newest window first and files each unit under the project directory.";

// ── Helpers ─────────────────────────────────────────────────────────────────

/// `<tmp>/brain/brain.r8`, `<tmp>/base`, `<tmp>/model` (empty), and project folders.
struct Lab {
    tmp: TempDir,
}

impl Lab {
    fn new() -> Lab {
        let tmp = TempDir::new().unwrap();
        for dir in ["brain", "base", "model"] {
            std::fs::create_dir_all(tmp.path().join(dir)).unwrap();
        }
        Lab { tmp }
    }

    fn brain_path(&self) -> PathBuf {
        self.tmp.path().join("brain").join("brain.r8")
    }

    fn open(&self) -> V2Brain {
        let opts = OpenOpts { agent: Some("test"), model: "hash-384", dim: 384, upgrade_v1: false };
        brain_v2::open_v2(&self.brain_path(), &opts).unwrap()
    }

    /// A folder with a `.git` entry: a project of its own. Returns it as a `cwd`.
    fn project(&self, name: &str) -> String {
        let dir = self.tmp.path().join(name);
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        dir.to_string_lossy().into_owned()
    }

    fn cfg_with(&self, embedder: &str, capture: &str) -> ExtConfig {
        let config = json!({
            "brain": self.brain_path().to_string_lossy(),
            "synaps_base": self.tmp.path().join("base").to_string_lossy(),
            "model_dir": self.tmp.path().join("model").to_string_lossy(),
            "embedder": embedder,
            "capture": capture,
        });
        ExtConfig::from_initialize(&json!({ "config": config }), self.tmp.path())
    }

    fn cfg(&self) -> ExtConfig {
        let cfg = self.cfg_with("off", "turns");
        assert!(cfg.capture);
        cfg
    }
}

fn scope_dir(cwd: &str) -> String {
    project::resolve(Some(cwd)).unwrap().scope_dir
}

fn tx<T>(conn: &Connection, work: impl FnOnce() -> T) -> T {
    brain_v2::begin_write(conn).unwrap();
    let out = work();
    conn.execute_batch("COMMIT").unwrap();
    out
}

/// One whole turn through the pairing: a prompt at `at`, its one reply a second later.
fn pair(session_id: &str, cwd: &str, prompt: &str, reply: &str, at: i64) -> CompletedTurn {
    let mut cap = CaptureState::default();
    cap.on_user(Some(session_id), Some(cwd), prompt, at);
    cap.on_assistant(Some(session_id), reply, false, at + 1000).expect("a turn")
}

fn write(brain: &V2Brain, cfg: &ExtConfig, turn: &CompletedTurn) -> Written {
    let written = capture::write_turn(&brain.conn, brain.signer.as_ref(), cfg, turn).unwrap();
    assert!(brain.conn.is_autocommit(), "write_turn left a transaction open");
    written
}

fn count(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |r| r.get(0)).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

fn captures(conn: &Connection) -> i64 {
    count(conn, "SELECT COUNT(*) FROM memory_ext WHERE origin_kind = 'capture'")
}

fn captures_of(conn: &Connection, session_id: &str) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM memory_ext WHERE origin_kind = 'capture' AND session_id = ?1",
        [session_id],
        |r| r.get(0),
    )
    .unwrap()
}

/// No text of any kind reached the tables a turn row is made of.
fn nothing_stored(conn: &Connection) {
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memories"), 0);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memory_ext"), 0);
}

fn content_of(conn: &Connection, id: &str) -> String {
    conn.query_row("SELECT content FROM memories WHERE id = ?1", [id], |r| r.get(0)).unwrap()
}

fn hidden_by(conn: &Connection, id: &str) -> Option<String> {
    conn.query_row("SELECT hidden_by FROM memory_ext WHERE mem_id = ?1", [id], |r| r.get(0)).unwrap()
}

/// What a search from another chat may return: rows that are not hidden.
fn visible(conn: &Connection) -> Vec<String> {
    let mut stmt = conn
        .prepare(
            "SELECT e.mem_id FROM memory_ext e JOIN memories m ON m.id = e.mem_id
              WHERE e.hidden_ms IS NULL AND m.superseded_by IS NULL ORDER BY e.mem_id",
        )
        .unwrap();
    let rows = stmt.query_map([], |r| r.get(0)).unwrap();
    rows.map(|r| r.unwrap()).collect()
}

fn id_of(turn: &CompletedTurn) -> String {
    capture::turn_id(&turn.session_id, turn.started_ms, &turn.prompt)
}

/// A token `redact` blanks and core's word list does not know, joined from two halves.
fn gitlab_token() -> String {
    format!("glpat-{}{}", "Zq7Lm2Xw9R", "bN4vT8yK1c")
}

fn boot(embedder: Option<Embedder>) -> (Boot, Arc<Status>) {
    let status = Arc::new(Status::default());
    let boot =
        Boot { embedder, clock: Arc::new(|| 1), test: Arc::new(TestHooks::default()), status: status.clone() };
    (boot, status)
}

fn hook(kind: &str, session_id: Option<&str>, cwd: Option<&str>, message: Option<&str>, data: Value, at: i64) -> HookEvent {
    HookEvent {
        kind: kind.to_string(),
        session_id: session_id.map(str::to_string),
        cwd: cwd.map(str::to_string),
        message: message.map(str::to_string),
        data,
        received_ms: at,
    }
}

fn user(st: &mut FrontState, session_id: &str, cwd: &str, text: &str, at: i64) {
    capture::on_user_event(st, &hook("before_message", Some(session_id), Some(cwd), Some(text), json!({}), at));
}

fn assistant(st: &mut FrontState, session_id: &str, cwd: &str, text: &str, has_tool_use: bool, at: i64) {
    let data = json!({ "has_tool_use": has_tool_use });
    capture::on_assistant_event(st, &hook("on_message_complete", Some(session_id), Some(cwd), Some(text), data, at));
}

/// The next capture job on the queue; other jobs are passed over.
fn next_capture(jobs: &Receiver<Job>) -> Option<CompletedTurn> {
    loop {
        match jobs.try_recv() {
            Ok(Job::Capture(turn)) => return Some(turn),
            Ok(_) => continue,
            Err(_) => return None,
        }
    }
}

/// An indexer with its brain made, and a front state connected to it.
fn threads(lab: &Lab, capture: &str) -> (IndexerCtx, FrontState, Receiver<Job>, Arc<Status>, Kept) {
    let (b, _) = boot(Some(Embedder::hashing()));
    let (ctx, ctx_jobs, vec_rx) = indexer_start::open_ctx(lab.cfg_with("hash", capture), b);
    assert_eq!(ctx.mode.load(Ordering::SeqCst), Mode::Ready as u8);
    let (b, status) = boot(None);
    let (st, jobs, vec_tx) = front::open_state(lab.cfg_with("off", capture), b);
    assert!(st.conn.is_some());
    (ctx, st, jobs, status, (ctx_jobs, vec_rx, vec_tx))
}

/// The other ends of the channels, held so that none of them reads as closed.
type Kept = (Receiver<Job>, Receiver<VecDelta>, Sender<VecDelta>);

fn ctx_conn(ctx: &IndexerCtx) -> &Connection {
    ctx.conn.as_ref().unwrap()
}

// ── 1. Pairing ──────────────────────────────────────────────────────────────

#[test]
fn t01_pairing() {
    let mut cap = CaptureState::default();

    // user, assistant with a tool use, assistant without: one turn with both texts.
    cap.on_user(Some("s"), Some("/work"), "first prompt", 1_000);
    assert_eq!(cap.on_assistant(Some("s"), "part one", true, 2_000), None);
    assert!(cap.is_pending("s"));
    let turn = cap.on_assistant(Some("s"), "part two", false, 3_000).unwrap();
    assert_eq!(turn.session_id, "s");
    assert_eq!(turn.cwd.as_deref(), Some("/work"));
    assert_eq!(turn.prompt, "first prompt");
    assert!(turn.reply.contains("part one") && turn.reply.contains("part two"), "{:?}", turn.reply);
    assert!(turn.reply.find("part one") < turn.reply.find("part two"));
    assert_eq!((turn.started_ms, turn.at_ms), (1_000, 3_000));
    assert!(!turn.suppressed && !turn.withheld);
    assert_eq!(cap.pending_len(), 0, "the buffer is dropped with the turn");
    // The turn is over: one more assistant message gives nothing.
    assert_eq!(cap.on_assistant(Some("s"), "late", false, 3_500), None);

    // A second `on_user` before the last reply replaces the first.
    cap.on_user(Some("s"), None, "prompt one", 4_000);
    assert_eq!(cap.on_assistant(Some("s"), "half an answer", true, 4_100), None);
    cap.on_user(Some("s"), None, "prompt two", 5_000);
    let turn = cap.on_assistant(Some("s"), "answer two", false, 6_000).unwrap();
    assert_eq!(turn.prompt, "prompt two");
    assert_eq!(turn.reply, "answer two");
    assert_eq!(turn.started_ms, 5_000);

    // An assistant message with no pending prompt gives nothing.
    assert_eq!(cap.on_assistant(Some("nobody"), "text", false, 6_500), None);

    // `session_id = null` gives nothing, on either hook.
    cap.on_user(None, Some("/work"), "a prompt of nobody", 7_000);
    assert_eq!(cap.pending_len(), 0);
    assert_eq!(cap.on_assistant(None, "text", false, 7_100), None);
    cap.on_user(Some("s"), None, "a prompt of s", 7_200);
    assert_eq!(cap.on_assistant(None, "text", false, 7_300), None, "no session takes another's turn");
    cap.drop_session("s");

    // 65 sessions pending: the oldest is dropped.
    for i in 0..65 {
        cap.on_user(Some(&format!("chat{i}")), None, "a prompt", 10_000 + i);
    }
    assert_eq!(cap.pending_len(), 64);
    assert!(!cap.is_pending("chat0"));
    assert!(cap.is_pending("chat1") && cap.is_pending("chat64"));
    assert_eq!(cap.on_assistant(Some("chat0"), "text", false, 11_000), None);
    // A chat that is pending already replaces its own buffer and evicts nobody.
    cap.on_user(Some("chat1"), None, "another prompt", 11_500);
    assert_eq!(cap.pending_len(), 64);
    assert!(cap.is_pending("chat2"));

    // R-16 item 1: no buffer is too old. A reply fifteen minutes after the prompt (a worker
    // ran inside the turn) is paired with it.
    let mut cap = CaptureState::default();
    cap.on_user(Some("s"), None, "a prompt with a long worker run", 0);
    assert_eq!(cap.on_assistant(Some("s"), "the worker is started", true, 1_000), None);
    let turn = cap.on_assistant(Some("s"), "fifteen minutes later", false, 15 * 60_000).unwrap();
    assert_eq!(turn.prompt, "a prompt with a long worker run");
    assert_eq!(turn.reply, "the worker is started\nfifteen minutes later");
    assert_eq!(cap.take_dropped(), 0);

    // `on_session_end` drops what is pending.
    cap.on_user(Some("s"), None, "a prompt", 0);
    cap.on_session_end("s");
    assert_eq!(cap.on_assistant(Some("s"), "text", false, 1), None);
}

#[test]
fn t01_the_reply_of_a_turn_is_cut_at_4096_bytes() {
    let mut cap = CaptureState::default();
    cap.on_user(Some("s"), None, "a prompt", 0);
    let long = "é".repeat(1_500);
    assert_eq!(long.len(), 3_000);
    assert_eq!(cap.on_assistant(Some("s"), &long, true, 1), None);
    let turn = cap.on_assistant(Some("s"), &long, false, 2).unwrap();
    assert!(turn.reply.len() <= 4_096, "{}", turn.reply.len());
    assert!(turn.reply.len() > 4_090);
    assert!(turn.reply.starts_with(&long));
}

// ── 2. Host messages ────────────────────────────────────────────────────────

#[test]
fn t02_host_messages_give_no_turn_and_only_an_event_clears_the_buffer() {
    let event = "<event id=\"7\" kind=\"worker_done\">the worker finished</event>";
    let event_after_space = "\n  <event id=\"8\">after white space</event>";
    let continuation = "[Context continuation: the earlier part of this chat was archived]";
    let advisory = "[Host context advisory: the window is nearly full]";
    let advisory_after_space = "  [Host context advisory: after white space]";

    // Alone: nothing is pending, and the reply that follows gives no turn.
    for host in [event, event_after_space, continuation, advisory, advisory_after_space] {
        let mut cap = CaptureState::default();
        cap.on_user(Some("s"), None, host, 1_000);
        assert_eq!(cap.pending_len(), 0, "{host}");
        assert_eq!(cap.on_assistant(Some("s"), "a reply to the host", false, 2_000), None, "{host}");
        assert_eq!(cap.take_dropped(), 0, "{host}");
    }

    // An event in the middle of a human turn: that turn is dropped, another chat's is not.
    for host in [event, event_after_space] {
        let mut cap = CaptureState::default();
        cap.on_user(Some("s"), None, "a human prompt", 3_000);
        cap.on_user(Some("t"), None, "another chat", 3_000);
        cap.on_user(Some("s"), None, host, 3_500);
        assert!(!cap.is_pending("s"), "{host}");
        assert_eq!(cap.on_assistant(Some("s"), "a reply", false, 4_000), None, "{host}");
        assert!(cap.on_assistant(Some("t"), "a reply", false, 4_000).is_some());
        assert_eq!(cap.take_dropped(), 1, "{host}");
    }

    // R-16 item 2: an advisory or a continuation in the middle of a human turn leaves it
    // alone. The turn keeps its prompt, its start, and what was said before and after.
    for host in [continuation, advisory, advisory_after_space] {
        let mut cap = CaptureState::default();
        cap.on_user(Some("s"), Some("/work"), "a human prompt", 3_000);
        assert_eq!(cap.on_assistant(Some("s"), "part one", true, 3_200), None);
        cap.on_user(Some("s"), Some("/elsewhere"), host, 3_500);
        assert!(cap.is_pending("s"), "{host}");
        let turn = cap.on_assistant(Some("s"), "part two", false, 4_000).unwrap();
        assert_eq!(turn.prompt, "a human prompt", "{host}");
        assert_eq!(turn.turn_key, archive::turn_key("a human prompt"));
        assert_eq!(turn.reply, "part one\npart two", "{host}");
        assert_eq!(turn.cwd.as_deref(), Some("/work"));
        assert_eq!((turn.started_ms, turn.at_ms), (3_000, 4_000));
        assert_eq!(cap.take_dropped(), 0, "{host}");
    }

    // A prompt that only mentions such a mark is a prompt.
    let mut cap = CaptureState::default();
    cap.on_user(Some("s"), None, "what does <event mean in the host log", 1);
    assert!(cap.is_pending("s"));
}

/// R-16 item 2, accepted as it is: the host calls the prompt hook again when the person
/// steers in the middle of a turn. The second prompt replaces the first; the turn is
/// stored under the second, with only what the model said after it. The first prompt and
/// the reply so far are not captured live (they come with the archive), and the turn is
/// not counted as dropped.
#[test]
fn r16_2_a_second_prompt_in_one_turn_replaces_the_first() {
    let lab = Lab::new();
    let (mut ctx, mut st, jobs, status, _kept) = threads(&lab, "on");
    let cwd = lab.project("alpha");
    let first = "look at the indexer and tell me why the newest window is read first";
    let second = "no, wait: look at the reader of the archive files instead, the slot files";

    user(&mut st, "S", &cwd, first, 1_000);
    assistant(&mut st, "S", &cwd, "I am opening the indexer to see how it orders the windows.", true, 2_000);
    // An advisory between the two changes nothing.
    user(&mut st, "S", &cwd, "[Host context advisory: the window is nearly full]", 2_500);
    user(&mut st, "S", &cwd, second, 3_000);
    assistant(&mut st, "S", &cwd, "The reader takes the slot files in the order of their numbers.", false, 4_000);

    let turn = next_capture(&jobs).unwrap();
    assert!(next_capture(&jobs).is_none(), "one turn, one job");
    assert_eq!(turn.prompt, second);
    assert_eq!(turn.turn_key, archive::turn_key(second));
    assert_eq!(turn.reply, "The reader takes the slot files in the order of their numbers.");
    assert_eq!((turn.started_ms, turn.at_ms), (3_000, 4_000));
    assert_eq!(dropped(&status), 0);

    indexer::dispatch(&mut ctx, Job::Capture(turn.clone()));
    let conn = ctx_conn(&ctx);
    assert_eq!(captures_of(conn, "S"), 1);
    let content = content_of(conn, &id_of(&turn));
    assert!(content.starts_with(&format!("User: {second}\n")), "{content}");
    assert!(!content.contains("newest window") && !content.contains("opening the indexer"), "{content}");
}

// ── 3. The row ──────────────────────────────────────────────────────────────

#[test]
fn t03_one_turn_is_one_row_under_the_project() {
    let lab = Lab::new();
    let brain = lab.open();
    let cfg = lab.cfg();
    let cwd = lab.project("alpha");
    let turn = pair("S", &cwd, PROMPT, REPLY, 1_700_000_000_000);

    assert_eq!(write(&brain, &cfg, &turn), Written::Stored);
    let conn = &brain.conn;
    assert_eq!(captures(conn), 1);
    let id = id_of(&turn);
    assert!(id.starts_with("mem_") && id.len() == 20, "{id}");

    let (category, project_key, title, content, tags, provenance, abstract_text): (
        String,
        String,
        String,
        String,
        String,
        String,
        String,
    ) = conn
        .query_row(
            "SELECT category, project_key, title, content, tags, provenance, abstract_text FROM memories WHERE id = ?1",
            [&id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?)),
        )
        .unwrap();
    assert_eq!(category, "episodic_turn");
    assert_eq!(project_key, scope_dir(&cwd), "filed under the project's directory");
    assert_eq!(content, format!("User: {PROMPT}\nAssistant: {REPLY}"));
    assert_eq!(title, "how does the indexer decide which archive files it reads first");
    assert_eq!(provenance, "axel:capture:turn");
    assert!(!abstract_text.is_empty());
    let tags: Vec<String> = serde_json::from_str(&tags).unwrap();
    assert_eq!(tags, vec!["session:S".to_string(), format!("turn:{}", turn.turn_key)]);

    let (origin_kind, origin_ref, session_id, logical, turn_key, created_ms, time_source, hidden_ms): (
        String,
        String,
        String,
        String,
        String,
        i64,
        String,
        Option<i64>,
    ) = conn
        .query_row(
            "SELECT origin_kind, origin_ref, session_id, logical, turn_key, created_ms, time_source, hidden_ms
               FROM memory_ext WHERE mem_id = ?1",
            [&id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?)),
        )
        .unwrap();
    assert_eq!((origin_kind.as_str(), origin_ref.as_str(), session_id.as_str()), ("capture", "S", "S"));
    assert_eq!(logical, project::logical("S"));
    assert_eq!(turn_key, archive::turn_key(PROMPT));
    assert_eq!(turn_key, turn.turn_key);
    assert_eq!((created_ms, time_source.as_str(), hidden_ms), (turn.at_ms, "exact", None));

    // The same turn written twice: one row.
    assert_eq!(write(&brain, &cfg, &turn), Written::Stored);
    assert_eq!(captures(conn), 1);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memories"), 1);

    // A title is the first 12 words of the prompt.
    let wordy = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen";
    let turn = pair("S", &cwd, wordy, REPLY, 1_700_000_100_000);
    assert_eq!(write(&brain, &cfg, &turn), Written::Stored);
    let title: String =
        conn.query_row("SELECT title FROM memories WHERE id = ?1", [id_of(&turn)], |r| r.get(0)).unwrap();
    assert_eq!(title, "one two three four five six seven eight nine ten eleven twelve");
    assert_eq!(captures(conn), 2);

    // A 40-char turn: no row. 60 chars is a turn.
    let turn = pair("S", &cwd, &"a".repeat(20), &"b".repeat(20), 1_700_000_200_000);
    assert_eq!(write(&brain, &cfg, &turn), Written::Skipped);
    let turn = pair("S", &cwd, &"a".repeat(30), &"b".repeat(29), 1_700_000_300_000);
    assert_eq!(write(&brain, &cfg, &turn), Written::Skipped);
    assert_eq!(captures(conn), 2);
    let turn = pair("S", &cwd, &"a".repeat(30), &"b".repeat(30), 1_700_000_400_000);
    assert_eq!(write(&brain, &cfg, &turn), Written::Stored);
    assert_eq!(captures(conn), 3);
}

#[test]
fn t03_no_project_no_row() {
    let lab = Lab::new();
    let brain = lab.open();
    let cfg = lab.cfg();
    let gone = lab.tmp.path().join("deleted-folder").to_string_lossy().into_owned();
    for cwd in [None, Some("relative/folder"), Some(gone.as_str())] {
        let mut cap = CaptureState::default();
        cap.on_user(Some("S"), cwd, PROMPT, 1_000);
        let turn = cap.on_assistant(Some("S"), REPLY, false, 2_000).unwrap();
        assert_eq!(write(&brain, &cfg, &turn), Written::Skipped, "{cwd:?}");
    }
    nothing_stored(&brain.conn);
}

/// `write_turn` owns its transaction: after an error nothing of the turn is in the brain,
/// no transaction is open, and the next turn is written.
#[test]
fn t03_an_error_rolls_the_turn_back() {
    let lab = Lab::new();
    let brain = lab.open();
    let cfg = lab.cfg();
    let cwd = lab.project("alpha");
    let conn = &brain.conn;
    conn.execute_batch(
        "CREATE TEMP TRIGGER capture_test_refuses BEFORE INSERT ON main.memory_ext
         BEGIN SELECT RAISE(ABORT, 'capture test: no write'); END;",
    )
    .unwrap();
    let turn = pair("S", &cwd, PROMPT, REPLY, 1_700_000_000_000);
    assert!(capture::write_turn(conn, brain.signer.as_ref(), &cfg, &turn).is_err());
    assert!(conn.is_autocommit(), "the failed turn left its transaction open");
    nothing_stored(conn);

    // Seen from another connection too: nothing was committed, and the lock is free.
    let other = brain_v2::connect(&lab.brain_path(), Role::Cli).unwrap();
    assert_eq!(count(&other, "SELECT COUNT(*) FROM memories"), 0);
    tx(&other, || ());

    conn.execute_batch("DROP TRIGGER capture_test_refuses").unwrap();
    assert_eq!(write(&brain, &cfg, &turn), Written::Stored);
    assert_eq!(count(&other, "SELECT COUNT(*) FROM memories"), 1);

    // Called inside somebody else's transaction it refuses, and leaves that one alone.
    brain_v2::begin_write(conn).unwrap();
    let second = pair("S", &cwd, PROMPT, REPLY, 1_700_000_500_000);
    assert!(capture::write_turn(conn, brain.signer.as_ref(), &cfg, &second).is_err());
    assert!(!conn.is_autocommit(), "the caller's transaction was ended");
    conn.execute_batch("ROLLBACK").unwrap();
    assert_eq!(captures(conn), 1);
}

// ── 4. The screen ───────────────────────────────────────────────────────────

#[test]
fn t04_a_token_is_blanked_and_what_core_withholds_is_not_captured() {
    let lab = Lab::new();
    let brain = lab.open();
    let cfg = lab.cfg();
    let cwd = lab.project("alpha");
    let conn = &brain.conn;

    // A token shape `redact` blanks, with none of core's words: stored, the token blanked.
    let token = gitlab_token();
    let prompt = format!("please rotate the gitlab key {token} before the release on friday");
    assert!(!screen::core_withholds(&prompt), "the test's token must not hold a word of core's");
    assert_eq!(screen::redact(&prompt).blanked, 1);
    let reply = format!("I rotated {token} and updated the runner configuration for the release.");
    let turn = pair("S", &cwd, &prompt, &reply, 1_700_000_000_000);
    assert!(!turn.prompt.contains(&token) && !turn.reply.contains(&token), "the buffer held the token");
    assert_eq!(write(&brain, &cfg, &turn), Written::Stored);
    let content = content_of(conn, &id_of(&turn));
    assert!(!content.contains(&token) && !content.contains("Zq7Lm2Xw9R"), "{content}");
    assert_eq!(
        content,
        "User: please rotate the gitlab key (redacted) before the release on friday\n\
         Assistant: I rotated (redacted) and updated the runner configuration for the release."
    );
    let title: String =
        conn.query_row("SELECT title FROM memories WHERE id = ?1", [id_of(&turn)], |r| r.get(0)).unwrap();
    assert!(!title.contains(&token));
    assert_eq!(captures(conn), 1);

    // A turn handed over with the token still in it (no pairing): the row writer blanks it.
    let raw = CompletedTurn {
        session_id: "R".to_string(),
        cwd: Some(cwd.clone()),
        prompt: prompt.clone(),
        turn_key: archive::turn_key(&screen::redact(&prompt).text),
        reply: reply.clone(),
        started_ms: 1_700_000_050_000,
        at_ms: 1_700_000_051_000,
        suppressed: false,
        withheld: false,
    };
    assert_eq!(write(&brain, &cfg, &raw), Written::Stored);
    assert!(!content_of(conn, &id_of(&raw)).contains(&token));
    assert_eq!(count(conn, &format!("SELECT COUNT(*) FROM memories WHERE content LIKE '%{token}%'")), 0);
    assert_eq!(captures(conn), 2);

    // Core withholds all three of these today: none is captured.
    let key = format!("sk-{}{}", "a1B2c3D4e5F6g7H8i9J0", "k1L2m3N4o5P6q7R8s9T0");
    let withheld = [
        format!("the key for the build machine is {key} and it must be rotated this week"),
        "the task-list for the indexer needs another look before we merge the branch".to_string(),
        "my secretary Nakia asked about the meeting notes of the indexer review".to_string(),
    ];
    for (i, prompt) in withheld.iter().enumerate() {
        let turn = pair("S", &cwd, prompt, REPLY, 1_700_000_100_000 + i as i64 * 10_000);
        assert!(turn.withheld, "{prompt}");
        assert_eq!(turn.prompt, "", "the turn carries a withheld prompt: {prompt}");
        assert_eq!(turn.reply, "", "the turn carries the reply to a withheld prompt: {prompt}");
        assert_eq!(write(&brain, &cfg, &turn), Written::Skipped, "{prompt}");
    }
    assert_eq!(captures(conn), 2);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM memories"), 2);
}

/// The rule holds in `write_turn` by itself: a turn that carries such text is refused even
/// when the flag says it is clean.
#[test]
fn t04_write_turn_refuses_what_core_withholds_whatever_the_flag() {
    let lab = Lab::new();
    let brain = lab.open();
    let cfg = lab.cfg();
    let cwd = lab.project("alpha");
    let clean = |prompt: &str, reply: &str, withheld: bool| CompletedTurn {
        session_id: "S".to_string(),
        cwd: Some(cwd.clone()),
        prompt: prompt.to_string(),
        turn_key: archive::turn_key(prompt),
        reply: reply.to_string(),
        started_ms: 1_700_000_000_000,
        at_ms: 1_700_000_001_000,
        suppressed: false,
        withheld,
    };
    let unflagged_prompt = clean("my password is hunter2, keep it for the deploy on monday", REPLY, false);
    let unflagged_reply = clean(PROMPT, "The Secret is in the vault, as agreed with the operations people.", false);
    let flagged = clean(PROMPT, REPLY, true);
    for turn in [&unflagged_prompt, &unflagged_reply, &flagged] {
        assert_eq!(write(&brain, &cfg, turn), Written::Skipped, "{turn:?}");
    }
    nothing_stored(&brain.conn);
    assert_eq!(write(&brain, &cfg, &clean(PROMPT, REPLY, false)), Written::Stored);
}

/// The text is redacted before it is cut: a token that straddles the cut leaves no head.
#[test]
fn t04_no_piece_of_a_token_survives_the_cut() {
    let token = gitlab_token();
    let mut cap = CaptureState::default();
    let prompt = format!("{} {token} and the rest", "x".repeat(4_085));
    cap.on_user(Some("s"), None, &prompt, 0);
    let reply = format!("{} {token} done", "y".repeat(4_085));
    let turn = cap.on_assistant(Some("s"), &reply, false, 1).unwrap();
    assert!(!turn.withheld);
    assert!(turn.prompt.len() <= 4_096 && turn.reply.len() <= 4_096);
    assert!(turn.prompt.len() > 4_085 && turn.reply.len() > 4_085);
    for text in [&turn.prompt, &turn.reply] {
        assert!(!text.contains("glpat"), "the head of the token was kept");
    }
}

// ── 5. Switches ─────────────────────────────────────────────────────────────

#[test]
fn t05_capture_off_stores_nothing() {
    let lab = Lab::new();
    let brain = lab.open();
    let off = lab.cfg_with("off", "off");
    assert!(!off.capture);
    let cwd = lab.project("alpha");
    let turn = pair("S", &cwd, PROMPT, REPLY, 1_700_000_000_000);
    assert_eq!(write(&brain, &off, &turn), Written::Skipped);
    nothing_stored(&brain.conn);
    assert_eq!(write(&brain, &lab.cfg(), &turn), Written::Stored);
}

#[test]
fn t05_a_learning_pause_stops_the_chat_and_only_the_chat() {
    let lab = Lab::new();
    let cfg = lab.cfg();
    let cwd = lab.project("alpha");
    let brain = lab.open();
    let s = project::logical("S");

    // A pause of recall alone stops no learning.
    tx(&brain.conn, || store::set_pause(&brain.conn, &s, Some("S"), "recall", "cli").unwrap());
    let before = pair("S", &cwd, PROMPT, REPLY, 1_700_000_000_000);
    assert_eq!(write(&brain, &cfg, &before), Written::Stored);
    assert_eq!(captures_of(&brain.conn, "S"), 1);

    // `learning` on S: the capture it had is hidden, and no new turn of S is written.
    let hid = tx(&brain.conn, || store::set_pause(&brain.conn, &s, Some("S"), "learning", "model").unwrap());
    assert_eq!(hid, 1);
    let turn_s = pair("S", &cwd, "what did we decide about the reader of note files", REPLY, 1_700_000_100_000);
    let turn_t = pair("T", &cwd, PROMPT, REPLY, 1_700_000_100_000);
    assert_eq!(write(&brain, &cfg, &turn_s), Written::Skipped);
    assert_eq!(write(&brain, &cfg, &turn_t), Written::Stored);
    assert_eq!(captures_of(&brain.conn, "S"), 1, "a turn of the paused chat was written");
    assert_eq!(captures_of(&brain.conn, "T"), 1);
    assert_eq!(visible(&brain.conn), vec![id_of(&turn_t)], "nothing of S can be found");

    // Still in force after the state and the connection are dropped and opened again.
    drop(brain);
    let brain = lab.open();
    let again = pair("S", &cwd, "and what about the forum, is it part of this wave", REPLY, 1_700_000_200_000);
    assert_eq!(write(&brain, &cfg, &again), Written::Skipped);
    assert_eq!(captures_of(&brain.conn, "S"), 1);

    // Lifted through a second connection (another process), honoured by the next turn.
    let other = brain_v2::connect(&lab.brain_path(), Role::Cli).unwrap();
    tx(&other, || store::clear_pause(&other, &s, Some("S"), "learning").unwrap());
    assert_eq!(write(&brain, &cfg, &again), Written::Stored);
    assert_eq!(captures_of(&brain.conn, "S"), 2);

    // A pause written through the second connection is honoured by the next turn. It is
    // keyed by the chat (`logical`), with no session id on the row.
    let t = project::logical("T");
    tx(&other, || store::set_pause(&other, &t, None, "learning", "cli").unwrap());
    let next_t = pair("T", &cwd, "is the status row written in every mode", REPLY, 1_700_000_300_000);
    assert_eq!(write(&brain, &cfg, &next_t), Written::Skipped);
    assert_eq!(captures_of(&brain.conn, "T"), 1);
    // S is not paused any more.
    let next_s = pair("S", &cwd, "is the status row written in every mode", REPLY, 1_700_000_300_000);
    assert_eq!(write(&brain, &cfg, &next_s), Written::Stored);

    // A pause row that names the session only (its `logical` is another value).
    tx(&other, || {
        other
            .execute(
                "INSERT INTO chat_pauses(logical, what, since_ms, set_by, session_id) VALUES ('other', 'learning', 1, 'cli', 'U')",
                [],
            )
            .unwrap()
    });
    let turn_u = pair("U", &cwd, PROMPT, REPLY, 1_700_000_400_000);
    assert_eq!(write(&brain, &cfg, &turn_u), Written::Skipped);
    assert_eq!(captures_of(&brain.conn, "U"), 0);
}

// ── 6. Suppress, the capture half ───────────────────────────────────────────

fn call_row(conn: &Connection, session_id: &str, turn_key: &str, from_ms: i64, to_ms: i64) {
    tx(conn, || {
        conn.execute(
            "INSERT INTO forgotten_turns(logical, session_id, source, turn_key, from_ms, to_ms, added_ms)
             VALUES (?1, ?2, 'call', ?3, ?4, ?5, ?5)",
            params![project::logical(session_id), session_id, turn_key, from_ms, to_ms],
        )
        .unwrap()
    });
}

fn to_ms(conn: &Connection, session_id: &str) -> i64 {
    conn.query_row(
        "SELECT to_ms FROM forgotten_turns WHERE source = 'call' AND logical = ?1",
        [project::logical(session_id)],
        |r| r.get(0),
    )
    .unwrap()
}

#[test]
fn t06_a_suppressed_turn_stores_no_row_and_closes_its_interval() {
    let lab = Lab::new();
    let brain = lab.open();
    let cfg = lab.cfg();
    let cwd = lab.project("alpha");
    let conn = &brain.conn;
    let prompt = "forget that I told you about the offer from Acme, it was not final";

    let mut cap = CaptureState::default();
    assert_eq!(cap.suppress("S"), None, "nothing is pending");
    cap.on_user(Some("S"), Some(&cwd), prompt, 1_000);
    assert_eq!(cap.on_assistant(Some("S"), "I will call the forget tool for the Acme offer.", true, 1_500), None);
    // `suppress` returns the prompt's hook time and its key.
    let (from_ms, key) = cap.suppress("S").unwrap();
    assert_eq!(from_ms, 1_000);
    assert_eq!(key, archive::turn_key(&screen::redact(prompt).text));
    // What the live call writes (commit 18a): the `call` row, ending at the call.
    call_row(conn, "S", &key, from_ms, 1_600);
    // Another chat's row with the same start is not this turn's.
    call_row(conn, "T", "", 1_000, 1_600);

    let turn = cap.on_assistant(Some("S"), "Done: the offer from Acme is forgotten, nothing of it is kept.", false, 5_000).unwrap();
    assert!(turn.suppressed);
    assert_eq!((turn.started_ms, turn.at_ms), (1_000, 5_000));
    assert_eq!(turn.turn_key, key);
    assert_eq!((turn.prompt.as_str(), turn.reply.as_str()), ("", ""), "a marked turn carries no text");
    assert_eq!(cap.suppress("S"), None, "no pending buffer");

    assert_eq!(write(&brain, &cfg, &turn), Written::Skipped);
    nothing_stored(conn);
    assert_eq!(to_ms(conn, "S"), 5_000, "the interval ends with the turn");
    assert_eq!(to_ms(conn, "T"), 1_600, "another chat's interval was moved");

    // An interval is never made shorter, and a second write changes nothing.
    assert_eq!(write(&brain, &cfg, &CompletedTurn { at_ms: 4_000, ..turn.clone() }), Written::Skipped);
    assert_eq!(to_ms(conn, "S"), 5_000);

    // A marked turn that does carry text is not stored either, and with capture off its
    // interval is closed all the same.
    let loud = CompletedTurn {
        prompt: prompt.to_string(),
        reply: REPLY.to_string(),
        at_ms: 6_000,
        ..turn.clone()
    };
    assert_eq!(write(&brain, &lab.cfg_with("off", "off"), &loud), Written::Skipped);
    assert_eq!(write(&brain, &cfg, &loud), Written::Skipped);
    nothing_stored(conn);
    assert_eq!(to_ms(conn, "S"), 6_000);
    assert_eq!(count(conn, "SELECT COUNT(*) FROM forgotten_turns"), 2, "write_turn writes no row of its own");

    // The same turn unmarked would have been stored: the mark is what kept it out.
    let unmarked = CompletedTurn { suppressed: false, ..loud };
    assert_eq!(write(&brain, &cfg, &unmarked), Written::Stored);
}

// ── 7. Filed elsewhere ──────────────────────────────────────────────────────

fn archive_row(conn: &Connection, scope: &str, name: &str, session_id: &str) {
    tx(conn, || {
        conn.execute(
            "INSERT INTO archive_index(scope_dir, rel_name, archive_id, logical, state, first_seen_ms)
             VALUES (?1, ?2, ?3, ?4, 'indexed', 1)",
            params![scope, format!("s000001/{name}.json"), name, project::logical(session_id)],
        )
        .unwrap()
    });
}

#[test]
fn t07_a_chat_filed_under_another_project_is_not_captured() {
    let lab = Lab::new();
    let brain = lab.open();
    let cfg = lab.cfg();
    let conn = &brain.conn;
    let (a, b) = (lab.project("alpha"), lab.project("beta"));
    assert_ne!(scope_dir(&a), scope_dir(&b));

    // The chat has an indexed archive under A and sends a turn with a `cwd` in B: no row.
    archive_row(conn, &scope_dir(&a), &"a".repeat(32), "S");
    let in_b = pair("S", &b, PROMPT, REPLY, 1_700_000_000_000);
    assert_eq!(write(&brain, &cfg, &in_b), Written::Skipped);
    nothing_stored(conn);
    // From A it is captured, and another chat in B is not affected.
    let in_a = pair("S", &a, PROMPT, REPLY, 1_700_000_000_000);
    assert_eq!(write(&brain, &cfg, &in_a), Written::Stored);
    let other = pair("T", &b, PROMPT, REPLY, 1_700_000_000_000);
    assert_eq!(write(&brain, &cfg, &other), Written::Stored);
    assert_eq!(captures_of(conn, "S"), 1);

    // After an archive of the chat exists under B, the next turn is captured under B.
    archive_row(conn, &scope_dir(&b), &"b".repeat(32), "S");
    let next = pair("S", &b, "and where is the next turn of this chat filed", REPLY, 1_700_000_100_000);
    assert_eq!(write(&brain, &cfg, &next), Written::Stored);
    let filed: String =
        conn.query_row("SELECT project_key FROM memories WHERE id = ?1", [id_of(&next)], |r| r.get(0)).unwrap();
    assert_eq!(filed, scope_dir(&b));

    // Rows alone file a chat too: a chat with captures under A and no archive anywhere
    // is not captured from B.
    let first = pair("V", &a, PROMPT, REPLY, 1_700_000_200_000);
    assert_eq!(write(&brain, &cfg, &first), Written::Stored);
    let moved = pair("V", &b, "does this chat belong to the other folder now", REPLY, 1_700_000_300_000);
    assert_eq!(write(&brain, &cfg, &moved), Written::Skipped);
    assert_eq!(captures_of(conn, "V"), 1);
}

// ── 8. No network ───────────────────────────────────────────────────────────

#[test]
fn t08_after_the_server_has_started_the_process_is_offline() {
    use velocirag::download::{ensure_model, is_offline, model_files_in, models_cache_dir, EMBEDDER_MODEL};

    let (b, _) = boot(None);
    let end = serve(std::io::Cursor::new(Vec::<u8>::new()), std::io::sink(), b);
    assert_eq!(end, ServeEnd::InputClosed);
    assert!(is_offline());

    // The gate runs this with HOME pointing at an empty folder and forbids a test to move
    // HOME itself: the model folder is the one under whatever home the test was given.
    let dir = models_cache_dir().join(EMBEDDER_MODEL.local_dir);
    if model_files_in(&dir).is_none() {
        let (dir_was, parent_was) = (dir.exists(), models_cache_dir().exists());
        assert!(ensure_model(&EMBEDDER_MODEL).is_err());
        assert_eq!(dir.exists(), dir_was, "ensure_model made the model folder");
        assert_eq!(models_cache_dir().exists(), parent_was, "ensure_model made the cache folder");
        assert!(model_files_in(&dir).is_none());
    }
    assert!(is_offline());
}

// ── 9. A pause hides ────────────────────────────────────────────────────────

#[test]
fn t09_a_pause_hides_the_chats_captures_and_resume_brings_them_back() {
    let lab = Lab::new();
    let brain = lab.open();
    let cfg = lab.cfg();
    let cwd = lab.project("alpha");
    let conn = &brain.conn;
    let prompts = [
        "what is the plan for the reader of archive files",
        "which folder do the notes of this project live in",
        "when does the embed thread stand aside for a write",
        "who reviews the forum part of this wave of commits",
    ];
    let of_s: Vec<String> = prompts
        .iter()
        .enumerate()
        .map(|(i, prompt)| {
            let turn = pair("S", &cwd, prompt, REPLY, 1_700_000_000_000 + i as i64 * 10_000);
            assert_eq!(write(&brain, &cfg, &turn), Written::Stored);
            id_of(&turn)
        })
        .collect();
    let turn_t = pair("T", &cwd, PROMPT, REPLY, 1_700_000_000_000);
    assert_eq!(write(&brain, &cfg, &turn_t), Written::Stored);
    let of_t = id_of(&turn_t);
    assert_eq!(visible(conn).len(), 5);

    // S's model hid one row before the pause.
    assert_eq!(tx(conn, || store::hide(conn, &of_s[3], "model:S").unwrap()), 1);

    let s = project::logical("S");
    let hidden = tx(conn, || store::set_pause(conn, &s, Some("S"), "learning", "model").unwrap());
    assert_eq!(hidden, 3);
    for id in &of_s[..3] {
        assert_eq!(hidden_by(conn, id).as_deref(), Some("pause"), "{id}");
    }
    assert_eq!(hidden_by(conn, &of_s[3]).as_deref(), Some("model:S"));
    assert_eq!(hidden_by(conn, &of_t), None, "T's capture was touched");
    assert_eq!(visible(conn), vec![of_t.clone()], "S's captures are in a search from another chat");

    // A turn that completes after the pause is not written: nothing of S comes after it.
    let late = pair("S", &cwd, "one more question about the archive reader", REPLY, 1_700_000_900_000);
    assert_eq!(write(&brain, &cfg, &late), Written::Skipped);
    assert_eq!(captures_of(conn, "S"), 4);

    let back = tx(conn, || store::clear_pause(conn, &s, Some("S"), "learning").unwrap());
    assert_eq!(back, 3);
    let mut expected: Vec<String> = of_s[..3].to_vec();
    expected.push(of_t);
    expected.sort();
    assert_eq!(visible(conn), expected, "the three are visible again");
    assert_eq!(hidden_by(conn, &of_s[3]).as_deref(), Some("model:S"), "the row S's model hid is still hidden");
}

// ── 10. drop_session, turn_key ──────────────────────────────────────────────

#[test]
fn t10_drop_session_and_the_key_of_the_whole_prompt() {
    // R-9e: a chat that lost a hook is not paired.
    let mut cap = CaptureState::default();
    cap.on_user(Some("S"), None, "one", 1_000);
    cap.on_user(Some("T"), None, "another chat", 1_000);
    cap.drop_session("S");
    assert_eq!(cap.on_assistant(Some("S"), "the reply to a later prompt", false, 2_000), None);
    assert!(cap.on_assistant(Some("T"), "a reply", false, 2_000).is_some());
    cap.drop_session("nobody");

    // B-2f: a 5,000-byte prompt has the key of the whole redacted text.
    let long = "word ".repeat(1_000);
    assert_eq!(long.len(), 5_000);
    cap.on_user(Some("S"), None, &long, 3_000);
    let turn = cap.on_assistant(Some("S"), "a reply", false, 4_000).unwrap();
    assert_eq!(turn.turn_key, archive::turn_key(&screen::redact(&long).text));
    assert_eq!(turn.turn_key, archive::turn_key(&long));
    assert_ne!(turn.turn_key, archive::turn_key(&long[..4_096]), "the key is of the cut prompt");
    assert_eq!(turn.prompt, &long[..4_096]);

    // A prompt holding a key-shaped token: the key of the redacted text, not of the raw.
    let token = gitlab_token();
    let with_token = format!("use {token} for the mirror job");
    cap.on_user(Some("S"), None, &with_token, 5_000);
    let turn = cap.on_assistant(Some("S"), "a reply", false, 6_000).unwrap();
    assert_eq!(turn.turn_key, archive::turn_key("use (redacted) for the mirror job"));
    assert_eq!(turn.turn_key, archive::turn_key(&screen::redact(&with_token).text));
    assert_ne!(turn.turn_key, archive::turn_key(&with_token));
    // `suppress` hands out the same key.
    cap.on_user(Some("S"), None, &with_token, 7_000);
    assert_eq!(cap.suppress("S"), Some((7_000, turn.turn_key.clone())));
    cap.drop_session("S");

    // Two prompts that differ only after byte 4,096 have different keys.
    let head = "ab ".repeat(1_400);
    assert!(head.len() > 4_096);
    let (one, two) = (format!("{head}first ending"), format!("{head}second ending"));
    cap.on_user(Some("S"), None, &one, 8_000);
    let turn_one = cap.on_assistant(Some("S"), "a reply", false, 8_500).unwrap();
    cap.on_user(Some("S"), None, &two, 9_000);
    let turn_two = cap.on_assistant(Some("S"), "a reply", false, 9_500).unwrap();
    assert_eq!(turn_one.prompt, turn_two.prompt, "the kept text is the same 4,096 bytes");
    assert_ne!(turn_one.turn_key, turn_two.turn_key);
}

// ── 11. What core would withhold ────────────────────────────────────────────

#[test]
fn t11_what_core_would_withhold_is_not_captured() {
    let lab = Lab::new();
    let (mut ctx, mut st, jobs, status, _kept) = threads(&lab, "turns");
    let cwd = lab.project("alpha");

    // "my password is hunter2" and a reply: no row, counted once.
    user(&mut st, "S", &cwd, "my password is hunter2", 1_000);
    assert!(st.capture.is_pending("S"));
    assistant(&mut st, "S", &cwd, "Noted. I will use it for the deploy to the staging machine tonight.", false, 2_000);
    assert_eq!(status.turns_withheld.load(Ordering::Relaxed), 1);
    let turn = next_capture(&jobs).expect("the turn goes to the indexer");
    assert!(turn.withheld);
    assert_eq!((turn.prompt.as_str(), turn.reply.as_str()), ("", ""), "the job carries text of a withheld turn");
    indexer::dispatch(&mut ctx, Job::Capture(turn));
    nothing_stored(ctx_conn(&ctx));

    // A clean prompt whose reply contains "secret": no row.
    user(&mut st, "S", &cwd, PROMPT, 3_000);
    assistant(&mut st, "S", &cwd, "It reads the newest window first.", true, 3_500);
    assistant(&mut st, "S", &cwd, "The secret of the order is the window number in the file name.", false, 4_000);
    assert_eq!(status.turns_withheld.load(Ordering::Relaxed), 2);
    let turn = next_capture(&jobs).unwrap();
    assert!(turn.withheld);
    assert_eq!((turn.prompt.as_str(), turn.reply.as_str()), ("", ""));
    indexer::dispatch(&mut ctx, Job::Capture(turn));
    nothing_stored(ctx_conn(&ctx));

    // A clean turn: one row, and no count.
    user(&mut st, "S", &cwd, PROMPT, 5_000);
    assistant(&mut st, "S", &cwd, REPLY, false, 6_000);
    let turn = next_capture(&jobs).unwrap();
    assert!(!turn.withheld);
    assert_eq!((turn.started_ms, turn.at_ms), (5_000, 6_000), "the times are the hooks' own");
    indexer::dispatch(&mut ctx, Job::Capture(turn.clone()));
    assert_eq!(captures(ctx_conn(&ctx)), 1);
    assert_eq!(content_of(ctx_conn(&ctx), &id_of(&turn)), format!("User: {PROMPT}\nAssistant: {REPLY}"));
    assert_eq!(status.turns_withheld.load(Ordering::Relaxed), 2);
    assert_eq!(ctx.status.write_errors.load(Ordering::Relaxed), 0);
    assert!(next_capture(&jobs).is_none(), "one job per turn");

    // Nowhere in the brain: not in the rows, not in the word index.
    let conn = ctx_conn(&ctx);
    for word in ["hunter2", "password", "secret"] {
        let sql = format!(
            "SELECT COUNT(*) FROM memories WHERE content LIKE '%{word}%' OR title LIKE '%{word}%' OR abstract_text LIKE '%{word}%'"
        );
        assert_eq!(count(conn, &sql), 0, "{word}");
    }
}

// ── The three calls of `ext::hooks` ─────────────────────────────────────────

#[test]
fn hook_events_pair_without_a_brain_and_the_indexer_writes_the_turn() {
    let lab = Lab::new();
    let cwd = lab.project("alpha");

    // No brain at all: pairing needs no connection (it does no SQL) and still hands the
    // turn over.
    let (b, _) = boot(None);
    let (mut bare, bare_jobs, _vec_tx) = front::open_state(lab.cfg(), b);
    assert!(bare.conn.is_none());
    user(&mut bare, "S", &cwd, PROMPT, 1_000);
    assistant(&mut bare, "S", &cwd, REPLY, false, 2_000);
    assert!(next_capture(&bare_jobs).is_some());
    assert!(!lab.brain_path().exists(), "the front thread made a brain");
    drop(bare);

    let (mut ctx, mut st, jobs, status, _kept) = threads(&lab, "turns");

    // A whole turn with a tool use in the middle.
    user(&mut st, "S", &cwd, PROMPT, 1_000);
    assistant(&mut st, "S", &cwd, "Let me look at the scan.", true, 1_500);
    assert!(next_capture(&jobs).is_none(), "the turn is still running");
    assistant(&mut st, "S", &cwd, REPLY, false, 2_000);
    let turn = next_capture(&jobs).unwrap();
    assert_eq!(turn.cwd.as_deref(), Some(cwd.as_str()));
    assert_eq!(turn.reply, format!("Let me look at the scan.\n{REPLY}"));
    // `indexer::dispatch` calls `write_turn` on a bare connection.
    assert!(ctx_conn(&ctx).is_autocommit());
    indexer::dispatch(&mut ctx, Job::Capture(turn.clone()));
    assert!(ctx_conn(&ctx).is_autocommit());
    assert_eq!(captures_of(ctx_conn(&ctx), "S"), 1);
    assert_eq!(ctx.status.write_errors.load(Ordering::Relaxed), 0);

    // No session id: nothing. No `has_tool_use`: the turn is dropped, not guessed.
    capture::on_user_event(&mut st, &hook("before_message", None, Some(&cwd), Some(PROMPT), json!({}), 3_000));
    assert_eq!(st.capture.pending_len(), 0);
    user(&mut st, "S", &cwd, PROMPT, 3_000);
    capture::on_assistant_event(&mut st, &hook("on_message_complete", Some("S"), Some(&cwd), Some(REPLY), json!({}), 3_500));
    assert!(!st.capture.is_pending("S"));
    assert!(next_capture(&jobs).is_none());

    // A host event in the prompt hook clears the chat's buffer.
    user(&mut st, "S", &cwd, PROMPT, 4_000);
    user(&mut st, "S", &cwd, "<event id=\"1\">a worker reported</event>", 4_100);
    assistant(&mut st, "S", &cwd, REPLY, false, 4_200);
    assert!(next_capture(&jobs).is_none());

    // `on_session_end` drops the pending turn.
    user(&mut st, "S", &cwd, PROMPT, 5_000);
    capture::on_session_end_event(&mut st, &hook("on_session_end", Some("S"), Some(&cwd), None, json!({}), 5_100));
    assistant(&mut st, "S", &cwd, REPLY, false, 5_200);
    assert!(next_capture(&jobs).is_none());

    // A marked turn goes to the indexer too (it closes the interval there), with no text.
    user(&mut st, "S", &cwd, "forget what I said about the offer from Acme last week", 6_000);
    assert!(st.capture.suppress("S").is_some());
    assistant(&mut st, "S", &cwd, "It is forgotten: nothing about the offer from Acme is kept.", false, 7_000);
    let marked = next_capture(&jobs).unwrap();
    assert!(marked.suppressed);
    assert_eq!((marked.prompt.as_str(), marked.reply.as_str()), ("", ""));
    indexer::dispatch(&mut ctx, Job::Capture(marked));
    assert_eq!(captures_of(ctx_conn(&ctx), "S"), 1);
    assert_eq!(status.turns_withheld.load(Ordering::Relaxed), 0);
    assert_eq!(ctx.status.write_errors.load(Ordering::Relaxed), 0);
}

#[test]
fn with_capture_off_the_indexer_stores_nothing_and_nothing_is_counted() {
    let lab = Lab::new();
    let (mut ctx, mut st, jobs, status, _kept) = threads(&lab, "off");
    let cwd = lab.project("alpha");
    user(&mut st, "S", &cwd, PROMPT, 1_000);
    assistant(&mut st, "S", &cwd, REPLY, false, 2_000);
    user(&mut st, "S", &cwd, "my password is hunter2", 3_000);
    assistant(&mut st, "S", &cwd, "Noted.", false, 4_000);
    while let Some(turn) = next_capture(&jobs) {
        indexer::dispatch(&mut ctx, Job::Capture(turn));
    }
    nothing_stored(ctx_conn(&ctx));
    assert_eq!(status.turns_withheld.load(Ordering::Relaxed), 0);
    // The pairing still runs: a forget call needs the turn's start and key.
    user(&mut st, "S", &cwd, PROMPT, 5_000);
    assert_eq!(st.capture.suppress("S"), Some((5_000, archive::turn_key(PROMPT))));
}

/// What the path of a project folder is made of does not reach the row.
#[test]
fn the_row_is_filed_by_directory_hash_not_by_path() {
    let lab = Lab::new();
    let brain = lab.open();
    let cwd = lab.project("alpha");
    let inside = Path::new(&cwd).join("src");
    std::fs::create_dir_all(&inside).unwrap();
    let turn = pair("S", &inside.to_string_lossy(), PROMPT, REPLY, 1_700_000_000_000);
    assert_eq!(write(&brain, &lab.cfg(), &turn), Written::Stored);
    let filed: String =
        brain.conn.query_row("SELECT project_key FROM memories WHERE id = ?1", [id_of(&turn)], |r| r.get(0)).unwrap();
    assert_eq!(filed, scope_dir(&cwd), "a folder inside the project is the project");
    assert_eq!(filed.len(), 64);
}

// ── R-16: what the review of wave 4 changed ─────────────────────────────────

fn dropped(status: &Status) -> u64 {
    status.turns_dropped.load(Ordering::Relaxed)
}

/// Item 1: every turn the buffer lets go is counted, each once; a turn that is replaced,
/// marked or withheld is not one; with capture off nothing is counted.
#[test]
fn r16_1_every_dropped_turn_is_counted() {
    let lab = Lab::new();
    let (_ctx, mut st, jobs, status, _kept) = threads(&lab, "turns");
    let cwd = lab.project("alpha");

    // A turn that is paired and sent is not dropped, however long it took.
    user(&mut st, "S", &cwd, PROMPT, 0);
    assistant(&mut st, "S", &cwd, "starting a worker", true, 1_000);
    assistant(&mut st, "S", &cwd, REPLY, false, 15 * 60_000);
    assert!(next_capture(&jobs).is_some(), "a reply fifteen minutes later is still paired");
    assert_eq!(dropped(&status), 0);

    // No `has_tool_use`: dropped, counted. With nothing pending there is nothing to count.
    let bare = |at| hook("on_message_complete", Some("S"), Some(&cwd), Some(REPLY), json!({}), at);
    user(&mut st, "S", &cwd, PROMPT, 1);
    capture::on_assistant_event(&mut st, &bare(2));
    assert_eq!(dropped(&status), 1);
    capture::on_assistant_event(&mut st, &bare(3));
    assert_eq!(dropped(&status), 1);

    // A host event clears the buffer.
    user(&mut st, "S", &cwd, PROMPT, 4);
    user(&mut st, "S", &cwd, "<event id=\"1\">a worker reported</event>", 5);
    assert_eq!(dropped(&status), 2);

    // The session ends in the middle of a turn.
    user(&mut st, "S", &cwd, PROMPT, 6);
    capture::on_session_end_event(&mut st, &hook("on_session_end", Some("S"), Some(&cwd), None, json!({}), 7));
    assert_eq!(dropped(&status), 3);
    capture::on_session_end_event(&mut st, &hook("on_session_end", Some("S"), Some(&cwd), None, json!({}), 8));
    assert_eq!(dropped(&status), 3);

    // A lost hook (`drop_session` is the front thread's call): counted at the next hook.
    user(&mut st, "S", &cwd, PROMPT, 9);
    st.capture.drop_session("S");
    user(&mut st, "T", &cwd, PROMPT, 10);
    assert_eq!(dropped(&status), 4);
    assistant(&mut st, "T", &cwd, REPLY, false, 11);
    assert!(next_capture(&jobs).is_some());

    // Eviction: 65 chats pending, the oldest goes, once.
    for i in 0..65 {
        user(&mut st, &format!("chat{i}"), &cwd, PROMPT, 100 + i);
    }
    assert_eq!(st.capture.pending_len(), 64);
    assert!(!st.capture.is_pending("chat0"));
    assert_eq!(dropped(&status), 5);

    // Not dropped: a prompt that replaces the pending one (the turn goes on), a turn a
    // forget call marked, a turn core's rule withholds.
    user(&mut st, "chat1", &cwd, "a second prompt for the same turn", 200);
    user(&mut st, "chat2", &cwd, "forget what I said about the offer from Acme", 201);
    assert!(st.capture.suppress("chat2").is_some());
    capture::on_session_end_event(&mut st, &hook("on_session_end", Some("chat2"), Some(&cwd), None, json!({}), 202));
    user(&mut st, "chat3", &cwd, "my password is hunter2", 203);
    capture::on_session_end_event(&mut st, &hook("on_session_end", Some("chat3"), Some(&cwd), None, json!({}), 204));
    assert_eq!(dropped(&status), 5);

    // Capture off: no turn is stored, so none is missing.
    let off = Lab::new();
    let (_ctx, mut st, _jobs, status, _kept) = threads(&off, "off");
    user(&mut st, "S", &cwd, PROMPT, 1);
    capture::on_session_end_event(&mut st, &hook("on_session_end", Some("S"), Some(&cwd), None, json!({}), 2));
    assert_eq!(dropped(&status), 0);
}

/// Item 1: a turn whose job finds the queue full is a dropped turn too.
#[test]
fn r16_1_a_turn_the_queue_refuses_is_counted() {
    let lab = Lab::new();
    let (_ctx, mut st, jobs, status, _kept) = threads(&lab, "turns");
    let cwd = lab.project("alpha");
    while st.jobs.try_send(Job::Tick).is_ok() {}
    user(&mut st, "S", &cwd, PROMPT, 1);
    assistant(&mut st, "S", &cwd, REPLY, false, 2);
    assert!(next_capture(&jobs).is_none());
    assert_eq!(dropped(&status), 1);
    assert_eq!(status.jobs_dropped.load(Ordering::Relaxed), 1);
}

fn log_row(conn: &Connection, session_id: &str, prompt: &str) {
    tx(conn, || {
        conn.execute(
            "INSERT INTO recall_log(ts_ms, session_id, logical, scope_dir, query_preview, query_hash, outcome, total_ms, detail)
             VALUES (1000, ?1, ?1, NULL, ?2, ?3, 'injected', 1.0, '{}')",
            params![session_id, prompt, recall::query_hash(prompt)],
        )
        .unwrap();
    });
}

fn log_rows(conn: &Connection, session_id: &str, prompt: &str) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM recall_log WHERE session_id = ?1 AND query_preview = ?2",
        params![session_id, prompt],
        |r| r.get(0),
    )
    .unwrap()
}

/// Item 3: the recall log wrote its row at the prompt. A turn that ends withheld takes
/// that row with it: the row of this prompt of this chat, and no other.
#[test]
fn r16_3_a_turn_that_ends_withheld_takes_its_recall_log_row() {
    let lab = Lab::new();
    let (mut ctx, mut st, jobs, _status, _kept) = threads(&lab, "turns");
    let cwd = lab.project("alpha");
    let other = "what did we decide about the order of the archive windows last week";

    // A clean prompt whose reply core would withhold.
    for chat in ["S", "T"] {
        log_row(ctx_conn(&ctx), chat, PROMPT);
        log_row(ctx_conn(&ctx), chat, other);
    }
    user(&mut st, "S", &cwd, PROMPT, 1_000);
    assistant(&mut st, "S", &cwd, "It reads them in the order the secret of the build names.", false, 2_000);
    let turn = next_capture(&jobs).unwrap();
    assert!(turn.withheld);
    assert_eq!(turn.turn_key, recall::query_hash(PROMPT), "capture's key is the log's hash");
    indexer::dispatch(&mut ctx, Job::Capture(turn));
    let conn = ctx_conn(&ctx);
    assert!(conn.is_autocommit());
    assert_eq!(log_rows(conn, "S", PROMPT), 0, "the log kept the prompt of a withheld turn");
    assert_eq!(log_rows(conn, "S", other), 1, "another prompt of the chat");
    assert_eq!(log_rows(conn, "T", PROMPT), 1, "the same prompt in another chat");
    assert_eq!(captures(conn), 0);
    assert_eq!(ctx.status.write_errors.load(Ordering::Relaxed), 0);

    // A turn that is stored keeps its row.
    user(&mut st, "T", &cwd, PROMPT, 3_000);
    assistant(&mut st, "T", &cwd, REPLY, false, 4_000);
    indexer::dispatch(&mut ctx, Job::Capture(next_capture(&jobs).unwrap()));
    let conn = ctx_conn(&ctx);
    assert_eq!(captures_of(conn, "T"), 1);
    assert_eq!(log_rows(conn, "T", PROMPT), 1);

    // `write_turn` by itself, for a turn whose flag says clean and whose text does not.
    let brain = Lab::new();
    let cfg = brain.cfg();
    let b = brain.open();
    log_row(&b.conn, "S", PROMPT);
    let turn = CompletedTurn {
        session_id: "S".to_string(),
        cwd: Some(brain.project("alpha")),
        prompt: PROMPT.to_string(),
        turn_key: archive::turn_key(PROMPT),
        reply: "the password of the build machine is in the vault".to_string(),
        started_ms: 1_000,
        at_ms: 2_000,
        suppressed: false,
        withheld: false,
    };
    assert_eq!(write(&b, &cfg, &turn), Written::Skipped);
    assert!(b.conn.is_autocommit());
    assert_eq!(log_rows(&b.conn, "S", PROMPT), 0);

    // Capture off: `write_turn` touches nothing (the log keeps no text of a prompt then).
    log_row(&b.conn, "S", PROMPT);
    let mut off = cfg.clone();
    off.capture = false;
    assert_eq!(write(&b, &off, &CompletedTurn { withheld: true, ..turn }), Written::Skipped);
    assert_eq!(log_rows(&b.conn, "S", PROMPT), 1);
}
