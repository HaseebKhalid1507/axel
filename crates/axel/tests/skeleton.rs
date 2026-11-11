//! The skeleton's contract (commit 3): every stub that has a fixed answer gives that answer.
//!
//! A later commit replaces a stub with the real thing and deletes the stub's line here. A
//! line that fails without that is a stub whose contract was changed by accident: another
//! commit of the same wave is built against it.
//!
//! The stub checks open no brain: their connection is in memory and their paths are never
//! opened. The indexer's tick has run the real `adopt_orphans` since commit 4, so that one
//! test makes a fresh version 2 brain in a temp folder.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{channel, sync_channel, Receiver};
use std::sync::Arc;
use std::time::Instant;

use rusqlite::Connection;
use serde_json::{json, Value};

use axel::archive::{self, ArchivePolicy, LiveArchive, Start};
use axel::brain_v2::{self, OpenOpts};
use axel::capture::{self, CaptureState, CompletedTurn, Written};
use axel::embed::EmbedService;
use axel::ext::config::{EmbedderKind, ExtConfig, ForumMode};
use axel::ext::types::{
    BrainArg, Caller, DegradedReason, FrontState, HookEvent, HookOut, IndexerCtx, Job, Mode,
    Status, TestHooks, ToolError,
};
use axel::ext::{front, hooks, tools, tools_core};
use axel::recall::{self, RecallConfig, RecallRequest, RecallTrace, View};
use axel::vindex::{ExactIndex, VecDelta, VecKey, VecScope};
use axel::{docsync, index_archives, index_notes, indexer, indexer_start, inject, project};

fn cfg() -> ExtConfig {
    ExtConfig {
        brain: PathBuf::from("/nonexistent/skeleton/brain.r8"),
        base: PathBuf::from("/nonexistent/skeleton/base"),
        data_dir: PathBuf::from("/nonexistent/skeleton/data"),
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
        model_dir: PathBuf::from("/nonexistent/skeleton/model"),
        log_queries: false,
        warnings: Vec::new(),
    }
}

/// A `FrontState` written field by field: when the list in `ext/types.rs` moves, this stops
/// compiling.
fn front_state(test: TestHooks) -> (FrontState, Receiver<Job>) {
    let (jobs, jobs_rx) = sync_channel::<Job>(1024);
    let (_vec_tx, vec_rx) = channel::<VecDelta>();
    let test = Arc::new(test);
    let st = FrontState {
        cfg: cfg(),
        conn: None,
        signer: None,
        index: ExactIndex::new(384),
        vec_rx,
        embed: EmbedService::start(None, jobs.clone(), test.clone()),
        jobs,
        capture: CaptureState::default(),
        status: Arc::new(Status::default()),
        mode: Arc::new(AtomicU8::new(Mode::Ready as u8)),
        forum_on: false,
        filed: HashMap::new(),
        forget_marks: Vec::new(),
        clock: Arc::new(|| 1_000),
        test,
        writing: Arc::new(AtomicBool::new(false)),
    };
    (st, jobs_rx)
}

fn indexer_ctx(test: TestHooks, conn: Option<Connection>) -> (IndexerCtx, Receiver<Job>) {
    let (jobs, jobs_rx) = sync_channel::<Job>(1024);
    let (vec_tx, _vec_rx) = channel::<VecDelta>();
    let test = Arc::new(test);
    let ctx = IndexerCtx {
        cfg: cfg(),
        conn,
        signer: None,
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
        clock: Arc::new(|| 1_000),
        test,
        writing: Arc::new(AtomicBool::new(false)),
    };
    (ctx, jobs_rx)
}

fn event(kind: &str) -> HookEvent {
    HookEvent {
        kind: kind.to_string(),
        session_id: Some("s1".to_string()),
        cwd: None,
        message: Some("what did we decide about the retry limit".to_string()),
        data: Value::Null,
        received_ms: 1_000,
    }
}

fn turn() -> CompletedTurn {
    CompletedTurn {
        session_id: "s1".to_string(),
        cwd: None,
        prompt: "p".to_string(),
        turn_key: "k".to_string(),
        reply: "r".to_string(),
        started_ms: 1,
        at_ms: 2,
        suppressed: false,
        withheld: false,
    }
}

fn live_archive() -> LiveArchive {
    LiveArchive {
        id: "0".repeat(32),
        scope: "scope".to_string(),
        logical: "logical".to_string(),
        digest: "digest".to_string(),
        created_ms: None,
        source_message_count: 0,
        note: String::new(),
        messages: Vec::new(),
        file_size: 0,
        carried: None,
        start: Start::Unknown,
    }
}

fn policy() -> ArchivePolicy {
    ArchivePolicy {
        event_sources: Vec::new(),
        forget_tools: vec!["axel_forget".to_string(), "memory_forget".to_string()],
        keep_forget_turns: false,
    }
}

#[test]
fn every_stub_gives_its_fixed_answer() {
    let missing = Path::new("/nonexistent/skeleton/brain.r8");
    let conn = Connection::open_in_memory().unwrap();

    // brain_v2 (commit 4) and screen (commit 8) are real: `brain_v2_tests.rs` and the tests
    // in `screen.rs` hold their contracts now.

    // archive
    let a = live_archive();
    assert!(archive::forgotten_keys(&a, &policy()).is_empty());
    assert_eq!(archive::carried_head(&a), None);
    assert!(archive::extract_units(&a, &policy()).is_empty());
    assert!(archive::list_project(missing).is_empty());
    assert!(archive::read_archive(missing, "scope", None).is_err());
    assert_eq!(Start::default(), Start::Unknown);

    // project
    assert!(project::derive_key(missing).is_none());
    assert!(project::resolve(Some("/nonexistent/skeleton")).is_none());

    // store, index_archives
    assert!(index_archives::mark_forgotten_turn(&conn, "l", Some("s1"), "tool", 0, 1, None)
        .unwrap()
        .is_empty());

    // capture
    let mut cap = CaptureState::default();
    cap.on_user(Some("s1"), None, "hello", 1);
    cap.drop_session("s1");
    cap.on_session_end("s1");
    assert_eq!(capture::write_turn(&conn, None, &cfg(), &turn()).unwrap(), Written::Skipped);

    // vindex, embed
    let mut index = ExactIndex::new(384);
    index.apply(VecDelta::Remove(vec![VecKey::Mem(1)]));
    assert_eq!(index.len(), 0);
    assert!(index.get(&VecKey::Doc(1)).is_none());
    assert!(index.search(&[0.0; 384], 5, &|_: &VecScope| true).is_empty());
    let (st, _jobs_rx) = front_state(TestHooks::default());
    assert_eq!(st.embed.model_id(), None);
    assert_eq!(st.embed.embed_query("q", Instant::now()), None);
    assert_eq!(st.embed.embed_wait(vec!["t".to_string()], Instant::now()), None);
    st.embed.submit_bulk(vec![(VecKey::Mem(1), "t".to_string())]);
    assert!(!st.embed.stopped());

    // recall
    let view = front::view_for(&Caller::default(), &st.cfg);
    assert!(view.project.is_none() && !view.global && !view.documents && view.own_session.is_none());
    let req = RecallRequest {
        query: "retry limit",
        view: &View { project: None, global: true, documents: true, own_session: None },
        kinds: None,
        since_ms: None,
        max_items: 5,
        deadline: Instant::now(),
        ambient: true,
    };
    let out = recall::recall(&conn, &st.index, &st.embed, &RecallConfig { now: 1_000 }, &req);
    assert!(out.picks.is_empty());

    // types
    for v in 0..=9u8 {
        assert_eq!(DegradedReason::from_u8(v) as u8, v);
    }
    assert_eq!(DegradedReason::from_u8(200), DegradedReason::None);
    assert_eq!(DegradedReason::IndexerStopped.as_str(), "IndexerStopped");
    assert_eq!((Mode::Starting as u8, Mode::Ready as u8, Mode::Degraded as u8), (0, 1, 2));
    assert_eq!(Mode::from_u8(0), Some(Mode::Starting));
    assert_eq!(Mode::from_u8(1), Some(Mode::Ready));
    assert_eq!(Mode::from_u8(2), Some(Mode::Degraded));
    assert_eq!(Mode::from_u8(3), None);
    let brain = BrainArg { path: missing.to_path_buf(), from_env: false, on_command_line: true };
    assert!(!brain.from_env && brain.on_command_line);
}

#[test]
fn the_hook_dispatch_answers_continue_for_every_kind() {
    let (mut st, _jobs_rx) = front_state(TestHooks::default());
    for kind in [
        "before_message",
        "on_message_complete",
        "on_session_start",
        "on_session_end",
        "on_context_archived",
        "on_memory_forgotten",
        "no_such_hook",
    ] {
        for recall_allowed in [true, false] {
            assert_eq!(hooks::handle(&mut st, &event(kind), recall_allowed), HookOut::Continue, "{kind}");
        }
    }
    assert_eq!(inject::on_session_start(&mut st, &event("on_session_start")), HookOut::Continue);
    assert_eq!(
        recall::on_before_message(&mut st, &event("before_message"), true),
        HookOut::Continue
    );
}

#[test]
fn the_tool_dispatch_declares_nothing_and_knows_no_name() {
    assert!(tools::declarations(false).is_empty());
    assert!(tools_core::tools().is_empty());

    let (mut st, _jobs_rx) = front_state(TestHooks::default());
    let caller = Caller::default();
    for name in ["search", "remember", "recall", "forget", "forum_post", "forum_read", "forum_forget"] {
        let got = tools::call(&mut st, name, &json!({}), &caller);
        assert!(matches!(got, Err(ToolError::Invalid(_))), "{name}: {got:?}");
    }
}

#[test]
fn the_tool_dispatch_panics_on_the_test_switch() {
    let (mut st, _jobs_rx) =
        front_state(TestHooks { panic_tool: Some("search".to_string()), ..TestHooks::default() });
    let caller = Caller::default();
    let got = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tools::call(&mut st, "search", &json!({}), &caller)
    }));
    assert!(got.is_err());
    assert!(matches!(
        tools::call(&mut st, "recall", &json!({}), &caller),
        Err(ToolError::Invalid(_))
    ));
}

#[test]
fn the_indexer_dispatch_and_tick_do_zero_work_and_survive_a_panic() {
    // A fresh version 2 brain: the tick's first call is the real `adopt_orphans`, which
    // reports an error on a connection without the tables (and must: on a real brain that
    // is damage).
    let dir = tempfile::TempDir::new().unwrap();
    let opts = OpenOpts { agent: None, model: "hash-384", dim: 384, upgrade_v1: false };
    let conn = brain_v2::open_v2(&dir.path().join("brain.r8"), &opts).unwrap().conn;
    let (mut ctx, _jobs_rx) = indexer_ctx(
        TestHooks { panic_on_forgotten_job: true, ..TestHooks::default() },
        Some(conn),
    );

    assert!(index_notes::pass(&mut ctx).is_ok());
    assert!(index_notes::on_forgotten(&mut ctx, "id", "p0000000000000000").is_ok());
    assert!(index_archives::scan(&mut ctx).is_ok());
    assert!(index_archives::on_forgotten(&mut ctx, "id", "p0000000000000000").is_ok());
    assert!(docsync::tick(&mut ctx).is_ok());
    indexer_start::ensure_open(&mut ctx);
    assert!(indexer_start::pump_embeddings(&mut ctx).is_ok());
    assert!(indexer_start::resync_vectors(&mut ctx).is_ok());
    assert!(indexer_start::write_status(&mut ctx).is_ok());
    assert!(indexer_start::on_embedded(&mut ctx, "m", Vec::new()).is_ok());
    assert!(recall::write_log(&mut ctx, &RecallTrace::default()).is_ok());
    assert!(recall::trim_log(&mut ctx).is_ok());

    let archive_job = Job::Archive {
        archive_id: "0".repeat(32),
        path: None,
        project_key: "p0000000000000000".to_string(),
        cwd: None,
        session_id: None,
        received_ms: 1_000,
        window: None,
        part: None,
    };
    assert!(index_archives::on_job(&mut ctx, &archive_job).is_ok());
    for job in [
        Job::Tick,
        archive_job,
        Job::Capture(turn()),
        Job::LogRecall(Box::default()),
        Job::Embedded { model: "m".to_string(), vectors: Vec::new() },
    ] {
        indexer::dispatch(&mut ctx, job);
    }
    assert_eq!(ctx.status.panics.load(Ordering::Relaxed), 0);
    assert_eq!(ctx.status.write_errors.load(Ordering::Relaxed), 0);

    // The switch makes a Forgotten job panic; the panic is caught and counted.
    indexer::dispatch(
        &mut ctx,
        Job::Forgotten {
            id: "id".to_string(),
            kind: "note".to_string(),
            project_key: "p0000000000000000".to_string(),
        },
    );
    assert_eq!(ctx.status.panics.load(Ordering::Relaxed), 1);

    // The first tick runs all eight calls; a second at the same time runs none.
    indexer::tick(&mut ctx);
    assert_eq!(ctx.last_run.len(), 8);
    indexer::tick(&mut ctx);
    assert_eq!(ctx.status.panics.load(Ordering::Relaxed), 1);
    assert_eq!(ctx.status.write_errors.load(Ordering::Relaxed), 0);
}

#[test]
fn the_indexer_loop_ends_on_shutdown_and_on_the_job_limit() {
    // The loop opens the brain its config names: one under a temporary folder.
    let dir = tempfile::TempDir::new().unwrap();
    let in_dir = |ctx: &mut IndexerCtx, name: &str| {
        ctx.cfg.data_dir = dir.path().join(name);
        ctx.cfg.brain = ctx.cfg.data_dir.join("brain.r8");
        ctx.cfg.base = dir.path().join("base");
    };
    let (mut ctx, jobs_rx) = indexer_ctx(TestHooks::default(), None);
    in_dir(&mut ctx, "one");
    ctx.jobs.try_send(Job::Shutdown).unwrap();
    indexer::run(ctx, jobs_rx);

    let (mut ctx, jobs_rx) = indexer_ctx(
        TestHooks { indexer_stop_after_jobs: Some(2), ..TestHooks::default() },
        None,
    );
    in_dir(&mut ctx, "two");
    ctx.jobs.try_send(Job::Tick).unwrap();
    ctx.jobs.try_send(Job::Tick).unwrap();
    indexer::run(ctx, jobs_rx);
}

/// The block's own check: "`axel --help` lists the new subcommands".
#[test]
fn help_lists_the_new_subcommands_and_not_mcp() {
    let out = Command::new(env!("CARGO_BIN_EXE_axel")).arg("--help").output().unwrap();
    assert!(out.status.success());
    let help = String::from_utf8(out.stdout).unwrap();
    let listed: Vec<&str> = help
        .lines()
        .filter(|l| l.starts_with("  ") && !l.starts_with("   "))
        .filter_map(|l| l.split_whitespace().next())
        .collect();
    for name in [
        "migrate", "plugin", "status", "why", "show", "pause", "resume", "restore", "model",
        "synaps-export", "forget", "recall", "search", "memories", "init", "extension",
    ] {
        assert!(listed.contains(&name), "{name} is not in --help:\n{help}");
    }
    assert!(!listed.contains(&"mcp"), "mcp is still in --help");
}
