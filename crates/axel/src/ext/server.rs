//! `run()`, `serve()`, thread start-up, the deadline pump, the breaker.
//!
//! The rpc thread is the caller of `serve`. It reads frames (through a small reader thread,
//! so that it can also watch the clock), answers what it can answer itself, and hands one
//! call at a time to the front thread, waiting on a channel with a deadline. It runs no SQL,
//! takes no lock and loads no model.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Once, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use velocirag::embedder::Embedder;

use crate::error::Result;
use crate::ext::config::{EmbedderKind, ExtConfig, ForumMode};
use crate::ext::front::{self, FrontAnswer, FrontCall, FrontReply, FrontWork};
use crate::ext::rpc::{self, FrameError, FrameReader, Response};
use crate::ext::tools;
use crate::ext::types::{
    Boot, Caller, Carry, Clock, DegradedReason, HookEvent, HookOut, IndexerCtx, Job, Mode, ServeEnd, Status,
    TestHooks, ToolError,
};
use crate::indexer;

/// The queue to the indexer: `sync_channel(1024)`; the rpc and front threads `try_send`.
pub const JOB_QUEUE: usize = 1024;
/// What the rpc thread waits for a hook other than `before_message`.
const HOOK_WAIT_MS: u64 = 150;
/// What the rpc thread waits for a tool.
const TOOL_WAIT_MS: u64 = 2_000;
/// A front thread still inside one call after this long ends the process (R-5).
const FRONT_HANG_MS: u64 = 30_000;
/// A front thread that has not handed over the indexer's queue this long after `initialize`
/// has hung at its start (wave 3, 12b ruling 2).
const FRONT_START_MS: u64 = 60_000;
/// `before_message` calls in a row that missed the budget before recall is switched off.
const TRIP_AFTER_MISSES: u32 = 3;
/// How long the breaker stays open.
const TRIP_OPEN_FOR: Duration = Duration::from_secs(600);
/// The log is renamed to `extension.log.1` when it passes this.
const LOG_LIMIT: u64 = 4 * 1024 * 1024;
/// How long `serve` waits for its threads after `shutdown` or end of input.
pub(crate) const EXIT_WAIT: Duration = Duration::from_millis(200);
// The front thread's last try of its kept forget marks waits `front::LAST_TRY_WAIT` for the
// write lock before it returns; that wait must end inside this one, or `serve` returns
// first and the marks and their log line are lost (v2 front follow-up, item 5).
const _: () = assert!(
    front::LAST_TRY_WAIT.as_nanos() < EXIT_WAIT.as_nanos(),
    "front::LAST_TRY_WAIT must be shorter than server::EXIT_WAIT"
);
/// The exit status of a process whose front thread died or hung.
const EXIT_FRONT_STUCK: i32 = 70;
/// The status file is written again after this long without a change.
const STATUS_FILE_EVERY: Duration = Duration::from_secs(30);
/// A status file older than this was left by a process that did not end cleanly.
const STATUS_FILE_OLD: Duration = Duration::from_secs(24 * 3600);

const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

/// `ToolError::Busy`, and what the rpc thread answers a tool at its deadline.
pub const BUSY_TEXT: &str = "Axel is busy; the outcome of this call is unknown; an exact retry is safe";

/// `ToolError::Unavailable(reason)` as the host is sent it.
pub fn unavailable_text(reason: &str) -> String {
    if reason == "BrainOldShape" {
        return format!(
            "Axel is unavailable: BrainOldShape (the brain {}). Synaps itself is not affected.",
            crate::brain_v2::old_shape()
        );
    }
    format!("Axel is unavailable: {reason}. Synaps itself is not affected.")
}

// ── Degrading ───────────────────────────────────────────────────────────────

/// The one way a thread of this process sets `Degraded`: the reason is stored first, then
/// the mode, so a reader that sees the mode always finds its reason. `at` is a file and
/// line, kept with this reason (the first place given for a reason stays): a reader never
/// finds the place of an earlier, different failure beside this one.
// RULING: wave4 R-12b item 6.
pub fn degrade(status: &Status, mode: &AtomicU8, reason: DegradedReason, at: Option<String>) {
    if let Some(at) = at {
        status.set_reason_place(reason, at);
    }
    status.reason.store(reason as u8, Ordering::SeqCst);
    mode.store(Mode::Degraded as u8, Ordering::SeqCst);
}

/// The one way back to `Ready`: the mode first, then the reason is cleared, so nobody reads
/// `Degraded` with no reason (the mirror of `degrade`).
// RULING: wave3-c12b ruling 10.
pub fn recover(status: &Status, mode: &AtomicU8) {
    mode.store(Mode::Ready as u8, Ordering::SeqCst);
    status.reason.store(DegradedReason::None as u8, Ordering::SeqCst);
}

/// The `<reason>` of the tool sentence: the variant's name when the mode is `Degraded`.
// SPEC: 990 says a tool answers "Axel is starting" before the brain is open and 3865 gives
// one sentence for everything outside `Ready`. The sentence of 3865 is used, with
// `starting` as its reason while the mode is `Starting` (there is no `DegradedReason` yet).
pub fn unavailable_reason(status: &Status, mode: &AtomicU8) -> &'static str {
    match Mode::from_u8(mode.load(Ordering::SeqCst)) {
        Some(Mode::Degraded) => DegradedReason::from_u8(status.reason.load(Ordering::SeqCst)).as_str(),
        _ => "starting",
    }
}

// ── The log ─────────────────────────────────────────────────────────────────

/// `<data_dir>/extension.log`: ids, counts, timings and error kinds, never text. Each line
/// is one append to a file opened for that line, so no thread holds a lock or a handle;
/// when the file passes its limit it is renamed to `extension.log.1` (one generation).
#[derive(Clone, Debug)]
pub struct Log {
    path: PathBuf,
    limit: u64,
    /// Held by the one thread of this process that is renaming the file.
    renaming: Arc<AtomicBool>,
}

impl Log {
    pub fn new(data_dir: &Path, limit: u64) -> Log {
        Log { path: data_dir.join("extension.log"), limit, renaming: Arc::new(AtomicBool::new(false)) }
    }

    pub fn line(&self, text: &str) {
        if let Some(dir) = self.path.parent() {
            if !dir.exists() {
                let _ = std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir);
            }
        }
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).mode(0o600).open(&self.path)
        else {
            return;
        };
        let _ = file.write_all(format!("{ms} {text}\n").as_bytes());
        let len = file.metadata().map(|m| m.len()).unwrap_or(0);
        drop(file);
        // RULING: wave3-c12b ruling 10. One thread renames at a time, and it looks at the
        // size again once it is the one: a second thread that saw the same full file would
        // otherwise put the few lines of the new file over the generation just kept. A
        // thread that finds the flag taken does not wait; the next line looks again.
        if len > self.limit && self.renaming.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok()
        {
            let still = std::fs::metadata(&self.path).map(|m| m.len()).unwrap_or(0);
            if still > self.limit {
                let mut old = self.path.clone().into_os_string();
                old.push(".1");
                let _ = std::fs::rename(&self.path, old);
            }
            self.renaming.store(false, Ordering::SeqCst);
        }
    }
}

// ── The status file ─────────────────────────────────────────────────────────

// RULING: wave3-c12b ruling 1. One file per process, beside the log: the mode and its
// reason, whatever the mode is. The brain's status row cannot say them: a degraded process
// must not write the brain, and a starting one has none.

/// `<data_dir>/extension.<pid>.status`.
pub fn status_file_path(data_dir: &Path, pid: u32) -> PathBuf {
    data_dir.join(format!("extension.{pid}.status"))
}

fn mode_name(mode: u8) -> &'static str {
    match Mode::from_u8(mode) {
        Some(Mode::Ready) => "ready",
        Some(Mode::Degraded) => "degraded",
        _ => "starting",
    }
}

/// `extension.<digits>.status`, or that name with `.tmp` after it.
fn is_status_file_name(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("extension.") else { return false };
    let rest = rest.strip_suffix(".tmp").unwrap_or(rest);
    let Some(pid) = rest.strip_suffix(".status") else { return false };
    !pid.is_empty() && pid.bytes().all(|b| b.is_ascii_digit())
}

/// At start: status files not written for 24 hours are removed. A younger file may be a
/// running process's, and is left whatever its pid.
fn remove_old_status_files(data_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(data_dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_str().is_some_and(is_status_file_name) {
            continue;
        }
        let old = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|at| at.elapsed().ok())
            .is_some_and(|age| age >= STATUS_FILE_OLD);
        if old {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Under a temporary name, then renamed: a reader never sees half an object. Mode 0600.
fn write_status_file(path: &Path, body: &Value) -> std::io::Result<()> {
    let mut tmp = path.to_path_buf().into_os_string();
    tmp.push(".tmp");
    let mut file = std::fs::OpenOptions::new().create(true).write(true).truncate(true).mode(0o600).open(&tmp)?;
    file.write_all(body.to_string().as_bytes())?;
    drop(file);
    std::fs::rename(&tmp, path)
}

thread_local! {
    /// The log of the process this thread belongs to (a test runs several in one process).
    static THREAD_LOG: RefCell<Option<Log>> = const { RefCell::new(None) };
    /// `file:line` of the last panic on this thread, for whoever catches it.
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// The log of a real process, for threads this file did not start (the embed thread).
static PROCESS_LOG: OnceLock<Log> = OnceLock::new();
/// Set by `run`: the panic hook then never hands a panic to Rust's default hook.
static QUIET: AtomicBool = AtomicBool::new(false);
static HOOK: Once = Once::new();

/// Gives this thread its log (`None` takes it away).
pub fn set_thread_log(log: Option<Log>) {
    let _ = THREAD_LOG.try_with(|l| {
        if let Ok(mut l) = l.try_borrow_mut() {
            *l = log;
        }
    });
}

/// One line in this thread's log; nothing when the thread has none.
pub fn log_line(text: &str) {
    let log = THREAD_LOG.try_with(|l| l.try_borrow().ok().and_then(|l| l.clone())).ok().flatten();
    if let Some(log) = log {
        log.line(text);
    }
}

/// `file:line` of the last panic on this thread, once.
pub fn take_last_panic() -> Option<String> {
    LAST_PANIC.try_with(|p| p.try_borrow_mut().ok().and_then(|mut p| p.take())).ok().flatten()
}

/// Installed first. A panic is logged as its file and line and nothing else: Rust's default
/// message can quote the string that was being cut, and the host forwards stderr to its log.
fn install_panic_hook() {
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let at = match info.location() {
                Some(l) => format!("{}:{}", l.file(), l.line()),
                None => "unknown".to_string(),
            };
            let _ = LAST_PANIC.try_with(|p| {
                if let Ok(mut p) = p.try_borrow_mut() {
                    *p = Some(at.clone());
                }
            });
            let own = THREAD_LOG.try_with(|l| l.try_borrow().ok().and_then(|l| l.clone())).ok().flatten();
            let quiet = QUIET.load(Ordering::SeqCst);
            match own.or_else(|| if quiet { PROCESS_LOG.get().cloned() } else { None }) {
                Some(log) => log.line(&format!("panic at {at}")),
                None if quiet => eprintln!("axel: panic at {at}"),
                None => {}
            }
            // Inside a test binary the other tests keep their messages. A real process
            // (`run`) never reaches Rust's default hook.
            if !quiet {
                previous(info);
            }
        }));
    });
}

/// Takes the thread's log away when `serve` returns on a thread it does not own.
struct ThreadLogGuard;
impl Drop for ThreadLogGuard {
    fn drop(&mut self) {
        set_thread_log(None);
    }
}

// ── run, serve ──────────────────────────────────────────────────────────────

/// `axel extension`: serves stdin and stdout, then ends the process with 0 (`shutdown`, end
/// of input) or 70 (the front thread died or hung). It does not return.
// SPEC: 4874 gives `run(brain_path)` and no line says what the path is for once the brain
// comes from `initialize.params.config.brain`. The config value is used when it is set;
// `brain_path` (the command line's `--brain`, `AXEL_BRAIN`, or the default) otherwise.
pub fn run(brain_path: &Path) -> Result<()> {
    QUIET.store(true, Ordering::SeqCst);
    install_panic_hook();
    let boot = Boot {
        embedder: None,
        clock: Arc::new(|| chrono::Utc::now().timestamp_millis()),
        test: Arc::new(TestHooks::default()),
        status: Arc::new(Status::default()),
    };
    let end = serve_from(std::io::stdin(), std::io::stdout(), boot, Some(brain_path.to_path_buf()));
    std::process::exit(match end {
        ServeEnd::Shutdown | ServeEnd::InputClosed => 0,
        ServeEnd::FrontStuck => EXIT_FRONT_STUCK,
    })
}

/// `Read`, not `BufRead`: the reader is wrapped here, so a test passes a pipe or a `Cursor`
/// as it is. Returns when the host says `shutdown`, when input ends, or when the front
/// thread is dead or stuck.
pub fn serve<R, W>(reader: R, writer: W, boot: Boot) -> ServeEnd
where
    R: Read + Send + 'static,
    W: Write + Send + 'static,
{
    serve_from(reader, writer, boot, None)
}

type Frame = std::result::Result<Vec<u8>, FrameError>;

fn serve_from<R, W>(reader: R, writer: W, boot: Boot, brain: Option<PathBuf>) -> ServeEnd
where
    R: Read + Send + 'static,
    W: Write + Send + 'static,
{
    install_panic_hook();
    let _unset = ThreadLogGuard;

    // The reader thread only moves frames; the rpc thread waits on this channel with a
    // timeout, which is how it sees a late or a hung front thread while no frame arrives.
    let (frames_tx, frames) = mpsc::sync_channel::<Frame>(256);
    let spawned = thread::Builder::new().name("axel-read".to_string()).spawn(move || {
        let mut frames_in = FrameReader::new(BufReader::new(reader));
        while let Some(frame) = frames_in.next_frame() {
            // A reader that fails goes on failing: `next_frame` would return the same error
            // for ever. It is the end of input.
            if matches!(frame, Err(FrameError::Io(_))) {
                return;
            }
            if frames_tx.send(frame).is_err() {
                return;
            }
        }
    });
    if spawned.is_err() {
        return ServeEnd::InputClosed;
    }

    let mut pump = Pump {
        writer,
        out_closed: false,
        status: boot.status.clone(),
        clock: boot.clock.clone(),
        test: boot.test.clone(),
        boot: Some(boot),
        brain,
        mode: Arc::new(AtomicU8::new(Mode::Starting as u8)),
        stop: Arc::new(AtomicBool::new(false)),
        log: None,
        tools: Vec::new(),
        budget_ms: HOOK_WAIT_MS,
        live: None,
        misses: 0,
        open_until: None,
        carry: Carry::default(),
        seq: 0,
        busy: None,
        status_file: None,
        status_seen: None,
        status_at: None,
        started: None,
        was_ready: false,
        front_gone: false,
    };
    let end = pump.pump(&frames);
    pump.finish(end)
}

// ── The pump ────────────────────────────────────────────────────────────────

/// The threads behind the pump, from `initialize` on.
struct Live {
    calls: Sender<FrontCall>,
    replies: Receiver<FrontReply>,
    /// The indexer's queue. The front thread makes it (after it has loaded the model, which
    /// the rpc thread must not wait for) and sends this end over `jobs_from`.
    jobs: Option<SyncSender<Job>>,
    jobs_from: Receiver<SyncSender<Job>>,
    /// Closed when the thread has ended; before that, the file and line of its panic.
    front_done: Receiver<String>,
    indexer_done: Receiver<()>,
}

/// The call the front thread is still inside, after its deadline.
struct Busy {
    since: Instant,
    method: &'static str,
    seq: u64,
}

enum Got {
    Answer(FrontAnswer),
    /// Forwarded; the deadline passed.
    Late,
    /// The front thread is still inside an earlier call.
    NotForwarded,
    /// The front thread has ended, or has been inside one call for too long.
    Dead,
}

struct Pump<W: Write> {
    writer: W,
    out_closed: bool,
    status: Arc<Status>,
    clock: Clock,
    test: Arc<TestHooks>,
    /// Taken by the first `initialize`.
    boot: Option<Boot>,
    brain: Option<PathBuf>,
    mode: Arc<AtomicU8>,
    /// Set before the indexer is told to end, so that its ending is not a failure.
    stop: Arc<AtomicBool>,
    log: Option<Log>,
    tools: Vec<Value>,
    budget_ms: u64,
    live: Option<Live>,
    // The breaker: in memory, on this thread. A restart clears it.
    misses: u32,
    open_until: Option<Instant>,
    carry: Carry,
    seq: u64,
    busy: Option<Busy>,
    /// From `initialize` on: this process's status file.
    status_file: Option<PathBuf>,
    /// The mode and reason last written to it, and when.
    status_seen: Option<(u8, u8)>,
    status_at: Option<Instant>,
    /// When `initialize` started the threads.
    started: Option<Instant>,
    /// This thread has seen the mode `Ready`.
    was_ready: bool,
    /// The front thread died or hung before that: the mode is `Degraded(FrontStopped)` for
    /// the rest of this process.
    front_gone: bool,
}

fn continue_result() -> Value {
    json!({"action": "continue"})
}

fn str_field(v: &Value, key: &str) -> Option<String> {
    v.get(key).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string)
}

fn is_hex32(s: &str) -> bool {
    s.len() == 32 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_project_key(s: &str) -> bool {
    s.len() == 17
        && s.starts_with('p')
        && s[1..].bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn positive(v: Option<&Value>) -> Option<i64> {
    v.and_then(Value::as_i64).filter(|n| *n > 0)
}

/// The name of a frame error, never its text.
fn frame_error_kind(e: &FrameError) -> &'static str {
    match e {
        FrameError::TooLarge(_) => "TooLarge",
        FrameError::Truncated => "Truncated",
        FrameError::Empty => "Empty",
        FrameError::BadHeader => "BadHeader",
        FrameError::MissingBlankLine => "MissingBlankLine",
        FrameError::Io(_) => "Io",
    }
}

const BEFORE_MESSAGE_LABEL: &str = "hook.handle before_message";

/// What the log calls a forwarded hook: a name this binary knows, never the host's string.
fn hook_label(kind: &str) -> &'static str {
    match kind {
        "before_message" => BEFORE_MESSAGE_LABEL,
        "on_message_complete" => "hook.handle on_message_complete",
        "on_session_start" => "hook.handle on_session_start",
        "on_session_end" => "hook.handle on_session_end",
        _ => "hook.handle",
    }
}

impl<W: Write> Pump<W> {
    fn log(&self, text: &str) {
        if let Some(log) = &self.log {
            log.line(text);
        }
    }

    /// Only end of input, `shutdown`, or a dead front thread ends the loop.
    fn pump(&mut self, frames: &Receiver<Frame>) -> ServeEnd {
        loop {
            let wait = Duration::from_millis(if self.busy.is_some() { 20 } else { 250 });
            match frames.recv_timeout(wait) {
                Ok(Ok(body)) => {
                    if let Some(end) = self.on_frame(&body) {
                        return end;
                    }
                }
                Ok(Err(e)) => self.log(&format!("frame skipped: {}", frame_error_kind(&e))),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => return ServeEnd::InputClosed,
            }
            if self.out_closed {
                return ServeEnd::InputClosed;
            }
            if self.front_is_dead() || self.front_has_stopped() {
                return ServeEnd::FrontStuck;
            }
            self.close_breaker_when_due();
            self.status_file_when_due();
        }
    }

    /// Looks at the front thread outside any call: it has ended (its end of `front_done`
    /// is gone), or it has not handed over the indexer's queue in time. True when the
    /// process must end (the mode was `Ready` before); before the first `Ready` the process
    /// stays, `Degraded(FrontStopped)`: the host would start it again into the same failure.
    // RULING: wave3-c12b ruling 2.
    fn front_has_stopped(&mut self) -> bool {
        if self.front_gone {
            // The indexer may have stored `Ready` in the same instant: the reason stands.
            let held = self.mode.load(Ordering::SeqCst) == Mode::Degraded as u8
                && self.status.reason.load(Ordering::SeqCst) == DegradedReason::FrontStopped as u8;
            if !held {
                degrade(&self.status, &self.mode, DegradedReason::FrontStopped, None);
            }
            return false;
        }
        if self.mode.load(Ordering::SeqCst) == Mode::Ready as u8 {
            self.was_ready = true;
        }
        let Some(live) = self.live.as_mut() else { return false };
        if live.jobs.is_none() {
            live.jobs = live.jobs_from.try_recv().ok();
        }
        let mut at = None;
        let ended = loop {
            match live.front_done.try_recv() {
                Ok(place) => at = Some(place),
                Err(TryRecvError::Empty) => break false,
                Err(TryRecvError::Disconnected) => break true,
            }
        };
        let limit = Duration::from_millis(self.test.front_start_limit_ms.unwrap_or(FRONT_START_MS));
        let hung = !ended && live.jobs.is_none() && self.started.is_some_and(|t| t.elapsed() >= limit);
        if !ended && !hung {
            return false;
        }
        self.log(if ended { "front thread ended" } else { "front thread did not start in time" });
        if self.was_ready {
            return true;
        }
        self.front_gone = true;
        degrade(&self.status, &self.mode, DegradedReason::FrontStopped, at);
        false
    }

    /// The status file, when the mode or the reason changed and every 30 s. This thread
    /// writes it because it is the one that is alive in every mode.
    fn status_file_when_due(&mut self) {
        let Some(path) = self.status_file.clone() else { return };
        // The mode first: whoever degrades stores the reason before it.
        let mode = self.mode.load(Ordering::SeqCst);
        let degraded = mode == Mode::Degraded as u8;
        let reason = if degraded { self.status.reason.load(Ordering::SeqCst) } else { DegradedReason::None as u8 };
        let unchanged = self.status_seen == Some((mode, reason));
        if unchanged && self.status_at.is_some_and(|at| at.elapsed() < STATUS_FILE_EVERY) {
            return;
        }
        self.status_seen = Some((mode, reason));
        self.status_at = Some(Instant::now());
        let body = json!({
            "pid": std::process::id(),
            "mode": mode_name(mode),
            "reason": DegradedReason::from_u8(reason).as_str(),
            "reason_at": if degraded { self.status.reason_place() } else { None },
            "written_ms": (self.clock)(),
            "started_ms": self.status.started_ms.load(Ordering::Relaxed),
            // RULING: v2 c18a review, item 5: `status` knows this process by these.
            "start_ticks": crate::cli::status::own_start_ticks(),
        });
        if write_status_file(&path, &body).is_err() {
            self.log("status file not written");
        }
    }

    fn on_frame(&mut self, body: &[u8]) -> Option<ServeEnd> {
        let request = match rpc::parse_request(body) {
            Ok(r) => r,
            Err(e) => {
                self.log(&format!("frame skipped: {e:?}"));
                return None;
            }
        };
        match request.method.as_str() {
            "initialize" => {
                self.on_initialize(request.id, &request.params);
                None
            }
            "hook.handle" => self.on_hook(request.id, &request.params),
            "tool.call" => self.on_tool(request.id, &request.params),
            "shutdown" => {
                self.stop.store(true, Ordering::SeqCst);
                self.answer(request.id, json!({}));
                Some(ServeEnd::Shutdown)
            }
            other => {
                if let Some(id) = request.id {
                    self.send(&Response::method_not_found(id, other));
                }
                None
            }
        }
    }

    fn send(&mut self, response: &Response) {
        if rpc::write_response(&mut self.writer, response).is_err() {
            self.out_closed = true;
        }
    }

    fn answer(&mut self, id: Option<Value>, result: Value) {
        if let Some(id) = id {
            self.send(&Response::result(id, result));
        }
    }

    fn answer_hook(&mut self, id: Option<Value>, out: HookOut) {
        let result = match out {
            HookOut::Continue => continue_result(),
            HookOut::Inject(content) => json!({"action": "inject", "content": content}),
        };
        self.answer(id, result);
    }

    /// Section 5: `Invalid` is the JSON-RPC error `-32602`; the other three kinds are a
    /// tool result with `is_error`.
    fn answer_tool(&mut self, id: Option<Value>, out: std::result::Result<String, ToolError>) {
        let Some(id) = id else { return };
        let response = match out {
            Ok(text) => Response::result(id, json!({"content": text, "is_error": false})),
            Err(ToolError::Invalid(message)) => Response::error(id, INVALID_PARAMS, message),
            Err(ToolError::Unavailable(reason)) => {
                Response::result(id, json!({"content": unavailable_text(&reason), "is_error": true}))
            }
            Err(ToolError::Busy) => Response::result(id, json!({"content": BUSY_TEXT, "is_error": true})),
            Err(ToolError::Refused(text)) => Response::result(id, json!({"content": text, "is_error": true})),
        };
        self.send(&response);
    }

    fn ready(&self) -> bool {
        self.live.is_some() && !self.front_gone && self.mode.load(Ordering::SeqCst) == Mode::Ready as u8
    }

    /// `ready()`, after waiting up to `until` while the brain is still opening. Nothing is
    /// waited for once the mode is `Degraded` (the open failed) or before `initialize`.
    // RULING: v2 real-binaries round, item 1: a call in the first moments of a process was
    // refused with `starting` while the brain was a few hundred ms from open.
    fn ready_by(&mut self, until: Instant) -> bool {
        while self.live.is_some()
            && !self.front_gone
            && self.mode.load(Ordering::SeqCst) == Mode::Starting as u8
            && Instant::now() < until
        {
            thread::sleep(Duration::from_millis(5));
        }
        self.ready()
    }

    // ── initialize ──────────────────────────────────────────────────────────

    /// Answered from the config alone, before the brain is open. The tools are declared
    /// whatever the mode.
    fn on_initialize(&mut self, id: Option<Value>, params: &Value) {
        if let Some(boot) = self.boot.take() {
            let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/"));
            let mut cfg = ExtConfig::from_initialize(params, &home);
            let named = params
                .get("config")
                .and_then(|c| c.get("brain"))
                .and_then(Value::as_str)
                .is_some_and(|s| !s.trim().is_empty());
            if let (false, Some(path)) = (named, self.brain.clone()) {
                cfg.data_dir = path
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."));
                cfg.brain = path;
            }
            // `<base>/config` is read once, here. The threads get the answer, not the
            // question, so the status row says what this `initialize` declared.
            let forum_on = cfg.forum_on();
            cfg.forum = if forum_on { ForumMode::On } else { ForumMode::Off };

            let log = Log::new(&cfg.data_dir, self.test.log_limit_bytes.unwrap_or(LOG_LIMIT));
            if QUIET.load(Ordering::SeqCst) {
                let _ = PROCESS_LOG.set(log.clone());
            }
            set_thread_log(Some(log.clone()));
            self.log = Some(log);
            self.tools = tools::declarations(forum_on);
            self.budget_ms = cfg.recall_budget_ms;
            self.status.started_ms.store((self.clock)(), Ordering::Relaxed);
            self.log(&format!(
                "initialize pid={} forum={} warnings={}",
                std::process::id(),
                if forum_on { "on" } else { "off" },
                cfg.warnings.len()
            ));
            remove_old_status_files(&cfg.data_dir);
            self.status_file = Some(status_file_path(&cfg.data_dir, std::process::id()));
            self.started = Some(Instant::now());
            self.live = self.start(cfg, boot);
            self.status_file_when_due();
        }
        let tools = self.tools.clone();
        self.answer(id, json!({"protocol_version": 1, "capabilities": {"tools": tools}}));
    }

    /// Starts the front thread, which loads the model, makes the shared state and starts
    /// the indexer. Nothing here waits for any of it.
    fn start(&mut self, cfg: ExtConfig, boot: Boot) -> Option<Live> {
        let (calls, calls_rx) = mpsc::channel::<FrontCall>();
        let (replies_tx, replies) = mpsc::channel::<FrontReply>();
        let (jobs_to, jobs_from) = mpsc::channel::<SyncSender<Job>>();
        let (front_alive, front_done) = mpsc::channel::<String>();
        let (indexer_alive, indexer_done) = mpsc::channel::<()>();
        let mode = self.mode.clone();
        let stop = self.stop.clone();
        let log = self.log.clone();
        let status = self.status.clone();

        let spawned = thread::Builder::new().name("axel-front".to_string()).spawn(move || {
            set_thread_log(log.clone());
            let panics = boot.status.clone();
            let ran = catch_unwind(AssertUnwindSafe(move || {
                let Boot { embedder, clock, test, status } = boot;
                if test.panic_front_start {
                    panic!("test switch: panic_front_start");
                }
                if test.hang_front_start {
                    while !stop.load(Ordering::SeqCst) {
                        thread::sleep(Duration::from_millis(20));
                    }
                    return;
                }
                let embedder = embedder.or_else(|| load_embedder(&cfg));
                let boot = Boot { embedder, clock, test, status };
                let (st, jobs_rx, vec_tx) = front::new_state(cfg.clone(), boot, mode);
                let ctx = IndexerCtx {
                    cfg,
                    conn: None,
                    signer: None,
                    embed: st.embed.clone(),
                    vec_tx,
                    jobs: st.jobs.clone(),
                    status: st.status.clone(),
                    mode: st.mode.clone(),
                    last_run: HashMap::new(),
                    open_tried: None,
                    vectors_sent: None,
                    stop: stop.clone(),
                    brain_file: None,
                    clock: st.clock.clone(),
                    test: st.test.clone(),
                    writing: st.writing.clone(),
                };
                let _ = jobs_to.send(st.jobs.clone());
                start_indexer(ctx, jobs_rx, stop, log, indexer_alive);
                front::run(st, calls_rx, replies_tx);
            }));
            // The place of a panic outside a call, for the rpc thread; then the channel
            // closes, which is how it knows this thread has ended.
            if ran.is_err() {
                panics.panics.fetch_add(1, Ordering::Relaxed);
                if let Some(at) = take_last_panic() {
                    let _ = front_alive.send(at);
                }
            }
        });
        if spawned.is_err() {
            self.front_gone = true;
            degrade(&status, &self.mode, DegradedReason::FrontStopped, Some(here(line!())));
            self.log("front thread not started");
            return None;
        }
        Some(Live { calls, replies, jobs: None, jobs_from, front_done, indexer_done })
    }

    // ── hooks ───────────────────────────────────────────────────────────────

    fn on_hook(&mut self, id: Option<Value>, params: &Value) -> Option<ServeEnd> {
        let kind = params.get("kind").and_then(Value::as_str).unwrap_or("");
        let session_id = str_field(params, "session_id");
        let cwd = str_field(params, "cwd");
        let data = params.get("data").cloned().unwrap_or(Value::Null);

        // Notify-only events: answered before any work is done, never forwarded.
        match kind {
            "on_context_archived" => {
                self.answer(id, continue_result());
                self.archived(session_id, cwd, &data);
                return None;
            }
            "on_memory_forgotten" => {
                self.answer(id, continue_result());
                self.forgotten(session_id, &data);
                return None;
            }
            _ => {}
        }

        let before_message = kind == "before_message";
        let wait = if before_message { self.budget_ms } else { HOOK_WAIT_MS };
        // The deadline this thread waits to, the brain's open included.
        let deadline = Instant::now() + Duration::from_millis(wait);
        // Outside `Ready` every hook is `continue`.
        // RULING: v2 real-binaries round, item 1: a hook waits for an open in progress, to
        // its own deadline.
        if !self.ready_by(deadline) {
            self.answer(id, continue_result());
            return None;
        }
        let recall_allowed = self.open_until.is_none();
        let label = hook_label(kind);
        let ev = HookEvent {
            kind: kind.to_string(),
            session_id: session_id.clone(),
            cwd,
            message: params.get("message").and_then(Value::as_str).map(str::to_string),
            data,
            received_ms: (self.clock)(),
        };
        // The front thread is handed the deadline this thread waits to: recall works to it,
        // whatever the `Carry` took before it (review of 15/16, the hook's deadline).
        match self.forward(FrontWork::Hook { ev, recall_allowed, deadline }, deadline, label) {
            Got::Answer(FrontAnswer::Hook(out)) => {
                if before_message {
                    self.misses = 0;
                }
                self.answer_hook(id, out);
                None
            }
            // A handler that panicked: the hook is `continue` and the next call works.
            Got::Answer(_) => {
                self.answer(id, continue_result());
                None
            }
            Got::Late => {
                self.answer(id, continue_result());
                self.missed(before_message);
                None
            }
            Got::NotForwarded => {
                self.answer(id, continue_result());
                // RULING: wave3-c12b ruling 3. Recall is not what is slow when the call the
                // front thread is inside is a tool or another hook: such a miss is counted,
                // and does not count toward the breaker.
                let behind_recall = self.busy.as_ref().is_some_and(|b| b.method == BEFORE_MESSAGE_LABEL);
                self.missed(before_message && behind_recall);
                self.skipped(session_id);
                None
            }
            Got::Dead => {
                self.answer(id, continue_result());
                Some(ServeEnd::FrontStuck)
            }
        }
    }

    /// A hook the pump answered itself. Three `before_message` in a row that recall made
    /// late (`for_breaker`) open the breaker.
    // SPEC: 4963 counts in `hook_misses` hooks answered "at a deadline, or the breaker
    // open", and 509 says a call is still forwarded while the breaker is open. Such a call
    // is forwarded with `recall: false` and counted only if the pump has to answer it.
    fn missed(&mut self, for_breaker: bool) {
        self.status.hook_misses.fetch_add(1, Ordering::Relaxed);
        if !for_breaker {
            return;
        }
        self.misses += 1;
        if self.misses >= TRIP_AFTER_MISSES && self.open_until.is_none() {
            let open_for = self.test.breaker_open_ms.map_or(TRIP_OPEN_FOR, Duration::from_millis);
            self.open_until = Some(Instant::now() + open_for);
            self.status.breaker_trips.fetch_add(1, Ordering::Relaxed);
            self.status.breaker_open.store(true, Ordering::Relaxed);
            self.log("breaker open");
        }
    }

    fn close_breaker_when_due(&mut self) {
        if self.open_until.is_some_and(|until| Instant::now() >= until) {
            self.open_until = None;
            self.misses = 0;
            self.status.breaker_open.store(false, Ordering::Relaxed);
            self.log("breaker closed");
        }
    }

    /// The chat of a hook, or of a `forget` call, that was answered without being forwarded
    /// (R-9e): its turn must not be paired with what the front thread still holds.
    fn skipped(&mut self, session_id: Option<String>) {
        let Some(session_id) = session_id else { return };
        if !self.carry.skipped.contains(&session_id) {
            self.carry.skipped.push(session_id);
            self.status.sessions_dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn archived(&mut self, session_id: Option<String>, cwd: Option<String>, data: &Value) {
        let archive_id = data.get("archive_id").and_then(Value::as_str).unwrap_or("");
        let project_key = data.get("project_key").and_then(Value::as_str).unwrap_or("");
        if !is_hex32(archive_id) || !is_project_key(project_key) {
            self.log("on_context_archived: not valid, no job");
            return;
        }
        self.send_job(Job::Archive {
            archive_id: archive_id.to_string(),
            path: str_field(data, "path"),
            project_key: project_key.to_string(),
            cwd,
            session_id,
            received_ms: (self.clock)(),
            window: positive(data.get("window")),
            part: positive(data.get("part")),
        });
    }

    fn forgotten(&mut self, session_id: Option<String>, data: &Value) {
        // The chat in which core's `memory_forget` ran is not to be learned (R3-2c),
        // whatever becomes of the job.
        if let Some(session_id) = session_id {
            if !self.carry.forgot.contains(&session_id) {
                self.carry.forgot.push(session_id);
            }
        }
        let id = data.get("id").and_then(Value::as_str).unwrap_or("");
        let kind = data.get("kind").and_then(Value::as_str).unwrap_or("");
        let project_key = data.get("project_key").and_then(Value::as_str).unwrap_or("");
        let id_ok = match kind {
            "archive" => is_hex32(id),
            "note" => !id.is_empty() && id.len() <= 256,
            _ => false,
        };
        if !id_ok || !is_project_key(project_key) {
            self.log("on_memory_forgotten: not valid, no job");
            return;
        }
        self.send_job(Job::Forgotten {
            id: id.to_string(),
            kind: kind.to_string(),
            project_key: project_key.to_string(),
        });
    }

    /// `try_send`: a full queue drops the job and counts it; a disconnected queue means the
    /// indexer thread has ended.
    fn send_job(&mut self, job: Job) {
        // RULING: v2 real-binaries round 2, item 2: a forget job dropped asks the indexer's
        // next tick for the cheap forget pass.
        let forget = matches!(job, Job::Forgotten { .. });
        let before = self.status.jobs_dropped.load(Ordering::Relaxed);
        self.queue_job(job);
        if forget && self.status.jobs_dropped.load(Ordering::Relaxed) != before {
            self.status.forget_pass_due.store(true, Ordering::SeqCst);
        }
    }

    fn queue_job(&mut self, job: Job) {
        let Some(live) = self.live.as_mut() else {
            self.status.jobs_dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if live.jobs.is_none() {
            live.jobs = live.jobs_from.try_recv().ok();
        }
        let Some(jobs) = live.jobs.as_ref() else {
            // The front thread has not made the queue yet; the scan finds the work.
            self.status.jobs_dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        match jobs.try_send(job) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                self.status.jobs_dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {
                if !self.stop.load(Ordering::SeqCst) {
                    degrade(&self.status, &self.mode, DegradedReason::IndexerStopped, Some(here(line!())));
                }
            }
        }
    }

    // ── tools ───────────────────────────────────────────────────────────────

    fn on_tool(&mut self, id: Option<Value>, params: &Value) -> Option<ServeEnd> {
        let Some(name) = str_field(params, "name") else {
            self.answer_tool(id, Err(ToolError::Invalid("tool.call needs a name".to_string())));
            return None;
        };
        // RULING: v2 real-binaries round, item 1: a tool waits up to its own wait for an
        // open in progress, then has that wait again for its answer.
        if !self.ready_by(Instant::now() + Duration::from_millis(TOOL_WAIT_MS)) {
            let reason = unavailable_reason(&self.status, &self.mode).to_string();
            self.answer_tool(id, Err(ToolError::Unavailable(reason)));
            return None;
        }
        let caller = Caller {
            session_id: str_field(params, "session_id"),
            cwd: str_field(params, "cwd"),
            // Only a JSON object is a caller: serde would also take an array of strings as
            // an `Author`, field by position, and the forum would then post under it.
            author: params
                .get("caller")
                .filter(|c| c.is_object())
                .cloned()
                .and_then(|c| serde_json::from_value(c).ok()),
        };
        // The host sends the name the extension declared (`forget`), never the name a model
        // sees (`axel_forget`): core's `tools/extension.rs` keeps `spec.name` for the call.
        let forget = name == "forget";
        let session_id = caller.session_id.clone();
        let input = params.get("input").cloned().unwrap_or(Value::Null);
        let work = FrontWork::Tool { name, input, caller };
        match self.forward(work, Instant::now() + Duration::from_millis(TOOL_WAIT_MS), "tool.call") {
            Got::Answer(FrontAnswer::Tool(out)) => {
                self.answer_tool(id, out);
                None
            }
            // A handler that panicked: an error, and the next call works.
            Got::Answer(_) => {
                if let Some(id) = id {
                    self.send(&Response::error(id, INTERNAL_ERROR, "internal error"));
                }
                None
            }
            Got::Late => {
                self.answer_tool(id, Err(ToolError::Busy));
                None
            }
            Got::NotForwarded => {
                self.answer_tool(id, Err(ToolError::Busy));
                if forget {
                    self.skipped(session_id);
                }
                None
            }
            Got::Dead => {
                self.answer_tool(id, Err(ToolError::Busy));
                Some(ServeEnd::FrontStuck)
            }
        }
    }

    // ── the front thread ────────────────────────────────────────────────────

    /// One call at a time. While the front thread is inside a call whose deadline has
    /// passed, nothing is forwarded; its late answer is discarded and forwarding resumes.
    fn forward(&mut self, work: FrontWork, deadline: Instant, method: &'static str) -> Got {
        if self.front_is_dead() {
            return Got::Dead;
        }
        if self.busy.is_some() {
            return Got::NotForwarded;
        }
        let Some(live) = self.live.as_ref() else { return Got::NotForwarded };
        self.seq += 1;
        let seq = self.seq;
        let carry = std::mem::take(&mut self.carry);
        let started = Instant::now();
        if live.calls.send(FrontCall { seq, carry, work }).is_err() {
            self.log("front thread ended");
            return Got::Dead;
        }
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match live.replies.recv_timeout(left) {
                Ok(reply) if reply.seq == seq => return Got::Answer(reply.answer),
                Ok(_) => {}
                Err(RecvTimeoutError::Timeout) => {
                    self.busy = Some(Busy { since: started, method, seq });
                    return Got::Late;
                }
                Err(RecvTimeoutError::Disconnected) => {
                    self.log("front thread ended");
                    return Got::Dead;
                }
            }
        }
    }

    /// Takes a late answer off the channel. True when the front thread has ended, or has
    /// been inside one call for longer than the limit (30 s): the process then ends.
    fn front_is_dead(&mut self) -> bool {
        let (Some(busy), Some(live)) = (self.busy.as_ref(), self.live.as_ref()) else { return false };
        loop {
            match live.replies.try_recv() {
                Ok(reply) if reply.seq == busy.seq => {
                    let ms = busy.since.elapsed().as_millis();
                    let method = busy.method;
                    self.busy = None;
                    self.log(&format!("front late in {method}: {ms} ms"));
                    return false;
                }
                Ok(_) => {}
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.log("front thread ended");
                    return true;
                }
            }
        }
        let limit = Duration::from_millis(self.test.front_hang_limit_ms.unwrap_or(FRONT_HANG_MS));
        if busy.since.elapsed() >= limit {
            let method = busy.method;
            self.log(&format!("front stuck in {method}"));
            return true;
        }
        false
    }

    // ── the end ─────────────────────────────────────────────────────────────

    /// Sets the stop flag, tells the threads, and waits a bounded time for them: work in a
    /// transaction is rolled back and found again at the next start.
    fn finish(&mut self, end: ServeEnd) -> ServeEnd {
        self.stop.store(true, Ordering::SeqCst);
        self.log(&format!("serve ends: {end:?}"));
        // A front thread that ended the process: the file that stays says so.
        if end == ServeEnd::FrontStuck {
            degrade(&self.status, &self.mode, DegradedReason::FrontStopped, None);
            self.status_file_when_due();
        }
        // A clean end leaves no status file behind.
        if let (Some(path), ServeEnd::Shutdown | ServeEnd::InputClosed) = (self.status_file.as_ref(), end) {
            let _ = std::fs::remove_file(path);
        }
        let Some(mut live) = self.live.take() else { return end };
        if live.jobs.is_none() {
            live.jobs = live.jobs_from.try_recv().ok();
        }
        if let Some(jobs) = live.jobs.take() {
            let _ = jobs.try_send(Job::Shutdown);
        }
        let Live { calls, front_done, indexer_done, .. } = live;
        drop(calls);
        let deadline = Instant::now() + EXIT_WAIT;
        let _ = indexer_done.recv_timeout(deadline.saturating_duration_since(Instant::now()));
        if end != ServeEnd::FrontStuck {
            let _ = front_done.recv_timeout(deadline.saturating_duration_since(Instant::now()));
        }
        end
    }
}

fn here(line: u32) -> String {
    format!("{}:{}", file!(), line)
}

/// The model the config names, when `Boot` brought none. Nothing is downloaded: a folder
/// without the files means words only.
fn load_embedder(cfg: &ExtConfig) -> Option<Embedder> {
    match cfg.embedder {
        EmbedderKind::Off => None,
        EmbedderKind::Hash => Some(Embedder::hashing()),
        EmbedderKind::Onnx => match Embedder::from_model_dir(&cfg.model_dir, 2, true) {
            Ok(embedder) => Some(embedder),
            Err(_) => {
                log_line("model not loaded: words only");
                None
            }
        },
    }
}

/// The indexer thread. Its own closure sets `Degraded(IndexerStopped at <file:line>)` when
/// `indexer::run` returns or unwinds for any reason but `shutdown` (R-5).
fn start_indexer(
    ctx: IndexerCtx,
    jobs_rx: Receiver<Job>,
    stop: Arc<AtomicBool>,
    log: Option<Log>,
    alive: Sender<()>,
) {
    let status = ctx.status.clone();
    let mode = ctx.mode.clone();
    let (status_2, mode_2) = (status.clone(), mode.clone());
    let spawned = thread::Builder::new().name("axel-indexer".to_string()).spawn(move || {
        let _alive = alive;
        set_thread_log(log);
        let ended = catch_unwind(AssertUnwindSafe(|| indexer::run(ctx, jobs_rx)));
        if stop.load(Ordering::SeqCst) {
            return;
        }
        let at = match ended {
            Ok(()) => here(line!()),
            Err(_) => take_last_panic().unwrap_or_else(|| here(line!())),
        };
        log_line(&format!("indexer stopped at {at}"));
        degrade(&status, &mode, DegradedReason::IndexerStopped, Some(at));
    });
    if spawned.is_err() {
        log_line("indexer thread not started");
        degrade(&status_2, &mode_2, DegradedReason::IndexerStopped, Some(here(line!())));
    }
}
