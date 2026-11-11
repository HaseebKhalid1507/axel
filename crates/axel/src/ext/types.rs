//! The types threads and commits share. Written whole in commit 3; nobody edits this file
//! afterwards (one more `Status` counter goes to the foreman, who adds it between waves).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use axel_memkoshi::security::MemorySigner;
use rusqlite::Connection;
use velocirag::embedder::Embedder;

use crate::capture::{CaptureState, CompletedTurn};
use crate::embed::EmbedService;
use crate::ext::config::ExtConfig;
use crate::recall::RecallTrace;
use crate::vindex::{ExactIndex, VecDelta, VecKey};

// SPEC: 4668-4738 give a derive only for Status, DegradedReason and TestHooks. The derives
// on the other plain-data types here are commit 3's addition (they change no signature, and
// nobody may add them later): without them a test cannot `assert_eq!` a `HookOut`, a
// `ServeEnd` or a `ToolError`.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Starting = 0,
    Ready = 1,
    Degraded = 2,
}

impl Mode {
    /// For readers of the mode's `AtomicU8`. A value that names no mode is `None`.
    pub fn from_u8(v: u8) -> Option<Mode> {
        match v {
            0 => Some(Mode::Starting),
            1 => Some(Mode::Ready),
            2 => Some(Mode::Degraded),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Caller {
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    pub author: Option<crate::forum::Author>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HookEvent {
    pub kind: String,
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    pub message: Option<String>,
    pub data: serde_json::Value,
    pub received_ms: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HookOut {
    Continue,
    Inject(String),
}

/// The whole list (4.1, build 2). Every field is `pub` and written with
/// `Ordering::Relaxed`; `indexer_start::write_status` copies them into the row.
/// Counts that are read from the brain (archives, hidden rows, forget turns, forum posts)
/// are NOT fields: `status` queries them.
#[derive(Default)]
pub struct Status {
    // rpc thread
    pub breaker_open: AtomicBool,
    pub breaker_trips: AtomicU64,
    /// Hooks the pump answered itself: a deadline, or the breaker open.
    pub hook_misses: AtomicU64,
    pub sessions_dropped: AtomicU64,
    // any thread
    /// `try_send` on a full queue.
    pub jobs_dropped: AtomicU64,
    pub tokens_blanked: AtomicU64,
    pub turns_withheld: AtomicU64,
    /// Turns the capture buffer let go without storing them and without core's rule being
    /// the reason: a buffer evicted, a reply with no `has_tool_use`, a buffer cleared.
    pub turns_dropped: AtomicU64,
    pub write_errors: AtomicU64,
    pub panics: AtomicU64,
    // indexer thread
    pub vectors_waiting: AtomicU64,
    /// Note files that needed a vector and did not get one in the last sync of the note
    /// folders (`docsync::tick`).
    pub docs_waiting: AtomicU64,
    pub model_loaded: AtomicBool,
    /// Memories with no `memory_ext` row whose `id` or `content` cannot be read as text:
    /// what `open_v2` and the last adoption counted (wave 4, R-12b item 2).
    pub unreadable_memories: AtomicU64,
    /// 0 pending, 1 running, 2 done.
    pub scan: AtomicU8,
    /// When this process answered `initialize`.
    pub started_ms: AtomicI64,
    pub status_written_ms: AtomicI64,
    // why the mode is Degraded: whoever sets the mode stores the reason FIRST, then the mode
    /// A `DegradedReason`.
    pub reason: AtomicU8,
    /// "file:line", kept with the reason it belongs to: one place per `DegradedReason`, the
    /// first writer of each wins. Read it with `reason_place` (wave 4, R-12b item 6).
    pub reason_at: [OnceLock<String>; DegradedReason::COUNT],
    /// Set once the brain file was seen replaced or gone; never cleared, whatever reason
    /// comes later (v2 r5 round 2, item 2).
    pub brain_replaced: AtomicBool,
}

impl Status {
    /// True once this process has seen its brain file replaced or gone.
    // RULING: v2 r5 round 2, item 2.
    pub fn brain_changed(&self) -> bool {
        self.brain_replaced.load(Ordering::SeqCst)
            || DegradedReason::from_u8(self.reason.load(Ordering::SeqCst)) == DegradedReason::BrainChanged
    }

    /// Records where `reason` was set; the first place given for a reason stays.
    pub fn set_reason_place(&self, reason: DegradedReason, at: String) {
        if let Some(slot) = self.reason_at.get(reason as usize) {
            let _ = slot.set(at);
        }
    }

    /// The place recorded for the reason that stands now; `None` when that reason has none,
    /// whatever an earlier reason recorded.
    pub fn reason_place(&self) -> Option<&str> {
        let reason = self.reason.load(Ordering::SeqCst) as usize;
        self.reason_at.get(reason)?.get().map(String::as_str)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DegradedReason {
    None = 0,
    NoBrain = 1,
    V1Brain = 2,
    BrainTooNew = 3,
    BrainDamaged = 4,
    BrainChanged = 5,
    BrainNotReady = 6,
    OtherBase = 7,
    IndexerStopped = 8,
    /// The front thread died or hung before the first `Ready` (wave 3, 12b ruling 2).
    FrontStopped = 9,
    /// A v2 brain of a shape before round 3 (`brain_v2::check_shape`; v2 c18b round 5).
    BrainOldShape = 10,
}

impl DegradedReason {
    /// How many reasons there are: `Status.reason_at` has one place for each.
    pub const COUNT: usize = 11;

    /// Every reason, in the order of its value (checked when the crate compiles, below).
    pub const ALL: [DegradedReason; Self::COUNT] = [
        Self::None,
        Self::NoBrain,
        Self::V1Brain,
        Self::BrainTooNew,
        Self::BrainDamaged,
        Self::BrainChanged,
        Self::BrainNotReady,
        Self::OtherBase,
        Self::IndexerStopped,
        Self::FrontStopped,
        Self::BrainOldShape,
    ];

    /// A value that names no reason reads as `None`.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::NoBrain,
            2 => Self::V1Brain,
            3 => Self::BrainTooNew,
            4 => Self::BrainDamaged,
            5 => Self::BrainChanged,
            6 => Self::BrainNotReady,
            7 => Self::OtherBase,
            8 => Self::IndexerStopped,
            9 => Self::FrontStopped,
            10 => Self::BrainOldShape,
            _ => Self::None,
        }
    }

    // SPEC: 4696 says `as_str` is final in commit 3 and no line gives the strings. These are
    // the variant names as the document writes them (`Degraded(V1Brain)`, 888, 3798-3809).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::NoBrain => "NoBrain",
            Self::V1Brain => "V1Brain",
            Self::BrainTooNew => "BrainTooNew",
            Self::BrainDamaged => "BrainDamaged",
            Self::BrainChanged => "BrainChanged",
            Self::BrainNotReady => "BrainNotReady",
            Self::OtherBase => "OtherBase",
            Self::IndexerStopped => "IndexerStopped",
            Self::FrontStopped => "FrontStopped",
            Self::BrainOldShape => "BrainOldShape",
        }
    }
}

// `ALL` holds `COUNT` reasons, each at its own value: the values are exactly `0..COUNT`,
// so every reason in it has its place in `Status.reason_at`.
const _: () = {
    let mut i = 0;
    while i < DegradedReason::COUNT {
        assert!(DegradedReason::ALL[i] as usize == i, "DegradedReason::ALL is not in the order of the values");
        i += 1;
    }
};

// No wildcard: a new reason stops the build here. Add it to `ALL` and raise `COUNT`.
const fn _every_reason_is_in_all(r: DegradedReason) {
    match r {
        DegradedReason::None
        | DegradedReason::NoBrain
        | DegradedReason::V1Brain
        | DegradedReason::BrainTooNew
        | DegradedReason::BrainDamaged
        | DegradedReason::BrainChanged
        | DegradedReason::BrainNotReady
        | DegradedReason::OtherBase
        | DegradedReason::IndexerStopped
        | DegradedReason::FrontStopped
        | DegradedReason::BrainOldShape => {}
    }
}

/// What `mark_forgotten_turn` is given for one session of `Carry.forgot`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForgetMark {
    pub session_id: String,
    pub from_ms: i64,
    pub to_ms: i64,
    pub turn_key: Option<String>,
}

pub struct FrontState {
    pub cfg: ExtConfig,
    pub conn: Option<Connection>,
    pub signer: Option<MemorySigner>,
    pub index: ExactIndex,
    pub vec_rx: Receiver<VecDelta>,
    pub embed: EmbedService,
    pub jobs: SyncSender<Job>,
    pub capture: CaptureState,
    pub status: Arc<Status>,
    pub mode: Arc<AtomicU8>,
    pub forum_on: bool,
    pub filed: HashMap<String, (i64, Option<Vec<String>>)>,
    /// Forget marks that could not get the write lock, oldest first (wave 3, 12b ruling 7).
    pub forget_marks: Vec<ForgetMark>,
    pub clock: Clock,
    pub test: Arc<TestHooks>,
    /// Section 7, "the indexer stands aside".
    pub writing: Arc<AtomicBool>,
}

pub struct IndexerCtx {
    pub cfg: ExtConfig,
    pub conn: Option<Connection>,
    pub signer: Option<MemorySigner>,
    pub embed: EmbedService,
    pub vec_tx: Sender<VecDelta>,
    pub jobs: SyncSender<Job>,
    pub status: Arc<Status>,
    pub mode: Arc<AtomicU8>,
    /// When each call of the tick last ran, on a monotonic clock (wave 3, 12b ruling 10).
    pub last_run: HashMap<&'static str, Instant>,
    /// When `ensure_open` last tried the brain.
    pub open_tried: Option<Instant>,
    /// What the front thread has been sent: count and highest id of the visible embedded
    /// documents, then of the memory rows. `None` until a `Replace` has gone.
    pub vectors_sent: Option<[i64; 4]>,
    /// Set by `serve` before it tells the threads to end; the loop looks at it every turn.
    pub stop: Arc<AtomicBool>,
    /// Device and inode of the brain file when it was opened (wave 3, 12b ruling 5).
    pub brain_file: Option<(u64, u64)>,
    pub clock: Clock,
    pub test: Arc<TestHooks>,
    pub writing: Arc<AtomicBool>,
}

// RULING: v2 r5 round 2, item 2. The thread's connection is given up through
// `brain_v2::release`: one to a brain file seen replaced is never closed through SQLite.
impl Drop for IndexerCtx {
    fn drop(&mut self) {
        crate::brain_v2::release(self.conn.take(), &self.status);
    }
}

// RULING: v2 r5 round 2, item 2.
impl Drop for FrontState {
    fn drop(&mut self) {
        crate::brain_v2::release(self.conn.take(), &self.status);
    }
}

#[derive(Clone, Debug)]
pub enum Job {
    Tick,
    Shutdown,
    /// 4.1: `window` and `part` are `data.window` and `data.part` (section 2.4).
    Archive {
        archive_id: String,
        path: Option<String>,
        project_key: String,
        cwd: Option<String>,
        session_id: Option<String>,
        received_ms: i64,
        window: Option<i64>,
        part: Option<i64>,
    },
    Forgotten {
        id: String,
        kind: String,
        project_key: String,
    },
    Capture(CompletedTurn),
    LogRecall(Box<RecallTrace>),
    /// `VecKey::Mem` only (B-2e). `out` is what the embed thread was given with the texts:
    /// which row each number stood for (wave 4, R-12b item 4). A job without it writes nothing.
    Embedded {
        model: String,
        vectors: Vec<(VecKey, Vec<f32>)>,
        out: Option<Arc<crate::embed::BatchOut>>,
    },
}

pub struct Boot {
    pub embedder: Option<Embedder>,
    pub clock: Clock,
    pub test: Arc<TestHooks>,
    /// A test keeps a clone and reads the counters.
    pub status: Arc<Status>,
}

/// B-2a. Run-time switches, all off by default: tests under `tests/` build the library
/// without `cfg(test)`, so these cannot be cfg switches.
#[derive(Default, Clone, Debug)]
pub struct TestHooks {
    pub front_sleep_ms: u64,
    pub indexer_sleep_ms: u64,
    pub panic_tool: Option<String>,
    pub panic_archive_id: Option<String>,
    pub panic_embed: bool,
    pub panic_on_forgotten_job: bool,
    pub log_limit_bytes: Option<u64>,
    /// The indexer's loop returns after this many jobs.
    pub indexer_stop_after_jobs: Option<u64>,
    /// Replaces the 30 s of section 7.
    pub front_hang_limit_ms: Option<u64>,
    /// The front thread panics before it has made anything.
    pub panic_front_start: bool,
    /// The front thread waits for the end of `serve` before it has made anything.
    pub hang_front_start: bool,
    /// Replaces the 60 s a front thread has to hand over the indexer's queue.
    pub front_start_limit_ms: Option<u64>,
    /// Replaces the 30 s between two status ticks of the indexer.
    pub status_tick_ms: Option<u64>,
    /// The embed thread sleeps this long inside every batch, before the model is called.
    pub embed_sleep_ms: u64,
    /// Replaces the 60 s after which a batch still running is a hang.
    pub embed_stuck_limit_ms: Option<u64>,
    /// Replaces the 60 s between two attempts to open the brain.
    pub open_retry_ms: Option<u64>,
    /// Replaces the 10 minutes the breaker stays open.
    pub breaker_open_ms: Option<u64>,
    /// The indexer sleeps this long before each attempt to open the brain.
    pub open_sleep_ms: u64,
}

/// What `serve` returns; `run` exits 0, 0, 70.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ServeEnd {
    Shutdown,
    InputClosed,
    FrontStuck,
}

/// B-2b.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainArg {
    pub path: PathBuf,
    /// `--brain` came from the `AXEL_BRAIN` environment variable.
    pub from_env: bool,
    /// `--brain` was written on the command line (not the environment, not the default
    /// path). `synaps-export` refuses a brain that was not.
    pub on_command_line: bool,
}

/// B3-3c; section 5 says what the host is sent for each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolError {
    /// Unknown name, non-object input, wrong type: JSON-RPC -32602.
    Invalid(String),
    /// The mode is not Ready.
    Unavailable(String),
    /// The write lock was not free in time; an exact retry is safe.
    Busy,
    /// The tool's own refusal: a worker, the cap, an id it may not take.
    Refused(String),
}

/// The rpc thread hands a `Carry` to the front thread with the next call it forwards.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Carry {
    /// Sessions whose hook the pump answered itself (R-9e).
    pub skipped: Vec<String>,
    /// Sessions named by `on_memory_forgotten` (R3-2c).
    pub forgot: Vec<String>,
}

/// Milliseconds.
pub type Clock = Arc<dyn Fn() -> i64 + Send + Sync>;
