//! The front thread's loop: it answers calls (hooks and tools) on its own connection and
//! owns the vector array. It never does background work.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TrySendError};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::brain_v2::{self, Role};
use crate::capture::CaptureState;
use crate::embed::EmbedService;
use crate::ext::config::ExtConfig;
use crate::ext::server::{self, degrade, log_line, JOB_QUEUE};
use crate::ext::types::{
    Boot, Caller, Carry, DegradedReason, FrontState, HookEvent, HookOut, Job, Mode, ToolError,
};
use crate::ext::{hooks, tools};
use crate::r8::EMBEDDING_DIM;
use crate::recall::View;
use crate::vindex::{ExactIndex, VecDelta};
use crate::{index_archives, project};

/// How long the loop waits for a call before it looks after its own state.
const IDLE: Duration = Duration::from_millis(250);

/// What the rpc thread hands over: one call, and with it the `Carry` it has collected.
pub struct FrontCall {
    pub seq: u64,
    pub carry: Carry,
    pub work: FrontWork,
}

pub enum FrontWork {
    Hook { ev: HookEvent, recall_allowed: bool },
    Tool { name: String, input: Value, caller: Caller },
}

pub struct FrontReply {
    pub seq: u64,
    pub answer: FrontAnswer,
}

pub enum FrontAnswer {
    Hook(HookOut),
    Tool(Result<String, ToolError>),
    /// The handler panicked: the hook is `continue`, the tool an error.
    Panicked,
}

/// The state without a connection, sharing `mode` with whoever made it.
pub(crate) fn new_state(
    cfg: ExtConfig,
    boot: Boot,
    mode: Arc<AtomicU8>,
) -> (FrontState, Receiver<Job>, Sender<VecDelta>) {
    let (jobs, jobs_rx) = mpsc::sync_channel::<Job>(JOB_QUEUE);
    let (vec_tx, vec_rx) = mpsc::channel::<VecDelta>();
    let embed = EmbedService::start(boot.embedder, jobs.clone(), boot.test.clone());
    let forum_on = cfg.forum_on();
    let st = FrontState {
        cfg,
        conn: None,
        signer: None,
        index: ExactIndex::new(EMBEDDING_DIM),
        vec_rx,
        embed,
        jobs,
        capture: CaptureState::default(),
        status: boot.status,
        mode,
        forum_on,
        filed: HashMap::new(),
        clock: boot.clock,
        test: boot.test,
        writing: Arc::new(AtomicBool::new(false)),
    };
    (st, jobs_rx, vec_tx)
}

/// The one constructor of a `FrontState` (B-7.1). It hands back the other ends of the
/// channels it makes (B3-3a): a state whose job receiver was dropped would turn the mode to
/// `Degraded(IndexerStopped)` at its first `try_send`.
///
/// When `cfg.brain` is a version 2 brain the state is connected to it and its mode is
/// `Ready`; otherwise it has no connection and the mode is `Starting`. It never creates or
/// migrates a brain: that is the indexer's (`indexer_start::open_ctx`).
pub fn open_state(cfg: ExtConfig, boot: Boot) -> (FrontState, Receiver<Job>, Sender<VecDelta>) {
    let mode = Arc::new(AtomicU8::new(Mode::Starting as u8));
    let (mut st, jobs_rx, vec_tx) = new_state(cfg, boot, mode);
    if connect(&mut st) {
        st.mode.store(Mode::Ready as u8, Ordering::SeqCst);
    }
    (st, jobs_rx, vec_tx)
}

/// `brain_v2::connect` (pragmas only), then the signer. True when the state has a
/// connection.
pub fn connect(st: &mut FrontState) -> bool {
    if st.conn.is_some() {
        return true;
    }
    match brain_v2::connect(&st.cfg.brain, Role::Front) {
        Ok(conn) => {
            st.signer = brain_v2::signer(&conn).ok().flatten();
            st.conn = Some(conn);
            true
        }
        Err(_) => false,
    }
}

/// Who may see what (section 5's table). Nobody is refused here: a worker gets a narrower
/// view, and a caller with no usable `cwd` gets `project: None`. A caller is a chat when
/// the call has a `session_id` and a worker when it has none.
pub fn view_for(caller: &Caller, cfg: &ExtConfig) -> View {
    let project = project::resolve(caller.cwd.as_deref());
    match &caller.session_id {
        Some(session_id) => View { project, global: true, documents: true, own_session: Some(session_id.clone()) },
        None => View {
            project,
            global: cfg.worker_scope_all,
            documents: cfg.worker_scope_all,
            own_session: None,
        },
    }
}

/// `try_send` to the indexer, for the handlers of this thread: a full queue drops the job
/// and counts it; a disconnected queue means the indexer thread has ended, and the mode
/// becomes `Degraded(IndexerStopped)`. True when the job was queued.
pub fn send_job(st: &FrontState, job: Job) -> bool {
    match st.jobs.try_send(job) {
        Ok(()) => true,
        Err(TrySendError::Full(_)) => {
            st.status.jobs_dropped.fetch_add(1, Ordering::Relaxed);
            false
        }
        Err(TrySendError::Disconnected(_)) => {
            degrade(
                &st.status,
                &st.mode,
                DegradedReason::IndexerStopped,
                Some(format!("{}:{}", file!(), line!())),
            );
            false
        }
    }
}

/// The loop. It ends when the rpc thread drops its end of `calls`.
pub fn run(mut st: FrontState, calls: Receiver<FrontCall>, replies: Sender<FrontReply>) {
    loop {
        match calls.recv_timeout(IDLE) {
            Ok(call) => {
                if st.test.front_sleep_ms > 0 {
                    std::thread::sleep(Duration::from_millis(st.test.front_sleep_ms));
                }
                let seq = call.seq;
                let answer = match catch_unwind(AssertUnwindSafe(|| handle(&mut st, call))) {
                    Ok(answer) => answer,
                    Err(_) => {
                        st.status.panics.fetch_add(1, Ordering::Relaxed);
                        // Whatever the handler had open is not kept. (A write guard has
                        // cleared the `writing` flag while unwinding.)
                        if let Some(conn) = st.conn.as_ref() {
                            let _ = conn.execute_batch("ROLLBACK");
                        }
                        FrontAnswer::Panicked
                    }
                };
                if replies.send(FrontReply { seq, answer }).is_err() {
                    return;
                }
            }
            Err(RecvTimeoutError::Timeout) => idle(&mut st),
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn is_ready(st: &FrontState) -> bool {
    st.mode.load(Ordering::SeqCst) == Mode::Ready as u8
}

/// Between calls: connect once the indexer reports the brain open, and take the vectors
/// that have arrived, so that a search finds a short queue.
fn idle(st: &mut FrontState) {
    if is_ready(st) {
        connect(st);
    }
    while let Ok(delta) = st.vec_rx.try_recv() {
        st.index.apply(delta);
    }
}

/// One call. The `Carry` first, before anything else.
pub fn handle(st: &mut FrontState, call: FrontCall) -> FrontAnswer {
    let ready = is_ready(st) && connect(st);
    apply_carry(st, &call.carry);
    match call.work {
        FrontWork::Hook { ev, recall_allowed } => {
            if !ready {
                return FrontAnswer::Hook(HookOut::Continue);
            }
            FrontAnswer::Hook(hooks::handle(st, &ev, recall_allowed))
        }
        FrontWork::Tool { name, input, caller } => {
            if !ready {
                let reason = match server::unavailable_reason(&st.status, &st.mode) {
                    // `Ready`, and no connection could be made.
                    "starting" if is_ready(st) => DegradedReason::BrainNotReady.as_str(),
                    reason => reason,
                };
                return FrontAnswer::Tool(Err(ToolError::Unavailable(reason.to_string())));
            }
            FrontAnswer::Tool(tools::call(st, &name, &input, &caller))
        }
    }
}

/// What the rpc thread collected while it could not forward (R-9e, R3-2c).
fn apply_carry(st: &mut FrontState, carry: &Carry) {
    if carry.skipped.is_empty() && carry.forgot.is_empty() {
        return;
    }
    log_line(&format!("front carry skipped={} forgot={}", carry.skipped.len(), carry.forgot.len()));
    // A chat that lost a hook is not captured live; the turn arrives with its archive.
    for session_id in &carry.skipped {
        st.capture.drop_session(session_id);
    }
    // A chat in which core's `memory_forget` was called: the turn is not to be learned.
    for session_id in &carry.forgot {
        mark_forgotten(st, session_id);
    }
}

/// Section 4, "the live call", for one session of `Carry.forgot`: `capture.suppress`, then
/// `mark_forgotten_turn` inside `begin_write_front`.
fn mark_forgotten(st: &mut FrontState, session_id: &str) {
    let now = (st.clock)();
    let pending = st.capture.suppress(session_id);
    if !is_ready(st) {
        return;
    }
    let Some(conn) = st.conn.as_ref() else { return };
    let (from_ms, turn_key) = match &pending {
        Some((started_ms, key)) => (*started_ms, Some(key.as_str())),
        None => (now, None),
    };
    let logical = project::logical(session_id);
    let guard = match brain_v2::begin_write_front(conn, &st.writing) {
        Ok(guard) => guard,
        Err(_) => {
            log_line("front carry: forget mark not written (no write lock)");
            return;
        }
    };
    let marked =
        index_archives::mark_forgotten_turn(conn, &logical, Some(session_id), "call", from_ms, now, turn_key);
    let ended = match marked {
        Ok(_) => conn.execute_batch("COMMIT"),
        Err(_) => conn.execute_batch("ROLLBACK").and(Err(rusqlite::Error::InvalidQuery)),
    };
    if ended.is_err() {
        let _ = conn.execute_batch("ROLLBACK");
        log_line("front carry: forget mark not written");
    }
    drop(guard);
}
