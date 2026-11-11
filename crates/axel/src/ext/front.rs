//! The front thread's loop: it answers calls (hooks and tools) on its own connection and
//! owns the vector array. It never does background work.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::brain_v2::{self, Role};
use crate::capture::CaptureState;
use crate::embed::EmbedService;
use crate::error::AxelError;
use crate::ext::config::ExtConfig;
use crate::ext::server::{self, degrade, log_line, JOB_QUEUE};
use crate::ext::types::{
    Boot, Caller, Carry, DegradedReason, ForgetMark, FrontState, HookEvent, HookOut, Job, Mode, ToolError,
};
use crate::ext::{hooks, tools};
use crate::r8::EMBEDDING_DIM;
use crate::recall::View;
use crate::vindex::{ExactIndex, VecDelta, VecKey};
use crate::{capture, index_archives, project, recall};

/// Forget marks kept for a later try (wave 3, 12b ruling 7; wave 4, R-12b item 1).
const FORGET_MARKS_KEPT: usize = 256;
/// How long the last try of the kept forget marks waits for the write lock. It must fit
/// inside `server::EXIT_WAIT` (200 ms), what `serve` waits for this thread at the end;
/// change the two together (`server.rs` asserts it when the crate compiles).
pub(crate) const LAST_TRY_WAIT: Duration = Duration::from_millis(150);
/// How long the loop waits for a call before it looks after its own state.
const IDLE: Duration = Duration::from_millis(250);

/// What the rpc thread hands over: one call, and with it the `Carry` it has collected.
pub struct FrontCall {
    pub seq: u64,
    pub carry: Carry,
    pub work: FrontWork,
}

pub enum FrontWork {
    /// `deadline`: when the rpc thread stops waiting for this answer.
    Hook { ev: HookEvent, recall_allowed: bool, deadline: Instant },
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
        forget_marks: Vec::new(),
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
                    break;
                }
                retry_forget_marks(&mut st);
            }
            Err(RecvTimeoutError::Timeout) => idle(&mut st),
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
    last_try_forget_marks(&mut st);
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
    retry_forget_marks(st);
}

/// One call. The `Carry` first, before anything else.
pub fn handle(st: &mut FrontState, call: FrontCall) -> FrontAnswer {
    let ready = is_ready(st) && connect(st);
    apply_carry(st, &call.carry);
    match call.work {
        FrontWork::Hook { ev, recall_allowed, deadline } => {
            if !ready {
                if let Some(sid) = ev.session_id.as_deref() { st.capture.drop_session(sid); }
                // RULING: v2 final capture round, item 2: a marked or withheld turn let go
                // here still reaches `write_turn`.
                capture::after_let_go(st);
                return FrontAnswer::Hook(HookOut::Continue);
            }
            // RULING: v2 front follow-up, item 2. Recall works to the rpc thread's deadline,
            // which the `Carry` above may have used up. `hooks::handle` has no deadline to
            // pass, so `before_message` is dispatched here, as its arm there does.
            if ev.kind == "before_message" {
                let out = recall::on_before_message_until(st, &ev, recall_allowed, deadline);
                capture::on_user_event(st, &ev);
                return FrontAnswer::Hook(out);
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
    // The turns let go here are counted now, not at the next capture hook of any chat.
    // RULING: v2 final capture round, item 2: and a marked or withheld one is sent.
    capture::after_let_go(st);
    // A chat in which core's `memory_forget` was called: the turn is not to be learned.
    mark_carried(st, &carry.forgot);
}

/// `mark_forgotten` for each chat of `Carry.forgot`.
// RULING: v2 c18a review round 2, item 1. Without a wait for the lock, as
// `retry_forget_marks`: a mark not written now is kept, and the call this carry came with
// starts with nothing in front of it.
fn mark_carried(st: &mut FrontState, forgot: &[String]) {
    if forgot.is_empty() {
        return;
    }
    let tried = match WaitBack::shorten(&mut *st, |_| Duration::ZERO) {
        Ok(mut st) => {
            for session_id in forgot {
                let _ = mark_forgotten(&mut st, session_id);
            }
            true
        }
        Err(_) => false,
    };
    // The wait could not be read: the marks are kept untried.
    if !tried {
        for session_id in forgot {
            let _ = mark(st, session_id, false);
        }
    }
}

/// The connection's wait for the write lock, shortened for a while and put back when this
/// is dropped, by a panic too. With no connection there is nothing to shorten or put back.
// RULING: v2 c18a review round 2, item 4.
pub struct WaitBack<S: Deref<Target = FrontState>> {
    st: S,
    waits: Option<Duration>,
}

impl<S: Deref<Target = FrontState>> WaitBack<S> {
    /// `to` is given the connection's own wait and returns the one to use meanwhile.
    pub fn shorten(st: S, to: impl FnOnce(Duration) -> Duration) -> rusqlite::Result<Self> {
        let Some(conn) = st.conn.as_ref() else { return Ok(Self { st, waits: None }) };
        let waits = Duration::from_millis(conn.query_row("PRAGMA busy_timeout", [], |r| r.get::<_, u64>(0))?);
        conn.busy_timeout(to(waits))?;
        Ok(Self { st, waits: Some(waits) })
    }
}

impl<S: Deref<Target = FrontState>> Deref for WaitBack<S> {
    type Target = FrontState;
    fn deref(&self) -> &FrontState {
        &self.st
    }
}

impl<S: DerefMut<Target = FrontState>> DerefMut for WaitBack<S> {
    fn deref_mut(&mut self) -> &mut FrontState {
        &mut self.st
    }
}

impl<S: Deref<Target = FrontState>> Drop for WaitBack<S> {
    fn drop(&mut self) {
        if let (Some(waits), Some(conn)) = (self.waits, self.st.conn.as_ref()) {
            let _ = conn.busy_timeout(waits);
        }
    }
}

/// Section 4, "the live call", for one session of `Carry.forgot`, and for the model's
/// `forget` tool (`tools_core::mark_turn`): `capture.suppress`, then `write_forget_mark`.
/// A mark that is not written now is kept. Returns what became of the try.
pub(crate) fn mark_forgotten(st: &mut FrontState, session_id: &str) -> MarkEnd {
    mark(st, session_id, true)
}

/// `mark_forgotten`; with `try_now` false every mark is kept without a try.
fn mark(st: &mut FrontState, session_id: &str, try_now: bool) -> MarkEnd {
    let now = (st.clock)();
    let replaced = st.capture.replaced_keys(session_id);
    let (from_ms, turn_key) = match st.capture.suppress(session_id) {
        Some((started_ms, key)) => (started_ms, Some(key)),
        None => (now, None),
    };
    // RULING: v2 final capture round, item 1. One keyed `call` mark for the turn's prompt
    // and one for each prompt it replaced, all from the turn's first message. Once one of
    // them meets a held lock the rest are kept without a try (each would wait as long).
    let keys = std::iter::once(turn_key).chain(replaced.into_iter().map(Some));
    let mut end = if try_now { MarkEnd::Written } else { MarkEnd::NotNow };
    for turn_key in keys {
        let mark = ForgetMark { session_id: session_id.to_string(), from_ms, to_ms: now, turn_key };
        // RULING: wave4 R-12b item 1. The `Carry` is gone, so nothing else holds this
        // session: a mark that is not written now is kept, whatever kept it out (the mode
        // is not `Ready`, there is no connection, the write lock is held).
        let this = if end != MarkEnd::NotNow && is_ready(st) && st.conn.is_some() {
            write_forget_mark(st, &mark)
        } else {
            MarkEnd::NotNow
        };
        match this {
            MarkEnd::Written => {}
            MarkEnd::Failed(kind) => {
                log_line(&format!("front carry: forget mark dropped: {kind}"));
                if end == MarkEnd::Written {
                    end = MarkEnd::Failed(kind);
                }
            }
            MarkEnd::NotNow => {
                keep_mark(st, mark);
                end = MarkEnd::NotNow;
            }
        }
    }
    end
}

/// A mark that could not be written now, kept for a later try.
fn keep_mark(st: &mut FrontState, mark: ForgetMark) {
    // RULING: v2 final capture round, item 7. A model that retries a busy call would
    // otherwise push copies of one mark, and at the limit they push other chats' marks out.
    // The one kept takes the later end.
    if let Some(kept) = st
        .forget_marks
        .iter_mut()
        .find(|m| m.session_id == mark.session_id && m.from_ms == mark.from_ms && m.turn_key == mark.turn_key)
    {
        kept.to_ms = kept.to_ms.max(mark.to_ms);
        log_line("front carry: forget mark not written, already kept");
        return;
    }
    // RULING: wave3-c12b ruling 7. At most `FORGET_MARKS_KEPT` wait; one more gives up the
    // oldest, so a brain that stays locked cannot grow the list without end.
    if st.forget_marks.len() >= FORGET_MARKS_KEPT {
        st.forget_marks.remove(0);
        log_line("front carry: forget mark given up (too many waiting)");
    }
    st.forget_marks.push(mark);
    log_line("front carry: forget mark not written, kept");
}

/// What became of one try to write a mark.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MarkEnd {
    /// The mark is in the brain.
    Written,
    /// The write lock was not free, or there is no connection: the mark is tried again.
    NotNow,
    /// Any other error, by its kind (never its text): the mark is dropped.
    Failed(&'static str),
}

/// The write lock is held by somebody else.
fn is_busy(e: &AxelError) -> bool {
    matches!(
        e,
        AxelError::Db(e) if matches!(
            e.sqlite_error_code(),
            Some(rusqlite::ErrorCode::DatabaseBusy) | Some(rusqlite::ErrorCode::DatabaseLocked)
        )
    )
}

fn error_kind(e: &AxelError) -> &'static str {
    match e {
        AxelError::BrainChanged => "BrainChanged",
        AxelError::BrainDamaged(_) => "BrainDamaged",
        AxelError::Db(_) => "Db",
        _ => "Other",
    }
}

/// One mark, and the vectors of the rows it purged out of the array, once its `COMMIT`
/// has held.
fn write_forget_mark(st: &mut FrontState, mark: &ForgetMark) -> MarkEnd {
    let (end, vids) = write_mark(st, mark);
    remove_vectors(st, vids);
    end
}

/// The vectors of purged rows leave the array. Called only with the `vid`s of a
/// transaction that has committed.
fn remove_vectors(st: &mut FrontState, vids: Vec<i64>) {
    if !vids.is_empty() {
        st.index.apply(VecDelta::Remove(vids.into_iter().map(VecKey::Mem).collect()));
    }
}

/// The mark in its own transaction. The `vid`s of the rows it purged are returned only
/// when the `COMMIT` succeeded: after a rollback the rows are still there.
fn write_mark(st: &FrontState, mark: &ForgetMark) -> (MarkEnd, Vec<i64>) {
    let Some(conn) = st.conn.as_ref() else { return (MarkEnd::NotNow, Vec::new()) };
    let logical = project::logical(&mark.session_id);
    let guard = match brain_v2::begin_write_front(conn, &st.writing) {
        Ok(guard) => guard,
        Err(e) if is_busy(&e) => return (MarkEnd::NotNow, Vec::new()),
        Err(e) => return (MarkEnd::Failed(error_kind(&e)), Vec::new()),
    };
    let marked = index_archives::mark_forgotten_turn_vids(
        conn,
        &logical,
        Some(&mark.session_id),
        "call",
        mark.from_ms,
        mark.to_ms,
        mark.turn_key.as_deref(),
    );
    let ended = match marked {
        Ok((_, vids)) => conn.execute_batch("COMMIT").map(|()| vids).map_err(AxelError::from),
        Err(e) => Err(e),
    };
    if ended.is_err() {
        let _ = conn.execute_batch("ROLLBACK");
    }
    drop(guard);
    match ended {
        Ok(vids) => (MarkEnd::Written, vids),
        Err(e) if is_busy(&e) => (MarkEnd::NotNow, Vec::new()),
        Err(e) => (MarkEnd::Failed(error_kind(&e)), Vec::new()),
    }
}

/// A marked or withheld turn whose `Job::Capture` the queue refused: what `write_turn` would
/// have done with it, on this thread's connection, in one write transaction. When that
/// cannot be done now, one log line; nothing is counted.
// RULING: v2 final small round, item 3. The `call` interval would end at the call and the
// log would keep the head of a prompt he asked to forget.
pub(crate) fn forget_unqueued_turn(st: &FrontState, turn: &capture::CompletedTurn) {
    let Some(conn) = st.conn.as_ref().filter(|_| is_ready(st)) else {
        log_line("front: a refused turn's forget rows not written: not ready");
        return;
    };
    let guard = match brain_v2::begin_write_front(conn, &st.writing) {
        Ok(guard) => guard,
        Err(e) => {
            let kind = if is_busy(&e) { "busy" } else { error_kind(&e) };
            log_line(&format!("front: a refused turn's forget rows not written: {kind}"));
            return;
        }
    };
    let ended = capture::forget_turn_rows(conn, turn)
        .and_then(|()| conn.execute_batch("COMMIT").map_err(AxelError::from));
    if let Err(e) = ended {
        let _ = conn.execute_batch("ROLLBACK");
        let kind = if is_busy(&e) { "busy" } else { error_kind(&e) };
        log_line(&format!("front: a refused turn's forget rows not written: {kind}"));
    }
    drop(guard);
}

/// `retry_marks` with `write_mark`; the vectors of what the written marks purged leave
/// the array after the round (each mark's `COMMIT` has held by then).
/// Public for its test.
pub fn retry_written(st: &mut FrontState) -> usize {
    let gone = RefCell::new(Vec::new());
    let written = retry_marks(st, |st, mark| {
        let (end, vids) = write_mark(st, mark);
        gone.borrow_mut().extend(vids);
        end
    });
    remove_vectors(st, gone.into_inner());
    written
}

/// The kept marks, oldest first, each given to `write` once. A mark that cannot be written
/// now stops the round and stays, with every mark behind it; a mark that failed for another
/// reason is dropped with a log line, and the round goes on. Returns how many were written.
/// Public for its test: `write` is `write_mark` everywhere else (`retry_written`).
// RULING: wave4 R-12b item 1. A mark that can never be written does not keep the ones
// behind it out.
pub fn retry_marks(st: &mut FrontState, write: impl Fn(&FrontState, &ForgetMark) -> MarkEnd) -> usize {
    let mut written = 0;
    while let Some(mark) = st.forget_marks.first().cloned() {
        match write(st, &mark) {
            MarkEnd::NotNow => break,
            MarkEnd::Written => written += 1,
            MarkEnd::Failed(kind) => log_line(&format!("front carry: forget mark dropped: {kind}")),
        }
        st.forget_marks.remove(0);
    }
    if written > 0 {
        log_line(&format!("front carry: forget marks written late: {written}"));
    }
    written
}

/// After each call's answer has been sent, and between calls. The lock is asked for without
/// waiting, so a brain that is still locked costs this thread nothing.
fn retry_forget_marks(st: &mut FrontState) {
    if st.forget_marks.is_empty() || !is_ready(st) || !connect(st) {
        return;
    }
    // RULING: v2 c18a review round 2, item 4: the wait is put back by `WaitBack`.
    if let Ok(mut st) = WaitBack::shorten(st, |_| Duration::ZERO) {
        retry_written(&mut st);
    }
}

/// Before the loop returns: the kept marks once more, waiting `LAST_TRY_WAIT` for the lock.
/// What is still not written is lost with the process, and the log says how many before
/// the thread returns.
// RULING: wave4 R-12b item 1. The list lives in memory only.
// RULING: v2 front follow-up, item 5. Not this thread's usual 1.5 s: `serve` would have
// returned first and the log line would be lost too. The archive carries the forget call
// anyway (index_archives step 3).
fn last_try_forget_marks(st: &mut FrontState) {
    if st.forget_marks.is_empty() {
        return;
    }
    if is_ready(st) && connect(st) {
        let waits = st.conn.as_ref().is_some_and(|conn| conn.busy_timeout(LAST_TRY_WAIT).is_ok());
        if waits {
            retry_written(st);
        }
    }
    if !st.forget_marks.is_empty() {
        log_line(&format!("front carry: forget marks lost at the end: {}", st.forget_marks.len()));
    }
}
