//! `EmbedService`: the embed thread, which owns the one embedder (section 3, 2.3, R-5). A query
//! queue, served first, and a bulk queue worked in batches of 4. A panic ends this thread only;
//! every caller then gets `None` at once, and every wait has a deadline.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use velocirag::embedder::Embedder;

use crate::ext::types::{Job, TestHooks};
use crate::vindex::VecKey;

/// Bulk texts embedded per step; the query queue is served between steps (2.3).
const BATCH: usize = 4;
/// How long the thread waits between attempts to hand a finished bulk to a full indexer queue.
const RETRY_SEND: Duration = Duration::from_millis(5);

/// A batch still inside the model after this long has hung there (wave 3, 12b ruling 9).
const STUCK_MS: u64 = 60_000;

// RULING: wave3-c12b ruling 9. When the embed thread began the batch it is inside. A thread
// that hangs in the model has not ended, so `stopped()` stays false; this is what says so.
struct BatchClock {
    epoch: Instant,
    /// 0 when no batch is running; else milliseconds from `epoch` to its start, plus 1.
    began: AtomicU64,
    limit: Duration,
}

impl BatchClock {
    fn new(test: &TestHooks) -> Arc<BatchClock> {
        let limit = Duration::from_millis(test.embed_stuck_limit_ms.unwrap_or(STUCK_MS));
        Arc::new(BatchClock { epoch: Instant::now(), began: AtomicU64::new(0), limit })
    }
    fn begin(&self) {
        self.began.store(self.epoch.elapsed().as_millis() as u64 + 1, Ordering::SeqCst);
    }
    fn end(&self) {
        self.began.store(0, Ordering::SeqCst);
    }
    fn stuck(&self) -> bool {
        match self.began.load(Ordering::SeqCst) {
            0 => false,
            began => self.epoch.elapsed().saturating_sub(Duration::from_millis(began - 1)) >= self.limit,
        }
    }
}

/// A field of both `FrontState` and `IndexerCtx`, so it is `Clone` (B-2c).
#[derive(Clone)]
pub struct EmbedService {
    tx: Option<Sender<Req>>,
    stopped: Arc<AtomicBool>,
    backlog: Arc<AtomicUsize>,
    batch: Arc<BatchClock>,
    model: Option<String>,
}

enum Req {
    Query { text: String, deadline: Instant, reply: Sender<Vec<f32>> },
    Bulk(Bulk),
}

struct Bulk {
    texts: Vec<String>,
    done: Vec<Vec<f32>>,
    to: BulkTo,
}

/// One memory row whose text goes to the embed thread (`submit_rows`).
pub struct BulkRow {
    pub vid: i64,
    pub mem_id: String,
    pub content_hash: String,
    pub text: String,
}

// RULING: wave4 R-12b item 4. `vid` alone does not name a row: SQLite gives the number of a
// purged newest row to the next row inserted. The batch carries what the row was when its
// text was read, and comes back with its vectors.
/// What a finished bulk carries beside its vectors: for each `vid` that was sent, the
/// `mem_id` and `content_hash` the row had then.
///
/// It also holds the batch's place in `backlog()`: the count comes off when this is dropped,
/// which is when the indexer has finished with the job (or the job was thrown away), not
/// when the job was put on the queue.
// RULING: wave4 R-12b item 5. A count taken off at the hand-over let the pump offer the same
// rows again while their vectors waited behind another job. The ruling names a method
// `done(n)`; a drop does the same and cannot be forgotten on a path that skips the handler
// (a degraded process, a shutdown, a panic).
#[derive(Debug)]
pub struct BatchOut {
    rows: HashMap<i64, (String, String)>,
    counted: usize,
    backlog: Arc<AtomicUsize>,
}

impl Drop for BatchOut {
    fn drop(&mut self) {
        unlog(&self.backlog, self.counted);
    }
}

impl BatchOut {
    /// `(mem_id, content_hash)` of the row that was sent under this number.
    pub fn row(&self, vid: i64) -> Option<(&str, &str)> {
        self.rows.get(&vid).map(|(mem_id, hash)| (mem_id.as_str(), hash.as_str()))
    }
}

enum BulkTo {
    /// `submit_bulk` and `submit_rows`: the result goes to the indexer as `Job::Embedded`.
    Rows(Vec<VecKey>, HashMap<i64, (String, String)>),
    /// `embed_wait`: the result goes back to the blocked caller.
    Caller { reply: Sender<Vec<Vec<f32>>>, deadline: Instant },
}

/// When the thread ends, by return or by unwinding: nothing it held will be handed over, so
/// the backlog is 0; then `stopped`.
struct StopGuard {
    stopped: Arc<AtomicBool>,
    backlog: Arc<AtomicUsize>,
}
impl Drop for StopGuard {
    fn drop(&mut self) {
        self.backlog.store(0, Ordering::SeqCst);
        self.stopped.store(true, Ordering::SeqCst);
    }
}

/// Takes `n` off the backlog, never below 0 (the guard may have zeroed it first).
fn unlog(backlog: &AtomicUsize, n: usize) {
    let _ = backlog.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |b| Some(b.saturating_sub(n)));
}

impl EmbedService {
    /// Spawns the thread; with `test.panic_embed` it panics on its first request.
    // SPEC: 5059: with no embedder no thread is spawned; every call answers `None` and
    // `stopped()` stays false, since nothing ended (`status` says the model is unavailable,
    // not `embedding: stopped`, 3889-3891).
    pub fn start(embedder: Option<Embedder>, jobs: SyncSender<Job>, test: Arc<TestHooks>) -> Self {
        let Some(mut embedder) = embedder else {
            let (stopped, backlog) = Default::default();
            return EmbedService { tx: None, stopped, backlog, batch: BatchClock::new(&test), model: None };
        };
        let model = embedder.model_id().to_string();
        Self::spawn(move |texts: &[&str]| embedder.embed(texts).ok(), model, jobs, test)
    }

    /// The thread around any embedding function (`None` is a failed batch).
    fn spawn<E>(embed: E, model: String, jobs: SyncSender<Job>, test: Arc<TestHooks>) -> Self
    where
        E: FnMut(&[&str]) -> Option<Vec<Vec<f32>>> + Send + 'static,
    {
        let stopped = Arc::new(AtomicBool::new(false));
        let backlog = Arc::new(AtomicUsize::new(0));
        let batch = BatchClock::new(&test);
        let thread_batch = batch.clone();
        let (tx, rx) = mpsc::channel();
        let guard = StopGuard { stopped: stopped.clone(), backlog: backlog.clone() };
        let thread_model = model.clone();
        let spawned = thread::Builder::new()
            .name("axel-embed".to_string())
            .spawn(move || {
                let backlog = guard.backlog.clone();
                let _guard = guard;
                run(embed, rx, jobs, test, thread_model, &backlog, &thread_batch);
            });
        // A thread that could not be spawned has ended before it began (its guard is dropped).
        let tx = spawned.ok().map(|_| tx);
        EmbedService { tx, stopped, backlog, batch, model: Some(model) }
    }

    pub fn model_id(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub fn embed_query(&self, text: &str, deadline: Instant) -> Option<Vec<f32>> {
        let tx = self.live(deadline)?;
        let (reply, answer) = mpsc::channel();
        tx.send(Req::Query { text: text.to_string(), deadline, reply }).ok()?;
        wait(&answer, deadline)
    }

    /// Memory rows; results arrive as `Job::Embedded`, whose `out` says which row each
    /// number stood for.
    pub fn submit_rows(&self, rows: Vec<BulkRow>) {
        let mut sent = HashMap::new();
        let mut items = Vec::with_capacity(rows.len());
        for row in rows {
            sent.insert(row.vid, (row.mem_id, row.content_hash));
            items.push((VecKey::Mem(row.vid), row.text));
        }
        self.submit(items, sent);
    }

    /// Texts by key alone; results arrive as `Job::Embedded`. The indexer writes no vector
    /// from such a batch: it does not say which rows it was read from.
    pub fn submit_bulk(&self, items: Vec<(VecKey, String)>) {
        self.submit(items, HashMap::new());
    }

    fn submit(&self, items: Vec<(VecKey, String)>, sent: HashMap<i64, (String, String)>) {
        let Some(tx) = &self.tx else { return };
        if items.is_empty() || self.stopped() {
            return;
        }
        let n = items.len();
        let (keys, texts): (Vec<VecKey>, Vec<String>) = items.into_iter().unzip();
        self.backlog.fetch_add(n, Ordering::SeqCst);
        if tx.send(Req::Bulk(Bulk { texts, done: Vec::new(), to: BulkTo::Rows(keys, sent) })).is_err() {
            unlog(&self.backlog, n);
        }
    }

    // RULING: wave2-c9 review finding 4: the depth of the bulk queue, for the indexer's pump.
    /// Texts accepted by `submit_rows` or `submit_bulk` whose `Job::Embedded` the indexer has
    /// not finished with, and which were not dropped (a failed batch, an indexer that is
    /// gone, an embed thread that ended).
    pub fn backlog(&self) -> usize {
        self.backlog.load(Ordering::SeqCst)
    }

    /// Through the bulk queue, blocking the caller; the indexer thread only (section 2.8).
    pub fn embed_wait(&self, texts: Vec<String>, deadline: Instant) -> Option<Vec<Vec<f32>>> {
        let tx = self.live(deadline)?;
        if texts.is_empty() {
            return Some(Vec::new());
        }
        let (reply, answer) = mpsc::channel();
        tx.send(Req::Bulk(Bulk { texts, done: Vec::new(), to: BulkTo::Caller { reply, deadline } })).ok()?;
        wait(&answer, deadline)
    }

    /// The embed thread has ended (section 7).
    pub fn stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }

    /// The embed thread has been inside one batch for longer than the limit (60 s): it has
    /// hung in the model. Every caller gets `None` at once until that batch returns.
    pub fn stuck(&self) -> bool {
        self.batch.stuck()
    }

    /// The channel to a running thread, when the deadline is still ahead.
    fn live(&self, deadline: Instant) -> Option<&Sender<Req>> {
        let tx = self.tx.as_ref()?;
        if self.stopped() || self.stuck() || Instant::now() >= deadline {
            return None;
        }
        Some(tx)
    }
}

/// `None` when the deadline passes or the thread drops the reply (it ended, or the work failed).
fn wait<T>(answer: &Receiver<T>, deadline: Instant) -> Option<T> {
    let left = deadline.checked_duration_since(Instant::now())?;
    answer.recv_timeout(left).ok()
}

fn run<E>(
    mut inner: E,
    rx: Receiver<Req>,
    jobs: SyncSender<Job>,
    test: Arc<TestHooks>,
    model: String,
    backlog: &Arc<AtomicUsize>,
    batch: &BatchClock,
) where
    E: FnMut(&[&str]) -> Option<Vec<Vec<f32>>>,
{
    // Every call into the model is clocked.
    let sleep_ms = test.embed_sleep_ms;
    let mut embed = |texts: &[&str]| {
        batch.begin();
        if sleep_ms > 0 {
            thread::sleep(Duration::from_millis(sleep_ms));
        }
        let out = inner(texts);
        batch.end();
        out
    };
    let mut queries: VecDeque<(String, Instant, Sender<Vec<f32>>)> = VecDeque::new();
    let mut bulks: VecDeque<Bulk> = VecDeque::new();
    // RULING: wave2-c9 review finding 3: finished bulks wait here for room on the indexer's
    // queue while bulk work goes on; what fits is sent at the top of each round.
    let mut pending: VecDeque<Job> = VecDeque::new();
    let mut open = true;

    loop {
        flush(&jobs, &mut pending);

        // Take every request that has arrived; block only when there is nothing to do.
        let mut first = None;
        if open && queries.is_empty() && bulks.is_empty() {
            let got = if pending.is_empty() {
                rx.recv().map_err(|_| RecvTimeoutError::Disconnected)
            } else {
                rx.recv_timeout(RETRY_SEND)
            };
            match got {
                Ok(req) => first = Some(req),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => open = false,
            }
        }
        let mut take = |req: Req| {
            if test.panic_embed {
                panic!("axel embed thread: panic_embed test hook");
            }
            match req {
                Req::Query { text, deadline, reply } => queries.push_back((text, deadline, reply)),
                Req::Bulk(b) => bulks.push_back(b),
            }
        };
        if let Some(req) = first {
            take(req);
        }
        while open {
            match rx.try_recv() {
                Ok(req) => take(req),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => open = false,
            }
        }

        // The query queue first, whole.
        while let Some((text, deadline, reply)) = queries.pop_front() {
            if Instant::now() >= deadline {
                continue;
            }
            if let Some(mut v) = embed(&[text.as_str()][..]) {
                if let Some(v) = v.pop() {
                    let _ = reply.send(v);
                }
            }
        }

        // Then one batch of the bulk queue.
        if let Some(job) = step(&mut embed, &mut bulks, &model, backlog) {
            pending.push_back(job);
        }

        if !open && bulks.is_empty() {
            if pending.is_empty() {
                return;
            }
            flush(&jobs, &mut pending);
            if !pending.is_empty() {
                thread::sleep(RETRY_SEND);
            }
        }
    }
}

/// Hands finished bulks to the indexer's queue, oldest first, until it is full. A send that
/// fails as disconnected drops the job: the indexer is gone. The backlog is not touched
/// here: a job takes its count with it (`BatchOut`).
fn flush(jobs: &SyncSender<Job>, pending: &mut VecDeque<Job>) {
    while let Some(job) = pending.pop_front() {
        match jobs.try_send(job) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => {}
            Err(TrySendError::Full(job)) => {
                pending.push_front(job);
                return;
            }
        }
    }
}

/// Embeds one batch of the oldest bulk request. Returns the `Job::Embedded` of a
/// `submit_bulk` request it finished; a finished `embed_wait` is answered here.
fn step<E>(embed: &mut E, bulks: &mut VecDeque<Bulk>, model: &str, backlog: &Arc<AtomicUsize>) -> Option<Job>
where
    E: FnMut(&[&str]) -> Option<Vec<Vec<f32>>>,
{
    let bulk = bulks.front_mut()?;
    if let BulkTo::Caller { deadline, .. } = &bulk.to {
        if Instant::now() >= *deadline {
            bulks.pop_front();
            return None;
        }
    }
    let from = bulk.done.len();
    let to = (from + BATCH).min(bulk.texts.len());
    let batch: Vec<&str> = bulk.texts[from..to].iter().map(String::as_str).collect();
    match embed(&batch[..]) {
        Some(vs) if vs.len() == batch.len() => bulk.done.extend(vs),
        // SPEC: 1789-1795: a failed batch is not retried here. A caller gets `None`; the
        // rows of `submit_bulk` are left out of `Job::Embedded` and keep `embedding = NULL`,
        // so the indexer's next pump offers them again.
        _ => {
            if let BulkTo::Caller { .. } = bulk.to {
                bulks.pop_front();
                return None;
            }
            unlog(backlog, batch.len());
            bulk.done.extend(std::iter::repeat_with(Vec::new).take(batch.len()));
        }
    }
    if bulk.done.len() < bulk.texts.len() {
        return None;
    }
    let bulk = bulks.pop_front()?;
    match bulk.to {
        BulkTo::Caller { reply, .. } => {
            let _ = reply.send(bulk.done);
            None
        }
        BulkTo::Rows(keys, rows) => {
            let vectors: Vec<(VecKey, Vec<f32>)> =
                keys.into_iter().zip(bulk.done).filter(|(_, v)| !v.is_empty()).collect();
            let out = BatchOut { rows, counted: vectors.len(), backlog: backlog.clone() };
            Some(Job::Embedded { model: model.to_string(), vectors, out: Some(Arc::new(out)) })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hash embedder cannot fail, so this one does: on any batch with a text containing
    /// "fail" (the indexer's `submit_bulk` path then leaves those rows out).
    #[test]
    fn a_failed_batch_leaves_the_backlog_at_zero() {
        let (jobs, jobs_rx) = mpsc::sync_channel(4);
        let embed = |texts: &[&str]| {
            if texts.iter().any(|t| t.contains("fail")) {
                None
            } else {
                Some(texts.iter().map(|_| vec![1.0, 0.0]).collect())
            }
        };
        let svc = EmbedService::spawn(embed, "fake".to_string(), jobs, Arc::new(TestHooks::default()));
        // Batch 1 (rows 0-3) fails, batch 2 (rows 4-7) works.
        let items: Vec<(VecKey, String)> =
            (0..8).map(|i| (VecKey::Mem(i), if i == 2 { "fail".to_string() } else { format!("row {i}") })).collect();
        svc.submit_bulk(items);
        match jobs_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Job::Embedded { vectors, .. }) => {
                assert_eq!(vectors.iter().map(|(k, _)| *k).collect::<Vec<_>>(), (4..8).map(VecKey::Mem).collect::<Vec<_>>());
            }
            other => panic!("expected Job::Embedded, got {other:?}"),
        }
        let until = Instant::now() + Duration::from_secs(2);
        while svc.backlog() != 0 && Instant::now() < until {
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(svc.backlog(), 0);

        // A bulk whose every batch fails is handed over empty and leaves nothing counted.
        svc.submit_bulk(vec![(VecKey::Mem(9), "fail".to_string())]);
        assert!(matches!(jobs_rx.recv_timeout(Duration::from_secs(5)), Ok(Job::Embedded { vectors, .. }) if vectors.is_empty()));
        let until = Instant::now() + Duration::from_secs(2);
        while svc.backlog() != 0 && Instant::now() < until {
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(svc.backlog(), 0);
        // An `embed_wait` with a failing batch is answered `None` at once, not at its deadline.
        let t0 = Instant::now();
        assert_eq!(svc.embed_wait(vec!["fail".to_string()], Instant::now() + Duration::from_secs(5)), None);
        assert!(t0.elapsed() < Duration::from_secs(2));
    }
}
