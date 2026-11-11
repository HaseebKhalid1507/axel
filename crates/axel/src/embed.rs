//! `EmbedService`: the embed thread, which owns the one embedder (section 3, 2.3, R-5). A query
//! queue, served first, and a bulk queue worked in batches of 4. A panic ends this thread only;
//! every caller then gets `None` at once, and every wait has a deadline.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// A field of both `FrontState` and `IndexerCtx`, so it is `Clone` (B-2c).
#[derive(Clone)]
pub struct EmbedService {
    tx: Option<Sender<Req>>,
    stopped: Arc<AtomicBool>,
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

enum BulkTo {
    /// `submit_bulk`: the result goes to the indexer as `Job::Embedded`.
    Rows(Vec<VecKey>),
    /// `embed_wait`: the result goes back to the blocked caller.
    Caller { reply: Sender<Vec<Vec<f32>>>, deadline: Instant },
}

/// Sets `stopped` when the thread ends, by return or by unwinding.
struct StopGuard(Arc<AtomicBool>);
impl Drop for StopGuard {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

impl EmbedService {
    /// Spawns the thread; with `test.panic_embed` it panics on its first request.
    // SPEC: 5059: with no embedder no thread is spawned; every call answers `None` and
    // `stopped()` stays false, since nothing ended (`status` says the model is unavailable,
    // not `embedding: stopped`, 3889-3891).
    pub fn start(embedder: Option<Embedder>, jobs: SyncSender<Job>, test: Arc<TestHooks>) -> Self {
        let stopped = Arc::new(AtomicBool::new(false));
        let Some(embedder) = embedder else {
            return EmbedService { tx: None, stopped, model: None };
        };
        let model = embedder.model_id().to_string();
        let (tx, rx) = mpsc::channel();
        let guard = StopGuard(stopped.clone());
        let thread_model = model.clone();
        let spawned = thread::Builder::new()
            .name("axel-embed".to_string())
            .spawn(move || {
                let _guard = guard;
                run(embedder, rx, jobs, test, thread_model);
            });
        // A thread that could not be spawned has ended before it began (its guard is dropped).
        let tx = spawned.ok().map(|_| tx);
        EmbedService { tx, stopped, model: Some(model) }
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

    /// Memory rows; results arrive as `Job::Embedded`.
    pub fn submit_bulk(&self, items: Vec<(VecKey, String)>) {
        let Some(tx) = &self.tx else { return };
        if items.is_empty() || self.stopped() {
            return;
        }
        let (keys, texts): (Vec<VecKey>, Vec<String>) = items.into_iter().unzip();
        let _ = tx.send(Req::Bulk(Bulk { texts, done: Vec::new(), to: BulkTo::Rows(keys) }));
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

    /// The channel to a running thread, when the deadline is still ahead.
    fn live(&self, deadline: Instant) -> Option<&Sender<Req>> {
        let tx = self.tx.as_ref()?;
        if self.stopped() || Instant::now() >= deadline {
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

fn run(mut embedder: Embedder, rx: Receiver<Req>, jobs: SyncSender<Job>, test: Arc<TestHooks>, model: String) {
    let mut queries: VecDeque<(String, Instant, Sender<Vec<f32>>)> = VecDeque::new();
    let mut bulks: VecDeque<Bulk> = VecDeque::new();
    let mut pending: Option<Job> = None;
    let mut open = true;

    loop {
        // Take every request that has arrived; block only when there is nothing to do.
        let mut first = None;
        if open && queries.is_empty() && (bulks.is_empty() || pending.is_some()) {
            let got = match pending {
                None => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
                Some(_) => rx.recv_timeout(RETRY_SEND),
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
            if let Ok(mut v) = embedder.embed(&[text.as_str()]) {
                if let Some(v) = v.pop() {
                    let _ = reply.send(v);
                }
            }
        }

        // A finished bulk waits for room on the indexer's queue; queries go on being served.
        if let Some(job) = pending.take() {
            match jobs.try_send(job) {
                Ok(()) | Err(TrySendError::Disconnected(_)) => {}
                Err(TrySendError::Full(job)) => {
                    pending = Some(job);
                    if !open {
                        thread::sleep(RETRY_SEND);
                    }
                    continue;
                }
            }
        }

        // Then one batch of the bulk queue.
        let Some(bulk) = bulks.front_mut() else {
            if !open && pending.is_none() {
                return;
            }
            continue;
        };
        if let BulkTo::Caller { deadline, .. } = &bulk.to {
            if Instant::now() >= *deadline {
                bulks.pop_front();
                continue;
            }
        }
        let from = bulk.done.len();
        let to = (from + BATCH).min(bulk.texts.len());
        let batch: Vec<&str> = bulk.texts[from..to].iter().map(String::as_str).collect();
        match embedder.embed(&batch) {
            Ok(vs) if vs.len() == batch.len() => bulk.done.extend(vs),
            // SPEC: 1789-1795: a failed batch is not retried here. A caller gets `None`; the
            // rows of `submit_bulk` are left out of `Job::Embedded` and keep `embedding = NULL`,
            // so the indexer's next pump offers them again.
            _ => {
                if let BulkTo::Caller { .. } = bulk.to {
                    bulks.pop_front();
                    continue;
                }
                bulk.done.extend(std::iter::repeat_with(Vec::new).take(batch.len()));
            }
        }
        if bulk.done.len() < bulk.texts.len() {
            continue;
        }
        let Some(bulk) = bulks.pop_front() else { continue };
        match bulk.to {
            BulkTo::Caller { reply, .. } => {
                let _ = reply.send(bulk.done);
            }
            BulkTo::Rows(keys) => {
                let vectors: Vec<(VecKey, Vec<f32>)> =
                    keys.into_iter().zip(bulk.done).filter(|(_, v)| !v.is_empty()).collect();
                pending = Some(Job::Embedded { model: model.clone(), vectors });
            }
        }
    }
}
