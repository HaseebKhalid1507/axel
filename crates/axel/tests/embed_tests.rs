//! Commit 9: the embed thread (`EmbedService`). Hash embedder only; every wait is bounded.

use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use axel::embed::EmbedService;
use axel::ext::types::{Job, TestHooks};
use axel::vindex::VecKey;
use velocirag::embedder::Embedder;

fn start(test: TestHooks) -> (EmbedService, Receiver<Job>) {
    let (jobs, jobs_rx) = sync_channel(16);
    (EmbedService::start(Some(Embedder::hashing()), jobs, Arc::new(test)), jobs_rx)
}

fn secs(n: u64) -> Instant {
    Instant::now() + Duration::from_secs(n)
}

fn hashed(text: &str) -> Vec<f32> {
    Embedder::hashing().embed(&[text]).unwrap().pop().unwrap()
}

/// `n` distinct texts of about `bytes` bytes each: slow enough for the hash embedder that
/// a bulk of them takes a measurable time.
fn big_texts(n: usize, bytes: usize) -> Vec<String> {
    let body = "alpha beta gamma delta epsilon zeta eta theta ".repeat(bytes / 47 + 1);
    (0..n).map(|i| format!("item{i} {body}")).collect()
}

#[test]
fn model_id_is_the_embedders() {
    let (svc, _jobs) = start(TestHooks::default());
    assert_eq!(svc.model_id(), Some(Embedder::hashing().model_id()));
    assert!(!svc.stopped());
}

#[test]
fn embed_query_returns_the_embedders_vector() {
    let (svc, _jobs) = start(TestHooks::default());
    let v = svc.embed_query("retry limit for the uploader", secs(5)).expect("a vector");
    assert_eq!(v.len(), 384);
    assert_eq!(v, hashed("retry limit for the uploader"));
}

#[test]
fn embed_wait_returns_one_vector_per_text() {
    let (svc, _jobs) = start(TestHooks::default());
    // 9 texts: two whole batches of 4 and one of 1; one text has no word (the zero vector).
    let texts: Vec<String> = (0..8).map(|i| format!("note number {i} about vectors")).chain(["!".to_string()]).collect();
    let got = svc.embed_wait(texts.clone(), secs(5)).expect("vectors");
    assert_eq!(got.len(), texts.len());
    for (t, v) in texts.iter().zip(&got) {
        assert_eq!(v.len(), 384);
        assert_eq!(*v, hashed(t));
    }
    assert_eq!(svc.embed_wait(Vec::new(), secs(5)), Some(Vec::new()));
}

#[test]
fn embed_wait_returns_none_past_its_deadline() {
    let (svc, _jobs) = start(TestHooks::default());
    let t0 = Instant::now();
    let got = svc.embed_wait(big_texts(64, 512 * 1024), Instant::now() + Duration::from_millis(20));
    let took = t0.elapsed();
    assert!(got.is_none(), "32 MiB cannot be embedded in 20 ms");
    assert!(took < Duration::from_secs(2), "embed_wait overran its deadline: {took:?}");
    // The thread abandons the late request and still serves.
    assert!(!svc.stopped());
    assert!(svc.embed_query("still here", secs(10)).is_some());
}

#[test]
fn a_deadline_in_the_past_returns_none() {
    let (svc, _jobs) = start(TestHooks::default());
    let past = Instant::now();
    thread::sleep(Duration::from_millis(2));
    let t0 = Instant::now();
    assert_eq!(svc.embed_query("retry limit", past), None);
    assert_eq!(svc.embed_wait(vec!["retry limit".to_string()], past), None);
    assert!(t0.elapsed() < Duration::from_millis(500));
    // The same call with time left is answered: the `None` was the deadline's.
    assert!(svc.embed_query("retry limit", secs(5)).is_some());
}

#[test]
fn no_embedder_returns_none() {
    let (jobs, jobs_rx) = sync_channel(16);
    let svc = EmbedService::start(None, jobs, Arc::new(TestHooks::default()));
    let t0 = Instant::now();
    assert_eq!(svc.model_id(), None);
    assert_eq!(svc.embed_query("retry limit", secs(5)), None);
    assert_eq!(svc.embed_wait(vec!["retry limit".to_string()], secs(5)), None);
    svc.submit_bulk(vec![(VecKey::Mem(1), "retry limit".to_string())]);
    assert!(t0.elapsed() < Duration::from_millis(500));
    assert!(jobs_rx.recv_timeout(Duration::from_millis(200)).is_err());
}

#[test]
fn panic_embed_ends_the_thread_only() {
    let (svc, _jobs) = start(TestHooks { panic_embed: true, ..Default::default() });
    let t0 = Instant::now();
    assert_eq!(svc.embed_query("first query", secs(5)), None);
    assert!(t0.elapsed() < Duration::from_secs(2), "the caller waited on a dead thread");

    let until = Instant::now() + Duration::from_secs(2);
    while !svc.stopped() && Instant::now() < until {
        thread::sleep(Duration::from_millis(5));
    }
    assert!(svc.stopped(), "stopped() must be true once the thread has ended");

    // Every later call is answered `None` at once, not at its deadline.
    let t1 = Instant::now();
    assert_eq!(svc.embed_query("second query", secs(5)), None);
    assert_eq!(svc.clone().embed_wait(vec!["x y".to_string()], secs(5)), None);
    svc.submit_bulk(vec![(VecKey::Mem(1), "x y".to_string())]);
    assert!(t1.elapsed() < Duration::from_millis(200), "{:?}", t1.elapsed());
}

#[test]
fn a_query_is_served_while_a_bulk_batch_of_64_is_queued() {
    // The indexer's queue is full (a dummy job in a queue of one), so no bulk can be handed
    // over until the test takes the dummy out: the query is answered while the bulk of 64 is
    // still the thread's, on any machine (review answer 3).
    let (svc, jobs_rx) = start_latched();
    svc.submit_bulk(rows(0, 4));
    svc.submit_bulk(rows(4, 64));

    let q = svc.embed_query("retry limit", secs(5)).expect("the query is answered");
    assert_eq!(q, hashed("retry limit"));
    assert_eq!(svc.backlog(), 68, "a bulk was handed over before the latch opened");

    // Then the queue gives, in order: the dummy, the bulk of 4, the bulk of 64.
    assert!(matches!(jobs_rx.recv_timeout(Duration::from_secs(5)), Ok(Job::Tick)));
    let first = embedded(&jobs_rx);
    assert_eq!(first.iter().map(|(k, _)| *k).collect::<Vec<_>>(), (0..4).map(VecKey::Mem).collect::<Vec<_>>());
    match jobs_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(Job::Embedded { model, vectors }) => {
            assert_eq!(model, Embedder::hashing().model_id());
            assert_eq!(vectors.len(), 64);
            for (i, (key, v)) in vectors.iter().enumerate() {
                let n = i as i64 + 4;
                assert_eq!(*key, VecKey::Mem(n));
                assert_eq!(*v, hashed(&format!("row number {n} of the bulk")));
            }
        }
        other => panic!("expected Job::Embedded, got {other:?}"),
    }
}

/// An embed service whose indexer queue holds one job and starts full (a dummy `Job::Tick`).
fn start_latched() -> (EmbedService, Receiver<Job>) {
    let (jobs, jobs_rx) = sync_channel(1);
    jobs.try_send(Job::Tick).unwrap();
    (EmbedService::start(Some(Embedder::hashing()), jobs, Arc::new(TestHooks::default())), jobs_rx)
}

fn rows(from: i64, n: i64) -> Vec<(VecKey, String)> {
    (from..from + n).map(|i| (VecKey::Mem(i), format!("row number {i} of the bulk"))).collect()
}

fn embedded(jobs_rx: &Receiver<Job>) -> Vec<(VecKey, Vec<f32>)> {
    match jobs_rx.recv_timeout(Duration::from_secs(5)) {
        Ok(Job::Embedded { vectors, .. }) => vectors,
        other => panic!("expected Job::Embedded, got {other:?}"),
    }
}

#[test]
fn bulk_work_goes_on_while_the_indexer_queue_is_full() {
    let (svc, jobs_rx) = start_latched();
    // The first bulk finishes and cannot be handed over; the queue stays full.
    svc.submit_bulk(rows(0, 4));
    svc.submit_bulk(rows(4, 4));
    // An `embed_wait` queued behind both, from another thread, as the indexer calls it.
    let other = svc.clone();
    let waiter = thread::spawn(move || {
        let t0 = Instant::now();
        let got = other.embed_wait(vec!["late note".to_string()], secs(5));
        (got, t0.elapsed())
    });
    let (got, took) = waiter.join().unwrap();
    assert_eq!(got, Some(vec![hashed("late note")]), "embed_wait was not served while a job waited");
    assert!(took < Duration::from_secs(2), "answered at its deadline, not inside it: {took:?}");

    // Nothing was lost: the dummy, then both bulks in order.
    assert!(matches!(jobs_rx.recv_timeout(Duration::from_secs(5)), Ok(Job::Tick)));
    let first = embedded(&jobs_rx);
    let second = embedded(&jobs_rx);
    assert_eq!(first.iter().map(|(k, _)| *k).collect::<Vec<_>>(), (0..4).map(VecKey::Mem).collect::<Vec<_>>());
    assert_eq!(second.iter().map(|(k, _)| *k).collect::<Vec<_>>(), (4..8).map(VecKey::Mem).collect::<Vec<_>>());
    assert_eq!(second[0].1, hashed("row number 4 of the bulk"));
}

/// Waits (bounded) for the backlog to reach `want`.
fn backlog_reaches(svc: &EmbedService, want: usize) -> usize {
    let until = Instant::now() + Duration::from_secs(2);
    while svc.backlog() != want && Instant::now() < until {
        thread::sleep(Duration::from_millis(2));
    }
    svc.backlog()
}

#[test]
fn backlog_counts_up_on_submit_and_is_zero_after_the_hand_over() {
    let (svc, jobs_rx) = start_latched();
    assert_eq!(svc.backlog(), 0);
    svc.submit_bulk(rows(0, 4));
    assert_eq!(svc.backlog(), 4);
    svc.submit_bulk(rows(4, 6));
    assert_eq!(svc.backlog(), 10);
    // Embedded and not handed over (the queue is full): still counted.
    assert_eq!(svc.embed_wait(vec!["after both".to_string()], secs(5)).map(|v| v.len()), Some(1));
    assert_eq!(svc.backlog(), 10);

    assert!(matches!(jobs_rx.recv_timeout(Duration::from_secs(5)), Ok(Job::Tick)));
    assert_eq!(embedded(&jobs_rx).len(), 4);
    assert_eq!(embedded(&jobs_rx).len(), 6);
    assert_eq!(backlog_reaches(&svc, 0), 0);
}

#[test]
fn backlog_is_zero_when_the_indexer_is_gone() {
    let (svc, jobs_rx) = start(TestHooks::default());
    drop(jobs_rx);
    svc.submit_bulk(rows(0, 5));
    assert_eq!(backlog_reaches(&svc, 0), 0);
}

#[test]
fn backlog_is_zero_after_the_threads_panic() {
    let (svc, _jobs) = start(TestHooks { panic_embed: true, ..Default::default() });
    svc.submit_bulk(rows(0, 8));
    let until = Instant::now() + Duration::from_secs(2);
    while !svc.stopped() && Instant::now() < until {
        thread::sleep(Duration::from_millis(2));
    }
    assert!(svc.stopped());
    assert_eq!(svc.backlog(), 0);
    // A submit to the dead thread is not counted either.
    svc.submit_bulk(rows(8, 3));
    assert_eq!(svc.backlog(), 0);
}
