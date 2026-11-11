//! Commit 9: the embed thread (`EmbedService`). Hash embedder only; every wait is bounded.

use std::sync::mpsc::{sync_channel, Receiver, TryRecvError};
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
    let (svc, jobs_rx) = start(TestHooks::default());
    let texts = big_texts(64, 1024 * 1024);
    let items: Vec<(VecKey, String)> = texts.iter().enumerate().map(|(i, t)| (VecKey::Mem(i as i64), t.clone())).collect();
    let t0 = Instant::now();
    svc.submit_bulk(items);
    // Let the thread take the bulk and start on it, so the query arrives behind it.
    thread::sleep(Duration::from_millis(20));

    let t1 = Instant::now();
    let q = svc.embed_query("retry limit", secs(30)).expect("the query is answered");
    let query_took = t1.elapsed();
    assert_eq!(q, hashed("retry limit"));
    // Had the query waited its turn, the bulk's `Job::Embedded` would be on the queue already:
    // the thread sends it before it turns to the next request.
    assert!(
        matches!(jobs_rx.try_recv(), Err(TryRecvError::Empty)),
        "the query was served only after the whole bulk"
    );

    match jobs_rx.recv_timeout(Duration::from_secs(60)) {
        Ok(Job::Embedded { model, vectors }) => {
            // The query waits for one batch of 4 at most; the bulk takes 16 of them.
            let bulk_took = t0.elapsed();
            assert!(bulk_took > Duration::from_millis(100), "the bulk was too quick to test: {bulk_took:?}");
            assert!(query_took * 4 < bulk_took, "query {query_took:?}, bulk {bulk_took:?}");
            assert_eq!(model, Embedder::hashing().model_id());
            assert_eq!(vectors.len(), 64);
            for (i, (key, v)) in vectors.iter().enumerate() {
                assert_eq!(*key, VecKey::Mem(i as i64));
                assert_eq!(*v, hashed(&texts[i]));
            }
        }
        other => panic!("expected Job::Embedded, got {other:?}"),
    }
}
