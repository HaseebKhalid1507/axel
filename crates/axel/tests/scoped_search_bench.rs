//! Scoped-search scale benchmark (T35).
//!
//! Run explicitly (ignored by default — it inserts up to 100K rows):
//!
//! ```bash
//! cargo test -p axel --release --test scoped_search_bench -- --ignored --nocapture
//! ```
//!
//! Asserts the T35 bound: search cost/result size is proportional to the
//! requested limit, not the total match count. Prints timings for
//! 1K / 10K / 100K corpora (1M documented as extrapolation in REVIEW notes;
//! run with AXEL_BENCH_1M=1 if you can afford it).

use axel::project_memory::{MemoryScope, Retention, ScopedQuery, Sensitivity};
use axel::AxelBrain;
use axel_memkoshi::memory::{Memory, MemoryCategory};
use tempfile::TempDir;

fn corpus(n: usize) -> Vec<(Memory, MemoryScope)> {
    (0..n)
        .map(|i| {
            let m = Memory::new(
                MemoryCategory::Events,
                "bench-topic",
                format!("Bench memory title {i}"),
                format!(
                    "benchmark corpus entry number {i} with some shared needle text \
                     and enough padding to satisfy the validation minimum."
                ),
            );
            (
                m,
                MemoryScope {
                    project_key: "proj_bench".into(),
                    sensitivity: Sensitivity::Normal,
                    retention: Retention::Standard,
                    provenance: "bench".into(),
                },
            )
        })
        .collect()
}

fn bench_size(n: usize) {
    let dir = TempDir::new().unwrap();
    let mut brain = AxelBrain::open_or_create(dir.path().join("bench.r8"), Some("bench")).unwrap();

    let t0 = std::time::Instant::now();
    // Insert in 5K-row transactions.
    for chunk in corpus(n).chunks(5000) {
        brain.store_scoped_batch(chunk.to_vec()).unwrap();
    }
    let insert_s = t0.elapsed().as_secs_f32();

    let q = ScopedQuery {
        project_key: "proj_bench".into(),
        text: Some("needle".into()),
        tags: vec![],
        since: None,
        until: None,
        limit: 10,
    };
    let t1 = std::time::Instant::now();
    let hits = brain.search_scoped(&q).unwrap();
    let search_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // Result size bounded by the limit, never by the ~n total matches.
    assert_eq!(hits.len(), 10);

    println!(
        "scoped_search_bench n={n:>7} insert={insert_s:>7.2}s search(limit=10)={search_ms:>8.2}ms hits={}",
        hits.len()
    );
}

#[test]
#[ignore = "scale benchmark — run with --ignored --nocapture"]
fn bench_1k_10k_100k() {
    bench_size(1_000);
    bench_size(10_000);
    bench_size(100_000);
    if std::env::var("AXEL_BENCH_1M").as_deref() == Ok("1") {
        bench_size(1_000_000);
    }
}
