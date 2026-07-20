//! Chat-history recall scale benchmark (continuous-memory task F3).
//!
//! Run serially in release mode:
//!
//! ```bash
//! cargo test -p axel --test recall_bench --release -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The default run covers 1K, 10K, and 100K episodic records. Set
//! `AXEL_BENCH_1M=1` to include the resource-intensive 1M corpus. Each JSON
//! line is a complete, machine-readable result for one independently created
//! corpus. Query latency covers the real bounded chat-history path:
//! candidate generation, stale/superseded filtering, diverse ranking, and
//! selection. `records_scanned` is the number of bounded descriptors crossing
//! the candidate-generator boundary; SQLite does not expose FTS row visits
//! through that API.

use std::fs;
use std::time::{Duration, Instant};

use axel::chat_history::{
    generate_candidates, Decision, EntityFact, EpisodicRecord, EpisodicTurn, Preference,
    UnresolvedTask,
};
use axel::project_memory::{self, MemoryScope, Retention, Sensitivity};
use axel::r8::Brain;
use axel::retrieval::{rank_and_select_with_metrics, MAX_SELECTED_CAP};
use axel_memkoshi::memory::{Memory, MemoryCategory};
use serde_json::json;
use tempfile::TempDir;

const PROJECT_KEY: &str = "recall-bench-project";
const QUERY: &str = "recallneedle benchmark";
const INGEST_BATCH_RECORDS: usize = 5_000;
const WARMUP_QUERIES: usize = 10;
const MEASURED_QUERIES: usize = 100;
const P95_TARGET_100K: Duration = Duration::from_millis(100);

fn episodic_record(index: usize) -> EpisodicRecord {
    let session = Some(format!("session-{}", index % 16));
    let turn = Some(format!("turn-{index}"));
    let text = format!(
        "recallneedle benchmark project chat history record {index} records a bounded retrieval fact"
    );
    match index % 5 {
        0 => EpisodicRecord::Turn(EpisodicTurn {
            content: text,
            source_session_id: session,
            source_turn_id: turn,
        }),
        1 => EpisodicRecord::Decision(Decision {
            decision: text,
            rationale: Some("measured chat-history decision".into()),
            source_session_id: session,
            source_turn_id: turn,
        }),
        2 => EpisodicRecord::Preference(Preference {
            preference: text,
            source_session_id: session,
            source_turn_id: turn,
        }),
        3 => EpisodicRecord::UnresolvedTask(UnresolvedTask {
            task: text,
            last_known_state: Some("awaiting benchmark recall".into()),
            source_session_id: session,
            source_turn_id: turn,
        }),
        _ => EpisodicRecord::EntityFact(EntityFact {
            entity: format!("bench-entity-{index}"),
            relationship: text,
            source_session_id: session,
            source_turn_id: turn,
        }),
    }
}

fn corpus_row(index: usize) -> (Memory, MemoryScope) {
    let record = episodic_record(index);
    let class = record.class();
    let mut memory = Memory::new(
        MemoryCategory::Events,
        class,
        format!("{class} chat record {index}"),
        serde_json::to_string(&record).unwrap(),
    );
    memory.id = format!("mem_{:016x}", index + 1);
    memory.abstract_text = format!("{QUERY} record {index}");
    memory.tags = vec![format!("session:session-{}", index % 16)];
    (
        memory,
        MemoryScope {
            project_key: PROJECT_KEY.into(),
            sensitivity: Sensitivity::Normal,
            retention: Retention::Standard,
            provenance: "recall-bench:synthetic-chat-history".into(),
        },
    )
}

fn resident_bytes(field: &str) -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with(field))?;
    let kib = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(kib.saturating_mul(1024))
}

fn descriptor_bytes(candidate: &axel::chat_history::RecallCandidate) -> usize {
    let descriptor = &candidate.descriptor;
    descriptor.id.len()
        + descriptor.project_key.len()
        + descriptor.title.len()
        + descriptor.snippet.as_ref().map_or(0, String::len)
        + descriptor.category.len()
        + descriptor.tags.iter().map(String::len).sum::<usize>()
        + descriptor.created.len()
        + descriptor.provenance.len()
}

fn selected_bytes(
    records: &[(
        axel::project_memory::ScopedDescriptor,
        Vec<axel::retrieval::RankReason>,
    )],
) -> usize {
    records
        .iter()
        .map(|(descriptor, _)| {
            descriptor.title.len() + descriptor.snippet.as_ref().map_or(0, String::len)
        })
        .sum()
}

fn percentile(sorted: &[Duration], percentile: usize) -> Duration {
    let rank = (sorted.len() * percentile).div_ceil(100);
    sorted[rank.saturating_sub(1)]
}

fn duration_micros(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

fn bench_size(records: usize) {
    let directory = TempDir::new().unwrap();
    let brain = Brain::create(
        &directory.path().join("recall-bench.r8"),
        Some("recall-bench"),
    )
    .unwrap();
    project_memory::migrate(brain.conn()).unwrap();

    let mut peak_resident_bytes = resident_bytes("VmRSS:").unwrap_or(0);
    let ingest_started = Instant::now();
    for start in (0..records).step_by(INGEST_BATCH_RECORDS) {
        let end = (start + INGEST_BATCH_RECORDS).min(records);
        let batch: Vec<_> = (start..end).map(corpus_row).collect();
        project_memory::store_scoped_batch(brain.conn(), &batch).unwrap();
        peak_resident_bytes = peak_resident_bytes.max(resident_bytes("VmRSS:").unwrap_or(0));
    }
    // The batch API stores the coarse enum category. Preserve each typed
    // chat-history class carried in `topic`, matching the capture path's
    // descriptor shape before exercising recall.
    brain
        .conn()
        .execute(
            "UPDATE memories SET category = topic WHERE project_key = ?1",
            [PROJECT_KEY],
        )
        .unwrap();
    let ingest_elapsed = ingest_started.elapsed();

    for _ in 0..WARMUP_QUERIES {
        let candidates = generate_candidates(
            brain.conn(),
            PROJECT_KEY,
            Some("session-7"),
            Some(QUERY),
            128,
        )
        .unwrap();
        let selection =
            rank_and_select_with_metrics(candidates, Some("session-7"), MAX_SELECTED_CAP);
        assert!(!selection.records.is_empty());
    }

    let mut latencies = Vec::with_capacity(MEASURED_QUERIES);
    let mut records_scanned = 0usize;
    let mut candidates_retained = 0usize;
    let mut selected_records = 0usize;
    let mut bytes_retained = 0usize;
    let mut candidate_bytes_retained = 0usize;
    for _ in 0..MEASURED_QUERIES {
        let started = Instant::now();
        let candidates = generate_candidates(
            brain.conn(),
            PROJECT_KEY,
            Some("session-7"),
            Some(QUERY),
            128,
        )
        .unwrap();
        records_scanned = records_scanned.max(candidates.len());
        candidates_retained = candidates_retained.max(candidates.len());
        candidate_bytes_retained =
            candidate_bytes_retained.max(candidates.iter().map(descriptor_bytes).sum::<usize>());
        let selection =
            rank_and_select_with_metrics(candidates, Some("session-7"), MAX_SELECTED_CAP);
        latencies.push(started.elapsed());
        selected_records = selected_records.max(selection.records.len());
        bytes_retained = bytes_retained.max(selected_bytes(&selection.records));
        assert!(selection.records.len() <= MAX_SELECTED_CAP);
        peak_resident_bytes = peak_resident_bytes.max(resident_bytes("VmRSS:").unwrap_or(0));
    }
    latencies.sort_unstable();
    let p50 = percentile(&latencies, 50);
    let p95 = percentile(&latencies, 95);
    peak_resident_bytes = peak_resident_bytes.max(resident_bytes("VmHWM:").unwrap_or(0));

    let target_status = if records == 100_000 {
        if p95 <= P95_TARGET_100K {
            assert!(p95 <= P95_TARGET_100K);
            "pass"
        } else {
            eprintln!(
                "RECALL_BENCH_100K_P95_TARGET_MISS measured_p95_us={} target_p95_us={}",
                duration_micros(p95),
                duration_micros(P95_TARGET_100K)
            );
            "measured_target_miss"
        }
    } else {
        "not_applicable"
    };

    println!(
        "{}",
        json!({
            "benchmark": "chat_history_recall",
            "records": records,
            "ingest_time_ms": duration_micros(ingest_elapsed) as f64 / 1_000.0,
            "query_p50_us": duration_micros(p50),
            "query_p95_us": duration_micros(p95),
            "records_scanned": records_scanned,
            "candidates_retained": candidates_retained,
            "selected_records": selected_records,
            "bytes_retained": bytes_retained,
            "candidate_bytes_retained": candidate_bytes_retained,
            "peak_resident_bytes": peak_resident_bytes,
            "peak_open_segments": 1,
            "peak_open_segment_records": INGEST_BATCH_RECORDS.min(records),
            "warm_lexical_100k_target_us": duration_micros(P95_TARGET_100K),
            "warm_lexical_100k_status": target_status,
        })
    );
}

#[test]
#[ignore = "release scale benchmark; run explicitly with --ignored --nocapture"]
fn recall_1k_10k_100k_and_optional_1m() {
    let max_records = std::env::var("AXEL_RECALL_BENCH_MAX_RECORDS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100_000);
    for records in [1_000, 10_000, 100_000] {
        if records <= max_records {
            bench_size(records);
        }
    }
    if std::env::var("AXEL_BENCH_1M").as_deref() == Ok("1") {
        bench_size(1_000_000);
    }
}
