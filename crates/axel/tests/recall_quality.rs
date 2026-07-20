//! End-to-end, labeled-corpus quality gate for continuous-memory retrieval.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use axel::chat_history::generate_candidates;
use axel::project_memory::{self, MemoryScope, RecordLinks, Retention, Sensitivity};
use axel::r8::Brain;
use axel::retrieval::{rank_and_select_with_metrics, MAX_SELECTED_CAP};
use axel_memkoshi::memory::{Memory, MemoryCategory};
use rusqlite::Connection;
use serde::Deserialize;
use tempfile::TempDir;

// The measured context is exactly each selected descriptor's title plus its
// optional snippet. This fixed budget is deliberately below downstream model
// context sizes and selection is additionally required to obey MAX_SELECTED_CAP.
const CONTEXT_BUDGET_BYTES: usize = 8 * 1024;

#[derive(Deserialize)]
struct Corpus {
    records: Vec<Record>,
    queries: Vec<Query>,
}

#[derive(Deserialize)]
struct Record {
    id: String,
    project: String,
    session: String,
    class: String,
    title: String,
    content: String,
    trap: String,
    #[serde(default)]
    superseded_by: Option<String>,
    #[serde(default)]
    sensitivity: Option<String>,
    #[serde(default)]
    retention: Option<String>,
}

#[derive(Deserialize)]
struct Query {
    project: String,
    session: String,
    text: String,
    expected_ids: Vec<String>,
}

fn setup() -> (TempDir, Brain) {
    let dir = TempDir::new().unwrap();
    let brain = Brain::create(&dir.path().join("quality.r8"), Some("recall-quality")).unwrap();
    project_memory::migrate(brain.conn()).unwrap();
    (dir, brain)
}

fn store(conn: &Connection, row: &Record) {
    let mut memory = Memory::new(
        MemoryCategory::Events,
        "recall-quality",
        row.title.clone(),
        row.content.clone(),
    );
    memory.id = row.id.clone();
    memory.tags.push(format!("session:{}", row.session));
    let scope = MemoryScope {
        project_key: row.project.clone(),
        sensitivity: if row.sensitivity.as_deref() == Some("secret") {
            Sensitivity::Secret
        } else {
            Sensitivity::Normal
        },
        retention: if row.retention.as_deref() == Some("local_only") {
            Retention::LocalOnly
        } else {
            Retention::Standard
        },
        provenance: "checked-in-recall-quality-corpus".into(),
    };
    let links = RecordLinks {
        superseded_by: row.superseded_by.clone(),
        ..RecordLinks::default()
    };
    project_memory::store_scoped_typed_with_links(conn, &memory, &scope, &row.class, &links)
        .unwrap();
}

fn percentile(sorted: &[u128], percentile: usize) -> u128 {
    sorted[(sorted.len() - 1) * percentile / 100]
}

#[test]
fn labeled_corpus_meets_recall_and_safety_quality_gates() {
    let corpus: Corpus =
        serde_json::from_str(include_str!("fixtures/recall_quality.json")).unwrap();
    let (_dir, brain) = setup();
    for row in &corpus.records {
        store(brain.conn(), row);
    }

    let traps: HashMap<&str, &str> = corpus
        .records
        .iter()
        .map(|record| (record.id.as_str(), record.trap.as_str()))
        .collect();
    let mut hits = [0usize; 3];
    let mut reciprocal_rank = 0.0;
    let mut selected_total = 0usize;
    let mut duplicate_selected = 0usize;
    let mut stale_selected = 0usize;
    let mut secret_leakage = 0usize;
    let mut cross_project_leakage = 0usize;
    let mut budget_violations = 0usize;
    let mut latencies = Vec::with_capacity(corpus.queries.len());

    for query in &corpus.queries {
        let started = Instant::now();
        let candidates = generate_candidates(
            brain.conn(),
            &query.project,
            Some(&query.session),
            Some(&query.text),
            128,
        )
        .unwrap();
        let selection = rank_and_select_with_metrics(candidates, Some(&query.session), 8);
        latencies.push(started.elapsed().as_micros());

        let selected = &selection.records;
        selected_total += selected.len();
        stale_selected += selection.metrics.stale_selected;
        let mut seen_ids = HashSet::new();
        let mut seen_titles = HashSet::new();
        let mut context_bytes = 0usize;
        for (descriptor, _) in selected {
            context_bytes += descriptor.title.len()
                + descriptor
                    .snippet
                    .as_ref()
                    .map_or(0, |snippet| snippet.len());
            if !seen_ids.insert(&descriptor.id)
                || !seen_titles.insert(descriptor.title.trim().to_ascii_lowercase())
            {
                duplicate_selected += 1;
            }
            match traps.get(descriptor.id.as_str()).copied() {
                Some("secret") => secret_leakage += 1,
                Some("cross_project") => cross_project_leakage += 1,
                _ => {}
            }
            if descriptor.project_key != query.project {
                cross_project_leakage += 1;
            }
        }
        if selected.len() > MAX_SELECTED_CAP || context_bytes > CONTEXT_BUDGET_BYTES {
            budget_violations += 1;
        }

        if let Some(rank) = selected
            .iter()
            .position(|(descriptor, _)| query.expected_ids.iter().any(|id| id == &descriptor.id))
        {
            reciprocal_rank += 1.0 / (rank + 1) as f64;
            for (index, cutoff) in [1usize, 5, 8].iter().enumerate() {
                if rank < *cutoff {
                    hits[index] += 1;
                }
            }
        }
    }

    latencies.sort_unstable();
    let count = corpus.queries.len() as f64;
    let recall = hits.map(|hit| hit as f64 / count);
    let mrr = reciprocal_rank / count;
    let stale_rate = stale_selected as f64 / selected_total.max(1) as f64;
    let duplicate_rate = duplicate_selected as f64 / selected_total.max(1) as f64;
    println!(
        "{{\"recall_at_1\":{:.6},\"recall_at_5\":{:.6},\"recall_at_8\":{:.6},\"mrr\":{:.6},\"duplicate_selected_rate\":{:.6},\"stale_superseded_selected_rate\":{:.6},\"secret_leakage_count\":{},\"cross_project_leakage_count\":{},\"context_budget_violations\":{},\"latency_p50_us\":{},\"latency_p95_us\":{}}}",
        recall[0], recall[1], recall[2], mrr, duplicate_rate, stale_rate,
        secret_leakage, cross_project_leakage, budget_violations,
        percentile(&latencies, 50), percentile(&latencies, 95)
    );

    assert!(recall[1] >= 0.85, "recall@5 {0}", recall[1]);
    assert!(stale_rate <= 0.05, "stale rate {stale_rate}");
    assert!(duplicate_rate <= 0.10, "duplicate rate {duplicate_rate}");
    assert_eq!(secret_leakage, 0, "secret records leaked");
    assert_eq!(cross_project_leakage, 0, "cross-project records leaked");
    assert_eq!(budget_violations, 0, "context budget violated");
}
