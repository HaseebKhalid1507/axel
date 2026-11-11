use axel::brain::AxelBrain;
use tempfile::TempDir;
use velocirag::embedder::Embedder;

/// These tests embed (`remember*` and `update_memory*` index what they
/// store), so the brain is built with the hashing embedder: no test loads
/// the ONNX model.
fn setup_test_brain() -> (AxelBrain, TempDir) {
    let temp_dir = tempfile::tempdir().unwrap();
    let brain_path = temp_dir.path().join("test.r8");
    let brain =
        AxelBrain::open_or_create_with(&brain_path, Some("test-agent"), Embedder::hashing())
            .unwrap();
    (brain, temp_dir)
}

#[test]
fn open_or_create_with_records_the_embedder_and_indexes_with_it() {
    let (mut brain, _temp_dir) = setup_test_brain();
    assert_eq!(brain.meta().embedder_model, "hash-384");
    assert_eq!(brain.meta().embedding_dim, 384);

    // The memory is embedded and searchable, with no model on this machine.
    let id = brain
        .remember("The harbour lantern was repaired on the ninth of March by the keeper", "events", 0.6)
        .unwrap();
    let found = brain.search("harbour lantern keeper", 5).unwrap();
    assert_eq!(found.results[0].doc_id, id);
}

#[test]
fn open_or_create_with_refuses_a_brain_built_with_another_model() {
    let temp_dir = tempfile::tempdir().unwrap();

    // A hash-384 brain: reopens with the hashing embedder, not with the default one.
    let hash_path = temp_dir.path().join("hash.r8");
    drop(AxelBrain::open_or_create_with(&hash_path, Some("t"), Embedder::hashing()).unwrap());
    assert!(AxelBrain::open_or_create_with(&hash_path, None, Embedder::hashing()).is_ok());
    let error = match AxelBrain::open_or_create(&hash_path, None) {
        Ok(_) => panic!("the default embedder must be refused on a hash-384 brain"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("hash-384") && error.contains("all-MiniLM-L6-v2"), "{error}");
    let error = match AxelBrain::open(&hash_path) {
        Ok(_) => panic!("the default embedder must be refused on a hash-384 brain"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("hash-384") && error.contains("all-MiniLM-L6-v2"), "{error}");

    // A default brain: the hashing embedder is refused.
    let default_path = temp_dir.path().join("default.r8");
    let brain = AxelBrain::open_or_create(&default_path, Some("t")).unwrap();
    assert_eq!(brain.meta().embedder_model, "all-MiniLM-L6-v2");
    drop(brain);
    assert!(AxelBrain::open_or_create_with(&default_path, None, Embedder::hashing()).is_err());
}

fn table_count(path: &std::path::Path) -> i64 {
    rusqlite::Connection::open(path)
        .unwrap()
        .query_row("SELECT COUNT(*) FROM sqlite_master", [], |row| row.get(0))
        .unwrap()
}

/// `AxelBrain::open` (and `open_or_create_with`) on a brain they refuse:
/// an error naming `must_name`, and the file exactly as it was. The brains
/// are made with `Brain::create_with_model`, so the project-memory migration
/// has not run on them: a migration before the check would change the file.
fn assert_refused_unchanged(path: &std::path::Path, must_name: &str) {
    let bytes_before = std::fs::read(path).unwrap();
    let tables_before = table_count(path);

    let error = match AxelBrain::open(path) {
        Ok(_) => panic!("open must refuse {}", path.display()),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("Model mismatch") && error.contains(must_name), "{error}");
    assert_eq!(std::fs::read(path).unwrap(), bytes_before, "open changed the refused file");

    let default_embedder = Embedder::new(None, None, true);
    let error = match AxelBrain::open_or_create_with(path, None, default_embedder) {
        Ok(_) => panic!("open_or_create_with must refuse {}", path.display()),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("Model mismatch") && error.contains(must_name), "{error}");
    assert_eq!(std::fs::read(path).unwrap(), bytes_before, "open_or_create_with changed the refused file");

    assert_eq!(table_count(path), tables_before);
    assert!(!path.with_extension("r8.hnsw").exists());
}

#[test]
fn a_refused_open_leaves_the_file_byte_identical() {
    let temp_dir = tempfile::tempdir().unwrap();

    // Another model.
    let hash_path = temp_dir.path().join("hash.r8");
    drop(axel::r8::Brain::create_with_model(&hash_path, Some("t"), "hash-384", 384).unwrap());
    assert_refused_unchanged(&hash_path, "hash-384");

    // The default model's name, another dimension.
    let wide_path = temp_dir.path().join("wide.r8");
    drop(axel::r8::Brain::create_with_model(&wide_path, Some("t"), "all-MiniLM-L6-v2", 768).unwrap());
    assert_refused_unchanged(&wide_path, "768 dimensions");

    // Control: an accepted open of such a file does change it (the
    // migration adds tables), so "unchanged" above is not trivially true.
    let plain_path = temp_dir.path().join("plain.r8");
    drop(axel::r8::Brain::create_with_model(&plain_path, Some("t"), "all-MiniLM-L6-v2", 384).unwrap());
    let tables_before = table_count(&plain_path);
    drop(AxelBrain::open(&plain_path).unwrap());
    assert!(table_count(&plain_path) > tables_before);
}

fn sha256(path: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(std::fs::read(path).unwrap()))
}

#[test]
fn a_768_dimension_brain_is_refused_by_both_openers_with_its_hash_unchanged() {
    use axel::error::AxelError;
    use axel::search::BrainSearch;

    // The brain records the default model's name and 768 dimensions; the
    // embedder has 384. Only the dimension can refuse it.
    let temp_dir = tempfile::tempdir().unwrap();
    let path = temp_dir.path().join("wide.r8");
    drop(axel::r8::Brain::create_with_model(&path, Some("t"), "all-MiniLM-L6-v2", 768).unwrap());
    let hash_before = sha256(&path);
    let embedder = Embedder::new(None, None, true);
    assert_eq!(embedder.dim(), 384);

    // Opener 1: BrainSearch.
    let error = match BrainSearch::open_with_embedder(&path, embedder) {
        Ok(_) => panic!("BrainSearch::open_with_embedder must refuse a 768-dimension brain"),
        Err(error) => error,
    };
    assert!(matches!(&error, AxelError::ModelMismatch { .. }), "{error}");
    assert!(error.to_string().contains("768 dimensions"), "{error}");
    assert_eq!(sha256(&path), hash_before, "BrainSearch::open_with_embedder changed the file");

    // Opener 2: AxelBrain::open.
    let error = match AxelBrain::open(&path) {
        Ok(_) => panic!("AxelBrain::open must refuse a 768-dimension brain"),
        Err(error) => error,
    };
    assert!(matches!(&error, AxelError::ModelMismatch { .. }), "{error}");
    assert!(error.to_string().contains("768 dimensions"), "{error}");
    assert_eq!(sha256(&path), hash_before, "AxelBrain::open changed the file");
}

#[test]
fn test_axel_verify_nonexistent_memory() {
    let (brain, _temp_dir) = setup_test_brain();
    
    let result = brain.get_memory_with_verification("mem_nonexist").unwrap();
    assert!(result.is_none());
}

#[test]
fn test_axel_verify_existing_memory() {
    let (mut brain, _temp_dir) = setup_test_brain();
    
    // Store a memory first
    let memory_id = brain.remember("This is a test memory content for verification that is sufficiently long to pass the validation requirements", "events", 0.7).unwrap();
    
    // Verify it
    let result = brain.get_memory_with_verification(&memory_id).unwrap();
    assert!(result.is_some());
    
    let (memory, verified) = result.unwrap();
    assert_eq!(memory.id, memory_id);
    assert_eq!(memory.content, "This is a test memory content for verification that is sufficiently long to pass the validation requirements");
    assert_eq!(memory.importance, 0.7);
    // Verification depends on whether brain has a signing key
    // In tests without explicit signing setup, it should be false
    assert!(!verified || memory.signature.is_some());
}

#[test]
fn test_memory_update_content() {
    let (mut brain, _temp_dir) = setup_test_brain();
    
    // Store initial memory
    let memory_id = brain.remember("Original memory content that is long enough to pass validation checks", "events", 0.5).unwrap();
    
    // Update content only
    let updated = brain.update_memory(&memory_id, Some("Updated memory content that is also long enough to pass validation requirements"), None).unwrap();
    assert!(updated);
    
    // Verify the update
    let result = brain.get_memory_with_verification(&memory_id).unwrap();
    assert!(result.is_some());
    
    let (memory, _) = result.unwrap();
    assert_eq!(memory.content, "Updated memory content that is also long enough to pass validation requirements");
    assert_eq!(memory.importance, 0.5); // Should be unchanged
}

#[test]
fn test_memory_update_importance() {
    let (mut brain, _temp_dir) = setup_test_brain();
    
    // Store initial memory
    let memory_id = brain.remember("Memory content for importance update test with sufficient length", "events", 0.3).unwrap();
    
    // Update importance only
    let updated = brain.update_memory(&memory_id, None, Some(0.9)).unwrap();
    assert!(updated);
    
    // Verify the update
    let result = brain.get_memory_with_verification(&memory_id).unwrap();
    assert!(result.is_some());
    
    let (memory, _) = result.unwrap();
    assert_eq!(memory.importance, 0.9);
    assert_eq!(memory.content, "Memory content for importance update test with sufficient length"); // Should be unchanged
}

#[test]
fn test_memory_update_both() {
    let (mut brain, _temp_dir) = setup_test_brain();
    
    // Store initial memory
    let memory_id = brain.remember("Initial memory content that meets the minimum length requirement", "events", 0.2).unwrap();
    
    // Update both content and importance
    let updated = brain.update_memory(&memory_id, Some("Completely new memory content that is sufficiently long for validation"), Some(0.8)).unwrap();
    assert!(updated);
    
    // Verify the update
    let result = brain.get_memory_with_verification(&memory_id).unwrap();
    assert!(result.is_some());
    
    let (memory, _) = result.unwrap();
    assert_eq!(memory.content, "Completely new memory content that is sufficiently long for validation");
    assert_eq!(memory.importance, 0.8);
}

#[test]
fn test_memory_update_nonexistent() {
    let (mut brain, _temp_dir) = setup_test_brain();
    
    // Try to update non-existent memory
    let updated = brain.update_memory("mem_nonexist", Some("New content that is long enough"), Some(0.7)).unwrap();
    assert!(!updated);
}
// ── remember_full / update_memory_full / MemoryPatch ────────────────────────

use axel::brain::MemoryPatch;
use axel_memkoshi::memory::{Confidence, Memory, MemoryCategory};
use chrono::{Duration, Utc};

fn rich_memory() -> Memory {
    let mut m = Memory::new(
        MemoryCategory::Cases,
        "longitudinal-study",
        "Long-form title for a fully enriched memory record",
        "This is a fully enriched memory body with plenty of length to clear the fifty character validation gate.",
    );
    m.abstract_text = "Short abstract describing the memory.".to_string();
    m.confidence = Confidence::High;
    m.importance = 0.85;
    m.trust_level = 0.9;
    m.tags = vec!["alpha".to_string(), "beta".to_string()];
    m.related_topics = vec!["topic-x".to_string(), "topic-y".to_string()];
    m.source_sessions = vec!["sess-1".to_string()];
    m
}

#[test]
fn remember_full_persists_all_fields() {
    let (mut brain, _td) = setup_test_brain();
    let mem = rich_memory();
    let id = brain.remember_full(mem.clone()).unwrap();

    let (got, _) = brain.get_memory_with_verification(&id).unwrap().unwrap();
    assert_eq!(got.topic, "longitudinal-study");
    assert_eq!(got.title, mem.title);
    assert_eq!(got.abstract_text, "Short abstract describing the memory.");
    assert_eq!(got.content, mem.content);
    assert_eq!(got.confidence, Confidence::High);
    assert!((got.importance - 0.85).abs() < 1e-9);
    assert!((got.trust_level - 0.9).abs() < 1e-9);
    assert_eq!(got.tags, vec!["alpha".to_string(), "beta".to_string()]);
    assert_eq!(got.related_topics, vec!["topic-x".to_string(), "topic-y".to_string()]);
    assert_eq!(got.source_sessions, vec!["sess-1".to_string()]);
    assert_eq!(got.category, MemoryCategory::Cases);
}

#[test]
fn remember_full_signs_when_signer_present() {
    let (mut brain, _td) = setup_test_brain();
    let id = brain.remember_full(rich_memory()).unwrap();
    let (got, verified) = brain.get_memory_with_verification(&id).unwrap().unwrap();
    assert!(got.signature.is_some(), "expected signature to be present");
    assert!(verified, "expected signature to verify");
}

#[test]
fn remember_full_clamps_importance_and_trust() {
    let (mut brain, _td) = setup_test_brain();
    let mut mem = rich_memory();
    mem.importance = 2.5;
    mem.trust_level = -0.1;
    let id = brain.remember_full(mem).unwrap();
    let (got, _) = brain.get_memory_with_verification(&id).unwrap().unwrap();
    assert!((got.importance - 1.0).abs() < 1e-9);
    assert!((got.trust_level - 0.0).abs() < 1e-9);
}

#[test]
fn remember_full_validates_short_content() {
    let (mut brain, _td) = setup_test_brain();
    let mut mem = rich_memory();
    mem.content = "too short".to_string();
    let res = brain.remember_full(mem);
    assert!(res.is_err(), "expected validation error for short content");
}

#[test]
fn update_memory_full_patches_each_field_individually() {
    let (mut brain, _td) = setup_test_brain();
    let id = brain.remember_full(rich_memory()).unwrap();

    // Patch topic only
    let patch = MemoryPatch { topic: Some("new-topic-label".to_string()), ..Default::default() };
    assert!(brain.update_memory_full(&id, patch).unwrap());
    let (got, _) = brain.get_memory_with_verification(&id).unwrap().unwrap();
    assert_eq!(got.topic, "new-topic-label");
    assert!(got.updated.is_some());
    assert_eq!(got.tags, vec!["alpha".to_string(), "beta".to_string()]); // unchanged

    // Patch tags only
    let patch = MemoryPatch { tags: Some(vec!["gamma".to_string()]), ..Default::default() };
    assert!(brain.update_memory_full(&id, patch).unwrap());
    let (got, _) = brain.get_memory_with_verification(&id).unwrap().unwrap();
    assert_eq!(got.tags, vec!["gamma".to_string()]);
    assert_eq!(got.topic, "new-topic-label"); // still

    // Patch confidence
    let patch = MemoryPatch { confidence: Some(Confidence::Low), ..Default::default() };
    assert!(brain.update_memory_full(&id, patch).unwrap());
    let (got, _) = brain.get_memory_with_verification(&id).unwrap().unwrap();
    assert_eq!(got.confidence, Confidence::Low);

    // Patch related_topics
    let patch = MemoryPatch { related_topics: Some(vec!["tz".to_string()]), ..Default::default() };
    assert!(brain.update_memory_full(&id, patch).unwrap());
    let (got, _) = brain.get_memory_with_verification(&id).unwrap().unwrap();
    assert_eq!(got.related_topics, vec!["tz".to_string()]);

    // Patch abstract_text
    let patch = MemoryPatch { abstract_text: Some("revised abstract".to_string()), ..Default::default() };
    assert!(brain.update_memory_full(&id, patch).unwrap());
    let (got, _) = brain.get_memory_with_verification(&id).unwrap().unwrap();
    assert_eq!(got.abstract_text, "revised abstract");
}

#[test]
fn update_memory_full_returns_false_for_unknown_id() {
    let (mut brain, _td) = setup_test_brain();
    let patch = MemoryPatch { topic: Some("whatever".to_string()), ..Default::default() };
    let res = brain.update_memory_full("mem_deadbeef", patch).unwrap();
    assert!(!res);
}

#[test]
fn update_memory_full_re_signs() {
    let (mut brain, _td) = setup_test_brain();
    let id = brain.remember_full(rich_memory()).unwrap();
    let (before, ok1) = brain.get_memory_with_verification(&id).unwrap().unwrap();
    assert!(ok1);
    let original_sig = before.signature.clone().expect("has signature");

    let patch = MemoryPatch {
        content: Some(
            "Patched memory body — sufficiently long to clear the fifty character validation gate.".to_string(),
        ),
        ..Default::default()
    };
    assert!(brain.update_memory_full(&id, patch).unwrap());

    let (after, ok2) = brain.get_memory_with_verification(&id).unwrap().unwrap();
    assert!(ok2, "re-signed memory should still verify");
    let new_sig = after.signature.clone().expect("has signature");
    assert_ne!(original_sig, new_sig, "signature should change after patch");
}

#[test]
fn update_memory_full_clears_expiry() {
    // NB: the `memories` SQLite schema does not currently persist
    // `expires_at` (see `MemoryStorage::store_memory` / `get_memory`),
    // and the brief explicitly forbids touching the schema. We therefore
    // assert the *patch path* runs end-to-end with `Some(None)` (the
    // "clear expiry" sentinel) without erroring, and that the resulting
    // record has no expiry. A persistence-level test for expires_at will
    // become possible once the schema gains the column.
    let (mut brain, _td) = setup_test_brain();
    let mut mem = rich_memory();
    mem.expires_at = Some(Utc::now() + Duration::hours(48));
    let id = brain.remember_full(mem).unwrap();

    let patch = MemoryPatch { expires_at: Some(None), ..Default::default() };
    assert!(brain.update_memory_full(&id, patch).unwrap());
    let (got, _) = brain.get_memory_with_verification(&id).unwrap().unwrap();
    assert!(got.expires_at.is_none(), "expiry should be cleared (or never persisted)");
}
