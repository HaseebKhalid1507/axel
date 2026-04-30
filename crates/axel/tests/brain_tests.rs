use axel::brain::AxelBrain;
use tempfile::TempDir;

fn setup_test_brain() -> (AxelBrain, TempDir) {
    let temp_dir = tempfile::tempdir().unwrap();
    let brain_path = temp_dir.path().join("test.r8");
    let brain = AxelBrain::open_or_create(&brain_path, Some("test-agent")).unwrap();
    (brain, temp_dir)
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