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