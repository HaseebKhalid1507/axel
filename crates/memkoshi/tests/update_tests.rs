use axel_memkoshi::memory::{Memory, MemoryCategory};
use axel_memkoshi::storage::MemoryStorage;
use tempfile::TempDir;

fn setup_test_storage() -> (MemoryStorage, TempDir) {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let storage = MemoryStorage::open(&db_path).unwrap();
    (storage, temp_dir)
}

#[test]
fn test_update_memory_content() {
    let (storage, _temp_dir) = setup_test_storage();
    
    // Create and store a memory
    let mut memory = Memory::new(
        MemoryCategory::Events,
        "test topic",
        "Test Memory Title for Update",
        "Original memory content that is long enough to pass validation checks and requirements",
    );
    memory.importance = 0.5;
    storage.store_memory(&memory).unwrap();
    
    // Update content
    let updated = storage.update_memory(&memory.id, Some("Updated memory content that is also long enough to pass validation requirements"), None).unwrap();
    assert!(updated);
    
    // Verify update
    let retrieved = storage.get_memory(&memory.id).unwrap().unwrap();
    assert_eq!(retrieved.content, "Updated memory content that is also long enough to pass validation requirements");
    assert_eq!(retrieved.importance, 0.5); // Should be unchanged
    assert!(retrieved.updated.is_some()); // Should have updated timestamp
}

#[test]
fn test_update_memory_importance() {
    let (storage, _temp_dir) = setup_test_storage();
    
    // Create and store a memory
    let mut memory = Memory::new(
        MemoryCategory::Events,
        "test topic",
        "Test Memory Title for Importance Update",
        "Memory content for importance update test with sufficient length to meet requirements",
    );
    memory.importance = 0.3;
    storage.store_memory(&memory).unwrap();
    
    // Update importance
    let updated = storage.update_memory(&memory.id, None, Some(0.9)).unwrap();
    assert!(updated);
    
    // Verify update
    let retrieved = storage.get_memory(&memory.id).unwrap().unwrap();
    assert_eq!(retrieved.importance, 0.9);
    assert_eq!(retrieved.content, "Memory content for importance update test with sufficient length to meet requirements"); // Should be unchanged
    assert!(retrieved.updated.is_some()); // Should have updated timestamp
}

#[test]
fn test_update_memory_both() {
    let (storage, _temp_dir) = setup_test_storage();
    
    // Create and store a memory
    let mut memory = Memory::new(
        MemoryCategory::Events,
        "test topic",
        "Test Memory Title for Full Update",
        "Initial memory content that meets the minimum length requirement for validation",
    );
    memory.importance = 0.2;
    storage.store_memory(&memory).unwrap();
    
    // Update both
    let updated = storage.update_memory(&memory.id, Some("Completely new memory content that is sufficiently long for validation requirements"), Some(0.8)).unwrap();
    assert!(updated);
    
    // Verify update
    let retrieved = storage.get_memory(&memory.id).unwrap().unwrap();
    assert_eq!(retrieved.content, "Completely new memory content that is sufficiently long for validation requirements");
    assert_eq!(retrieved.importance, 0.8);
    assert!(retrieved.updated.is_some()); // Should have updated timestamp
}

#[test]
fn test_update_memory_nonexistent() {
    let (storage, _temp_dir) = setup_test_storage();
    
    // Try to update non-existent memory
    let updated = storage.update_memory("mem_nonexist", Some("New content"), Some(0.7)).unwrap();
    assert!(!updated);
}

#[test]
fn test_update_memory_importance_clamping() {
    let (storage, _temp_dir) = setup_test_storage();
    
    // Create and store a memory
    let mut memory = Memory::new(
        MemoryCategory::Events,
        "test topic",
        "Test Memory Title for Importance Clamping",
        "Memory content for importance clamping test with sufficient length requirements",
    );
    memory.importance = 0.5;
    storage.store_memory(&memory).unwrap();
    
    // Update with out-of-range importance (should be clamped to 1.0)
    let updated = storage.update_memory(&memory.id, None, Some(1.5)).unwrap();
    assert!(updated);
    
    // Verify clamping
    let retrieved = storage.get_memory(&memory.id).unwrap().unwrap();
    assert_eq!(retrieved.importance, 1.0);
}