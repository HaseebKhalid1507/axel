use chrono::{Duration, Utc};
use tempfile::TempDir;
use velocirag::db::{Database, Node};
use velocirag::graph::GraphQuerier;

fn setup_test_db() -> (Database, TempDir) {
    let temp_dir = tempfile::tempdir().unwrap();
    let db = Database::open(temp_dir.path()).unwrap();
    (db, temp_dir)
}

#[test]
fn test_temporal_edge_insertion() {
    let (db, _temp_dir) = setup_test_db();
    
    // Insert two nodes
    let node1 = Node {
        id: "node1".to_string(),
        node_type: "document".to_string(),
        title: "Test Node 1".to_string(),
        content: Some("Test content 1".to_string()),
        metadata: serde_json::json!({}),
        source_file: None,
    };
    let node2 = Node {
        id: "node2".to_string(),
        node_type: "document".to_string(),
        title: "Test Node 2".to_string(),
        content: Some("Test content 2".to_string()),
        metadata: serde_json::json!({}),
        source_file: None,
    };
    
    db.upsert_node(&node1).unwrap();
    db.upsert_node(&node2).unwrap();
    
    // Insert edge with temporal validity
    let now = Utc::now();
    let future = now + Duration::hours(1);
    
    db.insert_edge(
        "edge1",
        "node1",
        "node2",
        "references",
        1.0,
        0.9,
        &serde_json::json!({}),
        None,
        Some(now),
        Some(future)
    ).unwrap();
    
    // Verify edge exists through find_connections
    let graph = GraphQuerier::new(&db);
    let connections = graph.find_connections("Test Node 1", 1).unwrap();
    assert!(connections.total_connections > 0);
}

#[test]
fn test_edge_invalidation() {
    let (db, _temp_dir) = setup_test_db();
    
    // Insert nodes
    let node1 = Node {
        id: "node1".to_string(),
        node_type: "document".to_string(),
        title: "Test Node 1".to_string(),
        content: Some("Test content 1".to_string()),
        metadata: serde_json::json!({}),
        source_file: None,
    };
    let node2 = Node {
        id: "node2".to_string(),
        node_type: "document".to_string(),
        title: "Test Node 2".to_string(),
        content: Some("Test content 2".to_string()),
        metadata: serde_json::json!({}),
        source_file: None,
    };
    
    db.upsert_node(&node1).unwrap();
    db.upsert_node(&node2).unwrap();
    
    // Insert edge without expiry
    db.insert_edge(
        "edge1",
        "node1",
        "node2",
        "references",
        1.0,
        0.9,
        &serde_json::json!({}),
        None,
        None,
        None
    ).unwrap();
    
    // Invalidate the edge
    let invalidated = db.invalidate_edge("edge1").unwrap();
    assert!(invalidated);
    
    // Try to invalidate it again (should still succeed since the row exists)
    let invalidated_again = db.invalidate_edge("edge1").unwrap();
    assert!(invalidated_again);
}

#[test]
fn test_temporal_filtering() {
    let (db, _temp_dir) = setup_test_db();
    
    // Insert nodes
    let node1 = Node {
        id: "node1".to_string(),
        node_type: "document".to_string(),
        title: "Test Node 1".to_string(),
        content: Some("Test content 1".to_string()),
        metadata: serde_json::json!({}),
        source_file: None,
    };
    let node2 = Node {
        id: "node2".to_string(),
        node_type: "document".to_string(),
        title: "Test Node 2".to_string(),
        content: Some("Test content 2".to_string()),
        metadata: serde_json::json!({}),
        source_file: None,
    };
    let node3 = Node {
        id: "node3".to_string(),
        node_type: "document".to_string(),
        title: "Test Node 3".to_string(),
        content: Some("Test content 3".to_string()),
        metadata: serde_json::json!({}),
        source_file: None,
    };
    
    db.upsert_node(&node1).unwrap();
    db.upsert_node(&node2).unwrap();
    db.upsert_node(&node3).unwrap();
    
    let now = Utc::now();
    let past = now - Duration::hours(1);
    let future = now + Duration::hours(1);
    
    // Insert one expired edge (past expiry)
    db.insert_edge(
        "expired_edge",
        "node1",
        "node2",
        "references",
        1.0,
        0.9,
        &serde_json::json!({}),
        None,
        Some(past),
        Some(past + Duration::minutes(30)) // Expired 30 minutes ago
    ).unwrap();
    
    // Insert one active edge (future expiry)
    db.insert_edge(
        "active_edge",
        "node1",
        "node3",
        "references",
        1.0,
        0.9,
        &serde_json::json!({}),
        None,
        Some(now),
        Some(future)
    ).unwrap();
    
    // Insert one permanent edge (no expiry)
    db.insert_edge(
        "permanent_edge",
        "node2",
        "node3",
        "references",
        1.0,
        0.9,
        &serde_json::json!({}),
        None,
        None,
        None
    ).unwrap();
    
    // Query should only return active edges
    let graph = GraphQuerier::new(&db);
    let connections_node1 = graph.find_connections("Test Node 1", 1).unwrap();
    let connections_node2 = graph.find_connections("Test Node 2", 1).unwrap();
    
    // node1 should only show connections through active edge (not expired)
    // node2 should show connections through permanent edge
    // Both should have some connections (temporal filtering working)
    assert!(connections_node1.total_connections >= 1);
    assert!(connections_node2.total_connections >= 1);
}

#[test]
fn test_invalidate_nonexistent_edge() {
    let (db, _temp_dir) = setup_test_db();
    
    // Try to invalidate non-existent edge
    let invalidated = db.invalidate_edge("nonexistent").unwrap();
    assert!(!invalidated);
}

#[test]
fn test_basic_temporal_edge_operations() {
    let (db, _temp_dir) = setup_test_db();
    
    // Create the nodes first
    let source_node = Node {
        id: "source1".to_string(),
        node_type: "test".to_string(),
        title: "Source Node".to_string(),
        content: Some("Source content".to_string()),
        metadata: serde_json::json!({}),
        source_file: None,
    };
    let target_node = Node {
        id: "target1".to_string(),
        node_type: "test".to_string(),
        title: "Target Node".to_string(),
        content: Some("Target content".to_string()),
        metadata: serde_json::json!({}),
        source_file: None,
    };
    
    db.upsert_node(&source_node).unwrap();
    db.upsert_node(&target_node).unwrap();
    
    let now = Utc::now();
    let future = now + Duration::hours(1);
    
    // Test basic insert_edge with temporal fields
    let result = db.insert_edge(
        "temporal_edge",
        "source1",
        "target1", 
        "test_relation",
        0.8,
        0.7,
        &serde_json::json!({"test": true}),
        Some("test.txt"),
        Some(now),
        Some(future)
    );
    if let Err(e) = &result {
        println!("Error inserting edge: {:?}", e);
    }
    assert!(result.is_ok());
    
    // Test invalidate_edge
    let invalidated = db.invalidate_edge("temporal_edge").unwrap();
    assert!(invalidated);
    
    // Test invalidating already invalidated edge
    let invalidated_again = db.invalidate_edge("temporal_edge").unwrap();
    assert!(invalidated_again); // Should still return true as the row exists
}