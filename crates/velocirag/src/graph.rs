//! Knowledge graph operations.
//!
//! Higher-level graph queries built on top of the unified database.
//! Provides path finding, similarity search, topic exploration, and hub discovery.
//! Port of velocirag/graph.py (GraphQuerier + GraphStore higher-level ops).

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

pub use crate::db::{Edge, Node};
use crate::db::Database;
use crate::error::{Result, VelociError};

// ── Types ───────────────────────────────────────────────────────────────────

/// A neighbor discovered during graph traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neighbor {
    pub node: Node,
    pub edge: Edge,
    pub distance: usize,
}

/// Result of a connection query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionMap {
    pub center_node: String,
    pub total_connections: usize,
    pub connections_by_type: HashMap<String, Vec<ConnectionEntry>>,
    pub max_depth: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionEntry {
    pub node_title: String,
    pub distance: usize,
    pub weight: f64,
    pub confidence: f64,
}

/// Result of a path query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathResult {
    pub path: Vec<String>,
    pub distance: usize,
    pub edges: Vec<Edge>,
}

/// A topic web — all nodes related to a topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicWeb {
    pub topic: String,
    pub related_nodes: Vec<TopicNode>,
    pub node_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicNode {
    pub id: String,
    pub title: String,
    pub node_type: String,
    pub connection_strength: f64,
}

/// A hub node — highly connected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubNode {
    pub node_id: String,
    pub title: String,
    pub node_type: String,
    pub connection_count: usize,
    pub importance_score: f64,
}

/// A similar node result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarNode {
    pub node: Node,
    pub similarity: f64,
    pub confidence: f64,
}

// ── Graph Querier ──────────────────────────────────────────────────────────

/// Query engine for the knowledge graph.
///
/// Path finding, similarity search, topic exploration, hub discovery.
pub struct GraphQuerier<'a> {
    db: &'a Database,
}

impl<'a> GraphQuerier<'a> {
    pub fn new(db: &'a Database) -> Self {
        Self { db }
    }

    /// Find all connections to a node by title within specified depth.
    pub fn find_connections(&self, node_title: &str, depth: usize) -> Result<ConnectionMap> {
        let depth = depth.min(3);

        // Find node by title (fuzzy match)
        let pattern = format!("%{}%", node_title.to_lowercase());
        let node_id: Option<String> = self.db.conn().query_row(
            "SELECT id FROM nodes WHERE LOWER(title) LIKE ?1 LIMIT 1",
            [&pattern],
            |row| row.get(0),
        ).ok();

        let node_id = node_id.ok_or_else(|| {
            VelociError::NotFound(format!("Node with title '{}' not found", node_title))
        })?;

        // Get node title
        let title: String = self.db.conn().query_row(
            "SELECT title FROM nodes WHERE id = ?1",
            [&node_id],
            |row| row.get(0),
        )?;

        // BFS traversal
        let neighbors = self.bfs_neighbors(&node_id, depth)?;

        // Organize by relationship type
        let mut connections_by_type: HashMap<String, Vec<ConnectionEntry>> = HashMap::new();
        for neighbor in &neighbors {
            connections_by_type
                .entry(neighbor.edge.edge_type.clone())
                .or_default()
                .push(ConnectionEntry {
                    node_title: neighbor.node.title.clone(),
                    distance: neighbor.distance,
                    weight: neighbor.edge.weight,
                    confidence: neighbor.edge.confidence,
                });
        }

        Ok(ConnectionMap {
            center_node: title,
            total_connections: neighbors.len(),
            connections_by_type,
            max_depth: depth,
        })
    }

    /// Find nodes similar to the given node (via SIMILAR_TO edges).
    pub fn find_similar(&self, node_id: &str, limit: usize) -> Result<Vec<SimilarNode>> {
        let mut stmt = self.db.conn().prepare(
            "SELECT target_id, weight, confidence FROM edges
             WHERE source_id = ?1 AND type = 'similar_to'
               AND (valid_to IS NULL OR valid_to > datetime('now'))
             ORDER BY weight DESC LIMIT ?2"
        )?;

        let rows = stmt.query_map(rusqlite::params![node_id, limit], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, f64>(1)?,
                row.get::<_, f64>(2)?,
            ))
        })?;

        let mut results = Vec::new();
        for row in rows {
            let (target_id, weight, confidence) = row?;
            if let Ok(Some(node)) = self.db.get_node(&target_id) {
                results.push(SimilarNode {
                    node,
                    similarity: weight,
                    confidence,
                });
            }
        }

        Ok(results)
    }

    /// Find shortest path between two nodes (BFS).
    pub fn find_path(&self, source_id: &str, target_id: &str) -> Result<Option<PathResult>> {
        if source_id == target_id {
            return Ok(Some(PathResult {
                path: vec![source_id.to_string()],
                distance: 0,
                edges: Vec::new(),
            }));
        }

        let mut queue: VecDeque<(String, Vec<String>, Vec<Edge>)> = VecDeque::new();
        queue.push_back((source_id.to_string(), vec![source_id.to_string()], Vec::new()));
        let mut visited = HashSet::new();
        visited.insert(source_id.to_string());

        let max_iterations = 200;
        let max_path_len = 6;

        for _ in 0..max_iterations {
            let Some((current_id, path, edges)) = queue.pop_front() else {
                break;
            };

            // Get all edges for current node
            let current_edges = self.get_edges_both(&current_id)?;

            for edge in current_edges {
                let neighbor_id = if edge.source_id == current_id {
                    edge.target_id.clone()
                } else {
                    edge.source_id.clone()
                };

                if neighbor_id == target_id {
                    let mut final_path = path.clone();
                    final_path.push(neighbor_id);
                    let mut final_edges = edges.clone();
                    final_edges.push(edge);
                    return Ok(Some(PathResult {
                        distance: final_path.len() - 1,
                        path: final_path,
                        edges: final_edges,
                    }));
                }

                if !visited.contains(&neighbor_id) && path.len() < max_path_len {
                    visited.insert(neighbor_id.clone());
                    let mut new_path = path.clone();
                    new_path.push(neighbor_id.clone());
                    let mut new_edges = edges.clone();
                    new_edges.push(edge);
                    queue.push_back((neighbor_id, new_path, new_edges));
                }
            }
        }

        Ok(None) // No path found
    }

    /// Get all nodes related to a specific topic.
    pub fn get_topic_web(&self, topic: &str) -> Result<TopicWeb> {
        let pattern = format!("%{}%", topic.to_lowercase());

        // Find topic node
        let topic_id: Option<String> = self.db.conn().query_row(
            "SELECT id FROM nodes WHERE type = 'topic' AND LOWER(title) LIKE ?1 LIMIT 1",
            [&pattern],
            |row| row.get(0),
        ).ok();

        let topic_id = topic_id.ok_or_else(|| {
            VelociError::NotFound(format!("Topic '{}' not found", topic))
        })?;

        // Find all nodes connected to this topic
        let mut stmt = self.db.conn().prepare(
            "SELECT DISTINCT n.id, n.title, n.type, e.weight
             FROM nodes n
             JOIN edges e ON (n.id = e.source_id OR n.id = e.target_id)
             WHERE (e.source_id = ?1 OR e.target_id = ?1) AND n.id != ?1
             ORDER BY e.weight DESC"
        )?;

        let rows = stmt.query_map([&topic_id], |row| {
            Ok(TopicNode {
                id: row.get(0)?,
                title: row.get(1)?,
                node_type: row.get(2)?,
                connection_strength: row.get(3)?,
            })
        })?;

        let related_nodes: Vec<TopicNode> = rows.filter_map(|r| r.ok()).collect();
        let node_count = related_nodes.len();

        Ok(TopicWeb {
            topic: topic.to_string(),
            related_nodes,
            node_count,
        })
    }

    /// Get the most connected nodes in the graph.
    pub fn get_hub_nodes(&self, limit: usize) -> Result<Vec<HubNode>> {
        let mut stmt = self.db.conn().prepare(
            "SELECT n.id, n.title, n.type,
                    COUNT(DISTINCT e.id) as connection_count
             FROM nodes n
             LEFT JOIN edges e ON (n.id = e.source_id OR n.id = e.target_id)
             GROUP BY n.id, n.title, n.type
             ORDER BY connection_count DESC
             LIMIT ?1"
        )?;

        let rows = stmt.query_map([limit], |row| {
            Ok(HubNode {
                node_id: row.get(0)?,
                title: row.get(1)?,
                node_type: row.get(2)?,
                connection_count: row.get(3)?,
                importance_score: 0.0, // filled in below
            })
        })?;

        let mut hubs: Vec<HubNode> = rows.filter_map(|r| r.ok()).collect();

        // Enrich with importance scores from metadata
        for hub in &mut hubs {
            if let Ok(Some(node)) = self.db.get_node(&hub.node_id) {
                if let Some(score) = node.metadata.get("importance_score").and_then(|v| v.as_f64()) {
                    hub.importance_score = score;
                }
            }
        }

        Ok(hubs)
    }

    // ── Internal ────────────────────────────────────────────────────────

    /// BFS neighbor traversal up to `depth` hops.
    fn bfs_neighbors(&self, start_id: &str, depth: usize) -> Result<Vec<Neighbor>> {
        let mut results = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(start_id.to_string());

        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((start_id.to_string(), 0));

        while let Some((current_id, current_depth)) = queue.pop_front() {
            if current_depth >= depth {
                continue;
            }

            let edges = self.get_edges_both(&current_id)?;
            for edge in edges {
                let neighbor_id = if edge.source_id == current_id {
                    edge.target_id.clone()
                } else {
                    edge.source_id.clone()
                };

                if visited.contains(&neighbor_id) {
                    continue;
                }
                visited.insert(neighbor_id.clone());

                if let Ok(Some(node)) = self.db.get_node(&neighbor_id) {
                    results.push(Neighbor {
                        node,
                        edge,
                        distance: current_depth + 1,
                    });
                    queue.push_back((neighbor_id, current_depth + 1));
                }
            }
        }

        Ok(results)
    }

    /// Get all edges touching a node (both directions).
    fn get_edges_both(&self, node_id: &str) -> Result<Vec<Edge>> {
        let mut stmt = self.db.conn().prepare(
            "SELECT id, source_id, target_id, type, weight, confidence, metadata, source_file, valid_from, valid_to
             FROM edges WHERE (source_id = ?1 OR target_id = ?1)
               AND (valid_to IS NULL OR valid_to > datetime('now'))"
        )?;

        let rows = stmt.query_map([node_id], |row| {
            Ok(Edge {
                id: row.get(0)?,
                source_id: row.get(1)?,
                target_id: row.get(2)?,
                edge_type: row.get(3)?,
                weight: row.get(4)?,
                confidence: row.get(5)?,
                metadata: row.get::<_, String>(6)
                    .ok()
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default(),
                source_file: row.get(7)?,
                valid_from: row.get::<_, Option<String>>(8)?.map(|s| chrono::DateTime::parse_from_rfc3339(&s).unwrap().into()),
                valid_to: row.get::<_, Option<String>>(9)?.map(|s| chrono::DateTime::parse_from_rfc3339(&s).unwrap().into()),
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_path_self() {
        let db = Database::open_memory().unwrap();
        let querier = GraphQuerier::new(&db);
        let result = querier.find_path("a", "a").unwrap();
        assert!(result.is_some());
        let path = result.unwrap();
        assert_eq!(path.distance, 0);
        assert_eq!(path.path, vec!["a"]);
    }

    #[test]
    fn test_find_path_simple() {
        let db = Database::open_memory().unwrap();

        // Insert nodes
        db.insert_node("a", "note", "Node A", None, &serde_json::json!({}), None).unwrap();
        db.insert_node("b", "note", "Node B", None, &serde_json::json!({}), None).unwrap();
        db.insert_node("c", "note", "Node C", None, &serde_json::json!({}), None).unwrap();

        // Insert edges: a → b → c
        db.insert_edge("e1", "a", "b", "references", 1.0, 1.0, &serde_json::json!({}), None, None, None).unwrap();
        db.insert_edge("e2", "b", "c", "references", 1.0, 1.0, &serde_json::json!({}), None, None, None).unwrap();

        let querier = GraphQuerier::new(&db);
        let result = querier.find_path("a", "c").unwrap();
        assert!(result.is_some());
        let path = result.unwrap();
        assert_eq!(path.distance, 2);
        assert_eq!(path.path, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_hub_nodes() {
        let db = Database::open_memory().unwrap();

        db.insert_node("hub", "note", "Hub Node", None, &serde_json::json!({}), None).unwrap();
        db.insert_node("a", "note", "A", None, &serde_json::json!({}), None).unwrap();
        db.insert_node("b", "note", "B", None, &serde_json::json!({}), None).unwrap();

        db.insert_edge("e1", "hub", "a", "references", 1.0, 1.0, &serde_json::json!({}), None, None, None).unwrap();
        db.insert_edge("e2", "hub", "b", "references", 1.0, 1.0, &serde_json::json!({}), None, None, None).unwrap();

        let querier = GraphQuerier::new(&db);
        let hubs = querier.get_hub_nodes(5).unwrap();
        assert!(!hubs.is_empty());
        assert_eq!(hubs[0].node_id, "hub");
        assert_eq!(hubs[0].connection_count, 2);
    }
}
