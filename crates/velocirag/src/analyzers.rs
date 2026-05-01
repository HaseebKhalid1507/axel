//! Graph analyzers for the knowledge graph pipeline.
//!
//! Port of velocirag/analyzers.py — six analyzers that extract structure from documents:
//!   1. ExplicitAnalyzer   — wiki-links and #tags
//!   2. TemporalAnalyzer   — date-based co-occurrence
//!   3. EntityAnalyzer     — regex-based named entity extraction
//!   4. TopicAnalyzer      — TF-IDF word frequency topic clustering
//!   5. SemanticAnalyzer   — embedding cosine similarity edges
//!   6. CentralityAnalyzer — degree + betweenness importance scores

use std::collections::{HashMap, HashSet, VecDeque};

use regex::Regex;


use crate::db::{Edge, Node};

use crate::frontmatter::{extract_tags_from_content, extract_wiki_links};

// ── Node/Edge type constants ────────────────────────────────────────────────

pub const NODE_NOTE: &str = "note";
pub const NODE_TAG: &str = "tag";
pub const NODE_ENTITY: &str = "entity";
pub const NODE_TOPIC: &str = "topic";

pub const REL_REFERENCES: &str = "references";
pub const REL_TAGGED_AS: &str = "tagged_as";
pub const REL_MENTIONS: &str = "mentions";
pub const REL_TEMPORAL: &str = "temporal";
pub const REL_DISCUSSES: &str = "discusses";
pub const REL_SIMILAR_TO: &str = "similar_to";

// ═══════════════════════════════════════════════════════════════════════════
// 1. ExplicitAnalyzer — wiki-links and #tags
// ═══════════════════════════════════════════════════════════════════════════

pub struct ExplicitAnalyzer;

impl ExplicitAnalyzer {
    pub fn analyze(&self, nodes: &[Node]) -> (Vec<Node>, Vec<Edge>) {
        let mut new_nodes = Vec::new();
        let mut new_edges = Vec::new();
        let mut all_tags = HashSet::new();

        // First pass: collect all tags
        for node in nodes {
            if node.node_type == NODE_NOTE {
                if let Some(ref content) = node.content {
                    for tag in extract_tags_from_content(content) {
                        all_tags.insert(tag);
                    }
                }
            }
        }

        // Create tag nodes
        let mut tag_node_map = HashMap::new();
        for tag in &all_tags {
            let tag_id = format!("tag_{}", tag.to_lowercase());
            new_nodes.push(Node {
                id: tag_id.clone(),
                node_type: NODE_TAG.to_string(),
                title: format!("#{}", tag),
                content: None,
                metadata: serde_json::json!({"tag_name": tag}),
                source_file: None,
            });
            tag_node_map.insert(tag.clone(), tag_id);
        }

        // Second pass: create relationships
        for node in nodes {
            if node.node_type != NODE_NOTE {
                continue;
            }
            let Some(ref content) = node.content else { continue };

            // Wiki-link relationships
            for link in extract_wiki_links(content) {
                if let Some(target) = find_node_by_title(nodes, &new_nodes, &link) {
                    let edge_id = format!("ref_{}_{}", node.id, target.id);
                    new_edges.push(Edge {
                        id: edge_id,
                        source_id: node.id.clone(),
                        target_id: target.id.clone(),
                        edge_type: REL_REFERENCES.to_string(),
                        weight: 0.8,
                        confidence: 0.9,
                        metadata: serde_json::json!({"link_text": link}),
                        source_file: None,
                        valid_from: None,
                        valid_to: None,
                    });
                }
            }

            // Tag relationships
            for tag in extract_tags_from_content(content) {
                if let Some(tag_id) = tag_node_map.get(&tag) {
                    let edge_id = format!("tagged_{}_{}", node.id, tag_id);
                    new_edges.push(Edge {
                        id: edge_id,
                        source_id: node.id.clone(),
                        target_id: tag_id.clone(),
                        edge_type: REL_TAGGED_AS.to_string(),
                        weight: 0.7,
                        confidence: 1.0,
                        metadata: serde_json::json!({"tag_name": tag}),
                        source_file: None,
                        valid_from: None,
                        valid_to: None,
                    });
                }
            }
        }

        tracing::info!("ExplicitAnalyzer: {} nodes, {} edges", new_nodes.len(), new_edges.len());
        (new_nodes, new_edges)
    }
}

fn find_node_by_title<'a>(nodes: &'a [Node], extra: &'a [Node], title: &str) -> Option<&'a Node> {
    let title_lower = title.to_lowercase();
    
    // First: try exact title match
    let exact = nodes.iter().chain(extra.iter()).find(|n| {
        n.title.to_lowercase() == title_lower
    });
    if exact.is_some() {
        return exact;
    }

    // Second: try exact match ignoring trailing/leading whitespace
    let exact_trimmed = nodes.iter().chain(extra.iter()).find(|n| {
        n.title.to_lowercase().trim() == title_lower.trim()
    });
    if exact_trimmed.is_some() {
        return exact_trimmed;
    }

    // Third: fuzzy containment (prefer shorter titles — more specific matches)
    let mut candidates: Vec<&Node> = nodes.iter().chain(extra.iter()).filter(|n| {
        let t = n.title.to_lowercase();
        t.contains(&title_lower) || title_lower.contains(&t)
    }).collect();
    
    // Sort by title length ascending — prefer exact-length matches
    candidates.sort_by_key(|n| n.title.len());
    candidates.into_iter().next()
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. TemporalAnalyzer — date-based co-occurrence
// ═══════════════════════════════════════════════════════════════════════════

pub struct TemporalAnalyzer {
    pub window_days: i64,
    pub max_temporal_edges: usize,
    pub max_edges_per_node: usize,
}

impl Default for TemporalAnalyzer {
    fn default() -> Self {
        Self {
            window_days: 7,
            max_temporal_edges: 50_000,
            max_edges_per_node: 20,
        }
    }
}

impl TemporalAnalyzer {
    pub fn analyze(&self, nodes: &[Node]) -> (Vec<Node>, Vec<Edge>) {
        let new_nodes = Vec::new();
        let mut new_edges = Vec::new();

        // Extract dates from nodes
        let mut dated: Vec<(&Node, chrono::NaiveDate)> = Vec::new();
        let date_re = Regex::new(r"(\d{4}-\d{2}-\d{2})").unwrap();

        for node in nodes {
            if node.node_type != NODE_NOTE { continue; }

            // Try metadata fields
            let mut date = None;
            if let Some(obj) = node.metadata.as_object() {
                for field in &["created_time", "modified_time", "file_date", "date"] {
                    if let Some(val) = obj.get(*field).and_then(|v| v.as_str()) {
                        if let Ok(d) = chrono::NaiveDate::parse_from_str(&val[..10.min(val.len())], "%Y-%m-%d") {
                            date = Some(d);
                            break;
                        }
                    }
                }
            }

            // Try filename
            if date.is_none() {
                if let Some(ref sf) = node.source_file {
                    if let Some(cap) = date_re.captures(sf) {
                        if let Ok(d) = chrono::NaiveDate::parse_from_str(&cap[1], "%Y-%m-%d") {
                            date = Some(d);
                        }
                    }
                }
            }

            if let Some(d) = date {
                dated.push((node, d));
            }
        }

        if dated.len() < 2 {
            return (new_nodes, new_edges);
        }

        // Sort by date
        dated.sort_by_key(|(_, d)| *d);

        let mut edge_counts: HashMap<&str, usize> = HashMap::new();

        for i in 0..dated.len() {
            let (src, src_date) = &dated[i];
            if *edge_counts.get(src.id.as_str()).unwrap_or(&0) >= self.max_edges_per_node {
                continue;
            }

            for (tgt, tgt_date) in &dated[(i + 1)..] {
                let diff = (*tgt_date - *src_date).num_days().unsigned_abs() as i64;

                if diff > self.window_days { break; }

                if *edge_counts.get(tgt.id.as_str()).unwrap_or(&0) >= self.max_edges_per_node {
                    continue;
                }

                let weight = (1.0 - (diff as f64 / self.window_days as f64) * 0.5).max(0.3);
                new_edges.push(Edge {
                    id: format!("temporal_{}_{}", src.id, tgt.id),
                    source_id: src.id.clone(),
                    target_id: tgt.id.clone(),
                    edge_type: REL_TEMPORAL.to_string(),
                    weight,
                    confidence: 0.7,
                    metadata: serde_json::json!({"temporal_type": "concurrent", "days_apart": diff}),
                    source_file: None,
                    valid_from: None,
                    valid_to: None,
                });

                *edge_counts.entry(src.id.as_str()).or_insert(0) += 1;
                *edge_counts.entry(tgt.id.as_str()).or_insert(0) += 1;

                if new_edges.len() >= self.max_temporal_edges { break; }
            }
            if new_edges.len() >= self.max_temporal_edges { break; }
        }

        tracing::info!("TemporalAnalyzer: {} temporal edges", new_edges.len());
        (new_nodes, new_edges)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. EntityAnalyzer — regex-based named entity extraction
// ═══════════════════════════════════════════════════════════════════════════

pub struct EntityAnalyzer {
    pub min_frequency: usize,
}

impl Default for EntityAnalyzer {
    fn default() -> Self {
        Self { min_frequency: 2 }
    }
}

static COMMON_WORDS: &[&str] = &[
    "The", "This", "That", "These", "Those", "And", "But", "Or", "So",
    "Then", "Now", "Here", "There", "When", "Where", "Why", "How",
    "What", "Which", "Who", "Whom", "Whose", "Will", "Would", "Could",
    "Should", "Might", "May", "Can", "Must", "Shall",
];

impl EntityAnalyzer {
    pub fn analyze(&self, nodes: &[Node]) -> (Vec<Node>, Vec<Edge>) {
        let mut new_nodes = Vec::new();
        let mut new_edges = Vec::new();

        // Regex patterns for entity extraction
        let person_re = Regex::new(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b").unwrap();
        let tech_re = Regex::new(r"\b(?:Python|JavaScript|React|Vue|Angular|Node\.js|Django|Flask|Docker|Kubernetes|AWS|Azure|GCP|Rust|TypeScript|Go|Java|Ruby|PHP|Swift)\b").unwrap();
        let cap_re = Regex::new(r"\b[A-Z][a-zA-Z]{2,}\b").unwrap();

        // Extract entities per node
        let mut node_entities: HashMap<&str, HashSet<String>> = HashMap::new();
        let mut entity_counts: HashMap<String, usize> = HashMap::new();

        for node in nodes {
            if node.node_type != NODE_NOTE { continue; }
            let Some(ref content) = node.content else { continue };

            let mut entities = HashSet::new();

            // Person names
            for m in person_re.find_iter(content) {
                let e = m.as_str().to_string();
                if e.len() > 2 { entities.insert(e); }
            }

            // Tech terms
            for m in tech_re.find_iter(content) {
                entities.insert(m.as_str().to_string());
            }

            // General capitalized words (lower priority)
            for m in cap_re.find_iter(content) {
                let e = m.as_str().to_string();
                if e.len() > 2 && !COMMON_WORDS.contains(&e.as_str()) {
                    entities.insert(e);
                }
            }

            // Wiki links as entities
            for link in extract_wiki_links(content) {
                entities.insert(link);
            }

            // Filter common words
            entities.retain(|e| !COMMON_WORDS.contains(&e.as_str()));

            for entity in &entities {
                *entity_counts.entry(entity.clone()).or_insert(0) += 1;
            }
            node_entities.insert(&node.id, entities);
        }

        // Create entity nodes for frequent entities
        let mut entity_node_map = HashMap::new();
        for (entity, count) in &entity_counts {
            if *count < self.min_frequency { continue; }

            let entity_id = format!("entity_{:x}", md5_hash_short(&entity.to_lowercase()));
            new_nodes.push(Node {
                id: entity_id.clone(),
                node_type: NODE_ENTITY.to_string(),
                title: entity.clone(),
                content: None,
                metadata: serde_json::json!({"entity_type": "concept", "frequency": count}),
                source_file: None,
            });
            entity_node_map.insert(entity.clone(), entity_id);
        }

        // Create mention edges
        for node in nodes {
            if let Some(entities) = node_entities.get(node.id.as_str()) {
                for entity in entities {
                    if let Some(entity_id) = entity_node_map.get(entity) {
                        new_edges.push(Edge {
                            id: format!("mentions_{}_{}", node.id, entity_id),
                            source_id: node.id.clone(),
                            target_id: entity_id.clone(),
                            edge_type: REL_MENTIONS.to_string(),
                            weight: 0.6,
                            confidence: 0.8,
                            metadata: serde_json::json!({"entity_name": entity}),
                            source_file: None,
                            valid_from: None,
                            valid_to: None,
                        });
                    }
                }
            }
        }

        tracing::info!("EntityAnalyzer: {} entities, {} mention edges", new_nodes.len(), new_edges.len());
        (new_nodes, new_edges)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. TopicAnalyzer — word frequency topic clustering
// ═══════════════════════════════════════════════════════════════════════════

pub struct TopicAnalyzer {
    pub n_topics: usize,
}

impl Default for TopicAnalyzer {
    fn default() -> Self {
        Self { n_topics: 10 }
    }
}

static STOP_WORDS: &[&str] = &[
    "that", "this", "with", "have", "will", "from", "they", "been", "were",
    "some", "more", "than", "also", "each", "just", "like", "into", "over",
    "such", "very", "when", "what", "your", "about", "which", "their", "would",
    "there", "could", "other", "after", "being", "these", "those",
];

impl TopicAnalyzer {
    pub fn analyze(&self, nodes: &[Node]) -> (Vec<Node>, Vec<Edge>) {
        let mut new_nodes = Vec::new();
        let mut new_edges = Vec::new();

        let note_nodes: Vec<&Node> = nodes.iter()
            .filter(|n| n.node_type == NODE_NOTE && n.content.is_some())
            .collect();

        if note_nodes.len() < 3 {
            tracing::info!("TopicAnalyzer: Too few notes for topic analysis");
            return (new_nodes, new_edges);
        }

        // Word frequency analysis (fallback — no sklearn in Rust, this is the simple path)
        let word_re = Regex::new(r"\b[a-zA-Z]{4,}\b").unwrap();
        let mut all_words: HashMap<String, usize> = HashMap::new();
        let mut note_words: HashMap<&str, HashMap<String, usize>> = HashMap::new();

        for node in &note_nodes {
            let content = node.content.as_ref().unwrap();
            let mut words: HashMap<String, usize> = HashMap::new();
            for m in word_re.find_iter(&content.to_lowercase()) {
                let w = m.as_str().to_string();
                if !STOP_WORDS.contains(&w.as_str()) {
                    *words.entry(w.clone()).or_insert(0) += 1;
                    *all_words.entry(w).or_insert(0) += 1;
                }
            }
            note_words.insert(&node.id, words);
        }

        // Get most common words as potential topics
        let mut common: Vec<(String, usize)> = all_words.into_iter()
            .filter(|(_, c)| *c >= 2)
            .collect();
        common.sort_by(|a, b| b.1.cmp(&a.1));
        common.truncate(self.n_topics * 2);

        // Group notes by shared words
        for (word, _) in &common {
            let related: Vec<String> = note_words.iter()
                .filter(|(_, words)| words.get(word).copied().unwrap_or(0) >= 2)
                .map(|(id, _)| id.to_string())
                .collect();

            if related.len() >= 2 {
                let topic_id = format!("topic_{:x}", md5_hash_short(word));
                let topic_name = format!("Topic: {}", word);

                new_nodes.push(Node {
                    id: topic_id.clone(),
                    node_type: NODE_TOPIC.to_string(),
                    title: topic_name.clone(),
                    content: None,
                    metadata: serde_json::json!({"note_count": related.len(), "topic_type": "cluster"}),
                    source_file: None,
                });

                for note_id in &related {
                    new_edges.push(Edge {
                        id: format!("discusses_{}_{}", note_id, topic_id),
                        source_id: note_id.clone(),
                        target_id: topic_id.clone(),
                        edge_type: REL_DISCUSSES.to_string(),
                        weight: 0.7,
                        confidence: 0.6,
                        metadata: serde_json::json!({"topic_name": topic_name}),
                        source_file: None,
                        valid_from: None,
                        valid_to: None,
                    });
                }
            }
        }

        tracing::info!("TopicAnalyzer: {} topics, {} discussion edges", new_nodes.len(), new_edges.len());
        (new_nodes, new_edges)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. SemanticAnalyzer — embedding cosine similarity
// ═══════════════════════════════════════════════════════════════════════════

pub struct SemanticAnalyzer {
    pub threshold: f64,
    pub max_edges_per_node: usize,
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self {
            threshold: 0.7,
            max_edges_per_node: 20,
        }
    }
}

impl SemanticAnalyzer {
    /// Analyze semantic similarity using precomputed embeddings.
    /// Takes (node_id, embedding) pairs instead of computing them internally.
    pub fn analyze(&self, node_embeddings: &[(String, Vec<f32>)]) -> (Vec<Node>, Vec<Edge>) {
        let new_nodes = Vec::new();
        let mut new_edges = Vec::new();

        if node_embeddings.len() < 2 {
            return (new_nodes, new_edges);
        }

        // Normalize embeddings for cosine similarity
        let normalized: Vec<Vec<f32>> = node_embeddings.iter()
            .map(|(_, emb)| {
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    emb.iter().map(|x| x / norm).collect()
                } else {
                    emb.clone()
                }
            })
            .collect();

        // Brute-force top-K similarity (for moderate corpus sizes)
        // For large corpora, this would use the VectorIndex (usearch)
        let mut edge_counts: HashMap<&str, usize> = HashMap::new();
        let mut seen_pairs: HashSet<(usize, usize)> = HashSet::new();

        for i in 0..normalized.len() {
            let src_id = &node_embeddings[i].0;
            if *edge_counts.get(src_id.as_str()).unwrap_or(&0) >= self.max_edges_per_node {
                continue;
            }

            // Compute similarities to all other nodes
            let mut sims: Vec<(usize, f64)> = Vec::new();
            for j in 0..normalized.len() {
                if i == j { continue; }
                let sim: f64 = normalized[i].iter()
                    .zip(normalized[j].iter())
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum();
                if sim >= self.threshold {
                    sims.push((j, sim));
                }
            }

            // Sort by similarity descending
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (j, sim) in sims.into_iter().take(self.max_edges_per_node) {
                let pair = if i < j { (i, j) } else { (j, i) };
                if seen_pairs.contains(&pair) { continue; }

                let tgt_id = &node_embeddings[j].0;
                if *edge_counts.get(tgt_id.as_str()).unwrap_or(&0) >= self.max_edges_per_node {
                    continue;
                }

                seen_pairs.insert(pair);
                let clamped = sim.min(1.0);

                new_edges.push(Edge {
                    id: format!("semantic_{}_{}", src_id, tgt_id),
                    source_id: src_id.clone(),
                    target_id: tgt_id.clone(),
                    edge_type: REL_SIMILAR_TO.to_string(),
                    weight: clamped,
                    confidence: clamped,
                    metadata: serde_json::json!({"similarity_score": sim, "analysis_method": "cosine"}),
                    source_file: None,
                    valid_from: None,
                    valid_to: None,
                });

                *edge_counts.entry(src_id.as_str()).or_insert(0) += 1;
                *edge_counts.entry(tgt_id.as_str()).or_insert(0) += 1;
            }
        }

        tracing::info!("SemanticAnalyzer: {} similarity edges", new_edges.len());
        (new_nodes, new_edges)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. CentralityAnalyzer — degree + betweenness importance
// ═══════════════════════════════════════════════════════════════════════════

pub struct CentralityAnalyzer;

impl CentralityAnalyzer {
    /// Calculate importance scores for all nodes.
    /// Returns a map of node_id → importance score [0, 1].
    pub fn analyze(&self, nodes: &[Node], edges: &[Edge]) -> HashMap<String, f64> {
        if nodes.is_empty() || edges.is_empty() {
            return HashMap::new();
        }

        let node_ids: HashSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();

        // Build adjacency graph
        let mut graph: HashMap<&str, HashSet<&str>> = HashMap::new();
        for edge in edges {
            if node_ids.contains(edge.source_id.as_str()) && node_ids.contains(edge.target_id.as_str()) {
                graph.entry(edge.source_id.as_str()).or_default().insert(edge.target_id.as_str());
                graph.entry(edge.target_id.as_str()).or_default().insert(edge.source_id.as_str());
            }
        }

        // Degree centrality
        let mut degree: HashMap<&str, f64> = HashMap::new();
        for &id in &node_ids {
            degree.insert(id, graph.get(id).map(|s| s.len()).unwrap_or(0) as f64);
        }

        // Approximate betweenness (sampled BFS)
        let betweenness = self.approx_betweenness(&graph, &node_ids, 500);

        // Normalize and combine
        let max_deg = degree.values().copied().fold(0.0f64, f64::max).max(1.0);
        let max_bet = betweenness.values().copied().fold(0.0f64, f64::max).max(1.0);

        let mut scores = HashMap::new();
        for &id in &node_ids {
            let d = degree.get(id).copied().unwrap_or(0.0) / max_deg;
            let b = betweenness.get(id).copied().unwrap_or(0.0) / max_bet;
            scores.insert(id.to_string(), 0.7 * d + 0.3 * b);
        }

        tracing::info!("CentralityAnalyzer: scored {} nodes", scores.len());
        scores
    }

    fn approx_betweenness<'a>(
        &self,
        graph: &HashMap<&'a str, HashSet<&'a str>>,
        node_ids: &HashSet<&'a str>,
        max_samples: usize,
    ) -> HashMap<&'a str, f64> {
        let mut betweenness: HashMap<&str, f64> = node_ids.iter().map(|&id| (id, 0.0)).collect();

        let mut sources: Vec<&str> = node_ids.iter().copied().collect();
        sources.truncate(max_samples);

        for &source in &sources {
            // BFS with parent pointers
            let mut parent: HashMap<&str, Option<&str>> = HashMap::new();
            let mut depth: HashMap<&str, usize> = HashMap::new();
            parent.insert(source, None);
            depth.insert(source, 0);

            let mut queue = VecDeque::new();
            queue.push_back(source);

            while let Some(current) = queue.pop_front() {
                if let Some(neighbors) = graph.get(current) {
                    for &neighbor in neighbors {
                        if !parent.contains_key(neighbor) {
                            parent.insert(neighbor, Some(current));
                            depth.insert(neighbor, depth[current] + 1);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            // Count intermediates
            for (&target, &d) in &depth {
                if d <= 1 { continue; }
                let mut node = parent.get(target).copied().flatten();
                while let Some(n) = node {
                    if n == source { break; }
                    *betweenness.entry(n).or_insert(0.0) += 1.0;
                    node = parent.get(n).copied().flatten();
                }
            }
        }

        betweenness
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn md5_hash_short(text: &str) -> u32 {
    use md5::{Digest, Md5};
    let mut hasher = Md5::new();
    hasher.update(text.as_bytes());
    let result = hasher.finalize();
    u32::from_le_bytes([result[0], result[1], result[2], result[3]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn make_note(id: &str, title: &str, content: &str) -> Node {
        Node {
            id: id.to_string(),
            node_type: NODE_NOTE.to_string(),
            title: title.to_string(),
            content: Some(content.to_string()),
            metadata: Value::Object(Default::default()),
            source_file: None,
        }
    }

    #[test]
    fn test_explicit_analyzer() {
        let nodes = vec![
            make_note("n1", "Note One", "Hello #rust and see [[Note Two]]"),
            make_note("n2", "Note Two", "Content with #python"),
        ];
        let analyzer = ExplicitAnalyzer;
        let (new_nodes, new_edges) = analyzer.analyze(&nodes);
        assert!(!new_nodes.is_empty()); // tag nodes
        assert!(!new_edges.is_empty()); // tag + reference edges
    }

    #[test]
    fn test_entity_analyzer() {
        let nodes = vec![
            make_note("n1", "Note", "John Smith works on the Docker project. John Smith is great."),
            make_note("n2", "Note2", "John Smith also uses Python and Docker here."),
        ];
        let analyzer = EntityAnalyzer { min_frequency: 2 };
        let (new_nodes, new_edges) = analyzer.analyze(&nodes);
        assert!(!new_nodes.is_empty());
        assert!(!new_edges.is_empty());
    }

    #[test]
    fn test_centrality_analyzer() {
        let nodes = vec![
            make_note("a", "A", ""), make_note("b", "B", ""), make_note("c", "C", ""),
        ];
        let edges = vec![
            Edge { id: "e1".into(), source_id: "a".into(), target_id: "b".into(),
                   edge_type: "test".into(), weight: 1.0, confidence: 1.0, metadata: Value::Null, source_file: None, valid_from: None, valid_to: None },
            Edge { id: "e2".into(), source_id: "b".into(), target_id: "c".into(),
                   edge_type: "test".into(), weight: 1.0, confidence: 1.0, metadata: Value::Null, source_file: None, valid_from: None, valid_to: None },
        ];
        let scores = CentralityAnalyzer.analyze(&nodes, &edges);
        // Node B should have highest centrality (bridge between A and C)
        assert!(scores["b"] >= scores["a"]);
        assert!(scores["b"] >= scores["c"]);
    }
}
