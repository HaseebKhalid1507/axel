//! Graph build pipeline for VelociRAG.
//!
//! 10-stage pipeline: scan → metadata → explicit → entity → temporal → topic → semantic → processing → centrality → store
//! Also handles basic document indexing (chunk + embed + store).
//! Port of velocirag/pipeline.py.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use tracing::info;
use walkdir::WalkDir;

use crate::analyzers::{
    CentralityAnalyzer, EntityAnalyzer, ExplicitAnalyzer, SemanticAnalyzer, TemporalAnalyzer,
    TopicAnalyzer, NODE_NOTE,
};
use crate::chunker::chunk_markdown;
use crate::db::{Database, Node};
use crate::embedder::Embedder;
use crate::error::Result;
use crate::frontmatter::{extract_tags_from_content, extract_wiki_links, parse_frontmatter};
use crate::index::VectorIndex;

// ── Document Indexing ──────────────────────────────────────────────────────

/// Index a directory of markdown files (chunk + embed + store).
pub fn index_directory(
    dir: impl AsRef<Path>,
    db: &mut Database,
    embedder: &mut Embedder,
    index: &mut VectorIndex,
) -> Result<IndexStats> {
    let dir = dir.as_ref();
    info!("Indexing directory: {}", dir.display());

    let mut stats = IndexStats::default();

    for entry in WalkDir::new(dir)
        .follow_links(true)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "md" || ext == "markdown" || ext == "txt")
                .unwrap_or(false)
        })
    {
        let path = entry.path();
        let rel_path = path
            .strip_prefix(dir)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();

        let mtime = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        if !db.file_needs_update(&rel_path, mtime)? {
            stats.skipped += 1;
            continue;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Failed to read {}: {}", path.display(), e);
                stats.errors += 1;
                continue;
            }
        };

        let chunks = chunk_markdown(&content, &rel_path);
        if chunks.is_empty() {
            stats.skipped += 1;
            continue;
        }

        db.delete_documents_by_file(&rel_path)?;

        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        let embeddings = embedder.embed(&texts)?;

        for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
            let rowid = db.insert_document(
                &chunk.doc_id,
                &chunk.content,
                &chunk.metadata,
                embedding,
                Some(&chunk.file_path),
            )?;
            index.add(rowid as u64, embedding)?;

            // Extract tags from content and frontmatter
            let tags = extract_tags_from_content(&chunk.content);
            for tag in &tags {
                if let Ok(tag_id) = db.upsert_tag(tag) {
                    let _ = db.tag_document(rowid, tag_id);
                }
            }

            // Extract wiki links as cross-references
            let links = extract_wiki_links(&chunk.content);
            for link in &links {
                let _ = db.insert_cross_ref(rowid, "wiki_link", link);
            }

            // Also extract frontmatter tags if present
            if let Some(fm_tags) = chunk.metadata.get("frontmatter")
                .and_then(|fm| fm.get("tags"))
                .and_then(|t| t.as_array())
            {
                for tag_val in fm_tags {
                    if let Some(tag_str) = tag_val.as_str() {
                        if let Ok(tag_id) = db.upsert_tag(&tag_str.to_lowercase()) {
                            let _ = db.tag_document(rowid, tag_id);
                        }
                    }
                }
            }
        }

        db.update_file_cache(&rel_path, mtime, "")?;
        stats.files_indexed += 1;
        stats.chunks_created += chunks.len();
    }

    stats.total_docs = db.document_count()?;
    info!(
        "Indexed {} files, {} chunks ({} skipped, {} errors). Total docs: {}",
        stats.files_indexed, stats.chunks_created, stats.skipped, stats.errors, stats.total_docs
    );

    Ok(stats)
}

#[derive(Debug, Default)]
pub struct IndexStats {
    pub files_indexed: usize,
    pub chunks_created: usize,
    pub skipped: usize,
    pub errors: usize,
    pub total_docs: usize,
}

// ── Graph Build Pipeline ───────────────────────────────────────────────────

/// Options for the graph build pipeline.
#[derive(Debug, Clone)]
pub struct GraphBuildOptions {
    pub force_rebuild: bool,
    pub skip_semantic: bool,
    pub min_edge_weight: f64,
    pub max_edges_per_node: usize,
}

impl Default for GraphBuildOptions {
    fn default() -> Self {
        Self {
            force_rebuild: false,
            skip_semantic: false,
            min_edge_weight: 0.3,
            max_edges_per_node: 50,
        }
    }
}

/// Statistics from the graph build.
#[derive(Debug, Default)]
pub struct GraphBuildStats {
    pub duration_secs: f64,
    pub files_scanned: usize,
    pub notes_created: usize,
    pub final_nodes: usize,
    pub final_edges: usize,
    pub stages: HashMap<String, StageStats>,
}

#[derive(Debug, Default, Clone)]
pub struct StageStats {
    pub duration_secs: f64,
    pub nodes_added: usize,
    pub edges_added: usize,
    pub extra: HashMap<String, String>,
}

/// Build the knowledge graph from a directory of markdown files.
///
/// 10-stage pipeline:
///   1. Scan files → note nodes
///   2. Metadata extraction (frontmatter)
///   3. Explicit analysis (wiki-links, tags)
///   4. Entity extraction (regex NER)
///   5. Temporal analysis (date co-occurrence)
///   6. Topic analysis (word frequency clustering)
///   7. Semantic analysis (embedding similarity)
///   8. Graph processing (dedup, prune)
///   9. Centrality analysis (degree + betweenness)
///  10. Storage (write to unified DB)
pub fn build_graph(
    source_dir: impl AsRef<Path>,
    db: &Database,
    embedder: Option<&mut Embedder>,
    opts: &GraphBuildOptions,
) -> Result<GraphBuildStats> {
    let source_dir = source_dir.as_ref();
    let pipeline_start = Instant::now();

    info!("Starting graph build from: {}", source_dir.display());

    // ═══ Incremental check ═══
    if !opts.force_rebuild {
        let (existing_nodes, existing_edges) = db.graph_stats()?;
        if existing_nodes > 0 {
            // Check if any files have changed since last graph build
            let mut any_changed = false;
            for entry in WalkDir::new(source_dir)
                .follow_links(true)
                .into_iter()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path().extension()
                        .map(|ext| ext == "md" || ext == "markdown")
                        .unwrap_or(false)
                })
            {
                let rel_path = entry.path()
                    .strip_prefix(source_dir)
                    .unwrap_or(entry.path())
                    .to_string_lossy()
                    .to_string();

                let mtime = entry.metadata().ok()
                    .and_then(|m| m.modified().ok())
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs_f64())
                    .unwrap_or(0.0);

                if db.file_needs_update(&format!("graph:{}", rel_path), mtime)? {
                    any_changed = true;
                    break;
                }
            }

            if !any_changed {
                info!("No changes detected, skipping graph rebuild ({} nodes, {} edges)", existing_nodes, existing_edges);
                return Ok(GraphBuildStats {
                    duration_secs: pipeline_start.elapsed().as_secs_f64(),
                    final_nodes: existing_nodes,
                    final_edges: existing_edges,
                    ..Default::default()
                });
            }
        }
    }

    let mut stats = GraphBuildStats::default();
    let mut nodes: Vec<Node> = Vec::new();
    let mut edges: Vec<crate::db::Edge> = Vec::new();

    // ═══ Stage 1: Scan files ═══
    {
        let t = Instant::now();
        let mut files_found = 0;

        for entry in WalkDir::new(source_dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "md" || ext == "markdown")
                    .unwrap_or(false)
            })
        {
            files_found += 1;
            let path = entry.path();

            let content = match std::fs::read_to_string(path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!("Failed to read {}: {}", path.display(), e);
                    continue;
                }
            };

            let rel_path = path
                .strip_prefix(source_dir)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            let title = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .replace('-', " ")
                .replace('_', " ");

            let node_id = format!("note_{:x}", md5_short(&rel_path));

            // Get file mtime
            let mtime = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);

            let word_count = content.split_whitespace().count();

            nodes.push(Node {
                id: node_id,
                node_type: NODE_NOTE.to_string(),
                title,
                content: Some(content),
                metadata: serde_json::json!({
                    "file_path": rel_path,
                    "word_count": word_count,
                    "modified_time": mtime,
                }),
                source_file: Some(rel_path),
            });
        }

        stats.files_scanned = files_found;
        stats.notes_created = nodes.len();
        stats.stages.insert("1_scan".into(), StageStats {
            duration_secs: t.elapsed().as_secs_f64(),
            nodes_added: nodes.len(),
            ..Default::default()
        });
        info!("Stage 1: {} files → {} notes ({:.1}s)", files_found, nodes.len(), t.elapsed().as_secs_f64());
    }

    // ═══ Stage 2: Metadata extraction ═══
    {
        let t = Instant::now();
        let mut metadata_enriched = 0;

        for node in &mut nodes {
            if node.node_type != NODE_NOTE {
                continue;
            }
            if let Some(ref content) = node.content {
                let (frontmatter, _body) = parse_frontmatter(content);
                if let Some(obj) = frontmatter.as_object() {
                    if !obj.is_empty() {
                        // Merge frontmatter into node metadata
                        if let Some(meta_obj) = node.metadata.as_object_mut() {
                            for (k, v) in obj {
                                meta_obj.insert(format!("fm_{}", k), v.clone());
                            }
                        }
                        metadata_enriched += 1;
                    }
                }
            }
        }

        stats.stages.insert("2_metadata".into(), StageStats {
            duration_secs: t.elapsed().as_secs_f64(),
            extra: [("enriched".into(), metadata_enriched.to_string())].into(),
            ..Default::default()
        });
        info!("Stage 2: Enriched {} notes with frontmatter ({:.1}s)", metadata_enriched, t.elapsed().as_secs_f64());
    }

    // ═══ Stage 3: Explicit analysis (wiki-links, tags) ═══
    {
        let t = Instant::now();
        let analyzer = ExplicitAnalyzer;
        let (new_nodes, new_edges) = analyzer.analyze(&nodes);

        stats.stages.insert("3_explicit".into(), StageStats {
            duration_secs: t.elapsed().as_secs_f64(),
            nodes_added: new_nodes.len(),
            edges_added: new_edges.len(),
            ..Default::default()
        });
        info!("Stage 3: +{} nodes, +{} edges ({:.1}s)", new_nodes.len(), new_edges.len(), t.elapsed().as_secs_f64());

        nodes.extend(new_nodes);
        edges.extend(new_edges);
    }

    // ═══ Stage 4: Entity extraction ═══
    {
        let t = Instant::now();
        let analyzer = EntityAnalyzer { min_frequency: 2 };
        let (new_nodes, new_edges) = analyzer.analyze(&nodes);

        stats.stages.insert("4_entity".into(), StageStats {
            duration_secs: t.elapsed().as_secs_f64(),
            nodes_added: new_nodes.len(),
            edges_added: new_edges.len(),
            ..Default::default()
        });
        info!("Stage 4: +{} entities, +{} edges ({:.1}s)", new_nodes.len(), new_edges.len(), t.elapsed().as_secs_f64());

        nodes.extend(new_nodes);
        edges.extend(new_edges);
    }

    // ═══ Stage 5: Temporal analysis ═══
    {
        let t = Instant::now();
        let analyzer = TemporalAnalyzer::default();
        let (new_nodes, new_edges) = analyzer.analyze(&nodes);

        stats.stages.insert("5_temporal".into(), StageStats {
            duration_secs: t.elapsed().as_secs_f64(),
            nodes_added: new_nodes.len(),
            edges_added: new_edges.len(),
            ..Default::default()
        });
        info!("Stage 5: +{} temporal edges ({:.1}s)", new_edges.len(), t.elapsed().as_secs_f64());

        nodes.extend(new_nodes);
        edges.extend(new_edges);
    }

    // ═══ Stage 6: Topic analysis ═══
    {
        let t = Instant::now();
        let analyzer = TopicAnalyzer::default();
        let (new_nodes, new_edges) = analyzer.analyze(&nodes);

        stats.stages.insert("6_topic".into(), StageStats {
            duration_secs: t.elapsed().as_secs_f64(),
            nodes_added: new_nodes.len(),
            edges_added: new_edges.len(),
            ..Default::default()
        });
        info!("Stage 6: +{} topics, +{} edges ({:.1}s)", new_nodes.len(), new_edges.len(), t.elapsed().as_secs_f64());

        nodes.extend(new_nodes);
        edges.extend(new_edges);
    }

    // ═══ Stage 7: Semantic analysis ═══
    if !opts.skip_semantic {
        if let Some(embedder) = embedder {
            let t = Instant::now();

            // Embed all note nodes
            let note_nodes: Vec<&Node> = nodes.iter()
                .filter(|n| n.node_type == NODE_NOTE && n.content.is_some())
                .collect();

            let mut node_embeddings: Vec<(String, Vec<f32>)> = Vec::new();

            // Batch embed
            let batch_size = 64;
            for chunk in note_nodes.chunks(batch_size) {
                let texts: Vec<&str> = chunk.iter()
                    .map(|n| n.content.as_deref().unwrap_or(""))
                    .collect();

                match embedder.embed(&texts) {
                    Ok(embs) => {
                        for (node, emb) in chunk.iter().zip(embs.into_iter()) {
                            node_embeddings.push((node.id.clone(), emb));
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Semantic embedding failed: {}", e);
                    }
                }
            }

            let analyzer = SemanticAnalyzer::default();
            let (new_nodes, new_edges) = analyzer.analyze(&node_embeddings);

            stats.stages.insert("7_semantic".into(), StageStats {
                duration_secs: t.elapsed().as_secs_f64(),
                nodes_added: new_nodes.len(),
                edges_added: new_edges.len(),
                extra: [("embeddings".into(), node_embeddings.len().to_string())].into(),
                ..Default::default()
            });
            info!("Stage 7: +{} semantic edges from {} embeddings ({:.1}s)",
                  new_edges.len(), node_embeddings.len(), t.elapsed().as_secs_f64());

            nodes.extend(new_nodes);
            edges.extend(new_edges);
        } else {
            info!("Stage 7: Skipped (no embedder provided)");
        }
    } else {
        info!("Stage 7: Skipped (skip_semantic=true)");
    }

    // ═══ Stage 8: Graph processing (dedup + prune) ═══
    {
        let t = Instant::now();
        let before_nodes = nodes.len();
        let before_edges = edges.len();

        nodes = merge_duplicates(nodes, &mut edges);
        edges = prune_weak_edges(edges, opts.min_edge_weight, opts.max_edges_per_node);

        stats.stages.insert("8_processing".into(), StageStats {
            duration_secs: t.elapsed().as_secs_f64(),
            extra: [
                ("nodes_merged".into(), (before_nodes - nodes.len()).to_string()),
                ("edges_pruned".into(), (before_edges - edges.len()).to_string()),
            ].into(),
            ..Default::default()
        });
        info!("Stage 8: {} nodes (-{}), {} edges (-{}) ({:.1}s)",
              nodes.len(), before_nodes - nodes.len(),
              edges.len(), before_edges - edges.len(),
              t.elapsed().as_secs_f64());
    }

    // ═══ Stage 9: Centrality analysis ═══
    {
        let t = Instant::now();
        let analyzer = CentralityAnalyzer;
        let scores = analyzer.analyze(&nodes, &edges);

        let mut scored = 0;
        for node in &mut nodes {
            if let Some(&score) = scores.get(&node.id) {
                if let Some(obj) = node.metadata.as_object_mut() {
                    obj.insert("importance_score".into(), serde_json::json!(score));
                }
                scored += 1;
            }
        }

        stats.stages.insert("9_centrality".into(), StageStats {
            duration_secs: t.elapsed().as_secs_f64(),
            extra: [("nodes_scored".into(), scored.to_string())].into(),
            ..Default::default()
        });
        info!("Stage 9: Scored {} nodes ({:.1}s)", scored, t.elapsed().as_secs_f64());
    }

    // ═══ Stage 10: Storage ═══
    {
        let t = Instant::now();

        // Clear content from nodes before storing (save DB space)
        for node in &mut nodes {
            if node.node_type == NODE_NOTE {
                node.content = None;
            }
        }

        // Store all nodes
        for node in &nodes {
            db.upsert_node(node)?;
        }

        // Store all edges
        for edge in &edges {
            db.upsert_edge(edge)?;
        }

        let (n, e) = db.graph_stats()?;
        stats.stages.insert("10_storage".into(), StageStats {
            duration_secs: t.elapsed().as_secs_f64(),
            nodes_added: n,
            edges_added: e,
            ..Default::default()
        });
        info!("Stage 10: Stored {} nodes, {} edges ({:.1}s)", n, e, t.elapsed().as_secs_f64());

        // Write file provenance for incremental detection
        for entry in WalkDir::new(source_dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path().extension()
                    .map(|ext| ext == "md" || ext == "markdown")
                    .unwrap_or(false)
            })
        {
            let rel_path = entry.path()
                .strip_prefix(source_dir)
                .unwrap_or(entry.path())
                .to_string_lossy()
                .to_string();

            let mtime = entry.metadata().ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);

            db.update_file_cache(&format!("graph:{}", rel_path), mtime, "graph")?;
        }
    }

    stats.final_nodes = nodes.len();
    stats.final_edges = edges.len();
    stats.duration_secs = pipeline_start.elapsed().as_secs_f64();

    info!(
        "Graph build complete: {} nodes, {} edges in {:.1}s",
        stats.final_nodes, stats.final_edges, stats.duration_secs
    );

    Ok(stats)
}

// ── Graph Processing Helpers ───────────────────────────────────────────────

/// Merge duplicate nodes by (type, title) and update edge references.
fn merge_duplicates(nodes: Vec<Node>, edges: &mut Vec<crate::db::Edge>) -> Vec<Node> {
    let mut unique_nodes = Vec::new();
    let mut node_mapping: HashMap<String, String> = HashMap::new();
    let mut seen: HashMap<(String, String), String> = HashMap::new();

    for node in nodes {
        let key = (node.node_type.clone(), node.title.to_lowercase().trim().to_string());

        if let Some(existing_id) = seen.get(&key) {
            node_mapping.insert(node.id.clone(), existing_id.clone());
        } else {
            seen.insert(key, node.id.clone());
            node_mapping.insert(node.id.clone(), node.id.clone());
            unique_nodes.push(node);
        }
    }

    // Update edge references
    for edge in edges.iter_mut() {
        if let Some(new_id) = node_mapping.get(&edge.source_id) {
            edge.source_id = new_id.clone();
        }
        if let Some(new_id) = node_mapping.get(&edge.target_id) {
            edge.target_id = new_id.clone();
        }
    }

    // Remove self-loops
    edges.retain(|e| e.source_id != e.target_id);

    let merged = node_mapping.len() - unique_nodes.len();
    if merged > 0 {
        info!("Merged {} duplicate nodes", merged);
    }

    unique_nodes
}

/// Remove weak edges and cap edges per node.
fn prune_weak_edges(
    mut edges: Vec<crate::db::Edge>,
    min_weight: f64,
    max_per_node: usize,
) -> Vec<crate::db::Edge> {
    let before = edges.len();

    // Filter by minimum weight
    edges.retain(|e| e.weight >= min_weight);

    // Sort by weight descending to keep strongest
    edges.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));

    // Cap edges per node
    let mut node_counts: HashMap<&str, usize> = HashMap::new();
    let mut kept = Vec::new();

    for edge in &edges {
        let src_count = node_counts.get(edge.source_id.as_str()).copied().unwrap_or(0);
        let tgt_count = node_counts.get(edge.target_id.as_str()).copied().unwrap_or(0);

        if src_count < max_per_node && tgt_count < max_per_node {
            *node_counts.entry(edge.source_id.as_str()).or_insert(0) += 1;
            *node_counts.entry(edge.target_id.as_str()).or_insert(0) += 1;
            kept.push(edge.clone());
        }
    }

    let pruned = before - kept.len();
    if pruned > 0 {
        info!("Pruned {} weak/excess edges", pruned);
    }

    kept
}

fn md5_short(text: &str) -> u32 {
    use md5::{Digest, Md5};
    let mut hasher = Md5::new();
    hasher.update(text.as_bytes());
    let result = hasher.finalize();
    u32::from_le_bytes([result[0], result[1], result[2], result[3]])
}
