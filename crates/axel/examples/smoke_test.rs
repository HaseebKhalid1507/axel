//! Smoke test: index real Jawz notes into a .r8 brain and search them.

use std::path::Path;
use std::time::Instant;

use axel::r8::Brain;
use axel::search::BrainSearch;

fn main() {
    let notes_dir = Path::new("/tmp/axel-sandbox/notes");
    let brain_path = Path::new("/tmp/axel-sandbox/jawz.r8");

    // Clean slate
    let _ = std::fs::remove_file(brain_path);

    println!("═══ AxelR8 Smoke Test ═══\n");

    // 1. Create brain
    let start = Instant::now();
    let _brain = Brain::create(brain_path, Some("jawz")).unwrap();
    println!("✓ Brain created: {} ({:.0}ms)", brain_path.display(), start.elapsed().as_millis());
    drop(_brain);

    // 2. Open search
    let start = Instant::now();
    let mut search = BrainSearch::open(brain_path).unwrap();
    println!("✓ Search initialized ({:.0}ms)\n", start.elapsed().as_millis());

    // 3. Index all markdown files
    let mut indexed = 0;
    let mut skipped = 0;
    let mut total_bytes = 0usize;
    let start = Instant::now();

    let mut files: Vec<_> = walkdir::WalkDir::new(notes_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
        .collect();
    files.sort_by_key(|e| e.path().to_path_buf());

    for entry in &files {
        let path = entry.path();
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => { skipped += 1; continue; }
        };

        // Skip tiny files
        if content.len() < 50 {
            skipped += 1;
            continue;
        }

        let rel_path = path.strip_prefix(notes_dir).unwrap_or(path);
        let doc_id = rel_path.to_string_lossy().replace('/', "::").replace(".md", "");

        // Extract title from first heading or filename
        let title = content.lines()
            .find(|l| l.starts_with("# "))
            .map(|l| l.trim_start_matches("# ").to_string())
            .unwrap_or_else(|| doc_id.clone());

        let metadata = serde_json::json!({
            "type": "note",
            "title": title,
            "path": rel_path.to_string_lossy(),
        });

        match search.index_document(&doc_id, &content, Some(metadata), Some(&rel_path.to_string_lossy())) {
            Ok(_) => {
                indexed += 1;
                total_bytes += content.len();
            }
            Err(e) => {
                eprintln!("  ✗ Failed to index {}: {}", doc_id, e);
                skipped += 1;
            }
        }
    }

    let index_time = start.elapsed();
    println!("✓ Indexed {indexed} documents ({skipped} skipped)");
    println!("  {:.1} KB of text", total_bytes as f64 / 1024.0);
    println!("  {:.1}s total ({:.0}ms per doc)\n",
        index_time.as_secs_f64(),
        index_time.as_millis() as f64 / indexed.max(1) as f64
    );

    // 4. Run test searches
    let queries = [
        "VelociRAG search architecture",
        "Memkoshi memory system",
        "Stelline session intelligence",
        "SynapsCLI agent runtime Rust",
        "D&D campaign Jawz character",
        "Haseeb internship job",
        "Docker security lab CS646",
        "homelab services pihole grafana",
        "JR Morton MahaMedia collaboration",
        "Dexter trading agent",
    ];

    println!("═══ Search Results ═══\n");

    for query in &queries {
        let start = Instant::now();
        match search.search(query, 3) {
            Ok(response) => {
                let ms = start.elapsed().as_millis();
                println!("🔍 \"{query}\" ({ms}ms, {}/{} layers hit)",
                    response.stats.final_count,
                    response.stats.vector_candidates
                        + response.stats.keyword_candidates
                        + response.stats.graph_candidates
                        + response.stats.metadata_candidates
                );
                for (i, result) in response.results.iter().take(3).enumerate() {
                    let preview: String = result.content.chars().take(80).collect();
                    let preview = preview.replace('\n', " ");
                    println!("  {}. [{}] {:.60}…", i + 1, result.doc_id, preview);
                }
                println!();
            }
            Err(e) => {
                println!("🔍 \"{query}\" — ERROR: {e}\n");
            }
        }
    }

    // 5. Brain stats
    let brain = Brain::open(brain_path).unwrap();
    let doc_count: i64 = brain.conn().query_row(
        "SELECT COUNT(*) FROM documents", [], |r| r.get(0)
    ).unwrap();
    let file_size = std::fs::metadata(brain_path).map(|m| m.len()).unwrap_or(0);

    println!("═══ Brain Stats ═══");
    println!("  Documents: {doc_count}");
    println!("  File size: {:.1} MB", file_size as f64 / 1024.0 / 1024.0);
    println!("  Path: {}", brain_path.display());
}
