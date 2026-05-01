//! MCP server — exposes Axel brain as tools for SynapsCLI.
//!
//! `axel mcp` speaks JSON-RPC 2.0 over stdin/stdout (newline-delimited,
//! matching SynapsCLI's MCP client). Exposes three tools:
//!
//! - `axel_search`   — search the brain for relevant documents/memories
//! - `axel_remember`  — store a new memory
//! - `axel_recall`    — get boot context (handoff + recent memories)

use std::io::{self, BufRead, Write};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::brain::AxelBrain;
use crate::error::Result;

// ── JSON-RPC types ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct RpcMessage {
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: String,
    id: Value,
    result: Value,
}

#[derive(Debug, Serialize)]
struct RpcError {
    jsonrpc: String,
    id: Value,
    error: RpcErrorBody,
}

#[derive(Debug, Serialize)]
struct RpcErrorBody {
    code: i64,
    message: String,
}

// ── Tool definitions ────────────────────────────────────────────────────

fn tool_definitions() -> Value {
    json!({
        "tools": [
            {
                "name": "axel_search",
                "description": "Search the agent's persistent brain (.r8) for relevant documents, memories, and knowledge graph connections. Use this when you need context about a topic, person, project, or past event.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query — natural language question or topic"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results to return (default: 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "axel_remember",
                "description": "Store a memory in the agent's persistent brain. Use this to save important facts, decisions, preferences, or events that should persist across sessions.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The memory content — what to remember"
                        },
                        "category": {
                            "type": "string",
                            "description": "Category: events, preferences, entities, or cases",
                            "enum": ["events", "preferences", "entities", "cases"]
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance 0.0–1.0 (default: 0.5)",
                            "default": 0.5
                        },
                        "ttl_hours": {
                            "type": "integer",
                            "description": "Time-to-live in hours. Memory will be auto-deleted after this time (optional)"
                        }
                    },
                    "required": ["content", "category"]
                }
            },
            {
                "name": "axel_recall",
                "description": "Get boot context from the agent's brain — last session handoff and recent memories. Use at session start or when you need to remember what happened previously.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "axel_verify",
                "description": "Verify a memory by ID — returns full record with provenance and signature status. Use this to inspect a specific memory's authenticity and metadata.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "Memory ID to verify (e.g. mem_12345678)"
                        }
                    },
                    "required": ["memory_id"]
                }
            },
            {
                "name": "axel_update",
                "description": "Update a memory's content and/or importance — wiki-style editing with re-signing and re-indexing. Use this to correct or enhance existing memories.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "Memory ID to update (e.g. mem_12345678)"
                        },
                        "content": {
                            "type": "string",
                            "description": "New content for the memory (optional)"
                        },
                        "importance": {
                            "type": "number",
                            "description": "New importance level 0.0-1.0 (optional)",
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["memory_id"]
                }
            },
            {
                "name": "axel_consolidate",
                "description": "Run a consolidation pass on the brain. Reindexes changed files, strengthens accessed documents, reorganizes graph edges, and prunes stale content. Use sparingly — typically runs on a timer.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "phase": {
                            "type": "string",
                            "enum": ["all", "reindex", "strengthen", "reorganize", "prune"],
                            "description": "Which phase to run. Default: all."
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "Preview changes without applying them."
                        }
                    }
                }
            }
        ]
    })
}

// ── Tool execution ──────────────────────────────────────────────────────

fn execute_tool(brain: &mut AxelBrain, name: &str, args: &Value) -> Value {
    match name {
        "axel_search" => {
            let query = args["query"].as_str().unwrap_or("");
            let limit = args["limit"].as_u64().unwrap_or(5).min(50) as usize;

            if query.is_empty() {
                return tool_error("Query cannot be empty");
            }

            match brain.search(query, limit) {
                Ok(results) => {
                    // Record search feedback for consolidation
                    brain.search_mut().record_search_feedback(query, &results.results);

                    let mut output = format!("🔍 {} results for \"{}\" ({:.0}ms)\n\n",
                        results.results.len(), query, results.stats.total_ms);

                    for (i, r) in results.results.iter().enumerate() {
                        let source = r.metadata.get("source_file")
                            .and_then(|v| v.as_str())
                            .unwrap_or(&r.doc_id);
                        let snippet: String = r.content.chars().take(300).collect();
                        output.push_str(&format!("{}. [{}] (score: {:.3})\n   {}\n\n",
                            i + 1, source, r.score,
                            snippet.replace('\n', "\n   ")));
                    }

                    tool_text(&output)
                }
                Err(e) => tool_error(&format!("Search failed: {}", e)),
            }
        }

        "axel_remember" => {
            let content = args["content"].as_str().unwrap_or("");
            let category = args["category"].as_str().unwrap_or("events");
            let importance = args["importance"].as_f64().unwrap_or(0.5);
            let ttl_hours = args["ttl_hours"].as_u64();

            if content.is_empty() {
                return tool_error("Content cannot be empty");
            }

            match brain.remember_with_ttl(content, category, importance, ttl_hours) {
                Ok(id) => {
                    let ttl_info = match ttl_hours {
                        Some(hours) => format!(" (expires in {}h)", hours),
                        None => "".to_string(),
                    };
                    tool_text(&format!("✅ Memory stored: {}{}", id, ttl_info))
                },
                Err(e) => tool_error(&format!("Failed to store memory: {}", e)),
            }
        }

        "axel_recall" => {
            match brain.boot_context() {
                Ok(ctx) => {
                    if ctx.formatted.is_empty() {
                        tool_text("No boot context available (no handoff or memories).")
                    } else {
                        tool_text(&format!("{}\n\n({} memories, ~{} tokens)",
                            ctx.formatted, ctx.included_ids.len(), ctx.estimated_tokens))
                    }
                }
                Err(e) => tool_error(&format!("Recall failed: {}", e)),
            }
        }

        "axel_verify" => {
            let memory_id = args["memory_id"].as_str().unwrap_or("");

            if memory_id.is_empty() {
                return tool_error("Memory ID cannot be empty");
            }

            match brain.get_memory_with_verification(memory_id) {
                Ok(Some((memory, verified))) => {
                    let verification_status = if memory.signature.is_some() {
                        if verified { "✅ VERIFIED" } else { "❌ FAILED" }
                    } else {
                        "⚠️ UNSIGNED"
                    };

                    let ttl_info = match memory.remaining_ttl_hours() {
                        Some(hours) if hours > 0 => format!("⏰ Expires in {}h", hours),
                        Some(_) => "🕐 EXPIRED".to_string(), // 0 or negative
                        None => "∞ No expiry".to_string(),
                    };

                    let superseded_info = if memory.is_superseded() {
                        format!("🔄 Superseded by: {}", memory.superseded_by.as_deref().unwrap_or("unknown"))
                    } else {
                        "✨ Active".to_string()
                    };

                    tool_text(&format!(
                        "🔍 Memory Verification: {}\n\n\
                        📝 **Content:** {}\n\n\
                        📊 **Metadata:**\n\
                        • ID: {}\n\
                        • Category: {:?}\n\
                        • Title: {}\n\
                        • Topic: {}\n\
                        • Importance: {:.2}\n\
                        • Confidence: {:?}\n\
                        • Trust Level: {:.2}\n\
                        • Created: {}\n\
                        • Updated: {}\n\
                        • Sessions: {}\n\
                        • Tags: {}\n\
                        • Related Topics: {}\n\n\
                        🔐 **Security:** {}\n\
                        ⏱️ **TTL:** {}\n\
                        🔄 **Status:** {}",
                        memory_id,
                        memory.content,
                        memory.id,
                        memory.category,
                        memory.title,
                        memory.topic,
                        memory.importance,
                        memory.confidence,
                        memory.trust_level,
                        memory.created.format("%Y-%m-%d %H:%M:%S UTC"),
                        memory.updated.map(|u| u.format("%Y-%m-%d %H:%M:%S UTC").to_string()).unwrap_or("Never".to_string()),
                        memory.source_sessions.join(", "),
                        memory.tags.join(", "),
                        memory.related_topics.join(", "),
                        verification_status,
                        ttl_info,
                        superseded_info
                    ))
                },
                Ok(None) => tool_error(&format!("Memory not found: {}", memory_id)),
                Err(e) => tool_error(&format!("Failed to verify memory: {}", e)),
            }
        }

        "axel_update" => {
            let memory_id = args["memory_id"].as_str().unwrap_or("");
            let new_content = args["content"].as_str();
            let new_importance = args["importance"].as_f64();

            if memory_id.is_empty() {
                return tool_error("Memory ID cannot be empty");
            }

            // Validate new content if provided
            if let Some(content) = new_content {
                if content.trim().is_empty() {
                    return tool_error("Content cannot be empty");
                }
                if content.chars().count() < 50 {
                    return tool_error("Content must be at least 50 characters");
                }
            }

            // Validate importance if provided
            if let Some(importance) = new_importance {
                if !(0.0..=1.0).contains(&importance) {
                    return tool_error("Importance must be between 0.0 and 1.0");
                }
            }

            match brain.update_memory(memory_id, new_content, new_importance) {
                Ok(true) => {
                    let updates: Vec<String> = [
                        new_content.map(|_| "content"),
                        new_importance.map(|_| "importance"),
                    ].into_iter().flatten().map(|s| s.to_string()).collect();
                    
                    tool_text(&format!("✅ Memory updated: {} ({})", memory_id, updates.join(", ")))
                },
                Ok(false) => tool_error(&format!("Memory not found: {}", memory_id)),
                Err(e) => tool_error(&format!("Failed to update memory: {}", e)),
            }
        }

        "axel_consolidate" => {
            use crate::consolidate::{self, Phase, ConsolidateOptions};
            use std::collections::HashSet;

            let dry_run = args["dry_run"].as_bool().unwrap_or(false);
            let phase_str = args["phase"].as_str().unwrap_or("all");

            // Honour ~/.config/axel/sources.toml so MCP and CLI runs agree.
            let sources = consolidate::load_sources(None);

            let phases: HashSet<Phase> = match phase_str {
                "reindex" => [Phase::Reindex].into(),
                "strengthen" => [Phase::Strengthen].into(),
                "reorganize" => [Phase::Reorganize].into(),
                "prune" => [Phase::Prune].into(),
                _ => HashSet::new(),
            };

            let opts = ConsolidateOptions { sources, phases, dry_run, verbose: false };

            match consolidate::consolidate(brain.search_mut(), &opts) {
                Ok(stats) => {
                    let mode = if dry_run { "dry run" } else { "complete" };
                    tool_text(&format!(
                        "🧠 Consolidation {mode}\n\
                         Phase 1 — Reindex:    {} checked, {} reindexed, {} pruned\n\
                         Phase 2 — Strengthen: {} boosted, {} decayed, {} extinction\n\
                         Phase 3 — Reorganize: +{} edges, ~{} updated, -{} removed\n\
                         Phase 4 — Prune:      {} removed, {} flagged, {} misaligned\n\
                         Duration: {:.1}s",
                        stats.reindex.checked, stats.reindex.reindexed, stats.reindex.pruned,
                        stats.strengthen.boosted, stats.strengthen.decayed, stats.strengthen.extinction_signals,
                        stats.reorganize.edges_added, stats.reorganize.edges_updated, stats.reorganize.edges_removed,
                        stats.prune.removed, stats.prune.flagged, stats.prune.misaligned,
                        stats.duration_secs,
                    ))
                }
                Err(e) => tool_error(&format!("Consolidation failed: {e}")),
            }
        }

        _ => tool_error(&format!("Unknown tool: {}", name)),
    }
}

fn tool_text(text: &str) -> Value {
    json!({
        "content": [{"type": "text", "text": text}]
    })
}

fn tool_error(msg: &str) -> Value {
    json!({
        "content": [{"type": "text", "text": msg}],
        "isError": true
    })
}

// ── Main loop ───────────────────────────────────────────────────────────

/// Run the MCP server.
///
/// Opens the brain once, then reads newline-delimited JSON-RPC from stdin
/// and writes responses to stdout.
pub fn run(brain_path: &std::path::Path) -> Result<()> {
    let mut brain = AxelBrain::open_or_create(brain_path, None)?;

    let stdin = io::stdin();
    let reader = stdin.lock();
    let stdout = io::stdout();
    let mut writer = io::BufWriter::new(stdout.lock());

    // Warm up search index
    let _ = brain.search("warmup", 1);

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        if line.trim().is_empty() {
            continue;
        }

        let msg: RpcMessage = match serde_json::from_str(&line) {
            Ok(m) => m,
            Err(_) => continue,
        };

        let method = msg.method.as_deref().unwrap_or("");

        match method {
            "initialize" => {
                let response = RpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: msg.id.unwrap_or(Value::Null),
                    result: json!({
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "axel",
                            "version": "0.1.0"
                        }
                    }),
                };
                writeln!(writer, "{}", serde_json::to_string(&response).unwrap())?;
                writer.flush()?;
            }

            "notifications/initialized" => {
                // Client acknowledged — no response needed
            }

            "tools/list" => {
                let response = RpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: msg.id.unwrap_or(Value::Null),
                    result: tool_definitions(),
                };
                writeln!(writer, "{}", serde_json::to_string(&response).unwrap())?;
                writer.flush()?;
            }

            "tools/call" => {
                let name = msg.params["name"].as_str().unwrap_or("");
                let arguments = msg.params.get("arguments").cloned().unwrap_or(json!({}));

                let result = execute_tool(&mut brain, name, &arguments);

                let response = RpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: msg.id.unwrap_or(Value::Null),
                    result,
                };
                writeln!(writer, "{}", serde_json::to_string(&response).unwrap())?;
                writer.flush()?;
            }

            _ => {
                // Unknown method — ignore notifications, error on requests
                if let Some(id) = msg.id {
                    let error = RpcError {
                        jsonrpc: "2.0".to_string(),
                        id,
                        error: RpcErrorBody {
                            code: -32601,
                            message: format!("Method not found: {}", method),
                        },
                    };
                    writeln!(writer, "{}", serde_json::to_string(&error).unwrap())?;
                    writer.flush()?;
                }
            }
        }
    }

    let _ = brain.flush();
    Ok(())
}
