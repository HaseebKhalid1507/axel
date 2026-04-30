//! SynapsCLI extension protocol — JSON-RPC over stdio.
//!
//! `axel extension` runs as a subprocess of SynapsCLI, speaking the
//! Content-Length framed JSON-RPC 2.0 protocol over stdin/stdout.
//!
//! No Python, no subprocess shelling. Brain opens once, stays warm,
//! all search happens in-process.

use std::io::{self, BufRead, Read, Write};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::brain::AxelBrain;
use crate::error::Result;

/// Minimum query length to bother searching.
const MIN_QUERY_LEN: usize = 5;

/// Max search results per query.
const SEARCH_LIMIT: usize = 5;

/// Max injection size in chars.
const MAX_INJECT_CHARS: usize = 3000;

// ── JSON-RPC types ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct RpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    jsonrpc: String,
    id: Value,
    result: Value,
}

// ── Protocol I/O ────────────────────────────────────────────────────────

fn read_message(reader: &mut impl BufRead) -> io::Result<Option<RpcRequest>> {
    // Read Content-Length header
    let mut header_line = String::new();
    loop {
        header_line.clear();
        let n = reader.read_line(&mut header_line)?;
        if n == 0 {
            return Ok(None); // EOF
        }
        let trimmed = header_line.trim();
        if trimmed.is_empty() {
            continue; // blank line between header and body, or between messages
        }
        if trimmed.to_lowercase().starts_with("content-length:") {
            break;
        }
    }

    let content_length: usize = header_line
        .trim()
        .split(':')
        .nth(1)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);

    if content_length == 0 {
        return Ok(None);
    }

    // Read blank line separator
    let mut blank = String::new();
    reader.read_line(&mut blank)?;

    // Read body
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;

    match serde_json::from_slice(&body) {
        Ok(req) => Ok(Some(req)),
        Err(_) => Ok(None),
    }
}

fn write_message(writer: &mut impl Write, response: &RpcResponse) -> io::Result<()> {
    let body = serde_json::to_string(response).unwrap();
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    writer.write_all(header.as_bytes())?;
    writer.write_all(body.as_bytes())?;
    writer.flush()
}

// ── Hook handlers ───────────────────────────────────────────────────────

fn handle_before_message(brain: &mut AxelBrain, params: &Value) -> Value {
    let message = params.get("message")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if message.len() < MIN_QUERY_LEN {
        return json!({"action": "continue"});
    }

    // Search the brain
    let results = match brain.search(message, SEARCH_LIMIT) {
        Ok(r) => r,
        Err(_) => return json!({"action": "continue"}),
    };

    if results.results.is_empty() {
        return json!({"action": "continue"});
    }

    // Format results for injection
    let mut injection = String::from("[Axel Brain — relevant context]\n");
    for r in &results.results {
        let snippet: String = r.content.chars().take(200).collect();
        let line = format!("• [{}] (score: {:.2}) {}\n",
            r.metadata.get("source_file")
                .and_then(|v| v.as_str())
                .or_else(|| r.metadata.get("title").and_then(|v| v.as_str()))
                .unwrap_or(&r.doc_id),
            r.score,
            snippet.replace('\n', " "),
        );
        if injection.len() + line.len() > MAX_INJECT_CHARS {
            break;
        }
        injection.push_str(&line);
    }
    injection.push_str("[End Axel context]");

    json!({
        "action": "inject",
        "content": injection,
    })
}

fn handle_session_start(brain: &mut AxelBrain) -> Value {
    // Inject handoff from last session if available
    match brain.get_handoff() {
        Ok(Some(handoff)) if !handoff.is_empty() => {
            let injection = format!(
                "[Last session handoff]\n{}\n[End handoff]",
                if handoff.len() > 800 { &handoff[..800] } else { &handoff }
            );
            json!({
                "action": "inject",
                "content": injection,
            })
        }
        _ => json!({"action": "continue"}),
    }
}

fn handle_session_end(_brain: &mut AxelBrain) -> Value {
    // Future: auto-store session summary
    json!({"action": "continue"})
}

// ── Main loop ───────────────────────────────────────────────────────────

/// Run the extension protocol loop.
///
/// Opens the brain once, then reads JSON-RPC requests from stdin and
/// writes responses to stdout until shutdown or EOF.
pub fn run(brain_path: &std::path::Path) -> Result<()> {
    let mut brain = AxelBrain::open_or_create(brain_path, None)?;

    let stdin = io::stdin();
    let mut reader = io::BufReader::new(stdin.lock());
    let stdout = io::stdout();
    let mut writer = io::BufWriter::new(stdout.lock());

    // Ensure search index is built on startup
    // (first search would build it anyway, but this avoids latency on first query)
    let _ = brain.search("warmup", 1);

    loop {
        let request = match read_message(&mut reader) {
            Ok(Some(req)) => req,
            Ok(None) => break, // EOF
            Err(_) => continue,
        };

        match request.method.as_str() {
            "hook.handle" => {
                let kind = request.params.get("kind")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let result = match kind {
                    "before_message" => handle_before_message(&mut brain, &request.params),
                    "on_session_start" => handle_session_start(&mut brain),
                    "on_session_end" => handle_session_end(&mut brain),
                    _ => json!({"action": "continue"}),
                };

                if let Some(id) = request.id {
                    let response = RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result,
                    };
                    let _ = write_message(&mut writer, &response);
                }
            }
            "shutdown" => {
                let _ = brain.flush();
                break;
            }
            _ => {
                // Unknown method — respond with continue
                if let Some(id) = request.id {
                    let response = RpcResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: json!({"action": "continue"}),
                    };
                    let _ = write_message(&mut writer, &response);
                }
            }
        }
    }

    Ok(())
}
