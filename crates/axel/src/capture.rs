//! Turn pairing, the turn row. Commit 16 writes this file.

use std::collections::HashMap;

use axel_memkoshi::security::MemorySigner;
use rusqlite::Connection;

use crate::error::Result;
use crate::ext::config::ExtConfig;
use crate::ext::types::{FrontState, HookEvent};

/// By `session_id`; at most 64, oldest evicted.
#[derive(Default)]
pub struct CaptureState {
    // Nothing reads it until commit 16.
    #[allow(dead_code)]
    pending: HashMap<String, TurnBuffer>,
}

// Never constructed until commit 16.
#[allow(dead_code)]
struct TurnBuffer {
    prompt: String,
    turn_key: String,
    cwd: Option<String>,
    started_ms: i64,
    last_ms: i64,
    assistant: String,
    suppressed: bool,
    withheld: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletedTurn {
    pub session_id: String,
    pub cwd: Option<String>,
    pub prompt: String,
    pub turn_key: String,
    pub reply: String,
    pub started_ms: i64,
    pub at_ms: i64,
    pub suppressed: bool,
    pub withheld: bool,
}

impl CaptureState {
    pub fn on_user(&mut self, session_id: Option<&str>, cwd: Option<&str>, text: &str, now_ms: i64) {
        let _ = (session_id, cwd, text, now_ms);
    }

    pub fn on_assistant(
        &mut self,
        session_id: Option<&str>,
        text: &str,
        has_tool_use: bool,
        now_ms: i64,
    ) -> Option<CompletedTurn> {
        let _ = (session_id, text, has_tool_use, now_ms);
        None
    }

    /// `(started_ms, turn_key)` of the pending turn, which it marks; `None` when nothing is
    /// pending. The stub returns `None`.
    pub fn suppress(&mut self, session_id: &str) -> Option<(i64, String)> {
        let _ = session_id;
        None
    }

    /// A hook of this chat was not forwarded.
    pub fn drop_session(&mut self, session_id: &str) {
        let _ = session_id;
    }

    pub fn on_session_end(&mut self, session_id: &str) {
        let _ = session_id;
    }
}

// SPEC: 2650 returns `Result<Written>` and no line defines `Written`. Two answers are
// assumed, a row or none; the stub writes nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Written {
    Stored,
    Skipped,
}

/// Indexer thread.
pub fn write_turn(
    conn: &Connection,
    signer: Option<&MemorySigner>,
    cfg: &ExtConfig,
    turn: &CompletedTurn,
) -> Result<Written> {
    let _ = (conn, signer, cfg, turn);
    Ok(Written::Skipped)
}

// SPEC: 4777-4779 give the three calls below as `capture::on_user_event(st, ev)`,
// `capture::on_assistant_event(st, ev)` and `capture::on_session_end_event(st, ev)`, with
// no result. None is assumed: the table in section 9 (4341-4344) answers these hooks
// without asking capture.

/// After recall has been answered (called by `ext::hooks::handle`).
pub fn on_user_event(st: &mut FrontState, ev: &HookEvent) {
    let _ = (st, ev);
}

/// `on_message_complete` (called by `ext::hooks::handle`).
pub fn on_assistant_event(st: &mut FrontState, ev: &HookEvent) {
    let _ = (st, ev);
}

/// `on_session_end` (called by `ext::hooks::handle`).
pub fn on_session_end_event(st: &mut FrontState, ev: &HookEvent) {
    let _ = (st, ev);
}
