//! Turn pairing, the turn row.
//!
//! A captured turn is a stand-in: it makes a turn findable from other chats before the
//! window it is in has been archived. Hook text has not been through core's credential
//! screen, so this file decides what is not stored at all: a turn core would withhold
//! (`screen::core_withholds`), the turn in which a forget tool was called, a turn of a
//! paused chat, a turn with no project, a chat filed under another project.
//!
//! Pairing runs on the front thread inside a hook's budget: no SQL, no embedding.
//! `write_turn` runs on the indexer thread and owns its transaction.

use std::collections::HashMap;

use axel_memkoshi::security::MemorySigner;
use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::ext::config::ExtConfig;
use crate::ext::front;
use crate::ext::types::{FrontState, HookEvent, Job};
use crate::project_memory::{RecordLinks, Sensitivity};
use crate::store::{self, Inserted, NewRow};
use crate::{archive, brain_v2, project, screen};

/// Pending turns kept at most; one more evicts the oldest.
pub const MAX_PENDING: usize = 64;
/// A prompt, and the whole reply of a turn, are cut here (bytes, on a char boundary).
pub const MAX_TEXT_BYTES: usize = 4096;
/// A buffer whose last hook is older than this is dropped (R17).
pub const STALE_MS: i64 = 10 * 60 * 1000;
/// A turn whose prompt plus reply is shorter is not stored.
pub const MIN_TURN_CHARS: usize = 60;

const TITLE_WORDS: usize = 12;

/// By `session_id`; at most 64, oldest evicted.
#[derive(Default)]
pub struct CaptureState {
    pending: HashMap<String, TurnBuffer>,
}

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

/// The longest head of `s` of at most `max` bytes that ends on a char boundary.
fn cut(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// A message the host wrote, not the person.
fn host_message(text: &str) -> bool {
    // SPEC: 3395-3397 say "starts"; white space before the mark is skipped, so that such a
    // message is dropped rather than paired.
    let head = text.trim_start();
    head.starts_with("<event ")
        || head.starts_with("[Context continuation")
        || head.starts_with("[Host context advisory")
}

impl CaptureState {
    /// How many turns are pending.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Whether a turn of this chat is pending.
    pub fn is_pending(&self, session_id: &str) -> bool {
        self.pending.contains_key(session_id)
    }

    pub fn on_user(&mut self, session_id: Option<&str>, cwd: Option<&str>, text: &str, now_ms: i64) {
        let Some(session_id) = session_id.filter(|s| !s.is_empty()) else {
            return;
        };
        if host_message(text) {
            self.pending.remove(session_id);
            return;
        }
        // Core's test is on the text as given, the whole of it. The key is of the whole
        // redacted text, before the cut (B-2f). The text is redacted before it is cut, so
        // the cut cannot leave the head of a token that is then too short to be a shape.
        let withheld = screen::core_withholds(text);
        let redacted = screen::redact(text);
        let turn_key = archive::turn_key(&redacted.text);
        let prompt = if withheld { String::new() } else { cut(&redacted.text, MAX_TEXT_BYTES).to_string() };

        if !self.pending.contains_key(session_id) && self.pending.len() >= MAX_PENDING {
            self.evict_oldest();
        }
        self.pending.insert(
            session_id.to_string(),
            TurnBuffer {
                prompt,
                turn_key,
                cwd: cwd.map(str::to_string),
                started_ms: now_ms,
                last_ms: now_ms,
                assistant: String::new(),
                suppressed: false,
                withheld,
            },
        );
    }

    // SPEC: 3371 says "oldest evicted" and not by which time: by the last hook, then by the
    // session id, so that the choice does not depend on the map's order.
    fn evict_oldest(&mut self) {
        let oldest = self
            .pending
            .iter()
            .min_by(|a, b| (a.1.last_ms, a.0).cmp(&(b.1.last_ms, b.0)))
            .map(|(session_id, _)| session_id.clone());
        if let Some(session_id) = oldest {
            self.pending.remove(&session_id);
        }
    }

    pub fn on_assistant(
        &mut self,
        session_id: Option<&str>,
        text: &str,
        has_tool_use: bool,
        now_ms: i64,
    ) -> Option<CompletedTurn> {
        let session_id = session_id?;
        let buf = self.pending.get_mut(session_id)?;
        if now_ms - buf.last_ms > STALE_MS {
            self.pending.remove(session_id);
            return None;
        }
        buf.last_ms = now_ms;
        // A marked turn keeps no text (`suppress`), a withheld one neither.
        if !buf.suppressed {
            if screen::core_withholds(text) {
                buf.withheld = true;
            }
            if buf.withheld {
                buf.prompt.clear();
                buf.assistant.clear();
            } else {
                let redacted = screen::redact(text);
                // SPEC: 3407 says "append" and not what stands between two responses: a
                // line break.
                if !buf.assistant.is_empty() && !redacted.text.is_empty() && buf.assistant.len() < MAX_TEXT_BYTES {
                    buf.assistant.push('\n');
                }
                let room = MAX_TEXT_BYTES - buf.assistant.len();
                buf.assistant.push_str(cut(&redacted.text, room));
            }
        }
        if has_tool_use {
            return None;
        }
        let buf = self.pending.remove(session_id)?;
        Some(CompletedTurn {
            session_id: session_id.to_string(),
            cwd: buf.cwd,
            prompt: buf.prompt,
            turn_key: buf.turn_key,
            reply: buf.assistant,
            started_ms: buf.started_ms,
            at_ms: now_ms,
            suppressed: buf.suppressed,
            withheld: buf.withheld,
        })
    }

    /// `(started_ms, turn_key)` of the pending turn, which it marks; `None` when nothing is
    /// pending.
    pub fn suppress(&mut self, session_id: &str) -> Option<(i64, String)> {
        let buf = self.pending.get_mut(session_id)?;
        buf.suppressed = true;
        // SPEC: 3409-3411: the indexer stores no row for a marked turn, so its text is not
        // kept either, here or on the queue. The buffer stays for its interval.
        buf.prompt.clear();
        buf.assistant.clear();
        Some((buf.started_ms, buf.turn_key.clone()))
    }

    /// A hook of this chat was not forwarded.
    pub fn drop_session(&mut self, session_id: &str) {
        self.pending.remove(session_id);
    }

    pub fn on_session_end(&mut self, session_id: &str) {
        self.pending.remove(session_id);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Written {
    /// A row was written, or the turn's row was there already.
    Stored,
    Skipped,
}

/// `"mem_" + hex(sha256("axel/live/1" || 0 || session_id || 0 || dec(started_ms) || 0 ||
/// sha256(prompt)))[..16]`; the prompt as `CompletedTurn` carries it.
pub fn turn_id(session_id: &str, started_ms: i64, prompt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"axel/live/1\0");
    hasher.update(session_id.as_bytes());
    hasher.update([0u8]);
    hasher.update(started_ms.to_string().as_bytes());
    hasher.update([0u8]);
    // SPEC: 3440-3441 do not say whether the inner hash goes in as bytes or as hex: the 32
    // bytes.
    hasher.update(Sha256::digest(prompt.as_bytes()));
    format!("mem_{}", &hex::encode(hasher.finalize())[..16])
}

/// One transaction of this file: `begin_write`, the work, `COMMIT`. After any error the
/// transaction is rolled back and never committed.
fn in_transaction<T>(conn: &Connection, work: impl FnOnce() -> Result<T>) -> Result<T> {
    // An error here leaves nothing of ours open: `begin_write` rolls its own `BEGIN` back,
    // and a transaction that was open before the call is the caller's.
    brain_v2::begin_write(conn)?;
    let out = work().and_then(|value| {
        conn.execute_batch("COMMIT")?;
        Ok(value)
    });
    if out.is_err() && !conn.is_autocommit() {
        let _ = conn.execute_batch("ROLLBACK");
    }
    out
}

/// Indexer thread. Called on a bare connection: it opens its own transaction, commits it,
/// and after any error rolls back and returns the error.
pub fn write_turn(
    conn: &Connection,
    signer: Option<&MemorySigner>,
    cfg: &ExtConfig,
    turn: &CompletedTurn,
) -> Result<Written> {
    let logical = project::logical(&turn.session_id);

    // The turn in which a forget tool was called: no row, whatever else holds; the
    // interval of the chat's `call` row is closed at the turn's end.
    if turn.suppressed {
        in_transaction(conn, || {
            // SPEC: 3435-3436 say "sets `to_ms` … to `at_ms`"; an interval is never made
            // shorter (a second call may have moved its end past this turn's clock).
            conn.execute(
                "UPDATE forgotten_turns SET to_ms = max(to_ms, ?1)
                  WHERE source = 'call' AND from_ms = ?2 AND (logical = ?3 OR session_id = ?4)",
                params![turn.at_ms, turn.started_ms, logical, turn.session_id],
            )?;
            Ok(())
        })?;
        return Ok(Written::Skipped);
    }
    if !cfg.capture {
        return Ok(Written::Skipped);
    }
    // What core would withhold. The flag is the pairing's, made from the whole text of
    // each hook; the test is made again on what the turn carries, so that no caller can
    // hand over a turn this function stores and core would not.
    if turn.withheld || screen::core_withholds(&turn.prompt) || screen::core_withholds(&turn.reply) {
        return Ok(Written::Skipped);
    }
    if turn.session_id.is_empty() {
        return Ok(Written::Skipped);
    }
    if turn.prompt.chars().count() + turn.reply.chars().count() < MIN_TURN_CHARS {
        return Ok(Written::Skipped);
    }
    let Some(project) = project::resolve(turn.cwd.as_deref()) else {
        return Ok(Written::Skipped);
    };

    in_transaction(conn, || {
        // The pause is read inside the write transaction, at each turn. `set_pause` writes
        // its row and hides the chat's captures in one transaction: this turn is either
        // committed before it, and hidden by it, or it sees the row and is not written.
        let paused = conn
            .query_row(
                "SELECT 1 FROM chat_pauses WHERE what = 'learning' AND (logical = ?1 OR session_id = ?2) LIMIT 1",
                params![logical, turn.session_id],
                |_| Ok(()),
            )
            .optional()?
            .is_some();
        if paused {
            return Ok(Written::Skipped);
        }

        // A chat filed elsewhere (R17).
        let filed: Vec<String> = {
            let mut stmt = conn.prepare(
                "SELECT DISTINCT scope_dir FROM archive_index WHERE logical = ?1
                 UNION
                 SELECT DISTINCT m.project_key FROM memory_ext e JOIN memories m ON m.id = e.mem_id
                  WHERE e.logical = ?1",
            )?;
            let rows = stmt.query_map([&logical], |r| r.get::<_, Option<String>>(0))?;
            rows.collect::<rusqlite::Result<Vec<_>>>()?.into_iter().flatten().collect()
        };
        if !filed.is_empty() && !filed.contains(&project.scope_dir) {
            return Ok(Written::Skipped);
        }

        let row = NewRow {
            id: turn_id(&turn.session_id, turn.started_ms, &turn.prompt),
            class: "episodic_turn",
            project_key: &project.scope_dir,
            title: turn.prompt.split_whitespace().take(TITLE_WORDS).collect::<Vec<_>>().join(" "),
            abstract_text: String::new(),
            // `insert_row` redacts title and content before anything is stored.
            content: format!("User: {}\nAssistant: {}", turn.prompt, turn.reply),
            tags: vec![format!("session:{}", turn.session_id), format!("turn:{}", turn.turn_key)],
            created_ms: turn.at_ms,
            time_source: "exact",
            sensitivity: Sensitivity::Normal,
            provenance: "axel:capture:turn".to_string(),
            expires_ms: None,
            importance: 0.5,
            origin_kind: "capture",
            origin_ref: &turn.session_id,
            unit_pos: None,
            session_id: Some(&turn.session_id),
            logical: Some(&logical),
            links: RecordLinks::default(),
            turn_key: Some(turn.turn_key.clone()),
            hidden: None,
        };
        Ok(match store::insert_row(conn, signer, &row)? {
            Inserted::New { .. } | Inserted::Exists => Written::Stored,
            // Forgotten for good: left alone, and no row is there.
            Inserted::Tombstoned => Written::Skipped,
        })
    })
}

/// After recall has been answered (called by `ext::hooks::handle`). No SQL.
pub fn on_user_event(st: &mut FrontState, ev: &HookEvent) {
    let Some(session_id) = ev.session_id.as_deref() else {
        return;
    };
    // SPEC: 3357 names `st.clock`; the time of a hook is `ev.received_ms`, which the rpc
    // thread read from that clock when the hook arrived. It is not later than anything
    // written for this turn, which is what a forget's `from_ms` needs.
    match ev.message.as_deref() {
        Some(text) => st.capture.on_user(Some(session_id), ev.cwd.as_deref(), text, ev.received_ms),
        // A prompt hook with no text: whatever was pending is not this message's turn.
        None => st.capture.drop_session(session_id),
    }
}

/// `on_message_complete` (called by `ext::hooks::handle`). No SQL: the turn goes to the
/// indexer as a job.
pub fn on_assistant_event(st: &mut FrontState, ev: &HookEvent) {
    let Some(session_id) = ev.session_id.as_deref() else {
        return;
    };
    // SPEC: 3340 says the hook carries `data.has_tool_use`. Without it nobody knows
    // whether the turn has ended: the turn is not captured live.
    let Some(has_tool_use) = ev.data.get("has_tool_use").and_then(serde_json::Value::as_bool) else {
        st.capture.drop_session(session_id);
        return;
    };
    let text = ev.message.as_deref().unwrap_or("");
    let Some(turn) = st.capture.on_assistant(Some(session_id), text, has_tool_use, ev.received_ms) else {
        return;
    };
    // `write_turn` is given no `Status`: the count is made here, where the turn ends.
    if turn.withheld && !turn.suppressed && st.cfg.capture {
        st.status.turns_withheld.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    // A full queue drops the turn (counted there); it arrives with its archive.
    front::send_job(st, Job::Capture(turn));
}

/// `on_session_end` (called by `ext::hooks::handle`).
pub fn on_session_end_event(st: &mut FrontState, ev: &HookEvent) {
    if let Some(session_id) = ev.session_id.as_deref() {
        st.capture.on_session_end(session_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn holds(buf: &TurnBuffer, needle: &str) -> bool {
        buf.prompt.contains(needle) || buf.assistant.contains(needle) || buf.turn_key.contains(needle)
    }

    // Section 4, test 11: the buffer never held the prompt's text.
    #[test]
    fn a_withheld_prompt_never_enters_the_buffer() {
        let mut cap = CaptureState::default();
        cap.on_user(Some("s"), None, "my password is hunter2", 1);
        let buf = &cap.pending["s"];
        assert!(buf.withheld);
        assert!(buf.prompt.is_empty());
        assert!(!holds(buf, "hunter2") && !holds(buf, "password"));

        // A reply core would withhold empties what the buffer held before it.
        cap.on_user(Some("t"), None, "what did the review of the indexer say", 1);
        assert!(cap.on_assistant(Some("t"), "it said the reader is fine", true, 2).is_none());
        assert!(cap.on_assistant(Some("t"), "the secret is plover", true, 3).is_none());
        let buf = &cap.pending["t"];
        assert!(buf.withheld);
        assert!(buf.prompt.is_empty() && buf.assistant.is_empty());
        // Nothing later is kept either.
        assert!(cap.on_assistant(Some("t"), "a clean line", true, 4).is_none());
        assert!(cap.pending["t"].assistant.is_empty());
    }

    // The turn of a forget call keeps no text from the call on.
    #[test]
    fn a_marked_buffer_holds_no_text() {
        let mut cap = CaptureState::default();
        cap.on_user(Some("s"), None, "forget that I told you about the offer from Acme", 1);
        assert!(cap.on_assistant(Some("s"), "I will forget the offer", true, 2).is_none());
        assert!(cap.suppress("s").is_some());
        let buf = &cap.pending["s"];
        assert!(buf.suppressed && buf.prompt.is_empty() && buf.assistant.is_empty());
        assert!(cap.on_assistant(Some("s"), "the offer from Acme is forgotten", true, 3).is_none());
        assert!(cap.pending["s"].assistant.is_empty());
    }

    #[test]
    fn cut_ends_on_a_char_boundary() {
        assert_eq!(cut("abc", 10), "abc");
        assert_eq!(cut("aé", 2), "a");
        assert_eq!(cut("aé", 3), "aé");
        assert_eq!(cut("abc", 0), "");
    }
}
