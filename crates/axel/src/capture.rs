//! Turn pairing, the turn row.
//!
//! A captured turn is a stand-in: it makes a turn findable from other chats before the
//! window it is in has been archived. Hook text has not been through core's credential
//! screen, so this file decides what is not stored at all: a turn core would withhold
//! (`screen::core_withholds`), the turn in which a forget tool was called, a turn of a
//! paused chat, a turn with no project, a chat filed under another project.
//!
//! Pairing runs on the front thread inside a hook's budget: no SQL, no embedding (but for a
//! marked or withheld turn the queue refuses: `front::forget_unqueued_turn`).
//! `write_turn` runs on the indexer thread and owns its transaction.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use axel_memkoshi::security::MemorySigner;
use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::ext::config::ExtConfig;
use crate::ext::front;
use crate::ext::types::{FrontState, HookEvent, Job};
use crate::project_memory::{RecordLinks, Sensitivity};
use crate::store::{self, Inserted, NewRow};
use crate::{archive, brain_v2, project, recall, screen};

/// Pending turns kept at most; one more evicts the oldest.
pub const MAX_PENDING: usize = 64;
/// A prompt, and the whole reply of a turn, are cut here (bytes, on a char boundary).
pub const MAX_TEXT_BYTES: usize = 4096;
/// A turn whose prompt plus reply is shorter is not stored.
pub const MIN_TURN_CHARS: usize = 60;

const TITLE_WORDS: usize = 12;
/// The keys of replaced prompts a pending turn keeps at most; later ones are not kept.
// RULING: v2 final capture round, item 1. R-16 set no bound for this; the oldest keys are
// kept, the first message's among them, and the time range covers the rest in the log.
pub const MAX_REPLACED_KEYS: usize = 16;

/// By `session_id`; at most 64, oldest evicted.
#[derive(Default)]
pub struct CaptureState {
    pending: HashMap<String, TurnBuffer>,
    /// Turns let go since `take_dropped` was last called.
    dropped: u64,
    /// Tokens blanked in the turns completed since `take_blanked` was last called.
    blanked: u64,
    /// Marked or withheld buffers let go before their turn ended, by session, for
    /// `take_released`.
    released: Vec<(String, TurnBuffer)>,
}

struct TurnBuffer {
    prompt: String,
    turn_key: String,
    cwd: Option<String>,
    started_ms: i64,
    /// The time of the turn's first message: a prompt that replaced another keeps it.
    turn_from_ms: i64,
    /// The keys of the prompts this one replaced in the same turn, oldest first.
    replaced_keys: Vec<String>,
    last_ms: i64,
    assistant: String,
    suppressed: bool,
    withheld: bool,
    /// Tokens `screen::redact` blanked in the text this buffer holds.
    blanked: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletedTurn {
    pub session_id: String,
    pub cwd: Option<String>,
    pub prompt: String,
    pub turn_key: String,
    pub reply: String,
    /// The hook time of the turn's prompt. For a marked or withheld turn (stored under no
    /// id) the time of the turn's first message, of a prompt it replaced when there was
    /// one: the mark's `from_ms`, and where the log delete starts.
    // RULING: v2 final capture round, item 1: carried here, not in a new field, so that
    // every `CompletedTurn` built elsewhere stays as it is.
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
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HostMessage {
    /// `<event `: something else begins. What was pending is over.
    Event,
    /// `[Context continuation`, `[Host context advisory`: the host speaks in the middle of
    /// a turn, which goes on. `[Host harness note` (core's nudge, "not a new user request",
    /// sent after a turn has ended): it starts no turn.
    Aside,
}

/// The marks a host message starts with, other than an event's.
const ASIDES: [&str; 3] = ["[Context continuation", "[Host context advisory", "[Host harness note"];

/// What is a host message, for capture and for recall's hook (`recall::on_before_message`
/// asks here): ONE list.
// RULING: v2 capture/recall review, item 6 (finding 8). The function is capture's; recall
// calls it. Core's Muse nudge `[Host harness note, not a new user request]` is one.
pub fn host_message(text: &str) -> Option<HostMessage> {
    // SPEC: 3395-3397 say "starts"; white space before the mark is skipped, so that such a
    // message is never paired as a prompt.
    let head = text.trim_start();
    if head.starts_with("<event ") {
        Some(HostMessage::Event)
    } else if ASIDES.iter().any(|mark| head.starts_with(mark)) {
        Some(HostMessage::Aside)
    } else {
        None
    }
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

    /// How many turns were let go since this was last called: a buffer evicted, a buffer
    /// cleared (`drop_session`, `on_session_end`, a host event). A turn that would not have
    /// been stored anyway, marked by a forget call or withheld by core's rule, is not one.
    // RULING: wave4 R-16 item 1. Pairing has no `Status`; the three `on_*_event` functions
    // move this count to `Status.turns_dropped`.
    pub fn take_dropped(&mut self) -> u64 {
        std::mem::take(&mut self.dropped)
    }

    /// Tokens blanked in the text of the turns `on_assistant` has completed since this was
    /// last called. A turn that carries no text (marked, withheld) adds nothing.
    // RULING: wave4 R-16 item 4. `CompletedTurn` keeps its fields; `on_assistant_event`
    // takes this right after a turn completes and adds it to `Status.tokens_blanked` when
    // the turn is sent.
    pub fn take_blanked(&mut self) -> u64 {
        std::mem::take(&mut self.blanked)
    }

    /// Removes the chat's buffer and counts the turn that goes with it. A marked or
    /// withheld buffer is not thrown away: `take_released` hands it out as a turn.
    fn let_go(&mut self, session_id: &str) {
        if let Some(buf) = self.pending.remove(session_id) {
            if !buf.suppressed && !buf.withheld {
                self.dropped += 1;
            } else {
                // RULING: v2 final capture round, item 2. Its mark's end must still move
                // and its log rows must still go: it reaches `write_turn` as a turn.
                self.released.push((session_id.to_string(), buf));
            }
        }
    }

    /// The marked or withheld buffers let go since this was last called, each as a
    /// completed turn that ends at `now_ms`. They carry no text.
    // RULING: v2 final capture round, item 2.
    pub fn take_released(&mut self, now_ms: i64) -> Vec<CompletedTurn> {
        std::mem::take(&mut self.released)
            .into_iter()
            .map(|(session_id, buf)| CompletedTurn {
                session_id,
                cwd: buf.cwd,
                prompt: String::new(),
                turn_key: buf.turn_key,
                reply: String::new(),
                started_ms: buf.turn_from_ms,
                at_ms: now_ms.max(buf.last_ms),
                suppressed: buf.suppressed,
                withheld: buf.withheld,
            })
            .collect()
    }

    /// Whether the chat's pending turn is marked by a forget call or withheld: nothing of
    /// it will be stored, so recall keeps no text of its messages.
    // RULING: v2 final capture round, item 2.
    pub fn holds_no_text(&self, session_id: &str) -> bool {
        self.pending.get(session_id).is_some_and(|buf| buf.suppressed || buf.withheld)
    }

    pub fn on_user(&mut self, session_id: Option<&str>, cwd: Option<&str>, text: &str, now_ms: i64) {
        let Some(session_id) = session_id.filter(|s| !s.is_empty()) else {
            return;
        };
        // RULING: v2 capture/recall review, item 1. A turn marked by a forget call, or one
        // core withholds, stays so to its end: a message in the middle of it (the person's,
        // or the host's) does not replace it, and none of its text is kept. The marked turn
        // then reaches `write_turn`, which moves the mark's end to the turn's end and takes
        // the turn's log rows.
        if self.pending.get(session_id).is_some_and(|buf| buf.suppressed || buf.withheld) {
            return;
        }
        // RULING: wave4 R-16 item 2. The host calls the prompt hook before each request to
        // the model, so an advisory or a continuation may arrive inside a person's turn:
        // it starts no turn and ends none. Only an event clears what is pending. A second
        // prompt of the person in one turn (steering) replaces the first, below: the turn
        // is then stored under the newer prompt.
        match host_message(text) {
            Some(HostMessage::Event) => {
                self.let_go(session_id);
                return;
            }
            Some(HostMessage::Aside) => return,
            None => {}
        }
        // Core's test is on the text as given, the whole of it. The key is of the whole
        // redacted text, before the cut (B-2f). The text is redacted before it is cut, so
        // the cut cannot leave the head of a token that is then too short to be a shape.
        let withheld = screen::core_withholds(text);
        let redacted = screen::redact(text);
        let turn_key = archive::turn_key(&redacted.text);
        let prompt = if withheld { String::new() } else { cut(&redacted.text, MAX_TEXT_BYTES).to_string() };

        // RULING: v2 final capture round, item 1. A prompt that replaces a pending one is
        // the same turn: the turn keeps the first message's time and the keys of the
        // prompts it replaced, so that a forget call later in the turn covers them too.
        let (turn_from_ms, replaced_keys) = match self.pending.get(session_id) {
            Some(prior) => {
                let mut keys = prior.replaced_keys.clone();
                if keys.len() < MAX_REPLACED_KEYS && prior.turn_key != turn_key && !keys.contains(&prior.turn_key) {
                    keys.push(prior.turn_key.clone());
                }
                keys.retain(|k| *k != turn_key);
                (prior.turn_from_ms, keys)
            }
            None => (now_ms, Vec::new()),
        };
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
                turn_from_ms,
                replaced_keys,
                last_ms: now_ms,
                assistant: String::new(),
                suppressed: false,
                withheld,
                blanked: if withheld { 0 } else { redacted.blanked as u64 },
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
            self.let_go(&session_id);
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
        // RULING: wave4 R-16 item 1. No buffer is too old to pair: a turn may run for as
        // long as its workers do. A new prompt replaces the buffer; a lost hook drops it
        // (`drop_session`).
        buf.last_ms = now_ms;
        // A marked turn keeps no text (`suppress`), a withheld one neither.
        if !buf.suppressed {
            if screen::core_withholds(text) {
                buf.withheld = true;
            }
            if buf.withheld {
                buf.prompt.clear();
                buf.assistant.clear();
                buf.blanked = 0;
            } else {
                let redacted = screen::redact(text);
                buf.blanked += redacted.blanked as u64;
                // SPEC: 3407 says "append" and not what stands between two responses: a
                // line break.
                // RULING: v2 c18b review round 3, item 3: each response joins the reply as
                // an archived unit joins its text blocks (`archive::reply_block`): trimmed,
                // and nothing for white space alone.
                if let Some(block) = archive::reply_block(&redacted.text) {
                    if !buf.assistant.is_empty() && buf.assistant.len() < MAX_TEXT_BYTES {
                        buf.assistant.push('\n');
                    }
                    let room = MAX_TEXT_BYTES - buf.assistant.len();
                    buf.assistant.push_str(cut(block, room));
                }
            }
        }
        if has_tool_use {
            return None;
        }
        let buf = self.pending.remove(session_id)?;
        self.blanked += buf.blanked;
        Some(CompletedTurn {
            session_id: session_id.to_string(),
            cwd: buf.cwd,
            prompt: buf.prompt,
            turn_key: buf.turn_key,
            reply: buf.assistant,
            started_ms: if buf.suppressed || buf.withheld { buf.turn_from_ms } else { buf.started_ms },
            at_ms: now_ms,
            suppressed: buf.suppressed,
            withheld: buf.withheld,
        })
    }

    /// `(from_ms, turn_key)` of the pending turn, which it marks; `None` when nothing is
    /// pending. `from_ms` is the time of the turn's first message (a prompt this one
    /// replaced, when there was one).
    pub fn suppress(&mut self, session_id: &str) -> Option<(i64, String)> {
        let buf = self.pending.get_mut(session_id)?;
        buf.suppressed = true;
        // SPEC: 3409-3411: the indexer stores no row for a marked turn, so its text is not
        // kept either, here or on the queue. The buffer stays for its interval.
        buf.prompt.clear();
        buf.assistant.clear();
        buf.blanked = 0;
        Some((buf.turn_from_ms, buf.turn_key.clone()))
    }

    /// The keys of the prompts the pending turn's prompt replaced, oldest first.
    // RULING: v2 final capture round, item 1: `mark_forgotten` writes a keyed mark for each.
    pub fn replaced_keys(&self, session_id: &str) -> Vec<String> {
        self.pending.get(session_id).map(|buf| buf.replaced_keys.clone()).unwrap_or_default()
    }

    /// A hook of this chat was not forwarded.
    pub fn drop_session(&mut self, session_id: &str) {
        self.let_go(session_id);
    }

    pub fn on_session_end(&mut self, session_id: &str) {
        self.let_go(session_id);
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
// RULING: wave4 R-16 item 5. `begin_write`, not `begin_write_front`: `write_turn` runs on the
// indexer thread. The rule that a write goes through `begin_write_front` is the front
// thread's, and nothing in this file writes from there.
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

/// A turn with capture on that is not stored takes its rows of the recall log with it: the
/// log decided at the prompt what only the turn's end can tell (a pause set in the middle
/// of the turn, a chat filed elsewhere since, a turn too short, a tombstone). On the bare
/// connection in its own transaction, inside `in_transaction` as part of it.
// RULING: v2 capture/recall review, item 2 (finding 6). The same delete as item 1's.
// RULING: v2 final capture round, item 5. Only where recall kept text of the turn. With
// `log_queries` off it never does, and no row is deleted. Each other reason can go either
// way, as recall decided it at the prompt and this is the turn's end (withheld: by the reply
// after a clean prompt; too short: a prompt recall took, a short reply; no project, filed
// elsewhere, paused: the file system, the archives, a pause changed during the turn; a
// tombstone: recall does not know it), so the delete stays for them.
fn not_stored(conn: &Connection, cfg: &ExtConfig, turn: &CompletedTurn) -> Result<Written> {
    if cfg.log_queries {
        recall::drop_logged_turn(conn, &turn.session_id, turn.started_ms, turn.at_ms)?;
    }
    Ok(Written::Skipped)
}

/// What a marked turn leaves behind when it ends, inside a transaction the caller holds: the
/// end of its `call` interval moves to the turn's end, and the turn's log rows go. For a
/// withheld turn that is not marked, the log rows only.
// RULING: v2 final small round, item 3: the statements of `write_turn`'s marked branch, so
// that the front thread runs the same ones for a turn whose job the queue refused.
pub(crate) fn forget_turn_rows(conn: &Connection, turn: &CompletedTurn) -> Result<()> {
    if turn.suppressed {
        // SPEC: 3435-3436 say "sets `to_ms` … to `at_ms`"; an interval is never made
        // shorter (a second call may have moved its end past this turn's clock).
        conn.execute(
            "UPDATE forgotten_turns SET to_ms = max(to_ms, ?1)
              WHERE source = 'call' AND from_ms = ?2 AND (logical = ?3 OR session_id = ?4)",
            params![turn.at_ms, turn.started_ms, project::logical(&turn.session_id), turn.session_id],
        )?;
    }
    recall::drop_logged_turn(conn, &turn.session_id, turn.started_ms, turn.at_ms)?;
    Ok(())
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
    // interval of the chat's `call` row is closed at the turn's end, and the turn's log
    // rows go (every message of it, by time: a message in the middle of the turn has its
    // own row and its own hash).
    if turn.suppressed {
        in_transaction(conn, || forget_turn_rows(conn, turn))?;
        return Ok(Written::Skipped);
    }
    if !cfg.capture {
        return Ok(Written::Skipped);
    }
    // What core would withhold. The flag is the pairing's, made from the whole text of
    // each hook; the test is made again on what the turn carries, so that no caller can
    // hand over a turn this function stores and core would not.
    if turn.withheld || screen::core_withholds(&turn.prompt) || screen::core_withholds(&turn.reply) {
        // RULING: wave4 R-16 item 3. The recall log wrote its row of this prompt at the
        // start of the turn, when the reply was not yet known, and may hold the prompt's
        // first characters and words: the row goes with the turn. This is the indexer
        // thread and the connection is bare, so the call makes its own write transaction;
        // the log's jobs were queued before this one, so the rows are there. With capture
        // off (above) the log keeps no text of a prompt and nothing is deleted.
        // RULING: v2 capture/recall review, item 1. By time, not by hash: every row of this
        // chat from the turn's first hook to its end, a message in the middle included, and
        // no row of another turn with the same prompt.
        return not_stored(conn, cfg, turn);
    }
    // RULING: v2 final capture round, item 5: recall keeps no text of a hook with an empty
    // session id (`never_captured`), so there is nothing of this turn to take out of the log.
    if turn.session_id.is_empty() {
        return Ok(Written::Skipped);
    }
    if turn.prompt.chars().count() + turn.reply.chars().count() < MIN_TURN_CHARS {
        return not_stored(conn, cfg, turn);
    }
    let Some(project) = project::resolve(turn.cwd.as_deref()) else {
        return not_stored(conn, cfg, turn);
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
            return not_stored(conn, cfg, turn);
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
            return not_stored(conn, cfg, turn);
        }

        let row = NewRow {
            id: turn_id(&turn.session_id, turn.started_ms, &turn.prompt),
            class: "episodic_turn",
            project_key: &project.scope_dir,
            title: turn.prompt.split_whitespace().take(TITLE_WORDS).collect::<Vec<_>>().join(" "),
            abstract_text: String::new(),
            // `insert_row` redacts title and content before anything is stored.
            // RULING: v2 c18b review round 3, item 3: composed as an archived unit is (the
            // prompt trimmed), so that the hash of a deleted turn's text meets its unit's.
            content: archive::exchange_text(&turn.prompt, &turn.reply),
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
        match store::insert_row(conn, signer, &row)? {
            Inserted::New { .. } | Inserted::Exists => Ok(Written::Stored),
            // Forgotten for good: left alone, and no row is there.
            Inserted::Tombstoned => not_stored(conn, cfg, turn),
        }
    })
}

/// After pairing has let buffers go: what it let go moves to `Status.turns_dropped` (with
/// capture off no turn is stored, so none is missing), and a marked or withheld buffer it
/// let go is sent as a turn that ends now. Called by the three `on_*_event` functions and
/// by the front thread wherever it lets a chat go.
pub(crate) fn after_let_go(st: &mut FrontState) {
    let dropped = st.capture.take_dropped();
    if dropped > 0 && st.cfg.capture {
        st.status.turns_dropped.fetch_add(dropped, Ordering::Relaxed);
    }
    // RULING: v2 final capture round, item 2.
    let now = (st.clock)();
    for turn in st.capture.take_released(now) {
        send_turn(st, turn, 0);
    }
}

/// After recall has been answered (called by `ext::hooks::handle`). No SQL.
pub fn on_user_event(st: &mut FrontState, ev: &HookEvent) {
    if let Some(session_id) = ev.session_id.as_deref() {
        // SPEC: 3357 names `st.clock`; the time of a hook is `ev.received_ms`, which the rpc
        // thread read from that clock when the hook arrived. It is not later than anything
        // written for this turn, which is what a forget's `from_ms` needs.
        match ev.message.as_deref() {
            Some(text) => st.capture.on_user(Some(session_id), ev.cwd.as_deref(), text, ev.received_ms),
            // A prompt hook with no text: whatever was pending is not this message's turn.
            None => st.capture.drop_session(session_id),
        }
    }
    after_let_go(st);
}

/// `on_message_complete` (called by `ext::hooks::handle`). No SQL: the turn goes to the
/// indexer as a job (a marked or withheld turn the queue refuses: see `pair_reply`).
pub fn on_assistant_event(st: &mut FrontState, ev: &HookEvent) {
    if let Some(session_id) = ev.session_id.as_deref() {
        pair_reply(st, session_id, ev);
    }
    after_let_go(st);
}

fn pair_reply(st: &mut FrontState, session_id: &str, ev: &HookEvent) {
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
    let blanked = st.capture.take_blanked();
    send_turn(st, turn, blanked);
}

/// A turn that has ended, to the indexer; `blanked`: the tokens blanked in its text.
fn send_turn(st: &mut FrontState, turn: CompletedTurn, blanked: u64) {
    let storable = st.cfg.capture && !turn.suppressed && !turn.withheld;
    // `write_turn` is given no `Status`: the count is made here, where the turn ends.
    if turn.withheld && !turn.suppressed && st.cfg.capture {
        st.status.turns_withheld.fetch_add(1, Ordering::Relaxed);
    }
    // RULING: v2 final small round, item 2: a forget mark of this turn that is still
    // waiting for the write lock ends where the turn ends, whenever it is written.
    if turn.suppressed {
        for mark in st.forget_marks.iter_mut() {
            if mark.session_id == turn.session_id && mark.from_ms == turn.started_ms {
                mark.to_ms = mark.to_ms.max(turn.at_ms);
            }
        }
    }
    // RULING: v2 final small round, item 3: a marked or withheld turn carries no text, so
    // it is kept for the case the queue refuses it.
    // RULING: v2 final capture round, item 5: a withheld turn's log rows hold no text with
    // `log_queries` off, as `not_stored` says.
    let unstored = (turn.suppressed || (turn.withheld && st.cfg.capture && st.cfg.log_queries)).then(|| turn.clone());
    // A full queue drops the turn (the job is counted there, the turn here); it arrives
    // with its archive.
    let sent = front::send_job(st, Job::Capture(turn));
    if !sent && storable {
        st.status.turns_dropped.fetch_add(1, Ordering::Relaxed);
    }
    if let (false, Some(turn)) = (sent, unstored) {
        front::forget_unqueued_turn(st, &turn);
    }
    if sent && storable {
        st.status.tokens_blanked.fetch_add(blanked, Ordering::Relaxed);
    }
}

/// `on_session_end` (called by `ext::hooks::handle`).
pub fn on_session_end_event(st: &mut FrontState, ev: &HookEvent) {
    if let Some(session_id) = ev.session_id.as_deref() {
        st.capture.on_session_end(session_id);
    }
    after_let_go(st);
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
