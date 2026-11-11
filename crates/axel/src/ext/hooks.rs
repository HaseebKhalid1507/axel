//! The hook dispatch: one arm per hook of the table in section 9, each calling its owner.
//! Final in commit 3.

use crate::ext::types::{FrontState, HookEvent, HookOut};
use crate::{capture, inject, recall};

pub fn handle(st: &mut FrontState, ev: &HookEvent, recall_allowed: bool) -> HookOut {
    match ev.kind.as_str() {
        "before_message" => {
            let out = recall::on_before_message(st, ev, recall_allowed);
            capture::on_user_event(st, ev);
            out
        }
        "on_message_complete" => {
            capture::on_assistant_event(st, ev);
            HookOut::Continue
        }
        "on_session_start" => inject::on_session_start(st, ev),
        "on_session_end" => {
            capture::on_session_end_event(st, ev);
            HookOut::Continue
        }
        // The rpc thread answers these two itself and sends the indexer a job; the front
        // thread is not involved.
        "on_context_archived" | "on_memory_forgotten" => HookOut::Continue,
        _ => HookOut::Continue,
    }
}
