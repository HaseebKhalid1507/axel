//! The forum contract, copied from core. Commit 10 copies
//! `CORE/crates/agent-core/src/memory/forum.rs` over this file with its tests.
//!
//! Commit 3 holds one type of it, because `ext::types::Caller` names it.

use serde::{Deserialize, Serialize};

// SPEC: 4670 uses `crate::forum::Author`, a type of commit 10's file. This is core's
// definition at 2ef1c708 (`forum.rs:16-23`), the struct alone, without its methods.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Author {
    pub actor: String,
    pub group: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
}
