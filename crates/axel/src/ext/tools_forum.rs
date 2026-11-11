//! The three forum tools. Commit 17 writes this file.

use serde_json::Value;

use crate::ext::types::{Caller, FrontState, ToolError};

pub fn tools() -> Vec<Value> {
    Vec::new()
}

// SPEC: the stub rule (4747-4758) names no neutral answer for a tool call. The stub
// declares no tool, so every name is unknown, and an unknown name is `ToolError::Invalid`
// (3184-3186).
pub fn call(
    st: &mut FrontState,
    name: &str,
    input: &Value,
    caller: &Caller,
) -> Result<String, ToolError> {
    let _ = (st, name, input, caller);
    Err(ToolError::Invalid("unknown tool".to_string()))
}
