//! The tool dispatch, asking each owner for its list. Final in commit 3.

use serde_json::Value;

use crate::ext::types::{Caller, FrontState, ToolError};
use crate::ext::{tools_core, tools_forum};

pub fn declarations(forum_on: bool) -> Vec<Value> {
    let mut tools = tools_core::tools();
    if forum_on {
        tools.extend(tools_forum::tools());
    }
    tools
}

pub fn call(
    st: &mut FrontState,
    name: &str,
    input: &Value,
    caller: &Caller,
) -> Result<String, ToolError> {
    if st.test.panic_tool.as_deref() == Some(name) {
        panic!("test switch: panic_tool");
    }
    if name.starts_with("forum_") {
        tools_forum::call(st, name, input, caller)
    } else {
        tools_core::call(st, name, input, caller)
    }
}
