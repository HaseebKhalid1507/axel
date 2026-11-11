//! The front thread's loop. Commit 12b writes this file.

use std::sync::mpsc::{Receiver, Sender};

use crate::ext::config::ExtConfig;
use crate::ext::types::{Boot, Caller, FrontState, Job};
use crate::recall::View;
use crate::vindex::VecDelta;

/// Hands back the other ends of the channels it makes (B3-3a).
pub fn open_state(cfg: ExtConfig, boot: Boot) -> (FrontState, Receiver<Job>, Sender<VecDelta>) {
    let _ = (cfg, boot);
    unimplemented!("commit 12b")
}

// SPEC: 3104 names `ext::front::view_for(caller, cfg)` without types or a result. A `View`
// is assumed (2314: "what this caller may see (section 5 fills it)"). The stub's view sees
// nothing.
pub fn view_for(caller: &Caller, cfg: &ExtConfig) -> View {
    let _ = (caller, cfg);
    View { project: None, global: false, documents: false, own_session: None }
}
