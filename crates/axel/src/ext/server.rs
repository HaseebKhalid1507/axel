//! `run()`, `serve()`, thread start-up, the deadline pump, the breaker. Commit 12b writes
//! this file; `extension::run` becomes a two-line wrapper around `run` then.

use std::io::{Read, Write};
use std::path::Path;

use crate::error::Result;
use crate::ext::types::{Boot, ServeEnd};

// SPEC: 530-531 and 4728 name `run` without a signature. This is `extension::run`'s, which
// is to wrap it.
pub fn run(brain_path: &Path) -> Result<()> {
    let _ = brain_path;
    unimplemented!("commit 12b")
}

// SPEC: 3917 gives `serve(reader, writer, boot)` and "returns a `ServeEnd`"; the bounds on
// the two pipes are not written anywhere. `Send + 'static` is assumed, for the threads.
pub fn serve<R, W>(reader: R, writer: W, boot: Boot) -> ServeEnd
where
    R: Read + Send + 'static,
    W: Write + Send + 'static,
{
    let _ = (reader, writer, boot);
    unimplemented!("commit 12b")
}
