//! Opening a file that another process of the same user may have swapped for something else.
//!
//! The readers of core's files (`archive.rs`, `notes.rs`) `lstat` a name, open it, and check
//! that what they opened is the regular file they saw. Between the two the name can be
//! replaced. A link is refused by `O_NOFOLLOW`; a FIFO with no writer would block a plain
//! open for ever and hang the indexer thread, so the open is `O_NONBLOCK` (which has no
//! effect on a regular file).

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

/// Opens for reading without following a link in the last place and without waiting.
/// The caller still checks, on the handle, that it is the regular file it expected.
pub(crate) fn open_no_wait(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    options.open(path)
}
