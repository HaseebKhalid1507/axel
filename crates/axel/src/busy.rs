//! How long a connection waits for a locked brain, measured on the wall clock.
//!
//! SQLite's own busy timeout (`sqlite3_busy_timeout`, `PRAGMA busy_timeout`) adds up the
//! sleeps it asked for (1, 2, 5, ... 100 ms), not the time that passed. When the machine is
//! loaded every sleep runs long, and the wait with it: on a GitHub macOS runner a 1.5 s wait
//! took 2.1 and 3.2 s, and a 150 ms one more than 200 ms, past the waits of `ext/server.rs`
//! that it must stay inside. [`set`] installs a handler with the same steps that stops when
//! the wall clock says the wait is over.
//!
//! The wait is kept as the connection's client data, so [`get`] answers what `PRAGMA
//! busy_timeout` answered before (that pragma now says 0 for a connection set here).

use std::cell::Cell;
use std::ffi::{c_int, c_void, CStr};
use std::time::{Duration, Instant};

use rusqlite::{ffi, Connection};

/// SQLite's own steps (`sqliteDefaultBusyCallback`), in milliseconds; the last repeats.
const STEPS_MS: [u64; 12] = [1, 2, 5, 10, 15, 20, 25, 25, 25, 50, 50, 100];

const KEY: &CStr = c"axel.busy_ms";

thread_local! {
    /// When the current wait began. A busy handler runs on the thread of the call that
    /// waits, and one call waits for one lock at a time.
    static SINCE: Cell<Option<Instant>> = const { Cell::new(None) };
}

/// `count` is 0 at the first call of a wait and goes up by one with each call after.
unsafe extern "C" fn wait_for_lock(arg: *mut c_void, count: c_int) -> c_int {
    let limit = Duration::from_millis(arg as usize as u64);
    let now = Instant::now();
    let since = SINCE
        .try_with(|since| match since.get() {
            Some(start) if count > 0 => start,
            _ => {
                since.set(Some(now));
                now
            }
        })
        .unwrap_or(now);
    let left = limit.saturating_sub(now.duration_since(since));
    if left.is_zero() {
        return 0;
    }
    let step = STEPS_MS[usize::try_from(count).unwrap_or(0).min(STEPS_MS.len() - 1)];
    std::thread::sleep(Duration::from_millis(step).min(left));
    1
}

/// The connection waits up to `wait` (wall clock) for a lock; zero: not at all.
pub fn set(conn: &Connection, wait: Duration) -> rusqlite::Result<()> {
    let ms = usize::try_from(wait.as_millis()).unwrap_or(usize::MAX).min(i32::MAX as usize);
    // SAFETY: the handle is this open connection's. The handler's argument is the wait
    // itself, carried in the pointer's value: nothing is pointed at, nothing is freed. The
    // client data is the same number under a key with static lifetime, with no destructor.
    let rc = unsafe {
        let db = conn.handle();
        let handler: Option<unsafe extern "C" fn(*mut c_void, c_int) -> c_int> =
            if ms == 0 { None } else { Some(wait_for_lock) };
        let rc = ffi::sqlite3_busy_handler(db, handler, ms as *mut c_void);
        if rc == ffi::SQLITE_OK {
            ffi::sqlite3_set_clientdata(db, KEY.as_ptr(), ms as *mut c_void, None)
        } else {
            rc
        }
    };
    if rc == ffi::SQLITE_OK {
        Ok(())
    } else {
        Err(rusqlite::Error::SqliteFailure(ffi::Error::new(rc), Some("busy handler".into())))
    }
}

/// The wait this connection has: SQLite's own timeout when one was set after [`set`] (a
/// non-zero `PRAGMA busy_timeout` means SQLite's handler is the one installed), else the
/// wait [`set`] gave it.
pub fn get(conn: &Connection) -> Duration {
    let pragma: i64 = conn.query_row("PRAGMA busy_timeout", [], |r| r.get(0)).unwrap_or(0);
    if pragma > 0 {
        return Duration::from_millis(u64::try_from(pragma).unwrap_or(0));
    }
    // SAFETY: reads the client data `set` stored, a number in a pointer's value.
    let ms = unsafe { ffi::sqlite3_get_clientdata(conn.handle(), KEY.as_ptr()) } as usize;
    Duration::from_millis(ms as u64)
}
