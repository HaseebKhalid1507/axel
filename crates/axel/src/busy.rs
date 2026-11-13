//! How long a connection waits for a locked brain, measured on the wall clock.
//!
//! SQLite's own busy timeout (`sqlite3_busy_timeout`, `PRAGMA busy_timeout`) adds up the
//! sleeps it asked for (1, 2, 5, ... 100 ms), not the time that passed. When the machine is
//! loaded every sleep runs long, and the wait with it: on a GitHub macOS runner a 1.5 s wait
//! took 2.1 and 3.2 s, and a 150 ms one more than 200 ms, past the waits of `ext/server.rs`
//! that it must stay inside. [`set`] installs a handler with the same steps that stops when
//! the wall clock says the wait is over.
//!
//! What one wait is. SQLite calls the handler with a count that is 0 at the first call and
//! goes up by one with each call after; it puts the count back to 0 only at the start of a
//! step and at the end of a prepare, not at the start of a prepare. So a count that follows
//! the previous one is not proof of the same wait: a prepare that waits after a step that
//! waited and got its lock goes on with the step's count. A call therefore goes on with the
//! current wait when its count is the previous one plus one AND it comes within [`GAP`] of
//! the previous call's return (a retry of the same lock comes at once). A count that does
//! not follow starts a new wait with the whole budget. A count that follows but comes late
//! starts a new wait once per run of following counts, and after that goes on with the
//! wait: the stale case needs only one, since the end of the prepare puts the count back to
//! 0, and a genuine wait whose thread lost the CPU, or whose retry was slow, is not started
//! again and again. So one wait lasts at most twice its budget plus one such stall (1.5 s
//! can become about 3 s), and it always ends while the lock is held. The state is the
//! connection's own, so two connections waiting on one thread do not share it.
//!
//! The budget covers the whole of one wait, including any work SQLite does between two calls
//! of it shorter than [`GAP`]. A step that waits, works and waits again, such as
//! `PRAGMA wal_checkpoint(TRUNCATE)` (it waits for the writer, backfills, then waits for the
//! readers), may so report busy after a backfill, a short wait sooner than its time; when
//! the work took longer than [`GAP`] the second wait has a budget of its own instead (the
//! one new start its run of counts is allowed). Callers
//! take busy as "blocked" and try again later, as they would anyway.
//!
//! The wait is kept as the connection's client data, so [`get`] answers what `PRAGMA
//! busy_timeout` answered before (that pragma now says 0 for a connection set here). Only
//! this module sets a connection's busy handler: rusqlite's `Connection::busy_timeout` and
//! `Connection::busy_handler` would replace the handler and leave the client data, and
//! `clippy.toml` refuses them outside it. `PRAGMA busy_timeout = N` in SQL and a raw
//! `ffi::sqlite3_busy_timeout` would do the same and no lint sees them; neither is used.

use std::cell::Cell;
use std::ffi::{c_int, c_void, CStr};
use std::time::{Duration, Instant};

use rusqlite::{ffi, Connection};

/// SQLite's own steps (`sqliteDefaultBusyCallback`), in milliseconds; the last repeats.
const STEPS_MS: [u64; 12] = [1, 2, 5, 10, 15, 20, 25, 25, 25, 50, 50, 100];

/// The longest time between a call's return and the next call of the same wait.
pub const GAP: Duration = Duration::from_millis(50);

const KEY: &CStr = c"axel.busy";

/// One connection's wait and the state of its current wait. Owned by the connection's
/// client data, which drops it when the connection closes or the entry is replaced.
#[derive(Debug)]
pub struct Wait {
    limit: Duration,
    /// When the current wait began.
    since: Cell<Option<Instant>>,
    /// The count of the previous call and the instant it returned.
    last: Cell<Option<(c_int, Instant)>>,
    /// A late following count has started the current wait again; cleared by a count that
    /// does not follow.
    restarted: Cell<bool>,
}

impl Wait {
    pub fn new(limit: Duration) -> Self {
        Self { limit, since: Cell::new(None), last: Cell::new(None), restarted: Cell::new(false) }
    }

    /// The call with this count, at `now`: how long to sleep before the next try, or `None`
    /// to give up.
    pub fn sleep_for(&self, count: c_int, now: Instant) -> Option<Duration> {
        let follows = match (self.since.get(), self.last.get()) {
            (Some(_), Some((prev, at))) if count > 0 && Some(count) == prev.checked_add(1) => Some(at),
            _ => None,
        };
        let goes_on = match follows {
            None => {
                self.restarted.set(false);
                false
            }
            Some(at) if now.saturating_duration_since(at) <= GAP => true,
            // Late: a new wait the first time in this run, the same wait after that.
            Some(_) => self.restarted.replace(true),
        };
        if !goes_on {
            self.since.set(Some(now));
        }
        let since = self.since.get().unwrap_or(now);
        let left = self.limit.saturating_sub(now.saturating_duration_since(since));
        if left.is_zero() {
            self.last.set(None);
            return None;
        }
        let step = STEPS_MS[usize::try_from(count).unwrap_or(0).min(STEPS_MS.len() - 1)];
        Some(Duration::from_millis(step).min(left))
    }

    /// The call with this count returned at `now`, after its sleep.
    pub fn returned(&self, count: c_int, now: Instant) {
        self.last.set(Some((count, now)));
    }
}

unsafe extern "C" fn wait_for_lock(arg: *mut c_void, count: c_int) -> c_int {
    // SAFETY: `arg` is the `Wait` the connection's client data owns (`set`); it lives until
    // the entry is replaced or the connection closes, and `set` installs the handler that
    // points at the new one before any other call on the connection.
    let wait = unsafe { &*(arg as *const Wait) };
    match wait.sleep_for(count, Instant::now()) {
        Some(step) => {
            std::thread::sleep(step);
            wait.returned(count, Instant::now());
            1
        }
        None => 0,
    }
}

unsafe extern "C" fn drop_wait(p: *mut c_void) {
    // SAFETY: `p` came from `Box::into_raw` in `set` and SQLite calls this once for it.
    drop(unsafe { Box::from_raw(p as *mut Wait) });
}

fn failure(rc: c_int) -> rusqlite::Error {
    rusqlite::Error::SqliteFailure(ffi::Error::new(rc), Some("busy handler".into()))
}

/// The connection waits up to `wait` (wall clock) for a lock; zero: not at all. On an error
/// the connection is left with no wait at all, and [`get`] says zero.
pub fn set(conn: &Connection, wait: Duration) -> rusqlite::Result<()> {
    // SAFETY: the handle is this open connection's, used on this thread only (`Connection`
    // is not `Sync`), so nothing runs on it between the calls below. The client data is set
    // first: replacing an entry frees the old `Wait` while the old handler still points at
    // it, and the handler is moved to the new one at once. A failed `set_clientdata`
    // (SQLITE_NOMEM, only for a key not yet there) has dropped the new `Wait` itself.
    unsafe {
        let db = conn.handle();
        if wait.is_zero() {
            ffi::sqlite3_busy_handler(db, None, std::ptr::null_mut());
            let rc = ffi::sqlite3_set_clientdata(db, KEY.as_ptr(), std::ptr::null_mut(), None);
            return if rc == ffi::SQLITE_OK { Ok(()) } else { Err(failure(rc)) };
        }
        let p = Box::into_raw(Box::new(Wait::new(wait))) as *mut c_void;
        let rc = ffi::sqlite3_set_clientdata(db, KEY.as_ptr(), p, Some(drop_wait));
        if rc != ffi::SQLITE_OK {
            ffi::sqlite3_busy_handler(db, None, std::ptr::null_mut());
            return Err(failure(rc));
        }
        let rc = ffi::sqlite3_busy_handler(db, Some(wait_for_lock), p);
        if rc != ffi::SQLITE_OK {
            ffi::sqlite3_busy_handler(db, None, std::ptr::null_mut());
            ffi::sqlite3_set_clientdata(db, KEY.as_ptr(), std::ptr::null_mut(), None);
            return Err(failure(rc));
        }
    }
    Ok(())
}

/// The wait this connection has: SQLite's own timeout when one is in force (a non-zero
/// `PRAGMA busy_timeout` means SQLite's handler is the one installed), else the wait [`set`]
/// gave it, else zero.
pub fn get(conn: &Connection) -> Duration {
    let pragma: i64 = conn.query_row("PRAGMA busy_timeout", [], |r| r.get(0)).unwrap_or(0);
    if pragma > 0 {
        return Duration::from_millis(u64::try_from(pragma).unwrap_or(0));
    }
    // SAFETY: the entry, when there is one, is the `Wait` `set` stored, alive while the
    // connection is and `conn` is borrowed.
    unsafe {
        let p = ffi::sqlite3_get_clientdata(conn.handle(), KEY.as_ptr()) as *const Wait;
        p.as_ref().map_or(Duration::ZERO, |w| w.limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: Duration = Duration::from_millis(1);

    #[test]
    fn a_count_that_follows_at_once_goes_on_with_the_wait() {
        let w = Wait::new(100 * MS);
        let t = Instant::now();
        assert_eq!(w.sleep_for(0, t), Some(MS));
        w.returned(0, t + 60 * MS);
        assert_eq!(w.sleep_for(1, t + 61 * MS), Some(2 * MS));
        w.returned(1, t + 99 * MS);
        assert_eq!(w.sleep_for(2, t + 100 * MS), None, "the budget is spent");
    }

    #[test]
    fn a_count_that_does_not_follow_gets_the_whole_wait_again() {
        let w = Wait::new(100 * MS);
        let t = Instant::now();
        w.sleep_for(0, t);
        w.returned(5, t + 99 * MS);
        // A lower count (a reset, or another statement) is a new wait.
        assert_eq!(w.sleep_for(3, t + 100 * MS), Some(10 * MS));
        // So is the count after a give-up.
        let w = Wait::new(10 * MS);
        w.sleep_for(0, t);
        w.returned(0, t + 9 * MS);
        assert_eq!(w.sleep_for(1, t + 10 * MS), None);
        assert_eq!(w.sleep_for(2, t + 11 * MS), Some(5 * MS));
    }

    #[test]
    fn a_run_of_following_counts_starts_again_once_however_late_its_calls_come() {
        let w = Wait::new(100 * MS);
        let t = Instant::now();
        assert_eq!(w.sleep_for(0, t), Some(MS));
        w.returned(0, t + MS);
        // Late: the one new start, from t + 52 ms.
        let first = t + MS + GAP + MS;
        assert_eq!(w.sleep_for(1, first), Some(2 * MS));
        w.returned(1, t + 60 * MS);
        // Late again: the same wait, 59 ms of it spent.
        assert_eq!(w.sleep_for(2, t + 60 * MS + GAP + MS), Some(5 * MS));
        w.returned(2, t + 150 * MS);
        // And it ends at its deadline, 100 ms after the new start, late calls or not.
        assert_eq!(w.sleep_for(3, first + 100 * MS + GAP), None);
        // A count that does not follow clears the mark: the next late one starts again.
        assert_eq!(w.sleep_for(0, t + 400 * MS), Some(MS));
        w.returned(0, t + 401 * MS);
        assert_eq!(w.sleep_for(1, t + 401 * MS + GAP + MS + 90 * MS), Some(2 * MS));
    }

    #[test]
    fn a_following_count_that_comes_late_is_a_new_wait() {
        // A step waited, got its lock, and a prepare after it waits with the next count.
        let w = Wait::new(100 * MS);
        let t = Instant::now();
        w.sleep_for(0, t);
        w.returned(7, t + 95 * MS);
        let late = t + 95 * MS + GAP + MS;
        assert_eq!(w.sleep_for(8, late), Some(25 * MS), "a new wait, not one already spent");
        w.returned(8, late + 100 * MS);
        assert_eq!(w.sleep_for(9, late + 100 * MS), None);
    }
}
