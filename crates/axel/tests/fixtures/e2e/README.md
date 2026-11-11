# End-to-end fixtures

Read only by `tests/ext_e2e_tests.rs` (commit 23). Hand-made live archive records in
core's version 1 format. `scope` is empty here: the test writes the project directory of
its temp project into it before it puts the file in place, under `<id>.json`. `logical`
and `digest` are made up.

- `new-archive.json`: the file that arrives with an `on_context_archived` event and is then
  replaced by its tombstone (marker word `quokkaledger`).
- `while-down.json`: written while no extension runs; the scan of the next start indexes
  it (marker word `plovertide`).
- `decoy.json`: put outside every project directory and named by an event's `path`; it
  must never be opened (marker `DECOYMARKER8812`).

The end-to-end test also uses `context-archives/01a8…c7/` (core's fixtures) and
`context-archives/c6c6…c6/s000001/b436b6196e6ca0758c70d5faed628f29.json` (the same
archive with `carried: 0`, so that its exchange is read whole), with `scope` rewritten.
