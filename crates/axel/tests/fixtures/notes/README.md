# Notes fixtures

- `project-p1111111111111111.jsonl`, `healed.jsonl`: written for `notes_reader_tests.rs`
  from core's `MemoryRecord` and `TombstoneLine`.
- `core-a373ff06.jsonl`: lines copied read-only from Synaps core's own tests at commit
  `a373ff06` (core has no notes fixture file; these are the literal lines of its tests), in
  this order:
  1-5. `crates/agent-core/tests/disclosure_retention.rs:235-252`, with `DAY_MS` (86,400,000)
       multiplied out: 80, 10, 99, 99 and 99 days.
  6.   `crates/agent-core/tests/memory_project_scope.rs:199`, the pre-T32 legacy line.
  7.   `crates/agent-core/src/memory/store.rs:2447`, a tombstone.
  8.   `crates/agent-core/src/memory/store.rs:2452`, `not json`.
  9.   `crates/agent-core/tests/memory_index.rs:206`, the torn append, with no line break.
