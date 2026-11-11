-- >>> MIGRATION 1: DEFINITIONS (begin; paste from this line to the end marker, V-2) >>>
CREATE TABLE IF NOT EXISTS axel_migrations(
  version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL, detail TEXT NOT NULL DEFAULT ''
);

-- one row per memories row: where it came from, when, its vector, whether it is hidden
CREATE TABLE IF NOT EXISTS memory_ext(
  vid          INTEGER PRIMARY KEY,
  mem_id       TEXT NOT NULL UNIQUE,     -- memories.id
  origin_kind  TEXT NOT NULL,            -- archive | note_file | capture | tool | cli | legacy
  origin_ref   TEXT NOT NULL DEFAULT '', -- archive id | notes file name | session id | ''
  unit_pos     TEXT,                     -- archive rows: '<kind><source_index>.<block_index>'
  session_id   TEXT,
  logical      TEXT,                     -- sha256(be64(len) || session_id), hex
  content_hash TEXT NOT NULL,            -- sha256 of the normalised body, hex
  turn_key     TEXT,                     -- chat turns: archive::turn_key of the whole redacted prompt
  created_ms   INTEGER NOT NULL,
  time_source  TEXT NOT NULL DEFAULT 'exact',  -- exact | archive | observed | mtime
  hidden_ms    INTEGER,                  -- NULL = visible
  hidden_by    TEXT,                     -- 'model:<session id>' | 'cli' | 'pause' | 'twin:<mem id>'
  embed_model  TEXT,
  embedding    BLOB                      -- f32 little-endian x dim; NULL until embedded
);
CREATE INDEX IF NOT EXISTS idx_memory_ext_origin  ON memory_ext(origin_kind, origin_ref);
CREATE INDEX IF NOT EXISTS idx_memory_ext_logical ON memory_ext(logical);
CREATE INDEX IF NOT EXISTS idx_memory_ext_turn    ON memory_ext(logical, turn_key) WHERE turn_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memory_ext_hash    ON memory_ext(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_ext_pending ON memory_ext(vid) WHERE embedding IS NULL;
CREATE INDEX IF NOT EXISTS idx_memory_ext_hidden  ON memory_ext(hidden_ms) WHERE hidden_ms IS NOT NULL;
CREATE TRIGGER IF NOT EXISTS memories_ad_ext AFTER DELETE ON memories
  BEGIN DELETE FROM memory_ext WHERE mem_id = OLD.id; END;

-- the delete guard (R-9i): a memories row can be deleted only inside store::purge_many, which
-- inserts the one row here at its start and deletes it at its end, in its own transaction
CREATE TABLE IF NOT EXISTS axel_purge(one INTEGER PRIMARY KEY CHECK (one = 1));
CREATE TRIGGER IF NOT EXISTS memories_bd_guard BEFORE DELETE ON memories
  WHEN NOT EXISTS (SELECT 1 FROM axel_purge)
  BEGIN SELECT RAISE(ABORT, 'this brain belongs to Axel v2; delete with axel-v2 forget'); END;

-- one row per archive FILE the extension has seen
CREATE TABLE IF NOT EXISTS archive_index(
  scope_dir    TEXT NOT NULL,            -- 64 hex: the project directory the file is in
  rel_name     TEXT NOT NULL,            -- 's000001/<id>.json' or '007.json'
  archive_id   TEXT NOT NULL,            -- 32 hex (from the name for a shelf file, from the file for a slot)
  logical      TEXT,
  digest       TEXT,
  state        TEXT NOT NULL,            -- indexed | tombstone | skipped
  skip_reason  TEXT,                     -- for skipped: the reader's reason | paused | forgotten_elsewhere | panic
  file_size    INTEGER,
  created_ms   INTEGER,
  time_source  TEXT,
  cover_ms     INTEGER,                  -- live captures of this chat up to here are covered by this archive
  session_id   TEXT,                     -- from a hook event, when there was one
  win_no       INTEGER,                  -- 4.1: the event's data.window, the number of the window that STARTS AFTER this file; NULL when no event named the file
  part_no      INTEGER,                  -- 4.1: the event's data.part, counted from 1; NULL when no event named the file
  source_message_count INTEGER, message_count INTEGER,
  rows_indexed INTEGER NOT NULL DEFAULT 0,
  forget_units INTEGER NOT NULL DEFAULT 0,      -- units left out because a forget tool was called (R3-5)
  keep_forget_turns INTEGER NOT NULL DEFAULT 0, -- 1: set by 'restore --forget-turns'; the forget rules are off for this file
  unmatched_hidden INTEGER NOT NULL DEFAULT 0,  -- hidden captures of its window that no unit matched (ruling 3)
  indexer_version INTEGER NOT NULL DEFAULT 1,
  first_seen_ms INTEGER NOT NULL,
  indexed_ms   INTEGER,
  PRIMARY KEY (scope_dir, rel_name)
);
CREATE INDEX IF NOT EXISTS idx_archive_id      ON archive_index(scope_dir, archive_id);
CREATE INDEX IF NOT EXISTS idx_archive_digest  ON archive_index(scope_dir, digest);
CREATE INDEX IF NOT EXISTS idx_archive_logical ON archive_index(logical);

CREATE TABLE IF NOT EXISTS projects(     -- display only; never used to decide scope
  scope_dir TEXT PRIMARY KEY, project_key TEXT, root_hint TEXT,
  first_seen_ms INTEGER NOT NULL, last_seen_ms INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS note_files(
  path TEXT PRIMARY KEY, project_key TEXT NOT NULL,
  size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
  ok INTEGER NOT NULL DEFAULT 1, reason TEXT,
  live_count INTEGER NOT NULL DEFAULT 0, scanned_ms INTEGER NOT NULL,
  -- the hashes of the lines the last pass could not parse, sorted, comma-joined; NULL = none
  -- (ruling on wave 2's review of commit 7: deletion by absence waits only while this changes)
  bad_lines TEXT, bad_lines_changed_ms INTEGER
);

CREATE TABLE IF NOT EXISTS chat_pauses(  -- section 4
  logical TEXT NOT NULL, what TEXT NOT NULL,          -- what: learning | recall
  since_ms INTEGER NOT NULL, set_by TEXT NOT NULL,    -- 'model' | 'cli'
  session_id TEXT,
  PRIMARY KEY (logical, what)
);
-- every turn in which a forget tool was called (section 4). ONE table (ruling 1e): revision 3's
-- suppressed_turns is gone, and the reviews' forgotten_turns / forgotten_prompts are this.
CREATE TABLE IF NOT EXISTS forgotten_turns(
  id         INTEGER PRIMARY KEY,
  logical    TEXT NOT NULL,              -- the chat
  session_id TEXT,
  source     TEXT NOT NULL,              -- call | archive | cli
  turn_key   TEXT NOT NULL DEFAULT '',   -- '' = no key (withheld prompt, no prompt in the file, no pending turn)
  from_ms    INTEGER NOT NULL,           -- call: the turn's first hook. archive: its cover_ms. cli: the row's created_ms
  to_ms      INTEGER NOT NULL,           -- call: the call, then the turn's end. archive, cli: = from_ms
  scope_dir  TEXT NOT NULL DEFAULT '',   -- archive: the file the mark was read from
  archive_id TEXT NOT NULL DEFAULT '',
  head       INTEGER NOT NULL DEFAULT 0, -- archive: the forget span is the file's leading span
  tail       INTEGER NOT NULL DEFAULT 0, -- archive: no human prompt follows the span in the file
  added_ms   INTEGER NOT NULL,
  UNIQUE (logical, source, from_ms, turn_key, archive_id)   -- the upsert's conflict target (R3-2e)
);
CREATE INDEX IF NOT EXISTS idx_forgotten_chat ON forgotten_turns(logical, from_ms);

CREATE TABLE IF NOT EXISTS forum_posts(
  id            TEXT PRIMARY KEY,        -- msg-<64hex>
  project_key   TEXT NOT NULL,           -- the project it is read in
  origin_project TEXT NOT NULL,          -- the project in the envelope and the digest
  thread_id     TEXT NOT NULL,
  reply_to      TEXT,
  title         TEXT NOT NULL,
  body          TEXT NOT NULL,
  author_actor  TEXT NOT NULL, author_group TEXT NOT NULL, author_parent TEXT,
  request_key   TEXT NOT NULL,
  retention_days INTEGER NOT NULL,
  digest        TEXT NOT NULL,
  timestamp_ms  INTEGER NOT NULL,
  expires_ms    INTEGER NOT NULL,
  origin        TEXT NOT NULL DEFAULT 'axel',  -- axel | brain.r8
  digest_ok     INTEGER NOT NULL DEFAULT 1,
  raw           TEXT                     -- imported posts: the source row, verbatim JSON
);
CREATE INDEX IF NOT EXISTS idx_forum_thread ON forum_posts(project_key, thread_id, timestamp_ms, id);
CREATE INDEX IF NOT EXISTS idx_forum_time   ON forum_posts(project_key, timestamp_ms, id);
CREATE TABLE IF NOT EXISTS forum_tombstones(
  id TEXT PRIMARY KEY, project_key TEXT NOT NULL, deleted_ms INTEGER NOT NULL,
  source TEXT NOT NULL DEFAULT 'tool',   -- tool | import | cli
  by_actor TEXT, by_group TEXT, by_parent TEXT, by_session TEXT
);

CREATE TABLE IF NOT EXISTS recall_log(
  id INTEGER PRIMARY KEY,
  ts_ms INTEGER NOT NULL, session_id TEXT, logical TEXT, scope_dir TEXT,
  query_preview TEXT,                    -- NULL unless the rule in section 3 allows it
  query_hash TEXT NOT NULL,              -- archive::turn_key of the prompt, so a purged turn takes its log row (R-7)
  outcome TEXT NOT NULL,
  total_ms REAL NOT NULL, embed_ms REAL,
  candidates INTEGER NOT NULL DEFAULT 0, selected INTEGER NOT NULL DEFAULT 0,
  detail TEXT NOT NULL                   -- JSON, shape in section 3
);
CREATE INDEX IF NOT EXISTS idx_recall_log_chat ON recall_log(logical, query_hash);

CREATE TABLE IF NOT EXISTS import_log(
  id INTEGER PRIMARY KEY, at TEXT NOT NULL,
  source TEXT NOT NULL, source_sha256 TEXT NOT NULL, item TEXT NOT NULL,
  rows_in INTEGER NOT NULL, rows_imported INTEGER NOT NULL,
  rows_present INTEGER NOT NULL, rows_flagged INTEGER NOT NULL, rows_skipped INTEGER NOT NULL DEFAULT 0,
  detail TEXT NOT NULL DEFAULT ''
);

-- the old binary cannot open this brain (below)
CREATE TRIGGER IF NOT EXISTS kv_v2_guard_insert BEFORE INSERT ON kv
  WHEN NEW.key = 'schema_version' AND CAST(NEW.value AS INTEGER) < 3
  BEGIN SELECT RAISE(ABORT, 'this brain belongs to Axel v2; use the axel-v2 binary'); END;
CREATE TRIGGER IF NOT EXISTS kv_v2_guard_update BEFORE UPDATE ON kv
  WHEN NEW.key = 'schema_version' AND CAST(NEW.value AS INTEGER) < 3
  BEGIN SELECT RAISE(ABORT, 'this brain belongs to Axel v2; use the axel-v2 binary'); END;
-- <<< MIGRATION 1: DEFINITIONS (end) <<<
